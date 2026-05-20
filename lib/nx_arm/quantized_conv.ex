defmodule NxArm.QuantizedConv do
  @moduledoc """
  CPU-side 2-D convolution with int8-quantised weights and f32
  activations. The weights live in BEAM-managed memory as binaries; the
  hot inner loop runs in a Rustler NIF (scalar today, NEON-vectorised
  next).

  Why this exists alongside the GPU stack:

    * The direct-conv kernel we tried writing for Rusticl hangs the
      a5xx IR3 shader compiler (see `project_nx_opencl_conv2d_blocked`),
      so the f32-on-GPU conv path is currently blocked.
    * Small CNN workloads (face unlock, person detection, wake-word with
      mel features) are inherently latency-sensitive and inherently
      small enough that CPU NEON int8 is competitive with — sometimes
      better than — our current GPU pipeline.
    * The same NIF works on every aarch64 Nerves target, not just the
      Fairphone, since the algorithm is architecture-portable Rust and
      the NEON intrinsics (when added) gate on `cfg(target_arch =
      "aarch64")`.

  ## Format

  Weights are quantised per output channel: for each output channel
  `co` (a row of the `[Cout, Kh*Kw*Cin]` weight matrix) we store
  `Kh*Kw*Cin` signed bytes plus one f32 scale. Dequantising a single
  weight is `scale[co] * weight_i8[co, kh*Kw*Cin + kw*Cin + ci]`.

  This matches `SmolLLM.Quantize`'s per-row scheme — the only
  difference is the rows are output channels of a conv kernel instead
  of rows of a linear matrix. Same `NXQ1` file format works for both.

  ## Building a `QuantizedConv` from an Axon / Bumblebee Conv2d kernel

  Bumblebee / Axon stores the kernel in HWIO order, shape
  `{Kh, Kw, Cin, Cout}`. To get to our layout:

      kernel
      |> Nx.transpose(axes: [3, 0, 1, 2])      # → {Cout, Kh, Kw, Cin}
      |> Nx.reshape({cout, kh * kw * cin})     # → flat per-channel rows

  Then `SmolLLM.Quantize.quantize_int8/1` produces the bytes + scales,
  which feed `new/4`.

  ## Convention

  Input and output are NHWC (channels-last), matching Axon and Bumblebee
  defaults. Stride and padding are 2-element / 4-element lists
  respectively: `stride = [sh, sw]`,
  `padding = [pad_top, pad_bottom, pad_left, pad_right]`.
  """

  alias NxArm.Native

  @enforce_keys [:weights, :scales, :bias, :cout, :kh, :kw, :cin]
  defstruct [:weights, :scales, :bias, :cout, :kh, :kw, :cin]

  @type t :: %__MODULE__{
          weights: binary(),
          scales: binary(),
          bias: binary(),
          cout: non_neg_integer(),
          kh: non_neg_integer(),
          kw: non_neg_integer(),
          cin: non_neg_integer()
        }

  @doc """
  Build a `QuantizedConv` from its component arrays.

    * `weights_int8` — raw int8 bytes, `Cout * Kh * Kw * Cin` long.
    * `scales` — list of `Cout` f32 scales (Elixir floats).
    * `bias` — list of `Cout` f32 biases, or `nil` for no bias.
    * `dims` — `{cout, cin, kh, kw}`.
  """
  @spec new(binary(), [float()], [float()] | nil, {pos_integer(), pos_integer(), pos_integer(), pos_integer()}) :: t()
  def new(weights_int8, scales, bias, {cout, cin, kh, kw})
      when is_binary(weights_int8) and is_list(scales) do
    expected = cout * kh * kw * cin

    if byte_size(weights_int8) != expected do
      raise ArgumentError,
            "weight buffer should be #{expected} bytes (Cout=#{cout} × Kh=#{kh} × Kw=#{kw} × Cin=#{cin}), got #{byte_size(weights_int8)}"
    end

    if length(scales) != cout do
      raise ArgumentError,
            "scales should have #{cout} entries, got #{length(scales)}"
    end

    bias_bin =
      case bias do
        nil ->
          <<>>

        list when is_list(list) and length(list) == cout ->
          for b <- list, into: <<>>, do: <<b::float-little-32>>

        list when is_list(list) ->
          raise ArgumentError,
                "bias should have #{cout} entries, got #{length(list)}"
      end

    scales_bin = for s <- scales, into: <<>>, do: <<s::float-little-32>>

    %__MODULE__{
      weights: weights_int8,
      scales: scales_bin,
      bias: bias_bin,
      cout: cout,
      kh: kh,
      kw: kw,
      cin: cin
    }
  end

  @doc """
  Apply the convolution to an input `Nx.Tensor` (NHWC f32).

  Options:
    * `:strides`  — `{sh, sw}` (default `{1, 1}`)
    * `:padding`  — `:valid` (default), `:same`, or an explicit
      `[{pt, pb}, {pl, pr}]` list (matching Nx's convention)

  Returns an `Nx.Tensor` on `Nx.BinaryBackend` (the NIF emits a
  CPU-resident binary; callers can transfer to the GPU explicitly if
  they want subsequent ops on the GPU).
  """
  @spec apply(Nx.Tensor.t(), t(), keyword()) :: Nx.Tensor.t()
  def apply(%Nx.Tensor{} = input, %__MODULE__{} = qc, opts \\ []) do
    {n, h_in, w_in, cin} =
      case Nx.shape(input) do
        {n, h, w, c} -> {n, h, w, c}
        other -> raise ArgumentError, "expected NHWC input, got shape #{inspect(other)}"
      end

    if cin != qc.cin do
      raise ArgumentError,
            "input Cin=#{cin} does not match weight Cin=#{qc.cin}"
    end

    {sh, sw} = Keyword.get(opts, :strides, {1, 1})

    {pt, pb, pl, pr} =
      case Keyword.get(opts, :padding, :valid) do
        :valid ->
          {0, 0, 0, 0}

        :same ->
          # Pad to keep H_out = ceil(H / sh), W_out = ceil(W / sw).
          pad_h = max(0, (div(h_in + sh - 1, sh) - 1) * sh + qc.kh - h_in)
          pad_w = max(0, (div(w_in + sw - 1, sw) - 1) * sw + qc.kw - w_in)
          {div(pad_h, 2), pad_h - div(pad_h, 2), div(pad_w, 2), pad_w - div(pad_w, 2)}

        [{pt, pb}, {pl, pr}] ->
          {pt, pb, pl, pr}

        other ->
          raise ArgumentError, "unsupported padding: #{inspect(other)}"
      end

    # Input must be on the host as raw f32 bytes; this is a CPU NIF.
    input_bin =
      input
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.as_type(:f32)
      |> Nx.to_binary()

    out_bin =
      Native.conv2d_int8_op(
        input_bin,
        qc.weights,
        qc.scales,
        qc.bias,
        [n, h_in, w_in, cin, qc.cout, qc.kh, qc.kw],
        [sh, sw],
        [pt, pb, pl, pr]
      )

    h_out = div(h_in + pt + pb - qc.kh, sh) + 1
    w_out = div(w_in + pl + pr - qc.kw, sw) + 1

    Nx.from_binary(out_bin, :f32, backend: Nx.BinaryBackend)
    |> Nx.reshape({n, h_out, w_out, qc.cout})
  end
end

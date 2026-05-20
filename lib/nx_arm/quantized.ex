defmodule NxArm.Quantized do
  @moduledoc """
  Quantized weight storage + matmul for ARM CPUs.

  This is the "weight-only int8" path used by llama.cpp Q8_0 and GPTQ-
  style quantization: weights stay int8 (4× memory savings vs f32),
  activations stay f32, the matmul dequantizes-and-accumulates in f32.

  On Cortex-A73 (FP3+ / Pi 4) without SDOT this is the realistic int8
  path; on ARMv8.2-A (Pi 5, newer Snapdragons) SDOT would let us run
  the matmul fully in int with another 2–4× speedup — a future
  feature gate.

  ## Building quantized weights

      f32_weight = Nx.tensor(...)           # {N, K}
      qw = NxArm.Quantized.from_f32(f32_weight)
      # qw is %NxArm.Quantized{weights: <<...>>, scales: <<...>>, shape: {n, k}}

  ## Running matmul

      out_f32 = NxArm.Quantized.matmul(qw, activations_f32)
  """

  @enforce_keys [:weights, :scales, :shape]
  defstruct [:weights, :scales, :shape]

  @type t :: %__MODULE__{
          weights: binary(),
          scales: binary(),
          shape: {pos_integer(), pos_integer()}
        }

  @doc """
  Quantize an f32 `{N, K}` weight tensor to int8 with per-row (per-
  output-channel) symmetric scales. `scale[i] = max(|row_i|) / 127`,
  `q[i, k] = clamp(round(w[i, k] / scale[i]), -128, 127)`.

  Returns an `%NxArm.Quantized{}` carrying the raw int8 + scale bytes.
  """
  def from_f32(%Nx.Tensor{shape: {n, k}, type: {:f, 32}} = weights) do
    cpu = Nx.backend_copy(weights, Nx.BinaryBackend)
    abs_max_per_row = cpu |> Nx.abs() |> Nx.reduce_max(axes: [1])

    # Avoid division-by-zero rows (all-zero weight) — use scale = 1.
    safe_abs_max =
      Nx.select(Nx.equal(abs_max_per_row, 0.0), Nx.tensor(1.0), abs_max_per_row)

    scales = Nx.divide(safe_abs_max, 127.0)
    # scales tensor of shape {n}.

    quantized =
      cpu
      |> Nx.divide(Nx.reshape(scales, {n, 1}))
      |> Nx.round()
      |> Nx.clip(-128, 127)

    # Pack to s8 little-endian.
    int8_bin =
      for v <- Nx.to_flat_list(quantized), into: <<>> do
        <<round(v)::signed-8>>
      end

    %__MODULE__{
      weights: int8_bin,
      scales: Nx.to_binary(scales),
      shape: {n, k}
    }
  end

  @doc """
  Matmul `act × q^T`, where `act` is an f32 NxArm/Nx tensor of shape
  `{..., K}` (any leading batch dims) and `q` is the quantized
  `{N, K}` weight. Output is f32 `{..., N}`.
  """
  def matmul(%__MODULE__{} = q, %Nx.Tensor{} = act) do
    {n, k} = q.shape

    act_shape = Nx.shape(act) |> Tuple.to_list()
    rank = length(act_shape)

    if Enum.at(act_shape, rank - 1) != k do
      raise ArgumentError,
            "matmul: act last axis #{Enum.at(act_shape, rank - 1)} != weight K #{k}"
    end

    lead = Enum.take(act_shape, rank - 1)
    m = Enum.reduce(lead, 1, &(&1 * &2))

    act_bin =
      act
      |> NxArm.Backend.__bin_of__()

    out_bin =
      NxArm.Native.dequant_matmul_int8_f32_op(act_bin, q.weights, q.scales, 1, m, n, k)

    out_shape = List.to_tuple(lead ++ [n])

    %Nx.Tensor{
      shape: out_shape,
      type: {:f, 32},
      names: List.duplicate(nil, rank),
      data: %NxArm.Backend{bin: out_bin}
    }
  end
end

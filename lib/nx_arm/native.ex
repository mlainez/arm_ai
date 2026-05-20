defmodule NxArm.Native do
  @moduledoc false

  # Derive Rust target triple from Nerves environment variables.
  # Mirrors the original NxCL.Native cross-compile logic so the NIF
  # builds cleanly under Nerves cross-toolchains.
  @rust_target (cond do
                  target = System.get_env("RUSTLER_TARGET") ->
                    target

                  (arch = System.get_env("TARGET_ARCH")) && System.get_env("TARGET_OS") ->
                    abi = System.get_env("TARGET_ABI") || "gnu"

                    case {arch, abi} do
                      {"aarch64", abi} -> "aarch64-unknown-linux-#{abi}"
                      {"arm", "gnueabihf"} -> "armv7-unknown-linux-gnueabihf"
                      {"arm", abi} -> "armv7-unknown-linux-#{abi}"
                      {"x86_64", abi} -> "x86_64-unknown-linux-#{abi}"
                      {arch, abi} -> "#{arch}-unknown-linux-#{abi}"
                    end

                  true ->
                    nil
                end)

  @linker_env (case {@rust_target, System.get_env("CC")} do
                 {nil, _} ->
                   []

                 {_, nil} ->
                   []

                 {target, cc} ->
                   triple_env = target |> String.upcase() |> String.replace("-", "_")
                   [{"CARGO_TARGET_#{triple_env}_LINKER", cc}]
               end)

  use Rustler,
    otp_app: :nx_arm,
    crate: "nx_arm_nif",
    target: @rust_target,
    env: @linker_env

  # ── Shape ops (dtype-agnostic) ─────────────────────────

  @doc """
  Generic n-D broadcast on raw bytes. `axes[i]` is the output axis that
  input axis `i` maps to (Nx semantics). `element_size` is the byte
  width of one element — 4 for f32, 8 for s64, 1 for u8, etc.
  """
  @spec broadcast_op(binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()], non_neg_integer()) :: binary()
  def broadcast_op(_input, _in_shape, _out_shape, _axes, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Generic n-D transpose. `axes[i]` is the input axis that becomes output axis `i`."
  @spec transpose_op(binary(), [non_neg_integer()], [non_neg_integer()], non_neg_integer()) :: binary()
  def transpose_op(_input, _in_shape, _axes, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Concatenate tensors along an axis. All shapes must match outside `axis`."
  @spec concatenate_op([binary()], [[non_neg_integer()]], non_neg_integer(), non_neg_integer()) :: binary()
  def concatenate_op(_tensors, _shapes, _axis, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Generic gather. `axes` lists which axes of `input` are indexed by
  the trailing dim of `indices`. Fast path when axes is a contiguous
  prefix (the embedding-lookup case).
  """
  @spec gather_op(
          binary(),
          [non_neg_integer()],
          binary(),
          [non_neg_integer()],
          non_neg_integer(),
          [non_neg_integer()],
          non_neg_integer()
        ) :: binary()
  def gather_op(_input, _in_shape, _indices, _idx_shape, _index_size, _axes, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Strided n-D slice. Output shape is `lengths`."
  @spec slice_op(
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()],
          non_neg_integer()
        ) :: binary()
  def slice_op(_input, _in_shape, _starts, _lengths, _strides, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Write `slice` into `tensor` at `starts`. Returns a fresh tensor."
  @spec put_slice_op(
          binary(),
          [non_neg_integer()],
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          non_neg_integer()
        ) :: binary()
  def put_slice_op(_tensor, _in_shape, _slice, _slice_shape, _starts, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  # ── f32 matmul ─────────────────────────────────────────

  @doc """
  Batched f32 matmul. `right_transposed = true` means right is laid out
  as `[B, N, K]` (the Q @ K^T pattern); `false` means `[B, K, N]`.
  `b = 1` covers the plain 2-D case.
  """
  @spec batched_matmul_f32_op(
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          boolean()
        ) :: binary()
  def batched_matmul_f32_op(_left, _right, _b, _m, _n, _k, _right_transposed),
    do: :erlang.nif_error(:nif_not_loaded)

  # ── f32 elementwise / unary / reduce ───────────────────

  @doc "Same-shape f32 elementwise binary op (add/subtract/multiply/divide/max/min/pow/atan2/remainder)."
  @spec elementwise_binary_f32_op(String.t(), binary(), binary()) :: binary()
  def elementwise_binary_f32_op(_op, _a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc ~S"""
  Scalar-broadcast f32 binary op. `side` is `"ab"` for `tensor OP scalar`,
  `"ba"` for `scalar OP tensor`.
  """
  @spec scalar_binary_f32_op(String.t(), String.t(), binary(), float()) :: binary()
  def scalar_binary_f32_op(_op, _side, _a, _scalar), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Elementwise f32 unary op."
  @spec elementwise_unary_f32_op(String.t(), binary()) :: binary()
  def elementwise_unary_f32_op(_op, _a), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused bias add: `out[i] = act[i] + bias[i mod inner]`. Bypasses
  `Nx.broadcast`'s wrapper overhead (~60–170 ms/call in our profile)
  by doing the implicit broadcast inside the NIF.
  """
  @spec bias_add_f32_op(binary(), binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def bias_add_f32_op(_act, _bias, _outer, _inner), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Reduce along the last axis. Input is `[n_outer × inner]` row-major; returns `n_outer` f32s."
  @spec reduce_axis_f32_op(String.t(), binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def reduce_axis_f32_op(_op, _input, _n_outer, _inner),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused softmax along the last axis. Input viewed as `[n_outer × inner]`
  row-major; each row gets the numerically-stable
  `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` in a single pass.
  """
  @spec softmax_f32_op(binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def softmax_f32_op(_input, _n_outer, _inner), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Fused GELU activation: `((erf(x/√2)+1)*x)/2` in one pass."
  @spec gelu_f32_op(binary()) :: binary()
  def gelu_f32_op(_input), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused LayerNorm along the last axis. Input as `[n_outer × inner]`;
  `gamma`/`beta` length-`inner`. Single pass per row: mean + variance,
  then `gamma * (x - mean) / sqrt(var + eps) + beta`.
  """
  @spec layernorm_f32_op(binary(), binary(), binary(), non_neg_integer(), non_neg_integer(), float()) :: binary()
  def layernorm_f32_op(_input, _gamma, _beta, _n_outer, _inner, _epsilon),
    do: :erlang.nif_error(:nif_not_loaded)

  # ── Conv2D (NEON int8 + f32) ───────────────────────────

  @doc "2-D NEON int8 conv (f32 activations × int8 weights × f32 per-out-channel scales)."
  @spec conv2d_int8_op(binary(), binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: binary()
  def conv2d_int8_op(_input, _weight, _scales, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Pure-f32 NEON conv2d. Same NHWC + flat-weight layout as conv2d_int8_op, no scales."
  @spec conv2d_f32_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: binary()
  def conv2d_f32_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Depthwise 2-D conv (feature_group_size == Cin). Kernel laid out as
  `{Cin, Kh, Kw}` raw f32 LE. NHWC input + NHWC output. Used by
  MobileNet/EfficientNet's per-channel spatial filter blocks.
  """
  @spec depthwise_conv2d_f32_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: binary()
  def depthwise_conv2d_f32_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)
end

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

  # Production loader: pull a precompiled NIF binary from the release
  # tarball matched to the runtime triple. Falls back to a local
  # rustc build when (a) `NX_ARM_BUILD=1` is set, (b) the runtime
  # triple isn't in the published target list, or (c) the tarball
  # checksum file isn't present (i.e. building from a git checkout).
  #
  # The `:targets` list is the supported deployment matrix; every
  # entry has a corresponding job in .github/workflows/release.yml.
  version = Mix.Project.config()[:version]

  force_build? =
    System.get_env("NX_ARM_BUILD") in ["1", "true"] or
      not File.exists?(Path.join([__DIR__, "..", "..", "checksum-Elixir.NxArm.Native.exs"]))

  use RustlerPrecompiled,
    otp_app: :nx_arm,
    crate: "nx_arm_nif",
    base_url: "https://github.com/marclainez/nx_arm/releases/download/v#{version}",
    version: version,
    nif_versions: ["2.16", "2.17"],
    targets: [
      "aarch64-unknown-linux-gnu",
      "aarch64-unknown-linux-musl",
      "armv7-unknown-linux-gnueabihf",
      "armv7-unknown-linux-musleabihf",
      "x86_64-unknown-linux-gnu",
      "x86_64-unknown-linux-musl",
      "x86_64-apple-darwin",
      "aarch64-apple-darwin"
    ],
    force_build: force_build?,
    # Nerves cross-build path: when force_build is on and a cross
    # target is set, forward that to cargo via the local-build path.
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
  Flash Attention V1 forward. Fused Q@K^T → scale → softmax → @V
  with streaming softmax — never materialises the (Sq, Sk) attention
  matrix.

    * `q`, `k`, `v` — `{B, H, S, D}` raw f32 LE bytes
    * `scale` — typically `1 / sqrt(D)`
    * `causal` — boolean, causal LM masking

  Output `{B, H, Sq, D}` raw f32 LE bytes.
  """
  @spec flash_attention_f32_op(binary(), binary(), binary(), float(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer(), boolean()) :: binary()
  def flash_attention_f32_op(_q, _k, _v, _scale, _b, _h, _sq, _sk, _d, _causal),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused linear: `out = act @ w^T + bias`. `bias` may be empty for
  no-bias linear. `activation` is one of `"none" | "relu" | "relu6"
  | "gelu" | "sigmoid" | "tanh"` (chained in the same pass).
  """
  @spec linear_f32_op(binary(), binary(), binary(), String.t(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def linear_f32_op(_act, _weights, _bias, _activation, _b, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused bias-add + activation in one pass.
  `activation` is one of `"none" | "relu" | "relu6" | "gelu" | "sigmoid" | "tanh"`.
  Saves the intermediate write+read between bias-add and activation.
  """
  @spec bias_add_activation_f32_op(binary(), binary(), String.t(), non_neg_integer(), non_neg_integer()) :: binary()
  def bias_add_activation_f32_op(_act, _bias, _activation, _outer, _inner),
    do: :erlang.nif_error(:nif_not_loaded)

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
  Winograd F(2, 3) convolution for 3×3 stride-1 NHWC. Weight layout
  `{Cout, 3, 3, Cin}` raw f32 LE. `dims` is `[N, H_in, W_in, Cin,
  Cout]`; `padding` is `[pad_top, pad_bottom, pad_left, pad_right]`.
  """
  @spec conv2d_f32_winograd_3x3_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()]) :: binary()
  def conv2d_f32_winograd_3x3_op(_input, _weight, _bias, _dims, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  General 2-D conv via im2col + GEMM. Same NHWC + flat-weight layout
  as `conv2d_f32_op`. Recommended when `Cin * Kh * Kw` is large
  (>~64): packs receptive fields into a contiguous matrix and reuses
  the cache-blocked NEON matmul kernel.
  """
  @spec conv2d_f32_im2col_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: binary()
  def conv2d_f32_im2col_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused depthwise + pointwise (1x1) conv -- the MobileNet /
  EfficientNet block. Activation between the two convs: 0 = none,
  1 = ReLU, 2 = ReLU6. Avoids materialising the intermediate Cin
  tensor by computing the depthwise scratch vector inline per output
  position.
  """
  @spec depthwise_pointwise_f32_op(
          binary(),
          binary(),
          binary(),
          binary(),
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()],
          non_neg_integer()
        ) :: binary()
  def depthwise_pointwise_f32_op(
        _input,
        _dw_weight,
        _dw_bias,
        _pw_weight,
        _pw_bias,
        _dims,
        _stride,
        _padding,
        _activation
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Depthwise 2-D conv (feature_group_size == Cin). Kernel laid out as
  `{Cin, Kh, Kw}` raw f32 LE. NHWC input + NHWC output. Used by
  MobileNet/EfficientNet's per-channel spatial filter blocks.
  """
  @spec depthwise_conv2d_f32_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: binary()
  def depthwise_conv2d_f32_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Bilinear resize for HWC u8 image buffers. Input is `in_h * in_w *
  channels` bytes; output is `out_h * out_w * channels` bytes.
  """
  @spec bilinear_resize_u8_op(binary(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def bilinear_resize_u8_op(_input, _in_h, _in_w, _channels, _out_h, _out_w),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "RMSNorm along the last axis. Pre-attention / pre-MLP norm in Llama/Mistral/Phi/Qwen."
  @spec rmsnorm_f32_op(binary(), binary(), non_neg_integer(), non_neg_integer(), float()) :: binary()
  def rmsnorm_f32_op(_input, _gamma, _n_outer, _inner, _epsilon),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Rotary Position Embedding (RoPE) for Q/K tensors. `positions` is
  s64 LE. `inv_freq` is the precomputed `1/base^(2k/head_dim)` series.
  """
  @spec rope_f32_op(binary(), binary(), binary(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def rope_f32_op(_input, _positions, _inv_freq, _n_rows, _head_dim, _heads_per_token),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "fp16 weight × f32 activation matmul. Weights converted via NEON vcvt_f32_f16 inline."
  @spec dequant_matmul_f16_f32_op(binary(), binary(), non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def dequant_matmul_f16_f32_op(_act, _weights_f16, _b, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Convert an f32 byte buffer to f16 (IEEE 754 binary16)."
  @spec f32_to_f16_op(binary()) :: binary()
  def f32_to_f16_op(_input), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Full int8 matmul: i8 activations × i8 weights × f32 per-row scales,
  returning f32. Uses SDOT (ARMv8.2-A `dotprod`) when runtime-detected,
  vmlal_s8 + vpadalq_s16 fallback otherwise.
  """
  @spec int8_matmul_f32_op(binary(), binary(), binary(), float(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def int8_matmul_f32_op(_a, _w, _w_scales, _act_scale, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Int4 (GGUF Q4_0 style) matmul: f32 activations × packed int4
  weights with per-group (group_size=32) f32 scales. Weight binary
  layout: `[N, K/2]` packed bytes (low nibble = even k); scales
  layout: `[N, K/32]` f32 LE. Returns f32 output `[M, N]`.
  """
  @spec int4_matmul_f32_op(binary(), binary(), binary(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def int4_matmul_f32_op(_a, _w_packed, _w_scales, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  NEON-accelerated Q4_0 GEMV (M=1). Same layout as
  `int4_matmul_f32_op` but specialised for single-row decode-time
  matmuls (lm_head, per-layer projections during token-by-token
  inference). Returns f32 output `[1, N]`.
  """
  @spec int4_matmul_gemv_neon_op(binary(), binary(), binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def int4_matmul_gemv_neon_op(_a, _w_packed, _w_scales, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Dequantize a GGML Q6_K blob to f32. `input` is the raw packed
  Q6_K bytes (210 bytes per 256-weight super-block); `n_elements`
  is the total weight count. Returns f32 LE.
  """
  @spec dequantize_q6_k_op(binary(), pos_integer()) :: binary()
  def dequantize_q6_k_op(_input, _n_elements),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Int8 matmul with per-token activation scales. `act_scales` is a
  length-M f32 LE binary -- one scale per row of A. Equivalent to
  `int8_matmul_f32_op` but with per-row dequant, recovering accuracy
  on outlier-heavy activation distributions.
  """
  @spec int8_matmul_f32_per_token_op(binary(), binary(), binary(), binary(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def int8_matmul_f32_per_token_op(_a, _w, _act_scales, _w_scales, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Quantize an `(M, K)` f32 activation matrix to int8 with symmetric
  per-token scales. Returns `{quantised_bin, scales_bin}`.
  """
  @spec quantize_int8_per_token_op(binary(), non_neg_integer(), non_neg_integer()) :: {binary(), binary()}
  def quantize_int8_per_token_op(_a, _m, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Open a file as a memory-mapped resource. Returns `{handle,
  file_size_bytes}`. The mmap lives as long as the handle reference;
  pages are demand-loaded from disk. Use `mmap_slice_op/3` to read a
  byte range out of it.
  """
  @spec mmap_open_op(String.t()) :: {reference(), non_neg_integer()}
  def mmap_open_op(_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Read `len` bytes at `offset` from a mmap'd file. Returns a BEAM-
  owned copy of the requested range -- the rest of the file stays on
  disk until accessed.
  """
  @spec mmap_slice_op(reference(), non_neg_integer(), non_neg_integer()) :: binary()
  def mmap_slice_op(_handle, _offset, _len),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Initialise the rayon global thread pool with `n` threads. Returns
  `:ok` on first call, `:already_initialised` if rayon has already
  started (the pool is one-shot at process startup). Call from
  application start callback before any parallel work runs.
  """
  @spec init_thread_pool_op(pos_integer()) :: :ok | :already_initialised
  def init_thread_pool_op(_n),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Reports the number of threads in the active rayon pool."
  @spec current_thread_count_op() :: pos_integer()
  def current_thread_count_op,
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Auto-detect big.LITTLE topology and pin rayon to the perf cluster.
  Returns `{status, n_threads, perf_core_ids, detection_source}`.
  Status is `:ok` on first call, `:already_initialised` if the pool
  is already up.
  """
  @spec init_perf_cluster_op() :: {:ok | :already_initialised, pos_integer(), [non_neg_integer()], String.t()}
  def init_perf_cluster_op,
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Inspect detected topology without altering the rayon pool. Returns
  `{perf_core_ids, all_core_ids, detection_source}`.
  """
  @spec detect_topology_op() :: {[non_neg_integer()], [non_neg_integer()], String.t()}
  def detect_topology_op,
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Pin the calling BEAM thread to the detected perf cluster. Use
  this from a startup warmup to make every dirty scheduler thread
  migrate once. The pin is cached thread-local; calling this on a
  thread that's already pinned is a no-op + a single read.
  """
  @spec pin_calling_thread_op() :: :ok
  def pin_calling_thread_op,
    do: :erlang.nif_error(:nif_not_loaded)

  # -------------------------------------------------------------
  # Production-readiness ops: replace BinaryBackend fallbacks for
  # argmax/argmin, select, as_type, clip, pad, gather, stack.
  # Dtype codes (matches Rust ops::Dtype):
  #   0=f32, 1=f64, 2=s8, 3=s16, 4=s32, 5=s64, 6=u8, 7=u16, 8=u32,
  #   9=u64, 10=bf16, 11=f16, 12=bool
  # -------------------------------------------------------------

  @spec argmax_axis_f32_op(binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def argmax_axis_f32_op(_input, _outer, _inner),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec argmin_axis_f32_op(binary(), non_neg_integer(), non_neg_integer()) :: binary()
  def argmin_axis_f32_op(_input, _outer, _inner),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec select_op(binary(), binary(), binary(), pos_integer()) :: binary()
  def select_op(_pred, _on_true, _on_false, _elem_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec as_type_op(binary(), non_neg_integer(), non_neg_integer(), non_neg_integer()) :: binary()
  def as_type_op(_input, _src_dtype, _dst_dtype, _n_elems),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec clip_op(binary(), non_neg_integer(), float(), float()) :: binary()
  def clip_op(_input, _dtype, _min, _max),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec pad_op(binary(), [non_neg_integer()], [non_neg_integer()], [{integer(), integer(), integer()}], binary(), pos_integer()) :: binary()
  def pad_op(_input, _in_shape, _out_shape, _pad_config, _fill, _elem_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec gather_axis0_op(binary(), binary(), pos_integer(), pos_integer()) :: binary()
  def gather_axis0_op(_input, _indices, _n_rows, _row_bytes),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec stack_axis0_op([binary()], pos_integer()) :: binary()
  def stack_axis0_op(_tensors, _tensor_bytes),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec sort_axis_f32_op(binary(), pos_integer(), pos_integer(), boolean()) :: binary()
  def sort_axis_f32_op(_input, _outer, _inner, _descending),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec argsort_axis_f32_op(binary(), pos_integer(), pos_integer(), boolean()) :: binary()
  def argsort_axis_f32_op(_input, _outer, _inner, _descending),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec reduce_all_u8_op(binary()) :: 0 | 1
  def reduce_all_u8_op(_input),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec reduce_any_u8_op(binary()) :: 0 | 1
  def reduce_any_u8_op(_input),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec reduce_product_f32_op(binary()) :: float()
  def reduce_product_f32_op(_input),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec reverse_op(binary(), [non_neg_integer()], [non_neg_integer()], pos_integer()) :: binary()
  def reverse_op(_input, _shape, _axes, _elem_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Quantize an `(N, K)` f32 weight matrix to Q4_0 packed int4 +
  scales. K must be a multiple of 32. Returns `{packed_bin,
  scales_bin}`.
  """
  @spec quantize_int4_q4_0_op(binary(), non_neg_integer(), non_neg_integer()) :: {binary(), binary()}
  def quantize_int4_q4_0_op(_w, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Weight-only int8 matmul: f32 activations × int8 weights × f32 per-row
  scales. Returns f32 output. Layout matches batched_matmul_f32 with
  `right_transposed = true` (acts {B,M,K} × weights {N,K} → out {B,M,N}).

  Used to run quantized LLMs (GPTQ Q8_0 / llama.cpp-style) on Nerves
  devices — weights stay int8 (4× memory savings vs f32) while the
  matmul itself dequantizes and accumulates in f32.
  """
  @spec dequant_matmul_int8_f32_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def dequant_matmul_int8_f32_op(_act, _weights, _scales, _b, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Generic n-D window reduction: `op` in `~w(max min sum product)`.
  Used as the backend for `window_max/min/sum/product` and via
  decomposition for max_pool/avg_pool.
  """
  @spec window_reduce_f32_op(
          String.t(),
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()],
          [integer()],
          [integer()]
        ) :: binary()
  def window_reduce_f32_op(_op, _input, _in_shape, _window_dims, _strides, _pad_low, _pad_high),
    do: :erlang.nif_error(:nif_not_loaded)
end

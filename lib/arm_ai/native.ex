defmodule ArmAI.Native do
  @moduledoc false

  # Derive the Rust target triple from the Nerves environment
  # (TARGET_ARCH / TARGET_OS / TARGET_ABI) so the NIF cross-compiles
  # under Nerves toolchains. RUSTLER_TARGET overrides it.
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
  # rustc build when (a) `ARM_AI_BUILD=1` is set, (b) the runtime
  # triple isn't in the published target list, or (c) the tarball
  # checksum file isn't present (i.e. building from a git checkout).
  # No release has been published yet, so today it always builds.
  #
  # The `:targets` list is the supported deployment matrix; every
  # entry has a corresponding job in .github/workflows/release.yml.
  version = Mix.Project.config()[:version]

  force_build? =
    System.get_env("ARM_AI_BUILD") in ["1", "true"] or
      not File.exists?(Path.join([__DIR__, "..", "..", "checksum-Elixir.ArmAI.Native.exs"]))

  # Cargo features. The crate's `default` is `full` (every
  # capability). Override per-application to shrink the binary:
  #
  #     config :arm_ai, features: ["chatbot"]           # 10 MB
  #     config :arm_ai, features: ["whisper"]           # 13 MB
  #     config :arm_ai, features: ["yolo"]              # 26 MB
  #     config :arm_ai, features: ["onnx", "vision"]    # compose your own
  #     config :arm_ai, features: []                    # 2.3 MB core only
  #
  # See docs/size_profile.md for the full matrix + presets.
  # Honoured by the local-build path; precompiled releases on
  # GitHub always ship the `full` set.
  cargo_features =
    case Application.compile_env(:arm_ai, :features, :default) do
      :default -> nil
      list when is_list(list) -> list |> Enum.map(&to_string/1)
    end

  use RustlerPrecompiled,
    otp_app: :arm_ai,
    crate: "arm_ai_nif",
    base_url: "https://github.com/mlainez/arm_ai/releases/download/v#{version}",
    version: version,
    nif_versions: ["2.16", "2.17"],
    targets: [
      "aarch64-unknown-linux-gnu",
      "armv7-unknown-linux-gnueabihf",
      "x86_64-unknown-linux-gnu",
      "aarch64-apple-darwin"
    ],
    force_build: force_build?,
    # When user override is given, pass it through to cargo and skip
    # default features. When unset, let cargo use the crate's
    # `default` (= `full`) so precompiled tarballs match.
    features: cargo_features || [],
    default_features: cargo_features == nil,
    target: @rust_target,
    env: @linker_env

  # ── Shape ops (dtype-agnostic) ─────────────────────────

  @doc """
  Generic n-D broadcast on raw bytes. `axes[i]` is the output axis that
  input axis `i` maps to (Nx semantics). `element_size` is the byte
  width of one element — 4 for f32, 8 for s64, 1 for u8, etc.
  """
  @spec broadcast_op(
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()],
          non_neg_integer()
        ) :: binary()
  def broadcast_op(_input, _in_shape, _out_shape, _axes, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Generic n-D transpose. `axes[i]` is the input axis that becomes output axis `i`."
  @spec transpose_op(binary(), [non_neg_integer()], [non_neg_integer()], non_neg_integer()) ::
          binary()
  def transpose_op(_input, _in_shape, _axes, _element_size),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Concatenate tensors along an axis. All shapes must match outside `axis`."
  @spec concatenate_op([binary()], [[non_neg_integer()]], non_neg_integer(), non_neg_integer()) ::
          binary()
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
  @spec flash_attention_f32_op(
          binary(),
          binary(),
          binary(),
          float(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          boolean()
        ) :: binary()
  def flash_attention_f32_op(_q, _k, _v, _scale, _b, _h, _sq, _sk, _d, _causal),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused linear: `out = act @ w^T + bias`. `bias` may be empty for
  no-bias linear. `activation` is one of `"none" | "relu" | "relu6"
  | "gelu" | "sigmoid" | "tanh"` (chained in the same pass).
  """
  @spec linear_f32_op(
          binary(),
          binary(),
          binary(),
          String.t(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def linear_f32_op(_act, _weights, _bias, _activation, _b, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused bias-add + activation in one pass.
  `activation` is one of `"none" | "relu" | "relu6" | "gelu" | "sigmoid" | "tanh"`.
  Saves the intermediate write+read between bias-add and activation.
  """
  @spec bias_add_activation_f32_op(
          binary(),
          binary(),
          String.t(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
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
  @spec layernorm_f32_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          float()
        ) :: binary()
  def layernorm_f32_op(_input, _gamma, _beta, _n_outer, _inner, _epsilon),
    do: :erlang.nif_error(:nif_not_loaded)

  # ── Conv2D (NEON int8 + f32) ───────────────────────────

  @doc "2-D NEON int8 conv (f32 activations × int8 weights × f32 per-out-channel scales)."
  @spec conv2d_int8_op(
          binary(),
          binary(),
          binary(),
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()]
        ) :: binary()
  def conv2d_int8_op(_input, _weight, _scales, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Pure-f32 NEON conv2d. Same NHWC + flat-weight layout as conv2d_int8_op, no scales."
  @spec conv2d_f32_op(binary(), binary(), binary(), [non_neg_integer()], [non_neg_integer()], [
          non_neg_integer()
        ]) :: binary()
  def conv2d_f32_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Winograd F(2, 3) convolution for 3×3 stride-1 NHWC. Weight layout
  `{Cout, 3, 3, Cin}` raw f32 LE. `dims` is `[N, H_in, W_in, Cin,
  Cout]`; `padding` is `[pad_top, pad_bottom, pad_left, pad_right]`.
  """
  @spec conv2d_f32_winograd_3x3_op(binary(), binary(), binary(), [non_neg_integer()], [
          non_neg_integer()
        ]) :: binary()
  def conv2d_f32_winograd_3x3_op(_input, _weight, _bias, _dims, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  General 2-D conv via im2col + GEMM. Same NHWC + flat-weight layout
  as `conv2d_f32_op`. Recommended when `Cin * Kh * Kw` is large
  (>~64): packs receptive fields into a contiguous matrix and reuses
  the cache-blocked NEON matmul kernel.
  """
  @spec conv2d_f32_im2col_op(
          binary(),
          binary(),
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()]
        ) :: binary()
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
  @spec depthwise_conv2d_f32_op(
          binary(),
          binary(),
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [non_neg_integer()]
        ) :: binary()
  def depthwise_conv2d_f32_op(_input, _weight, _bias, _dims, _stride, _padding),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Bilinear resize for HWC u8 image buffers. Input is `in_h * in_w *
  channels` bytes; output is `out_h * out_w * channels` bytes.
  """
  @spec bilinear_resize_u8_op(
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def bilinear_resize_u8_op(_input, _in_h, _in_w, _channels, _out_h, _out_w),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "RMSNorm along the last axis. Pre-attention / pre-MLP norm in Llama/Mistral/Phi/Qwen."
  @spec rmsnorm_f32_op(binary(), binary(), non_neg_integer(), non_neg_integer(), float()) ::
          binary()
  def rmsnorm_f32_op(_input, _gamma, _n_outer, _inner, _epsilon),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Rotary Position Embedding (RoPE) for Q/K tensors. `positions` is
  s64 LE. `inv_freq` is the precomputed `1/base^(2k/head_dim)` series.
  """
  @spec rope_f32_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def rope_f32_op(_input, _positions, _inv_freq, _n_rows, _head_dim, _heads_per_token),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "fp16 weight × f32 activation matmul. Weights converted via NEON vcvt_f32_f16 inline."
  @spec dequant_matmul_f16_f32_op(
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
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
  @spec int8_matmul_f32_op(
          binary(),
          binary(),
          binary(),
          float(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def int8_matmul_f32_op(_a, _w, _w_scales, _act_scale, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Int4 (GGUF Q4_0 style) matmul: f32 activations × packed int4
  weights with per-group (group_size=32) f32 scales. Weight binary
  layout: `[N, K/2]` packed bytes (low nibble = even k); scales
  layout: `[N, K/32]` f32 LE. Returns f32 output `[M, N]`.
  """
  @spec int4_matmul_f32_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def int4_matmul_f32_op(_a, _w_packed, _w_scales, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  NEON-accelerated Q4_0 GEMV (M=1). Same layout as
  `int4_matmul_f32_op` but specialised for single-row decode-time
  matmuls (lm_head, per-layer projections during token-by-token
  inference). Returns f32 output `[1, N]`.
  """
  @spec int4_matmul_gemv_neon_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def int4_matmul_gemv_neon_op(_a, _w_packed, _w_scales, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  NEON Q4_0 × Q8_0 GEMV (M=1). Pre-quantises activations to Q8_0,
  keeps the inner FMA in int8 lanes (vmull_s8 → i16 → i32, scale
  once per 32-weight group). 2-4× faster than the f32 path on
  ARMv8.0 (A53/A72/A73) which lacks dotprod. Same return layout
  as `int4_matmul_gemv_neon_op`.
  """
  @spec int4_matmul_gemv_q4_x_q8_neon_op(
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def int4_matmul_gemv_q4_x_q8_neon_op(_a, _w_packed, _w_scales, _n, _k),
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
  Fused softmax along the last axis. Replaces the 5-NIF Elixir
  chain (reduce_max + subtract + exp + sum + divide) with one
  fused pass. Input layout: `outer * inner` f32 LE.
  """
  @spec softmax_last_axis_f32_op(binary(), pos_integer(), pos_integer()) :: binary()
  def softmax_last_axis_f32_op(_input, _outer, _inner),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Fused `silu(gate) * up` (SwiGLU FFN inner). Replaces sigmoid +
  multiply + multiply (3 NIFs) with one pass. Both inputs same
  length, returned tensor same length.
  """
  @spec silu_gate_mul_up_f32_op(binary(), binary()) :: binary()
  def silu_gate_mul_up_f32_op(_gate, _up),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Fast Q4_0 row dequant for embedding-table lookups."
  @spec q4_0_dequant_row_op(binary(), binary(), non_neg_integer(), pos_integer()) :: binary()
  def q4_0_dequant_row_op(_packed, _scales, _row, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Fast Q8_0 row dequant for embedding-table lookups."
  @spec q8_0_dequant_row_op(binary(), binary(), non_neg_integer(), pos_integer()) :: binary()
  def q8_0_dequant_row_op(_weights, _scales, _row, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Load a quantised Llama-family GGUF file via the upstream `candle`
  crate. Returns a model handle that subsequent `llama_candle_*` NIFs
  consume. Supports Q4_0 / Q4_K / Q5_0 / Q5_K / Q6_K / Q8_0 — every
  k-quant ggml ships, courtesy of candle's GGML reader.
  """
  @spec llama_candle_load_op(String.t()) :: reference()
  def llama_candle_load_op(_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Greedy decode via `candle`. Returns `{generated_token_ids,
  prefill_us, decode_us}`.
  """
  @spec llama_candle_generate_op(
          reference(),
          [non_neg_integer()],
          non_neg_integer(),
          [non_neg_integer()]
        ) :: {[non_neg_integer()], non_neg_integer(), non_neg_integer()}
  def llama_candle_generate_op(_model, _prompt, _max_new, _stop_tokens),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Same as `llama_candle_generate_op/4`, also sending `{:llama_token, id}`
  to `pid` as each token is chosen.
  """
  @spec llama_candle_stream_op(
          reference(),
          [non_neg_integer()],
          non_neg_integer(),
          [non_neg_integer()],
          pid()
        ) :: {[non_neg_integer()], non_neg_integer(), non_neg_integer()}
  def llama_candle_stream_op(_model, _prompt, _max_new, _stop_tokens, _pid),
    do: :erlang.nif_error(:nif_not_loaded)

  # --- Embeddings / RAG primitives ---

  @spec l2_normalize_rows_f32_op(binary(), pos_integer(), pos_integer()) :: binary()
  def l2_normalize_rows_f32_op(_data, _n, _d), do: :erlang.nif_error(:nif_not_loaded)

  @spec cosine_similarity_f32_op(binary(), binary(), pos_integer(), pos_integer()) :: binary()
  def cosine_similarity_f32_op(_q, _corpus, _n, _d), do: :erlang.nif_error(:nif_not_loaded)

  @spec top_k_indices_f32_op(binary(), pos_integer()) :: [non_neg_integer()]
  def top_k_indices_f32_op(_scores, _k), do: :erlang.nif_error(:nif_not_loaded)

  # --- Scatter ops (indexed_add / indexed_put) ---

  @spec indexed_add_f32_op(binary(), binary(), binary()) :: binary()
  def indexed_add_f32_op(_tensor, _indices, _updates),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec indexed_put_f32_op(binary(), binary(), binary()) :: binary()
  def indexed_put_f32_op(_tensor, _indices, _updates),
    do: :erlang.nif_error(:nif_not_loaded)

  # --- FFT (rustfft) ---

  @spec fft_complex_op(binary()) :: binary()
  def fft_complex_op(_input), do: :erlang.nif_error(:nif_not_loaded)

  @spec ifft_complex_op(binary()) :: binary()
  def ifft_complex_op(_input), do: :erlang.nif_error(:nif_not_loaded)

  @spec rfft_op(binary()) :: binary()
  def rfft_op(_input), do: :erlang.nif_error(:nif_not_loaded)

  # --- Whisper (candle-transformers) ---

  @spec whisper_load_op(String.t(), String.t(), String.t(), String.t()) :: reference()
  def whisper_load_op(_model, _tok, _mel, _config),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec whisper_transcribe_op(reference(), binary()) :: String.t()
  def whisper_transcribe_op(_handle, _pcm), do: :erlang.nif_error(:nif_not_loaded)

  # --- Vision (image + fast_image_resize) ---

  @spec vision_decode_to_rgb8_op(String.t()) :: {binary(), pos_integer(), pos_integer()}
  def vision_decode_to_rgb8_op(_path), do: :erlang.nif_error(:nif_not_loaded)

  @spec vision_load_for_classifier_op(
          String.t(),
          pos_integer(),
          pos_integer(),
          {float(), float(), float()},
          {float(), float(), float()},
          :nchw | :nhwc
        ) :: binary()
  def vision_load_for_classifier_op(_path, _h, _w, _mean, _std, _layout),
    do: :erlang.nif_error(:nif_not_loaded)

  # --- Audio (symphonia + rubato) ---

  @spec audio_decode_file_op(String.t()) :: {binary(), pos_integer(), pos_integer()}
  def audio_decode_file_op(_path), do: :erlang.nif_error(:nif_not_loaded)

  @spec audio_to_mono_op(binary(), pos_integer()) :: binary()
  def audio_to_mono_op(_samples, _channels), do: :erlang.nif_error(:nif_not_loaded)

  @spec audio_resample_op(binary(), pos_integer(), pos_integer()) :: binary()
  def audio_resample_op(_samples, _from_hz, _to_hz), do: :erlang.nif_error(:nif_not_loaded)

  @spec audio_load_for_whisper_op(String.t()) :: binary()
  def audio_load_for_whisper_op(_path), do: :erlang.nif_error(:nif_not_loaded)

  @spec audio_write_wav_op(String.t(), binary(), pos_integer(), pos_integer()) :: :ok
  def audio_write_wav_op(_path, _samples, _sr, _channels),
    do: :erlang.nif_error(:nif_not_loaded)

  # --- ONNX (tract-onnx) ---

  @doc """
  Load an ONNX model file. Returns `{model_handle, input_names,
  output_names}`. Names match the ONNX graph (BERT inputs are
  typically `input_ids`, `attention_mask`, etc.).
  """
  @spec onnx_load_op(String.t()) :: {reference(), [String.t()], [String.t()]}
  def onnx_load_op(_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Run an ONNX model. `inputs` is a list of `{name, shape, bin}`
  where `bin` is f32-little binary of length `Enum.product(shape) * 4`.
  Returns `[{name, shape, bin}]` for each output, all f32.
  """
  @spec onnx_run_op(reference(), [{String.t(), [non_neg_integer()], binary()}]) ::
          [{String.t(), [non_neg_integer()], binary()}]
  def onnx_run_op(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Int8 matmul with per-token activation scales. `act_scales` is a
  length-M f32 LE binary -- one scale per row of A. Equivalent to
  `int8_matmul_f32_op` but with per-row dequant, recovering accuracy
  on outlier-heavy activation distributions.
  """
  @spec int8_matmul_f32_per_token_op(
          binary(),
          binary(),
          binary(),
          binary(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: binary()
  def int8_matmul_f32_per_token_op(_a, _w, _act_scales, _w_scales, _m, _n, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Quantize an `(M, K)` f32 activation matrix to int8 with symmetric
  per-token scales. Returns `{quantised_bin, scales_bin}`.
  """
  @spec quantize_int8_per_token_op(binary(), non_neg_integer(), non_neg_integer()) ::
          {binary(), binary()}
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
  @spec init_perf_cluster_op() ::
          {:ok | :already_initialised, pos_integer(), [non_neg_integer()], String.t()}
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
  Full topology: `{perf_cores, efficiency_cores, all_cores, source}`.
  On homogeneous chips `efficiency_cores` is empty.
  """
  @spec detect_topology_full_op() ::
          {[non_neg_integer()], [non_neg_integer()], [non_neg_integer()], String.t()}
  def detect_topology_full_op,
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Pin the calling OS thread to the given list of CPUs via
  sched_setaffinity. Best-effort: returns :ok regardless.
  """
  @spec pin_thread_to_cores_op([non_neg_integer()]) :: :ok
  def pin_thread_to_cores_op(_cores),
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

  @spec pad_op(
          binary(),
          [non_neg_integer()],
          [non_neg_integer()],
          [{integer(), integer(), integer()}],
          binary(),
          pos_integer()
        ) :: binary()
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
  @spec quantize_int4_q4_0_op(binary(), non_neg_integer(), non_neg_integer()) ::
          {binary(), binary()}
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

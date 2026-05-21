defmodule NxArm.Models.Llama do
  @moduledoc """
  Llama-architecture inference using NxArm primitives. Loads a GGUF
  Q4_0 file and runs greedy / sampled decode on-device.

  Architecture: pre-norm RMSNorm + SwiGLU FFN + RoPE + GQA-or-MHA.
  Covers TinyLlama 1.1B, SmolLM, Phi (Llama mode), and any Llama-2
  / Llama-3 lineage that ships as GGUF.

  Usage:

      {:ok, model} = NxArm.Models.Llama.load("/tmp/tinyllama.gguf")
      {tokens, stats} = NxArm.Models.Llama.generate(model,
        prompt_tokens: [1, 12345, ...],
        max_new: 32
      )

  `stats` reports prefill_ms and decode_ms_per_tok so callers can
  put real numbers next to the model.
  """

  alias NxArm.GGUF

  defstruct [
    :gguf,
    :config,
    :embed,
    :layers,
    :final_norm,
    :lm_head,
    :inv_freq,
    :kv_cache
  ]

  defmodule Config do
    @moduledoc false
    defstruct [
      :vocab,
      :hidden,
      :ff_dim,
      :n_layers,
      :n_heads,
      :n_kv_heads,
      :head_dim,
      :rope_theta,
      :rms_eps,
      :max_seq
    ]
  end

  defmodule Layer do
    @moduledoc false
    defstruct [
      :attn_norm,
      :w_q, :w_k, :w_v, :w_o,
      :ffn_norm,
      :w_gate, :w_up, :w_down
    ]
  end

  # ----------------------------------------------------------------
  # Loading
  # ----------------------------------------------------------------

  @doc """
  Parse a GGUF file and pull in tensor references. Quantised tensors
  stay packed; we unpack on demand inside the matmul calls.
  """
  def load(path) do
    with {:ok, gguf} <- GGUF.read_mmap(path) do
      config = build_config(gguf.metadata)
      {:ok, build(gguf, config)}
    end
  end

  defp build_config(meta) do
    %Config{
      vocab:      meta["llama.vocab_size"]      || meta["tokenizer.ggml.tokens"] |> safe_len(),
      hidden:     meta["llama.embedding_length"],
      ff_dim:     meta["llama.feed_forward_length"],
      n_layers:   meta["llama.block_count"],
      n_heads:    meta["llama.attention.head_count"],
      n_kv_heads: meta["llama.attention.head_count_kv"] || meta["llama.attention.head_count"],
      head_dim:   div(meta["llama.embedding_length"], meta["llama.attention.head_count"]),
      rope_theta: meta["llama.rope.freq_base"] || 10_000.0,
      rms_eps:    meta["llama.attention.layer_norm_rms_epsilon"] || 1.0e-5,
      max_seq:    meta["llama.context_length"] || 2048
    }
  end

  defp safe_len(l) when is_list(l), do: length(l)
  defp safe_len(_), do: nil

  defp build(gguf, %Config{} = c) do
    layers =
      for i <- 0..(c.n_layers - 1) do
        prefix = "blk.#{i}"

        %Layer{
          attn_norm: gguf_unpack(gguf, "#{prefix}.attn_norm.weight"),
          w_q:       gguf_unpack(gguf, "#{prefix}.attn_q.weight"),
          w_k:       gguf_unpack(gguf, "#{prefix}.attn_k.weight"),
          w_v:       gguf_unpack(gguf, "#{prefix}.attn_v.weight"),
          w_o:       gguf_unpack(gguf, "#{prefix}.attn_output.weight"),
          ffn_norm:  gguf_unpack(gguf, "#{prefix}.ffn_norm.weight"),
          w_gate:    gguf_unpack(gguf, "#{prefix}.ffn_gate.weight"),
          w_up:      gguf_unpack(gguf, "#{prefix}.ffn_up.weight"),
          w_down:    gguf_unpack(gguf, "#{prefix}.ffn_down.weight")
        }
      end

    final_norm = gguf_unpack(gguf, "output_norm.weight")

    # Many GGUF files tie the LM head to the input embedding; the
    # "output.weight" tensor is optional in that case.
    lm_head =
      case Map.fetch(gguf.tensors, "output.weight") do
        {:ok, _} -> gguf_unpack(gguf, "output.weight")
        :error -> gguf_unpack(gguf, "token_embd.weight")
      end

    embed = gguf_unpack(gguf, "token_embd.weight")

    inv_freq = NxArm.LLM.rope_inv_freq(c.head_dim, c.rope_theta)

    %__MODULE__{
      gguf: gguf,
      config: c,
      embed: embed,
      layers: layers,
      final_norm: final_norm,
      lm_head: lm_head,
      inv_freq: inv_freq,
      kv_cache: nil
    }
  end

  # Returns a tagged tensor:
  #   {:q4_0, %{packed: bin, scales: bin, n: ..., k: ...}}
  #   {:f32, Nx.Tensor.t}
  # so the matmul site can pick the right NIF.
  defp gguf_unpack(gguf, name) do
    case Map.fetch!(gguf.tensors, name) do
      %{dtype: :q4_0} ->
        {:ok, q} = GGUF.q4_0_unpack(gguf, name)
        {:q4_0, q}

      %{dtype: :q8_0} ->
        {:ok, q} = GGUF.q8_0_unpack(gguf, name)
        {:q8_0, q}

      %{dtype: :q6_k} = info ->
        # Pragmatic: dequantize Q6_K to f32 at load time. We only see
        # Q6_K on the LM head in TinyLlama, which is ~256 MB at f32;
        # a dedicated Q6_K matmul kernel can land later if we want
        # to save the RAM. Tensor is consumed as a weight (N, K)
        # row-major in ggml — shape after reverse is (N, K).
        {:ok, bin} = GGUF.tensor_bytes(gguf, name)
        n_elements = Enum.reduce(info.shape, 1, &(&1 * &2))
        f32_bin = NxArm.Native.dequantize_q6_k_op(bin, n_elements)

        shape = info.shape |> Enum.reverse() |> List.to_tuple()

        t =
          Nx.from_binary(f32_bin, :f32)
          |> Nx.reshape(shape)
          |> Nx.backend_copy(NxArm.Backend)

        {:f32, t}

      %{dtype: :f32} = info ->
        {:ok, bin} = GGUF.tensor_bytes(gguf, name)
        shape = info.shape |> Enum.reverse() |> List.to_tuple()
        t =
          Nx.from_binary(bin, :f32)
          |> Nx.reshape(shape)
          |> Nx.backend_copy(NxArm.Backend)

        {:f32, t}

      %{dtype: dt} = info ->
        raise "unsupported tensor dtype #{inspect(dt)} for #{name} shape=#{inspect(info.shape)}"
    end
  end

  # ----------------------------------------------------------------
  # Inference
  # ----------------------------------------------------------------

  @doc """
  Greedy decode from a list of prompt token ids. Returns
  `{tokens, stats}` where `tokens` is the list of all tokens
  including the prompt, and `stats` carries timing.
  """
  def generate(%__MODULE__{} = model, opts) do
    prompt = Keyword.fetch!(opts, :prompt_tokens)
    max_new = Keyword.get(opts, :max_new, 16)
    eos = Keyword.get(opts, :eos_token)

    cache = NxArm.KVCache.new(
      model.config.n_layers,
      model.config.n_kv_heads,
      model.config.max_seq,
      model.config.head_dim
    )

    {prefill_us, {logits_last, cache, n_prompt}} =
      :timer.tc(fn -> prefill(model, prompt, cache) end)

    next = greedy(logits_last)

    {decode_us, {tokens, _cache}} =
      :timer.tc(fn ->
        decode_loop(model, [next], cache, n_prompt, max_new - 1, eos)
      end)

    all_tokens = prompt ++ [next | tokens]

    stats = %{
      n_prompt: n_prompt,
      n_new: length(all_tokens) - n_prompt,
      prefill_ms: prefill_us / 1000.0,
      decode_total_ms: decode_us / 1000.0,
      decode_ms_per_tok:
        case length(all_tokens) - n_prompt - 1 do
          0 -> nil
          n -> decode_us / 1000.0 / n
        end,
      prefill_tokens_per_sec: n_prompt * 1.0e6 / max(prefill_us, 1)
    }

    {all_tokens, stats}
  end

  defp prefill(model, tokens, cache) do
    x = embed_tokens(model, tokens)
    seq = length(tokens)

    {final_x, cache} = run_layers(model, x, cache, 0)
    logits = project_lm(model, last_row(final_x))
    {logits, cache, seq}
  end

  defp decode_loop(_model, acc, cache, _pos, 0, _eos), do: {Enum.reverse(acc), cache}

  defp decode_loop(model, [last | _] = acc, cache, pos, remaining, eos) do
    if eos != nil and last == eos do
      {Enum.reverse(acc), cache}
    else
      x = embed_tokens(model, [last])
      {final_x, cache} = run_layers(model, x, cache, pos)
      logits = project_lm(model, last_row(final_x))
      next = greedy(logits)
      decode_loop(model, [next | acc], cache, pos + 1, remaining - 1, eos)
    end
  end

  defp last_row(x) do
    {rows, hidden} = Nx.shape(x)
    Nx.slice(x, [rows - 1, 0], [1, hidden])
  end

  defp embed_tokens(model, token_ids) do
    d = model.config.hidden

    case model.embed do
      {:f32, embed} ->
        rows = for tok <- token_ids, do: Nx.slice(embed, [tok, 0], [1, d])
        case rows do
          [single] -> single
          many -> Nx.concatenate(many, axis: 0)
        end

      {:q8_0, q} ->
        rows =
          for tok <- token_ids do
            row_bin = GGUF.q8_0_dequant_row(q, tok)
            Nx.from_binary(row_bin, :f32) |> Nx.reshape({1, d}) |> Nx.backend_copy(NxArm.Backend)
          end

        case rows do
          [single] -> single
          many -> Nx.concatenate(many, axis: 0)
        end

      {:q4_0, q} ->
        rows =
          for tok <- token_ids do
            row_bin = GGUF.q4_0_dequant_row(q, tok)
            Nx.from_binary(row_bin, :f32) |> Nx.reshape({1, d}) |> Nx.backend_copy(NxArm.Backend)
          end

        case rows do
          [single] -> single
          many -> Nx.concatenate(many, axis: 0)
        end
    end
  end

  defp run_layers(model, x, cache, start_pos) do
    {final_x, cache_out} =
      Enum.reduce(Enum.with_index(model.layers), {x, cache}, fn {layer, li}, {x_acc, c_acc} ->
        layer_forward(model, layer, x_acc, c_acc, li, start_pos)
      end)

    # Final RMSNorm.
    final_normed = rmsnorm(final_x, model.final_norm, model.config.rms_eps)
    {final_normed, cache_out}
  end

  defp layer_forward(model, layer, x, cache, layer_idx, start_pos) do
    %Config{n_heads: nh, n_kv_heads: nkv, head_dim: hd, hidden: d} = model.config
    {seq, _} = Nx.shape(x)

    # Pre-norm RMSNorm + QKV.
    x_norm = rmsnorm(x, layer.attn_norm, model.config.rms_eps)

    q_flat = matmul_quant_rows(x_norm, layer.w_q, nh * hd, seq)
    k_flat = matmul_quant_rows(x_norm, layer.w_k, nkv * hd, seq)
    v_flat = matmul_quant_rows(x_norm, layer.w_v, nkv * hd, seq)

    # Reshape into per-head: {seq, n_heads, head_dim} then transpose
    # to {n_heads, seq, head_dim} for attention dot.
    q = q_flat |> Nx.reshape({seq, nh, hd}) |> Nx.transpose(axes: [1, 0, 2])
    k = k_flat |> Nx.reshape({seq, nkv, hd}) |> Nx.transpose(axes: [1, 0, 2])
    v = v_flat |> Nx.reshape({seq, nkv, hd}) |> Nx.transpose(axes: [1, 0, 2])

    # RoPE.
    positions =
      Nx.tensor(Enum.map(0..(seq - 1), &(&1 + start_pos)), type: :s64)
      |> Nx.backend_copy(NxArm.Backend)

    q_roped = NxArm.LLM.rope(Nx.reshape(q, {1, seq, nh, hd}), positions, model.inv_freq)
                |> Nx.reshape({nh, seq, hd})

    k_roped = NxArm.LLM.rope(Nx.reshape(k, {1, seq, nkv, hd}), positions, model.inv_freq)
                |> Nx.reshape({nkv, seq, hd})

    # Push the whole (seq) of K and V into this layer's cache slot,
    # one timestep at a time. No more 22-layer zero-tensor allocation.
    cache =
      Enum.reduce(0..(seq - 1), cache, fn t, c_acc ->
        kt = Nx.slice(k_roped, [0, t, 0], [nkv, 1, hd])
        vt = Nx.slice(v,        [0, t, 0], [nkv, 1, hd])
        NxArm.KVCache.append_layer(c_acc, layer_idx, kt, vt)
      end)

    # On the last layer of this timestep, the caller advances the
    # cursor. We do it after every layer here to keep the API simple;
    # length will be the same after every layer for the same step.
    cache =
      if layer_idx == model.config.n_layers - 1 do
        # Advance once per timestep for all `seq` steps we wrote.
        Enum.reduce(1..seq, cache, fn _, c -> NxArm.KVCache.advance(c) end)
      else
        cache
      end

    # Cache "view length" — the prefix this layer's attention should
    # see is (existing length) + (this step's writes), regardless of
    # whether the cursor has bumped yet.
    view_length = cache.length + if(layer_idx == model.config.n_layers - 1, do: 0, else: seq)

    {k_all_layer, v_all_layer} =
      layer_view_with_length(cache, layer_idx, view_length, nkv, hd)

    # GQA expansion: each query head attends to floor(n_heads /
    # n_kv_heads) consecutive KV heads. We expand K and V along the
    # head dimension via Nx.broadcast over a reshape so one batched
    # matmul covers all heads.
    head_groups = div(nh, nkv)

    {k_expanded, v_expanded} =
      if head_groups == 1 do
        {k_all_layer, v_all_layer}
      else
        k_e = expand_kv_heads(k_all_layer, head_groups, nh, view_length, hd)
        v_e = expand_kv_heads(v_all_layer, head_groups, nh, view_length, hd)
        {k_e, v_e}
      end

    # Batched attention: one Nx.dot for Q@K^T across all heads.
    # Q: {nh, seq, hd}, K_expanded: {nh, view_length, hd}.
    # Result: {nh, seq, view_length}.
    scale = 1.0 / :math.sqrt(hd * 1.0)
    scores = Nx.dot(q_roped, [2], [0], k_expanded, [2], [0]) |> Nx.multiply(scale)
    scores_masked = apply_causal_mask_batched(scores, seq, view_length, start_pos, nh)

    m = Nx.reduce_max(scores_masked, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(scores_masked, m))
    s = Nx.sum(e, axes: [-1], keep_axes: true)
    attn = Nx.divide(e, s)

    # attn: {nh, seq, view_length}, V_expanded: {nh, view_length, hd}.
    # Result: {nh, seq, hd} → flatten heads → {seq, d}.
    attn_out =
      Nx.dot(attn, [2], [0], v_expanded, [1], [0])
      |> Nx.transpose(axes: [1, 0, 2])
      |> Nx.reshape({seq, d})

    o = matmul_quant_rows(attn_out, layer.w_o, d, seq)
    x1 = Nx.add(x, o)

    x1_norm = rmsnorm(x1, layer.ffn_norm, model.config.rms_eps)
    gate = matmul_quant_rows(x1_norm, layer.w_gate, model.config.ff_dim, seq)
    up = matmul_quant_rows(x1_norm, layer.w_up, model.config.ff_dim, seq)
    silu = Nx.multiply(gate, Nx.sigmoid(gate))
    fused = Nx.multiply(silu, up)
    down = matmul_quant_rows(fused, layer.w_down, d, seq)

    x_out = Nx.add(x1, down)
    {x_out, cache}
  end

  # Layer view, but force-overriding the visible length so attention
  # can read uncommitted writes from the current timestep.
  defp layer_view_with_length(cache, layer_idx, length, n_kv_heads, head_dim) do
    k = Map.fetch!(cache.k_layers, layer_idx)
    v = Map.fetch!(cache.v_layers, layer_idx)
    {Nx.slice(k, [0, 0, 0], [n_kv_heads, length, head_dim]),
     Nx.slice(v, [0, 0, 0], [n_kv_heads, length, head_dim])}
  end

  # Expand KV heads to match Q-head count. Each KV head broadcasts
  # to `head_groups` consecutive Q heads. Memory-cheap via broadcast.
  defp expand_kv_heads(kv, head_groups, n_heads, seq_len, head_dim) do
    n_kv = div(n_heads, head_groups)
    # kv: {n_kv, seq_len, head_dim} → {n_kv, 1, seq_len, head_dim}
    # broadcast → {n_kv, head_groups, seq_len, head_dim}
    # reshape → {n_heads, seq_len, head_dim}
    kv
    |> Nx.reshape({n_kv, 1, seq_len, head_dim})
    |> Nx.broadcast({n_kv, head_groups, seq_len, head_dim})
    |> Nx.reshape({n_heads, seq_len, head_dim})
  end

  defp apply_causal_mask_batched(scores, seq, kv_len, start_pos, _n_heads) do
    if seq == 1 do
      scores
    else
      i = Nx.iota({seq, 1}) |> Nx.add(start_pos)
      j = Nx.iota({1, kv_len})
      allowed = Nx.greater_equal(i, j)
      mask = Nx.select(allowed, Nx.tensor(0.0), Nx.tensor(-1.0e9))
      # mask: {seq, kv_len}; scores: {n_heads, seq, kv_len}.
      # Broadcast-add via reshape to {1, seq, kv_len}.
      Nx.add(scores, Nx.reshape(mask, {1, seq, kv_len}))
    end
  end

  defp rmsnorm(x, {:f32, gamma}, eps) do
    NxArm.LLM.rmsnorm(x, gamma, eps)
  end

  # ----------------------------------------------------------------
  # Matmul site: dispatches on the tensor's tag.
  #   x is f32 (rows, K). w is either {:f32, w_tensor} or {:q4_0, q}.
  #   Output (rows, N) f32.
  # ----------------------------------------------------------------
  defp matmul_quant_rows(x, w, n, rows) do
    case rows do
      1 -> matmul_quant_gemv(x, w, n)
      _ -> matmul_quant_gemm(x, w, n)
    end
  end

  defp matmul_quant_gemv(x, {:q4_0, q}, n) do
    x_bin = NxArm.Backend.__bin_of__(x)
    out_bin = NxArm.Native.int4_matmul_gemv_neon_op(x_bin, q.packed, q.scales, n, q.k)
    Nx.from_binary(out_bin, :f32) |> Nx.reshape({1, n}) |> Nx.backend_copy(NxArm.Backend)
  end

  defp matmul_quant_gemv(x, {:q8_0, q}, _n) do
    # Convert Q8_0 weight to f32 lazily and Nx.dot — gives us the
    # NEON GEMV from backend.dot for free. For lm_head this happens
    # once per token; reasonable for now. A dedicated Q8_0 GEMV
    # would skip the dequant copy but isn't critical at SmolLM size.
    w_f32 = q8_0_to_f32_tensor(q)
    Nx.dot(x, [1], w_f32, [1])
  end

  defp matmul_quant_gemv(x, {:f32, w}, _n) do
    Nx.dot(x, [1], w, [1])
  end

  defp matmul_quant_gemm(x, {:q4_0, q}, n) do
    {rows, _} = Nx.shape(x)
    x_bin = NxArm.Backend.__bin_of__(x)
    out_bin = NxArm.Native.int4_matmul_f32_op(x_bin, q.packed, q.scales, rows, n, q.k)
    Nx.from_binary(out_bin, :f32) |> Nx.reshape({rows, n}) |> Nx.backend_copy(NxArm.Backend)
  end

  defp matmul_quant_gemm(x, {:q8_0, q}, _n) do
    w_f32 = q8_0_to_f32_tensor(q)
    Nx.dot(x, [1], w_f32, [1])
  end

  defp matmul_quant_gemm(x, {:f32, w}, _n) do
    Nx.dot(x, [1], w, [1])
  end

  # ETS-memoised Q8_0→f32 dequantisation. Each weight matrix gets
  # dequantised at most once across the inference loop.
  defp q8_0_to_f32_tensor(%{n: n, k: k} = q) do
    cache_key = {:erlang.phash2(q.weights), n, k}

    case :persistent_term.get({:nx_arm_q8_cache, cache_key}, nil) do
      nil ->
        # Dequant: for each row, for each group, multiply 32 i8s by f32 scale.
        bin = q8_0_dequant_all(q)
        t = Nx.from_binary(bin, :f32) |> Nx.reshape({n, k}) |> Nx.backend_copy(NxArm.Backend)
        :persistent_term.put({:nx_arm_q8_cache, cache_key}, t)
        t

      cached ->
        cached
    end
  end

  defp q8_0_dequant_all(%{weights: w, scales: s, n: n, k: k}) do
    n_groups = div(k, 32)

    for row <- 0..(n - 1), into: <<>> do
      row_w = binary_part(w, row * k, k)
      row_s = binary_part(s, row * n_groups * 4, n_groups * 4)

      for g <- 0..(n_groups - 1), into: <<>> do
        group_w = binary_part(row_w, g * 32, 32)
        <<scale::float-32-little>> = binary_part(row_s, g * 4, 4)

        for <<v::signed-8 <- group_w>>, into: <<>> do
          <<v * scale::float-32-little>>
        end
      end
    end
  end

  defp project_lm(%__MODULE__{} = model, x_row) do
    matmul_quant_rows(x_row, model.lm_head, model.config.vocab, 1)
  end

  defp greedy(logits) do
    list =
      logits
      |> Nx.reshape({Nx.size(logits)})
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    list
    |> Enum.with_index()
    |> Enum.max_by(fn {v, _} -> v end)
    |> elem(1)
  end
end

defmodule NxArm.Bench.TinyLM do
  @moduledoc """
  Synthetic tiny transformer LM benchmark. Builds a deterministic-
  weight model (no training, no real tokenizer) and times prefill
  + decode on whatever backend is current.

  Defaults: 4 layers, d_model=128, 4 heads, d_ff=512, vocab=4000.
  ~3M params, ~12 MB at f32. Fits comfortably on a FP3 (~1 GB RAM).

      iex> NxArm.Bench.TinyLM.run()
      %{
        config: %{...},
        prefill_ms: 412.3,
        decode_ms_per_tok: 21.4,
        decode_total_ms: 343.0,
        tokens_per_sec: 46.8,
        param_count: 3_245_184
      }
  """

  defstruct [
    :n_layers,
    :d_model,
    :n_heads,
    :d_ff,
    :vocab,
    :seq_prefill,
    :n_decode,
    :head_dim,
    :embed,
    :layers,
    :out_proj
  ]

  @doc """
  Build the model with deterministic weights so two runs are
  comparable. `opts` overrides the defaults.
  """
  def build(opts \\ []) do
    n_layers = Keyword.get(opts, :n_layers, 4)
    d_model = Keyword.get(opts, :d_model, 128)
    n_heads = Keyword.get(opts, :n_heads, 4)
    d_ff = Keyword.get(opts, :d_ff, 512)
    vocab = Keyword.get(opts, :vocab, 4000)
    seq_prefill = Keyword.get(opts, :seq_prefill, 64)
    n_decode = Keyword.get(opts, :n_decode, 16)
    head_dim = div(d_model, n_heads)
    backend = Keyword.get(opts, :backend, NxArm.Backend)

    embed =
      Nx.iota({vocab, d_model}, type: :f32)
      |> Nx.divide(vocab * 1.0)
      |> Nx.sin()
      |> Nx.backend_copy(backend)

    layers =
      for layer_id <- 0..(n_layers - 1) do
        seed = layer_id * 1000

        %{
          norm1: weight({d_model}, seed + 1, backend),
          norm2: weight({d_model}, seed + 2, backend),
          w_q: weight({d_model, d_model}, seed + 3, backend),
          w_k: weight({d_model, d_model}, seed + 4, backend),
          w_v: weight({d_model, d_model}, seed + 5, backend),
          w_o: weight({d_model, d_model}, seed + 6, backend),
          w_gate: weight({d_model, d_ff}, seed + 7, backend),
          w_up: weight({d_model, d_ff}, seed + 8, backend),
          w_down: weight({d_ff, d_model}, seed + 9, backend)
        }
      end

    out_proj = weight({d_model, vocab}, 9999, backend)

    %__MODULE__{
      n_layers: n_layers,
      d_model: d_model,
      n_heads: n_heads,
      d_ff: d_ff,
      vocab: vocab,
      seq_prefill: seq_prefill,
      n_decode: n_decode,
      head_dim: head_dim,
      embed: embed,
      layers: layers,
      out_proj: out_proj
    }
  end

  defp weight(shape, seed, backend) do
    n = Tuple.product(shape)

    Nx.iota({n}, type: :f32)
    |> Nx.add(seed * 1.0)
    |> Nx.divide(1000)
    |> Nx.sin()
    |> Nx.reshape(shape)
    |> Nx.backend_copy(backend)
  end

  @doc "Run the benchmark and return a stats map."
  def run(opts \\ []) do
    model = build(opts)
    %{
      n_layers: nl,
      d_model: d,
      n_heads: nh,
      d_ff: dff,
      vocab: v,
      seq_prefill: sp,
      n_decode: nd
    } = model

    param_count =
      v * d +
        nl * (4 * d * d + 3 * d * dff + 2 * d) +
        d * v

    # Random-ish token ids for prefill.
    prompt = Enum.map(0..(sp - 1), &rem(&1 * 31 + 7, v))

    # Warm-up: one prefill + one decode (NIF first-call cost, JIT, page-in).
    _ = prefill(model, prompt)

    # Real measurement.
    {prefill_us, last_logits} = :timer.tc(fn -> prefill(model, prompt) end)

    {decode_us, _} =
      :timer.tc(fn ->
        decode_loop(model, last_logits, nd)
      end)

    decode_total_ms = decode_us / 1000.0
    decode_per_tok_ms = decode_total_ms / nd
    prefill_ms = prefill_us / 1000.0

    %{
      config: %{
        n_layers: nl,
        d_model: d,
        n_heads: nh,
        d_ff: dff,
        vocab: v,
        seq_prefill: sp,
        n_decode: nd
      },
      param_count: param_count,
      prefill_ms: Float.round(prefill_ms, 1),
      prefill_tokens_per_sec: Float.round(sp * 1000.0 / prefill_ms, 1),
      decode_total_ms: Float.round(decode_total_ms, 1),
      decode_ms_per_tok: Float.round(decode_per_tok_ms, 2),
      decode_tokens_per_sec: Float.round(1000.0 / decode_per_tok_ms, 1)
    }
  end

  # --- forward pieces ---

  defp prefill(model, prompt) do
    x = embed_tokens(model, prompt)
    mask = NxArm.LLM.causal_mask(model.seq_prefill)

    h =
      Enum.reduce(model.layers, x, fn ws, acc ->
        layer_forward(acc, ws, mask, model.n_heads)
      end)

    # Last position logits for sampling.
    last = Nx.slice(h, [model.seq_prefill - 1, 0], [1, model.d_model])
    Nx.dot(last, model.out_proj)
  end

  defp decode_loop(model, last_logits, n) do
    Enum.reduce(1..n, last_logits, fn _, logits ->
      tok = greedy(logits)
      x = embed_tokens(model, [tok])
      mask = NxArm.LLM.causal_mask(1)

      h =
        Enum.reduce(model.layers, x, fn ws, acc ->
          layer_forward(acc, ws, mask, model.n_heads)
        end)

      Nx.dot(h, model.out_proj)
    end)
  end

  defp greedy(logits) do
    # Avoid argmax fallback by going through to_flat_list.
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

  defp embed_tokens(model, token_ids) do
    # Avoid gather fallback by manually slicing each row and stacking
    # back into one tensor via concatenate.
    rows =
      for tok <- token_ids do
        Nx.slice(model.embed, [tok, 0], [1, model.d_model])
      end

    case rows do
      [single] -> single
      many -> Nx.concatenate(many, axis: 0)
    end
  end

  defp layer_forward(x, ws, mask, n_heads) do
    a = rmsnorm(x, ws.norm1)
    a = attention(a, ws.w_q, ws.w_k, ws.w_v, ws.w_o, mask, n_heads)
    x1 = Nx.add(x, a)
    f = rmsnorm(x1, ws.norm2)
    f = ffn(f, ws.w_gate, ws.w_up, ws.w_down)
    Nx.add(x1, f)
  end

  defp rmsnorm(x, gamma) do
    sq = Nx.multiply(x, x)
    mean = Nx.mean(sq, axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(mean, 1.0e-5))) |> Nx.multiply(gamma)
  end

  defp attention(x, w_q, w_k, w_v, w_o, mask, n_heads) do
    {seq, d_model} = Nx.shape(x)
    head_dim = div(d_model, n_heads)
    scale = 1.0 / :math.sqrt(head_dim * 1.0)

    q =
      Nx.dot(x, w_q)
      |> Nx.reshape({seq, n_heads, head_dim})
      |> Nx.transpose(axes: [1, 0, 2])

    k =
      Nx.dot(x, w_k)
      |> Nx.reshape({seq, n_heads, head_dim})
      |> Nx.transpose(axes: [1, 0, 2])

    v =
      Nx.dot(x, w_v)
      |> Nx.reshape({seq, n_heads, head_dim})
      |> Nx.transpose(axes: [1, 0, 2])

    qk = Nx.dot(q, [2], [0], k, [2], [0]) |> Nx.multiply(scale)

    qk_masked =
      case Nx.shape(qk) do
        {h, sq, sk} when sq == sk ->
          Nx.add(qk, Nx.broadcast(mask, {h, sq, sk}))

        _ ->
          qk
      end

    m = Nx.reduce_max(qk_masked, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(qk_masked, m))
    s = Nx.sum(e, axes: [-1], keep_axes: true)
    attn = Nx.divide(e, s)

    out = Nx.dot(attn, [2], [0], v, [1], [0])

    flat =
      out
      |> Nx.transpose(axes: [1, 0, 2])
      |> Nx.reshape({seq, d_model})

    Nx.dot(flat, w_o)
  end

  defp ffn(x, w_gate, w_up, w_down) do
    gate = Nx.dot(x, w_gate)
    up = Nx.dot(x, w_up)
    silu = Nx.multiply(gate, Nx.sigmoid(gate))
    fused = Nx.multiply(silu, up)
    Nx.dot(fused, w_down)
  end
end

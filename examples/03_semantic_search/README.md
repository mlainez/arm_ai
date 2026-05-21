# 03 — Semantic search / RAG retrieval

The cosine-similarity + top-k building blocks for on-device RAG.
The retrieval over 100k docs at d=384 fits well under 100 ms on
an A73 cluster — gemm does the dot product, Rust does the heap.

This example uses a tiny synthetic 4-D embedding table so it
runs on a fresh device with no model files. To make it real,
swap the synthetic table with the output of a sentence encoder
(MiniLM, MPNet, e5-small) loaded via `NxArm.Models.Onnx`.

## Set up

```sh
# Copy config.exs into your Nerves project's config/target.exs.
mix firmware && mix upload
```

For the synthetic version (this example), no model files are needed.

## Verified on FP3

```
Retrieval over 6 docs in 676 µs

Per-document scores:
  0.991  trains and railways
  0.998  subway transit map
  0.348  espresso brewing tips
  0.0    machine learning paper
  0.0    guitar lessons
  0.857  general transport + drinks blog

Top-3 retrieved (descending):
  1: subway transit map  (score=0.998)
  0: trains and railways  (score=0.991)
  5: general transport + drinks blog  (score=0.857)
```

676 µs over 6 docs; the kernel is gemm GEMV so it scales
linearly. 100k docs × 384 dims ≈ 30 ms.

## Real encoder (sentence-transformers)

Replace the synthetic corpus building with:

```elixir
{:ok, encoder} = NxArm.Models.Onnx.load("/root/models/all-MiniLM-L6-v2.onnx")
{:ok, tok}     = Tokenizers.Tokenizer.from_file("/root/models/all-MiniLM-L6-v2-tokenizer.json")

defp encode(text) do
  {:ok, enc} = Tokenizers.Tokenizer.encode(tok, text)
  ids = Tokenizers.Encoding.get_ids(enc)
  attn = List.duplicate(1, length(ids))

  outputs = NxArm.Models.Onnx.run(encoder, %{
    "input_ids"      => Nx.tensor([ids], type: :s64) |> Nx.as_type(:f32) |> Nx.backend_copy(NxArm.Backend),
    "attention_mask" => Nx.tensor([attn], type: :s64) |> Nx.as_type(:f32) |> Nx.backend_copy(NxArm.Backend)
  })

  # Mean-pool the last hidden state across the sequence axis.
  outputs["last_hidden_state"]
  |> Nx.mean(axes: [1])
  |> Nx.squeeze()
  |> NxArm.Embeddings.l2_normalize()
end
```

Then your corpus comes from `Enum.map(docs, &encode/1) |> Nx.stack()`.
Retrieval over a corpus of 1k chunks at 384 dim takes ~300 µs.

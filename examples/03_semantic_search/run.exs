#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 03 — Semantic search / RAG retrieval.
#
# Demonstrates the on-device retrieval primitives: L2 normalisation,
# cosine similarity over a corpus, and top-k.
#
# In a real RAG pipeline the embeddings come from a sentence-encoder
# (e.g. all-MiniLM-L6-v2 ONNX, ~90 MB). For this example we use a
# tiny synthetic embedding table so it runs in a couple of ms with
# zero model files — perfect for a copy-paste smoke test on a fresh
# device.
#
# To make this real:
#   * Add the sentence encoder to `config :nx_arm, :models, ...`.
#   * Replace `encode_query/1` and `encode_corpus/1` with
#     `ArmAI.Onnx.run/3` against the loaded model.
# ---------------------------------------------------------------

# Toy 4-D embedding table representing 6 short documents.
# In production these would come from a real encoder.
corpus = Nx.tensor([
  [0.9,  0.1, 0.0,  0.0],  # 0: "trains and railways"
  [0.85, 0.2, 0.05, 0.0],  # 1: "subway transit map"
  [0.1,  0.9, 0.0,  0.0],  # 2: "espresso brewing tips"
  [0.0,  0.0, 0.95, 0.05], # 3: "machine learning paper"
  [0.0,  0.0, 0.1,  0.9],  # 4: "guitar lessons"
  [0.5,  0.5, 0.0,  0.0]   # 5: "general transport + drinks blog"
]) |> Nx.backend_copy(NxArm.Backend)

doc_titles = [
  "trains and railways",
  "subway transit map",
  "espresso brewing tips",
  "machine learning paper",
  "guitar lessons",
  "general transport + drinks blog"
]

# L2-normalise once at index time.
corpus_normed = ArmAI.Embeddings.l2_normalize(corpus)

# Build a query — in real life this is the encoder output for a
# user's question. Here we hand-craft something close to docs 0+1.
query =
  Nx.tensor([0.8, 0.2, 0.0, 0.0])
  |> Nx.backend_copy(NxArm.Backend)
  |> ArmAI.Embeddings.l2_normalize()

# Score + retrieve.
{us, _} = :timer.tc(fn ->
  scores = ArmAI.Embeddings.cosine_similarity(query, corpus_normed)
  top3 = ArmAI.Embeddings.top_k(scores, 3)
  {scores, top3}
end)

scores = ArmAI.Embeddings.cosine_similarity(query, corpus_normed)
top3 = ArmAI.Embeddings.top_k(scores, 3)

IO.puts("Retrieval over #{length(doc_titles)} docs in #{us} µs")
IO.puts("")

scores_list = scores |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

IO.puts("Per-document scores:")
for {title, score} <- Enum.zip(doc_titles, scores_list) do
  IO.puts("  #{Float.round(score, 3)}  #{title}")
end

IO.puts("")
IO.puts("Top-3 retrieved (descending):")
for idx <- top3 do
  IO.puts("  #{idx}: #{Enum.at(doc_titles, idx)}  (score=#{Float.round(Enum.at(scores_list, idx), 3)})")
end

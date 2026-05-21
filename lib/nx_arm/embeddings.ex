defmodule NxArm.Embeddings do
  @moduledoc """
  On-device semantic search building blocks.

  The canonical RAG pipeline on Nerves:
    1. Encode documents into f32 embedding vectors (via Bumblebee,
       candle, or a tract-onnx sentence encoder).
    2. L2-normalise the corpus matrix once at index time.
    3. At query time: encode → L2-normalise → cosine similarity
       against the corpus → top-k indices.

  Cosine similarity collapses to a single GEMV under the hood (we
  hand it to `gemm`), and `top_k` is O(n log k) via a min-heap in
  Rust. The whole retrieval over 100k documents at d=384 fits well
  under 100 ms on a Cortex-A73 cluster.

      # Index time (run once, persist `corpus_norm`):
      corpus = encode_docs(...)                        # {n, d} f32
      corpus_norm = NxArm.Embeddings.l2_normalize(corpus)

      # Query time:
      q = encode_query("...")                          # {d} f32
      q_norm = NxArm.Embeddings.l2_normalize(q)
      scores = NxArm.Embeddings.cosine_similarity(q_norm, corpus_norm)
      top_5 = NxArm.Embeddings.top_k(scores, 5)
  """

  @doc """
  L2-normalise every row of an `(n, d)` matrix (or a 1-D `(d,)`
  vector). Output is the same shape, where each row has unit norm.
  """
  @spec l2_normalize(Nx.Tensor.t()) :: Nx.Tensor.t()
  def l2_normalize(t) do
    {n, d} =
      case Nx.shape(t) do
        {d} -> {1, d}
        {n, d} -> {n, d}
        other ->
          raise ArgumentError, "expected 1-D or 2-D tensor, got shape #{inspect(other)}"
      end

    bin = NxArm.Backend.__bin_of__(t)
    out_bin = NxArm.Native.l2_normalize_rows_f32_op(bin, n, d)

    Nx.from_binary(out_bin, :f32)
    |> Nx.reshape(Nx.shape(t))
    |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Cosine similarity between a query vector `q` (shape `{d}`) and
  every row of `corpus` (shape `{n, d}`). Returns `{n}` scores.

  **Both inputs must be L2-normalised first** (cosine sim is then
  a plain dot product, which is what this op computes).
  """
  @spec cosine_similarity(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def cosine_similarity(q, corpus) do
    {d_q} =
      case Nx.shape(q) do
        {d} -> {d}
        {1, d} -> {d}
        other -> raise ArgumentError, "query must be {d} or {1, d}, got #{inspect(other)}"
      end

    {n, d_c} =
      case Nx.shape(corpus) do
        {n, d} -> {n, d}
        other -> raise ArgumentError, "corpus must be {n, d}, got #{inspect(other)}"
      end

    if d_q != d_c do
      raise ArgumentError, "query d=#{d_q} ≠ corpus d=#{d_c}"
    end

    q_bin = NxArm.Backend.__bin_of__(q)
    c_bin = NxArm.Backend.__bin_of__(corpus)
    out_bin = NxArm.Native.cosine_similarity_f32_op(q_bin, c_bin, n, d_c)

    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Return the indices of the top-`k` highest values in `scores`,
  descending. O(n log k) via a min-heap in Rust.

  Returns a list of integers; pair with the caller's document IDs.
  """
  @spec top_k(Nx.Tensor.t(), pos_integer()) :: [non_neg_integer()]
  def top_k(scores, k) do
    bin = NxArm.Backend.__bin_of__(scores)
    NxArm.Native.top_k_indices_f32_op(bin, k)
  end
end

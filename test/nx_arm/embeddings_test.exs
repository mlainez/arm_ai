defmodule NxArm.EmbeddingsTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  test "l2_normalize gives unit-norm rows" do
    t = Nx.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, -5.0]])
    normed = NxArm.Embeddings.l2_normalize(arm(t))
    rows = normed |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list()
    for row <- rows do
      norm = row |> Enum.map(&(&1 * &1)) |> Enum.sum() |> :math.sqrt()
      assert_in_delta norm, 1.0, 1.0e-5
    end
  end

  test "cosine similarity after normalisation" do
    corpus = Nx.tensor([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ])

    corpus_n = NxArm.Embeddings.l2_normalize(arm(corpus))

    q = Nx.tensor([1.0, 1.0, 0.0])
    q_n = NxArm.Embeddings.l2_normalize(arm(q))

    scores = NxArm.Embeddings.cosine_similarity(q_n, corpus_n)
    list = scores |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

    # q == row 2 normalised → score 1.0
    assert_in_delta Enum.at(list, 2), 1.0, 1.0e-5
    # q is at 45° to row 0 / 1 → cos(45°) ≈ 0.707
    assert_in_delta Enum.at(list, 0), 0.7071, 1.0e-3
    assert_in_delta Enum.at(list, 1), 0.7071, 1.0e-3
    # q is orthogonal to row 3 → score 0
    assert_in_delta Enum.at(list, 3), 0.0, 1.0e-5
  end

  test "top_k returns indices in descending score order" do
    scores = Nx.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2]) |> arm()
    top3 = NxArm.Embeddings.top_k(scores, 3)
    assert top3 == [1, 3, 4]
  end

  test "end-to-end mini retrieval" do
    docs = Nx.tensor([
      [0.9, 0.1, 0.0],
      [0.1, 0.9, 0.0],
      [0.5, 0.5, 0.0],
      [0.0, 0.0, 1.0]
    ]) |> arm() |> NxArm.Embeddings.l2_normalize()

    query =
      Nx.tensor([0.8, 0.2, 0.0])
      |> arm()
      |> NxArm.Embeddings.l2_normalize()

    scores = NxArm.Embeddings.cosine_similarity(query, docs)
    top2 = NxArm.Embeddings.top_k(scores, 2)

    # Closest to the [0.9, 0.1, 0.0] direction are docs 0 then 2.
    assert top2 == [0, 2]
  end
end

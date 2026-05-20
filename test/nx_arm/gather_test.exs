defmodule NxArm.GatherTest do
  @moduledoc """
  Correctness tests for `NxArm.Backend.gather/4` against
  `Nx.BinaryBackend`. The fast path handles axes=[0..depth-1]
  (the common embedding-lookup pattern).
  """

  use NxArmCase, async: true

  describe "embedding-table lookup (axes=[0])" do
    test "1-D table, 1-D scalar indices" do
      table = Nx.tensor([10.0, 20.0, 30.0, 40.0])
      indices = Nx.tensor([[1], [3], [0]])

      assert_arm_matches_ref_n(
        fn t, i -> Nx.gather(t, i, axes: [0]) end,
        [table, indices]
      )
    end

    test "2-D table {V, H}, indices {N, 1} (the BERT-style token embedding)" do
      vocab = 50
      hidden = 8
      table = Nx.iota({vocab, hidden}, type: :f32) |> Nx.divide(10)
      indices = Nx.tensor([[0], [7], [49], [23]])

      assert_arm_matches_ref_n(
        fn t, i -> Nx.gather(t, i, axes: [0]) end,
        [table, indices]
      )
    end

    test "Nx.take is gather underneath (axis: 0)" do
      table = Nx.iota({30, 4}, type: :f32) |> Nx.divide(10)
      ids = Nx.tensor([[2, 5, 8], [11, 14, 17]])

      assert_arm_matches_ref_n(
        fn t, i -> Nx.take(t, i, axis: 0) end,
        [table, ids]
      )
    end

    test "f32 embedding table, larger shapes" do
      table = Nx.iota({256, 32}, type: :f32) |> Nx.divide(100)
      ids = Nx.iota({1, 16, 1}, type: :s64) |> Nx.remainder(256)

      assert_arm_matches_ref_n(
        fn t, i -> Nx.gather(t, i, axes: [0]) end,
        [table, ids]
      )
    end
  end

  describe "fallback to BinaryBackend" do
    test "non-prefix axes go through fallback" do
      # axes=[1] is non-prefix; backend should still produce
      # correct results via fallback.
      table = Nx.iota({4, 6}, type: :f32) |> Nx.divide(10)
      indices = Nx.tensor([[2], [5]])

      assert_arm_matches_ref_n(
        fn t, i -> Nx.gather(t, i, axes: [1]) end,
        [table, indices]
      )
    end
  end

  describe "realistic transformer-sized table" do
    @tag :slow
    test "BERT-base sized embedding table {30522, 768}, batch of 128 tokens" do
      # Use moderately sized hidden=64 so the test is fast on host;
      # the lookup is the part we care about, not the row width.
      table = Nx.iota({30522, 64}, type: :f32) |> Nx.divide(10000)
      indices = Nx.iota({1, 128, 1}, type: :s64) |> Nx.remainder(30522)

      assert_arm_matches_ref_n(
        fn t, i -> Nx.gather(t, i, axes: [0]) end,
        [table, indices]
      )
    end
  end
end

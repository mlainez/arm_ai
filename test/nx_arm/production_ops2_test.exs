defmodule NxArm.ProductionOps2Test do
  @moduledoc """
  Phase 2 op coverage: sort, argsort, all, any, product, reverse.
  Each was on the BinaryBackend fallback path until this commit.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  # --- sort ---

  test "sort 1-D ascending" do
    t = Nx.tensor([3.0, 1.0, 4.0, 1.5, 9.0])
    got = Nx.sort(arm(t))
    assert Nx.to_flat_list(got) == [1.0, 1.5, 3.0, 4.0, 9.0]
  end

  test "sort 1-D descending" do
    t = Nx.tensor([3.0, 1.0, 4.0, 1.5, 9.0])
    got = Nx.sort(arm(t), direction: :desc)
    assert Nx.to_flat_list(got) == [9.0, 4.0, 3.0, 1.5, 1.0]
  end

  test "sort 2-D along last axis" do
    t = Nx.tensor([[3.0, 1.0, 4.0], [9.0, 2.0, 6.0]])
    got = Nx.sort(arm(t), axis: -1)
    assert Nx.to_flat_list(got) == [1.0, 3.0, 4.0, 2.0, 6.0, 9.0]
  end

  test "sort 2-D along axis 0 falls back to BinaryBackend (correct)" do
    t = Nx.tensor([[5.0, 1.0], [3.0, 4.0], [1.0, 2.0]])
    got = Nx.sort(arm(t), axis: 0)
    ref = Nx.sort(t, axis: 0)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  # --- argsort ---

  test "argsort 1-D ascending" do
    t = Nx.tensor([3.0, 1.0, 4.0, 1.5, 9.0])
    got = Nx.argsort(arm(t))
    assert Nx.to_flat_list(got) == [1, 3, 0, 2, 4]
  end

  test "argsort descending = top-k indices for sampling" do
    logits = Nx.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
    got = Nx.argsort(arm(logits), direction: :desc)
    assert Nx.to_flat_list(got) == [3, 1, 2, 4, 0]

    # Top-3 indices.
    top3 = Nx.slice(got, [0], [3]) |> Nx.to_flat_list()
    assert top3 == [3, 1, 2]
  end

  test "argsort 2-D along last axis" do
    t = Nx.tensor([[3.0, 1.0, 4.0], [9.0, 2.0, 6.0]])
    got = Nx.argsort(arm(t), axis: -1)
    assert Nx.to_flat_list(got) == [1, 0, 2, 1, 2, 0]
  end

  # --- all / any / product ---

  test "all on all-true bool tensor" do
    t = Nx.tensor([1, 1, 1, 1], type: :u8)
    got = Nx.all(arm(t))
    assert Nx.to_number(got) == 1
  end

  test "all returns 0 if any element is zero" do
    t = Nx.tensor([1, 1, 0, 1], type: :u8)
    got = Nx.all(arm(t))
    assert Nx.to_number(got) == 0
  end

  test "any returns 1 if at least one element is non-zero" do
    t = Nx.tensor([0, 0, 1, 0], type: :u8)
    got = Nx.any(arm(t))
    assert Nx.to_number(got) == 1
  end

  test "any returns 0 on all-zeros" do
    t = Nx.tensor([0, 0, 0, 0], type: :u8)
    got = Nx.any(arm(t))
    assert Nx.to_number(got) == 0
  end

  test "all over f32 tensor (non-zero check)" do
    t = Nx.tensor([1.5, 2.5, 3.0])
    got = Nx.all(arm(t))
    assert Nx.to_number(got) == 1
  end

  test "product over f32 tensor" do
    t = Nx.tensor([2.0, 3.0, 4.0])
    got = Nx.product(arm(t))
    assert_in_delta Nx.to_number(got), 24.0, 1.0e-6
  end

  # --- reverse ---

  test "reverse 1-D" do
    t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    got = Nx.reverse(arm(t))
    assert Nx.to_flat_list(got) == [4.0, 3.0, 2.0, 1.0]
  end

  test "reverse 2-D along all axes" do
    t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = Nx.reverse(arm(t))
    assert Nx.to_flat_list(got) == [4.0, 3.0, 2.0, 1.0]
  end

  test "reverse 2-D along single axis" do
    t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = Nx.reverse(arm(t), axes: [1])
    assert Nx.to_flat_list(got) == [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
  end

  # --- integration: top-k sampling ---

  test "top-k sampling end-to-end via argsort" do
    logits = Nx.tensor([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4]) |> arm()

    sorted_idx = Nx.argsort(logits, direction: :desc)
    top_k = Nx.slice(sorted_idx, [0], [3])

    assert Nx.to_flat_list(top_k) == [3, 5, 1]
  end
end

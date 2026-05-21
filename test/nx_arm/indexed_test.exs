defmodule NxArm.IndexedTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  test "indexed_add accumulates at repeated indices" do
    t = Nx.iota({5}, type: :f32) |> arm()                # [0,1,2,3,4]
    indices = Nx.tensor([[1], [1], [3]], type: :s64) |> arm()
    updates = Nx.tensor([10.0, 100.0, 5.0]) |> arm()

    got = Nx.indexed_add(t, indices, updates) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.to_flat_list(got) == [0.0, 111.0, 2.0, 8.0, 4.0]
  end

  test "indexed_put overwrites at indices (last-write-wins)" do
    t = Nx.broadcast(0.0, {4}) |> arm()
    indices = Nx.tensor([[0], [2], [2]], type: :s64) |> arm()
    updates = Nx.tensor([1.0, 2.0, 99.0]) |> arm()

    got = Nx.indexed_put(t, indices, updates) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.to_flat_list(got) == [1.0, 0.0, 99.0, 0.0]
  end
end

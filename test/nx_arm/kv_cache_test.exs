defmodule NxArm.KVCacheTest do
  use ExUnit.Case, async: true

  test "new + append + prefix" do
    c = NxArm.KVCache.new(2, 3, 8, 4)

    assert c.length == 0
    assert_raise ArgumentError, fn -> NxArm.KVCache.prefix(c) end

    k_step = Nx.iota({2, 3, 1, 4}, type: :f32) |> Nx.divide(10)
    v_step = Nx.iota({2, 3, 1, 4}, type: :f32) |> Nx.add(100) |> Nx.divide(10)

    c1 = NxArm.KVCache.append(c, k_step, v_step)
    assert c1.length == 1

    {k1, v1} = NxArm.KVCache.prefix(c1)
    assert Nx.shape(k1) == {2, 3, 1, 4}
    assert Nx.to_flat_list(k1) == Nx.to_flat_list(k_step)
    assert Nx.to_flat_list(v1) == Nx.to_flat_list(v_step)
  end

  test "append two steps preserves history" do
    c = NxArm.KVCache.new(1, 2, 8, 3)
    s1_k = Nx.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
    s1_v = Nx.tensor([[[[10.0, 20.0, 30.0]], [[40.0, 50.0, 60.0]]]])
    s2_k = Nx.tensor([[[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]]])
    s2_v = Nx.tensor([[[[70.0, 80.0, 90.0]], [[100.0, 110.0, 120.0]]]])

    c =
      c
      |> NxArm.KVCache.append(s1_k, s1_v)
      |> NxArm.KVCache.append(s2_k, s2_v)

    assert c.length == 2

    {k, v} = NxArm.KVCache.prefix(c)
    assert Nx.shape(k) == {1, 2, 2, 3}

    # Layer 0, head 0, positions 0 and 1 should hold s1_k row, s2_k row.
    assert Nx.to_flat_list(k[[0, 0, 0]]) == [1.0, 2.0, 3.0]
    assert Nx.to_flat_list(k[[0, 0, 1]]) == [7.0, 8.0, 9.0]
    assert Nx.to_flat_list(v[[0, 1, 1]]) == [100.0, 110.0, 120.0]
  end

  test "layer/2 returns one layer's K/V" do
    c = NxArm.KVCache.new(3, 2, 4, 2)
    k_step = Nx.iota({3, 2, 1, 2}, type: :f32)
    v_step = Nx.iota({3, 2, 1, 2}, type: :f32) |> Nx.add(100)
    c = NxArm.KVCache.append(c, k_step, v_step)

    {k_l1, v_l1} = NxArm.KVCache.layer(c, 1)
    assert Nx.shape(k_l1) == {2, 1, 2}
    # Layer 1 starts at offset (n_heads * head_dim) = 2*2 = 4 in iota.
    assert Nx.to_flat_list(k_l1) == [4.0, 5.0, 6.0, 7.0]
    assert Nx.to_flat_list(v_l1) == [104.0, 105.0, 106.0, 107.0]
  end

  test "append past max_seq raises" do
    c = NxArm.KVCache.new(1, 1, 1, 2)
    step = Nx.tensor([[[[1.0, 2.0]]]])
    c1 = NxArm.KVCache.append(c, step, step)
    assert_raise ArgumentError, fn -> NxArm.KVCache.append(c1, step, step) end
  end

  test "reset moves cursor without clearing storage" do
    c = NxArm.KVCache.new(1, 1, 4, 2)
    step = Nx.tensor([[[[1.0, 2.0]]]])
    c = NxArm.KVCache.append(c, step, step) |> NxArm.KVCache.reset()
    assert c.length == 0

    # After reset, prefix raises (empty) but a fresh append works.
    new_step = Nx.tensor([[[[3.0, 4.0]]]])
    c = NxArm.KVCache.append(c, new_step, new_step)
    {k, _} = NxArm.KVCache.prefix(c)
    assert Nx.shape(k) == {1, 1, 1, 2}
    assert Nx.to_flat_list(k) == [3.0, 4.0]
  end
end

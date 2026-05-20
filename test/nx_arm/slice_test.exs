defmodule NxArm.SliceTest do
  use NxArmCase, async: true

  describe "Nx.slice" do
    test "1-D unit-stride slice" do
      t = Nx.tensor([10, 20, 30, 40, 50, 60, 70], type: :f32)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [2], [4]) end, t)
    end

    test "2-D unit-stride slice, full rows" do
      t = Nx.iota({5, 4}, type: :f32)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [1, 0], [3, 4]) end, t)
    end

    test "2-D unit-stride slice, partial rows" do
      t = Nx.iota({5, 6}, type: :f32)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [1, 2], [3, 3]) end, t)
    end

    test "3-D KV-cache style slice (full inner axes)" do
      # cache: {batch, heads, max_seq, head_dim}, take current prefix
      cache = Nx.iota({1, 4, 16, 8}, type: :f32) |> Nx.divide(100)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [0, 0, 0, 0], [1, 4, 5, 8]) end, cache)
    end

    test "strided slice" do
      t = Nx.iota({10}, type: :f32)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [0], [5], strides: [2]) end, t)
    end

    test "4-D slice covering middle of all axes" do
      t = Nx.iota({3, 4, 5, 6}, type: :f32) |> Nx.divide(10)
      assert_arm_matches_ref(fn x -> Nx.slice(x, [1, 1, 1, 1], [1, 2, 3, 4]) end, t)
    end
  end

  describe "Nx.put_slice" do
    test "2-D put_slice writes a sub-matrix" do
      t = Nx.iota({4, 6}, type: :f32)
      patch = Nx.broadcast(Nx.tensor(99.0, type: :f32), {2, 3})
      assert_arm_matches_ref_n(fn t, p -> Nx.put_slice(t, [1, 1], p) end, [t, patch])
    end

    test "3-D KV-cache append-style write" do
      cache = Nx.broadcast(Nx.tensor(0.0, type: :f32), {1, 4, 8, 8})
      new_kv = Nx.iota({1, 4, 2, 8}, type: :f32) |> Nx.divide(100)
      # Write 2 new tokens at position 3..4
      assert_arm_matches_ref_n(
        fn c, k -> Nx.put_slice(c, [0, 0, 3, 0], k) end,
        [cache, new_kv]
      )
    end

    test "1-D put_slice at offset" do
      t = Nx.iota({10}, type: :f32)
      patch = Nx.tensor([100.0, 200.0, 300.0])
      assert_arm_matches_ref_n(fn t, p -> Nx.put_slice(t, [4], p) end, [t, patch])
    end
  end
end

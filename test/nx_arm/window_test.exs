defmodule NxArm.WindowTest do
  use NxArmCase, async: true

  describe "window_max (max_pool primitive)" do
    test "2x2 stride 1" do
      t = Nx.iota({1, 4, 4, 1}, type: :f32)

      assert_arm_matches_ref(
        fn x -> Nx.window_max(x, {1, 2, 2, 1}, strides: [1, 1, 1, 1], padding: [{0, 0}, {0, 0}, {0, 0}, {0, 0}]) end,
        t
      )
    end

    test "2x2 stride 2" do
      t = Nx.iota({1, 4, 4, 3}, type: :f32)

      assert_arm_matches_ref(
        fn x -> Nx.window_max(x, {1, 2, 2, 1}, strides: [1, 2, 2, 1], padding: [{0, 0}, {0, 0}, {0, 0}, {0, 0}]) end,
        t
      )
    end

    test "3x3 with padding" do
      t = Nx.iota({1, 5, 5, 2}, type: :f32)

      assert_arm_matches_ref(
        fn x -> Nx.window_max(x, {1, 3, 3, 1}, strides: [1, 1, 1, 1], padding: [{0, 0}, {1, 1}, {1, 1}, {0, 0}]) end,
        t
      )
    end
  end

  describe "window_sum (avg_pool primitive)" do
    test "2x2 stride 2" do
      t = Nx.iota({1, 4, 4, 1}, type: :f32)

      assert_arm_matches_ref(
        fn x -> Nx.window_sum(x, {1, 2, 2, 1}, strides: [1, 2, 2, 1], padding: [{0, 0}, {0, 0}, {0, 0}, {0, 0}]) end,
        t
      )
    end

    test "3x3 same-pad" do
      t = Nx.iota({1, 5, 5, 2}, type: :f32) |> Nx.divide(10)

      assert_arm_matches_ref(
        fn x -> Nx.window_sum(x, {1, 3, 3, 1}, strides: [1, 1, 1, 1], padding: [{0, 0}, {1, 1}, {1, 1}, {0, 0}]) end,
        t,
        tol: 1.0e-4
      )
    end
  end

  describe "window_min" do
    test "2x2 stride 1" do
      t = Nx.iota({1, 4, 4, 1}, type: :f32) |> Nx.subtract(10)

      assert_arm_matches_ref(
        fn x -> Nx.window_min(x, {1, 2, 2, 1}, strides: [1, 1, 1, 1], padding: [{0, 0}, {0, 0}, {0, 0}, {0, 0}]) end,
        t
      )
    end
  end

  describe "max_pool decomposition" do
    test "Axon-style max_pool (2x2 valid)" do
      t = Nx.iota({1, 8, 8, 4}, type: :f32) |> Nx.divide(100)

      assert_arm_matches_ref(
        fn x ->
          Axon.Layers.max_pool(x, kernel_size: {2, 2}, strides: [2, 2], padding: :valid, channels: :last)
        end,
        t,
        tol: 1.0e-4
      )
    end

    test "Axon-style avg_pool (3x3 same)" do
      t = Nx.iota({1, 8, 8, 4}, type: :f32) |> Nx.divide(100)

      assert_arm_matches_ref(
        fn x ->
          Axon.Layers.avg_pool(x, kernel_size: {3, 3}, strides: [1, 1], padding: :same, channels: :last)
        end,
        t,
        tol: 1.0e-4
      )
    end
  end
end

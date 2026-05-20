defmodule NxArm.DepthwiseConvTest do
  @moduledoc """
  Depthwise convolution correctness vs Nx.BinaryBackend.
  In Nx parlance, depthwise = `feature_group_size == Cin`.
  """

  use NxArmCase, async: true

  describe "depthwise conv" do
    test "3x3 stride 1 valid padding (basic)" do
      # NHWC input — Nx.conv expects input_permutation to map.
      input = Nx.iota({1, 8, 8, 4}, type: :f32) |> Nx.divide(10)
      # Kernel OIHW = {Cin, 1, Kh, Kw} for depthwise.
      kernel = Nx.iota({4, 1, 3, 3}, type: :f32) |> Nx.divide(10)

      assert_arm_matches_ref_n(
        fn i, k ->
          Nx.conv(i, k,
            strides: 1,
            padding: :valid,
            input_permutation: [0, 3, 1, 2],
            kernel_permutation: [0, 1, 2, 3],
            output_permutation: [0, 3, 1, 2],
            feature_group_size: 4
          )
        end,
        [input, kernel],
        tol: 1.0e-3
      )
    end

    test "MobileNet-style 3x3 stride 1 SAME padding, 16 channels" do
      input = Nx.iota({1, 12, 12, 16}, type: :f32) |> Nx.divide(100)
      kernel = Nx.iota({16, 1, 3, 3}, type: :f32) |> Nx.divide(10)

      assert_arm_matches_ref_n(
        fn i, k ->
          Nx.conv(i, k,
            strides: 1,
            padding: [{1, 1}, {1, 1}],
            input_permutation: [0, 3, 1, 2],
            kernel_permutation: [0, 1, 2, 3],
            output_permutation: [0, 3, 1, 2],
            feature_group_size: 16
          )
        end,
        [input, kernel],
        tol: 1.0e-3
      )
    end

    test "stride 2" do
      input = Nx.iota({1, 16, 16, 8}, type: :f32) |> Nx.divide(100)
      kernel = Nx.iota({8, 1, 3, 3}, type: :f32) |> Nx.divide(10)

      assert_arm_matches_ref_n(
        fn i, k ->
          Nx.conv(i, k,
            strides: 2,
            padding: [{1, 1}, {1, 1}],
            input_permutation: [0, 3, 1, 2],
            kernel_permutation: [0, 1, 2, 3],
            output_permutation: [0, 3, 1, 2],
            feature_group_size: 8
          )
        end,
        [input, kernel],
        tol: 1.0e-3
      )
    end

    test "5x5 kernel" do
      input = Nx.iota({1, 10, 10, 4}, type: :f32) |> Nx.divide(100)
      kernel = Nx.iota({4, 1, 5, 5}, type: :f32) |> Nx.divide(10)

      assert_arm_matches_ref_n(
        fn i, k ->
          Nx.conv(i, k,
            strides: 1,
            padding: :valid,
            input_permutation: [0, 3, 1, 2],
            kernel_permutation: [0, 1, 2, 3],
            output_permutation: [0, 3, 1, 2],
            feature_group_size: 4
          )
        end,
        [input, kernel],
        tol: 1.0e-3
      )
    end
  end
end

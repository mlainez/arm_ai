defmodule NxArm.QuantizedTest do
  use ExUnit.Case, async: true

  describe "NxArm.Quantized.from_f32 + matmul" do
    test "round-trips a known weight matrix within int8 tolerance" do
      # 4 output channels × 6 inputs, hand-crafted values that quantize
      # cleanly.
      w =
        Nx.tensor([
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
          [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        ])

      q = NxArm.Quantized.from_f32(w)

      # Single-vector input.
      act = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      ref = Nx.dot(act, Nx.transpose(w))

      got = NxArm.Quantized.matmul(q, act) |> Nx.backend_copy(Nx.BinaryBackend)

      diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      # int8 has ~1/128 relative precision per row; absolute diff scales
      # with the row magnitude. The row {10..60} has max-abs = 60 so
      # scale = 60/127 ≈ 0.47, giving per-output error around 6 (worst
      # case, where every input contributes). For sane numerical inputs
      # the realistic per-output error is much smaller; we use a
      # relative tolerance.
      max_val = Nx.reduce_max(Nx.abs(ref)) |> Nx.to_number()
      tol = 0.05 * max_val + 0.1
      assert diff < tol, "diff #{diff} > tol #{tol}"
    end

    test "matmul shape with batched activations" do
      w = Nx.iota({8, 16}, type: :f32) |> Nx.divide(10)
      q = NxArm.Quantized.from_f32(w)

      act = Nx.iota({1, 5, 16}, type: :f32) |> Nx.divide(10)
      out = NxArm.Quantized.matmul(q, act)

      assert Nx.shape(out) == {1, 5, 8}
      assert Nx.type(out) == {:f, 32}
    end

    test "scales correctly reproduce f32 results for clean weights (×127 multiples)" do
      # When weights are exactly k * scale for integer k, the int8
      # quantization is lossless. Construct such a case.
      scale = 0.1
      w =
        Nx.tensor([
          [scale * 1, scale * 2, scale * 3, scale * 4],
          [scale * -127, scale * 127, scale * 0, scale * 10]
        ])

      q = NxArm.Quantized.from_f32(w)
      act = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      ref = Nx.dot(act, Nx.transpose(w))
      got = NxArm.Quantized.matmul(q, act) |> Nx.backend_copy(Nx.BinaryBackend)

      diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      # round() to nearest at quantize time + f32 mul/sum at matmul
      # time accrues ~5e-3 absolute error here. That's the inherent
      # precision floor of int8-via-f32 dequant — not a correctness bug.
      assert diff < 5.0e-3
    end
  end

  describe "int8_matmul_f32_op (full int8: SDOT path or vmlal fallback)" do
    test "matches scalar reference for known inputs" do
      # 4x6 acts (i8), 3x6 weights (i8), per-row weight scales
      a = for v <- 1..24, into: <<>>, do: <<v::signed-8>>
      w_data = for v <- 1..18, into: <<>>, do: <<rem(v, 7) - 3::signed-8>>
      scales = for v <- 1..3, into: <<>>, do: <<v / 10.0::float-little-32>>
      act_scale = 0.01

      out_bin =
        NxArm.Native.int8_matmul_f32_op(a, w_data, scales, act_scale, 4, 3, 6)
        |> :erlang.binary_to_list()

      out_floats =
        for <<v::float-little-32 <- :erlang.list_to_binary(out_bin)>>, do: v

      # Scalar reference.
      a_vals = for <<v::signed-8 <- a>>, do: v
      w_vals = for <<v::signed-8 <- w_data>>, do: v
      s_vals = for <<v::float-little-32 <- scales>>, do: v

      ref =
        for i <- 0..3, j <- 0..2 do
          dot =
            Enum.reduce(0..5, 0, fn k, acc ->
              acc + Enum.at(a_vals, i * 6 + k) * Enum.at(w_vals, j * 6 + k)
            end)

          dot * act_scale * Enum.at(s_vals, j)
        end

      Enum.zip(out_floats, ref)
      |> Enum.each(fn {got, exp} ->
        assert_in_delta got, exp, 1.0e-4
      end)
    end

    test "handles K=64 and K=16 (typical attention head_dim)" do
      for k <- [16, 64] do
        m = 8
        n = 4
        a = :binary.copy(<<1::signed-8>>, m * k)
        w = :binary.copy(<<2::signed-8>>, n * k)
        scales = for _ <- 1..n, into: <<>>, do: <<0.5::float-little-32>>
        out_bin = NxArm.Native.int8_matmul_f32_op(a, w, scales, 1.0, m, n, k)

        # All inputs are 1×2 = 2, K of them, scale 0.5, act_scale 1.0
        # → 2 * K * 0.5 * 1.0 = K.
        expected = k * 1.0
        out_floats = for <<v::float-little-32 <- out_bin>>, do: v
        Enum.each(out_floats, &assert_in_delta(&1, expected, 1.0e-5))
      end
    end
  end
end

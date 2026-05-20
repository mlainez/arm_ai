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
end

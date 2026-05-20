defmodule NxArm.Fp16Test do
  use ExUnit.Case, async: true

  describe "f32_to_f16_op" do
    test "round-trips simple values via the NEON convert path" do
      values = [0.0, 1.0, -1.0, 2.5, -2.5, 100.0, 0.5, -0.5]
      f32_bin = for v <- values, into: <<>>, do: <<v::float-little-32>>
      f16_bin = NxArm.Native.f32_to_f16_op(f32_bin)

      assert byte_size(f16_bin) == length(values) * 2

      # Verify by running fp16 matmul against an identity-ish input.
      # Build a 1x8 act = values, weights = 1.0 → out = sum(values) per row.
      act_bin = f32_bin
      w_f32 = for _ <- values, into: <<>>, do: <<1.0::float-little-32>>
      w_f16 = NxArm.Native.f32_to_f16_op(w_f32)

      out_bin =
        NxArm.Native.dequant_matmul_f16_f32_op(act_bin, w_f16, 1, 1, 1, length(values))

      [sum] = for <<v::float-little-32 <- out_bin>>, do: v
      expected = Enum.sum(values)
      assert_in_delta sum, expected, 1.0e-3
    end
  end

  describe "fp16 matmul" do
    test "matches f32 reference within fp16 precision" do
      # Random-ish input matrices.
      m = 4
      k = 16
      n = 8

      act_f32 =
        for i <- 0..(m * k - 1), into: <<>>,
            do: <<:math.sin(i * 0.1)::float-little-32>>

      w_f32 =
        for i <- 0..(n * k - 1), into: <<>>,
            do: <<:math.cos(i * 0.1)::float-little-32>>

      w_f16 = NxArm.Native.f32_to_f16_op(w_f32)

      out_bin = NxArm.Native.dequant_matmul_f16_f32_op(act_f32, w_f16, 1, m, n, k)
      out = for <<v::float-little-32 <- out_bin>>, do: v

      # Reference: act @ w^T in f32.
      a_vals = for <<v::float-little-32 <- act_f32>>, do: v
      w_vals = for <<v::float-little-32 <- w_f32>>, do: v

      ref =
        for i <- 0..(m - 1), j <- 0..(n - 1) do
          Enum.reduce(0..(k - 1), 0.0, fn kk, acc ->
            acc + Enum.at(a_vals, i * k + kk) * Enum.at(w_vals, j * k + kk)
          end)
        end

      Enum.zip(out, ref)
      |> Enum.each(fn {got, exp} ->
        # fp16 mantissa is 10 bits → relative precision ~1e-3.
        rel_err = abs(got - exp) / (abs(exp) + 1.0e-6)
        assert rel_err < 1.0e-2, "rel_err #{rel_err}, got #{got}, exp #{exp}"
      end)
    end
  end
end

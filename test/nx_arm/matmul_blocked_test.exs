defmodule NxArm.MatmulBlockedTest do
  @moduledoc """
  Verify the cache-blocked matmul path produces identical results to
  the reference for shapes that trigger it (K > 256).
  """

  use NxArmCase, async: true

  # Use sin-based small inputs so accumulator magnitudes stay bounded
  # and f32 round-off doesn't dominate the absolute-diff tolerance.
  defp small(shape) do
    Nx.iota(shape, type: :f32)
    |> Nx.multiply(0.001)
    |> Nx.sin()
  end

  describe "cache-blocked matmul (K > 256)" do
    test "K=768 N=192 (MLP-down shape)" do
      a = small({197, 768})
      b = small({768, 192})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-3)
    end

    test "K=512 N=256" do
      a = small({64, 512})
      b = small({512, 256})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-3)
    end

    test "M tail (m=199 doesn't divide 4)" do
      a = small({199, 384})
      b = small({384, 64})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-3)
    end

    test "N tail (n=23 doesn't divide 8)" do
      a = small({32, 384})
      b = small({384, 23})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-3)
    end
  end

  describe "unblocked path (K ≤ 256) still works" do
    test "K=192 (ViT linear shape)" do
      a = small({197, 192})
      b = small({192, 192})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-4)
    end

    test "K=64 (per-head dim)" do
      a = small({16, 64})
      b = small({64, 16})
      assert_arm_matches_ref_n(&Nx.dot/2, [a, b], tol: 1.0e-5)
    end
  end
end

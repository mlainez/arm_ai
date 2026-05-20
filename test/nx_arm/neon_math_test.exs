defmodule NxArm.NeonMathTest do
  @moduledoc """
  Correctness of NEON-vectorised exp / sigmoid / tanh against the
  scalar libm reference. Tolerance accounts for the polynomial
  approximation (range-reduced degree-5 Taylor) vs libm's accurate
  exp. ~1 ULP across the safe domain.
  """

  use NxArmCase, async: true

  describe "exp" do
    test "small inputs match libm" do
      x = Nx.tensor([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0])
      assert_arm_matches_ref(&Nx.exp/1, x, tol: 1.0e-5)
    end

    test "wider range" do
      x = Nx.tensor([-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0])
      assert_arm_matches_ref(&Nx.exp/1, x, tol: 1.0e-3)
    end

    test "vector-length sweep stresses NEON tail handling" do
      for n <- 1..16 do
        x = Nx.iota({n}, type: :f32) |> Nx.subtract(Nx.divide(Nx.tensor(n, type: :f32), 2))
        assert_arm_matches_ref(&Nx.exp/1, x, tol: 1.0e-4)
      end
    end

    test "large 2-D shape" do
      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(50)
      assert_arm_matches_ref(&Nx.exp/1, x, tol: 1.0e-3)
    end
  end

  describe "sigmoid" do
    test "matches reference across saturating range" do
      x = Nx.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
      assert_arm_matches_ref(&Nx.sigmoid/1, x, tol: 1.0e-5)
    end

    test "vector tail correctness" do
      for n <- 1..16 do
        x = Nx.iota({n}, type: :f32) |> Nx.subtract(Nx.divide(Nx.tensor(n, type: :f32), 2))
        assert_arm_matches_ref(&Nx.sigmoid/1, x, tol: 1.0e-5)
      end
    end
  end

  describe "tanh" do
    test "matches reference" do
      x = Nx.tensor([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
      assert_arm_matches_ref(&Nx.tanh/1, x, tol: 1.0e-5)
    end

    test "approaches ±1 at saturating ends" do
      x = Nx.tensor([100.0, -100.0])
      out = Nx.tanh(Nx.backend_copy(x, NxArm.Backend)) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
      [pos, neg] = out
      assert_in_delta pos, 1.0, 1.0e-6
      assert_in_delta neg, -1.0, 1.0e-6
    end
  end

  describe "softmax (uses NEON exp internally)" do
    test "matches reference at attention scale" do
      x = Nx.iota({1, 3, 32, 32}, type: :f32) |> Nx.divide(100)

      assert_arm_matches_ref(
        fn t ->
          NxArm.softmax(t)
        end,
        x,
        tol: 1.0e-5
      )
    end

    test "rows sum to 1" do
      x = Nx.iota({4, 16}, type: :f32) |> Nx.divide(10)
      arm_x = Nx.backend_copy(x, NxArm.Backend)
      out = NxArm.softmax(arm_x) |> Nx.backend_copy(Nx.BinaryBackend)
      sums = Nx.sum(out, axes: [-1]) |> Nx.to_flat_list()

      for s <- sums do
        assert_in_delta s, 1.0, 1.0e-5
      end
    end
  end
end

defmodule NxArm.ConformanceTest do
  @moduledoc """
  Cross-backend conformance: every op exercised here is computed on
  both NxArm.Backend and Nx.BinaryBackend (the reference) and the
  results compared element-wise. Uses StreamData to fuzz shapes and
  values so we hit corners the hand-written tests miss.

  Tolerances are deliberately loose to absorb f32 FMA reorder noise
  on the NEON path, but tight enough to catch any actual computation
  bug.
  """

  use ExUnit.Case, async: true
  use ExUnitProperties

  @tol 1.0e-4

  # ------- shape generators -------

  defp small_dim, do: StreamData.integer(1..6)
  defp medium_dim, do: StreamData.integer(1..16)

  defp shape_1d, do: StreamData.list_of(small_dim(), length: 1) |> StreamData.map(&List.to_tuple/1)

  defp shape_2d,
    do: StreamData.list_of(small_dim(), length: 2) |> StreamData.map(&List.to_tuple/1)

  defp shape_3d,
    do: StreamData.list_of(small_dim(), length: 3) |> StreamData.map(&List.to_tuple/1)

  # ------- helpers -------

  defp tensor(shape) do
    n = Tuple.product(shape)

    StreamData.list_of(StreamData.float(min: -3.0, max: 3.0), length: n)
    |> StreamData.map(fn xs ->
      Nx.tensor(xs, type: :f32) |> Nx.reshape(shape)
    end)
  end

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp run_and_compare(f, args) do
    got = apply(f, Enum.map(args, &arm/1)) |> Nx.backend_copy(Nx.BinaryBackend)
    expected = apply(f, Enum.map(args, &ref/1))

    diff =
      Nx.subtract(got, expected)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff < @tol, "diff=#{diff}, got=#{inspect(got)}, expected=#{inspect(expected)}"
  end

  # ------- elementwise unary -------

  property "abs matches BinaryBackend" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.abs/1, [t])
    end
  end

  property "negate matches BinaryBackend" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.negate/1, [t])
    end
  end

  property "sqrt matches BinaryBackend (positive input)" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.sqrt/1, [Nx.add(Nx.abs(t), 0.01)])
    end
  end

  property "exp matches BinaryBackend (clipped to avoid Inf)" do
    check all shape <- shape_1d(), t <- tensor(shape) do
      # exp(big) → Inf which breaks diff comparison; clamp.
      run_and_compare(&Nx.exp/1, [Nx.clip(t, -5.0, 5.0)])
    end
  end

  property "tanh matches BinaryBackend" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.tanh/1, [t])
    end
  end

  # ------- elementwise binary -------

  property "add same-shape matches BinaryBackend" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      run_and_compare(&Nx.add/2, [a, b])
    end
  end

  property "subtract matches BinaryBackend" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      run_and_compare(&Nx.subtract/2, [a, b])
    end
  end

  property "multiply matches BinaryBackend" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      run_and_compare(&Nx.multiply/2, [a, b])
    end
  end

  property "divide matches BinaryBackend (clamped to avoid /0)" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      b_safe = Nx.add(b, Nx.tensor(2.0))
      run_and_compare(&Nx.divide/2, [a, b_safe])
    end
  end

  property "max matches BinaryBackend" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      run_and_compare(&Nx.max/2, [a, b])
    end
  end

  property "min matches BinaryBackend" do
    check all shape <- shape_2d(), a <- tensor(shape), b <- tensor(shape) do
      run_and_compare(&Nx.min/2, [a, b])
    end
  end

  # ------- reductions -------

  property "sum across all axes matches" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.sum/1, [t])
    end
  end

  property "mean across all axes matches" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.mean/1, [t])
    end
  end

  property "reduce_max matches" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.reduce_max/1, [t])
    end
  end

  # ------- matmul -------

  property "2-D dot matches BinaryBackend" do
    check all m <- medium_dim(),
              k <- medium_dim(),
              n <- medium_dim(),
              a <- tensor({m, k}),
              b <- tensor({k, n}) do
      run_and_compare(&Nx.dot/2, [a, b])
    end
  end

  # ------- shape ops -------

  property "transpose 2-D matches" do
    check all shape <- shape_2d(), t <- tensor(shape) do
      run_and_compare(&Nx.transpose/1, [t])
    end
  end

  property "broadcast scalar matches" do
    check all shape <- shape_2d() do
      scalar = Nx.tensor(1.5)

      got =
        scalar
        |> arm()
        |> Nx.broadcast(shape)
        |> Nx.backend_copy(Nx.BinaryBackend)

      ref = Nx.broadcast(scalar, shape)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
    end
  end

  property "softmax along last axis matches" do
    check all shape <- shape_3d(), t <- tensor(shape) do
      # Stable softmax: subtract max, exp, divide by sum.
      f = fn x ->
        m = Nx.reduce_max(x, axes: [-1], keep_axes: true)
        e = Nx.exp(Nx.subtract(x, m))
        s = Nx.sum(e, axes: [-1], keep_axes: true)
        Nx.divide(e, s)
      end

      run_and_compare(f, [t])
    end
  end
end

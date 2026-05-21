defmodule NxArm.ConcurrencyTest do
  @moduledoc """
  Multi-process / multi-task NIF stress tests. The Rust kernels use
  rayon under the hood, so the BEAM is calling into a parallel pool
  from concurrent dirty schedulers. This catches any rayon /
  thread-safety regression and any missing Send/Sync impl.
  """

  use ExUnit.Case, async: true

  test "matmul correctness under 16 concurrent tasks" do
    a = Nx.iota({32, 64}, type: :f32) |> Nx.divide(100) |> Nx.sin()
    b = Nx.iota({64, 32}, type: :f32) |> Nx.divide(100) |> Nx.cos()

    a_arm = Nx.backend_copy(a, NxArm.Backend)
    b_arm = Nx.backend_copy(b, NxArm.Backend)
    expected = Nx.dot(a, b)

    tasks =
      for _ <- 1..16 do
        Task.async(fn -> Nx.dot(a_arm, b_arm) |> Nx.backend_copy(Nx.BinaryBackend) end)
      end

    results = Task.await_many(tasks, 30_000)

    for got <- results do
      diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-4
    end
  end

  test "100 different matmuls in parallel all match reference" do
    tasks =
      for seed <- 1..100 do
        Task.async(fn ->
          a = Nx.iota({16, 32}, type: :f32) |> Nx.add(seed * 1.0) |> Nx.divide(1000) |> Nx.sin()
          b = Nx.iota({32, 16}, type: :f32) |> Nx.add(seed * 1.0) |> Nx.divide(1000) |> Nx.cos()
          a_arm = Nx.backend_copy(a, NxArm.Backend)
          b_arm = Nx.backend_copy(b, NxArm.Backend)

          got = Nx.dot(a_arm, b_arm) |> Nx.backend_copy(Nx.BinaryBackend)
          expected = Nx.dot(a, b)
          diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          diff
        end)
      end

    diffs = Task.await_many(tasks, 60_000)
    assert Enum.max(diffs) < 1.0e-3
  end

  test "mixed-op concurrent workload survives" do
    base_a = Nx.iota({16, 16}, type: :f32) |> Nx.divide(50)
    base_b = Nx.iota({16, 16}, type: :f32) |> Nx.divide(50)

    ops = [
      fn -> Nx.dot(Nx.backend_copy(base_a, NxArm.Backend), Nx.backend_copy(base_b, NxArm.Backend)) end,
      fn -> Nx.add(Nx.backend_copy(base_a, NxArm.Backend), Nx.backend_copy(base_b, NxArm.Backend)) end,
      fn -> Nx.multiply(Nx.backend_copy(base_a, NxArm.Backend), Nx.backend_copy(base_b, NxArm.Backend)) end,
      fn -> Nx.exp(Nx.backend_copy(base_a, NxArm.Backend) |> Nx.clip(-5, 5)) end,
      fn -> Nx.tanh(Nx.backend_copy(base_a, NxArm.Backend)) end,
      fn -> Nx.transpose(Nx.backend_copy(base_a, NxArm.Backend)) end,
      fn -> Nx.sum(Nx.backend_copy(base_a, NxArm.Backend)) end
    ]

    tasks =
      for _ <- 1..50 do
        op = Enum.random(ops)
        Task.async(fn -> op.() |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.shape() end)
      end

    # Just survive without crash, all shapes are valid.
    results = Task.await_many(tasks, 30_000)
    assert length(results) == 50
  end
end

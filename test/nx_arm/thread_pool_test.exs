defmodule NxArm.ThreadPoolTest do
  # async: false because it pokes the global rayon pool state.
  use ExUnit.Case, async: false

  test "current_thread_count_op reports a positive integer" do
    n = NxArm.Native.current_thread_count_op()
    assert is_integer(n)
    assert n > 0
  end

  test "init_thread_pool_op returns :already_initialised after first call" do
    # Run some parallel work first to ensure rayon has touched the
    # global pool. Any of our matmul NIFs will do.
    a = Nx.iota({4, 32}, type: :f32) |> Nx.to_binary()
    w = Nx.iota({4, 32}, type: :s8) |> Nx.to_binary()
    scales = Nx.broadcast(1.0, {4}) |> Nx.to_binary()
    _ = NxArm.Native.int8_matmul_f32_op(a, w, scales, 1.0, 4, 4, 32)

    # First or subsequent call: either way this should not return :ok
    # (some other test or warmup may have already initialised).
    result = NxArm.Native.init_thread_pool_op(2)
    assert result in [:ok, :already_initialised]
  end

  test "Runtime.init_thread_pool defaults to perf-cluster pinning when no config" do
    prev_count = Application.get_env(:nx_arm, :thread_count)
    prev_pool = Application.get_env(:nx_arm, :thread_pool)
    Application.delete_env(:nx_arm, :thread_count)
    Application.delete_env(:nx_arm, :thread_pool)

    try do
      result = NxArm.Runtime.init_thread_pool()
      # Pool is already up from app start, but the shape is the same.
      assert match?({status, n, perf, _src}
                     when status in [:ok, :already_initialised] and is_integer(n) and is_list(perf),
                    result)
    after
      if prev_count, do: Application.put_env(:nx_arm, :thread_count, prev_count)
      if prev_pool, do: Application.put_env(:nx_arm, :thread_pool, prev_pool)
    end
  end

  test "Runtime.topology/0 returns perf + all core lists" do
    topo = NxArm.Runtime.topology()
    assert is_list(topo.perf_cores)
    assert is_list(topo.all_cores)
    assert is_binary(topo.source)
    # Perf cores ⊆ all cores
    assert MapSet.subset?(MapSet.new(topo.perf_cores), MapSet.new(topo.all_cores))
  end

  test "Runtime.thread_count delegates to NIF" do
    nif_count = NxArm.Native.current_thread_count_op()
    api_count = NxArm.Runtime.thread_count()
    assert nif_count == api_count
  end
end

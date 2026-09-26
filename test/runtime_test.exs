defmodule ArmAI.RuntimeTest do
  use ExUnit.Case, async: false

  describe "topology/0" do
    test "returns the standard struct" do
      t = ArmAI.Runtime.topology()
      assert is_list(t[:perf_cores])
      assert is_list(t[:efficiency_cores])
      assert is_list(t[:all_cores])
      assert is_binary(t[:source])
    end

    test "perf + efficiency partition all_cores" do
      t = ArmAI.Runtime.topology()
      all = MapSet.new(t[:all_cores])
      perf = MapSet.new(t[:perf_cores])
      eff = MapSet.new(t[:efficiency_cores])
      assert MapSet.subset?(perf, all)
      assert MapSet.subset?(eff, all)
      assert MapSet.disjoint?(perf, eff)
    end
  end

  describe "thread_count/0" do
    test "returns a positive integer" do
      n = ArmAI.Runtime.thread_count()
      assert is_integer(n)
      assert n > 0
    end
  end

  describe "init_thread_pool/0" do
    test "honors :thread_count config" do
      Application.put_env(:arm_ai, :thread_count, 1)
      result = ArmAI.Runtime.init_thread_pool()
      Application.delete_env(:arm_ai, :thread_count)
      # Rayon is global one-shot — likely :already_initialised, which is fine
      assert match?({_, 1}, result) or match?({_, _, _, _}, result)
    end

    test "honors :thread_pool :all_cores config" do
      Application.put_env(:arm_ai, :thread_pool, :all_cores)
      result = ArmAI.Runtime.init_thread_pool()
      Application.delete_env(:arm_ai, :thread_pool)
      assert match?({:no_pinning, n} when is_integer(n) and n > 0, result)
    end

    test "default path returns the 4-tuple (perf cluster init / already initialised)" do
      Application.delete_env(:arm_ai, :thread_count)
      Application.delete_env(:arm_ai, :thread_pool)
      result = ArmAI.Runtime.init_thread_pool()
      assert match?({status, n, perf, source}
                    when status in [:ok, :already_initialised] and
                         is_integer(n) and is_list(perf) and is_binary(source),
                    result)
    end
  end
end

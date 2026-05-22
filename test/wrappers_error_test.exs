defmodule ArmAI.WrappersErrorTest do
  use ExUnit.Case, async: true

  # Contract tests for arm_ai's shim modules — they must never
  # raise on unhappy paths. The Audio / Safetensors / Tokenizers
  # tests that used to live here moved to:
  #   - Audio.* (in the audio package)
  #   - safetensors + tokenizers contracts (upstream Hex packages)

  describe "ArmAI.Performance (shim → CpuGovernor.Performance)" do
    test "with_performance/1 runs and returns the fun's value" do
      assert :hello = ArmAI.Performance.with_performance(fn -> :hello end)
    end

    test "with_performance/1 still returns even when the fun raises" do
      # Restores prior governor even on a raise (after-clause guarantee).
      assert_raise RuntimeError, "boom", fn ->
        ArmAI.Performance.with_performance(fn -> raise "boom" end)
      end
    end

    test "current_governors/1 on empty list returns empty map" do
      assert ArmAI.Performance.current_governors([]) == %{}
    end

    test "current_governors/1 returns a map keyed by cpu" do
      result = ArmAI.Performance.current_governors([0, 1])
      assert is_map(result)
      assert Map.keys(result) |> Enum.sort() == [0, 1]
    end

    test "set_governor/2 on empty cores returns :ok (no-op)" do
      assert :ok = ArmAI.Performance.set_governor([], "performance")
    end

    test "restore_governors/1 with empty map is a no-op" do
      assert :ok = ArmAI.Performance.restore_governors(%{})
    end

    test "max_cpu_temp_c/0 returns a non-negative float" do
      val = ArmAI.Performance.max_cpu_temp_c()
      assert is_float(val)
      assert val >= 0.0
    end
  end

  describe "ArmAI.Runtime (shim → CpuGovernor.Topology)" do
    test "topology/0 returns the expected struct shape" do
      topo = ArmAI.Runtime.topology()
      assert is_list(topo[:perf_cores])
      assert is_list(topo[:efficiency_cores])
      assert is_list(topo[:all_cores])
      assert is_binary(topo[:source])
    end
  end

  describe "ArmAI.Hub (shim → ModelHub)" do
    test "path/1 returns :not_configured for an unknown id" do
      assert {:error, :not_configured} = ArmAI.Hub.path(:__no_such_model)
    end
  end

  describe "ArmAI.StorageResizer (shim → FwupDataResize)" do
    test "run/0 returns :disabled when no config set" do
      Application.delete_env(:nx_arm, :storage_resizer)
      assert :disabled = ArmAI.StorageResizer.run()
    end
  end
end

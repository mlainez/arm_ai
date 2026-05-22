defmodule ArmAI.HubTest do
  @moduledoc """
  Shim tests for `ArmAI.Hub` — the wrapper around `ModelHub`
  that reads from `:nx_arm, :models` config (back-compat with
  pre-extraction users).
  """

  use ExUnit.Case, async: false

  setup do
    Application.delete_env(:nx_arm, :models)
    on_exit(fn -> Application.delete_env(:nx_arm, :models) end)
    :ok
  end

  describe "ensure_all/0" do
    test "returns {:ok, %{}} when no models configured" do
      assert {:ok, %{}} = ArmAI.Hub.ensure_all()
    end

    test "delegates to ModelHub.ensure_all with :nx_arm namespace" do
      path = Path.join(System.tmp_dir!(), "arm_ai_hub_cached_#{System.unique_integer([:positive])}.bin")
      File.write!(path, "ok")
      on_exit(fn -> File.rm(path) end)

      Application.put_env(:nx_arm, :models, [
        cached: [source: {:url, "http://0.0.0.0:1/no"}, path: path]
      ])

      assert {:ok, %{cached: ^path}} = ArmAI.Hub.ensure_all()
    end
  end

  describe "path/1" do
    test "returns :not_configured when id absent" do
      assert {:error, :not_configured} = ArmAI.Hub.path(:__unknown)
    end

    test "returns the configured path when present" do
      path = "/tmp/arm_ai_hub_path_test"
      File.write!(path, "x")
      on_exit(fn -> File.rm(path) end)

      Application.put_env(:nx_arm, :models, [
        foo: [source: {:url, "http://0.0.0.0:1/n"}, path: path]
      ])

      assert {:ok, ^path} = ArmAI.Hub.path(:foo)
    end
  end

  describe "ensure_one/2" do
    test "delegates correctly" do
      path = "/tmp/arm_ai_hub_one_#{System.unique_integer([:positive])}.bin"
      File.write!(path, "ok")
      on_exit(fn -> File.rm(path) end)

      assert {:ok, ^path} =
               ArmAI.Hub.ensure_one(:test_id,
                 source: {:url, "http://0.0.0.0:1/never"},
                 path: path
               )
    end
  end
end

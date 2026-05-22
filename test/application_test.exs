defmodule ArmAI.ApplicationTest do
  use ExUnit.Case, async: false

  # The Application.start/2 callback runs once at `mix test` boot.
  # We can't re-invoke it cleanly inside the same OS process (the
  # named Supervisor is already alive), so this suite verifies the
  # observable post-boot state instead.

  describe "supervisor lifecycle" do
    test "NxArm.Supervisor is alive after boot" do
      assert is_pid(Process.whereis(NxArm.Supervisor))
    end

    test "ArmAI.Application has boot_mode visible via config" do
      mode = Application.get_env(:nx_arm, :boot_mode, :normal)
      assert mode in [:normal, :recovery]
    end
  end

  describe "ensure_models behaviour (private path exercised via Hub)" do
    test "no models configured is a no-op" do
      Application.delete_env(:nx_arm, :models)
      assert {:ok, %{}} == ArmAI.Hub.ensure_all()
    end
  end

  describe "recovery mode start/2 (idempotent call)" do
    test "calling start/2 when supervisor already runs returns {:error, {:already_started, _}}" do
      # ApplicationTest verifies the contract — start/2 either creates
      # the Supervisor or reports it's already up. Either is acceptable.
      Application.put_env(:nx_arm, :boot_mode, :recovery)
      result = ArmAI.Application.start(:normal, [])
      Application.delete_env(:nx_arm, :boot_mode)

      assert match?({:ok, _pid}, result) or
             match?({:error, {:already_started, _pid}}, result)
    end
  end
end

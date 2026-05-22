defmodule ArmAI.StorageResizer do
  @moduledoc """
  Thin shim around `FwupDataResize`. The first-boot F2FS-grow logic
  was extracted from nx_arm into its own package; new code should
  call `FwupDataResize.run/1` directly.

  Reads config under `:nx_arm, :storage_resizer` (the old namespace)
  so existing target.exs files keep working without changes.
  """

  @doc "See `FwupDataResize.run/1`."
  def run do
    config = Application.get_env(:nx_arm, :storage_resizer) || []
    FwupDataResize.run(config)
  end
end

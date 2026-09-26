defmodule ArmAI.Performance do
  @moduledoc """
  Thin shim delegating to `CpuGovernor.Performance` in the
  standalone `cpu_governor` package.

  New code should call `CpuGovernor.Performance` directly.
  """

  defdelegate with_performance(fun), to: CpuGovernor.Performance
  defdelegate current_governors(cores), to: CpuGovernor.Performance
  defdelegate set_governor(cores, governor), to: CpuGovernor.Performance
  defdelegate restore_governors(prev), to: CpuGovernor.Performance
  defdelegate max_cpu_temp_c(), to: CpuGovernor.Performance
end

defmodule NxArm.Performance do
  @moduledoc """
  Scoped CPU governor control. Delegates to `NervesCPU.Performance`
  in the standalone `nerves_cpu` package — kept here as a thin
  compatibility shim so internal nx_arm callers
  (`NxArm.Models.LlamaCandle.generate/2`, etc.) continue to work.

  New code should call `NervesCPU.Performance` directly.
  """

  defdelegate with_performance(fun), to: NervesCPU.Performance
  defdelegate current_governors(cores), to: NervesCPU.Performance
  defdelegate set_governor(cores, governor), to: NervesCPU.Performance
  defdelegate restore_governors(prev), to: NervesCPU.Performance
  defdelegate max_cpu_temp_c(), to: NervesCPU.Performance
end

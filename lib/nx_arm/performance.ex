defmodule NxArm.Performance do
  @moduledoc """
  Scoped CPU performance governor control for inference bursts.

  On phone-class ARM SoCs the default CPU governor (`schedutil`,
  `interactive`, etc.) takes 30–80 ms to ramp from idle frequency
  to max. ML inference is short bursts of compute, so by the time
  the governor reacts the burst is already over — measured
  throughput ends up 2–3× lower than the silicon can deliver.

  This module flips the perf-cluster cores to `performance` for a
  scoped block, then restores the previous governor:

      NxArm.Performance.with_performance(fn ->
        NxArm.Models.Llama.generate(model, ...)
      end)

  Running cores at max frequency continuously will heat the device
  and can trigger thermal throttling. Use this only during the
  actual inference call so the cores can cool between requests.

  The Application *does not* apply the performance governor at
  boot — that would defeat the cooling story. Set
  `config :nx_arm, governor_at_boot: :performance` if you really
  want always-on performance mode.
  """

  require Logger

  @doc """
  Run `fun` with the perf-cluster cores pinned to `performance`,
  restoring the previous governor afterward (even on crash).

  Returns whatever `fun` returns.
  """
  @spec with_performance((-> any())) :: any()
  def with_performance(fun) when is_function(fun, 0) do
    cores =
      case NxArm.Runtime.topology() do
        %{perf_cores: c} when is_list(c) and c != [] -> c
        _ -> []
      end

    if cores == [] do
      fun.()
    else
      previous = current_governors(cores)

      _ = set_governor(cores, "performance")

      try do
        fun.()
      after
        restore_governors(previous)
      end
    end
  end

  @doc "Read the currently-active CPU governor for each given core."
  @spec current_governors([non_neg_integer()]) :: %{non_neg_integer() => String.t()}
  def current_governors(cores) do
    Map.new(cores, fn cpu ->
      gov =
        case File.read("/sys/devices/system/cpu/cpu#{cpu}/cpufreq/scaling_governor") do
          {:ok, s} -> String.trim(s)
          _ -> "schedutil"
        end

      {cpu, gov}
    end)
  end

  @doc """
  Set `governor` on `cores`. Best-effort: failures are logged at
  debug (sysfs writes can fail if the kernel lacks the governor or
  if the caller isn't root).
  """
  @spec set_governor([non_neg_integer()], String.t()) :: :ok | {:error, term()}
  def set_governor([], _governor), do: :ok

  def set_governor(cores, governor) do
    Enum.each(cores, fn cpu ->
      path = "/sys/devices/system/cpu/cpu#{cpu}/cpufreq/scaling_governor"

      case File.write(path, governor) do
        :ok -> :ok
        {:error, reason} -> Logger.debug("[nx_arm] gov set failed cpu#{cpu}: #{inspect(reason)}")
      end
    end)
  end

  @doc "Restore per-core governors from a map captured by `current_governors/1`."
  @spec restore_governors(%{non_neg_integer() => String.t()}) :: :ok
  def restore_governors(prev) do
    Enum.each(prev, fn {cpu, gov} ->
      _ = set_governor([cpu], gov)
    end)
  end

  @doc """
  Read the highest CPU temperature reported by any thermal zone
  whose name starts with `cpu`. Useful for guarding inference
  against runaway heat.
  """
  @spec max_cpu_temp_c() :: float()
  def max_cpu_temp_c do
    Path.wildcard("/sys/class/thermal/thermal_zone*")
    |> Enum.flat_map(fn zone ->
      with {:ok, type} <- File.read("#{zone}/type"),
           type = String.trim(type),
           true <- String.contains?(type, "cpu"),
           {:ok, raw} <- File.read("#{zone}/temp"),
           {milli, _} <- Integer.parse(String.trim(raw)) do
        [milli / 1000.0]
      else
        _ -> []
      end
    end)
    |> case do
      [] -> 0.0
      vals -> Enum.max(vals)
    end
  end
end

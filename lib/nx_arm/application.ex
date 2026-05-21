defmodule NxArm.Application do
  @moduledoc false

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    auto_init_thread_pool()
    Supervisor.start_link([], strategy: :one_for_one, name: NxArm.Supervisor)
  end

  defp auto_init_thread_pool do
    case NxArm.Runtime.init_thread_pool() do
      {:ok, n, perf, source} ->
        Logger.info(
          "[nx_arm] pinned rayon pool to #{n} perf core(s) #{inspect(perf)} (source: #{source})"
        )

      {:already_initialised, n, perf, source} ->
        Logger.info(
          "[nx_arm] rayon pool already initialised; perf cluster detected as #{n} core(s) #{inspect(perf)} (source: #{source})"
        )

      {:no_pinning, n} ->
        Logger.info("[nx_arm] using all #{n} cores (no perf-cluster pinning)")
    end
  rescue
    e ->
      # Detection or NIF call failed (e.g. running on host without
      # the latest firmware). Log and continue with rayon defaults.
      Logger.warning("[nx_arm] thread pool init failed: #{Exception.message(e)}; rayon defaults will apply")
  end
end

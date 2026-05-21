defmodule NxArm.Application do
  @moduledoc false

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    auto_init_thread_pool()
    warm_up_dirty_schedulers()
    Supervisor.start_link([], strategy: :one_for_one, name: NxArm.Supervisor)
  end

  # Make sure every BEAM dirty-CPU scheduler thread gets a chance to
  # land on a perf core. We spawn a task per online scheduler that
  # calls the pin NIF on its dirty scheduler thread. The thread_local
  # cache in topology.rs means subsequent calls are cheap, so this
  # only runs the migration syscall once per thread.
  defp warm_up_dirty_schedulers do
    if function_exported?(NxArm.Native, :pin_calling_thread_op, 0) do
      n =
        case :erlang.system_info(:dirty_cpu_schedulers_online) do
          n when is_integer(n) and n > 0 -> n
          _ -> :erlang.system_info(:logical_processors) || 4
        end

      # Spawn 4× the scheduler count so the BEAM rotates tasks
      # through every dirty scheduler at least once.
      tasks =
        for _ <- 1..(n * 4) do
          Task.async(fn -> NxArm.Native.pin_calling_thread_op() end)
        end

      _ = Task.await_many(tasks, 5_000)
    end
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
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

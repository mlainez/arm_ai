defmodule ArmAI.Application do
  @moduledoc false

  use Application
  require Logger

  # ArmAI.Application owns ONLY the NIF-runtime side effects:
  # rayon pool init, dirty-scheduler warm-up, scheduler affinity to
  # the efficiency cluster, and the optional boot governor.
  #
  # First-boot disk resize and model-hub downloads live in the
  # application layer above us — `nerves_ai` wires those in by
  # depending on `:fwup_data_resize` and `:model_hub` directly.

  @impl true
  def start(_type, _args) do
    case Application.get_env(:nx_arm, :boot_mode, :normal) do
      :recovery ->
        Logger.warning(
          "[arm_ai] BOOT_MODE=:recovery — no thread-pool pinning, no governor changes"
        )

        Supervisor.start_link([], strategy: :one_for_one, name: NxArm.Supervisor)

      _ ->
        auto_init_thread_pool()
        warm_up_dirty_schedulers()
        pin_normal_schedulers_to_efficiency()
        maybe_apply_boot_governor()
        Supervisor.start_link([], strategy: :one_for_one, name: NxArm.Supervisor)
    end
  end

  # On big.LITTLE chips, the right partition is:
  #   * perf cluster      → rayon pool + BEAM dirty CPU schedulers
  #     (these run our heavy NEON compute NIFs)
  #   * efficiency cluster → BEAM normal schedulers
  #     (Elixir glue code, GC, message-passing — latency-sensitive
  #     but not compute-hungry)
  #
  # rayon + dirty CPU schedulers are already pinned to the perf
  # cluster (see auto_init_thread_pool + warm_up_dirty_schedulers).
  # Here we migrate the BEAM normal-scheduler threads to the
  # efficiency cluster so they don't share L1/L2 with the NEON
  # workers.
  defp pin_normal_schedulers_to_efficiency do
    if not function_exported?(ArmAI.Native, :pin_thread_to_cores_op, 1) do
      :skip
    else
      eff =
        case ArmAI.Runtime.topology() do
          %{efficiency_cores: c} when is_list(c) and c != [] -> c
          _ -> []
        end

      if eff == [] do
        :no_efficiency_cluster
      else
        # Spawn one task per normal scheduler — each runs on a
        # different scheduler thread, pins itself to the eff cluster.
        n = :erlang.system_info(:schedulers_online)

        tasks =
          for sched_id <- 1..n do
            Task.async(fn ->
              # Bias this lightweight task toward the matching scheduler.
              :erlang.process_flag(:scheduler, sched_id)
              ArmAI.Native.pin_thread_to_cores_op(eff)
            end)
          end

        _ = Task.await_many(tasks, 5_000)
        Logger.info(
          "[nx_arm] BEAM normal schedulers (#{n}) migrated to efficiency cluster #{inspect(eff)}"
        )
      end
    end
  rescue
    e -> Logger.warning("[nx_arm] normal-scheduler pinning failed: #{Exception.message(e)}")
  catch
    _, _ -> :ok
  end

  # By default we do NOT pin CPUs to `performance` at boot —
  # ML inference is bursty, and pinning to max frequency
  # continuously heats the device until thermal throttling kicks
  # in. Use `ArmAI.Performance.with_performance/1` to scope it to
  # the actual inference call.
  #
  # Opt-in to always-on max clock via:
  #
  #     config :nx_arm, governor_at_boot: :performance
  #
  defp maybe_apply_boot_governor do
    case Application.get_env(:nx_arm, :governor_at_boot, :default) do
      :default ->
        :ok

      governor when is_atom(governor) ->
        cores =
          case ArmAI.Runtime.topology() do
            %{perf_cores: c} when is_list(c) and c != [] -> c
            _ -> []
          end

        ArmAI.Performance.set_governor(cores, Atom.to_string(governor))
        Logger.info("[nx_arm] applied CPU governor #{governor} to #{inspect(cores)}")
    end
  rescue
    e -> Logger.warning("[nx_arm] boot governor policy crashed: #{Exception.message(e)}")
  end

  # Make sure every BEAM dirty-CPU scheduler thread gets a chance to
  # land on a perf core. We spawn a task per online scheduler that
  # calls the pin NIF on its dirty scheduler thread. The thread_local
  # cache in topology.rs means subsequent calls are cheap, so this
  # only runs the migration syscall once per thread.
  defp warm_up_dirty_schedulers do
    if function_exported?(ArmAI.Native, :pin_calling_thread_op, 0) do
      n =
        case :erlang.system_info(:dirty_cpu_schedulers_online) do
          n when is_integer(n) and n > 0 -> n
          _ -> :erlang.system_info(:logical_processors) || 4
        end

      # Spawn 4× the scheduler count so the BEAM rotates tasks
      # through every dirty scheduler at least once.
      tasks =
        for _ <- 1..(n * 4) do
          Task.async(fn -> ArmAI.Native.pin_calling_thread_op() end)
        end

      _ = Task.await_many(tasks, 5_000)
    end
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
  end

  defp auto_init_thread_pool do
    case ArmAI.Runtime.init_thread_pool() do
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

defmodule NxArm.Runtime do
  @moduledoc """
  Runtime configuration for the nx_arm CPU backend.

  ## Thread pool

  Rayon's global thread pool is sized to logical core count on first
  use. For Nerves devices that's usually fine, but two cases where
  you want to override it:

    * big.LITTLE chips where running on every core hurts vs pinning
      to the perf cluster.
    * Workloads that want headroom for other BEAM schedulers, audio
      threads, or async I/O.

  Set the count from application config:

      config :nx_arm, thread_count: 4

  …then call `NxArm.Runtime.init_thread_pool/0` at app start (before
  any matmul or convolution touches rayon). Once initialised the
  pool is fixed for the OS process lifetime.
  """

  @doc """
  Initialise the rayon thread pool. Default policy: detect
  big.LITTLE topology and pin to the perf cluster. The generic
  ARM backend should *always* leverage the best cores first; on
  homogeneous chips this collapses to the obvious "use everything"
  case.

  Override via app config:

      config :nx_arm,
        thread_pool: :perf_cluster      # default
      # or
      config :nx_arm,
        thread_pool: :all_cores
      # or
      config :nx_arm,
        thread_count: 4                 # explicit count, no pinning

  Returns a tuple describing what was set up:

      {:ok, n_threads, perf_core_ids, source}
      {:already_initialised, n_threads, perf_core_ids, source}
      {:no_pinning, n_threads}     # explicit thread_count path
  """
  @spec init_thread_pool() ::
          {:ok | :already_initialised, pos_integer(), [non_neg_integer()], String.t()}
          | {:no_pinning, pos_integer()}
  def init_thread_pool do
    cond do
      n = Application.get_env(:nx_arm, :thread_count) ->
        status = NxArm.Native.init_thread_pool_op(n)
        {status, n}

      Application.get_env(:nx_arm, :thread_pool, :perf_cluster) == :all_cores ->
        # No pinning: rayon picks default count = logical CPU count.
        n = NxArm.Native.current_thread_count_op()
        {:no_pinning, n}

      true ->
        NxArm.Native.init_perf_cluster_op()
    end
  end

  @doc "Active thread count in the rayon pool."
  @spec thread_count() :: pos_integer()
  def thread_count, do: NxArm.Native.current_thread_count_op()

  @doc """
  Inspect detected big.LITTLE topology. Useful for diagnostics from
  iex on the device.
  """
  @spec topology() :: %{
          perf_cores: [non_neg_integer()],
          efficiency_cores: [non_neg_integer()],
          all_cores: [non_neg_integer()],
          source: String.t()
        }
  def topology, do: CpuGovernor.Topology.detect()
end

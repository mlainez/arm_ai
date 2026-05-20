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
  Initialise the rayon thread pool from `Application.get_env(:nx_arm,
  :thread_count)`. Returns `:ok`, `:already_initialised`, or
  `:no_config` if no thread_count is set.
  """
  @spec init_thread_pool() :: :ok | :already_initialised | :no_config
  def init_thread_pool do
    case Application.get_env(:nx_arm, :thread_count) do
      nil ->
        :no_config

      n when is_integer(n) and n > 0 ->
        NxArm.Native.init_thread_pool_op(n)
    end
  end

  @doc "Active thread count in the rayon pool."
  @spec thread_count() :: pos_integer()
  def thread_count, do: NxArm.Native.current_thread_count_op()
end

defmodule NxArm do
  @moduledoc """
  Nx backend for ARM CPUs via NEON intrinsics.

  See `NxArm.Backend` for the backend implementation and supported ops.

  ## Usage

      Nx.global_default_backend(NxArm.Backend)

  ## Fused ops

  These call hand-tuned NEON kernels directly, bypassing the
  primitive-by-primitive Axon/Nx.Defn decomposition. Use them where
  speed matters more than uniform Nx semantics:

    * `softmax/2` — fused softmax along an axis.
  """

  @doc """
  Fused softmax along `axis` (default `-1`). Reads the input tensor
  back as a binary, calls `NxArm.Native.softmax_f32_op/3`, returns a
  fresh `NxArm.Backend` tensor.

  ## Example

      iex> x = Nx.tensor([[1.0, 2.0, 3.0]], backend: NxArm.Backend)
      iex> NxArm.softmax(x)
  """
  def softmax(%Nx.Tensor{} = tensor, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    shape = Nx.shape(tensor) |> Tuple.to_list()
    rank = length(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if axis != rank - 1 do
      raise ArgumentError,
            "NxArm.softmax currently only supports the last axis (got #{axis} for rank #{rank})"
    end

    if Nx.type(tensor) != {:f, 32} do
      raise ArgumentError, "NxArm.softmax requires :f32 input"
    end

    inner = elem(Nx.shape(tensor), rank - 1)
    n_outer = div(Nx.size(tensor), inner)

    bin = NxArm.Backend.__bin_of__(tensor)
    out_bin = NxArm.Native.softmax_f32_op(bin, n_outer, inner)

    %{tensor | data: %NxArm.Backend{bin: out_bin}}
  end
end

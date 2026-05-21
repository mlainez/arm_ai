defmodule NxArm.KVCache do
  @moduledoc """
  Append-style key/value cache for transformer decoders.

  Stores K and V tensors of shape `{n_layers, n_heads, max_seq, head_dim}`
  preallocated up front, plus a current-length cursor. Each decode
  step writes the new (key, value) row at index `length` and bumps the
  cursor. Attention reads the live prefix `[0..length)` via
  `prefix/2`.

  Preallocation matters: growing tensors per step would allocate at
  every token. Here the underlying binaries are reused — only the
  cursor moves.
  """

  defstruct [:k, :v, :length, :max_seq, :n_layers, :n_heads, :head_dim, :type]

  @type t :: %__MODULE__{
          k: Nx.Tensor.t(),
          v: Nx.Tensor.t(),
          length: non_neg_integer(),
          max_seq: pos_integer(),
          n_layers: pos_integer(),
          n_heads: pos_integer(),
          head_dim: pos_integer(),
          type: Nx.Type.t()
        }

  @doc """
  Allocate an empty cache. K and V are zero tensors of shape
  `{n_layers, n_heads, max_seq, head_dim}`. The cursor starts at 0.
  """
  @spec new(pos_integer(), pos_integer(), pos_integer(), pos_integer(), Keyword.t()) :: t()
  def new(n_layers, n_heads, max_seq, head_dim, opts \\ []) do
    type = Keyword.get(opts, :type, {:f, 32})
    shape = {n_layers, n_heads, max_seq, head_dim}

    %__MODULE__{
      k: Nx.broadcast(Nx.tensor(0, type: type), shape),
      v: Nx.broadcast(Nx.tensor(0, type: type), shape),
      length: 0,
      max_seq: max_seq,
      n_layers: n_layers,
      n_heads: n_heads,
      head_dim: head_dim,
      type: type
    }
  end

  @doc """
  Append one decode step's K and V at the current cursor position.

  `k_step` and `v_step` are `{n_layers, n_heads, 1, head_dim}` tensors
  (one new token's contribution at every layer). Returns the updated
  cache with `length` incremented.
  """
  @spec append(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: t()
  def append(%__MODULE__{} = c, k_step, v_step) do
    if c.length >= c.max_seq do
      raise ArgumentError,
            "KV cache full: length=#{c.length} max_seq=#{c.max_seq}"
    end

    k = Nx.put_slice(c.k, [0, 0, c.length, 0], k_step)
    v = Nx.put_slice(c.v, [0, 0, c.length, 0], v_step)
    %{c | k: k, v: v, length: c.length + 1}
  end

  @doc """
  Return the live K/V prefix as `{k_prefix, v_prefix}` of shape
  `{n_layers, n_heads, length, head_dim}`. The trailing unused slots
  are not part of the result.
  """
  @spec prefix(t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def prefix(%__MODULE__{length: 0}) do
    raise ArgumentError,
          "KV cache is empty -- call append/3 at least once before prefix/1, " <>
            "or check length == 0 yourself"
  end

  def prefix(%__MODULE__{} = c) do
    k = Nx.slice(c.k, [0, 0, 0, 0], [c.n_layers, c.n_heads, c.length, c.head_dim])
    v = Nx.slice(c.v, [0, 0, 0, 0], [c.n_layers, c.n_heads, c.length, c.head_dim])
    {k, v}
  end

  @doc """
  Return K/V for a single layer at index `l`. Shape:
  `{n_heads, length, head_dim}`. Used inside the per-layer attention
  loop.
  """
  @spec layer(t(), non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def layer(%__MODULE__{} = c, l) when l >= 0 do
    {k_all, v_all} = prefix(c)
    {Nx.squeeze(Nx.slice(k_all, [l, 0, 0, 0], [1, c.n_heads, c.length, c.head_dim]), axes: [0]),
     Nx.squeeze(Nx.slice(v_all, [l, 0, 0, 0], [1, c.n_heads, c.length, c.head_dim]), axes: [0])}
  end

  @doc "Reset the cursor without reallocating the underlying storage."
  @spec reset(t()) :: t()
  def reset(%__MODULE__{} = c), do: %{c | length: 0}
end

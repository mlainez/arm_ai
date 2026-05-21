defmodule NxArm.KVCache do
  @moduledoc """
  Append-style key/value cache for transformer decoders.

  Each layer has its own preallocated K and V tensor of shape
  `{n_kv_heads, max_seq, head_dim}`. A single shared cursor tracks
  the current sequence length. Decode-step writes hit one layer's
  K/V row directly — no zero-tensor allocation per layer-step.

      cache = NxArm.KVCache.new(n_layers, n_kv_heads, max_seq, head_dim)
      cache = NxArm.KVCache.append_layer(cache, layer_idx, k_row, v_row)
      {k_for_attn, v_for_attn} = NxArm.KVCache.layer_view(cache, layer_idx)

  The cursor advances when the caller bumps it explicitly via
  `advance/1`, since within one timestep we write to every layer
  with the same cursor value.
  """

  defstruct [:k_layers, :v_layers, :length, :max_seq, :n_layers, :n_heads, :head_dim, :type]

  @type t :: %__MODULE__{
          k_layers: %{non_neg_integer() => Nx.Tensor.t()},
          v_layers: %{non_neg_integer() => Nx.Tensor.t()},
          length: non_neg_integer(),
          max_seq: pos_integer(),
          n_layers: pos_integer(),
          n_heads: pos_integer(),
          head_dim: pos_integer(),
          type: Nx.Type.t()
        }

  @doc """
  Allocate an empty cache. Per-layer K/V tensors live in maps; each
  is `{n_kv_heads, max_seq, head_dim}` zero-initialised. Cursor 0.
  """
  @spec new(pos_integer(), pos_integer(), pos_integer(), pos_integer(), Keyword.t()) :: t()
  def new(n_layers, n_heads, max_seq, head_dim, opts \\ []) do
    type = Keyword.get(opts, :type, {:f, 32})
    backend = Keyword.get(opts, :backend, NxArm.Backend)
    layer_shape = {n_heads, max_seq, head_dim}

    zero =
      Nx.broadcast(Nx.tensor(0, type: type), layer_shape)
      |> Nx.backend_copy(backend)

    k_layers = for li <- 0..(n_layers - 1), into: %{}, do: {li, zero}
    v_layers = for li <- 0..(n_layers - 1), into: %{}, do: {li, zero}

    %__MODULE__{
      k_layers: k_layers,
      v_layers: v_layers,
      length: 0,
      max_seq: max_seq,
      n_layers: n_layers,
      n_heads: n_heads,
      head_dim: head_dim,
      type: type
    }
  end

  @doc """
  Write one timestep's K and V into layer `layer_idx` at the
  current cursor position.

  `k_step` and `v_step` are `{n_kv_heads, 1, head_dim}` tensors.
  Returns the updated cache *without* advancing the cursor — that
  must be done explicitly with `advance/1` after writing every
  layer for the same timestep.
  """
  @spec append_layer(t(), non_neg_integer(), Nx.Tensor.t(), Nx.Tensor.t()) :: t()
  def append_layer(%__MODULE__{} = c, layer_idx, k_step, v_step) do
    if c.length >= c.max_seq do
      raise ArgumentError,
            "KV cache full: length=#{c.length} max_seq=#{c.max_seq}"
    end

    current_k = Map.fetch!(c.k_layers, layer_idx)
    current_v = Map.fetch!(c.v_layers, layer_idx)

    new_k = Nx.put_slice(current_k, [0, c.length, 0], k_step)
    new_v = Nx.put_slice(current_v, [0, c.length, 0], v_step)

    %{
      c
      | k_layers: Map.put(c.k_layers, layer_idx, new_k),
        v_layers: Map.put(c.v_layers, layer_idx, new_v)
    }
  end

  @doc "Bump the shared cursor by one timestep."
  @spec advance(t()) :: t()
  def advance(%__MODULE__{} = c), do: %{c | length: c.length + 1}

  @doc """
  Read the live K/V slice for one layer as
  `{n_kv_heads, length, head_dim}` tensors.
  """
  @spec layer_view(t(), non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def layer_view(%__MODULE__{length: 0}, _layer_idx) do
    raise ArgumentError, "KV cache empty -- write to it before reading"
  end

  def layer_view(%__MODULE__{} = c, layer_idx) do
    k = Map.fetch!(c.k_layers, layer_idx)
    v = Map.fetch!(c.v_layers, layer_idx)
    {Nx.slice(k, [0, 0, 0], [c.n_heads, c.length, c.head_dim]),
     Nx.slice(v, [0, 0, 0], [c.n_heads, c.length, c.head_dim])}
  end

  @doc "Reset the cursor without reallocating storage."
  @spec reset(t()) :: t()
  def reset(%__MODULE__{} = c), do: %{c | length: 0}

  # -----------------------------------------------------------
  # Legacy single-tensor API kept for backward compatibility
  # with the older tests. New code should use append_layer/4 +
  # advance/1 + layer_view/2.
  # -----------------------------------------------------------

  @doc false
  @spec append(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: t()
  def append(%__MODULE__{} = c, k_step, v_step) do
    if c.length >= c.max_seq do
      raise ArgumentError,
            "KV cache full: length=#{c.length} max_seq=#{c.max_seq}"
    end

    # k_step / v_step shape: {n_layers, n_kv_heads, 1, head_dim}.
    # Split per layer and write each into its own slot.
    cache_after =
      Enum.reduce(0..(c.n_layers - 1), c, fn li, c_acc ->
        k_l = Nx.slice(k_step, [li, 0, 0, 0], [1, c.n_heads, 1, c.head_dim])
              |> Nx.reshape({c.n_heads, 1, c.head_dim})

        v_l = Nx.slice(v_step, [li, 0, 0, 0], [1, c.n_heads, 1, c.head_dim])
              |> Nx.reshape({c.n_heads, 1, c.head_dim})

        append_layer(c_acc, li, k_l, v_l)
      end)

    advance(cache_after)
  end

  @doc false
  @spec prefix(t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def prefix(%__MODULE__{length: 0}) do
    raise ArgumentError, "KV cache empty"
  end

  def prefix(%__MODULE__{} = c) do
    # Rebuild the {n_layers, n_kv_heads, length, head_dim} view by
    # stacking per-layer slices. Only used by legacy callers.
    ks = for li <- 0..(c.n_layers - 1) do
      Nx.slice(c.k_layers[li], [0, 0, 0], [c.n_heads, c.length, c.head_dim])
    end

    vs = for li <- 0..(c.n_layers - 1) do
      Nx.slice(c.v_layers[li], [0, 0, 0], [c.n_heads, c.length, c.head_dim])
    end

    {Nx.stack(ks), Nx.stack(vs)}
  end

  @doc false
  @spec layer(t(), non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def layer(c, l), do: layer_view(c, l)
end

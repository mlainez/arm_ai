defmodule NxArm.LLM do
  @moduledoc """
  Helpers for transformer-LM architectures (Llama / Mistral / Phi / Qwen):
  RMSNorm, RoPE, plus small utilities.

  These are direct NIF wrappers — not yet picked up by `NxArm.Compiler`
  pattern fusion. Use them directly in your forward function (or
  in a custom Axon layer) until the compiler-side support lands.
  """

  @doc """
  Fused linear layer: `y = activation(x · W^T + b)`.

  * `x` shape: `{M, K}` or `{B, M, K}` (any leading-batch).
  * `weight` shape: `{N, K}` (output-major — matches how Bumblebee
    + Axon lay dense weights).
  * `bias` shape: `{N}` or `nil`.
  * `activation`: `:none | :relu | :relu6 | :sigmoid | :tanh | :gelu`.

  Avoids the BEAM round-trips of separate `Nx.dot` + `Nx.add` +
  activation calls. The same `linear_f32` NIF that powers the
  compiler-side bias-add pattern; this is the direct-call form
  for hot paths that don't want to rely on pattern recognition.
  """
  def linear(%Nx.Tensor{} = x, %Nx.Tensor{} = weight, bias, activation \\ :none) do
    {b_dim, m_dim, k_dim, out_shape, n_dim} =
      case {Nx.shape(x), Nx.shape(weight)} do
        {{m, k}, {n, k2}} when k == k2 ->
          {1, m, k, {m, n}, n}

        {{b, m, k}, {n, k2}} when k == k2 ->
          {b, m, k, {b, m, n}, n}

        {x_shape, w_shape} ->
          raise ArgumentError,
                "linear: incompatible shapes x=#{inspect(x_shape)} weight=#{inspect(w_shape)}"
      end

    act_str =
      case activation do
        :none -> "none"
        :relu -> "relu"
        :relu6 -> "relu6"
        :sigmoid -> "sigmoid"
        :tanh -> "tanh"
        :gelu -> "gelu"
      end

    x_bin = NxArm.Backend.__bin_of__(x)
    w_bin = NxArm.Backend.__bin_of__(weight)
    b_bin = if bias, do: NxArm.Backend.__bin_of__(bias), else: <<>>

    out_bin =
      NxArm.Native.linear_f32_op(x_bin, w_bin, b_bin, act_str, b_dim, m_dim, n_dim, k_dim)

    %{x | data: %NxArm.Backend{bin: out_bin}, shape: out_shape, type: {:f, 32}}
  end

  @doc """
  RMSNorm along the last axis: `(x / sqrt(mean(x²) + eps)) * gamma`.

  ## Example

      out = NxArm.LLM.rmsnorm(x, gamma, 1.0e-5)
  """
  def rmsnorm(%Nx.Tensor{} = x, %Nx.Tensor{} = gamma, epsilon \\ 1.0e-5) do
    shape = Nx.shape(x) |> Tuple.to_list()
    rank = length(shape)
    inner = Enum.at(shape, rank - 1)
    n_outer = div(Nx.size(x), inner)

    bin = NxArm.Backend.__bin_of__(x)
    gamma_bin = NxArm.Backend.__bin_of__(gamma)
    out_bin = NxArm.Native.rmsnorm_f32_op(bin, gamma_bin, n_outer, inner, epsilon)

    %{x | data: %NxArm.Backend{bin: out_bin}}
  end

  @doc """
  Rotary Position Embedding applied to a Q or K tensor.

  Tensor layout (after attention head split): `{batch, seq, heads, head_dim}`.
  `positions` is shape `{batch * seq}` of integer token positions.
  `inv_freq` is shape `{head_dim/2}` of the precomputed
  `1.0 / base^(2k/head_dim)` series.
  """
  def rope(
        %Nx.Tensor{} = qk,
        %Nx.Tensor{} = positions,
        %Nx.Tensor{} = inv_freq
      ) do
    shape = Nx.shape(qk) |> Tuple.to_list()
    rank = length(shape)
    head_dim = Enum.at(shape, rank - 1)
    heads = Enum.at(shape, rank - 2)
    n_rows = div(Nx.size(qk), head_dim)
    n_tokens = div(n_rows, heads)

    if Nx.size(positions) != n_tokens do
      raise ArgumentError,
            "rope: positions size #{Nx.size(positions)} != n_tokens #{n_tokens}"
    end

    bin = NxArm.Backend.__bin_of__(qk)
    pos_bin = positions |> Nx.as_type(:s64) |> NxArm.Backend.__bin_of__()
    inv_freq_bin = NxArm.Backend.__bin_of__(inv_freq)

    out_bin = NxArm.Native.rope_f32_op(bin, pos_bin, inv_freq_bin, n_rows, head_dim, heads)

    %{qk | data: %NxArm.Backend{bin: out_bin}}
  end

  @doc """
  Precompute the inverse-frequency vector for RoPE.

  Returns a `{head_dim / 2}` f32 tensor with `inv_freq[k] = 1.0 /
  base^(2k / head_dim)`.
  """
  def rope_inv_freq(head_dim, base \\ 10_000.0) when rem(head_dim, 2) == 0 do
    Enum.map(0..(div(head_dim, 2) - 1), fn k ->
      1.0 / :math.pow(base, 2 * k / head_dim)
    end)
    |> Nx.tensor(type: :f32, backend: NxArm.Backend)
  end

  @doc """
  Build a causal attention mask of shape `{seq_len, seq_len}` for
  prefill. `mask[i, j] = 0` when j ≤ i (allowed), `-inf` otherwise
  (masked out before softmax). Allocated once per prefill batch.

  The default fill value is `-1.0e9`, large enough to drive exp() to
  zero in f32 but well clear of overflow when chained with other
  additive offsets.
  """
  @spec causal_mask(pos_integer(), Keyword.t()) :: Nx.Tensor.t()
  def causal_mask(seq_len, opts \\ []) do
    type = Keyword.get(opts, :type, :f32)
    masked_value = Keyword.get(opts, :masked_value, -1.0e9)

    i = Nx.iota({seq_len, 1})
    j = Nx.iota({1, seq_len})
    allowed = Nx.greater_equal(i, j)
    Nx.select(allowed, Nx.tensor(0.0, type: type), Nx.tensor(masked_value, type: type))
  end

  @doc """
  Build a decode-step causal mask: shape `{1, kv_len}` where the
  single query token can attend to all `kv_len` cached positions.
  This is the trivial mask for autoregressive decode after prefill —
  every cached key is in the past so nothing gets masked.

  Returned as a zero tensor (no positions masked); kept as a function
  to make the call site self-documenting and to let callers pass the
  same dtype as their attention scores.
  """
  @spec decode_mask(pos_integer(), Keyword.t()) :: Nx.Tensor.t()
  def decode_mask(kv_len, opts \\ []) do
    type = Keyword.get(opts, :type, :f32)
    Nx.broadcast(Nx.tensor(0.0, type: type), {1, kv_len})
  end

  @doc """
  Apply RoPE at a single position to a tensor of shape
  `{..., 1, head_dim}` (the decode step). Equivalent to building a
  one-element `positions` tensor and calling `rope/3`.
  """
  @spec rope_at(Nx.Tensor.t(), non_neg_integer(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def rope_at(qk, pos, inv_freq) do
    positions = Nx.tensor([pos], type: :s64)
    rope(qk, positions, inv_freq)
  end
end

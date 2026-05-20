defmodule NxArm.LLM do
  @moduledoc """
  Helpers for transformer-LM architectures (Llama / Mistral / Phi / Qwen):
  RMSNorm, RoPE, plus small utilities.

  These are direct NIF wrappers — not yet picked up by `NxArm.Compiler`
  pattern fusion. Use them directly in your forward function (or
  in a custom Axon layer) until the compiler-side support lands.
  """

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
end

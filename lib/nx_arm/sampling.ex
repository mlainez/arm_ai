defmodule NxArm.Sampling do
  @moduledoc """
  Sampling helpers for autoregressive LM token generation.

  Inputs to `sample/2` and `greedy/1` are expected to be 1-D logits
  tensors of shape `{vocab_size}` — i.e. one token's worth of logits.
  This matches the way single-stream autoregressive generation
  feeds the model.

  ## Strategies

    * `greedy/1` — argmax over the vocab axis. Deterministic.
    * `sample/2` — temperature + top-k + top-p sampling. Pass options
      to control the strategy.

  ## Example

      logits = forward_model.(tokens)            # {vocab}
      next_id = NxArm.Sampling.sample(logits, top_k: 40, top_p: 0.9, temperature: 0.8)
  """

  @doc """
  Greedy: argmax over the last axis. Returns an integer token id.
  """
  def greedy(%Nx.Tensor{} = logits) do
    logits
    |> Nx.argmax(axis: -1)
    |> Nx.to_number()
  end

  @doc """
  Sample from logits with temperature + top-k + top-p filtering.

  Returns an integer token id.

  ## Options

    * `:temperature` — float, default 1.0. 0 falls through to greedy.
    * `:top_k` — int (or nil). Keep only the top-k logits.
    * `:top_p` — float in (0, 1] (or nil). Keep the smallest set of
      tokens whose cumulative probability ≥ top_p.
  """
  def sample(%Nx.Tensor{} = logits, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    top_k = Keyword.get(opts, :top_k)
    top_p = Keyword.get(opts, :top_p)
    rep_penalty = Keyword.get(opts, :repetition_penalty)
    rep_tokens = Keyword.get(opts, :recent_tokens, [])

    cond do
      temperature == 0.0 and rep_penalty in [nil, 1.0] ->
        greedy(logits)

      true ->
        flat = Nx.flatten(logits) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

        flat =
          if rep_penalty && rep_penalty != 1.0 && rep_tokens != [],
            do: apply_repetition_penalty(flat, rep_tokens, rep_penalty),
            else: flat

        if temperature == 0.0 do
          # Repetition-penalty-modified greedy.
          {idx, _} = Enum.with_index(flat) |> Enum.max_by(fn {v, _} -> v end) |> then(&{elem(&1, 1), elem(&1, 0)})
          idx
        else
          scaled = Enum.map(flat, &(&1 / temperature))

          scaled =
            if top_k && top_k < length(scaled),
              do: apply_top_k_list(scaled, top_k),
              else: scaled

          probs = softmax_list(scaled)

          probs =
            if top_p,
              do: apply_top_p_list(probs, top_p),
              else: probs

          draw_categorical(probs)
        end
    end
  end

  @doc """
  Apply a CTRL-style repetition penalty: for every token id in
  `recent_tokens`, divide its logit by `penalty` if positive, multiply
  if negative. Penalty > 1.0 discourages repetition; 1.0 is a no-op.
  Returns the modified logits as an Elixir list.
  """
  @spec apply_repetition_penalty([number()], [non_neg_integer()], number()) :: [float()]
  def apply_repetition_penalty(logits, recent_tokens, penalty)
      when is_list(logits) and penalty > 0 do
    seen = MapSet.new(recent_tokens)

    Enum.with_index(logits)
    |> Enum.map(fn {logit, idx} ->
      if MapSet.member?(seen, idx) do
        if logit > 0, do: logit / penalty, else: logit * penalty
      else
        logit * 1.0
      end
    end)
  end

  @doc """
  Stream of generated tokens.

  `step_fn` is `(state, last_token) -> {logits, new_state}` where
  `logits` is a 1-D tensor `{vocab}`. The stream emits one token
  per step.

      tokens =
        NxArm.Sampling.generate(initial_state, &model_step/2,
          start_token: bos,
          sampling: [top_k: 50, top_p: 0.95, temperature: 0.8]
        )
        |> Stream.take_while(&(&1 != eos_token))
        |> Enum.to_list()
  """
  def generate(state, step_fn, opts) do
    start = Keyword.fetch!(opts, :start_token)
    sampling = Keyword.get(opts, :sampling, [])

    Stream.unfold({state, start}, fn {st, last} ->
      {logits, new_state} = step_fn.(st, last)
      next = sample(logits, sampling)
      {next, {new_state, next}}
    end)
  end

  # ── internals (work on Elixir lists for clarity + correctness) ──

  defp softmax_list(scores) do
    m = Enum.max(scores)
    exps = Enum.map(scores, &:math.exp(&1 - m))
    total = Enum.sum(exps)
    Enum.map(exps, &(&1 / total))
  end

  defp apply_top_k_list(scores, k) do
    threshold =
      scores
      |> Enum.sort(:desc)
      |> Enum.at(k - 1)

    Enum.map(scores, fn s -> if s >= threshold, do: s, else: -1.0e30 end)
  end

  defp apply_top_p_list(probs, p) do
    indexed = Enum.with_index(probs)
    sorted = Enum.sort_by(indexed, fn {pr, _} -> -pr end)

    {kept, _} =
      Enum.reduce(sorted, {[], 0.0}, fn {pr, idx}, {acc, cum} ->
        if cum < p do
          {[{pr, idx} | acc], cum + pr}
        else
          {acc, cum}
        end
      end)

    keep_idx = MapSet.new(Enum.map(kept, fn {_, idx} -> idx end))

    masked =
      Enum.with_index(probs)
      |> Enum.map(fn {pr, i} -> if MapSet.member?(keep_idx, i), do: pr, else: 0.0 end)

    total = Enum.sum(masked)
    Enum.map(masked, &(&1 / max(total, 1.0e-30)))
  end

  defp draw_categorical(probs) do
    u = :rand.uniform()

    {idx, _} =
      Enum.reduce_while(Enum.with_index(probs), {0, 0.0}, fn {p, i}, {_, acc} ->
        new_acc = acc + p
        if new_acc >= u, do: {:halt, {i, new_acc}}, else: {:cont, {i, new_acc}}
      end)

    idx
  end
end

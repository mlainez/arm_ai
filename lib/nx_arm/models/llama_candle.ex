defmodule NxArm.Models.LlamaCandle do
  @moduledoc """
  Llama-family inference via the upstream `candle` crate.

  Per the "use the crate if it's faster" policy: candle's Q4_0 /
  Q4_K / Q8_0 matmul kernels are NEON-tuned across more ARM cores
  than we have, its Llama implementation is widely-used + tested,
  and its KV cache is in-place (no per-step BEAM↔NIF round-trips).

  This module is the recommended path for running Llama / TinyLlama /
  SmolLM / Phi / Mistral / Qwen on Nerves. Keep `NxArm.Models.Llama`
  around for users who want the pure-Nx integration path.

      {:ok, model} = NxArm.Models.LlamaCandle.load("/root/tinyllama.gguf")
      {tokens, stats} = NxArm.Models.LlamaCandle.generate(model,
        prompt_tokens: [1, 1724, 338, 263],
        max_new: 32
      )
  """

  defstruct [:handle]

  @doc "Load a GGUF Llama-family model via candle."
  @spec load(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path) do
    try do
      handle = NxArm.Native.llama_candle_load_op(path)
      {:ok, %__MODULE__{handle: handle}}
    rescue
      e -> {:error, e}
    catch
      :error, reason -> {:error, reason}
    end
  end

  @doc """
  Greedy-decode `max_new` tokens after `prompt_tokens`. Returns
  `{all_tokens, stats}` where `all_tokens` is `prompt ++ generated`
  and `stats` carries timing (matches `NxArm.Models.Llama.generate`).

  Wraps the call in `NxArm.Performance.with_performance/1` so the
  CPU governor flips to `performance` for the burst and restores
  on exit (default; pass `performance_governor: false` to disable).
  """
  @spec generate(%__MODULE__{}, keyword()) :: {[non_neg_integer()], map()}
  def generate(%__MODULE__{handle: handle}, opts) do
    prompt = Keyword.fetch!(opts, :prompt_tokens)
    max_new = Keyword.get(opts, :max_new, 16)
    scope_governor? = Keyword.get(opts, :performance_governor, true)

    do_run = fn ->
      {tokens, prefill_us, decode_us} =
        NxArm.Native.llama_candle_generate_op(handle, prompt, max_new)

      all = prompt ++ tokens
      n_new = length(tokens)

      stats = %{
        n_prompt: length(prompt),
        n_new: n_new,
        prefill_ms: prefill_us / 1000.0,
        decode_total_ms: decode_us / 1000.0,
        decode_ms_per_tok:
          if(n_new > 1, do: decode_us / 1000.0 / max(n_new - 1, 1), else: nil),
        prefill_tokens_per_sec: length(prompt) * 1.0e6 / max(prefill_us, 1),
        cpu_temp_c: NxArm.Performance.max_cpu_temp_c()
      }

      {all, stats}
    end

    if scope_governor? do
      NxArm.Performance.with_performance(do_run)
    else
      do_run.()
    end
  end
end

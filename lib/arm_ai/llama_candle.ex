defmodule ArmAI.LlamaCandle do
  @moduledoc """
  Llama-family inference via the upstream `candle` crate.

  Per the "use the crate if it's faster" policy: candle's Q4_0 /
  Q4_K / Q8_0 matmul kernels are NEON-tuned across more ARM cores
  than we have, its Llama implementation is widely-used + tested,
  and its KV cache is in-place (no per-step BEAM↔NIF round-trips).

  This module is the recommended path for running Llama / TinyLlama /
  SmolLM / Phi / Mistral / Qwen on Nerves. Keep `ArmAI.Llama`
  around for users who want the pure-Nx integration path.

      {:ok, model} = ArmAI.LlamaCandle.load("/root/tinyllama.gguf")
      {tokens, stats} = ArmAI.LlamaCandle.generate(model,
        prompt_tokens: [1, 1724, 338, 263],
        max_new: 32
      )
  """

  defstruct [:handle, :tokenizer]

  @doc """
  Load a GGUF Llama-family model via candle. Optionally also load a
  `tokenizer.json` so `generate/2` can take string prompts.

      {:ok, model} = ArmAI.LlamaCandle.load("/root/tinyllama.gguf",
                       tokenizer: "/root/tinyllama-tokenizer.json")

  Requires the `llm` Cargo feature (default-on). If the binary was
  built with `features: []` you'll get `{:error, :llm_feature_disabled}`.
  """
  @spec load(Path.t(), keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path, opts \\ []) do
    if not function_exported?(ArmAI.Native, :llama_candle_load_op, 1) do
      {:error, :llm_feature_disabled}
    else
    try do
      case ArmAI.Native.llama_candle_load_op(path) do
        {:error, reason} ->
          {:error, reason}

        handle ->
          tokenizer =
            case Keyword.get(opts, :tokenizer) do
              nil ->
                nil

              tok_path ->
                case Tokenizers.Tokenizer.from_file(tok_path) do
                  {:ok, tok} -> tok
                  {:error, _} -> nil
                end
            end

          {:ok, %__MODULE__{handle: handle, tokenizer: tokenizer}}
      end
    rescue
      e -> {:error, e}
    end
    end
  end

  @doc """
  Greedy-decode `max_new` tokens after `prompt_tokens`. Returns
  `{all_tokens, stats}` where `all_tokens` is `prompt ++ generated`
  and `stats` carries timing (matches `ArmAI.Llama.generate`).

  Wraps the call in `ArmAI.Performance.with_performance/1` so the
  CPU governor flips to `performance` for the burst and restores
  on exit (default; pass `performance_governor: false` to disable).
  """
  @spec generate(%__MODULE__{}, keyword()) :: {[non_neg_integer()] | String.t(), map()}
  def generate(%__MODULE__{handle: handle, tokenizer: tokenizer} = model, opts) do
    {prompt, return_string?} =
      cond do
        text = Keyword.get(opts, :prompt) ->
          if tokenizer == nil do
            raise ArgumentError,
                  "load/2 with `tokenizer:` path to use string :prompt — got plain text but no tokenizer"
          end

          {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
          {Tokenizers.Encoding.get_ids(encoding), true}

        prompt = Keyword.get(opts, :prompt_tokens) ->
          {prompt, false}

        true ->
          raise ArgumentError, "pass :prompt (string) or :prompt_tokens (list)"
      end

    _ = model
    max_new = Keyword.get(opts, :max_new, 16)
    scope_governor? = Keyword.get(opts, :performance_governor, true)

    do_run = fn ->
      {tokens, prefill_us, decode_us} =
        ArmAI.Native.llama_candle_generate_op(handle, prompt, max_new)

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
        cpu_temp_c: ArmAI.Performance.max_cpu_temp_c()
      }

      result =
        if return_string? do
          {:ok, text} = Tokenizers.Tokenizer.decode(tokenizer, all)
          text
        else
          all
        end

      {result, stats}
    end

    if scope_governor? do
      ArmAI.Performance.with_performance(do_run)
    else
      do_run.()
    end
  end
end

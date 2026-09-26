defmodule ArmAI.LlamaCandle do
  @moduledoc """
  Quantized Llama-architecture inference via the upstream `candle` crate.

  Loads GGUF files whose metadata uses the `llama.*` keys (Llama,
  TinyLlama, SmolLM, Mistral-style exports). Other architectures
  (Phi, Qwen, Gemma) are not supported by candle's quantized Llama
  loader. Decoding is greedy.

      {:ok, model} = ArmAI.LlamaCandle.load("/data/models/tinyllama.gguf",
                       tokenizer: "/data/models/tinyllama-tokenizer.json")
      {reply, stats} = ArmAI.LlamaCandle.generate(model,
        prompt: "<|user|>\\nWhat is Elixir?</s>\\n<|assistant|>\\n",
        max_new: 64,
        stop_tokens: [2]
      )

  Requires the `llm` Cargo feature (on by default).
  """

  defstruct [:handle, :tokenizer]

  @type t :: %__MODULE__{handle: reference(), tokenizer: Tokenizers.Tokenizer.t() | nil}

  @doc """
  Load a GGUF model. Pass `tokenizer:` (a `tokenizer.json` path) to let
  `generate/2` take and return strings.

  Returns `{:error, reason}` if the model or the tokenizer can't be loaded.
  """
  @spec load(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load(path, opts \\ []) do
    with {:ok, tokenizer} <- load_tokenizer(Keyword.get(opts, :tokenizer)),
         {:ok, handle} <- load_model(path) do
      {:ok, %__MODULE__{handle: handle, tokenizer: tokenizer}}
    end
  end

  defp load_tokenizer(nil), do: {:ok, nil}

  defp load_tokenizer(path) do
    case Tokenizers.Tokenizer.from_file(path) do
      {:ok, tok} -> {:ok, tok}
      {:error, reason} -> {:error, {:tokenizer, reason}}
    end
  end

  defp load_model(path) do
    if File.regular?(path) do
      {:ok, ArmAI.Native.llama_candle_load_op(path)}
    else
      {:error, {:enoent, path}}
    end
  rescue
    e in ErlangError -> {:error, e.original}
    e -> {:error, e}
  end

  @doc """
  Greedy-decode up to `max_new` tokens.

  ## Options

    * `:prompt` — a string (requires `tokenizer:` at load time), or
    * `:prompt_tokens` — a list of token ids.
    * `:max_new` — maximum number of generated tokens. Default 16.
    * `:stop_tokens` — token ids that end generation (e.g. the model's
      EOS id). The stop token is not included in the result. Default `[]`.
    * `:performance_governor` — scope the `performance` CPU governor
      to the call via `CpuGovernor.Performance`. Default `true`.

  Returns `{generated, stats}`. `generated` holds only the new tokens:
  a string when `:prompt` was given, a list of ids otherwise.
  """
  @spec generate(t(), keyword()) :: {[non_neg_integer()] | String.t(), map()}
  def generate(%__MODULE__{handle: handle, tokenizer: tokenizer}, opts) do
    {prompt, return_string?} =
      cond do
        text = Keyword.get(opts, :prompt) ->
          if tokenizer == nil do
            raise ArgumentError, "a string :prompt requires load/2 with a `tokenizer:` path"
          end

          {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
          {Tokenizers.Encoding.get_ids(encoding), true}

        prompt = Keyword.get(opts, :prompt_tokens) ->
          {prompt, false}

        true ->
          raise ArgumentError, "pass :prompt (string) or :prompt_tokens (list)"
      end

    max_new = Keyword.get(opts, :max_new, 16)
    stop_tokens = Keyword.get(opts, :stop_tokens, [])

    do_run = fn ->
      {tokens, prefill_us, decode_us} =
        ArmAI.Native.llama_candle_generate_op(handle, prompt, max_new, stop_tokens)

      tokens =
        case List.last(tokens) do
          nil -> tokens
          last -> if last in stop_tokens, do: List.delete_at(tokens, -1), else: tokens
        end

      n_new = length(tokens)

      stats = %{
        n_prompt: length(prompt),
        n_new: n_new,
        prefill_ms: prefill_us / 1000.0,
        decode_total_ms: decode_us / 1000.0,
        decode_ms_per_tok: if(n_new > 1, do: decode_us / 1000.0 / (n_new - 1), else: nil),
        prefill_tokens_per_sec: length(prompt) * 1.0e6 / max(prefill_us, 1),
        cpu_temp_c: CpuGovernor.Performance.max_cpu_temp_c()
      }

      result =
        if return_string? do
          {:ok, text} = Tokenizers.Tokenizer.decode(tokenizer, tokens)
          text
        else
          tokens
        end

      {result, stats}
    end

    if Keyword.get(opts, :performance_governor, true) do
      CpuGovernor.Performance.with_performance(do_run)
    else
      do_run.()
    end
  end
end

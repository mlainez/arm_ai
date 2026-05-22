defmodule ArmAI.LLMBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferLLM.Backend` — Whisper STT via
  `candle-transformers`.

      config :infer_llm, backend: ArmAI.LLMBackend

  Decoder LLMs (Llama / TinyLlama / SmolLM / Phi / Qwen / Mistral)
  go through `ArmAI.LlamaCandle` directly with the binary token API
  — there's no Nx-tensor LLM wrapper because the candle path doesn't
  need one (token IDs in, token IDs out).
  """

  if Code.ensure_loaded?(InferLLM.Backend) do
    @behaviour InferLLM.Backend
  end

  @impl true
  def whisper_load(opts) do
    model = Keyword.fetch!(opts, :model)
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    mel = Keyword.fetch!(opts, :mel_filters)
    config = Keyword.fetch!(opts, :config)

    if not function_exported?(ArmAI.Native, :whisper_load_op, 4) do
      {:error, :whisper_feature_disabled}
    else
      try do
        case ArmAI.Native.whisper_load_op(model, tokenizer, mel, config) do
          {:error, reason} -> {:error, reason}
          handle -> {:ok, handle}
        end
      rescue
        e -> {:error, e}
      end
    end
  end

  @impl true
  def whisper_transcribe(handle, pcm, opts) do
    bin =
      cond do
        is_binary(pcm) -> pcm
        match?(%Nx.Tensor{}, pcm) -> Nx.to_binary(pcm)
      end

    run = fn -> ArmAI.Native.whisper_transcribe_op(handle, bin) end

    result =
      if Keyword.get(opts, :performance_governor, true) do
        ArmAI.Performance.with_performance(run)
      else
        run.()
      end

    case result do
      {:error, _} = err -> err
      text when is_binary(text) -> {:ok, text}
      other -> {:ok, other}
    end
  end
end

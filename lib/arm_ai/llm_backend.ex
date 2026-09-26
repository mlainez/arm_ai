defmodule ArmAI.LLMBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferLLM.Backend`: Whisper speech-to-text
  via `candle-transformers` (requires the `whisper` Cargo feature).

      config :infer_llm, backend: ArmAI.LLMBackend

  Decoder LLMs go through `ArmAI.LlamaCandle` directly with the token
  API; there is no Nx-tensor LLM wrapper.
  """

  # Implements `InferLLM.Backend`. The behaviour isn't declared because that
  # package depends on arm_ai, so it isn't loaded when this compiles;
  # nerves_ai's test suite checks every callback is present.

  def whisper_load(opts) do
    model = Keyword.fetch!(opts, :model)
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    mel = Keyword.fetch!(opts, :mel_filters)
    config = Keyword.fetch!(opts, :config)

    case ArmAI.Native.whisper_load_op(model, tokenizer, mel, config) do
      {:error, reason} -> {:error, reason}
      handle -> {:ok, handle}
    end
  rescue
    e in ErlangError -> {:error, e.original}
    e -> {:error, e}
  end

  def whisper_transcribe(_handle, {:error, _} = err, _opts), do: err

  def whisper_transcribe(handle, pcm, opts) do
    bin =
      case pcm do
        bin when is_binary(bin) -> bin
        %Nx.Tensor{} = t -> t |> Nx.as_type(:f32) |> Nx.to_binary()
      end

    run = fn -> ArmAI.Native.whisper_transcribe_op(handle, bin) end

    result =
      if Keyword.get(opts, :performance_governor, true) do
        CpuGovernor.Performance.with_performance(run)
      else
        run.()
      end

    case result do
      {:error, _} = err -> err
      text when is_binary(text) -> {:ok, String.trim(text)}
    end
  rescue
    e in ErlangError -> {:error, e.original}
  end
end

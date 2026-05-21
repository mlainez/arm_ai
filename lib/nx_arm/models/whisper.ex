defmodule NxArm.Models.Whisper do
  @moduledoc """
  Speech-to-text via `candle-transformers/whisper`.

  Loads a Whisper model (safetensors or GGUF), the matching
  `tokenizer.json`, the precomputed mel filter banks, and the
  `config.json`. Transcribes a 16 kHz mono f32 audio buffer to text.

      {:ok, w} = NxArm.Models.Whisper.load(
        model:        "/root/whisper-tiny.gguf",
        tokenizer:    "/root/whisper-tokenizer.json",
        mel_filters:  "/root/mel_filters.bin",
        config:       "/root/whisper-config.json"
      )

      pcm  = NxArm.Audio.load_for_whisper("/root/clip.mp3")
      text = NxArm.Models.Whisper.transcribe(w, pcm)

  Mel filters are shipped alongside Whisper checkpoints; download
  the `mel_filters.npz` or `melfilters.bytes` from the HuggingFace
  repo and pass it here.

  Greedy decode covers the first 30 s only — chunked /
  timestamp-based decoding is future work. Requires the `whisper`
  Cargo feature (default-on).
  """

  defstruct [:handle]

  @doc "Load a Whisper checkpoint with its tokenizer + mel filters + config."
  @spec load(keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(opts) do
    model = Keyword.fetch!(opts, :model)
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    mel = Keyword.fetch!(opts, :mel_filters)
    config = Keyword.fetch!(opts, :config)

    if not function_exported?(NxArm.Native, :whisper_load_op, 4) do
      {:error, :whisper_feature_disabled}
    else
      try do
        case NxArm.Native.whisper_load_op(model, tokenizer, mel, config) do
          {:error, reason} -> {:error, reason}
          handle -> {:ok, %__MODULE__{handle: handle}}
        end
      rescue
        e -> {:error, e}
      end
    end
  end

  @doc """
  Transcribe a 16 kHz mono f32 audio tensor (or raw f32 binary) to
  text. Wraps in the scoped CPU governor by default.
  """
  @spec transcribe(%__MODULE__{}, Nx.Tensor.t() | binary(), keyword()) :: String.t()
  def transcribe(%__MODULE__{handle: handle}, pcm, opts \\ []) do
    bin =
      cond do
        is_binary(pcm) -> pcm
        match?(%Nx.Tensor{}, pcm) -> NxArm.Backend.__bin_of__(pcm)
      end

    run = fn -> NxArm.Native.whisper_transcribe_op(handle, bin) end

    if Keyword.get(opts, :performance_governor, true) do
      NxArm.Performance.with_performance(run)
    else
      run.()
    end
  end
end

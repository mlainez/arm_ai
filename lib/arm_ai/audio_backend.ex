defmodule ArmAI.AudioBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferAudio.Backend` — symphonia for
  decode, rubato for resample, tract-onnx for Silero VAD and
  Piper TTS.

      config :infer_audio, backend: ArmAI.AudioBackend
  """

  if Code.ensure_loaded?(InferAudio.Backend) do
    @behaviour InferAudio.Backend
  end

  @impl true
  def decode_file(path) do
    if not function_exported?(ArmAI.Native, :audio_decode_file_op, 1) do
      {:error, :audio_feature_disabled}
    else
      case ArmAI.Native.audio_decode_file_op(path) do
        {:error, _} = err ->
          err

        {bin, sample_rate, channels} ->
          samples = Nx.from_binary(bin, :f32)
          {:ok, %{samples: samples, sample_rate: sample_rate, channels: channels}}
      end
    end
  end

  @impl true
  def resample(samples, from_hz, to_hz) when from_hz == to_hz, do: samples

  def resample(samples, from_hz, to_hz) do
    bin = Nx.to_binary(samples)
    out_bin = ArmAI.Native.audio_resample_op(bin, from_hz, to_hz)
    Nx.from_binary(out_bin, :f32)
  end

  @impl true
  def load_for_whisper(path) do
    if not function_exported?(ArmAI.Native, :audio_load_for_whisper_op, 1) do
      {:error, :audio_feature_disabled}
    else
      case ArmAI.Native.audio_load_for_whisper_op(path) do
        {:error, _} = err -> err
        bin when is_binary(bin) -> Nx.from_binary(bin, :f32)
      end
    end
  end

  @impl true
  def write_wav(path, samples, opts) do
    sample_rate = Keyword.get(opts, :sample_rate, 22_050)
    channels = Keyword.get(opts, :channels, 1)

    bin =
      cond do
        is_binary(samples) -> samples
        match?(%Nx.Tensor{}, samples) -> Nx.to_binary(samples)
      end

    ArmAI.Native.audio_write_wav_op(path, bin, sample_rate, channels)
  end

  @impl true
  def silero_vad_load(path) do
    if not function_exported?(ArmAI.Native, :onnx_load_op, 1) do
      {:error, :onnx_feature_disabled}
    else
      case ArmAI.Native.onnx_load_op(path) do
        {:error, _} = err -> err
        handle -> {:ok, handle}
      end
    end
  end

  @impl true
  def piper_load(path, _opts) do
    if not function_exported?(ArmAI.Native, :onnx_load_op, 1) do
      {:error, :onnx_feature_disabled}
    else
      case ArmAI.Native.onnx_load_op(path) do
        {:error, _} = err -> err
        handle -> {:ok, handle}
      end
    end
  end
end

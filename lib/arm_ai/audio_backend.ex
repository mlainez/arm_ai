defmodule ArmAI.AudioBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferAudio.Backend`: symphonia for
  decode, rubato for resample, and a minimal PCM WAV writer.

      config :infer_audio, backend: ArmAI.AudioBackend
  """

  # Implements `InferAudio.Backend`. The behaviour isn't declared because that
  # package depends on arm_ai, so it isn't loaded when this compiles;
  # nerves_ai's test suite checks every callback is present.

  # ---------------- audio I/O ----------------

  def decode_file(path) do
    case ArmAI.Native.audio_decode_file_op(path) do
      {:error, _} = err ->
        err

      {bin, sample_rate, channels} ->
        {:ok, %{samples: Nx.from_binary(bin, :f32), sample_rate: sample_rate, channels: channels}}
    end
  rescue
    e in ErlangError -> {:error, e.original}
  end

  def resample(samples, from_hz, to_hz) when from_hz == to_hz, do: samples

  def resample(samples, from_hz, to_hz) do
    samples
    |> Nx.as_type(:f32)
    |> Nx.to_binary()
    |> ArmAI.Native.audio_resample_op(from_hz, to_hz)
    |> Nx.from_binary(:f32)
  end

  def load_for_whisper(path) do
    case ArmAI.Native.audio_load_for_whisper_op(path) do
      {:error, _} = err -> err
      bin when is_binary(bin) -> Nx.from_binary(bin, :f32)
    end
  rescue
    e in ErlangError -> {:error, e.original}
  end

  def write_wav(path, samples, opts) do
    sample_rate = Keyword.get(opts, :sample_rate, 22_050)
    channels = Keyword.get(opts, :channels, 1)

    bin =
      case samples do
        bin when is_binary(bin) -> bin
        %Nx.Tensor{} = t -> t |> Nx.as_type(:f32) |> Nx.to_binary()
      end

    ArmAI.Native.audio_write_wav_op(path, bin, sample_rate, channels)
  end
end

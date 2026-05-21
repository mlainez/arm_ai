defmodule NxArm.Audio do
  @moduledoc """
  Audio decode + resample helpers for speech models on Nerves.

  Wraps `symphonia` (decode MP3/WAV/FLAC/Opus/OGG/Vorbis/PCM) and
  `rubato` (sinc-resampler, the standard rate-converter in
  Whisper-class pipelines).

      # One-shot: file path → Whisper-ready 16 kHz mono f32 tensor.
      samples = NxArm.Audio.load_for_whisper("/data/clip.mp3")
      # samples is an Nx tensor on NxArm.Backend, shape {n_samples}.

      # Or step-by-step when you need intermediate state:
      {raw, sr, channels} = NxArm.Audio.decode_file("/data/clip.wav")
      mono = NxArm.Audio.to_mono(raw, channels)
      sixteen = NxArm.Audio.resample(mono, sr, 16_000)

  Returns Nx tensors so the result flows into Bumblebee/Axon
  speech models, candle-whisper, or anything else.

  Requires the `audio` Cargo feature (default-on).
  """

  @doc """
  Decode an audio file (any format symphonia understands) into
  interleaved f32 samples + the source sample rate + channel count.
  Returns `{tensor, sample_rate, channels}` where `tensor` is shape
  `{n_samples * channels}` on `NxArm.Backend`.
  """
  @spec decode_file(Path.t()) :: {Nx.Tensor.t(), pos_integer(), pos_integer()}
  def decode_file(path) do
    if not function_exported?(NxArm.Native, :audio_decode_file_op, 1) do
      raise "NxArm built without the `audio` feature"
    end

    {bin, sr, channels} = NxArm.Native.audio_decode_file_op(path)
    samples = Nx.from_binary(bin, :f32) |> Nx.backend_copy(NxArm.Backend)
    {samples, sr, channels}
  end

  @doc "Mix multi-channel interleaved samples to mono by averaging."
  @spec to_mono(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def to_mono(samples, channels) do
    bin = NxArm.Backend.__bin_of__(samples)
    out_bin = NxArm.Native.audio_to_mono_op(bin, channels)
    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end

  @doc "Sinc-resample mono f32 samples from `from_hz` to `to_hz`."
  @spec resample(Nx.Tensor.t(), pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def resample(samples, from_hz, to_hz) do
    bin = NxArm.Backend.__bin_of__(samples)
    out_bin = NxArm.Native.audio_resample_op(bin, from_hz, to_hz)
    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  One-shot: decode → mono → resample to 16 kHz. Returns an Nx
  tensor of shape `{n_samples}` ready for Whisper / wav2vec2 /
  any 16 kHz speech model.
  """
  @spec load_for_whisper(Path.t()) :: Nx.Tensor.t()
  def load_for_whisper(path) do
    if not function_exported?(NxArm.Native, :audio_load_for_whisper_op, 1) do
      raise "NxArm built without the `audio` feature"
    end

    bin = NxArm.Native.audio_load_for_whisper_op(path)
    Nx.from_binary(bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end
end

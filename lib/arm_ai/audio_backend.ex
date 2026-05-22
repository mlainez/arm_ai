defmodule ArmAI.AudioBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferAudio.Backend`:

  * symphonia for decode, rubato for resample
  * tract-onnx (via `ArmAI.VisionBackend`'s internal helpers) for
    Silero VAD scoring and Piper TTS synthesis

      config :infer_audio, backend: ArmAI.AudioBackend
  """

  if Code.ensure_loaded?(InferAudio.Backend) do
    @behaviour InferAudio.Backend
  end

  # Defaults for Silero VAD V4 (32 ms × 16 kHz windows, h/c LSTM state).
  @silero_window_samples 512
  @silero_sample_rate 16_000

  # ---------------- audio I/O ----------------

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

  # ---------------- Silero VAD ----------------

  @impl true
  def silero_vad_load(path, _opts) do
    onnx_load_via_arm_ai(path)
  end

  @impl true
  def silero_vad_scores(handle, audio, _opts) do
    samples = audio |> Nx.flatten() |> Nx.backend_copy(Nx.BinaryBackend)
    n = Nx.size(samples)
    n_windows = div(n, @silero_window_samples)

    h0 = Nx.broadcast(0.0, {2, 1, 64})
    c0 = Nx.broadcast(0.0, {2, 1, 64})
    sr = Nx.tensor([@silero_sample_rate], type: :s64)

    {scores_acc, _h, _c} =
      Enum.reduce(0..(n_windows - 1), {[], h0, c0}, fn i, {acc, h, c} ->
        window =
          Nx.slice(samples, [i * @silero_window_samples], [@silero_window_samples])
          |> Nx.reshape({1, @silero_window_samples})

        outputs =
          onnx_run_via_arm_ai(handle, %{
            "input" => window,
            "sr" => sr,
            "h" => h,
            "c" => c
          })

        prob = outputs["output"] |> Nx.to_flat_list() |> hd()
        h_new = outputs["hn"] || h
        c_new = outputs["cn"] || c

        {[prob | acc], h_new, c_new}
      end)

    scores_acc |> Enum.reverse() |> Nx.tensor(type: :f32)
  end

  # ---------------- Piper TTS ----------------

  @impl true
  def piper_load(path, _opts) do
    onnx_load_via_arm_ai(path)
  end

  @impl true
  def piper_synthesize(handle, phoneme_ids, opts) do
    speaker_id = Keyword.get(opts, :speaker_id, 0)
    length_scale = Keyword.get(opts, :length_scale, 1.0)
    noise_scale = Keyword.get(opts, :noise_scale, 0.667)
    noise_w = Keyword.get(opts, :noise_w, 0.8)

    input_lengths = Nx.tensor([length(phoneme_ids)], type: :f32)

    input_ids =
      phoneme_ids
      |> Enum.map(&(&1 / 1))
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({1, length(phoneme_ids)})

    scales = Nx.tensor([noise_scale, length_scale, noise_w], type: :f32)
    speaker = Nx.tensor([speaker_id], type: :f32)

    inputs = %{
      "input" => input_ids,
      "input_lengths" => input_lengths,
      "scales" => scales,
      "sid" => speaker
    }

    outputs = onnx_run_via_arm_ai(handle, inputs)

    {_name, audio_tensor} = Enum.at(outputs, 0)
    Nx.flatten(audio_tensor)
  end

  # ---------------- helpers ----------------

  # Direct calls into ArmAI.Native — duplicated from
  # ArmAI.VisionBackend so InferAudio doesn't depend on InferVision.
  defp onnx_load_via_arm_ai(path) do
    if not function_exported?(ArmAI.Native, :onnx_load_op, 1) do
      {:error, :onnx_feature_disabled}
    else
      try do
        case ArmAI.Native.onnx_load_op(path) do
          {:error, _} = err -> err
          handle -> {:ok, handle}
        end
      rescue
        e -> {:error, e}
      end
    end
  end

  defp onnx_run_via_arm_ai(handle, inputs) when is_map(inputs) do
    bins =
      Enum.map(inputs, fn {name, tensor} ->
        {name, Nx.to_binary(tensor), Tuple.to_list(Nx.shape(tensor))}
      end)

    ArmAI.Native.onnx_run_op(handle, bins)
    |> Enum.into(%{}, fn {name, bin, shape, _dtype} ->
      {name, Nx.from_binary(bin, :f32) |> Nx.reshape(List.to_tuple(shape))}
    end)
  end
end

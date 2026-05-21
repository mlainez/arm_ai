defmodule NxArm.Models.Piper do
  @moduledoc """
  Text-to-speech via Piper VITS models (ONNX export from
  https://github.com/rhasspy/piper).

  Piper distributes its voices as `<voice>.onnx` + `<voice>.onnx.json`
  config. The ONNX graph takes phoneme IDs and emits f32 audio
  samples at the model's sample rate (typically 22050 Hz). We route
  it through the existing `NxArm.Models.Onnx` bridge — tract handles
  every op Piper uses.

  ## Synthesis pipeline

      {:ok, piper} = NxArm.Models.Piper.load("/root/en_US-amy-medium.onnx")
      # phoneme_ids: list of integers (the user / their phonemiser
      # produces these; eSpeak NG is the canonical source).
      samples = NxArm.Models.Piper.synthesize(piper, phoneme_ids)
      NxArm.Audio.write_wav("/data/out.wav", samples, sample_rate: 22050)

  Phonemisation is **not** done here — eSpeak NG is a C library and
  on-device phonemisation is rare. For most Nerves use cases the
  phonemes are computed on the host or via a service, then sent to
  the device.
  """

  defstruct [:onnx, :sample_rate]

  @doc """
  Load a Piper voice. `sample_rate` defaults to 22050 (Piper's
  default for medium-quality voices; some voices use 16000 — pass
  the value from the voice's `.onnx.json` config).
  """
  @spec load(Path.t(), keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path, opts \\ []) do
    sample_rate = Keyword.get(opts, :sample_rate, 22050)

    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model, sample_rate: sample_rate}}
      err -> err
    end
  end

  @doc """
  Synthesise audio from a list of phoneme IDs. Returns the f32 audio
  tensor (shape `{n_samples}`) at the model's sample rate.

  ## Options

    * `:speaker_id` — for multi-speaker voices (default 0)
    * `:length_scale` — speech rate (1.0 = normal, < 1 faster)
    * `:noise_scale` — Piper VITS noise injection (default 0.667)
    * `:noise_w` — duration predictor noise (default 0.8)
  """
  @spec synthesize(%__MODULE__{}, [non_neg_integer()], keyword()) :: Nx.Tensor.t()
  def synthesize(%__MODULE__{onnx: onnx}, phoneme_ids, opts \\ []) do
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

    inputs =
      Map.filter(inputs, fn {name, _} -> name in onnx.input_names end)

    outputs = NxArm.Models.Onnx.run(onnx, inputs)

    # Piper's single output is named "output" by convention; pick
    # whatever the model actually exposes and reshape to 1-D.
    {_, audio_tensor} = Enum.at(outputs, 0)
    audio_tensor |> Nx.flatten() |> Nx.backend_copy(NxArm.Backend)
  end
end

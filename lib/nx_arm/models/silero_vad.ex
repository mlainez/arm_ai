defmodule NxArm.Models.SileroVAD do
  @moduledoc """
  Voice activity detection via Silero VAD (ONNX export).

  Silero VAD is a 1.8 MB ONNX model that classifies 16 kHz audio
  windows as speech / non-speech. Runs through our existing
  `NxArm.Models.Onnx` bridge.

  Use case on Nerves: gate Whisper transcription on actually
  hearing a voice (saves cycles vs running Whisper on silence).

  ## Pipeline

      {:ok, vad} = NxArm.Models.SileroVAD.load("/root/silero_vad.onnx")
      pcm = NxArm.Audio.load_for_whisper("/data/clip.wav")
      segments = NxArm.Models.SileroVAD.detect(vad, pcm, threshold: 0.5)
      # segments = [%{start_ms: 320, end_ms: 1840}, ...]

  Download: <https://github.com/snakers4/silero-vad>
  """

  defstruct [:onnx]

  # Silero VAD processes audio in fixed 30 ms windows (480 samples
  # at 16 kHz). The model is stateful via an h/c LSTM hidden state.
  @window_samples 512
  @sample_rate 16_000

  @doc "Load the Silero VAD ONNX model."
  @spec load(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path) do
    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model}}
      err -> err
    end
  end

  @doc """
  Slide the model across 30 ms windows and return the per-window
  speech-probability tensor of shape `{n_windows}`.
  """
  @spec scores(%__MODULE__{}, Nx.Tensor.t()) :: Nx.Tensor.t()
  def scores(%__MODULE__{onnx: model}, audio) do
    samples = audio |> Nx.flatten() |> Nx.backend_copy(Nx.BinaryBackend)
    n = Nx.size(samples)
    n_windows = div(n, @window_samples)

    # State buffers — Silero V4 uses h, c each shape {2, 1, 64}.
    h0 = Nx.broadcast(0.0, {2, 1, 64})
    c0 = Nx.broadcast(0.0, {2, 1, 64})
    sr = Nx.tensor([@sample_rate], type: :s64)

    {scores_acc, _h, _c} =
      Enum.reduce(0..(n_windows - 1), {[], h0, c0}, fn i, {acc, h, c} ->
        window =
          Nx.slice(samples, [i * @window_samples], [@window_samples])
          |> Nx.reshape({1, @window_samples})
          |> Nx.backend_copy(NxArm.Backend)

        outputs =
          NxArm.Models.Onnx.run(model, %{
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

    scores_acc
    |> Enum.reverse()
    |> Nx.tensor(type: :f32)
    |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Run VAD and return speech segments as `[%{start_ms, end_ms}]`.

  Options:
    * `:threshold` — probability cutoff (default 0.5)
    * `:min_speech_ms` — drop segments shorter than this (default 250)
    * `:min_silence_ms` — gap below threshold needed to break a
      segment (default 100)
  """
  @spec detect(%__MODULE__{}, Nx.Tensor.t(), keyword()) ::
          [%{start_ms: non_neg_integer(), end_ms: non_neg_integer()}]
  def detect(%__MODULE__{} = vad, audio, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.5)
    min_speech_ms = Keyword.get(opts, :min_speech_ms, 250)
    min_silence_ms = Keyword.get(opts, :min_silence_ms, 100)

    window_ms = div(@window_samples * 1000, @sample_rate)

    probs = vad |> scores(audio) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

    {_state, segments_rev} =
      Enum.with_index(probs)
      |> Enum.reduce({nil, []}, fn {p, i}, {state, segs} ->
        is_speech = p >= threshold
        t = i * window_ms

        case {state, is_speech} do
          {nil, true} -> {{:speech, t, t + window_ms, 0}, segs}
          {nil, false} -> {nil, segs}
          {{:speech, s, _e, _sil}, true} -> {{:speech, s, t + window_ms, 0}, segs}
          {{:speech, s, e, sil}, false} ->
            new_sil = sil + window_ms
            if new_sil >= min_silence_ms do
              if e - s >= min_speech_ms do
                {nil, [%{start_ms: s, end_ms: e} | segs]}
              else
                {nil, segs}
              end
            else
              {{:speech, s, e, new_sil}, segs}
            end
        end
      end)

    Enum.reverse(segments_rev)
  end
end

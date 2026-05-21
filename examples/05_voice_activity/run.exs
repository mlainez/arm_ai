#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 05 — Voice activity detection with Silero VAD.
# Slides a 30 ms window across an audio file and returns the
# speech segments. Pair with example 02 to gate Whisper on actual
# speech.
# ---------------------------------------------------------------

audio_path = "/root/clip.wav"
model_path = "/root/models/silero_vad.onnx"

unless File.exists?(model_path) and File.exists?(audio_path) do
  IO.puts("Missing #{model_path} or #{audio_path}")
  IO.puts("See config.exs for the NxArm.Hub config that fetches Silero VAD (~1.8 MB).")
  System.halt(1)
end

IO.puts("Loading Silero VAD...")
{:ok, vad} = NxArm.Models.SileroVAD.load(model_path)

IO.puts("Decoding audio...")
pcm = NxArm.Audio.load_for_whisper(audio_path)

IO.puts("Running VAD over #{Float.round(Nx.size(pcm) / 16_000, 2)} s of audio...")
{us, segments} =
  :timer.tc(fn ->
    NxArm.Models.SileroVAD.detect(vad, pcm,
      threshold: 0.5,
      min_speech_ms: 250,
      min_silence_ms: 100
    )
  end)

IO.puts("")
IO.puts("Found #{length(segments)} speech segments in #{div(us, 1000)} ms:")

for %{start_ms: s, end_ms: e} <- segments do
  IO.puts("  [#{s} ms .. #{e} ms]  (#{e - s} ms)")
end

#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 02 — Voice transcription with Whisper.
#
# Decodes an audio file (anything symphonia understands) to 16 kHz
# mono and runs candle-whisper. Audio longer than 30 s is handled
# automatically (sliding-window decode).
# ---------------------------------------------------------------

audio_path = "/root/clip.wav"  # change to your file

model_path     = "/root/models/whisper-tiny.gguf"
tokenizer_path = "/root/models/whisper-tokenizer.json"
mel_path       = "/root/models/whisper-mel-filters.bin"
config_path    = "/root/models/whisper-config.json"

required = [model_path, tokenizer_path, mel_path, config_path, audio_path]

unless Enum.all?(required, &File.exists?/1) do
  IO.puts("Missing one or more required files:")
  for p <- required, do: IO.puts("  #{if File.exists?(p), do: "OK", else: "MISSING"}  #{p}")
  IO.puts("")
  IO.puts("See config.exs for the ArmAI.Hub config that downloads the model bundle.")
  System.halt(1)
end

IO.puts("Decoding audio at #{audio_path}...")
pcm = ArmAI.Audio.load_for_whisper(audio_path)
IO.puts("  → #{Nx.size(pcm)} samples at 16 kHz (#{Float.round(Nx.size(pcm) / 16_000, 2)} s)")

IO.puts("Loading Whisper...")
{:ok, whisper} =
  ArmAI.Whisper.load(
    model: model_path,
    tokenizer: tokenizer_path,
    mel_filters: mel_path,
    config: config_path
  )

IO.puts("Transcribing...")
{us, text} = :timer.tc(fn -> ArmAI.Whisper.transcribe(whisper, pcm) end)

IO.puts("")
IO.puts("Transcript (#{div(us, 1000)} ms):")
IO.puts("  #{text}")

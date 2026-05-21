#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 06 — Text-to-speech with Piper.
#
# Piper voices ship as ONNX. The model expects phoneme IDs as
# input (ARPAbet / eSpeak NG symbols). On Nerves the canonical
# pattern is to compute phonemes on the host side and ship phoneme
# IDs to the device — eSpeak NG is a C library that doesn't
# cross-compile cleanly.
#
# This example uses the built-in tiny English G2P
# (NxArm.Phonemizer.simple_english_phonemize) so the demo runs
# without any external dependency. Production users should plug
# in a real phonemizer.
# ---------------------------------------------------------------

model_path = "/root/models/piper-amy-medium.onnx"

unless File.exists?(model_path) do
  IO.puts("Missing #{model_path}.")
  IO.puts("See config.exs for the Piper voice download (~25 MB).")
  System.halt(1)
end

IO.puts("Loading Piper voice...")
{:ok, piper} = NxArm.Models.Piper.load(model_path, sample_rate: 22050)

text = "Hello world. This is your Nerves device speaking."

IO.puts("Phonemizing: \"#{text}\"")
phonemes = NxArm.Phonemizer.simple_english_phonemize(text)
IO.inspect(phonemes, label: "phonemes")

# A real Piper voice ships a phoneme_id_map in its .onnx.json config.
# For this example we use a stub: each unique ARPAbet symbol gets a
# sequential ID. Replace with the voice's real map.
unique = phonemes |> Enum.uniq()
phoneme_id_map = unique |> Enum.with_index() |> Enum.into(%{})

ids = NxArm.Phonemizer.to_phoneme_ids(phonemes, phoneme_id_map)
IO.inspect(ids, label: "phoneme IDs")

IO.puts("Synthesizing audio...")
{us, samples} = :timer.tc(fn -> NxArm.Models.Piper.synthesize(piper, ids) end)
IO.puts("  → #{Nx.size(samples)} samples (#{Float.round(Nx.size(samples) / 22_050, 2)} s) in #{div(us, 1000)} ms")

out_path = "/root/tts_output.wav"
:ok = NxArm.Audio.write_wav(out_path, samples, sample_rate: 22_050)
IO.puts("Wrote #{out_path}")

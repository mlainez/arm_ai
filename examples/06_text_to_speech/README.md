# 06 — Text-to-speech (Piper)

Generates a WAV file from text using a Piper voice (ONNX).

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. First boot pulls ~25 MB.

## How phoneme IDs work

A Piper voice ships two files:

- `voice.onnx` — the synthesizer
- `voice.onnx.json` — speaker config including `phoneme_id_map`

For a production app, parse the `.onnx.json` once at boot and
keep `phoneme_id_map` in memory. This example uses a stub map
(unique-symbol → sequential index) so the demo runs without
the json parser. The audio will play but will sound like noise
until the voice's real ID map is wired in.

## Phonemizer

Piper expects eSpeak NG phoneme symbols. Cross-compiling eSpeak NG
for Nerves is a pain; the recommended pattern is to:

1. Compute phonemes on the host / a server, ship phoneme IDs.
2. Or use `NxArm.Phonemizer.simple_english_phonemize/1` for
   quick demos (≈80 word dictionary, naive letter fallback).
3. Or implement `NxArm.Phonemizer.callback` to point at your own
   G2P (cmudict lookup, neural g2p, whatever).

## Honest status

Not yet tested end-to-end on FP3. The bridge calls Piper's ONNX
forward via `tract-onnx`; both the load and the synth path
compile clean. End-to-end playback verification requires a real
`phoneme_id_map` parse, which is left as a one-line JSON read for
the integrator.

## Expected output

```
Loading Piper voice...
Phonemizing: "Hello world. This is your Nerves device speaking."
phonemes: ["HH", "EH", "L", "OW", " ", "W", "ER", "L", "D", ".", ...]
phoneme IDs: [4, 11, 2, 7, 0, 14, 18, 2, 9, 22, ...]
Synthesizing audio...
  → 95040 samples (4.31 s) in 1820 ms
Wrote /root/tts_output.wav
```

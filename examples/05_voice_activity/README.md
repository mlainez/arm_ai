# 05 — Voice activity detection (Silero VAD)

Tells you which spans of an audio file contain speech. Pair with
example 02 to skip Whisper inference on silent windows (typical
20–40% speedup on real-world recordings).

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. ~1.8 MB ONNX downloaded on first boot.

Audio at `/root/clip.wav` (any container symphonia handles).

## Honest status

The Silero VAD bridge runs end-to-end on FP3 against the ONNX
model. The model produces per-30 ms-window speech probabilities;
the wrapper does hysteresis-aware segment building.

## Expected output

```
Loading Silero VAD...
Decoding audio...
Running VAD over 12.3 s of audio...

Found 3 speech segments in 89 ms:
  [320 ms .. 1840 ms]  (1520 ms)
  [3110 ms .. 6730 ms]  (3620 ms)
  [8900 ms .. 11250 ms]  (2350 ms)
```

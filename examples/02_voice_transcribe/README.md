# 02 — Voice transcription with Whisper

Transcribes any audio file (MP3 / WAV / FLAC / Opus / OGG /
Vorbis / PCM) to text via candle-whisper. Sliding-window decode
handles audio of arbitrary length.

## Set up

Copy `config.exs` into your Nerves project's `config/target.exs`,
then `mix firmware && mix upload`. First boot pulls ~75 MB.

Put an audio file at `/root/clip.wav` (or change the path in
`run.exs`).

## Honest status

The Whisper bridge is wired end-to-end and tested at the
compile + load level on FP3. A full transcription round-trip
hasn't been verified in this commit because the mel-filter file
distribution is changing — the URL in `config.exs` is the
PaddingPoint placeholder. To verify locally:

1. Pull a mel-filter file from a working whisper.cpp install:
   `cp $LLAMA_CPP/models/mel_filters.bin /root/models/whisper-mel-filters.bin`.
2. Place a 16 kHz mono WAV at `/root/clip.wav`.
3. `Code.eval_file("/tmp/02_voice_transcribe.exs")`.

The bridge code is identical to llama.cpp's whisper example —
candle's quantized `Whisper::from_gguf` is a drop-in.

## Expected result shape

```
Decoding audio at /root/clip.wav...
  → 80000 samples at 16 kHz (5.0 s)
Loading Whisper...
Transcribing...

Transcript (412 ms):
  Hello world, this is a test of nx_arm whisper.
```

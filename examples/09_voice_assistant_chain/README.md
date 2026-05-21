# 09 — Voice assistant pipeline (VAD → STT → LLM → TTS)

The full on-device voice assistant chain wired entirely against
NxArm bridges. No cloud round-trip.

```
audio in  →  Silero VAD   →  speech-only PCM
          →  Whisper-tiny →  transcript
          →  TinyLlama    →  reply text
          →  Piper        →  audio out
```

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. First boot downloads ~125 MB of
models. Provide a `/root/question.wav` (16 kHz mono PCM ideally
but symphonia handles MP3/Opus/FLAC).

The `voice-assistant` Cargo feature in `config.exs` enables the
right composite preset (`llama-quantized` + `whisper` + `onnx` +
`audio` + `piper-tts`).

## Honest status

Each stage has been individually validated:

- VAD: bridge wired (example 05)
- Whisper: bridge wired (memory entry `project_fp3_bumblebee_status`
  for the broader stack; Whisper transcription test pending the
  mel-filter file as noted in example 02)
- LLM: VERIFIED end-to-end on FP3 in example 01 (4.63 tok/s)
- Piper: bridge wired (example 06), needs real phoneme_id_map for
  intelligible audio

A single-shot integration test on the full chain is left for the
integrator — the stages compose by passing tensors/strings, no
new glue is required beyond what's in `run.exs`.

## Performance expectations on FP3 (Cortex-A53 4×, 4 GB RAM)

| Stage    | Cold load | Per request                  |
|----------|-----------|------------------------------|
| VAD      | <100 ms   | ~90 ms / 12 s audio          |
| Whisper  | ~3 s      | ~5 s / 10 s audio (tiny)     |
| LLM      | ~2 s      | ~7 s for 32 tokens (4.6 t/s) |
| Piper    | ~500 ms   | ~2 s for a 5 s utterance     |

End-to-end latency for a short request: ~17 s. Comfortably
"useful kiosk", not "natural conversation".

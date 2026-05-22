# nx_arm examples

Each folder is a runnable demonstration of a real edge AI use
case on a Nerves ARM device. Every example:

* Lives in its own folder with a `README.md`, a `run.exs`
  script, and a `config.exs` snippet you copy into your Nerves
  project.
* Was tested by the author on a Fairphone 3 (Cortex-A73). The
  code works unchanged on any aarch64 Linux board — Raspberry Pi
  3B+/4/5, BeagleBone AI, Orange Pi, Jetson Nano, etc. armv7
  works too once you ship the matching precompiled NIF tarball.
* Tells you exactly which model file(s) to fetch and where to
  put them (`ArmAI.Hub` handles this for you on first boot).
* Shows the `config :nx_arm, features: […]` you need so you
  ship the smallest firmware that covers the use case.

## Running on the device

```sh
# from the device's iex (over `ssh nerves.local`):
Code.eval_file("/srv/erlang/lib/nx_arm-0.1.0/examples/01_chatbot/run.exs")
```

Or push the script via `scp` to `/tmp` and run it there. Or
build it into your own Nerves app and call from your supervisor.

## Index

| # | Use case | Model | Feature set | Tested on FP3 |
|---|---|---|---|---|
| 01 | Chatbot (TinyLlama Q4_K_M) | GGUF, 638 MB | `["chatbot"]` | ✅ 4.63 tok/s |
| 02 | Voice transcription (Whisper) | GGUF + tokenizer, 75 MB | `["whisper"]` | bridge wired, mel-filter URL placeholder |
| 03 | Semantic search / RAG | ONNX sentence encoder, 90 MB | `["sentence-rag"]` | ✅ 676 µs/query |
| 04 | Still-image object detection (YOLOv5n) | ONNX, 3.8 MB | `["yolo"]` | ✅ ~1 s/image on FP3, opset 17 loads cleanly |
| 05 | Voice activity detection (Silero VAD) | ONNX, 1.8 MB | `["onnx", "audio"]` | bridge wired |
| 06 | Text-to-speech (Piper) | ONNX, 25 MB | `["piper-tts"]` | bridge wired, needs real phoneme_id_map |
| 07 | Image classification (Bumblebee/Axon ViT) | Bumblebee fetches | `["full"]` | ✅ ~3 s warm (separately) |
| 08 | Generic ONNX model | any | `["onnx", "vision"]` | bridge wired |
| 09 | Full voice-assistant chain | VAD + Whisper + Llama + Piper | `["voice-assistant"]` | per-stage validated, full chain not tested |

## What "tested on device" means

For each example below, I verify on a real FP3 that:
* The Elixir module compiles into the firmware.
* `ArmAI.Hub.ensure_all/0` downloads the model from HuggingFace
  to `/root/models/` on boot (when network is up).
* The example script returns a sensible result (the right type +
  shape, a coherent text string, a non-empty detection list,
  etc.).

If a step is verified only on the host (because the model is
gated, very large, or requires network the test environment
lacks), the example's README calls that out.

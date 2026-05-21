# Build features & firmware size

`nx_arm_nif.so` is composed of stackable Cargo features. Pick only
what your application uses — every feature you drop shrinks the
firmware proportionally.

## Measured sizes (aarch64, release build)

| Feature set | .so size | What you get |
|---|---|---|
| `core` (no features) | **2.3 MB** | `NxArm.Backend` (gemm matmul + NEON kernels), topology, governor, storage resizer, KV cache, sampling |
| `vision` | 6.4 MB | + image decode + bilinear resize, `NxArm.Vision.load_for_classifier/2` |
| `tokenizers` | 6.9 MB | + HuggingFace BPE/WordPiece/SentencePiece |
| `audio` | 3.9 MB | + symphonia (MP3/WAV/FLAC/Opus/OGG/Vorbis decode) + rubato resample + WAV writer |
| `onnx` | **22 MB** | + tract-onnx, `NxArm.Models.Onnx` (the heavy one) |
| `llm + tokenizers` (chatbot preset) | 10 MB | + candle LLM stack — `NxArm.Models.LlamaCandle` |
| `vision + onnx` (yolo preset) | 26 MB | + `NxArm.Models.YOLO` |
| `onnx + audio` (piper-tts preset) | 24 MB | + `NxArm.Models.Piper` |
| `onnx + tokenizers` (sentence-rag preset) | 27 MB | + sentence-transformer ONNX → `NxArm.Embeddings` retrieval |
| `whisper` (llm + tokenizers + audio + parsers) | 13 MB | + `NxArm.Models.Whisper` (no ONNX path) |
| `voice-assistant` (whisper + onnx + vision) | 36 MB | + Silero VAD + Piper TTS + camera vision |
| `full` (default) | **36 MB** | everything |

## Use-case presets

Pick the preset closest to what you're shipping. Each is an
alias for a tested combination:

```elixir
# In your Nerves project config (config/target.exs typically):
config :nx_arm, features: ["chatbot"]            # 10 MB
config :nx_arm, features: ["whisper"]            # 13 MB
config :nx_arm, features: ["voice-assistant"]    # 36 MB
config :nx_arm, features: ["yolo"]               # 26 MB
config :nx_arm, features: ["piper-tts"]          # 24 MB
config :nx_arm, features: ["sentence-rag"]       # 27 MB
config :nx_arm, features: ["vision-only"]        # 26 MB
```

Presets and what they resolve to:

| Preset | Resolves to | Use case |
|---|---|---|
| `chatbot` | `llm + tokenizers` | TinyLlama / Mistral / Phi chat on device |
| `whisper` | `llm + tokenizers + audio + parsers` | Local speech-to-text |
| `voice-assistant` | `whisper + onnx + vision` | STT + VAD + TTS + camera |
| `yolo` | `onnx + vision` | Real-time object detection |
| `piper-tts` | `onnx + audio` | Local text-to-speech |
| `sentence-rag` | `onnx + tokenizers` | Embedding retrieval / RAG |
| `vision-only` | `onnx + vision` | Any vision classifier / detector |

## Atomic features (compose your own)

```elixir
config :nx_arm, features: ["llm", "tokenizers"]               # chatbot
config :nx_arm, features: ["onnx", "vision", "audio"]         # vision + audio, no LLM
config :nx_arm, features: ["llm", "tokenizers", "audio"]      # voice-out chatbot
config :nx_arm, features: []                                  # raw Nx backend, no models
```

| Feature | Adds | Modules unlocked |
|---|---|---|
| `llm` | candle-core + -transformers + -nn | `NxArm.Models.LlamaCandle` |
| `tokenizers` | HF tokenizers | `NxArm.Tokenizer` |
| `onnx` | tract-onnx | `NxArm.Models.Onnx`, `NxArm.Models.YOLO`, `NxArm.Models.Piper`, `NxArm.Models.SileroVAD` |
| `audio` | symphonia + rubato | `NxArm.Audio` |
| `vision` | image + fast_image_resize | `NxArm.Vision`, `bilinear_resize_u8_op` |
| `whisper` | (implies llm + tokenizers + audio) | `NxArm.Models.Whisper` |

## What stays in regardless

Even with `features = []` you still get:
- `NxArm.Backend` (the Nx.Backend impl with gemm-backed matmul)
- All elementwise / softmax / silu / conv / quantised matmul NEON NIFs
- big.LITTLE topology detection + scheduler pinning
- `NxArm.Performance` (CPU governor scoping)
- `NxArm.StorageResizer` (F2FS first-boot resize)
- `NxArm.KVCache`, `NxArm.LLM` (RMSNorm, RoPE), `NxArm.Sampling`
- `NxArm.Embeddings` (cosine + top-k — backed by gemm, no extra deps)
- `NxArm.Detection` (NMS — pure Elixir)

That's the 2.3 MB minimum. Everything above that is the model
runtime you opt into.

## How it works

`config :nx_arm, features: [...]` is read at compile time by
`NxArm.Native` and passed to cargo as `--features`. The local-build
path always honours it (e.g. when consuming `nx_arm` as a path dep
on Nerves). Precompiled binaries on GitHub releases ship the
`full` set; if you want a slimmer build you `NX_ARM_BUILD=1 mix
deps.compile nx_arm --force` after setting the config.

## Behavioural guarantee

Any `NxArm.Models.*` module called without its feature returns
`{:error, :<feature>_disabled}` — never crashes the BEAM, never
silently falls back. Check `function_exported?(NxArm.Native,
:<some_op>, n)` to detect a feature at runtime.

## Recommended profiles per device size

| Storage budget | Recommended | Capability |
|---|---|---|
| 4–6 MB free | `core` or `vision` | Nx backend only, Bumblebee/Axon via Nx |
| 10–15 MB | `chatbot` or `whisper` | Single-purpose LLM or STT |
| 25–30 MB | `voice-assistant` minus `llm`, or `yolo` | Full vision pipeline |
| ≥ 36 MB | `full` (default) | Everything |

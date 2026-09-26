# arm_ai

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
> There are no stability guarantees and APIs will change without notice.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

Edge AI inference NIF for ARM CPUs, built for Nerves devices.

One Rustler NIF with:

* hand-tuned NEON kernels (matmul, conv, int8/int4, attention, norms)
  used by [`nx_arm`](https://github.com/mlainez/nx_arm) and the generic
  libraries;
* [candle](https://github.com/huggingface/candle) for quantized Llama
  GGUF inference and Whisper speech-to-text;
* [tract](https://github.com/sonos/tract) for ONNX inference;
* symphonia + rubato for audio decode and resampling, and `image` +
  `fast_image_resize` for image preprocessing.

## Public API

* `ArmAI.LlamaCandle` — load a GGUF file, generate greedily, get timing
  stats. No Nx needed.
* `ArmAI.NxPrimitivesBackend`, `ArmAI.LLMBackend`, `ArmAI.VisionBackend`,
  `ArmAI.AudioBackend` — backends for `nx_primitives`, `infer_llm`,
  `infer_vision` and `infer_audio`.
* `ArmAI.Runtime` — rayon thread pool and CPU topology.

```elixir
{:ok, model} =
  ArmAI.LlamaCandle.load("/data/models/tinyllama.gguf",
    tokenizer: "/data/models/tinyllama-tokenizer.json")

{reply, stats} =
  ArmAI.LlamaCandle.generate(model,
    prompt: "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n",
    max_new: 64,
    stop_tokens: [2])
# reply => "The capital of France is Paris."
```

See `examples/chatbot` for a runnable script.

## Install

```elixir
defp deps do
  [{:arm_ai, github: "mlainez/arm_ai"}]
end
```

No precompiled release has been published yet, so the NIF always builds
from source and the build machine needs a Rust toolchain. For a Nerves
target, add its Rust target (for example
`rustup target add aarch64-unknown-linux-gnu`); `ArmAI.Native` picks up
the target and cross linker from the Nerves environment.

### Cargo features

The crate builds every capability by default (`full`). To shrink the
NIF, pick features in your config:

```elixir
config :arm_ai, features: ["chatbot"]   # llm
config :arm_ai, features: ["whisper"]   # llm + audio + tokenizers
config :arm_ai, features: ["yolo"]      # onnx + vision
config :arm_ai, features: ["onnx", "vision", "audio", "fft"]   # compose your own
```

Atomic features: `llm`, `whisper`, `onnx`, `audio`, `vision`, `fft`.
Calling a function whose feature was left out raises.

### Runtime configuration

```elixir
config :arm_ai,
  thread_pool: :perf_cluster,     # default: pin rayon to the big cores
  # thread_pool: :all_cores,
  # thread_count: 4,              # explicit size, no pinning
  governor_at_boot: :default      # or e.g. :performance
```

## Limits

* ONNX: inputs must be float tensors (f16 models run in f32). Models with
  integer inputs, such as text encoders, Silero VAD or Piper TTS, are not
  supported yet.
* LLM: GGUF files with `llama.*` metadata (Llama, TinyLlama, SmolLM,
  Mistral-style). Greedy decoding only.
* Whisper: candle GGUF or safetensors checkpoints; whisper.cpp `ggml-*.bin`
  files do not load. Greedy, no timestamps.

## Performance

On a Fairphone 3 (Snapdragon 632, 4× Cortex-A73), TinyLlama 1.1B Q4_K_M
decodes at about 4.7 tokens/s, close to the memory-bandwidth limit. See
`docs/perf_llm.md` for the measurements.

## Toolchain

Built and tested with Erlang/OTP 29.1.1, Elixir 1.20.4 and stable Rust,
matching the official Nerves systems (see `.tool-versions`).

## License

Apache-2.0.

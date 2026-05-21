# Architecture

Layer cake of what runs where, why, and how the pieces talk.

```
┌─────────────────────────────────────────────────────────────┐
│  USER CODE                                                  │
│  examples/, hello_nerves, your Nerves app                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  HIGH-LEVEL MODEL WRAPPERS (Elixir)                         │
│  NxArm.Models.LlamaCandle   — quantised Llama family        │
│  NxArm.Models.WhisperCandle — STT, sliding-window decode    │
│  NxArm.Models.Onnx          — generic ONNX                  │
│  NxArm.Models.YOLO          — detection (v5/v8 layouts)     │
│  NxArm.Models.SileroVAD     — voice activity                │
│  NxArm.Models.Piper         — text-to-speech                │
│  NxArm.Models.OCR / .Face / .StableDiffusion                │
│                                                             │
│  Helpers: NxArm.Hub, NxArm.Vision, NxArm.Audio,             │
│           NxArm.Detection, NxArm.Embeddings,                │
│           NxArm.Phonemizer, NxArm.SafeTensors,              │
│           NxArm.Tokenizer, NxArm.FFT                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  NX BACKEND (Elixir)                                        │
│  NxArm.Backend                                              │
│    — implements every Nx.Backend callback                   │
│    — dispatches to NEON NIFs on the hot path                │
│    — falls back to Nx.BinaryBackend for niche ops           │
│    — manages CPU governor scoping via NxArm.Performance     │
│    — manages rayon thread pool pinning via NxArm.Runtime    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  RUST NIF (single .so, feature-gated)                       │
│  native/nx_arm_nif/src/                                     │
│    lib.rs       — rustler entry points                      │
│    ops.rs       — element-wise, argmax, sort, etc.          │
│    shape_ops.rs — matmul, conv, softmax, RMSNorm, RoPE,     │
│                   int8 matmul, FP16 dequant, gather, ...    │
│    conv_int8.rs — int8 conv + dispatch                      │
│    audio.rs / vision.rs / onnx.rs / tokenizer.rs            │
│    fft.rs / safetensors_load.rs / topology.rs               │
│    llama_candle.rs — Llama bridge (greedy decode driver)    │
│    quantized_llama_inplace.rs — vendored quantized Llama    │
│                                  with in-place KvCache      │
│    whisper_candle.rs — Whisper bridge (chunked decode)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  UPSTREAM CRATES                                            │
│  candle-core / candle-transformers / candle-nn              │
│      — Q4_K_M / Q4_0 / Q8_0 matmul (NEON inline asm)        │
│      — Llama / Whisper model definitions                    │
│  gemm                                                       │
│      — f32 GEMM (NEON-tuned)                                │
│  tract-onnx                                                 │
│      — ONNX inference (NEON conv kernels)                   │
│  tokenizers                                                 │
│      — HF-format tokenizer.json                             │
│  symphonia + rubato                                         │
│      — audio decode + resample (NEON SIMD)                  │
│  image + fast_image_resize                                  │
│      — JPEG/PNG decode + NEON-tuned resize                  │
│  rustfft                                                    │
│      — FFT/IFFT (NEON)                                      │
│  safetensors                                                │
│      — HF SafeTensors file format                           │
│  std::arch::aarch64 (NEON intrinsics)                       │
│      — our own hand-rolled kernels in shape_ops.rs          │
└─────────────────────────────────────────────────────────────┘
```

## Why this layering

* **Single .so**: every Cargo feature compiles into one NIF. Saves
  cross-NIF FFI, simplifies cross-compilation. Atomic features
  (`llm`, `onnx`, `vision`, `audio`, `tokenizers`, `safetensors`,
  `fft`) let the Nerves config slim the binary down to just what's
  used. See `docs/size_profile.md`.
* **NxArm.Backend implements `Nx.Backend`**: makes nx_arm a
  drop-in for Bumblebee, Axon, anything in the Nx ecosystem. The
  user doesn't have to learn nx_arm-specific APIs to get NEON
  acceleration for their Axon model.
* **Model wrappers vs raw NIF**: high-level wrappers
  (`NxArm.Models.LlamaCandle`) compose the NIF calls into a
  request → response shape. Raw NIF still callable for users who
  want the building blocks.
* **Vendored `quantized_llama_inplace.rs`**: candle's upstream
  `quantized_llama.rs` re-cats KV cache every decode step (2
  `Tensor::cat` + 1 `contiguous` allocations per attention layer
  per token). Vendoring lets us swap that for
  `candle_nn::kv_cache::KvCache` — pre-allocated, in-place. Pure
  generic-ARM win, no chip-specific hackery.

## Cross-compile flow

For Nerves targets:

```
mix firmware  →  RustlerPrecompiled  →
   (precompiled tarball matched to target triple)
   OR
   (NX_ARM_BUILD=1 → local cargo build with the target triple)
   |
   → Nerves toolchain (gcc/musl) as the C linker
   → Rustler builds the .so → Nerves bundles it into the firmware
```

Targets supported:

* `aarch64-unknown-linux-gnu` (FP3 / most modern ARM Nerves)
* `aarch64-unknown-linux-musl` (slimmer rootfs Nerves)
* `armv7-unknown-linux-gnueabihf` (RPi 0/1/2/3 32-bit)
* `armv7-unknown-linux-musleabihf` (slimmer 32-bit Nerves)
* `x86_64-unknown-linux-gnu` (CI + host development)

## Where time goes (TinyLlama Q4_K_M, decode step on FP3)

Per-token at ~200 ms:

| Phase | Approx ms | Where it runs |
|---|---|---|
| Q/K/V/O matmuls (4× per layer × 22 layers) | ~100 ms | candle NEON Q4_K kernel |
| FFN matmuls (gate/up/down × 22) | ~70 ms | candle NEON Q4_K kernel |
| RMSNorm × 44 | ~5 ms | shape_ops `rmsnorm_f32` NEON |
| RoPE × 22 | ~3 ms | shape_ops `rope_f32` NEON |
| Softmax × 22 | ~2 ms | shape_ops `softmax_f32` NEON |
| KV cache append (in-place) | ~1 ms | candle_nn::KvCache slice_set |
| BEAM ↔ NIF + driver overhead | ~5-10 ms | rustler |
| Sampling (argmax over vocab) | <1 ms | local scan in `llama_candle.rs` |

DRAM bandwidth is the floor — TinyLlama 1.1B Q4_K_M weights are
~660 MB, streaming the full set per token at 6 GB/s = ~110 ms.
We're at ~200 ms, so roughly 2× memory-bound. The remaining gap is
matmul efficiency on the A73 NEON unit + scheduling.

## Things that intentionally don't exist

* **GPU offload from this codebase.** nx_opencl is a separate
  project; the FP3 a5xx CL conv2d hang means it can't help YOLO
  today. Keeping nx_arm CPU-only keeps the deployment story
  simple.
* **A custom tensor library.** Nx is the tensor library. nx_arm
  is a backend.
* **Auto-quantisation tooling.** Use Ultralytics / candle / HF
  tools to quantise. nx_arm runs the quantised result.
* **Model fine-tuning.** nx_arm is inference-only. Train on
  desktop, deploy here.

# Upstream Rust crates — the "use it if it's faster" inventory

Policy: if a NEON-optimised Rust crate already does what Bumblebee
needs better than our hand-rolled code, bridge to it. nx_arm focuses
on (a) being a great Nx backend, (b) wiring upstream into the Nerves
+ Bumblebee story.

## LLM inference (immediate target)

| Crate | What it gives us | Status |
|---|---|---|
| `candle-core` | Rust ML framework, ARM NEON kernels for Q4_0 / Q4_K / Q8_0, GGUF loader | **Adopt for `NxArm.Models.Llama` and quantised matmul** |
| `candle-transformers` | Llama-1/2/3, Mistral, Phi-3, Qwen, Gemma forward passes | **Adopt — kills our hand-rolled `models/llama.ex`** |
| `candle-nn` | Linear, RMSNorm, RoPE, scaled-dot-product attention | **Adopt — replaces `NxArm.LLM.*` for the LLM path** |
| `mistralrs` | Paged attention, speculative decoding, faster decode loop | Watch — adopt for >1B param production runs |
| `llm` (rustformers) | Older Rust LLM runtime | Skip (candle covers it) |

## Matrix multiplication

| Crate | What it gives us | Status |
|---|---|---|
| `gemm` (sarah-quinones) | Fast Rust GEMM with NEON, competitive with OpenBLAS on ARM | **Adopt for `NxArm.Backend.dot` f32 path** |
| `matrixmultiply` | Older Rust GEMM, less NEON-tuned | Skip |
| `ndarray-linalg` | LAPACK bindings | Only if we need QR/SVD/etc. |

## Tokenisation (needed for full LLM story)

| Crate | What it gives us | Status |
|---|---|---|
| `tokenizers` (HF) | Fast BPE/WordPiece/SentencePiece, used by candle | **Adopt — prompts can take strings instead of token IDs** |
| `sentencepiece` | Google SentencePiece bindings | Used by candle internally |

## Quantisation primitives

| Crate | What it gives us | Status |
|---|---|---|
| `gguf-rs` | GGUF v3 reader | **Already replicated; switch when we adopt candle (which uses ggml directly)** |
| `safetensors` | SafeTensors reader | **Adopt — drop `lib/nx_arm/safetensors.ex`** |
| `half` | f16/bf16 types | **Adopt — replace our manual bit-level f16 ↔ f32** |

## Numerical / SIMD

| Crate | What it gives us | Status |
|---|---|---|
| `wide` | Portable SIMD (f32x4, etc.) | Skip — direct NEON intrinsics are clearer for our case |
| `rayon` | Data parallelism | **Already using** |

## ONNX (a future Bumblebee path)

| Crate | What it gives us | Status |
|---|---|---|
| `tract-onnx` | ONNX runtime with ARM NEON, Rust-native | **Worth adopting once we want generic ONNX models on Nerves** |
| `wonnx` | WebGPU ONNX | Skip (no GPU on most Nerves targets) |

## Image / audio (for vision / speech models)

| Crate | What it gives us | Status |
|---|---|---|
| `image` | Decoding + basic transforms | Adopt if we add vision models |
| `fast-image-resize` | SIMD-accelerated resize | Adopt instead of `bilinear_resize_u8_op` once we benchmark |
| `rubato` | Audio resampling | Adopt for Whisper / speech models |

## What stays hand-rolled

* `NxArm.Backend` — Nx callback dispatch, this is our integration layer
* big.LITTLE topology detection + pinning — Nerves-specific
* `StorageResizer`, `Performance` (governor scoping) — Nerves-specific
* Application boot hooks — Nerves-specific

## Migration order (proposed)

1. **`gemm`** for f32 matmul — kills hand-tuned `matmul_2d_neon_blocked`, expected better perf on every shape.
2. **`candle-core` + `candle-transformers`** for Llama — kills `NxArm.Models.Llama` internals, expected 3–5× decode tok/s on FP3.
3. **`tokenizers`** — lets the public API take strings.
4. **`safetensors` + `half`** — drops two hand-rolled chunks of decoder code.
5. **`tract-onnx`** when a Bumblebee user actually needs it.

Each migration is a separate PR; nx_arm keeps the Nx interface, the
swap is internal.

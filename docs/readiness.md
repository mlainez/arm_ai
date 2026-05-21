# Readiness — what works in production, what doesn't, and where the NEON is

This is the honest map of what nx_arm is ready to ship in real
products versus what's still bridge-quality. Updated 2026-05-21.

## Production-ready use cases

| Use case | What it does | Verified ceiling |
|---|---|---|
| **Quantised LLM chat** (TinyLlama / SmolLM / Phi-Q4) | Q4_K_M GGUF inference via `NxArm.Models.LlamaCandle` | 4.7 tok/s on FP3 (Snapdragon 632, 4× A73 @ 1.8 GHz). 12–15 tok/s on A76+. See [perf_llm.md](perf_llm.md) for the ceiling analysis. |
| **Sentence embeddings + RAG** | `NxArm.Embeddings` cosine search + top-k | ~700 µs/query for ~10 docs; scales linearly to ~10 k docs |
| **Image classification** (ViT-tiny, MobileNet, EfficientNet-Q) | Bumblebee/Axon or `NxArm.Models.Onnx` | ViT-tiny warm forward ~3 s on FP3 |
| **Generic ONNX inference** (conv/gemm/relu/softmax/resize) | `NxArm.Models.Onnx` over tract-onnx 0.21 | Works for most CV models post-2020; older exports need an opset bump (see `examples/04_yolo_detection/export_to_tract_opset.py`) |
| **Still-image object detection** (yolov5n) | `NxArm.Models.YOLO` with `:v5` layout | ~1 s/image on FP3 at 640×640 fp32 |

For these, ship them. The hot paths are all on NEON kernels.

## Bridge-quality (wired but not field-validated)

| Use case | Module | What's missing |
|---|---|---|
| Speech-to-text | `NxArm.Models.WhisperCandle` | mel-filter binary distribution + an audio-in integration test |
| Real-time object detection | `NxArm.Models.YOLO` | Latency improvements (int8 export, smaller input, tracker between frames). See [examples/04 README](../examples/04_yolo_detection/README.md). |
| Voice activity detection | `NxArm.Models.SileroVAD` | End-to-end test against a real WAV |
| Text-to-speech | `NxArm.Models.Piper` | Real `phoneme_id_map` parser from the voice's `.onnx.json` |
| OCR | `NxArm.Models.OCR` | Test against PaddleOCR / Tesseract-class exports |
| Face detection / landmarks | `NxArm.Models.Face` | Same as OCR |
| Stable Diffusion | `NxArm.Models.StableDiffusion` | A53/A73 hardware can't run this at usable latency. Demo only. |

## Where NEON is engaged

Every line-bottleneck path has a NEON-tuned kernel:

| Operation | Path | Source |
|---|---|---|
| f32 GEMM | `gemm` crate, hand-tuned NEON | `Cargo.toml` |
| Q4_K_M / Q4_0 / Q8_0 GEMM | candle `vec_dot_q4k_q8k` NEON inline-asm | `candle-core 0.8.4` |
| f32 dot products | local fused-multiply-add NEON | `shape_ops.rs:646` |
| Softmax | NEON variant | `shape_ops.rs:2401` |
| RMSNorm | NEON variant | `shape_ops.rs:977` |
| RoPE (rotary embeddings) | NEON variant | `shape_ops.rs:1041` |
| Bias-add + activation fusion | NEON | `shape_ops.rs:701` |
| int8 matmul (static + per-token quantised) | NEON dispatch | `shape_ops.rs:1170`, `shape_ops.rs:1223` |
| f16/f32 dequant matmul | NEON | `shape_ops.rs:1387` |
| 2D conv f32 (NCHW, dilation=1) | NEON via im2col + gemm | `lib/nx_arm/backend.ex` (`do_neon_conv`) |
| FFT / IFFT (complex-64) | rustfft NEON | `fft.rs` |
| Audio resample | rubato NEON-SIMD | `audio.rs` |
| Image decode + resize | `image` + `fast_image_resize` NEON | `vision.rs` |
| Bilinear resize (u8) | NEON | `shape_ops.rs:1110` |
| Quantised weights via ONNX | tract-onnx NEON conv/gemm | `onnx.rs` |
| Q4_K LLM forward pass | candle quantised\_llama (vendored with in-place KvCache) | `quantized_llama_inplace.rs` |

## Where NEON is NOT engaged (fallback to `Nx.BinaryBackend`)

These fall back to scalar Erlang binaries. Correct but slow. Ranked
by likelihood of being hit on a real inference workload:

| Op | Frequency in inference | Impact |
|---|---|---|
| `reduce` (generic with a fun) | Rare — networks use sum/mean/max directly | Low |
| `window_reduce`, `window_scatter_max/min` | If a model uses an exotic pooling primitive | Low-Med |
| `gather` (n-D index) | Hot path is embedding lookup, which uses the indexed-array fast path; n-D index hits the slow path | Medium for some attention variants |
| `pad` with non-zero fill | Rare; zero-pad has a NEON memcpy path | Low |
| `triangular_solve`, `lu`, Cholesky | Almost never in inference | Negligible |
| `dot` outside the batched-f32 (≥3-D) path | f32 contraction shapes that aren't batched matmul | Medium |
| `conv` with `feature_group_size > 1` | Grouped/depthwise-with-expansion (some efficient nets) | Medium |
| `fft` / `ifft` for non-complex-64 input | Rare | Low |
| `to_batched`, `reverse`, `stack`, `as_type`, `clip`, `select` outside the same-shape fast path | Glue, not arithmetic | Low (memcpy-bound) |

The biggest realistic next win is **grouped conv with `feature_group_size > 1`**.
Nothing in the example set hits it yet — that's why it hasn't shipped.

## Per-target ceilings

Generic ARM, but here's how the same code scales across the CPU
classes you'd realistically deploy onto:

| SoC class (representative) | LLM (TinyLlama Q4_K_M) | YOLOv5n 640² fp32 | ViT-tiny |
|---|---|---|---|
| Cortex-A53 (RPi 3, FP2) | 1-2 tok/s | 5-7 s/frame | 8-10 s |
| Cortex-A53/A73 (FP3, mid-range 2018-2020 phones) | **5+ tok/s** | ~1 s/frame | ~3 s |
| Cortex-A72 (RPi 4) | 4-5 tok/s | 1.5-2 s/frame | 4-5 s |
| Cortex-A76/A78 (RK3588, Pixel-6-class) | 12-15 tok/s | 100 ms/frame | ~1 s |
| Cortex-X1/X2 + A78 | 20+ tok/s | <60 ms/frame | <500 ms |

For an "edge AI on ARM" deployment, the FP3-class row is the
sensible reference point. Anything below A53 isn't enough RAM;
anything above A78 doesn't need this stack.

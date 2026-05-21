# Changelog

## Unreleased

### Added

* **Production-readiness op coverage.** Replaced 13 `BinaryBackend`
  fallbacks with native NIFs covering the LLM/CV hot path:
  argmax/argmin, select, as_type, clip, pad, gather (axis 0),
  stack (axis 0), sort/argsort, all/any/product, reverse.
* **big.LITTLE auto-pinning.** `NxArm.Application` detects the perf
  cluster at boot via `cpu_capacity` / `cpufreq` / `midr_el1` and
  pins both the rayon worker pool and the BEAM dirty CPU
  schedulers to it. Generic ARM CPUs and homogeneous chips are
  unaffected (no-op fallback).
* **Reverse-mode autodiff coverage.** Five new tests verify
  `Nx.Defn.grad` works through the backend, including a 300-step
  SGD loop that converges to ground truth.
* **Precompiled NIF infrastructure.** Switched to
  `rustler_precompiled` with a deployment matrix covering 8
  triples (aarch64 + armv7 gnu/musl, x86_64 gnu/musl, mac
  x86_64/aarch64). CI workflows for both PR tests and tag
  releases.
* **Battle-test coverage.** 18 StreamData properties fuzzing
  every elementwise/reduction/dot/transpose/softmax op against
  `Nx.BinaryBackend`. Edge cases (NaN, Inf, large tensors,
  extreme aspect ratios), concurrency stress (16 parallel
  matmuls), NIF robustness (bad inputs, leak detection), and a
  2-layer transformer e2e diff.
* **LLM helpers.** KV cache (`NxArm.KVCache`), causal +
  decode-step masks, single-position RoPE (`rope_at/3`),
  CTRL-style repetition penalty for sampling.
* **Model file readers.** GGUF v3 with Q4_0 unpack, SafeTensors,
  memory-mapped loading via `memmap2` so multi-GB models page in
  from disk on demand.

### Op kernels (NEON)

* Cache-blocked 4×8 register-tile matmul (M=64..K_block=128).
* Vector exp/sigmoid/tanh polynomial approximations.
* Window reductions (max/sum/min/product) NHWC.
* SDOT path for int8 matmul on ARMv8.2-A with runtime feature
  detection, vmlal_s8 + vpadalq fallback on ARMv8.0.
* fp16/bf16 weight storage with vectorised dequant (scalar
  conversion + NEON FMA).
* Fused bias-add + activation (ReLU, ReLU6, sigmoid, tanh, GELU).
* Winograd F(2, 3) conv2d (3×3 stride-1).
* im2col + GEMM for general kernel sizes.
* Depthwise + pointwise fusion (MobileNet inverted residual).
* Flash Attention V1 (streaming softmax, no Sq×Sk
  materialisation).

### Compiler

* Pattern fusion for softmax, multiply→softmax→divide, layernorm,
  GELU, fused linear, fused attention, fused bias-add.
* Dead-broadcast elimination, dropout elimination, constant
  folding.

### Known fallbacks

* `triangular_solve`, `lu`, `fft`/`ifft`,
  `indexed_add`/`indexed_put`, `window_scatter_max`/`min` route
  through `Nx.BinaryBackend`. File an issue if you need any of
  these on the hot path.

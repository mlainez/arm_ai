# NxArm optimisation roadmap

Living list of optimisations for Axon/Bumblebee workloads on ARM CPUs.
Driven by the FP3+ ViT-tiny baseline; should generalise to any aarch64.

## Done

- [x] **Tier-0** — NEON-vectorised primitives via Rust auto-vectoriser
  (rayon-parallel chunks). 30–50× over pure-Elixir BinaryBackend.
- [x] **Tier-1.1** — Register-tiled 4×4 NEON matmul kernel
  (`matmul_kernel_4x4`). 5–8× on ViT-tiny shapes vs the
  auto-vectorised loop. Q @ K^T uses a vfmaq + vaddvq dot variant.

## In progress

- [ ] **Tier-1.2** — Fused softmax NIF (`softmax_f32_op`). Today
  Axon's softmax decomposes into 7 backend calls + 7 `Vec<f32>`
  allocations and costs 109 ms on `{1,3,197,197}`. Fused should be
  10–20 ms (one pass: max → exp(x-max) → sum → divide).

## Next up

- [ ] **Tier-1.3** — Fused GELU NIF. Currently divide+erf+add+multiply
  +divide (5 backend calls). Fused: one pass with polynomial erf
  inline.
- [ ] **Tier-1.4** — Fused LayerNorm NIF. Currently 2 reduces + 2
  broadcasts + subtract + multiply + add + scale + bias (~9 calls).
  Fused: one pass over each `[..., hidden]` row.
- [ ] **Tier-1.5** — Cache-blocked matmul. Current 4×4 tile keeps
  16 fp32 accumulators in registers but reads B at full N stride
  every k step — wastes L1 (32 KB on A73) on the big (197×768)
  matmul. Block K by ~128 to keep B-panel in L1.
- [x] **Tier-1.6** — Wider matmul register tile (4×8). Done 2026-05-20
  (commit `6ba8f83`). 1.1–1.2× on N-divisible-by-8 shapes; ViT-tiny
  e2e in the noise — bottleneck has moved off compute.

## Tier 2 — `NxArm.Compiler` (Nx.Defn.Compiler implementation)

Single sweep over the `Nx.Defn.Expr` graph before execution.

- [ ] Skeleton compiler that walks Expr, emits NIF-call plan, runs it.
- [ ] Pattern-fuse softmax / GELU / LayerNorm to single NIF calls
      (vs hand-detection inside backend).
- [ ] Pre-allocate intermediate buffers; reuse when previous SSA
      value is dead (saves `Vec<f32>` per op).
- [ ] Elide redundant `Nx.broadcast` calls (when the broadcast result
      already matches the consumer's needs).
- [ ] Constant-fold trivial subgraphs (bias broadcasts, scale literals).
- [ ] Expected gain: another 1.5–2× on warm Bumblebee forwards.

## Tier 3 — Numerical / hardware

- [ ] **Int8 quantised matmul** (`vmlal` pattern). Per-channel int8
      weights, f32 activations, f32 accumulation. Memory-bound ops
      get 2–4× from cutting bandwidth. Cortex-A73 (pre-ARMv8.2) has
      no SDOT, but `vmlal_s8` + `vpadalq_s16` works.
- [ ] **Int4 quantised matmul**. Packed nibbles, unpack-on-the-fly.
      4× memory.
- [ ] **bf16 / fp16 storage** with f32 compute (load + convert).
      Half the memory bandwidth for weights. We can't do native fp16
      compute on a5xx GPU but on A73 CPU it's a load-time conversion.
- [ ] **ARMv8.2 SDOT/UDOT** path, gated on `is_aarch64_feature_detected!`.
      Not present on SDM632 (FP3+), but on newer Snapdragons / Apple M.

## Tier 4 — Algorithmic

- [ ] **Flash Attention** style attention (avoid materialising the
      `[seq, seq]` matrix). Saves memory + bandwidth for long
      sequences. Less impactful at seq=197.
- [ ] **Winograd 3×3 conv** for residual networks (not ViT).
- [ ] **ARM Compute Library (libACL)** as an optional dependency for
      its hand-tuned gemm/conv. Cross-compile concerns; evaluate.

## Tier 5 — Model / serving level

- [ ] **KV cache** for generative LLMs (Mistral / Llama via Bumblebee).
      Massive speedup on chat-style serving.
- [ ] **Speculative decoding** for chat workloads.
- [ ] **Static-shape JIT** — once a model's input shape is known,
      pre-plan/specialise the whole graph (a Tier-2 compiler refinement).

## Measurement discipline

Every tier change must include:
1. A targeted microbenchmark in `perf_probe.exs`.
2. A full end-to-end ViT-tiny (and eventually Mistral) before/after
   number.
3. Numerical equality vs `Nx.BinaryBackend` (or last-known-good
   reference) to catch regressions.

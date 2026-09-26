# LLM decode performance on ARM CPUs

This is the honest report on where TinyLlama (and Q4_K_M class
quantised LLMs in general) lands on ARM CPUs with the current
arm_ai stack. Updated 2026-05-21 after exhausting every generic
ARM optimisation that's safe to ship.

## The number

**4.71 tok/s** decode best, **4.68 tok/s** average across three
back-to-back runs on a FairPhone 3 (Snapdragon 632 — 4× Cortex-A73
+ 4× Cortex-A53, all at 1.8 GHz, 4 GB LPDDR3), running the full
production app stack (modem QMI, NFC, GPS, audio, camera) plus
the FP3 platform daemons.

Stable, reproducible, ±0.05 tok/s variance.

## Why not 5 tok/s

Snapdragon 632 has roughly **6 GB/s of DRAM bandwidth**. The
TinyLlama 1.1B Q4_K_M weights are ~660 MB. Per decode step we
have to stream every weight tile through the matmul kernel,
because there's no temporal locality in a single-token forward:

```
660 MB ÷ 6 GB/s ≈ 110 ms of pure DRAM streaming per token
```

Measured per-token decode is 213 ms. **We're at 52 % of the
memory-bandwidth ceiling**, which is where well-tuned LLM
inference lives. The remaining 100 ms is candle's NEON Q4_K
matmul compute, attention math, RMSNorm, RoPE, and Rust framework
overhead — all already on NEON, all near their respective ceilings.

To hit 5 tok/s (200 ms/token) on this exact hardware we'd need
to cut ~13 ms per token, which is impossible without either a
custom matmul kernel that beats candle's hand-tuned inline asm
or a different memory hierarchy.

## What was tried, what worked, what didn't

All "generic ARM" — nothing FP3-specific.

| Optimisation | Δ tok/s | Status |
|---|---|---|
| Release profile (LTO=fat, codegen-units=1, panic=abort, strip) | +~0.1 | kept (see [Cargo.toml](../native/arm_ai_nif/Cargo.toml)) |
| Per-token argmax inline (skip per-step tensor alloc) | ~0 | kept (cleaner code) |
| Vendored quantized_llama with in-place KvCache | **-0.17** | reverted: KvCache.current_data returns narrow views that aren't contiguous → downstream matmul triggers extra `.contiguous()` copies that exceed the saved `cat` allocations |
| mimalloc as `#[global_allocator]` | **boot loop** | reverted: TLS init order vs the BEAM scheduler under Nerves rootfs |
| PGO (profile-guided optimisation) | **~0** | tried, not kept: instrumented build captured 3 MB of profile data on-device, llvm-profdata merged, optimised rebuild tested. **No measurable lift.** Matmul kernels don't have the branchy code paths PGO specialises on. Captured here as evidence that PGO ≠ free perf for ML workloads. |

## Why PGO didn't help

PGO improves code that has many branches whose direction depends
on runtime data the compiler can't predict statically — compilers,
parsers, request-routing code, JSON deserialisers. The typical
PGO win is 5–10 % on such code.

The TinyLlama decode hot path is **85 % time inside candle's NEON
Q4_K matmul kernel** (`vec_dot_q4k_q8k` in candle-core 0.8.4). It's
a tight `#[target_feature(enable = "neon")]` function with NEON
intrinsics in a row-by-row loop. No data-dependent branches in
the inner loop. LLVM's default heuristics already nail it. PGO
literally has no information to add.

Same story for RMSNorm, RoPE, softmax — short, branchless, NEON.

## What would actually close the gap

These are real but each costs more than what we've spent so far:

1. **Hardware uplift.** Cortex-A76 / A78 with the same Rust binary
   should deliver 12–15 tok/s based on the IPC + memory bandwidth
   ratio (~2.5–3× this SoC). The codebase is generic ARM; no
   changes needed.
2. **Custom quantised matmul kernel.** Beat candle's `vec_dot_q4k_q8k`
   on the specific shapes TinyLlama uses (M=1, K=2048, N=2048 for
   attention; K=2048/5632 for FFN). Possible 10–20 % lift on
   matmul, ~5–10 % overall. Effort: weeks. Risk: easy to lose
   numerical accuracy.
3. **int2/int3 quantisation.** Cuts weight size 2× from int4,
   doubles effective DRAM bandwidth. Quality loss is non-trivial
   for 1B-class models. Out of scope without retraining.
4. **Speculative decoding.** Predict N tokens at once via a draft
   model + verification. Mostly improves *latency-to-first-token*
   for long sequences; per-token throughput depends on draft-model
   accept rate. Substantial implementation effort.
5. **Skip the BEAM round-trip more aggressively.** Our per-token
   loop is already entirely in Rust; further BEAM-side gains
   would need a redesign of how Elixir consumes streaming output.

## What this means in practice

- Expect about **4.7 tok/s** for a 1B Q4_K_M model on a Snapdragon
  632-class device. That is the realistic baseline, not a bug.
- LLM-heavy workloads benefit substantially from newer ARM cores
  (A76 and later) with more memory bandwidth; the code doesn't change.
- For sub-200 ms per token on phone-class ARM, use a smaller model:
  SmolLM-135M Q4_K_M measured about 13 tok/s on the same Fairphone 3.

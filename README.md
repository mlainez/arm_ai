# nx_arm

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
>
> It is **not an actively maintained project** (yet). There are no
> stability guarantees, APIs will change without notice, and parts of it
> are wired-but-unproven. Treat it as a starting point to hack on, not as
> a dependency to build a product on.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

An [Nx](https://github.com/elixir-nx/nx) backend for ARM
CPUs via NEON intrinsics + rayon. Built for Nerves devices —
Raspberry Pi, FP3, BeagleBone — where you need real ML throughput
on the CPU without dragging in an OpenCL / NPU stack.

## Status

* **217 tests, 18 property tests** (~2000 randomized cases), 0
  failures on every commit.
* End-to-end autodiff verified: 300-step SGD training loop runs
  on `NxArm.Backend` and converges to ground truth.
* On-device benchmarks on FP3 (Cortex-A73 cluster):
  * 256×256×256 f32 matmul: **9.8 GFLOPS sustained**
  * 64×128×4000 prefill lm_head: **10.9 GFLOPS**
  * Tiny-LM (4 layers, 2M params): prefill 196 ms, decode 83 ms/tok

## What's in the box

### Op coverage (native NIFs, not fallbacks)

* **Elementwise**: NEON-vectorised add/subtract/multiply/divide,
  exp / sigmoid / tanh polynomial approximations.
* **Matmul**: cache-blocked NEON kernel with 4×8 register tile,
  prefetching, K-blocking for L1 fit.
* **Convolution**: direct conv2d, Winograd F(2,3) for 3×3
  stride-1, im2col+GEMM for general kernels, depthwise +
  pointwise fusion (the MobileNet block).
* **Quantisation**: full int8 matmul (SDOT on ARMv8.2-A,
  vmlal_s8+vpadalq fallback), int4 packed weights (GGUF Q4_0),
  per-token activation scales, fp16/bf16 weight storage.
* **LLM kernels**: Flash Attention V1 (streaming softmax),
  RMSNorm, RoPE, KV cache, causal mask, repetition penalty,
  top-k / top-p sampling.
* **Production ops**: argmax/argmin, select, as_type, clip, pad,
  gather, stack, sort/argsort, all/any/product, reverse — all
  native NIFs covering the LLM/CV hot path.
* **File loading**: GGUF v3 reader (header + metadata + tensor
  catalogue + Q4_0 unpack), SafeTensors, memory-mapped loading.

### Compiler-side patterns

`NxArm.Compiler` recognises and fuses softmax,
multiply→softmax→divide, layernorm, GELU, bias-add, dropout
elimination, dead broadcast elimination, constant folding.

### big.LITTLE topology

On heterogeneous chips (Snapdragon big.LITTLE, Tegra, Apple, Pi
5) the backend automatically detects the perf cluster at boot
and pins both the rayon worker pool **and** the BEAM dirty CPU
schedulers to it. Detection priority:

1. `/sys/devices/system/cpu/cpu*/cpu_capacity` (DT-derived).
2. `cpufreq/cpuinfo_max_freq` (highest cluster).
3. `regs/identification/midr_el1` part-number table (A35..A715
   + Kryo Gold/Silver + Apple silicon).
4. Fallback: use all cores (no pinning).

```elixir
iex> ArmAI.Runtime.topology()
%{source: "cpu_capacity", perf_cores: [4, 5, 6, 7], all_cores: [0..7]}
```

## Installation

```elixir
defp deps do
  [{:nx_arm, "~> 0.1"}]
end
```

Activate as the default backend in your project config:

```elixir
# config/config.exs
config :nx, default_backend: NxArm.Backend
```

The NIF binaries are precompiled and fetched on first
`mix deps.compile`; no Rust toolchain required. See
[`docs/deployment.md`](docs/deployment.md) for the full target
matrix and override knobs.

## Quick example

```elixir
# Inference: a 2-layer transformer forward
mask = ArmAI.LLM.causal_mask(seq) |> Nx.broadcast({n_heads, seq, seq})
x =
  Enum.reduce(layers, x, fn ws, acc ->
    a = ArmAI.LLM.rmsnorm(acc, ws.norm1)
    a = attention(a, ws.w_q, ws.w_k, ws.w_v, ws.w_o, mask, n_heads)
    x1 = Nx.add(acc, a)
    f = ArmAI.LLM.rmsnorm(x1, ws.norm2)
    Nx.add(x1, ffn(f, ws.w_gate, ws.w_up, ws.w_down))
  end)
```

```elixir
# Training: SGD with autodiff
defn step(w, b, x, y, lr) do
  {loss, {dw, db}} =
    value_and_grad({w, b}, fn {ww, bb} ->
      pred = Nx.dot(x, ww) + bb
      Nx.sum((pred - y) ** 2)
    end)

  {w - dw * lr, b - db * lr, loss}
end
```

## Architecture targets

| Triple                              | Typical board                    |
|-------------------------------------|----------------------------------|
| `aarch64-unknown-linux-gnu`         | Pi 3B+/4/5 (64-bit), FP3         |
| `aarch64-unknown-linux-musl`        | Nerves aarch64 musl              |
| `armv7-unknown-linux-gnueabihf`     | Pi Zero 2W, BeagleBone Black     |
| `armv7-unknown-linux-musleabihf`    | Nerves armv7 musl                |
| `x86_64-unknown-linux-{gnu,musl}`   | Dev, CI                          |
| `{x86_64,aarch64}-apple-darwin`     | Mac dev box                      |

## Honest limitations

* **Fallback ops**: `triangular_solve`, `lu`, `fft`/`ifft`,
  `indexed_add`/`put`, `window_scatter_max`/`min` still route to
  `Nx.BinaryBackend`. Won't break correctness; will be slow if
  hit in a hot loop. Open an issue with the use case and they'll
  jump in priority.
* **Linear algebra**: no QR/SVD; same as above — file an issue
  if you need them.
* **No GPU/NPU**: this is a CPU-only backend by design.
  `nx_opencl` exists for the Adreno/Mali side of the same
  hardware.

## Layout

```
lib/nx_arm/
├── application.ex      # boot: topology detect + thread-pool pin
├── backend.ex          # Nx.Backend impl (op dispatch)
├── compiler.ex         # Nx.Defn.Compiler pattern fusion
├── llm.ex              # transformer helpers
├── kv_cache.ex         # decoder cache
├── sampling.ex         # greedy/top-k/top-p + repetition penalty
├── gguf.ex             # GGUF v3 reader
├── safetensors.ex      # SafeTensors reader
├── runtime.ex          # topology / thread-pool diagnostics
└── bench/              # microbenchmarks
native/nx_arm_nif/src/
├── lib.rs              # NIF entry points
├── shape_ops.rs        # NEON kernels: matmul, conv, attention, ...
├── conv_int8.rs        # quantised conv
├── ops.rs              # production ops (argmax, select, sort, ...)
└── topology.rs         # CPU cluster detection + pinning
```

## License

Apache-2.0.

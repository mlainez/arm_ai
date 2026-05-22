# Nx op coverage on `NxArm.Backend`

The lib implements most of the `Nx.Backend` callback surface
natively (NEON kernels, gemm-backed matmul, or thin bridges to
upstream crates). A small number of ops still fall back to
`Nx.BinaryBackend` — they're either niche enough that nobody's
hit them yet, or they need linear-algebra heavy implementations
that haven't been wired.

## Native (fast)

* Elementwise unary (negate, exp, log, tanh, sigmoid, sqrt, ...)
* Elementwise binary (add, subtract, multiply, divide, max, min, ...)
* Reductions (sum, reduce_max, reduce_min, mean, all, any, product)
* Reshape / transpose / broadcast / concatenate / stack / slice / put_slice
* Dot (every contracting-axis pattern: `[1,0]`, `[1,1]`, batched 3-D+)
* Conv (general direct, Winograd 3×3, im2col + GEMM, depthwise + pointwise)
* Pad / clip / reverse / as_type / select / gather
* argmax / argmin (last axis, f32)
* sort / argsort (last axis, f32)
* indexed_add / indexed_put (f32)
* FFT / IFFT (complex-64 via rustfft)
* Window reductions (max-pool, sum-pool, etc.)
* Softmax / silu fused
* RMSNorm + RoPE (fused, via `ArmAI.LLM`)

## Falls back to BinaryBackend

These are still pure-Elixir scalar in `Nx.BinaryBackend`. Calling
them on `NxArm.Backend` automatically routes there with a `to_binary
→ run → from_binary` round-trip. Functionally correct, just slow.

| Op | Why still on fallback | Realistic impact |
|---|---|---|
| `triangular_solve` | Linear algebra (forward/back-substitution). Used in Cholesky, regression solvers. | Low: rare in NN inference; matters for Bayesian / kalman pipelines. |
| `lu` | LU decomposition. Used inside `solve`, `det`. | Low: same niche as above. |
| `window_scatter_max`/`min` | Pooling gradients during training. | Low: training on-device is rare; inference doesn't use them. |
| `reduce` / `window_reduce` with arbitrary 2-arg fun | Generic reducers — needs to call back into BEAM per element. | Medium: specific reducers (sum, max, min, product, all, any) are native; only the user-supplied-fun path falls back. |
| `argsort` / `sort` for non-f32 dtypes | Native only covers f32. | Low: f32 is the dominant case. |

## When to care

If you're shipping an LLM, vision classifier, voice agent, or
embedding pipeline you'll hit none of the fallbacks. They show up
in:
* Bayesian regression / Kalman filters → `triangular_solve`, `lu`
* Custom Defn reducers with non-standard accumulators
* On-device training of pooling layers → `window_scatter_*`

If any of these is your bottleneck, file an issue with the
specific shape you're hitting and we'll prioritise the kernel.

## How to detect fallbacks at runtime

```elixir
# Bench any op; if it suddenly costs many ms for tiny tensors,
# you're probably on the fallback path.
:timer.tc(fn -> Nx.lu(NxArm.Backend.copy(matrix)) end)
```

For systematic discovery, `mix test --seed 0 --include slow`
runs the conformance fuzz suite against `Nx.BinaryBackend` — any
op that diverges, hangs, or is dramatically slower is on
fallback.

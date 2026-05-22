# Deployment guide

## Precompiled NIFs (production path)

`nx_arm` ships precompiled NIF binaries via `rustler_precompiled`.
At dependency resolution time, mix pulls the binary matching the
runtime's Erlang NIF version + target triple from the project's
GitHub releases. No Rust toolchain is required on the deployment
target.

### Supported deployment targets

| Triple                              | Typical board                   |
|-------------------------------------|---------------------------------|
| `aarch64-unknown-linux-gnu`         | Raspberry Pi 3B+/4/5 (64-bit), FP3 |
| `aarch64-unknown-linux-musl`        | Nerves aarch64 musl systems     |
| `armv7-unknown-linux-gnueabihf`     | Pi Zero 2 W, BeagleBone Black   |
| `armv7-unknown-linux-musleabihf`    | Nerves armv7 musl systems       |
| `x86_64-unknown-linux-gnu`          | Dev box, CI                     |
| `x86_64-unknown-linux-musl`         | Container images                |
| `x86_64-apple-darwin`               | Mac dev box (Intel)             |
| `aarch64-apple-darwin`              | Mac dev box (Apple Silicon)     |

### Forcing a local build

Set `NX_ARM_BUILD=1` to bypass the precompiled fetch and build
from the Rust source. Useful during development, when adding a
new target, or when running on a triple that isn't in the matrix
yet. The Rust toolchain must be available.

```sh
NX_ARM_BUILD=1 mix deps.compile nx_arm --force
```

## Nerves cross-build

For Nerves firmware builds, the Rustler config picks up the
cross-toolchain environment variables (`TARGET_ARCH`, `TARGET_OS`,
`TARGET_ABI`, `CC`) and translates them into the matching Rust
target triple. Today the cross-build path runs whenever
`NX_ARM_BUILD=1` is set or when the project is consumed as a path
dependency without a checksums file (i.e. a git checkout, not a
published Hex version).

Once a precompiled artifact for the Nerves target triple is on
GitHub releases, the precompiled fetch takes over and the Rust
toolchain is no longer required on the build host either.

## big.LITTLE topology

`ArmAI.Application` detects the CPU topology at boot. On
heterogeneous chips (e.g. FP3 / Snapdragon 632 with 4× A73 + 4×
A53) the rayon worker threads pin themselves to the perf cluster
and the BEAM dirty CPU schedulers migrate there too.

To inspect or override:

```elixir
# Diagnostics
ArmAI.Runtime.topology()
# %{source: "cpu_capacity", perf_cores: [4, 5, 6, 7], all_cores: [0..7]}

# Force a specific thread count (no pinning)
config :nx_arm, thread_count: 4

# Disable perf-cluster pinning
config :nx_arm, thread_pool: :all_cores
```

## Adding a new deployment target

1. Add the triple to the `:targets` list in
   `lib/nx_arm/native.ex`.
2. Add the corresponding row to the matrix in
   `.github/workflows/release.yml`.
3. Tag a new version. The release workflow builds and uploads the
   precompiled tarball; `rustler_precompiled` finds it on next
   `mix deps.compile`.

## Verifying a deployment

```elixir
# In the device's iex
ArmAI.Runtime.topology()
ArmAI.Runtime.thread_count()

# Quick correctness check
Nx.iota({4, 4}, type: :f32)
|> Nx.backend_copy(NxArm.Backend)
|> Nx.dot(Nx.iota({4, 4}, type: :f32) |> Nx.backend_copy(NxArm.Backend))
|> Nx.to_flat_list()

# Microbench: the cache-blocked matmul should hit several GFLOPS
# on any aarch64 board.
ArmAI.Bench.TinyLM.run(seq_prefill: 16, n_decode: 4)
```

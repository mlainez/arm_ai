# Binary size & feature set

`nx_arm_nif.so` carries different upstream crates depending on
which Cargo features are enabled. Pick the slimmest set that
covers your workload.

## Feature matrix

| Feature | Crates | Approx .so size on aarch64 |
|---|---|---|
| (none) | gemm + half + fast_image_resize + image + safetensors | **~4 MB** |
| `llm` | + candle-core / -transformers / -nn + tokenizers | **~7 MB** |
| `onnx` | + tract-onnx | **~25 MB** |
| `llm + onnx` (default) | both | **~28 MB** |

For Nerves devices with multi-GB partitions and plenty of RAM, the
default is fine. For tight deployments, pick what you need:

```elixir
# In your Nerves project's config.exs (mix env :prod typical)
config :nx_arm, features: ["llm"]            # candle only, ~7 MB
config :nx_arm, features: []                 # raw Nx backend only, ~4 MB
config :nx_arm, features: ["onnx"]           # tract only, ~25 MB
```

Default is `["llm", "onnx"]` and matches the precompiled NIF tarballs
in the GitHub release.

## What disabling does

* `features: []` — drops candle, tokenizers, tract. `NxArm.Backend`
  still works (matmul via gemm, all the elementwise/reduction/conv
  NIFs), image resize works, SafeTensors loader works. Calls to
  `NxArm.Models.LlamaCandle.load/2` or `NxArm.Models.Onnx.load/1`
  return `{:error, :llm_feature_disabled}` / `:onnx_feature_disabled`.

* `features: ["llm"]` — Llama-family inference works (TinyLlama,
  Mistral, Phi-3, Qwen via candle). ONNX is unavailable.

* `features: ["onnx"]` — ONNX runs (BERT, sentence encoders, vision
  models exported from HuggingFace). LlamaCandle is unavailable; use
  the ONNX export of your LLM instead (slower than candle for GGUF
  Q4_K_M but works).

## What we actually shipped

The local-build path (Nerves cross-compile) honours
`config :nx_arm, features: [...]`. Precompiled releases on GitHub
always ship the `default` set so consumers don't have to pick at
download time; if you need a slimmer build, set `NX_ARM_BUILD=1`
and your config feature list, and `mix deps.compile nx_arm --force`
will rebuild from source with the trim.

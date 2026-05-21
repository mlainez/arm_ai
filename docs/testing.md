# Testing

What's tested where, and what's intentionally device-only.

## Test layers

```
test/nx_arm/*.exs        — host tests, run via `mix test`
examples/*/run.exs       — device smoke tests, run via mix upload + ssh
```

Host tests are the safety net. Examples are the field-validation
suite — they need a real device because most of the win in this
codebase is in the NEON kernels and the cross-compiled NIF.

## What `mix test` covers (host, x86_64 or ARM dev box)

Runnable in seconds, no Nerves toolchain needed.

| Suite | Purpose |
|---|---|
| `smoke_test.exs` | Backend boot, NIF load, basic op surface |
| `bias_act_test.exs` | Fused bias+activation NEON path |
| `concurrency_test.exs` | NIF re-entrancy / dirty-CPU scheduler safety |
| `conformance_test.exs` | Bit-exact match vs Nx.BinaryBackend for the supported op set |
| `depthwise_conv_test.exs`, `dw_pw_fusion_test.exs` | Depthwise + depthwise-pointwise conv kernels |
| `detection_test.exs` | IoU + NMS correctness |
| `edge_cases_test.exs` | Empty tensors, axis-of-size-1, etc. |
| `embeddings_test.exs` | l2-normalise, cosine sim, top-k |
| `fft_test.exs` | FFT / IFFT correctness against scipy reference |
| `flash_attention_test.exs` | Attention kernel (no-cache variant) |
| `fp16_test.exs` | f16 ↔ f32 dequant matmul |
| `gather_test.exs` | Indexed gather, both fast path and n-D fallback |
| `gradient_test.exs` | Backward-pass support for trainable use |
| `hub_test.exs` | `NxArm.Hub` — caching, SHA verification, transient failures |
| `im2col_conv_test.exs` | Generic im2col conv path |
| `image_test.exs` | Image preprocess transforms |
| `indexed_test.exs` | `indexed_add` / `indexed_put` |
| `int4_test.exs` | int4-packed dequant + matmul |
| `kv_cache_test.exs` | Manual KV cache helper |
| `linear_test.exs`, `llm_linear_test.exs` | Linear layer (bias optional) |
| `llm_test.exs`, `llm_g_test.exs` | LLM op set (no model load) |
| `matmul_blocked_test.exs` | Blocked f32 matmul |
| `mini_transformer_test.exs` | End-to-end transformer block via the backend |
| `mmap_test.exs` | mmap resource lifetime |
| `models_api_test.exs` | Model wrapper error-path contracts |
| `neon_math_test.exs` | NEON intrinsic primitives |
| `nif_robustness_test.exs` | Bad-input handling |
| `per_token_int8_test.exs` | Per-token int8 quantised matmul |
| `phonemizer_test.exs` | Dictionary lookup, OOV fallback, to_phoneme_ids |
| `production_ops_test.exs`, `production_ops2_test.exs` | Production op suite |
| `quantized_test.exs` | Quantised tensor formats |
| `repetition_penalty_test.exs` | Sampling penalty |
| `sampling_test.exs` | Top-k / top-p / temperature |
| `slice_test.exs` | Slicing across the dispatch matrix |
| `thread_pool_test.exs` | Rayon pool init / topology detection |
| `vision_test.exs` | JPEG/PNG decode, classifier preprocess |
| `window_test.exs` | Sliding windows |
| `winograd_test.exs` | 3×3 Winograd conv variant |
| `yolo_test.exs` | YOLO wrapper struct + load error path |

Total: **~245 tests, 18 property tests**, ~10 s wall-clock.

## What only runs on a real device

These need the cross-compiled NIF on real ARM silicon. Run them
via `mix upload` + `ssh nerves.local 'Code.eval_file(...)'`.

| Test | Why device-only |
|---|---|
| `examples/01_chatbot/run.exs` | Needs a 660 MB GGUF, NEON on real ARM |
| `examples/02_voice_transcribe/run.exs` | Needs Whisper GGUF + mel-filter file |
| `examples/03_semantic_search/run.exs` | Same as #1 — small embedder model + corpus |
| `examples/04_yolo_detection/run.exs` | Needs yolov5n.onnx + a JPEG |
| `examples/05_voice_activity/run.exs` | Needs silero_vad.onnx + a WAV |
| `examples/06_text_to_speech/run.exs` | Needs Piper voice + phoneme map |
| `examples/07_image_classifier/run.exs` | Bumblebee + Axon on-device |
| `examples/08_generic_onnx/run.exs` | Any ONNX model |
| `examples/09_voice_assistant_chain/run.exs` | Full chain |

Each example folder has:
* `config.exs` — Nerves config snippet
* `run.exs` — the runnable script
* `README.md` — setup + expected output + honest status

## Recommended workflow

For nx_arm contributors:

```
# Host loop (fast iteration)
mix test                                      # ~10 s
mix test test/nx_arm/<one>_test.exs           # ~1 s

# Device validation (slow, ~5 min cycle)
cd ../hello_nerves
MIX_TARGET=nerves_system_fp3 NX_ARM_BUILD=1 mix firmware
MIX_TARGET=nerves_system_fp3 mix upload
ssh nerves.local 'Code.eval_file("/srv/erlang/lib/nx_arm-0.1.0/examples/01_chatbot/run.exs")'
```

For nx_arm *users* (people deploying to a real device):

```
# 1. Pick the use case (one of the examples folders).
# 2. Copy its config.exs into your config/target.exs.
# 3. mix firmware && mix upload.
# 4. ssh in and run the script.
```

## What we explicitly do *not* test

* **Accuracy of upstream models.** We test that the bridge runs;
  model accuracy is the model's problem (we don't fine-tune or
  re-quantise).
* **Real-time scheduling guarantees.** BEAM is not an RTOS;
  inference latency has tail behaviour. The `with_performance`
  scope helps but doesn't bound it.
* **Power consumption.** Pinning to the perf cluster + holding
  governor "performance" is by design a heat/power tradeoff
  (`NxArm.Performance` docs cover this).
* **Cross-arch determinism.** f32 matmul order is rayon-dependent;
  conformance tests assert within tolerance, not bit-exact.

## Adding a test

Match the style of the closest existing suite. The conventions:

```elixir
defmodule NxArm.MyOpTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  test "describes one specific behaviour" do
    out =
      Nx.tensor([...])
      |> arm()
      |> NxArm.MyOp.run()
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    assert_in_delta Enum.at(out, 0), 1.234, 1.0e-5
  end
end
```

Three rules:
1. `async: true` unless the test mutates global state (governor,
   thread pool).
2. Use `Nx.backend_copy/2` to cross the host backend boundary,
   never re-create tensors with the wrong backend.
3. Assert on tolerance, not equality, for any float path that
   touches NEON (`assert_in_delta`, not `assert ==`).

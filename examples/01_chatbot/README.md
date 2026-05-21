# 01 — On-device chatbot (TinyLlama 1.1B)

A local LLM with no network at inference time. Streams ~5 tokens/sec
on a Fairphone 3 (Cortex-A73 cluster). 638 MB model.

## Set up

1. Copy `config.exs` into your Nerves project's
   `config/target.exs`. It does two things:
   * Trims the firmware to `["chatbot"]` (only candle + tokenizers, no
     ONNX or audio crate weight) — **`.so` ≈ 10 MB**.
   * Tells `NxArm.Hub` to fetch the TinyLlama GGUF + tokenizer on
     first boot.
2. `mix firmware && mix upload` — first boot pulls ~640 MB from
   HuggingFace.
3. `ssh nerves.local` then `Code.eval_file("path/to/run.exs")`.

## Verified on FP3

```
$ ssh nerves.local
iex(1)> Code.eval_file("/tmp/run.exs")
Loading TinyLlama 1.1B Q4_K_M from /root/models/tinyllama.gguf...
Warming up...
Generating 32 tokens after the prompt...

Generated token IDs:
[1, 1724, 338, 263, 1781, 982, 304, 1207, 263, 3632, 331, 1943, 282, 24990, 2181, 504, 393, 338, 1716, 2181, 275, 2272, 322, 923, 12822, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13, 1762, 1207]

Stats:
%{cpu_temp_c: 60.9, decode_ms_per_tok: 214.94, decode_total_ms: 6663.4,
  n_new: 32, n_prompt: 6, prefill_ms: 1124.9,
  prefill_tokens_per_sec: 5.33}

=> 4.65 tokens/sec on this device.
```

## What to change

* For a different model: change the `source:` in `config.exs`. Any
  Llama-family GGUF works (Mistral, Phi, Qwen — candle's
  `quantized_llama` covers them all).
* For string-in/string-out: load with `tokenizer:` pointing at the
  fetched `tokenizer.json` and pass `prompt: "..."` to `generate/2`.
* For different quants: `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`
  all work. `Q4_K_M` is the recommended sweet spot for FP3-class
  hardware.

## What this is NOT good for

* Multi-turn chat: this example calls `generate/2` once. Build the
  loop yourself with the same model handle (no re-load needed
  between turns).
* Streaming output: candle returns the full token list. To stream,
  wrap `LlamaCandle` in a GenServer that loops and emits each token.

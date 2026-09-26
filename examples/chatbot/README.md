# On-device chatbot (TinyLlama 1.1B)

A local LLM with no network at inference time: TinyLlama 1.1B Chat
Q4_K_M (638 MB) through `ArmAI.LlamaCandle`.

## Run it

1. Merge `config.exs` into your firmware config. It builds only the
   `chatbot` Cargo feature and has `nerves_ai`'s model hub fetch the
   GGUF and tokenizer to `/data/models/` on first boot.
2. From iex on the device: `Code.eval_file("/path/to/run.exs")`.

On a host, pass the paths explicitly:

```sh
mix run examples/chatbot/run.exs tinyllama.gguf tinyllama-tokenizer.json
```

## Expected output

On an x86_64 desktop:

```
The capital of France is Paris.

7 tokens, prefill 2896 ms
6.38 tokens/s decode
```

Generation stops at TinyLlama's end-of-sequence token (id 2). On a
Fairphone 3 decode runs at about 4.7 tokens/s; a desktop x86 CPU is
several times faster.

## Changing the model

Any GGUF whose metadata uses `llama.*` keys works (Llama, TinyLlama,
SmolLM, Mistral-style exports), in Q4_0, Q4_K_M, Q5_K_M, Q6_K or Q8_0.
Use the model's own chat template and EOS id for `stop_tokens:`.

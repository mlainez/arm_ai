#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 01 — Chatbot with TinyLlama 1.1B on-device.
#
# Run on a Nerves device after the model is in place (the
# ArmAI.Hub config in `config.exs` downloads it on first boot).
#
# Usage from iex over SSH:
#   Code.eval_file("/path/to/this/run.exs")
# ---------------------------------------------------------------

model_path = "/root/models/tinyllama.gguf"

unless File.exists?(model_path) do
  IO.puts("Model missing at #{model_path}.")
  IO.puts("Either set up the hub config (see config.exs) and reboot,")
  IO.puts("or run ArmAI.Hub.ensure_all() manually now.")
  System.halt(1)
end

IO.puts("Loading TinyLlama 1.1B Q4_K_M from #{model_path}...")
{:ok, model} = ArmAI.LlamaCandle.load(model_path)

# Warmup pass so the first measured run doesn't include cold caches.
IO.puts("Warming up...")
{_, _} = ArmAI.LlamaCandle.generate(model,
  prompt_tokens: [1, 1724, 338, 263],
  max_new: 2
)

# Token IDs come from the TinyLlama tokenizer; if you wired in a
# tokenizer (see config.exs `tokenizer:` line), call with
# `prompt: "..."` instead.
prompt_tokens = [1, 1724, 338, 263, 1781, 982, 304, 1207]  # "What is a good way to make"
IO.puts("Generating 32 tokens after the prompt...")

{tokens, stats} =
  ArmAI.LlamaCandle.generate(model,
    prompt_tokens: prompt_tokens,
    max_new: 32
  )

IO.puts("")
IO.puts("Generated token IDs:")
IO.inspect(tokens)
IO.puts("")
IO.puts("Stats:")
IO.inspect(stats, pretty: true)
IO.puts("")
IO.puts("=> #{Float.round(1000.0 / stats.decode_ms_per_tok, 2)} tokens/sec on this device.")

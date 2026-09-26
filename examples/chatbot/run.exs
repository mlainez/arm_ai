# Example — chatbot with TinyLlama 1.1B Q4_K_M, on device.
#
# Usage from iex on the device (or `mix run` on a host):
#
#     Code.eval_file("run.exs")
#
# Pass different paths as arguments: run.exs MODEL.gguf TOKENIZER.json

[model_path, tokenizer_path] =
  case System.argv() do
    [m, t] -> [m, t]
    _ -> ["/data/models/tinyllama.gguf", "/data/models/tinyllama-tokenizer.json"]
  end

for path <- [model_path, tokenizer_path], not File.exists?(path) do
  raise "missing #{path} — see config.exs for the model download"
end

{:ok, model} = ArmAI.LlamaCandle.load(model_path, tokenizer: tokenizer_path)

# TinyLlama-Chat's template. Token 2 is its end-of-sequence token.
ask = fn question ->
  prompt = "<|user|>\n#{question}</s>\n<|assistant|>\n"
  ArmAI.LlamaCandle.generate(model, prompt: prompt, max_new: 64, stop_tokens: [2])
end

{reply, stats} = ask.("What is the capital of France?")

IO.puts(reply)
IO.puts("")
IO.puts("#{stats.n_new} tokens, prefill #{round(stats.prefill_ms)} ms")

if stats.decode_ms_per_tok do
  IO.puts("#{Float.round(1000.0 / stats.decode_ms_per_tok, 2)} tokens/s decode")
end

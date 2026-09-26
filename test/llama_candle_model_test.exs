defmodule ArmAI.LlamaCandleModelTest do
  # Needs, in $NERVES_AI_MODELS:
  #   tinyllama.gguf            TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF, Q4_K_M
  #   tinyllama-tokenizer.json  TinyLlama/TinyLlama-1.1B-Chat-v1.0 tokenizer.json
  use ExUnit.Case, async: false

  @moduletag :models
  @moduletag timeout: 300_000

  setup_all do
    dir = System.fetch_env!("NERVES_AI_MODELS")

    {:ok, model} =
      ArmAI.LlamaCandle.load(Path.join(dir, "tinyllama.gguf"),
        tokenizer: Path.join(dir, "tinyllama-tokenizer.json")
      )

    %{model: model}
  end

  @prompt "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n"

  test "answers a question and stops at EOS", %{model: model} do
    {reply, stats} =
      ArmAI.LlamaCandle.generate(model, prompt: @prompt, max_new: 48, stop_tokens: [2])

    assert reply =~ "Paris"
    assert stats.n_new < 48
  end

  test "max_new: 0 generates nothing", %{model: model} do
    assert {"", %{n_new: 0}} = ArmAI.LlamaCandle.generate(model, prompt: @prompt, max_new: 0)
  end
end

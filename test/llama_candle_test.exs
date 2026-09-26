defmodule ArmAI.LlamaCandleTest do
  @moduledoc """
  Surface-area tests for `ArmAI.LlamaCandle`. The real inference
  paths need a GGUF model file + a target ARM device — those are
  driven by the examples in `examples/chatbot/` from an actual
  device. Here we lock down the Elixir contract:

  * `load/2` surfaces errors cleanly (missing file, missing
    tokenizer, NIF feature absent)
  * argument parsing for `generate/2` raises on the right inputs
  """

  use ExUnit.Case, async: true

  describe "load/2 error paths" do
    test "missing GGUF returns {:error, _}" do
      assert {:error, _} = ArmAI.LlamaCandle.load("/tmp/__nope.gguf")
    end

    test "missing tokenizer file returns {:error, _}" do
      assert {:error, _} =
               ArmAI.LlamaCandle.load("/tmp/__nope.gguf",
                 tokenizer: "/tmp/__nope_tokenizer.json"
               )
    end

    test "empty path returns {:error, _}" do
      assert {:error, _} = ArmAI.LlamaCandle.load("")
    end
  end

  describe "generate/2 argument validation" do
    test "raises when neither :prompt nor :prompt_tokens passed" do
      # Build a struct that looks like LlamaCandle's output but with a
      # nil handle so we can hit only the arg-parsing arm.
      fake = %ArmAI.LlamaCandle{handle: nil, tokenizer: nil}

      assert_raise ArgumentError, ~r/prompt/i, fn ->
        ArmAI.LlamaCandle.generate(fake, max_new: 4)
      end
    end

    test "raises when :prompt (string) is passed without a tokenizer" do
      fake = %ArmAI.LlamaCandle{handle: nil, tokenizer: nil}

      assert_raise ArgumentError, ~r/tokenizer/, fn ->
        ArmAI.LlamaCandle.generate(fake, prompt: "hello", max_new: 4)
      end
    end
  end

  describe "struct contract" do
    test "fields are stable" do
      m = %ArmAI.LlamaCandle{handle: :placeholder, tokenizer: nil}
      assert m.handle == :placeholder
      assert m.tokenizer == nil
    end
  end
end

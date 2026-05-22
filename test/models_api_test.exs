defmodule ArmAI.ModelsApiTest do
  use ExUnit.Case, async: true

  # arm_ai ships exactly one model wrapper at the Nx-free layer:
  # ArmAI.LlamaCandle. Every other model bridge (Whisper, ONNX,
  # YOLO, etc.) lives in a domain-specific Nx-tensor sibling
  # package (llm, vision, audio) and is tested there.
  #
  # These contract tests confirm the LlamaCandle Elixir surface
  # never raises on the unhappy path — missing file, bad tokenizer
  # path, etc. — even when the NIF is loaded.

  describe "ArmAI.LlamaCandle.load/2" do
    test "missing GGUF returns {:error, _}, never raises" do
      assert {:error, _} = ArmAI.LlamaCandle.load("/tmp/__no_such_gguf.gguf")
    end

    test "accepts a :tokenizer keyword without crashing on missing tokenizer.json" do
      assert {:error, _} =
               ArmAI.LlamaCandle.load("/tmp/__nope.gguf",
                 tokenizer: "/tmp/__nope_tokenizer.json"
               )
    end
  end
end

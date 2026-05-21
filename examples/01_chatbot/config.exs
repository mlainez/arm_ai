# Copy this block into your Nerves project's `config/target.exs`.
import Config

# Smallest firmware that runs an LLM (no ONNX, no vision, no audio).
config :nx_arm, features: ["chatbot"]

# First-boot model download. The device fetches both files to
# /root/models/ on the first boot where they're missing; subsequent
# boots short-circuit.
config :nx_arm,
  models: [
    tinyllama: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
      path: "/root/models/tinyllama.gguf"
    ],
    tinyllama_tokenizer: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tokenizer.json"},
      path: "/root/models/tinyllama-tokenizer.json"
    ]
  ]

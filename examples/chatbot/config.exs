import Config

# Fetch the model and tokenizer on first boot with nerves_ai's model hub.
config :nerves_ai,
  models: [
    tinyllama: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
      path: "/data/models/tinyllama.gguf"
    ],
    tinyllama_tokenizer: [
      source: {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.json"},
      path: "/data/models/tinyllama-tokenizer.json"
    ]
  ]

# Optional: build only the Cargo features this example needs.
config :arm_ai, features: ["chatbot"]

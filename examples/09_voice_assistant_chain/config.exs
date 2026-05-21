import Config

config :nx_arm, features: ["voice-assistant"]

# Pulls the four models needed for the chain. ~125 MB total.
config :nx_arm,
  models: [
    silero_vad: [
      source: {:hf, "snakers4/silero-vad", "src/silero_vad/data/silero_vad.onnx"},
      path: "/root/models/silero_vad.onnx"
    ],
    whisper_tiny: [
      source: {:hf, "ggerganov/whisper.cpp", "ggml-tiny.bin"},
      path: "/root/models/whisper-tiny.gguf"
    ],
    whisper_tokenizer: [
      source: {:hf, "openai/whisper-tiny", "tokenizer.json"},
      path: "/root/models/whisper-tokenizer.json"
    ],
    tinyllama: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
      path: "/root/models/tinyllama.gguf"
    ],
    tinyllama_tokenizer: [
      source: {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.json"},
      path: "/root/models/tinyllama-tokenizer.json"
    ],
    piper_amy: [
      source: {:hf, "rhasspy/piper-voices",
                    "en/en_US/amy/medium/en_US-amy-medium.onnx"},
      path: "/root/models/piper-amy-medium.onnx"
    ]
  ]

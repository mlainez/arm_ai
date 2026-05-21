import Config

config :nx_arm, features: ["whisper"]

# Whisper-tiny GGUF bundle from https://huggingface.co/ggerganov/whisper.cpp
# (the canonical Whisper export host). ~75 MB total.
config :nx_arm,
  models: [
    whisper_tiny: [
      source: {:hf, "ggerganov/whisper.cpp", "ggml-tiny.bin"},
      path: "/root/models/whisper-tiny.gguf"
    ],
    whisper_tokenizer: [
      source: {:hf, "openai/whisper-tiny", "tokenizer.json"},
      path: "/root/models/whisper-tokenizer.json"
    ],
    whisper_config: [
      source: {:hf, "openai/whisper-tiny", "config.json"},
      path: "/root/models/whisper-config.json"
    ],
    whisper_mel_filters: [
      source: {:hf, "openai/whisper-tiny.en", "preprocessor_config.json"},
      path: "/root/models/whisper-mel-filters.bin"
    ]
  ]

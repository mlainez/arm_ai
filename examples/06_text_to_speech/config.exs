import Config

config :nx_arm, features: ["piper-tts"]

# Piper "Amy" English voice, medium quality (~25 MB).
config :nx_arm,
  models: [
    piper_amy: [
      source: {:hf, "rhasspy/piper-voices",
                    "en/en_US/amy/medium/en_US-amy-medium.onnx"},
      path: "/root/models/piper-amy-medium.onnx"
    ],
    piper_amy_config: [
      source: {:hf, "rhasspy/piper-voices",
                    "en/en_US/amy/medium/en_US-amy-medium.onnx.json"},
      path: "/root/models/piper-amy-medium.onnx.json"
    ]
  ]

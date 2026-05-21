import Config

# VAD only needs ONNX + audio decode; no LLM/tokenizer/vision.
config :nx_arm, features: ["onnx", "audio"]

config :nx_arm,
  models: [
    silero_vad: [
      source: {:hf, "snakers4/silero-vad", "src/silero_vad/data/silero_vad.onnx"},
      path: "/root/models/silero_vad.onnx"
    ]
  ]

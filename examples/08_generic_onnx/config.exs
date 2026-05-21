import Config

config :nx_arm, features: ["onnx", "vision"]

# MobileNetV2 ONNX from the ONNX Model Zoo (~13 MB).
config :nx_arm,
  models: [
    mobilenetv2: [
      source: {:url, "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"},
      path: "/root/models/mobilenetv2.onnx"
    ]
  ]

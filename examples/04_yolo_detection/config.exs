import Config

config :nx_arm, features: ["yolo"]

# YOLOv8n ONNX (~12 MB) from the official Ultralytics export.
config :nx_arm,
  models: [
    yolov8n: [
      source: {:url, "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"},
      path: "/root/models/yolov8n.onnx"
    ]
  ]

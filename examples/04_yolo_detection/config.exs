import Config

config :nx_arm, features: ["yolo"]

# YOLOv5n ONNX (~3.8 MB), opset 17, official Ultralytics release.
# Already uses Resize (not Upsample) so tract-onnx loads it without
# any opset conversion. 3× smaller and ~30% less compute than v8n.
config :nx_arm,
  models: [
    yolov5n: [
      source: {:url, "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"},
      path: "/root/models/yolov5n.onnx"
    ]
  ]

#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 04 — Still-image object detection with YOLOv5n.
#
# Snapshot in, JSON detections out. Optimised for capture-and-
# analyse workflows (kiosk, inventory scan, snapshot-triggered
# security, wildlife cam-trap) where ~1 s of latency per image
# is acceptable and the 640×640 fp32 accuracy is worth keeping.
#
# Why YOLOv5n (vs v8): the official Ultralytics release ships at
# opset 17 with `Resize`, so it loads cleanly into tract-onnx with
# no extra conversion step. It's also ~3× smaller (3.8 MB) and a
# bit faster on CPU.
# ---------------------------------------------------------------

image_path = "/root/sample.jpg"
model_path = "/root/models/yolov5n.onnx"

unless File.exists?(model_path) and File.exists?(image_path) do
  IO.puts("Missing #{model_path} or #{image_path}.")
  IO.puts("See config.exs for the NxArm.Hub config that fetches the model.")
  System.halt(1)
end

IO.puts("Loading YOLOv5n...")
{:ok, yolo} = NxArm.Models.YOLO.load(model_path, layout: :v5, input_shape: {640, 640})

IO.puts("Preprocessing image...")
input =
  NxArm.Vision.load_for_classifier(image_path,
    size: {640, 640},
    layout: :nchw,
    # YOLO expects 0-1 normalised RGB (no ImageNet mean/std).
    mean: {0.0, 0.0, 0.0},
    std: {1.0, 1.0, 1.0}
  )

IO.puts("Running detection...")
{us, detections} =
  :timer.tc(fn ->
    NxArm.Models.YOLO.detect(yolo, input,
      iou_threshold: 0.45,
      score_threshold: 0.25,
      max_output: 50
    )
  end)

IO.puts("")
IO.puts("Detected #{length(detections)} objects in #{div(us, 1000)} ms")

for {det, i} <- Enum.with_index(detections) do
  %{box: {x1, y1, x2, y2}, class: c, score: s} = det
  IO.puts("  ##{i}: class=#{c} score=#{Float.round(s, 3)} box=[#{Float.round(x1, 0)}, #{Float.round(y1, 0)}, #{Float.round(x2, 0)}, #{Float.round(y2, 0)}]")
end

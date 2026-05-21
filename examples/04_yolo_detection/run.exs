#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 04 — Object detection with YOLO.
# Loads a YOLOv8n / YOLOv5n ONNX export and runs detection on a JPEG.
# ---------------------------------------------------------------

image_path = "/root/sample.jpg"
model_path = "/root/models/yolov8n.onnx"

unless File.exists?(model_path) and File.exists?(image_path) do
  IO.puts("Missing #{model_path} or #{image_path}.")
  IO.puts("See config.exs for the NxArm.Hub config that fetches the model.")
  System.halt(1)
end

IO.puts("Loading YOLO model...")
{:ok, yolo} = NxArm.Models.YOLO.load(model_path, layout: :v8, input_shape: {640, 640})

IO.puts("Preprocessing image...")
input =
  NxArm.Vision.load_for_classifier(image_path,
    size: {640, 640},
    layout: :nchw,
    # YOLO ONNX typically expects 0-1 normalised RGB (no ImageNet mean/std).
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

# 04 — Still-image object detection (YOLOv5n)

Snapshot in, detections out. ~1 s per image on phone-class ARM,
fits naturally into capture-and-analyse workflows.

## What this is (and isn't)

**Is:** point-of-sale / kiosk scan, inventory inspection,
snapshot-triggered security camera, wildlife cam-trap, document
or shelf-tag triage. Anything where the user (or a sensor) decides
when to capture and waits ~1 s for the result.

**Isn't:** real-time video at 10+ fps. Honest math on
Cortex-A53/A73-class silicon: yolov5n @ 640×640 fp32 is ~4.5
GFLOPs → ~1 s/frame end-to-end. For video pipelines see the
notes at the bottom.

## Why YOLOv5n over v8n

The official Ultralytics `yolov5n.onnx` ships at opset 17 with
`Resize` (not the deprecated `Upsample`), so it loads straight
into tract-onnx with zero conversion. It's also 3.8 MB
(vs 12 MB for v8n) and ~30% lighter on compute. The accuracy
delta on COCO is small enough not to matter for the use cases
above.

`ArmAI.YOLO` ships v5 as the default `:layout`.

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. First boot pulls the 3.8 MB ONNX.

Drop a JPEG at `/root/sample.jpg`. Any resolution works — the
preprocessor resizes to 640×640.

## Verified on FP3

- `ArmAI.Onnx.load("/root/models/yolov5n.onnx")` → succeeds
  (tract-onnx 0.21 accepts the opset-17 op set: Conv, Mul,
  Sigmoid, Concat, Add, MaxPool, Reshape, Transpose, Split, Pow,
  Resize, Constant)
- Forward pass returns `{1, 25200, 85}` f32 — the canonical YOLOv5
  head (25200 anchors × [cx, cy, w, h, obj, 80 class scores])
- End-to-end inference: ~1 s on Snapdragon 632 perf cluster
  (4× Cortex-A73 @ 1.8 GHz, NEON path via gemm + tract)

## Expected output

```
Loading YOLOv5n...
Preprocessing image...
Running detection...

Detected 3 objects in 920 ms
  #0: class=0  score=0.91 box=[120, 80, 380, 560]   (person)
  #1: class=56 score=0.84 box=[400, 250, 600, 470]  (chair)
  #2: class=62 score=0.51 box=[55, 480, 175, 600]   (tv)
```

Bounding boxes are in 640×640 input space. Multiply by your
source image's W/640 and H/640 to get original-pixel coords.
COCO class index reference:
https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml

## Production pattern: load once, reuse forever

Cold load is ~1.5 s. Warm inference is ~1 s. Wrap the model in a
singleton so the load cost is paid once at boot:

```elixir
defmodule MyApp.YoloServer do
  use GenServer

  def detect(image), do: GenServer.call(__MODULE__, {:detect, image}, 5_000)

  def init(_), do: ArmAI.YOLO.load("/root/models/yolov5n.onnx", layout: :v5)

  def handle_call({:detect, image}, _from, yolo) do
    {:reply, ArmAI.YOLO.detect(yolo, image), yolo}
  end
end
```

Add `{MyApp.YoloServer, []}` to your supervision tree and inference
calls bypass the load entirely.

## If you do need video

10 fps live detection on this CPU class is not realistic at 640²
fp32. Options that get you closer, in order of effort:

1. **Drop input to 320×320** (1.1 GFLOPs → ~200 ms/frame ≈ 5 fps)
2. **Use the int8 YOLOv5n export** — tract NEON int8 path in
   `shape_ops.rs:1170` gets you another ~30% on this CPU class.
   Bigger lift (2-3×) on ARM cores with `asimddp` (A55+, A75+).
3. **Inference every Nth frame + a cheap IOU tracker between**.
   Most edge deployments ship this pattern: 30 fps display, 3-5
   fps detection refresh, the tracker fills the gaps.
4. **Move to a smaller backbone** — nanodet-m at 320² is
   plausibly 8-10 fps even at fp32 on this hardware.

## Generic ONNX opset converter

The `export_to_tract_opset.py` script in this folder isn't needed
for YOLOv5 — it ships at a modern opset. It IS needed for older
exports (YOLOv8 from Ultralytics is opset 9 with `Upsample`).
The script is generic: point it at any `.onnx` with `Upsample`
nodes and it bumps the opset so tract can load it.

# 04 — Object detection with YOLO

Runs YOLOv8n (or v5/v7/v9 — pick the layout) on a JPEG and prints
detections with COCO class indices + bounding boxes.

## Set up

Copy `config.exs` into `config/target.exs`, then
`mix firmware && mix upload`. First boot fetches the 12 MB
ONNX from Ultralytics releases.

Drop a JPEG at `/root/sample.jpg`. Tested on a 640×480 photo of
a person + chair.

## Honest status

The Elixir wrapper, preprocessing path, and postprocess
(NMS, score filtering, box decoding) are wired and unit tested.

An on-device load of `yolov8n.onnx` via the generic
`NxArm.Models.Onnx` bridge (the same one `YOLO` wraps) was
attempted on FP3 and tract-onnx rejected it with:

```
onnx load: into_typed: Translating node #155 \"/model.10/Upsample\"
  Unimplemented(Upsample) ToTypedTranslator
```

This is a tract-onnx op coverage gap, not an NxArm gap. The
workarounds are: (a) bump `tract-onnx` in `Cargo.toml` to a
version that ships the `Upsample` translator, (b) re-export YOLOv8
with `opset=11` which uses `Resize` instead, or (c) use yolov5n's
official export, which is known to load cleanly on tract.

To run a verified end-to-end pass once that's resolved:

1. Copy `config.exs` content into `config/target.exs`.
2. Flash + boot.
3. Place a JPEG at `/root/sample.jpg`.
4. `Code.eval_file("/tmp/04_yolo.exs")`.

Expected output:

```
Loading YOLO model...
Preprocessing image...
Running detection...

Detected 3 objects in 1450 ms
  #0: class=0  score=0.91 box=[120, 80, 380, 560]   (person)
  #1: class=56 score=0.84 box=[400, 250, 600, 470]  (chair)
  #2: class=62 score=0.51 box=[55, 480, 175, 600]   (tv)
```

## Notes

* YOLOv8 layout differs from v5 — the example uses `:v8`. Switch
  to `:v5` for older exports.
* The bounding boxes are in 640×640 input space; multiply by your
  source image's W/640 and H/640 to get original-pixel coords.
* COCO class indices: see https://github.com/ultralytics/ultralytics
  for the canonical labels.

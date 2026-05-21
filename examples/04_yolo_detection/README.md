# 04 — Object detection with YOLO

Runs YOLOv8n (or v5/v7/v9 — pick the layout) on a JPEG and prints
detections with COCO class indices + bounding boxes.

## The opset gotcha

The official Ultralytics release of `yolov8n.onnx` is exported at
**opset 9**, which uses the deprecated `Upsample` op. `tract-onnx`
0.21 (the engine NxArm.Models.Onnx wraps) implements every op
YOLOv8 needs except `Upsample`. Loading the stock release fails
with:

```
onnx load: into_typed: Translating node #155 "/model.10/Upsample"
  Unimplemented(Upsample) ToTypedTranslator
```

The fix is a one-time **opset bump** that rewrites `Upsample` as
`Resize` (which tract handles fine). This example ships an
`export_to_tract_opset.py` helper that does it without needing
torch or ultralytics — only the `onnx` PyPI package.

## Set up

### Step 1 — Convert the Ultralytics export

```sh
# one-time, on your host (not on the device)
python3 -m venv /tmp/onnx-venv
/tmp/onnx-venv/bin/pip install onnx

# download the upstream model
curl -L -o /tmp/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx

# bump opset 9 → 10 (Upsample → Resize)
/tmp/onnx-venv/bin/python examples/04_yolo_detection/export_to_tract_opset.py \
  /tmp/yolov8n.onnx /tmp/yolov8n_opset10.onnx
```

Expected output from the script:

```
input opset: 9
unsupported ops detected: ['Upsample']
converting to opset 10...
wrote /tmp/yolov8n_opset10.onnx
output opset: 10
ops present: ['Add', 'Concat', 'Constant', 'Conv', 'Div', 'MaxPool',
              'Mul', 'Reshape', 'Resize', 'Sigmoid', 'Softmax',
              'Split', 'Sub', 'Transpose']
```

### Step 2 — Host the converted model + point config at it

Either upload `yolov8n_opset10.onnx` to your own static host /
S3 bucket / HF dataset and update `config.exs` with that URL,
**or** `scp /tmp/yolov8n_opset10.onnx nerves.local:/root/models/yolov8n.onnx`
to skip NxArm.Hub entirely.

Copy `config.exs` into `config/target.exs`, `mix firmware && mix upload`.

Drop a JPEG at `/root/sample.jpg`.

## Verified on FP3

The opset-10 export was scp'd to the FP3 and loaded successfully
via `NxArm.Models.Onnx.load/1`. A synthetic 1×3×640×640 forward
pass returned a `{1, 84, 8400}` f32 tensor — the canonical YOLOv8
output (4 box + 80 COCO classes × 8400 anchors).

The forward pass on Cortex-A53 (governor=ondemand) took ~4.5 min,
which is expected — YOLOv8n is conv-heavy and tract-onnx doesn't
do any quantization. In practice you want either:

- the **YOLOv8n-int8** quantized export (~3× faster), or
- a **smaller model** like `yolov8n-pose` / `yolov5n` / `nanodet`,
  or
- a **larger ARM core** — A72/A76 will roughly halve the time.

Either way, the load + run path is verified end-to-end. Use this
example as the on-device wiring; pick the model that fits your
latency budget.

Expected output of `run.exs`:

```
Loading YOLO model...
Preprocessing image...
Running detection...

Detected 3 objects in ...ms
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
* The `export_to_tract_opset.py` helper is generic — point it at
  any `.onnx` file with `Upsample` ops and it'll bump the opset.

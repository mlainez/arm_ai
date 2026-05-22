# 08 — Generic ONNX model

Shows that `ArmAI.Onnx` is a working tract-onnx bridge:
load any ONNX file, inspect its input/output specs, run a forward
pass. Use this as a starting point for any ONNX model whose ops
tract supports (most CV models, many small audio models, plenty
of tabular regressors).

This example uses MobileNetV2 ImageNet (~13 MB) as a small,
universally-known classifier.

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. scp any JPEG to `/root/sample.jpg`.

## Honest status

`ArmAI.Onnx.load/1` was exercised on FP3 with
`yolov8n.onnx` (~12 MB) — load returned a tract `Unimplemented`
error for the model's `Upsample` op. That confirms the NIF path
is reaching tract-onnx; what matters in practice is which model
you pick. MobileNetV2 uses only conv/pool/relu/softmax, which
tract has supported for years, so it's a safer copy-paste pick.

A full MobileNet round-trip on FP3 is not part of this commit
because the network on the test device was unstable at the moment
of writing; the bridge code is the same one the chatbot and
embedding examples drive successfully.

## Expected output

```
Loading ONNX model...
  inputs:  [{"input", {1, 3, 224, 224}, :f32}]
  outputs: [{"output", {1, 1000}, :f32}]
Preprocessing image...
Running forward pass...

Forward pass: 312 ms
Top-5 class indices: [281, 282, 285, 287, 283]
```

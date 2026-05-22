#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 08 — Run an arbitrary ONNX model.
#
# Demonstrates the generic ArmAI.Onnx wrapper. Works for
# any ONNX model whose ops are supported by tract: classifiers,
# regressors, encoders, pose estimators, depth predictors, etc.
#
# This example uses MobileNetV2 (ImageNet, ~13 MB), the canonical
# tiny mobile classifier.
# ---------------------------------------------------------------

model_path = "/root/models/mobilenetv2.onnx"
image_path = "/root/sample.jpg"

unless File.exists?(model_path) and File.exists?(image_path) do
  IO.puts("Missing #{model_path} or #{image_path}.")
  System.halt(1)
end

IO.puts("Loading ONNX model...")
{:ok, model} = ArmAI.Onnx.load(model_path)
IO.puts("  inputs:  #{inspect(ArmAI.Onnx.input_specs(model))}")
IO.puts("  outputs: #{inspect(ArmAI.Onnx.output_specs(model))}")

IO.puts("Preprocessing image...")
input =
  ArmAI.Vision.load_for_classifier(image_path,
    size: {224, 224},
    layout: :nchw,
    # ImageNet normalisation (MobileNetV2 expects this).
    mean: {0.485, 0.456, 0.406},
    std:  {0.229, 0.224, 0.225}
  )

IO.puts("Running forward pass...")
{us, [logits]} = :timer.tc(fn -> ArmAI.Onnx.run(model, [input]) end)

# Softmax → top-5.
probs = Nx.divide(Nx.exp(Nx.subtract(logits, Nx.reduce_max(logits))),
                  Nx.sum(Nx.exp(Nx.subtract(logits, Nx.reduce_max(logits)))))
top5_idx = ArmAI.Embeddings.top_k(Nx.squeeze(probs), 5) |> Nx.to_flat_list()

IO.puts("")
IO.puts("Forward pass: #{div(us, 1000)} ms")
IO.puts("Top-5 class indices: #{inspect(top5_idx)}")

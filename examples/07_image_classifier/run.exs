#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 07 — Image classification via Bumblebee + Axon + NxArm.
#
# Loads a Bumblebee-supported model (here: ViT-tiny ImageNet) and
# classifies a JPEG. Demonstrates that NxArm is a drop-in Nx
# backend for the Bumblebee/Axon stack on ARM Nerves devices.
# ---------------------------------------------------------------

Nx.global_default_backend(NxArm.Backend)

image_path = "/root/cat.jpg"

unless File.exists?(image_path) do
  IO.puts("Missing #{image_path}.")
  IO.puts("scp any JPEG to /root/cat.jpg to run this example.")
  System.halt(1)
end

IO.puts("Loading ViT-tiny model + preprocessor...")
{:ok, model_info} = Bumblebee.load_model({:hf, "WinKawaks/vit-tiny-patch16-224"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, "WinKawaks/vit-tiny-patch16-224"})

IO.puts("Loading image and featurizing...")
image = Image.open!(image_path)
inputs = Bumblebee.apply_featurizer(featurizer, image)

IO.puts("Building forward pass...")
{model, params} = {model_info.model, model_info.params}

IO.puts("Running prediction...")
{us, output} = :timer.tc(fn -> Axon.predict(model, params, inputs, compiler: Nx.Defn.Evaluator) end)

logits = output.logits |> Nx.squeeze()
top5 = NxArm.Embeddings.top_k(Nx.negate(Nx.negate(logits)), 5) |> Nx.to_flat_list()

IO.puts("")
IO.puts("Forward pass: #{div(us, 1000)} ms")
IO.puts("Top-5 ImageNet class indices: #{inspect(top5)}")

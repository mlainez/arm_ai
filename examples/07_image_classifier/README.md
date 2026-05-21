# 07 — Image classifier (Bumblebee + Axon)

Classifies a JPEG against ImageNet using a tiny ViT model and
the standard Bumblebee/Axon stack. Demonstrates that NxArm is a
working `Nx.global_default_backend/1` for end-to-end Bumblebee
inference on ARM/Nerves.

## Set up

Copy `config.exs` into `config/target.exs`,
`mix firmware && mix upload`. Bumblebee fetches and caches model
weights on first run; on Nerves we cache to `/root/.bumblebee` so
it survives a re-flash of the rootfs (which is read-only).

scp any JPEG to `/root/cat.jpg`.

Add to your `mix.exs`:

```elixir
{:bumblebee, "~> 0.5"},
{:image, "~> 0.54"}
```

## Honest status

Verified on FP3 (the project memory entry "FP3 Bumblebee status"
documents the prior end-to-end run, ~3 s warm ViT-tiny forward
pass). This example formalises that run as a copy-paste recipe.

## Expected output

```
Loading ViT-tiny model + preprocessor...
Loading image and featurizing...
Building forward pass...
Running prediction...

Forward pass: 2987 ms
Top-5 ImageNet class indices: [281, 282, 285, 287, 283]
```
(281 = "tabby cat", 282 = "tiger cat", 285 = "Egyptian cat", etc.)

# First-boot model downloads (`ArmAI.Hub`)

Models can be fetched on first boot from HuggingFace or any HTTPS
URL, so the firmware itself stays small and the user doesn't have
to scp weights to every device.

## Quick start

```elixir
# config/target.exs (or config/config.exs)
config :nx_arm,
  models: [
    tinyllama: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
      path: "/root/models/tinyllama.gguf"
    ],
    tinyllama_tokenizer: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tokenizer.json"},
      path: "/root/models/tinyllama-tokenizer.json"
    ]
  ]
```

`ArmAI.Application` calls `ArmAI.Hub.ensure_all/0` on boot. Missing
files are streamed to disk synchronously, then your supervisor
starts. Successful boots after the first reuse the cached files.

## Looking up paths at runtime

```elixir
{:ok, model_path} = ArmAI.Hub.path(:tinyllama)
{:ok, tok_path}   = ArmAI.Hub.path(:tinyllama_tokenizer)

{:ok, model} = ArmAI.LlamaCandle.load(model_path, tokenizer: tok_path)
```

## Source forms

* `{:hf, "owner/repo", "file"}` — HuggingFace, `main` branch
* `{:hf, "owner/repo", "file", revision: "v1.0.0"}` — a specific
  branch or commit
* `{:url, "https://example.com/file.bin"}` — any HTTPS URL

## Integrity check (optional)

Add a `:sha256` hex digest to fail-closed against tampering or
truncation:

```elixir
config :nx_arm,
  models: [
    tinyllama: [
      source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
      path: "/root/models/tinyllama.gguf",
      sha256: "deadbeef..."
    ]
  ]
```

If the cached file's SHA doesn't match, `ArmAI.Hub` deletes it and
re-downloads.

## Networking expectations

`ArmAI.Hub` uses Erlang's stdlib `:httpc` — no new Rust crates,
no extra runtime deps. You're responsible for bringing up
connectivity before nx_arm starts; on Nerves the typical pattern is
to let `vintage_net` come up first.

If you boot offline, `ensure_all/0` returns `{:error, [{id, reason}]}`
for every missing model and the Application log records each
failure but keeps booting. Cached files are still usable.

## Atomicity

Each download writes to `<path>.partial` and renames on success. A
crash mid-download leaves the partial file; the next boot retries
from scratch. There is no resume support today (planned).

## Storage location

Combine with `ArmAI.StorageResizer`: by default
`/dev/mmcblk0p62p3` is auto-grown to fill the data partition on
first boot, so a 700 MB TinyLlama GGUF + 50 MB tokenizer fits on
a stock FP3 image. Models live under `/root/models/` by convention
— pick any path you like.

## Per-model logging

Each download logs:

```
[nx_arm.hub] tinyllama: fetching https://... → /root/models/tinyllama.gguf
[nx_arm.hub] tinyllama: OK (638 MB)
[nx_arm] model hub: 2 model(s) ready
```

On failure:

```
[nx_arm.hub] tinyllama: fetching ...
[nx_arm] model hub: tinyllama failed: {:http_status, 404}
```

## Pattern: bundle the config with each release

For each application, ship one `config/target.exs` per
deployment variant:

```elixir
# config/target.exs (production release)
config :nx_arm,
  features: ["chatbot"],
  models: [
    llama: [
      source: {:hf, "your-org/your-llama", "model.gguf"},
      path: "/root/models/llama.gguf"
    ],
    tokenizer: [
      source: {:hf, "your-org/your-llama", "tokenizer.json"},
      path: "/root/models/tokenizer.json"
    ]
  ]
```

The same config selects the firmware's feature set (slim builds via
`config :nx_arm, features: ["chatbot"]`) and the runtime models.
Devices boot, fetch, and run with no further intervention.

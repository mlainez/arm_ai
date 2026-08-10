defmodule ArmAI do
  @moduledoc """
  Edge AI inference NIF for ARM CPUs.

  This package ships the Rust NIF (`ArmAI.Native`) with the
  NEON-tuned compute primitives, candle bridge for quantised
  LLMs and Whisper, tract-onnx bridge, symphonia + rubato for
  audio, `image` + `fast_image_resize` for vision preprocessing.

  ## Public Nx-free APIs

  * `ArmAI.LlamaCandle` — quantised Llama-family inference
    (token IDs in → token IDs out + stats)
  * `ArmAI.Phonemizer` — text → phoneme symbols (G2P)

  ## Shims to companion packages

  * `ArmAI.Performance` → `CpuGovernor.Performance`
  * `ArmAI.Runtime` → `CpuGovernor.Topology`

  Things `arm_ai` deliberately does **not** know about:

  * `nerves_model_hub` (first-boot HF/URL downloads) — application
    concern; depend on it from your `:nerves_ai` (or your own
    application) layer.
  * `nerves_data_resize` (first-boot F2FS grow) — same: orchestrated
    by `nerves_ai`, not by the NIF.

  ## Backend implementations for the generic libraries

  Higher-level Nx-tensor APIs live in sibling packages with
  pluggable backends. `arm_ai` provides the ARM-NEON impl of each:

  | Generic library     | Behaviour              | ARM impl in `arm_ai`        |
  |---|---|---|
  | `nx_arm`            | `Nx.Backend`           | `NxArm.Backend`             |
  | `nx_primitives`     | `NxPrimitives.Backend` | `ArmAI.NxPrimitivesBackend` |
  | `infer_llm`         | `InferLLM.Backend`          | `ArmAI.LLMBackend`          |
  | `infer_vision`      | `InferVision.Backend`       | `ArmAI.VisionBackend`       |
  | `infer_audio`       | `InferAudio.Backend`        | `ArmAI.AudioBackend`        |

  Activate the ARM impls in your app config (the `nerves_ai`
  meta-package does this automatically at boot):

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
      config :infer_llm,           backend: ArmAI.LLMBackend
      config :infer_vision,        backend: ArmAI.VisionBackend
      config :infer_audio,         backend: ArmAI.AudioBackend

  `arm_ai` itself has no `Nx` dependency. Depend on it directly
  for embedded inference apps that don't need the Nx ecosystem.
  """
end

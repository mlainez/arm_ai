defmodule ArmAI do
  @moduledoc """
  Edge AI inference NIF for ARM CPUs.

  This package ships the Rust NIF (`ArmAI.Native`): NEON-tuned compute
  kernels, candle for quantized Llama and Whisper, tract for ONNX,
  symphonia + rubato for audio, and `image` + `fast_image_resize` for
  image preprocessing.

  ## Public API

  * `ArmAI.LlamaCandle` — quantized Llama-architecture GGUF inference
    (token ids or strings in, generated tokens out, plus timing stats)
  * `ArmAI.Runtime` — rayon thread-pool setup and CPU topology
  * `ArmAI.Performance` — delegates to `CpuGovernor.Performance`

  ## Backends for the generic libraries

  | Generic library | Behaviour              | Implementation in `arm_ai`  |
  |---|---|---|
  | `nx_primitives` | `NxPrimitives.Backend` | `ArmAI.NxPrimitivesBackend` |
  | `infer_llm`     | `InferLLM.Backend`     | `ArmAI.LLMBackend`          |
  | `infer_vision`  | `InferVision.Backend`  | `ArmAI.VisionBackend`       |
  | `infer_audio`   | `InferAudio.Backend`   | `ArmAI.AudioBackend`        |

  `nx_arm` builds its `Nx.Backend` on the same kernels.

  Activate the backends in your config (the `nerves_ai` meta-package
  does this at boot):

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
      config :infer_llm, backend: ArmAI.LLMBackend
      config :infer_vision, backend: ArmAI.VisionBackend
      config :infer_audio, backend: ArmAI.AudioBackend

  `ArmAI.LlamaCandle` needs no Nx. The backend modules use Nx, which the
  generic libraries bring in.
  """
end

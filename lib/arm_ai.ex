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

  * `ArmAI.Hub` → `ModelHub`
  * `ArmAI.Performance` → `CpuGovernor.Performance`
  * `ArmAI.Runtime` → `CpuGovernor.Topology`
  * `ArmAI.StorageResizer` → `FwupDataResize`

  ## Companion packages

  Higher-level Nx-tensor APIs live in sibling packages:

  * `nx_arm` — `Nx.Backend` impl over `ArmAI.Native`
  * `arm_nx_primitives` — FFT, embeddings, quantized matmul + conv
  * `arm_llm` — LLM + STT Nx wrappers (Llama, Whisper)
  * `arm_vision` — YOLO, OCR, Face, ONNX, vision preprocessing
  * `arm_audio` — Silero VAD, Piper, audio decode/resample

  `arm_ai` itself has no `Nx` dependency. Depend on it directly
  for embedded inference apps that don't need the Nx ecosystem.
  """
end

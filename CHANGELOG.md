# Changelog

## Unreleased

### Fixed

* Whisper transcription produced garbage: the greedy decoder fed only the
  last token each step and the mel spectrogram was not trimmed to 30 s.
  Transcripts are now correct.
* The ONNX backend passed the wrong handle, argument order and output
  shape to the NIF, so every ONNX call failed. Half-precision models are
  now translated to f32 at load time.
* `ArmAI.LlamaCandle.generate/2` honours `stop_tokens:` and returns only
  the generated tokens. `load/2` reports tokenizer load failures.
* An explicit `thread_count` made `ArmAI.Runtime.init_thread_pool/0`
  return a shape its caller didn't match.
* `bias_add_activation_f32_op` aborted the VM on an unknown activation
  name or empty input; it now returns an error.
* `remainder` used Euclidean instead of truncated semantics.

### Changed

* Config keys and the build env var are now `:arm_ai` /
  `ARM_AI_BUILD` (were `:nx_arm` / `NX_ARM_BUILD`). The supervisor is
  `ArmAI.Supervisor`.
* `rustler` and `tokenizers` are required dependencies.
* Toolchain: Erlang/OTP 29.1.1 and Elixir 1.20.4, matching the official
  Nerves systems. Requires Elixir `~> 1.17` and Nx `~> 0.12.0`.

### Removed

* `ArmAI.Phonemizer`, the Silero VAD and Piper TTS backend callbacks,
  and the `piper-tts`, `voice-assistant`, `sentence-rag`, `vision-only`
  and `pgo` Cargo features. None of them worked: the ONNX path cannot
  feed the integer inputs these models need.
* The `quantized_matmul` / `quantized_conv2d` backend callbacks, which
  only raised.
* The musl targets from CI, the release matrix and the precompiled target
  list. Nerves toolchains are glibc, and the musl builds didn't compile.
* Elixir stubs with no NIF behind them (`tokenizer_*`,
  `safetensors_load_op`), and the uncompiled
  `quantized_llama_inplace.rs` experiment.
* Docs describing modules that no longer exist in this package.

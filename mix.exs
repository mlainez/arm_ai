defmodule ArmAI.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :arm_ai,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "ArmAI",
      description:
        "Edge AI inference NIF for ARM CPUs — Llama / Whisper / ONNX via Candle + tract-onnx, with hand-tuned NEON kernels",
      docs: [main: "readme", extras: ["README.md"]],
      package: package()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger],
      mod: {ArmAI.Application, []}
    ]
  end

  def package do
    [
      name: :arm_ai,
      licenses: ["Apache-2.0"],
      files: ~w(lib native/arm_ai_nif/src native/arm_ai_nif/Cargo.toml
                native/arm_ai_nif/Cargo.lock README.md mix.exs
                checksum-Elixir.ArmAI.Native.exs),
      links: %{"GitHub" => "https://github.com/marclainez/arm_ai"}
    ]
  end

  defp deps do
    [
      # CPU governor scoping + topology — used by ArmAI.Performance
      # and ArmAI.Runtime shim modules. Generic Linux helpers.
      {:cpu_governor, path: "../cpu_governor"},
      # First-boot model downloader — used by ArmAI.Hub shim.
      {:model_hub, path: "../model_hub"},
      # First-boot F2FS resize — used by ArmAI.StorageResizer shim.
      {:fwup_data_resize, path: "../fwup_data_resize"},
      # HF tokenizers — used by ArmAI.LlamaCandle for string prompts.
      # Itself NIF-wrapped, no Nx dep. Optional from arm_ai's POV;
      # if absent, only the token-id-list prompt API works.
      {:tokenizers, "~> 0.5", optional: true},
      # Optional Nx for the *Backend modules. When the host app pulls
      # nx_primitives, llm, vision, or audio (which require Nx), this
      # gets satisfied; arm_ai itself doesn't need it.
      {:nx, "~> 0.9", optional: true},
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"}
    ]
  end
end

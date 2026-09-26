defmodule ArmAI.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :arm_ai,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "ArmAI",
      description:
        "Edge AI inference NIF for ARM CPUs — quantized Llama and Whisper via candle, ONNX via tract, with hand-tuned NEON kernels",
      docs: [main: "readme", extras: ["README.md", "CHANGELOG.md", "docs/perf_llm.md"]],
      package: package()
    ]
  end

  defp elixirc_paths(:test), do: ["lib"]
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
      links: %{"GitHub" => "https://github.com/mlainez/arm_ai"}
    ]
  end

  defp deps do
    [
      # CPU governor scoping + topology — used by ArmAI.Performance
      # at inference time, and by ArmAI.Application for scheduler
      # affinity on big.LITTLE. Generic Linux helper.
      {:cpu_governor, github: "mlainez/cpu_governor"},
      # HF tokenizers — used by ArmAI.LlamaCandle for string prompts.
      {:tokenizers, "~> 0.5"},
      # Optional Nx for the *Backend modules. When the host app pulls
      # nx_primitives, llm, vision, or audio (which require Nx), this
      # gets satisfied; arm_ai itself doesn't need it.
      {:nx, "~> 0.12.0", optional: true},
      # No precompiled release is published yet, so the NIF is always
      # built from source and rustler is a hard requirement.
      {:rustler, "~> 0.37"},
      {:rustler_precompiled, "~> 0.8"}
    ]
  end
end

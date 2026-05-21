defmodule NxArm.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_arm,
      version: "0.1.0",
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxArm",
      description: "Nx backend for ARM CPUs via NEON intrinsics",
      docs: docs(),
      package: package()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger],
      mod: {NxArm.Application, []}
    ]
  end

  def docs do
    [
      main: "readme",
      extras: ["README.md", "docs/deployment.md"]
    ]
  end

  def package do
    [
      name: :nx_arm,
      licenses: ["Apache-2.0"],
      files: ~w(lib native/nx_arm_nif/src native/nx_arm_nif/Cargo.toml
                native/nx_arm_nif/Cargo.lock README.md docs mix.exs
                checksum-Elixir.NxArm.Native.exs),
      links: %{"GitHub" => "https://github.com/marclainez/nx_arm"}
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      # Upstream tokenizer + safetensors (replacing nx_arm's own NIF
      # wrappers, which were duplicating these crates).
      {:tokenizers, "~> 0.5"},
      {:safetensors, "~> 0.1"},
      # CPU governor scoping + big.LITTLE topology — extracted from
      # nx_arm into its own package since it isn't Nx-specific.
      {:nerves_cpu, path: "../nerves_cpu"},
      {:axon, "~> 0.7", only: [:test]},
      {:bumblebee, "~> 0.6", only: [:test]},
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:stream_data, "~> 1.1", only: [:test]}
    ]
  end
end

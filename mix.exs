defmodule NxArm.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_arm,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxArm",
      description: "Nx backend for ARM CPUs via NEON intrinsics",
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def docs do
    [
      main: "readme",
      extras: ["README.md"]
    ]
  end

  def package do
    [
      name: :nx_arm,
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/TODO/nx_arm"}
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7", only: [:test]},
      {:rustler, "~> 0.36"}
    ]
  end
end

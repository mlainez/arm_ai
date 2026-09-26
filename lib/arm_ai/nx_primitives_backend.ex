defmodule ArmAI.NxPrimitivesBackend do
  @moduledoc """
  `arm_ai`'s implementation of the `NxPrimitives.Backend`
  behaviour: FFT via `rustfft` (requires the `fft` Cargo feature).

  Activate it in your app config:

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
  """

  # Implements `NxPrimitives.Backend`. The behaviour isn't declared because that
  # package depends on arm_ai, so it isn't loaded when this compiles;
  # nerves_ai's test suite checks every callback is present.

  def fft(input), do: input |> Nx.to_binary() |> ArmAI.Native.fft_complex_op() |> Nx.from_binary(:f32)

  def ifft(input), do: input |> Nx.to_binary() |> ArmAI.Native.ifft_complex_op() |> Nx.from_binary(:f32)

  def rfft(input), do: input |> Nx.to_binary() |> ArmAI.Native.rfft_op() |> Nx.from_binary(:f32)
end

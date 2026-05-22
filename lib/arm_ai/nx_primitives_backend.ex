defmodule ArmAI.NxPrimitivesBackend do
  @moduledoc """
  `arm_ai`'s implementation of the `NxPrimitives.Backend`
  behaviour — the NEON-tuned compute path via `rustfft` and the
  hand-tuned NIF kernels.

  Activate it in your app config:

      config :nx_primitives, backend: ArmAI.NxPrimitivesBackend
  """

  if Code.ensure_loaded?(NxPrimitives.Backend) do
    @behaviour NxPrimitives.Backend
  end

  @impl true
  def fft(input) do
    if not function_exported?(ArmAI.Native, :fft_complex_op, 1) do
      raise "arm_ai built without the `fft` Cargo feature"
    end

    bin = Nx.to_binary(input)
    out_bin = ArmAI.Native.fft_complex_op(bin)
    Nx.from_binary(out_bin, :f32)
  end

  @impl true
  def ifft(input) do
    if not function_exported?(ArmAI.Native, :ifft_complex_op, 1) do
      raise "arm_ai built without the `fft` Cargo feature"
    end

    bin = Nx.to_binary(input)
    out_bin = ArmAI.Native.ifft_complex_op(bin)
    Nx.from_binary(out_bin, :f32)
  end

  @impl true
  def rfft(input) do
    if not function_exported?(ArmAI.Native, :rfft_op, 1) do
      raise "arm_ai built without the `fft` Cargo feature"
    end

    bin = Nx.to_binary(input)
    out_bin = ArmAI.Native.rfft_op(bin)
    Nx.from_binary(out_bin, :f32)
  end

  @impl true
  def quantized_matmul(_act, _w_int8, _scales, _bias) do
    # The current quantized matmul kernel is exposed via the
    # `NxPrimitives.Quantized.matmul/2` struct-based API; this
    # callback is the future-facing slot for backend abstraction.
    raise "Use NxPrimitives.Quantized.matmul/2 instead — backend abstraction TODO."
  end

  @impl true
  def quantized_conv2d(_opts) do
    raise "Use NxPrimitives.QuantizedConv.apply/3 instead — backend abstraction TODO."
  end
end

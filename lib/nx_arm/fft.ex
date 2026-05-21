defmodule NxArm.FFT do
  @moduledoc """
  Fast Fourier Transform via the upstream `rustfft` crate
  (NEON-tuned on aarch64). Used by audio analysis (STFT,
  spectrograms, MFCC), some classical signal-processing models, and
  anything that needs frequency-domain features on the edge.

  All inputs/outputs are Nx tensors on `NxArm.Backend`.

      # 1024-point complex FFT (input: shape {2N} interleaved re,im,...)
      x = Nx.iota({2048}) |> NxArm.Backend.copy()
      y = NxArm.FFT.fft(x)              # complex → complex

      # Real-input FFT (input: shape {N} real, output: shape {2*(N/2+1)})
      y_real = NxArm.FFT.rfft(x)

  Requires the `fft` Cargo feature (in `full` by default).
  """

  @doc """
  Complex → complex forward FFT. Input is interleaved
  `[re0, im0, re1, im1, ...]` of length `2*N`. Output has the same
  length.
  """
  @spec fft(Nx.Tensor.t()) :: Nx.Tensor.t()
  def fft(input) do
    if not function_exported?(NxArm.Native, :fft_complex_op, 1) do
      raise "NxArm built without `fft` feature"
    end

    bin = NxArm.Backend.__bin_of__(input)
    out_bin = NxArm.Native.fft_complex_op(bin)
    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Complex → complex inverse FFT. Same layout as `fft/1`. Output is
  scaled by `1/N` so a forward+inverse round-trip is the identity.
  """
  @spec ifft(Nx.Tensor.t()) :: Nx.Tensor.t()
  def ifft(input) do
    if not function_exported?(NxArm.Native, :ifft_complex_op, 1) do
      raise "NxArm built without `fft` feature"
    end

    bin = NxArm.Backend.__bin_of__(input)
    out_bin = NxArm.Native.ifft_complex_op(bin)
    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Real-input FFT. Input is a length-`N` real signal; output is the
  non-redundant half as interleaved `[re, im, re, im, ...]` of
  length `2*(N/2 + 1)`.
  """
  @spec rfft(Nx.Tensor.t()) :: Nx.Tensor.t()
  def rfft(input) do
    if not function_exported?(NxArm.Native, :rfft_op, 1) do
      raise "NxArm built without `fft` feature"
    end

    bin = NxArm.Backend.__bin_of__(input)
    out_bin = NxArm.Native.rfft_op(bin)
    Nx.from_binary(out_bin, :f32) |> Nx.backend_copy(NxArm.Backend)
  end
end

defmodule NxArm.FFTTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  test "fft → ifft round-trips to within f32 noise" do
    # Interleaved re,im for an 8-point complex signal.
    signal = Nx.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0])
    spectrum = NxArm.FFT.fft(arm(signal))
    recovered = NxArm.FFT.ifft(spectrum) |> Nx.backend_copy(Nx.BinaryBackend)
    diff = Nx.subtract(recovered, signal) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4
  end

  test "rfft length is 2*(N/2 + 1)" do
    signal = Nx.iota({1024}, type: :f32) |> arm()
    spectrum = NxArm.FFT.rfft(signal)
    # 1024 → 513 complex bins → 1026 interleaved floats.
    assert Nx.size(spectrum) == 1026
  end
end

defmodule NxArm.Int4Test do
  use ExUnit.Case, async: true

  defp matmul_via_dequant(a, packed, scales, m, n, k) do
    # Reference: dequantize packed weights then run f32 matmul.
    w_f32 = dequantize_q4_0(packed, scales, n, k)
    w_tensor = Nx.from_binary(w_f32, :f32) |> Nx.reshape({n, k})
    a_tensor = Nx.from_binary(a, :f32) |> Nx.reshape({m, k})

    Nx.dot(a_tensor, [1], w_tensor, [1])
  end

  defp dequantize_q4_0(packed, scales, n, k) do
    n_groups = div(k, 32)
    bytes_per_row = div(k, 2)

    for ni <- 0..(n - 1), into: <<>> do
      for g <- 0..(n_groups - 1), into: <<>> do
        scale_offset = (ni * n_groups + g) * 4
        <<scale::float-32-little>> = binary_part(scales, scale_offset, 4)

        row_offset = ni * bytes_per_row + g * 16

        for j <- 0..15, into: <<>> do
          byte = :binary.at(packed, row_offset + j)
          lo = Bitwise.band(byte, 0x0F) - 8
          hi = Bitwise.band(Bitwise.bsr(byte, 4), 0x0F) - 8
          <<lo * scale::float-32-little, hi * scale::float-32-little>>
        end
      end
    end
  end

  test "round-trip quantize → dequant within 1 / 7 · max_abs per group" do
    n = 4
    k = 64
    # Use sinusoidal weights so each group has a distinct range.
    w_tensor =
      Nx.iota({n, k}, type: :f32)
      |> Nx.divide(50)
      |> Nx.sin()

    w_bin = Nx.to_binary(w_tensor)
    {packed, scales} = NxArm.Native.quantize_int4_q4_0_op(w_bin, n, k)
    assert byte_size(packed) == n * div(k, 2)
    assert byte_size(scales) == n * div(k, 32) * 4

    deq_bin = dequantize_q4_0(packed, scales, n, k)
    deq = Nx.from_binary(deq_bin, :f32) |> Nx.reshape({n, k})

    # Max error per group is scale/2 because of round-to-nearest;
    # group max_abs / 7 is the scale, so error ≤ max_abs/14. We
    # check the overall max-abs error is bounded by the largest
    # group scale / 2.
    max_input = Nx.reduce_max(Nx.abs(w_tensor)) |> Nx.to_number()
    diff = Nx.subtract(deq, w_tensor) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff <= max_input / 14 + 1.0e-6, "diff = #{diff}, bound = #{max_input / 14}"
  end

  test "int4 matmul agrees with f32 matmul on dequantized weights" do
    m = 3
    n = 5
    k = 64

    a_tensor =
      Nx.iota({m, k}, type: :f32) |> Nx.divide(20) |> Nx.sin()

    w_tensor =
      Nx.iota({n, k}, type: :f32) |> Nx.divide(30) |> Nx.cos()

    a_bin = Nx.to_binary(a_tensor)
    w_bin = Nx.to_binary(w_tensor)

    {packed, scales} = NxArm.Native.quantize_int4_q4_0_op(w_bin, n, k)
    out_bin = NxArm.Native.int4_matmul_f32_op(a_bin, packed, scales, m, n, k)
    got = Nx.from_binary(out_bin, :f32) |> Nx.reshape({m, n})

    ref = matmul_via_dequant(a_bin, packed, scales, m, n, k)

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "LLM-shape vector × weight matrix (M=1)" do
    # Decoder one-token forward: hidden vector × output projection.
    m = 1
    n = 32
    k = 128

    a_tensor =
      Nx.iota({m, k}, type: :f32) |> Nx.divide(50) |> Nx.sin()

    w_tensor =
      Nx.iota({n, k}, type: :f32) |> Nx.divide(80) |> Nx.cos()

    a_bin = Nx.to_binary(a_tensor)
    w_bin = Nx.to_binary(w_tensor)

    {packed, scales} = NxArm.Native.quantize_int4_q4_0_op(w_bin, n, k)
    out_bin = NxArm.Native.int4_matmul_f32_op(a_bin, packed, scales, m, n, k)
    got = Nx.from_binary(out_bin, :f32) |> Nx.reshape({m, n})

    ref = matmul_via_dequant(a_bin, packed, scales, m, n, k)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "memory footprint is ~8× smaller than f32" do
    n = 8
    k = 64

    w_tensor =
      Nx.iota({n, k}, type: :f32) |> Nx.divide(100) |> Nx.sin()

    {packed, scales} = NxArm.Native.quantize_int4_q4_0_op(Nx.to_binary(w_tensor), n, k)

    f32_bytes = n * k * 4
    int4_bytes = byte_size(packed) + byte_size(scales)
    # Packed: N*K/2 = 256, scales: N*K/32*4 = 64; total 320.
    # f32: N*K*4 = 2048. Ratio 6.4×.
    assert int4_bytes * 6 <= f32_bytes
    assert int4_bytes < f32_bytes / 6
  end
end

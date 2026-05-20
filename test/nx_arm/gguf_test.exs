defmodule NxArm.GGUFTest do
  use ExUnit.Case, async: true

  # Helpers to assemble a synthetic GGUF blob in tests. Matches the
  # llama.cpp v3 spec.

  defp str(s), do: <<byte_size(s)::little-64, s::binary>>

  defp kv_u32(k, v), do: str(k) <> <<4::little-32, v::little-32>>
  defp kv_f32(k, v), do: str(k) <> <<6::little-32, v::float-32-little>>
  defp kv_str(k, v), do: str(k) <> <<8::little-32, str(v)::binary>>

  defp tensor_info(name, shape, ttype, offset) do
    dim_bytes = for d <- shape, into: <<>>, do: <<d::little-64>>
    str(name) <> <<length(shape)::little-32, dim_bytes::binary, ttype::little-32, offset::little-64>>
  end

  defp pad_to(bin, align) do
    pad = rem(align - rem(byte_size(bin), align), align)
    bin <> :binary.copy(<<0>>, pad)
  end

  test "parses header + metadata + tensor catalogue" do
    metadata =
      kv_str("general.architecture", "test") <>
        kv_u32("general.alignment", 32) <>
        kv_f32("test.scalar", 1.5)

    # Two tensors: 8x4 f32 at offset 0, 4x8 f32 at offset 128.
    n_tensors = 2

    tensors =
      tensor_info("weight.a", [4, 8], 0, 0) <>
        tensor_info("weight.b", [8, 4], 0, 128)

    header =
      <<"GGUF", 3::little-32, n_tensors::little-64, 3::little-64>> <>
        metadata <>
        tensors

    aligned_header = pad_to(header, 32)

    # 32 floats + 32 floats = 256 bytes of data.
    data_a = for i <- 0..31, into: <<>>, do: <<:math.sin(i / 10.0)::float-32-little>>
    data_b = for i <- 0..31, into: <<>>, do: <<:math.cos(i / 10.0)::float-32-little>>

    file = aligned_header <> data_a <> data_b

    {:ok, gguf} = NxArm.GGUF.parse(file)

    assert gguf.version == 3
    assert gguf.metadata["general.architecture"] == "test"
    assert gguf.metadata["general.alignment"] == 32
    assert_in_delta gguf.metadata["test.scalar"], 1.5, 1.0e-6

    assert Map.keys(gguf.tensors) |> Enum.sort() == ["weight.a", "weight.b"]
    info_a = gguf.tensors["weight.a"]
    assert info_a.shape == [4, 8]
    assert info_a.dtype == :f32
    assert info_a.byte_size == 4 * 8 * 4

    {:ok, bytes_a} = NxArm.GGUF.tensor_bytes(gguf, "weight.a")
    assert byte_size(bytes_a) == 128
    assert bytes_a == data_a
  end

  test "Q4_0 unpack yields ints4-matmul-compatible {packed, scales}" do
    # Build a 2x32 Q4_0 weight matrix manually. Each row is one
    # block of 32 weights with one f16 scale.
    # We'll set scale=0.25 on both rows, and pick nibbles so that
    # decoded values are -2.0, -1.75, ..., 5.25 (32 distinct values
    # across the [-8, 7] · 0.25 range).
    f16_scale_025 = f32_to_f16(0.25)

    # Generate qs bytes: 16 bytes per block, each byte = (hi << 4) | lo.
    qs_row =
      for i <- 0..15, into: <<>> do
        lo = rem(i * 2, 16)
        hi = rem(i * 2 + 1, 16)
        <<Bitwise.bor(Bitwise.bsl(hi, 4), lo)::8>>
      end

    block = <<f16_scale_025::little-16, qs_row::binary>>
    blocks = block <> block

    # Header + tensor info for a single 2x32 Q4_0 tensor.
    tinfo = tensor_info("w", [32, 2], 2, 0)

    header =
      <<"GGUF", 3::little-32, 1::little-64, 1::little-64>> <>
        kv_u32("general.alignment", 32) <>
        tinfo

    file = pad_to(header, 32) <> blocks

    {:ok, gguf} = NxArm.GGUF.parse(file)

    info = gguf.tensors["w"]
    assert info.dtype == :q4_0
    assert info.shape == [32, 2]
    assert info.byte_size == 2 * 18

    {:ok, %{packed: packed, scales: scales, n: n, k: k}} = NxArm.GGUF.q4_0_unpack(gguf, "w")
    assert n == 2
    assert k == 32
    assert byte_size(packed) == n * div(k, 2)
    assert byte_size(scales) == n * div(k, 32) * 4

    # Scales should round-trip back to 0.25 f32.
    <<s0::float-32-little, s1::float-32-little>> = scales
    assert_in_delta s0, 0.25, 1.0e-3
    assert_in_delta s1, 0.25, 1.0e-3

    # Feeding these into int4_matmul_f32_op should produce the same
    # result as f32 matmul against the manually-decoded weights.
    a = Nx.iota({1, k}, type: :f32) |> Nx.divide(20)
    a_bin = Nx.to_binary(a)

    out_bin = NxArm.Native.int4_matmul_f32_op(a_bin, packed, scales, 1, n, k)
    got = Nx.from_binary(out_bin, :f32) |> Nx.reshape({1, n})

    # Decode the weight rows in Elixir for the reference.
    decoded_row =
      for i <- 0..15, into: [] do
        lo = rem(i * 2, 16) - 8
        hi = rem(i * 2 + 1, 16) - 8
        [lo * 0.25, hi * 0.25]
      end
      |> List.flatten()

    w_ref = Nx.tensor([decoded_row, decoded_row], type: :f32)
    ref = Nx.dot(a, [1], w_ref, [1])

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-3, "diff = #{diff}"
  end

  test "rejects non-GGUF input" do
    assert {:error, :not_a_gguf_file} = NxArm.GGUF.parse(<<"PNG\n", 0::little-32>>)
  end

  # Bit-level f32 → f16 for the test fixture.
  defp f32_to_f16(x) do
    <<bits::little-32>> = <<x::float-32-little>>
    sign = Bitwise.band(Bitwise.bsr(bits, 31), 0x1)
    exp = Bitwise.band(Bitwise.bsr(bits, 23), 0xFF)
    mant = Bitwise.band(bits, 0x7FFFFF)

    cond do
      exp == 0 and mant == 0 ->
        Bitwise.bsl(sign, 15)

      exp == 255 ->
        Bitwise.bor(Bitwise.bsl(sign, 15), 0x7C00) |> Bitwise.bor(if mant != 0, do: 1, else: 0)

      true ->
        new_exp = exp - 127 + 15

        cond do
          new_exp >= 31 ->
            Bitwise.bor(Bitwise.bsl(sign, 15), 0x7C00)

          new_exp <= 0 ->
            Bitwise.bsl(sign, 15)

          true ->
            new_mant = Bitwise.bsr(mant, 13)
            Bitwise.bsl(sign, 15)
            |> Bitwise.bor(Bitwise.bsl(new_exp, 10))
            |> Bitwise.bor(new_mant)
        end
    end
  end
end

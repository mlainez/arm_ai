defmodule NxArm.MmapTest do
  use ExUnit.Case, async: true

  @tmp_dir System.tmp_dir!()

  defp write_tmp(name, bytes) do
    path = Path.join(@tmp_dir, "nx_arm_mmap_test_#{name}_#{System.unique_integer([:positive])}")
    File.write!(path, bytes)
    path
  end

  test "mmap_open returns handle + file size" do
    payload = :crypto.strong_rand_bytes(8192)
    path = write_tmp("basic", payload)

    {handle, len} = NxArm.Native.mmap_open_op(path)
    assert is_reference(handle)
    assert len == 8192

    File.rm!(path)
  end

  test "mmap_slice reads correct bytes" do
    payload = for i <- 0..1023, into: <<>>, do: <<rem(i, 256)::8>>
    path = write_tmp("slice", payload)

    {handle, 1024} = NxArm.Native.mmap_open_op(path)

    assert NxArm.Native.mmap_slice_op(handle, 0, 16) == binary_part(payload, 0, 16)
    assert NxArm.Native.mmap_slice_op(handle, 256, 64) == binary_part(payload, 256, 64)
    assert NxArm.Native.mmap_slice_op(handle, 1020, 4) == binary_part(payload, 1020, 4)

    File.rm!(path)
  end

  test "mmap_slice rejects out-of-bounds reads" do
    path = write_tmp("oob", :crypto.strong_rand_bytes(64))
    {handle, 64} = NxArm.Native.mmap_open_op(path)

    assert match?({:error, _}, NxArm.Native.mmap_slice_op(handle, 60, 8))

    File.rm!(path)
  end

  test "mmap_open errors on missing file" do
    assert match?({:error, _}, NxArm.Native.mmap_open_op("/nonexistent/path/should/not/exist"))
  end

  test "GGUF.read_mmap loads tensor bytes lazily" do
    # Synthesize a tiny GGUF: 1 tensor of 32 f32 values.
    payload = for i <- 0..31, into: <<>>, do: <<i / 10.0::float-32-little>>

    str = fn s -> <<byte_size(s)::little-64, s::binary>> end
    kv_u32 = fn k, v -> str.(k) <> <<4::little-32, v::little-32>> end

    tinfo =
      str.("w") <> <<1::little-32, 32::little-64, 0::little-32, 0::little-64>>

    header =
      <<"GGUF", 3::little-32, 1::little-64, 1::little-64>> <>
        kv_u32.("general.alignment", 32) <>
        tinfo

    pad = rem(32 - rem(byte_size(header), 32), 32)
    file = header <> :binary.copy(<<0>>, pad) <> payload

    path = write_tmp("gguf", file)

    {:ok, gguf} = NxArm.GGUF.read_mmap(path)
    assert gguf.mmap_handle != nil
    assert gguf.file_contents == nil
    info = gguf.tensors["w"]
    assert info.dtype == :f32
    assert info.shape == [32]

    {:ok, bytes} = NxArm.GGUF.tensor_bytes(gguf, "w")
    assert bytes == payload

    File.rm!(path)
  end
end

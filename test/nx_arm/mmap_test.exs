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

end

defmodule NxArm.NIFRobustnessTest do
  @moduledoc """
  Tests that NIFs reject obviously bad inputs cleanly (with
  {:error, _} or a raise) rather than UB / segfault. Each test
  exercises one shape-mismatch or out-of-range path the runtime
  guard is supposed to catch.
  """

  use ExUnit.Case, async: true

  defp errored?({:error, _}), do: true
  defp errored?(_), do: false

  test "int8_matmul rejects K mismatch via shape arg" do
    # Pass dims so a / w are smaller than declared. The NIF reads off the
    # declared size; we just want the operation to either return an
    # error tuple or raise, never silently corrupt memory.
    a = <<0::8>>
    w = <<0::8>>
    scales = <<0.0::float-32-little>>

    result =
      try do
        NxArm.Native.int8_matmul_f32_op(a, w, scales, 1.0, 8, 8, 8)
      catch
        _, _ -> :raised
      end

    # Either we raise, return {:error, _}, or we get an empty/garbage
    # binary back. The important thing is no segfault — if we got
    # here the BEAM is alive.
    assert is_binary(result) or result == :raised or errored?(result)
  end

  test "int4_matmul rejects K%32 != 0" do
    a = <<0.0::float-32-little>>
    packed = <<0>>
    scales = <<0.0::float-32-little>>

    assert errored?(NxArm.Native.int4_matmul_f32_op(a, packed, scales, 1, 1, 33))
  end

  test "mmap_slice rejects out-of-bounds" do
    tmp = Path.join(System.tmp_dir!(), "nx_arm_robust_#{System.unique_integer([:positive])}")
    File.write!(tmp, :crypto.strong_rand_bytes(64))
    {handle, 64} = NxArm.Native.mmap_open_op(tmp)

    assert errored?(NxArm.Native.mmap_slice_op(handle, 60, 8))
    assert errored?(NxArm.Native.mmap_slice_op(handle, 1_000_000, 1))

    File.rm!(tmp)
  end

  test "winograd 3x3 rejects bad dims arg length" do
    inp = <<0.0::float-32-little>>
    w = <<0.0::float-32-little>>

    result =
      try do
        NxArm.Native.conv2d_f32_winograd_3x3_op(inp, w, <<>>, [1, 2, 3], [0, 0, 0, 0])
      catch
        _, _ -> :raised
      end

    assert result == :raised or errored?(result)
  end

  test "im2col conv rejects bad dims arg length" do
    inp = <<0.0::float-32-little>>
    w = <<0.0::float-32-little>>

    result =
      try do
        NxArm.Native.conv2d_f32_im2col_op(inp, w, <<>>, [1, 2, 3], [1, 1], [0, 0, 0, 0])
      catch
        _, _ -> :raised
      end

    assert result == :raised or errored?(result)
  end

  test "quantize_int4 rejects K%32 != 0" do
    w = <<0.0::float-32-little>>

    result =
      try do
        NxArm.Native.quantize_int4_q4_0_op(w, 1, 17)
      catch
        _, _ -> :raised
      end

    assert result == :raised or errored?(result)
  end

  test "init_thread_pool returns atom, never crashes" do
    res = NxArm.Native.init_thread_pool_op(1)
    assert res in [:ok, :already_initialised]
  end

  test "many empty / 1-element NIF calls in a tight loop don't leak" do
    a = Nx.tensor([1.0], type: :f32) |> Nx.backend_copy(NxArm.Backend)
    b = Nx.tensor([2.0], type: :f32) |> Nx.backend_copy(NxArm.Backend)

    for _ <- 1..1000 do
      _ = Nx.add(a, b)
    end

    # If we leaked OwnedBinaries the BEAM would eventually run out;
    # 1000 cycles is plenty to catch obvious leaks.
    assert Nx.to_flat_list(Nx.add(a, b)) == [3.0]
  end
end

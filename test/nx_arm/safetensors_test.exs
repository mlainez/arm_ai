defmodule NxArm.SafeTensorsTest do
  use ExUnit.Case, async: true

  @tmpdir System.tmp_dir!()

  # Build a SafeTensors-format file in memory for testing.
  defp build_safetensors(specs_and_data) do
    # specs_and_data :: [{name, dtype_str, shape :: list, raw_bytes}]
    {entries, _final_offset, data_bytes} =
      Enum.reduce(specs_and_data, {[], 0, <<>>}, fn {name, dtype, shape, raw},
                                                   {acc, offset, data_acc} ->
        len = byte_size(raw)
        spec = %{
          "dtype" => dtype,
          "shape" => shape,
          "data_offsets" => [offset, offset + len]
        }
        {[{name, spec} | acc], offset + len, data_acc <> raw}
      end)

    header_map = Map.new(entries) |> Map.put("__metadata__", %{"format" => "pt"})
    header_bytes = :json.encode(header_map) |> IO.iodata_to_binary()
    # Pad header to 8-byte alignment as SafeTensors recommends.
    pad = rem(8 - rem(byte_size(header_bytes), 8), 8)
    padded_header = header_bytes <> :binary.copy(<<32>>, pad)

    <<byte_size(padded_header)::little-64, padded_header::binary, data_bytes::binary>>
  end

  defp tmp_path(name), do: Path.join(@tmpdir, "nx_arm_st_test_#{:erlang.unique_integer([:positive])}_#{name}")

  test "loads a single f32 tensor" do
    raw = for v <- [1.0, 2.0, 3.0, 4.0], into: <<>>, do: <<v::float-little-32>>
    bin = build_safetensors([{"weight", "F32", [2, 2], raw}])

    path = tmp_path("single.safetensors")
    File.write!(path, bin)

    on_exit(fn -> File.rm(path) end)

    {:ok, tensors} = NxArm.SafeTensors.load(path)

    assert Map.keys(tensors) == ["weight"]
    w = tensors["weight"]
    assert Nx.shape(w) == {2, 2}
    assert Nx.type(w) == {:f, 32}
    assert Nx.to_flat_list(w) == [1.0, 2.0, 3.0, 4.0]
  end

  test "loads multiple tensors of mixed dtypes" do
    f32_data = for v <- [1.0, 2.0, 3.0], into: <<>>, do: <<v::float-little-32>>
    s64_data = for v <- [10, 20, 30, 40], into: <<>>, do: <<v::little-signed-64>>

    bin =
      build_safetensors([
        {"a.weight", "F32", [3], f32_data},
        {"b.idx", "I64", [2, 2], s64_data}
      ])

    path = tmp_path("multi.safetensors")
    File.write!(path, bin)
    on_exit(fn -> File.rm(path) end)

    {:ok, tensors} = NxArm.SafeTensors.load(path)
    assert MapSet.new(Map.keys(tensors)) == MapSet.new(["a.weight", "b.idx"])
    assert Nx.type(tensors["a.weight"]) == {:f, 32}
    assert Nx.type(tensors["b.idx"]) == {:s, 64}
    assert Nx.to_flat_list(tensors["a.weight"]) == [1.0, 2.0, 3.0]
    assert Nx.to_flat_list(tensors["b.idx"]) == [10, 20, 30, 40]
  end

  test "info/1 returns metadata without loading data" do
    raw = for v <- [1.0, 2.0, 3.0, 4.0], into: <<>>, do: <<v::float-little-32>>
    bin = build_safetensors([{"weight", "F32", [2, 2], raw}])

    path = tmp_path("info.safetensors")
    File.write!(path, bin)
    on_exit(fn -> File.rm(path) end)

    {:ok, info} = NxArm.SafeTensors.info(path)
    assert info == %{"weight" => %{"dtype" => "F32", "shape" => [2, 2], "data_offsets" => [0, 16]}}
  end

  test "only: option loads selected tensors only" do
    f32 = for v <- [1.0, 2.0, 3.0], into: <<>>, do: <<v::float-little-32>>
    s64 = for v <- [10, 20, 30, 40], into: <<>>, do: <<v::little-signed-64>>

    bin = build_safetensors([
      {"a", "F32", [3], f32},
      {"b", "I64", [2, 2], s64}
    ])

    path = tmp_path("only.safetensors")
    File.write!(path, bin)
    on_exit(fn -> File.rm(path) end)

    {:ok, tensors} = NxArm.SafeTensors.load(path, only: ["a"])
    assert Map.keys(tensors) == ["a"]
  end

  test "missing header length returns an error" do
    assert {:error, _} = NxArm.SafeTensors.load(unique_path_or_create("nope"))
  end

  defp unique_path_or_create(content) do
    p = tmp_path("err.safetensors")
    File.write!(p, content)
    p
  end
end

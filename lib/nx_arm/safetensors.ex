defmodule NxArm.SafeTensors do
  @moduledoc """
  Pure-Elixir loader for the SafeTensors format (the standard
  Hugging Face file format for model weights).

  SafeTensors file layout:

    1. 8 bytes — little-endian u64, length of the JSON header.
    2. N bytes — UTF-8 JSON header describing each tensor (dtype,
       shape, byte offsets within the data block, plus an optional
       `__metadata__` map).
    3. Raw bytes — tensor data, packed back-to-back in the order
       declared by `data_offsets`.

  See: https://huggingface.co/docs/safetensors/index

  ## Example

      {:ok, weights} = NxArm.SafeTensors.load("/data/model.safetensors")
      %{
        "model.embed_tokens.weight" => emb_tensor,
        "model.norm.weight" => norm_tensor,
        ...
      } = weights

  By default each tensor is built on `NxArm.Backend`. Pass
  `backend:` to override.
  """

  @doc """
  Load a SafeTensors file into a map of `name -> Nx.Tensor`.

  ## Options

    * `:backend` — Nx backend module to materialise tensors on.
      Default: `NxArm.Backend`.
    * `:only` — list of tensor names to load (the rest are skipped,
      useful for partial loads).
  """
  def load(path, opts \\ []) do
    backend = Keyword.get(opts, :backend, NxArm.Backend)
    only = Keyword.get(opts, :only)

    with {:ok, data} <- File.read(path),
         {:ok, header_bytes, data_bytes} <- split_header(data),
         {:ok, header} <- decode_json(header_bytes),
         {:ok, tensors} <- build_tensors(header, data_bytes, backend, only) do
      {:ok, tensors}
    end
  end

  @doc "Same as load/2 but raises on error."
  def load!(path, opts \\ []) do
    case load(path, opts) do
      {:ok, tensors} -> tensors
      {:error, reason} -> raise "SafeTensors load failed: #{inspect(reason)}"
    end
  end

  @doc """
  Read just the header (tensor metadata) without loading any data.
  Useful for inspecting a file's tensor names + shapes + dtypes.
  """
  def info(path) do
    with {:ok, data} <- File.read(path),
         {:ok, header_bytes, _data_bytes} <- split_header(data),
         {:ok, header} <- decode_json(header_bytes) do
      {:ok, header |> Map.delete("__metadata__")}
    end
  end

  # ── internals ─────────────────────────────────────────

  defp split_header(<<header_len::little-64, rest::binary>>) do
    if byte_size(rest) < header_len do
      {:error, :file_too_short}
    else
      <<header_bytes::binary-size(header_len), data_bytes::binary>> = rest
      {:ok, header_bytes, data_bytes}
    end
  end

  defp split_header(_), do: {:error, :missing_header_length}

  defp decode_json(bytes) do
    case :json.decode(bytes) do
      decoded when is_map(decoded) -> {:ok, decoded}
    end
  rescue
    e -> {:error, {:json_decode_failed, Exception.message(e)}}
  end

  defp build_tensors(header, data_bytes, backend, only) do
    header
    |> Enum.reject(fn {k, _v} -> k == "__metadata__" end)
    |> Enum.filter(fn {name, _spec} ->
      only == nil or name in only
    end)
    |> Enum.reduce_while({:ok, %{}}, fn {name, spec}, {:ok, acc} ->
      case build_tensor(name, spec, data_bytes, backend) do
        {:ok, t} -> {:cont, {:ok, Map.put(acc, name, t)}}
        {:error, _} = err -> {:halt, err}
      end
    end)
  end

  defp build_tensor(name, spec, data_bytes, backend) do
    with {:ok, dtype} <- decode_dtype(spec["dtype"]),
         shape when is_list(shape) <- spec["shape"],
         [start, stop] <- spec["data_offsets"] do
      slice_len = stop - start

      if start + slice_len > byte_size(data_bytes) do
        {:error, {:tensor_out_of_range, name}}
      else
        slice = binary_part(data_bytes, start, slice_len)
        shape_tuple = List.to_tuple(shape)
        tensor = Nx.from_binary(slice, dtype, backend: backend) |> Nx.reshape(shape_tuple)
        {:ok, tensor}
      end
    else
      err -> {:error, {:bad_tensor_spec, name, err}}
    end
  end

  defp decode_dtype("F64"), do: {:ok, {:f, 64}}
  defp decode_dtype("F32"), do: {:ok, {:f, 32}}
  defp decode_dtype("F16"), do: {:ok, {:f, 16}}
  defp decode_dtype("BF16"), do: {:ok, {:bf, 16}}
  defp decode_dtype("I64"), do: {:ok, {:s, 64}}
  defp decode_dtype("I32"), do: {:ok, {:s, 32}}
  defp decode_dtype("I16"), do: {:ok, {:s, 16}}
  defp decode_dtype("I8"), do: {:ok, {:s, 8}}
  defp decode_dtype("U64"), do: {:ok, {:u, 64}}
  defp decode_dtype("U32"), do: {:ok, {:u, 32}}
  defp decode_dtype("U16"), do: {:ok, {:u, 16}}
  defp decode_dtype("U8"), do: {:ok, {:u, 8}}
  defp decode_dtype("BOOL"), do: {:ok, {:u, 8}}
  defp decode_dtype(other), do: {:error, {:unknown_dtype, other}}
end

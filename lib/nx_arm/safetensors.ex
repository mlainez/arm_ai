defmodule NxArm.SafeTensors do
  @moduledoc """
  Read HuggingFace SafeTensors files via the upstream `safetensors`
  crate.

  Returns a map `%{name => Nx.Tensor}` with all tensors loaded on
  `NxArm.Backend` as f32. Original dtype (f16/bf16/f64/i32/i64/u8)
  is captured in metadata; callers can re-cast via `Nx.as_type/2` if
  they need to.

      {:ok, tensors} = NxArm.SafeTensors.load("/root/model.safetensors")
      Map.keys(tensors) |> Enum.take(5)
      #=> ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", ...]

      tensors["model.embed_tokens.weight"] |> Nx.shape()
      #=> {32000, 4096}

  Requires the `safetensors` Cargo feature (in `full` by default).
  """

  @doc """
  Load every tensor in a SafeTensors file. Returns
  `{:ok, %{name => Nx.Tensor}}`.

  All tensors are loaded as f32 on `NxArm.Backend`. Use the
  `metadata/1` companion when you need to know the on-disk dtype.
  """
  @spec load(Path.t()) :: {:ok, %{String.t() => Nx.Tensor.t()}} | {:error, term()}
  def load(path) do
    if not function_exported?(NxArm.Native, :safetensors_load_op, 1) do
      {:error, :safetensors_feature_disabled}
    else
      try do
        triples = NxArm.Native.safetensors_load_op(path)

        tensors =
          for {name, shape, _dtype, bin} <- triples, into: %{} do
            t =
              bin
              |> Nx.from_binary(:f32)
              |> Nx.reshape(List.to_tuple(shape))
              |> Nx.backend_copy(NxArm.Backend)

            {name, t}
          end

        {:ok, tensors}
      rescue
        e -> {:error, e}
      catch
        :error, reason -> {:error, reason}
      end
    end
  end

  @doc """
  Lightweight metadata view (without loading tensor bytes into Nx).
  Returns `[%{name, shape, dtype}]` — useful for inspecting a file
  before deciding what to load.
  """
  @spec metadata(Path.t()) :: {:ok, [%{name: String.t(), shape: [non_neg_integer()], dtype: String.t()}]} | {:error, term()}
  def metadata(path) do
    if not function_exported?(NxArm.Native, :safetensors_load_op, 1) do
      {:error, :safetensors_feature_disabled}
    else
      try do
        triples = NxArm.Native.safetensors_load_op(path)
        list = Enum.map(triples, fn {name, shape, dtype, _bin} ->
          %{name: name, shape: shape, dtype: dtype}
        end)
        {:ok, list}
      rescue
        e -> {:error, e}
      catch
        :error, reason -> {:error, reason}
      end
    end
  end
end

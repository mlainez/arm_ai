defmodule ArmAI.VisionBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferVision.Backend` — tract-onnx ONNX
  runtime, plus `image` + `fast_image_resize` for preprocessing.

      config :infer_vision, backend: ArmAI.VisionBackend
  """

  # Implements `InferVision.Backend`. The behaviour isn't declared because that
  # package depends on arm_ai, so it isn't loaded when this compiles;
  # nerves_ai's test suite checks every callback is present.

  # tract-onnx runs models with f32 inputs only: the NIF builds f32
  # tensors and tract checks them against the model's declared input
  # types. Outputs are always returned as f32.

  def onnx_load(path, _opts) do
    case ArmAI.Native.onnx_load_op(path) do
      {resource, input_names, output_names} ->
        {:ok, %{resource: resource, input_names: input_names, output_names: output_names}}

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    e in ErlangError -> {:error, e.original}
    e -> {:error, e}
  end

  def onnx_run(%{resource: resource}, inputs, _opts) when is_map(inputs) do
    with {:ok, args} <- encode_onnx_inputs(inputs) do
      case ArmAI.Native.onnx_run_op(resource, args) do
        {:error, _} = err ->
          err

        outputs when is_list(outputs) ->
          Map.new(outputs, fn {name, shape, bin} ->
            {name, bin |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))}
          end)
      end
    end
  rescue
    e in ErlangError -> {:error, e.original}
  end

  def onnx_input_names(%{input_names: names}), do: names

  def onnx_output_names(%{output_names: names}), do: names

  defp encode_onnx_inputs(inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn {name, tensor}, {:ok, acc} ->
      case Nx.type(tensor) do
        {:f, _} ->
          t = Nx.as_type(tensor, :f32)
          {:cont, {:ok, [{name, Tuple.to_list(Nx.shape(t)), Nx.to_binary(t)} | acc]}}

        type ->
          {:halt, {:error, {:unsupported_input_type, name, type}}}
      end
    end)
  end

  def decode_to_rgb8(path) do
    case ArmAI.Native.vision_decode_to_rgb8_op(path) do
      {:error, _} = err -> err
      {bin, w, h} -> {:ok, {bin, w, h}}
    end
  rescue
    e in ErlangError -> {:error, e.original}
  end

  def load_for_classifier(path, opts) do
    {out_h, out_w} = Keyword.fetch!(opts, :size)
    mean = Keyword.get(opts, :mean, {0.485, 0.456, 0.406})
    std = Keyword.get(opts, :std, {0.229, 0.224, 0.225})
    layout = Keyword.get(opts, :layout, :nchw)

    bin =
      case ArmAI.Native.vision_load_for_classifier_op(path, out_h, out_w, mean, std, layout) do
        {:error, reason} -> raise ArgumentError, "cannot load image #{inspect(path)}: #{inspect(reason)}"
        bin -> bin
      end

    shape =
      case layout do
        :nchw -> {3, out_h, out_w}
        :nhwc -> {out_h, out_w, 3}
      end

    Nx.from_binary(bin, :f32) |> Nx.reshape(shape)
  end
end

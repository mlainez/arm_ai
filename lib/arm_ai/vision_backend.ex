defmodule ArmAI.VisionBackend do
  @moduledoc """
  `arm_ai`'s implementation of `InferVision.Backend` — tract-onnx ONNX
  runtime, plus `image` + `fast_image_resize` for preprocessing.

      config :infer_vision, backend: ArmAI.VisionBackend
  """

  if Code.ensure_loaded?(InferVision.Backend) do
    @behaviour InferVision.Backend
  end

  @impl true
  def onnx_load(path, _opts) do
    if not function_exported?(ArmAI.Native, :onnx_load_op, 1) do
      {:error, :onnx_feature_disabled}
    else
      try do
        case ArmAI.Native.onnx_load_op(path) do
          {:error, reason} -> {:error, reason}
          handle -> {:ok, handle}
        end
      rescue
        e -> {:error, e}
      end
    end
  end

  @impl true
  def onnx_run(handle, inputs, _opts) when is_map(inputs) do
    bins =
      Enum.map(inputs, fn {name, tensor} ->
        {name, Nx.to_binary(tensor), Tuple.to_list(Nx.shape(tensor))}
      end)

    case ArmAI.Native.onnx_run_op(handle, bins) do
      {:error, _} = err ->
        err

      outputs when is_list(outputs) ->
        outputs
        |> Enum.into(%{}, fn {name, bin, shape, _dtype} ->
          {name, Nx.from_binary(bin, :f32) |> Nx.reshape(List.to_tuple(shape))}
        end)
    end
  end

  @impl true
  def onnx_input_specs(handle), do: ArmAI.Native.onnx_input_specs_op(handle)

  @impl true
  def onnx_output_specs(handle), do: ArmAI.Native.onnx_output_specs_op(handle)

  @impl true
  def decode_to_rgb8(path) do
    if not function_exported?(ArmAI.Native, :vision_decode_to_rgb8_op, 1) do
      {:error, :vision_feature_disabled}
    else
      case ArmAI.Native.vision_decode_to_rgb8_op(path) do
        {:error, _} = err -> err
        {bin, w, h} -> {:ok, {bin, w, h}}
      end
    end
  end

  @impl true
  def load_for_classifier(path, opts) do
    {out_h, out_w} = Keyword.fetch!(opts, :size)
    mean = Keyword.get(opts, :mean, {0.485, 0.456, 0.406})
    std = Keyword.get(opts, :std, {0.229, 0.224, 0.225})
    layout = Keyword.get(opts, :layout, :nchw)

    bin =
      ArmAI.Native.vision_load_for_classifier_op(path, out_h, out_w, mean, std, layout)

    shape =
      case layout do
        :nchw -> {3, out_h, out_w}
        :nhwc -> {out_h, out_w, 3}
      end

    Nx.from_binary(bin, :f32) |> Nx.reshape(shape)
  end
end

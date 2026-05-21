defmodule NxArm.Models.YOLO do
  @moduledoc """
  YOLO object detection via the ONNX bridge + the existing
  `NxArm.Detection` NMS helpers.

  Works with any YOLO export that follows the v5/v8 convention of
  emitting a single `(1, n_boxes, 5 + n_classes)` tensor where each
  row is `[cx, cy, w, h, obj_conf, class_0, ..., class_K]`. v8/v9
  exports use `(1, 4 + n_classes, n_boxes)` — pass `:layout, :v8`
  to handle that variant.

      {:ok, model} = NxArm.Models.YOLO.load("/root/yolov8n.onnx")
      image = NxArm.Vision.load_for_classifier("/data/cat.jpg",
                size: {640, 640},
                mean: {0.0, 0.0, 0.0},
                std:  {1.0, 1.0, 1.0},
                layout: :nchw)
      detections = NxArm.Models.YOLO.detect(model, image,
                     iou_threshold: 0.45,
                     score_threshold: 0.25)
      # detections = [%{box: {x1, y1, x2, y2}, class: 16, score: 0.91}, ...]
  """

  defstruct [:onnx, :input_name, :input_shape, :layout]

  @doc "Load a YOLO ONNX file. `layout` is `:v5` (default) or `:v8`."
  @spec load(Path.t(), keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path, opts \\ []) do
    layout = Keyword.get(opts, :layout, :v5)
    input_shape = Keyword.get(opts, :input_shape, {640, 640})

    case NxArm.Models.Onnx.load(path) do
      {:ok, model} ->
        input_name = List.first(model.input_names) || "images"
        {:ok, %__MODULE__{onnx: model, input_name: input_name, input_shape: input_shape, layout: layout}}

      err ->
        err
    end
  end

  @doc """
  Run detection. `image` is a preprocessed `{1, 3, H, W}` tensor
  (use `NxArm.Vision.load_for_classifier/2` to build it).

  Returns a list of `%{box: {x1, y1, x2, y2}, class: integer, score: float}`,
  with box coordinates in the input image's pixel space (0..H/W).
  """
  @spec detect(%__MODULE__{}, Nx.Tensor.t(), keyword()) :: [
          %{box: {float(), float(), float(), float()}, class: non_neg_integer(), score: float()}
        ]
  def detect(%__MODULE__{onnx: onnx, input_name: in_name, layout: layout}, image, opts \\ []) do
    iou_threshold = Keyword.get(opts, :iou_threshold, 0.45)
    score_threshold = Keyword.get(opts, :score_threshold, 0.25)
    max_output = Keyword.get(opts, :max_output, 100)

    # Ensure batch dim.
    image =
      case Nx.shape(image) do
        {_n, _c, _h, _w} -> image
        {c, h, w} -> Nx.reshape(image, {1, c, h, w})
      end

    outputs = NxArm.Models.Onnx.run(onnx, %{in_name => image})
    {_, raw} = Enum.at(outputs, 0)

    boxes_scores = decode(raw, layout)

    case boxes_scores do
      {boxes, scores, classes} ->
        kept =
          NxArm.Detection.nms(boxes, scores,
            iou_threshold: iou_threshold,
            score_threshold: score_threshold,
            max_output: max_output
          )

        boxes_list = Nx.to_list(boxes)
        scores_list = Nx.to_flat_list(scores)
        classes_list = Nx.to_flat_list(classes)

        for idx <- kept do
          [x1, y1, x2, y2] = Enum.at(boxes_list, idx)
          %{
            box: {x1 * 1.0, y1 * 1.0, x2 * 1.0, y2 * 1.0},
            class: Enum.at(classes_list, idx),
            score: Enum.at(scores_list, idx)
          }
        end
    end
  end

  # YOLOv5: (1, n_boxes, 5 + n_classes) with [cx, cy, w, h, obj, cls...]
  defp decode(raw, :v5) do
    {1, n, channels} = Nx.shape(raw)
    flat = Nx.reshape(raw, {n, channels})

    boxes_xywh = Nx.slice(flat, [0, 0], [n, 4])
    obj = Nx.slice(flat, [0, 4], [n, 1]) |> Nx.reshape({n})
    class_probs = Nx.slice(flat, [0, 5], [n, channels - 5])

    classes = Nx.argmax(class_probs, axis: -1)
    max_class = Nx.reduce_max(class_probs, axes: [-1])

    scores = Nx.multiply(obj, max_class)
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    {boxes_xyxy, scores, classes}
  end

  # YOLOv8: (1, 4 + n_classes, n_boxes) — channels-first.
  defp decode(raw, :v8) do
    {1, channels, n} = Nx.shape(raw)
    flat = Nx.reshape(raw, {channels, n}) |> Nx.transpose(axes: [1, 0])

    boxes_xywh = Nx.slice(flat, [0, 0], [n, 4])
    class_probs = Nx.slice(flat, [0, 4], [n, channels - 4])

    classes = Nx.argmax(class_probs, axis: -1)
    scores = Nx.reduce_max(class_probs, axes: [-1])

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    {boxes_xyxy, scores, classes}
  end

  defp xywh_to_xyxy(boxes_xywh) do
    cx = Nx.slice(boxes_xywh, [0, 0], [Nx.axis_size(boxes_xywh, 0), 1])
    cy = Nx.slice(boxes_xywh, [0, 1], [Nx.axis_size(boxes_xywh, 0), 1])
    w = Nx.slice(boxes_xywh, [0, 2], [Nx.axis_size(boxes_xywh, 0), 1])
    h = Nx.slice(boxes_xywh, [0, 3], [Nx.axis_size(boxes_xywh, 0), 1])

    x1 = Nx.subtract(cx, Nx.divide(w, 2))
    y1 = Nx.subtract(cy, Nx.divide(h, 2))
    x2 = Nx.add(cx, Nx.divide(w, 2))
    y2 = Nx.add(cy, Nx.divide(h, 2))

    Nx.concatenate([x1, y1, x2, y2], axis: 1)
  end
end

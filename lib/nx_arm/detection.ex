defmodule NxArm.Detection do
  @moduledoc """
  Object-detection post-processing helpers: non-max suppression and
  anchor-box decoding. These are the building blocks every YOLO /
  SSD / RetinaNet variant uses after the convolutional head.

  Everything in here works on Nx tensors but does the actual loops
  in Elixir — the workloads are tiny (typically <1000 boxes) and
  putting them in NIFs is not the bottleneck.

  ## Example (YOLOv5-style)

      logits = model_forward(image)       # {N, 4 + 1 + n_classes}
      boxes_xyxy = NxArm.Detection.decode_xywh_to_xyxy(logits[..., 0..3])
      scores = logits[..., 4] * Nx.reduce_max(logits[..., 5..-1], axes: [-1])
      kept = NxArm.Detection.nms(boxes_xyxy, scores, iou_threshold: 0.45, score_threshold: 0.25)
  """

  @doc """
  Non-maximum suppression on `{N, 4}` boxes in `[x1, y1, x2, y2]` and
  `{N}` scores. Returns the indices of kept boxes, in score-descending
  order.

  ## Options

    * `:iou_threshold` — boxes with IoU above this against a kept
      box are suppressed. Default 0.5.
    * `:score_threshold` — boxes below this score are discarded
      before NMS runs. Default 0.0 (no pre-filter).
    * `:max_output` — cap on number of kept boxes. Default `nil`
      (no cap).
  """
  def nms(boxes, scores, opts \\ []) do
    iou_threshold = Keyword.get(opts, :iou_threshold, 0.5)
    score_threshold = Keyword.get(opts, :score_threshold, 0.0)
    max_output = Keyword.get(opts, :max_output)

    boxes_cpu = boxes |> Nx.backend_copy(Nx.BinaryBackend)
    scores_cpu = scores |> Nx.backend_copy(Nx.BinaryBackend)

    {n, 4} = Nx.shape(boxes_cpu)
    if Nx.shape(scores_cpu) != {n} do
      raise ArgumentError,
            "nms: boxes shape {N, 4} vs scores shape {N} mismatch: #{n} vs #{inspect(Nx.shape(scores_cpu))}"
    end

    score_list = Nx.to_flat_list(scores_cpu)
    box_list = Nx.to_flat_list(boxes_cpu) |> Enum.chunk_every(4)

    # Filter by score threshold while keeping original indices.
    survivors =
      score_list
      |> Enum.with_index()
      |> Enum.filter(fn {s, _i} -> s >= score_threshold end)
      |> Enum.sort_by(fn {s, _i} -> -s end)

    do_nms(survivors, box_list, iou_threshold, max_output, [])
  end

  defp do_nms([], _boxes, _thr, _max, acc), do: Enum.reverse(acc)

  defp do_nms(_survivors, _boxes, _thr, max, acc) when is_integer(max) and length(acc) >= max,
    do: Enum.reverse(acc)

  defp do_nms([{_s, head_idx} | rest], boxes, thr, max, acc) do
    head_box = Enum.at(boxes, head_idx)
    filtered =
      Enum.reject(rest, fn {_s, idx} ->
        iou(head_box, Enum.at(boxes, idx)) > thr
      end)

    do_nms(filtered, boxes, thr, max, [head_idx | acc])
  end

  @doc """
  Intersection-over-Union of two `[x1, y1, x2, y2]` boxes.
  """
  def iou([a_x1, a_y1, a_x2, a_y2], [b_x1, b_y1, b_x2, b_y2]) do
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a_x2 - a_x1) * max(0.0, a_y2 - a_y1)
    area_b = max(0.0, b_x2 - b_x1) * max(0.0, b_y2 - b_y1)
    union = area_a + area_b - inter

    if union <= 0.0, do: 0.0, else: inter / union
  end

  @doc """
  Convert `{N, 4}` boxes from `[cx, cy, w, h]` to `[x1, y1, x2, y2]`.
  YOLO heads typically emit the cx/cy/w/h form.
  """
  def decode_xywh_to_xyxy(boxes_xywh) do
    cpu = Nx.backend_copy(boxes_xywh, Nx.BinaryBackend)
    cx = Nx.slice_along_axis(cpu, 0, 1, axis: -1)
    cy = Nx.slice_along_axis(cpu, 1, 1, axis: -1)
    w = Nx.slice_along_axis(cpu, 2, 1, axis: -1)
    h = Nx.slice_along_axis(cpu, 3, 1, axis: -1)

    half_w = Nx.divide(w, 2.0)
    half_h = Nx.divide(h, 2.0)

    x1 = Nx.subtract(cx, half_w)
    y1 = Nx.subtract(cy, half_h)
    x2 = Nx.add(cx, half_w)
    y2 = Nx.add(cy, half_h)

    Nx.concatenate([x1, y1, x2, y2], axis: -1)
    |> Nx.backend_copy(NxArm.Backend)
  end

  @doc """
  Decode YOLOv5-style anchor predictions to absolute `[cx, cy, w, h]`
  in input-image pixel coordinates.

  Inputs:
    * `pred` — `{N, 4}` raw model outputs `[tx, ty, tw, th]`.
    * `anchors` — `{N, 4}` corresponding anchor centres + sizes
      `[ax, ay, aw, ah]` in input-image pixels.
    * `:stride` — feature-map stride relative to input image.

  Decoded:
      cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
      cy = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
      w  = (sigmoid(tw) * 2)^2 * anchor_w
      h  = (sigmoid(th) * 2)^2 * anchor_h

  This module provides the box-form decode; callers who need the
  raw anchor-grid wiring can call `iou/2` + `nms/3` directly with
  their own decoded boxes.
  """
  def decode_yolov5(_pred, _anchors, _opts \\ []) do
    raise "decode_yolov5 needs a worked YOLOv5 pipeline to validate; provided as a stub for now."
  end
end

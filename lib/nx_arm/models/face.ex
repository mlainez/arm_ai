defmodule NxArm.Models.Face do
  @moduledoc """
  Face detection + recognition via ONNX models.

  Like the OCR module, this is glue for the standard pipelines
  rather than a bundled model:

    * **Detection** — MTCNN, RetinaFace, SCRFD, YuNet. All export
      to ONNX. Output is bounding boxes (and usually 5 landmarks
      per face).
    * **Recognition** — ArcFace, FaceNet, InsightFace. Take a
      cropped+aligned face, emit a 128- or 512-dim embedding.
      Use `NxArm.Embeddings.cosine_similarity/2` against a known-face
      DB for identification.

  ## Pipeline

  ```elixir
  {:ok, det} = NxArm.Models.Face.load_detector("/data/yunet.onnx")
  {:ok, rec} = NxArm.Models.Face.load_recognizer("/data/arcface.onnx")

  image = NxArm.Vision.load_for_classifier("/data/photo.jpg",
            size: {640, 640}, layout: :nchw,
            mean: {0.0, 0.0, 0.0}, std: {1.0, 1.0, 1.0})

  faces = NxArm.Models.Face.detect(det, image,
                                    iou_threshold: 0.4,
                                    score_threshold: 0.6)
  # faces = [%{box: {x1, y1, x2, y2}, score: 0.93, landmarks: [{x,y}, ...]}, ...]

  for face <- faces do
    crop = NxArm.Models.Face.align(image, face)
    embedding = NxArm.Models.Face.embed(rec, crop)
    # embedding is shape {512}, compare with NxArm.Embeddings.cosine_similarity/2
  end
  ```

  The `align/2` step (face alignment via landmarks → 112×112 crop)
  is the bit you'll typically tune per detector. We provide a
  simple bbox-crop fallback.
  """

  defstruct [:onnx, :stage]

  @doc "Load a face-detection ONNX (YuNet, RetinaFace, SCRFD, MTCNN)."
  @spec load_detector(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load_detector(path) do
    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model, stage: :detection}}
      err -> err
    end
  end

  @doc "Load a face-recognition / embedding ONNX (ArcFace, FaceNet)."
  @spec load_recognizer(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load_recognizer(path) do
    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model, stage: :recognition}}
      err -> err
    end
  end

  @doc """
  Run a face detector on a preprocessed image tensor. The raw
  output layout depends on the model — pass a `:decoder` callback
  to translate logits → `[%{box, score, landmarks}]`.

  Default decoder assumes YuNet-style output:
    `{1, n_priors, 14}` where `[0..3] = box xyxy`,
    `[4..13] = 5 landmarks`, plus a parallel score tensor.

  This default is a placeholder — most users will provide their
  own decoder once they pick a detector. We just expose the model
  + give callers the raw outputs to decode however they want.
  """
  @spec detect(%__MODULE__{}, Nx.Tensor.t(), keyword()) :: any()
  def detect(%__MODULE__{onnx: onnx, stage: :detection}, image, opts \\ []) do
    decoder = Keyword.get(opts, :decoder, &(&1))
    input_name = List.first(onnx.input_names) || "input"

    image4d =
      case Nx.shape(image) do
        {_n, _c, _h, _w} -> image
        {c, h, w} -> Nx.reshape(image, {1, c, h, w})
        {h, w, c} -> Nx.reshape(image, {1, c, h, w})
      end

    outputs = NxArm.Models.Onnx.run(onnx, %{input_name => image4d})
    decoder.(outputs)
  end

  @doc """
  Compute a face embedding for a preprocessed crop. Returns the
  raw embedding tensor — typically `{512}` (ArcFace) or `{128}`
  (FaceNet) — on `NxArm.Backend`.
  """
  @spec embed(%__MODULE__{}, Nx.Tensor.t()) :: Nx.Tensor.t()
  def embed(%__MODULE__{onnx: onnx, stage: :recognition}, crop) do
    input_name = List.first(onnx.input_names) || "data"

    crop4d =
      case Nx.shape(crop) do
        {_n, _c, _h, _w} -> crop
        {c, h, w} -> Nx.reshape(crop, {1, c, h, w})
      end

    outputs = NxArm.Models.Onnx.run(onnx, %{input_name => crop4d})
    {_, embedding} = Enum.at(outputs, 0)
    Nx.flatten(embedding)
  end

  @doc """
  Quick bbox-only crop helper. For landmark-based alignment (the
  ArcFace/InsightFace expected input), implement per-detector and
  pass the warped crop to `embed/2`.
  """
  @spec crop_box(Nx.Tensor.t(), {number(), number(), number(), number()}) :: Nx.Tensor.t()
  def crop_box(image, {x1, y1, x2, y2}) do
    {_, _, h, w} = Nx.shape(image)
    x1_i = trunc(max(0, x1))
    y1_i = trunc(max(0, y1))
    cw = trunc(min(x2, w - 1) - x1_i)
    ch = trunc(min(y2, h - 1) - y1_i)
    Nx.slice(image, [0, 0, y1_i, x1_i], [1, 3, ch, cw])
  end
end

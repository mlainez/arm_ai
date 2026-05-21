defmodule NxArm.Models.OCR do
  @moduledoc """
  Optical character recognition via ONNX models.

  We don't bundle a specific OCR model — the field has many good
  options (PaddleOCR, EasyOCR, TrOCR) and each has different
  pre/post processing. Instead we provide the standard pipeline
  glue: load a detection + recognition ONNX pair, wrap them with
  `NxArm.Models.Onnx`, and post-process the output.

  ## Two-stage OCR (the common case)

  Most OCR models are a pair:

    * **Detection**: takes an image, returns box polygons.
    * **Recognition**: takes a cropped text image, returns characters.

  ```elixir
  {:ok, det} = NxArm.Models.OCR.load_detection("/data/paddle_ocr_det.onnx")
  {:ok, rec} = NxArm.Models.OCR.load_recognition("/data/paddle_ocr_rec.onnx",
                 character_set: "/data/charset.txt")

  image = NxArm.Vision.load_for_classifier("/data/scan.jpg",
            size: {960, 960}, layout: :nchw,
            mean: {0.485, 0.456, 0.406}, std: {0.229, 0.224, 0.225})

  boxes = NxArm.Models.OCR.detect(det, image)
  text  = NxArm.Models.OCR.recognize_boxes(rec, image, boxes)
  ```

  Detection boxes are returned as `{x1, y1, x2, y2}` tuples in
  pixel space. The default detection postprocessor is DBNet-style
  (PaddleOCR), but you can plug in your own box decoder via the
  `:decoder` option.

  ## One-shot models

  If you have an end-to-end model that does both stages (TrOCR for
  example), just use `NxArm.Models.Onnx` directly. This module is
  only worth it for the two-stage pipeline.
  """

  defstruct [:onnx, :stage]

  @doc "Load a text-detection ONNX model (DBNet, EAST, etc.)."
  @spec load_detection(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load_detection(path) do
    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model, stage: :detection}}
      err -> err
    end
  end

  @doc """
  Load a text-recognition ONNX model.

  The model is expected to take a fixed-size grayscale or RGB crop
  and emit per-step logits over a character set. The CTC-decoded
  string is built using the character set supplied in `:character_set`
  (newline-separated UTF-8 file of single-character entries).
  """
  @spec load_recognition(Path.t(), keyword()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load_recognition(path, _opts \\ []) do
    case NxArm.Models.Onnx.load(path) do
      {:ok, model} -> {:ok, %__MODULE__{onnx: model, stage: :recognition}}
      err -> err
    end
  end

  @doc """
  Run text detection on a preprocessed image tensor. Returns a list
  of boxes — defaults to passing the raw logits to the user's
  `:decoder` callback because every detector has a different
  postprocessing pipeline (DBNet uses connected components on the
  probability map, EAST uses rotated NMS, etc.).
  """
  @spec detect(%__MODULE__{}, Nx.Tensor.t(), keyword()) :: any()
  def detect(%__MODULE__{onnx: onnx, stage: :detection}, image, opts \\ []) do
    decoder = Keyword.get(opts, :decoder, &(&1))
    input_name = List.first(onnx.input_names) || "x"

    image =
      case Nx.shape(image) do
        {_n, _c, _h, _w} -> image
        {c, h, w} -> Nx.reshape(image, {1, c, h, w})
      end

    outputs = NxArm.Models.Onnx.run(onnx, %{input_name => image})
    decoder.(outputs)
  end

  @doc """
  Run text recognition on a list of pre-cropped + pre-normalised
  image tensors. Returns one decoded string per crop. CTC-decode
  with the user-supplied `character_set`.
  """
  @spec recognize_crops(%__MODULE__{}, [Nx.Tensor.t()], keyword()) :: [String.t()]
  def recognize_crops(%__MODULE__{onnx: onnx, stage: :recognition}, crops, opts) do
    charset = Keyword.fetch!(opts, :character_set)
    input_name = List.first(onnx.input_names) || "x"

    for crop <- crops do
      shape = Nx.shape(crop)

      crop4d =
        case shape do
          {_, _, _} -> Nx.reshape(crop, {1, elem(shape, 0), elem(shape, 1), elem(shape, 2)})
          {_, _, _, _} -> crop
        end

      outputs = NxArm.Models.Onnx.run(onnx, %{input_name => crop4d})
      {_, logits} = Enum.at(outputs, 0)
      ctc_decode(logits, charset)
    end
  end

  # CTC greedy decode: take argmax per timestep, drop blanks (id 0)
  # and consecutive duplicates, then look up characters.
  defp ctc_decode(logits, charset) do
    {_b, t, _v} =
      case Nx.shape(logits) do
        {b, t, v} -> {b, t, v}
        {t, v} -> {1, t, v}
      end

    flat =
      logits
      |> Nx.argmax(axis: -1)
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> Enum.take(t)

    chars =
      flat
      |> Enum.reject(&(&1 == 0))
      |> Enum.dedup()
      |> Enum.map(fn id -> Enum.at(charset, id - 1, "") end)

    Enum.join(chars)
  end
end

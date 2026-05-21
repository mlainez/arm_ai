defmodule NxArm.Vision do
  @moduledoc """
  Image preprocessing for vision models on Nerves.

  Wraps `image` (decode JPEG/PNG/WebP) + `fast_image_resize`
  (SIMD bilinear resize) + Rust normalisation into the one-line
  "image on disk → ViT-ready tensor" pipeline that Bumblebee
  callers would otherwise reinvent.

      # ImageNet-normalised 224×224 RGB tensor for ViT / EfficientNet:
      x = NxArm.Vision.load_for_classifier("/data/cat.jpg",
            size: {224, 224},
            mean: {0.485, 0.456, 0.406},
            std:  {0.229, 0.224, 0.225},
            layout: :nchw
          )
      # x is shape {3, 224, 224} on NxArm.Backend.

  Or the step-by-step path when you need raw pixels (e.g. for a
  custom OpenCV-style pipeline):

      {bin, w, h} = NxArm.Vision.decode_to_rgb8("/data/cat.jpg")
  """

  @doc """
  Decode any container `image` understands (JPEG/PNG/WebP — features
  gated in our Cargo.toml) into raw RGB8 bytes plus dimensions.
  Returns `{rgb_binary, width, height}`.
  """
  @spec decode_to_rgb8(Path.t()) :: {binary(), pos_integer(), pos_integer()}
  def decode_to_rgb8(path), do: NxArm.Native.vision_decode_to_rgb8_op(path)

  @doc """
  Decode + resize + normalise an image into a model-ready f32
  tensor.

  ## Options

    * `:size` — `{height, width}` (required). The model's input shape.
    * `:mean` — `{r, g, b}` per-channel mean in [0, 1] (default ImageNet).
    * `:std` — `{r, g, b}` per-channel stddev in [0, 1] (default ImageNet).
    * `:layout` — `:nchw` (PyTorch / Bumblebee default) or `:nhwc`
      (TensorFlow). Default `:nchw`.

  Returns an Nx tensor on `NxArm.Backend`:
    * Shape `{3, h, w}` when `:layout == :nchw`
    * Shape `{h, w, 3}` when `:layout == :nhwc`
  """
  @spec load_for_classifier(Path.t(), keyword()) :: Nx.Tensor.t()
  def load_for_classifier(path, opts) do
    {out_h, out_w} = Keyword.fetch!(opts, :size)
    {m_r, m_g, m_b} = Keyword.get(opts, :mean, {0.485, 0.456, 0.406})
    {s_r, s_g, s_b} = Keyword.get(opts, :std, {0.229, 0.224, 0.225})
    layout = Keyword.get(opts, :layout, :nchw)

    bin =
      NxArm.Native.vision_load_for_classifier_op(
        path,
        out_h,
        out_w,
        {m_r, m_g, m_b},
        {s_r, s_g, s_b},
        layout
      )

    shape =
      case layout do
        :nchw -> {3, out_h, out_w}
        :nhwc -> {out_h, out_w, 3}
      end

    Nx.from_binary(bin, :f32) |> Nx.reshape(shape) |> Nx.backend_copy(NxArm.Backend)
  end
end

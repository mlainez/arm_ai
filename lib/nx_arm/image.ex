defmodule NxArm.Image do
  @moduledoc """
  Image preprocessing for vision models.

  This module covers the "raw RGB u8 buffer → normalised f32 NHWC
  tensor" pipeline. Decoding JPEG/PNG to a raw RGB buffer is the
  caller's responsibility — on Nerves that typically comes from a
  camera library (`fp3_camera`, GStreamer) or from `stb_image` if a
  pure-Elixir JPEG decoder is needed.

  ## Standard ViT/ImageNet pipeline

      raw_rgb = camera_capture()           # H × W × 3 u8 binary
      tensor =
        raw_rgb
        |> NxArm.Image.from_raw_rgb(camera_h, camera_w)
        |> NxArm.Image.resize_bilinear(224, 224)
        |> NxArm.Image.to_f32_normalized(
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225]
        )

      # tensor :: {1, 224, 224, 3} f32 NxArm.Backend, model-ready
  """

  @doc """
  Wrap a raw HWC u8 RGB binary as a `{H, W, 3}` u8 NxArm tensor.
  Zero-copy aside from the BEAM binary handoff.
  """
  def from_raw_rgb(bin, h, w) when is_binary(bin) and is_integer(h) and is_integer(w) do
    if byte_size(bin) != h * w * 3 do
      raise ArgumentError,
            "from_raw_rgb: bytes #{byte_size(bin)} != H*W*3 = #{h * w * 3}"
    end

    Nx.from_binary(bin, :u8, backend: NxArm.Backend) |> Nx.reshape({h, w, 3})
  end

  @doc """
  Bilinear-resize an HWC u8 image to `{out_h, out_w, channels}`. Works
  on any `{H, W, C}` u8 tensor.
  """
  def resize_bilinear(%Nx.Tensor{shape: {h, w, c}, type: {:u, 8}} = img, out_h, out_w) do
    bin = NxArm.Backend.__bin_of__(img)
    out_bin = NxArm.Native.bilinear_resize_u8_op(bin, h, w, c, out_h, out_w)

    Nx.from_binary(out_bin, :u8, backend: NxArm.Backend)
    |> Nx.reshape({out_h, out_w, c})
  end

  @doc """
  Cast a u8 image tensor to f32 in `[0, 1)`, normalise per-channel
  `(x - mean) / std`, then add a leading batch dim → `{1, H, W, C}`.

  `:mean` and `:std` default to ImageNet stats (RGB order).
  """
  def to_f32_normalized(%Nx.Tensor{shape: {h, w, c}, type: {:u, 8}} = img, opts \\ []) do
    mean = Keyword.get(opts, :mean, [0.485, 0.456, 0.406])
    std = Keyword.get(opts, :std, [0.229, 0.224, 0.225])

    if length(mean) != c or length(std) != c do
      raise ArgumentError,
            "to_f32_normalized: mean/std length must match channels (#{c})"
    end

    f32 =
      img
      |> Nx.as_type(:f32)
      |> Nx.divide(255.0)

    mean_t = Nx.tensor(mean, type: :f32, backend: NxArm.Backend)
    std_t = Nx.tensor(std, type: :f32, backend: NxArm.Backend)

    f32
    |> Nx.subtract(mean_t)
    |> Nx.divide(std_t)
    |> Nx.reshape({1, h, w, c})
  end
end

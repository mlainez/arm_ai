defmodule NxArm.ImageTest do
  use ExUnit.Case, async: true

  describe "from_raw_rgb" do
    test "wraps an HxWx3 u8 binary" do
      bin = <<255, 0, 0,  0, 255, 0,  0, 0, 255,  128, 128, 128>>
      img = NxArm.Image.from_raw_rgb(bin, 2, 2)
      assert Nx.shape(img) == {2, 2, 3}
      assert Nx.type(img) == {:u, 8}
      assert Nx.to_flat_list(img) == [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]
    end

    test "rejects wrong byte size" do
      assert_raise ArgumentError, fn -> NxArm.Image.from_raw_rgb(<<1, 2, 3>>, 4, 4) end
    end
  end

  describe "resize_bilinear" do
    test "identity when target == source" do
      bin = for i <- 0..47, into: <<>>, do: <<rem(i, 256)>>
      img = NxArm.Image.from_raw_rgb(bin, 4, 4)
      resized = NxArm.Image.resize_bilinear(img, 4, 4)
      assert Nx.to_flat_list(img) == Nx.to_flat_list(resized)
    end

    test "downsample 4x4 → 2x2 produces sensible averages" do
      # A 4x4 RGB image where all pixels = (100, 100, 100). Downsampled
      # bilinear should still be ~(100, 100, 100).
      bin = :binary.copy(<<100, 100, 100>>, 16)
      img = NxArm.Image.from_raw_rgb(bin, 4, 4)
      resized = NxArm.Image.resize_bilinear(img, 2, 2)
      assert Nx.shape(resized) == {2, 2, 3}
      flat = Nx.to_flat_list(resized)
      assert Enum.all?(flat, &(&1 == 100))
    end

    test "upsample 2x2 → 4x4 produces interpolated values" do
      # Single-channel-equivalent: red=0 on left, red=255 on right.
      bin = <<0, 0, 0,  255, 0, 0,    0, 0, 0,  255, 0, 0>>
      img = NxArm.Image.from_raw_rgb(bin, 2, 2)
      resized = NxArm.Image.resize_bilinear(img, 4, 4)
      assert Nx.shape(resized) == {4, 4, 3}
      # First row should interpolate red from 0 → 255 across 4 pixels.
      [r0, _, _, r1, _, _, r2, _, _, r3, _, _ | _] = Nx.to_flat_list(resized)
      assert r0 == 0
      assert r3 == 255
      assert r1 > r0 and r1 < r2
      assert r2 > r1 and r2 < r3
    end
  end

  describe "to_f32_normalized" do
    test "normalises with ImageNet stats and adds batch dim" do
      bin = <<255, 0, 0>> <> <<0, 255, 0>> <> <<0, 0, 255>> <> <<128, 128, 128>>
      img = NxArm.Image.from_raw_rgb(bin, 2, 2)
      out = NxArm.Image.to_f32_normalized(img)
      assert Nx.shape(out) == {1, 2, 2, 3}
      assert Nx.type(out) == {:f, 32}
    end

    test "custom mean/std" do
      bin = :binary.copy(<<128, 128, 128>>, 4)
      img = NxArm.Image.from_raw_rgb(bin, 2, 2)
      out =
        NxArm.Image.to_f32_normalized(img,
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5]
        )
      # 128/255 ≈ 0.502; (0.502 - 0.5) / 0.5 ≈ 0.004
      [v | _] = Nx.to_flat_list(out)
      assert_in_delta v, 0.004, 1.0e-3
    end
  end

  describe "full pipeline" do
    test "ViT-tiny preprocessing: raw 320x240 RGB → 1x224x224x3 f32" do
      # Synthetic gradient: r = x, g = y, b = (x+y) mod 256
      bin =
        for y <- 0..(240 - 1), x <- 0..(320 - 1), into: <<>> do
          <<x::8, y::8, rem(x + y, 256)::8>>
        end

      out =
        NxArm.Image.from_raw_rgb(bin, 240, 320)
        |> NxArm.Image.resize_bilinear(224, 224)
        |> NxArm.Image.to_f32_normalized()

      assert Nx.shape(out) == {1, 224, 224, 3}
      assert Nx.type(out) == {:f, 32}
    end
  end
end

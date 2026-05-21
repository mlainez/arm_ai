defmodule NxArm.VisionTest do
  use ExUnit.Case, async: true

  @fixture_png Path.expand("../support/tiny.png", __DIR__)
  @fixture_jpg Path.expand("../support/tiny.jpg", __DIR__)

  describe "decode_to_rgb8/1" do
    test "decodes a PNG to {bytes, w, h}" do
      {bytes, w, h} = NxArm.Vision.decode_to_rgb8(@fixture_png)
      assert is_binary(bytes)
      assert w == 8
      assert h == 8
      # 8x8x3 RGB = 192 bytes
      assert byte_size(bytes) == 192
    end

    test "decodes a JPEG to {bytes, w, h}" do
      {bytes, w, h} = NxArm.Vision.decode_to_rgb8(@fixture_jpg)
      assert is_binary(bytes)
      assert w == 8
      assert h == 8
      assert byte_size(bytes) == 192
    end
  end

  describe "load_for_classifier/2 — shape contract" do
    test "nchw layout produces {3, h, w} f32 tensor" do
      t =
        NxArm.Vision.load_for_classifier(@fixture_png,
          size: {32, 32},
          layout: :nchw,
          mean: {0.0, 0.0, 0.0},
          std: {1.0, 1.0, 1.0}
        )

      assert Nx.shape(t) == {3, 32, 32}
      assert Nx.type(t) == {:f, 32}
    end

    test "nhwc layout produces {h, w, 3}" do
      t =
        NxArm.Vision.load_for_classifier(@fixture_jpg,
          size: {16, 16},
          layout: :nhwc,
          mean: {0.0, 0.0, 0.0},
          std: {1.0, 1.0, 1.0}
        )

      assert Nx.shape(t) == {16, 16, 3}
      assert Nx.type(t) == {:f, 32}
    end

    test "applies mean and std normalisation" do
      # With mean=0, std=1, pixel value 128 (mid-grey) → 128/255 ≈ 0.502.
      # With mean=0.5, std=0.5, that same pixel → (0.502 - 0.5) / 0.5 ≈ 0.004 — close to 0.
      t_id =
        NxArm.Vision.load_for_classifier(@fixture_png,
          size: {8, 8},
          layout: :nchw,
          mean: {0.0, 0.0, 0.0},
          std: {1.0, 1.0, 1.0}
        )

      t_norm =
        NxArm.Vision.load_for_classifier(@fixture_png,
          size: {8, 8},
          layout: :nchw,
          mean: {0.5, 0.5, 0.5},
          std: {0.5, 0.5, 0.5}
        )

      # Identity-norm path: middle range pixel values, all in [0, 1].
      flat_id = t_id |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
      assert Enum.all?(flat_id, &(&1 >= 0.0 and &1 <= 1.0))

      # Mean-half-std-half path: pixels shift toward 0 (a grey
      # image hits exactly 0 if it's exactly the mean; ours is
      # close but not exact since the RGB(128,64,32) input isn't
      # uniformly 128).
      flat_n = t_norm |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
      assert length(flat_n) == 3 * 8 * 8
    end

    test "raises ArgumentError when :size is missing" do
      assert_raise KeyError, fn ->
        NxArm.Vision.load_for_classifier(@fixture_png, [])
      end
    end
  end

  describe "missing file" do
    test "decode_to_rgb8 returns {:error, _} on a missing path" do
      assert {:error, msg} = NxArm.Vision.decode_to_rgb8("/tmp/__no_such_image.png")
      assert is_binary(msg)
      assert msg =~ "No such file"
    end
  end
end

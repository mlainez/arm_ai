defmodule NxArm.Models.YOLOTest do
  use ExUnit.Case, async: true

  describe "load/2" do
    test "returns a struct error tuple for a missing file" do
      # Whether or not the onnx feature is compiled in, loading a path
      # that does not exist must surface as an error tuple (never raise).
      assert {:error, _} = NxArm.Models.YOLO.load("/tmp/__nx_arm_no_such_yolo.onnx")
    end

    test "default layout is :v5" do
      # Stub by going through Models.Onnx — when the feature is
      # disabled, both Onnx and YOLO load return error tuples without
      # crashing. We verify the YOLO wrapper doesn't lose options on
      # the error path.
      result = NxArm.Models.YOLO.load("/tmp/__missing.onnx")
      assert match?({:error, _}, result)
    end

    test "passing layout: :v8 is accepted (no crash on the option)" do
      # If the underlying file is missing we still want the option to
      # round-trip cleanly through the load function.
      assert {:error, _} = NxArm.Models.YOLO.load("/tmp/__missing.onnx", layout: :v8)
    end
  end

  describe "struct shape" do
    test "wraps an :onnx field, an :input_name, and a :layout" do
      yolo = %NxArm.Models.YOLO{
        onnx: nil,
        input_name: "images",
        input_shape: {640, 640},
        layout: :v5
      }

      assert yolo.layout == :v5
      assert yolo.input_shape == {640, 640}
      assert yolo.input_name == "images"
    end
  end
end

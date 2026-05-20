defmodule NxArm.DetectionTest do
  use ExUnit.Case, async: true

  describe "iou" do
    test "identical boxes → 1.0" do
      assert NxArm.Detection.iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) == 1.0
    end

    test "disjoint boxes → 0.0" do
      assert NxArm.Detection.iou([0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 15.0, 15.0]) == 0.0
    end

    test "half-overlap" do
      # box A: 0..10 × 0..10 (area 100)
      # box B: 5..15 × 0..10 (area 100)
      # intersection: 5..10 × 0..10 (area 50)
      # union: 100 + 100 - 50 = 150
      # iou = 50/150 ≈ 0.333
      iou = NxArm.Detection.iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0])
      assert_in_delta iou, 1.0 / 3.0, 1.0e-6
    end
  end

  describe "nms" do
    test "single box passes through" do
      boxes = Nx.tensor([[0.0, 0.0, 10.0, 10.0]])
      scores = Nx.tensor([0.9])
      assert NxArm.Detection.nms(boxes, scores) == [0]
    end

    test "overlapping boxes get suppressed by lower-scored one" do
      boxes =
        Nx.tensor([
          [0.0, 0.0, 10.0, 10.0],
          [1.0, 1.0, 11.0, 11.0],
          [50.0, 50.0, 60.0, 60.0]
        ])

      scores = Nx.tensor([0.9, 0.8, 0.7])
      kept = NxArm.Detection.nms(boxes, scores, iou_threshold: 0.5)
      # Box 0 has higher score than box 1 (overlapping); box 2 is separate.
      assert kept == [0, 2]
    end

    test "score_threshold filters before NMS" do
      boxes =
        Nx.tensor([
          [0.0, 0.0, 10.0, 10.0],
          [50.0, 50.0, 60.0, 60.0]
        ])

      scores = Nx.tensor([0.9, 0.05])
      kept = NxArm.Detection.nms(boxes, scores, score_threshold: 0.1)
      assert kept == [0]
    end

    test "max_output caps the result" do
      boxes =
        Nx.tensor([
          [0.0, 0.0, 10.0, 10.0],
          [20.0, 20.0, 30.0, 30.0],
          [40.0, 40.0, 50.0, 50.0],
          [60.0, 60.0, 70.0, 70.0]
        ])

      scores = Nx.tensor([0.95, 0.85, 0.75, 0.65])
      kept = NxArm.Detection.nms(boxes, scores, max_output: 2)
      assert kept == [0, 1]
    end

    test "high IoU threshold keeps overlapping boxes" do
      boxes =
        Nx.tensor([
          [0.0, 0.0, 10.0, 10.0],
          [1.0, 1.0, 11.0, 11.0]
        ])

      scores = Nx.tensor([0.9, 0.8])
      kept = NxArm.Detection.nms(boxes, scores, iou_threshold: 0.95)
      assert kept == [0, 1]
    end
  end

  describe "decode_xywh_to_xyxy" do
    test "converts centre+size to corners" do
      # Box at (5, 5) with size (4, 6) → (3, 2, 7, 8)
      xywh = Nx.tensor([[5.0, 5.0, 4.0, 6.0]])
      xyxy = NxArm.Detection.decode_xywh_to_xyxy(xywh) |> Nx.backend_copy(Nx.BinaryBackend)
      assert Nx.to_flat_list(xyxy) == [3.0, 2.0, 7.0, 8.0]
    end

    test "batch of boxes" do
      xywh =
        Nx.tensor([
          [10.0, 10.0, 20.0, 20.0],
          [50.0, 50.0, 10.0, 10.0]
        ])

      xyxy = NxArm.Detection.decode_xywh_to_xyxy(xywh) |> Nx.backend_copy(Nx.BinaryBackend)
      assert Nx.to_flat_list(xyxy) == [0.0, 0.0, 20.0, 20.0, 45.0, 45.0, 55.0, 55.0]
    end
  end
end

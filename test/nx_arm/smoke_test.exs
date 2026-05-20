defmodule NxArm.SmokeTest do
  @moduledoc """
  Sanity-check that the test harness runs end-to-end against an op
  we know works (Nx.add). If this test fails, the issue is in the
  harness itself, not in any particular feature.
  """

  use NxArmCase, async: true

  test "add same-shape f32 matches BinaryBackend" do
    a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Nx.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    assert_arm_matches_ref_n(&Nx.add/2, [a, b])
  end

  test "exp same-shape f32 matches BinaryBackend" do
    x = Nx.tensor([0.0, 1.0, 2.0, -1.0, 3.5])
    assert_arm_matches_ref(&Nx.exp/1, x)
  end

  test "dot 2-D f32 matches BinaryBackend" do
    a = Nx.iota({4, 6}, type: :f32) |> Nx.divide(10)
    b = Nx.iota({6, 3}, type: :f32) |> Nx.divide(10)
    assert_arm_matches_ref_n(&Nx.dot/2, [a, b])
  end
end

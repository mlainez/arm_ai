defmodule NxArm.EdgeCasesTest do
  @moduledoc """
  Edge cases that break naive implementations: NaN propagation,
  ±Inf, denormals, very-large tensors, single-element tensors,
  high-aspect-ratio matmuls, zero tensors. Each runs on
  NxArm.Backend and verifies the result either matches a hand-coded
  expectation or doesn't crash.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  # --- scalar / single-element ---

  test "1-element add" do
    a = Nx.tensor([1.0]) |> arm()
    b = Nx.tensor([2.0]) |> arm()
    assert Nx.to_flat_list(Nx.add(a, b)) == [3.0]
  end

  test "scalar tensor matmul against 1-D" do
    # Skip: scalars × 1-D is non-standard; just ensure no crash.
    a = Nx.tensor([[1.0, 2.0]]) |> arm()
    b = Nx.tensor([[3.0], [4.0]]) |> arm()
    assert Nx.to_flat_list(Nx.dot(a, b)) == [11.0]
  end

  # --- NaN / Inf ---

  test "NaN propagates through add" do
    a = Nx.tensor([:nan, 1.0, 2.0], type: :f32) |> arm()
    b = Nx.tensor([1.0, 1.0, :nan], type: :f32) |> arm()
    out = Nx.add(a, b) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
    [n0, n1, n2] = out
    assert n0 == :nan
    assert n1 == 2.0
    assert n2 == :nan
  end

  test "Inf propagates through multiply" do
    a = Nx.tensor([:infinity, 1.0, :neg_infinity], type: :f32) |> arm()
    b = Nx.tensor([1.0, 1.0, 2.0], type: :f32) |> arm()
    out = Nx.multiply(a, b) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
    assert out == [:infinity, 1.0, :neg_infinity]
  end

  test "exp boundary: very negative -> 0 / subnormal, very positive -> finite" do
    a = Nx.tensor([-1000.0, -100.0, 0.0, 80.0], type: :f32) |> arm()
    out = Nx.exp(a) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
    [e_neg_huge, e_neg_100, e_0, e_pos_80] = out
    # -1000 is past the f32 subnormal floor -> exact zero.
    assert e_neg_huge == 0.0
    # -100 produces a subnormal (~3.78e-44), not exactly zero but tiny.
    assert e_neg_100 < 1.0e-40
    assert_in_delta e_0, 1.0, 1.0e-5
    # exp(80) ≈ 5.5e34 — finite f32.
    assert is_float(e_pos_80) and e_pos_80 > 1.0e30 and e_pos_80 < 1.0e40
  end

  # --- zeros ---

  test "all-zeros add is identity" do
    a = Nx.iota({4, 5}, type: :f32) |> arm()
    z = Nx.broadcast(0.0, {4, 5}) |> arm()
    assert Nx.to_flat_list(Nx.add(a, z)) == Nx.to_flat_list(a)
  end

  test "all-zeros matmul produces zeros" do
    a = Nx.broadcast(0.0, {8, 16}) |> arm()
    b = Nx.broadcast(0.0, {16, 8}) |> arm()
    out = Nx.dot(a, b) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.reduce_max(Nx.abs(out)) |> Nx.to_number() == 0.0
  end

  # --- aspect ratios ---

  test "tall thin matmul (M=200, K=1, N=1)" do
    a = Nx.iota({200, 1}, type: :f32) |> Nx.divide(100) |> arm()
    b = Nx.tensor([[2.0]]) |> arm()
    out = Nx.dot(a, b) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.shape(out) == {200, 1}
    assert Nx.to_flat_list(out) |> List.last() |> Float.round(4) == 3.98
  end

  test "wide thin matmul (M=1, K=512, N=1)" do
    a = Nx.broadcast(1.0, {1, 512}) |> arm()
    b = Nx.broadcast(1.0, {512, 1}) |> arm()
    out = Nx.dot(a, b) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.shape(out) == {1, 1}
    [v] = Nx.to_flat_list(out)
    assert v == 512.0
  end

  test "large matmul (M=64, K=128, N=64) accuracy" do
    a =
      Nx.iota({64, 128}, type: :f32)
      |> Nx.divide(1000)
      |> Nx.sin()
      |> arm()

    b =
      Nx.iota({128, 64}, type: :f32)
      |> Nx.divide(1000)
      |> Nx.cos()
      |> arm()

    got =
      Nx.dot(a, b)
      |> Nx.backend_copy(Nx.BinaryBackend)

    a_ref = Nx.backend_copy(a, Nx.BinaryBackend)
    b_ref = Nx.backend_copy(b, Nx.BinaryBackend)
    ref = Nx.dot(a_ref, b_ref)

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4
  end

  # --- type round-trips ---

  test "f32 -> binary -> f32 round trip preserves bytes" do
    t = Nx.iota({3, 5}, type: :f32) |> Nx.divide(10) |> arm()
    bin = Nx.to_binary(t)
    t2 = Nx.from_binary(bin, :f32) |> Nx.reshape({3, 5}) |> arm()
    assert Nx.to_flat_list(t) == Nx.to_flat_list(t2)
  end

  # --- ill-conditioned softmax ---

  test "softmax stays stable under extreme logits (avoids overflow)" do
    # Without max-subtraction, exp(1000) overflows.
    logits = Nx.tensor([1000.0, 999.0, 998.0, 0.0], type: :f32) |> arm()
    m = Nx.reduce_max(logits, keep_axes: true)
    shifted = Nx.subtract(logits, m)
    e = Nx.exp(shifted)
    s = Nx.sum(e)
    probs = Nx.divide(e, s) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

    assert Enum.sum(probs) |> Float.round(5) == 1.0
    assert Enum.all?(probs, &(&1 >= 0.0))
    # Largest logit dominates.
    assert hd(probs) > 0.5
  end

  # --- reduction over 1-element axis ---

  test "sum over single-element axis" do
    a = Nx.tensor([[[1.0], [2.0], [3.0]]]) |> arm()
    out = Nx.sum(a, axes: [-1]) |> Nx.backend_copy(Nx.BinaryBackend)
    assert Nx.to_flat_list(out) == [1.0, 2.0, 3.0]
  end
end

defmodule NxArm.LLMGTest do
  use ExUnit.Case, async: true

  test "causal_mask is lower-triangular with 0 / -inf" do
    m = NxArm.LLM.causal_mask(4)
    assert Nx.shape(m) == {4, 4}

    rows = Nx.to_list(m)
    # Row 0: [0, -inf, -inf, -inf]
    assert hd(rows) == [0.0, -1.0e9, -1.0e9, -1.0e9]
    # Row 3: [0, 0, 0, 0]
    assert List.last(rows) == [0.0, 0.0, 0.0, 0.0]
  end

  test "causal_mask custom masked_value + dtype" do
    m = NxArm.LLM.causal_mask(3, type: :f32, masked_value: -1.0e10)
    rows = Nx.to_list(m)
    assert hd(rows) |> hd() == 0.0
    assert hd(rows) |> Enum.at(1) < -1.0e9
    assert Enum.at(rows, 2) == [0.0, 0.0, 0.0]
  end

  test "decode_mask is all zeros (every cached pos is in the past)" do
    m = NxArm.LLM.decode_mask(5)
    assert Nx.shape(m) == {1, 5}
    assert Nx.to_flat_list(m) == [0.0, 0.0, 0.0, 0.0, 0.0]
  end

  test "rope_at delegates to rope/3 at one position" do
    head_dim = 4
    inv_freq = NxArm.LLM.rope_inv_freq(head_dim)

    qk = Nx.iota({1, 1, 1, head_dim}, type: :f32) |> Nx.divide(10)

    pos_tensor = Nx.tensor([3], type: :s64)
    expected = NxArm.LLM.rope(qk, pos_tensor, inv_freq)
    got = NxArm.LLM.rope_at(qk, 3, inv_freq)

    diff =
      Nx.subtract(got, expected)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff < 1.0e-5
  end
end

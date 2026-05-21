defmodule NxArm.ProductionOpsTest do
  @moduledoc """
  Coverage for the production-readiness ops that used to fall back
  to BinaryBackend: argmax, argmin, select, as_type, clip, pad,
  stack. Each replaces a hot-path fallback that any non-trivial
  model hits.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  defp ref_matches(arm_tensor, ref_tensor, opts \\ []) do
    tol = Keyword.get(opts, :tol, 0.0)
    arm_b = Nx.backend_copy(arm_tensor, Nx.BinaryBackend)

    if Nx.type(arm_b) == Nx.type(ref_tensor) and Nx.shape(arm_b) == Nx.shape(ref_tensor) do
      if tol == 0.0 do
        Nx.to_flat_list(arm_b) == Nx.to_flat_list(ref_tensor)
      else
        diff = Nx.subtract(arm_b, ref_tensor) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        diff <= tol
      end
    else
      false
    end
  end

  # --- argmax / argmin ---

  test "argmax along last axis 1-D" do
    t = Nx.tensor([0.1, 0.9, 0.3, 0.7])
    assert Nx.argmax(arm(t), axis: -1) |> Nx.to_number() == 1
  end

  test "argmax along last axis 2-D" do
    t = Nx.tensor([[0.1, 0.9, 0.3], [0.5, 0.2, 0.8]])
    got = Nx.argmax(arm(t), axis: -1)
    assert Nx.to_flat_list(got) == [1, 2]
  end

  test "argmin along last axis 2-D" do
    t = Nx.tensor([[0.1, 0.9, 0.3], [0.5, 0.2, 0.8]])
    got = Nx.argmin(arm(t), axis: -1)
    assert Nx.to_flat_list(got) == [0, 1]
  end

  test "argmax over whole tensor (no axis)" do
    t = Nx.tensor([3.0, 1.0, 4.0, 1.5])
    assert Nx.argmax(arm(t)) |> Nx.to_number() == 2
  end

  test "argmax 3-D over last axis matches BinaryBackend" do
    t = Nx.iota({2, 3, 5}, type: :f32) |> Nx.multiply(0.1) |> Nx.sin()
    got = Nx.argmax(arm(t), axis: -1)
    ref = Nx.argmax(t, axis: -1)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  # --- select ---

  test "select with same-shape mask + values" do
    pred = Nx.tensor([[1, 0], [0, 1]], type: :u8)
    a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])

    got = Nx.select(arm(pred), arm(a), arm(b))
    assert Nx.to_flat_list(got) == [1.0, 20.0, 30.0, 4.0]
  end

  test "select with f32 values and bool pred" do
    pred = Nx.tensor([true, false, true, false])
    a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    b = Nx.tensor([-1.0, -2.0, -3.0, -4.0])

    got = Nx.select(arm(pred), arm(a), arm(b))
    assert Nx.to_flat_list(got) == [1.0, -2.0, 3.0, -4.0]
  end

  # --- as_type ---

  test "as_type f32 -> s8 (truncate toward zero, matches Nx default)" do
    t = Nx.tensor([-128.7, -1.4, 0.0, 1.6, 127.4], type: :f32)
    got = Nx.as_type(arm(t), {:s, 8})
    # truncate: -128.7 saturates to -128, -1.4 → -1, 1.6 → 1, 127.4 → 127
    assert Nx.to_flat_list(got) == [-128, -1, 0, 1, 127]
  end

  test "as_type s8 -> f32 (dequantisation-style)" do
    t = Nx.tensor([-128, -1, 0, 2, 127], type: :s8)
    got = Nx.as_type(arm(t), {:f, 32})
    assert Nx.to_flat_list(got) == [-128.0, -1.0, 0.0, 2.0, 127.0]
  end

  test "as_type f32 -> s64 (integer indexing)" do
    # Truncation toward zero matches Nx's default cast semantics.
    t = Nx.tensor([1.0, -2.0, 100.0, 0.0], type: :f32)
    got = Nx.as_type(arm(t), {:s, 64})
    assert Nx.to_flat_list(got) == [1, -2, 100, 0]
  end

  test "as_type s32 -> u8 with saturation" do
    t = Nx.tensor([-5, 0, 100, 300, 255], type: :s32)
    got = Nx.as_type(arm(t), {:u, 8})
    assert Nx.to_flat_list(got) == [0, 0, 100, 255, 255]
  end

  # --- clip ---

  test "clip f32" do
    t = Nx.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    got = Nx.clip(arm(t), Nx.tensor(-1.0), Nx.tensor(1.0))
    assert Nx.to_flat_list(got) == [-1.0, -0.5, 0.0, 0.5, 1.0]
  end

  test "clip s32" do
    t = Nx.tensor([-10, -1, 0, 5, 100], type: :s32)
    got = Nx.clip(arm(t), Nx.tensor(0, type: :s32), Nx.tensor(10, type: :s32))
    assert Nx.to_flat_list(got) == [0, 0, 0, 5, 10]
  end

  # --- pad ---

  test "pad 1-D low + high" do
    t = Nx.tensor([1.0, 2.0, 3.0])
    got = Nx.pad(arm(t), 0.0, [{2, 1, 0}])
    assert Nx.to_flat_list(got) == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]
  end

  test "pad 2-D zero on all sides" do
    t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = Nx.pad(arm(t), 0.0, [{1, 1, 0}, {1, 1, 0}])
    ref = Nx.pad(t, 0.0, [{1, 1, 0}, {1, 1, 0}])
    assert ref_matches(got, ref)
  end

  test "pad with non-zero fill" do
    t = Nx.tensor([[1.0, 2.0]])
    got = Nx.pad(arm(t), -1.0, [{1, 0, 0}, {0, 2, 0}])
    assert Nx.to_flat_list(got) == [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0]
  end

  # --- stack ---

  test "stack along axis 0 (the fast path)" do
    a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
    got = Nx.stack([arm(a), arm(b)], axis: 0)
    assert Nx.shape(got) == {2, 2, 2}
    assert Nx.to_flat_list(got) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  end

  test "stack along non-zero axis still works via fallback" do
    a = Nx.tensor([1.0, 2.0, 3.0])
    b = Nx.tensor([4.0, 5.0, 6.0])
    got = Nx.stack([arm(a), arm(b)], axis: 1)
    ref = Nx.stack([a, b], axis: 1)
    assert ref_matches(got, ref)
  end

  # --- integration: greedy decode with NIF argmax ---

  test "greedy decode loop uses NIF argmax + gather" do
    vocab = 10
    d = 4
    embed = Nx.iota({vocab, d}, type: :f32) |> Nx.divide(10) |> arm()
    proj = Nx.iota({d, vocab}, type: :f32) |> Nx.divide(10) |> arm()

    next_token = fn last ->
      x = Nx.slice(embed, [last, 0], [1, d])
      logits = Nx.dot(x, proj) |> Nx.reshape({vocab})
      Nx.argmax(logits, axis: -1) |> Nx.to_number()
    end

    tokens =
      Stream.unfold(0, fn t ->
        nxt = next_token.(t)
        {nxt, nxt}
      end)
      |> Enum.take(5)

    # Deterministic given the synthetic weights; just verify it doesn't
    # crash and stays in vocab range.
    assert length(tokens) == 5
    assert Enum.all?(tokens, &(&1 >= 0 and &1 < vocab))
  end
end

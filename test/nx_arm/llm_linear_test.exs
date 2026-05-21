defmodule NxArm.LLMLinearTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)

  defp ref_linear(x, w, b, activation) do
    # Reference: x @ w^T + b followed by activation, on BinaryBackend.
    y = Nx.dot(x, [Nx.rank(x) - 1], w, [1])
    y_b = if b, do: Nx.add(y, b), else: y

    case activation do
      :none -> y_b
      :relu -> Nx.max(y_b, 0)
      :relu6 -> Nx.min(Nx.max(y_b, 0), 6)
      :sigmoid -> Nx.sigmoid(y_b)
      :tanh -> Nx.tanh(y_b)
      :gelu ->
        inv_sqrt2 = 1.0 / :math.sqrt(2.0)
        Nx.multiply(0.5, Nx.multiply(y_b, Nx.add(1.0, Nx.erf(Nx.multiply(y_b, inv_sqrt2)))))
    end
  end

  test "linear matches reference (2-D, no bias, no activation)" do
    x = Nx.iota({3, 8}, type: :f32) |> Nx.divide(100) |> Nx.sin()
    w = Nx.iota({5, 8}, type: :f32) |> Nx.divide(100) |> Nx.cos()

    got = NxArm.LLM.linear(arm(x), arm(w), nil)
    ref = ref_linear(x, w, nil, :none)

    diff = Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5
  end

  test "linear with bias + ReLU" do
    x = Nx.iota({2, 4}, type: :f32) |> Nx.divide(50) |> Nx.subtract(0.5)
    w = Nx.iota({3, 4}, type: :f32) |> Nx.divide(50) |> Nx.subtract(0.5)
    b = Nx.tensor([0.1, -0.2, 0.3])

    got = NxArm.LLM.linear(arm(x), arm(w), arm(b), :relu)
    ref = ref_linear(x, w, b, :relu)

    diff = Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5
  end

  test "linear with bias + GELU" do
    x = Nx.iota({2, 6}, type: :f32) |> Nx.divide(20) |> Nx.subtract(0.5)
    w = Nx.iota({4, 6}, type: :f32) |> Nx.divide(20) |> Nx.subtract(0.5)
    b = Nx.tensor([0.0, 0.1, -0.1, 0.2])

    got = NxArm.LLM.linear(arm(x), arm(w), arm(b), :gelu)
    ref = ref_linear(x, w, b, :gelu)

    diff = Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4
  end

  test "linear 3-D batched (B, M, K) × (N, K)" do
    x = Nx.iota({2, 3, 4}, type: :f32) |> Nx.divide(50)
    w = Nx.iota({5, 4}, type: :f32) |> Nx.divide(50)

    got = NxArm.LLM.linear(arm(x), arm(w), nil, :none)
    ref = ref_linear(x, w, nil, :none)

    assert Nx.shape(got) == {2, 3, 5}
    diff = Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4
  end

  test "linear decode-step (M=1)" do
    x = Nx.iota({1, 128}, type: :f32) |> Nx.divide(100) |> Nx.sin()
    w = Nx.iota({4000, 128}, type: :f32) |> Nx.divide(100) |> Nx.cos()

    got = NxArm.LLM.linear(arm(x), arm(w), nil, :none)
    ref = ref_linear(x, w, nil, :none)

    assert Nx.shape(got) == {1, 4000}
    diff = Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4
  end
end

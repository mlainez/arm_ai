defmodule NxArm.GradientTest do
  @moduledoc """
  Reverse-mode autodiff through NxArm.Backend. Nx.Defn.grad emits
  a chain of forward + backward ops — every op in the graph runs
  on whichever backend the input tensors are pinned to. So if
  gradient values match BinaryBackend, our backend implements all
  the required forward ops + their VJPs without divergence.

  Mirror models: a single linear, a small MLP, and the canonical
  "loss = sum((Wx + b - y)^2)" regression chain. The training
  signal — gradient of loss w.r.t. each parameter — must agree
  between backends to within f32 noise.
  """

  use ExUnit.Case, async: true

  import Nx.Defn

  defn loss_linear(w, b, x, y) do
    pred = Nx.dot(x, w) + b
    diff = pred - y
    Nx.sum(diff * diff)
  end

  defn mlp_loss(w1, b1, w2, b2, x, y) do
    h1 = Nx.dot(x, w1) + b1
    h1_act = Nx.tanh(h1)
    pred = Nx.dot(h1_act, w2) + b2
    diff = pred - y
    Nx.sum(diff * diff)
  end

  defn sigmoid_bce(w, b, x, y) do
    z = Nx.dot(x, w) + b
    p = Nx.sigmoid(z)
    -Nx.sum(y * Nx.log(p + 1.0e-7) + (1 - y) * Nx.log(1 - p + 1.0e-7))
  end

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp diff_max(a, b) do
    a_ref = Nx.backend_copy(a, Nx.BinaryBackend)
    Nx.subtract(a_ref, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  test "grad of linear regression loss matches BinaryBackend" do
    w = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = Nx.tensor([0.1, 0.2])
    x = Nx.tensor([[1.0, 2.0, 3.0]])
    y = Nx.tensor([[0.5, 1.0]])

    grad_fn = fn w_, b_ -> grad({w_, b_}, fn {ww, bb} -> loss_linear(ww, bb, x, y) end) end

    {ref_dw, ref_db} = grad_fn.(w, b)
    {arm_dw, arm_db} = grad_fn.(arm(w), arm(b))

    assert diff_max(arm_dw, ref_dw) < 1.0e-4, "dw diverged"
    assert diff_max(arm_db, ref_db) < 1.0e-4, "db diverged"
  end

  test "grad of MLP loss with tanh activation" do
    w1 = Nx.iota({3, 4}, type: :f32) |> Nx.divide(20) |> Nx.subtract(0.2)
    b1 = Nx.iota({4}, type: :f32) |> Nx.divide(40)
    w2 = Nx.iota({4, 2}, type: :f32) |> Nx.divide(15)
    b2 = Nx.iota({2}, type: :f32) |> Nx.divide(50)
    x = Nx.tensor([[0.5, -0.3, 0.7]])
    y = Nx.tensor([[1.0, 0.0]])

    grad_fn = fn w1_, b1_, w2_, b2_ ->
      grad({w1_, b1_, w2_, b2_}, fn {a, b, c, d} -> mlp_loss(a, b, c, d, x, y) end)
    end

    {ref_dw1, ref_db1, ref_dw2, ref_db2} = grad_fn.(w1, b1, w2, b2)
    {arm_dw1, arm_db1, arm_dw2, arm_db2} =
      grad_fn.(arm(w1), arm(b1), arm(w2), arm(b2))

    assert diff_max(arm_dw1, ref_dw1) < 1.0e-3
    assert diff_max(arm_db1, ref_db1) < 1.0e-3
    assert diff_max(arm_dw2, ref_dw2) < 1.0e-3
    assert diff_max(arm_db2, ref_db2) < 1.0e-3
  end

  test "grad of binary cross-entropy with sigmoid" do
    w = Nx.tensor([[0.3], [-0.5], [0.7]])
    b = Nx.tensor([0.1])
    x = Nx.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.2]])
    y = Nx.tensor([[1.0], [0.0]])

    grad_fn = fn w_, b_ -> grad({w_, b_}, fn {ww, bb} -> sigmoid_bce(ww, bb, x, y) end) end

    {ref_dw, ref_db} = grad_fn.(w, b)
    {arm_dw, arm_db} = grad_fn.(arm(w), arm(b))

    assert diff_max(arm_dw, ref_dw) < 1.0e-3
    assert diff_max(arm_db, ref_db) < 1.0e-3
  end

  defn mse_loss(w, x, y) do
    pred = Nx.dot(x, w)
    diff = pred - y
    Nx.sum(diff * diff)
  end

  defn sgd_step(w, b, x, y, lr) do
    {loss, {dw, db}} =
      value_and_grad(
        {w, b},
        fn {ww, bb} ->
          pred = Nx.dot(x, ww) + bb
          diff = pred - y
          Nx.sum(diff * diff)
        end
      )

    new_w = w - dw * lr
    new_b = b - db * lr
    {new_w, new_b, loss}
  end

  defn val_grad_mse(w, x, y) do
    value_and_grad(w, fn ww ->
      pred = Nx.dot(x, ww)
      diff = pred - y
      Nx.sum(diff * diff)
    end)
  end

  test "value-and-grad: forward pass + backward in one call" do
    w = Nx.tensor([[0.1, 0.2], [0.3, 0.4]])
    x = Nx.tensor([[1.0, 1.0]])
    y = Nx.tensor([[2.0, 3.0]])

    {ref_val, ref_grad} = val_grad_mse(w, x, y)
    {arm_val, arm_grad} = val_grad_mse(arm(w), arm(x), arm(y))

    assert abs(Nx.to_number(arm_val) - Nx.to_number(ref_val)) < 1.0e-4
    assert diff_max(arm_grad, ref_grad) < 1.0e-4
  end

  test "300-step SGD training loop converges on NxArm.Backend" do
    # Target: y = Wx + b with W=[2,3], b=1. Learn it from data.
    w_true = Nx.tensor([[2.0], [3.0]])
    b_true = Nx.tensor([1.0])

    x_host =
      Nx.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
      ])

    y_host = Nx.add(Nx.dot(x_host, w_true), b_true)

    # Init parameters at zero, on NxArm.
    w0 = arm(Nx.tensor([[0.0], [0.0]]))
    b0 = arm(Nx.tensor([0.0]))
    x = arm(x_host)
    y = arm(y_host)
    lr = arm(Nx.tensor(0.05))

    {w_final, b_final, final_loss} =
      Enum.reduce(1..300, {w0, b0, 1.0e9}, fn _step, {w_, b_, _} ->
        {new_w, new_b, loss} = sgd_step(w_, b_, x, y, lr)
        {new_w, new_b, Nx.to_number(loss)}
      end)

    w_diff = diff_max(w_final, w_true)
    b_diff = diff_max(b_final, b_true)

    assert final_loss < 1.0e-3, "loss didn't go to ~0: #{final_loss}"
    assert w_diff < 0.05, "w didn't converge, diff=#{w_diff}"
    assert b_diff < 0.05, "b didn't converge, diff=#{b_diff}"
  end
end

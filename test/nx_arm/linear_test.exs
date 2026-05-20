defmodule NxArm.LinearTest do
  use ExUnit.Case, async: true

  defp linear(act_t, w_t, bias_t, activation \\ "none") do
    {b, m, k} =
      case Nx.shape(act_t) do
        {b, m, k} -> {b, m, k}
        {m, k} -> {1, m, k}
      end

    {n, ^k} = Nx.shape(w_t)
    act_bin = Nx.to_binary(act_t)
    w_bin = Nx.to_binary(w_t)
    bias_bin = if bias_t, do: Nx.to_binary(bias_t), else: <<>>

    out_bin = NxArm.Native.linear_f32_op(act_bin, w_bin, bias_bin, activation, b, m, n, k)
    {b, m, n, out_bin}
  end

  test "linear no-bias matches dot" do
    a = Nx.iota({4, 8}, type: :f32) |> Nx.divide(10)
    w = Nx.iota({16, 8}, type: :f32) |> Nx.divide(10)

    {_, _, _, out_bin} = linear(a, w, nil)
    got = for <<v::float-little-32 <- out_bin>>, do: v

    # Reference: a @ w^T
    ref = Nx.dot(a, Nx.transpose(w)) |> Nx.to_flat_list()
    Enum.zip(got, ref) |> Enum.each(fn {g, e} -> assert_in_delta g, e, 1.0e-3 end)
  end

  test "linear with bias" do
    a = Nx.tensor([[1.0, 2.0, 3.0]])
    w = Nx.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    bias = Nx.tensor([100.0, 200.0])

    {_, _, _, out_bin} = linear(a, w, bias)
    got = for <<v::float-little-32 <- out_bin>>, do: v

    # a @ w^T = [[6, 12]], +bias = [[106, 212]]
    assert got == [106.0, 212.0]
  end

  test "linear with relu" do
    a = Nx.tensor([[1.0, 2.0]])
    w = Nx.tensor([[1.0, -1.0], [-1.0, 1.0]])
    bias = Nx.tensor([0.0, 0.0])

    {_, _, _, out_bin} = linear(a, w, bias, "relu")
    got = for <<v::float-little-32 <- out_bin>>, do: v

    # a @ w^T = [[1-2, -1+2]] = [[-1, 1]]; relu → [0, 1]
    assert got == [0.0, 1.0]
  end

  test "linear with gelu matches separate ops" do
    a = Nx.tensor([[0.5, -0.5, 1.0]])
    w = Nx.tensor([[1.0, 1.0, 1.0]])
    bias = Nx.tensor([0.0])

    {_, _, _, out_bin} = linear(a, w, bias, "gelu")
    [got] = for <<v::float-little-32 <- out_bin>>, do: v

    # a @ w^T = [[1.0]]; gelu(1.0) = 0.5 * 1 * (1 + erf(1/√2)) ≈ 0.841
    expected = 0.5 * 1.0 * (1.0 + Nx.to_number(Nx.erf(Nx.tensor(1.0 / :math.sqrt(2.0)))))
    assert_in_delta got, expected, 1.0e-4
  end
end

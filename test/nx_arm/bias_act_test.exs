defmodule NxArm.BiasActTest do
  use ExUnit.Case, async: true

  defp call(act, bias, activation, outer, inner) do
    act_bin = for v <- act, into: <<>>, do: <<v::float-little-32>>
    bias_bin = for v <- bias, into: <<>>, do: <<v::float-little-32>>
    out_bin = NxArm.Native.bias_add_activation_f32_op(act_bin, bias_bin, activation, outer, inner)
    for <<v::float-little-32 <- out_bin>>, do: v
  end

  test "none == plain bias add" do
    act = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bias = [10.0, 20.0, 30.0]
    got = call(act, bias, "none", 2, 3)
    assert got == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
  end

  test "relu" do
    act = [-5.0, 0.0, 5.0, -1.0]
    bias = [2.0, 2.0]
    # row 0: [-5+2=-3, 0+2=2] → [0, 2]
    # row 1: [5+2=7, -1+2=1] → [7, 1]
    got = call(act, bias, "relu", 2, 2)
    assert got == [0.0, 2.0, 7.0, 1.0]
  end

  test "relu6 clips at 6" do
    act = [0.0, 5.0, 10.0, 100.0]
    bias = [0.0, 0.0]
    got = call(act, bias, "relu6", 2, 2)
    assert got == [0.0, 5.0, 6.0, 6.0]
  end

  test "sigmoid" do
    act = [0.0, 100.0, -100.0]
    bias = [0.0, 0.0, 0.0]
    got = call(act, bias, "sigmoid", 1, 3)
    assert_in_delta Enum.at(got, 0), 0.5, 1.0e-5
    assert_in_delta Enum.at(got, 1), 1.0, 1.0e-5
    assert_in_delta Enum.at(got, 2), 0.0, 1.0e-5
  end

  test "tanh" do
    act = [0.0, 100.0, -100.0]
    bias = [0.0, 0.0, 0.0]
    got = call(act, bias, "tanh", 1, 3)
    assert_in_delta Enum.at(got, 0), 0.0, 1.0e-5
    assert_in_delta Enum.at(got, 1), 1.0, 1.0e-5
    assert_in_delta Enum.at(got, 2), -1.0, 1.0e-5
  end

  test "gelu matches decomposed" do
    act = [0.0, 1.0, -1.0, 2.0]
    bias = [0.0, 0.0, 0.0, 0.0]
    got = call(act, bias, "gelu", 1, 4)

    # Reference: 0.5*x*(1 + erf(x/√2))
    inv_sqrt2 = 1.0 / :math.sqrt(2.0)
    expected =
      Enum.map(act, fn x ->
        # We need erf — use Nx.
        e = Nx.erf(Nx.tensor(x * inv_sqrt2)) |> Nx.to_number()
        0.5 * x * (1.0 + e)
      end)

    Enum.zip(got, expected)
    |> Enum.each(fn {g, e} -> assert_in_delta g, e, 1.0e-4 end)
  end
end

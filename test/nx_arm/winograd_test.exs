defmodule NxArm.WinogradTest do
  use ExUnit.Case, async: true

  # Reference: Nx.conv with NHWC inputs, NHWC outputs, kernel
  # laid out as {Cout, 3, 3, Cin}. We use Nx default NCHW and
  # transpose for the comparison.
  defp ref_conv(input, weight, bias, padding) do
    # input: {N, H, W, Cin}, weight: {Cout, 3, 3, Cin}
    nchw = Nx.transpose(input, axes: [0, 3, 1, 2])
    w_oihw = Nx.transpose(weight, axes: [0, 3, 1, 2])

    out =
      Nx.conv(nchw, w_oihw,
        strides: [1, 1],
        padding: padding,
        input_permutation: [0, 1, 2, 3],
        kernel_permutation: [0, 1, 2, 3],
        output_permutation: [0, 1, 2, 3]
      )

    out_nhwc = Nx.transpose(out, axes: [0, 2, 3, 1])

    if bias do
      Nx.add(out_nhwc, Nx.reshape(bias, {1, 1, 1, :auto}))
    else
      out_nhwc
    end
  end

  defp call_winograd(input, weight, bias, padding) do
    {n, h_in, w_in, c_in} = Nx.shape(input)
    {c_out, _, _, _} = Nx.shape(weight)
    [{pt, pb}, {pl, pr}] = padding

    inp_bin = Nx.to_binary(input)
    w_bin = Nx.to_binary(weight)
    bias_bin = if bias, do: Nx.to_binary(bias), else: <<>>

    out_bin =
      NxArm.Native.conv2d_f32_winograd_3x3_op(
        inp_bin,
        w_bin,
        bias_bin,
        [n, h_in, w_in, c_in, c_out],
        [pt, pb, pl, pr]
      )

    h_out = h_in + pt + pb - 2
    w_out = w_in + pl + pr - 2
    Nx.from_binary(out_bin, :f32) |> Nx.reshape({n, h_out, w_out, c_out})
  end

  defp small(shape, divisor \\ 100) do
    Nx.iota(shape, type: :f32)
    |> Nx.divide(divisor)
    |> Nx.sin()
  end

  test "valid 3x3 conv matches reference (small)" do
    input = small({1, 4, 4, 2})
    weight = small({3, 3, 3, 2}, 10)
    got = call_winograd(input, weight, nil, [{0, 0}, {0, 0}])
    ref = ref_conv(input, weight, nil, [{0, 0}, {0, 0}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "SAME-padded 3x3 conv matches reference" do
    input = small({1, 8, 8, 3})
    weight = small({4, 3, 3, 3}, 10)
    got = call_winograd(input, weight, nil, [{1, 1}, {1, 1}])
    ref = ref_conv(input, weight, nil, [{1, 1}, {1, 1}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "odd output dimensions work" do
    # H_in=5, pad=1 → H_out=5 (odd). Tile loop must trim.
    input = small({1, 5, 5, 2})
    weight = small({3, 3, 3, 2}, 10)
    got = call_winograd(input, weight, nil, [{1, 1}, {1, 1}])
    ref = ref_conv(input, weight, nil, [{1, 1}, {1, 1}])
    assert Nx.shape(got) == {1, 5, 5, 3}
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "with bias" do
    input = small({1, 4, 4, 2})
    weight = small({3, 3, 3, 2}, 10)
    bias = Nx.tensor([0.1, -0.2, 0.3], type: :f32)
    got = call_winograd(input, weight, bias, [{0, 0}, {0, 0}])
    ref = ref_conv(input, weight, bias, [{0, 0}, {0, 0}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "larger batch + channels (ResNet-ish block)" do
    input = small({2, 14, 14, 8})
    weight = small({16, 3, 3, 8}, 50)
    got = call_winograd(input, weight, nil, [{1, 1}, {1, 1}])
    ref = ref_conv(input, weight, nil, [{1, 1}, {1, 1}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end
end

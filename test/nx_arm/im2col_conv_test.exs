defmodule NxArm.Im2colConvTest do
  use ExUnit.Case, async: true

  defp ref_conv(input, weight, bias, strides, padding) do
    nchw = Nx.transpose(input, axes: [0, 3, 1, 2])
    w_oihw = Nx.transpose(weight, axes: [0, 3, 1, 2])

    out =
      Nx.conv(nchw, w_oihw,
        strides: strides,
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

  defp call_im2col(input, weight, bias, strides, padding) do
    {n, h_in, w_in, c_in} = Nx.shape(input)
    {c_out, kh, kw, _} = Nx.shape(weight)
    [{pt, pb}, {pl, pr}] = padding
    [sh, sw] = strides

    inp_bin = Nx.to_binary(input)
    w_bin = Nx.to_binary(weight)
    bias_bin = if bias, do: Nx.to_binary(bias), else: <<>>

    out_bin =
      NxArm.Native.conv2d_f32_im2col_op(
        inp_bin,
        w_bin,
        bias_bin,
        [n, h_in, w_in, c_in, c_out, kh, kw],
        [sh, sw],
        [pt, pb, pl, pr]
      )

    h_out = div(h_in + pt + pb - kh, sh) + 1
    w_out = div(w_in + pl + pr - kw, sw) + 1
    Nx.from_binary(out_bin, :f32) |> Nx.reshape({n, h_out, w_out, c_out})
  end

  defp small(shape, divisor \\ 100) do
    Nx.iota(shape, type: :f32)
    |> Nx.divide(divisor)
    |> Nx.sin()
  end

  test "1x1 conv (pointwise) matches reference" do
    input = small({1, 8, 8, 4})
    weight = small({6, 1, 1, 4}, 10)
    got = call_im2col(input, weight, nil, [1, 1], [{0, 0}, {0, 0}])
    ref = ref_conv(input, weight, nil, [1, 1], [{0, 0}, {0, 0}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "3x3 stride-1 conv matches reference" do
    input = small({1, 7, 7, 3})
    weight = small({5, 3, 3, 3}, 10)
    got = call_im2col(input, weight, nil, [1, 1], [{1, 1}, {1, 1}])
    ref = ref_conv(input, weight, nil, [1, 1], [{1, 1}, {1, 1}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "5x5 conv with stride 2" do
    input = small({1, 12, 12, 4})
    weight = small({8, 5, 5, 4}, 20)
    got = call_im2col(input, weight, nil, [2, 2], [{2, 2}, {2, 2}])
    ref = ref_conv(input, weight, nil, [2, 2], [{2, 2}, {2, 2}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "with bias" do
    input = small({1, 5, 5, 2})
    weight = small({3, 3, 3, 2}, 10)
    bias = Nx.tensor([0.1, -0.2, 0.3], type: :f32)
    got = call_im2col(input, weight, bias, [1, 1], [{0, 0}, {0, 0}])
    ref = ref_conv(input, weight, bias, [1, 1], [{0, 0}, {0, 0}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "ViT patch16 stem: 16x16 stride 16 (im2col is the canonical path)" do
    input = small({1, 32, 32, 3})
    weight = small({12, 16, 16, 3}, 200)
    got = call_im2col(input, weight, nil, [16, 16], [{0, 0}, {0, 0}])
    ref = ref_conv(input, weight, nil, [16, 16], [{0, 0}, {0, 0}])
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    # gemm uses a different accumulation order than our hand-rolled
    # tile kernel; for big K (K = 16·16·3 = 768) f32 rounding diverges
    # at the 6th decimal. Loosen to match the reference within the
    # tolerance other large-K tests already use.
    assert diff < 1.0e-3, "diff = #{diff}"
  end
end

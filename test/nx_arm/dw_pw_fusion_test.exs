defmodule NxArm.DwPwFusionTest do
  use ExUnit.Case, async: true

  # Reference: depthwise then activation then pointwise, all via Nx.
  defp ref_dw_pw(input, dw_w, dw_b, pw_w, pw_b, strides, padding, activation) do
    # Depthwise: build a (Cin, 1, Kh, Kw) kernel, use feature_group_size = Cin
    {_, _, _, c_in} = Nx.shape(input)
    {_c_in, kh, kw} = Nx.shape(dw_w)

    nchw = Nx.transpose(input, axes: [0, 3, 1, 2])
    dw_kernel = Nx.reshape(dw_w, {c_in, 1, kh, kw})

    dw_out =
      Nx.conv(nchw, dw_kernel,
        strides: strides,
        padding: padding,
        feature_group_size: c_in
      )

    dw_with_bias =
      if dw_b do
        Nx.add(dw_out, Nx.reshape(dw_b, {1, :auto, 1, 1}))
      else
        dw_out
      end

    activated =
      case activation do
        0 -> dw_with_bias
        1 -> Nx.max(dw_with_bias, 0)
        2 -> Nx.min(Nx.max(dw_with_bias, 0), 6)
      end

    {c_out, _} = Nx.shape(pw_w)
    pw_kernel = Nx.reshape(pw_w, {c_out, c_in, 1, 1})

    pw_out =
      Nx.conv(activated, pw_kernel,
        strides: [1, 1],
        padding: [{0, 0}, {0, 0}]
      )

    pw_with_bias =
      if pw_b do
        Nx.add(pw_out, Nx.reshape(pw_b, {1, :auto, 1, 1}))
      else
        pw_out
      end

    Nx.transpose(pw_with_bias, axes: [0, 2, 3, 1])
  end

  defp call_fused(input, dw_w, dw_b, pw_w, pw_b, strides, padding, activation) do
    {n, h_in, w_in, c_in} = Nx.shape(input)
    {c_out, _} = Nx.shape(pw_w)
    {_, kh, kw} = Nx.shape(dw_w)
    [{pt, pb}, {pl, pr}] = padding
    [sh, sw] = strides

    out_bin =
      NxArm.Native.depthwise_pointwise_f32_op(
        Nx.to_binary(input),
        Nx.to_binary(dw_w),
        if(dw_b, do: Nx.to_binary(dw_b), else: <<>>),
        Nx.to_binary(pw_w),
        if(pw_b, do: Nx.to_binary(pw_b), else: <<>>),
        [n, h_in, w_in, c_in, c_out, kh, kw],
        [sh, sw],
        [pt, pb, pl, pr],
        activation
      )

    h_out = div(h_in + pt + pb - kh, sh) + 1
    w_out = div(w_in + pl + pr - kw, sw) + 1
    Nx.from_binary(out_bin, :f32) |> Nx.reshape({n, h_out, w_out, c_out})
  end

  defp small(shape, divisor \\ 100) do
    Nx.iota(shape, type: :f32) |> Nx.divide(divisor) |> Nx.sin()
  end

  test "no activation matches reference" do
    input = small({1, 6, 6, 4})
    dw_w = small({4, 3, 3}, 10)
    pw_w = small({8, 4}, 10)
    got = call_fused(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 0)
    ref = ref_dw_pw(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 0)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "ReLU activation between dw and pw" do
    input = small({1, 5, 5, 3})
    dw_w = small({3, 3, 3}, 10)
    pw_w = small({6, 3}, 10)
    got = call_fused(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 1)
    ref = ref_dw_pw(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 1)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "ReLU6 activation (MobileNet's default)" do
    input = small({1, 7, 7, 4}, 5)
    dw_w = small({4, 3, 3}, 2)
    pw_w = small({8, 4}, 2)
    got = call_fused(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 2)
    ref = ref_dw_pw(input, dw_w, nil, pw_w, nil, [1, 1], [{1, 1}, {1, 1}], 2)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "with both biases" do
    input = small({1, 4, 4, 3})
    dw_w = small({3, 3, 3}, 10)
    pw_w = small({5, 3}, 10)
    dw_b = Nx.tensor([0.1, -0.05, 0.2], type: :f32)
    pw_b = Nx.tensor([0.01, 0.02, 0.03, 0.04, 0.05], type: :f32)
    got = call_fused(input, dw_w, dw_b, pw_w, pw_b, [1, 1], [{1, 1}, {1, 1}], 1)
    ref = ref_dw_pw(input, dw_w, dw_b, pw_w, pw_b, [1, 1], [{1, 1}, {1, 1}], 1)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "diff = #{diff}"
  end

  test "stride 2 (MobileNet downsample block)" do
    input = small({1, 14, 14, 6})
    dw_w = small({6, 3, 3}, 10)
    pw_w = small({12, 6}, 10)
    got = call_fused(input, dw_w, nil, pw_w, nil, [2, 2], [{1, 1}, {1, 1}], 2)
    ref = ref_dw_pw(input, dw_w, nil, pw_w, nil, [2, 2], [{1, 1}, {1, 1}], 2)
    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end
end

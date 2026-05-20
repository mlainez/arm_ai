defmodule NxArm.PerTokenInt8Test do
  use ExUnit.Case, async: true

  test "per-token quantize round-trip" do
    m = 4
    k = 32

    a =
      Nx.iota({m, k}, type: :f32)
      |> Nx.divide(20)
      |> Nx.sin()

    {q_bin, s_bin} = NxArm.Native.quantize_int8_per_token_op(Nx.to_binary(a), m, k)
    assert byte_size(q_bin) == m * k
    assert byte_size(s_bin) == m * 4

    scales = Nx.from_binary(s_bin, :f32) |> Nx.reshape({m})
    q = Nx.from_binary(q_bin, :s8) |> Nx.reshape({m, k})

    deq = Nx.multiply(Nx.as_type(q, :f32), Nx.reshape(scales, {m, 1}))

    # Per-row max error ≤ scale/2.
    diff = Nx.subtract(deq, a) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    max_scale = Nx.reduce_max(scales) |> Nx.to_number()
    assert diff <= max_scale / 2 + 1.0e-6
  end

  test "per-token int8 matmul agrees with f32 matmul on dequantized values" do
    m = 3
    n = 5
    k = 32

    # Distinct per-row magnitudes so per-token vs global scale differ.
    a =
      Nx.tensor(
        for r <- 1..m do
          for j <- 0..(k - 1), do: r * :math.sin(j / 5.0)
        end,
        type: :f32
      )

    w_f32 =
      Nx.iota({n, k}, type: :f32) |> Nx.divide(50) |> Nx.cos()

    # Quantize.
    {a_q_bin, a_s_bin} =
      NxArm.Native.quantize_int8_per_token_op(Nx.to_binary(a), m, k)

    # Weights: simple per-row symmetric int8 quant in Elixir.
    w_max_abs = Nx.reduce_max(Nx.abs(w_f32), axes: [1], keep_axes: true)
    w_scales_tensor = Nx.divide(w_max_abs, 127)
    w_int = Nx.divide(w_f32, w_scales_tensor) |> Nx.round() |> Nx.clip(-127, 127) |> Nx.as_type(:s8)
    w_q_bin = Nx.to_binary(w_int)
    w_s_bin = Nx.to_binary(Nx.reshape(w_scales_tensor, {n}))

    out_bin =
      NxArm.Native.int8_matmul_f32_per_token_op(a_q_bin, w_q_bin, a_s_bin, w_s_bin, m, n, k)

    got = Nx.from_binary(out_bin, :f32) |> Nx.reshape({m, n})

    # Reference: dequantize both sides and run f32 matmul.
    a_scales = Nx.from_binary(a_s_bin, :f32) |> Nx.reshape({m, 1})
    a_int = Nx.from_binary(a_q_bin, :s8) |> Nx.reshape({m, k})
    a_deq = Nx.multiply(Nx.as_type(a_int, :f32), a_scales)
    w_deq = Nx.multiply(Nx.as_type(w_int, :f32), Nx.reshape(w_scales_tensor, {n, 1}))
    ref = Nx.dot(a_deq, [1], w_deq, [1])

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-3, "diff = #{diff}"
  end

  test "per-token scale preserves accuracy on outlier-heavy distributions better than a single global scale" do
    # Row 0 is mostly small values with one large outlier; row 1 is all
    # tiny values. A single global scale would have to be set by the
    # outlier and would crush row 1 to (near) zero.
    m = 2
    k = 32
    row_a = List.duplicate(0.01, k - 1) ++ [50.0]
    row_b = List.duplicate(0.05, k)
    a = Nx.tensor([row_a, row_b], type: :f32)

    w_f32 =
      Nx.iota({4, k}, type: :f32) |> Nx.divide(64) |> Nx.cos()

    {a_q_bin, a_s_bin} = NxArm.Native.quantize_int8_per_token_op(Nx.to_binary(a), m, k)

    w_max_abs = Nx.reduce_max(Nx.abs(w_f32), axes: [1], keep_axes: true)
    w_scales_tensor = Nx.divide(w_max_abs, 127)
    w_int = Nx.divide(w_f32, w_scales_tensor) |> Nx.round() |> Nx.clip(-127, 127) |> Nx.as_type(:s8)
    w_q_bin = Nx.to_binary(w_int)
    w_s_bin = Nx.to_binary(Nx.reshape(w_scales_tensor, {4}))

    out_bin =
      NxArm.Native.int8_matmul_f32_per_token_op(a_q_bin, w_q_bin, a_s_bin, w_s_bin, m, 4, k)

    got = Nx.from_binary(out_bin, :f32) |> Nx.reshape({m, 4})

    ref_fp32 = Nx.dot(a, [1], w_f32, [1])

    diff = Nx.subtract(got, ref_fp32) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    # With per-token scales, row 1 stays near its true magnitude (~0.05 * sum)
    # rather than collapsing to 0. Tolerance: ~0.5 (loose to allow for
    # per-row scale=50/127 quantization of the outlier row).
    assert diff < 0.6, "diff = #{diff}"
  end
end

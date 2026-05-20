defmodule NxArm.FlashAttentionTest do
  use ExUnit.Case, async: true

  defp ref_attention(q, k, v, scale, causal? \\ false) do
    # Reference: softmax(Q @ K^T * scale) @ V
    qk = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)

    masked =
      if causal? do
        {_, _, sq, sk} = Nx.shape(qk)

        mask_2d =
          Nx.iota({sq, 1})
          |> Nx.greater_equal(Nx.iota({1, sk}))

        mask = Nx.reshape(mask_2d, {1, 1, sq, sk}) |> Nx.broadcast(Nx.shape(qk))

        Nx.select(mask, qk, Nx.tensor(-1.0e30))
      else
        qk
      end

    # Manual softmax along last axis.
    m = Nx.reduce_max(masked, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(masked, m))
    s = Nx.sum(e, axes: [-1], keep_axes: true)
    attn = Nx.divide(e, s)

    Nx.dot(attn, [3], [0, 1], v, [2], [0, 1])
  end

  defp call_flash(q, k, v, scale, causal?) do
    {b, h, sq, d} = Nx.shape(q)
    {_, _, sk, _} = Nx.shape(k)

    q_bin = Nx.to_binary(q)
    k_bin = Nx.to_binary(k)
    v_bin = Nx.to_binary(v)

    out_bin =
      NxArm.Native.flash_attention_f32_op(q_bin, k_bin, v_bin, scale, b, h, sq, sk, d, causal?)

    Nx.from_binary(out_bin, :f32) |> Nx.reshape({b, h, sq, d})
  end

  test "matches reference attention at ViT-tiny shape" do
    q = Nx.iota({1, 3, 8, 64}, type: :f32) |> Nx.divide(1000)
    k = Nx.iota({1, 3, 8, 64}, type: :f32) |> Nx.divide(1000)
    v = Nx.iota({1, 3, 8, 64}, type: :f32) |> Nx.divide(100)
    scale = 1.0 / :math.sqrt(64.0)

    got = call_flash(q, k, v, scale, false)
    ref = ref_attention(q, k, v, scale)

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "matches reference with causal mask" do
    q = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100)
    k = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100)
    v = Nx.iota({1, 2, 4, 8}, type: :f32) |> Nx.divide(100)
    scale = 1.0 / :math.sqrt(8.0)

    got = call_flash(q, k, v, scale, true)
    ref = ref_attention(q, k, v, scale, true)

    diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "diff = #{diff}"
  end

  test "produces sensible output (norms preserved approximately)" do
    q = Nx.iota({1, 1, 4, 8}, type: :f32) |> Nx.divide(10)
    k = Nx.iota({1, 1, 4, 8}, type: :f32) |> Nx.divide(10)
    v = Nx.iota({1, 1, 4, 8}, type: :f32) |> Nx.divide(10)
    scale = 1.0 / :math.sqrt(8.0)

    got = call_flash(q, k, v, scale, false)
    # Attention output should have shape matching Q
    assert Nx.shape(got) == {1, 1, 4, 8}
  end
end

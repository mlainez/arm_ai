defmodule NxArm.LLMTest do
  use ExUnit.Case, async: true

  describe "rmsnorm" do
    test "matches decomposed reference" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
      gamma = Nx.tensor([0.5, 1.0, 1.5, 2.0])
      eps = 1.0e-5

      got =
        NxArm.LLM.rmsnorm(x, gamma, eps)
        |> Nx.backend_copy(Nx.BinaryBackend)

      # Reference via primitives.
      rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true), eps))
      ref = Nx.multiply(Nx.divide(x, rms), gamma)

      diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "larger transformer-style shape {1, 16, 768}" do
      x = Nx.iota({1, 16, 768}, type: :f32) |> Nx.divide(100)
      gamma = Nx.iota({768}, type: :f32) |> Nx.divide(100)

      got =
        NxArm.LLM.rmsnorm(x, gamma)
        |> Nx.backend_copy(Nx.BinaryBackend)

      rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true), 1.0e-5))
      ref = Nx.multiply(Nx.divide(x, rms), gamma)

      diff = Nx.subtract(got, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-3, "diff #{diff}"
    end
  end

  describe "rope" do
    test "round-trip at position 0 is identity (cos(0)=1, sin(0)=0)" do
      head_dim = 8
      qk = Nx.iota({1, 1, 2, head_dim}, type: :f32) |> Nx.divide(10)
      inv_freq = NxArm.LLM.rope_inv_freq(head_dim)
      positions = Nx.tensor([0], type: :s64)

      got =
        NxArm.LLM.rope(qk, positions, inv_freq)
        |> Nx.backend_copy(Nx.BinaryBackend)

      # At pos=0, cos(0*freq)=1 and sin(0*freq)=0, so RoPE is identity.
      diff = Nx.subtract(got, qk) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "preserves vector magnitudes (rotation is unitary)" do
      head_dim = 8
      qk = Nx.iota({1, 4, 2, head_dim}, type: :f32) |> Nx.divide(10)
      inv_freq = NxArm.LLM.rope_inv_freq(head_dim)
      positions = Nx.tensor([0, 1, 2, 3], type: :s64)

      got = NxArm.LLM.rope(qk, positions, inv_freq)
      got_b = Nx.backend_copy(got, Nx.BinaryBackend)

      # Per (batch, token, head), the L2 norm of head_dim values should
      # be unchanged.
      orig_norm = Nx.sum(Nx.pow(qk, 2), axes: [-1])
      new_norm = Nx.sum(Nx.pow(got_b, 2), axes: [-1])

      diff = Nx.subtract(orig_norm, new_norm) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-4, "norm preservation: diff #{diff}"
    end

    test "matches a hand-decomposed pair rotation" do
      # head_dim = 2 (single token, single head, position = 1).
      qk = Nx.tensor([[[[3.0, 4.0]]]], type: :f32)
      inv_freq = Nx.tensor([1.0], type: :f32, backend: NxArm.Backend)
      positions = Nx.tensor([1], type: :s64)

      got =
        NxArm.LLM.rope(qk, positions, inv_freq)
        |> Nx.backend_copy(Nx.BinaryBackend)

      # theta = 1 * 1.0 = 1.0; cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
      cos = :math.cos(1.0)
      sin = :math.sin(1.0)
      expected_0 = 3.0 * cos - 4.0 * sin
      expected_1 = 3.0 * sin + 4.0 * cos

      flat = Nx.to_flat_list(got)
      assert_in_delta Enum.at(flat, 0), expected_0, 1.0e-5
      assert_in_delta Enum.at(flat, 1), expected_1, 1.0e-5
    end
  end
end

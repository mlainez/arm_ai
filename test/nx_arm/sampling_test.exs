defmodule NxArm.SamplingTest do
  use ExUnit.Case, async: true

  describe "greedy" do
    test "picks argmax" do
      logits = Nx.tensor([1.0, 5.0, 3.0, 2.0])
      assert NxArm.Sampling.greedy(logits) == 1
    end

    test "argmax on 1-D" do
      logits = Nx.tensor([0.1, 0.2, 0.7])
      assert NxArm.Sampling.greedy(logits) == 2
    end
  end

  describe "sample" do
    test "temperature=0 is greedy" do
      logits = Nx.tensor([1.0, 5.0, 3.0])
      for _ <- 1..50 do
        assert NxArm.Sampling.sample(logits, temperature: 0.0) == 1
      end
    end

    test "always returns an int in [0, vocab)" do
      logits = Nx.tensor([1.0, 5.0, 3.0, 2.0, 8.0])
      for _ <- 1..100 do
        tok = NxArm.Sampling.sample(logits, temperature: 0.8)
        assert tok in 0..4
      end
    end

    test "top_k=1 is identical to greedy regardless of temperature" do
      logits = Nx.tensor([1.0, 5.0, 3.0, 2.0])
      for _ <- 1..50 do
        assert NxArm.Sampling.sample(logits, top_k: 1, temperature: 0.8) == 1
      end
    end

    test "top_p=1.0 is identical to plain temperature sample" do
      logits = Nx.tensor([1.0, 5.0, 3.0, 2.0])
      # Both should produce the same distribution; we can't compare
      # token-for-token (RNG), but we can verify the result is valid.
      for _ <- 1..50 do
        tok = NxArm.Sampling.sample(logits, top_p: 1.0, temperature: 1.0)
        assert tok in 0..3
      end
    end

    test "tight top_p concentrates on top tokens" do
      # With clearly dominant token, top_p=0.5 should heavily favour it.
      logits = Nx.tensor([0.0, 0.0, 100.0, 0.0])
      results = for _ <- 1..50, do: NxArm.Sampling.sample(logits, top_p: 0.5, temperature: 1.0)
      assert Enum.all?(results, &(&1 == 2))
    end
  end

  describe "generate" do
    test "emits a stream that terminates at eos" do
      # Toy model: append the previous token + 1 mod 5; stop at 4.
      step_fn = fn state, last ->
        next_logit_target = rem(last + 1, 5)
        logits =
          0..4
          |> Enum.map(fn i -> if i == next_logit_target, do: 100.0, else: 0.0 end)
          |> Nx.tensor()
        {logits, state}
      end

      tokens =
        NxArm.Sampling.generate(nil, step_fn,
          start_token: 0,
          sampling: [temperature: 0.0]
        )
        |> Stream.take_while(&(&1 != 4))
        |> Enum.to_list()

      assert tokens == [1, 2, 3]
    end
  end
end

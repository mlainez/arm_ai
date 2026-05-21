defmodule NxArm.RepetitionPenaltyTest do
  use ExUnit.Case, async: true

  test "no effect when recent_tokens empty or penalty 1.0" do
    logits = [1.0, 2.0, 3.0, 4.0]
    assert NxArm.Sampling.apply_repetition_penalty(logits, [], 1.5) == [1.0, 2.0, 3.0, 4.0]
    assert NxArm.Sampling.apply_repetition_penalty(logits, [0, 1], 1.0) == [1.0, 2.0, 3.0, 4.0]
  end

  test "positive logits get divided by penalty" do
    logits = [1.0, 2.0, 3.0, 4.0]
    out = NxArm.Sampling.apply_repetition_penalty(logits, [2], 2.0)
    assert out == [1.0, 2.0, 1.5, 4.0]
  end

  test "negative logits get multiplied (pushed further down)" do
    logits = [-1.0, -2.0, 5.0]
    out = NxArm.Sampling.apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out == [-2.0, -4.0, 5.0]
  end

  test "sample with repetition_penalty + greedy steers around recent tokens" do
    # Without penalty, token 2 wins greedy (logit 10).
    # With penalty=10 and 2 in recent_tokens, its effective logit
    # becomes 1.0, so token 1 (logit 5) wins.
    logits = Nx.tensor([1.0, 5.0, 10.0, 0.0])

    assert NxArm.Sampling.sample(logits, temperature: 0.0) == 2

    assert NxArm.Sampling.sample(logits,
             temperature: 0.0,
             repetition_penalty: 10.0,
             recent_tokens: [2]
           ) == 1
  end
end

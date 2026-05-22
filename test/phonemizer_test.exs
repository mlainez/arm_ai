defmodule ArmAI.PhonemizerTest do
  use ExUnit.Case, async: true

  describe "phonemize/2" do
    test "delegates to the supplied callback" do
      cb = fn text -> {:called_with, text} end
      assert ArmAI.Phonemizer.phonemize("hi there", cb) == {:called_with, "hi there"}
    end
  end

  describe "to_phoneme_ids/2" do
    test "looks up scalar-int mapping" do
      map = %{"HH" => 4, "AH" => 11, "L" => 2, "OW" => 7}
      assert ArmAI.Phonemizer.to_phoneme_ids(["HH", "AH", "L", "OW"], map) == [4, 11, 2, 7]
    end

    test "looks up list-of-int mapping (Piper convention) and takes the first" do
      map = %{"HH" => [4], "AH" => [11, 99], "L" => [2], "OW" => [7]}
      assert ArmAI.Phonemizer.to_phoneme_ids(["HH", "AH", "L", "OW"], map) == [4, 11, 2, 7]
    end

    test "drops symbols missing from the map" do
      map = %{"HH" => 4, "OW" => 7}
      assert ArmAI.Phonemizer.to_phoneme_ids(["HH", "ZZ", "OW", "WAT"], map) == [4, 7]
    end

    test "empty input → empty output" do
      assert ArmAI.Phonemizer.to_phoneme_ids([], %{"A" => 1}) == []
    end
  end

  describe "simple_english_phonemize/1" do
    test "looks up common dictionary words" do
      # "hello world" — both in the bundled mini-dict
      assert ArmAI.Phonemizer.simple_english_phonemize("hello world") ==
               ["HH", "AH", "L", "OW", "W", "ER", "L", "D"]
    end

    test "is case-insensitive and ignores punctuation" do
      assert ArmAI.Phonemizer.simple_english_phonemize("Hello, World!") ==
               ["HH", "AH", "L", "OW", "W", "ER", "L", "D"]
    end

    test "OOV words fall back to letter-by-letter phonemes" do
      # "xyz" is not in the dict — every letter falls back. We don't
      # assert the exact phoneme list (the letter table is allowed
      # to evolve) but we expect a non-empty result with the right
      # rough cardinality.
      out = ArmAI.Phonemizer.simple_english_phonemize("xyz")
      assert is_list(out)
      assert length(out) >= 3
      assert Enum.all?(out, &is_binary/1)
    end

    test "mixes dictionary hits with OOV fallback" do
      # "hello qwerty" — first word hits the dict, second falls back.
      out = ArmAI.Phonemizer.simple_english_phonemize("hello qwerty")
      # Hello's phonemes must appear at the start.
      assert Enum.take(out, 4) == ["HH", "AH", "L", "OW"]
      # Something must follow for "qwerty".
      assert length(out) > 4
    end

    test "empty / punctuation-only input → []" do
      assert ArmAI.Phonemizer.simple_english_phonemize("") == []
      assert ArmAI.Phonemizer.simple_english_phonemize("...!?") == []
    end
  end

  describe "phonemize → to_phoneme_ids pipeline" do
    test "complete demo path produces a token sequence" do
      symbols = ArmAI.Phonemizer.simple_english_phonemize("hello world")
      # Build a stub ID map from the unique symbols.
      id_map = symbols |> Enum.uniq() |> Enum.with_index() |> Enum.into(%{})
      ids = ArmAI.Phonemizer.to_phoneme_ids(symbols, id_map)

      assert length(ids) == length(symbols)
      assert Enum.all?(ids, &is_integer/1)
    end
  end
end

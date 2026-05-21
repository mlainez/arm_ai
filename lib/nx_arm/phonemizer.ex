defmodule NxArm.Phonemizer do
  @moduledoc """
  Text → phoneme conversion for TTS pipelines.

  This is the gap between `NxArm.Models.Piper.synthesize/3` (which
  takes phoneme IDs) and a string prompt from the user.

  ## Reality check

  Production-grade G2P for English typically uses eSpeak NG (a C
  library) or a learned seq2seq model. eSpeak NG isn't friendly to
  Nerves cross-compiles; a learned model adds another model file.
  This module provides:

    * `phonemize/2` — pluggable interface taking a `:fn` callback so
      callers can wire any phonemizer they like (host-side service,
      a candle G2P model, eSpeak NG via NIF if they ship the C dep).
    * `to_phoneme_ids/2` — given a phoneme symbol → ID map (the
      `phoneme_id_map` from a Piper voice's `.onnx.json`), turn a
      list of phoneme symbols into the IDs the model expects.
    * `simple_english_phonemize/1` — a 200-line "good enough for
      demos" English phonemizer using a small dictionary + a naive
      letter-to-sound fallback. **Not production quality**;
      pronounces most common words correctly, butchers proper
      nouns and complex words.

  ## Two recommended production paths

  ### A) Phonemize off-device (recommended)

  Most Nerves voice-assistant deployments have a cloud companion.
  Compute phonemes there with the proper eSpeak NG / G2P model
  and ship phoneme IDs to the device. The device runs only the
  ONNX synthesis step.

  ### B) Bring your own NIF

  If you must phonemize on-device, vendor an eSpeak NG NIF or a
  small G2P model into your app. Pass its output to `to_phoneme_ids/2`
  here.

  ## API

      # Plug in your own phonemizer:
      phoneme_strs = NxArm.Phonemizer.phonemize("Hello world",
                       fn text -> MyApp.G2P.run(text) end)

      # Map symbols to IDs using the voice's phoneme_id_map:
      ids = NxArm.Phonemizer.to_phoneme_ids(phoneme_strs, phoneme_id_map)

      # Synthesize:
      samples = NxArm.Models.Piper.synthesize(piper, ids)

      # The demo English phonemizer (low fidelity):
      phoneme_strs = NxArm.Phonemizer.simple_english_phonemize("hello world")
  """

  @doc """
  Phonemize text via a user-provided callback. Returns the
  callback's result. Centralised entry point so users can swap
  phonemizers without changing the surrounding pipeline.
  """
  @spec phonemize(String.t(), (String.t() -> any())) :: any()
  def phonemize(text, fun) when is_function(fun, 1) do
    fun.(text)
  end

  @doc """
  Translate a list of phoneme symbols (e.g. `["HH", "AH", "L", "OW"]`)
  to integer IDs using the voice-specific lookup table.

  `phoneme_id_map` follows the Piper convention: each phoneme symbol
  maps to a list of IDs (multiple IDs per symbol cover variants like
  silence padding). We take the first ID for each.
  """
  @spec to_phoneme_ids([String.t()], %{String.t() => [non_neg_integer()] | non_neg_integer()}) ::
          [non_neg_integer()]
  def to_phoneme_ids(symbols, phoneme_id_map) do
    Enum.flat_map(symbols, fn sym ->
      case Map.get(phoneme_id_map, sym) do
        nil -> []
        [id | _] -> [id]
        id when is_integer(id) -> [id]
      end
    end)
  end

  @doc """
  Best-effort English G2P. Splits on whitespace + punctuation,
  looks each word up in a tiny built-in dictionary, falls back to
  per-letter mapping for OOV. Outputs ARPAbet symbols (the set used
  by CMU and most Piper English voices).

  **Use this for demos, smoke tests, and proofs of concept only.**
  Production TTS should use a real phonemizer.
  """
  @spec simple_english_phonemize(String.t()) :: [String.t()]
  def simple_english_phonemize(text) do
    text
    |> String.downcase()
    |> String.split(~r/[^a-z']+/, trim: true)
    |> Enum.flat_map(&word_to_phonemes/1)
  end

  # Very small CMU-style dictionary subset for the most common words.
  # Real eSpeak handles ~135k entries; this gets 100% on toy demos.
  @minidict %{
    "the" => ["DH", "AH"],
    "and" => ["AE", "N", "D"],
    "is" => ["IH", "Z"],
    "a" => ["AH"],
    "an" => ["AE", "N"],
    "i" => ["AY"],
    "you" => ["Y", "UW"],
    "of" => ["AH", "V"],
    "to" => ["T", "UW"],
    "in" => ["IH", "N"],
    "that" => ["DH", "AE", "T"],
    "it" => ["IH", "T"],
    "for" => ["F", "AO", "R"],
    "was" => ["W", "AH", "Z"],
    "with" => ["W", "IH", "DH"],
    "as" => ["AE", "Z"],
    "be" => ["B", "IY"],
    "are" => ["AA", "R"],
    "this" => ["DH", "IH", "S"],
    "have" => ["HH", "AE", "V"],
    "from" => ["F", "R", "AH", "M"],
    "or" => ["AO", "R"],
    "had" => ["HH", "AE", "D"],
    "by" => ["B", "AY"],
    "hot" => ["HH", "AA", "T"],
    "word" => ["W", "ER", "D"],
    "but" => ["B", "AH", "T"],
    "what" => ["W", "AH", "T"],
    "some" => ["S", "AH", "M"],
    "we" => ["W", "IY"],
    "can" => ["K", "AE", "N"],
    "out" => ["AW", "T"],
    "other" => ["AH", "DH", "ER"],
    "were" => ["W", "ER"],
    "all" => ["AO", "L"],
    "there" => ["DH", "EH", "R"],
    "when" => ["W", "EH", "N"],
    "up" => ["AH", "P"],
    "use" => ["Y", "UW", "Z"],
    "your" => ["Y", "AO", "R"],
    "how" => ["HH", "AW"],
    "said" => ["S", "EH", "D"],
    "each" => ["IY", "CH"],
    "she" => ["SH", "IY"],
    "do" => ["D", "UW"],
    "their" => ["DH", "EH", "R"],
    "time" => ["T", "AY", "M"],
    "if" => ["IH", "F"],
    "will" => ["W", "IH", "L"],
    "way" => ["W", "EY"],
    "about" => ["AH", "B", "AW", "T"],
    "many" => ["M", "EH", "N", "IY"],
    "then" => ["DH", "EH", "N"],
    "them" => ["DH", "EH", "M"],
    "would" => ["W", "UH", "D"],
    "like" => ["L", "AY", "K"],
    "so" => ["S", "OW"],
    "these" => ["DH", "IY", "Z"],
    "her" => ["HH", "ER"],
    "long" => ["L", "AO", "NG"],
    "make" => ["M", "EY", "K"],
    "thing" => ["TH", "IH", "NG"],
    "see" => ["S", "IY"],
    "him" => ["HH", "IH", "M"],
    "two" => ["T", "UW"],
    "has" => ["HH", "AE", "Z"],
    "look" => ["L", "UH", "K"],
    "more" => ["M", "AO", "R"],
    "day" => ["D", "EY"],
    "could" => ["K", "UH", "D"],
    "go" => ["G", "OW"],
    "come" => ["K", "AH", "M"],
    "did" => ["D", "IH", "D"],
    "no" => ["N", "OW"],
    "most" => ["M", "OW", "S", "T"],
    "my" => ["M", "AY"],
    "over" => ["OW", "V", "ER"],
    "know" => ["N", "OW"],
    "than" => ["DH", "AE", "N"],
    "first" => ["F", "ER", "S", "T"],
    "been" => ["B", "IH", "N"],
    "call" => ["K", "AO", "L"],
    "who" => ["HH", "UW"],
    "its" => ["IH", "T", "S"],
    "now" => ["N", "AW"],
    "find" => ["F", "AY", "N", "D"],
    "down" => ["D", "AW", "N"],
    "people" => ["P", "IY", "P", "AH", "L"],
    "hello" => ["HH", "AH", "L", "OW"],
    "world" => ["W", "ER", "L", "D"],
    "yes" => ["Y", "EH", "S"]
  }

  defp word_to_phonemes(word) do
    case Map.fetch(@minidict, word) do
      {:ok, phs} -> phs
      :error -> naive_letter_phonemes(word)
    end
  end

  # Letter-by-letter fallback. Horrendous for real English but
  # better than zero.
  @letters %{
    "a" => ["AH"], "b" => ["B"], "c" => ["K"], "d" => ["D"], "e" => ["EH"],
    "f" => ["F"], "g" => ["G"], "h" => ["HH"], "i" => ["IH"], "j" => ["JH"],
    "k" => ["K"], "l" => ["L"], "m" => ["M"], "n" => ["N"], "o" => ["OW"],
    "p" => ["P"], "q" => ["K", "W"], "r" => ["R"], "s" => ["S"], "t" => ["T"],
    "u" => ["AH"], "v" => ["V"], "w" => ["W"], "x" => ["K", "S"], "y" => ["Y"],
    "z" => ["Z"]
  }

  defp naive_letter_phonemes(word) do
    word
    |> String.graphemes()
    |> Enum.flat_map(fn ch -> Map.get(@letters, ch, []) end)
  end
end

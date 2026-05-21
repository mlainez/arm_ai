defmodule NxArm.Tokenizer do
  @moduledoc """
  Thin wrapper over the HuggingFace `tokenizers` crate.

      {:ok, tok} = NxArm.Tokenizer.load("/root/tinyllama-tokenizer.json")
      ids = NxArm.Tokenizer.encode(tok, "Hello, world", add_special_tokens: true)
      "Hello, world" = NxArm.Tokenizer.decode(tok, ids)

  Supports BPE, WordPiece, SentencePiece — every tokenizer format
  HuggingFace serializes to `tokenizer.json`. Use this with any
  GGUF model that ships a matching tokenizer.json (Llama, Mistral,
  Phi, Qwen, Gemma, etc).
  """

  defstruct [:handle]

  @doc "Load a `tokenizer.json` file."
  @spec load(Path.t()) :: {:ok, %__MODULE__{}} | {:error, term()}
  def load(path) do
    try do
      handle = NxArm.Native.tokenizer_load_op(path)
      {:ok, %__MODULE__{handle: handle}}
    rescue
      e -> {:error, e}
    catch
      :error, reason -> {:error, reason}
    end
  end

  @doc """
  Encode a UTF-8 string to a list of token IDs.

  Options:
    * `:add_special_tokens` — include BOS/EOS markers per the tokenizer
      config (default `true`).
  """
  @spec encode(%__MODULE__{}, String.t(), Keyword.t()) :: [non_neg_integer()]
  def encode(%__MODULE__{handle: handle}, text, opts \\ []) do
    add_special = Keyword.get(opts, :add_special_tokens, true)
    NxArm.Native.tokenizer_encode_op(handle, text, add_special)
  end

  @doc """
  Decode a list of token IDs back to a UTF-8 string.

  Options:
    * `:skip_special_tokens` — drop BOS/EOS/pad in the output
      (default `true`).
  """
  @spec decode(%__MODULE__{}, [non_neg_integer()], Keyword.t()) :: String.t()
  def decode(%__MODULE__{handle: handle}, ids, opts \\ []) do
    skip_special = Keyword.get(opts, :skip_special_tokens, true)
    NxArm.Native.tokenizer_decode_op(handle, ids, skip_special)
  end
end

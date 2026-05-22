#!/usr/bin/env elixir

# ---------------------------------------------------------------
# Example 09 — Full voice-assistant pipeline.
#
#   audio in  →  VAD  →  Whisper  →  LLM  →  Piper  →  audio out
#
# Demonstrates that the four model bridges chain cleanly without
# leaving the device. Each stage uses the same model bridges as
# examples 01, 02, 05, 06.
# ---------------------------------------------------------------

audio_in  = "/root/question.wav"
audio_out = "/root/answer.wav"

# Model paths (ArmAI.Hub installs these on first boot — see config.exs).
vad_path     = "/root/models/silero_vad.onnx"
whisper_gguf = "/root/models/whisper-tiny.gguf"
whisper_tok  = "/root/models/whisper-tokenizer.json"
whisper_mel  = "/root/models/whisper-mel-filters.bin"
llama_gguf   = "/root/models/tinyllama.gguf"
piper_onnx   = "/root/models/piper-amy-medium.onnx"

paths = [vad_path, whisper_gguf, whisper_tok, whisper_mel, llama_gguf, piper_onnx, audio_in]
missing = Enum.reject(paths, &File.exists?/1)

unless missing == [] do
  IO.puts("Missing files:\n  - " <> Enum.join(missing, "\n  - "))
  System.halt(1)
end

IO.puts("== 1. VAD ==")
{:ok, vad} = ArmAI.SileroVAD.load(vad_path)
pcm = ArmAI.Audio.load_for_whisper(audio_in)
segs = ArmAI.SileroVAD.detect(vad, pcm, threshold: 0.5)
IO.puts("  → #{length(segs)} speech segments")

# Concat speech-only PCM (skip silence — saves Whisper time).
speech_pcm =
  segs
  |> Enum.map(fn %{start_ms: s, end_ms: e} ->
    si = trunc(s * 16)
    ei = trunc(e * 16)
    Nx.slice(pcm, [si], [ei - si])
  end)
  |> case do
    [] -> pcm
    [single] -> single
    many -> Nx.concatenate(many)
  end

IO.puts("== 2. Whisper ==")
{:ok, whisper} = ArmAI.WhisperCandle.load(
  gguf_path: whisper_gguf,
  tokenizer_path: whisper_tok,
  mel_filters_path: whisper_mel
)
{:ok, transcript} = ArmAI.WhisperCandle.transcribe(whisper, speech_pcm)
IO.puts("  → \"#{transcript}\"")

IO.puts("== 3. LLM ==")
{:ok, llm} = ArmAI.LlamaCandle.load(llama_gguf)
prompt = "Q: #{transcript}\nA:"
{:ok, tokenizer} = Tokenizers.Tokenizer.from_file("/root/models/tinyllama-tokenizer.json")
{:ok, prompt_enc} = Tokenizers.Tokenizer.encode(tokenizer, prompt)
prompt_tokens = Tokenizers.Encoding.get_ids(prompt_enc)
{response_tokens, _stats} =
  ArmAI.LlamaCandle.generate(llm,
    prompt_tokens: prompt_tokens,
    max_new: 48,
    stop_tokens: [2]
  )
{:ok, answer} = Tokenizers.Tokenizer.decode(tokenizer, response_tokens)
IO.puts("  → \"#{answer}\"")

IO.puts("== 4. Piper ==")
{:ok, piper} = ArmAI.Piper.load(piper_onnx, sample_rate: 22050)
phonemes = ArmAI.Phonemizer.simple_english_phonemize(answer)
phoneme_id_map = phonemes |> Enum.uniq() |> Enum.with_index() |> Enum.into(%{})
ids = ArmAI.Phonemizer.to_phoneme_ids(phonemes, phoneme_id_map)
samples = ArmAI.Piper.synthesize(piper, ids)
:ok = ArmAI.Audio.write_wav(audio_out, samples, sample_rate: 22_050)

IO.puts("Done. Wrote #{audio_out} (#{Float.round(Nx.size(samples) / 22050, 2)} s)")

//! Whisper speech-to-text via candle-transformers.
//!
//! Loads a safetensors or candle-quantized GGUF Whisper checkpoint
//! (whisper-tiny.en is the Nerves-sized choice) and runs 16 kHz mono
//! f32 audio through mel → encode → greedy decode → detokenize, one
//! 30 s window at a time.
//!
//! `InferAudio.Decoder.load_for_whisper/1` produces the input.

use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{
    self as m,
    audio::pcm_to_mel,
    Config,
};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub struct WhisperResource {
    pub config: Config,
    pub model: Mutex<Variant>,
    pub tokenizer: Mutex<Tokenizer>,
    pub mel_filters: Vec<f32>,
    pub sot: u32,
    pub eot: u32,
    pub no_timestamps: u32,
    pub transcribe: u32,
    // Reserved for the "no-speech" probability threshold path —
    // populated at load time but not consumed yet. Prefix with `_`
    // to silence the dead-field warning without changing the wire
    // format of the resource.
    pub _no_speech: u32,
}

pub enum Variant {
    F32(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Variant {
    fn reset_kv_cache(&mut self) {
        match self {
            Variant::F32(w) => w.reset_kv_cache(),
            Variant::Quantized(w) => w.reset_kv_cache(),
        }
    }

    fn encoder_forward(&mut self, mel: &Tensor, flush: bool) -> Result<Tensor, String> {
        match self {
            Variant::F32(w) => w
                .encoder
                .forward(mel, flush)
                .map_err(|e| format!("encoder: {}", e)),
            Variant::Quantized(w) => w
                .encoder
                .forward(mel, flush)
                .map_err(|e| format!("encoder: {}", e)),
        }
    }

    fn decoder_forward(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
        flush: bool,
    ) -> Result<Tensor, String> {
        match self {
            Variant::F32(w) => w
                .decoder
                .forward(tokens, audio_features, flush)
                .map_err(|e| format!("decoder: {}", e)),
            Variant::Quantized(w) => w
                .decoder
                .forward(tokens, audio_features, flush)
                .map_err(|e| format!("decoder: {}", e)),
        }
    }

    fn decoder_final_linear(&self, x: &Tensor) -> Result<Tensor, String> {
        match self {
            Variant::F32(w) => w
                .decoder
                .final_linear(x)
                .map_err(|e| format!("final_linear: {}", e)),
            Variant::Quantized(w) => w
                .decoder
                .final_linear(x)
                .map_err(|e| format!("final_linear: {}", e)),
        }
    }
}

unsafe impl Send for WhisperResource {}
unsafe impl Sync for WhisperResource {}

/// Load Whisper from either:
///   * a `.gguf` file (quantised path, the smallest deployment)
///   * a `.safetensors` file + accompanying `config.json` (f32 path)
///
/// `tokenizer_json_path` points to the matching `tokenizer.json`,
/// `mel_filters_path` points to the precomputed mel filter banks
/// (whisper-tiny ships these as `mel_filters.safetensors` or
/// `melfilters.bytes`).
pub fn load(
    model_path: &str,
    tokenizer_json_path: &str,
    mel_filters_path: &str,
    config_path: &str,
) -> Result<WhisperResource, String> {
    let device = Device::Cpu;

    let config_bytes = std::fs::read(config_path)
        .map_err(|e| format!("read config {}: {}", config_path, e))?;
    let config: Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| format!("config parse: {}", e))?;

    let model_ext = Path::new(model_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let model = match model_ext {
        "gguf" => {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                model_path, &device,
            )
            .map_err(|e| format!("gguf VarBuilder: {}", e))?;
            let w = m::quantized_model::Whisper::load(&vb, config.clone())
                .map_err(|e| format!("Whisper::load (quantized): {}", e))?;
            Variant::Quantized(w)
        }
        _ => {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[model_path],
                    candle_core::DType::F32,
                    &device,
                )
                .map_err(|e| format!("safetensors VarBuilder: {}", e))?
            };
            let w = m::model::Whisper::load(&vb, config.clone())
                .map_err(|e| format!("Whisper::load (f32): {}", e))?;
            Variant::F32(w)
        }
    };

    let tokenizer = Tokenizer::from_file(tokenizer_json_path)
        .map_err(|e| format!("Tokenizer::from_file({}): {}", tokenizer_json_path, e))?;

    // Mel filters: shipped alongside whisper checkpoints as raw f32
    // bytes (`mel_filters_path` → 80 mel bins × 201 fft bins by default).
    let mel_bytes =
        std::fs::read(mel_filters_path).map_err(|e| format!("read mel filters: {}", e))?;
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    use byteorder::{ByteOrder, LittleEndian};
    LittleEndian::read_f32_into(&mel_bytes, &mut mel_filters);

    let sot = token_id(&tokenizer, m::SOT_TOKEN)?;
    let eot = token_id(&tokenizer, m::EOT_TOKEN)?;
    let no_timestamps = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
    let transcribe = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
    let no_speech = m::NO_SPEECH_TOKENS
        .iter()
        .find_map(|t| token_id(&tokenizer, t).ok())
        .unwrap_or(u32::MAX);

    Ok(WhisperResource {
        config,
        model: Mutex::new(model),
        tokenizer: Mutex::new(tokenizer),
        mel_filters,
        sot,
        eot,
        no_timestamps,
        transcribe,
        _no_speech: no_speech,
    })
}

fn token_id(tok: &Tokenizer, sym: &str) -> Result<u32, String> {
    tok.token_to_id(sym)
        .ok_or_else(|| format!("tokenizer missing symbol: {}", sym))
}

/// Transcribe an arbitrarily-long 16 kHz mono f32 buffer by
/// sliding a 30-second window. Each window contributes a chunk of
/// text; chunks are joined with a single space. For overlapping
/// windows we suppress the leading EOT / SOT special tokens that
/// would otherwise repeat.
pub fn transcribe(res: &WhisperResource, pcm: &[f32]) -> Result<String, String> {
    const STEP: usize = m::N_SAMPLES; // 30 s, no overlap
    if pcm.is_empty() {
        return Ok(String::new());
    }

    let mut out = String::new();
    let mut start = 0usize;

    while start < pcm.len() {
        let end = (start + STEP).min(pcm.len());
        let chunk_text = transcribe_chunk(res, &pcm[start..end])?;

        if !chunk_text.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(chunk_text.trim());
        }

        start += STEP;
    }

    Ok(out)
}

fn transcribe_chunk(res: &WhisperResource, pcm: &[f32]) -> Result<String, String> {
    let device = Device::Cpu;

    // Pad/truncate to 30 s.
    let mut window = pcm.to_vec();
    window.resize(m::N_SAMPLES, 0.0);

    // 1. Mel spectrogram.
    let mel = pcm_to_mel(&res.config, &window, &res.mel_filters);
    let mel_len = mel.len();
    let mel_tensor = Tensor::from_vec(mel, (1, res.config.num_mel_bins, mel_len / res.config.num_mel_bins), &device)
        .map_err(|e| format!("mel tensor: {}", e))?;
    // candle's pcm_to_mel appends its own padding; the encoder takes
    // exactly one 30 s window (N_FRAMES = 3000 frames).
    let mel_tensor = mel_tensor
        .narrow(2, 0, m::N_FRAMES)
        .map_err(|e| format!("mel narrow: {}", e))?;

    let mut model = res.model.lock().map_err(|e| format!("lock model: {}", e))?;
    model.reset_kv_cache();

    // 2. Encoder.
    let audio_features = model.encoder_forward(&mel_tensor, true)?;

    // 3. Greedy decoder loop.
    let mut tokens: Vec<u32> = vec![res.sot, res.transcribe, res.no_timestamps];
    let max_new = res.config.max_target_positions.min(224);

    // candle's whisper decoder has no self-attention KV cache: every
    // step must see the whole token sequence (SOT prefix + history).
    // `flush` resets the cross-attention cache on the first step only.
    for step in 0..max_new {
        let tokens_tensor =
            Tensor::new(tokens.as_slice(), &device).map_err(|e| format!("tokens tensor: {}", e))?;
        let tokens_tensor = tokens_tensor
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze: {}", e))?;

        let dec = model.decoder_forward(&tokens_tensor, &audio_features, step == 0)?;
        let seq_len = tokens.len();
        let dec = dec
            .i((..1, seq_len - 1..))
            .map_err(|e| format!("slice last position: {}", e))?;
        let logits = model.decoder_final_linear(&dec)?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| format!("logits squeeze: {}", e))?;

        let logits = logits
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| format!("logits dtype: {}", e))?;
        let argmax = logits
            .argmax(logits.rank() - 1)
            .map_err(|e| format!("argmax: {}", e))?;
        let last = argmax
            .to_vec1::<u32>()
            .map_err(|e| format!("argmax to_vec1: {}", e))?;
        let next = *last.last().unwrap_or(&res.eot);
        tokens.push(next);

        if next == res.eot {
            break;
        }
    }

    let tokenizer = res.tokenizer.lock().map_err(|e| format!("lock tok: {}", e))?;
    let text = tokenizer
        .decode(&tokens, true)
        .map_err(|e| format!("decode: {}", e))?;
    Ok(text)
}

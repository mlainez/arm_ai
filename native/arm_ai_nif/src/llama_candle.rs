//! Llama / TinyLlama / SmolLM forward pass via `candle`.
//!
//! Per the upstream-crates policy: candle ships hand-tuned ARM NEON
//! kernels for Q4_0/Q4_K/Q8_0 matmul, a proven Llama implementation
//! with KV cache + RoPE + GQA, and a GGUF reader. We bridge to it
//! rather than maintain a parallel hand-rolled stack.
//!
//! The Elixir side calls:
//!   * `llama_candle_load_op(path)` → opens the GGUF, builds a
//!     `ResourceArc<LlamaModelResource>` holding the loaded model.
//!   * `llama_candle_generate_op(model, prompt, max_new)` → greedy
//!     decode, returns the generated token IDs.
//!
//! Compared to the hand-rolled `ArmAI.Llama`:
//!   * Q4_0 matmul uses candle's NEON kernel (which has been
//!     profiled and tuned on a wider set of ARM cores than we have
//!     access to).
//!   * KV cache is candle-internal, no per-step put_slice copies.
//!   * The whole decoder layer runs in one Rust function — saves
//!     hundreds of BEAM↔NIF round-trips per token.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

/// Wraps a loaded Llama model + its KV cache state. The Mutex
/// serialises calls so concurrent inference is safe (callers
/// typically have one per process).
pub struct LlamaResource {
    pub model: Mutex<ModelWeights>,
}

unsafe impl Send for LlamaResource {}
unsafe impl Sync for LlamaResource {}

/// Load a GGUF Llama-family model.
pub fn load_model(path: &str) -> Result<LlamaResource, String> {
    let device = Device::Cpu;
    let mut file =
        File::open(Path::new(path)).map_err(|e| format!("open {} failed: {}", path, e))?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|e| format!("gguf read failed: {}", e))?;

    let model = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|e| format!("ModelWeights::from_gguf failed: {}", e))?;

    Ok(LlamaResource {
        model: Mutex::new(model),
    })
}

/// One forward step: takes `prompt` token IDs, runs the model up to
/// `max_new` greedy decode steps. Returns the new tokens (only the
/// generated ones, not the prompt).
///
/// Timing instrumentation: the caller receives prefill + decode
/// timings via the return tuple so the Elixir side can report
/// tokens/sec.
pub fn generate_greedy(
    res: &LlamaResource,
    prompt: &[u32],
    max_new: usize,
) -> Result<GenerateResult, String> {
    let mut model = res.model.lock().map_err(|e| format!("lock: {}", e))?;
    let device = Device::Cpu;

    let prefill_start = std::time::Instant::now();

    // Prefill: feed the whole prompt at offset 0, take the logits
    // of the final position to predict the first new token.
    let prompt_tensor = Tensor::new(prompt, &device)
        .map_err(|e| format!("Tensor::new prompt: {}", e))?
        .unsqueeze(0)
        .map_err(|e| format!("unsqueeze: {}", e))?;

    let logits = model
        .forward(&prompt_tensor, 0)
        .map_err(|e| format!("forward prefill: {}", e))?;

    let prefill_us = prefill_start.elapsed().as_micros() as u64;

    let last_logits = logits
        .squeeze(0)
        .map_err(|e| format!("squeeze prefill logits: {}", e))?;

    let mut next = argmax_u32(&last_logits)?;
    let mut generated = vec![next];

    let decode_start = std::time::Instant::now();

    let mut offset = prompt.len();
    for _ in 1..max_new {
        let step_tensor = Tensor::new(&[next], &device)
            .map_err(|e| format!("Tensor::new step: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze step: {}", e))?;

        let step_logits = model
            .forward(&step_tensor, offset)
            .map_err(|e| format!("forward step: {}", e))?
            .squeeze(0)
            .map_err(|e| format!("squeeze step: {}", e))?;

        next = argmax_u32(&step_logits)?;
        generated.push(next);
        offset += 1;
    }

    let decode_us = decode_start.elapsed().as_micros() as u64;

    Ok(GenerateResult {
        tokens: generated,
        prefill_us,
        decode_us,
    })
}

pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub prefill_us: u64,
    pub decode_us: u64,
}

fn argmax_u32(t: &Tensor) -> Result<u32, String> {
    // `quantized_llama` returns f32 logits, so `to_dtype` is a cheap
    // Arc-clone (candle returns Ok(self.clone()) when dtype matches).
    // We then pull a contiguous f32 slice once and scan it ourselves —
    // candle's own `argmax` allocates a fresh 1-element index tensor
    // per call which is wasted at decode (we call it once per token).
    let logits = t
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| format!("to_dtype: {}", e))?;

    // Flatten to 1-D so to_vec1 gives us the whole vocab.
    let flat = logits
        .flatten_all()
        .map_err(|e| format!("flatten: {}", e))?;

    let v: Vec<f32> = flat
        .to_vec1::<f32>()
        .map_err(|e| format!("to_vec1: {}", e))?;

    let (best_idx, _best_val) = v
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        });

    Ok(best_idx as u32)
}

//! ONNX inference via the upstream `tract-onnx` crate.
//!
//! Bumblebee + Axon cover BERT/GPT-2/ViT directly via Nx. For
//! everything else in the HuggingFace zoo — Whisper, CLIP,
//! detection models, embedding models, classifiers — the
//! universal path is HuggingFace -> ONNX export -> tract-onnx.
//!
//! tract-onnx is pure Rust (no ONNX Runtime C dep), ships ARM
//! NEON kernels for the common ops, and runs everywhere our other
//! NIFs run.

use std::collections::HashMap;
use std::sync::Mutex;
use tract_onnx::prelude::*;

pub struct OnnxModelResource {
    pub model: Mutex<TypedRunnableModel<TypedModel>>,
    /// Capture input/output names so the Elixir caller can address
    /// them by name (more robust than positional indexes).
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

unsafe impl Send for OnnxModelResource {}
unsafe impl Sync for OnnxModelResource {}

pub fn load(path: &str) -> Result<OnnxModelResource, String> {
    let proto = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| format!("onnx::model_for_path({}): {}", path, e))?;

    let typed = proto
        .into_typed()
        .map_err(|e| format!("into_typed: {}", e))?;

    let decluttered = typed
        .into_decluttered()
        .map_err(|e| format!("into_decluttered: {}", e))?;

    let input_names: Vec<String> = decluttered
        .input_outlets()
        .map_err(|e| format!("input_outlets: {}", e))?
        .iter()
        .map(|outlet| {
            decluttered
                .node(outlet.node)
                .name
                .clone()
        })
        .collect();

    let output_names: Vec<String> = decluttered
        .output_outlets()
        .map_err(|e| format!("output_outlets: {}", e))?
        .iter()
        .map(|outlet| {
            decluttered
                .node(outlet.node)
                .name
                .clone()
        })
        .collect();

    let runnable = decluttered
        .into_runnable()
        .map_err(|e| format!("into_runnable: {}", e))?;

    Ok(OnnxModelResource {
        model: Mutex::new(runnable),
        input_names,
        output_names,
    })
}

pub struct RunResult {
    pub outputs: Vec<(String, Vec<usize>, Vec<f32>)>,
}

/// Run inference. Inputs is a map name → (shape, flat f32 buffer).
/// Outputs are returned f32-only (we cast everything via tract's
/// `cast_to_dt(F32)` at the boundary — convenient for Nx interop).
pub fn run(
    res: &OnnxModelResource,
    inputs: HashMap<String, (Vec<usize>, Vec<f32>)>,
) -> Result<RunResult, String> {
    let model = res.model.lock().map_err(|e| format!("lock: {}", e))?;

    // Build tract Tensors in declared input order.
    let mut input_tensors: Vec<TValue> = Vec::with_capacity(res.input_names.len());
    for name in &res.input_names {
        let (shape, data) = inputs
            .get(name)
            .ok_or_else(|| format!("missing input '{}'", name))?;

        let t = tract_ndarray::Array::from_shape_vec(shape.clone(), data.clone())
            .map_err(|e| format!("input '{}' reshape: {}", name, e))?
            .into_tensor();

        input_tensors.push(t.into());
    }

    let outputs = model
        .run(input_tensors.into())
        .map_err(|e| format!("model.run: {}", e))?;

    let mut result = Vec::with_capacity(outputs.len());
    for (i, t) in outputs.iter().enumerate() {
        let name = res
            .output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", i));

        let shape = t.shape().to_vec();

        let f32_tensor = t
            .cast_to::<f32>()
            .map_err(|e| format!("cast output '{}' to f32: {}", name, e))?;

        let flat = f32_tensor
            .as_slice::<f32>()
            .map_err(|e| format!("as_slice output '{}': {}", name, e))?
            .to_vec();

        result.push((name, shape, flat));
    }

    Ok(RunResult { outputs: result })
}

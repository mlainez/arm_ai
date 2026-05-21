// nx_arm — Nx backend for ARM CPUs via NEON intrinsics + rayon.
//
// Pure-CPU compute: no OpenCL, no GPU, no device context. Every NIF
// takes raw `Binary` bytes in, returns raw `Binary` bytes out. The
// Elixir backend (NxArm.Backend) stores tensors as plain binaries
// and dispatches each Nx callback to one of these.

mod conv_int8;
mod llama_candle;
mod ops;
mod shape_ops;
mod topology;

use rustler::{Env, NifResult, OwnedBinary, ResourceArc};

// ---------------------------------------------------------------
// E1: memory-mapped model file resource.
//
// Wraps a `memmap2::Mmap` in a rustler resource so a single mmap can
// live for the lifetime of an Elixir reference. Slicing the mmap into
// a binary still copies (see comment on mmap_slice_op) but the file
// pages themselves are demand-loaded by the kernel — we never pull a
// multi-GB file into BEAM heap up front.
// ---------------------------------------------------------------
pub struct MmapResource(memmap2::Mmap);

unsafe impl Send for MmapResource {}
unsafe impl Sync for MmapResource {}

// ── helpers ─────────────────────────────────────────────

fn f32_vec_to_bin<'a>(env: Env<'a>, v: &[f32]) -> NifResult<rustler::Binary<'a>> {
    let bytes = v.len() * 4;
    let mut bin = OwnedBinary::new(bytes)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let src: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
    bin.as_mut_slice().copy_from_slice(src);
    Ok(bin.release(env))
}

fn bytes_to_bin<'a>(env: Env<'a>, v: &[u8]) -> NifResult<rustler::Binary<'a>> {
    let mut bin = OwnedBinary::new(v.len())
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    bin.as_mut_slice().copy_from_slice(v);
    Ok(bin.release(env))
}

/// Allocate a fresh `OwnedBinary` of `n_elements * 4` bytes and view it
/// as a `&mut [f32]`. Used by NIFs that can write directly into the
/// output buffer, skipping the Vec<f32> intermediate + memcpy pair.
fn alloc_f32_bin(n_elements: usize) -> NifResult<(OwnedBinary, *mut f32)> {
    let bin = OwnedBinary::new(n_elements * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let ptr = bin.as_slice().as_ptr() as *mut f32;
    Ok((bin, ptr))
}

// ── shape ops (dtype-agnostic via element_size) ─────────

#[rustler::nif(schedule = "DirtyCpu")]
fn broadcast_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    axes: Vec<usize>,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::broadcast(input.as_slice(), &in_shape, &out_shape, &axes, element_size)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn transpose_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    axes: Vec<usize>,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::transpose(input.as_slice(), &in_shape, &axes, element_size)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

/// Generic gather. `axes` lists which `tensor` axes are indexed by
/// the trailing dim of `indices`. Most callers (token-embedding
/// lookup) use `axes = [0]` — that goes through the fast contiguous
/// memcpy path.
/// Bilinear resize for HWC u8 image buffers. The image featurizer
/// entry point for vision models.
#[rustler::nif(schedule = "DirtyCpu")]
fn bilinear_resize_u8_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_h: usize,
    in_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::bilinear_resize_u8(input.as_slice(), in_h, in_w, channels, out_h, out_w)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

/// RMSNorm — Llama/Mistral/Phi/Qwen pre-attention/pre-MLP norm.
#[rustler::nif(schedule = "DirtyCpu")]
fn rmsnorm_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    gamma: rustler::Binary<'a>,
    n_outer: usize,
    inner: usize,
    epsilon: f64,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4)
    };
    let gamma_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(gamma.as_ptr() as *const f32, gamma.len() / 4)
    };
    let out = shape_ops::rmsnorm_f32(input_slice, gamma_slice, n_outer, inner, epsilon as f32)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Rotary Position Embedding for Q/K tensors.
#[rustler::nif(schedule = "DirtyCpu")]
fn rope_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    positions: rustler::Binary<'a>,
    inv_freq: rustler::Binary<'a>,
    n_rows: usize,
    head_dim: usize,
    heads_per_token: usize,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n_rows * head_dim)
    };
    let positions_slice: &[i64] = unsafe {
        std::slice::from_raw_parts(positions.as_ptr() as *const i64, positions.len() / 8)
    };
    let inv_freq_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(inv_freq.as_ptr() as *const f32, head_dim / 2)
    };
    let out = shape_ops::rope_f32(input_slice, positions_slice, inv_freq_slice, n_rows, head_dim, heads_per_token)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// fp16 weight × f32 activation matmul. Weights stored as 2 bytes
/// each, converted to f32 inline via NEON vcvt_f32_f16. Halves weight
/// memory footprint vs f32 with negligible accuracy loss.
#[rustler::nif(schedule = "DirtyCpu")]
fn dequant_matmul_f16_f32_op<'a>(
    env: Env<'a>,
    act: rustler::Binary<'a>,
    weights: rustler::Binary<'a>,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let act_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(act.as_ptr() as *const f32, b * m * k)
    };
    let weights_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(weights.as_ptr() as *const u16, n * k)
    };

    let out = shape_ops::dequant_matmul_f16_f32(act_slice, weights_slice, b, m, n, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Convert an f32 raw byte buffer to f16 (IEEE 754 binary16) bytes.
/// Used to build half-precision weight stores from existing f32 params.
#[rustler::nif(schedule = "DirtyCpu")]
fn f32_to_f16_op<'a>(env: Env<'a>, input: rustler::Binary<'a>) -> NifResult<rustler::Binary<'a>> {
    let n = input.len() / 4;
    let input_slice: &[f32] = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, n) };
    let halves = shape_ops::f32_to_f16_array(input_slice);

    let bytes = halves.len() * 2;
    let mut bin = OwnedBinary::new(bytes)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let src: &[u8] = unsafe { std::slice::from_raw_parts(halves.as_ptr() as *const u8, bytes) };
    bin.as_mut_slice().copy_from_slice(src);
    Ok(bin.release(env))
}

/// Full int8 matmul: i8 acts × i8 weights with f32 scales out. Uses
/// SDOT on ARMv8.2-A when available, vmlal_s8+vpadalq fallback otherwise.
#[rustler::nif(schedule = "DirtyCpu")]
fn int8_matmul_f32_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    w: rustler::Binary<'a>,
    w_scales: rustler::Binary<'a>,
    act_scale: f64,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let a_slice: &[i8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const i8, m * k) };
    let w_slice: &[i8] = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const i8, n * k) };
    let w_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w_scales.as_ptr() as *const f32, n)
    };

    let out = shape_ops::int8_matmul_f32(a_slice, w_slice, act_scale as f32, w_scales_slice, m, n, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Int8 matmul with per-token activation scales. act_scales is a
/// length-M f32 vector.
#[rustler::nif(schedule = "DirtyCpu")]
fn int8_matmul_f32_per_token_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    w: rustler::Binary<'a>,
    act_scales: rustler::Binary<'a>,
    w_scales: rustler::Binary<'a>,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let a_slice: &[i8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const i8, m * k) };
    let w_slice: &[i8] = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const i8, n * k) };
    let act_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(act_scales.as_ptr() as *const f32, m)
    };
    let w_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w_scales.as_ptr() as *const f32, n)
    };

    let out = shape_ops::int8_matmul_f32_per_token(
        a_slice, w_slice, act_scales_slice, w_scales_slice, m, n, k,
    )
    .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Quantize an (M, K) f32 activation matrix to int8 with symmetric
/// per-token scales. Returns {quantised_bytes, scales_bytes}.
#[rustler::nif(schedule = "DirtyCpu")]
fn quantize_int8_per_token_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    m: usize,
    k: usize,
) -> NifResult<(rustler::Binary<'a>, rustler::Binary<'a>)> {
    let a_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(a.as_ptr() as *const f32, m * k)
    };
    let (quantised, scales) = shape_ops::quantize_int8_per_token(a_slice, m, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    let q_bytes = unsafe {
        std::slice::from_raw_parts(quantised.as_ptr() as *const u8, quantised.len())
    };
    let mut q_bin = OwnedBinary::new(quantised.len())
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    q_bin.as_mut_slice().copy_from_slice(q_bytes);

    let s_bytes = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
    };
    let mut s_bin = OwnedBinary::new(scales.len() * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    s_bin.as_mut_slice().copy_from_slice(s_bytes);

    Ok((q_bin.release(env), s_bin.release(env)))
}

/// Fused softmax along the last axis. Replaces a 5-call Elixir
/// chain in the attention path with one NIF.
#[rustler::nif(schedule = "DirtyCpu")]
fn softmax_last_axis_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let src: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, outer * inner)
    };
    let mut out_bin = OwnedBinary::new(outer * inner * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let dst: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, outer * inner)
    };
    shape_ops::softmax_last_axis_f32(src, dst, outer, inner);
    Ok(out_bin.release(env))
}

/// Fused `silu(gate) * up` for the SwiGLU FFN. Replaces sigmoid +
/// multiply + multiply (3 NIFs) with one fused pass.
#[rustler::nif(schedule = "DirtyCpu")]
fn silu_gate_mul_up_f32_op<'a>(
    env: Env<'a>,
    gate: rustler::Binary<'a>,
    up: rustler::Binary<'a>,
) -> NifResult<rustler::Binary<'a>> {
    let g: &[f32] = unsafe {
        std::slice::from_raw_parts(gate.as_ptr() as *const f32, gate.as_slice().len() / 4)
    };
    let u: &[f32] = unsafe {
        std::slice::from_raw_parts(up.as_ptr() as *const f32, up.as_slice().len() / 4)
    };
    let mut out_bin = OwnedBinary::new(g.len() * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let dst: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, g.len())
    };
    shape_ops::silu_gate_mul_up_f32(g, u, dst);
    Ok(out_bin.release(env))
}

/// Dequantize a GGML Q6_K blob into a fresh f32 binary. Used to
/// materialise LM head weights at model load time since Q6_K
/// kernels aren't yet implemented inline.
#[rustler::nif(schedule = "DirtyCpu")]
fn dequantize_q6_k_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    n_elements: usize,
) -> NifResult<rustler::Binary<'a>> {
    let mut out = vec![0.0f32; n_elements];
    shape_ops::dequantize_q6_k(input.as_slice(), &mut out);
    f32_vec_to_bin(env, &out)
}

/// Dequantize one Q4_0 row to f32 — fast embedding-table lookup.
#[rustler::nif]
fn q4_0_dequant_row_op<'a>(
    env: Env<'a>,
    packed: rustler::Binary<'a>,
    scales: rustler::Binary<'a>,
    row: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let scales_f: &[f32] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const f32, scales.as_slice().len() / 4)
    };
    let mut out_bin = OwnedBinary::new(k * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let dst: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, k)
    };
    shape_ops::q4_0_dequant_row_f32(packed.as_slice(), scales_f, row, k, dst);
    Ok(out_bin.release(env))
}

/// Dequantize one Q8_0 row to f32 — fast embedding-table lookup.
#[rustler::nif]
fn q8_0_dequant_row_op<'a>(
    env: Env<'a>,
    weights: rustler::Binary<'a>,
    scales: rustler::Binary<'a>,
    row: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let w_i8: &[i8] = unsafe {
        std::slice::from_raw_parts(weights.as_ptr() as *const i8, weights.as_slice().len())
    };
    let scales_f: &[f32] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const f32, scales.as_slice().len() / 4)
    };
    let mut out_bin = OwnedBinary::new(k * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let dst: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, k)
    };
    shape_ops::q8_0_dequant_row_f32(w_i8, scales_f, row, k, dst);
    Ok(out_bin.release(env))
}

/// NEON Q4_0 × Q8_0 GEMV (M=1, dotprod-free). Best path for LLM
/// decode-time matmuls on A73 / generic ARMv8.0+ ARM cores.
#[rustler::nif(schedule = "DirtyCpu")]
fn int4_matmul_gemv_q4_x_q8_neon_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    w_packed: rustler::Binary<'a>,
    w_scales: rustler::Binary<'a>,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let a_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(a.as_ptr() as *const f32, k)
    };
    let w_packed_slice: &[u8] = w_packed.as_slice();
    let w_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w_scales.as_ptr() as *const f32, n * (k / 32))
    };

    let out = shape_ops::int4_matmul_gemv_q4_x_q8_neon(a_slice, w_packed_slice, w_scales_slice, n, k);
    f32_vec_to_bin(env, &out)
}

/// NEON Q4_0 GEMV (M=1 specialised). Pulled out so decode-time
/// LLM matmuls (lm_head, per-layer projections) skip the scalar
/// path and hit ~5-10× over `int4_matmul_f32_op` for M=1.
#[rustler::nif(schedule = "DirtyCpu")]
fn int4_matmul_gemv_neon_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    w_packed: rustler::Binary<'a>,
    w_scales: rustler::Binary<'a>,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let a_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(a.as_ptr() as *const f32, k)
    };
    let w_packed_slice: &[u8] = w_packed.as_slice();
    let w_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w_scales.as_ptr() as *const f32, n * (k / 32))
    };

    let out = shape_ops::int4_matmul_gemv_neon(a_slice, w_packed_slice, w_scales_slice, n, k);
    f32_vec_to_bin(env, &out)
}

/// Int4 (Q4_0) matmul: f32 activations × packed int4 weights with
/// per-group (group_size=32) f32 scales. Returns f32 outputs.
#[rustler::nif(schedule = "DirtyCpu")]
fn int4_matmul_f32_op<'a>(
    env: Env<'a>,
    a: rustler::Binary<'a>,
    w_packed: rustler::Binary<'a>,
    w_scales: rustler::Binary<'a>,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let a_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(a.as_ptr() as *const f32, m * k)
    };
    let w_packed_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(w_packed.as_ptr(), n * (k / 2))
    };
    let w_scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w_scales.as_ptr() as *const f32, n * (k / 32))
    };

    let out = shape_ops::int4_matmul_f32(a_slice, w_packed_slice, w_scales_slice, m, n, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Quantize an (N, K) f32 weight matrix to Q4_0 packed int4. Returns
/// {packed_bytes_binary, scales_binary}. K must be a multiple of 32.
#[rustler::nif(schedule = "DirtyCpu")]
fn quantize_int4_q4_0_op<'a>(
    env: Env<'a>,
    w: rustler::Binary<'a>,
    n: usize,
    k: usize,
) -> NifResult<(rustler::Binary<'a>, rustler::Binary<'a>)> {
    let w_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const f32, n * k)
    };
    let (packed, scales) = shape_ops::quantize_int4_q4_0(w_slice, n, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    let mut packed_bin = OwnedBinary::new(packed.len())
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    packed_bin.as_mut_slice().copy_from_slice(&packed);

    let scales_bytes = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
    };
    let mut scales_bin = OwnedBinary::new(scales.len() * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    scales_bin.as_mut_slice().copy_from_slice(scales_bytes);

    Ok((packed_bin.release(env), scales_bin.release(env)))
}

/// Weight-only int8 matmul: f32 acts × int8 weights × f32 per-row
/// scales. Layout matches batched_matmul_f32 with `right_transposed=true`
/// (acts and weights both contract on last axis).
#[rustler::nif(schedule = "DirtyCpu")]
fn dequant_matmul_int8_f32_op<'a>(
    env: Env<'a>,
    act: rustler::Binary<'a>,
    weights: rustler::Binary<'a>,
    scales: rustler::Binary<'a>,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let act_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(act.as_ptr() as *const f32, b * m * k)
    };
    let weights_slice: &[i8] = unsafe {
        std::slice::from_raw_parts(weights.as_ptr() as *const i8, n * k)
    };
    let scales_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const f32, n)
    };

    let out = shape_ops::dequant_matmul_int8_f32(act_slice, weights_slice, scales_slice, b, m, n, k)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    f32_vec_to_bin(env, &out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn window_reduce_f32_op<'a>(
    env: Env<'a>,
    op: String,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    window_dims: Vec<usize>,
    strides: Vec<usize>,
    pad_low: Vec<i64>,
    pad_high: Vec<i64>,
) -> NifResult<rustler::Binary<'a>> {
    let n = input.len() / 4;
    let in_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, n) };

    let padding: Vec<(isize, isize)> = pad_low
        .iter()
        .zip(pad_high.iter())
        .map(|(&lo, &hi)| (lo as isize, hi as isize))
        .collect();

    let (out_vec, _out_shape) =
        shape_ops::window_reduce_f32(&op, in_slice, &in_shape, &window_dims, &strides, &padding)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    f32_vec_to_bin(env, &out_vec)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn slice_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    starts: Vec<usize>,
    lengths: Vec<usize>,
    strides: Vec<usize>,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::slice(
        input.as_slice(),
        &in_shape,
        &starts,
        &lengths,
        &strides,
        element_size,
    )
    .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn put_slice_op<'a>(
    env: Env<'a>,
    tensor: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    slice: rustler::Binary<'a>,
    slice_shape: Vec<usize>,
    starts: Vec<usize>,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::put_slice(
        tensor.as_slice(),
        &in_shape,
        slice.as_slice(),
        &slice_shape,
        &starts,
        element_size,
    )
    .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn gather_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    indices: rustler::Binary<'a>,
    idx_shape: Vec<usize>,
    index_size: usize,
    axes: Vec<usize>,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let out = shape_ops::gather(
        input.as_slice(),
        &in_shape,
        indices.as_slice(),
        &idx_shape,
        index_size,
        &axes,
        element_size,
    )
    .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn concatenate_op<'a>(
    env: Env<'a>,
    tensors: Vec<rustler::Binary<'a>>,
    shapes: Vec<Vec<usize>>,
    axis: usize,
    element_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    if tensors.len() != shapes.len() {
        return Err(rustler::Error::Term(Box::new(format!(
            "tensors len {} != shapes len {}",
            tensors.len(),
            shapes.len()
        ))));
    }
    let slices: Vec<&[u8]> = tensors.iter().map(|b| b.as_slice()).collect();
    let out = shape_ops::concatenate(&slices, &shapes, axis, element_size)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    bytes_to_bin(env, &out)
}

// ── f32 matmul (batched, b=1 covers the 2-D case) ───────

#[rustler::nif(schedule = "DirtyCpu")]
fn batched_matmul_f32_op<'a>(
    env: Env<'a>,
    left: rustler::Binary<'a>,
    right: rustler::Binary<'a>,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    right_transposed: bool,
) -> NifResult<rustler::Binary<'a>> {
    let need_left = b * m * k;
    let need_right = b * k * n;
    if left.len() != need_left * 4 {
        return Err(rustler::Error::Term(Box::new(format!(
            "left bytes {} != B*M*K*4 = {}",
            left.len(),
            need_left * 4
        ))));
    }
    if right.len() != need_right * 4 {
        return Err(rustler::Error::Term(Box::new(format!(
            "right bytes {} != B*K*N*4 (or B*N*K*4) = {}",
            right.len(),
            need_right * 4
        ))));
    }

    let left_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(left.as_ptr() as *const f32, need_left) };
    let right_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(right.as_ptr() as *const f32, need_right) };

    let out = shape_ops::batched_matmul_f32(left_f32, right_f32, b, m, n, k, right_transposed)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    f32_vec_to_bin(env, &out)
}

// ── f32 elementwise binary / scalar / unary ─────────────

#[rustler::nif(schedule = "DirtyCpu")]
fn elementwise_binary_f32_op<'a>(
    env: Env<'a>,
    op: String,
    a: rustler::Binary<'a>,
    b: rustler::Binary<'a>,
) -> NifResult<rustler::Binary<'a>> {
    let n = a.len() / 4;
    let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, n) };
    let b_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) };

    // Write directly into the OwnedBinary — skip the Vec<f32> →
    // OwnedBinary memcpy that adds ~50–100 µs per call.
    let mut bin = OwnedBinary::new(n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    {
        let out_slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(bin.as_mut_slice().as_mut_ptr() as *mut f32, n) };
        shape_ops::elementwise_binary_f32_into(&op, a_slice, b_slice, out_slice)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    }
    Ok(bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn scalar_binary_f32_op<'a>(
    env: Env<'a>,
    op: String,
    side: String,
    a: rustler::Binary<'a>,
    scalar: f64,
) -> NifResult<rustler::Binary<'a>> {
    let n = a.len() / 4;
    let a_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, n) };
    let mut bin = OwnedBinary::new(n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    {
        let out_slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(bin.as_mut_slice().as_mut_ptr() as *mut f32, n) };
        shape_ops::scalar_binary_f32_into(&op, &side, a_slice, scalar as f32, out_slice)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    }
    Ok(bin.release(env))
}

/// Flash Attention V1 forward — fused Q@K^T → softmax → @V without
/// materialising the (Sq, Sk) attention matrix.
#[rustler::nif(schedule = "DirtyCpu")]
fn flash_attention_f32_op<'a>(
    env: Env<'a>,
    q: rustler::Binary<'a>,
    k: rustler::Binary<'a>,
    v: rustler::Binary<'a>,
    scale: f64,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
) -> NifResult<rustler::Binary<'a>> {
    let q_slice: &[f32] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const f32, b * h * sq * d) };
    let k_slice: &[f32] = unsafe { std::slice::from_raw_parts(k.as_ptr() as *const f32, b * h * sk * d) };
    let v_slice: &[f32] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, b * h * sk * d) };

    let out = shape_ops::flash_attention_f32(q_slice, k_slice, v_slice, scale as f32, b, h, sq, sk, d, causal)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Fused linear: `out = act @ w^T + bias`. Optionally chains an
/// activation in the same pass.
#[rustler::nif(schedule = "DirtyCpu")]
fn linear_f32_op<'a>(
    env: Env<'a>,
    act: rustler::Binary<'a>,
    weights: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    activation: String,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> NifResult<rustler::Binary<'a>> {
    let need_act = b * m * k;
    let act_slice: &[f32] = unsafe { std::slice::from_raw_parts(act.as_ptr() as *const f32, need_act) };
    let w_slice: &[f32] = unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const f32, n * k) };
    let bias_slice = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, n) })
    };

    let out = shape_ops::linear_f32(act_slice, w_slice, bias_slice, b, m, n, k, &activation)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Fused bias-add + activation. Replaces the
/// `out = activation(linear + bias)` pattern with one pass.
#[rustler::nif(schedule = "DirtyCpu")]
fn bias_add_activation_f32_op<'a>(
    env: Env<'a>,
    act: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    activation: String,
    outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let act_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(act.as_ptr() as *const f32, act.len() / 4) };
    let bias_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len() / 4) };

    let n = outer * inner;
    let mut bin = OwnedBinary::new(n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    {
        let out_slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(bin.as_mut_slice().as_mut_ptr() as *mut f32, n) };
        shape_ops::bias_add_activation_f32_into(act_slice, bias_slice, out_slice, outer, inner, &activation)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    }
    Ok(bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn bias_add_f32_op<'a>(
    env: Env<'a>,
    act: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let act_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(act.as_ptr() as *const f32, act.len() / 4)
    };
    let bias_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len() / 4)
    };

    let n = outer * inner;
    let mut bin = OwnedBinary::new(n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    {
        let out_slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(bin.as_mut_slice().as_mut_ptr() as *mut f32, n) };
        shape_ops::bias_add_f32_into(act_slice, bias_slice, out_slice, outer, inner)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    }
    Ok(bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn elementwise_unary_f32_op<'a>(
    env: Env<'a>,
    op: String,
    a: rustler::Binary<'a>,
) -> NifResult<rustler::Binary<'a>> {
    let n = a.len() / 4;
    let a_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, n) };
    let mut bin = OwnedBinary::new(n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    {
        let out_slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(bin.as_mut_slice().as_mut_ptr() as *mut f32, n) };
        shape_ops::elementwise_unary_f32_into(&op, a_slice, out_slice)
            .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    }
    Ok(bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn reduce_axis_f32_op<'a>(
    env: Env<'a>,
    op: String,
    input: rustler::Binary<'a>,
    n_outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4)
    };
    let out = shape_ops::reduce_axis_f32(&op, input_slice, n_outer, inner)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Fused GELU. Replaces Axon's 5-primitive defn decomposition
/// `((erf(x/√2)+1)*x)/2` with one NIF call.
#[rustler::nif(schedule = "DirtyCpu")]
fn gelu_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4)
    };
    let out = shape_ops::gelu_f32(input_slice)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Fused LayerNorm along the last axis. `gamma`/`beta` length-`inner`.
#[rustler::nif(schedule = "DirtyCpu")]
fn layernorm_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    gamma: rustler::Binary<'a>,
    beta: rustler::Binary<'a>,
    n_outer: usize,
    inner: usize,
    epsilon: f64,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4)
    };
    let gamma_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(gamma.as_ptr() as *const f32, gamma.len() / 4)
    };
    let beta_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(beta.as_ptr() as *const f32, beta.len() / 4)
    };
    let out = shape_ops::layernorm_f32(input_slice, gamma_slice, beta_slice, n_outer, inner, epsilon as f32)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

/// Fused softmax along the last axis. Replaces Axon's 7-primitive defn
/// decomposition with one NIF call (one pass per row: max, exp, sum,
/// divide).
#[rustler::nif(schedule = "DirtyCpu")]
fn softmax_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    n_outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let input_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4)
    };
    let out = shape_ops::softmax_f32(input_slice, n_outer, inner)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    f32_vec_to_bin(env, &out)
}

// ── conv2d (NEON int8 + f32) ────────────────────────────

#[rustler::nif(schedule = "DirtyCpu")]
fn conv2d_int8_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    weight: rustler::Binary<'a>,
    scales: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 7 {
        return Err(rustler::Error::Term(Box::new(
            "dims must be [N, H, W, Cin, Cout, Kh, Kw]".to_string(),
        )));
    }
    if stride.len() != 2 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "stride must be 2 ints, padding 4".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, c_out, kh, kw) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let weight_i8: &[i8] = unsafe {
        std::slice::from_raw_parts(weight.as_ptr() as *const i8, c_out * kh * kw * c_in)
    };
    let scales_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const f32, c_out) };

    let bias_slice: Option<&[f32]> = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, c_out) })
    };

    let out_n = n * h_out * w_out * c_out;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };

    conv_int8::conv2d_int8(
        input_f32, weight_i8, scales_f32, bias_slice, out_f32,
        n, h_in, w_in, c_in, c_out, kh, kw,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    Ok(out_bin.release(env))
}

/// Depthwise 2-D conv (feature_group_size == Cin). Kernel layout
/// `{Cin, Kh, Kw}` (the {Cin, 1, Kh, Kw} Nx tensor with the 1-dim
/// squeezed). Same NHWC input + NHWC output layout as conv2d_f32_op.
#[rustler::nif(schedule = "DirtyCpu")]
fn depthwise_conv2d_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    weight: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 6 {
        return Err(rustler::Error::Term(Box::new(
            "dims must be [N, H, W, Cin, Kh, Kw]".to_string(),
        )));
    }
    if stride.len() != 2 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "stride must be 2 ints, padding 4".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, kh, kw) = (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(weight.as_ptr() as *const f32, c_in * kh * kw)
    };
    let bias_slice: Option<&[f32]> = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, c_in) })
    };

    let out_n = n * h_out * w_out * c_in;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };

    conv_int8::depthwise_conv2d_f32(
        input_f32, weight_f32, bias_slice, out_f32,
        n, h_in, w_in, c_in, kh, kw,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    Ok(out_bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn conv2d_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    weight: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 7 || stride.len() != 2 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "bad dims/stride/padding".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, c_out, kh, kw) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(weight.as_ptr() as *const f32, c_out * kh * kw * c_in)
    };
    let bias_slice: Option<&[f32]> = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, c_out) })
    };

    let out_n = n * h_out * w_out * c_out;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };

    conv_int8::conv2d_f32(
        input_f32, weight_f32, bias_slice, out_f32,
        n, h_in, w_in, c_in, c_out, kh, kw,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    Ok(out_bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn depthwise_pointwise_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    dw_weight: rustler::Binary<'a>,
    dw_bias: rustler::Binary<'a>,
    pw_weight: rustler::Binary<'a>,
    pw_bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    activation: u8,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 7 || stride.len() != 2 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "bad dims/stride/padding".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, c_out, kh, kw) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let dw_weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(dw_weight.as_ptr() as *const f32, c_in * kh * kw)
    };
    let pw_weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(pw_weight.as_ptr() as *const f32, c_out * c_in)
    };
    let dw_bias_slice: Option<&[f32]> = if dw_bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(dw_bias.as_ptr() as *const f32, c_in) })
    };
    let pw_bias_slice: Option<&[f32]> = if pw_bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(pw_bias.as_ptr() as *const f32, c_out) })
    };

    let out_n = n * h_out * w_out * c_out;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };

    conv_int8::depthwise_pointwise_f32(
        input_f32, dw_weight_f32, dw_bias_slice,
        pw_weight_f32, pw_bias_slice, out_f32,
        n, h_in, w_in, c_in, c_out, kh, kw,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
        activation,
    );

    Ok(out_bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn conv2d_f32_im2col_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    weight: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 7 || stride.len() != 2 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "bad dims/stride/padding".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, c_out, kh, kw) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(weight.as_ptr() as *const f32, c_out * kh * kw * c_in)
    };
    let bias_slice: Option<&[f32]> = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, c_out) })
    };

    let out_n = n * h_out * w_out * c_out;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };

    conv_int8::conv2d_f32_im2col(
        input_f32, weight_f32, bias_slice, out_f32,
        n, h_in, w_in, c_in, c_out, kh, kw,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    Ok(out_bin.release(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn conv2d_f32_winograd_3x3_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    weight: rustler::Binary<'a>,
    bias: rustler::Binary<'a>,
    dims: Vec<usize>,
    padding: Vec<usize>,
) -> NifResult<rustler::Binary<'a>> {
    if dims.len() != 5 || padding.len() != 4 {
        return Err(rustler::Error::Term(Box::new(
            "bad dims/padding".to_string(),
        )));
    }

    let (n, h_in, w_in, c_in, c_out) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let (pad_top, pad_bottom, pad_left, pad_right) =
        (padding[0], padding[1], padding[2], padding[3]);

    let (h_out, w_out) = conv_int8::output_dims(
        h_in, w_in, 3, 3, 1, 1, pad_top, pad_bottom, pad_left, pad_right,
    );

    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, n * h_in * w_in * c_in)
    };
    let weight_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(weight.as_ptr() as *const f32, c_out * 3 * 3 * c_in)
    };
    let bias_slice: Option<&[f32]> = if bias.is_empty() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, c_out) })
    };

    let out_n = n * h_out * w_out * c_out;
    let mut out_bin = OwnedBinary::new(out_n * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    let out_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out_bin.as_mut_slice().as_mut_ptr() as *mut f32, out_n)
    };
    // Zero the buffer so untouched border cells (if H_out or W_out
    // is odd) stay at 0 rather than holding uninitialised bytes.
    for v in out_f32.iter_mut() {
        *v = 0.0;
    }

    conv_int8::conv2d_f32_winograd_3x3(
        input_f32, weight_f32, bias_slice, out_f32,
        n, h_in, w_in, c_in, c_out,
        pad_top, pad_bottom, pad_left, pad_right,
    );

    Ok(out_bin.release(env))
}

// ---------------------------------------------------------------
// E3: rayon thread pool tuning.
//
// Rayon's global pool is sized to logical core count on first use.
// On Cortex-A73 (4 cores) this is fine; on big.LITTLE chips you
// typically want to pin to the perf cluster or cap below the total.
//
// `init_thread_pool_op(n)` calls ThreadPoolBuilder::build_global with
// the requested thread count. Must be called before any rayon
// par_iter touches the global pool (i.e. at app boot, before any
// matmul). Returns `:ok` on first call, `:already_initialised`
// otherwise (rayon's global pool is one-shot).
//
// `current_thread_count_op/0` always reports the active pool size.
// ---------------------------------------------------------------

#[rustler::nif]
fn init_thread_pool_op(n: usize) -> rustler::Atom {
    let res = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    match res {
        Ok(_) => atoms::ok(),
        Err(_) => atoms::already_initialised(),
    }
}

/// Auto-detect big.LITTLE topology and pin rayon to the perf cluster.
/// On homogeneous chips it's a no-op pin (rayon sizes itself normally).
/// Idempotent: returns :already_initialised if rayon's global pool has
/// already been touched.
#[rustler::nif]
fn init_perf_cluster_op() -> (rustler::Atom, usize, Vec<usize>, String) {
    let topo = topology::Topology::detect();
    let perf = topo.perf_cores.clone();
    let n = perf.len().max(1);

    // Seed the per-thread pin cache so any BEAM dirty scheduler
    // that later enters our NIFs pins itself to the same cluster.
    topology::set_perf_cluster_cache(perf.clone());

    let pinning_cores = perf.clone();
    let res = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(move |_idx| {
            topology::pin_current_thread_to(&pinning_cores);
        })
        .build_global();

    let status = match res {
        Ok(_) => atoms::ok(),
        Err(_) => atoms::already_initialised(),
    };

    (status, n, perf, topo.source.to_string())
}

/// Pin the calling BEAM dirty scheduler thread to the perf cluster.
/// Cheap on subsequent calls (thread-local cache). Called from
/// `NxArm.Runtime` on warmup to ensure all live dirty schedulers
/// migrate to the perf cluster once.
#[rustler::nif(schedule = "DirtyCpu")]
fn pin_calling_thread_op() -> rustler::Atom {
    topology::ensure_thread_pinned();
    // Sleep briefly so the BEAM doesn't immediately re-use this
    // thread for the next warmup task — we want every dirty
    // scheduler to get its own pin call.
    std::thread::sleep(std::time::Duration::from_millis(5));
    atoms::ok()
}

#[rustler::nif]
fn detect_topology_op() -> (Vec<usize>, Vec<usize>, String) {
    let topo = topology::Topology::detect();
    (topo.perf_cores, topo.all_cores, topo.source.to_string())
}

#[rustler::nif]
fn detect_topology_full_op() -> (Vec<usize>, Vec<usize>, Vec<usize>, String) {
    let topo = topology::Topology::detect();
    (
        topo.perf_cores,
        topo.efficiency_cores,
        topo.all_cores,
        topo.source.to_string(),
    )
}

/// Pin the *calling thread* to the given list of CPUs. Used by
/// startup hooks to migrate BEAM normal-scheduler threads off the
/// perf cluster so the perf cores stay reserved for compute NIFs.
#[rustler::nif]
fn pin_thread_to_cores_op(cores: Vec<usize>) -> rustler::Atom {
    topology::pin_current_thread_to(&cores);
    atoms::ok()
}

#[rustler::nif]
fn current_thread_count_op() -> usize {
    rayon::current_num_threads()
}

mod atoms {
    rustler::atoms! { ok, already_initialised }
}

#[rustler::nif(schedule = "DirtyIo")]
fn mmap_open_op(path: String) -> NifResult<(ResourceArc<MmapResource>, usize)> {
    let file = std::fs::File::open(&path)
        .map_err(|e| rustler::Error::Term(Box::new(format!("open failed: {}", e))))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| rustler::Error::Term(Box::new(format!("mmap failed: {}", e))))?;
    let len = mmap.len();
    Ok((ResourceArc::new(MmapResource(mmap)), len))
}

/// Read `len` bytes at `offset` from a mmap'd file into a fresh
/// binary. The mmap'd pages are demand-loaded from disk; the
/// returned binary is a BEAM-owned copy of just the requested range,
/// so even multi-GB files only ever touch RAM lazily and in slices.
#[rustler::nif(schedule = "DirtyIo")]
fn mmap_slice_op<'a>(
    env: Env<'a>,
    handle: ResourceArc<MmapResource>,
    offset: usize,
    len: usize,
) -> NifResult<rustler::Binary<'a>> {
    let bytes = &handle.0;
    if offset.checked_add(len).map_or(true, |end| end > bytes.len()) {
        return Err(rustler::Error::Term(Box::new(format!(
            "mmap_slice: offset {} + len {} > file len {}",
            offset, len, bytes.len()
        ))));
    }
    bytes_to_bin(env, &bytes[offset..offset + len])
}

// ---------------------------------------------------------------
// Production-readiness ops: argmax/argmin, select, as_type, clip,
// pad, gather, stack. Each replaces a fallback path.
// ---------------------------------------------------------------

#[rustler::nif]
fn argmax_axis_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let mut out_bin = OwnedBinary::new(outer * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::argmax_axis_f32(input.as_slice(), out_bin.as_mut_slice(), outer, inner);
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn argmin_axis_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
) -> NifResult<rustler::Binary<'a>> {
    let mut out_bin = OwnedBinary::new(outer * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::argmin_axis_f32(input.as_slice(), out_bin.as_mut_slice(), outer, inner);
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn select_op<'a>(
    env: Env<'a>,
    pred: rustler::Binary<'a>,
    on_true: rustler::Binary<'a>,
    on_false: rustler::Binary<'a>,
    elem_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let n = pred.as_slice().len();
    let mut out_bin = OwnedBinary::new(n * elem_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::select_same_shape(
        pred.as_slice(),
        on_true.as_slice(),
        on_false.as_slice(),
        out_bin.as_mut_slice(),
        elem_size,
    );
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn as_type_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    src_dtype: u8,
    dst_dtype: u8,
    n_elems: usize,
) -> NifResult<rustler::Binary<'a>> {
    let src_dt = ops::Dtype::from_u8(src_dtype)
        .ok_or_else(|| rustler::Error::Term(Box::new(format!("bad src dtype {}", src_dtype))))?;
    let dst_dt = ops::Dtype::from_u8(dst_dtype)
        .ok_or_else(|| rustler::Error::Term(Box::new(format!("bad dst dtype {}", dst_dtype))))?;

    let mut out_bin = OwnedBinary::new(n_elems * dst_dt.size())
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::as_type(input.as_slice(), out_bin.as_mut_slice(), src_dt, dst_dt)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn clip_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    dtype: u8,
    min: f64,
    max: f64,
) -> NifResult<rustler::Binary<'a>> {
    let dt = ops::Dtype::from_u8(dtype)
        .ok_or_else(|| rustler::Error::Term(Box::new(format!("bad dtype {}", dtype))))?;

    let mut out_bin = OwnedBinary::new(input.as_slice().len())
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::clip(input.as_slice(), out_bin.as_mut_slice(), dt, min, max)
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn pad_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    pad_config: Vec<(i64, i64, i64)>,
    fill: rustler::Binary<'a>,
    elem_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let n_out: usize = out_shape.iter().product();
    let mut out_bin = OwnedBinary::new(n_out * elem_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::pad(
        input.as_slice(),
        out_bin.as_mut_slice(),
        &in_shape,
        &out_shape,
        &pad_config,
        fill.as_slice(),
        elem_size,
    );
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn gather_axis0_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    indices: rustler::Binary<'a>,
    n_rows: usize,
    row_bytes: usize,
) -> NifResult<rustler::Binary<'a>> {
    let idx_bytes = indices.as_slice();
    let idx_slice: &[i64] = unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const i64, idx_bytes.len() / 8)
    };

    let mut out_bin = OwnedBinary::new(idx_slice.len() * row_bytes)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::gather_axis0(
        input.as_slice(),
        idx_slice,
        out_bin.as_mut_slice(),
        n_rows,
        row_bytes,
    )
    .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn stack_axis0_op<'a>(
    env: Env<'a>,
    tensors: Vec<rustler::Binary<'a>>,
    tensor_bytes: usize,
) -> NifResult<rustler::Binary<'a>> {
    let slices: Vec<&[u8]> = tensors.iter().map(|t| t.as_slice()).collect();
    let mut out_bin = OwnedBinary::new(tensors.len() * tensor_bytes)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::stack_axis0(&slices, out_bin.as_mut_slice(), tensor_bytes);
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn sort_axis_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
    descending: bool,
) -> NifResult<rustler::Binary<'a>> {
    let mut out_bin = OwnedBinary::new(outer * inner * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::sort_axis_f32(input.as_slice(), out_bin.as_mut_slice(), outer, inner, descending);
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn argsort_axis_f32_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    outer: usize,
    inner: usize,
    descending: bool,
) -> NifResult<rustler::Binary<'a>> {
    let mut out_bin = OwnedBinary::new(outer * inner * 4)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::argsort_axis_f32(input.as_slice(), out_bin.as_mut_slice(), outer, inner, descending);
    Ok(out_bin.release(env))
}

#[rustler::nif]
fn reduce_all_u8_op(input: rustler::Binary) -> u8 {
    ops::reduce_all_u8(input.as_slice())
}

#[rustler::nif]
fn reduce_any_u8_op(input: rustler::Binary) -> u8 {
    ops::reduce_any_u8(input.as_slice())
}

#[rustler::nif]
fn reduce_product_f32_op(input: rustler::Binary) -> f32 {
    ops::reduce_product_f32(input.as_slice())
}

#[rustler::nif]
fn reverse_op<'a>(
    env: Env<'a>,
    input: rustler::Binary<'a>,
    shape: Vec<usize>,
    axes: Vec<usize>,
    elem_size: usize,
) -> NifResult<rustler::Binary<'a>> {
    let n: usize = shape.iter().product();
    let mut out_bin = OwnedBinary::new(n * elem_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("OwnedBinary alloc failed".to_string())))?;
    ops::reverse_axes(
        input.as_slice(),
        out_bin.as_mut_slice(),
        &shape,
        &axes,
        elem_size,
    );
    Ok(out_bin.release(env))
}

// ---------------------------------------------------------------
// candle bridge for Llama-family models.
// ---------------------------------------------------------------

#[rustler::nif(schedule = "DirtyCpu")]
fn llama_candle_load_op(path: String) -> NifResult<ResourceArc<llama_candle::LlamaResource>> {
    let res = llama_candle::load_model(&path)
        .map_err(|e| rustler::Error::Term(Box::new(format!("candle load: {}", e))))?;
    Ok(ResourceArc::new(res))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn llama_candle_generate_op(
    model: ResourceArc<llama_candle::LlamaResource>,
    prompt: Vec<u32>,
    max_new: usize,
) -> NifResult<(Vec<u32>, u64, u64)> {
    let result = llama_candle::generate_greedy(&model, &prompt, max_new)
        .map_err(|e| rustler::Error::Term(Box::new(format!("candle generate: {}", e))))?;
    Ok((result.tokens, result.prefill_us, result.decode_us))
}

fn load(env: Env, _info: rustler::Term) -> bool {
    rustler::resource!(MmapResource, env);
    rustler::resource!(llama_candle::LlamaResource, env);
    true
}

rustler::init!("Elixir.NxArm.Native", load = load);

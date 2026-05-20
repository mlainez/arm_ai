// nx_arm — Nx backend for ARM CPUs via NEON intrinsics + rayon.
//
// Pure-CPU compute: no OpenCL, no GPU, no device context. Every NIF
// takes raw `Binary` bytes in, returns raw `Binary` bytes out. The
// Elixir backend (NxArm.Backend) stores tensors as plain binaries
// and dispatches each Nx callback to one of these.

mod conv_int8;
mod shape_ops;

use rustler::{Env, NifResult, OwnedBinary};

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

rustler::init!("Elixir.NxArm.Native");

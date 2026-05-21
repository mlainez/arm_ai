//! General-purpose Nx op kernels: argmax/argmin, select, as_type,
//! clip, pad, gather, stack, sort/argsort. These replace the
//! BinaryBackend fallbacks that production LLM/CV workloads kept
//! hitting. Each function is dtype-dispatched at the NIF boundary.

use rayon::prelude::*;

/// Dtype tag. We pass these as `u8` over the NIF boundary to keep
/// the API stable.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32 = 0,
    F64 = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    BF16 = 10,
    F16 = 11,
    BOOL = 12,
}

impl Dtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        use Dtype::*;
        match v {
            0 => Some(F32),
            1 => Some(F64),
            2 => Some(S8),
            3 => Some(S16),
            4 => Some(S32),
            5 => Some(S64),
            6 => Some(U8),
            7 => Some(U16),
            8 => Some(U32),
            9 => Some(U64),
            10 => Some(BF16),
            11 => Some(F16),
            12 => Some(BOOL),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        use Dtype::*;
        match self {
            F32 | S32 | U32 => 4,
            F64 | S64 | U64 => 8,
            S8 | U8 | BOOL => 1,
            S16 | U16 | BF16 | F16 => 2,
        }
    }
}

// --- helpers ---

fn as_slice<T: Copy>(bytes: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / std::mem::size_of::<T>(),
        )
    }
}

fn write_slice<T: Copy>(dst: &mut [u8], src: &[T]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * std::mem::size_of::<T>())
    };
    dst[..bytes.len()].copy_from_slice(bytes);
}

// ---------------------------------------------------------------
// argmax / argmin along the last axis, returning s64.
//
// The most common LLM use is greedy-decode + classification head,
// always argmax over the vocab/class axis. We special-case "no axis
// given = argmax over whole tensor" and "axis = -1 / inner".
// ---------------------------------------------------------------

pub fn argmax_axis_f32(input: &[u8], out: &mut [u8], outer: usize, inner: usize) {
    let src = as_slice::<f32>(input);
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut i32, outer)
    };

    dst.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let row = &src[o * inner..(o + 1) * inner];
        let mut best_idx = 0;
        let mut best_val = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        *slot = best_idx as i32;
    });
}

pub fn argmin_axis_f32(input: &[u8], out: &mut [u8], outer: usize, inner: usize) {
    let src = as_slice::<f32>(input);
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut i32, outer)
    };

    dst.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let row = &src[o * inner..(o + 1) * inner];
        let mut best_idx = 0;
        let mut best_val = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v < best_val {
                best_val = v;
                best_idx = i;
            }
        }
        *slot = best_idx as i32;
    });
}

// ---------------------------------------------------------------
// select: same-shape predicate (u8 0/1), then on_true / on_false
// of arbitrary numeric dtype. Output dtype == on_true/on_false.
//
// Hot path for causal masks and any Nx.select conditional.
// ---------------------------------------------------------------

pub fn select_same_shape(
    pred: &[u8],
    on_true: &[u8],
    on_false: &[u8],
    out: &mut [u8],
    elem_size: usize,
) {
    let n = pred.len();
    debug_assert_eq!(on_true.len(), n * elem_size);
    debug_assert_eq!(on_false.len(), n * elem_size);
    debug_assert_eq!(out.len(), n * elem_size);

    for i in 0..n {
        let src = if pred[i] != 0 { on_true } else { on_false };
        let dst_off = i * elem_size;
        let src_off = i * elem_size;
        out[dst_off..dst_off + elem_size].copy_from_slice(&src[src_off..src_off + elem_size]);
    }
}

// ---------------------------------------------------------------
// as_type: dtype conversion between numeric types.
//
// Covers the common Bumblebee/Axon paths: f32→s8/u8 (quantisation),
// s8/u8→f32 (dequantisation), f32→bf16 (mixed precision), f32→s64
// (integer indexing), and the obvious widening conversions.
// ---------------------------------------------------------------

pub fn as_type(input: &[u8], out: &mut [u8], src_dt: Dtype, dst_dt: Dtype) -> Result<(), String> {
    if src_dt == dst_dt {
        out.copy_from_slice(input);
        return Ok(());
    }

    macro_rules! cast {
        ($S:ty, $D:ty, $convert:expr) => {{
            let src = as_slice::<$S>(input);
            let n = src.len();
            let mut tmp = Vec::with_capacity(n);
            for &v in src {
                tmp.push($convert(v));
            }
            write_slice(out, &tmp);
        }};
    }

    use Dtype::*;
    // Float→int casts truncate toward zero (matches XLA/Nx default,
    // not round-half-away-from-zero). `as` on f32→iN already does this.
    match (src_dt, dst_dt) {
        (F32, S8)  => cast!(f32, i8,  |v: f32| v.clamp(-128.0, 127.0) as i8),
        (F32, U8)  => cast!(f32, u8,  |v: f32| v.clamp(0.0, 255.0) as u8),
        (F32, S16) => cast!(f32, i16, |v: f32| v.clamp(-32768.0, 32767.0) as i16),
        (F32, U16) => cast!(f32, u16, |v: f32| v.clamp(0.0, 65535.0) as u16),
        (F32, S32) => cast!(f32, i32, |v: f32| v as i32),
        (F32, U32) => cast!(f32, u32, |v: f32| v.max(0.0) as u32),
        (F32, S64) => cast!(f32, i64, |v: f32| v as i64),
        (F32, U64) => cast!(f32, u64, |v: f32| v.max(0.0) as u64),
        (F32, F64) => cast!(f32, f64, |v: f32| v as f64),

        (F64, F32) => cast!(f64, f32, |v: f64| v as f32),
        (F64, S64) => cast!(f64, i64, |v: f64| v as i64),
        (F64, S32) => cast!(f64, i32, |v: f64| v as i32),

        (S8,  F32) => cast!(i8,  f32, |v: i8|  v as f32),
        (U8,  F32) => cast!(u8,  f32, |v: u8|  v as f32),
        (S16, F32) => cast!(i16, f32, |v: i16| v as f32),
        (U16, F32) => cast!(u16, f32, |v: u16| v as f32),
        (S32, F32) => cast!(i32, f32, |v: i32| v as f32),
        (U32, F32) => cast!(u32, f32, |v: u32| v as f32),
        (S64, F32) => cast!(i64, f32, |v: i64| v as f32),
        (U64, F32) => cast!(u64, f32, |v: u64| v as f32),

        (S64, S32) => cast!(i64, i32, |v: i64| v as i32),
        (S32, S64) => cast!(i32, i64, |v: i32| v as i64),
        (S64, U8)  => cast!(i64, u8,  |v: i64| v.clamp(0, 255) as u8),
        (U8,  S64) => cast!(u8,  i64, |v: u8|  v as i64),
        (S32, U8)  => cast!(i32, u8,  |v: i32| v.clamp(0, 255) as u8),
        (S8,  S32) => cast!(i8,  i32, |v: i8|  v as i32),

        (F32, BOOL) => cast!(f32, u8, |v: f32| if v != 0.0 { 1u8 } else { 0u8 }),
        (S64, BOOL) => cast!(i64, u8, |v: i64| if v != 0 { 1u8 } else { 0u8 }),
        (BOOL, F32) => cast!(u8, f32, |v: u8| v as f32),
        (BOOL, S64) => cast!(u8, i64, |v: u8| v as i64),

        _ => return Err(format!("as_type: {:?} → {:?} not implemented", src_dt, dst_dt)),
    }

    Ok(())
}

// ---------------------------------------------------------------
// clip: clamp each element into [min, max]. Min and max are
// scalar f64 (gets cast to the actual dtype) — matches Nx's
// `Nx.clip(t, lo, hi)` API.
// ---------------------------------------------------------------

pub fn clip(input: &[u8], out: &mut [u8], dt: Dtype, min: f64, max: f64) -> Result<(), String> {
    macro_rules! clip_t {
        ($T:ty, $min:expr, $max:expr) => {{
            let src = as_slice::<$T>(input);
            let mut tmp: Vec<$T> = Vec::with_capacity(src.len());
            let mn = $min as $T;
            let mx = $max as $T;
            for &v in src {
                tmp.push(if v < mn { mn } else if v > mx { mx } else { v });
            }
            write_slice(out, &tmp);
        }};
    }

    use Dtype::*;
    match dt {
        F32 => clip_t!(f32, min as f32, max as f32),
        F64 => clip_t!(f64, min, max),
        S8  => clip_t!(i8, min, max),
        S16 => clip_t!(i16, min, max),
        S32 => clip_t!(i32, min, max),
        S64 => clip_t!(i64, min, max),
        U8  => clip_t!(u8, min, max),
        U16 => clip_t!(u16, min, max),
        U32 => clip_t!(u32, min, max),
        U64 => clip_t!(u64, min, max),
        _ => return Err(format!("clip: dtype {:?} not implemented", dt)),
    }

    Ok(())
}

// ---------------------------------------------------------------
// pad: per-axis (low, high, interior) padding with a fill value.
// Interior padding (insert N zeros between every adjacent element)
// is rare; we support it but the typical use is low/high only.
// ---------------------------------------------------------------

pub fn pad(
    input: &[u8],
    out: &mut [u8],
    in_shape: &[usize],
    out_shape: &[usize],
    pad_config: &[(i64, i64, i64)],
    fill: &[u8],
    elem_size: usize,
) {
    debug_assert_eq!(in_shape.len(), out_shape.len());
    debug_assert_eq!(pad_config.len(), in_shape.len());
    debug_assert_eq!(fill.len(), elem_size);

    // First, fill the entire output with the pad value.
    let n_out: usize = out_shape.iter().product();
    for i in 0..n_out {
        let dst_off = i * elem_size;
        out[dst_off..dst_off + elem_size].copy_from_slice(fill);
    }

    // Walk the input in row-major order and place each element at
    // its mapped position in the output.
    let rank = in_shape.len();
    let n_in: usize = in_shape.iter().product();

    let in_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * in_shape[i + 1];
        }
        s
    };
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    for flat in 0..n_in {
        let mut rem = flat;
        let mut out_flat = 0usize;
        for axis in 0..rank {
            let idx = rem / in_strides[axis];
            rem %= in_strides[axis];
            let (lo, _hi, interior) = pad_config[axis];
            let mapped = lo + (idx as i64) * (1 + interior);
            if mapped < 0 || (mapped as usize) >= out_shape[axis] {
                // Outside the output window — skip this input element.
                // (Negative padding can crop.)
                out_flat = usize::MAX;
                break;
            }
            out_flat += (mapped as usize) * out_strides[axis];
        }

        if out_flat != usize::MAX {
            let src_off = flat * elem_size;
            let dst_off = out_flat * elem_size;
            out[dst_off..dst_off + elem_size]
                .copy_from_slice(&input[src_off..src_off + elem_size]);
        }
    }
}

// ---------------------------------------------------------------
// gather along axis 0: out[i] = input[indices[i]]. Indices are
// s64. The full Nx.gather supports arbitrary index shapes; this
// covers the embedding-lookup case which is what we actually need
// at decode time.
// ---------------------------------------------------------------

pub fn gather_axis0(
    input: &[u8],
    indices: &[i64],
    out: &mut [u8],
    n_rows: usize,
    row_bytes: usize,
) -> Result<(), String> {
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || (idx as usize) >= n_rows {
            return Err(format!(
                "gather_axis0: index {} out of range [0, {})",
                idx, n_rows
            ));
        }
        let src_off = (idx as usize) * row_bytes;
        let dst_off = i * row_bytes;
        out[dst_off..dst_off + row_bytes]
            .copy_from_slice(&input[src_off..src_off + row_bytes]);
    }
    Ok(())
}

// ---------------------------------------------------------------
// stack: concatenate N tensors along a new leading axis. Each
// tensor must have identical shape and dtype.
// ---------------------------------------------------------------

pub fn stack_axis0(tensors: &[&[u8]], out: &mut [u8], tensor_bytes: usize) {
    for (i, t) in tensors.iter().enumerate() {
        let dst_off = i * tensor_bytes;
        out[dst_off..dst_off + tensor_bytes].copy_from_slice(&t[..tensor_bytes]);
    }
}

// ---------------------------------------------------------------
// Embedding ops — the building blocks for on-device semantic
// search / RAG / retrieval.
// ---------------------------------------------------------------

/// L2-normalise rows of an `(n, d)` matrix in place. Cheap pre-step
/// so cosine sim becomes a plain dot product.
pub fn l2_normalize_rows_f32(data: &mut [f32], n: usize, d: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(data.len(), n * d);
    data.par_chunks_mut(d).for_each(|row| {
        let mut sum_sq = 0.0f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let inv = if sum_sq > 0.0 {
            1.0 / sum_sq.sqrt()
        } else {
            1.0
        };
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

/// Cosine similarity between a query vector `q` (length `d`) and
/// every row of `corpus` (shape `(n, d)`). Returns `(n,)` scores.
///
/// Both inputs must be pre-L2-normalised — this then collapses to
/// `q · corpus[i]`. We hand off to `gemm` for the dot product so the
/// throughput matches our matmul kernels.
pub fn cosine_similarity_f32(
    q: &[f32],
    corpus: &[f32],
    n: usize,
    d: usize,
) -> Vec<f32> {
    debug_assert_eq!(q.len(), d);
    debug_assert_eq!(corpus.len(), n * d);

    // Treat q as (1, d), corpus as (n, d). Result (1, n) = q · corpus^T.
    // Reuse the batched matmul wrapper from shape_ops.
    crate::shape_ops::batched_matmul_f32(q, corpus, 1, 1, n, d, true)
        .expect("cosine_similarity dimensions checked above")
}

/// Scatter-add f32 updates into `tensor` at the given indices.
/// `tensor` is flat-rank-1 here (after caller reshape). Mirrors
/// `Nx.indexed_add/3` semantics: out-of-bounds indices wrap, repeated
/// indices accumulate.
pub fn indexed_add_f32(tensor: &[f32], indices: &[i64], updates: &[f32]) -> Result<Vec<f32>, String> {
    if indices.len() != updates.len() {
        return Err(format!(
            "indexed_add: indices ({}) != updates ({})",
            indices.len(),
            updates.len()
        ));
    }
    let n = tensor.len();
    if n == 0 {
        return Ok(tensor.to_vec());
    }

    let mut out = tensor.to_vec();
    for (i, &idx) in indices.iter().enumerate() {
        let pos = idx.rem_euclid(n as i64) as usize;
        out[pos] += updates[i];
    }
    Ok(out)
}

/// Scatter-overwrite f32 updates into `tensor` at the given
/// indices. Last write wins for repeated indices.
pub fn indexed_put_f32(tensor: &[f32], indices: &[i64], updates: &[f32]) -> Result<Vec<f32>, String> {
    if indices.len() != updates.len() {
        return Err(format!(
            "indexed_put: indices ({}) != updates ({})",
            indices.len(),
            updates.len()
        ));
    }
    let n = tensor.len();
    if n == 0 {
        return Ok(tensor.to_vec());
    }

    let mut out = tensor.to_vec();
    for (i, &idx) in indices.iter().enumerate() {
        let pos = idx.rem_euclid(n as i64) as usize;
        out[pos] = updates[i];
    }
    Ok(out)
}

/// Return the indices of the top `k` highest values in `scores`,
/// ordered descending. O(n log k) via a min-heap.
pub fn top_k_indices_f32(scores: &[f32], k: usize) -> Vec<i32> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if k == 0 || scores.is_empty() {
        return Vec::new();
    }

    #[derive(PartialEq)]
    struct Entry(f32, i32);
    impl Eq for Entry {}
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            // Min-heap: reverse the natural ordering on f32.
            other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);
    for (i, &s) in scores.iter().enumerate() {
        if heap.len() < k {
            heap.push(Entry(s, i as i32));
        } else if let Some(top) = heap.peek() {
            if s > top.0 {
                heap.pop();
                heap.push(Entry(s, i as i32));
            }
        }
    }

    let mut out: Vec<(f32, i32)> = heap.into_iter().map(|e| (e.0, e.1)).collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out.into_iter().map(|(_, i)| i).collect()
}

// ---------------------------------------------------------------
// sort + argsort along the last axis. Stable sort via Rust's std.
// Directions: 0 = ascending, 1 = descending.
//
// The common LLM use is `Nx.argsort(logits, direction: :desc)` for
// top-k sampling; the CV use is sort + slice for NMS.
// ---------------------------------------------------------------

pub fn sort_axis_f32(input: &[u8], out: &mut [u8], outer: usize, inner: usize, descending: bool) {
    let src = as_slice::<f32>(input);
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, outer * inner)
    };

    dst.par_chunks_mut(inner).enumerate().for_each(|(o, row)| {
        let src_row = &src[o * inner..(o + 1) * inner];
        row.copy_from_slice(src_row);
        if descending {
            row.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
    });
}

pub fn argsort_axis_f32(input: &[u8], out: &mut [u8], outer: usize, inner: usize, descending: bool) {
    let src = as_slice::<f32>(input);
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut i32, outer * inner)
    };

    dst.par_chunks_mut(inner).enumerate().for_each(|(o, row)| {
        let src_row = &src[o * inner..(o + 1) * inner];
        let mut idx: Vec<u32> = (0..inner as u32).collect();
        if descending {
            idx.sort_by(|&a, &b| {
                src_row[b as usize]
                    .partial_cmp(&src_row[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            idx.sort_by(|&a, &b| {
                src_row[a as usize]
                    .partial_cmp(&src_row[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for (i, &v) in idx.iter().enumerate() {
            row[i] = v as i32;
        }
    });
}

// ---------------------------------------------------------------
// Bool reductions: all / any / product / sum over a u8-boolean
// tensor (Nx represents booleans as u8 elements 0 or 1).
//
// `all` returns 1 iff every element is non-zero.
// `any` returns 1 iff any element is non-zero.
// Result type matches Nx convention: u8 for all/any, f32 for product.
// ---------------------------------------------------------------

pub fn reduce_all_u8(input: &[u8]) -> u8 {
    if input.iter().all(|&v| v != 0) { 1 } else { 0 }
}

pub fn reduce_any_u8(input: &[u8]) -> u8 {
    if input.iter().any(|&v| v != 0) { 1 } else { 0 }
}

pub fn reduce_product_f32(input: &[u8]) -> f32 {
    let src = as_slice::<f32>(input);
    src.iter().copied().product()
}

// ---------------------------------------------------------------
// reverse along arbitrary axes. We reverse one axis at a time;
// most callers reverse a single axis (image flip, sequence
// reversal), and the multi-axis case is rare enough that the per-
// pass copy is fine.
// ---------------------------------------------------------------

pub fn reverse_axes(
    input: &[u8],
    out: &mut [u8],
    shape: &[usize],
    axes: &[usize],
    elem_size: usize,
) {
    let rank = shape.len();
    let n: usize = shape.iter().product();

    // Pre-compute per-axis stride in elements.
    let strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };

    let reverse_set: std::collections::HashSet<usize> = axes.iter().copied().collect();

    for flat in 0..n {
        let mut rem = flat;
        let mut out_flat = 0usize;
        for axis in 0..rank {
            let idx = rem / strides[axis];
            rem %= strides[axis];
            let mapped = if reverse_set.contains(&axis) {
                shape[axis] - 1 - idx
            } else {
                idx
            };
            out_flat += mapped * strides[axis];
        }
        let src_off = flat * elem_size;
        let dst_off = out_flat * elem_size;
        out[dst_off..dst_off + elem_size]
            .copy_from_slice(&input[src_off..src_off + elem_size]);
    }
}

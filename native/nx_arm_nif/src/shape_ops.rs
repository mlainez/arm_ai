// Fast CPU shape ops on raw bytes — broadcast, transpose, concatenate,
// batched matmul, plus f32 elementwise binary/unary.
//
// Why these are CPU and not GPU: a5xx Rusticl runs the elementwise GPU
// kernel at ~2 MFLOPS (dispatch overhead + single-thread-per-cell
// pattern + non-tiled memory access). Cortex-A73 NEON via the auto-
// vectoriser is ~50× faster for the same compute. The GPU↔CPU
// roundtrip is cheap (sub-ms for the tensor sizes ViT-tiny uses).
//
// These exist so NxCL.Backend can keep tensors resident on the GPU but
// still avoid the pure-Elixir BinaryBackend fallback path for shape
// manipulations that Rusticl/a5xx can't (yet) do natively. The flow is:
//
//   1. backend.ex reads input bytes from the GPU
//   2. calls one of these NIFs
//   3. writes the output bytes back to the GPU
//
// Rust + rayon doing the actual work is 10–100× faster than pure-Elixir
// Nx.BinaryBackend, so even with the roundtrip we come out well ahead.

use rayon::prelude::*;

/// Broadcast `input` of `in_shape` up to `out_shape` according to `axes`
/// (Nx semantics: `axes[i]` is the output axis that input axis `i` lands
/// on; in_shape[i] must equal 1 or out_shape[axes[i]]).
///
/// Dtype-agnostic via `element_size` (bytes per element). Output length
/// is `prod(out_shape) * element_size`.
pub fn broadcast(
    input: &[u8],
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &[usize],
    element_size: usize,
) -> Result<Vec<u8>, String> {
    if in_shape.len() != axes.len() {
        return Err(format!(
            "broadcast: axes length {} != in_shape rank {}",
            axes.len(),
            in_shape.len()
        ));
    }

    let out_rank = out_shape.len();
    let out_total: usize = out_shape.iter().product();
    let in_total: usize = in_shape.iter().product();

    if input.len() != in_total * element_size {
        return Err(format!(
            "broadcast: input bytes {} != prod(in_shape) {} * element_size {}",
            input.len(),
            in_total,
            element_size
        ));
    }

    // Precompute strides per output axis (element units, row-major).
    let mut out_strides = vec![0usize; out_rank];
    if out_rank > 0 {
        out_strides[out_rank - 1] = 1;
        for k in (0..out_rank - 1).rev() {
            out_strides[k] = out_strides[k + 1] * out_shape[k + 1];
        }
    }

    // Per-OUTPUT-axis contribution to the input flat index. If an input
    // axis maps to this output axis with in_shape > 1, the contribution
    // is that input axis's row-major stride; if it has in_shape == 1
    // (broadcast across) it's 0. Axes that don't appear in `axes` at all
    // also contribute 0 (the input has no such axis).
    let mut in_strides = vec![0usize; in_shape.len()];
    if !in_shape.is_empty() {
        in_strides[in_shape.len() - 1] = 1;
        for k in (0..in_shape.len() - 1).rev() {
            in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
        }
    }

    let mut contrib = vec![0isize; out_rank];
    for (i, &out_axis) in axes.iter().enumerate() {
        if out_axis >= out_rank {
            return Err(format!(
                "broadcast: axes[{}] = {} out of range for out_rank {}",
                i, out_axis, out_rank
            ));
        }
        if in_shape[i] == 1 {
            contrib[out_axis] = 0;
        } else if in_shape[i] == out_shape[out_axis] {
            contrib[out_axis] = in_strides[i] as isize;
        } else {
            return Err(format!(
                "broadcast: in_shape[{}] = {} cannot broadcast to out_shape[{}] = {}",
                i, in_shape[i], out_axis, out_shape[out_axis]
            ));
        }
    }

    let mut output = vec![0u8; out_total * element_size];

    // Per-chunk parallel fill. Each chunk gets its own slice; no
    // contention. Tune chunk count by output total to keep overhead low.
    let chunk_size = ((out_total + 31) / 32).max(1024).min(out_total);

    output
        .par_chunks_mut(chunk_size * element_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            for j in 0..(chunk.len() / element_size) {
                let oi = start + j;
                let mut src = 0usize;
                let mut rem = oi;
                for k in 0..out_rank {
                    let s = out_strides[k];
                    let idx_k = if s == 0 { 0 } else { rem / s };
                    rem -= idx_k * s;
                    src = src.wrapping_add((contrib[k] as usize).wrapping_mul(idx_k));
                }
                let src_byte = src * element_size;
                let dst_byte = j * element_size;
                chunk[dst_byte..dst_byte + element_size]
                    .copy_from_slice(&input[src_byte..src_byte + element_size]);
            }
        });

    Ok(output)
}

/// Generic n-D transpose. `axes[i]` is the input axis that becomes
/// output axis `i` (Nx semantics, matching `Nx.transpose(t, axes: ...)`).
pub fn transpose(
    input: &[u8],
    in_shape: &[usize],
    axes: &[usize],
    element_size: usize,
) -> Result<Vec<u8>, String> {
    let rank = in_shape.len();
    if axes.len() != rank {
        return Err(format!(
            "transpose: axes len {} != rank {}",
            axes.len(),
            rank
        ));
    }

    let total: usize = in_shape.iter().product();
    if input.len() != total * element_size {
        return Err(format!(
            "transpose: input bytes {} != prod(shape) {} * element_size {}",
            input.len(),
            total,
            element_size
        ));
    }

    // Input row-major strides.
    let mut in_strides = vec![0usize; rank];
    if rank > 0 {
        in_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
        }
    }

    // Output shape and strides: out_shape[i] = in_shape[axes[i]].
    let out_shape: Vec<usize> = axes.iter().map(|&a| in_shape[a]).collect();
    let mut out_strides = vec![0usize; rank];
    if rank > 0 {
        out_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            out_strides[k] = out_strides[k + 1] * out_shape[k + 1];
        }
    }

    // For each OUTPUT element, compute the source flat index. axes[i] is
    // the input axis at output position i; so out flat → out multi-index
    // → permute through axes → input multi-index → input flat.
    let mut output = vec![0u8; total * element_size];

    let chunk_size = ((total + 31) / 32).max(1024).min(total.max(1));

    output
        .par_chunks_mut(chunk_size * element_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            for j in 0..(chunk.len() / element_size) {
                let oi = start + j;
                let mut rem = oi;
                let mut src = 0usize;
                for i in 0..rank {
                    let s = out_strides[i];
                    let idx_i = if s == 0 { 0 } else { rem / s };
                    rem -= idx_i * s;
                    // input axis `axes[i]` gets value `idx_i`
                    src += in_strides[axes[i]] * idx_i;
                }
                let src_byte = src * element_size;
                let dst_byte = j * element_size;
                chunk[dst_byte..dst_byte + element_size]
                    .copy_from_slice(&input[src_byte..src_byte + element_size]);
            }
        });

    Ok(output)
}

/// Concatenate `tensors` along `axis`. All tensors must share the same
/// shape outside the concatenation axis. `shapes` is parallel to
/// `tensors`; `axis` is the axis to concat along; `element_size` is the
/// byte width of one element.
pub fn concatenate(
    tensors: &[&[u8]],
    shapes: &[Vec<usize>],
    axis: usize,
    element_size: usize,
) -> Result<Vec<u8>, String> {
    if tensors.is_empty() {
        return Err("concatenate: empty input list".into());
    }
    let rank = shapes[0].len();
    if axis >= rank {
        return Err(format!(
            "concatenate: axis {} out of range for rank {}",
            axis, rank
        ));
    }
    for s in shapes.iter() {
        if s.len() != rank {
            return Err("concatenate: rank mismatch across tensors".into());
        }
        for k in 0..rank {
            if k != axis && s[k] != shapes[0][k] {
                return Err(format!(
                    "concatenate: shape mismatch on axis {} (axis {} differs)",
                    k, axis
                ));
            }
        }
    }

    // Outer = product of axes [0, axis) — number of outer iterations.
    // Inner = product of axes (axis, rank) * element_size — bytes per
    // axis-element per tensor per outer iteration.
    let outer: usize = shapes[0][..axis].iter().product();
    let inner_elems: usize = shapes[0][axis + 1..].iter().product();
    let inner_bytes = inner_elems * element_size;

    // Total axis length in the output:
    let axis_total: usize = shapes.iter().map(|s| s[axis]).sum();
    let total_bytes = outer * axis_total * inner_bytes;

    let mut output = vec![0u8; total_bytes];

    // For each outer slice o, copy tensor t's chunk of bytes
    //   tensors[t][o * shapes[t][axis] * inner_bytes ..
    //              (o+1) * shapes[t][axis] * inner_bytes]
    // into
    //   output[o * axis_total * inner_bytes + offset_t * inner_bytes ..]
    let mut tensor_offsets_elems = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for s in shapes.iter() {
        tensor_offsets_elems.push(off);
        off += s[axis];
    }

    output
        .par_chunks_mut(axis_total * inner_bytes)
        .enumerate()
        .for_each(|(o, slot)| {
            for (t, &input) in tensors.iter().enumerate() {
                let n_axis_t = shapes[t][axis];
                let src_start = o * n_axis_t * inner_bytes;
                let src_end = src_start + n_axis_t * inner_bytes;
                let dst_start = tensor_offsets_elems[t] * inner_bytes;
                let dst_end = dst_start + n_axis_t * inner_bytes;
                slot[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
            }
        });

    Ok(output)
}

/// Batched f32 matmul. `left` is [B, M, K], `right` is either
/// [B, K, N] (when `right_transposed == false`) or [B, N, K]
/// (when `right_transposed == true`). Output is [B, M, N].
///
/// Standard layout uses a hand-written 4-row × 4-col NEON register
/// kernel (16 accumulators kept in V registers across the K loop).
/// Transposed layout (Q @ K^T) uses a 4-way dot-product NEON kernel.
pub fn batched_matmul_f32(
    left: &[f32],
    right: &[f32],
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    right_transposed: bool,
) -> Result<Vec<f32>, String> {
    let need_left = b * m * k;
    let need_right = b * k * n;
    if left.len() != need_left {
        return Err(format!(
            "batched_matmul_f32: left len {} != B*M*K = {}",
            left.len(),
            need_left
        ));
    }
    if right.len() != need_right {
        return Err(format!(
            "batched_matmul_f32: right len {} != B*K*N (or B*N*K) = {}",
            right.len(),
            need_right
        ));
    }

    let mut out = vec![0.0f32; b * m * n];

    // Parallelise across batches. Each batch slice is independent;
    // the inner tiled kernel runs single-threaded within one batch so
    // its register/cache locality isn't shared with other threads.
    //
    // For b=1 (the dominant Bumblebee case), this means single-threaded
    // for the inner matmul — we make up for it by splitting the M-tile
    // parallelism inside `matmul_2d_*` below.
    out.par_chunks_mut(m * n)
        .enumerate()
        .for_each(|(batch, c_slice)| {
            let a_slice = &left[batch * m * k..(batch + 1) * m * k];

            if right_transposed {
                let r_slice = &right[batch * n * k..(batch + 1) * n * k];
                matmul_2d_neon_qkt(a_slice, r_slice, c_slice, m, n, k);
            } else {
                let r_slice = &right[batch * k * n..(batch + 1) * k * n];
                matmul_2d_neon(a_slice, r_slice, c_slice, m, n, k);
            }
        });

    Ok(out)
}

/// Standard 2-D matmul: C[m, n] = A[m, k] @ B[k, n].
///
/// 4×4 register-tiled inner kernel via NEON intrinsics on aarch64; the
/// outer loop over row-tiles is rayon-parallel. Edges where M or N
/// aren't multiples of 4 are handled by scalar fallback for those
/// remaining cells only.
fn matmul_2d_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let n_tiles = n / 4;
    let n_tail_start = n_tiles * 4;
    let m_tiles_count = m / 4;
    let m_tail_start = m_tiles_count * 4;

    // Transport the mut pointer across threads as a usize — guarantees
    // Send + Sync without unsafe-impl shenanigans. Disjointness across
    // the (mi, nj) tile space ensures no two threads ever target the
    // same output cell, so race-free in practice.
    let c_addr = c.as_mut_ptr() as usize;

    (0..m_tiles_count).into_par_iter().for_each(|mi| {
        let row_base = mi * 4;
        let c_ptr = c_addr as *mut f32;
        unsafe {
            for nj in 0..n_tiles {
                let col_base = nj * 4;
                matmul_kernel_4x4(a, b, c_ptr, row_base, col_base, n, k);
            }
            // N tail (n % 4 columns) for these 4 rows.
            for col in n_tail_start..n {
                for row_off in 0..4 {
                    let row = row_base + row_off;
                    let a_row = std::slice::from_raw_parts(a.as_ptr().add(row * k), k);
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += a_row[kk] * b[kk * n + col];
                    }
                    *c_ptr.add(row * n + col) = acc;
                }
            }
        }
    });

    // M tail (m % 4 rows): scalar fallback, runs serially after the
    // parallel section — small enough to not matter.
    for row in m_tail_start..m {
        let a_row = &a[row * k..row * k + k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_row[kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

/// 4-row × 4-col register-tiled NEON kernel.
///
/// Computes C[row_base..row_base+4, col_base..col_base+4] = A·B
/// where A is row-major [M, K], B is row-major [K, N].
///
/// Inner loop reads one 4-wide B vector per K-step and broadcasts
/// 4 A scalars (one per output row) via vfmaq_n_f32, accumulating
/// into 4 vector registers held the whole K loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matmul_kernel_4x4(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
) {
    use core::arch::aarch64::*;

    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    let a_row0 = a.as_ptr().add(row_base * k);
    let a_row1 = a.as_ptr().add((row_base + 1) * k);
    let a_row2 = a.as_ptr().add((row_base + 2) * k);
    let a_row3 = a.as_ptr().add((row_base + 3) * k);
    let b_ptr = b.as_ptr();

    for kk in 0..k {
        let b_vec = vld1q_f32(b_ptr.add(kk * n + col_base));
        c0 = vfmaq_n_f32(c0, b_vec, *a_row0.add(kk));
        c1 = vfmaq_n_f32(c1, b_vec, *a_row1.add(kk));
        c2 = vfmaq_n_f32(c2, b_vec, *a_row2.add(kk));
        c3 = vfmaq_n_f32(c3, b_vec, *a_row3.add(kk));
    }

    vst1q_f32(c_ptr.add(row_base * n + col_base), c0);
    vst1q_f32(c_ptr.add((row_base + 1) * n + col_base), c1);
    vst1q_f32(c_ptr.add((row_base + 2) * n + col_base), c2);
    vst1q_f32(c_ptr.add((row_base + 3) * n + col_base), c3);
}

/// Scalar fallback for non-aarch64 hosts (e.g. CI on x86_64).
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn matmul_kernel_4x4(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
) {
    for row_off in 0..4 {
        let row = row_base + row_off;
        let a_row = &a[row * k..row * k + k];
        for col_off in 0..4 {
            let col = col_base + col_off;
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_row[kk] * b[kk * n + col];
            }
            *c_ptr.add(row * n + col) = acc;
        }
    }
}

/// Q @ K^T pattern: C[m, n] = A[m, k] · B[n, k] (contract on last axis
/// of both). NEON-vectorised 4-way dot product per (row, col) pair:
/// for each (row, col) we load 4 K-values from both, FMA into a vector
/// accumulator, then horizontal-add at the end.
fn matmul_2d_neon_qkt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let c_addr = c.as_mut_ptr() as usize;

    (0..m).into_par_iter().for_each(|row| {
        let c_ptr = c_addr as *mut f32;
        unsafe {
            let a_row = a.as_ptr().add(row * k);
            for col in 0..n {
                let b_row = b.as_ptr().add(col * k);
                let acc = dot4_neon(a_row, b_row, k);
                *c_ptr.add(row * n + col) = acc;
            }
        }
    });
}

/// 4-wide NEON dot product of two length-K f32 arrays. Tail handled
/// scalar.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot4_neon(a: *const f32, b: *const f32, k: usize) -> f32 {
    use core::arch::aarch64::*;

    let mut acc = vdupq_n_f32(0.0);
    let k_tiles = k / 4;
    let k_tail_start = k_tiles * 4;

    for kk in 0..k_tiles {
        let av = vld1q_f32(a.add(kk * 4));
        let bv = vld1q_f32(b.add(kk * 4));
        acc = vfmaq_f32(acc, av, bv);
    }

    let mut sum = vaddvq_f32(acc);
    for kk in k_tail_start..k {
        sum += *a.add(kk) * *b.add(kk);
    }
    sum
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn dot4_neon(a: *const f32, b: *const f32, k: usize) -> f32 {
    let mut acc = 0.0f32;
    for kk in 0..k {
        acc += *a.add(kk) * *b.add(kk);
    }
    acc
}

// ── f32 elementwise (CPU NEON via auto-vectoriser) ──────

/// Same-shape elementwise binary op on f32 arrays.
pub fn elementwise_binary_f32(op: &str, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err(format!("len mismatch: {} vs {}", a.len(), b.len()));
    }
    let n = a.len();
    let mut out = vec![0.0f32; n];

    // Parallelise across cores at a coarse granularity. Inside each
    // chunk a plain `for` loop autovectorises to NEON FMA/ADD on
    // aarch64 (we don't need explicit intrinsics — the compiler
    // generates the right code for tight in-order f32 arithmetic).
    let chunk = ((n + 7) / 8).max(8192).min(n.max(1));

    match op {
        "add" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j] + b[s + j];
            }
        }),
        "subtract" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j] - b[s + j];
            }
        }),
        "multiply" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j] * b[s + j];
            }
        }),
        "divide" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j] / b[s + j];
            }
        }),
        "max" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j].max(b[s + j]);
            }
        }),
        "min" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j].min(b[s + j]);
            }
        }),
        "pow" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j].powf(b[s + j]);
            }
        }),
        "atan2" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j].atan2(b[s + j]);
            }
        }),
        "remainder" => out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
            let s = i * chunk;
            for j in 0..slot.len() {
                slot[j] = a[s + j].rem_euclid(b[s + j]);
            }
        }),
        other => return Err(format!("unknown binary op: {}", other)),
    }

    Ok(out)
}

/// Scalar-broadcast binary op: `tensor OP scalar` (`side == "ab"`) or
/// `scalar OP tensor` (`side == "ba"`).
pub fn scalar_binary_f32(op: &str, side: &str, a: &[f32], scalar: f32) -> Result<Vec<f32>, String> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let chunk = ((n + 7) / 8).max(8192).min(n.max(1));

    macro_rules! run {
        ($body:expr) => {
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                let s = i * chunk;
                for j in 0..slot.len() {
                    let x = a[s + j];
                    slot[j] = $body(x, scalar);
                }
            })
        };
    }
    macro_rules! run_rev {
        ($body:expr) => {
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                let s = i * chunk;
                for j in 0..slot.len() {
                    let x = a[s + j];
                    slot[j] = $body(scalar, x);
                }
            })
        };
    }

    let forward = side == "ab";

    match (op, forward) {
        ("add", _) => run!(|x: f32, y: f32| x + y),
        ("subtract", true) => run!(|x: f32, y: f32| x - y),
        ("subtract", false) => run_rev!(|x: f32, y: f32| x - y),
        ("multiply", _) => run!(|x: f32, y: f32| x * y),
        ("divide", true) => run!(|x: f32, y: f32| x / y),
        ("divide", false) => run_rev!(|x: f32, y: f32| x / y),
        ("max", _) => run!(|x: f32, y: f32| x.max(y)),
        ("min", _) => run!(|x: f32, y: f32| x.min(y)),
        ("pow", true) => run!(|x: f32, y: f32| x.powf(y)),
        ("pow", false) => run_rev!(|x: f32, y: f32| x.powf(y)),
        (other, _) => return Err(format!("unknown scalar op: {}", other)),
    }

    Ok(out)
}

/// Elementwise unary op on f32 array.
pub fn elementwise_unary_f32(op: &str, a: &[f32]) -> Result<Vec<f32>, String> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let chunk = ((n + 7) / 8).max(8192).min(n.max(1));

    macro_rules! run {
        ($body:expr) => {
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                let s = i * chunk;
                for j in 0..slot.len() {
                    slot[j] = $body(a[s + j]);
                }
            })
        };
    }

    match op {
        "negate" => run!(|x: f32| -x),
        "exp" => run!(|x: f32| x.exp()),
        "log" => run!(|x: f32| x.ln()),
        "tanh" => run!(|x: f32| x.tanh()),
        "sigmoid" => run!(|x: f32| 1.0 / (1.0 + (-x).exp())),
        "abs" => run!(|x: f32| x.abs()),
        "sqrt" => run!(|x: f32| x.sqrt()),
        "rsqrt" => run!(|x: f32| 1.0 / x.sqrt()),
        "cbrt" => run!(|x: f32| x.cbrt()),
        "expm1" => run!(|x: f32| x.exp_m1()),
        "log1p" => run!(|x: f32| x.ln_1p()),
        "sin" => run!(|x: f32| x.sin()),
        "cos" => run!(|x: f32| x.cos()),
        "tan" => run!(|x: f32| x.tan()),
        "asin" => run!(|x: f32| x.asin()),
        "acos" => run!(|x: f32| x.acos()),
        "atan" => run!(|x: f32| x.atan()),
        "sinh" => run!(|x: f32| x.sinh()),
        "cosh" => run!(|x: f32| x.cosh()),
        "ceil" => run!(|x: f32| x.ceil()),
        "floor" => run!(|x: f32| x.floor()),
        "round" => run!(|x: f32| x.round()),
        "sign" => run!(|x: f32| x.signum()),
        "erf" => run!(erf_f32),
        "erfc" => run!(|x: f32| 1.0 - erf_f32(x)),
        "asinh" => run!(|x: f32| x.asinh()),
        "acosh" => run!(|x: f32| x.acosh()),
        "atanh" => run!(|x: f32| x.atanh()),
        other => return Err(format!("unknown unary op: {}", other)),
    }

    Ok(out)
}

// Abramowitz & Stegun 7.1.26 — max abs error ~1.5e-7, well within f32.
// erf is what Axon's GELU calls through Nx.erf, so we need a fast
// builtin here (Rust std has no erf).
fn erf_f32(x: f32) -> f32 {
    let a1: f32 = 0.254829592;
    let a2: f32 = -0.284496736;
    let a3: f32 = 1.421413741;
    let a4: f32 = -1.453152027;
    let a5: f32 = 1.061405429;
    let p: f32 = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Reduce along the LAST axis. Input viewed as [n_outer × inner];
/// output is n_outer f32s, each the reduction of `inner` consecutive
/// values.
pub fn reduce_axis_f32(op: &str, input: &[f32], n_outer: usize, inner: usize) -> Result<Vec<f32>, String> {
    if input.len() != n_outer * inner {
        return Err(format!(
            "reduce_axis: len {} != n_outer*inner = {}",
            input.len(),
            n_outer * inner
        ));
    }
    let mut out = vec![0.0f32; n_outer];

    let init = match op {
        "sum" => 0.0,
        "max" => f32::NEG_INFINITY,
        "min" => f32::INFINITY,
        other => return Err(format!("unknown reduce op: {}", other)),
    };

    match op {
        "sum" => out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &input[i * inner..(i + 1) * inner];
            let mut acc = init;
            for &v in row {
                acc += v;
            }
            *o = acc;
        }),
        "max" => out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &input[i * inner..(i + 1) * inner];
            let mut acc = init;
            for &v in row {
                if v > acc {
                    acc = v;
                }
            }
            *o = acc;
        }),
        "min" => out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &input[i * inner..(i + 1) * inner];
            let mut acc = init;
            for &v in row {
                if v < acc {
                    acc = v;
                }
            }
            *o = acc;
        }),
        _ => unreachable!(),
    }

    Ok(out)
}

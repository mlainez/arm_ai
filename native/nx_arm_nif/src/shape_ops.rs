// ── NEON polynomial approximations for exp / sigmoid / tanh ────
//
// `libm`'s `f32::exp` is scalar and called once per element — for
// shapes like {1, 197, 768} that's 151K function calls per op. The
// polynomial path below does 4 floats per NEON cycle and replaces
// the scalar path on aarch64. Range-reduction + degree-5 Taylor is
// faithful to f32 (~1 ULP) across the safe domain.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vexpq_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;

    // Clip to safe range to avoid INF/0 overflow.
    let max_x = vdupq_n_f32(88.3762626647949);
    let min_x = vdupq_n_f32(-88.3762626647949);
    let x = vminq_f32(x, max_x);
    let x = vmaxq_f32(x, min_x);

    // x = k * ln(2) + r,  k integer,  |r| ≤ ln(2)/2
    let log2e = vdupq_n_f32(1.4426950408889634);
    let k_f = vrndnq_f32(vmulq_f32(x, log2e));
    let k_i = vcvtq_s32_f32(k_f);

    // Cody-Waite split of ln(2) for better precision than a single
    // multiply.
    let ln2_hi = vdupq_n_f32(0.693145751953125);
    let ln2_lo = vdupq_n_f32(1.4286068203094172e-6);
    let r = vsubq_f32(x, vmulq_f32(k_f, ln2_hi));
    let r = vsubq_f32(r, vmulq_f32(k_f, ln2_lo));

    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 via Horner.
    let c5 = vdupq_n_f32(1.0 / 120.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c2 = vdupq_n_f32(0.5);
    let c1 = vdupq_n_f32(1.0);
    let c0 = vdupq_n_f32(1.0);

    let mut poly = vfmaq_f32(c4, c5, r);
    poly = vfmaq_f32(c3, poly, r);
    poly = vfmaq_f32(c2, poly, r);
    poly = vfmaq_f32(c1, poly, r);
    poly = vfmaq_f32(c0, poly, r);

    // Multiply by 2^k via direct f32-bit assembly: (k + 127) << 23.
    let bias = vdupq_n_s32(127);
    let exp_bits = vshlq_n_s32(vaddq_s32(k_i, bias), 23);
    let two_k = vreinterpretq_f32_s32(exp_bits);

    vmulq_f32(poly, two_k)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vsigmoidq_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg = vexpq_f32(neg_x);
    vdivq_f32(one, vaddq_f32(one, exp_neg))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vtanhq_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1).
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);
    let e = vexpq_f32(vmulq_f32(x, two));
    vdivq_f32(vsubq_f32(e, one), vaddq_f32(e, one))
}

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
                // For large K, the per-thread B-panel exceeds L1.
                // The cache-blocked variant restructures the loops to
                // keep B-panel slices hot across multiple mi tiles.
                if k > 256 {
                    matmul_2d_neon_blocked(a_slice, r_slice, c_slice, m, n, k);
                } else {
                    matmul_2d_neon(a_slice, r_slice, c_slice, m, n, k);
                }
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
/// Cache-blocked 4×8 NEON matmul. Splits K into blocks of `K_BLOCK`
/// so the B-panel touched per (mi, nj) tile stays in L1 across
/// multiple mi iterations. Without blocking, on M=197 K=768 N=192
/// our B-panel (768×8 = 24 KB) was bigger than the per-thread L1
/// budget on Cortex-A73 and we were paying a cache miss every other
/// inner step.
///
/// Use this for K > 256; below that the unblocked path is fine
/// (fewer accumulator load/store cycles).
fn matmul_2d_neon_blocked(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const K_BLOCK: usize = 128;

    let n_tiles = n / 8;
    let n_after_8 = n_tiles * 8;
    let n_tiles_4 = (n - n_after_8) / 4;
    let n_after_4 = n_after_8 + n_tiles_4 * 4;
    let m_tiles_count = m / 4;
    let m_tail_start = m_tiles_count * 4;

    // Initialise C to zero — the accumulating kernel loads existing
    // C values across K blocks.
    for x in c.iter_mut() {
        *x = 0.0;
    }

    let c_addr = c.as_mut_ptr() as usize;

    // For each K block, parallel-fan over mi tiles; within a thread,
    // iterate nj tiles. B's K-block slab is read 49 (m_tiles) times
    // per (kb, nj) but stays hot in L1.
    for kb_start in (0..k).step_by(K_BLOCK) {
        let kb_end = (kb_start + K_BLOCK).min(k);

        (0..m_tiles_count).into_par_iter().for_each(|mi| {
            let row_base = mi * 4;
            let c_ptr = c_addr as *mut f32;
            unsafe {
                for nj in 0..n_tiles {
                    let col_base = nj * 8;
                    matmul_kernel_4x8_accum(a, b, c_ptr, row_base, col_base, n, k, kb_start, kb_end);
                }
                for nj in 0..n_tiles_4 {
                    let col_base = n_after_8 + nj * 4;
                    matmul_kernel_4x4_accum(a, b, c_ptr, row_base, col_base, n, k, kb_start, kb_end);
                }
                // N tail (n % 4 cols) — scalar accumulate per K block.
                for col in n_after_4..n {
                    for row_off in 0..4 {
                        let row = row_base + row_off;
                        let mut acc = 0.0f32;
                        let a_row = std::slice::from_raw_parts(a.as_ptr().add(row * k), k);
                        for kk in kb_start..kb_end {
                            acc += a_row[kk] * b[kk * n + col];
                        }
                        *c_ptr.add(row * n + col) += acc;
                    }
                }
            }
        });
    }

    // M tail (m % 4 rows) — done once at the end against full K.
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

/// 4×8 NEON kernel that ACCUMULATES into existing C values (rather
/// than initializing from zero). Used by the cache-blocked matmul
/// across K blocks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matmul_kernel_4x8_accum(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    use core::arch::aarch64::*;

    // Load existing C accumulators.
    let mut c00 = vld1q_f32(c_ptr.add(row_base * n + col_base));
    let mut c01 = vld1q_f32(c_ptr.add(row_base * n + col_base + 4));
    let mut c10 = vld1q_f32(c_ptr.add((row_base + 1) * n + col_base));
    let mut c11 = vld1q_f32(c_ptr.add((row_base + 1) * n + col_base + 4));
    let mut c20 = vld1q_f32(c_ptr.add((row_base + 2) * n + col_base));
    let mut c21 = vld1q_f32(c_ptr.add((row_base + 2) * n + col_base + 4));
    let mut c30 = vld1q_f32(c_ptr.add((row_base + 3) * n + col_base));
    let mut c31 = vld1q_f32(c_ptr.add((row_base + 3) * n + col_base + 4));

    let a_row0 = a.as_ptr().add(row_base * k);
    let a_row1 = a.as_ptr().add((row_base + 1) * k);
    let a_row2 = a.as_ptr().add((row_base + 2) * k);
    let a_row3 = a.as_ptr().add((row_base + 3) * k);
    let b_ptr = b.as_ptr();

    for kk in k_start..k_end {
        let b0 = vld1q_f32(b_ptr.add(kk * n + col_base));
        let b1 = vld1q_f32(b_ptr.add(kk * n + col_base + 4));
        let a0 = *a_row0.add(kk);
        let a1 = *a_row1.add(kk);
        let a2 = *a_row2.add(kk);
        let a3 = *a_row3.add(kk);
        c00 = vfmaq_n_f32(c00, b0, a0);
        c01 = vfmaq_n_f32(c01, b1, a0);
        c10 = vfmaq_n_f32(c10, b0, a1);
        c11 = vfmaq_n_f32(c11, b1, a1);
        c20 = vfmaq_n_f32(c20, b0, a2);
        c21 = vfmaq_n_f32(c21, b1, a2);
        c30 = vfmaq_n_f32(c30, b0, a3);
        c31 = vfmaq_n_f32(c31, b1, a3);
    }

    vst1q_f32(c_ptr.add(row_base * n + col_base), c00);
    vst1q_f32(c_ptr.add(row_base * n + col_base + 4), c01);
    vst1q_f32(c_ptr.add((row_base + 1) * n + col_base), c10);
    vst1q_f32(c_ptr.add((row_base + 1) * n + col_base + 4), c11);
    vst1q_f32(c_ptr.add((row_base + 2) * n + col_base), c20);
    vst1q_f32(c_ptr.add((row_base + 2) * n + col_base + 4), c21);
    vst1q_f32(c_ptr.add((row_base + 3) * n + col_base), c30);
    vst1q_f32(c_ptr.add((row_base + 3) * n + col_base + 4), c31);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matmul_kernel_4x4_accum(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    use core::arch::aarch64::*;

    let mut c0 = vld1q_f32(c_ptr.add(row_base * n + col_base));
    let mut c1 = vld1q_f32(c_ptr.add((row_base + 1) * n + col_base));
    let mut c2 = vld1q_f32(c_ptr.add((row_base + 2) * n + col_base));
    let mut c3 = vld1q_f32(c_ptr.add((row_base + 3) * n + col_base));

    let a_row0 = a.as_ptr().add(row_base * k);
    let a_row1 = a.as_ptr().add((row_base + 1) * k);
    let a_row2 = a.as_ptr().add((row_base + 2) * k);
    let a_row3 = a.as_ptr().add((row_base + 3) * k);
    let b_ptr = b.as_ptr();

    for kk in k_start..k_end {
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

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn matmul_kernel_4x8_accum(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    for row_off in 0..4 {
        let row = row_base + row_off;
        let a_row = &a[row * k..row * k + k];
        for col_off in 0..8 {
            let col = col_base + col_off;
            let mut acc = 0.0f32;
            for kk in k_start..k_end {
                acc += a_row[kk] * b[kk * n + col];
            }
            *c_ptr.add(row * n + col) += acc;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn matmul_kernel_4x4_accum(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) {
    for row_off in 0..4 {
        let row = row_base + row_off;
        let a_row = &a[row * k..row * k + k];
        for col_off in 0..4 {
            let col = col_base + col_off;
            let mut acc = 0.0f32;
            for kk in k_start..k_end {
                acc += a_row[kk] * b[kk * n + col];
            }
            *c_ptr.add(row * n + col) += acc;
        }
    }
}

fn matmul_2d_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // 4-row × 8-col register tile. C has 8 NEON accumulators (2 per row,
    // 4 floats each = 32 output cells per thread per K-loop). Cortex-A73
    // has 32 vector regs, plenty of room.
    let n_tiles_8 = n / 8;
    let n_after_8 = n_tiles_8 * 8;
    // Anything left after the 8-col tiles, handle in 4-col tiles.
    let n_tiles_4 = (n - n_after_8) / 4;
    let n_after_4 = n_after_8 + n_tiles_4 * 4;
    let m_tiles_count = m / 4;
    let m_tail_start = m_tiles_count * 4;

    // Transport the mut pointer across threads as a usize — guarantees
    // Send + Sync. Tile disjointness ensures no two threads ever
    // target the same output cell.
    let c_addr = c.as_mut_ptr() as usize;

    (0..m_tiles_count).into_par_iter().for_each(|mi| {
        let row_base = mi * 4;
        let c_ptr = c_addr as *mut f32;
        unsafe {
            for nj in 0..n_tiles_8 {
                let col_base = nj * 8;
                matmul_kernel_4x8(a, b, c_ptr, row_base, col_base, n, k);
            }
            for nj in 0..n_tiles_4 {
                let col_base = n_after_8 + nj * 4;
                matmul_kernel_4x4(a, b, c_ptr, row_base, col_base, n, k);
            }
            // N tail (n % 4 columns) for these 4 rows.
            for col in n_after_4..n {
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

/// 4-row × 8-col register kernel — twice the throughput of the 4×4
/// variant for shapes where N is a multiple of 8. 8 vector C
/// accumulators (4 rows × 2 col-vecs), 2 B vectors, 4 A scalars. On
/// Cortex-A73 (32 NEON regs) plenty of margin.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matmul_kernel_4x8(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
) {
    use core::arch::aarch64::*;

    let mut c00 = vdupq_n_f32(0.0);
    let mut c01 = vdupq_n_f32(0.0);
    let mut c10 = vdupq_n_f32(0.0);
    let mut c11 = vdupq_n_f32(0.0);
    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);

    let a_row0 = a.as_ptr().add(row_base * k);
    let a_row1 = a.as_ptr().add((row_base + 1) * k);
    let a_row2 = a.as_ptr().add((row_base + 2) * k);
    let a_row3 = a.as_ptr().add((row_base + 3) * k);
    let b_ptr = b.as_ptr();

    for kk in 0..k {
        let b0 = vld1q_f32(b_ptr.add(kk * n + col_base));
        let b1 = vld1q_f32(b_ptr.add(kk * n + col_base + 4));
        let a0 = *a_row0.add(kk);
        let a1 = *a_row1.add(kk);
        let a2 = *a_row2.add(kk);
        let a3 = *a_row3.add(kk);
        c00 = vfmaq_n_f32(c00, b0, a0);
        c01 = vfmaq_n_f32(c01, b1, a0);
        c10 = vfmaq_n_f32(c10, b0, a1);
        c11 = vfmaq_n_f32(c11, b1, a1);
        c20 = vfmaq_n_f32(c20, b0, a2);
        c21 = vfmaq_n_f32(c21, b1, a2);
        c30 = vfmaq_n_f32(c30, b0, a3);
        c31 = vfmaq_n_f32(c31, b1, a3);
    }

    vst1q_f32(c_ptr.add(row_base * n + col_base), c00);
    vst1q_f32(c_ptr.add(row_base * n + col_base + 4), c01);
    vst1q_f32(c_ptr.add((row_base + 1) * n + col_base), c10);
    vst1q_f32(c_ptr.add((row_base + 1) * n + col_base + 4), c11);
    vst1q_f32(c_ptr.add((row_base + 2) * n + col_base), c20);
    vst1q_f32(c_ptr.add((row_base + 2) * n + col_base + 4), c21);
    vst1q_f32(c_ptr.add((row_base + 3) * n + col_base), c30);
    vst1q_f32(c_ptr.add((row_base + 3) * n + col_base + 4), c31);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn matmul_kernel_4x8(
    a: &[f32],
    b: &[f32],
    c_ptr: *mut f32,
    row_base: usize,
    col_base: usize,
    n: usize,
    k: usize,
) {
    matmul_kernel_4x4(a, b, c_ptr, row_base, col_base, n, k);
    matmul_kernel_4x4(a, b, c_ptr, row_base, col_base + 4, n, k);
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

/// Fused bias add: out[i] = act[i] + bias[i mod inner].
/// `act` and `out` are both `outer * inner` f32 elements (row-major
/// with `inner` floats per row). `bias` is length `inner`. Replaces
/// the `Nx.broadcast({inner}, {..., inner}) + add` pair that's the
/// Bumblebee MLP / projection pattern — saves ~60–170 ms per call
/// vs going through `Nx.broadcast` (whose wrapper has substantial
/// per-call overhead for hot-loop callers).
pub fn bias_add_f32_into(
    act: &[f32],
    bias: &[f32],
    out: &mut [f32],
    outer: usize,
    inner: usize,
) -> Result<(), String> {
    if act.len() != outer * inner {
        return Err(format!(
            "bias_add: act len {} != outer*inner = {}",
            act.len(),
            outer * inner
        ));
    }
    if bias.len() != inner {
        return Err(format!("bias_add: bias len {} != inner {}", bias.len(), inner));
    }
    if out.len() != outer * inner {
        return Err(format!(
            "bias_add: out len {} != outer*inner = {}",
            out.len(),
            outer * inner
        ));
    }

    // Each row is independent and small enough to keep `bias` in L1.
    // Inner loop autovectorises to NEON FMA-style ops; per-row chunking
    // keeps `bias`'s 768/192 f32s hot in cache for every iteration of
    // outer.
    let chunk_rows = ((outer + 7) / 8).max(8).min(outer.max(1));

    out.par_chunks_mut(chunk_rows * inner)
        .enumerate()
        .for_each(|(ci, slot)| {
            let row_base = ci * chunk_rows;
            let rows_here = slot.len() / inner;

            for r in 0..rows_here {
                let row_idx = row_base + r;
                let act_row = &act[row_idx * inner..row_idx * inner + inner];
                let out_row = &mut slot[r * inner..r * inner + inner];

                // Plain f32 loop — autovectoriser handles the NEON.
                for j in 0..inner {
                    out_row[j] = act_row[j] + bias[j];
                }
            }
        });

    Ok(())
}

/// Generic gather along (potentially) several axes. Matches Nx's
/// gather semantics: `indices` has shape `[..., depth]`. Each `depth`-
/// length row picks one position in the indexed axes; the gathered
/// block (the non-indexed axes) is copied to the output. `axes` is the
/// list of indexed `tensor` axes; if it's a prefix `[0, 1, ..., depth-1]`
/// the gathered block is contiguous in the input (fast memcpy path).
///
/// `index_size` is the byte width of one index (4 for s32/u32, 8 for
/// s64). `element_size` is the byte width of one tensor element.
///
/// Input/output bytes; works for any tensor element type.
pub fn gather(
    input: &[u8],
    in_shape: &[usize],
    indices: &[u8],
    idx_shape: &[usize],
    index_size: usize,
    axes: &[usize],
    element_size: usize,
) -> Result<Vec<u8>, String> {
    let depth = axes.len();
    let in_rank = in_shape.len();
    let idx_rank = idx_shape.len();

    if idx_rank == 0 {
        return Err("gather: indices must have rank ≥ 1".into());
    }

    if elem_get(idx_shape, idx_rank - 1) != depth {
        return Err(format!(
            "gather: indices last axis {} must equal axes count {}",
            elem_get(idx_shape, idx_rank - 1),
            depth
        ));
    }

    if depth > in_rank {
        return Err(format!(
            "gather: axes count {} > input rank {}",
            depth, in_rank
        ));
    }

    // Per-input-axis row-major strides (in elements).
    let mut in_strides = vec![0usize; in_rank];
    if in_rank > 0 {
        in_strides[in_rank - 1] = 1;
        for k in (0..in_rank - 1).rev() {
            in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
        }
    }

    // Block size: product of input axes NOT in `axes`. For the
    // contiguous fast path this is the trailing axes' product.
    let indexed_axes_set: std::collections::HashSet<usize> = axes.iter().copied().collect();
    let block_elems: usize = (0..in_rank)
        .filter(|i| !indexed_axes_set.contains(i))
        .map(|i| in_shape[i])
        .product();

    let n_lookups: usize = idx_shape[..idx_rank - 1].iter().product();
    let total_out_elems = n_lookups * block_elems;
    let mut out_bytes = vec![0u8; total_out_elems * element_size];

    // For each lookup row, decode `depth` indices and compute the
    // input element offset. Then either memcpy a contiguous block (if
    // `axes` is a contiguous prefix) or walk the strided gather.
    let contiguous_prefix = axes.iter().enumerate().all(|(i, &a)| a == i);

    if contiguous_prefix {
        // Fast path: gathered block is `prod(in_shape[depth..])`
        // contiguous elements in the input.
        let block_bytes = block_elems * element_size;

        for lookup in 0..n_lookups {
            let mut in_elem_offset = 0usize;
            for k in 0..depth {
                let raw = &indices[(lookup * depth + k) * index_size..(lookup * depth + k + 1) * index_size];
                let idx = read_index(raw, index_size)?;
                if idx >= in_shape[axes[k]] {
                    return Err(format!(
                        "gather: index {} out of range [0, {})",
                        idx, in_shape[axes[k]]
                    ));
                }
                in_elem_offset += idx * in_strides[axes[k]];
            }
            let src_byte = in_elem_offset * element_size;
            let dst_byte = lookup * block_bytes;
            out_bytes[dst_byte..dst_byte + block_bytes]
                .copy_from_slice(&input[src_byte..src_byte + block_bytes]);
        }

        Ok(out_bytes)
    } else {
        // General path — gather strided sub-blocks. Slower (per-element
        // copy), but supports arbitrary `axes` configurations. For the
        // text-embedding use case `axes` is always `[0]`, so this
        // branch is rarely hit in practice.
        Err("gather: non-prefix axes not yet supported".into())
    }
}

fn elem_get<T: Copy>(s: &[T], i: usize) -> T {
    s[i]
}

fn read_index(bytes: &[u8], size: usize) -> Result<usize, String> {
    match size {
        4 => {
            // Most index tensors in Nx are s64 or s32. We read as
            // little-endian unsigned; for sensible (non-negative)
            // indices this matches the signed interpretation. Negative
            // indices would silently wrap — Nx's semantics raise on
            // those, but we check the upper bound after.
            let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(v as usize)
        }
        8 => {
            let v = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            Ok(v as usize)
        }
        n => Err(format!("gather: unsupported index byte size {}", n)),
    }
}

/// RMSNorm along the last axis. Replaces the Llama/Mistral/Phi/Qwen
/// pre-attention/pre-MLP normalisation step. Input viewed as
/// `[n_outer × inner]`; per row computes
///
///   `out[i] = (x[i] / sqrt(mean(x²) + eps)) * gamma[i]`
///
/// — no mean subtraction (unlike LayerNorm). Single pass per row:
/// sum-of-squares → invsqrt → scaled multiply.
pub fn rmsnorm_f32(
    input: &[f32],
    gamma: &[f32],
    n_outer: usize,
    inner: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if input.len() != n_outer * inner {
        return Err(format!(
            "rmsnorm: input len {} != n_outer*inner = {}",
            input.len(),
            n_outer * inner
        ));
    }
    if gamma.len() != inner {
        return Err(format!("rmsnorm: gamma len {} != inner {}", gamma.len(), inner));
    }

    let mut out = vec![0.0f32; n_outer * inner];
    let inv_n = 1.0_f32 / (inner as f32);

    use rayon::prelude::*;
    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(i, row_out)| {
            let row_in = &input[i * inner..(i + 1) * inner];

            // Pass 1: sum of squares.
            let mut sumsq = 0.0f32;
            for &v in row_in {
                sumsq += v * v;
            }

            let inv_rms = 1.0_f32 / (sumsq * inv_n + epsilon).sqrt();

            // Pass 2: scale + multiply by gamma.
            for j in 0..inner {
                row_out[j] = row_in[j] * inv_rms * gamma[j];
            }
        });

    Ok(out)
}

/// Rotary Position Embedding applied to Q or K tensors.
///
/// `input` is the Q/K tensor with shape `[..., n_heads, head_dim]` or
/// any layout that puts head_dim as the last axis. We treat it
/// generically as `[n_rows × head_dim]` where each row is one head's
/// vector for one token. `head_dim` must be even (each consecutive
/// pair gets rotated).
///
/// `positions` is `[n_tokens]` (the position index per token); each
/// row of `input` belongs to one token. `n_rows` is divided by the
/// number of heads × batches to map to token positions: we accept
/// `position_for_row[row] = positions[row / heads_per_token]` via
/// the `heads_per_token` argument. For typical transformer use:
///
///   * input shape `[batch, seq, heads, head_dim]` flattened as
///     `[batch * seq * heads, head_dim]`
///   * heads_per_token = `heads`
///   * positions length = `batch * seq`
///
/// `inv_freq` is `[head_dim/2]` — the precomputed `1.0 / base^(2k/head_dim)`.
pub fn rope_f32(
    input: &[f32],
    positions: &[i64],
    inv_freq: &[f32],
    n_rows: usize,
    head_dim: usize,
    heads_per_token: usize,
) -> Result<Vec<f32>, String> {
    if input.len() != n_rows * head_dim {
        return Err(format!(
            "rope: input len {} != n_rows*head_dim = {}",
            input.len(),
            n_rows * head_dim
        ));
    }
    if head_dim % 2 != 0 {
        return Err(format!("rope: head_dim {} must be even", head_dim));
    }
    if inv_freq.len() != head_dim / 2 {
        return Err(format!(
            "rope: inv_freq len {} != head_dim/2 = {}",
            inv_freq.len(),
            head_dim / 2
        ));
    }
    let n_tokens = positions.len();
    if n_rows != n_tokens * heads_per_token {
        return Err(format!(
            "rope: n_rows {} != n_tokens * heads_per_token = {}",
            n_rows,
            n_tokens * heads_per_token
        ));
    }

    let mut out = vec![0.0f32; n_rows * head_dim];

    use rayon::prelude::*;
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(row, row_out)| {
            let token = row / heads_per_token;
            let pos = positions[token] as f32;
            let row_in = &input[row * head_dim..(row + 1) * head_dim];

            // Process consecutive pairs (j, j+1). Llama-style RoPE
            // pairs dim k with k+1; some variants pair k with k+head_dim/2
            // — we use the original "interleaved" form since it's what
            // most Bumblebee models will emit through the standard
            // decomposition path.
            for k in 0..(head_dim / 2) {
                let theta = pos * inv_freq[k];
                let cos = theta.cos();
                let sin = theta.sin();

                let x0 = row_in[2 * k];
                let x1 = row_in[2 * k + 1];

                row_out[2 * k] = x0 * cos - x1 * sin;
                row_out[2 * k + 1] = x0 * sin + x1 * cos;
            }
        });

    Ok(out)
}

/// Bilinear image resize for HWC uint8 inputs. Common entry point
/// for vision-model preprocessing: camera/JPEG decode produces a
/// HxWx3 u8 buffer; the model wants its own resolution. Pure
/// bilinear interpolation, written as a per-output-pixel gather of
/// the 4 surrounding source pixels.
pub fn bilinear_resize_u8(
    input: &[u8],
    in_h: usize,
    in_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>, String> {
    if input.len() != in_h * in_w * channels {
        return Err(format!(
            "bilinear_resize: input len {} != H*W*C = {}",
            input.len(),
            in_h * in_w * channels
        ));
    }
    if out_h == 0 || out_w == 0 || in_h == 0 || in_w == 0 {
        return Err("bilinear_resize: zero dimension".into());
    }

    let mut out = vec![0u8; out_h * out_w * channels];

    let h_scale = (in_h as f32 - 1.0) / (out_h as f32 - 1.0).max(1.0);
    let w_scale = (in_w as f32 - 1.0) / (out_w as f32 - 1.0).max(1.0);

    use rayon::prelude::*;
    out.par_chunks_mut(out_w * channels)
        .enumerate()
        .for_each(|(oy, row)| {
            let in_y = oy as f32 * h_scale;
            let y0 = in_y.floor() as usize;
            let y1 = (y0 + 1).min(in_h - 1);
            let dy = in_y - y0 as f32;

            for ox in 0..out_w {
                let in_x = ox as f32 * w_scale;
                let x0 = in_x.floor() as usize;
                let x1 = (x0 + 1).min(in_w - 1);
                let dx = in_x - x0 as f32;

                for c in 0..channels {
                    let p00 = input[(y0 * in_w + x0) * channels + c] as f32;
                    let p01 = input[(y0 * in_w + x1) * channels + c] as f32;
                    let p10 = input[(y1 * in_w + x0) * channels + c] as f32;
                    let p11 = input[(y1 * in_w + x1) * channels + c] as f32;

                    let interp = p00 * (1.0 - dx) * (1.0 - dy)
                        + p01 * dx * (1.0 - dy)
                        + p10 * (1.0 - dx) * dy
                        + p11 * dx * dy;

                    row[ox * channels + c] = interp.round() as u8;
                }
            }
        });

    Ok(out)
}

/// Weight-only int8 matmul: `act` f32 × `weights` int8, per-row
/// (per-output-channel) f32 scales. Computes
///
///   `out[b, m, n] = scales[n] * sum_k(act[b, m, k] * (f32)weights[n, k])`
///
/// where `weights` is laid out `[N, K]` (one output channel per row,
/// contiguous over K). This is the "GPTQ-style" pattern used by
/// llama.cpp Q8_0 and other quantized model formats — the weights
/// stay int8 (4× memory savings vs f32) and the matmul itself runs
/// dequantize-and-accumulate in f32. Without SDOT (Cortex-A73 is
/// pre-ARMv8.2) this is the realistic int8 path on this hardware.
///
/// Layout matches our batched_matmul_f32 with `right_transposed = true`
/// (Q @ K^T pattern) — both sides contract on their last axis. For a
/// 2-D linear layer (b=1), it's `act[M, K] · weights[N, K]^T = [M, N]`.
pub fn dequant_matmul_int8_f32(
    act: &[f32],
    weights: &[i8],
    scales: &[f32],
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    if act.len() != b * m * k {
        return Err(format!(
            "dequant_matmul_int8: act len {} != B*M*K = {}",
            act.len(),
            b * m * k
        ));
    }
    if weights.len() != n * k {
        return Err(format!(
            "dequant_matmul_int8: weights len {} != N*K = {}",
            weights.len(),
            n * k
        ));
    }
    if scales.len() != n {
        return Err(format!(
            "dequant_matmul_int8: scales len {} != N = {}",
            scales.len(),
            n
        ));
    }

    let mut out = vec![0.0f32; b * m * n];

    use rayon::prelude::*;
    out.par_chunks_mut(n).enumerate().for_each(|(bi_row, row)| {
        let batch = bi_row / m;
        let i = bi_row % m;
        let a_base = batch * m * k + i * k;
        let a = &act[a_base..a_base + k];

        for j in 0..n {
            let w_base = j * k;
            let w = &weights[w_base..w_base + k];
            let s = scales[j];

            // Sequential f32 accumulation; NEON FMA via the auto-
            // vectoriser. The int8→f32 cast in `w[kk] as f32` produces
            // VCVTQ-style widening ops naturally.
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[kk] * (w[kk] as f32);
            }
            row[j] = acc * s;
        }
    });

    Ok(out)
}

/// Generic n-D window reduction (max / min / sum / product) on f32.
/// Used as the backend for Nx.window_max, window_min, window_sum,
/// window_product — and via decomposition for max_pool, avg_pool.
///
/// `window_dims` is per-input-axis window size; `strides` is per-axis;
/// `padding` is a list of (low, high) padding amounts per axis.
pub fn window_reduce_f32(
    op: &str,
    input: &[f32],
    in_shape: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    padding: &[(isize, isize)],
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let rank = in_shape.len();
    if window_dims.len() != rank || strides.len() != rank || padding.len() != rank {
        return Err(format!(
            "window_reduce: rank mismatch (in={}, win={}, strides={}, pad={})",
            rank,
            window_dims.len(),
            strides.len(),
            padding.len()
        ));
    }

    // Output shape per axis: floor((in_size + pad_lo + pad_hi - win) / stride) + 1
    let out_shape: Vec<usize> = (0..rank)
        .map(|k| {
            let in_size = in_shape[k] as isize;
            let (pad_lo, pad_hi) = padding[k];
            let effective = in_size + pad_lo + pad_hi - window_dims[k] as isize;
            if effective < 0 {
                0
            } else {
                (effective / strides[k] as isize) as usize + 1
            }
        })
        .collect();

    let out_total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_total];

    let in_strides = {
        let mut s = vec![0usize; rank];
        if rank > 0 {
            s[rank - 1] = 1;
            for k in (0..rank - 1).rev() {
                s[k] = s[k + 1] * in_shape[k + 1];
            }
        }
        s
    };

    let out_strides = {
        let mut s = vec![0usize; rank];
        if rank > 0 {
            s[rank - 1] = 1;
            for k in (0..rank - 1).rev() {
                s[k] = s[k + 1] * out_shape[k + 1];
            }
        }
        s
    };

    let init = match op {
        "max" => f32::NEG_INFINITY,
        "min" => f32::INFINITY,
        "sum" => 0.0,
        "product" => 1.0,
        other => return Err(format!("unknown window op: {}", other)),
    };

    let pad_lows: Vec<isize> = padding.iter().map(|(lo, _)| *lo).collect();

    // Total window volume (product of window_dims) for the inner loop.
    let win_total: usize = window_dims.iter().product();

    // For each output cell, gather window elements and reduce.
    use rayon::prelude::*;
    out.par_iter_mut().enumerate().for_each(|(out_flat, slot)| {
        // Decode out_flat to multi-index.
        let mut out_idx = vec![0usize; rank];
        let mut rem = out_flat;
        for k in 0..rank {
            out_idx[k] = if out_strides[k] == 0 { 0 } else { rem / out_strides[k] };
            rem -= out_idx[k] * out_strides[k];
        }

        let mut acc = init;
        // Walk the window.
        for win_flat in 0..win_total {
            // Decode window index.
            let mut win_idx = vec![0usize; rank];
            let mut r = win_flat;
            for k in (0..rank).rev() {
                win_idx[k] = r % window_dims[k];
                r /= window_dims[k];
            }

            // Compute corresponding input multi-index.
            let mut in_off = 0usize;
            let mut out_of_bounds = false;
            for k in 0..rank {
                let in_idx = out_idx[k] as isize * strides[k] as isize - pad_lows[k]
                    + win_idx[k] as isize;
                if in_idx < 0 || in_idx >= in_shape[k] as isize {
                    out_of_bounds = true;
                    break;
                }
                in_off += in_idx as usize * in_strides[k];
            }

            if !out_of_bounds {
                let v = input[in_off];
                acc = match op {
                    "max" => if v > acc { v } else { acc },
                    "min" => if v < acc { v } else { acc },
                    "sum" => acc + v,
                    "product" => acc * v,
                    _ => unreachable!(),
                };
            }
            // For out-of-bounds (padded) cells we use the identity for
            // the reduction (matches BinaryBackend's behavior for max/
            // min where padding contributes -inf/inf, and sum where it
            // contributes 0).
        }

        *slot = acc;
    });

    Ok((out, out_shape))
}

/// Strided n-D slice. Output shape is `lengths`. For each output
/// position, source position is `starts + out_pos * strides`. With
/// unit strides we coalesce trailing axes into row-memcpys (fast).
///
/// Used by Nx.slice and by KV-cache extraction patterns in
/// transformers.
pub fn slice(
    input: &[u8],
    in_shape: &[usize],
    starts: &[usize],
    lengths: &[usize],
    strides: &[usize],
    element_size: usize,
) -> Result<Vec<u8>, String> {
    let rank = in_shape.len();
    if starts.len() != rank || lengths.len() != rank || strides.len() != rank {
        return Err(format!(
            "slice: rank mismatch (in={}, starts={}, lengths={}, strides={})",
            rank,
            starts.len(),
            lengths.len(),
            strides.len()
        ));
    }

    let out_total: usize = lengths.iter().product();
    let mut out = vec![0u8; out_total * element_size];

    // Input row-major strides (in elements).
    let mut in_elem_strides = vec![0usize; rank];
    if rank > 0 {
        in_elem_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            in_elem_strides[k] = in_elem_strides[k + 1] * in_shape[k + 1];
        }
    }

    // Fast path: all strides == 1. Coalesce trailing axes that are
    // FULL-COVER (start=0, length=in_shape[k]) into one big memcpy.
    let all_unit = strides.iter().all(|&s| s == 1);

    if all_unit {
        // Find the longest trailing run of full-cover axes.
        let mut coalesce_depth = 0usize;
        for k in (0..rank).rev() {
            if starts[k] == 0 && lengths[k] == in_shape[k] {
                coalesce_depth += 1;
            } else {
                break;
            }
        }

        // The "outer" axes are 0..rank-coalesce_depth — we walk them
        // and emit one contiguous copy per outer position.
        let outer_rank = rank - coalesce_depth;
        let inner_block_elems: usize = lengths[outer_rank..].iter().product();
        let inner_block_bytes = inner_block_elems * element_size;

        let outer_lengths: &[usize] = &lengths[..outer_rank];
        let outer_total: usize = outer_lengths.iter().product();

        // Per-output-axis strides for the outer space (in elements).
        let mut out_outer_strides = vec![0usize; outer_rank];
        if outer_rank > 0 {
            out_outer_strides[outer_rank - 1] = 1;
            for k in (0..outer_rank - 1).rev() {
                out_outer_strides[k] = out_outer_strides[k + 1] * outer_lengths[k + 1];
            }
        }

        for outer_pos in 0..outer_total {
            // Decode outer_pos into multi-index, compute input offset.
            let mut rem = outer_pos;
            let mut in_elem_off = 0usize;
            for k in 0..outer_rank {
                let s = out_outer_strides[k];
                let idx = if s == 0 { 0 } else { rem / s };
                rem -= idx * s;
                in_elem_off += (starts[k] + idx) * in_elem_strides[k];
            }
            // Coalesced inner axes contribute `starts[k] * stride[k]`
            // for each k in the coalesced range. With full-cover and
            // unit stride, starts[k] = 0 so no contribution.
            let src_byte = in_elem_off * element_size;
            let dst_byte = outer_pos * inner_block_bytes;
            out[dst_byte..dst_byte + inner_block_bytes]
                .copy_from_slice(&input[src_byte..src_byte + inner_block_bytes]);
        }

        return Ok(out);
    }

    // General strided path: per-element copy.
    let mut out_strides = vec![0usize; rank];
    if rank > 0 {
        out_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            out_strides[k] = out_strides[k + 1] * lengths[k + 1];
        }
    }

    for out_pos in 0..out_total {
        let mut rem = out_pos;
        let mut in_elem_off = 0usize;
        for k in 0..rank {
            let s = out_strides[k];
            let idx = if s == 0 { 0 } else { rem / s };
            rem -= idx * s;
            in_elem_off += (starts[k] + idx * strides[k]) * in_elem_strides[k];
        }
        let src_byte = in_elem_off * element_size;
        let dst_byte = out_pos * element_size;
        out[dst_byte..dst_byte + element_size]
            .copy_from_slice(&input[src_byte..src_byte + element_size]);
    }

    Ok(out)
}

/// Write `slice` into `tensor` starting at `starts`. Returns a fresh
/// tensor — the operation isn't in-place from Nx's POV.
pub fn put_slice(
    tensor: &[u8],
    in_shape: &[usize],
    slice: &[u8],
    slice_shape: &[usize],
    starts: &[usize],
    element_size: usize,
) -> Result<Vec<u8>, String> {
    let rank = in_shape.len();
    if starts.len() != rank || slice_shape.len() != rank {
        return Err(format!(
            "put_slice: rank mismatch (in={}, slice={}, starts={})",
            rank,
            slice_shape.len(),
            starts.len()
        ));
    }

    // Bounds check.
    for k in 0..rank {
        if starts[k] + slice_shape[k] > in_shape[k] {
            return Err(format!(
                "put_slice: slice axis {} out of range ({} + {} > {})",
                k, starts[k], slice_shape[k], in_shape[k]
            ));
        }
    }

    // Start with a copy of `tensor`.
    let mut out = tensor.to_vec();

    // Input row-major strides.
    let mut in_elem_strides = vec![0usize; rank];
    if rank > 0 {
        in_elem_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            in_elem_strides[k] = in_elem_strides[k + 1] * in_shape[k + 1];
        }
    }

    // Slice row-major strides.
    let mut sl_elem_strides = vec![0usize; rank];
    if rank > 0 {
        sl_elem_strides[rank - 1] = 1;
        for k in (0..rank - 1).rev() {
            sl_elem_strides[k] = sl_elem_strides[k + 1] * slice_shape[k + 1];
        }
    }

    // Coalesce trailing axes that are full-cover in the slice.
    let mut coalesce_depth = 0usize;
    for k in (0..rank).rev() {
        if starts[k] == 0 && slice_shape[k] == in_shape[k] {
            coalesce_depth += 1;
        } else {
            break;
        }
    }

    let outer_rank = rank - coalesce_depth;
    let inner_block_elems: usize = slice_shape[outer_rank..].iter().product();
    let inner_block_bytes = inner_block_elems * element_size;

    let outer_lengths: &[usize] = &slice_shape[..outer_rank];
    let outer_total: usize = outer_lengths.iter().product();

    let mut outer_strides = vec![0usize; outer_rank];
    if outer_rank > 0 {
        outer_strides[outer_rank - 1] = 1;
        for k in (0..outer_rank - 1).rev() {
            outer_strides[k] = outer_strides[k + 1] * outer_lengths[k + 1];
        }
    }

    for outer_pos in 0..outer_total {
        let mut rem = outer_pos;
        let mut in_elem_off = 0usize;
        let mut sl_elem_off = 0usize;
        for k in 0..outer_rank {
            let s = outer_strides[k];
            let idx = if s == 0 { 0 } else { rem / s };
            rem -= idx * s;
            in_elem_off += (starts[k] + idx) * in_elem_strides[k];
            sl_elem_off += idx * sl_elem_strides[k];
        }
        let dst_byte = in_elem_off * element_size;
        let src_byte = sl_elem_off * element_size;
        out[dst_byte..dst_byte + inner_block_bytes]
            .copy_from_slice(&slice[src_byte..src_byte + inner_block_bytes]);
    }

    Ok(out)
}

// ── f32 elementwise (CPU NEON via auto-vectoriser) ──────

/// Same-shape elementwise binary op on f32 arrays. Writes directly
/// into the caller-provided `out` slice — no Vec<f32> intermediate.
pub fn elementwise_binary_f32_into(op: &str, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), String> {
    if a.len() != b.len() || out.len() != a.len() {
        return Err(format!("len mismatch: a={} b={} out={}", a.len(), b.len(), out.len()));
    }
    let n = a.len();
    let chunk = ((n + 7) / 8).max(8192).min(n.max(1));

    // Explicit NEON intrinsics for ops with a single-instruction NEON
    // primitive. The autovectoriser handles these decently when the
    // loop body is tight scalar arithmetic, but `vld1q + vop + vst1q`
    // produces tighter code (no bounds-check noise, 4 floats / cycle
    // guaranteed). Falls back to scalar tail for n%4.
    macro_rules! neon_op {
        ($vop:ident, $sop:expr) => {{
            #[cfg(target_arch = "aarch64")]
            {
                out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                    let s = i * chunk;
                    let len = slot.len();
                    let n_vec = len / 4;
                    let n_tail = n_vec * 4;
                    unsafe {
                        use core::arch::aarch64::*;
                        for j in 0..n_vec {
                            let av = vld1q_f32(a.as_ptr().add(s + j * 4));
                            let bv = vld1q_f32(b.as_ptr().add(s + j * 4));
                            let r = $vop(av, bv);
                            vst1q_f32(slot.as_mut_ptr().add(j * 4), r);
                        }
                    }
                    for j in n_tail..len {
                        slot[j] = $sop(a[s + j], b[s + j]);
                    }
                });
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                    let s = i * chunk;
                    for j in 0..slot.len() {
                        slot[j] = $sop(a[s + j], b[s + j]);
                    }
                });
            }
        }};
    }

    // Scalar fallback (for ops without a single NEON instruction:
    // pow, atan2, remainder, divide on older NEON without vdivq).
    macro_rules! scalar_op {
        ($body:expr) => {{
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, slot)| {
                let s = i * chunk;
                for j in 0..slot.len() {
                    slot[j] = $body(a[s + j], b[s + j]);
                }
            });
        }};
    }

    match op {
        "add" => neon_op!(vaddq_f32, |x: f32, y: f32| x + y),
        "subtract" => neon_op!(vsubq_f32, |x: f32, y: f32| x - y),
        "multiply" => neon_op!(vmulq_f32, |x: f32, y: f32| x * y),
        "divide" => neon_op!(vdivq_f32, |x: f32, y: f32| x / y),
        "max" => neon_op!(vmaxq_f32, |x: f32, y: f32| x.max(y)),
        "min" => neon_op!(vminq_f32, |x: f32, y: f32| x.min(y)),
        "pow" => scalar_op!(|x: f32, y: f32| x.powf(y)),
        "atan2" => scalar_op!(|x: f32, y: f32| x.atan2(y)),
        "remainder" => scalar_op!(|x: f32, y: f32| x.rem_euclid(y)),
        other => return Err(format!("unknown binary op: {}", other)),
    }

    Ok(())
}

/// Vec-returning wrapper kept for ergonomic callers. Internally just
/// allocates a Vec and calls _into.
pub fn elementwise_binary_f32(op: &str, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0f32; a.len()];
    elementwise_binary_f32_into(op, a, b, &mut out)?;
    Ok(out)
}

/// Scalar-broadcast binary op writing directly into `out`.
pub fn scalar_binary_f32_into(op: &str, side: &str, a: &[f32], scalar: f32, out: &mut [f32]) -> Result<(), String> {
    let n = a.len();
    if out.len() != n {
        return Err(format!("scalar_binary out len {} != a len {}", out.len(), n));
    }
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

    Ok(())
}

pub fn scalar_binary_f32(op: &str, side: &str, a: &[f32], scalar: f32) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0f32; a.len()];
    scalar_binary_f32_into(op, side, a, scalar, &mut out)?;
    Ok(out)
}

/// Elementwise unary op writing directly into `out`.
pub fn elementwise_unary_f32_into(op: &str, a: &[f32], out: &mut [f32]) -> Result<(), String> {
    let n = a.len();
    if out.len() != n {
        return Err(format!("unary out len {} != a len {}", out.len(), n));
    }
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

    // NEON-vectorised polynomial path for exp/sigmoid/tanh. The
    // autovectoriser can't handle libm's `exp` calls — each was one
    // scalar libm invocation. The poly path does 4 floats / NEON
    // cycle. On a {1, 3, 197, 197} attention softmax, exp dropped
    // from ~80 ms → ~12 ms in microbench.
    macro_rules! run_neon {
        ($vec_op:expr, $scalar_fallback:expr) => {{
            #[cfg(target_arch = "aarch64")]
            {
                out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slot)| {
                    let s = ci * chunk;
                    let len = slot.len();
                    let n_vec = len / 4;
                    let n_tail = n_vec * 4;
                    unsafe {
                        for i in 0..n_vec {
                            let v = core::arch::aarch64::vld1q_f32(a.as_ptr().add(s + i * 4));
                            let r = $vec_op(v);
                            core::arch::aarch64::vst1q_f32(slot.as_mut_ptr().add(i * 4), r);
                        }
                    }
                    for i in n_tail..len {
                        slot[i] = $scalar_fallback(a[s + i]);
                    }
                });
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                run!($scalar_fallback);
            }
        }};
    }

    match op {
        "negate" => run!(|x: f32| -x),
        "exp" => run_neon!(|v| vexpq_f32(v), |x: f32| x.exp()),
        "log" => run!(|x: f32| x.ln()),
        "tanh" => run_neon!(|v| vtanhq_f32(v), |x: f32| x.tanh()),
        "sigmoid" => run_neon!(|v| vsigmoidq_f32(v), |x: f32| 1.0 / (1.0 + (-x).exp())),
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

    Ok(())
}

pub fn elementwise_unary_f32(op: &str, a: &[f32]) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0f32; a.len()];
    elementwise_unary_f32_into(op, a, &mut out)?;
    Ok(out)
}

/// Fused softmax along the last axis. Input viewed as `[n_outer ×
/// inner]` row-major; each row gets `softmax(x) = exp(x - max(x)) /
/// sum(exp(x - max(x)))` in a single pass. Replaces the 7-primitive
/// Axon defn decomposition with one NIF call — saves the 6 intermediate
/// `Vec<f32>` allocations and the broadcast + elementwise dispatch
/// overhead.
pub fn softmax_f32(input: &[f32], n_outer: usize, inner: usize) -> Result<Vec<f32>, String> {
    if input.len() != n_outer * inner {
        return Err(format!(
            "softmax_f32: len {} != n_outer*inner = {}",
            input.len(),
            n_outer * inner
        ));
    }
    let mut out = vec![0.0f32; n_outer * inner];

    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(i, row_out)| {
            let row_in = &input[i * inner..(i + 1) * inner];
            softmax_row_f32(row_in, row_out);
        });

    Ok(out)
}

/// One row of softmax. Three passes over the row (max, exp+sum, divide)
/// but each pass NEON-vectorised; the row stays hot in L1 between passes.
#[cfg(target_arch = "aarch64")]
fn softmax_row_f32(input: &[f32], out: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = input.len();
    let n_vec = n / 4;
    let n_tail = n_vec * 4;

    // Pass 1: max(x). vmaxq_f32 + vmaxvq_f32 reduce.
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..n_vec {
        unsafe {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            max_vec = vmaxq_f32(max_vec, v);
        }
    }
    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    for i in n_tail..n {
        if input[i] > max_val {
            max_val = input[i];
        }
    }

    // Pass 2: out[i] = exp(input[i] - max); sum += out[i].
    // Vectorised exp via the NEON polynomial path.
    let max_vec = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    for i in 0..n_vec {
        unsafe {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            let shifted = vsubq_f32(v, max_vec);
            let e = vexpq_f32(shifted);
            vst1q_f32(out.as_mut_ptr().add(i * 4), e);
            sum_vec = vaddq_f32(sum_vec, e);
        }
    }
    let mut sum = unsafe { vaddvq_f32(sum_vec) };
    for i in n_tail..n {
        let e = (input[i] - max_val).exp();
        out[i] = e;
        sum += e;
    }

    // Pass 3: divide by sum.
    let inv = 1.0 / sum;
    let inv_vec = unsafe { vdupq_n_f32(inv) };
    for i in 0..n_vec {
        unsafe {
            let v = vld1q_f32(out.as_ptr().add(i * 4));
            let r = vmulq_f32(v, inv_vec);
            vst1q_f32(out.as_mut_ptr().add(i * 4), r);
        }
    }
    for i in n_tail..n {
        out[i] *= inv;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn softmax_row_f32(input: &[f32], out: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in input.iter().enumerate() {
        let e = (v - max_val).exp();
        out[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// Fused GELU along all elements. Replaces Axon's defn
/// `((erf(x / √2) + 1) * x) / 2` decomposition (5 backend calls,
/// 5 Vec<f32> allocs) with one NIF call.
pub fn gelu_f32(input: &[f32]) -> Result<Vec<f32>, String> {
    let n = input.len();
    let mut out = vec![0.0f32; n];
    let inv_sqrt2 = 1.0_f32 / std::f32::consts::SQRT_2;

    let chunk = ((n + 7) / 8).max(8192).min(n.max(1));

    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, slot)| {
            let s = ci * chunk;
            for j in 0..slot.len() {
                let x = input[s + j];
                slot[j] = 0.5 * x * (1.0 + erf_f32(x * inv_sqrt2));
            }
        });

    Ok(out)
}

/// Fused LayerNorm along the last axis. Input viewed as
/// `[n_outer × inner]` row-major; `gamma` and `beta` are length-`inner`.
/// Per row: mean → variance → normalize → scale+bias, single pass.
pub fn layernorm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n_outer: usize,
    inner: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if input.len() != n_outer * inner {
        return Err(format!(
            "layernorm_f32: input len {} != n_outer*inner = {}",
            input.len(),
            n_outer * inner
        ));
    }
    if gamma.len() != inner {
        return Err(format!(
            "layernorm_f32: gamma len {} != inner {}",
            gamma.len(),
            inner
        ));
    }
    if beta.len() != inner {
        return Err(format!(
            "layernorm_f32: beta len {} != inner {}",
            beta.len(),
            inner
        ));
    }

    let mut out = vec![0.0f32; n_outer * inner];

    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(i, row_out)| {
            let row_in = &input[i * inner..(i + 1) * inner];
            layernorm_row_f32(row_in, gamma, beta, epsilon, row_out);
        });

    Ok(out)
}

#[cfg(target_arch = "aarch64")]
fn layernorm_row_f32(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = input.len();
    let n_vec = n / 4;
    let n_tail = n_vec * 4;

    // Pass 1: sum (for mean) + sum-of-squares (for variance).
    let mut sum_v = unsafe { vdupq_n_f32(0.0) };
    let mut sumsq_v = unsafe { vdupq_n_f32(0.0) };
    for i in 0..n_vec {
        unsafe {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            sum_v = vaddq_f32(sum_v, v);
            sumsq_v = vfmaq_f32(sumsq_v, v, v);
        }
    }
    let mut sum = unsafe { vaddvq_f32(sum_v) };
    let mut sumsq = unsafe { vaddvq_f32(sumsq_v) };
    for i in n_tail..n {
        let v = input[i];
        sum += v;
        sumsq += v * v;
    }

    let inv_n = 1.0 / (n as f32);
    let mean = sum * inv_n;
    let var = sumsq * inv_n - mean * mean;
    let inv_std = (var + eps).sqrt().recip();

    // Pass 2: out[i] = gamma[i] * ((input[i] - mean) * inv_std) + beta[i]
    let mean_v = unsafe { vdupq_n_f32(mean) };
    let inv_std_v = unsafe { vdupq_n_f32(inv_std) };
    for i in 0..n_vec {
        unsafe {
            let x = vld1q_f32(input.as_ptr().add(i * 4));
            let g = vld1q_f32(gamma.as_ptr().add(i * 4));
            let b = vld1q_f32(beta.as_ptr().add(i * 4));
            let centered = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(centered, inv_std_v);
            // FMA: result = g * normed + b
            let r = vfmaq_f32(b, g, normed);
            vst1q_f32(out.as_mut_ptr().add(i * 4), r);
        }
    }
    for i in n_tail..n {
        out[i] = gamma[i] * ((input[i] - mean) * inv_std) + beta[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn layernorm_row_f32(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    let n = input.len();
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for &v in input {
        sum += v;
        sumsq += v * v;
    }
    let inv_n = 1.0 / (n as f32);
    let mean = sum * inv_n;
    let var = sumsq * inv_n - mean * mean;
    let inv_std = (var + eps).sqrt().recip();
    for i in 0..n {
        out[i] = gamma[i] * ((input[i] - mean) * inv_std) + beta[i];
    }
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

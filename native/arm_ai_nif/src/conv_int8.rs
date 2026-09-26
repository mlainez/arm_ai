//! CPU 2-D convolution with int8-quantised weights and f32 activations.
//!
//! Layout (per-output-channel int8 quantization, as used by
//! `NxPrimitives.QuantizedConv`):
//!
//! * `input`   : `[N, H, W, Cin]` f32, channels-last (NHWC).
//! * `weight`  : `[Cout, Kh*Kw*Cin]` int8, row-major. Each output channel
//!   is a flat vector indexed by `((kh * Kw) + kw) * Cin + ci`, which
//!   matches a row-major flatten of a Bumblebee / Axon HWIO kernel
//!   `[Kh, Kw, Cin, Cout]` after transposing to put `Cout` first.
//! * `scales`  : `[Cout]` f32, one symmetric scale per output channel.
//! * `bias`    : `[Cout]` f32, or empty for no bias.
//!
//! Output: `[N, H_out, W_out, Cout]` f32, with
//! `H_out = (H + pad_top + pad_bottom - Kh) / stride_h + 1`
//! and the analogous formula for `W_out`.
//!
//! The inner channel dot product has a NEON path on aarch64 and a scalar
//! fallback elsewhere. `nx_primitives` tests the result against `Nx.conv`.

use crate::shape_ops;

/// Dot product of one row of weights (int8) against one row of
/// activations (f32). Pulled out as a separate function so the NEON
/// path can specialise it on aarch64 without uglifying the outer
/// conv loop.
///
/// Returns `sum_i (weights[i] as f32) * input[i]`. The per-output-
/// channel scale is applied by the caller, not here, so we can keep
/// the accumulator type narrow.
#[inline(always)]
fn ci_dot(weights: &[i8], input: &[f32]) -> f32 {
    debug_assert_eq!(weights.len(), input.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        ci_dot_neon(weights, input)
    }

    #[cfg(not(target_arch = "aarch64"))]
    ci_dot_scalar(weights, input)
}

#[inline(always)]
#[allow(dead_code)]
fn ci_dot_scalar(weights: &[i8], input: &[f32]) -> f32 {
    let mut acc: f32 = 0.0;
    for i in 0..weights.len() {
        acc += weights[i] as f32 * input[i];
    }
    acc
}

/// NEON int8 → f32 dot product. Processes 8 channels per iteration:
/// load 8 int8 weights, widen to int16 then to two int32 halves,
/// convert to f32, FMA against the matching 8 f32 inputs. Falls back
/// to scalar for the trailing 0-7 channels.
///
/// This is the simplest correct NEON shape for "weight-only int8"
/// (a.k.a. int8 storage, f32 compute). It does not extract the int8
/// throughput a fully-int8 dot product (sdot / udot, ARMv8.2-A) would,
/// because Snapdragon 632 / Cortex-A73 predates those instructions.
/// What it does buy is 8-wide f32 FMA in the inner loop instead of
/// one scalar multiply-add per iteration: ~4x throughput on this CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ci_dot_neon(weights: &[i8], input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = weights.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i + 8 <= n {
        // 8 int8 weights -> int16x8
        let w_i8 = vld1_s8(weights.as_ptr().add(i));
        let w_i16 = vmovl_s8(w_i8);

        // Split int16x8 into two int32x4 halves, convert to f32x4.
        let w_lo_i32 = vmovl_s16(vget_low_s16(w_i16));
        let w_hi_i32 = vmovl_high_s16(w_i16);
        let w_lo_f32 = vcvtq_f32_s32(w_lo_i32);
        let w_hi_f32 = vcvtq_f32_s32(w_hi_i32);

        // Matching f32 inputs.
        let x_lo = vld1q_f32(input.as_ptr().add(i));
        let x_hi = vld1q_f32(input.as_ptr().add(i + 4));

        // acc += w * x  (fused multiply-add).
        acc0 = vfmaq_f32(acc0, w_lo_f32, x_lo);
        acc1 = vfmaq_f32(acc1, w_hi_f32, x_hi);

        i += 8;
    }

    // Combine the two accumulators and reduce horizontally.
    let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));

    // Scalar tail for the trailing 0-7 channels.
    while i < n {
        acc += weights[i] as f32 * input[i];
        i += 1;
    }

    acc
}

/// Compute output H/W from input dims, kernel dims, stride and padding.
#[inline]
pub fn output_dims(
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> (usize, usize) {
    let h_out = (h_in + pad_top + pad_bottom - kh) / stride_h + 1;
    let w_out = (w_in + pad_left + pad_right - kw) / stride_w + 1;
    (h_out, w_out)
}

/// Run the conv. Writes `output_f32` (must be sized
/// `n * h_out * w_out * c_out`).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_int8(
    input_f32: &[f32],
    weight_i8: &[i8],
    scales_f32: &[f32],
    bias_f32: Option<&[f32]>,
    output_f32: &mut [f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) {
    let (h_out, w_out) =
        output_dims(h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);

    debug_assert_eq!(input_f32.len(), n * h_in * w_in * c_in);
    debug_assert_eq!(weight_i8.len(), c_out * kh * kw * c_in);
    debug_assert_eq!(scales_f32.len(), c_out);
    if let Some(b) = bias_f32 {
        debug_assert_eq!(b.len(), c_out);
    }
    debug_assert_eq!(output_f32.len(), n * h_out * w_out * c_out);

    let kk = kh * kw * c_in;
    let in_stride_n = h_in * w_in * c_in;
    let in_stride_h = w_in * c_in;
    let out_stride_n = h_out * w_out * c_out;
    let out_stride_h = w_out * c_out;

    for ni in 0..n {
        let in_base = ni * in_stride_n;
        let out_base = ni * out_stride_n;

        for ho in 0..h_out {
            for wo in 0..w_out {
                // For this output spatial location, accumulate one f32
                // per output channel.
                let out_offset = out_base + ho * out_stride_h + wo * c_out;

                // Initialise accumulators with bias (or zero).
                for co in 0..c_out {
                    output_f32[out_offset + co] = match bias_f32 {
                        Some(b) => b[co],
                        None => 0.0,
                    };
                }

                // Walk the kernel window.
                for kh_i in 0..kh {
                    // Compute the input row this kernel row reads from,
                    // accounting for stride and padding.
                    let h_in_signed = (ho * stride_h) as isize + kh_i as isize - pad_top as isize;
                    if h_in_signed < 0 || h_in_signed >= h_in as isize {
                        continue;
                    }
                    let h_in_i = h_in_signed as usize;

                    for kw_i in 0..kw {
                        let w_in_signed =
                            (wo * stride_w) as isize + kw_i as isize - pad_left as isize;
                        if w_in_signed < 0 || w_in_signed >= w_in as isize {
                            continue;
                        }
                        let w_in_i = w_in_signed as usize;

                        // The input patch row for this kernel position.
                        let in_row =
                            in_base + h_in_i * in_stride_h + w_in_i * c_in;
                        // Position within the flat weight vector.
                        let w_row_offset = (kh_i * kw + kw_i) * c_in;

                        // Per-output-channel accumulation. The inner
                        // ci-loop is the NEON vectorisation target.
                        for co in 0..c_out {
                            let w_base = co * kk + w_row_offset;
                            let acc = ci_dot(
                                &weight_i8[w_base..w_base + c_in],
                                &input_f32[in_row..in_row + c_in],
                            );
                            // scales[co] is applied once per (ho, wo, co)
                            // — pulled outside the (kh, kw) loop because
                            // every weight in row `co` shares the same
                            // scale.
                            output_f32[out_offset + co] += acc * scales_f32[co];
                        }
                    }
                }
            }
        }
    }
}

/// Pure-f32 conv2d. Same NHWC + flat-weight layout as `conv2d_int8`
/// but with f32 weights and no per-channel scale.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    input_f32: &[f32],
    weight_f32: &[f32],
    bias_f32: Option<&[f32]>,
    output_f32: &mut [f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) {
    let (h_out, w_out) =
        output_dims(h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);

    debug_assert_eq!(input_f32.len(), n * h_in * w_in * c_in);
    debug_assert_eq!(weight_f32.len(), c_out * kh * kw * c_in);
    if let Some(b) = bias_f32 {
        debug_assert_eq!(b.len(), c_out);
    }
    debug_assert_eq!(output_f32.len(), n * h_out * w_out * c_out);

    let kk = kh * kw * c_in;
    let in_stride_n = h_in * w_in * c_in;
    let in_stride_h = w_in * c_in;
    let out_stride_n = h_out * w_out * c_out;
    let out_stride_h = w_out * c_out;

    for ni in 0..n {
        let in_base = ni * in_stride_n;
        let out_base = ni * out_stride_n;

        for ho in 0..h_out {
            for wo in 0..w_out {
                let out_offset = out_base + ho * out_stride_h + wo * c_out;

                for co in 0..c_out {
                    output_f32[out_offset + co] = match bias_f32 {
                        Some(b) => b[co],
                        None => 0.0,
                    };
                }

                for kh_i in 0..kh {
                    let h_in_signed = (ho * stride_h) as isize + kh_i as isize - pad_top as isize;
                    if h_in_signed < 0 || h_in_signed >= h_in as isize {
                        continue;
                    }
                    let h_in_i = h_in_signed as usize;

                    for kw_i in 0..kw {
                        let w_in_signed =
                            (wo * stride_w) as isize + kw_i as isize - pad_left as isize;
                        if w_in_signed < 0 || w_in_signed >= w_in as isize {
                            continue;
                        }
                        let w_in_i = w_in_signed as usize;

                        let in_row = in_base + h_in_i * in_stride_h + w_in_i * c_in;
                        let w_row_offset = (kh_i * kw + kw_i) * c_in;

                        for co in 0..c_out {
                            let w_base = co * kk + w_row_offset;
                            let acc = ci_dot_f32(
                                &weight_f32[w_base..w_base + c_in],
                                &input_f32[in_row..in_row + c_in],
                            );
                            output_f32[out_offset + co] += acc;
                        }
                    }
                }
            }
        }
    }
}

/// f32 ⋅ f32 row dot. NEON path processes 8 f32 per iteration via two
/// vfmaq_f32 calls. Scalar fallback for non-aarch64 hosts.
#[inline(always)]
/// Winograd F(2, 3) convolution: 3×3 stride-1 conv via the 16-multiply
/// transform-based algorithm. Halves MAC count vs direct conv at the
/// cost of extra add/sub work. NHWC layout, processes output in 2×2
/// tiles.
///
/// For a 3×3 conv producing an O×O output, direct convolution does
/// 9·O² multiplies per channel pair. Winograd F(2,3) groups outputs
/// into 2×2 tiles and produces each tile with 16 multiplies — 16/(2²·9)
/// = 0.44× the multiplies, a 56% reduction. Extra add/sub work brings
/// realised speedup to ~1.5–2× on cache-friendly shapes.
///
/// Currently only the math kernel is exposed (`winograd_f23_tile`);
/// wiring into `backend.conv` is future work guarded by shape
/// detection (kh==kw==3, sh==sw==1, no dilation).
#[allow(dead_code)]
pub fn winograd_f23_tile_f32(
    input_tile: &[f32; 16],
    kernel_3x3: &[f32; 9],
    out_tile: &mut [f32; 4],
) {
    // Pre-transform the 3×3 kernel to 4×4 (typically done once per
    // kernel at model load, but we recompute here for the per-tile
    // function): U = G g G^T where
    //   G = [[1,    0,    0],
    //        [1/2,  1/2,  1/2],
    //        [1/2, -1/2,  1/2],
    //        [0,    0,    1]]
    let mut u = [0.0f32; 16];
    {
        // G g, shape (4, 3).
        let mut gg = [0.0f32; 12];
        for j in 0..3 {
            let g00 = kernel_3x3[0 * 3 + j];
            let g10 = kernel_3x3[1 * 3 + j];
            let g20 = kernel_3x3[2 * 3 + j];
            gg[0 * 3 + j] = g00;
            gg[1 * 3 + j] = 0.5 * (g00 + g10 + g20);
            gg[2 * 3 + j] = 0.5 * (g00 - g10 + g20);
            gg[3 * 3 + j] = g20;
        }
        // (G g) G^T, shape (4, 4).
        for i in 0..4 {
            let r0 = gg[i * 3 + 0];
            let r1 = gg[i * 3 + 1];
            let r2 = gg[i * 3 + 2];
            u[i * 4 + 0] = r0;
            u[i * 4 + 1] = 0.5 * (r0 + r1 + r2);
            u[i * 4 + 2] = 0.5 * (r0 - r1 + r2);
            u[i * 4 + 3] = r2;
        }
    }

    // Transform the 4×4 input tile: V = B^T d B where
    //   B^T = [[ 1,  0, -1,  0],
    //          [ 0,  1,  1,  0],
    //          [ 0, -1,  1,  0],
    //          [ 0,  1,  0, -1]]
    let mut v = [0.0f32; 16];
    {
        // B^T d
        let mut btd = [0.0f32; 16];
        for j in 0..4 {
            let d0 = input_tile[0 * 4 + j];
            let d1 = input_tile[1 * 4 + j];
            let d2 = input_tile[2 * 4 + j];
            let d3 = input_tile[3 * 4 + j];
            btd[0 * 4 + j] = d0 - d2;
            btd[1 * 4 + j] = d1 + d2;
            btd[2 * 4 + j] = -d1 + d2;
            btd[3 * 4 + j] = d1 - d3;
        }
        // (B^T d) B
        for i in 0..4 {
            let r0 = btd[i * 4 + 0];
            let r1 = btd[i * 4 + 1];
            let r2 = btd[i * 4 + 2];
            let r3 = btd[i * 4 + 3];
            v[i * 4 + 0] = r0 - r2;
            v[i * 4 + 1] = r1 + r2;
            v[i * 4 + 2] = -r1 + r2;
            v[i * 4 + 3] = r1 - r3;
        }
    }

    // Element-wise multiply U ⊙ V (16 multiplies — the speedup
    // happens here vs the 36 of direct conv).
    let mut m = [0.0f32; 16];
    for i in 0..16 {
        m[i] = u[i] * v[i];
    }

    // Inverse transform: Y = A^T M A, with
    //   A^T = [[1, 1,  1,  0],
    //          [0, 1, -1, -1]]
    // Result is 2×2.
    let mut atm = [0.0f32; 8];
    for j in 0..4 {
        let m0 = m[0 * 4 + j];
        let m1 = m[1 * 4 + j];
        let m2 = m[2 * 4 + j];
        let m3 = m[3 * 4 + j];
        atm[0 * 4 + j] = m0 + m1 + m2;
        atm[1 * 4 + j] = m1 - m2 - m3;
    }
    for i in 0..2 {
        let r0 = atm[i * 4 + 0];
        let r1 = atm[i * 4 + 1];
        let r2 = atm[i * 4 + 2];
        let r3 = atm[i * 4 + 3];
        out_tile[i * 2 + 0] = r0 + r1 + r2;
        out_tile[i * 2 + 1] = r1 - r2 - r3;
    }
}

/// Full 3×3 stride-1 NHWC convolution via Winograd F(2, 3) tiles.
/// Input layout (N, H, W, C_in), weight layout (C_out, 3, 3, C_in),
/// output layout (N, H_out, W_out, C_out). H_out and W_out come from
/// the standard formula with stride 1, dilation 1.
///
/// Output is produced in 2×2 tiles; rows/columns past the true
/// output are trimmed at write time, so any H_out × W_out is fine.
pub fn conv2d_f32_winograd_3x3(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) {
    let h_out = h_in + pad_top + pad_bottom - 2;
    let w_out = w_in + pad_left + pad_right - 2;

    // Pre-transform all weights G g G^T into U[c_out, c_in, 16].
    let mut u_all = vec![0.0f32; c_out * c_in * 16];
    for oc in 0..c_out {
        for ic in 0..c_in {
            let mut g = [0.0f32; 9];
            for ky in 0..3 {
                for kx in 0..3 {
                    g[ky * 3 + kx] = weight[((oc * 3 + ky) * 3 + kx) * c_in + ic];
                }
            }
            // G g, shape (4, 3).
            let mut gg = [0.0f32; 12];
            for j in 0..3 {
                let g0 = g[0 * 3 + j];
                let g1 = g[1 * 3 + j];
                let g2 = g[2 * 3 + j];
                gg[0 * 3 + j] = g0;
                gg[1 * 3 + j] = 0.5 * (g0 + g1 + g2);
                gg[2 * 3 + j] = 0.5 * (g0 - g1 + g2);
                gg[3 * 3 + j] = g2;
            }
            // (G g) G^T, shape (4, 4).
            let base = (oc * c_in + ic) * 16;
            for i in 0..4 {
                let r0 = gg[i * 3 + 0];
                let r1 = gg[i * 3 + 1];
                let r2 = gg[i * 3 + 2];
                u_all[base + i * 4 + 0] = r0;
                u_all[base + i * 4 + 1] = 0.5 * (r0 + r1 + r2);
                u_all[base + i * 4 + 2] = 0.5 * (r0 - r1 + r2);
                u_all[base + i * 4 + 3] = r2;
            }
        }
    }

    let n_th = (h_out + 1) / 2;
    let n_tw = (w_out + 1) / 2;

    for nn in 0..n {
        for th in 0..n_th {
            for tw in 0..n_tw {
                let oh0 = th * 2;
                let ow0 = tw * 2;

                let mut acc_oc = vec![[0.0f32; 4]; c_out];

                for ic in 0..c_in {
                    // Gather 4×4 input tile with zero-padding.
                    let mut d = [0.0f32; 16];
                    for i in 0..4 {
                        let ih = oh0 as isize + i as isize - pad_top as isize;
                        if ih < 0 || (ih as usize) >= h_in {
                            continue;
                        }
                        let ih_u = ih as usize;
                        for j in 0..4 {
                            let iw = ow0 as isize + j as isize - pad_left as isize;
                            if iw < 0 || (iw as usize) >= w_in {
                                continue;
                            }
                            let iw_u = iw as usize;
                            let idx = ((nn * h_in + ih_u) * w_in + iw_u) * c_in + ic;
                            d[i * 4 + j] = input[idx];
                        }
                    }

                    // V = B^T d B.
                    let mut v = [0.0f32; 16];
                    let mut btd = [0.0f32; 16];
                    for j in 0..4 {
                        let d0 = d[0 * 4 + j];
                        let d1 = d[1 * 4 + j];
                        let d2 = d[2 * 4 + j];
                        let d3 = d[3 * 4 + j];
                        btd[0 * 4 + j] = d0 - d2;
                        btd[1 * 4 + j] = d1 + d2;
                        btd[2 * 4 + j] = -d1 + d2;
                        btd[3 * 4 + j] = d1 - d3;
                    }
                    for i in 0..4 {
                        let r0 = btd[i * 4 + 0];
                        let r1 = btd[i * 4 + 1];
                        let r2 = btd[i * 4 + 2];
                        let r3 = btd[i * 4 + 3];
                        v[i * 4 + 0] = r0 - r2;
                        v[i * 4 + 1] = r1 + r2;
                        v[i * 4 + 2] = -r1 + r2;
                        v[i * 4 + 3] = r1 - r3;
                    }

                    for oc in 0..c_out {
                        let base = (oc * c_in + ic) * 16;
                        let u = &u_all[base..base + 16];

                        let mut m = [0.0f32; 16];
                        for k in 0..16 {
                            m[k] = u[k] * v[k];
                        }

                        let mut atm = [0.0f32; 8];
                        for j in 0..4 {
                            let m0 = m[0 * 4 + j];
                            let m1 = m[1 * 4 + j];
                            let m2 = m[2 * 4 + j];
                            let m3 = m[3 * 4 + j];
                            atm[0 * 4 + j] = m0 + m1 + m2;
                            atm[1 * 4 + j] = m1 - m2 - m3;
                        }
                        for i in 0..2 {
                            let r0 = atm[i * 4 + 0];
                            let r1 = atm[i * 4 + 1];
                            let r2 = atm[i * 4 + 2];
                            let r3 = atm[i * 4 + 3];
                            acc_oc[oc][i * 2 + 0] += r0 + r1 + r2;
                            acc_oc[oc][i * 2 + 1] += r1 - r2 - r3;
                        }
                    }
                }

                for oc in 0..c_out {
                    let b = bias.map(|bs| bs[oc]).unwrap_or(0.0);
                    for i in 0..2 {
                        for j in 0..2 {
                            let oh = oh0 + i;
                            let ow = ow0 + j;
                            if oh < h_out && ow < w_out {
                                let idx = ((nn * h_out + oh) * w_out + ow) * c_out + oc;
                                output[idx] = acc_oc[oc][i * 2 + j] + b;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Depthwise 2-D convolution. Per channel c in 0..Cin we compute
///   `out[n, ho, wo, c] = sum_{kh, kw} input[n, ho*Sh+kh-Pt, wo*Sw+kw-Pl, c]
///                                     * kernel[c, kh, kw]`
/// then add `bias[c]` if provided.
///
/// Used by MobileNet/EfficientNet families' depthwise+pointwise
/// alternating blocks. With feature_group_size == Cin, Nx hands us
/// kernel {Cin, 1, Kh, Kw}; we treat the 1-dim as squeezed to
/// {Cin, Kh, Kw}.
///
/// Layout (NHWC) input + per-channel weights matches our existing
/// conv2d_f32 conventions; output is NHWC f32.
pub fn depthwise_conv2d_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    _n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) {
    let (h_out, w_out) = output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right,
    );

    // Parallelise over (n, ho). Per work unit fills one row of one
    // batch's output: w_out * c_in output cells. Picking (n, ho) as
    // the parallel axes keeps the kernel + per-channel state hot in
    // L1 within the loop and gives plenty of work units (n * h_out =
    // e.g. 1 * 56 = 56 for a MobileNet block).
    use rayon::prelude::*;

    let c_in_ = c_in;
    let w_out_ = w_out;
    let h_in_ = h_in as isize;
    let w_in_ = w_in as isize;
    let pad_top_ = pad_top as isize;
    let pad_left_ = pad_left as isize;

    out.par_chunks_mut(w_out_ * c_in_)
        .enumerate()
        .for_each(|(nh, row_out)| {
            let batch = nh / h_out;
            let ho = nh % h_out;
            let in_batch_off = batch * h_in * w_in * c_in;

            for wo in 0..w_out {
                let cell_out_off = wo * c_in_;
                let h_orig_base = ho as isize * stride_h as isize - pad_top_;
                let w_orig_base = wo as isize * stride_w as isize - pad_left_;

                // Per-channel accumulator, reset each output cell.
                // The inner loops over kh × kw are short (typically 3
                // or 5) — the compiler can unroll + vectorise across
                // the contiguous c_in run inside.
                for c in 0..c_in_ {
                    let mut acc = bias.map(|b| b[c]).unwrap_or(0.0);

                    for kh_i in 0..kh {
                        let h_in_idx = h_orig_base + kh_i as isize;
                        if h_in_idx < 0 || h_in_idx >= h_in_ {
                            continue;
                        }
                        let h_in_idx = h_in_idx as usize;

                        for kw_i in 0..kw {
                            let w_in_idx = w_orig_base + kw_i as isize;
                            if w_in_idx < 0 || w_in_idx >= w_in_ {
                                continue;
                            }
                            let w_in_idx = w_in_idx as usize;

                            let in_off =
                                in_batch_off + (h_in_idx * w_in + w_in_idx) * c_in_ + c;
                            let wt_off = c * kh * kw + kh_i * kw + kw_i;
                            acc += input[in_off] * weight[wt_off];
                        }
                    }

                    row_out[cell_out_off + c] = acc;
                }
            }

            // Silence unused-var warnings when bias is None.
            let _ = (&bias, pad_bottom, pad_right);
        });
}

fn ci_dot_f32(weights: &[f32], input: &[f32]) -> f32 {
    debug_assert_eq!(weights.len(), input.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        ci_dot_f32_neon(weights, input)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = 0.0_f32;
        for i in 0..weights.len() {
            acc += weights[i] * input[i];
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ci_dot_f32_neon(weights: &[f32], input: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let n = weights.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i + 8 <= n {
        let w_lo = vld1q_f32(weights.as_ptr().add(i));
        let w_hi = vld1q_f32(weights.as_ptr().add(i + 4));
        let x_lo = vld1q_f32(input.as_ptr().add(i));
        let x_hi = vld1q_f32(input.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, w_lo, x_lo);
        acc1 = vfmaq_f32(acc1, w_hi, x_hi);
        i += 8;
    }

    let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        acc += weights[i] * input[i];
        i += 1;
    }
    acc
}

/// Fused depthwise + 1×1 pointwise conv (the MobileNet / EfficientNet
/// building block). Avoids materialising the intermediate
/// `(N, H_out, W_out, Cin)` activation tensor: for each output
/// position we compute the depthwise vector into a `Cin`-sized scratch
/// buffer, optionally apply an activation, then do the pointwise GEMV
/// against the (Cout, Cin) pointwise weight in one shot — keeping the
/// depthwise output values in L1 across both consumers.
///
/// `activation`: 0 = none, 1 = ReLU, 2 = ReLU6 (the MobileNet default).
///
/// Layouts: input NHWC, depthwise weight `(Cin, Kh, Kw)`, pointwise
/// weight `(Cout, Cin)`, both biases optional.
pub fn depthwise_pointwise_f32(
    input: &[f32],
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    output: &mut [f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    activation: u8,
) {
    let (h_out, w_out) = output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w,
        pad_top, pad_bottom, pad_left, pad_right,
    );

    let mut dw_scratch = vec![0.0f32; c_in];

    for nn in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                // 1. Compute depthwise vector (Cin floats) at (oh, ow).
                for ci in 0..c_in {
                    let mut acc = 0.0f32;
                    for ky in 0..kh {
                        let ih = oh as isize * stride_h as isize + ky as isize
                            - pad_top as isize;
                        if ih < 0 || (ih as usize) >= h_in {
                            continue;
                        }
                        let ih_u = ih as usize;
                        for kx in 0..kw {
                            let iw = ow as isize * stride_w as isize + kx as isize
                                - pad_left as isize;
                            if iw < 0 || (iw as usize) >= w_in {
                                continue;
                            }
                            let iw_u = iw as usize;
                            let inp = input[((nn * h_in + ih_u) * w_in + iw_u) * c_in + ci];
                            let wt = dw_weight[(ci * kh + ky) * kw + kx];
                            acc += inp * wt;
                        }
                    }
                    if let Some(b) = dw_bias {
                        acc += b[ci];
                    }
                    // Activation between depthwise and pointwise.
                    acc = match activation {
                        1 => acc.max(0.0),
                        2 => acc.max(0.0).min(6.0),
                        _ => acc,
                    };
                    dw_scratch[ci] = acc;
                }

                // 2. Pointwise GEMV: out[oh, ow, :] = pw_weight @ dw_scratch.
                let out_base = ((nn * h_out + oh) * w_out + ow) * c_out;
                for oc in 0..c_out {
                    let mut acc = 0.0f32;
                    let w_base = oc * c_in;
                    for ci in 0..c_in {
                        acc += pw_weight[w_base + ci] * dw_scratch[ci];
                    }
                    if let Some(b) = pw_bias {
                        acc += b[oc];
                    }
                    output[out_base + oc] = acc;
                }
            }
        }
    }
}

/// im2col + GEMM convolution. For general `Kh`, `Kw`, stride, dilation
/// (=1) and padding this packs each output position's receptive field
/// into one row of an `M × K` activation matrix
/// (`M = N * H_out * W_out`, `K = Kh * Kw * Cin`), then performs a
/// single matmul against a weight matrix shape `K × Cout` to produce
/// the `M × Cout` output. Reuses the cache-blocked NEON kernel from
/// `shape_ops`.
///
/// The temporary activation matrix uses `M * K * 4` bytes — for ViT
/// patch16 stem on a 224 image this is ~1.5 MB, well within budget.
/// For very large feature maps it should still fit in L2.
pub fn conv2d_f32_im2col(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) {
    let (h_out, w_out) = output_dims(
        h_in, w_in, kh, kw, stride_h, stride_w,
        pad_top, pad_bottom, pad_left, pad_right,
    );

    let m = n * h_out * w_out;
    let k_inner = kh * kw * c_in;

    // Pack receptive fields into a row-major (M, K) matrix.
    let mut a = vec![0.0f32; m * k_inner];
    for nn in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = (nn * h_out + oh) * w_out + ow;
                let row_base = row * k_inner;
                for ky in 0..kh {
                    let ih = oh as isize * stride_h as isize + ky as isize
                        - pad_top as isize;
                    if ih < 0 || (ih as usize) >= h_in {
                        continue;
                    }
                    let ih_u = ih as usize;
                    for kx in 0..kw {
                        let iw = ow as isize * stride_w as isize + kx as isize
                            - pad_left as isize;
                        if iw < 0 || (iw as usize) >= w_in {
                            continue;
                        }
                        let iw_u = iw as usize;
                        let col_base = (ky * kw + kx) * c_in;
                        let inp_base = ((nn * h_in + ih_u) * w_in + iw_u) * c_in;
                        a[row_base + col_base..row_base + col_base + c_in]
                            .copy_from_slice(&input[inp_base..inp_base + c_in]);
                    }
                }
            }
        }
    }

    // Repack weight from (Cout, Kh, Kw, Cin) to (Kh*Kw*Cin, Cout) row-major.
    let mut b = vec![0.0f32; k_inner * c_out];
    for oc in 0..c_out {
        for ky in 0..kh {
            for kx in 0..kw {
                for ic in 0..c_in {
                    let src = ((oc * kh + ky) * kw + kx) * c_in + ic;
                    let dst = ((ky * kw + kx) * c_in + ic) * c_out + oc;
                    b[dst] = weight[src];
                }
            }
        }
    }

    // im2col + GEMM via gemm crate. `a` is (M, K=k_inner), `b` is
    // (K=k_inner, N=c_out), output is (M, N=c_out) — all row-major.
    let _ = shape_ops::batched_matmul_f32(&a, &b, 1, m, c_out, k_inner, false)
        .map(|v| output.copy_from_slice(&v));

    if let Some(bs) = bias {
        for i in 0..m {
            let base = i * c_out;
            for oc in 0..c_out {
                output[base + oc] += bs[oc];
            }
        }
    }
    // Use `out` to silence dead-write warning when bias is None — the
    // matmul has already written it.
    let _ = h_out;
    let _ = w_out;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1×3×3×1 input, 1×3×3×1 kernel, no padding, stride 1: a single
    /// dot product. Lets us check the inner loop without any of the
    /// indexing machinery.
    #[test]
    fn single_window_correctness() {
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight: Vec<i8> = vec![1; 9];
        let scales = vec![1.0_f32];
        let mut out = vec![0.0_f32; 1];

        conv2d_int8(
            &input, &weight, &scales, None, &mut out,
            1, 3, 3, 1, 1, 3, 3, 1, 1, 0, 0, 0, 0,
        );

        assert_eq!(out[0], 45.0); // 1+2+3+4+5+6+7+8+9
    }

    /// Same but with a per-channel scale of 0.5 — output should halve.
    #[test]
    fn per_channel_scale() {
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight: Vec<i8> = vec![1; 9];
        let scales = vec![0.5_f32];
        let mut out = vec![0.0_f32; 1];

        conv2d_int8(
            &input, &weight, &scales, None, &mut out,
            1, 3, 3, 1, 1, 3, 3, 1, 1, 0, 0, 0, 0,
        );

        assert_eq!(out[0], 22.5);
    }

    /// Two output channels, distinct scales and biases. Confirms the
    /// per-channel scale + bias application is correct.
    #[test]
    fn two_output_channels_with_bias() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 1x2x2x1
        // Cout=2, Kh=2, Kw=2, Cin=1 → 4 weights per Cout.
        let weight: Vec<i8> = vec![
            1, 1, 1, 1,   // co=0: sum
            1, 0, 0, -1,  // co=1: top-left minus bottom-right
        ];
        let scales = vec![1.0_f32, 2.0_f32];
        let bias = vec![10.0_f32, -5.0_f32];
        let mut out = vec![0.0_f32; 2];

        conv2d_int8(
            &input, &weight, &scales, Some(&bias), &mut out,
            1, 2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0,
        );

        // co=0: (1+2+3+4) * 1.0 + 10 = 20
        // co=1: (1 - 4)   * 2.0 - 5  = -11
        assert_eq!(out[0], 20.0);
        assert_eq!(out[1], -11.0);
    }

    /// Padding test: 3×3 input, 3×3 kernel of all-ones with stride 1 and
    /// pad 1 on every side → output 3×3, with corners summing only the
    /// in-bounds region.
    #[test]
    fn padding_zeroes_at_boundary() {
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight: Vec<i8> = vec![1; 9];
        let scales = vec![1.0_f32];
        let mut out = vec![0.0_f32; 9];

        conv2d_int8(
            &input, &weight, &scales, None, &mut out,
            1, 3, 3, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1,
        );

        // top-left output sees: 0 0 0 / 0 1 2 / 0 4 5 = 12
        // centre output sees the full window = 45
        // bottom-right sees: 5 6 0 / 8 9 0 / 0 0 0 = 28
        assert_eq!(out[0], 12.0);
        assert_eq!(out[4], 45.0);
        assert_eq!(out[8], 28.0);
    }
}

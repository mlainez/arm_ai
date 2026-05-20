//! CPU 2-D convolution with int8-quantised weights and f32 activations.
//!
//! Layout (matches what `SmolLLM.Quantize.write!` produces, extended to
//! 4-D conv kernels):
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
//! First version: scalar Rust, no NEON intrinsics. Used to nail down
//! correctness and the wire format before reaching for SIMD. The hot
//! ci-loop is the obvious vectorisation target — once correctness is
//! validated against `Nx.conv` on host and on-device, swap the inner
//! loop for `core::arch::aarch64` int8x8 widening + f32 FMA.

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
/// but with f32 weights and no per-channel scale. Used by the generic
/// `NxCL.Backend.conv` dispatch when the kernel is a normal `Nx.Tensor`
/// rather than a pre-quantised `NxCL.QuantizedConv`.
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
    n: usize,
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

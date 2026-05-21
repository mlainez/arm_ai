//! FFT / IFFT via `rustfft`.
//!
//! Real → complex (for audio analysis, MFCC, etc.):
//!   rfft(samples: &[f32]) → Vec<f32> of length 2*(n/2 + 1)
//!     interleaved (re, im, re, im, ...).
//!
//! Complex → real (the inverse):
//!   irfft(freq: &[f32], n: usize) → Vec<f32> of length n.
//!
//! Complex → complex (matches Nx's :fft / :ifft callbacks):
//!   fft(input: &[f32]) → Vec<f32> interleaved.

use rustfft::{num_complex::Complex32, FftPlanner};

pub fn fft_complex(input_interleaved: &[f32]) -> Result<Vec<f32>, String> {
    if input_interleaved.len() % 2 != 0 {
        return Err("fft_complex: input length must be even (interleaved re/im)".into());
    }
    let n = input_interleaved.len() / 2;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex32> = input_interleaved
        .chunks_exact(2)
        .map(|c| Complex32::new(c[0], c[1]))
        .collect();

    fft.process(&mut buffer);

    let mut out = Vec::with_capacity(buffer.len() * 2);
    for c in buffer {
        out.push(c.re);
        out.push(c.im);
    }
    Ok(out)
}

pub fn ifft_complex(input_interleaved: &[f32]) -> Result<Vec<f32>, String> {
    if input_interleaved.len() % 2 != 0 {
        return Err("ifft_complex: input length must be even (interleaved re/im)".into());
    }
    let n = input_interleaved.len() / 2;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n);

    let mut buffer: Vec<Complex32> = input_interleaved
        .chunks_exact(2)
        .map(|c| Complex32::new(c[0], c[1]))
        .collect();

    ifft.process(&mut buffer);

    let scale = 1.0 / (n as f32);
    let mut out = Vec::with_capacity(buffer.len() * 2);
    for c in buffer {
        out.push(c.re * scale);
        out.push(c.im * scale);
    }
    Ok(out)
}

/// Real-input FFT — typical audio path. Returns the
/// non-redundant half (`n/2 + 1` complex values) as an
/// interleaved f32 buffer of length `2 * (n/2 + 1)`.
pub fn rfft(samples: &[f32]) -> Result<Vec<f32>, String> {
    let n = samples.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex32> = samples.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    fft.process(&mut buffer);

    // Keep first n/2 + 1 bins (the redundant conjugate half is dropped).
    let keep = n / 2 + 1;
    let mut out = Vec::with_capacity(keep * 2);
    for c in &buffer[..keep] {
        out.push(c.re);
        out.push(c.im);
    }
    Ok(out)
}

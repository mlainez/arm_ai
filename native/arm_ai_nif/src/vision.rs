//! Vision preprocessing pipeline via `image` + `fast_image_resize`.
//!
//! The canonical "JPEG/PNG/WebP on disk → ViT-ready f32 tensor"
//! pipeline:
//!   1. Decode whatever container the user has.
//!   2. Convert to RGB8.
//!   3. Resize to model input (bilinear via fast_image_resize, NEON).
//!   4. Optionally normalise (mean / std) per-channel.
//!   5. Lay out as either NHWC or NCHW f32.

use fast_image_resize as fir;
use image::{GenericImageView, ImageReader};

#[derive(Clone, Copy)]
pub enum Layout {
    Nhwc,
    Nchw,
}

pub fn decode_to_rgb8(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let img = ImageReader::open(path)
        .map_err(|e| format!("open {}: {}", path, e))?
        .with_guessed_format()
        .map_err(|e| format!("guess format: {}", e))?
        .decode()
        .map_err(|e| format!("decode: {}", e))?;

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    Ok((w, h, rgb.into_raw()))
}

pub fn resize_rgb8(
    input: &[u8],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, String> {
    let src = fir::images::Image::from_vec_u8(in_w, in_h, input.to_vec(), fir::PixelType::U8x3)
        .map_err(|e| format!("fir source: {}", e))?;
    let mut dst = fir::images::Image::new(out_w, out_h, fir::PixelType::U8x3);
    let opts = fir::ResizeOptions::new()
        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(&src, &mut dst, &opts)
        .map_err(|e| format!("fir resize: {}", e))?;
    Ok(dst.into_vec())
}

/// Normalise + layout. `mean` and `std` are length-3 (per-RGB-channel).
/// Output is `out_h * out_w * 3` f32 in the requested layout.
pub fn normalize(
    rgb8: &[u8],
    h: u32,
    w: u32,
    mean: [f32; 3],
    std: [f32; 3],
    layout: Layout,
) -> Vec<f32> {
    let h = h as usize;
    let w = w as usize;
    let mut out = vec![0.0f32; h * w * 3];

    match layout {
        Layout::Nhwc => {
            for i in 0..(h * w) {
                for c in 0..3 {
                    let v = rgb8[i * 3 + c] as f32 / 255.0;
                    out[i * 3 + c] = (v - mean[c]) / std[c];
                }
            }
        }
        Layout::Nchw => {
            let plane = h * w;
            for c in 0..3 {
                for i in 0..plane {
                    let v = rgb8[i * 3 + c] as f32 / 255.0;
                    out[c * plane + i] = (v - mean[c]) / std[c];
                }
            }
        }
    }

    out
}

/// One-shot: decode → resize → normalise → produce f32 in requested
/// layout. The canonical "image on disk → model input" pipeline.
pub fn load_for_classifier(
    path: &str,
    out_h: u32,
    out_w: u32,
    mean: [f32; 3],
    std: [f32; 3],
    layout: Layout,
) -> Result<Vec<f32>, String> {
    let (in_w, in_h, rgb) = decode_to_rgb8(path)?;
    let resized = resize_rgb8(&rgb, in_w, in_h, out_w, out_h)?;
    Ok(normalize(&resized, out_h, out_w, mean, std, layout))
}

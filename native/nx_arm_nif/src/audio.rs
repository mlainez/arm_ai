//! Audio decode + resample via `symphonia` + `rubato`.
//!
//! The canonical "audio to Whisper" pipeline:
//!   * Decode whatever container the user has (MP3 / WAV / FLAC / Opus
//!     / OGG / Vorbis / raw PCM) — symphonia covers all of these.
//!   * Mix down to mono.
//!   * Resample to 16 kHz f32 — rubato's quality-preserving sinc
//!     resampler is what whisper.cpp uses.
//!
//! Output is a flat f32 buffer of monophonic samples at the target
//! sample rate, ready to feed into Whisper (or any other 16 kHz
//! speech model).

use rubato::Resampler;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Decode an audio file to interleaved f32 samples at the source
/// sample rate.
pub fn decode_file(path: &str) -> Result<DecodedAudio, String> {
    let file = std::fs::File::open(Path::new(path))
        .map_err(|e| format!("open {}: {}", path, e))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("probe: {}", e))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("make decoder: {}", e))?;

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "missing sample_rate".to_string())?;
    let channels = codec_params
        .channels
        .ok_or_else(|| "missing channels".to_string())?
        .count() as u16;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("next_packet: {}", e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let buf_ref = decoder
            .decode(&packet)
            .map_err(|e| format!("decode: {}", e))?;

        append_buf(&buf_ref, &mut samples);
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
    })
}

fn append_buf(buf: &AudioBufferRef, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::U8(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push((b.chan(c)[f] as f32 - 128.0) / 128.0);
                }
            }
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push(b.chan(c)[f] as f32 / 32768.0);
                }
            }
        }
        AudioBufferRef::S24(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push(b.chan(c)[f].inner() as f32 / 8_388_608.0);
                }
            }
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push(b.chan(c)[f] as f32 / 2_147_483_648.0);
                }
            }
        }
        AudioBufferRef::F32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push(b.chan(c)[f]);
                }
            }
        }
        AudioBufferRef::F64(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count();
            for f in 0..frames {
                for c in 0..chans {
                    out.push(b.chan(c)[f] as f32);
                }
            }
        }
        _ => {} // skip unsupported sample formats
    }
}

/// Mix multi-channel interleaved samples down to mono by averaging.
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let c = channels as usize;
    let frames = samples.len() / c;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..c {
            sum += samples[f * c + ch];
        }
        mono.push(sum / c as f32);
    }
    mono
}

/// Resample mono f32 samples from `from_hz` to `to_hz` using
/// rubato's quality-preserving sinc resampler.
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, String> {
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }

    let ratio = to_hz as f64 / from_hz as f64;
    let chunk = 1024;

    let params = rubato::SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: rubato::SincInterpolationType::Linear,
        window: rubato::WindowFunction::BlackmanHarris2,
    };

    let mut resampler = rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| format!("SincFixedIn::new: {}", e))?;

    let mut input = vec![samples.to_vec()];
    let mut output: Vec<Vec<f32>> = vec![Vec::with_capacity(
        (samples.len() as f64 * ratio).ceil() as usize,
    )];

    let chunk_len = chunk;
    let mut pos = 0;
    while pos + chunk_len <= input[0].len() {
        let in_chunk = vec![input[0][pos..pos + chunk_len].to_vec()];
        let out_chunk = resampler
            .process(&in_chunk, None)
            .map_err(|e| format!("resample: {}", e))?;
        output[0].extend_from_slice(&out_chunk[0]);
        pos += chunk_len;
    }

    if pos < input[0].len() {
        // Pad the trailing partial chunk with zeros to satisfy the
        // fixed-in resampler, then truncate to the expected length.
        let mut tail = input[0][pos..].to_vec();
        let pad = chunk_len - tail.len();
        tail.extend(std::iter::repeat(0.0).take(pad));
        let in_chunk = vec![tail];
        let out_chunk = resampler
            .process(&in_chunk, None)
            .map_err(|e| format!("resample tail: {}", e))?;
        let useful = ((input[0].len() - pos) as f64 * ratio).round() as usize;
        output[0].extend_from_slice(&out_chunk[0][..useful.min(out_chunk[0].len())]);
    }

    input.clear();
    Ok(output.pop().unwrap())
}

/// One-shot pipeline: decode a file, mix to mono, resample to
/// `target_hz`. Returns the final f32 buffer.
pub fn decode_to_mono_at(path: &str, target_hz: u32) -> Result<Vec<f32>, String> {
    let decoded = decode_file(path)?;
    let mono = to_mono(&decoded.samples, decoded.channels);
    resample(&mono, decoded.sample_rate, target_hz)
}

/// Write mono f32 samples in `[-1.0, 1.0]` to a 16-bit PCM WAV file.
/// Small enough that pulling in `hound` would be heavier than the
/// 44-byte header we write by hand here.
pub fn write_wav_pcm16(
    path: &str,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    use std::io::Write;

    let bytes_per_sample = 2;
    let n_samples = samples.len();
    let data_bytes = (n_samples * bytes_per_sample) as u32;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;

    let mut file = std::fs::File::create(path).map_err(|e| format!("create {}: {}", path, e))?;

    let header_writes: Result<(), std::io::Error> = (|| {
        file.write_all(b"RIFF")?;
        file.write_all(&(36 + data_bytes).to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // fmt chunk size
        file.write_all(&1u16.to_le_bytes())?; // PCM format
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&(channels * bytes_per_sample as u16).to_le_bytes())?;
        file.write_all(&16u16.to_le_bytes())?; // bits per sample
        file.write_all(b"data")?;
        file.write_all(&data_bytes.to_le_bytes())?;
        Ok(())
    })();
    header_writes.map_err(|e| format!("wav header: {}", e))?;

    // Convert f32 [-1.0, 1.0] → i16. Clip on overflow.
    let mut buf = Vec::with_capacity(n_samples * 2);
    for &s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let v = (clipped * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&buf)
        .map_err(|e| format!("wav data: {}", e))?;

    Ok(())
}

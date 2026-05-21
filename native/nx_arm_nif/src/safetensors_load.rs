//! SafeTensors loader bridge.
//!
//! Returns the list of tensors in a file as
//! `[(name, shape, dtype_atom, f32_bin)]` so the Elixir side can
//! turn them into Nx tensors on NxArm.Backend.
//!
//! All tensors are cast to f32 at the boundary — keeps the API
//! tiny. If a caller needs the original dtype they can pass
//! `:keep_dtype` (future work).

use safetensors::{Dtype, SafeTensors};

pub struct LoadedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_f32: Vec<f32>,
    pub original_dtype: &'static str,
}

pub fn load_all(path: &str) -> Result<Vec<LoadedTensor>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("SafeTensors::deserialize: {}", e))?;

    let mut out = Vec::with_capacity(st.tensors().len());

    for (name, tv) in st.tensors() {
        let shape = tv.shape().to_vec();
        let dtype = tv.dtype();
        let raw = tv.data();
        let n_elems: usize = shape.iter().product();

        let (data_f32, dtype_str) = match dtype {
            Dtype::F32 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    v[i] = f32::from_le_bytes([
                        raw[i * 4],
                        raw[i * 4 + 1],
                        raw[i * 4 + 2],
                        raw[i * 4 + 3],
                    ]);
                }
                (v, "f32")
            }
            Dtype::F16 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                    v[i] = half::f16::from_bits(bits).to_f32();
                }
                (v, "f16")
            }
            Dtype::BF16 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                    v[i] = half::bf16::from_bits(bits).to_f32();
                }
                (v, "bf16")
            }
            Dtype::F64 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    let val = f64::from_le_bytes([
                        raw[i * 8],
                        raw[i * 8 + 1],
                        raw[i * 8 + 2],
                        raw[i * 8 + 3],
                        raw[i * 8 + 4],
                        raw[i * 8 + 5],
                        raw[i * 8 + 6],
                        raw[i * 8 + 7],
                    ]);
                    v[i] = val as f32;
                }
                (v, "f64")
            }
            Dtype::I32 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    let val = i32::from_le_bytes([
                        raw[i * 4],
                        raw[i * 4 + 1],
                        raw[i * 4 + 2],
                        raw[i * 4 + 3],
                    ]);
                    v[i] = val as f32;
                }
                (v, "i32")
            }
            Dtype::I64 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    let val = i64::from_le_bytes([
                        raw[i * 8],
                        raw[i * 8 + 1],
                        raw[i * 8 + 2],
                        raw[i * 8 + 3],
                        raw[i * 8 + 4],
                        raw[i * 8 + 5],
                        raw[i * 8 + 6],
                        raw[i * 8 + 7],
                    ]);
                    v[i] = val as f32;
                }
                (v, "i64")
            }
            Dtype::U8 => {
                let mut v = vec![0f32; n_elems];
                for i in 0..n_elems {
                    v[i] = raw[i] as f32;
                }
                (v, "u8")
            }
            other => {
                return Err(format!(
                    "safetensors: unsupported dtype {:?} for tensor '{}'",
                    other, name
                ))
            }
        };

        out.push(LoadedTensor {
            name: name.to_string(),
            shape,
            data_f32,
            original_dtype: dtype_str,
        });
    }

    Ok(out)
}

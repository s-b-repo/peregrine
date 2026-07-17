//! Safetensors writing + weight quantization — the inverse of [`crate::safetensors`]
//! and [`crate::qt`]. Used by tools and tests to emit model directories in the
//! int4/int8 container format the engine reads (a numpy-free synthetic model
//! generator), without pulling in torch.

use crate::dtype::bf16_to_f32;
use std::path::Path;

/// One tensor to embed: name, safetensors dtype string, shape, raw LE bytes.
pub struct Blob {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl Blob {
    pub fn new(name: impl Into<String>, dtype: &str, shape: Vec<i64>, bytes: Vec<u8>) -> Blob {
        Blob { name: name.into(), dtype: dtype.into(), shape, bytes }
    }
}

/// Write a single-shard `model.safetensors` into `dir` (created if needed).
pub fn write_safetensors(dir: &Path, blobs: &[Blob]) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut cursor: i64 = 0;
    let mut data: Vec<u8> = Vec::new();
    for b in blobs {
        let start = cursor;
        let end = start + b.bytes.len() as i64;
        header.insert(
            b.name.clone(),
            serde_json::json!({"dtype": b.dtype, "shape": b.shape, "data_offsets": [start, end]}),
        );
        data.extend_from_slice(&b.bytes);
        cursor = end;
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut out = Vec::with_capacity(8 + hdr.len() + data.len());
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&data);
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("model.safetensors"), out)
}

pub fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

/// Round-trip check helper: bf16 encode then decode.
pub fn bf16_roundtrip(v: f32) -> f32 {
    bf16_to_f32((v.to_bits() >> 16) as u16)
}

/// Quantize a weight `[O, I]` to per-row int8: returns `(bytes[O*I], scale[O])`.
pub fn quant_i8(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = vec![0u8; o * i];
    let mut sc = vec![0f32; o];
    for oo in 0..o {
        let row = &w[oo * i..oo * i + i];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / 127.0).max(1e-12);
        sc[oo] = s;
        for ii in 0..i {
            q[oo * i + ii] = ((row[ii] / s).round_ties_even() as i32 as i8) as u8;
        }
    }
    (q, sc)
}

/// Quantize a weight `[O, I]` to per-row packed int4: returns
/// `(bytes[O*ceil(I/2)], scale[O])`. Nibbles are biased by +8 into `[0,15]`.
pub fn quant_i4(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
    let rb = i.div_ceil(2);
    let mut q = vec![0u8; o * rb];
    let mut sc = vec![0f32; o];
    for oo in 0..o {
        let row = &w[oo * i..oo * i + i];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / 7.0).max(1e-12);
        sc[oo] = s;
        for ii in 0..i {
            let v = (row[ii] / s).round_ties_even().clamp(-8.0, 7.0) as i32;
            let bias = (v + 8) as u8 & 0x0F;
            if ii & 1 == 0 {
                q[oo * rb + (ii >> 1)] |= bias;
            } else {
                q[oo * rb + (ii >> 1)] |= bias << 4;
            }
        }
    }
    (q, sc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QtFmt, QtInfo, SafeTensors};

    #[test]
    fn written_model_reads_back() {
        let dir = std::env::temp_dir().join(format!("coli_pack_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (o, i) = (3usize, 8usize);
        let w: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.1) - 1.0).collect();
        let (q4, s4) = quant_i4(&w, o, i);
        write_safetensors(
            &dir,
            &[
                Blob::new("w", "U8", vec![o as i64, (i / 2) as i64], q4),
                Blob::new("w.qs", "F32", vec![o as i64], f32_bytes(&s4)),
                Blob::new("norm", "F32", vec![4], f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();
        let st = SafeTensors::open(&dir).unwrap();
        assert_eq!(QtInfo::detect(&st, "w", o as i64, i as i64).fmt, QtFmt::Int4);
        let mut n = [0f32; 4];
        st.read_f32("norm", &mut n).unwrap();
        assert_eq!(n, [1.0, 2.0, 3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

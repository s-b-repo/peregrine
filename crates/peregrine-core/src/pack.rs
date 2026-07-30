//! Safetensors writing + weight quantization — the inverse of [`crate::safetensors`]
//! and [`crate::qt`]. Used by tools and tests to emit model directories in the
//! int4/int8 container format the engine reads (a numpy-free synthetic model
//! generator), without pulling in torch.

use crate::compress::{encode, Compression};
use crate::dtype::bf16_to_f32;
use std::path::Path;

/// One tensor to embed: name, safetensors dtype string, shape, raw LE bytes.
pub struct Blob {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
    /// Optional per-blob compression scheme. When set, the payload is
    /// compressed before being embedded and the tensor entry gains a
    /// `"compression": "zstd"` field plus an `"uncompressed_nbytes"` field.
    /// [`Compression::None`] emits the historical (raw) format so a fresh
    /// writer stays byte-compatible with the reader that predates compression.
    pub compression: Compression,
    /// Optional on-disk byte-layout tag. `None` = the kernels' native row-major
    /// layout (historical). `Some(("kblock", gs))` = the group-block transposed
    /// layout ([`to_kblock`]) with group size `gs`; the loader auto-converts
    /// back to native at read time ("tensor layout auto-conversion").
    pub layout: Option<(&'static str, usize)>,
}

impl Blob {
    pub fn new(name: impl Into<String>, dtype: &str, shape: Vec<i64>, bytes: Vec<u8>) -> Blob {
        Blob { name: name.into(), dtype: dtype.into(), shape, bytes, compression: Compression::None, layout: None }
    }

    /// Set the compression scheme for this blob.
    pub fn with_compression(mut self, c: Compression) -> Blob {
        self.compression = c;
        self
    }

    /// Re-tile this blob's bytes into the K-block layout (see [`to_kblock`])
    /// and tag the header accordingly. `o` = rows, `gs_bytes` = bytes per
    /// group-block per row; `bytes.len()` must be `o * n_groups * gs_bytes`.
    pub fn with_kblock_layout(mut self, o: usize, gs_bytes: usize) -> Blob {
        if let Some(t) = to_kblock(&self.bytes, o, gs_bytes) {
            self.bytes = t;
            self.layout = Some(("kblock", gs_bytes));
        }
        self
    }
}

/// Transpose row-major group blocks into group-major order: native layout is
/// `[o][g]` blocks of `gs_bytes` (each row's groups contiguous); K-block is
/// `[g][o]` (each *group column* contiguous across all rows) — sequential for
/// per-group streaming access patterns. Pure byte permutation; its own inverse
/// is [`from_kblock`]. `None` when the length doesn't tile.
pub fn to_kblock(bytes: &[u8], o: usize, gs_bytes: usize) -> Option<Vec<u8>> {
    if o == 0 || gs_bytes == 0 || !bytes.len().is_multiple_of(o * gs_bytes) {
        return None;
    }
    let n_groups = bytes.len() / (o * gs_bytes);
    let mut out = vec![0u8; bytes.len()];
    for row in 0..o {
        for g in 0..n_groups {
            let src = (row * n_groups + g) * gs_bytes;
            let dst = (g * o + row) * gs_bytes;
            out[dst..dst + gs_bytes].copy_from_slice(&bytes[src..src + gs_bytes]);
        }
    }
    Some(out)
}

/// Inverse of [`to_kblock`]: group-major blocks back to the kernels' native
/// row-major layout.
pub fn from_kblock(bytes: &[u8], o: usize, gs_bytes: usize) -> Option<Vec<u8>> {
    if o == 0 || gs_bytes == 0 || !bytes.len().is_multiple_of(o * gs_bytes) {
        return None;
    }
    let n_groups = bytes.len() / (o * gs_bytes);
    let mut out = vec![0u8; bytes.len()];
    for g in 0..n_groups {
        for row in 0..o {
            let src = (g * o + row) * gs_bytes;
            let dst = (row * n_groups + g) * gs_bytes;
            out[dst..dst + gs_bytes].copy_from_slice(&bytes[src..src + gs_bytes]);
        }
    }
    Some(out)
}

/// Write a single-shard `model.safetensors` into `dir` (created if needed).
/// Blobs with [`Compression::Zstd`] have their payload zstd-encoded (level 3);
/// the reader detects and decompresses via the `"compression"` header field.
pub fn write_safetensors(dir: &Path, blobs: &[Blob]) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut cursor: i64 = 0;
    let mut data: Vec<u8> = Vec::new();
    for b in blobs {
        // Compress the payload once. The tensor entry's `data_offsets` describe
        // the *on-disk* bytes (compressed or raw); the reader also learns the
        // original size from `uncompressed_nbytes` so it can preallocate cleanly.
        let payload = encode(&b.bytes, b.compression, 3).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("compress '{}': {e}", b.name))
        })?;
        let start = cursor;
        let end = start + payload.len() as i64;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String(b.dtype.clone()));
        entry.insert("shape".into(), serde_json::json!(b.shape));
        entry.insert("data_offsets".into(), serde_json::json!([start, end]));
        if let Some(tag) = b.compression.tag() {
            entry.insert("compression".into(), serde_json::Value::String(tag.to_string()));
            entry.insert("uncompressed_nbytes".into(), serde_json::json!(b.bytes.len()));
        }
        if let Some((tag, gs_bytes)) = b.layout {
            entry.insert("layout".into(), serde_json::Value::String(tag.to_string()));
            entry.insert("layout_gs_bytes".into(), serde_json::json!(gs_bytes));
        }
        header.insert(b.name.clone(), serde_json::Value::Object(entry));
        data.extend_from_slice(&payload);
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

/// Quantize a weight `[O, I]` to grouped packed int4 (colibrì fmt 4): one scale
/// per `gs`-element group along the input dim. Returns `(bytes[O*ceil(I/2)],
/// scale[O*ceil(I/gs)])` with scales laid out `sc[o*ng + g]` — the layout
/// `convert_fp8_to_int4.py --group-size gs` emits and [`crate::qt`] detects.
pub fn quant_i4_grouped(w: &[f32], o: usize, i: usize, gs: usize) -> (Vec<u8>, Vec<f32>) {
    let rb = i.div_ceil(2);
    let ng = i.div_ceil(gs);
    let mut q = vec![0u8; o * rb];
    let mut sc = vec![0f32; o * ng];
    for oo in 0..o {
        let row = &w[oo * i..oo * i + i];
        for g in 0..ng {
            let (s, e) = (g * gs, ((g + 1) * gs).min(i));
            let amax = row[s..e].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = (amax / 7.0).max(1e-12);
            sc[oo * ng + g] = scale;
            for ii in s..e {
                let v = (row[ii] / scale).round_ties_even().clamp(-8.0, 7.0) as i32;
                let bias = (v + 8) as u8 & 0x0F;
                if ii & 1 == 0 {
                    q[oo * rb + (ii >> 1)] |= bias;
                } else {
                    q[oo * rb + (ii >> 1)] |= bias << 4;
                }
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
    fn written_model_reads_back() -> Result<(), crate::Error> {
        let dir = std::env::temp_dir().join(format!("coli_pack_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
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
        )?;
        let st = SafeTensors::open(&dir)?;
        assert_eq!(QtInfo::detect(&st, "w", o as i64, i as i64).fmt, QtFmt::Int4);
        let mut n = [0f32; 4];
        st.read_f32("norm", &mut n)?;
        assert_eq!(n, [1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn kblock_layout_round_trips_to_native_bytes() -> Result<(), crate::Error> {
        // A kblock-tagged tensor must read back byte-identical to its native
        // form: the writer permutes, the reader inverts — auto-conversion.
        let dir = std::env::temp_dir().join(format!("coli_pack_kblock_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        // native payload: 4 rows × 3 groups × 8 bytes, recognizable pattern
        let (o, n_groups, gsb) = (4usize, 3usize, 8usize);
        let native: Vec<u8> = (0..o * n_groups * gsb).map(|k| (k % 251) as u8).collect();
        write_safetensors(
            &dir,
            &[
                Blob::new("w", "U8", vec![o as i64, (n_groups * gsb) as i64], native.clone())
                    .with_kblock_layout(o, gsb),
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        let mut got = vec![0u8; native.len()];
        st.read_raw("w", &mut got)?;
        assert_eq!(got, native, "reader must undo the kblock permutation");
        // and pure permutation round-trip
        let t = to_kblock(&native, o, gsb).ok_or(crate::Error::Format("tile".into()))?;
        assert_ne!(t, native, "kblock actually permutes");
        let back = from_kblock(&t, o, gsb).ok_or(crate::Error::Format("untile".into()))?;
        assert_eq!(back, native);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn compressed_written_model_reads_back() -> Result<(), crate::Error> {
        // Round-trip a compressed tensor: the reader must decompress and yield
        // byte-identical values to what the writer supplied. Correctness-neutral
        // is the whole point of the compression feature.
        let dir = std::env::temp_dir().join(format!("coli_pack_zstd_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        // A larger tensor so compression actually engages a real zstd frame.
        let vals: Vec<f32> = (0..1024).map(|k| ((k as f32 * 0.13) % 5.0) - 2.5).collect();
        write_safetensors(
            &dir,
            &[
                Blob::new("weights", "F32", vec![vals.len() as i64], f32_bytes(&vals))
                    .with_compression(Compression::Zstd),
                // Uncompressed norm to confirm mixed shards work.
                Blob::new("norm", "F32", vec![4], f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        assert!(st.has_compressed_tensors());
        assert_eq!(st.compression("weights"), Compression::Zstd);
        assert_eq!(st.compression("norm"), Compression::None);
        // read_f32 decompresses transparently.
        let mut got = vec![0f32; vals.len()];
        st.read_f32("weights", &mut got)?;
        assert_eq!(got, vals);
        // The uncompressed sibling still reads back correctly.
        let mut n = [0f32; 4];
        st.read_f32("norm", &mut n)?;
        assert_eq!(n, [1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

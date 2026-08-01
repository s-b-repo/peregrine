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

/// Quantize a weight `[O, I]` to per-row packed int2 (colibrì fmt 3): returns
/// `(bytes[O*ceil(I/4)], scale[O])`. Four 2-bit fields per byte, low field first;
/// each is biased by `+2` into `[0,3]`.
///
/// **This is the format's first producer.** int2 has been fully *consumable*
/// since M1 — the container detector, the scalar and AVX2 dot kernels, the
/// dequantizer and the CUDA `row_bytes`/`weight_at` decode all handle fmt 3 —
/// but nothing could write it, so no checkpoint ever used it. At GLM-5.2 shapes
/// it halves the payload that dominates a disk-bound run: 18.92 → 9.48 MB per
/// expert, 11.35 → 5.69 GB per token.
///
/// ## The scale convention (defined here, because none existed)
///
/// The decoders read a field as `field - 2`, so the representable levels are
/// `{-2, -1, 0, +1}` — **asymmetric**, unlike int8's `[-128, 127]` or int4's
/// `[-8, 7]` where the positive max is what the scale divides by. Following the
/// same rule (`scale = amax / max_positive_level`) gives `amax / 1`, so the
/// largest positive weight maps exactly to `+1` and nothing clips.
///
/// The trade that buys: because `|v| <= amax` by construction, `v / s` never
/// reaches `-2`, so this convention uses three of the four levels and is
/// effectively ternary. Reaching `-2` requires a scale below `amax`, which clips
/// the positive extreme instead — a straight swap of one distortion for another,
/// and which wins is an accuracy question that needs a real checkpoint to answer.
/// No-clipping is the safer default to establish; revisit it with
/// `Model::prediction_flip_rate` and a real model, not by intuition.
///
/// Lossy by construction — 2 bits cannot round-trip. Gate a checkpoint built with
/// this on `Model::prediction_flip_rate`, not on a bit-identity test.
pub fn quant_i2(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
    let rb = i.div_ceil(4);
    let mut q = vec![0u8; o * rb];
    let mut sc = vec![0f32; o];
    for oo in 0..o {
        let row = &w[oo * i..oo * i + i];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // `/ 1.0` is the max *positive* level, matching int4's `/ 7.0` and int8's
        // `/ 127.0`. Written out rather than elided so the convention is legible.
        let s = (amax / 1.0).max(1e-12);
        sc[oo] = s;
        for ii in 0..i {
            let v = (row[ii] / s).round_ties_even().clamp(-2.0, 1.0) as i32;
            let field = (v + 2) as u8 & 0x03;
            q[oo * rb + (ii >> 2)] |= field << ((ii & 3) * 2);
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

    /// Decode one packed int2 field exactly as every consumer does
    /// (`idot.rs::dot_i2i8_scalar`, `weight.rs`, `backend_cuda.cu::weight_at`):
    /// the `2·(i&3)`-shifted 2-bit field of byte `i>>2`, biased by −2.
    fn decode_i2(q: &[u8], row: usize, rb: usize, i: usize) -> i32 {
        let byte = q[row * rb + (i >> 2)];
        (((byte >> (2 * (i & 3))) & 0x03) as i32) - 2
    }

    #[test]
    fn quant_i2_packs_four_fields_per_byte_in_decoder_order() {
        // The producer must agree with the decoders that already shipped. One row
        // spanning both level extremes and the middle.
        let (o, i) = (1usize, 8usize);
        // amax = 4.0 → s = 4.0, so each value lands at round_ties_even(v/4).
        let w = vec![4.0f32, -4.0, 0.0, 2.0, -4.0, 1.0, -2.0, 3.9];
        let (q, sc) = quant_i2(&w, o, i);
        assert_eq!(q.len(), o * i.div_ceil(4), "ceil(I/4) bytes per row");
        assert!((sc[0] - 4.0).abs() < 1e-6, "scale is amax / max-positive-level");
        let rb = i.div_ceil(4);
        let got: Vec<i32> = (0..i).map(|k| decode_i2(&q, 0, rb, k)).collect();
        // 1, -1, 0, 0.5→0 (ties-even), -1, 0.25→0, -0.5→0 (ties-even), 0.975→1.
        assert_eq!(got, vec![1, -1, 0, 0, -1, 0, 0, 1]);
    }

    #[test]
    fn quant_i2_clamps_to_the_asymmetric_level_range() {
        // Levels are {-2,-1,0,+1}: positives cannot exceed +1, negatives reach -2.
        // A row whose extreme is negative must not wrap or overflow the field.
        let (o, i) = (1usize, 4usize);
        let w = vec![-10.0f32, 10.0, -10.0, 10.0];
        let (q, _) = quant_i2(&w, o, i);
        let rb = i.div_ceil(4);
        for k in 0..i {
            let v = decode_i2(&q, 0, rb, k);
            assert!((-2..=1).contains(&v), "field {k} decoded to {v}, outside [-2,1]");
        }
    }

    #[test]
    fn quant_i2_halves_int4_and_matches_the_container_detector() -> Result<(), crate::Error> {
        // Byte count must be exactly half int4's, and `QtInfo::detect` must
        // classify the payload as fmt 3 from its size alone — that inference is
        // the only thing that tells the loader what it is reading.
        let (o, i) = (4usize, 64usize);
        let w: Vec<f32> = (0..o * i).map(|k| ((k % 17) as f32 - 8.0) * 0.1).collect();
        let (q2, sc2) = quant_i2(&w, o, i);
        let (q4, _) = quant_i4(&w, o, i);
        assert_eq!(q2.len() * 2, q4.len(), "int2 is exactly half of int4");

        let dir = std::env::temp_dir().join(format!("coli_i2_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        write_safetensors(
            &dir,
            &[
                Blob::new("w2", "U8", vec![o as i64, (i / 4) as i64], q2),
                Blob::new("w2.qs", "F32", vec![o as i64], f32_bytes(&sc2)),
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        assert_eq!(QtInfo::detect(&st, "w2", o as i64, i as i64).fmt, QtFmt::Int2, "byte count must infer fmt 3");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

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

//! Safetensors writing + weight quantization — the inverse of [`crate::safetensors`]
//! and [`crate::qt`]. Used by tools and tests to emit model directories in the
//! int4/int8 container format the engine reads (a numpy-free synthetic model
//! generator), without pulling in torch.

use crate::compress::{encode, Compression};
use crate::dtype::bf16_to_f32;
use crate::qt::QtFmt;
use std::path::Path;

/// Values per int3-g64 group.
pub const I3_GROUP: usize = 64;
/// Low plane bytes per group (2 bits per value, packed like int2).
pub const I3_LOW_BYTES: usize = 16;
/// Total bytes per int3-g64 group: 16-byte low plane + 8-byte high plane.
pub const I3_GROUP_BYTES: usize = 24;

/// Values per int2-g64 group. Matches [`I3_GROUP`] deliberately: the two
/// formats share a group geometry, and int2-g64's payload *is* int3-g64's low
/// plane, so a reader that understands one understands the other's packing.
pub const I2G_GROUP: usize = 64;
/// Bytes per int2-g64 group: 64 values × 2 bits. Identical to [`I3_LOW_BYTES`].
pub const I2G_GROUP_BYTES: usize = 16;
/// f32 per int2-g64 group in the `.qs` sibling: **scale then zero-point,
/// interleaved** as `[s0, z0, s1, z1, …]`.
///
/// Two f32 rather than a separate `.qz` tensor, and that is a deliberate trade.
/// A third sibling per weight would take the streamed expert read from 6 regions
/// to 9 — `prefetch_hint_item` returns a fixed `[(RawFd, u64, usize); 6]`, and
/// widening it ripples through the prefetch lane, the batched submit and
/// `rebuild`. Interleaving keeps every one of those untouched, and it makes the
/// container *less* ambiguous rather than more: `2·o·ng` scales is a cardinality
/// no other format produces, which is exactly what [`crate::qt::QtInfo::detect`]
/// needs to tell a grouped 2-bit tensor from a per-row one whose byte count is
/// identical.
pub const I2G_SCALES_PER_GROUP: usize = 2;

/// Decode value `k` of an int2-g64 group: a 2-bit field, four per byte, in the
/// same bit order as per-row int2. Stored **unsigned** `[0, 3]` — unlike every
/// other format here the bias is not a constant, it is the per-group zero-point.
#[inline]
pub fn i2g_field(byte: u8, k: usize) -> i32 {
    ((byte >> (2 * (k & 3))) & 0x03) as i32
}

/// Quantize `[o, i]` to **affine** int2 with a scale *and a zero-point* per
/// 64-value group. Returns `(bytes[o·ng·16], scales[2·o·ng])` interleaved
/// `[s, z]` per group.
///
/// **Why this format exists.** peregrine's per-row [`quant_i2`] has two
/// independent defects against the 2-bit recipe that is actually evidenced for
/// GLM-5.2-class checkpoints. It scales per *row*, where the reference groups
/// by 64; and its symmetric convention `s = amax / 1` with a `[-2, 1]` clamp
/// makes the `-2` level **unreachable** — hitting it needs `|w| ≥ 1.5·amax`,
/// impossible when `amax` is the row's own maximum. One of four levels is dead
/// in every row it can write, so the format is effectively ternary.
///
/// An affine mapping fixes both by construction: `[min, max]` of each group maps
/// onto `{0,1,2,3}`, so all four levels carry weight and nothing clips. The cost
/// is one extra f32 per group (≈0.06 bits/weight at g64) and one subtraction per
/// group in the dot product.
///
/// A constant group (`max == min`) gets `s = 0`, `z = min`, which dequantizes
/// every element back to exactly `min` — the degenerate case has to round-trip,
/// not divide by zero.
pub fn quant_i2_g64(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
    let ng = i.div_ceil(I2G_GROUP);
    let mut q = vec![0u8; o * ng * I2G_GROUP_BYTES];
    let mut scale = vec![0f32; o * ng * I2G_SCALES_PER_GROUP];
    for row in 0..o {
        for g in 0..ng {
            let lo_i = g * I2G_GROUP;
            let hi_i = ((g + 1) * I2G_GROUP).min(i);
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for k in lo_i..hi_i {
                let v = w[row * i + k];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            // A ragged final group has no elements at all when i is a multiple
            // of the group size; guard the empty range rather than propagating
            // the infinities into the scale.
            if !lo.is_finite() || !hi.is_finite() {
                lo = 0.0;
                hi = 0.0;
            }
            let s = (hi - lo) / 3.0;
            let (s, z) = if s > 0.0 { (s, lo) } else { (0.0, lo) };
            let si = (row * ng + g) * I2G_SCALES_PER_GROUP;
            scale[si] = s;
            scale[si + 1] = z;
            let base = (row * ng + g) * I2G_GROUP_BYTES;
            for k in lo_i..hi_i {
                // Ties to even, matching `quant_i3_g64` and colibrì's `np.rint`.
                let u = if s > 0.0 { (((w[row * i + k] - z) / s).round_ties_even()).clamp(0.0, 3.0) as u8 } else { 0 };
                q[base + ((k - lo_i) >> 2)] |= (u & 0x03) << (2 * ((k - lo_i) & 3));
            }
            // Padding past `i` stays at field 0, which dequantizes to `z` — the
            // group's own minimum rather than a spurious extreme.
        }
    }
    (q, scale)
}

/// Decode value `k` of an int3-g64 group from its two planes. The low plane
/// holds bits 0-1 in the int2 layout; the high plane holds bit 2, one bit per
/// value, eight values per byte. Stored biased `+4`, so the result is `[-4, 3]`.
#[inline]
pub fn i3_value(lo_byte: u8, hi_byte: u8, k: usize) -> i32 {
    let low = ((lo_byte >> (2 * (k & 3))) & 0x03) as i32;
    let high = ((hi_byte >> (k & 7)) & 0x01) as i32;
    (low | (high << 2)) - 4
}

/// Quantize `[o, i]` to int3 with one scale per 64-value group — colibrì's
/// fmt 5, byte-for-byte. Positive extreme is exact (`amax / 3`); the `-4` level
/// exists but only rounding reaches it.
pub fn quant_i3_g64(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
    let ng = i.div_ceil(I3_GROUP);
    let mut q = vec![0u8; o * ng * I3_GROUP_BYTES];
    let mut scale = vec![0f32; o * ng];
    for row in 0..o {
        for g in 0..ng {
            let lo_i = g * I3_GROUP;
            let hi_i = ((g + 1) * I3_GROUP).min(i);
            let mut amax = 0f32;
            for k in lo_i..hi_i {
                amax = amax.max(w[row * i + k].abs());
            }
            let s = (amax / 3.0).max(1e-8);
            scale[row * ng + g] = s;
            let base = (row * ng + g) * I3_GROUP_BYTES;
            for k in lo_i..hi_i {
                // Padding past `i` stays at the biased zero (4), matching the
                // reference converter, so a partial final group decodes to 0.
                // Ties to even, not away from zero: the reference encoder uses
                // `np.rint`, whose own comment in that file reads "np.rint =
                // lrintf" — C's default rounding mode. `f32::round()` disagrees
                // on exact .5 values, which showed up as 3 differing bytes in 480
                // when this was diffed against colibrì's encoder. Byte-identity
                // is the whole point of supporting this format.
                let v = (w[row * i + k] / s).round_ties_even().clamp(-4.0, 3.0) as i32;
                let u = (v + 4) as u8;
                q[base + ((k - lo_i) >> 2)] |= (u & 0x03) << (2 * ((k - lo_i) & 3));
                q[base + I3_LOW_BYTES + ((k - lo_i) >> 3)] |= ((u >> 2) & 0x01) << ((k - lo_i) & 7);
            }
            // A partial final group pads with biased 4, i.e. value 0: low bits
            // already zero, high bit set. (Writing the low plane instead would
            // encode -3 and quietly corrupt the tail of every ragged row.)
            for k in hi_i..(g + 1) * I3_GROUP {
                let kk = k - lo_i;
                q[base + I3_LOW_BYTES + (kk >> 3)] |= 1 << (kk & 7);
            }
        }
    }
    (q, scale)
}

/// A borrowed view of one quantized weight's on-disk bytes — enough to read rows
/// back out without the model stack.
///
/// The offline tools requantize a checkpoint (int4 → int2, or a heat-tiered mix)
/// and so must decode what a container already holds. They depend on this crate
/// alone, deliberately: a multi-hour batch job has no business pulling in
/// io_uring, the scheduler, or CUDA. `peregrine-model`'s `QtWeight` remains the
/// engine's hot path; this is the same arithmetic for offline use, pinned to it
/// by `core_dequant_matches_qtweight` in that crate.
pub struct QtView<'a> {
    pub fmt: QtFmt,
    /// rows (output dim)
    pub o: usize,
    /// columns (input dim)
    pub i: usize,
    /// group size for [`QtFmt::Int4Grouped`], else 0
    pub gs: usize,
    pub q: &'a [u8],
    pub scale: &'a [f32],
}

impl QtView<'_> {
    /// Packed bytes per row for this format, or `None` if the format has no
    /// row-major packed form (`F32`/`Unknown`).
    pub fn row_bytes(fmt: QtFmt, i: usize) -> Option<usize> {
        match fmt {
            QtFmt::Int8 => Some(i),
            QtFmt::Int4 | QtFmt::Int4Grouped => Some(i.div_ceil(2)),
            QtFmt::Int2 => Some(i.div_ceil(4)),
            QtFmt::Int2G64 => Some(i.div_ceil(I2G_GROUP) * I2G_GROUP_BYTES),
            QtFmt::Int3G64 => Some(i.div_ceil(I3_GROUP) * I3_GROUP_BYTES),
            QtFmt::F32 | QtFmt::Unknown => None,
        }
    }

    /// Dequantize row `row` into `out` (which must hold at least `i` floats; a
    /// shorter buffer is left untouched, matching `QtWeight::dequant_row_into`).
    /// Out-of-range indices produce zeros rather than a panic — an offline tool
    /// reading a malformed container should report, not abort.
    pub fn dequant_row_into(&self, row: usize, out: &mut [f32]) {
        if out.len() < self.i {
            return;
        }
        let out = &mut out[..self.i];
        let Some(rb) = Self::row_bytes(self.fmt, self.i) else {
            out.fill(0.0);
            return;
        };
        let base = row * rb;
        match self.fmt {
            QtFmt::Int8 => {
                let s = self.scale.get(row).copied().unwrap_or(0.0);
                for (i, dst) in out.iter_mut().enumerate() {
                    let v = self.q.get(base + i).copied().unwrap_or(0) as i8;
                    *dst = v as f32 * s;
                }
            }
            QtFmt::Int4 => {
                let s = self.scale.get(row).copied().unwrap_or(0.0);
                for (i, dst) in out.iter_mut().enumerate() {
                    let byte = self.q.get(base + (i >> 1)).copied().unwrap_or(0);
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    *dst = (nib - 8) as f32 * s;
                }
            }
            QtFmt::Int4Grouped => {
                let ng = if self.gs > 0 { self.i.div_ceil(self.gs) } else { 1 };
                for (i, dst) in out.iter_mut().enumerate() {
                    let g = i.checked_div(self.gs).unwrap_or(0);
                    let s = self.scale.get(row * ng + g).copied().unwrap_or(0.0);
                    let byte = self.q.get(base + (i >> 1)).copied().unwrap_or(0);
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    *dst = (nib - 8) as f32 * s;
                }
            }
            QtFmt::Int2 => {
                let s = self.scale.get(row).copied().unwrap_or(0.0);
                for (i, dst) in out.iter_mut().enumerate() {
                    let byte = self.q.get(base + (i >> 2)).copied().unwrap_or(0);
                    let field = ((byte >> (2 * (i & 3))) & 0x03) as i32;
                    *dst = (field - 2) as f32 * s;
                }
            }
            QtFmt::Int2G64 => {
                // Affine: the stored field is unsigned and the bias is the
                // group's own zero-point, so this is `q * s + z`, not
                // `(q - bias) * s` like every other format here.
                let ng = self.i.div_ceil(I2G_GROUP);
                for (i, dst) in out.iter_mut().enumerate() {
                    let g = i / I2G_GROUP;
                    let k = i % I2G_GROUP;
                    let si = (row * ng + g) * I2G_SCALES_PER_GROUP;
                    let s = self.scale.get(si).copied().unwrap_or(0.0);
                    let z = self.scale.get(si + 1).copied().unwrap_or(0.0);
                    let byte = self.q.get(base + g * I2G_GROUP_BYTES + (k >> 2)).copied().unwrap_or(0);
                    *dst = i2g_field(byte, k) as f32 * s + z;
                }
            }
            QtFmt::Int3G64 => {
                let ng = self.i.div_ceil(I3_GROUP);
                for (i, dst) in out.iter_mut().enumerate() {
                    let g = i / I3_GROUP;
                    let k = i % I3_GROUP;
                    let s = self.scale.get(row * ng + g).copied().unwrap_or(0.0);
                    let gb = base + g * I3_GROUP_BYTES;
                    let lo = self.q.get(gb + (k >> 2)).copied().unwrap_or(0);
                    let hi = self.q.get(gb + I3_LOW_BYTES + (k >> 3)).copied().unwrap_or(0);
                    *dst = i3_value(lo, hi, k) as f32 * s;
                }
            }
            QtFmt::F32 | QtFmt::Unknown => out.fill(0.0),
        }
    }
}

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
    #[test]
    fn int3_g64_bytes_match_colibri_byte_for_byte() {
        // Cross-implementation compatibility, frozen. This vector was produced by
        // colibrì's own `tools/convert_fp8_to_int4.py::quant_int3_g64` and is the
        // reason peregrine can read a container that engine wrote (and vice
        // versa) rather than merely one that looks similar.
        //
        // It caught a real defect: `f32::round()` rounds halves away from zero
        // where `np.rint` rounds to even, which differed in 3 of 480 bytes on the
        // first attempt. A round-trip test cannot see that — both encoders
        // decode their own output correctly.
        const PAYLOAD: [u8; 96] = [
            201, 201, 218, 218, 30, 30, 39, 103, 107, 123, 120, 152, 156, 172, 173, 225, 102, 102,
            238, 204, 204, 157, 153, 185, 225, 114, 114, 182, 182, 135, 135, 201, 201, 218, 26, 30,
            46, 39, 103, 107, 59, 51, 115, 103, 102, 238, 206, 204, 123, 120, 156, 156, 173, 173,
            225, 225, 114, 178, 182, 182, 135, 199, 201, 217, 220, 153, 153, 187, 51, 51, 119, 102,
            218, 30, 30, 38, 39, 107, 107, 120, 120, 156, 156, 173, 237, 225, 97, 114, 230, 206,
            204, 220, 157, 153, 185, 51,
        ];
        // f32 bit patterns of the reference scales, so no decimal literal is
        // silently rounded on the way in.
        const SCALES: [u32; 4] = [0x3f2a_aaab, 0x3f28_46ff, 0x3f2a_5349, 0x3f28_f5c3];

        let (o, i) = (2usize, 128usize);
        // Same deterministic input the reference was generated from.
        let w: Vec<f32> =
            (0..o * i).map(|k| (((k * 2654435761usize) % 1000) as f32 - 500.0) / 250.0).collect();
        let (q, sc) = quant_i3_g64(&w, o, i);
        assert_eq!(q.len(), PAYLOAD.len(), "payload size");
        assert_eq!(q, PAYLOAD, "payload must be byte-identical to colibri fmt 5");
        assert_eq!(sc.len(), SCALES.len(), "scale count");
        for (k, (a, b)) in sc.iter().zip(SCALES.iter()).enumerate() {
            assert_eq!(a.to_bits(), *b, "scale {k} must be bit-identical");
        }
    }

    #[test]
    fn int2_g64_round_trips_within_its_step() {
        let (o, i) = (3usize, 200usize); // 4 groups, last one ragged (8 of 64)
        let w: Vec<f32> = (0..o * i).map(|k| ((k % 37) as f32 - 18.0) / 6.0).collect();
        let (q, sc) = quant_i2_g64(&w, o, i);
        let ng = i.div_ceil(I2G_GROUP);
        assert_eq!(q.len(), o * ng * I2G_GROUP_BYTES, "16 B per group per row");
        assert_eq!(sc.len(), o * ng * I2G_SCALES_PER_GROUP, "scale AND zero per group");
        let view = QtView { fmt: QtFmt::Int2G64, o, i, gs: I2G_GROUP, q: &q, scale: &sc };
        let mut row = vec![0f32; i];
        for r in 0..o {
            view.dequant_row_into(r, &mut row);
            for k in 0..i {
                let step = sc[(r * ng + k / I2G_GROUP) * I2G_SCALES_PER_GROUP];
                let err = (row[k] - w[r * i + k]).abs();
                assert!(err <= step * 0.5 + 1e-6, "row {r} col {k}: err {err} > half step {step}");
            }
        }
    }

    #[test]
    fn int2_g64_reaches_all_four_levels_where_per_row_int2_reaches_three() {
        // The defect this format exists to fix. Per-row int2 uses `s = amax / 1`
        // and clamps to [-2, 1], so hitting -2 needs |w| >= 1.5*amax — impossible
        // when amax IS the row maximum. One level is dead in every row it writes.
        let i = I2G_GROUP;
        let w: Vec<f32> = (0..i).map(|k| -1.0 + 2.0 * (k as f32) / (i - 1) as f32).collect();

        let (q2, _) = quant_i2(&w, 1, i);
        let mut seen_row = std::collections::BTreeSet::new();
        for k in 0..i {
            seen_row.insert((q2[k >> 2] >> (2 * (k & 3))) & 0x03);
        }
        assert_eq!(seen_row.len(), 3, "per-row int2 is effectively ternary: {seen_row:?}");
        assert!(!seen_row.contains(&0), "field 0 (value -2) is the unreachable one");

        // Affine grouping maps [min, max] onto {0,1,2,3}, so all four are used.
        let (qg, _) = quant_i2_g64(&w, 1, i);
        let mut seen_g = std::collections::BTreeSet::new();
        for k in 0..i {
            seen_g.insert(i2g_field(qg[k >> 2], k) as u8);
        }
        assert_eq!(seen_g.len(), 4, "affine int2-g64 must use every level: {seen_g:?}");
    }

    #[test]
    fn int2_g64_beats_per_row_int2_on_reconstruction_error() {
        // The point of the format, stated as a measurement rather than a claim:
        // finer scales plus a live fourth level must reduce error on the same
        // weights. A skewed distribution is where an affine zero-point earns its
        // keep, since a symmetric grid wastes half its range.
        let (o, i) = (4usize, 256usize);
        let w: Vec<f32> = (0..o * i).map(|k| 0.3 + ((k % 53) as f32 / 53.0).powi(2)).collect();
        let err = |fmt, q: &[u8], sc: &[f32], gs| {
            let view = QtView { fmt, o, i, gs, q, scale: sc };
            let mut row = vec![0f32; i];
            let mut worst = 0f32;
            for r in 0..o {
                view.dequant_row_into(r, &mut row);
                for k in 0..i {
                    worst = worst.max((row[k] - w[r * i + k]).abs());
                }
            }
            worst
        };
        let (q2, s2) = quant_i2(&w, o, i);
        let (qg, sg) = quant_i2_g64(&w, o, i);
        let e_row = err(QtFmt::Int2, &q2, &s2, 0);
        let e_grp = err(QtFmt::Int2G64, &qg, &sg, I2G_GROUP);
        assert!(e_grp < e_row, "int2-g64 err {e_grp} must beat per-row int2 err {e_row}");
    }

    #[test]
    fn int2_g64_handles_a_constant_group_without_dividing_by_zero() {
        // max == min gives a zero scale. Every element must still come back as
        // exactly that constant rather than NaN.
        let i = I2G_GROUP;
        let w = vec![0.75f32; i];
        let (q, sc) = quant_i2_g64(&w, 1, i);
        assert_eq!(sc[0], 0.0, "a constant group has no spread");
        assert_eq!(sc[1], 0.75, "…and its zero-point is the constant");
        let view = QtView { fmt: QtFmt::Int2G64, o: 1, i, gs: I2G_GROUP, q: &q, scale: &sc };
        let mut row = vec![0f32; i];
        view.dequant_row_into(0, &mut row);
        assert!(row.iter().all(|&v| v == 0.75), "constant group must round-trip exactly");
    }

    #[test]
    fn int3_g64_round_trips_within_its_step() {
        // 3 bits over [-4,3] with a per-64 group scale: every value must come
        // back within half a quantization step of where it went in.
        let (o, i) = (3usize, 200usize); // 200 => 4 groups, last one ragged (8 of 64)
        let w: Vec<f32> = (0..o * i).map(|k| ((k % 37) as f32 - 18.0) / 6.0).collect();
        let (q, sc) = quant_i3_g64(&w, o, i);
        assert_eq!(q.len(), o * i.div_ceil(I3_GROUP) * I3_GROUP_BYTES, "24 B per group per row");
        assert_eq!(sc.len(), o * i.div_ceil(I3_GROUP), "one scale per group per row");
        let view = QtView { fmt: QtFmt::Int3G64, o, i, gs: I3_GROUP, q: &q, scale: &sc };
        let mut row = vec![0f32; i];
        for r in 0..o {
            view.dequant_row_into(r, &mut row);
            for k in 0..i {
                let step = sc[r * i.div_ceil(I3_GROUP) + k / I3_GROUP];
                let err = (row[k] - w[r * i + k]).abs();
                assert!(err <= step * 0.5 + 1e-6, "row {r} col {k}: err {err} > half step {step}");
            }
        }
    }

    #[test]
    fn int3_g64_plane_layout_matches_the_reference_encoder() {
        // The two planes are the compatibility surface with colibrì's fmt 5: the
        // low plane is the int2 layout (4 values/byte), the high plane is one bit
        // per value, 8 per byte. Encode known values and read the bytes directly
        // rather than through our own decoder, which would hide a shared error.
        let i = I3_GROUP;
        // scale 1.0 => value v encodes as biased v+4; pick the extremes and some middles
        let mut w = vec![0f32; i];
        // amax must be 3.0 for the scale to land on 1.0, so -4 is reached by
        // clamping rather than by being an input.
        w[0] = 3.0; // v = 3 -> u = 7 -> low 3, high 1
        w[1] = -3.0; // v = -3 -> u = 1 -> low 1, high 0
        w[2] = -1.0; // v = -1 -> u = 3 -> low 3, high 0
        w[8] = 0.0; // v = 0 -> u = 4 -> low 0, high 1
        let (q, sc) = quant_i3_g64(&w, 1, i);
        assert!((sc[0] - 1.0).abs() < 1e-6, "amax 3 / 3 = 1.0 scale");
        assert_eq!(q[0] & 0x03, 3, "value 0 low bits");
        assert_eq!((q[0] >> 2) & 0x03, 1, "value 1 low bits");
        assert_eq!((q[0] >> 4) & 0x03, 3, "value 2 low bits");
        assert_eq!(q[I3_LOW_BYTES] & 0x01, 1, "value 0 high bit");
        assert_eq!((q[I3_LOW_BYTES] >> 1) & 0x01, 0, "value 1 high bit");
        assert_eq!((q[I3_LOW_BYTES] >> 2) & 0x01, 0, "value 2 high bit");
        assert_eq!(q[I3_LOW_BYTES + 1] & 0x01, 1, "value 8 high bit lives in the next byte");
        // and the decoder agrees with the bytes
        assert_eq!(i3_value(q[0], q[I3_LOW_BYTES], 0), 3);
        assert_eq!(i3_value(q[0], q[I3_LOW_BYTES], 1), -3);
        assert_eq!(i3_value(q[0], q[I3_LOW_BYTES], 2), -1);
    }

    #[test]
    fn int3_g64_pads_a_ragged_group_with_zero_not_minus_three() {
        // Padding is stored as biased 4 (value 0). Writing the low plane instead
        // encodes -3, which would corrupt the tail of every row whose width is
        // not a multiple of 64 — silently, since dequant never reads that far.
        let i = 70usize; // 2 groups, second holds 6 real values + 58 pad
        let w: Vec<f32> = (0..i).map(|k| if k < 70 { 1.0 } else { 0.0 }).collect();
        let (q, sc) = quant_i3_g64(&w, 1, i);
        let gb = I3_GROUP_BYTES; // second group
        for k in 6..I3_GROUP {
            let v = i3_value(q[gb + (k >> 2)], q[gb + I3_LOW_BYTES + (k >> 3)], k);
            assert_eq!(v, 0, "pad slot {k} must decode to 0");
        }
        assert!(sc[1] > 0.0);
    }

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

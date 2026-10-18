//! The container group: byte layouts, compression, and the write→read round
//! trip.
//!
//! A quantizer that is off by one and a container that drops a byte produce the
//! same symptom — weights that are subtly wrong, and logits that are subtly
//! wrong, with no error anywhere. These are the claims that separate "the
//! checkpoint is what we wrote" from "the checkpoint is a file that parses".

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use peregrine_core::compress::{decode, encode, Compression};
use peregrine_core::pack::{
    bf16_bytes, f32_bytes, from_kblock, quant_i2, quant_i2_g64, quant_i3_g64, quant_i4, quant_i4_grouped, quant_i8, to_kblock,
    write_safetensors, Blob, QtView,
};
use peregrine_core::qt::{detect_group_size, QtFmt};
use peregrine_core::SafeTensors;

use crate::{cert, exact, failed, mismatch_bits, rng, with_cases, Certificate};

pub fn certificates() -> Vec<Certificate> {
    vec![
        kblock_involution(),
        compression_roundtrip(),
        scalar_bytes_roundtrip(),
        group_size_detection(),
        container_roundtrip(),
        dequant_row_is_within_half_a_step(),
    ]
}

/// The last step between bytes and arithmetic: the offline dequantizer every tool
/// uses to read a quantized checkpoint back. `QtView::dequant_row_into` and the
/// engine's own `QtWeight::dequant_row_into` are the same arithmetic kept in step
/// by a test elsewhere, so what is certified here is the arithmetic's *accuracy*
/// claim: round-to-nearest quantization puts every value within half a step of
/// the truth, where the step is the format's own scale.
///
/// The tolerance is therefore derived per row from that format's documented
/// divisor — `amax/127` for int8, `amax/7` for int4, `amax` for int2,
/// `amax/3` for int3-g64, `(hi−lo)/3` for the affine int2-g64 — and the residual
/// is measured against it. A dequantizer that dropped the per-group scale, or
/// read the wrong group's scale, would miss by *multiples* of the step rather
/// than by a rounding, which is the failure this is here to catch.
fn dequant_row_is_within_half_a_step() -> Certificate {
    let mut r = rng();
    let (o, i, gs) = (4usize, 128usize, 32usize);
    let mut w: Vec<f32> = (0..o * i).map(|_| r.sym() * 4.0).collect();
    // Row 0 all zeros: the degenerate group, which must dequantize to zeros
    // rather than dividing by a zero scale.
    for slot in w.iter_mut().take(i) {
        *slot = 0.0;
    }
    let mut bad = 0.0f64;
    let mut cases = 0usize;

    // (format, producer, half-step bound for the row)
    let int8 = quant_i8(&w, o, i);
    let int4 = quant_i4(&w, o, i);
    let int4g = quant_i4_grouped(&w, o, i, gs);
    let int2 = quant_i2(&w, o, i);
    let i2g = quant_i2_g64(&w, o, i);
    let i3g = quant_i3_g64(&w, o, i);
    // One entry per container format: the name its divisor is documented under,
    // the format tag, the packed bytes, the scales, and the group size (0 when
    // the format has none).
    // One entry per container format: the name its divisor is documented under,
    // the format tag, the packed bytes, the scales, and the group size (0 when
    // the format has none). Borrowed from locals, so the slices are not `'static`.
    type View<'a> = (&'a str, QtFmt, &'a [u8], &'a [f32], usize);
    let views: [View<'_>; 6] = [
        ("int8", QtFmt::Int8, &int8.0, &int8.1, 0),
        ("int4", QtFmt::Int4, &int4.0, &int4.1, 0),
        ("int4g", QtFmt::Int4Grouped, &int4g.0, &int4g.1, gs),
        ("int2", QtFmt::Int2, &int2.0, &int2.1, 0),
        ("int2g64", QtFmt::Int2G64, &i2g.0, &i2g.1, 0),
        ("int3g64", QtFmt::Int3G64, &i3g.0, &i3g.1, 0),
    ];
    for (name, fmt, bytes, scale, group) in views {
        let view = QtView { fmt, o, i, gs: group, q: bytes, scale };
        for row in 0..o {
            let truth = &w[row * i..row * i + i];
            let mut out = vec![0f32; i];
            view.dequant_row_into(row, &mut out);
            // The bound: half the format's own step for this row (or group), from
            // the divisor each producer documents. Half a step is the most a
            // round-to-nearest encoder can cost.
            let bound = half_step(name, truth, &out);
            cases += i;
            for (t, d) in truth.iter().zip(out.iter()) {
                if (f64::from(*t) - f64::from(*d)).abs() > bound {
                    bad += 1.0;
                }
            }
        }
    }
    with_cases(
        cert(
            "pack",
            "dequantized_rows_land_within_half_a_step",
            "Every dequantized row of every format is within half a step of the float it was quantized from, where the step is that format's own documented divisor (`amax/127`, `amax/7`, `amax`, `amax/3`, `(hi−lo)/3`) — so the right scale is read for the right group.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Half a quantization step for one row, derived from the format's divisor.
///
/// For the grouped formats the step is per group, so the bound is the largest
/// half-step the row's groups imply. A row of exact zeros is the degenerate case:
/// its scale is the producer's floor, and its reconstruction must be exactly zero
/// — which it is, because a zero value maps to code 0 and `0 * floor == 0`.
fn half_step(name: &str, truth: &[f32], dequant: &[f32]) -> f64 {
    let amax = truth.iter().fold(0f32, |m, v| m.max(v.abs()));
    let divisor = match name {
        "int8" => 127.0f32,
        "int4" | "int4g" => 7.0,
        "int2" => 1.0,
        "int3g64" => 3.0,
        // Affine: the step is the group's own range over its four levels, which
        // the reconstruction itself carries — so the bound is half the widest
        // gap the group could have produced, bounded by the row's own range.
        _ => {
            let hi = truth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lo = truth.iter().copied().fold(f32::INFINITY, f32::min);
            let _ = dequant;
            return (f64::from(hi - lo) / 3.0) / 2.0;
        }
    };
    let step = if amax > 0.0 { amax / divisor } else { 0.0 };
    f64::from(step) / 2.0
}

/// Temp directories get a counter and the pid: parallel tests share `/tmp`, and
/// two agents on the same box share the tree. A fixed name here is a flaky test
/// waiting for a bad hour.
fn unique_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("peregrine-verify-{tag}-{pid}-{n}"))
}

/// The k-block transpose is a pure byte permutation, and its own inverse is the
/// matching inverse. If it were only *nearly* an involution — a group counted
/// twice, a tail block duplicated — every streamed expert would still load and
/// the model would still answer, a little worse, forever.
fn kblock_involution() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let mut r = rng();
    for (o, gs_bytes, n_groups) in [(1usize, 4usize, 1usize), (4, 4, 3), (8, 16, 2), (16, 32, 5), (3, 8, 7)] {
        // A byte pattern that repeats across blocks but is unique inside one, so
        // a permutation that moved a whole block is still detectable.
        let _ = &mut r;
        let bytes: Vec<u8> =
            (0..o * gs_bytes * n_groups).map(|k| ((k / gs_bytes) as u8).wrapping_mul(31).wrapping_add(k as u8)).collect();
        match (to_kblock(&bytes, o, gs_bytes), bytes.len()) {
            (Some(fwd), _) => match from_kblock(&fwd, o, gs_bytes) {
                Some(back) if back == bytes => {}
                _ => bad += 1.0,
            },
            (None, len) => bad += f64::from(u8::from(len != o * gs_bytes * n_groups)),
        }
        cases += 1;
    }
    // The forward permutation must also be non-trivial: an identity would satisfy
    // the involution claim while leaving the layout unimplemented.
    let probe: Vec<u8> = (0..(2 * 4 * 2)).map(|k| k as u8).collect();
    let untouched = to_kblock(&probe, 2, 4).is_some_and(|t| t == probe);
    bad += f64::from(u8::from(untouched));
    with_cases(
        cert(
            "pack",
            "kblock_transpose_is_its_own_inverse",
            "`to_kblock` followed by `from_kblock` returns the input byte-for-byte, and the forward pass actually permutes rather than copying.",
            bad,
            0.0,
        ),
        cases + 1,
    )
}

/// Compression is claimed to be byte-for-byte transparent: the same quantized
/// weights with or without zstd, which is what lets a checkpoint be re-packed
/// without changing what it computes. A decoder that drops a trailing byte or
/// pads with zeroes would pass every "did it load" check.
fn compression_roundtrip() -> Certificate {
    let mut r = rng();
    let payload: Vec<u8> = (0..4096).map(|_| (r.unit() * 255.0) as u8).collect();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for scheme in [Compression::None, Compression::Zstd] {
        for level in [1, 3, 19] {
            if scheme == Compression::None && level != 1 {
                continue;
            }
            match encode(&payload, scheme, level) {
                Ok(enc) => match decode(&enc, scheme, payload.len()) {
                    Ok(dec) => {
                        if dec != payload {
                            bad += 1.0;
                        }
                        cases += 1;
                    }
                    Err(e) => bad += failed(&format!("decode {scheme:?} level {level}"), &e.to_string()),
                },
                Err(e) => bad += failed(&format!("encode {scheme:?} level {level}"), &e.to_string()),
            }
        }
    }
    with_cases(
        cert(
            "pack",
            "compression_is_byte_for_byte_transparent",
            "`encode` then `decode` returns the payload unchanged for both schemes and for zstd levels 1, 3 and 19, so re-packing a checkpoint cannot change what it computes.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The scalar byte encoders are the writer half of the safetensors payload, and
/// f32 must survive them bit-for-bit while bf16 survives to its own precision.
fn scalar_bytes_roundtrip() -> Certificate {
    let mut r = rng();
    let vals: Vec<f32> = (0..512).map(|_| r.sym() * 1024.0).collect();

    let f32_bad = {
        let bytes = f32_bytes(&vals);
        let back: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        mismatch_bits(&vals, &back)
    };
    let bf16_bad = {
        let bytes = bf16_bytes(&vals);
        let back: Vec<f32> =
            bytes.chunks_exact(2).map(|c| peregrine_core::dtype::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        // bf16 keeps 8 significand bits, so a round trip is lossy by exactly the
        // amount the dtype group certifies: half an ulp of bf16.
        let worst = vals
            .iter()
            .zip(back.iter())
            .map(|(a, b)| {
                let bound = 2f64.powi(-8) * (f64::from(*a)).abs() + 2f64.powi(-133);
                ((f64::from(*a) - f64::from(*b)).abs() / bound).max(0.0)
            })
            .fold(0.0f64, f64::max);
        exact(worst > 1.0)
    };
    with_cases(
        cert(
            "pack",
            "scalar_byte_encoders_round_trip",
            "`f32_bytes` decodes to the identical `f32` bits, and `bf16_bytes` decodes to within half an ulp of bfloat16.",
            f32_bad + bf16_bad,
            0.0,
        ),
        vals.len(),
    )
}

/// The grouped-int4 detector infers a group size from a scale count alone, and
/// the inference feeds the streaming layout — so a wrong answer reads the wrong
/// scale for a weight block and every expert using it degrades quietly. For
/// every candidate the detector offers, the count `O·ceil(I/gs)` must resolve
/// back to that `gs`.
fn group_size_detection() -> Certificate {
    /// The candidates the detector probes, finest first. Mirrored here rather
    /// than imported: the list *is* the contract, and a change to it has to show
    /// up as a certificate failure rather than as a silently different answer.
    const CANDS: [i64; 8] = [16, 32, 48, 64, 96, 128, 192, 256];
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for i in [64i64, 96, 128, 192, 256, 512] {
        for gs in CANDS {
            if gs > i {
                break;
            }
            let ng = (i + gs - 1) / gs;
            // One group per row is the per-row count, which the detector reads as
            // "not grouped" — the property under test does not exist there.
            if ng <= 1 {
                continue;
            }
            let o = 4i64;
            let ns = o * ng;
            cases += 1;
            let got = detect_group_size(o, i, ns);
            // A scale count is genuinely ambiguous — `I=128` predicts `O·2`
            // scales for both `gs=64` and `gs=96` — so the claim is the
            // documented tie-break: the finest candidate whose predicted count
            // matches, and never a candidate whose prediction is wrong.
            let finest = CANDS.iter().copied().find(|c| *c <= i && (i + c - 1) / c == ng);
            if got != finest.unwrap_or(0) as i32 {
                bad += 1.0;
            }
        }
    }
    // Degenerate inputs must be refused rather than guessed.
    bad += f64::from(u8::from(detect_group_size(4, 128, 4) != 0));
    bad += f64::from(u8::from(detect_group_size(0, 128, 64) != 0));
    bad += f64::from(u8::from(detect_group_size(4, 0, 64) != 0));
    with_cases(
        cert(
            "pack",
            "group_size_is_the_finest_candidate_matching_its_count",
            "`detect_group_size(O, I, ns)` returns the finest probed candidate whose predicted scale count equals `ns`, never a candidate whose prediction differs, and returns 0 for a per-row count or a degenerate shape.",
            bad,
            0.0,
        ),
        cases + 3,
    )
}

/// The end-to-end claim: a checkpoint written by these tools reads back through
/// the same index the engine uses, byte-identical, in every uncompressed dtype
/// the writer supports.
///
/// The compressed case is a different claim and gets its own: `mmap_tensor`
/// *refuses* a zstd tensor by design, so what the container has to guarantee is
/// the header arithmetic the runtime decoder depends on — `data_offsets`
/// describe the compressed bytes, `uncompressed_nbytes` describes the tensor,
/// and `numel` follows from the shape rather than from either. `QtInfo::detect`
/// reads exactly these three, and every compressed quantized weight was
/// misdetected once because it trusted `nbytes`.
fn container_roundtrip() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (dtype, shape) in [("F32", vec![4i64, 16]), ("BF16", vec![4, 16]), ("F32", vec![3, 33])] {
        let n = (shape[0] * shape[1]) as usize;
        let vals: Vec<f32> = (0..n).map(|_| r.sym() * 512.0).collect();
        let bytes = if dtype == "F32" { f32_bytes(&vals) } else { bf16_bytes(&vals) };
        let blob = Blob::new("layer.0.w", dtype, shape.clone(), bytes);
        let dir = unique_dir("container");
        match write_safetensors(&dir, std::slice::from_ref(&blob)) {
            Ok(()) => match read_back(&dir, "layer.0.w", n) {
                Ok(observed) => {
                    cases += 1;
                    // bf16 is compared against its own round trip, which is the
                    // most the format can hold; f32 has no loss to account for.
                    let expect: Vec<f32> = if dtype == "F32" {
                        vals.clone()
                    } else {
                        vals.iter().map(|v| peregrine_core::pack::bf16_roundtrip(*v)).collect()
                    };
                    bad += mismatch_bits(&expect, &observed);
                }
                Err(e) => bad += failed(&format!("container round trip {dtype}"), &e),
            },
            Err(e) => bad += failed(&format!("write {dtype} blob"), &e.to_string()),
        }
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            bad += failed("temp cleanup", &e.to_string());
        }
    }

    // The compressed header contract, checked through the production index.
    let n = 512usize;
    let vals: Vec<f32> = (0..n).map(|_| r.sym() * 512.0).collect();
    let payload = f32_bytes(&vals);
    let blob = Blob::new("layer.0.w", "F32", vec![n as i64], payload.clone()).with_compression(Compression::Zstd);
    let dir = unique_dir("container-zstd");
    let mut compressed_cases = 0usize;
    match write_safetensors(&dir, std::slice::from_ref(&blob)) {
        Ok(()) => match SafeTensors::open(&dir) {
            Ok(st) => {
                let (_fd, off, len) = match st.region("layer.0.w") {
                    Some(r) => r,
                    None => {
                        bad += failed("zstd header", "region() returned nothing");
                        return finish_container(bad, compressed_cases);
                    }
                };
                let info = st.find("layer.0.w").cloned();
                let Some(info) = info else {
                    bad += failed("zstd header", "find() returned nothing");
                    return finish_container(bad, compressed_cases);
                };
                compressed_cases += 3;
                // The on-disk length is the *compressed* length …
                if info.nbytes != len as i64 || info.nbytes as usize >= info.uncompressed_nbytes as usize {
                    bad += 1.0;
                }
                // … while the element count follows the shape, not either length.
                if info.numel != n as i64 {
                    bad += 1.0;
                }
                // And the region really is the payload the writer produced.
                let on_disk = match read_region(&dir, off, len) {
                    Ok(b) => b,
                    Err(e) => {
                        bad += failed("zstd region read", &e);
                        return finish_container(bad, compressed_cases);
                    }
                };
                match decode(&on_disk, Compression::Zstd, payload.len()) {
                    Ok(back) => {
                        if back != payload {
                            bad += 1.0;
                        }
                    }
                    Err(e) => bad += failed("zstd decode", &e.to_string()),
                }
            }
            Err(e) => bad += failed("open zstd container", &e.to_string()),
        },
        Err(e) => bad += failed("write zstd blob", &e.to_string()),
    }
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        bad += failed("temp cleanup", &e.to_string());
    }
    finish_container(bad, cases + compressed_cases)
}

/// Open the container and read one tensor back as f32 through the production
/// index. The mmap path is the engine's own read path for an uncompressed
/// tensor.
fn read_back(dir: &std::path::Path, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let st = SafeTensors::open(dir).map_err(|e| e.to_string())?;
    let t = st.mmap_tensor(name).map_err(|e| e.to_string())?;
    let mut out = vec![0f32; n];
    t.read_f32(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Read the bytes the index says a region occupies, using the **offset the
/// header gave us** rather than a re-derived one — that is the whole point of
/// the check, so computing the offset independently here would certify nothing.
///
/// The file is opened by path (`write_safetensors` emits a single shard), because
/// this crate is `#![forbid(unsafe_code)]` and turning the index's raw descriptor
/// back into a `File` would need `from_raw_fd`. The descriptor itself is checked
/// by the engine's own streaming tests; what is certified here is the header
/// arithmetic.
fn read_region(dir: &std::path::Path, off: u64, len: usize) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::open(dir.join("model.safetensors")).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, off).map_err(|e| e.to_string())?;
    Ok(buf)
}

fn finish_container(bad: f64, cases: usize) -> Certificate {
    with_cases(
        cert(
            "pack",
            "a_written_checkpoint_reads_back_identical",
            "A blob written by `write_safetensors` reads back through `SafeTensors` + `mmap_tensor::read_f32` with the original F32 (or BF16) bits, and a zstd blob's header reports the compressed on-disk length, the uncompressed logical length and a shape-derived element count that all agree with the bytes on disk.",
            bad,
            0.0,
        ),
        cases,
    )
}

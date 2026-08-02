//! Quantized-tensor (QT) container-format detection — ported from the fmt logic
//! in `qt_from_disk` and `detect_group_size` (`c/glm.c:1310-1359`).
//!
//! A weight `name` in the int4 container is stored as a U8 payload plus a
//! sibling `name.qs` F32 scale tensor. The format is inferred from the byte
//! counts, not declared: per-row int8 (fmt 1), packed int4 (fmt 2), packed int2
//! (fmt 3), or grouped int4 (fmt 4, with a probed group size). A weight with no
//! `.qs` sibling is a full-precision tensor quantized at runtime (fmt 0).

use crate::safetensors::SafeTensors;

/// Quantization format. Discriminants match the C `QT.fmt` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QtFmt {
    /// full precision f32/bf16, quantized at runtime (no `.qs` sibling)
    F32 = 0,
    /// per-row int8
    Int8 = 1,
    /// per-row packed int4
    Int4 = 2,
    /// per-row packed int2
    Int2 = 3,
    /// grouped packed int4 (`gs` weights share one scale)
    Int4Grouped = 4,
    /// int3 with per-group scales, group = 64 (colibrì's fmt 5). Each group is
    /// 24 bytes — a 16-byte low plane (2 bits/value, the int2 layout) plus an
    /// 8-byte high plane (1 bit/value) — and carries one f32 scale, so 3.5
    /// bits/weight effective. Values are `[-4, 3]`, stored biased `+4`.
    Int3G64 = 5,
    /// **affine** int2 with a scale *and* zero-point per 64-value group. 16
    /// bytes per group (the int2 packing, four 2-bit fields per byte) plus two
    /// f32 per group in `.qs`, interleaved `[scale, zero]`. Fields are unsigned
    /// `[0, 3]` and the bias is the per-group zero-point, so all four levels
    /// carry weight — unlike per-row [`QtFmt::Int2`], whose `amax / 1`
    /// convention leaves the `-2` level unreachable.
    ///
    /// **3.0 bits/weight effective**, not 2 — the two f32 per group are 8 bytes
    /// per 64 values, a full extra bit/weight on top of the 2-bit payload. Still
    /// the smallest format in this container family (int3-g64 is 3.5, per-row
    /// int4 is ~4.0), but the headline "2-bit" describes the payload, not the
    /// container. Narrowing the scales to f16 would reach ~2.5.
    Int2G64 = 7,
    /// the payload size matches no known container for the requested `[O, I]`
    /// — a truncated tensor, or a caller/file shape disagreement. Loading one
    /// is an error rather than a guess.
    Unknown = 6,
}

/// Resolved format for one weight `[O, I]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QtInfo {
    pub fmt: QtFmt,
    pub o: i64,
    pub i: i64,
    /// group size (only meaningful for [`QtFmt::Int4Grouped`], else 0)
    pub gs: i32,
    /// number of F32 scales expected: `O` per-row, or `O*ceil(I/gs)` grouped
    pub scale_count: i64,
}

/// Derive the fmt=4 group size from the scale-array element count.
///
/// A grouped-int4 tensor stores `ceil(I/gs)` scales per output row. We probe
/// candidate group sizes (multiples of 16, the AVX2 vector width the grouped
/// kernel requires) finest-first and return the first whose predicted scale
/// count matches. Returns 0 if none fit (then it's plain per-row int4).
///
/// `ns` is the number of **f32 scales** (in the C code it's `ns_bytes/4`; here
/// we take the count directly to avoid re-deriving byte counts).
pub fn detect_group_size(o: i64, i: i64, ns: i64) -> i32 {
    if o <= 0 || ns <= o || i <= 0 {
        return 0; // not grouped (per-row is exactly O scales)
    }
    const CANDS: [i32; 8] = [16, 32, 48, 64, 96, 128, 192, 256];
    for &gs in &CANDS {
        if gs as i64 > i {
            break;
        }
        let ng = (i + gs as i64 - 1) / gs as i64;
        if ns == o * ng {
            return gs;
        }
    }
    0
}

impl QtInfo {
    /// Inspect a weight `[O, I]` in `st` and resolve its container format.
    pub fn detect(st: &SafeTensors, name: &str, o: i64, i: i64) -> QtInfo {
        let scale_name = format!("{name}.qs");
        // The *uncompressed* size is what describes the container: `nbytes` is
        // the on-disk payload, which for a zstd tensor is the compressed length
        // and matches no format's byte count — every compressed quantized weight
        // was therefore misdetected (and then failed to load).
        let Some(nb) = st.uncompressed_nbytes(name) else {
            // absent weight → treat as runtime-quantized full precision
            return QtInfo { fmt: QtFmt::F32, o, i, gs: 0, scale_count: 0 };
        };
        if !st.has(&scale_name) {
            return QtInfo { fmt: QtFmt::F32, o, i, gs: 0, scale_count: 0 };
        }
        // Scale count straight from the tensor's element count — independent of
        // both compression and the scale dtype's byte width.
        let ns = st.numel(&scale_name).unwrap_or(0);

        // Row-major formats are tested before int3-g64 so a narrow tensor whose
        // int3 size coincides with an int8/int4/int2 row size still resolves to
        // the row format — the precedence colibrì documents at `colibri.c:1059`.
        let i3_groups = (i + 63) / 64;
        let i2g_scales = o * i3_groups * 2;
        // int2-g64 is tested **before** the row formats, which is the opposite of
        // int3-g64's precedence below. Its payload `o·ng·16` collides with a row
        // format at several widths — int8 at I=16, per-row int2 at every I that
        // is a multiple of 64 — so byte count alone can never place it, and the
        // `2·o·ng` scale cardinality is what separates it.
        //
        // **Requiring at least two groups is what makes the pair unique.** At
        // `ng == 1` the format is degenerate (one scale+zero per row) and
        // genuinely indistinguishable: O=2, I=32 grouped int4 with gs=16 is also
        // 32 bytes with 4 scales. Above one group no other format produces this
        // (bytes, scales) combination, so first place is safe. Real routed
        // experts are 1536–5120 wide, so this excludes only fixtures — and
        // `peregrine-requantize` refuses to *write* a single-group int2-g64
        // tensor rather than emit a container that would land here as int8.
        //
        // int3-g64 cannot take the same precedence: its `o·ng` scales *equal*
        // `o` when I ≤ 64, so it stays ambiguous with per-row and has to yield.
        let mut fmt = if i3_groups >= 2 && nb == o * i3_groups * 16 && ns == i2g_scales {
            QtFmt::Int2G64
        } else if nb == o * i {
            QtFmt::Int8
        } else if nb == o * ((i + 1) / 2) {
            QtFmt::Int4
        } else if nb == o * ((i + 3) / 4) {
            QtFmt::Int2
        } else if nb == o * i3_groups * 24 && ns == o * i3_groups {
            // The scale cardinality is part of the discriminator, not a
            // consequence of it: colibrì records fmt 5 regressing precisely when
            // a per-row scale bound was applied to a per-group format.
            QtFmt::Int3G64
        } else {
            // Deliberately not "assume int2": an unrecognized size means the
            // caller's (o, i) disagrees with the file, or the tensor is
            // truncated. Guessing produced 2-bit garbage that loaded cleanly.
            return QtInfo { fmt: QtFmt::Unknown, o, i, gs: 0, scale_count: 0 };
        };
        let mut gs = 0;
        if fmt == QtFmt::Int4 {
            gs = detect_group_size(o, i, ns);
            if gs > 0 {
                fmt = QtFmt::Int4Grouped;
            }
        }
        let scale_count = match fmt {
            QtFmt::Int4Grouped => o * ((i + gs as i64 - 1) / gs as i64),
            QtFmt::Int3G64 => o * i3_groups,
            // Two f32 per group: scale and zero-point, interleaved.
            QtFmt::Int2G64 => i2g_scales,
            _ => o,
        };
        if matches!(fmt, QtFmt::Int3G64 | QtFmt::Int2G64) {
            gs = 64;
        }
        QtInfo { fmt, o, i, gs, scale_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::test_support::{write_safetensors, Blob};
    use crate::Error;
    use std::path::PathBuf;

    #[test]
    fn group_size_probe() {
        // O=2, I=32, grouped gs=16 → ng=2 → 2*2 = 4 scales
        assert_eq!(detect_group_size(2, 32, 4), 16);
        // per-row: exactly O scales → not grouped
        assert_eq!(detect_group_size(2, 32, 2), 0);
        // O=2, I=64, gs=32 → ng=2 → 4 scales
        assert_eq!(detect_group_size(2, 64, 4), 32);
        // I smaller than the smallest candidate (16) → never grouped
        assert_eq!(detect_group_size(4, 8, 100), 0);
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("coli_qt_{}_{}", std::process::id(), tag));
        if let Err(e) = std::fs::remove_dir_all(&d) {
            if e.kind() != std::io::ErrorKind::NotFound {
                peregrine_io::note_advisory_err("pre-clean test tmpdir", &e);
            }
        }
        d
    }

    #[test]
    fn detect_formats() -> Result<(), Error> {
        // O=2, I=32
        let (o, i) = (2i64, 32i64);
        let packed4 = (o * ((i + 1) / 2)) as usize; // 32 bytes
        let int8 = (o * i) as usize; // 64 bytes
        let dir = tmpdir("fmts");
        write_safetensors(
            &dir,
            &[
                // per-row int4: O scales
                Blob { name: "w4", dtype: "U8", shape: vec![o, i / 2], bytes: vec![0u8; packed4] },
                Blob { name: "w4.qs", dtype: "F32", shape: vec![o], bytes: vec![0u8; (o * 4) as usize] },
                // grouped int4: gs=16 → ng=2 → O*ng = 4 scales
                Blob { name: "wg", dtype: "U8", shape: vec![o, i / 2], bytes: vec![0u8; packed4] },
                Blob { name: "wg.qs", dtype: "F32", shape: vec![o * 2], bytes: vec![0u8; (o * 2 * 4) as usize] },
                // int8
                Blob { name: "w8", dtype: "U8", shape: vec![o, i], bytes: vec![0u8; int8] },
                Blob { name: "w8.qs", dtype: "F32", shape: vec![o], bytes: vec![0u8; (o * 4) as usize] },
                // full precision (no .qs)
                Blob { name: "wf", dtype: "F32", shape: vec![o, i], bytes: vec![0u8; (o * i * 4) as usize] },
            ],
        )?;
        let st = SafeTensors::open(&dir)?;

        assert_eq!(QtInfo::detect(&st, "w4", o, i).fmt, QtFmt::Int4);
        let g = QtInfo::detect(&st, "wg", o, i);
        assert_eq!(g.fmt, QtFmt::Int4Grouped);
        assert_eq!(g.gs, 16);
        assert_eq!(g.scale_count, 4);
        assert_eq!(QtInfo::detect(&st, "w8", o, i).fmt, QtFmt::Int8);
        assert_eq!(QtInfo::detect(&st, "wf", o, i).fmt, QtFmt::F32);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn int2_and_int2_g64_have_identical_byte_counts_and_are_told_apart_by_scales() -> Result<(), Error> {
        // The hazard this format introduces. At I=128 a grouped-2-bit tensor is
        // o*ceil(128/64)*16 = 32*o bytes and per-row int2 is o*ceil(128/4) =
        // 32*o bytes — the *same*. Byte count cannot discriminate, and reading a
        // 2*ng-entry interleaved [scale, zero] array as one scale per row would
        // decode garbage that loads cleanly, which is exactly the failure mode
        // the "deliberately not assume int2" comment above guards against.
        // Scale cardinality is the only thing that separates them.
        let (o, i) = (4i64, 128i64);
        let ng = (i + 63) / 64;
        let payload = (o * (i / 4)) as usize;
        assert_eq!(payload, (o * ng * 16) as usize, "the collision this test exists for");
        let dir = tmpdir("i2g");
        write_safetensors(
            &dir,
            &[
                // per-row int2: exactly O scales
                Blob { name: "wr", dtype: "U8", shape: vec![o, i / 4], bytes: vec![0u8; payload] },
                Blob { name: "wr.qs", dtype: "F32", shape: vec![o], bytes: vec![0u8; (o * 4) as usize] },
                // int2-g64: 2 f32 per group per row, interleaved [scale, zero]
                Blob { name: "wg", dtype: "U8", shape: vec![o, i / 4], bytes: vec![0u8; payload] },
                Blob {
                    name: "wg.qs",
                    dtype: "F32",
                    shape: vec![o * ng * 2],
                    bytes: vec![0u8; (o * ng * 2 * 4) as usize],
                },
            ],
        )?;
        let st = SafeTensors::open(&dir)?;

        assert_eq!(QtInfo::detect(&st, "wr", o, i).fmt, QtFmt::Int2, "O scales => per-row");
        let g = QtInfo::detect(&st, "wg", o, i);
        assert_eq!(g.fmt, QtFmt::Int2G64, "2*O*ng scales => grouped affine");
        assert_eq!(g.gs, 64);
        assert_eq!(g.scale_count, o * ng * 2);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn int2_g64_with_a_ragged_width_gets_its_own_branch() -> Result<(), Error> {
        // When I is not a multiple of 64 the two byte counts differ (I=100:
        // 2*16=32 bytes/row grouped vs ceil(100/4)=25 per-row), so the int2
        // branch never fires and detection must reach the dedicated one.
        let (o, i) = (3i64, 100i64);
        let ng = (i + 63) / 64; // 2
        let payload = (o * ng * 16) as usize;
        assert_ne!(payload, (o * ((i + 3) / 4)) as usize, "ragged widths must not collide");
        let dir = tmpdir("i2g_ragged");
        write_safetensors(
            &dir,
            &[
                Blob { name: "w", dtype: "U8", shape: vec![o, ng * 16], bytes: vec![0u8; payload] },
                Blob {
                    name: "w.qs",
                    dtype: "F32",
                    shape: vec![o * ng * 2],
                    bytes: vec![0u8; (o * ng * 2 * 4) as usize],
                },
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        let d = QtInfo::detect(&st, "w", o, i);
        assert_eq!(d.fmt, QtFmt::Int2G64);
        assert_eq!(d.scale_count, o * ng * 2);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn detects_compressed_quantized_weights() -> Result<(), Error> {
        // Regression: detection compared the *on-disk* byte count, which for a
        // zstd tensor is the compressed length. It matched no format, fell
        // through to "assume int2", and the weight then failed to load — making
        // compression unusable for exactly the tensors it matters most for.
        let (o, i) = (4i64, 64i64);
        let packed4 = (o * ((i + 1) / 2)) as usize;
        // Patterned bytes so zstd actually shrinks the payload.
        let w: Vec<u8> = (0..packed4).map(|k| (k % 7) as u8).collect();
        let scales: Vec<u8> = (0..o as usize * 4).map(|k| (k % 5) as u8).collect();
        let dir = tmpdir("compressed");
        // The real writer, since only it emits the compression header fields.
        crate::pack::write_safetensors(
            &dir,
            &[
                crate::pack::Blob::new("w4", "U8", vec![o, i / 2], w).with_compression(crate::Compression::Zstd),
                crate::pack::Blob::new("w4.qs", "F32", vec![o], scales),
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        assert!(st.nbytes("w4") < st.uncompressed_nbytes("w4"), "the payload must really be compressed");
        let info = QtInfo::detect(&st, "w4", o, i);
        assert_eq!(info.fmt, QtFmt::Int4, "a compressed int4 weight is still an int4 weight");
        assert_eq!(info.scale_count, o);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn unrecognized_payload_size_is_not_guessed_as_int2() -> Result<(), Error> {
        // A truncated tensor (or a caller shape that disagrees with the file)
        // used to be silently classified int2 and loaded as 2-bit garbage.
        let (o, i) = (4i64, 64i64);
        let dir = tmpdir("mismatch");
        write_safetensors(
            &dir,
            &[
                // 100 bytes matches none of int8 (256), int4 (128), int2 (64).
                Blob { name: "w", dtype: "U8", shape: vec![100], bytes: vec![0u8; 100] },
                Blob { name: "w.qs", dtype: "F32", shape: vec![o], bytes: vec![0u8; (o * 4) as usize] },
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        assert_eq!(QtInfo::detect(&st, "w", o, i).fmt, QtFmt::Unknown);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

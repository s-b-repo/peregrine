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
    /// the payload size matches no known container for the requested `[O, I]`
    /// — a truncated tensor, or a caller/file shape disagreement. Loading one
    /// is an error rather than a guess.
    Unknown = 5,
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

        let mut fmt = if nb == o * i {
            QtFmt::Int8
        } else if nb == o * ((i + 1) / 2) {
            QtFmt::Int4
        } else if nb == o * ((i + 3) / 4) {
            QtFmt::Int2
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
        let scale_count = if fmt == QtFmt::Int4Grouped {
            let ng = (i + gs as i64 - 1) / gs as i64;
            o * ng
        } else {
            o
        };
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

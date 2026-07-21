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
        let Some(nb) = st.nbytes(name) else {
            // absent weight → treat as runtime-quantized full precision
            return QtInfo { fmt: QtFmt::F32, o, i, gs: 0, scale_count: 0 };
        };
        if !st.has(&scale_name) {
            return QtInfo { fmt: QtFmt::F32, o, i, gs: 0, scale_count: 0 };
        }
        // scales are F32; count = bytes / 4
        let ns = st.nbytes(&scale_name).unwrap_or(0) / 4;

        let mut fmt = if nb == o * i {
            QtFmt::Int8
        } else if nb == o * ((i + 1) / 2) {
            QtFmt::Int4
        } else {
            QtFmt::Int2
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
        let _ = std::fs::remove_dir_all(&d);
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
}

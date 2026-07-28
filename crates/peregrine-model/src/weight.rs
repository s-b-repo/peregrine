//! `QtWeight` — the bridge from `peregrine-core` container formats to `peregrine-kernels`
//! matmuls. Owns one quantized weight `[O, I]` (packed bytes + per-row scales)
//! and applies it to f32 activations via the IDOT path (quantize activations
//! with `qrow_i8`, then integer-dot).
//!
//! Supports every container format GLM-5.2 experts ship in — per-row int8
//! (fmt 1), per-row packed int4 (fmt 2), grouped packed int4 (fmt 4, the
//! coherence-critical format for GLM-5.2), and packed int2 (fmt 3).

use peregrine_core::{Error, QtFmt, QtInfo, SafeTensors};
use peregrine_io::Bytes;
use peregrine_kernels::{
    dot_i2i8, matmul_i4_from_f32, matmul_i4g_from_f32, matmul_i8_from_f32, qrow_i8,
};

/// The quantized formats the compute path supports. F32 is rejected at load, so
/// it is deliberately *unrepresentable* here — this makes `apply`/`dequant`
/// exhaustive with no `unreachable!`/panic branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFmt {
    Int8,
    Int4,
    Int4Grouped,
    Int2,
}

impl QuantFmt {
    /// Narrow a detected container format to a computable one; `F32` (a tensor
    /// with no `.qs` scale sibling) has no quantized compute path here.
    pub fn from_qt(f: QtFmt) -> Option<QuantFmt> {
        match f {
            QtFmt::Int8 => Some(QuantFmt::Int8),
            QtFmt::Int4 => Some(QuantFmt::Int4),
            QtFmt::Int4Grouped => Some(QuantFmt::Int4Grouped),
            QtFmt::Int2 => Some(QuantFmt::Int2),
            QtFmt::F32 => None,
        }
    }
}

/// One quantized weight matrix `[O, I]`.
pub struct QtWeight {
    pub fmt: QuantFmt,
    pub o: usize,
    pub i: usize,
    /// packed weight bytes: int8 as u8 (reinterpreted), int4 as nibbles, int2 as
    /// 2-bit fields (4/byte). A [`Bytes`] region so a streamed expert can own its
    /// O_DIRECT aligned DMA buffer directly (zero-copy); resident loads use a plain
    /// `Vec`. Every kernel reads it as `&[u8]` via `Deref`, so the shape is invisible
    /// to the compute path.
    q: Bytes,
    /// scales: `O` per-row (int8/int4/int2) or `O*ceil(I/gs)` grouped (fmt 4),
    /// laid out `scale[o*ng + g]`
    scale: Vec<f32>,
    /// group size for [`QtFmt::Int4Grouped`] (weights per shared scale), else 0
    gs: usize,
}

impl QtWeight {
    /// Build from already-quantized per-row data (also the test constructor).
    /// For grouped-int4 use [`Self::new_grouped`].
    pub fn new(fmt: QuantFmt, o: usize, i: usize, q: impl Into<Bytes>, scale: Vec<f32>) -> QtWeight {
        debug_assert!(
            matches!(fmt, QuantFmt::Int8 | QuantFmt::Int4 | QuantFmt::Int2),
            "QtWeight::new is for per-row formats (got {fmt:?}); use new_grouped for grouped int4"
        );
        QtWeight { fmt, o, i, q: q.into(), scale, gs: 0 }
    }

    /// Build a grouped-int4 weight: `scale` holds `o*ceil(i/gs)` entries laid out
    /// `scale[o*ng + g]`.
    pub fn new_grouped(o: usize, i: usize, q: impl Into<Bytes>, scale: Vec<f32>, gs: usize) -> QtWeight {
        debug_assert!(gs > 0 && gs.is_multiple_of(16), "grouped-int4 gs must be a positive multiple of 16");
        QtWeight { fmt: QuantFmt::Int4Grouped, o, i, q: q.into(), scale, gs }
    }

    /// Load a container weight `[O, I]` (`name` + `name.qs`) from a model dir.
    pub fn load(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<QtWeight, Error> {
        let info = QtInfo::detect(st, name, o as i64, i as i64);
        let fmt = QuantFmt::from_qt(info.fmt).ok_or_else(|| {
            Error::Format(format!(
                "weight '{name}': no `.qs` scale sibling (runtime-f32 weights are not \
                 supported on the quantized expert path)"
            ))
        })?;
        let nb = match fmt {
            QuantFmt::Int8 => o * i,
            QuantFmt::Int4 | QuantFmt::Int4Grouped => o * (i.div_ceil(2)),
            QuantFmt::Int2 => o * (i.div_ceil(4)),
        };
        let mut q = vec![0u8; nb];
        st.read_raw(name, &mut q)?;
        let mut scale = vec![0f32; info.scale_count as usize];
        st.read_f32(&format!("{name}.qs"), &mut scale)?;
        Ok(QtWeight { fmt, o, i, q: q.into(), scale, gs: info.gs as usize })
    }

    /// Raw quantized payload: `(packed_bytes, per_row_scales)`. Lets the
    /// scheduler serialize a resident expert to a disk blob and reconstruct it
    /// after streaming (`QtWeight::new`).
    pub fn raw(&self) -> (&[u8], &[f32]) {
        (&self.q[..], &self.scale)
    }

    /// `q` viewed as int8 (only meaningful for [`QuantFmt::Int8`]). `u8` and `i8`
    /// are both `Pod` with identical layout, so `bytemuck` reinterprets the slice
    /// with no `unsafe` and no copy.
    fn as_i8(&self) -> &[i8] {
        bytemuck::cast_slice::<u8, i8>(&self.q[..])
    }

    /// `y[s_n, O] = apply(self, x[s_n, I])`. Caller provides int8 activation
    /// scratch `xq[s_n*I]`, per-row scale scratch `sx[s_n]`, and output `y`.
    pub fn apply(&self, x: &[f32], s_n: usize, xq: &mut [i8], sx: &mut [f32], y: &mut [f32]) {
        match self.fmt {
            QuantFmt::Int8 => matmul_i8_from_f32(y, x, self.as_i8(), &self.scale, s_n, self.i, self.o, xq, sx),
            QuantFmt::Int4 => matmul_i4_from_f32(y, x, &self.q[..], &self.scale, s_n, self.i, self.o, xq, sx),
            QuantFmt::Int4Grouped => {
                matmul_i4g_from_f32(y, x, &self.q[..], &self.scale, s_n, self.i, self.o, self.gs, xq, sx)
            }
            QuantFmt::Int2 => {
                // Quantize activations, then int2·int8 dot each row (AVX2 when
                // available, else scalar — bit-identical). int2 is rare (extra
                // compression); the scalar path remains the token-exact reference.
                for s in 0..s_n {
                    sx[s] = qrow_i8(&x[s * self.i..s * self.i + self.i], &mut xq[s * self.i..s * self.i + self.i]);
                }
                let rb = self.i.div_ceil(4);
                for o in 0..self.o {
                    let w = &self.q[o * rb..o * rb + rb];
                    let sc = self.scale[o];
                    for s in 0..s_n {
                        let d = dot_i2i8(w, &xq[s * self.i..s * self.i + self.i], self.i) as f32;
                        y[s * self.o + o] = d * sc * sx[s];
                    }
                }
            }
        }
    }

    /// Allocating convenience over [`Self::apply`]. Output rows are independent, so
    /// the matmul runs as batched row-chunks on the persistent compute pool above
    /// `PAR_MATMUL_MIN` — each chunk does one batched `apply` with its own quantized
    /// scratch, so the result is bit-identical to the serial whole-matrix call
    /// (guarded by `apply_vec_parallel_matches_serial`). Serial below the gate or
    /// when nested (e.g. an expert matmul inside a parallel MoE), so no
    /// oversubscription. This parallelizes every projection, lm_head, and expert.
    pub fn apply_vec(&self, x: &[f32], s_n: usize) -> Vec<f32> {
        let mut y = vec![0f32; s_n * self.o];
        // Only parallelize when the per-row matmul work (`i·o` MACs) is large enough
        // that the pool dispatch pays off; tiny matrices (e.g. the test model) stay
        // serial regardless of batch, avoiding overhead-dominated slowdowns. Real
        // GLM-5.2 projections (6144×6144 ≈ 38M) clear this by ~36×.
        let gate = if self.i * self.o >= 1 << 20 { peregrine_par::PAR_MATMUL_MIN } else { usize::MAX };
        peregrine_par::par_chunks_mut(&mut y, self.o, s_n, gate, |start, end, y_chunk| {
            let n = end - start;
            let mut xq = vec![0i8; n * self.i];
            let mut sx = vec![0f32; n];
            self.apply(&x[start * self.i..end * self.i], n, &mut xq, &mut sx, y_chunk);
        });
        y
    }

    /// Dequantize a single output row `o` to f32 `[I]` — used by the MLA
    /// absorption path (`qt_addrow` / `qt_matvec_rows`), which reads individual
    /// `kv_b` rows rather than a batched matmul.
    pub fn dequant_row(&self, o: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.i];
        match self.fmt {
            QuantFmt::Int8 => {
                let s = self.scale[o];
                let q = self.as_i8();
                for i in 0..self.i {
                    out[i] = q[o * self.i + i] as f32 * s;
                }
            }
            QuantFmt::Int4 => {
                let s = self.scale[o];
                let rb = self.i.div_ceil(2);
                for i in 0..self.i {
                    let byte = self.q[o * rb + (i >> 1)];
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    out[i] = (nib - 8) as f32 * s;
                }
            }
            QuantFmt::Int4Grouped => {
                let rb = self.i.div_ceil(2);
                let ng = self.i.div_ceil(self.gs);
                for i in 0..self.i {
                    let s = self.scale[o * ng + i / self.gs];
                    let byte = self.q[o * rb + (i >> 1)];
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    out[i] = (nib - 8) as f32 * s;
                }
            }
            QuantFmt::Int2 => {
                let s = self.scale[o];
                let rb = self.i.div_ceil(4);
                for i in 0..self.i {
                    let byte = self.q[o * rb + (i >> 2)];
                    let field = ((byte >> (2 * (i & 3))) & 0x03) as i32;
                    out[i] = (field - 2) as f32 * s;
                }
            }
        }
        out
    }

    /// Dequantize to a full f32 `[O, I]` matrix — for reference/validation paths.
    pub fn dequant(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.o * self.i];
        for o in 0..self.o {
            out[o * self.i..o * self.i + self.i].copy_from_slice(&self.dequant_row(o));
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Quantize an f32 weight `[O, I]` into a [`super::QtWeight`] for tests.
    use super::QtWeight;
    use super::QuantFmt;

    pub fn quant_i8(w: &[f32], o: usize, i: usize) -> QtWeight {
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
        QtWeight::new(QuantFmt::Int8, o, i, q, sc)
    }

    /// Grouped-int4 quantizer (matches colibrì `quant_int4_grouped`): one scale
    /// per `gs`-element group along the input dim, `sc[o*ng + g]`.
    pub fn quant_i4_grouped(w: &[f32], o: usize, i: usize, gs: usize) -> QtWeight {
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
        QtWeight::new_grouped(o, i, q, sc, gs)
    }

    pub fn quant_i4(w: &[f32], o: usize, i: usize) -> QtWeight {
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
        QtWeight::new(QuantFmt::Int4, o, i, q, sc)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::QtWeight;
    use peregrine_kernels::matmul_f32;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn grouped_int4_loads_from_disk_and_matches() -> Result<(), peregrine_core::Error> {
        // Emit a grouped-int4 weight in the container format the converter
        // writes, then load it through QtWeight::load (format detection + read)
        // and confirm the loaded weight forwards identically to the in-memory one.
        use super::QuantFmt;
        use peregrine_core::pack::{f32_bytes, quant_i4_grouped, write_safetensors, Blob};
        use peregrine_core::SafeTensors;

        let (o, i, gs) = (4usize, 64usize, 16usize); // ng=4 → 16 scales > o → grouped
        let mut rng = Lcg(0x515e);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let (q, s) = quant_i4_grouped(&wf, o, i, gs);

        let dir = std::env::temp_dir().join(format!("coli_wg_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        write_safetensors(
            &dir,
            &[
                Blob::new("w", "U8", vec![o as i64, i.div_ceil(2) as i64], q.clone()),
                Blob::new("w.qs", "F32", vec![(o * (i / gs)) as i64], f32_bytes(&s)),
            ],
        )?;

        let st = SafeTensors::open(&dir)?;
        let loaded = QtWeight::load(&st, "w", o, i)?;
        assert_eq!(loaded.fmt, QuantFmt::Int4Grouped);
        assert_eq!(loaded.gs, gs);

        let mem = QtWeight::new_grouped(o, i, q, s, gs);
        let x: Vec<f32> = (0..2 * i).map(|_| rng.f()).collect();
        assert_eq!(loaded.apply_vec(&x, 2), mem.apply_vec(&x, 2), "disk-loaded grouped == in-memory");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn dequant_apply_tracks_f32() {
        let (o, i, s_n) = (5usize, 40usize, 3usize);
        let mut rng = Lcg(0x77);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();

        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            // apply() (quantized activations) vs a full-f32 matmul with the
            // dequantized weights — must agree within quant error.
            let y = w.apply_vec(&xf, s_n);
            let wdq = w.dequant();
            let mut yref = vec![0f32; s_n * o];
            matmul_f32(&mut yref, &xf, &wdq, s_n, i, o);
            for k in 0..s_n * o {
                let tol = 0.03 * i as f32;
                assert!((y[k] - yref[k]).abs() < tol, "fmt {:?} k={k} y={} ref={}", w.fmt, y[k], yref[k]);
            }
        }
    }

    #[test]
    fn apply_vec_parallel_matches_serial() {
        // apply_vec runs batched row-chunks on the pool when the matrix is large
        // enough (i·o ≥ 1<<20) and s_n ≥ PAR_MATMUL_MIN. Use 1024×1024 so the parallel
        // path actually engages, and assert it is bit-identical to one whole serial
        // `apply` for every format.
        let (o, i, s_n) = (1024usize, 1024usize, 16usize);
        let mut rng = Lcg(0x1234);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();
        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            let par = w.apply_vec(&xf, s_n);
            let mut xq = vec![0i8; s_n * i];
            let mut sx = vec![0f32; s_n];
            let mut ser = vec![0f32; s_n * o];
            w.apply(&xf, s_n, &mut xq, &mut sx, &mut ser);
            assert!(
                par.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()),
                "apply_vec parallel must match serial for {:?}",
                w.fmt
            );
        }
    }
}

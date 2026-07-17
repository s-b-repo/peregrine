//! `QtWeight` — the bridge from `peregrine-core` container formats to `peregrine-kernels`
//! matmuls. Owns one quantized weight `[O, I]` (packed bytes + per-row scales)
//! and applies it to f32 activations via the IDOT path (quantize activations
//! with `qrow_i8`, then integer-dot).
//!
//! Supports the two formats GLM-5.2 experts ship in — per-row int8 (fmt 1) and
//! packed int4 (fmt 2). Grouped-int4 (fmt 4) and int2 (fmt 3) need dedicated
//! kernels in `peregrine-kernels` and are rejected at load time until then.

use peregrine_core::{Error, QtFmt, QtInfo, SafeTensors};
use peregrine_kernels::{matmul_i4_from_f32, matmul_i8_from_f32};

/// One quantized weight matrix `[O, I]`.
pub struct QtWeight {
    pub fmt: QtFmt,
    pub o: usize,
    pub i: usize,
    /// packed weight bytes: int8 stored as u8 (reinterpreted), int4 as nibbles
    q: Vec<u8>,
    /// per-row scales (`O` entries)
    scale: Vec<f32>,
}

impl QtWeight {
    /// Build from already-quantized data (also the test constructor).
    ///
    /// Only [`QtFmt::Int8`] and [`QtFmt::Int4`] are supported today; constructing
    /// another format would make `apply`/`dequant` hit an `unreachable!` deep in
    /// the compute path, so we assert the invariant here where the misuse is.
    pub fn new(fmt: QtFmt, o: usize, i: usize, q: Vec<u8>, scale: Vec<f32>) -> QtWeight {
        debug_assert!(
            matches!(fmt, QtFmt::Int8 | QtFmt::Int4),
            "QtWeight supports Int8/Int4 only (got {fmt:?})"
        );
        QtWeight { fmt, o, i, q, scale }
    }

    /// Load a container weight `[O, I]` (`name` + `name.qs`) from a model dir.
    pub fn load(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<QtWeight, Error> {
        let info = QtInfo::detect(st, name, o as i64, i as i64);
        let nb = match info.fmt {
            QtFmt::Int8 => o * i,
            QtFmt::Int4 => o * (i.div_ceil(2)),
            other => {
                return Err(Error::Format(format!(
                    "weight '{name}': fmt {other:?} not yet supported in the Rust engine"
                )))
            }
        };
        let mut q = vec![0u8; nb];
        st.read_raw(name, &mut q)?;
        let mut scale = vec![0f32; info.scale_count as usize];
        st.read_f32(&format!("{name}.qs"), &mut scale)?;
        Ok(QtWeight { fmt: info.fmt, o, i, q, scale })
    }

    /// Raw quantized payload: `(packed_bytes, per_row_scales)`. Lets the
    /// scheduler serialize a resident expert to a disk blob and reconstruct it
    /// after streaming (`QtWeight::new`).
    pub fn raw(&self) -> (&[u8], &[f32]) {
        (&self.q, &self.scale)
    }

    /// `q` reinterpreted as int8 (only valid for [`QtFmt::Int8`]). u8↔i8 share
    /// layout and every bit pattern is valid, so the cast is sound.
    fn as_i8(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.q.as_ptr() as *const i8, self.q.len()) }
    }

    /// `y[s_n, O] = apply(self, x[s_n, I])`. Caller provides int8 activation
    /// scratch `xq[s_n*I]`, per-row scale scratch `sx[s_n]`, and output `y`.
    pub fn apply(&self, x: &[f32], s_n: usize, xq: &mut [i8], sx: &mut [f32], y: &mut [f32]) {
        match self.fmt {
            QtFmt::Int8 => matmul_i8_from_f32(y, x, self.as_i8(), &self.scale, s_n, self.i, self.o, xq, sx),
            QtFmt::Int4 => matmul_i4_from_f32(y, x, &self.q, &self.scale, s_n, self.i, self.o, xq, sx),
            _ => unreachable!("unsupported fmt reached apply (rejected at construction)"),
        }
    }

    /// Allocating convenience over [`Self::apply`].
    pub fn apply_vec(&self, x: &[f32], s_n: usize) -> Vec<f32> {
        let mut xq = vec![0i8; s_n * self.i];
        let mut sx = vec![0f32; s_n];
        let mut y = vec![0f32; s_n * self.o];
        self.apply(x, s_n, &mut xq, &mut sx, &mut y);
        y
    }

    /// Dequantize a single output row `o` to f32 `[I]` — used by the MLA
    /// absorption path (`qt_addrow` / `qt_matvec_rows`), which reads individual
    /// `kv_b` rows rather than a batched matmul.
    pub fn dequant_row(&self, o: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.i];
        let s = self.scale[o];
        match self.fmt {
            QtFmt::Int8 => {
                let q = self.as_i8();
                for i in 0..self.i {
                    out[i] = q[o * self.i + i] as f32 * s;
                }
            }
            QtFmt::Int4 => {
                let rb = self.i.div_ceil(2);
                for i in 0..self.i {
                    let byte = self.q[o * rb + (i >> 1)];
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    out[i] = (nib - 8) as f32 * s;
                }
            }
            _ => unreachable!(),
        }
        out
    }

    /// Dequantize to a full f32 `[O, I]` matrix — for reference/validation paths.
    pub fn dequant(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.o * self.i];
        match self.fmt {
            QtFmt::Int8 => {
                let q = self.as_i8();
                for o in 0..self.o {
                    let s = self.scale[o];
                    for i in 0..self.i {
                        out[o * self.i + i] = q[o * self.i + i] as f32 * s;
                    }
                }
            }
            QtFmt::Int4 => {
                let rb = self.i.div_ceil(2);
                for o in 0..self.o {
                    let s = self.scale[o];
                    for i in 0..self.i {
                        let byte = self.q[o * rb + (i >> 1)];
                        let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                        out[o * self.i + i] = (nib - 8) as f32 * s;
                    }
                }
            }
            _ => unreachable!(),
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Quantize an f32 weight `[O, I]` into a [`super::QtWeight`] for tests.
    use super::QtWeight;
    use peregrine_core::QtFmt;

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
        QtWeight::new(QtFmt::Int8, o, i, q, sc)
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
        QtWeight::new(QtFmt::Int4, o, i, q, sc)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use peregrine_kernels::matmul_f32;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn dequant_apply_tracks_f32() {
        let (o, i, s_n) = (5usize, 40usize, 3usize);
        let mut rng = Lcg(0x77);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();

        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i)] {
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
}

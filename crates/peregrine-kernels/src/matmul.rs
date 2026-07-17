//! Quantized matmuls — port of `matmul_q_idot` / `matmul_i4_idot`
//! (`c/glm.c:933,942`), the IDOT decode path.
//!
//! Output `y[s*O + o] = dot(w_o, xq_s) as f32 * scale[o] * sx[s]`, with the
//! multiply evaluated left-to-right exactly as in C so the f32 result matches
//! bit-for-bit. Weights are pre-quantized (per-row `scale`); activations are
//! pre-quantized to int8 with per-row `sx` (see [`crate::quant::qrow_i8`]).
//!
//! Single-threaded and deterministic here — the CPU-lane threading arrives with
//! the M4 scheduler; correctness and reproducibility come first.

use crate::idot::{dot_i4i8, dot_i8i8};
use crate::quant::qrow_i8;

/// Plain f32 matmul `y[S,O] = x[S,D] · wᵀ` where weights `w` are row-major
/// `[O, D]` (one row per output). Port of `matmul` (`c/glm.c:331`), used for the
/// small full-precision tensors the engine keeps in f32 (the MoE router, norms).
pub fn matmul_f32(y: &mut [f32], x: &[f32], w: &[f32], s_n: usize, d_n: usize, o_n: usize) {
    for s in 0..s_n {
        let xs = &x[s * d_n..s * d_n + d_n];
        for o in 0..o_n {
            let wo = &w[o * d_n..o * d_n + d_n];
            let mut acc = 0f32;
            for d in 0..d_n {
                acc += xs[d] * wo[d];
            }
            y[s * o_n + o] = acc;
        }
    }
}

/// int8 weights `q[O*I]`, int8 activations `xq[S*I]` → `y[S*O]`.
#[allow(clippy::too_many_arguments)] // matmul shapes are inherently wide
pub fn matmul_q_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q: &[i8], scale: &[f32], s_n: usize, i_n: usize, o_n: usize) {
    for o in 0..o_n {
        let w = &q[o * i_n..o * i_n + i_n];
        let sc = scale[o];
        for s in 0..s_n {
            let xs = &xq[s * i_n..s * i_n + i_n];
            let d = dot_i8i8(w, xs, i_n) as f32;
            y[s * o_n + o] = d * sc * sx[s];
        }
    }
}

/// packed-int4 weights `q4[O*ceil(I/2)]`, int8 activations `xq[S*I]` → `y[S*O]`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i4_idot(y: &mut [f32], xq: &[i8], sx: &[f32], q4: &[u8], scale: &[f32], s_n: usize, i_n: usize, o_n: usize) {
    let rb = i_n.div_ceil(2);
    for o in 0..o_n {
        let w = &q4[o * rb..o * rb + rb];
        let sc = scale[o];
        for s in 0..s_n {
            let xs = &xq[s * i_n..s * i_n + i_n];
            let d = dot_i4i8(w, xs, i_n) as f32;
            y[s * o_n + o] = d * sc * sx[s];
        }
    }
}

/// Convenience: quantize f32 activations row-wise (`qrow_i8`) into `xq`/`sx`
/// scratch, then run [`matmul_q_idot`]. `xq` is `S*I`, `sx` is `S`.
#[allow(clippy::too_many_arguments)] // matmul shape (dims + scratch) is inherently wide
pub fn matmul_i8_from_f32(
    y: &mut [f32],
    x: &[f32],
    q: &[i8],
    scale: &[f32],
    s_n: usize,
    i_n: usize,
    o_n: usize,
    xq: &mut [i8],
    sx: &mut [f32],
) {
    for s in 0..s_n {
        sx[s] = qrow_i8(&x[s * i_n..s * i_n + i_n], &mut xq[s * i_n..s * i_n + i_n]);
    }
    matmul_q_idot(y, xq, sx, q, scale, s_n, i_n, o_n);
}

/// Convenience: same as [`matmul_i8_from_f32`] for packed-int4 weights.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i4_from_f32(
    y: &mut [f32],
    x: &[f32],
    q4: &[u8],
    scale: &[f32],
    s_n: usize,
    i_n: usize,
    o_n: usize,
    xq: &mut [i8],
    sx: &mut [f32],
) {
    for s in 0..s_n {
        sx[s] = qrow_i8(&x[s * i_n..s * i_n + i_n], &mut xq[s * i_n..s * i_n + i_n]);
    }
    matmul_i4_idot(y, xq, sx, q4, scale, s_n, i_n, o_n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idot::{dot_i4i8_scalar, dot_i8i8_scalar};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 // [-1,1)
        }
    }

    /// Per-row int8 weight quantizer (test helper): scale = amax/127.
    fn quant_i8_rows(w: &[f32], o_n: usize, i_n: usize) -> (Vec<i8>, Vec<f32>) {
        let mut q = vec![0i8; o_n * i_n];
        let mut sc = vec![0f32; o_n];
        for o in 0..o_n {
            let row = &w[o * i_n..o * i_n + i_n];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = (amax / 127.0).max(1e-12);
            sc[o] = s;
            for i in 0..i_n {
                q[o * i_n + i] = (row[i] / s).round_ties_even() as i8;
            }
        }
        (q, sc)
    }

    /// Per-row int4 weight quantizer (test helper): scale = amax/7, nibbles [-8,7].
    fn quant_i4_rows(w: &[f32], o_n: usize, i_n: usize) -> (Vec<u8>, Vec<f32>) {
        let rb = i_n.div_ceil(2);
        let mut q = vec![0u8; o_n * rb];
        let mut sc = vec![0f32; o_n];
        for o in 0..o_n {
            let row = &w[o * i_n..o * i_n + i_n];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = (amax / 7.0).max(1e-12);
            sc[o] = s;
            for i in 0..i_n {
                let v = (row[i] / s).round_ties_even().clamp(-8.0, 7.0) as i32;
                let bias = (v + 8) as u8 & 0x0F;
                if i & 1 == 0 {
                    q[o * rb + (i >> 1)] |= bias;
                } else {
                    q[o * rb + (i >> 1)] |= bias << 4;
                }
            }
        }
        (q, sc)
    }

    #[test]
    fn matmul_matches_scalar_dot_defn() {
        // The dispatched matmul must equal the by-definition scalar computation.
        let (s_n, i_n, o_n) = (3usize, 70usize, 5usize);
        let mut rng = Lcg(0xabcd);
        let xf: Vec<f32> = (0..s_n * i_n).map(|_| rng.f()).collect();
        let wf: Vec<f32> = (0..o_n * i_n).map(|_| rng.f()).collect();
        let (q8, sc8) = quant_i8_rows(&wf, o_n, i_n);

        let mut xq = vec![0i8; s_n * i_n];
        let mut sx = vec![0f32; s_n];
        let mut y = vec![0f32; s_n * o_n];
        matmul_i8_from_f32(&mut y, &xf, &q8, &sc8, s_n, i_n, o_n, &mut xq, &mut sx);

        for s in 0..s_n {
            for o in 0..o_n {
                let d = dot_i8i8_scalar(&q8[o * i_n..o * i_n + i_n], &xq[s * i_n..s * i_n + i_n], i_n) as f32;
                let expect = d * sc8[o] * sx[s];
                assert_eq!(y[s * o_n + o], expect, "int8 s={s} o={o}");
            }
        }
    }

    #[test]
    fn i4_matmul_matches_scalar_dot_defn() {
        let (s_n, i_n, o_n) = (2usize, 96usize, 4usize);
        let mut rng = Lcg(0x0f0f);
        let xf: Vec<f32> = (0..s_n * i_n).map(|_| rng.f()).collect();
        let wf: Vec<f32> = (0..o_n * i_n).map(|_| rng.f()).collect();
        let (q4, sc4) = quant_i4_rows(&wf, o_n, i_n);
        let rb = i_n.div_ceil(2);

        let mut xq = vec![0i8; s_n * i_n];
        let mut sx = vec![0f32; s_n];
        let mut y = vec![0f32; s_n * o_n];
        matmul_i4_from_f32(&mut y, &xf, &q4, &sc4, s_n, i_n, o_n, &mut xq, &mut sx);

        for s in 0..s_n {
            for o in 0..o_n {
                let d = dot_i4i8_scalar(&q4[o * rb..o * rb + rb], &xq[s * i_n..s * i_n + i_n], i_n) as f32;
                let expect = d * sc4[o] * sx[s];
                assert_eq!(y[s * o_n + o], expect, "int4 s={s} o={o}");
            }
        }
    }

    #[test]
    fn int8_matmul_approximates_f32() {
        // End-to-end: quantized matmul should track the full-f32 matmul within
        // quant error. Guards against index/layout mistakes the exact tests miss.
        let (s_n, i_n, o_n) = (4usize, 128usize, 6usize);
        let mut rng = Lcg(0x5151);
        let xf: Vec<f32> = (0..s_n * i_n).map(|_| rng.f()).collect();
        let wf: Vec<f32> = (0..o_n * i_n).map(|_| rng.f()).collect();
        let (q8, sc8) = quant_i8_rows(&wf, o_n, i_n);

        let mut xq = vec![0i8; s_n * i_n];
        let mut sx = vec![0f32; s_n];
        let mut y = vec![0f32; s_n * o_n];
        matmul_i8_from_f32(&mut y, &xf, &q8, &sc8, s_n, i_n, o_n, &mut xq, &mut sx);

        for s in 0..s_n {
            for o in 0..o_n {
                let mut r = 0f32;
                for i in 0..i_n {
                    r += wf[o * i_n + i] * xf[s * i_n + i];
                }
                let got = y[s * o_n + o];
                let tol = 0.02 * i_n as f32; // ~2% per-term over the accumulation
                assert!((got - r).abs() < tol, "s={s} o={o} got={got} ref={r}");
            }
        }
    }
}

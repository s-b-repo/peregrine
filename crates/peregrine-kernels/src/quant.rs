//! Activation quantization — port of `qrow_i8` (`c/glm.c:581`).
//!
//! The IDOT path quantizes each activation row to int8 with a single per-row
//! scale, then does an integer dot against the int8/int4 weights. Reproduced
//! exactly (including the `1e-12` scale floor and round-ties-to-even) so the
//! Rust decode path is bit-identical to the C engine's on the S=1 hot path.

/// Quantize one activation row `x[0..I]` into `q[0..I]` (int8), returning the
/// per-row scale `s` such that `x[i] ≈ q[i] * s`.
///
/// Mirrors `qrow_i8`: `s = amax/127` (floored at 1e-12), `q[i] = round(x[i]/s)`
/// with round-half-to-even (the default `lrintf` rounding mode).
#[inline]
pub fn qrow_i8(x: &[f32], q: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), q.len());
    let mut amax = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let mut s = amax / 127.0;
    if s < 1e-12 {
        s = 1e-12;
    }
    let inv = 1.0 / s;
    for (qi, &xi) in q.iter_mut().zip(x) {
        // lrintf: round-half-to-even. |x*inv| <= 127 by construction.
        *qi = (xi * inv).round_ties_even() as i8;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dequant_roundtrip() {
        let x = [0.5f32, -1.25, 3.0, -0.001, 2.7, -3.0];
        let mut q = [0i8; 6];
        let s = qrow_i8(&x, &mut q);
        // amax = 3.0 → s = 3/127; the ±3.0 entries map to ±127.
        assert_eq!(q[2], 127);
        assert_eq!(q[5], -127);
        for (xi, qi) in x.iter().zip(q) {
            assert!((xi - qi as f32 * s).abs() <= s); // within one quant step
        }
    }

    #[test]
    fn all_zero_row() {
        let x = [0.0f32; 8];
        let mut q = [1i8; 8];
        let s = qrow_i8(&x, &mut q);
        assert_eq!(s, 1e-12);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn ties_to_even() {
        // 1.5 and 2.5 both round to 2 (ties-to-even) when the scale is 1.0.
        // Build x so inv == 1: amax = 127 → s = 1.0.
        let mut x = [0.0f32; 4];
        x[0] = 127.0;
        x[1] = 1.5;
        x[2] = 2.5;
        x[3] = -1.5;
        let mut q = [0i8; 4];
        let s = qrow_i8(&x, &mut q);
        assert_eq!(s, 1.0);
        assert_eq!(q[1], 2); // 1.5 → 2
        assert_eq!(q[2], 2); // 2.5 → 2
        assert_eq!(q[3], -2); // -1.5 → -2
    }
}

//! The token-exactness group: SIMD against scalar, packed-row producer against
//! packed-row consumer, and parallel against serial.
//!
//! This is the group that matters most. Most identities in the suite fail
//! loudly enough to notice; a SIMD integer-dot that disagrees with its scalar
//! reference by one unit of accumulation does not — it emits different tokens
//! and nothing anywhere reports an error.

use peregrine_core::pack::{quant_i2, quant_i2_g64, quant_i3_g64, quant_i4};
use peregrine_kernels::{
    dot_i2i8, dot_i2i8_g64, dot_i2i8_g64_scalar, dot_i2i8_scalar, dot_i3i8_g64, dot_i3i8_g64_scalar, dot_i4i8, dot_i4i8_grouped,
    dot_i4i8_grouped_scalar, dot_i4i8_scalar, dot_i8i8, dot_i8i8_scalar, matmul_f32, matmul_q_idot, MatShape,
};
use peregrine_par::par_rows_mut;

use crate::{cert, mismatch_bits, rng, with_cases, Certificate, EPS_F32};

/// Lengths chosen to straddle every vector boundary the kernels have: a
/// remainder of 1 in each of the 8-lane (AVX2) and 16-lane (AVX-VNNI) groupings,
/// a full 16, and an odd tail for the packed formats.
const LENGTHS: [usize; 8] = [1, 2, 3, 7, 8, 15, 16, 17];

pub fn certificates() -> Vec<Certificate> {
    vec![
        dot_i8i8_cert(),
        dot_i4i8_cert(),
        dot_i2i8_cert(),
        dot_i4i8_grouped_cert(),
        dot_i2i8_g64_cert(),
        dot_i3i8_g64_cert(),
        int4_packed_row_cert(),
        int2_packed_row_cert(),
        int2g64_affine_row_cert(),
        int3g64_row_cert(),
        matmul_q_idot_layout_cert(),
        parallel_matmul_cert(),
    ]
}

/// `dot_i8i8` is the AVX2/AVX-VNNI path; `dot_i8i8_scalar` is the reference the
/// C engine's `dot_i8i8` is transcribed from. The claim is *bit* identity, not
/// closeness: an integer dot has no tolerance to argue about, and a one-unit
/// difference is a different token.
fn dot_i8i8_cert() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in LENGTHS {
        for _ in 0..8 {
            let w = r.i8_vec(n);
            let x = r.i8_vec(n);
            if dot_i8i8(&w, &x, n) != dot_i8i8_scalar(&w, &x, n) {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "dot_i8i8_is_bit_identical_to_scalar",
            "Every SIMD int8 dot equals the scalar reference exactly, for every length that straddles the 8- and 16-lane vector boundaries.",
            bad,
            0.0,
        ),
        cases,
    )
}

fn dot_i4i8_cert() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in LENGTHS {
        for _ in 0..8 {
            // Nibbles are read as `nibble - 8`, so the packed bytes must be
            // biased on the producer side or the two sides would not agree by
            // construction and the check would be measuring the fixture.
            let w = bias8_row(&mut r, n);
            let x = r.i8_vec(n);
            if dot_i4i8(&w, &x, n) != dot_i4i8_scalar(&w, &x, n) {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "dot_i4i8_is_bit_identical_to_scalar",
            "Every SIMD packed-int4 dot equals the scalar reference exactly; the bias-8 decode and the odd-nibble tail agree on both sides.",
            bad,
            0.0,
        ),
        cases,
    )
}

fn dot_i2i8_cert() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in LENGTHS {
        for _ in 0..8 {
            let w = bias2_row(&mut r, n);
            let x = r.i8_vec(n);
            if dot_i2i8(&w, &x, n) != dot_i2i8_scalar(&w, &x, n) {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "dot_i2i8_is_bit_identical_to_scalar",
            "Every SIMD packed-int2 dot (four 2-bit fields per byte, biased +2) equals the scalar reference exactly.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// A packed int4 row of `n` values, biased the way a producer must bias it:
/// `q = value + 8` in `[0, 15]`, low nibble first. Returned as bytes so the SIMD
/// and scalar readers consume the same buffer.
fn bias8_row(r: &mut crate::Rng, n: usize) -> Vec<u8> {
    let rb = n.div_ceil(2);
    let mut out = vec![0u8; rb];
    for k in 0..n {
        let v = ((r.sym() * 8.0).round() as i32).clamp(-8, 7);
        let nib = ((v + 8) as u8) & 0x0F;
        if k & 1 == 0 {
            out[k >> 1] |= nib;
        } else {
            out[k >> 1] |= nib << 4;
        }
    }
    out
}

/// A packed int2 row of `n` values, biased `+2` into `[0, 3]`, low field first.
fn bias2_row(r: &mut crate::Rng, n: usize) -> Vec<u8> {
    let rb = n.div_ceil(4);
    let mut out = vec![0u8; rb];
    for k in 0..n {
        let v = ((r.sym() * 2.0).round() as i32).clamp(-2, 1);
        let field = ((v + 2) as u8) & 0x03;
        out[k >> 2] |= field << (2 * (k & 3));
    }
    out
}

/// The grouped variants accumulate in f32, so "bit-identical" is a statement
/// about `f32::to_bits`, not about `==` — a negative zero or a NaN payload would
/// satisfy `==` and still be a different answer.
fn dot_i4i8_grouped_cert() -> Certificate {
    let mut r = rng();
    let (i_n, gs) = (64usize, 32usize);
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for _ in 0..16 {
        let w = bias8_row(&mut r, i_n);
        let x = r.i8_vec(i_n);
        let sc: Vec<f32> = (0..i_n.div_ceil(gs)).map(|_| r.sym().abs() + 0.5).collect();
        let simd = dot_i4i8_grouped(&w, &x, &sc, i_n, gs);
        let scalar = dot_i4i8_grouped_scalar(&w, &x, &sc, i_n, gs);
        if simd.to_bits() != scalar.to_bits() {
            bad += 1.0;
        }
        cases += 1;
    }
    with_cases(
        cert(
            "kernels",
            "dot_i4i8_grouped_is_bit_identical_to_scalar",
            "Every SIMD grouped-int4 dot equals the scalar reference bit-for-bit (f32 `to_bits`), so per-group scale folding and the ragged final group agree on both sides.",
            bad,
            0.0,
        ),
        cases,
    )
}

fn dot_i2i8_g64_cert() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    // 64 = exactly one group, 128 = two, 65 = a ragged tail.
    for n in [64usize, 128, 65] {
        for _ in 0..8 {
            let w = bias2_row(&mut r, n);
            let x = r.i8_vec(n);
            let sc: Vec<f32> = (0..2 * n.div_ceil(64)).map(|_| r.sym().abs() + 0.5).collect();
            let simd = dot_i2i8_g64(&w, &x, &sc, n);
            let scalar = dot_i2i8_g64_scalar(&w, &x, &sc, n);
            if simd.to_bits() != scalar.to_bits() {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "dot_i2i8_g64_is_bit_identical_to_scalar",
            "Every SIMD affine int2-g64 dot equals the scalar reference bit-for-bit, including the interleaved per-group scale/zero-point pairs and the `n % 64` tail.",
            bad,
            0.0,
        ),
        cases,
    )
}

fn dot_i3i8_g64_cert() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in [64usize, 128, 65] {
        for _ in 0..8 {
            // One 24-byte group per 64 values: a 16-byte low plane (2 bits per
            // value) plus an 8-byte high plane (1 bit per value).
            let w = r.u8_vec(n.div_ceil(64) * 24);
            let x = r.i8_vec(n);
            let sc: Vec<f32> = (0..n.div_ceil(64)).map(|_| r.sym().abs() + 0.5).collect();
            let simd = dot_i3i8_g64(&w, &x, &sc, n);
            let scalar = dot_i3i8_g64_scalar(&w, &x, &sc, n);
            if simd.to_bits() != scalar.to_bits() {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "dot_i3i8_g64_is_bit_identical_to_scalar",
            "Every SIMD int3-g64 dot equals the scalar reference bit-for-bit, including the two-plane (low 2 bits, high 1 bit) layout and the biased +4 decode.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The int4 bias convention, end to end, on values the format represents
/// **exactly**. `quant_i4` divides a row by `amax / 7`, so a row of small
/// integers whose maximum is 7 gets `scale == 1.0`, every element lands on the
/// grid with no rounding at all, and the dot of the packed row against a vector
/// of ones must return the row's own integer sum.
///
/// This is the certificate that catches the class of bug documented in
/// `docs/BAD_PATTERNS.md` `[X1]`: host int4 is bias-8 while the CUDA upload ran
/// `offset_to_signed_s4` and produced two's complement. Both conventions map
/// `0..15` onto `-8..7` by different permutations, so a mismatch yields
/// plausible garbage rather than an error — and a test built on random weights
/// would still pass, because the defect is in the *encoding*, not the arithmetic.
fn int4_packed_row_cert() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (o, i) in [(1usize, 8usize), (2, 17), (3, 64)] {
        let mut w = vec![0f32; o * i];
        for (row, wr) in w.chunks_mut(i).enumerate() {
            for (col, slot) in wr.iter_mut().enumerate() {
                // Levels `[-7, 6]` — every level the format can hold *and* encode
                // at scale 1.0. The divisor is the largest positive level (`7`),
                // not the largest magnitude (`8`), so a row containing `-8` would
                // give `scale = 8/7` and the row would not land on the grid at
                // all. That asymmetry is the format's, and pinning `amax` to
                // exactly `7.0` is what makes this an exact identity.
                *slot = (((row * 7 + col) % 14) as f32) - 7.0;
            }
            wr[0] = 7.0;
        }
        let (q4, scale) = quant_i4(&w, o, i);
        let ones = vec![1i8; i];
        let rb = i.div_ceil(2);
        for row in 0..o {
            let got = dot_i4i8(&q4[row * rb..row * rb + rb], &ones, i) as f32 * scale[row];
            let expect: f32 = w[row * i..row * i + i].iter().sum();
            if got.to_bits() != expect.to_bits() {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "int4_packed_row_dots_to_its_own_sum",
            "A row quantized by `quant_i4` onto the integer grid (scale 1.0) dots against a vector of ones to exactly the row's own sum: producer bias and consumer decode are one convention.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The same identity for per-row int2, whose bias is `+2` into `[0,3]` and whose
/// scale divides by 1 (the largest positive level), so a row bounded by 1.0
/// lands on the grid exactly.
fn int2_packed_row_cert() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (o, i) in [(1usize, 12usize), (2, 17)] {
        let mut w = vec![0f32; o * i];
        for (row, wr) in w.chunks_mut(i).enumerate() {
            for (col, slot) in wr.iter_mut().enumerate() {
                // Levels `{-1, 0, 1}`: `quant_i2`'s scale divides by the largest
                // positive level (1), which makes `-2` unreachable at any scale
                // the row's own maximum allows — the format is effectively
                // ternary, and documented as such at `quant_i2`.
                *slot = (((row + col) % 3) as f32) - 1.0;
            }
        }
        let (q2, scale) = quant_i2(&w, o, i);
        let ones = vec![1i8; i];
        let rb = i.div_ceil(4);
        for row in 0..o {
            let got = dot_i2i8(&q2[row * rb..row * rb + rb], &ones, i) as f32 * scale[row];
            let expect: f32 = w[row * i..row * i + i].iter().sum();
            if got.to_bits() != expect.to_bits() {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "int2_packed_row_dots_to_its_own_sum",
            "A row quantized by `quant_i2` onto the levels a single scale can represent dots against a vector of ones to exactly the row's integer sum.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The affine int2-g64 convention: `s = (hi - lo) / 3`, `z = lo`, and the dot
/// subtracts `z · Σx` from the field sum. With activations of one, the result
/// must be the row's own sum when the group endpoints are exactly
/// representable. Unlike the biased formats this one carries a zero-point, so
/// it is the case where an off-by-one in the `[s, z]` interleaving would be
/// invisible to the per-row int2 certificate above.
fn int2g64_affine_row_cert() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let (o, i) = (2usize, 128usize);
    let mut w = vec![0f32; o * i];
    for (row, wr) in w.chunks_mut(i).enumerate() {
        for (col, slot) in wr.iter_mut().enumerate() {
            // Exactly representable endpoints (`lo = -1`, `hi = 2`, both exact),
            // and the values are the four levels those endpoints imply.
            let u = ((col + row) % 4) as f32;
            *slot = -1.0 + u;
        }
    }
    let (q, scale) = quant_i2_g64(&w, o, i);
    let ones = vec![1i8; i];
    let ng = i.div_ceil(64);
    for row in 0..o {
        let bytes = &q[row * ng * 16..(row + 1) * ng * 16];
        // Two f32 per group, interleaved `[s, z]`, `ng` groups per row.
        let sc = &scale[row * 2 * ng..(row + 1) * 2 * ng];
        let got = dot_i2i8_g64(bytes, &ones, sc, i);
        let expect: f32 = w[row * i..row * i + i].iter().sum();
        // One f32 add and two f32 multiplies per group (64 values), so the
        // accumulated error is bounded by `2n` ulps of the running sum. The
        // bound is the arithmetic's, not a fit to the observed value.
        let tol = (2.0 * i as f64) * EPS_F32 * (1.0 + expect.abs() as f64);
        if (got as f64 - expect as f64).abs() > tol {
            bad += 1.0;
        }
        cases += 1;
    }
    with_cases(
        cert(
            "kernels",
            "int2_g64_affine_row_dots_to_its_own_sum",
            "An affine int2-g64 row over exactly representable group endpoints dots against a vector of ones to its own sum, within `2n` f32 ulps of the accumulator.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The int3-g64 encode/decode pair on a grid of exactly representable levels:
/// the decoder's `+4` bias spans `[-4, 3]`, so a row whose maximum is `3.0`
/// gets `scale == 1.0` and every value survives encode and decode unchanged.
fn int3g64_row_cert() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (o, i) in [(1usize, 64usize), (2, 129)] {
        let mut w = vec![0f32; o * i];
        for (row, wr) in w.chunks_mut(i).enumerate() {
            for (col, slot) in wr.iter_mut().enumerate() {
                *slot = (((col + row) % 8) as f32) - 4.0;
            }
        }
        let (q, scale) = quant_i3_g64(&w, o, i);
        let ones = vec![1i8; i];
        let ng = i.div_ceil(64);
        for row in 0..o {
            let got = dot_i3i8_g64(&q[row * ng * 24..(row + 1) * ng * 24], &ones, &scale[row * ng..(row + 1) * ng], i);
            let expect: f32 = w[row * i..row * i + i].iter().sum();
            let tol = 2.0 * ng as f64 * EPS_F32 * (1.0 + expect.abs() as f64);
            if (got as f64 - expect as f64).abs() > tol {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "kernels",
            "int3_g64_row_dots_to_its_own_sum",
            "An int3-g64 row on exactly representable levels dots against a vector of ones to its own sum, within `2·groups` f32 ulps.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// `matmul_q_idot` claims that output `(s, o)` is
/// `dot_i8i8(q[o..], xq[s..], i) * scale[o] * sx[s]`, with the two multiplies
/// applied left to right. Checked against the *kernel* rather than a
/// hand-written loop, so the arithmetic under test is the production arithmetic
/// and only the layout is the claim: the expert row and position offsets are
/// read where the layout says they are. Powers-of-two scales keep every product
/// exact, so the comparison can be bit-for-bit.
fn matmul_q_idot_layout_cert() -> Certificate {
    let (s_n, i_n, o_n) = (5usize, 33usize, 7usize);
    let mut r = rng();
    let q: Vec<i8> = r.i8_vec(o_n * i_n);
    let xq: Vec<i8> = r.i8_vec(s_n * i_n);
    let scale: Vec<f32> = (0..o_n).map(|o| (1 << (o % 5)) as f32).collect();
    let sx: Vec<f32> = (0..s_n).map(|s| (1 << (s % 4)) as f32).collect();

    let mut expected = Vec::with_capacity(s_n * o_n);
    for s in 0..s_n {
        for o in 0..o_n {
            let d = dot_i8i8(&q[o * i_n..o * i_n + i_n], &xq[s * i_n..s * i_n + i_n], i_n) as f32;
            expected.push(d * scale[o] * sx[s]);
        }
    }
    let mut got = vec![0f32; s_n * o_n];
    matmul_q_idot(&mut got, &xq, &sx, &q, &scale, MatShape::new(s_n, i_n, o_n));

    with_cases(
        cert(
            "kernels",
            "matmul_q_idot_is_bit_exact_on_exact_scales",
            "With power-of-two scales and small integer operands, `matmul_q_idot` is `to_bits`-identical to `dot_i8i8` of the addressed row and position, scaled left to right.",
            mismatch_bits(&expected, &got),
            0.0,
        ),
        s_n * o_n,
    )
}

/// Parallel == serial, exactly. `peregrine-par` exists to run the same
/// per-output loop on many workers, so the only acceptable relationship to the
/// serial reference is bit-identity — and it is achievable because each output
/// element keeps its own left-to-right accumulation and no partial sum is shared
/// between workers. A change that batched rows into one reduction would break
/// this while every throughput benchmark kept improving.
fn parallel_matmul_cert() -> Certificate {
    let (s_n, i_n, o_n) = (64usize, 128usize, 256usize);
    let mut r = rng();
    let x = r.f32_vec(s_n * i_n);
    let w = r.f32_vec(o_n * i_n);

    let mut serial = vec![0f32; s_n * o_n];
    matmul_f32(&mut serial, &x, &w, s_n, i_n, o_n);

    // The parallel arm must run the identical inner loop; only the partitioning
    // differs, which is the whole content of the claim.
    let mut parallel = vec![0f32; s_n * o_n];
    par_rows_mut(&mut parallel, o_n, s_n, 2, |s, row| {
        let xs = &x[s * i_n..s * i_n + i_n];
        for o in 0..o_n {
            let wo = &w[o * i_n..o * i_n + i_n];
            let mut acc = 0f32;
            for d in 0..i_n {
                acc += xs[d] * wo[d];
            }
            row[o] = acc;
        }
    });

    with_cases(
        cert(
            "kernels",
            "parallel_matmul_is_bit_identical_to_serial",
            "A row-parallel matmul through `peregrine_par::par_rows_mut` is `f32::to_bits`-identical to the serial `matmul_f32` for every output element: worker partitioning cannot change an accumulation.",
            mismatch_bits(&serial, &parallel),
            0.0,
        ),
        s_n * o_n,
    )
}
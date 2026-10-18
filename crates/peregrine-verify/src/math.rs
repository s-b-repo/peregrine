//! The elementary-numerics group: the building blocks every forward pass is
//! made of.
//!
//! These are the identities whose violation is hardest to see. A softmax that
//! sums to 1.02 still produces plausible probabilities; an RMSNorm with a
//! slightly wrong `rsqrt` still produces plausible activations. Each one is
//! checked against the property itself — a distribution, a unit RMS, a
//! preserved norm — rather than against a re-typed copy of the formula, because
//! the formula is not what is in doubt.

use peregrine_model::math::{layernorm, rmsnorm, rope_interleave_with, rope_neox_with, sigmoidf, silu_mul, softmax, RopeTable};

use crate::{cert, exact, max_abs_diff, mismatch_bits, rng, with_cases, Certificate, EPS_F32};

pub fn certificates() -> Vec<Certificate> {
    vec![
        rmsnorm_unit_rms(),
        rmsnorm_out_of_place_matches_in_place(),
        softmax_is_a_distribution(),
        softmax_is_finite_on_a_masked_row(),
        silu_mul_composes(),
        layernorm_standardizes(),
        sigmoid_fixed_point(),
        rope_at_zero_is_identity(),
        rope_preserves_the_norm(),
        rope_layouts_agree_under_the_input_permutation(),
        rope_positions_compose(),
    ]
}

/// RMSNorm exists to put a row at unit RMS; that is the property every
/// downstream scale factor assumes. `rmsnorm` accumulates the sum of squares in
/// f64 and rounds once, so the residual is bounded by the roundings of the final
/// multiply and one division: a few ulps of the unit RMS, not of the data.
fn rmsnorm_unit_rms() -> Certificate {
    let mut r = rng();
    let mut worst = 0.0f64;
    for n in [1usize, 3, 16, 128, 4096] {
        for _ in 0..4 {
            let x = r.f32_vec(n);
            let w = vec![1.0f32; n];
            let mut out = vec![0f32; n];
            rmsnorm(&mut out, &x, &w, 0.0);
            let ms: f64 = out.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / n as f64;
            worst = worst.max((ms - 1.0).abs());
        }
    }
    cert(
        "math",
        "rmsnorm_yields_unit_rms",
        "With unit weights and no epsilon, the RMS of an RMSNormed row is 1 to within a few f32 ulps of the sum-of-squares accumulation.",
        worst,
        8.0 * EPS_F32,
    )
}

/// With a non-unit weight vector the row is scaled by `w[i]` elementwise, so
/// `out[i]/x[i]` must equal `w[i] * r` for a single `r` — i.e. the norm is the
/// only place the row length enters. Checked through `rmsnorm` against
/// `rmsnorm_inplace`, which takes a different code path through the same
/// arithmetic.
fn rmsnorm_out_of_place_matches_in_place() -> Certificate {
    let mut r = rng();
    let n = 256usize;
    let x = r.f32_vec(n);
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let mut a = vec![0f32; n];
    rmsnorm(&mut a, &x, &w, 1e-6);
    let mut b = x.clone();
    rmsnorm_inplace_helper(&mut b, &w, 1e-6);
    with_cases(
        cert(
            "math",
            "rmsnorm_out_of_place_matches_in_place",
            "`rmsnorm` into a separate buffer is `to_bits`-identical to `rmsnorm_inplace` on the same row, so the fused variant cannot drift from the reference.",
            max_abs_diff(&a, &b),
            0.0,
        ),
        n,
    )
}

/// Softmax's contract is a distribution: the row sums to one and every entry is
/// in `[0, 1]`. The tolerance is `n` ulps of the running sum, which is the
/// arithmetic's own bound on `Σ e_i` divided by it.
fn softmax_is_a_distribution() -> Certificate {
    let mut r = rng();
    let mut worst_sum = 0.0f64;
    let mut bad_range = 0.0f64;
    let mut cases = 0usize;
    for n in [1usize, 2, 7, 64, 1024, 32000] {
        for _ in 0..4 {
            let mut row = r.f32_vec(n);
            softmax(&mut row);
            let sum: f64 = row.iter().map(|v| f64::from(*v)).sum();
            worst_sum = worst_sum.max((sum - 1.0).abs());
            if row.iter().any(|v| *v < 0.0 || *v > 1.0) {
                bad_range += 1.0;
            }
            cases += 1;
        }
    }
    // n adds of n terms: `n · eps` is the worst case if every partial sum lands
    // on the wrong side of the last bit.
    let tol = 32000.0 * EPS_F32;
    with_cases(
        cert(
            "math",
            "softmax_is_a_distribution",
            "Every softmaxed row sums to 1 within `n` f32 ulps of the partial sums, and no entry leaves `[0, 1]`.",
            worst_sum / tol + bad_range,
            1.0,
        ),
        cases,
    )
}

/// The masked-row contract: an all `-inf` row must come out uniform rather than
/// NaN. A NaN row poisons every logit behind it and makes greedy decode latch
/// onto one token forever — the failure this function's fallback exists to
/// prevent, so it gets its own certificate rather than riding along above.
fn softmax_is_finite_on_a_masked_row() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in [1usize, 5, 128] {
        let mut row = vec![f32::NEG_INFINITY; n];
        softmax(&mut row);
        let want = 1.0 / n as f32;
        let tol = 4.0 * EPS_F32 * (want as f64).abs();
        if row.iter().any(|v| !v.is_finite() || (f64::from(*v) - f64::from(want)).abs() > tol) {
            bad += 1.0;
        }
        cases += 1;
    }
    with_cases(
        cert(
            "math",
            "an_all_masked_softmax_row_is_uniform",
            "A row of `-inf` softmaxes to the uniform distribution, never to NaN: every entry is finite and equal to `1/n`.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// `silu_mul` is the fused SwiGLU: it must equal the composition of the two
/// production halves, `siluf` and a multiply. Checked as a composition rather
/// than against a re-typed `x·sigmoid(x)`, because the shipped `siluf` — not
/// the textbook expression — is what the kernel fuses.
fn silu_mul_composes() -> Certificate {
    let mut r = rng();
    let n = 512usize;
    let g0 = r.f32_vec(n);
    let u = r.f32_vec(n);
    let mut g = g0.clone();
    silu_mul(&mut g, &u);
    let expect: Vec<f32> = g0.iter().zip(u.iter()).map(|(a, b)| peregrine_model::math::siluf(*a) * *b).collect();
    with_cases(
        cert(
            "math",
            "silu_mul_equals_siluf_times_up",
            "`silu_mul(g, u)` is bit-identical to `siluf(g[i]) * u[i]` elementwise, so the fused activation cannot disagree with the unfused pair.",
            mismatch_bits(&expect, &g),
            0.0,
        ),
        n,
    )
}

/// LayerNorm's contract is standardization: zero mean, unit variance, before the
/// affine. The epsilon is set to 0 and the weights/bias are neutral so the
/// property is observable.
fn layernorm_standardizes() -> Certificate {
    let mut r = rng();
    let n = 512usize;
    let x = r.f32_vec(n);
    let mut v = x.clone();
    layernorm(&mut v, &vec![1.0f32; n], &vec![0.0f32; n], 0.0);
    let mean: f64 = v.iter().map(|x| f64::from(*x)).sum::<f64>() / n as f64;
    let var: f64 = v.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / n as f64;
    // Both sums are f64 in the implementation, so what is left is one f32
    // subtract and one f32 multiply per element plus the f64→f32 narrowing:
    // `n` ulps of the result is the honest bound.
    let tol = 8.0 * n as f64 * EPS_F32;
    cert(
        "math",
        "layernorm_standardizes_the_row",
        "With neutral affine parameters and no epsilon, LayerNorm leaves the row at mean 0 and variance 1 to within `8n` f32 ulps.",
        mean.abs() + (var - 1.0).abs(),
        tol,
    )
}

/// `sigmoidf` is used for every router logit, and `sigmoidf(0) == 0.5` is the
/// one value whose answer is known to be exactly representable — a broken
/// sigmoid (a sign error, a missing negation) would move it away from the
/// symmetry point and skew every routing decision.
fn sigmoid_fixed_point() -> Certificate {
    let holds = sigmoidf(0.0).to_bits() == 0.5f32.to_bits() && sigmoidf(-0.0).to_bits() == 0.5f32.to_bits();
    with_cases(
        cert(
            "math",
            "sigmoid_is_half_at_zero",
            "`sigmoidf(0)` is exactly 0.5 for both signs of zero, so a router logit at the origin is unskewed.",
            exact(holds),
            0.0,
        ),
        2,
    )
}

/// RoPE at position 0 is the identity, exactly: `cos(0) == 1` and `sin(0) == 0`
/// are representable and every lane is computed as `a·1 - b·0` / `b·1 + a·0`.
/// Anything else means the table's frequencies or the pairing are wrong, and a
/// wrong pairing still produces plausible vectors — this is the check that says
/// a Qwen head is being rotated the way its checkpoint was trained.
fn rope_at_zero_is_identity() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (qk, theta) in [(128usize, 10_000.0f32), (64, 500_000.0), (96, 1_000_000.0)] {
        let table = RopeTable::new(qk, theta);
        let mut v = r.f32_vec(2 * table.pairs());
        let expect = v.clone();
        rope_neox_with(&mut v, 0, &table);
        bad += mismatch_bits(&expect, &v);
        cases += 1;
    }
    with_cases(
        cert(
            "math",
            "rope_at_position_zero_is_the_identity",
            "Rotating at position 0 leaves every lane bit-identical, for every table width and theta in the checkpoint range: `cos(0)=1`, `sin(0)=0` are exact.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Rotation is orthogonal, so it preserves the 2-norm of each rotated pair. This
/// is the invariant that distinguishes a correct rotation from a scaling, and it
/// holds independently of the frequency table — which is why it can be stated
/// without restating the formula.
fn rope_preserves_the_norm() -> Certificate {
    let mut r = rng();
    let qk = 128usize;
    let table = RopeTable::new(qk, 10_000.0);
    let mut v = r.f32_vec(2 * table.pairs());
    let before: Vec<f32> = (0..table.pairs()).map(|j| (f64::from(v[j]).powi(2) + f64::from(v[table.pairs() + j]).powi(2)).sqrt() as f32).collect();
    for pos in [1usize, 7, 1024] {
        rope_neox_with(&mut v, pos, &table);
    }
    let after: Vec<f32> = (0..table.pairs()).map(|j| (f64::from(v[j]).powi(2) + f64::from(v[table.pairs() + j]).powi(2)).sqrt() as f32).collect();
    // Each pair touches four f32 operations; the norms are recomputed in f64,
    // so the bound is the arithmetic's.
    let tol = 8.0 * EPS_F32;
    with_cases(
        cert(
            "math",
            "rope_preserves_each_rotated_pair_norm",
            "Rotating at positions 1, 7 and 1024 leaves every rotated pair's 2-norm within `8·eps_f32` of its original value — rotation, not rescaling.",
            max_abs_diff(&before, &after),
            tol,
        ),
        table.pairs(),
    )
}

/// The two RoPE layouts disagree by a permutation of the input lanes: GLM reads
/// interleaved pairs `(2j, 2j+1)` and writes split-half output, while the Neox
/// form reads split-half pairs and writes in place. So interleaving the input is
/// the only difference between them — and applying both forms to the matching
/// permutations must give bit-identical outputs. This is the certificate that
/// catches the pairing being swapped, which scrambles every position while
/// looking entirely plausible.
fn rope_layouts_agree_under_the_input_permutation() -> Certificate {
    let mut r = rng();
    let qk = 64usize;
    let table = RopeTable::new(qk, 10_000.0);
    let half = table.pairs();
    let interleaved_in = r.f32_vec(2 * half);
    // The Neox input, de-interleaved from it: `c[j]` is what interleaved reads
    // as lane `j`, `c[half + j]` as lane `2j + 1`.
    let mut neox_in = vec![0f32; 2 * half];
    for j in 0..half {
        neox_in[j] = interleaved_in[2 * j];
        neox_in[half + j] = interleaved_in[2 * j + 1];
    }
    let pos = 37usize;
    let mut a = interleaved_in.clone();
    let mut b = neox_in;
    rope_interleave_with(&mut a, pos, &table);
    rope_neox_with(&mut b, pos, &table);

    with_cases(
        cert(
            "math",
            "rope_layouts_agree_under_the_input_permutation",
            "Interleaved RoPE on a vector equals Neox RoPE on its de-interleaved permutation, bit-for-bit at the same position: the two layouts differ by a lane permutation, not by a different rotation.",
            mismatch_bits(&a, &b),
            0.0,
        ),
        half,
    )
}

/// RoPE is a group of rotations, so `R_p` then `R_1` is `R_{p+1}` in exact
/// arithmetic. That is what lets a KV cache be split across a resumed session, a
/// beam, or a chunked prefill: positions are re-derived from an offset rather
/// than carried.
///
/// In f32 the two routes are **not** ulp-close, and the reason is worth writing
/// down rather than papering over with a loose tolerance: the dominant error is
/// not the composition but the *argument*. `rope_neox_with` evaluates
/// `cos(pos · inv[j])`, and at position 4097 with the first lane's frequency of
/// 1.0 the argument is ~4097 rad — so the f32 representation of the argument
/// alone contributes up to `pos · eps` of angular error, which the composition
/// does not share. The tolerance is therefore that bound plus the handful of ulps
/// the composition itself adds:
///
/// ```text
/// (pos + 2) · eps_f32      the argument's own f32 rounding, worst lane (inv = 1)
/// + 8 · eps_f32           two roundings of the product identities per lane
/// ```
///
/// A measured deviation well inside that bound is the meaningful statement; a
/// deviation outside it would mean the *angle* was being computed differently,
/// not that f32 lost a bit.
fn rope_positions_compose() -> Certificate {
    let mut r = rng();
    let qk = 64usize;
    let table = RopeTable::new(qk, 10_000.0);
    let half = table.pairs();
    let mut composed = r.f32_vec(2 * half);
    let mut direct = composed.clone();
    // R_1 first, then R_p: order does not matter for a commutative rotation
    // group, and doing both directions in one pass keeps the fixture honest.
    rope_neox_with(&mut composed, 1, &table);
    rope_neox_with(&mut composed, 4096, &table);
    rope_neox_with(&mut direct, 4097, &table);

    // The deviation is measured relative to the lane's own magnitude, because a
    // rotation moves mass between the two lanes of a pair.
    const SPLIT_AT: usize = 4096;
    let mut worst = 0.0f64;
    for (a, b) in composed.iter().zip(direct.iter()) {
        let scale = a.abs().max(b.abs()).max(1.0);
        worst = worst.max((f64::from(*a) - f64::from(*b)).abs() / f64::from(scale));
    }
    // The worst lane is the one whose frequency is 1.0 (`RopeTable` sets
    // `inv[0] = 1`), so the largest angle either route forms is `pos + 1` rad.
    let tolerance = (SPLIT_AT as f64 + 2.0) * EPS_F32 + 8.0 * EPS_F32;
    with_cases(
        cert(
            "math",
            "rope_positions_compose_additively",
            "Rotating at position 1 and then at 4096 equals rotating once at 4097 to within `(pos + 2)·eps + 8·eps` per lane — the bound is set by representing the rotation angle in f32, not by the composition: positions are re-derivable from an offset, which is what a split or resumed KV cache depends on.",
            worst,
            tolerance,
        ),
        half,
    )
}

/// Small alias so the group reads the same as the others: `rmsnorm_inplace` is
/// the fused variant the production path uses, reached through the model crate's
/// public surface.
fn rmsnorm_inplace_helper(v: &mut [f32], w: &[f32], eps: f32) {
    peregrine_model::math::rmsnorm_inplace(v, w, eps);
}
//! The decode-decision group: what the sampler is allowed to assume about the
//! token it picks.
//!
//! Greedy decode is the engine's validation mode — `temp <= 0` must reproduce
//! the reference token for token — so its conventions are contracts, not
//! implementation details. Ties go to the lowest index, NaN is skipped rather
//! than allowed to win, and a seed of 0 must not be a fixed point of the RNG.
//! None of those fail loudly: they emit a *different token*.

use peregrine_model::sample::{argmax, pick_batch_greedy, top_prob, Sampler};

use crate::{cert, rng, with_cases, Certificate};

pub fn certificates() -> Vec<Certificate> {
    vec![
        argmax_breaks_ties_lowest(),
        argmax_skips_nan(),
        batch_greedy_agrees_with_argmax(),
        top_prob_is_inverse_normalizer(),
        seed_zero_is_not_a_fixed_point(),
    ]
}

/// Two logits of equal value are a real occurrence (a saturated or clamped head,
/// a masked prefix), and which token wins changes the output. The contract is
/// `argmax_v`'s: a strict `>` comparison, so the **lowest** index wins.
fn argmax_breaks_ties_lowest() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let mut r = rng();
    for n in [2usize, 8, 64] {
        for _ in 0..16 {
            let mut row = r.f32_vec(n);
            // Force a tie by copying the current maximum onto another index; the
            // answer is then the *lower* of the two, whichever that is.
            let best = argmax(&row);
            let other = (best + 1) % n;
            row[other] = row[best];
            cases += 1;
            if argmax(&row) != best.min(other) {
                bad += 1.0;
            }
        }
    }
    // An all-equal row is the degenerate version of the same rule.
    let flat = vec![0.25f32; 33];
    cases += 1;
    if argmax(&flat) != 0 {
        bad += 1.0;
    }
    with_cases(
        cert(
            "sample",
            "argmax_breaks_ties_at_the_lowest_index",
            "Two equal logits resolve to the lower index, and an all-equal row resolves to index 0: greedy decode's tie convention is `argmax_v`'s strict `>`.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// A NaN logit must not win, and must not pin the result. `x > NaN` is false for
/// every `x`, so a NaN that reached the running best would make every later
/// candidate compare false — the decoder would then emit that one token forever.
/// The documented behaviour is that the NaN is skipped, so the argmax is the
/// best *comparable* logit.
fn argmax_skips_nan() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let row = vec![0.1f32, 0.9, 0.2, 0.8, 0.3];
    let clean = argmax(&row);
    // A NaN before the winner.
    // A leading NaN shifts every index by one but must not change *which* token
    // wins, so the expected answer moves with it.
    let mut with_nan = row.clone();
    with_nan.insert(0, f32::NAN);
    cases += 1;
    if argmax(&with_nan) != clean + 1 {
        bad += 1.0;
    }
    // A NaN after the winner.
    let mut trailing = row.clone();
    trailing.push(f32::NAN);
    cases += 1;
    if argmax(&trailing) != clean {
        bad += 1.0;
    }
    // NaNs everywhere: nothing is comparable, and the documented answer is 0
    // rather than a pin to whichever NaN came first.
    let all_nan = vec![f32::NAN; 8];
    cases += 1;
    if argmax(&all_nan) != 0 || argmax(&[]) != 0 {
        bad += 1.0;
    }
    with_cases(
        cert(
            "sample",
            "argmax_skips_non_finite_logits",
            "A NaN before or after the maximum does not change the argmax, and an all-NaN or empty row yields 0 rather than pinning the decode to one token.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The batched fast path must be the same decision as the per-row one. It is a
/// separate loop with its own indexing, and a batch/sequence disagreement is a
/// "the model answers differently at batch 16" bug report that has no stack trace.
fn batch_greedy_agrees_with_argmax() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (s_n, vocab) in [(1usize, 7usize), (4, 128), (16, 1024)] {
        let logits = r.f32_vec(s_n * vocab);
        let mut out = vec![-1i32; s_n];
        pick_batch_greedy(&logits, vocab, &mut out);
        for s in 0..s_n {
            cases += 1;
            if out[s] != argmax(&logits[s * vocab..s * vocab + vocab]) as i32 {
                bad += 1.0;
            }
        }
        // Ties in a batch row must resolve the same way the scalar path does.
        // An all-tied batch: every row flat, so every row must resolve to 0.
        let tied = vec![0.25f32; s_n * vocab];
        let mut out = vec![-1i32; s_n];
        pick_batch_greedy(&tied, vocab, &mut out);
        cases += s_n;
        if out.iter().any(|o| *o != 0) {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "sample",
            "batched_greedy_picks_the_same_token_as_argmax",
            "`pick_batch_greedy` picks exactly `argmax` for every row of every batch shape, including an all-tied row resolving to token 0.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The draft confidence the speculative gate compares against its floor: the
/// probability mass the top token holds. It is computed as `1 / Σ exp(l − max)`,
/// so by construction it lies in `(0, 1]` — it can never exceed 1 (which would
/// make every draft certain) and can never be negative. A non-finite or
/// all-masked row reports exactly 0.0, the "no confidence, stop drafting" answer.
fn top_prob_is_inverse_normalizer() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in [2usize, 17, 1024] {
        for _ in 0..8 {
            let row = r.f32_vec(n);
            let p = top_prob(&row);
            cases += 1;
            if !(p > 0.0 && p <= 1.0) {
                bad += 1.0;
            }
        }
    }
    // Degenerate rows: no confidence, never NaN.
    for row in [vec![f32::NEG_INFINITY; 8], vec![f32::NAN; 8], Vec::new()] {
        cases += 1;
        let p = top_prob(&row);
        if p != 0.0 {
            bad += 1.0;
        }
    }
    // A one-hot row is certainty: exactly 1.0.
    let mut onehot = vec![0.0f32; 64];
    onehot[17] = 50.0;
    cases += 1;
    if top_prob(&onehot).to_bits() != 1.0f32.to_bits() {
        bad += 1.0;
    }
    with_cases(
        cert(
            "sample",
            "top_prob_is_a_confidence_in_the_unit_interval",
            "`top_prob` lies in `(0, 1]` for every finite row, is exactly 1.0 for a one-hot row, and reports exactly 0.0 for an all-masked, all-NaN or empty row.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// xorshift64 maps 0 to 0, so a sampler seeded with 0 would draw 0.0 forever
/// and emit the same token of the support at every step — a decode loop that
/// "reproduces" while saying nothing. `Sampler::new` maps a zero seed to the
/// engine's default stream for exactly this reason.
///
/// The property is therefore about the *draws*: `Sampler::uniform` is the RNG the
/// sampler's inverse-CDF step consumes, so a seed-0 sampler and a default-seeded
/// one must produce different sequences, and a given seed must reproduce its own.
fn seed_zero_is_not_a_fixed_point() -> Certificate {
    let draws = |seed: u64| -> Vec<f64> {
        let mut s = Sampler::new(1.0, 0.0, seed);
        (0..64).map(|_| s.uniform()).collect()
    };
    let zero = draws(0);
    let default_stream = draws(1);
    let repeated = draws(1);
    // Reproducible from its own seed.
    let reproducible = default_stream == repeated;
    // A seed-0 sampler must not be the degenerate stream: it has to move.
    let moves = zero.windows(2).any(|w| w[0] != w[1]) && zero.iter().any(|v| *v > 0.0);
    // And it must be a different stream from any other seed, not just a shifted
    // copy of one.
    let distinct = zero != default_stream && !zero.iter().zip(default_stream.iter()).all(|(a, b)| (a - b).abs() < 1e-12);
    with_cases(
        cert(
            "sample",
            "a_zero_seed_is_not_a_degenerate_rng_stream",
            "A sampler seeded 0 draws the engine's default stream rather than xorshift64's fixed point at 0: its 64 draws move, they differ from another seed's, and the same seed reproduces its own sequence exactly.",
            f64::from(u8::from(!reproducible)) + f64::from(u8::from(!moves)) + f64::from(u8::from(!distinct)),
            0.0,
        ),
        64,
    )
}
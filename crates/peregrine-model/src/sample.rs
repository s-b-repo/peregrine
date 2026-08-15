//! Token sampling — port of `argmax_v` / `dist_build` / `dist_sample` /
//! `pick_tok` and the xorshift64 RNG (`c/glm.c:4063-4101`).
//!
//! `temp <= 0` is greedy argmax (deterministic — the token-exact validation
//! mode). Otherwise: softmax at `temp`, optional nucleus (top-p) truncation,
//! then inverse-CDF sampling with an optional banned token (used by speculative
//! rejection sampling so the draft stays invisible to the output distribution).

/// The C engine's default xorshift64 seed. Also the substitute for a caller
/// seed of 0, which xorshift64 maps to itself forever (a dead RNG stream).
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Greedy argmax; lowest index wins ties (strict `>`), matching `argmax_v`.
///
/// NaN logits are skipped rather than allowed to win: `x > NaN` is false, so a
/// NaN sitting at the running best would make every later candidate compare
/// false and pin the result to that index — greedy decode would then emit the
/// same token forever instead of surfacing the numerical fault. An all-NaN row
/// still returns 0 (nothing is comparable), which the caller sees as a normal
/// token id.
pub fn argmax(lo: &[f32]) -> usize {
    let mut b = match lo.iter().position(|v| !v.is_nan()) {
        Some(i) => i,
        None => return 0, // empty, or every logit is NaN
    };
    for i in b + 1..lo.len() {
        if lo[i] > lo[b] {
            b = i;
        }
    }
    b
}

/// Probability the argmax token would receive under `softmax(lo)` — the draft
/// confidence the speculative gate (`COLI_SPEC_CONF`) compares against its
/// floor. Stable by construction: shifting by the max makes the top term
/// `exp(0)`, so the result is `1 / Σ exp(l − max)` and nothing overflows.
///
/// NaN logits are skipped, matching [`argmax`]'s stance. An empty, all-NaN or
/// `+inf`-contaminated row reports `0.0` — under any floor that reads as "no
/// confidence" and stops the draft, which is the safe side: a numerical fault
/// costs draft depth, never a wrong token (acceptance is separate).
pub fn top_prob(lo: &[f32]) -> f32 {
    let mut mx = f32::NEG_INFINITY;
    for &v in lo {
        if !v.is_nan() && v > mx {
            mx = v;
        }
    }
    if !mx.is_finite() {
        return 0.0;
    }
    let mut denom = 0f64;
    for &v in lo {
        if !v.is_nan() {
            denom += f64::from(v - mx).exp();
        }
    }
    if denom > 0.0 {
        (1.0 / denom) as f32
    } else {
        0.0
    }
}

/// Greedy argmax over a batch of `out.len()` logit rows `logits[n, vocab]`, one
/// token per row into `out`. The all-greedy fast path for batched decode
/// (`temp <= 0`) — skips per-sequence `Sampler` dispatch. Sampled decode keeps a
/// per-sequence [`Sampler`] and calls [`Sampler::pick`] on each row instead, so
/// each sequence draws from its own RNG stream.
pub fn pick_batch_greedy(logits: &[f32], vocab: usize, out: &mut [i32]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = argmax(&logits[s * vocab..s * vocab + vocab]) as i32;
    }
}

/// Stateful sampler: holds the RNG stream and reused distribution buffers so a
/// decode loop is single-threaded and reproducible from `seed`.
pub struct Sampler {
    rng: u64,
    pub temp: f32,
    pub nucleus: f32,
    p: Vec<f32>,
    idx: Vec<usize>,
}

impl Sampler {
    /// `temp <= 0` → greedy. `nucleus` in (0,1) enables top-p truncation.
    /// The default RNG seed matches the C engine (`0x9E3779B97F4A7C15`).
    pub fn new(temp: f32, nucleus: f32, seed: u64) -> Sampler {
        // xorshift64 has 0 as a fixed point: seeded with it, every `rndu()`
        // returns 0.0 and sampling degenerates to "always the first token".
        // `--seed 0` is the most natural determinism knob a caller would reach
        // for, so map it to the documented default stream.
        let rng = if seed == 0 { DEFAULT_SEED } else { seed };
        Sampler { rng, temp, nucleus, p: Vec::new(), idx: Vec::new() }
    }

    /// xorshift64 → uniform double in [0,1). Port of `rndu`.
    fn rndu(&mut self) -> f64 {
        let mut g = self.rng;
        g ^= g << 13;
        g ^= g >> 7;
        g ^= g << 17;
        self.rng = g;
        (g >> 11) as f64 * (1.0 / 9007199254740992.0) // 2^53
    }

    /// Build the target distribution into `self.p`: softmax(lo/temp), optionally
    /// truncated to the top-p `nucleus` mass and renormalized. Port of `dist_build`.
    fn dist_build(&mut self, lo: &[f32]) {
        let v = lo.len();
        self.p.resize(v, 0.0);
        let mx = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let invt = 1.0 / self.temp.max(1e-4);
        let mut s = 0f64;
        for (p, &l) in self.p.iter_mut().zip(lo.iter()).take(v) {
            *p = ((l - mx) * invt).exp();
            s += *p as f64;
        }
        for p in self.p.iter_mut().take(v) {
            *p /= s as f32;
        }
        // `nucleus >= 1` (or NaN) means "keep everything". A `nucleus` of 0 is a
        // request for the tightest possible nucleus, i.e. the single most likely
        // token — the loop below reaches `cum >= 0` at the first entry and keeps
        // exactly one. Gating on `> 0.0` instead disabled truncation entirely,
        // which is the opposite of what `top_p: 0` asks for.
        if self.nucleus < 1.0 && !self.nucleus.is_nan() {
            self.idx.clear();
            self.idx.extend(0..v);
            let p = &self.p;
            // total_cmp is a total order over f32 (NaN-safe) — `partial_cmp` would
            // return None on a NaN probability and panic on unwrap, and a NaN in a
            // sort comparator also violates sort's ordering contract.
            self.idx.sort_by(|&a, &b| p[b].total_cmp(&p[a])); // desc
            let mut cum = 0f64;
            let mut keep = v;
            for i in 0..v {
                cum += self.p[self.idx[i]] as f64;
                if cum >= self.nucleus as f64 {
                    keep = i + 1;
                    break;
                }
            }
            for i in keep..v {
                self.p[self.idx[i]] = 0.0;
            }
            let mut s2 = 0f64;
            for i in 0..keep {
                s2 += self.p[self.idx[i]] as f64;
            }
            for i in 0..keep {
                self.p[self.idx[i]] /= s2 as f32;
            }
        }
    }

    /// Inverse-CDF sample from `self.p`; `ban >= 0` excludes that token,
    /// renormalizing on the fly. Port of `dist_sample`.
    fn dist_sample(&mut self, v: usize, ban: i32) -> usize {
        // Resolve the ban to an in-range index once: a ban past the vocabulary
        // (a draft id from a mismatched tokenizer, a stale id after a vocab
        // change) would otherwise index `self.p` out of range and abort.
        let ban_idx: Option<usize> = usize::try_from(ban).ok().filter(|&b| b < v.min(self.p.len()));
        let ban: i32 = match ban_idx {
            Some(b) => b as i32,
            None => -1,
        };
        let banned = match ban_idx {
            Some(b) => self.p[b] as f64,
            None => 0.0,
        };
        let mut z = 1.0 - banned;
        if z <= 1e-12 {
            z = 1e-12;
        }
        let u = self.rndu() * z;
        let mut cum = 0f64;
        for i in 0..v {
            if i as i32 == ban {
                continue;
            }
            cum += self.p[i] as f64;
            // `cum > u` (not `>=`) so a draw of exactly 0.0 cannot select a
            // token the nucleus truncation zeroed out.
            if cum > u {
                return i;
            }
        }
        for i in (0..v).rev() {
            if i as i32 != ban && self.p[i] > 0.0 {
                return i;
            }
        }
        // Nothing carries positive probability. Return any non-banned index —
        // returning 0 unconditionally could hand back the banned token itself,
        // which would defeat speculative rejection sampling's correctness.
        (0..v).find(|&i| i as i32 != ban).unwrap_or(0)
    }

    /// Next token from logits. Greedy if `temp <= 0`, else sampled. `ban < 0`
    /// means no banned token.
    pub fn pick(&mut self, lo: &[f32], ban: i32) -> usize {
        if self.temp <= 0.0 {
            return argmax(lo);
        }
        self.dist_build(lo);
        self.dist_sample(lo.len(), ban)
    }

    /// This sampler's target distribution over `lo`: softmax at `temp`, nucleus-
    /// truncated and renormalized — **the exact transform [`Self::pick`]
    /// samples from**, which is the entire reason this is a method on `Sampler`
    /// rather than a free function.
    ///
    /// [`crate::speculative_sample`]'s guarantee is that the emitted token is
    /// distributed as `p`. It compares `p[d]/q[d]`, so if `p` and `q` come from
    /// two different transforms — one nucleus-truncated and one not, one at the
    /// request's temperature and one at the draft head's — the ratio is a
    /// number with no meaning and the "provably distribution-preserving" claim
    /// silently becomes false. Sharing this one builder is what makes it true.
    ///
    /// At `temp <= 0` the distribution is the one-hot at the argmax, which is
    /// what greedy decoding *is* as a distribution.
    pub fn distribution(&mut self, lo: &[f32]) -> &[f32] {
        if self.temp <= 0.0 {
            self.p.clear();
            self.p.resize(lo.len(), 0.0);
            if !lo.is_empty() {
                self.p[argmax(lo)] = 1.0;
            }
            return &self.p;
        }
        self.dist_build(lo);
        &self.p
    }

    /// Sample a token **and** return the distribution it was drawn from.
    ///
    /// One call, because the pair is a precondition rather than two
    /// conveniences: `speculative_sample` assumes `drafted ~ q`, and a caller
    /// that sampled with one call and described the distribution with another
    /// could drift between them (a re-`dist_build` on different logits, a
    /// nucleus change) with nothing detecting it.
    pub fn pick_with_distribution(&mut self, lo: &[f32]) -> (usize, Vec<f32>) {
        if self.temp <= 0.0 {
            let t = argmax(lo);
            let mut q = vec![0.0; lo.len()];
            if !lo.is_empty() {
                q[t] = 1.0;
            }
            return (t, q);
        }
        self.dist_build(lo);
        let q = self.p.clone();
        (self.dist_sample(lo.len(), -1), q)
    }

    /// One uniform in `[0,1)` from this sampler's stream.
    ///
    /// Exposed for [`crate::speculative_sample`]'s two draws. Taking them from
    /// the *request's* stream rather than a private one is deliberate: it keeps
    /// a seeded request reproducible end to end, and it is the only stream whose
    /// consumption is already part of the request's contract.
    pub fn uniform(&mut self) -> f64 {
        self.rndu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_lowest_index_wins_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 5.0, 1.0]), 0); // tie → lowest index
    }

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(0.0, 0.9, 1);
        assert_eq!(s.pick(&[0.1, 0.9, 0.3, 0.2], -1), 1);
    }

    #[test]
    fn top_prob_is_uniform_share_on_flat_logits_and_one_on_a_spike() {
        // Flat logits: every token holds exactly 1/n of the mass.
        let flat = [0.5f32; 8];
        assert!((top_prob(&flat) - 1.0 / 8.0).abs() < 1e-6);
        // A dominant logit takes essentially all of it.
        let spike = [0.0f32, 30.0, 0.0, 0.0];
        assert!(top_prob(&spike) > 0.999);
        // Sharpening the same shape can only raise the top share — the
        // monotonicity a confidence floor relies on.
        let soft = [0.0f32, 2.0, 0.0, 0.0];
        assert!(top_prob(&spike) > top_prob(&soft));
        assert!(top_prob(&soft) > top_prob(&flat));
    }

    #[test]
    fn top_prob_reports_zero_confidence_on_degenerate_rows() {
        // Empty, all-NaN, and +inf-contaminated rows must all read as "no
        // confidence" so a floor stops the draft instead of trusting a fault.
        assert_eq!(top_prob(&[]), 0.0);
        assert_eq!(top_prob(&[f32::NAN, f32::NAN]), 0.0);
        assert_eq!(top_prob(&[1.0, f32::INFINITY]), 0.0);
        // A NaN alongside real logits is skipped, not fatal.
        let with_nan = [0.0f32, f32::NAN, 30.0];
        assert!(top_prob(&with_nan) > 0.999);
    }

    #[test]
    fn pick_batch_greedy_argmaxes_each_row() {
        // three rows of vocab=4; each row's token is its own argmax
        let logits = [0.1, 0.9, 0.3, 0.2, /**/ 5.0, 1.0, 2.0, 3.0, /**/ 0.0, 0.0, 0.0, 1.0];
        let mut out = [0i32; 3];
        pick_batch_greedy(&logits, 4, &mut out);
        assert_eq!(out, [1, 0, 3]);
    }

    #[test]
    fn sampling_is_reproducible() {
        let lo = [1.0f32, 2.0, 0.5, 3.0, 1.5];
        let mut a = Sampler::new(1.0, 0.9, 42);
        let mut b = Sampler::new(1.0, 0.9, 42);
        for _ in 0..20 {
            assert_eq!(a.pick(&lo, -1), b.pick(&lo, -1));
        }
    }

    #[test]
    fn peaked_logits_mostly_pick_peak() {
        // one dominant logit → sampled overwhelmingly (softmax still stochastic)
        let lo = [0.0f32, 0.0, 12.0, 0.0];
        let mut s = Sampler::new(1.0, 1.0, 7);
        let hits = (0..200).filter(|_| s.pick(&lo, -1) == 2).count();
        assert!(hits > 190, "peak picked {hits}/200");
    }

    #[test]
    fn nucleus_excludes_tail() {
        // tiny nucleus keeps only the top token → tail never sampled
        let lo = [0.0f32, 1.0, 2.0, 5.0];
        let mut s = Sampler::new(1.0, 0.3, 3);
        for _ in 0..100 {
            assert_eq!(s.pick(&lo, -1), 3);
        }
    }

    #[test]
    fn nan_logits_do_not_panic() {
        // a NaN logit (e.g. from a corrupted forward) must not panic the sort or
        // the sampler — total_cmp gives a total order and the result stays valid.
        let lo = [1.0f32, f32::NAN, 2.0, 0.5];
        let mut s = Sampler::new(1.0, 0.9, 5);
        let t = s.pick(&lo, -1);
        assert!(t < lo.len());
    }

    #[test]
    fn ban_is_never_sampled() {
        let lo = [2.0f32, 2.0, 2.0, 2.0];
        let mut s = Sampler::new(1.0, 1.0, 99);
        for _ in 0..200 {
            assert_ne!(s.pick(&lo, 1), 1);
        }
    }

    #[test]
    fn argmax_skips_nan_and_finds_the_real_peak() {
        // `x > NaN` is false, so a NaN at the running best used to pin the result
        // to index 0 — greedy decode then emitted token 0 forever.
        assert_eq!(argmax(&[f32::NAN, 1.0, 9.0, 2.0]), 2);
        assert_eq!(argmax(&[f32::NAN, f32::NAN, 3.0]), 2);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0, "all-NaN still returns a valid index");
        assert_eq!(argmax(&[]), 0, "empty row returns a valid index");
    }

    #[test]
    fn top_p_zero_selects_the_single_most_likely_token() {
        // `top_p: 0` asks for the tightest nucleus (top-1). The old `> 0.0` guard
        // skipped truncation entirely and sampled the FULL distribution — the
        // exact opposite. The HTTP layer clamps top_p to [0,1], so 0 is reachable.
        let lo = [0.0f32, 1.0, 2.0, 5.0];
        let mut s = Sampler::new(1.0, 0.0, 11);
        for _ in 0..200 {
            assert_eq!(s.pick(&lo, -1), 3, "top_p=0 must behave as top-1");
        }
    }

    #[test]
    fn out_of_range_ban_is_ignored_not_fatal() {
        // A ban id past the vocabulary indexed the probability buffer out of
        // bounds; under `panic = "abort"` that killed the server.
        let lo = [1.0f32, 2.0, 3.0];
        let mut s = Sampler::new(1.0, 1.0, 21);
        for ban in [3i32, 99, i32::MAX] {
            let t = s.pick(&lo, ban);
            assert!(t < lo.len(), "ban {ban} must be ignored, got {t}");
        }
    }

    #[test]
    fn zero_seed_still_produces_a_live_rng_stream() {
        // xorshift64 maps 0 to 0 forever: `rndu()` would return 0.0 every draw
        // and sampling would always return the first token.
        let lo = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut s = Sampler::new(1.0, 1.0, 0);
        let picks: std::collections::HashSet<usize> = (0..200).map(|_| s.pick(&lo, -1)).collect();
        assert!(picks.len() > 1, "seed 0 must not collapse to one token: {picks:?}");
    }

    #[test]
    fn banned_token_is_not_returned_by_the_fallback() {
        // Degenerate case: the only token with mass is the banned one. Returning
        // 0 unconditionally would re-emit exactly the token being rejected.
        let lo = [10.0f32, -50.0];
        let mut s = Sampler::new(1.0, 1.0, 4);
        for _ in 0..50 {
            assert_ne!(s.pick(&lo, 0), 0, "the fallback must respect the ban");
        }
    }
}

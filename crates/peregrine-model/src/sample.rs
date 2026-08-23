//! Token sampling — port of `argmax_v` / `dist_build` / `dist_sample` /
//! `pick_tok` and the xorshift64 RNG (`c/glm.c:4063-4101`).
//!
//! `temp <= 0` is greedy argmax (deterministic — the token-exact validation
//! mode). Otherwise: softmax at `temp`, optional nucleus (top-p) truncation,
//! then inverse-CDF sampling with an optional banned token (used by speculative
//! rejection sampling so the draft stays invisible to the output distribution).

use crate::draftdist::DraftDist;

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
    /// Rank cutoff: keep only the `top_k` most likely tokens. `0` is off.
    ///
    /// Set through [`Sampler::with_top_k`] rather than [`Sampler::new`] so the
    /// twenty-odd existing construction sites — nearly all of them tests
    /// asserting something about temperature or the nucleus — keep saying what
    /// they mean instead of carrying a third argument none of them care about.
    pub top_k: usize,
    p: Vec<f32>,
    /// Support of the distribution `dist_build` last built, in descending
    /// probability order. Empty means "the whole vocabulary" — see
    /// [`Sampler::support`].
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
        Sampler { rng, temp, nucleus, top_k: 0, p: Vec::new(), idx: Vec::new() }
    }

    /// Also truncate to the `k` most likely tokens. `0` disables it, which is
    /// the default and the historical behaviour.
    ///
    /// Ported from DFlash's `_sampling_probs`, which applies `torch.topk`
    /// *before* the softmax and `top_p` after it — the composition order this
    /// reproduces (see [`Self::truncate`]). It closes a real gap: the model
    /// families this engine serves ship `top_k` in their own recommended
    /// sampling settings (Qwen3 and GLM both publish `top_k` alongside `top_p`),
    /// and an OpenAI-compatible server that silently ignored the field was
    /// answering a different question than the client asked.
    pub fn with_top_k(mut self, k: usize) -> Sampler {
        self.top_k = k;
        self
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
    /// truncated to a top-k rank cutoff and the top-p `nucleus` mass, then
    /// renormalized. Port of `dist_build`, with DFlash's truncation shape.
    fn dist_build(&mut self, lo: &[f32]) {
        let v = lo.len();
        self.p.resize(v, 0.0);
        self.idx.clear();
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
        // token — the walk below reaches `cum >= 0` at the first entry and keeps
        // exactly one. Gating on `> 0.0` instead disabled truncation entirely,
        // which is the opposite of what `top_p: 0` asks for.
        let nucleus_on = self.nucleus < 1.0 && !self.nucleus.is_nan();
        let hard_k = if self.top_k > 0 { self.top_k.min(v) } else { v };
        if (hard_k >= v && !nucleus_on) || v == 0 {
            return; // support is the whole vocabulary: nothing to cut, nothing to record
        }
        self.truncate(v, hard_k.max(1), nucleus_on);
    }

    /// Cut `self.p` down to its top-k / nucleus support, renormalize over what
    /// survives, and leave the surviving indices in `self.idx` in descending
    /// probability order.
    ///
    /// **This replaces a full sort of the vocabulary on every sampled token.**
    /// `sort_by` over 154 880 indices with an indirect comparator answers a
    /// question whose answer is usually the first few dozen entries. DFlash's
    /// `_sampling_probs` has the shape that avoids it — `torch.topk` first,
    /// order only the survivors — and that is what this does. A top-k gives the
    /// rank boundary outright; a nucleus does not, so [`Self::nucleus_bound`]
    /// derives one in a single pass and the selection and sort behind it stay
    /// small. Measured on this box at GLM-5.2's 154 880-token vocabulary
    /// (`cargo run --release -p peregrine-model --example nucleusbench`, and
    /// `docs/dflash.md` for the method): **4.8–5.1× faster** on decode-shaped
    /// rows, 2.6× where the nucleus keeps ~17 000 tokens, 1.2× where it keeps
    /// half the vocabulary, and 0.95× in the one regime where it cannot win — a
    /// nucleus keeping ~90 % of the vocabulary, where the bound costs a pass and
    /// saves nothing.
    ///
    /// The two cutoffs compose in DFlash's order: top-k first, renormalize over
    /// it, then apply the nucleus to *those* probabilities. Doing it the other
    /// way round would measure the nucleus against mass the top-k has already
    /// discarded, so the same `(top_k, top_p)` pair would mean something
    /// different here than at the client that sent it.
    fn truncate(&mut self, v: usize, hard_k: usize, nucleus_on: bool) {
        // A count the answer provably cannot exceed: the top-k cutoff itself
        // when there is one, and otherwise the exponent bound.
        let mut ranked = if hard_k < v { hard_k } else { self.nucleus_bound(v) };
        self.rank_top(v, ranked);
        let mut keep = ranked;
        if nucleus_on {
            // Denominator of the post-top-k renormalization. Without a top-k it
            // is the whole softmax, which already sums to 1.
            let mass: f64 = if hard_k < v {
                self.idx.iter().map(|&i| f64::from(self.p[i])).sum()
            } else {
                1.0
            };
            keep = match self.nucleus_prefix(mass) {
                Some(k) => k,
                // The bound promised mass the exact prefix did not quite reach
                // — the same values summed in a different order, a couple of
                // ULP apart. Rank everything and settle it exactly rather than
                // approximately. Not reachable under a top-k, where the mass was
                // measured over the very entries being walked.
                None if hard_k >= v && ranked < v => {
                    ranked = v;
                    self.rank_top(v, v);
                    self.nucleus_prefix(1.0).unwrap_or(v)
                }
                None => ranked,
            };
        }
        self.normalize_support(keep, ranked >= v);
    }

    /// How many of the most likely entries the nucleus can possibly need, from
    /// one pass over the probabilities.
    ///
    /// Buckets by IEEE-754 exponent — 256 of them, every member of a bucket
    /// within 2× of every other. What makes this a **bound** and not a heuristic
    /// is that every entry outside the top `n` buckets is strictly smaller than
    /// every entry inside them, so those `n` entries *are* the ranked top `n`.
    /// Once the buckets walked from the top hold the nucleus mass, the prefix
    /// that reaches it lies inside them.
    ///
    /// Returns `v` when the mass is never reached, which is what a NaN row does:
    /// the caller then ranks everything, exactly as the full sort always did.
    fn nucleus_bound(&self, v: usize) -> usize {
        let mut counts = [0u32; 256];
        let mut mass = [0f64; 256];
        for &x in self.p.iter() {
            let e = ((x.to_bits() >> 23) & 0xFF) as usize;
            counts[e] += 1;
            mass[e] += f64::from(x);
        }
        let target = f64::from(self.nucleus);
        let mut cum = 0f64;
        let mut n = 0usize;
        for e in (0..256).rev() {
            n += counts[e] as usize;
            cum += mass[e];
            if cum >= target {
                return n.clamp(1, v);
            }
        }
        v
    }

    /// Length of the shortest prefix of the ranked support holding `nucleus` of
    /// `mass`, or `None` if the whole ranked support does not hold it.
    fn nucleus_prefix(&self, mass: f64) -> Option<usize> {
        let target = f64::from(self.nucleus) * mass;
        let mut cum = 0f64;
        for (rank, &i) in self.idx.iter().enumerate() {
            cum += f64::from(self.p[i]);
            if cum >= target {
                return Some(rank + 1);
            }
        }
        None
    }

    /// Zero everything outside `self.idx[..keep]`, renormalize over it, and
    /// leave `self.idx` holding exactly the support.
    ///
    /// `ranked_all` says `self.idx` covers the vocabulary, in which case the cut
    /// entries are exactly its tail — cheaper than clearing `p` and scattering
    /// the survivors back, and the difference is measurable precisely in the
    /// regime where the nucleus keeps most of the vocabulary.
    fn normalize_support(&mut self, keep: usize, ranked_all: bool) {
        let Sampler { p, idx, .. } = self;
        let mut s2 = 0f64;
        for i in 0..keep {
            s2 += f64::from(p[idx[i]]);
        }
        if ranked_all {
            for i in keep..idx.len() {
                p[idx[i]] = 0.0;
            }
        } else {
            let kept: Vec<f32> = idx[..keep].iter().map(|&i| p[i]).collect();
            p.iter_mut().for_each(|x| *x = 0.0);
            for (&i, &x) in idx[..keep].iter().zip(kept.iter()) {
                p[i] = x;
            }
        }
        for i in 0..keep {
            p[idx[i]] /= s2 as f32;
        }
        idx.truncate(keep);
    }

    /// Leave the `k` highest-probability indices in `self.idx`, descending.
    ///
    /// The partial comparator breaks probability ties by **ascending index**,
    /// making it a total order with no ties at all. That is load-bearing: the
    /// selection is unstable, so without a tiebreak two equally likely tokens
    /// could rank either way and a seeded request would stop being reproducible.
    /// Ranking the whole vocabulary needs no such tiebreak — a *stable* sort
    /// already leaves equal probabilities in ascending index order, which is the
    /// same order and one comparison cheaper.
    fn rank_top(&mut self, v: usize, k: usize) {
        let Sampler { p, idx, .. } = self;
        let p: &[f32] = p;
        idx.clear();
        idx.extend(0..v);
        if k >= v {
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            return;
        }
        let cmp = |&a: &usize, &b: &usize| p[b].total_cmp(&p[a]).then(a.cmp(&b));
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
        idx.sort_unstable_by(cmp);
    }

    /// Support of the distribution [`Self::dist_build`] last built: `Some` when
    /// a top-k or nucleus cut it down, `None` when it is the whole vocabulary
    /// and there is no smaller way to say so.
    ///
    /// The pair DFlash's `_sampling_probs` returns, where `indices` is `None`
    /// exactly when neither cutoff removed anything.
    pub fn support(&self) -> Option<&[usize]> {
        if self.idx.is_empty() {
            None
        } else {
            Some(&self.idx)
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
            // Clear the support too, or [`Self::support`] would keep describing
            // whatever distribution this sampler built last. Nothing reads it on
            // the greedy path today; leaving a stale answer reachable is the
            // kind of thing that is only ever found by the bug it causes.
            self.idx.clear();
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

    /// Sample a token **and** return the distribution it was drawn from, held at
    /// its support.
    ///
    /// The sparse twin of [`Self::pick_with_distribution`], and the one the
    /// engine's speculative path uses. Same single call, so the token and its
    /// `q` still cannot come apart; the difference is only that `q` is a
    /// [`DraftDist`] rather than a vocabulary of mostly zeros. At GLM-5.2's
    /// vocabulary that is ~620 KB per draft step per sequence saved — see
    /// [`crate::draftdist`] for why it is a change of representation and not of
    /// distribution.
    pub fn pick_with_draft_dist(&mut self, lo: &[f32]) -> (usize, DraftDist) {
        if self.temp <= 0.0 {
            let t = argmax(lo);
            return (t, DraftDist::one_hot(lo.len(), t));
        }
        self.dist_build(lo);
        // Sparse only while it is actually smaller. An index costs a `u32`
        // alongside the `f32`, so the support form is 8 B/entry against the
        // dense 4 B/entry and stops paying for itself past half the vocabulary
        // — which a nucleus over a genuinely flat distribution does reach. The
        // dense form is also `O(1)` to look up where the support form is
        // `O(k)`, so the two thresholds point the same way. Both are the same
        // type and the same rule reads them, so this choice cannot desync
        // anything; it only picks the cheaper encoding of one distribution.
        let q = match self.support() {
            Some(sup) if sup.len() * 2 < lo.len() => {
                DraftDist::sparse(lo.len(), sup.iter().map(|&i| (i as u32, self.p[i])))
            }
            _ => DraftDist::dense(self.p.clone()),
        };
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

    /// The pre-DFlash `dist_build`, verbatim: softmax, a **full stable sort of
    /// the vocabulary**, keep the smallest prefix reaching the nucleus,
    /// renormalize. The reference the `O(v)` selection has to reproduce.
    fn reference_dist(lo: &[f32], temp: f32, nucleus: f32) -> Vec<f32> {
        let v = lo.len();
        let mut p = vec![0f32; v];
        let mx = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let invt = 1.0 / temp.max(1e-4);
        let mut s = 0f64;
        for (pi, &l) in p.iter_mut().zip(lo.iter()) {
            *pi = ((l - mx) * invt).exp();
            s += *pi as f64;
        }
        for pi in p.iter_mut() {
            *pi /= s as f32;
        }
        if nucleus < 1.0 && !nucleus.is_nan() {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            let mut cum = 0f64;
            let mut keep = v;
            for i in 0..v {
                cum += p[idx[i]] as f64;
                if cum >= nucleus as f64 {
                    keep = i + 1;
                    break;
                }
            }
            for i in keep..v {
                p[idx[i]] = 0.0;
            }
            let mut s2 = 0f64;
            for i in 0..keep {
                s2 += p[idx[i]] as f64;
            }
            for i in 0..keep {
                p[idx[i]] /= s2 as f32;
            }
        }
        p
    }

    struct Lcg(u64);
    impl Lcg {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / 16_777_216.0) * 12.0 - 6.0
        }
    }

    #[test]
    fn the_o_v_selection_reproduces_the_full_sort_bit_for_bit() {
        // Replacing a full vocabulary sort with a growing top-k selection is
        // only allowed to be faster, not different: the same seed must keep
        // emitting the same tokens, which means the truncated distribution has
        // to match the old one exactly — including which of two equally likely
        // tokens survives at the boundary.
        let mut rng = Lcg(0x00C0_FFEE_1234);
        for trial in 0..40 {
            // A vocabulary wider than GROW_START so the growth loop is exercised,
            // and flat enough that a large nucleus needs several rounds of it.
            let v = 200;
            let lo: Vec<f32> = (0..v)
                .map(|i| if trial % 3 == 0 { (i % 7) as f32 } else { rng.f32() })
                .collect();
            for &nucleus in &[0.0f32, 0.1, 0.5, 0.9, 0.95, 0.999, 1.0] {
                let want = reference_dist(&lo, 0.8, nucleus);
                let got = Sampler::new(0.8, nucleus, 1).distribution(&lo).to_vec();
                assert_eq!(got, want, "trial {trial}, nucleus {nucleus}: truncation changed");
            }
        }
    }

    #[test]
    fn top_k_keeps_exactly_k_and_composes_after_it() {
        let lo = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        // top-k alone: the k most likely survive, renormalized among themselves.
        let mut s = Sampler::new(1.0, 1.0, 1).with_top_k(2);
        let p = s.distribution(&lo).to_vec();
        assert_eq!(p.iter().filter(|&&x| x > 0.0).count(), 2);
        assert!(p[4] > 0.0 && p[5] > 0.0, "the two highest logits must be the survivors");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5, "and must be renormalized");

        // The nucleus then applies to the *renormalized* top-k, DFlash's order:
        // token 5 alone already holds ~73% of the two-token mass.
        let mut s = Sampler::new(1.0, 0.5, 1).with_top_k(2);
        let p = s.distribution(&lo).to_vec();
        assert_eq!(p.iter().filter(|&&x| x > 0.0).count(), 1);
        assert_eq!(p[5], 1.0);

        // A k at or past the vocabulary cuts nothing.
        let mut s = Sampler::new(1.0, 1.0, 1).with_top_k(99);
        assert!(s.distribution(&lo).iter().all(|&x| x > 0.0));
    }

    #[test]
    fn top_k_bounds_what_can_ever_be_sampled() {
        let lo = [5.0f32, 4.9, 4.8, 4.7, 4.6];
        let mut s = Sampler::new(2.0, 1.0, 77).with_top_k(2);
        for _ in 0..500 {
            assert!(s.pick(&lo, -1) < 2, "a token outside the top-2 was sampled");
        }
    }

    #[test]
    fn top_k_is_inert_under_greedy_decoding() {
        // Greedy is the top-1 already; a rank cutoff cannot move it, and must
        // not perturb the deterministic path that the token-exact gate runs on.
        let lo = [0.1f32, 0.9, 0.3];
        assert_eq!(Sampler::new(0.0, 0.9, 1).with_top_k(1).pick(&lo, -1), 1);
        assert_eq!(Sampler::new(0.0, 0.9, 1).with_top_k(3).pick(&lo, -1), 1);
    }

    #[test]
    fn support_is_none_only_when_nothing_was_cut() {
        let lo = [0.0f32, 1.0, 2.0, 3.0];
        let mut s = Sampler::new(1.0, 1.0, 1);
        s.distribution(&lo);
        assert_eq!(s.support(), None, "an untruncated distribution has no smaller form");

        let mut s = Sampler::new(1.0, 0.5, 1);
        s.distribution(&lo);
        let sup = s.support().unwrap_or(&[]);
        assert!(!sup.is_empty() && sup.len() < lo.len());
        assert_eq!(sup[0], 3, "the support is ordered by descending probability");
    }

    #[test]
    fn the_sparse_draw_is_the_dense_draw() {
        // `pick_with_draft_dist` is `pick_with_distribution` with a smaller `q`.
        // Same seed, same logits: same token, and the same distribution written
        // out. If these ever diverge, speculation stops being distribution-
        // preserving and no other test would say so.
        let mut rng = Lcg(0xFEED_BEEF);
        let lo: Vec<f32> = (0..300).map(|_| rng.f32()).collect();
        for &(temp, nucleus, k) in &[(0.7f32, 0.9f32, 0usize), (1.0, 1.0, 0), (1.0, 0.95, 32), (0.5, 1.0, 8)] {
            let mut dense = Sampler::new(temp, nucleus, 4242).with_top_k(k);
            let mut sparse = Sampler::new(temp, nucleus, 4242).with_top_k(k);
            for step in 0..25 {
                let (td, qd) = dense.pick_with_distribution(&lo);
                let (ts, qs) = sparse.pick_with_draft_dist(&lo);
                assert_eq!(td, ts, "temp {temp} nucleus {nucleus} k {k} step {step}: token");
                assert_eq!(qd, qs.to_dense(), "temp {temp} nucleus {nucleus} k {k} step {step}: q");
            }
        }
    }

    #[test]
    fn a_truncated_draft_distribution_is_orders_of_magnitude_smaller() {
        // The point of the whole exercise, stated as a number rather than a hope.
        // GLM-5.2's vocabulary, and a decode distribution shaped like a real one:
        // a decaying head over a floor the softmax puts nothing on.
        const V: usize = 154_880;
        let peaked: Vec<f32> = (0..V).map(|i| if i < 64 { 10.0 - i as f32 * 0.3 } else { -20.0 }).collect();
        let (_, q) = Sampler::new(1.0, 0.95, 9).pick_with_draft_dist(&peaked);
        assert!(q.bytes() < 4096, "a peaked draft q should be kilobytes, got {} B", q.bytes());
        assert_eq!(DraftDist::dense(vec![0.0; V]).bytes(), 619_520, "what it replaces");

        // Greedy drafts are one token wide.
        let (_, q) = Sampler::new(0.0, 0.95, 9).pick_with_draft_dist(&peaked);
        assert_eq!(q.support_len(), 1);

        // A **flat** distribution is the honest counter-case: a 0.95 nucleus over
        // one keeps most of the vocabulary, and the support form would be larger
        // than the dense one. It must fall back rather than grow.
        let flat = vec![0.0f32; V];
        let (_, q) = Sampler::new(1.0, 0.95, 9).pick_with_draft_dist(&flat);
        assert!(q.is_dense(), "a near-full support must not be stored as a support");
        assert!(q.bytes() <= 619_520, "and can never cost more than the dense form");

        // Which is what `top_k` is for: it bounds the support no matter how flat
        // the distribution underneath it is.
        let (_, q) = Sampler::new(1.0, 1.0, 9).with_top_k(64).pick_with_draft_dist(&flat);
        assert_eq!(q.support_len(), 64);
        assert_eq!(q.bytes(), 64 * 8);
        // With the nucleus on top it can only be tighter — 61 of 64 equal
        // masses reach 0.95, which is the composition order working.
        let (_, q) = Sampler::new(1.0, 0.95, 9).with_top_k(64).pick_with_draft_dist(&flat);
        assert_eq!(q.support_len(), 61);
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

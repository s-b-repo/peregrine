//! The distribution a speculative draft was drawn from, stored at its support.
//!
//! [`crate::speculative_sample`] needs `q(drafted)` and the residual `(p − q)+`,
//! and nothing else. It was handed a **dense** `q` — one `f32` per vocabulary
//! entry per draft step — because that is the shape a softmax comes out in, and
//! [`Model::mtp_draft_sampled`](crate::Model::mtp_draft_sampled) priced it
//! honestly in its own doc comment: *"a depth-`g` draft holds `g * vocab` floats
//! per sequence between ticks — ~2.4 MB per sequence at GLM-5.2's vocab and
//! `g = 4`"*. It then declined to fix it, for a reason that was correct at the
//! time: a sparse `q` alongside a dense one is *two* representations of the same
//! distribution, and the failure mode of speculation is exactly the two coming
//! apart.
//!
//! DFlash ([z-lab/dflash](https://github.com/z-lab/dflash), `dflash/model.py`)
//! answers that objection by making the sparse form the **only** form. Its
//! `_rejection_sample` takes `draft_probs` with an optional `draft_indices`, and
//! branches on `draft_indices is None` in the two places a support matters:
//!
//! ```text
//! q(drafted)  =  (draft_probs * (draft_indices == drafted)).sum(-1)
//! residual    =  target_probs.scatter_add_(0, draft_indices, -draft_probs).clamp_min_(0)
//! ```
//!
//! That is this type. There is no dense twin to drift from, because the dense
//! case is a variant of the same value (`idx: None`) rather than a parallel
//! code path — the same shape DFlash's `draft_indices: Tensor | None` has.
//!
//! # Why it is exactly equivalent, not an approximation
//!
//! `q` is zero off its support, so `q(t) = 0` for any `t` outside it and
//! `(p − q)+ = p` there. Both quantities the acceptance rule reads are therefore
//! unchanged by storing only the support. This is a change of *representation*;
//! [`crate::mtp::speculative_sample_at`] is gated on producing the same token as
//! the dense reference for the same inputs and uniforms.
//!
//! # What it costs, and what it saves
//!
//! [`Sampler`](crate::Sampler) already truncates to a nucleus and (since the
//! DFlash port) an optional top-k, so the support is *already* small — the
//! engine was storing a mostly-zero vector to hold it. At GLM-5.2's 154 880-entry
//! vocabulary a dense `q` is 619 520 B; a nucleus that keeps 64 entries is 512 B
//! in this form, and a greedy draft's one-hot is 8 B. The scan in
//! [`DraftDist::prob_of`] is `O(support)` against the dense `O(1)` index, which
//! at those sizes is a cache line or two against a 620 KB allocation per draft
//! step per sequence.

/// The distribution a draft token was drawn from.
///
/// Sparse (`idx = Some`) when the sampler truncated to a nucleus or a top-k,
/// which is every configuration that is not "sample the raw softmax"; dense
/// otherwise, because then the support really is the whole vocabulary and there
/// is nothing to compress.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DraftDist {
    /// Vocabulary this distribution is over — kept even in the sparse form so a
    /// shape fault (a `q` from a different vocabulary) is still detectable, which
    /// is what the dense form's `q.len()` used to provide.
    vocab: usize,
    /// Support, as vocabulary indices. `None` is the dense form, where the
    /// support is `0..vocab` and `p` is indexed directly.
    idx: Option<Vec<u32>>,
    /// Probabilities: one per entry of `idx`, or one per vocabulary entry.
    p: Vec<f32>,
}

impl DraftDist {
    /// A dense distribution over `p.len()` tokens. `p` is the sampler's own
    /// normalized buffer — the whole vocabulary, most of it usually zero.
    pub fn dense(p: Vec<f32>) -> DraftDist {
        DraftDist { vocab: p.len(), idx: None, p }
    }

    /// A distribution held at its support.
    ///
    /// Takes `(token, probability)` **pairs** rather than two vectors: the
    /// indices and the probabilities cannot be given different lengths, and no
    /// caller can zip them in the wrong order. The same reason
    /// [`Sampler::pick_with_draft_dist`](crate::Sampler::pick_with_draft_dist)
    /// draws and describes in one call.
    pub fn sparse<I: IntoIterator<Item = (u32, f32)>>(vocab: usize, support: I) -> DraftDist {
        let (idx, p): (Vec<u32>, Vec<f32>) = support.into_iter().unzip();
        DraftDist { vocab, idx: Some(idx), p }
    }

    /// The distribution greedy decoding samples from: all mass on one token.
    ///
    /// Two words of storage instead of a vocabulary of zeros with a single 1.0
    /// in it, which is what the dense form of this was.
    pub fn one_hot(vocab: usize, token: usize) -> DraftDist {
        match u32::try_from(token) {
            Ok(t) if token < vocab => DraftDist { vocab, idx: Some(vec![t]), p: vec![1.0] },
            // A token id that is out of range or past `u32` describes no
            // distribution. An empty support says exactly that, and
            // `accept_run_sampled` rejects a draft whose `q` gives it no mass
            // rather than accepting it on a fabricated one.
            _ => DraftDist { vocab, idx: Some(Vec::new()), p: Vec::new() },
        }
    }

    /// The vocabulary this distribution is over.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// How many tokens carry (possibly) non-zero probability. The whole
    /// vocabulary in the dense form.
    pub fn support_len(&self) -> usize {
        match &self.idx {
            Some(i) => i.len(),
            None => self.p.len(),
        }
    }

    /// Whether this is the dense form — for tests and the sparsity telemetry.
    pub fn is_dense(&self) -> bool {
        self.idx.is_none()
    }

    /// Heap bytes this distribution holds. The number the sparse form exists to
    /// reduce, so it is reportable rather than merely asserted.
    pub fn bytes(&self) -> usize {
        self.p.len() * std::mem::size_of::<f32>()
            + self.idx.as_ref().map_or(0, |i| i.len() * std::mem::size_of::<u32>())
    }

    /// `q(t)` — zero for any token off the support, which is what a truncated
    /// distribution assigns it.
    ///
    /// The sparse scan is DFlash's `(draft_probs * (draft_indices == token)).sum(-1)`
    /// written as a loop: the support is tens of entries, so this is a linear
    /// walk of one or two cache lines.
    pub fn prob_of(&self, t: usize) -> f32 {
        match &self.idx {
            None => self.p.get(t).copied().unwrap_or(0.0),
            Some(idx) => {
                let Ok(t) = u32::try_from(t) else { return 0.0 };
                idx.iter().position(|&i| i == t).and_then(|j| self.p.get(j).copied()).unwrap_or(0.0)
            }
        }
    }

    /// `resid -= q`, in place and elementwise — DFlash's
    /// `residual.scatter_add_(0, draft_indices, -draft_probs)`.
    ///
    /// In `f32`, deliberately: [`crate::speculative_sample`] forms the residual
    /// as `(p − q).max(0.0)` in `f32` before widening, and the sparse path has to
    /// produce the same bits to be a representation change rather than a second
    /// algorithm. Entries off the support are untouched, which is the same
    /// subtraction of zero.
    pub fn subtract_from(&self, resid: &mut [f32]) {
        match &self.idx {
            None => {
                for (r, &q) in resid.iter_mut().zip(self.p.iter()) {
                    *r -= q;
                }
            }
            Some(idx) => {
                for (&i, &q) in idx.iter().zip(self.p.iter()) {
                    if let Some(r) = resid.get_mut(i as usize) {
                        *r -= q;
                    }
                }
            }
        }
    }

    /// This distribution written out densely — the reference form, for tests
    /// that check the sparse path against [`crate::speculative_sample`].
    pub fn to_dense(&self) -> Vec<f32> {
        match &self.idx {
            None => self.p.clone(),
            Some(idx) => {
                let mut out = vec![0.0f32; self.vocab];
                for (&i, &q) in idx.iter().zip(self.p.iter()) {
                    if let Some(slot) = out.get_mut(i as usize) {
                        *slot = q;
                    }
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_and_dense_answer_the_same_two_questions() {
        // The only two things the acceptance rule asks a `q`.
        let dense = DraftDist::dense(vec![0.0, 0.25, 0.0, 0.75, 0.0]);
        let sparse = DraftDist::sparse(5, [(3u32, 0.75f32), (1, 0.25)]);
        for t in 0..6 {
            assert_eq!(dense.prob_of(t), sparse.prob_of(t), "q({t})");
        }
        let p = [0.2f32, 0.2, 0.2, 0.2, 0.2];
        let (mut a, mut b) = (p, p);
        dense.subtract_from(&mut a);
        sparse.subtract_from(&mut b);
        assert_eq!(a, b, "the residual must not depend on the representation");
    }

    #[test]
    fn the_sparse_form_is_the_smaller_one() {
        let dense = DraftDist::dense(vec![0.0; 154_880]);
        let sparse = DraftDist::sparse(154_880, (0u32..64).map(|i| (i, 1.0 / 64.0)));
        assert_eq!(dense.bytes(), 154_880 * 4);
        assert_eq!(sparse.bytes(), 64 * 8);
        assert_eq!(DraftDist::one_hot(154_880, 7).bytes(), 8);
    }

    #[test]
    fn one_hot_is_greedys_distribution() {
        let g = DraftDist::one_hot(4, 2);
        assert_eq!(g.to_dense(), vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(g.prob_of(2), 1.0);
        assert_eq!(g.prob_of(0), 0.0);
    }

    #[test]
    fn an_unrepresentable_token_yields_no_mass_rather_than_a_wrong_one() {
        // Out of range, and past u32: both describe no distribution, and must
        // not silently become "all the mass on token 0".
        for bad in [4usize, usize::MAX] {
            let g = DraftDist::one_hot(4, bad);
            assert_eq!(g.support_len(), 0);
            assert!((0..4).all(|t| g.prob_of(t) == 0.0));
        }
    }

    #[test]
    fn to_dense_round_trips_the_support() {
        let sparse = DraftDist::sparse(6, [(5u32, 0.5f32), (0, 0.5)]);
        assert_eq!(sparse.to_dense(), vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.5]);
        assert_eq!(sparse.support_len(), 2);
        assert!(!sparse.is_dense());
        assert_eq!(sparse.vocab(), 6);
    }
}

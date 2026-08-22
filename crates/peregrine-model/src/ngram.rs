//! Prompt-lookup drafting: a speculative draft source with no weights, no
//! forward pass, and no training.
//!
//! Every other draft source this engine has costs something before the verify
//! forward runs. The MTP head costs a full sparse-MoE layer per draft step —
//! on the streaming container that is ~300 MB of SSD at topk=8, *per step*,
//! because a draft runs at `s_n = 1` and gets no batch-union amortization. A
//! separate draft model would cost its own weights.
//!
//! This one costs a backward scan of the token history. It proposes the
//! continuation that followed the last time the current suffix appeared, which
//! is the right guess exactly when the output repeats something already in
//! context — quoted code, an edited file, a repeated identifier, a list the
//! model is walking. That is a large fraction of what this box actually serves.
//!
//! **Why it matters more here than on a compute-bound engine.** Both peregrine
//! and colibrì measured naive speculation as a net loss on streamed experts,
//! and for the same reason: a *rejected* draft still pays for its verify row's
//! expert reads. `docs/ideas-from-colibri.md` states the corollary — speculation on
//! a disk-bound engine only loses when drafts are rejected, so a draft source
//! whose acceptance approaches 1 sidesteps the failure mode entirely. A
//! literal repeat from context is that source.
//!
//! Correctness is not this module's problem: whatever it proposes is checked by
//! `accept_run` against the model's own argmax, so a bad guess costs a verify
//! row and never a token.

/// Longest suffix this drafter will try to match. Beyond ~3 the extra
/// specificity buys little — a 4-gram that repeats almost always contains a
/// 3-gram that repeats in the same place — while each extra length is another
/// backward scan.
pub const DEFAULT_MAX_NGRAM: usize = 3;

/// Shortest suffix it will accept as evidence. **Two, not one.** A single-token
/// match fires constantly (any repeated `,` or `the`) and predicts almost
/// nothing, and on the streaming track every one of those wrong drafts is a
/// verify row streaming its own expert union. The floor is where the
/// acceptance-≈1 argument above stops holding.
pub const DEFAULT_MIN_NGRAM: usize = 2;

/// Proposes drafts by finding the current suffix earlier in the same sequence.
#[derive(Clone, Copy, Debug)]
pub struct NgramDrafter {
    max_n: usize,
    min_n: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        NgramDrafter { max_n: DEFAULT_MAX_NGRAM, min_n: DEFAULT_MIN_NGRAM }
    }
}

impl NgramDrafter {
    /// A drafter matching suffixes of up to `max_n` tokens. `max_n` below
    /// [`DEFAULT_MIN_NGRAM`] disables it, which is the honest reading of
    /// "match shorter than the floor".
    pub fn new(max_n: usize) -> NgramDrafter {
        NgramDrafter { max_n, min_n: DEFAULT_MIN_NGRAM }
    }

    /// Whether this drafter can propose anything at all.
    pub fn is_enabled(&self) -> bool {
        self.max_n >= self.min_n
    }

    /// Draft up to `depth` tokens to follow `hist ++ [next]`.
    ///
    /// `next` is the engine's pending token — already emitted, fed at the next
    /// tick's row 0 — so it is part of the pattern being matched but is not
    /// itself in `hist` yet. Taking it as a separate argument avoids copying
    /// the whole history every tick just to append one element.
    ///
    /// Returns the tokens that followed the **most recent** earlier occurrence
    /// of the longest matching suffix, or empty when nothing matches. Most
    /// recent rather than first because locality is the whole signal: the loop
    /// body the model is repeating now is the one it wrote a moment ago, not
    /// the one from the top of the prompt.
    pub fn draft(&self, hist: &[i32], next: i32, depth: usize) -> Vec<i32> {
        if depth == 0 || !self.is_enabled() {
            return Vec::new();
        }
        // The sequence being matched is `hist ++ [next]`, indexed without
        // materializing it.
        let vlen = hist.len() + 1;
        let at = |i: usize| -> i32 {
            if i < hist.len() {
                hist[i]
            } else {
                next
            }
        };
        // Longest first: a 3-gram match is stronger evidence than the 2-gram
        // inside it, and finding it first means the weaker match is never used
        // when the stronger one exists.
        let longest = self.max_n.min(vlen.saturating_sub(1));
        for n in (self.min_n..=longest).rev() {
            let suffix_at = vlen - n;
            // A match must start strictly before the suffix, so its window
            // `start..start + n` lies entirely inside `hist` and can never
            // include `next` itself.
            let mut start = suffix_at;
            while start > 0 {
                start -= 1;
                if (0..n).all(|j| at(start + j) == at(suffix_at + j)) {
                    let from = start + n;
                    let take = depth.min(vlen - from);
                    if take > 0 {
                        return (from..from + take).map(at).collect();
                    }
                }
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_the_continuation_of_a_repeated_suffix() {
        // "1 2 3 4 5" earlier; the sequence now ends "… 3 4", so 5 follows.
        let d = NgramDrafter::new(3);
        let hist = [1i32, 2, 3, 4, 5, 9, 9, 1, 2, 3];
        assert_eq!(d.draft(&hist, 4, 2), vec![5, 9]);
    }

    #[test]
    fn the_pending_token_is_part_of_the_pattern() {
        // Without `next` in the suffix this matches on "…2" and predicts 3.
        // With it the suffix is "2 3" and the prediction is 4 — the point of
        // taking the pending token rather than drafting from `hist` alone.
        let d = NgramDrafter::new(3);
        let hist = [1i32, 2, 3, 4, 5, 8, 1, 2];
        assert_eq!(d.draft(&hist, 3, 1), vec![4]);
    }

    #[test]
    fn prefers_the_most_recent_occurrence() {
        // "7 7" is followed by 1 the first time and by 2 the second. Locality
        // says the recent continuation is the live one.
        let d = NgramDrafter::new(2);
        let hist = [7i32, 7, 1, 5, 5, 7, 7, 2, 6, 7];
        assert_eq!(d.draft(&hist, 7, 1), vec![2]);
    }

    #[test]
    fn prefers_a_longer_match_over_a_shorter_more_recent_one() {
        // A 2-gram "5 6" matches later in the history than the 3-gram
        // "4 5 6" does. The longer match is stronger evidence and must win,
        // or the length loop is pointless.
        let d = NgramDrafter::new(3);
        let hist = [4i32, 5, 6, 111, 0, 0, 5, 6, 222, 0, 4, 5];
        assert_eq!(d.draft(&hist, 6, 1), vec![111]);
        // With 3-grams disabled the same history gives the shorter match.
        assert_eq!(NgramDrafter::new(2).draft(&hist, 6, 1), vec![222]);
    }

    #[test]
    fn no_match_drafts_nothing() {
        let d = NgramDrafter::new(3);
        assert!(d.draft(&[1i32, 2, 3], 9, 4).is_empty());
    }

    #[test]
    fn never_reads_past_the_end_of_history() {
        // The most recent possible match ends one token short of the sequence,
        // so only one token follows it however deep the request. Asking for
        // four must yield one, not four — and must not read out of bounds.
        let d = NgramDrafter::new(2);
        assert_eq!(d.draft(&[7i32, 5, 5], 5, 4), vec![5]);
        // An earlier match has more room, and the drafter takes all of it —
        // "1 2" was followed by "7 1 2", so three tokens is the honest answer.
        assert_eq!(d.draft(&[1i32, 2, 7, 1], 2, 4), vec![7, 1, 2]);
    }

    #[test]
    fn degenerate_inputs_are_empty_not_panics() {
        let d = NgramDrafter::new(3);
        assert!(d.draft(&[], 1, 4).is_empty(), "no history");
        assert!(d.draft(&[1i32, 1, 1], 1, 0).is_empty(), "zero depth");
        assert!(!NgramDrafter::new(1).is_enabled(), "below the floor is off");
        assert!(d.draft(&[1i32], 1, 2).is_empty(), "one token cannot form a 2-gram match");
    }

    #[test]
    fn a_period_one_repeat_matches_but_does_not_extrapolate() {
        // "a a a a" — the suffix "a a" occurs one position back, and the
        // window arithmetic has to allow a match ending exactly where the
        // suffix begins. It yields one token, not two: this drafter only ever
        // *replays* what the history holds, it never invents a continuation.
        // A run of identical tokens therefore drafts one per tick rather than
        // filling the depth, which is the conservative behaviour — on the
        // streaming track an invented token that misses costs a verify row's
        // worth of expert reads.
        let d = NgramDrafter::new(2);
        assert_eq!(d.draft(&[5i32, 5, 5], 5, 4), vec![5]);
    }
}

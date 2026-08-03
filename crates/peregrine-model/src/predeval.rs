//! Predictor evaluation — scoring what the prefetch predictors *said* against what
//! the router then *did*.
//!
//! Every predictor in this engine is correctness-neutral, which means none of them
//! can be wrong in a way a test would catch. A predictor that has quietly degraded to
//! noise costs throughput and nothing else, and it costs it silently. This module is
//! the instrument that makes the difference visible: `COLI_PREDICT_EVAL=1` scores
//! each predictor's ranked candidates for layer `L+1` against the set layer `L+1`
//! actually routed, on the same forward, and reports recall and precision-by-rank at
//! shutdown.
//!
//! It exists because of a specific and expensive mistake made twice in this problem
//! space. WASTE measured a cross-layer co-occurrence predictor at 29.0 % recall@16,
//! computed a 60 % break-even from the read economics, and closed the question — and
//! then found (their `LEARNED.md` §29 vs §34) that they had measured *one* predictor
//! and concluded about *the question*. Asking the next layer's router directly scored
//! 59.0 % on the same trace. The shape of the data their format happened to reserve
//! had decided the shape of the experiment.
//!
//! The lesson taken here is not "the router look-ahead is better" — that is their
//! number, on their container, for their model. It is that a predictor's worth is
//! measurable cheaply and should not be argued about. Hence arms, not an arm.
//!
//! Pure: no model, no I/O, no clock. The engine feeds it ids.

use std::collections::HashMap;

/// One predictor's running score.
#[derive(Clone, Debug)]
struct Arm {
    name: String,
    /// `rank_hit[r]`: times the candidate offered at rank `r` was in the actual set.
    rank_hit: Vec<u64>,
    /// `rank_n[r]`: times *any* candidate was offered at rank `r`. Separate from
    /// `rank_hit` because a predictor may offer fewer than `width` candidates (an
    /// empty history has nothing to say), and dividing by `width` regardless would
    /// report a predictor that abstained as a predictor that guessed wrong.
    rank_n: Vec<u64>,
    /// Distinct actual experts this predictor's whole offer covered.
    covered: u64,
    /// Actual routed experts, summed over scored layers — recall's denominator.
    actual: u64,
    /// Layers this arm was asked about, including those where it offered nothing.
    asked: u64,
    /// Layers where it offered nothing at all (cold history, typically).
    silent: u64,
}

impl Arm {
    fn new(name: &str, width: usize) -> Arm {
        Arm {
            name: name.to_string(),
            rank_hit: vec![0; width],
            rank_n: vec![0; width],
            covered: 0,
            actual: 0,
            asked: 0,
            silent: 0,
        }
    }
}

/// A finished per-predictor score, for reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmReport {
    pub name: String,
    /// Fraction of the actually-routed experts the offer contained.
    pub recall: f64,
    /// Fraction of the offered candidates that were actually routed, over all ranks.
    pub precision: f64,
    /// `precision_at[r]`: fraction correct among candidates offered at rank `r`.
    /// This is the profile that decides how wide the look-ahead should be — a flat
    /// profile says take everything or nothing, a steep one says take the head.
    pub precision_at: Vec<f64>,
    /// Layers scored, and of those, layers where this predictor offered nothing.
    pub asked: u64,
    pub silent: u64,
}

/// Scores several predictors against the authoritative routing, layer by layer.
///
/// The engine [`stash`](Self::stash)es each predictor's ranked candidates for layer
/// `L+1` while it is still at layer `L`, then [`score`](Self::score)s them once layer
/// `L+1` has actually routed. Predictions that never get scored (the forward ended
/// first) are simply dropped.
#[derive(Debug)]
pub struct PredictEval {
    width: usize,
    arms: Vec<Arm>,
    /// Layer → one ranked candidate list per arm, awaiting that layer's real routing.
    pending: HashMap<usize, Vec<Vec<i32>>>,
    scored_layers: u64,
}

impl PredictEval {
    /// `width` candidates per arm are scored; `names` fixes the arm order that
    /// [`stash`](Self::stash) must then follow.
    pub fn new(width: usize, names: &[&str]) -> PredictEval {
        PredictEval {
            width,
            arms: names.iter().map(|n| Arm::new(n, width)).collect(),
            pending: HashMap::new(),
            scored_layers: 0,
        }
    }

    /// How many candidates each arm is scored on.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of arms — the length `stash` expects.
    pub fn arms(&self) -> usize {
        self.arms.len()
    }

    /// Record each arm's ranked candidates for `layer`, in the order `new` was given.
    ///
    /// A mis-sized `per_arm` is dropped rather than scored: attributing one
    /// predictor's candidates to another's name would corrupt every number this
    /// module exists to produce, and silently. Overwrites any unscored prediction
    /// for the same layer (the newer one is the one about to be tested).
    pub fn stash(&mut self, layer: usize, per_arm: Vec<Vec<i32>>) {
        if per_arm.len() != self.arms.len() {
            return;
        }
        self.pending.insert(layer, per_arm);
    }

    /// Score every arm's stashed prediction for `layer` against `actual` — the set
    /// that layer really routed. A layer with no stashed prediction, or with an empty
    /// actual set (a dense layer), is skipped.
    pub fn score(&mut self, layer: usize, actual: &[i32]) {
        let Some(per_arm) = self.pending.remove(&layer) else {
            return;
        };
        if actual.is_empty() {
            return;
        }
        self.scored_layers += 1;
        for (arm, pred) in self.arms.iter_mut().zip(per_arm) {
            arm.asked += 1;
            arm.actual += actual.len() as u64;
            if pred.is_empty() {
                arm.silent += 1;
                continue;
            }
            let mut covered = 0u64;
            for (rank, e) in pred.iter().take(self.width).enumerate() {
                arm.rank_n[rank] += 1;
                if actual.contains(e) {
                    arm.rank_hit[rank] += 1;
                    covered += 1;
                }
            }
            arm.covered += covered;
        }
    }

    /// Layers scored so far.
    pub fn scored_layers(&self) -> u64 {
        self.scored_layers
    }

    /// Per-arm scores. Empty when nothing has been scored — an evaluation with no
    /// evidence reports no number rather than a division by zero dressed as 0.0.
    pub fn report(&self) -> Vec<ArmReport> {
        if self.scored_layers == 0 {
            return Vec::new();
        }
        self.arms
            .iter()
            .map(|a| {
                let offered: u64 = a.rank_n.iter().sum();
                let hit: u64 = a.rank_hit.iter().sum();
                ArmReport {
                    name: a.name.clone(),
                    recall: ratio(a.covered, a.actual),
                    precision: ratio(hit, offered),
                    precision_at: a
                        .rank_hit
                        .iter()
                        .zip(&a.rank_n)
                        .map(|(&h, &n)| ratio(h, n))
                        .collect(),
                    asked: a.asked,
                    silent: a.silent,
                }
            })
            .collect()
    }
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfect_and_a_useless_arm_score_as_such() {
        let mut ev = PredictEval::new(4, &["oracle", "noise"]);
        let actual = vec![7i32, 3, 9, 1];
        ev.stash(5, vec![vec![7, 3, 9, 1], vec![100, 101, 102, 103]]);
        ev.score(5, &actual);
        let r = ev.report();
        assert_eq!(r[0].recall, 1.0);
        assert_eq!(r[0].precision, 1.0);
        assert_eq!(r[0].precision_at, vec![1.0; 4]);
        assert_eq!(r[1].recall, 0.0);
        assert_eq!(r[1].precision, 0.0);
    }

    #[test]
    fn precision_by_rank_is_what_separates_two_equal_recalls() {
        // The whole reason `precision_at` exists: these two arms have identical
        // recall, and only one of them is worth truncating to its head. A steeply
        // ranked predictor can be prefetched six deep; a flat one cannot.
        let mut ev = PredictEval::new(4, &["steep", "flat"]);
        let actual = vec![1i32, 2];
        //                        hits at ranks 0,1        hits at ranks 1,3
        ev.stash(0, vec![vec![1, 2, 90, 91], vec![90, 1, 91, 2]]);
        ev.score(0, &actual);
        let r = ev.report();
        assert_eq!(r[0].recall, r[1].recall, "same recall by construction");
        assert_eq!(r[0].precision_at, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(r[1].precision_at, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn an_arm_that_abstains_is_not_an_arm_that_guessed_wrong() {
        // A cold history offers nothing. That must show as `silent`, and must not
        // dilute the per-rank precision of the ranks it never filled — otherwise a
        // predictor looks worst exactly where it is honest about knowing nothing.
        let mut ev = PredictEval::new(3, &["cold"]);
        ev.stash(0, vec![vec![]]);
        ev.score(0, &[1, 2]);
        ev.stash(1, vec![vec![1, 2]]);
        ev.score(1, &[1, 2]);
        let r = ev.report();
        assert_eq!(r[0].asked, 2);
        assert_eq!(r[0].silent, 1);
        assert_eq!(r[0].precision_at[0], 1.0, "the one real offer at rank 0 was right");
        assert_eq!(r[0].precision_at[2], 0.0, "rank 2 was never offered");
        // Recall is over both layers: 2 covered of 4 actual.
        assert_eq!(r[0].recall, 0.5);
    }

    #[test]
    fn unscored_and_mismatched_predictions_are_dropped_not_misattributed() {
        let mut ev = PredictEval::new(2, &["a", "b"]);
        // Wrong arm count — dropping it is the only safe move; scoring it would
        // credit arm "a" with arm "b"'s candidates.
        ev.stash(0, vec![vec![1]]);
        ev.score(0, &[1]);
        assert!(ev.report().is_empty(), "nothing was validly stashed, so nothing is scored");
        // A layer that never routes (dense) leaves its prediction unscored.
        ev.stash(1, vec![vec![1], vec![2]]);
        ev.score(1, &[]);
        assert_eq!(ev.scored_layers(), 0);
    }

    #[test]
    fn duplicate_stash_keeps_the_newer_prediction() {
        // Each token re-predicts the same layer; the score must be against the
        // prediction that was actually live when the layer ran.
        let mut ev = PredictEval::new(2, &["a"]);
        ev.stash(3, vec![vec![90, 91]]);
        ev.stash(3, vec![vec![1, 2]]);
        ev.score(3, &[1, 2]);
        assert_eq!(ev.report()[0].recall, 1.0);
    }

    #[test]
    fn recall_counts_distinct_coverage_not_repeated_candidates() {
        // A predictor that offers the same right answer four times has covered one
        // expert, not four. Guarding this stops a degenerate arm from reporting
        // recall > 1.
        let mut ev = PredictEval::new(4, &["repeat"]);
        ev.stash(0, vec![vec![1, 1, 1, 1]]);
        ev.score(0, &[1, 2, 3, 4]);
        let r = ev.report();
        assert!(r[0].recall <= 1.0, "recall must not exceed 1, got {}", r[0].recall);
        assert_eq!(r[0].recall, 0.25, "one of four actual experts covered");
    }
}

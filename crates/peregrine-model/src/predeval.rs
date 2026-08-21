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

/// Name of the **control arm**: a predictor that is known to be worthless.
///
/// Every other arm here is one someone believed in, which means a scoreboard
/// that always reported "fine" would look exactly like a working one. Nothing
/// in this module had ever been shown to *detect* a bad predictor — only to
/// rank good ones — and an instrument that has never produced a negative is not
/// yet known to be able to.
///
/// So one arm is deliberately degraded: uniform-random expert ids at the same
/// emission rate as the real arms. Whatever recall it scores is the **floor**
/// that comes free from guessing, and a real arm's number only means something
/// as a margin over it. On a top-k of `k` out of `n` experts the expectation is
/// `k/n` — small, but not zero, and it is the "not zero" that makes reading a
/// raw recall figure misleading.
pub const CONTROL_ARM: &str = "control/noise";

/// Deterministic uniform-random candidates for the control arm.
///
/// Seeded from `(layer, token)` rather than a clock so two runs of the same
/// trace produce the same control. A control that varied run to run would let
/// anyone reroll until the real arms won, which is the same unfalsifiability
/// the control exists to remove.
pub fn control_candidates(width: usize, n_experts: usize, layer: usize, token: u64) -> Vec<i32> {
    if n_experts == 0 || width == 0 {
        return Vec::new();
    }
    // splitmix64 — a few lines, no dependency, and good enough that the control
    // is not accidentally correlated with expert id order.
    let mut z = (layer as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(token.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    // Distinct ids, like a real routed set: sampling with replacement would let
    // the control offer the same expert twice and understate its own floor.
    let mut out: Vec<i32> = Vec::with_capacity(width.min(n_experts));
    let mut guard = 0;
    while out.len() < width.min(n_experts) && guard < width * 16 + 64 {
        let e = (next() % n_experts as u64) as i32;
        if !out.contains(&e) {
            out.push(e);
        }
        guard += 1;
    }
    out
}

/// What the control arm says about the scoreboard itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Separation {
    /// Recall of the best non-control arm.
    pub best_real: f64,
    /// Recall the control arm got for free.
    pub control: f64,
    /// Name of the best non-control arm.
    pub best_name: String,
}

impl Separation {
    /// Best real arm's recall as a multiple of the control's. `<= 1` means the
    /// scoreboard cannot distinguish the predictors it is scoring from noise.
    pub fn ratio(&self) -> f64 {
        if self.control > 0.0 {
            self.best_real / self.control
        } else if self.best_real > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// The verdict about the **instrument**, not about the predictors.
    pub fn verdict(&self) -> String {
        let r = self.ratio();
        let head = format!(
            "[predict-eval] separation: best real arm `{}` recall {:.3} vs control/noise {:.3} ({:.1}x)\n",
            self.best_name, self.best_real, self.control, r
        );
        // 2x is a low bar on purpose. It is not "is the predictor good", it is
        // "can this scoreboard tell a predictor from a coin flip at all". A
        // scoreboard that fails this cannot support any of its other columns.
        if r >= 2.0 {
            format!("{head}[predict-eval] the scoreboard separates signal from noise; its other columns are readable.\n")
        } else {
            format!(
                "{head}[predict-eval] WARNING: the best real arm is within 2x of uniform noise. \
                 Either every predictor has degraded, or this scoreboard cannot discriminate at \
                 this width and sample count — and its recall columns should not be quoted until \
                 which one is known.\n"
            )
        }
    }
}

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

    /// Layers scored so far. Used as the varying half of the control arm's
    /// seed: it advances once per scored layer, so the control differs between
    /// layers and between tokens without the forward loop having to thread a
    /// position through.
    pub fn scored(&self) -> u64 {
        self.scored_layers
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
            let mut seen: Vec<i32> = Vec::with_capacity(self.width);
            let mut covered = 0u64;
            for (rank, e) in pred.iter().take(self.width).enumerate() {
                arm.rank_n[rank] += 1;
                if actual.contains(e) {
                    arm.rank_hit[rank] += 1;
                    // Recall counts *distinct* coverage. An arm that offers the same
                    // right answer twice has covered one expert, and counting it twice
                    // would let a degenerate arm report recall above 1.
                    if !seen.contains(e) {
                        covered += 1;
                    }
                }
                seen.push(*e);
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
    /// Compare the best real arm against the control arm.
    ///
    /// `None` when no control arm is present, which is itself worth surfacing:
    /// a scoreboard without one is reporting recall figures whose floor nobody
    /// has measured.
    pub fn separation(&self) -> Option<Separation> {
        let reports = self.report();
        let control = reports.iter().find(|r| r.name == CONTROL_ARM)?;
        let best = reports
            .iter()
            .filter(|r| r.name != CONTROL_ARM)
            .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(Separation {
            best_real: best.recall,
            control: control.recall,
            best_name: best.name.clone(),
        })
    }

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

#[cfg(test)]
mod control_tests {
    use super::*;

    #[test]
    fn the_control_is_deterministic_and_offers_distinct_experts() {
        // Determinism is what makes the control falsifiable: a control that
        // varied run to run could be rerolled until the real arms won.
        let a = control_candidates(8, 64, 3, 11);
        let b = control_candidates(8, 64, 3, 11);
        assert_eq!(a, b, "same (layer, token) must give the same control");
        assert_ne!(a, control_candidates(8, 64, 4, 11), "a different layer must differ");
        assert_ne!(a, control_candidates(8, 64, 3, 12), "a different token must differ");
        assert_eq!(a.len(), 8, "the control must offer at the same rate as the real arms");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), a.len(), "duplicate ids would understate the control's own floor");
        assert!(a.iter().all(|&e| (0..64).contains(&e)), "ids must be in range: {a:?}");
    }

    #[test]
    fn the_control_cannot_offer_more_experts_than_exist() {
        // Otherwise the loop that samples distinct ids spins to its guard.
        let c = control_candidates(16, 4, 0, 0);
        assert_eq!(c.len(), 4, "width is capped by the expert count: {c:?}");
        assert!(control_candidates(8, 0, 0, 0).is_empty(), "no experts means no candidates");
    }

    #[test]
    fn a_scoreboard_that_cannot_beat_noise_says_so() {
        // The instrument's own verdict. A real arm scoring at the noise floor
        // means either every predictor has degraded or the scoreboard cannot
        // discriminate — and the report must refuse to be quoted either way,
        // rather than printing a recall column that looks like a measurement.
        let mut ev = PredictEval::new(2, &["real", CONTROL_ARM]);
        for layer in 0..50 {
            // Both arms guess the same wrong thing: no separation exists.
            ev.stash(layer, vec![vec![90, 91], vec![92, 93]]);
            ev.score(layer, &[1, 2]);
        }
        assert!(ev.separation().is_some(), "a control arm is present");
        let sep = ev.separation().unwrap_or(Separation {
            best_real: -1.0,
            control: -1.0,
            best_name: String::new(),
        });
        assert_eq!(sep.control, 0.0);
        assert!(sep.verdict().contains("WARNING"), "{}", sep.verdict());
        assert!(sep.verdict().contains("should not be quoted"), "{}", sep.verdict());
    }

    #[test]
    fn a_scoreboard_that_separates_says_that_too() {
        let mut ev = PredictEval::new(2, &["real", CONTROL_ARM]);
        for layer in 0..50 {
            ev.stash(layer, vec![vec![1, 2], vec![90, 91]]);
            ev.score(layer, &[1, 2]);
        }
        assert!(ev.separation().is_some(), "a control arm is present");
        let sep = ev.separation().unwrap_or(Separation {
            best_real: -1.0,
            control: -1.0,
            best_name: String::new(),
        });
        assert!((sep.best_real - 1.0).abs() < 1e-9, "the real arm caught everything");
        assert_eq!(sep.best_name, "real");
        assert!(sep.ratio() > 2.0);
        assert!(sep.verdict().contains("separates signal from noise"), "{}", sep.verdict());
    }

    #[test]
    fn separation_is_none_without_a_control_arm() {
        // Surfacing the absence rather than defaulting to "fine": a scoreboard
        // with no control has an unmeasured floor under every number it prints.
        let mut ev = PredictEval::new(2, &["a", "b"]);
        ev.stash(0, vec![vec![1], vec![2]]);
        ev.score(0, &[1, 2]);
        assert!(ev.separation().is_none());
    }

    #[test]
    fn the_control_scores_above_zero_when_it_guesses_widely() {
        // The floor is small but NOT zero, which is the whole reason a raw
        // recall figure is misleading: offering half the experts catches half
        // the routed set by construction.
        let mut ev = PredictEval::new(8, &["real", CONTROL_ARM]);
        for layer in 0..100u64 {
            let c = control_candidates(8, 16, layer as usize, layer);
            ev.stash(layer as usize, vec![vec![0], c]);
            ev.score(layer as usize, &[1, 2, 3, 4]);
        }
        assert!(ev.separation().is_some(), "control present");
        let sep = ev.separation().unwrap_or(Separation {
            best_real: -1.0,
            control: -1.0,
            best_name: String::new(),
        });
        assert!(sep.control > 0.0, "uniform guessing over 16 experts must catch some of 4");
    }
}

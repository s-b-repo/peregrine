//! RLM (Recursive Language Model) controller — decides when a token's
//! forward pass needs additional recursive reasoning passes before the
//! logits are sampled.
//!
//! Conceptually, after the main forward produces hidden and logits at
//! step `t`, the RLM controller inspects the logits' top-2 confidence
//! margin (for greedy) or distribution entropy (for sampled) and, if the
//! pass was "uncertain", triggers one or more recursive passes. Each
//! recursive pass re-runs the hidden state through a configurable subset
//! of transformer layers, allowing the MoE router to potentially select
//! different experts on the second pass (since the hidden has been refined).
//!
//! This is correctness-neutral when off (`COLI_RLM` unset or = 0): the
//! `generate()` loop checks `should_recurse()` which returns `false`, so
//! no extra passes run and the output is bit-identical to the non-RLM path.
//!
//! When on, the recursive passes modify the logits before sampling. The
//! emitted token stream may differ from non-RLM decode — this is the
//! intended tradeoff (extra compute for better quality on hard steps).
//!
//! The actual recursive forward pass lives on `Model` (it needs access to
//! private fields like `layers`, `kv`, `st`, etc.). This module provides
//! the decision/policy logic and env-gating.

/// Whether RLM is globally active (`COLI_RLM=1`). Off by default.
pub fn rlm_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_RLM").as_deref(), Ok("1") | Ok("true")))
}

/// Maximum recursive passes per token (`COLI_RLM_DEPTH`, default 2, cap 4).
pub fn rlm_max_depth() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_RLM_DEPTH")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 4)
    })
}

/// How many layers the recursive pass re-runs (`COLI_RLM_LAYERS`, default 4).
/// Fewer layers = cheaper recursion but less refinement. Must be <= n_layers.
pub fn rlm_layers() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_RLM_LAYERS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(4)
            .max(1)
    })
}

/// Confidence margin below which (in the top-2 logit gap) a token is
/// considered "hard" and eligible for recursive refinement
/// (`COLI_RLM_MARGIN`, default 0.1). Only applies to greedy decode (temp <= 0).
pub fn rlm_margin() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_RLM_MARGIN")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|f| f.is_finite())
            .unwrap_or(0.1)
    })
}

/// The RLM controller state.
///
/// Holds the recursion counter for the current token-in-progress and
/// accumulates per-session statistics (how often recursion fired, how many
/// passes total) for telemetry.
pub struct RLMController {
    /// Current depth of recursion for the token being generated (reset to 0
    /// at the start of each new token).
    depth: usize,
    /// Total recursive passes triggered across the session.
    passes_emitted: u64,
    /// Total tokens that triggered at least one recursive pass.
    tokens_recursed: u64,
}

impl RLMController {
    /// Create a new controller. If `COLI_RLM` is not set, the controller is
    /// inert — `should_recurse` always returns `false`.
    pub fn new() -> Self {
        RLMController {
            depth: 0,
            passes_emitted: 0,
            tokens_recursed: 0,
        }
    }

    /// Reset depth at the start of a new token.
    pub fn reset(&mut self) {
        self.depth = 0;
    }

    /// Decide whether to trigger a recursive pass for the current token,
    /// based on the greedy confidence margin in `logits`.
    ///
    /// Returns `true` if:
    /// - RLM is globally enabled
    /// - We have not exceeded `rlm_max_depth()` passes for this token
    /// - The top-2 logit gap is below `rlm_margin()` (i.e., the model is
    ///   uncertain between its top candidates)
    ///
    /// For non-greedy (temp > 0) sampling, the entropy of the distribution
    /// is used instead of the top-2 margin.
    pub fn should_recurse(&mut self, logits: &[f32], temp: f32) -> bool {
        if !rlm_enabled() {
            return false;
        }
        if self.depth >= rlm_max_depth() {
            return false;
        }

        let need_recurse = if temp <= 0.0 {
            // Greedy: check top-2 margin.
            let (top1, top2) = top_two_logits(logits);
            (top1 - top2) < rlm_margin()
        } else {
            // Sampled: check distribution entropy (normalized).
            let ent = normalized_entropy(logits, temp);
            ent > 0.5 // threshold: high entropy = uncertain
        };

        if need_recurse {
            self.depth += 1;
            self.passes_emitted += 1;
            if self.depth == 1 {
                self.tokens_recursed += 1;
            }
            true
        } else {
            false
        }
    }

    /// Current recursion depth for the active token.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Session telemetry: total recursive passes triggered.
    pub fn passes_emitted(&self) -> u64 {
        self.passes_emitted
    }

    /// Session telemetry: total tokens that triggered at least one pass.
    pub fn tokens_recursed(&self) -> u64 {
        self.tokens_recursed
    }
}

impl Default for RLMController {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the top two logit values (descending). The rest of the logits are
/// irrelevant — we only need the gap to judge confidence.
fn top_two_logits(logits: &[f32]) -> (f32, f32) {
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &l in logits {
        if l > top1 {
            top2 = top1;
            top1 = l;
        } else if l > top2 {
            top2 = l;
        }
    }
    (top1, top2)
}

/// Normalized Shannon entropy of the softmax distribution at temperature `temp`.
/// Returns 0.0 for a degenerate (one-hot) distribution, 1.0 for uniform.
/// Uses log-sum-exp for numerical stability.
fn normalized_entropy(logits: &[f32], temp: f32) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let inv_temp = 1.0 / temp.max(1e-4);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0f64;
    for &l in logits {
        sum_exp += (((l - max) * inv_temp) as f64).exp();
    }
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return 0.0;
    }
    let mut ent = 0f64;
    for &l in logits {
        let p = (((l - max) * inv_temp) as f64).exp() / sum_exp;
        if p > 0.0 {
            ent -= p * p.ln();
        }
    }
    let max_ent = (logits.len() as f64).ln();
    if max_ent <= 0.0 {
        return 0.0;
    }
    (ent / max_ent) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlm_disabled_by_default() {
        assert!(!rlm_enabled(), "RLM must be off by default for correctness");
    }

    #[test]
    fn top_two_logits_finds_correct_gap() {
        let logits = [1.0f32, 5.0, 3.0, 2.0];
        let (t1, t2) = top_two_logits(&logits);
        assert_eq!(t1, 5.0);
        assert_eq!(t2, 3.0);
    }

    #[test]
    fn top_two_logits_all_negative() {
        let logits = [-5.0f32, -1.0, -3.0];
        let (t1, t2) = top_two_logits(&logits);
        assert_eq!(t1, -1.0);
        assert_eq!(t2, -3.0);
    }

    #[test]
    fn normalized_entropy_uniform_is_one() {
        let logits = [1.0f32; 4];
        let ent = normalized_entropy(&logits, 1.0);
        assert!(
            (ent - 1.0).abs() < 0.01,
                 "uniform should have entropy near 1.0, got {ent}"
        );
    }

    #[test]
    fn normalized_entropy_peaked_is_near_zero() {
        let logits = [100.0f32, 0.0, 0.0, 0.0];
        let ent = normalized_entropy(&logits, 1.0);
        assert!(
            ent < 0.01,
                 "peaked distribution should have entropy near 0, got {ent}"
        );
    }

    #[test]
    fn controller_inert_when_disabled() {
        let mut c = RLMController::new();
        let logits = [1.0f32, 0.9, 0.8]; // very close gap — would trigger if enabled
        assert!(
            !c.should_recurse(&logits, 0.0),
                 "must be inert when COLI_RLM is off"
        );
        assert_eq!(c.depth(), 0);
        assert_eq!(c.passes_emitted(), 0);
    }

    // The integration smoke test belongs in `rlm.rs` rather than `model.rs`
    // because the integration is the unit of value here: keeping the wiring alive
    // by a single stale `mod` declaration would have silently regressed RLM had
    // that exact accident already happened (it did — Brownfield bug: `rlm.rs`
    // sat on disk without `pub mod rlm;` in `lib.rs` for ~2 weeks; the module's
    // tests never executed). The smoke rebuilds the tiny model and exercises the
    // `Model→forward_hidden_recursive→rlm_stats` surface directly, without env
    // mutation (see `mla_absorb_knob_is_inert_when_off_and_live_when_on` for why
    // `cargo test`'s parallel threads make env-var tests dangerous here).
    #[test]
    fn forward_hidden_recursive_runs_on_the_tiny_model() -> Result<(), peregrine_core::Error> {
        use crate::testkit::build_tiny_model;
        use crate::Model;
        let dir = std::env::temp_dir().join(format!(
            "peregrine_rlm_smoke_{}_{}",
            std::process::id(),
            std::line!()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let m = Model::load(&dir)?;
        let d = m.cfg.hidden as usize;
        let vocab = m.cfg.vocab as usize;
        let mut h = vec![0.5f32; d];
        let lg = m.forward_hidden_recursive(&mut h, 1, 0)?;
        assert_eq!(lg.len(), vocab, "RLM recursive pass must emit exactly [vocab] logits");
        assert!(lg.iter().all(|v| v.is_finite()), "recursive pass logits must be finite");
        assert!(h.iter().all(|v| v.is_finite()), "refined hidden must remain finite");
        assert_eq!(m.rlm_stats(), (0, 0), "fresh model: no recursive passes yet");
        // The audit treats `.ok();` on a `Result` as silent error swallowing
        // (BAD_PATTERNS.md §B). The other temp-dir tests in this crate use `?`
        // for cleanup, so we reciprocate — the temp dir is in the process's
        // `std::env::temp_dir()`, so a left-over is harmless but not silent.
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

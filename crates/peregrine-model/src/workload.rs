//! Workload classification & phase detection.
//!
//! Two independent observables:
//!
//! - [`TokenClass`] — a coarse bucket over recent tokens by their id-space
//!   statistics (proxying "is this code / prose / JSON / math?"). The HTTP
//!   handler is the natural producer since it has the tokenizer; the engine
//!   consumes a class tag from the request.
//!
//! - [`PhaseTracker`] — an EWMA over frame-to-frame routing-set Jaccard
//!   distance. Sharp spikes mark a phase transition (a topic/task shift), which
//!   the prefetch predictor uses to temporarily inflate momentum weight so it
//!   catches up faster.
//!
//! Both are correctness-neutral. They only steer prefetch/eviction/admission,
//! never the numerics.

use std::collections::HashSet;

/// A coarse workload class inferred from the recent token stream. Deliberately
/// broad — the class is a *prefetch hint*, not a routing decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TokenClass {
    #[default]
    Prose,
    Code,
    Json,
    Math,
    Mixed,
}

/// Bucketize a short string over a raw slice of decoded characters. Meant for
/// the *tail* of the prompt (say, the last 128 characters — long enough to
/// be representative, short enough to be O(1)). Deterministic and stateless.
pub fn classify_str(tail: &str) -> TokenClass {
    if tail.trim().is_empty() {
        return TokenClass::Prose;
    }
    let (mut alpha, mut digits, mut punct, mut brace_curly, mut brace_paren, mut brace_bracket, mut ws) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let mut math_ops = 0u32;
    for ch in tail.chars() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => ws += 1,
            c if c.is_ascii_alphabetic() => alpha += 1,
            c if c.is_ascii_digit() => digits += 1,
            '{' | '}' => brace_curly += 1,
            '(' | ')' => brace_paren += 1,
            '[' | ']' => brace_bracket += 1,
            '+' | '-' | '*' | '/' | '=' | '<' | '>' | '^' => math_ops += 1,
            c if c.is_ascii_punctuation() => punct += 1,
            _ => {}
        }
    }
    let total = alpha + digits + punct + brace_curly + brace_paren + brace_bracket + ws + math_ops;
    if total == 0 {
        return TokenClass::Prose;
    }
    let punct_ratio = (punct + brace_curly + brace_paren + brace_bracket) as f32 / total as f32;
    let digit_ratio = digits as f32 / total as f32;
    let math_ratio = math_ops as f32 / total as f32;
    // JSON: curly braces present AND a dense punctuation stream (quotes,
    // colons, commas — all counted as `punct`). Short JSON snippets rarely
    // exceed a 10 % curly-brace ratio, so gate on presence rather than share.
    if brace_curly >= 2 && punct_ratio > 0.30 {
        return TokenClass::Json;
    }
    // Math: heavy on operators + digits.
    if math_ratio > 0.10 && digit_ratio > 0.10 {
        return TokenClass::Math;
    }
    // Code: many brackets/parens or high punctuation.
    if brace_paren + brace_bracket > 3 || punct_ratio > 0.15 {
        return TokenClass::Code;
    }
    // Mixed vs Prose: if the alpha share is modest, it is likely mixed content.
    if alpha as f32 / total as f32 > 0.60 {
        TokenClass::Prose
    } else {
        TokenClass::Mixed
    }
}

/// Phase-shift detector over the routed-expert sets a stream produces frame by
/// frame. Maintains an EWMA of the Jaccard distance between consecutive
/// frames' top-k sets — an abrupt jump signals a topic/task transition and
/// prompts the predictor to re-weight recency.
pub struct PhaseTracker {
    ewma: f32,
    alpha: f32,
    /// Threshold on the *instantaneous* Jaccard distance above which a phase
    /// change is declared. Tunable via `COLI_PHASE_THRESHOLD` (default 0.6).
    threshold: f32,
    /// Number of forwards since the last declared phase change (used to
    /// re-inflate momentum weight for a short window).
    since_change: u32,
    last_set: Option<Vec<i32>>,
}

impl PhaseTracker {
    pub fn new() -> PhaseTracker {
        let threshold = std::env::var("COLI_PHASE_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|&v: &f32| (0.0..=1.0).contains(&v))
            .unwrap_or(0.6);
        PhaseTracker { ewma: 0.0, alpha: 0.3, threshold, since_change: u32::MAX, last_set: None }
    }

    /// Feed one layer's routed set for this token. Returns `true` if a phase
    /// change was detected on this frame.
    pub fn observe(&mut self, routed: &[i32]) -> bool {
        let d = self.jaccard_distance(routed);
        self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * d;
        self.since_change = self.since_change.saturating_add(1);
        let changed = d > self.threshold;
        if changed {
            self.since_change = 0;
        }
        self.last_set = Some(routed.to_vec());
        changed
    }

    /// Frames since the last declared phase change (`u32::MAX` before the first
    /// observation).
    pub fn since_change(&self) -> u32 {
        self.since_change
    }

    /// Current EWMA of frame-over-frame Jaccard distance (0 → identical, 1 →
    /// fully disjoint).
    pub fn stability(&self) -> f32 {
        self.ewma
    }

    fn jaccard_distance(&self, cur: &[i32]) -> f32 {
        let Some(prev) = &self.last_set else { return 0.0 };
        if cur.is_empty() && prev.is_empty() {
            return 0.0;
        }
        let a: HashSet<i32> = prev.iter().copied().collect();
        let b: HashSet<i32> = cur.iter().copied().collect();
        let inter = a.intersection(&b).count();
        let uni = a.union(&b).count();
        if uni == 0 {
            0.0
        } else {
            1.0 - (inter as f32 / uni as f32)
        }
    }
}

impl Default for PhaseTracker {
    fn default() -> PhaseTracker {
        PhaseTracker::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_json_and_code() {
        assert_eq!(classify_str(r#"{"key": "value", "n": 42}"#), TokenClass::Json);
        assert_eq!(classify_str("fn foo(x: i32) -> i32 { x + 1 }"), TokenClass::Code);
    }

    #[test]
    fn classify_math_and_prose() {
        assert_eq!(classify_str("2 + 3 = 5 and 6 * 7 = 42"), TokenClass::Math);
        assert_eq!(classify_str("The quick brown fox jumps over the lazy dog."), TokenClass::Prose);
    }

    #[test]
    fn phase_tracker_flags_shift() {
        let mut p = PhaseTracker::new();
        // Steady-state: same experts every frame → no change flagged after the first.
        assert!(!p.observe(&[1, 2, 3]));
        assert!(!p.observe(&[1, 2, 3]));
        // Big shift: fully disjoint set → change flagged.
        let changed = p.observe(&[10, 11, 12]);
        assert!(changed, "disjoint routing set is a phase change");
        assert_eq!(p.since_change(), 0);
    }
}

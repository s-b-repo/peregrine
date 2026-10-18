//! Eviction and memory residency — research-backed additions.
//!
//! Implements the three residency items from the consolidated gap analysis:
//! * reactivation-aware eviction (probability an evicted item is needed again),
//! * partial-session eviction (evict part of a session footprint, not whole sessions),
//! * a near-LRU recency-weighted baseline (which production traces show beats
//!   clever workload-aware policies in agentic settings).

use std::collections::HashMap;

/// Predictor for the probability that an evicted memory item will be needed
/// again soon. Importance is not always monotonic: an item evicted as "low
/// value" can resurface and cost more to reload than it saved.
#[derive(Debug, Clone)]
pub struct ReactivationPredictor {
    pub recency_weight: f32,
    pub frequency_weight: f32,
    pub utility_weight: f32,
    pub bias: f32,
}

impl Default for ReactivationPredictor {
    fn default() -> Self {
        Self {
            recency_weight: 0.6,
            frequency_weight: 0.3,
            utility_weight: 0.2,
            bias: -0.5,
        }
    }
}

impl ReactivationPredictor {
    /// Probability in `[0, 1]` that an item will be reactivated soon.
    pub fn probability(&self, recency: f32, frequency: f32, utility: f32) -> f32 {
        let z = self.bias
            + self.recency_weight * recency
            + self.frequency_weight * frequency
            + self.utility_weight * utility;
        1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
    }

    pub fn predict(&self, recency: f32, frequency: f32, utility: f32) -> f32 {
        self.probability(recency, frequency, utility)
    }

    pub fn reactivation_probability(&self, recency: f32, frequency: f32, utility: f32) -> f32 {
        self.probability(recency, frequency, utility)
    }
}

/// Eviction policy selection. Defaults to near-LRU per production traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    Lru,
    ReactivationAware,
    #[default]
    RecencyWeighted,
}

/// A candidate memory item considered for eviction.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub id: String,
    pub session: String,
    pub recency: f32,
    pub frequency: f32,
    pub utility: f32,
    pub cost: usize,
}

/// Partial-session eviction: evict part of a session's footprint rather than
/// only whole sessions, since real-world session sizes are heavy-tailed.
#[derive(Debug, Clone, Default)]
pub struct PartialSessionEvictor {
    pub predictor: ReactivationPredictor,
}

impl PartialSessionEvictor {
    /// Returns the subset of candidates that should be evicted to free at
    /// least `budget` units, preferring low-reactivation items.
    pub fn evict_part(
        &self,
        candidates: &[EvictionCandidate],
        budget: usize,
    ) -> Vec<String> {
        let mut scored: Vec<(&EvictionCandidate, f32)> = candidates
            .iter()
            .map(|c| {
                let p = self.predictor.probability(c.recency, c.frequency, c.utility);
                (c, p)
            })
            .collect();
        // Evict lowest reactivation probability first.
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.id.cmp(&b.0.id)));
        let mut freed = 0usize;
        let mut out = Vec::new();
        for (c, _) in scored {
            if freed >= budget {
                break;
            }
            out.push(c.id.clone());
            freed += c.cost;
        }
        out
    }

    pub fn evict(&self, candidates: &[EvictionCandidate], budget: usize) -> Vec<String> {
        self.evict_part(candidates, budget)
    }
}

/// Recency-weighted baseline: near-LRU ordering by recency alone.
pub fn recency_weighted_order(candidates: &[EvictionCandidate]) -> Vec<String> {
    let mut v: Vec<&EvictionCandidate> = candidates.iter().collect();
    v.sort_by(|a, b| b.recency.total_cmp(&a.recency).then(a.id.cmp(&b.id)));
    v.into_iter().map(|c| c.id.clone()).collect()
}

// --- Aliases for naming variants -------------------------------------------

pub type ReactivationAwareEviction = ReactivationPredictor;
pub type PartialSessionEviction = PartialSessionEvictor;
pub type RecencyWeightedBaseline = EvictionPolicy;

/// Session footprint helper for partial eviction bookkeeping.
#[derive(Debug, Clone, Default)]
pub struct SessionFootprint {
    pub session: String,
    pub items: Vec<String>,
    pub bytes: usize,
}

impl SessionFootprint {
    pub fn evict_part(&self, count: usize) -> Vec<String> {
        self.items.iter().take(count).cloned().collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrossSessionEvictionPriority {
    pub weights: HashMap<String, f32>,
}

impl CrossSessionEvictionPriority {
    pub fn priority_for(&self, session: &str) -> f32 {
        self.weights.get(session).copied().unwrap_or(0.0)
    }
}

// Unused import guard.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactivation_probability_in_range() {
        let p = ReactivationPredictor::default();
        let x = p.probability(0.5, 0.5, 0.5);
        assert!((0.0..=1.0).contains(&x));
    }

    #[test]
    fn partial_eviction_frees_budget() {
        let e = PartialSessionEvictor::default();
        let cands = vec![
            EvictionCandidate { id: "a".into(), session: "s".into(), recency: 0.1, frequency: 0.1, utility: 0.1, cost: 5 },
            EvictionCandidate { id: "b".into(), session: "s".into(), recency: 0.9, frequency: 0.9, utility: 0.9, cost: 5 },
        ];
        let out = e.evict_part(&cands, 5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], "a");
    }

    #[test]
    fn recency_order_descending() {
        let cands = vec![
            EvictionCandidate { id: "a".into(), session: "s".into(), recency: 0.1, frequency: 0.0, utility: 0.0, cost: 1 },
            EvictionCandidate { id: "b".into(), session: "s".into(), recency: 0.9, frequency: 0.0, utility: 0.0, cost: 1 },
        ];
        let order = recency_weighted_order(&cands);
        assert_eq!(order[0], "b");
    }
}

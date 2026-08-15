//! Topic-based smart routing: per-[`crate::workload::TokenClass`] expert-usage
//! profiles that steer warm-cache residency toward the experts the *current*
//! topic actually reuses.
//!
//! The problem this addresses: the warm cache's eviction protection already
//! favours experts the per-stream predictor expects to reuse, with global
//! routing *heat* as the tiebreak — but heat is a single distribution over all
//! traffic, and it only exists when a GPU tier does. On a CPU-only box, and
//! across a mixed request stream, two equally-predicted experts are
//! indistinguishable to the evictor even when one is hot *for this topic* and
//! the other is not.
//!
//! [`TopicProfiles`] accumulates, per class, how often each `(layer, expert)`
//! was routed while that class was the active workload. The evictor then breaks
//! ties by the *active topic's* frequency instead of the global one, so a
//! coding request keeps coding-hot experts resident through an interleaved
//! prose request and vice-versa. It is correctness-neutral by construction —
//! the profile only reorders eviction victims, never what a `get`/`insert`
//! returns — and works with no GPU tier. Off unless `COLI_TOPIC_ROUTING=1`.
//!
//! Accumulation is once-per-decode-forward, off the hot path (the same cadence
//! as cache-protection), so the per-token cost is ~one increment per routed
//! expert. Profiles persist to and load from a `topic_profiles.json` sidecar so
//! a fresh boot starts warm on a known workload mix.

use std::sync::atomic::{AtomicU64, Ordering};

use peregrine_core::Error;

use crate::workload::TokenClass;

/// The classes a profile is kept for, in a fixed order so the flat per-class
/// storage and the sidecar layout are stable. Mirrors [`TokenClass`]; `Mixed`
/// gets its own bucket rather than folding into another so an ambiguous stream
/// does not pollute a specific topic's profile.
pub const PROFILE_CLASSES: [TokenClass; 5] = [
    TokenClass::Code,
    TokenClass::Json,
    TokenClass::Math,
    TokenClass::Prose,
    TokenClass::Mixed,
];

/// Dense index of a class within [`PROFILE_CLASSES`].
fn class_index(class: TokenClass) -> usize {
    match class {
        TokenClass::Code => 0,
        TokenClass::Json => 1,
        TokenClass::Math => 2,
        TokenClass::Prose => 3,
        TokenClass::Mixed => 4,
    }
}

/// Per-class `(layer, expert)` routing-frequency counters.
///
/// One flat `[n_classes][n_layers * n_experts]` array of atomics: accumulation
/// is a shared-borrow `fetch_add`, so the batch engine bumps it while holding a
/// shared model borrow, same as [`crate::Model::set_workload_class`].
pub struct TopicProfiles {
    n_layers: usize,
    n_experts: usize,
    /// `counts[class_index * stride + layer * n_experts + expert]`.
    counts: Vec<AtomicU64>,
}

impl TopicProfiles {
    pub fn new(n_layers: usize, n_experts: usize) -> TopicProfiles {
        let stride = n_layers * n_experts;
        TopicProfiles {
            n_layers,
            n_experts,
            counts: (0..PROFILE_CLASSES.len() * stride).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn stride(&self) -> usize {
        self.n_layers * self.n_experts
    }

    fn slot(&self, class: TokenClass, layer: usize, expert: usize) -> Option<usize> {
        if layer >= self.n_layers || expert >= self.n_experts {
            return None;
        }
        Some(class_index(class) * self.stride() + layer * self.n_experts + expert)
    }

    /// Record that, under `class`, `layer` routed the given experts. Called once
    /// per decode forward per layer from the cache-protection cadence — never
    /// from inside the batched matmul, so the atomics are uncontended.
    pub fn note(&self, class: TokenClass, layer: usize, experts: &[i32]) {
        for &e in experts {
            if e < 0 {
                continue;
            }
            if let Some(s) = self.slot(class, layer, e as usize) {
                self.counts[s].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// This class's routing frequency for `(layer, expert)`, saturated into the
    /// `u32` tiebreak the evictor packs. `0` for an unseen slot.
    pub fn heat_for(&self, class: TokenClass, layer: usize, expert: usize) -> u32 {
        match self.slot(class, layer, expert) {
            Some(s) => self.counts[s].load(Ordering::Relaxed).min(u32::MAX as u64) as u32,
            None => 0,
        }
    }

    /// Whether `class` has observed anything yet — the caller falls back to the
    /// global heat tiebreak while a class is still cold, so a fresh profile never
    /// makes protection *worse* than the pre-topic behaviour.
    pub fn is_warm(&self, class: TokenClass) -> bool {
        let base = class_index(class) * self.stride();
        self.counts[base..base + self.stride()]
            .iter()
            .any(|c| c.load(Ordering::Relaxed) != 0)
    }

    /// Serialize to the `topic_profiles.json` sidecar shape. Sparse per class —
    /// only non-zero slots are written, since a warm profile touches a fraction
    /// of the `n_classes * n_layers * n_experts` space.
    pub fn to_json(&self, tag: &str) -> serde_json::Value {
        let stride = self.stride();
        let classes: Vec<serde_json::Value> = PROFILE_CLASSES
            .iter()
            .enumerate()
            .map(|(ci, class)| {
                let base = ci * stride;
                let entries: Vec<serde_json::Value> = (0..stride)
                    .filter_map(|i| {
                        let c = self.counts[base + i].load(Ordering::Relaxed);
                        (c != 0).then(|| {
                            serde_json::json!([i / self.n_experts, i % self.n_experts, c])
                        })
                    })
                    .collect();
                serde_json::json!({ "class": format!("{class:?}"), "entries": entries })
            })
            .collect();
        serde_json::json!({
            "version": 1,
            "tag": tag,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "classes": classes,
        })
    }

    /// Load a sidecar, rejecting a shape mismatch (a profile from another
    /// checkpoint is silently useless, so refuse it rather than mis-index).
    /// `None` on any malformed or mismatched input — the caller then starts cold.
    pub fn from_json(v: &serde_json::Value, tag: &str, n_layers: usize, n_experts: usize) -> Option<TopicProfiles> {
        if v.get("version")?.as_u64()? != 1 || v.get("tag")?.as_str()? != tag {
            return None;
        }
        if v.get("n_layers")?.as_u64()? as usize != n_layers
            || v.get("n_experts")?.as_u64()? as usize != n_experts
        {
            return None;
        }
        let profiles = TopicProfiles::new(n_layers, n_experts);
        for c in v.get("classes")?.as_array()? {
            let cname = c.get("class")?.as_str()?;
            let class = PROFILE_CLASSES.iter().copied().find(|k| format!("{k:?}") == cname)?;
            for e in c.get("entries")?.as_array()? {
                let a = e.as_array()?;
                let (layer, expert, count) = (a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize, a.get(2)?.as_u64()?);
                if let Some(s) = profiles.slot(class, layer, expert) {
                    profiles.counts[s].store(count, Ordering::Relaxed);
                }
            }
        }
        Some(profiles)
    }
}

/// Save `profiles` to `path` atomically (the `topic_profiles.json` artifact a
/// model writes on shutdown and auto-loads from its checkpoint dir).
pub fn save_profiles(profiles: &TopicProfiles, tag: &str, path: &std::path::Path) -> Result<(), Error> {
    let json = serde_json::to_vec(&profiles.to_json(tag))
        .map_err(|e| Error::Format(format!("serialize topic profiles: {e}")))?;
    peregrine_core::write_atomic(path, &json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_accumulates_under_the_active_class_only() {
        let p = TopicProfiles::new(4, 8);
        p.note(TokenClass::Code, 2, &[3, 5, 3]); // 3 twice, 5 once
        assert_eq!(p.heat_for(TokenClass::Code, 2, 3), 2);
        assert_eq!(p.heat_for(TokenClass::Code, 2, 5), 1);
        // A different class saw nothing — profiles do not bleed.
        assert_eq!(p.heat_for(TokenClass::Prose, 2, 3), 0);
        assert!(p.is_warm(TokenClass::Code));
        assert!(!p.is_warm(TokenClass::Prose));
    }

    #[test]
    fn out_of_range_and_negative_experts_are_ignored_not_panicking() {
        let p = TopicProfiles::new(4, 8);
        p.note(TokenClass::Json, 99, &[1]); // layer OOR
        p.note(TokenClass::Json, 1, &[99, -1]); // expert OOR / sentinel
        assert!(!p.is_warm(TokenClass::Json));
        assert_eq!(p.heat_for(TokenClass::Json, 1, 99), 0);
    }

    #[test]
    fn a_warmer_expert_for_the_active_topic_wins_the_tiebreak() {
        // The evictor packs (predictor_score << 16 | topic_heat). Two experts
        // the predictor scores equally must be ordered by the ACTIVE topic's
        // frequency — the whole point of topic routing. Reproduces that packing
        // locally so the ordering contract is pinned without a built Model.
        let p = TopicProfiles::new(4, 8);
        p.note(TokenClass::Code, 1, &[2, 2, 2]); // expert 2 hot for Code
        p.note(TokenClass::Code, 1, &[5]); // expert 5 lukewarm for Code
        let pack = |score: u32, heat: u32| ((score.min(0xFFFF) << 16) | heat.min(0xFFFF)).saturating_add(1);
        let score = 3; // equal predictor score for both
        let a = pack(score, p.heat_for(TokenClass::Code, 1, 2));
        let b = pack(score, p.heat_for(TokenClass::Code, 1, 5));
        assert!(a > b, "the topic-hotter expert must outrank the equal-scored one");
        // Under a different active topic the tiebreak is 0 for both (cold), so
        // the packing collapses to the predictor score alone — no bias leaks.
        let a2 = pack(score, p.heat_for(TokenClass::Prose, 1, 2));
        let b2 = pack(score, p.heat_for(TokenClass::Prose, 1, 5));
        assert_eq!(a2, b2);
    }

    #[test]
    fn sidecar_round_trips_and_rejects_a_shape_mismatch() -> Result<(), Error> {
        let p = TopicProfiles::new(6, 16);
        p.note(TokenClass::Code, 3, &[7, 7, 2]);
        p.note(TokenClass::Math, 5, &[1]);
        let v = p.to_json("L6E16");
        let back = TopicProfiles::from_json(&v, "L6E16", 6, 16)
            .ok_or_else(|| Error::Format("round trip should reconstruct the profile".into()))?;
        assert_eq!(back.heat_for(TokenClass::Code, 3, 7), 2);
        assert_eq!(back.heat_for(TokenClass::Math, 5, 1), 1);
        // Wrong tag and wrong shape are both refused.
        assert!(TopicProfiles::from_json(&v, "OTHER", 6, 16).is_none());
        assert!(TopicProfiles::from_json(&v, "L6E16", 6, 32).is_none());
        Ok(())
    }
}

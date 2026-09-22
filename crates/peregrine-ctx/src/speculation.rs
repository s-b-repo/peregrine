//! Speculative context — prefetching the next-needed state.
//!
//! `P(m_i | q_t, S_t)` is estimated by online logistic regression over cheap
//! features (query similarity, graph proximity, historical read-through,
//! recency, staleness). Nodes above `tau` are materialised at their cheap
//! levels *before* the model asks.
//!
//! Two rules the spec is emphatic about and that are enforced here:
//!
//! * speculation is budgeted **separately** from `B_attention` — prefetch
//!   consumes storage/compute, never the active window, so it cannot starve the
//!   current computation;
//! * the predictor is fed back the truth (was the prefetch actually used?),
//!   which is the only thing that keeps `tau` honest.

use std::collections::{HashMap, HashSet};

use crate::ladder::Ladder;
use crate::nodes::{Node, NodeIdx, Status};
use crate::spans::RawStore;

pub const FEATURE_COUNT: usize = 7;
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "bias",
    "query_sim",
    "proximity",
    "read_rate",
    "recency",
    "co_used",
    "stale",
];

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Features {
    pub bias: f32,
    pub query_sim: f32,
    pub proximity: f32,
    pub read_rate: f32,
    pub recency: f32,
    pub co_used: f32,
    pub stale: f32,
}

impl Features {
    pub fn as_array(&self) -> [f32; FEATURE_COUNT] {
        [
            self.bias,
            self.query_sim,
            self.proximity,
            self.read_rate,
            self.recency,
            self.co_used,
            self.stale,
        ]
    }
}

/// Tiny online logistic model. Deterministic init, SGD on feedback.
#[derive(Debug, Clone)]
pub struct Predictor {
    pub weights: [f32; FEATURE_COUNT],
    pub lr: f32,
    pub updates: u32,
}

impl Default for Predictor {
    fn default() -> Self {
        Self {
            weights: [-1.2, 2.4, 1.6, 1.1, 0.7, 1.3, -2.5],
            lr: 0.15,
            updates: 0,
        }
    }
}

impl Predictor {
    pub fn score(&self, features: &Features) -> f32 {
        let z: f32 = self
            .weights
            .iter()
            .zip(features.as_array())
            .map(|(w, f)| w * f)
            .sum();
        1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
    }

    pub fn observe(&mut self, features: &Features, used: bool) {
        let p = self.score(features);
        let error = if used { 1.0 } else { 0.0 } - p;
        for (weight, feature) in self.weights.iter_mut().zip(features.as_array()) {
            *weight += self.lr * error * feature;
        }
        self.updates += 1;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub node: NodeIdx,
    pub p: f32,
    pub features: Features,
}

#[derive(Debug)]
pub struct Speculator {
    pub predictor: Predictor,
    pub tau: f32,
    /// Number of level-builds allowed per turn. Storage/compute budget —
    /// deliberately *not* denominated in attention tokens.
    pub materialization_budget: usize,
    outstanding: Vec<(NodeIdx, Features)>,
    co_use: HashMap<NodeIdx, HashMap<NodeIdx, u32>>,
    last_used: Vec<NodeIdx>,
    pub issued: u32,
    pub hits: u32,
    pub misses: u32,
    pub wasted_builds: u32,
}

impl Default for Speculator {
    fn default() -> Self {
        Self {
            predictor: Predictor::default(),
            tau: 0.5,
            materialization_budget: 24,
            outstanding: Vec::new(),
            co_use: HashMap::new(),
            last_used: Vec::new(),
            issued: 0,
            hits: 0,
            misses: 0,
            wasted_builds: 0,
        }
    }
}

impl Speculator {
    /// `tau` is the materialisation threshold; `materialization_budget` caps
    /// level-builds per turn and is denominated in builds, not tokens.
    pub fn new(tau: f32, materialization_budget: usize) -> Self {
        Self {
            tau,
            materialization_budget,
            ..Default::default()
        }
    }

    pub fn features(
        &self,
        node: &Node,
        idx: NodeIdx,
        query_sim: f32,
        proximity: f32,
        now: u64,
    ) -> Features {
        let age = now.saturating_sub(node.timestamp).max(1) as f32;
        let co_used = self
            .last_used
            .iter()
            .map(|&last| self.co_use_rate(last, idx))
            .fold(0.0f32, f32::max);
        Features {
            bias: 1.0,
            query_sim: query_sim.max(0.0),
            proximity,
            read_rate: node.read_rate(),
            recency: 1.0 / (1.0 + age.ln()),
            co_used,
            stale: if node.status == Status::Stale {
                1.0
            } else {
                0.0
            },
        }
    }

    fn co_use_rate(&self, a: NodeIdx, b: NodeIdx) -> f32 {
        match self.co_use.get(&a) {
            Some(row) => {
                let total: u32 = row.values().sum();
                if total == 0 {
                    0.0
                } else {
                    row.get(&b).copied().unwrap_or(0) as f32 / total as f32
                }
            }
            None => 0.0,
        }
    }

    pub fn predict(&self, scored: &[(NodeIdx, &Node, f32, f32)], now: u64) -> Vec<Prediction> {
        let mut out: Vec<Prediction> = scored
            .iter()
            .map(|&(idx, node, sim, proximity)| {
                let features = self.features(node, idx, sim, proximity, now);
                Prediction {
                    node: idx,
                    p: self.predictor.score(&features),
                    features,
                }
            })
            .collect();
        out.sort_by(|a, b| b.p.total_cmp(&a.p).then(a.node.cmp(&b.node)));
        out
    }

    /// Materialise cheap levels for predictions above `tau`, within budget.
    pub fn prefetch(
        &mut self,
        predictions: &[Prediction],
        nodes: &[Node],
        ladder: &Ladder,
        raw: &RawStore,
    ) -> Vec<Prediction> {
        let mut issued = Vec::new();
        let mut spent = 0usize;
        for prediction in predictions {
            if prediction.p < self.tau || spent >= self.materialization_budget {
                break;
            }
            let Some(node) = nodes.get(prediction.node) else {
                continue;
            };
            spent += ladder.prewarm(raw, node);
            self.outstanding
                .push((prediction.node, prediction.features));
            issued.push(*prediction);
        }
        self.issued += issued.len() as u32;
        issued
    }

    /// Was the prefetch used? Trains the predictor and reports hit rate.
    pub fn feedback(&mut self, used: &HashSet<NodeIdx>) -> (usize, usize) {
        let outstanding = std::mem::take(&mut self.outstanding);
        let issued = outstanding.len();
        let mut hits = 0usize;
        for (node, features) in outstanding {
            let was_used = used.contains(&node);
            self.predictor.observe(&features, was_used);
            if was_used {
                hits += 1;
                self.hits += 1;
            } else {
                self.misses += 1;
                self.wasted_builds += 1;
            }
        }
        // Record co-occurrence so the next turn can use "what was read with
        // what" as a signal.
        for &a in used {
            let row = self.co_use.entry(a).or_default();
            for &b in used {
                if a != b {
                    *row.entry(b).or_insert(0) += 1;
                }
            }
        }
        self.last_used = {
            let mut v: Vec<NodeIdx> = used.iter().copied().collect();
            v.sort_unstable();
            v
        };
        (issued, hits)
    }

    pub fn co_use_row(&self, node: NodeIdx) -> Option<&HashMap<NodeIdx, u32>> {
        self.co_use.get(&node)
    }

    pub fn hit_rate(&self) -> Option<f32> {
        let total = self.hits + self.misses;
        if total == 0 {
            None
        } else {
            Some(self.hits as f32 / total as f32)
        }
    }

    pub fn stats(&self) -> String {
        format!(
            "prefetch_issued: {}, prefetch_hits: {}, prefetch_hit_rate: {}, wasted_builds: {}, \
             tau: {}, predictor_updates: {}",
            self.issued,
            self.hits,
            self.hit_rate()
                .map(|r| format!("{r:.3}"))
                .unwrap_or_else(|| "n/a".into()),
            self.wasted_builds,
            self.tau,
            self.predictor.updates
        )
    }
}

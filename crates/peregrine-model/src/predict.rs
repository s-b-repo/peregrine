//! Prefetch prediction: which experts the next forward will route, so the prefetch
//! lane can warm them ahead of the critical path. Everything here affects only
//! *which* experts are prefetched — never the model's output. A wrong guess just
//! re-streams identical bytes on demand, so all prediction is correctness-neutral.
//!
//! The substrate is [`RouteHistory`], a bounded per-layer ring of the most recent
//! routed expert sets. [`PredictSource`] consumes it to produce a *ranked* candidate
//! list per layer. Today the only source is recency-weighted [`Momentum`]; the
//! offline transition automaton is layered in as an additional source without
//! changing consumers.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// Bounded per-layer history of recently routed expert sets, newest first. One ring
/// per layer, capped at `depth` frames. Written once per forward at the single
/// post-reduce point (exactly one writer → no race), read by the predictor. Storing
/// owned `Vec<i32>` frames and rotating them is allocation-neutral versus the previous
/// single-frame store — `batch_union` already allocates the set we take ownership of.
pub struct RouteHistory {
    layers: Vec<VecDeque<Vec<i32>>>,
    depth: usize,
}

impl RouteHistory {
    /// History for `n_layers` layers, keeping the last `depth` routed sets per layer.
    /// `depth` is floored at 1, so a depth-1 history reproduces the legacy predictor
    /// ("the next token routes like the last one").
    pub fn new(n_layers: usize, depth: usize) -> RouteHistory {
        let depth = depth.max(1);
        RouteHistory {
            layers: (0..n_layers).map(|_| VecDeque::with_capacity(depth)).collect(),
            depth,
        }
    }

    /// Record `uniq` — this forward's routed experts at `layer` — as the newest frame,
    /// dropping the oldest when already at `depth`. No-op for an out-of-range layer.
    pub fn push_layer(&mut self, layer: usize, uniq: Vec<i32>) {
        if let Some(ring) = self.layers.get_mut(layer) {
            if ring.len() == self.depth {
                ring.pop_back();
            }
            ring.push_front(uniq);
        }
    }

    /// Frames for `layer`, newest first. Empty for a dense layer or before any forward.
    pub fn frames(&self, layer: usize) -> impl Iterator<Item = &Vec<i32>> {
        self.layers.get(layer).into_iter().flatten()
    }

    /// The most-recent routed set for `layer` (the legacy single-frame prediction).
    pub fn latest(&self, layer: usize) -> Option<&Vec<i32>> {
        self.layers.get(layer).and_then(VecDeque::front)
    }

    /// Number of layers this history tracks.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Configured ring depth (max frames kept per layer).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Drop every layer's frames (per-sequence reset), keeping allocated capacity.
    pub fn clear(&mut self) {
        for ring in &mut self.layers {
            ring.clear();
        }
    }

    /// Serialize to a flat, ordered JSON document (deterministic output). Frames
    /// preserve newest-first order per layer. Companion to [`Self::from_json`].
    pub fn to_json(&self, tag: &str) -> serde_json::Value {
        let layers: Vec<serde_json::Value> = self
            .layers
            .iter()
            .map(|ring| {
                let frames: Vec<serde_json::Value> =
                    ring.iter().map(|frame| serde_json::json!(frame)).collect();
                serde_json::Value::Array(frames)
            })
            .collect();
        serde_json::json!({
            "tag": tag,
            "n_layers": self.layers.len(),
            "depth": self.depth,
            "layers": layers,
        })
    }

    /// Reconstruct from [`Self::to_json`] output. `None` when the document is
    /// malformed or its `tag` differs from `expected_tag` (config fingerprint
    /// staleness check — same policy as [`TransitionTable::from_json`]).
    pub fn from_json(v: &serde_json::Value, expected_tag: &str) -> Option<RouteHistory> {
        let tag = v.get("tag")?.as_str()?;
        if tag != expected_tag {
            return None;
        }
        let n_layers = v.get("n_layers")?.as_u64()? as usize;
        let depth = v.get("depth")?.as_u64()? as usize;
        let layers = v.get("layers")?.as_array()?;
        if layers.len() != n_layers {
            return None;
        }
        let mut out = RouteHistory::new(n_layers, depth);
        for (layer_idx, layer) in layers.iter().enumerate() {
            let frames = layer.as_array()?;
            let ring = out.layers.get_mut(layer_idx)?;
            // frames are newest-first in the serialized form; push in reverse so
            // push_front keeps them newest-first after the loop.
            for frame in frames.iter().rev() {
                let arr = frame.as_array()?;
                let mut ids: Vec<i32> = Vec::with_capacity(arr.len());
                for x in arr {
                    ids.push(x.as_i64()? as i32);
                }
                if ring.len() == depth.max(1) {
                    ring.pop_back();
                }
                ring.push_front(ids);
            }
        }
        Some(out)
    }
}

/// Recency-weighted momentum predictor. Each recent routed set votes for its experts;
/// a frame `i` steps back from the newest contributes `weight(i)`, so recent routing
/// dominates while experts that persist across frames accumulate. Depth-1 history
/// reduces this to "predict exactly the last routed set".
#[derive(Default)]
pub struct Momentum {
    /// Frames to consider (capped by the history depth). 0 means "all frames".
    pub window: usize,
}

impl Momentum {
    fn weight(&self, frame_idx: usize, considered: usize) -> u32 {
        // linear recency: the newest frame weighs `considered`, the oldest weighs 1.
        (considered - frame_idx) as u32
    }
}

/// A first-order expert-transition automaton, built offline from a routing corpus.
/// Per layer it counts how often expert `from` (routed at token *t*) is followed by
/// expert `to` (routed at token *t+1*). At runtime, given the current routed set it
/// predicts the next experts by summing transition counts out of the current set —
/// branch-prediction for MoE routing. Correctness-neutral: it only ranks prefetch
/// candidates. Tagged with a config fingerprint so a stale artifact is ignored.
pub struct TransitionTable {
    tag: String,
    n_layers: usize,
    /// per layer: `from_expert -> (to_expert -> count)`.
    layers: Vec<HashMap<u32, HashMap<u32, u32>>>,
}

impl TransitionTable {
    /// An empty table for `n_layers`, fingerprinted with `tag`.
    pub fn new(n_layers: usize, tag: String) -> TransitionTable {
        TransitionTable { tag, n_layers, layers: (0..n_layers).map(|_| HashMap::new()).collect() }
    }

    /// The config fingerprint this table was built for (staleness check on load).
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Record one transition at `layer`: every `from` in the previous routed set is
    /// followed by every `to` in the current routed set.
    pub fn observe(&mut self, layer: usize, from_set: &[i32], to_set: &[i32]) {
        let Some(map) = self.layers.get_mut(layer) else {
            return;
        };
        for &f in from_set {
            if f < 0 {
                continue;
            }
            let row = map.entry(f as u32).or_default();
            for &t in to_set {
                if t >= 0 {
                    *row.entry(t as u32).or_insert(0) += 1;
                }
            }
        }
    }

    /// Ranked next-expert prediction for `layer` given the `current` routed set:
    /// `score[to] = Σ_{from ∈ current} count[from][to]`, sorted score-desc then id-asc.
    /// Empty when the layer/state is unseen.
    pub fn predict_layer(&self, layer: usize, current: &[i32]) -> Vec<(u32, u32)> {
        let Some(map) = self.layers.get(layer) else {
            return Vec::new();
        };
        let mut scores: Vec<(u32, u32)> = Vec::new();
        for &f in current {
            if f < 0 {
                continue;
            }
            if let Some(row) = map.get(&(f as u32)) {
                for (&to, &c) in row {
                    match scores.iter_mut().find(|(e, _)| *e == to) {
                        Some(entry) => entry.1 += c,
                        None => scores.push((to, c)),
                    }
                }
            }
        }
        scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        scores
    }

    /// Serialize to a flat, sorted JSON document (deterministic output).
    pub fn to_json(&self) -> serde_json::Value {
        let mut edges: Vec<(usize, u32, u32, u32)> = Vec::new();
        for (layer, map) in self.layers.iter().enumerate() {
            for (&f, row) in map {
                for (&t, &c) in row {
                    edges.push((layer, f, t, c));
                }
            }
        }
        edges.sort_unstable();
        let edges: Vec<serde_json::Value> =
            edges.into_iter().map(|(l, f, t, c)| serde_json::json!([l, f, t, c])).collect();
        serde_json::json!({ "tag": self.tag, "n_layers": self.n_layers, "edges": edges })
    }

    /// Parse a table from [`Self::to_json`] output. `None` on a malformed document.
    pub fn from_json(v: &serde_json::Value) -> Option<TransitionTable> {
        let tag = v.get("tag")?.as_str()?.to_string();
        let n_layers = v.get("n_layers")?.as_u64()? as usize;
        let mut table = TransitionTable::new(n_layers, tag);
        for e in v.get("edges")?.as_array()? {
            let a = e.as_array()?;
            let (l, f, t, c) = (a.first()?.as_u64()?, a.get(1)?.as_u64()?, a.get(2)?.as_u64()?, a.get(3)?.as_u64()?);
            if let Some(map) = table.layers.get_mut(l as usize) {
                *map.entry(f as u32).or_default().entry(t as u32).or_insert(0) += c as u32;
            }
        }
        Some(table)
    }
}

/// Where prefetch candidates come from. New sources (e.g. the offline transition
/// automaton) are added as variants — a different *source*, not a different ranking —
/// so consumers stay unchanged.
pub enum PredictSource {
    /// Recency-weighted vote over recent routing history.
    Momentum(Momentum),
    /// Offline transition automaton, with a momentum fallback for unseen states.
    Automaton { table: Arc<TransitionTable>, fallback: Momentum },
    /// Phase-aware wrapper: delegate to `inner` normally, but when a phase-shift
    /// is detected (Jaccard distance between the newest two frames exceeds
    /// `threshold_bp / 10000`), fold in an extra momentum vote weighted heavily
    /// on the newest frame so prefetch tracks the new routing distribution
    /// faster than steady-state momentum would. Correctness-neutral — the
    /// scores are still sorted by (score-desc, expert-asc).
    PhaseAware { inner: Box<PredictSource>, threshold_bp: u32, boost: u32 },
}

impl Default for PredictSource {
    fn default() -> PredictSource {
        PredictSource::Momentum(Momentum::default())
    }
}

impl PredictSource {
    /// Ranked prefetch candidates for `layer`: `(expert, score)` in score-descending,
    /// then expert-id-ascending order — a deterministic tie-break so counter-based
    /// tests are stable and prefetch ordering never leaks `HashMap` iteration order.
    /// Empty when the layer has no history yet.
    pub fn predict_layer(&self, layer: usize, hist: &RouteHistory) -> Vec<(u32, u32)> {
        match self {
            PredictSource::Momentum(m) => momentum_vote(layer, hist, m),
            PredictSource::Automaton { table, fallback } => {
                // automaton prediction from the current set, blended with momentum so
                // cold/unseen states still predict something.
                let mut scores = hist.latest(layer).map(|cur| table.predict_layer(layer, cur)).unwrap_or_default();
                merge_scores(&mut scores, momentum_vote(layer, hist, fallback));
                scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                scores
            }
            PredictSource::PhaseAware { inner, threshold_bp, boost } => {
                let mut scores = inner.predict_layer(layer, hist);
                // Cheap two-frame phase check: distance between the newest two
                // frames. A bump above `threshold` means routing just shifted —
                // temporarily amplify the newest frame's contribution so
                // prefetch chases the new distribution instead of averaging.
                let (top, prev) = {
                    let mut it = hist.frames(layer);
                    (it.next(), it.next())
                };
                if let (Some(cur), Some(prev)) = (top, prev) {
                    if jaccard_distance_bp(cur, prev) > *threshold_bp {
                        // Add one high-weight vote per expert in the newest set.
                        let extra: Vec<(u32, u32)> = cur
                            .iter()
                            .filter(|&&e| e >= 0)
                            .map(|&e| (e as u32, *boost))
                            .collect();
                        merge_scores(&mut scores, extra);
                        scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    }
                }
                scores
            }
        }
    }
}

/// Jaccard distance between two integer sets, expressed in basis points
/// (0 → identical, 10000 → fully disjoint). Ignores negative ids ("no route").
/// Zero when both sets are empty (no distance = no shift signal).
fn jaccard_distance_bp(a: &[i32], b: &[i32]) -> u32 {
    let sa: std::collections::HashSet<i32> = a.iter().copied().filter(|&x| x >= 0).collect();
    let sb: std::collections::HashSet<i32> = b.iter().copied().filter(|&x| x >= 0).collect();
    let inter = sa.intersection(&sb).count();
    let uni = sa.union(&sb).count();
    if uni == 0 {
        0
    } else {
        (10_000 - (inter as u32 * 10_000 / uni as u32)).min(10_000)
    }
}

/// Merge `extra` `(expert, score)` votes into `into`, summing scores per expert.
/// Leaves `into` unsorted (the caller re-sorts).
fn merge_scores(into: &mut Vec<(u32, u32)>, extra: Vec<(u32, u32)>) {
    for (e, s) in extra {
        match into.iter_mut().find(|(ex, _)| *ex == e) {
            Some(entry) => entry.1 = entry.1.saturating_add(s),
            None => into.push((e, s)),
        }
    }
}

fn momentum_vote(layer: usize, hist: &RouteHistory, m: &Momentum) -> Vec<(u32, u32)> {
    let total = hist.frames(layer).count();
    if total == 0 {
        return Vec::new();
    }
    let considered = if m.window == 0 { total } else { m.window.min(total) };
    // `scores` holds one entry per distinct expert (a small set — the routed union
    // per layer). A linear probe keeps the order deterministic without hashing.
    let mut scores: Vec<(u32, u32)> = Vec::new();
    for (i, frame) in hist.frames(layer).take(considered).enumerate() {
        let w = m.weight(i, considered);
        for &e in frame {
            if e < 0 {
                continue;
            }
            let e = e as u32;
            match scores.iter_mut().find(|(ex, _)| *ex == e) {
                Some(entry) => entry.1 += w,
                None => scores.push((e, w)),
            }
        }
    }
    scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    scores
}

/// Adaptive prefetch-distance controller. It tracks how often prefetched experts are
/// actually used vs. wasted (evicted unused) with an EWMA over per-forward deltas, and
/// grows or shrinks the per-layer prefetch breadth `distance` accordingly: good
/// predictions → prefetch more; wasted bandwidth → prefetch less. `distance` is always
/// clamped to `[1, d_max]`, so it only bounds prefetch volume — never correctness.
pub struct PrefetchTuner {
    distance: usize,
    d_max: usize,
    alpha: f32,
    ewma_used: f32,
    ewma_wasted: f32,
    last_used: u64,
    last_wasted: u64,
}

impl PrefetchTuner {
    /// Controller starting at `initial` distance, capped at `d_max` (both floored at 1).
    pub fn new(initial: usize, d_max: usize) -> PrefetchTuner {
        let d_max = d_max.max(1);
        PrefetchTuner {
            distance: initial.clamp(1, d_max),
            d_max,
            alpha: 0.3,
            ewma_used: 0.0,
            ewma_wasted: 0.0,
            last_used: 0,
            last_wasted: 0,
        }
    }

    /// Current per-layer prefetch breadth.
    pub fn distance(&self) -> usize {
        self.distance
    }

    /// Feed the *cumulative* used/wasted counters. Updates the EWMA on this forward's
    /// delta and steps `distance` up (recent prefetches mostly used) or down (mostly
    /// wasted), clamped to `[1, d_max]`. Returns the new distance.
    pub fn observe(&mut self, used_total: u64, wasted_total: u64) -> usize {
        // The counters can be cleared out from under us (cache reset / tests). If they
        // went backwards, rebase so this forward contributes a zero delta rather than
        // a spurious spike.
        if used_total < self.last_used || wasted_total < self.last_wasted {
            self.last_used = used_total;
            self.last_wasted = wasted_total;
        }
        let used = used_total.saturating_sub(self.last_used) as f32;
        let wasted = wasted_total.saturating_sub(self.last_wasted) as f32;
        self.last_used = used_total;
        self.last_wasted = wasted_total;
        self.ewma_used = (1.0 - self.alpha) * self.ewma_used + self.alpha * used;
        self.ewma_wasted = (1.0 - self.alpha) * self.ewma_wasted + self.alpha * wasted;
        if self.ewma_used > self.ewma_wasted && self.distance < self.d_max {
            self.distance += 1;
        } else if self.ewma_wasted > self.ewma_used && self.distance > 1 {
            self.distance -= 1;
        }
        self.distance
    }

    /// Reset the smoothing + delta baselines for a new sequence (keeps `distance`).
    pub fn reset(&mut self) {
        self.ewma_used = 0.0;
        self.ewma_wasted = 0.0;
        self.last_used = 0;
        self.last_wasted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_newest_first_and_capped() {
        let mut h = RouteHistory::new(2, 2); // depth 2
        h.push_layer(1, vec![1, 2]);
        h.push_layer(1, vec![3, 4]);
        h.push_layer(1, vec![5, 6]); // evicts [1,2]
        let frames: Vec<&Vec<i32>> = h.frames(1).collect();
        assert_eq!(frames.len(), 2, "capped at depth");
        assert_eq!(frames[0], &vec![5, 6], "newest first");
        assert_eq!(frames[1], &vec![3, 4]);
        assert_eq!(h.latest(1), Some(&vec![5, 6]));
    }

    #[test]
    fn empty_and_dense_layers_yield_no_frames() {
        let h = RouteHistory::new(3, 4);
        assert_eq!(h.frames(0).count(), 0);
        assert_eq!(h.latest(0), None);
        assert_eq!(h.frames(99).count(), 0); // out of range
    }

    #[test]
    fn route_history_json_round_trip_preserves_newest_first_order() {
        let mut h = RouteHistory::new(3, 2);
        h.push_layer(1, vec![10, 20]);
        h.push_layer(1, vec![30, 40]);
        h.push_layer(2, vec![5]);
        let j = h.to_json("TAG");
        let h2 = RouteHistory::from_json(&j, "TAG");
        assert!(h2.is_some(), "round trip must succeed");
        if let Some(h2) = h2 {
            assert_eq!(h2.n_layers(), 3);
            assert_eq!(h2.depth(), 2);
            assert_eq!(h2.latest(1), Some(&vec![30, 40]));
            let frames: Vec<&Vec<i32>> = h2.frames(1).collect();
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0], &vec![30, 40]);
            assert_eq!(frames[1], &vec![10, 20]);
            assert_eq!(h2.latest(2), Some(&vec![5]));
            assert_eq!(h2.latest(0), None);
        }
    }

    #[test]
    fn route_history_json_rejects_wrong_tag() {
        let mut h = RouteHistory::new(1, 1);
        h.push_layer(0, vec![7]);
        let j = h.to_json("REAL");
        assert!(RouteHistory::from_json(&j, "OTHER").is_none());
    }

    #[test]
    fn phase_aware_boosts_newest_on_shift() {
        // Two frames with fully-disjoint routing → phase shift detected → the
        // newest frame's experts must dominate the returned ranking, even if
        // the inner Momentum source averages older frames.
        let mut h = RouteHistory::new(1, 4);
        h.push_layer(0, vec![1, 2, 3]); // older
        h.push_layer(0, vec![9, 8, 7]); // newest, disjoint from older
        let inner = Box::new(PredictSource::Momentum(Momentum { window: 0 }));
        // 6000 bp = 0.6 threshold; boost weight = 100 (bigger than any per-frame
        // momentum weight for depth-4 history, so it dominates the ranking).
        let src = PredictSource::PhaseAware { inner, threshold_bp: 6000, boost: 100 };
        let ranked = src.predict_layer(0, &h);
        // 9, 8, 7 (the newest frame) should be ranked above 1, 2, 3.
        let top3: Vec<u32> = ranked.iter().take(3).map(|&(e, _)| e).collect();
        assert!(top3.contains(&9) && top3.contains(&8) && top3.contains(&7));
    }

    #[test]
    fn phase_aware_defers_to_inner_when_steady() {
        // Two identical frames → Jaccard distance = 0 → below threshold → the
        // wrapper must be a pure passthrough (bit-identical to the inner source).
        let mut h = RouteHistory::new(1, 4);
        h.push_layer(0, vec![1, 2, 3]);
        h.push_layer(0, vec![1, 2, 3]);
        let inner_ref = PredictSource::Momentum(Momentum { window: 0 });
        let expected = inner_ref.predict_layer(0, &h);
        let src = PredictSource::PhaseAware {
            inner: Box::new(PredictSource::Momentum(Momentum { window: 0 })),
            threshold_bp: 6000,
            boost: 100,
        };
        let got = src.predict_layer(0, &h);
        assert_eq!(got, expected, "steady-state PhaseAware == inner passthrough");
    }

    #[test]
    fn momentum_depth1_equals_last_set() {
        // Depth-1 momentum must predict exactly the last routed set (as a set),
        // the parity anchor for the legacy "predict = last routed" behaviour.
        let mut h = RouteHistory::new(1, 1);
        h.push_layer(0, vec![7, 2, 5]);
        let src = PredictSource::default();
        let mut got: Vec<u32> = src.predict_layer(0, &h).into_iter().map(|(e, _)| e).collect();
        got.sort_unstable();
        assert_eq!(got, vec![2, 5, 7]);
    }

    #[test]
    fn momentum_ranks_recent_and_persistent_higher() {
        // frames newest→oldest: [3,4], [2,3], [1,2]  (weights 3,2,1)
        // scores: 3→5, 4→3, 2→3, 1→1  → 3, then (2,4 tie broken by id), then 1.
        let mut h = RouteHistory::new(1, 3);
        h.push_layer(0, vec![1, 2]);
        h.push_layer(0, vec![2, 3]);
        h.push_layer(0, vec![3, 4]);
        let src = PredictSource::default();
        let ranked: Vec<(u32, u32)> = src.predict_layer(0, &h);
        assert_eq!(ranked, vec![(3, 5), (2, 3), (4, 3), (1, 1)]);
    }

    #[test]
    fn clear_empties_all_layers() {
        let mut h = RouteHistory::new(2, 2);
        h.push_layer(0, vec![1]);
        h.push_layer(1, vec![2]);
        h.clear();
        assert_eq!(h.frames(0).count(), 0);
        assert_eq!(h.frames(1).count(), 0);
    }

    #[test]
    fn tuner_grows_on_useful_prefetch_and_shrinks_on_waste() {
        let mut t = PrefetchTuner::new(4, 8);
        // sustained useful prefetch (used delta >> wasted) climbs to d_max
        let mut used = 0u64;
        for _ in 0..30 {
            used += 10;
            t.observe(used, 0);
        }
        assert_eq!(t.distance(), 8, "sustained useful prefetch grows to d_max");
        // now waste dominates (used flat, wasted climbs) → falls to the floor
        let mut wasted = 0u64;
        for _ in 0..30 {
            wasted += 10;
            t.observe(used, wasted);
        }
        assert_eq!(t.distance(), 1, "sustained waste shrinks to the floor");
    }

    #[test]
    fn tuner_stays_within_bounds() {
        let mut t = PrefetchTuner::new(1, 3);
        assert_eq!(t.distance(), 1);
        let mut used = 0u64;
        for _ in 0..100 {
            used += 5;
            t.observe(used, 0);
        }
        assert!(t.distance() <= 3, "never exceeds d_max");
        let mut wasted = 0u64;
        for _ in 0..100 {
            wasted += 5;
            t.observe(used, wasted);
        }
        assert!(t.distance() >= 1, "never below the floor of 1");
    }

    #[test]
    fn tuner_clamps_initial_and_floors_dmax() {
        assert_eq!(PrefetchTuner::new(99, 4).distance(), 4, "initial clamped to d_max");
        assert_eq!(PrefetchTuner::new(0, 0).distance(), 1, "d_max and initial floored at 1");
    }

    #[test]
    fn automaton_predicts_recurring_transition() {
        // Teach: at layer 1, set {1,2} is repeatedly followed by {3,4}. The automaton
        // must then predict 3 and 4 as the top candidates from the current set {1,2}.
        let mut t = TransitionTable::new(2, "tag".into());
        for _ in 0..5 {
            t.observe(1, &[1, 2], &[3, 4]);
        }
        let ranked = t.predict_layer(1, &[1, 2]);
        let top: Vec<u32> = ranked.iter().take(2).map(|(e, _)| *e).collect();
        assert!(top.contains(&3) && top.contains(&4), "predicts the recurring next set, got {ranked:?}");
        // strength scales with count: each of 3,4 got 2 froms × 5 obs = 10.
        assert_eq!(ranked.iter().find(|(e, _)| *e == 3).map(|(_, s)| *s), Some(10));
        assert!(t.predict_layer(1, &[9]).is_empty(), "unseen state predicts nothing");
    }

    #[test]
    fn automaton_json_round_trips() {
        let mut t = TransitionTable::new(3, "cfg-abc".into());
        t.observe(1, &[1], &[2, 3]);
        t.observe(2, &[5], &[5]);
        let v = t.to_json();
        let back = TransitionTable::from_json(&v);
        assert!(back.is_some(), "round-trip must parse");
        if let Some(back) = back {
            assert_eq!(back.tag(), "cfg-abc");
            assert_eq!(back.predict_layer(1, &[1]), t.predict_layer(1, &[1]));
            assert_eq!(back.predict_layer(2, &[5]), vec![(5, 1)]);
        }
    }

    #[test]
    fn automaton_source_blends_momentum_fallback() {
        // Empty automaton → prediction falls back to momentum over history.
        let mut h = RouteHistory::new(2, 2);
        h.push_layer(1, vec![7, 8]);
        let src = PredictSource::Automaton {
            table: Arc::new(TransitionTable::new(2, "t".into())),
            fallback: Momentum::default(),
        };
        let mut got: Vec<u32> = src.predict_layer(1, &h).into_iter().map(|(e, _)| e).collect();
        got.sort_unstable();
        assert_eq!(got, vec![7, 8], "unseen automaton state falls back to momentum");
    }
}

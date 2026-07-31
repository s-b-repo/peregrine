//! GPU VRAM expert tier — the third concurrent lane (feature `cuda`).
//!
//! At load, as many routed experts as fit in VRAM are dequantized to f32 and
//! uploaded (fmt=0). During the MoE forward, [`crate::concurrent`] dispatches the
//! GPU-resident experts of a layer in one batched `expert_group` call — running
//! on its own thread, concurrently with the io_uring disk lane and the CPU pool.
//!
//! Numerics note: GPU experts compute in **f32** (more accurate than the CPU
//! int4 path), so enabling the tier changes low-order bits versus a pure-CPU run
//! — expected, and why the bit-exact determinism test stays CPU-only. The tier is
//! opt-in (built only when `COLI_GPU` is set) so default behavior is unchanged.
//!
//! The type exists in both builds: the real tier under `cuda`, and a never-built
//! empty stub otherwise, so the scheduler/`Model` stay feature-agnostic.

#[cfg(feature = "cuda")]
pub use real::GpuTier;
#[cfg(not(feature = "cuda"))]
pub use stub::GpuTier;

/// Choose which `(layer, expert)` pairs to hold VRAM-resident, given how many
/// experts fit in `budget`. Spreads residency **round-robin across all sparse
/// layers** (expert 0 of every layer, then expert 1 of every layer, …) instead of
/// greedily filling the earliest layers — so every sparse layer gets a VRAM share
/// and no layer is left streaming 100% from disk. Pure and deterministic, so the
/// placement policy is unit-testable without a GPU (the actual upload is not).
///
/// This is the static B1(a) policy; a later refinement can rank experts by routing
/// frequency (a `heat` accumulator) instead of by index, reusing this same shape.
pub fn plan_residency(
    n_layers: usize,
    first_dense: usize,
    n_experts: usize,
    bytes_per_expert: usize,
    budget: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if bytes_per_expert == 0 || n_experts == 0 || first_dense >= n_layers {
        return out;
    }
    let capacity = budget / bytes_per_expert; // number of experts that fit
    if capacity == 0 {
        return out;
    }
    'fill: for e in 0..n_experts {
        for layer in first_dense..n_layers {
            if out.len() >= capacity {
                break 'fill;
            }
            out.push((layer, e));
        }
    }
    out
}

use std::cmp::Reverse;
use std::sync::atomic::{AtomicU32, Ordering};

/// Routing-frequency accumulator over `(layer, expert)` — the "heat" that drives
/// dynamic VRAM residency. Bumped once per routed expert per layer during the
/// forward (lock-free), read by `GpuTier::reheat` to keep the hottest experts
/// resident. Present in every build (the `Model` holds one when a GPU tier
/// exists); the ranking is pure, so it is unit-testable without a GPU.
pub struct HeatTable {
    n_experts: usize,
    counts: Vec<AtomicU32>,
}

impl HeatTable {
    /// A zeroed table for `n_layers × n_experts` routed experts.
    pub fn new(n_layers: usize, n_experts: usize) -> HeatTable {
        HeatTable { n_experts, counts: (0..n_layers * n_experts).map(|_| AtomicU32::new(0)).collect() }
    }

    /// Record one routing of `expert` in `layer` (lock-free; out-of-range ignored).
    pub fn bump(&self, layer: usize, expert: usize) {
        // Bound the expert *within its layer*: checking only the flattened index
        // let `expert == n_experts` credit (layer+1, 0), and a wildly out-of-range
        // id wrap the multiply in release (no overflow checks) into a valid but
        // wrong slot.
        if expert >= self.n_experts {
            return;
        }
        if let Some(c) = layer.checked_mul(self.n_experts).and_then(|b| b.checked_add(expert)).and_then(|i| self.counts.get(i)) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One slot's current count (0 for out-of-range) — a single atomic load, so
    /// hot paths (cache-admission gating) can consult it without a snapshot.
    pub fn get(&self, layer: usize, expert: usize) -> u32 {
        if expert >= self.n_experts {
            return 0; // see `bump`: the expert id is bounded within its layer
        }
        layer
            .checked_mul(self.n_experts)
            .and_then(|b| b.checked_add(expert))
            .and_then(|i| self.counts.get(i))
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// A plain snapshot of the counts, row-major `[layer * n_experts + expert]`.
    pub fn snapshot(&self) -> Vec<u32> {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Restore counts from a snapshot (mismatched-length input is truncated at
    /// the shorter of the two). Used by the cross-session persistence path so a
    /// fresh model starts with the previous session's routing heat instead of a
    /// cold zero table.
    pub fn restore(&self, snapshot: &[u32]) -> bool {
        // Exact shape or nothing: `zip` silently truncates, so a snapshot from a
        // model with a different expert count would reinterpret a flat
        // row-major array against the new stride and land every count on the
        // wrong (layer, expert) pair. Heat is additive, so that never washes out.
        if snapshot.len() != self.counts.len() {
            return false;
        }
        for (c, &v) in self.counts.iter().zip(snapshot.iter()) {
            c.store(v, Ordering::Relaxed);
        }
        true
    }

    /// Total slots (`n_layers * n_experts`) — length of [`Self::snapshot`].
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether the table is empty (no layers × no experts).
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// Persistent Expert Residency Solver. Greedy knapsack over `(layer, expert)`
/// pairs, maximizing total heat within `budget` bytes. Handles heterogeneous
/// expert sizes (per-layer inter-size variation) by sorting on `heat /
/// bytes_per_expert` — a small expert with modest heat can beat a larger cold
/// one. Falls back to [`plan_residency`] round-robin when the heat table is
/// all-zero. Deterministic ties (equal ratio → lower layer, lower expert).
pub fn solve_residency_greedy(
    counts: &[u32],
    n_layers: usize,
    first_dense: usize,
    n_experts: usize,
    bytes_per_expert: usize,
    budget: usize,
) -> Vec<(usize, usize)> {
    solve_residency_sized(counts, n_layers, first_dense, n_experts, budget, |_, _| bytes_per_expert)
}

/// [`solve_residency_greedy`] with a per-candidate size, which is what makes it
/// a knapsack rather than a top-N: `bytes_of(layer, expert)` reports what THAT
/// expert will actually occupy, so a tier that promotes some residents to f32
/// (8× the int4 footprint) still fits its VRAM budget. Ordering is by heat
/// density (`heat / bytes`), deterministic on ties.
pub fn solve_residency_sized(
    counts: &[u32],
    n_layers: usize,
    first_dense: usize,
    n_experts: usize,
    budget: usize,
    bytes_of: impl Fn(usize, usize) -> usize,
) -> Vec<(usize, usize)> {
    if n_experts == 0 || first_dense >= n_layers || budget == 0 {
        return Vec::new();
    }
    // All-zero → deterministic fallback to the round-robin cold start, sized by
    // the first candidate (uniform on a cold tier).
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        let uniform = bytes_of(first_dense, 0);
        if uniform == 0 {
            return Vec::new();
        }
        return plan_residency(n_layers, first_dense, n_experts, uniform, budget);
    }
    // Score each candidate by heat density (heat per byte), so a small expert
    // with modest heat can outrank a larger cold one. Compared as a cross
    // product to avoid integer-division ties collapsing distinct ratios.
    let mut cand: Vec<((usize, usize), u32, usize)> = Vec::new();
    for e in 0..n_experts {
        for layer in first_dense..n_layers {
            let heat = counts.get(layer * n_experts + e).copied().unwrap_or(0);
            let bytes = bytes_of(layer, e);
            if bytes == 0 || bytes > budget {
                continue; // cannot be resident at any ordering
            }
            cand.push(((layer, e), heat, bytes));
        }
    }
    cand.sort_by(|a, b| {
        let lhs = a.1 as u128 * b.2 as u128; // heat_a / bytes_a vs heat_b / bytes_b
        let rhs = b.1 as u128 * a.2 as u128;
        rhs.cmp(&lhs).then(a.0 .0.cmp(&b.0 .0)).then(a.0 .1.cmp(&b.0 .1))
    });
    // Greedy fill; keep scanning past a too-large candidate so smaller ones can
    // still use the remaining room.
    let mut out = Vec::new();
    let mut used = 0usize;
    for (key, _, bytes) in cand {
        if used.saturating_add(bytes) > budget {
            continue;
        }
        out.push(key);
        used += bytes;
    }
    out
}

/// Per-expert VRAM residency precision. The hottest experts earn the
/// high-precision f32 residency (better numerics on the most-used weights);
/// the long tail stays per-row int4 (8× denser → more experts resident).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPrecision {
    F32,
    Int4,
}

/// Per-expert adaptive precision planner: given the residency set and the heat
/// table, promote the hottest `f32_frac` of residents to f32 and leave the rest
/// int4. Deterministic ties (equal heat → lower layer, lower expert first).
/// Pure — unit-testable without a GPU; the `cuda` tier applies it in `reheat`.
pub fn plan_precision(
    counts: &[u32],
    n_experts: usize,
    resident: &[(usize, usize)],
    f32_frac: f32,
) -> Vec<((usize, usize), ExpertPrecision)> {
    let f32_frac = f32_frac.clamp(0.0, 1.0);
    let n_f32 = ((resident.len() as f32) * f32_frac).ceil() as usize;
    let heat = |&(l, e): &(usize, usize)| counts.get(l * n_experts + e).copied().unwrap_or(0);
    let mut order: Vec<(usize, usize)> = resident.to_vec();
    order.sort_by(|a, b| heat(b).cmp(&heat(a)).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    order
        .into_iter()
        .enumerate()
        .map(|(i, k)| (k, if i < n_f32 { ExpertPrecision::F32 } else { ExpertPrecision::Int4 }))
        .collect()
}

/// Rank sparse `(layer, expert)` pairs by heat and take the hottest `capacity`.
/// Ties (equal heat — including a cold all-zero table) keep the round-robin order
/// of [`plan_residency`], so cold start reproduces the static placement and
/// residency only *shifts* toward hot experts as heat accumulates. Pure.
pub fn rank_by_heat(counts: &[u32], n_layers: usize, first_dense: usize, n_experts: usize, capacity: usize) -> Vec<(usize, usize)> {
    if n_experts == 0 || first_dense >= n_layers || capacity == 0 {
        return Vec::new();
    }
    // candidates in round-robin order — the stable tiebreak (== static placement)
    let mut cand: Vec<(usize, usize)> = Vec::new();
    for e in 0..n_experts {
        for layer in first_dense..n_layers {
            cand.push((layer, e));
        }
    }
    let heat = |&(l, e): &(usize, usize)| counts.get(l * n_experts + e).copied().unwrap_or(0);
    cand.sort_by_key(|c| Reverse(heat(c))); // stable: equal heat keeps round-robin order
    cand.truncate(capacity);
    cand
}

#[cfg(test)]
mod residency_tests {
    use super::{solve_residency_sized, HeatTable};

    #[test]
    fn sized_solver_respects_a_byte_budget_with_mixed_formats() {
        // The failure this guards: a plan sized in *experts* while some
        // residents are f32 (~8× the int4 footprint) overcommits VRAM, and every
        // upload past the real limit fails — each reheat, forever.
        let (n_layers, first_dense, n_experts) = (2usize, 0usize, 4usize);
        // Layer 0 experts are "f32-promoted" (big), layer 1 stays int4 (small).
        let (big, small) = (800usize, 100usize);
        let bytes_of = |layer: usize, _e: usize| if layer == 0 { big } else { small };
        // Hot: layer 0 expert 0, then layer 1 experts.
        let mut counts = vec![0u32; n_layers * n_experts];
        counts[0] = 100; // (0,0) big + hottest
        counts[n_experts + 1] = 90; // (1,1) small
        counts[n_experts + 2] = 80; // (1,2) small
        let budget = 1000usize;
        let plan = solve_residency_sized(&counts, n_layers, first_dense, n_experts, budget, bytes_of);
        let total: usize = plan.iter().map(|&(l, e)| bytes_of(l, e)).sum();
        assert!(total <= budget, "plan takes {total} bytes of a {budget} budget: {plan:?}");
        assert!(!plan.is_empty(), "a workable plan exists");
        // Density ordering: the two small hot experts beat the big one per byte.
        assert!(plan.contains(&(1, 1)) && plan.contains(&(1, 2)), "hot small experts are selected: {plan:?}");
    }

    #[test]
    fn sized_solver_skips_candidates_that_cannot_fit() {
        let bytes_of = |_l: usize, e: usize| if e == 0 { 10_000 } else { 10 };
        let counts = vec![100u32, 50, 25, 10];
        let plan = solve_residency_sized(&counts, 1, 0, 4, 100, bytes_of);
        assert!(!plan.contains(&(0, 0)), "an expert larger than the whole budget is skipped");
        assert!(!plan.is_empty(), "smaller experts still use the room");
    }

    #[test]
    fn heat_table_rejects_a_mismatched_snapshot() {
        // A snapshot from a checkpoint with a different expert count would
        // reinterpret a flat row-major array against the new stride, landing
        // every count on the wrong (layer, expert) pair — and heat is additive,
        // so it never washes out.
        let t = HeatTable::new(2, 4); // 8 slots
        assert!(!t.restore(&[1, 2, 3]), "short snapshot rejected");
        assert!(!t.restore(&[1u32; 12]), "long snapshot rejected");
        assert!(t.restore(&[7u32; 8]), "exact-shape snapshot accepted");
        assert_eq!(t.get(1, 3), 7);
    }

    #[test]
    fn heat_table_ignores_out_of_range_experts() {
        let t = HeatTable::new(2, 4);
        t.bump(0, 4); // == n_experts: used to credit (1, 0)
        t.bump(0, usize::MAX); // used to wrap the multiply in release
        assert_eq!(t.get(1, 0), 0, "no cross-slot credit");
        assert_eq!(t.get(0, 4), 0, "out-of-range reads are zero, not a panic");
        t.bump(0, 3);
        assert_eq!(t.get(0, 3), 1);
    }
}

#[cfg(test)]
mod precision_tests {
    use super::{plan_precision, ExpertPrecision};

    #[test]
    fn hottest_fraction_promoted_to_f32() {
        // 4 residents on one layer; heat 30, 10, 20, 0 → frac 0.5 promotes the
        // top two (experts 0 and 2).
        let counts = vec![30u32, 10, 20, 0];
        let resident = vec![(0usize, 0usize), (0, 1), (0, 2), (0, 3)];
        let plan = plan_precision(&counts, 4, &resident, 0.5);
        let p = |e: usize| plan.iter().find(|((_, x), _)| *x == e).map(|&(_, p)| p);
        assert_eq!(p(0), Some(ExpertPrecision::F32));
        assert_eq!(p(2), Some(ExpertPrecision::F32));
        assert_eq!(p(1), Some(ExpertPrecision::Int4));
        assert_eq!(p(3), Some(ExpertPrecision::Int4));
    }

    #[test]
    fn frac_zero_and_one_are_uniform() {
        let counts = vec![5u32; 4];
        let resident = vec![(0usize, 0usize), (0, 1), (0, 2), (0, 3)];
        assert!(plan_precision(&counts, 4, &resident, 0.0).iter().all(|&(_, p)| p == ExpertPrecision::Int4));
        assert!(plan_precision(&counts, 4, &resident, 1.0).iter().all(|&(_, p)| p == ExpertPrecision::F32));
    }

    #[test]
    fn ties_break_deterministically() {
        // equal heat → (layer, expert) ascending decides who gets the f32 slots.
        let counts = vec![7u32; 4];
        let resident = vec![(0usize, 3usize), (0, 1), (0, 0), (0, 2)];
        let plan = plan_precision(&counts, 4, &resident, 0.5);
        let f32s: Vec<usize> = plan.iter().filter(|&&(_, p)| p == ExpertPrecision::F32).map(|&((_, e), _)| e).collect();
        assert_eq!(f32s, vec![0, 1], "lowest ids win the tie");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_residency_tests {
    use super::real::GpuTier;
    use crate::testkit::build_tiny_model;
    use peregrine_core::{Cfg, Error, SafeTensors};

    // These tests share the single GPU (context, VRAM, global scratch), so they
    // must run serially; `unwrap_or_else` recovers a poisoned lock (a panicked
    // test) without tripping the no-`unwrap` gate.
    static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn tier_spans_multiple_sparse_layers() -> Result<(), Error> {
        // On a real GPU, the round-robin placement must put residents in BOTH of
        // the tiny model's sparse layers (1 and 2) — the old greedy fill would have
        // packed layer 1 first. Skips gracefully when no GPU is available.
        let _g = gpu_guard();
        let dir = std::env::temp_dir().join(format!("peregrine_gputier_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let tier = GpuTier::build(&st, &cfg, 0)?; // headroom 0: tiny experts always fit
        std::fs::remove_dir_all(&dir)?;
        let Some(tier) = tier else {
            return Ok(()); // no CUDA device on this host → skip
        };
        assert!(tier.has(1, 0) && tier.has(2, 0), "VRAM residency must span both sparse layers");
        Ok(())
    }

    #[test]
    fn reheat_keeps_hot_experts_resident() -> Result<(), Error> {
        // Build a tier over the tiny model (all experts fit → capacity == total),
        // then reheat with synthetic heat. At full capacity the hottest set is the
        // whole set, so reheat must keep the resident count and the hot expert —
        // exercising the rank → retain → skip-upload path end to end on the GPU.
        let _g = gpu_guard();
        let dir = std::env::temp_dir().join(format!("peregrine_reheat_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let tier = GpuTier::build(&st, &cfg, 0)?;
        let Some(mut tier) = tier else {
            std::fs::remove_dir_all(&dir)?;
            return Ok(()); // no CUDA device on this host → skip
        };
        let before = tier.len();
        let n_experts = cfg.n_experts as usize;
        let mut counts = vec![0u32; cfg.n_layers as usize * n_experts];
        counts[2 * n_experts] = 99; // make (layer 2, expert 0) the hottest
        let after = tier.reheat(&st, &cfg, &counts)?;
        std::fs::remove_dir_all(&dir)?;
        assert!(before > 0, "tiny experts must fit VRAM");
        assert_eq!(after, before, "reheat at full capacity keeps the resident count");
        assert!(tier.has(2, 0), "the hottest expert must remain resident");
        Ok(())
    }

    #[test]
    fn int4_tier_builds_on_per_row_int4() -> Result<(), Error> {
        // The tiny model's experts are per-row int4, so an int4 tier (8× denser than
        // f32) must build and span the sparse layers — the density path end to end.
        let _g = gpu_guard();
        let dir = std::env::temp_dir().join(format!("peregrine_int4tier_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let tier = GpuTier::build_with(&st, &cfg, 0, true)?;
        std::fs::remove_dir_all(&dir)?;
        let Some(tier) = tier else {
            return Ok(()); // no CUDA device on this host → skip
        };
        assert!(!tier.is_empty(), "int4 tier must hold experts on a per-row-int4 model");
        assert!(tier.has(1, 0) || tier.has(2, 0), "int4 residency spans the sparse layers");
        Ok(())
    }
}

#[cfg(test)]
mod placement_tests {
    use super::plan_residency;
    use std::collections::BTreeSet;

    #[test]
    fn spreads_across_all_sparse_layers() {
        // 8 layers, first_dense=2 → 6 sparse layers; budget fits 12 experts.
        // Round-robin must cover every sparse layer (2 experts each), not just the
        // first two (which the old greedy fill would have done).
        let placed = plan_residency(8, 2, 16, 100, 12 * 100);
        assert_eq!(placed.len(), 12);
        let layers: BTreeSet<usize> = placed.iter().map(|&(l, _)| l).collect();
        assert_eq!(layers, (2..8).collect(), "every sparse layer must get a share");
        // exactly experts {0,1} in each layer
        for l in 2..8 {
            let mut es: Vec<usize> = placed.iter().filter(|&&(pl, _)| pl == l).map(|&(_, e)| e).collect();
            es.sort_unstable();
            assert_eq!(es, vec![0, 1]);
        }
    }

    #[test]
    fn respects_capacity_and_partial_round() {
        // budget fits 7 experts across 3 sparse layers (5..8): first round places 3
        // (one per layer), second round places 3, then 1 more on the first layer.
        let placed = plan_residency(8, 5, 16, 10, 7 * 10);
        assert_eq!(placed.len(), 7);
        let layers: BTreeSet<usize> = placed.iter().map(|&(l, _)| l).collect();
        assert_eq!(layers, (5..8).collect(), "all sparse layers still covered");
    }

    #[test]
    fn edge_cases_yield_empty() {
        assert!(plan_residency(4, 4, 8, 100, 1 << 30).is_empty()); // no sparse layers
        assert!(plan_residency(8, 2, 0, 100, 1 << 30).is_empty()); // no experts
        assert!(plan_residency(8, 2, 16, 0, 1 << 30).is_empty()); // zero bytes/expert
        assert!(plan_residency(8, 2, 16, 100, 0).is_empty()); // zero budget
        assert!(plan_residency(8, 2, 16, 100, 50).is_empty()); // budget < one expert
    }

    #[test]
    fn rank_by_heat_cold_matches_static() {
        // an all-zero (cold) heat table must reproduce plan_residency exactly, so
        // enabling heat never changes cold-start residency.
        let counts = vec![0u32; 8 * 16];
        let capacity = 12;
        let heat = super::rank_by_heat(&counts, 8, 2, 16, capacity);
        let stat = plan_residency(8, 2, 16, 100, capacity * 100);
        assert_eq!(heat, stat, "cold heat ranking must equal the static round-robin");
    }

    #[test]
    fn rank_by_heat_prefers_hot_experts() {
        // make (layer 5, expert 9) and (layer 3, expert 1) the hottest; with a small
        // capacity they must rank first regardless of round-robin index order.
        let (n_layers, first_dense, n_experts) = (8usize, 2usize, 16usize);
        let mut counts = vec![0u32; n_layers * n_experts];
        counts[5 * n_experts + 9] = 100;
        counts[3 * n_experts + 1] = 50;
        let top = super::rank_by_heat(&counts, n_layers, first_dense, n_experts, 2);
        assert_eq!(top, vec![(5, 9), (3, 1)], "hottest experts must rank first");
    }
}

#[cfg(feature = "cuda")]
mod real {
    use std::collections::{HashMap, HashSet};

    use peregrine_core::{Cfg, Error, SafeTensors};
    use peregrine_cuda::GpuExpert;

    use crate::weight::QtWeight;

    /// VRAM-resident experts, keyed by `(layer, expert index)`. `capacity` is how
    /// many fit the VRAM budget (fixed at build), so [`Self::reheat`] re-selects the
    /// same-size hottest set as heat accumulates. `int4` picks the residency format:
    /// per-row int4 (8× denser) vs dequantized f32. When `adaptive_f32_frac` is
    /// set (`COLI_GPU_F32_FRAC`, int4 tiers only), `reheat` promotes that fraction
    /// of the hottest residents to f32 per [`super::plan_precision`], tracking each
    /// expert's current format in `precision` and re-uploading on a change.
    pub struct GpuTier {
        device: i32,
        experts: HashMap<(usize, usize), GpuExpert>,
        capacity: usize,
        /// VRAM the resident set may occupy. `capacity` alone could not express
        /// this once adaptive precision existed: an f32-promoted expert is ~8×
        /// its int4 size, so "N experts" silently became "up to 8N experts'
        /// worth of VRAM" and `reheat` OOM'd every generation.
        budget_bytes: usize,
        /// Per-expert footprint in each residency format, `(int4, f32)`.
        expert_bytes: (usize, usize),
        int4: bool,
        adaptive_f32_frac: Option<f32>,
        /// Current per-expert residency format (`true` = int4). Only consulted
        /// on the adaptive path; uniform tiers leave it empty.
        precision: HashMap<(usize, usize), bool>,
    }

    /// Load and upload one expert to VRAM: raw per-row int4 (`fmt=2`, ~8× denser)
    /// when `int4` and the source is per-row int4, else dequantized f32. Shared by
    /// [`GpuTier::build`] and [`GpuTier::reheat`] so both formats stay in one place.
    fn upload_expert(st: &SafeTensors, cfg: &Cfg, layer: usize, e: usize, device: i32, int4: bool) -> Result<GpuExpert, Error> {
        let hidden = cfg.hidden as usize;
        let inter = cfg.moe_inter as usize;
        let pe = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
        let gate = QtWeight::load(st, &pe("gate_proj.weight"), inter, hidden)?;
        let up = QtWeight::load(st, &pe("up_proj.weight"), inter, hidden)?;
        let down = QtWeight::load(st, &pe("down_proj.weight"), hidden, inter)?;
        if int4 {
            use crate::weight::QuantFmt;
            if gate.fmt != QuantFmt::Int4 || up.fmt != QuantFmt::Int4 || down.fmt != QuantFmt::Int4 {
                return Err(Error::Format(format!(
                    "COLI_GPU_INT4 needs per-row int4 experts; layer {layer} expert {e} is {:?} \
                     (grouped int4 / int8 sources need the f32 tier or a requantize first)",
                    gate.fmt
                )));
            }
            GpuExpert::upload_int4(device, gate.raw(), up.raw(), down.raw(), hidden, inter)
        } else {
            GpuExpert::upload(device, &gate.dequant(), &up.dequant(), &down.dequant(), hidden, inter)
        }
    }

    impl GpuTier {
        /// Build the tier by dequantizing routed experts to f32 and uploading as
        /// many as fit within `free VRAM - headroom`, iterating sparse layers and
        /// experts in order. `Ok(None)` when CUDA is unavailable or nothing fits.
        /// Build the tier, choosing int4-resident (8× denser) vs f32 residency from
        /// the `COLI_GPU_INT4` env var. See [`Self::build_with`].
        pub fn build(st: &SafeTensors, cfg: &Cfg, headroom_bytes: usize) -> Result<Option<GpuTier>, Error> {
            Self::build_with(st, cfg, headroom_bytes, std::env::var("COLI_GPU_INT4").is_ok())
        }

        /// Build by uploading as many routed experts as fit `free VRAM − headroom`,
        /// spread round-robin across all sparse layers. `int4` uploads per-row int4
        /// weights directly (~8× denser; needs per-row int4 sources), else
        /// dequantized f32. `Ok(None)` when CUDA is unavailable or nothing fits.
        /// Takes `int4` explicitly so it is testable without racing process env.
        pub fn build_with(st: &SafeTensors, cfg: &Cfg, headroom_bytes: usize, int4: bool) -> Result<Option<GpuTier>, Error> {
            if peregrine_cuda::init(&[0]) < 1 {
                return Ok(None);
            }
            let device = 0;
            let (free, _total) = peregrine_cuda::mem_info(device)?;
            let hidden = cfg.hidden as usize;
            let inter = cfg.moe_inter as usize;
            // gate + up ([inter,hidden]) + down ([hidden,inter]); int4 = packed
            // nibbles + f32 row scales (~8× smaller than the f32 residency).
            let weights = 2 * inter * hidden + hidden * inter;
            let int4_bytes = weights / 2 + (2 * inter + hidden) * 4;
            let f32_bytes = weights * 4;
            let bytes_per_expert = if int4 { int4_bytes } else { f32_bytes };
            let budget = free.saturating_sub(headroom_bytes);

            let placement = super::plan_residency(
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                bytes_per_expert,
                budget,
            );
            let mut capacity = placement.len(); // expert-count view of the budget
            let mut experts = HashMap::new();
            for (layer, e) in placement {
                match upload_expert(st, cfg, layer, e, device, int4) {
                    Ok(ge) => {
                        experts.insert((layer, e), ge);
                    }
                    // A tail-end upload can fail on allocation granularity even
                    // though the byte arithmetic said it fits. Settle for the
                    // experts that did land rather than failing the whole model
                    // load, and shrink the capacity to match reality.
                    Err(e_up) => {
                        peregrine_io::note_advisory_err("gpu residency upload (tier truncated)", &e_up);
                        capacity = experts.len();
                        break;
                    }
                }
            }

            if experts.is_empty() {
                Ok(None)
            } else {
                // Adaptive per-expert precision: COLI_GPU_F32_FRAC=<0..1> promotes
                // that fraction of the hottest residents to f32 at each reheat.
                // Meaningful only on an int4 tier (a f32 tier is already all-f32).
                let adaptive_f32_frac = if int4 {
                    std::env::var("COLI_GPU_F32_FRAC")
                        .ok()
                        .and_then(|v| v.trim().parse::<f32>().ok())
                        // "inf" and "NaN" both parse: inf clamps to 1.0 and
                        // promotes *every* resident to f32 (the worst case for
                        // the VRAM budget), while NaN silently disables the
                        // feature. Neither is what the operator asked for.
                        .filter(|f| f.is_finite() && (0.0..=1.0).contains(f))
                } else {
                    None
                };
                Ok(Some(GpuTier {
                    device,
                    experts,
                    capacity,
                    budget_bytes: budget,
                    expert_bytes: (int4_bytes, f32_bytes),
                    int4,
                    adaptive_f32_frac,
                    precision: HashMap::new(),
                }))
            }
        }

        /// Re-select the VRAM-resident set as the `capacity` hottest experts by
        /// `counts` (routing frequency), evicting experts that cooled and uploading
        /// newly-hot ones. Reuses [`Self::build`]'s dequantize+upload path. Called
        /// between forwards with `&mut self`, so residency adapts to the workload
        /// without a rewrite. Returns the resident count after re-selection.
        pub fn reheat(&mut self, st: &SafeTensors, cfg: &Cfg, counts: &[u32]) -> Result<usize, Error> {
            // Re-read free VRAM each generation: another process may have taken
            // some since load, and the budget must reflect what is available now
            // (plus what this tier already holds, which it is free to reuse).
            let held: usize = self
                .experts
                .keys()
                .map(|k| {
                    let int4 = self.precision.get(k).copied().unwrap_or(self.int4);
                    if int4 { self.expert_bytes.0 } else { self.expert_bytes.1 }
                })
                .sum();
            let budget = match peregrine_cuda::mem_info(self.device) {
                Ok((free, _)) => self.budget_bytes.min(free.saturating_add(held)),
                // Query failure: keep the load-time budget rather than guessing.
                Err(e) => {
                    peregrine_io::note_advisory_err("gpu mem_info during reheat", &e);
                    self.budget_bytes
                }
            };
            let want = super::rank_by_heat(
                counts,
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                self.capacity,
            );
            // Per-expert precision for this residency generation: adaptive
            // (hottest frac → f32) or the tier's uniform format.
            let precision_of: HashMap<(usize, usize), bool> = match self.adaptive_f32_frac {
                Some(frac) => super::plan_precision(counts, cfg.n_experts as usize, &want, frac)
                    .into_iter()
                    .map(|(k, p)| (k, matches!(p, super::ExpertPrecision::Int4)))
                    .collect(),
                None => want.iter().map(|&k| (k, self.int4)).collect(),
            };
            // Trim the wish list to what actually fits *in bytes*. With adaptive
            // precision the hottest residents are f32 (~8× the int4 footprint),
            // so a plan sized purely by expert count overcommits VRAM by up to
            // that factor and every upload past the real limit fails.
            let (int4_bytes, f32_bytes) = self.expert_bytes;
            let mut fitted: Vec<(usize, usize)> = Vec::with_capacity(want.len());
            let mut used = 0usize;
            for key in want {
                let bytes = if precision_of.get(&key).copied().unwrap_or(self.int4) { int4_bytes } else { f32_bytes };
                if used.saturating_add(bytes) > budget {
                    continue;
                }
                used += bytes;
                fitted.push(key);
            }
            let want = fitted;
            let want_set: HashSet<(usize, usize)> = want.iter().copied().collect();
            // evict experts that cooled off — their `Drop` frees the VRAM slot
            self.experts.retain(|k, _| want_set.contains(k));
            self.precision.retain(|k, _| want_set.contains(k));
            for (layer, e) in want {
                let key = (layer, e);
                let want_int4 = precision_of.get(&key).copied().unwrap_or(self.int4);
                // An expert uploaded by `build` (absent from `precision`) is in the
                // tier's uniform format — treat missing as that, so a non-adaptive
                // reheat never re-uploads a format-correct resident.
                let cur_int4 = self.precision.get(&key).copied().unwrap_or(self.int4);
                if self.experts.contains_key(&key) && cur_int4 == want_int4 {
                    continue;
                }
                // Re-upload on a format change (remove first so the old tensor's
                // Drop frees its VRAM before the new allocation).
                self.experts.remove(&key);
                match upload_expert(st, cfg, layer, e, self.device, want_int4) {
                    Ok(ge) => {
                        self.experts.insert(key, ge);
                        self.precision.insert(key, want_int4);
                    }
                    // Keep the generation that did upload instead of bailing with
                    // the tier holding fewer experts than it thinks and no record
                    // of which are missing. The next reheat retries from a
                    // consistent state.
                    Err(e_up) => {
                        peregrine_io::note_advisory_err("gpu reheat upload (residency kept partial)", &e_up);
                        self.precision.remove(&key);
                        break;
                    }
                }
            }
            Ok(self.experts.len())
        }

        /// Whether expert `e` of `layer` is resident in VRAM.
        pub fn has(&self, layer: usize, e: usize) -> bool {
            self.experts.contains_key(&(layer, e))
        }

        /// Number of experts resident (for logging).
        pub fn len(&self) -> usize {
            self.experts.len()
        }

        /// Whether the tier holds no experts.
        pub fn is_empty(&self) -> bool {
            self.experts.is_empty()
        }

        /// The device this tier lives on.
        pub fn device(&self) -> i32 {
            self.device
        }

        /// Compute a batch of this layer's GPU-resident experts. `jobs[k]` is
        /// `(expert index, gathered input rows [nr*hidden])`; returns each
        /// expert's SwiGLU output `[nr*hidden]`, in the same order.
        pub fn compute(&self, layer: usize, jobs: &[(usize, Vec<f32>)], hidden: usize) -> Result<Vec<Vec<f32>>, Error> {
            if jobs.is_empty() {
                return Ok(Vec::new());
            }
            let mut refs = Vec::with_capacity(jobs.len());
            let mut rows = Vec::with_capacity(jobs.len());
            let mut x = Vec::new();
            for (e, xg) in jobs {
                let ge = self
                    .experts
                    .get(&(layer, *e))
                    .ok_or_else(|| Error::Format(format!("gpu expert ({layer},{e}) not resident")))?;
                if hidden == 0 || !xg.len().is_multiple_of(hidden) {
                    return Err(Error::Format("gpu compute: ragged gathered rows".into()));
                }
                refs.push(ge);
                rows.push((xg.len() / hidden) as i32);
                x.extend_from_slice(xg);
            }
            let y = peregrine_cuda::expert_group(&refs, &rows, &x, hidden)?;
            let mut out = Vec::with_capacity(jobs.len());
            let mut off = 0usize;
            for &r in &rows {
                let n = r as usize * hidden;
                out.push(y[off..off + n].to_vec());
                off += n;
            }
            Ok(out)
        }
    }

    impl Drop for GpuTier {
        fn drop(&mut self) {
            // GpuExpert handles free themselves; release the device contexts too.
            self.experts.clear();
            peregrine_cuda::shutdown();
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod stub {
    use peregrine_core::{Cfg, Error, SafeTensors};

    /// Empty tier for non-`cuda` builds — `build` always yields `None`, so it is
    /// never constructed and the scheduler always takes the CPU/disk path.
    pub struct GpuTier {
        _never: (),
    }

    impl GpuTier {
        pub fn build(_st: &SafeTensors, _cfg: &Cfg, _headroom_bytes: usize) -> Result<Option<GpuTier>, Error> {
            Ok(None)
        }
        pub fn has(&self, _layer: usize, _e: usize) -> bool {
            false
        }
        pub fn len(&self) -> usize {
            0
        }
        pub fn is_empty(&self) -> bool {
            true
        }
        pub fn compute(&self, _layer: usize, _jobs: &[(usize, Vec<f32>)], _hidden: usize) -> Result<Vec<Vec<f32>>, Error> {
            Err(Error::Format("gpu tier not built (no cuda feature)".into()))
        }
        pub fn reheat(&mut self, _st: &SafeTensors, _cfg: &Cfg, _counts: &[u32]) -> Result<usize, Error> {
            Ok(0)
        }
    }
}

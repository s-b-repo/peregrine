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

/// Bytes one resident expert actually occupies in VRAM.
///
/// `raw_int4` must reflect what the *container* holds, not what the operator
/// asked for. Raw int4 residency requires all three projections to be per-row
/// int4 (`upload_int4`); every other source format — grouped int4, int8,
/// int3-g64, int2-g64 — is uploaded **dequantized to f32**, which is 8× the
/// int4 figure.
///
/// Planning with the int4 number on a container that cannot use it budgets `N`
/// experts and then uploads `8N` worth. The runtime byte tracker in
/// `build_with` stops the upload before VRAM overruns, so this does not crash —
/// it silently delivers a tier roughly 8× smaller than the operator asked for,
/// which is worse than crashing because nothing reports it. `validation-runbook`
/// §4 flags exactly this for an int3 container; every sub-4-bit format added
/// since makes it likelier.
pub fn resident_bytes_per_expert(hidden: usize, inter: usize, raw_int4: bool) -> usize {
    // gate + up ([inter,hidden]) + down ([hidden,inter])
    let weights = 2 * inter * hidden + hidden * inter;
    if raw_int4 {
        // packed nibbles + f32 row scales
        weights / 2 + (2 * inter + hidden) * 4
    } else {
        weights * 4
    }
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

/// Plan a mixed-precision resident set that actually fits `budget` bytes.
///
/// [`plan_precision`] promotes a fraction of an *already chosen* resident set,
/// which is circular: promoting to a wider format shrinks how many experts fit,
/// which changes the set the fraction was taken over. Done naively — rank the
/// hottest `capacity` experts (a count derived from the *narrow* format), then
/// promote `frac` of them, then drop whatever overruns — the wide residents eat
/// the budget before the narrow tail is ever reached. At the repo's GLM-5.2
/// shape (18.9 MB int4 / 151 MB f32) with a 10 GB budget and `frac = 0.25`,
/// that turns 542 residents into ~67: a 25% *quality* request costing 88% of
/// *residency*.
///
/// Solved directly instead. If the final set holds `R` experts of which
/// `frac·R` are wide, then `budget = R·(frac·hi + (1-frac)·lo)`, so
/// `R = budget / (frac·hi + (1-frac)·lo)`. Ranking every sparse candidate and
/// promoting the hottest `frac·R` makes the fraction self-consistent with the
/// set it describes, and the greedy fit then packs the narrow tail into any
/// remainder. Pure — unit-testable without a GPU.
/// `bytes` is `(wide, narrow)` — the per-expert footprint of each format.
pub fn plan_precision_fitted(
    counts: &[u32],
    n_layers: usize,
    first_dense: usize,
    n_experts: usize,
    budget: usize,
    hi_frac: f32,
    bytes: (usize, usize),
) -> Vec<((usize, usize), ExpertPrecision)> {
    let (hi_bytes, lo_bytes) = bytes;
    if n_experts == 0 || first_dense >= n_layers || budget == 0 || lo_bytes == 0 {
        return Vec::new();
    }
    let hi_frac = hi_frac.clamp(0.0, 1.0) as f64;
    // Self-consistent resident count, then the wide share of it.
    let avg = hi_frac * hi_bytes as f64 + (1.0 - hi_frac) * lo_bytes as f64;
    let n_hi = if avg > 0.0 { ((budget as f64 / avg) * hi_frac).ceil() as usize } else { 0 };

    // Candidates in round-robin order, so equal heat (including a cold all-zero
    // table) reproduces `plan_residency`'s static placement — same tiebreak as
    // `rank_by_heat`.
    let mut cand: Vec<(usize, usize)> = Vec::with_capacity((n_layers - first_dense) * n_experts);
    for e in 0..n_experts {
        for layer in first_dense..n_layers {
            cand.push((layer, e));
        }
    }
    let heat = |&(l, e): &(usize, usize)| counts.get(l * n_experts + e).copied().unwrap_or(0);
    cand.sort_by_key(|c| Reverse(heat(c))); // stable: equal heat keeps round-robin order

    // Greedy fit over the full ranked list. Keep scanning past a candidate that
    // does not fit so the narrow tail can still use the remaining room.
    let mut out = Vec::new();
    let mut used = 0usize;
    for (rank, key) in cand.into_iter().enumerate() {
        // A promoted candidate that no longer fits at the wide size falls back to
        // the narrow one rather than being dropped. Without this the *hottest*
        // expert is the one evicted whenever the remaining room is smaller than a
        // wide slot — precisely backwards, since rank is what earned it residency.
        let (prec, bytes) = match rank < n_hi {
            true if used.saturating_add(hi_bytes) <= budget => (ExpertPrecision::F32, hi_bytes),
            _ => (ExpertPrecision::Int4, lo_bytes),
        };
        if used.saturating_add(bytes) > budget {
            continue;
        }
        used += bytes;
        out.push((key, prec));
    }
    out
}

/// How many of `costs` a single reheat generation may upload before hitting
/// `budget_bytes`. `costs` is in placement order, which is heat order, so the
/// answer is a prefix length.
///
/// `reheat` otherwise re-uploads every expert whose heat rank moved, unbounded,
/// once per 256 decode steps — at ~18.9 MB (int4) or ~151 MB (f32) each, a
/// churny generation can push gigabytes across PCIe in one burst and stall the
/// lane it is supposed to be feeding. Capping the burst spreads the same
/// migration over several generations; the experts deferred here are the coldest
/// in the plan, and the next generation reconsiders them.
///
/// `budget_bytes == 0` means unlimited — the default, and bit-identical to having
/// no governor at all. A budget smaller than the first upload still admits one,
/// so residency always makes progress instead of deadlocking on a too-tight knob.
///
/// Pure — no CUDA measurement is involved; the byte costs are known from the
/// residency format alone.
pub fn admit_uploads(costs: &[usize], budget_bytes: usize) -> usize {
    if budget_bytes == 0 {
        return costs.len();
    }
    let mut used = 0usize;
    for (i, &c) in costs.iter().enumerate() {
        if used.saturating_add(c) > budget_bytes {
            return i.max(1).min(costs.len()); // never stall completely
        }
        used += c;
    }
    costs.len()
}

/// Per-generation PCIe upload budget in bytes from `COLI_PCIE_BUDGET_MB`.
/// `0`/unset/invalid → unlimited, which is the untouched behaviour.
pub fn pcie_budget_bytes() -> usize {
    std::env::var("COLI_PCIE_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(0)
}

/// Group job indices by residency format, preserving each group's relative order.
///
/// The CUDA expert-group kernel picks its dispatch path from `all_s4`, computed
/// over the *whole* call (`backend_cuda.cu:638,645`): if any expert in the group
/// is f32, every expert in that call falls off the int4 Tensor-Core and packed-W4
/// ladders onto the generic scalar path. Issuing one call per format class keeps
/// the int4 majority on the fast path.
///
/// Returns one index list per non-empty class (int4 first, then f32) — never an
/// empty list, since the kernel rejects a group with no rows. A homogeneous input
/// yields exactly one class, i.e. the single call made today.
///
/// Pure — testable without a GPU.
pub fn partition_by_format(fmt: &[bool]) -> Vec<Vec<usize>> {
    let mut int4: Vec<usize> = Vec::new();
    let mut f32s: Vec<usize> = Vec::new();
    for (i, &is_int4) in fmt.iter().enumerate() {
        if is_int4 {
            int4.push(i);
        } else {
            f32s.push(i);
        }
    }
    [int4, f32s].into_iter().filter(|c| !c.is_empty()).collect()
}

/// Undo [`partition_by_format`]: place each class's results back at their original
/// job positions.
///
/// `results[c][i]` is the output for job `classes[c][i]`. Returns `None` if the
/// shapes disagree or an index is out of range or repeated — the caller turns that
/// into an error rather than silently emitting misaligned outputs, which would
/// attach every expert's result to the wrong rows with no other symptom.
pub fn scatter_by_index<T>(total: usize, classes: &[Vec<usize>], results: Vec<Vec<T>>) -> Option<Vec<T>> {
    if classes.len() != results.len() {
        return None;
    }
    let mut slots: Vec<Option<T>> = (0..total).map(|_| None).collect();
    for (idxs, vals) in classes.iter().zip(results) {
        if idxs.len() != vals.len() {
            return None;
        }
        for (&i, v) in idxs.iter().zip(vals) {
            let slot = slots.get_mut(i)?;
            if slot.is_some() {
                return None; // duplicate index — the permutation is not a bijection
            }
            *slot = Some(v);
        }
    }
    slots.into_iter().collect()
}

/// Whether `reheat` must evict and re-upload a resident expert this generation.
///
/// Re-upload is expensive — a full host dequantize (~151 MB for an f32 expert)
/// plus the PCIe transfer — so it must happen only on a real format change.
///
/// The subtle case is `unsatisfiable`: an expert whose source is grouped-int4 or
/// int8 cannot be int4-resident, so an int4 request lands f32. The wanted format
/// is recomputed from the tier's uniform format every generation and keeps asking,
/// while the resident is recorded f32 — so a plain `cur != want` test never
/// converges and re-uploads that expert every 256 decode steps forever. Recording
/// the *landed* format does not fix it; the request side has to stop asking.
///
/// Pure — the churn bug is unit-testable without a GPU.
pub fn needs_reupload(resident: bool, cur_int4: bool, want_int4: bool, unsatisfiable: bool) -> bool {
    if !resident {
        return true; // not there yet — must upload
    }
    if want_int4 && unsatisfiable {
        return false; // asked once, answered f32; asking again changes nothing
    }
    cur_int4 != want_int4
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
    use super::{resident_bytes_per_expert, solve_residency_sized, HeatTable};

    #[test]
    fn resident_bytes_track_the_format_the_container_can_actually_upload() {
        // GLM-5.2 shapes. Raw int4 residency is ~8x denser than the dequantized
        // f32 fallback, and *only* per-row int4 sources can take the raw path.
        let (hidden, inter) = (5120usize, 1536usize);
        let i4 = resident_bytes_per_expert(hidden, inter, true);
        let f32b = resident_bytes_per_expert(hidden, inter, false);
        let ratio = f32b as f64 / i4 as f64;
        assert!((7.0..8.1).contains(&ratio), "f32 residency should cost ~8x int4, got {ratio:.2}x");

        // The bug this helper exists to prevent: planning with the int4 figure
        // on a container that must fall back to f32 budgets N experts and
        // uploads ~8N worth. Expressed as the capacity the planner would hand
        // out for a fixed budget.
        let budget = 10usize << 30;
        assert!(
            budget / i4 > 7 * (budget / f32b),
            "an int4-sized plan hands out ~8x the experts an f32 upload can hold"
        );
    }

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

    // GLM-5.2 shape, from `peregrine-cuda/src/lib.rs`: 18.9 MB per int4 expert,
    // 151 MB per f32 expert, 75 layers x 256 experts, ~10 GB of usable VRAM.
    const INT4_B: usize = 18_900_000;
    const F32_B: usize = 151_000_000;
    const BUDGET: usize = 10 * 1024 * 1024 * 1024;
    const LAYERS: usize = 75;
    const EXPERTS: usize = 256;

    /// The regression this planner exists for: promoting a fraction of residents
    /// to f32 must not gut the resident set.
    ///
    /// The old path ranked the hottest `capacity` experts — a count derived from
    /// the *int4* footprint — promoted `frac` of them to f32, then dropped
    /// whatever overran the budget. Because f32 is ~8x int4, the promotions alone
    /// exhausted VRAM and the int4 tail was never reached: 542 residents became
    /// ~67. Sizing the set and its f32 share together keeps the count sane.
    #[test]
    fn promotion_does_not_collapse_residency() {
        let counts: Vec<u32> = (0..LAYERS * EXPERTS).map(|i| (i % 977) as u32).collect();
        let plan = super::plan_precision_fitted(&counts, LAYERS, 0, EXPERTS, BUDGET, 0.25, (F32_B, INT4_B));

        let n_f32 = plan.iter().filter(|&&(_, p)| p == ExpertPrecision::F32).count();
        let total = plan.len();
        // Naive count-then-promote lands near 67 here; anything in that range means
        // the collapse is back.
        assert!(total > 150, "promotion collapsed residency to {total} experts");
        // The f32 share must match what was actually asked for, not just be small.
        let share = n_f32 as f64 / total as f64;
        assert!((0.20..0.30).contains(&share), "f32 share {share:.3} should be ~0.25");
        // And it must genuinely fit.
        let used: usize = plan
            .iter()
            .map(|&(_, p)| if p == ExpertPrecision::F32 { F32_B } else { INT4_B })
            .sum();
        assert!(used <= BUDGET, "plan overruns the budget: {used} > {BUDGET}");
    }

    #[test]
    fn zero_fraction_matches_a_pure_int4_tier() {
        let counts: Vec<u32> = (0..LAYERS * EXPERTS).map(|i| (i % 977) as u32).collect();
        let plan = super::plan_precision_fitted(&counts, LAYERS, 0, EXPERTS, BUDGET, 0.0, (F32_B, INT4_B));
        assert!(plan.iter().all(|&(_, p)| p == ExpertPrecision::Int4));
        // Every slot the budget can hold is used.
        assert_eq!(plan.len(), BUDGET / INT4_B);
    }

    #[test]
    fn unset_pcie_budget_admits_every_upload() {
        // The knob defaults off, and off must mean "exactly what it did before".
        let costs = vec![INT4_B; 40];
        assert_eq!(super::admit_uploads(&costs, 0), 40);
        assert_eq!(super::admit_uploads(&[], 0), 0);
    }

    #[test]
    fn pcie_budget_caps_the_generation_at_a_heat_ordered_prefix() {
        // 4 int4 uploads fit in a 4-expert budget; the 5th waits for next time.
        let costs = vec![INT4_B; 10];
        assert_eq!(super::admit_uploads(&costs, 4 * INT4_B), 4);
        assert_eq!(super::admit_uploads(&costs, 4 * INT4_B + INT4_B / 2), 4);
        // Mixed costs: one f32 promotion crowds out several int4 uploads.
        let mixed = vec![F32_B, INT4_B, INT4_B, INT4_B];
        assert_eq!(super::admit_uploads(&mixed, F32_B + 2 * INT4_B), 3);
    }

    #[test]
    fn a_too_tight_budget_still_makes_progress() {
        // A budget smaller than a single upload must not freeze residency
        // forever — one is always admitted.
        assert_eq!(super::admit_uploads(&[F32_B, F32_B], 1), 1);
        assert_eq!(super::admit_uploads(&[F32_B], 0_usize.saturating_add(1)), 1);
    }

    #[test]
    fn homogeneous_group_makes_one_class() {
        // The common case must stay exactly what it is today: a single call.
        assert_eq!(super::partition_by_format(&[true, true, true]), vec![vec![0, 1, 2]]);
        assert_eq!(super::partition_by_format(&[false, false]), vec![vec![0, 1]]);
        // Empty input yields no classes — the kernel rejects a group with no rows,
        // so an empty class must never be issued.
        assert!(super::partition_by_format(&[]).is_empty());
        assert_eq!(super::partition_by_format(&[true]), vec![vec![0]]);
    }

    #[test]
    fn mixed_group_splits_preserving_relative_order() {
        // int4 class first, then f32; within each, original order is kept.
        let classes = super::partition_by_format(&[true, false, true, false, true]);
        assert_eq!(classes, vec![vec![0, 2, 4], vec![1, 3]]);
    }

    #[test]
    fn scatter_inverts_partition_exactly() {
        // scatter ∘ partition == identity, for every mix including the edges.
        for fmt in [
            vec![true, false, true, false, true],
            vec![true; 4],
            vec![false; 4],
            vec![false, true],
            vec![true],
        ] {
            let classes = super::partition_by_format(&fmt);
            // Each job's "result" is just its own index, so the round-trip is
            // checkable by equality with 0..n.
            let per_class: Vec<Vec<usize>> = classes.to_vec();
            let got = super::scatter_by_index(fmt.len(), &classes, per_class);
            assert_eq!(got, Some((0..fmt.len()).collect::<Vec<_>>()), "round-trip failed for {fmt:?}");
        }
    }

    #[test]
    fn scatter_rejects_malformed_input_instead_of_misaligning() {
        // A misaligned scatter would silently attach every expert's output to the
        // wrong rows, so these must be errors, not best-effort results.
        let classes = vec![vec![0usize, 2], vec![1usize]];
        // Wrong number of result groups.
        assert!(super::scatter_by_index(3, &classes, vec![vec![0, 2]]).is_none());
        // Group length mismatch.
        assert!(super::scatter_by_index(3, &classes, vec![vec![0], vec![1]]).is_none());
        // Index out of range.
        assert!(super::scatter_by_index(2, &classes, vec![vec![0, 2], vec![1]]).is_none());
        // Duplicate index (not a bijection) leaves a hole.
        let dup = vec![vec![0usize, 0], vec![1usize]];
        assert!(super::scatter_by_index(3, &dup, vec![vec![0, 0], vec![1]]).is_none());
    }

    /// The regression: an expert whose source cannot be int4 must be asked once,
    /// not once per generation. Before `forced_f32`, an int4 tier holding a
    /// grouped-int4 or int8 expert evicted and re-uploaded it every 256 decode
    /// steps forever, because the wanted format is recomputed unconditionally and
    /// the resident is recorded f32 — so the two never agreed.
    #[test]
    fn unsatisfiable_int4_request_settles_after_one_upload() {
        // Not yet resident → upload, whatever the formats say.
        assert!(super::needs_reupload(false, true, true, false));
        assert!(super::needs_reupload(false, false, true, true));
        // Resident and already in the wanted format → leave it alone.
        assert!(!super::needs_reupload(true, true, true, false));
        assert!(!super::needs_reupload(true, false, false, false));
        // Resident, wants int4, source could do it but the resident is f32 →
        // a real format change, re-upload.
        assert!(super::needs_reupload(true, false, true, false));
        // Same, but the source can never be int4 → must NOT re-upload. This is
        // the case that used to churn forever.
        assert!(!super::needs_reupload(true, false, true, true));
        // Demotion to f32 is still honoured even for an unsatisfiable source —
        // `unsatisfiable` only suppresses the int4 *request*.
        assert!(super::needs_reupload(true, true, false, true));
    }

    #[test]
    fn fitted_plan_is_heat_ordered_and_deterministic() {
        // Distinct heats: expert 0 hottest, descending. The hottest must be the
        // one promoted, and two runs must agree exactly.
        let mut counts = vec![0u32; 2 * 4];
        for (e, slot) in counts.iter_mut().take(4).enumerate() {
            *slot = (4 - e) as u32; // layer 0
        }
        let a = super::plan_precision_fitted(&counts, 2, 0, 4, 3 * INT4_B, 0.34, (F32_B, INT4_B));
        let b = super::plan_precision_fitted(&counts, 2, 0, 4, 3 * INT4_B, 0.34, (F32_B, INT4_B));
        assert_eq!(a, b, "planning is deterministic");
        assert!(!a.is_empty());
        assert_eq!(a[0].0, (0, 0), "the hottest expert is placed first");
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
        // headroom 0: tiny experts always fit. `&[]` = cold heat table, which
        // `solve_residency_sized` answers with the round-robin spread this asserts.
        let tier = GpuTier::build(&st, &cfg, 0, &[])?;
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
        let tier = GpuTier::build(&st, &cfg, 0, &[])?;
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
        let tier = GpuTier::build_with(&st, &cfg, 0, true, &[])?;
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
    /// of the hottest residents to f32 per [`super::plan_precision_fitted`], tracking each
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
        /// Current per-expert residency format (`true` = int4), recorded for
        /// every resident on every path — `build` and `reheat` both insert the
        /// format that actually landed, so this is authoritative for uniform and
        /// adaptive tiers alike. Dispatch reads it to group same-format experts.
        precision: HashMap<(usize, usize), bool>,
        /// Residents whose source cannot be int4 (grouped-int4 or int8), so an
        /// int4 request fell back to f32. Without this the tier re-asks every
        /// generation: `reheat` wants `self.int4` for every expert on a uniform
        /// tier, the resident is recorded f32, the formats never match, and the
        /// expert is evicted and re-uploaded (a full ~151 MB host dequantize plus
        /// PCIe transfer) every 256 decode steps forever.
        forced_f32: std::collections::HashSet<(usize, usize)>,
    }

    /// Load and upload one expert to VRAM: raw per-row int4 (`fmt=2`, ~8× denser)
    /// when `int4` and the source is per-row int4, else dequantized f32. Shared by
    /// [`GpuTier::build`] and [`GpuTier::reheat`] so both formats stay in one place.
    /// Upload one expert's three projections to VRAM. Returns the expert and the
    /// format it actually landed in (`true` = int4-resident).
    ///
    /// `int4` is a *preference*, not a requirement. int4 residency needs per-row
    /// int4 sources; a grouped-int4 or int8 expert falls back to the dequantized
    /// f32 path for that expert alone. Previously this returned `Err`, which
    /// truncated the whole tier at the first non-int4 expert (the caller treats an
    /// upload error as "stop, keep what landed") — so one odd expert could cost
    /// every expert behind it. Residency is therefore mixed-format, which the byte
    /// accounting already models via the per-expert `precision` map.
    fn upload_expert(
        st: &SafeTensors,
        cfg: &Cfg,
        layer: usize,
        e: usize,
        device: i32,
        int4: bool,
    ) -> Result<(GpuExpert, bool), Error> {
        let hidden = cfg.hidden as usize;
        let inter = cfg.moe_inter as usize;
        let pe = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
        let gate = QtWeight::load(st, &pe("gate_proj.weight"), inter, hidden)?;
        let up = QtWeight::load(st, &pe("up_proj.weight"), inter, hidden)?;
        let down = QtWeight::load(st, &pe("down_proj.weight"), hidden, inter)?;
        use crate::weight::QuantFmt;
        let all_int4 = gate.fmt == QuantFmt::Int4 && up.fmt == QuantFmt::Int4 && down.fmt == QuantFmt::Int4;
        if int4 && all_int4 {
            Ok((GpuExpert::upload_int4(device, gate.raw(), up.raw(), down.raw(), hidden, inter)?, true))
        } else {
            Ok((
                GpuExpert::upload(device, &gate.dequant(), &up.dequant(), &down.dequant(), hidden, inter)?,
                false,
            ))
        }
    }

    /// Whether the first sparse layer's expert 0 stores all three projections as
    /// per-row int4 — the only source shape `upload_int4` can take raw.
    ///
    /// One expert is a sound probe because a container is written by one
    /// converter run at one target; a *tiered* container mixes formats, and this
    /// deliberately answers "no" for it rather than planning as if the whole tier
    /// were int4. Under-planning wastes VRAM; over-planning silently truncates
    /// the tier, so the conservative answer is the right one.
    fn experts_are_per_row_int4(st: &SafeTensors, cfg: &Cfg) -> bool {
        use peregrine_core::QtInfo;
        let (hidden, inter) = (cfg.hidden as i64, cfg.moe_inter as i64);
        let l = cfg.first_dense as usize;
        let pe = |t: &str| format!("model.layers.{l}.mlp.experts.0.{t}");
        [(pe("gate_proj.weight"), inter, hidden), (pe("up_proj.weight"), inter, hidden), (pe("down_proj.weight"), hidden, inter)]
            .iter()
            .all(|(name, o, i)| QtInfo::detect(st, name, *o, *i).fmt == peregrine_core::QtFmt::Int4)
    }

    impl GpuTier {
        /// Build the tier by uploading as many routed experts as fit within
        /// `free VRAM - headroom`, choosing int4-resident (8× denser) vs f32
        /// residency from the `COLI_GPU_INT4` env var. `counts` is last session's
        /// routing heat (empty for a cold start). `Ok(None)` when CUDA is
        /// unavailable or nothing fits. See [`Self::build_with`].
        pub fn build(
            st: &SafeTensors,
            cfg: &Cfg,
            headroom_bytes: usize,
            counts: &[u32],
        ) -> Result<Option<GpuTier>, Error> {
            Self::build_with(st, cfg, headroom_bytes, std::env::var("COLI_GPU_INT4").is_ok(), counts)
        }

        /// Build by uploading as many routed experts as fit `free VRAM − headroom`,
        /// placed by the heat/bytes knapsack over `counts`. `int4` uploads per-row
        /// int4 weights directly (~8× denser), falling back to dequantized f32 per
        /// expert for sources that aren't per-row int4. `Ok(None)` when CUDA is
        /// unavailable or nothing fits. Takes `int4` and `counts` explicitly so it
        /// is testable without racing process env.
        pub fn build_with(
            st: &SafeTensors,
            cfg: &Cfg,
            headroom_bytes: usize,
            int4: bool,
            counts: &[u32],
        ) -> Result<Option<GpuTier>, Error> {
            if peregrine_cuda::init(&[0]) < 1 {
                return Ok(None);
            }
            let device = 0;
            let (free, _total) = peregrine_cuda::mem_info(device)?;
            let hidden = cfg.hidden as usize;
            let inter = cfg.moe_inter as usize;
            // The two possible per-expert costs; which one an expert actually
            // pays is decided per upload, below.
            let int4_bytes = super::resident_bytes_per_expert(hidden, inter, true);
            let f32_bytes = super::resident_bytes_per_expert(hidden, inter, false);
            // `int4` is what the operator asked for; `raw_int4` is what this
            // container can actually deliver. Sizing the *plan* from the request
            // rather than the container is how an int3/int2-g64 checkpoint plans
            // N experts and uploads 8N worth — the per-expert tracking below then
            // truncates the tier, silently, to ~1/8 of what was asked for.
            let raw_int4 = int4 && experts_are_per_row_int4(st, cfg);
            if int4 && !raw_int4 {
                eprintln!(
                    "peregrine: COLI_GPU_INT4 set, but this container's experts are not per-row \
                     int4 — they upload dequantized to f32 (8x), so the VRAM plan is sized for f32"
                );
            }
            let bytes_per_expert = if raw_int4 { int4_bytes } else { f32_bytes };
            let budget = free.saturating_sub(headroom_bytes);

            // Heat-density knapsack over the persisted routing counts. On a cold
            // table (no `route_stats.json`, or a fingerprint mismatch) this falls
            // back internally to the same round-robin `plan_residency` placement,
            // so a first run is byte-for-byte what it always was.
            let placement = super::solve_residency_sized(
                counts,
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                budget,
                |_, _| bytes_per_expert,
            );
            let mut capacity = placement.len(); // expert-count view of the budget
            let mut experts = HashMap::new();
            let mut precision: HashMap<(usize, usize), bool> = HashMap::new();
            let mut forced_f32: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
            // Track bytes as uploaded, not as planned: an expert that falls back
            // from int4 to f32 costs 8× what the placement budgeted for it, so a
            // few fallbacks would otherwise overrun VRAM.
            let mut used = 0usize;
            for (layer, e) in placement {
                match upload_expert(st, cfg, layer, e, device, int4) {
                    Ok((ge, landed_int4)) => {
                        let bytes = if landed_int4 { int4_bytes } else { f32_bytes };
                        if used.saturating_add(bytes) > budget {
                            // This one doesn't fit at its real size. Drop it and
                            // stop — the placement is heat-ordered, so everything
                            // after it is colder and no more valuable.
                            capacity = experts.len();
                            break;
                        }
                        used += bytes;
                        experts.insert((layer, e), ge);
                        precision.insert((layer, e), landed_int4);
                        if int4 && !landed_int4 {
                            forced_f32.insert((layer, e)); // source can't be int4 — don't re-ask
                        }
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
                // Meaningful only when some resident is int4 (an all-f32 tier has
                // nothing left to promote).
                let any_int4 = precision.values().any(|&p| p);
                let adaptive_f32_frac = if any_int4 {
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
                // Registered before the value exists so the matching `Drop` can
                // never run against a count that was not incremented.
                LIVE_TIERS.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                Ok(Some(GpuTier {
                    device,
                    experts,
                    capacity,
                    budget_bytes: budget,
                    expert_bytes: (int4_bytes, f32_bytes),
                    int4,
                    adaptive_f32_frac,
                    precision,
                    forced_f32,
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
            let (int4_bytes, f32_bytes) = self.expert_bytes;
            // Per-expert precision for this residency generation.
            let (want, precision_of): ResidencyGeneration =
                match self.adaptive_f32_frac {
                    // Adaptive: size the resident set and its f32 share together,
                    // over every sparse candidate. Ranking only `self.capacity`
                    // first would cap the pool at a count derived from the int4
                    // footprint, and the f32 promotions (~8× each) would then eat
                    // that budget before the int4 tail was ever considered.
                    Some(frac) => {
                        let plan = super::plan_precision_fitted(
                            counts,
                            cfg.n_layers as usize,
                            cfg.first_dense as usize,
                            cfg.n_experts as usize,
                            budget,
                            frac,
                            (f32_bytes, int4_bytes),
                        );
                        let prec = plan
                            .iter()
                            .map(|&(k, p)| (k, matches!(p, super::ExpertPrecision::Int4)))
                            .collect();
                        (plan.into_iter().map(|(k, _)| k).collect(), prec)
                    }
                    // Uniform format: every resident is the same size, so the
                    // count-based rank already respects the byte budget.
                    None => {
                        let want = super::rank_by_heat(
                            counts,
                            cfg.n_layers as usize,
                            cfg.first_dense as usize,
                            cfg.n_experts as usize,
                            self.capacity,
                        );
                        let uniform = if self.int4 { int4_bytes } else { f32_bytes };
                        let fit = budget / uniform.max(1);
                        let prec = want.iter().map(|&k| (k, self.int4)).collect();
                        (want.into_iter().take(fit).collect(), prec)
                    }
                };
            let want_set: HashSet<(usize, usize)> = want.iter().copied().collect();
            // evict experts that cooled off — their `Drop` frees the VRAM slot
            self.experts.retain(|k, _| want_set.contains(k));
            self.precision.retain(|k, _| want_set.contains(k));
            // `forced_f32` is a property of the *source*, not of residency, but
            // pruning it with the rest keeps it bounded by the resident set instead
            // of growing across every expert the tier ever touched.
            self.forced_f32.retain(|k| want_set.contains(k));

            // Which of `want` would actually cross PCIe this generation, in heat
            // order, and what each would cost. Everything else is already resident
            // in the wanted format and is free.
            let fmt_of = |key: &(usize, usize)| precision_of.get(key).copied().unwrap_or(self.int4);
            let upload_costs: Vec<usize> = want
                .iter()
                .filter(|key| {
                    let want_int4 = fmt_of(key);
                    let cur_int4 = self.precision.get(key).copied().unwrap_or(self.int4);
                    super::needs_reupload(
                        self.experts.contains_key(key),
                        cur_int4,
                        want_int4,
                        self.forced_f32.contains(key),
                    )
                })
                .map(|key| if fmt_of(key) { int4_bytes } else { f32_bytes })
                .collect();
            // Unlimited by default, so this is bit-identical with the knob unset.
            let mut upload_quota = super::admit_uploads(&upload_costs, super::pcie_budget_bytes());

            for (layer, e) in want {
                let key = (layer, e);
                let want_int4 = precision_of.get(&key).copied().unwrap_or(self.int4);
                // An expert uploaded by `build` (absent from `precision`) is in the
                // tier's uniform format — treat missing as that, so a non-adaptive
                // reheat never re-uploads a format-correct resident.
                let cur_int4 = self.precision.get(&key).copied().unwrap_or(self.int4);
                let resident = self.experts.contains_key(&key);
                if !super::needs_reupload(resident, cur_int4, want_int4, self.forced_f32.contains(&key)) {
                    continue;
                }
                if upload_quota == 0 {
                    // Budget spent. Leave the rest for the next generation rather
                    // than bursting: they are the coldest in the plan, and an expert
                    // that stays non-resident simply streams from the CPU lane.
                    break;
                }
                upload_quota -= 1;
                // Re-upload on a format change (remove first so the old tensor's
                // Drop frees its VRAM before the new allocation).
                self.experts.remove(&key);
                match upload_expert(st, cfg, layer, e, self.device, want_int4) {
                    // Record the format that actually landed, not the one asked for.
                    Ok((ge, landed_int4)) => {
                        self.experts.insert(key, ge);
                        self.precision.insert(key, landed_int4);
                        if want_int4 && !landed_int4 {
                            self.forced_f32.insert(key);
                        }
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
            // Validate every job up front, so a ragged input fails before any
            // kernel is dispatched rather than after the first class has run.
            for (e, xg) in jobs {
                if !self.experts.contains_key(&(layer, *e)) {
                    return Err(Error::Format(format!("gpu expert ({layer},{e}) not resident")));
                }
                if hidden == 0 || !xg.len().is_multiple_of(hidden) {
                    return Err(Error::Format("gpu compute: ragged gathered rows".into()));
                }
            }
            // One call per residency format: a single f32 resident in the group
            // would otherwise drop every expert in the call off the int4 fast
            // paths. A homogeneous group still makes exactly one call.
            let fmt: Vec<bool> =
                jobs.iter().map(|(e, _)| self.precision.get(&(layer, *e)).copied().unwrap_or(self.int4)).collect();
            let classes = super::partition_by_format(&fmt);

            let mut per_class = Vec::with_capacity(classes.len());
            for idxs in &classes {
                let mut refs = Vec::with_capacity(idxs.len());
                let mut rows = Vec::with_capacity(idxs.len());
                let mut x = Vec::new();
                for &i in idxs {
                    let (e, xg) = &jobs[i];
                    let ge = self
                        .experts
                        .get(&(layer, *e))
                        .ok_or_else(|| Error::Format(format!("gpu expert ({layer},{e}) not resident")))?;
                    refs.push(ge);
                    rows.push((xg.len() / hidden) as i32);
                    x.extend_from_slice(xg);
                }
                let y = peregrine_cuda::expert_group(&refs, &rows, &x, hidden)?;
                let mut outs = Vec::with_capacity(idxs.len());
                let mut off = 0usize;
                for &r in &rows {
                    let n = r as usize * hidden;
                    let end = off.checked_add(n).filter(|&e| e <= y.len()).ok_or_else(|| {
                        Error::Format("gpu compute: expert_group returned a short buffer".into())
                    })?;
                    outs.push(y[off..end].to_vec());
                    off = end;
                }
                per_class.push(outs);
            }
            // Back to job order: `concurrent.rs` zips the returned Vec positionally
            // against its plans, and the reduce accumulates in that order — so the
            // permutation must not escape this function.
            super::scatter_by_index(jobs.len(), &classes, per_class)
                .ok_or_else(|| Error::Format("gpu compute: format partition is not a bijection".into()))
        }
    }

    /// One residency generation: the experts to hold, plus each one's format
    /// (`true` = int4). Produced together because the format decides the byte
    /// cost, which decides how many fit.
    type ResidencyGeneration = (Vec<(usize, usize)>, HashMap<(usize, usize), bool>);

    /// Live `GpuTier`s in this process. `peregrine_cuda::shutdown()` is global —
    /// it loops every device context — so a tier that called it from its own
    /// `Drop` would tear down the CUDA state its siblings are still using, and
    /// their `GpuExpert` handles would then free against destroyed contexts. Only
    /// the last tier out shuts the backend down.
    static LIVE_TIERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    impl Drop for GpuTier {
        fn drop(&mut self) {
            // GpuExpert handles free themselves; the contexts outlive them unless
            // this is the last tier in the process.
            self.experts.clear();
            if LIVE_TIERS.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                peregrine_cuda::shutdown();
            }
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
        pub fn build(
            _st: &SafeTensors,
            _cfg: &Cfg,
            _headroom_bytes: usize,
            _counts: &[u32],
        ) -> Result<Option<GpuTier>, Error> {
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

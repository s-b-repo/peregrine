//! GPU VRAM expert tier — the third concurrent lane (feature `cuda`).
//!
//! At load, as many routed experts as fit in VRAM are dequantized to f32 and
//! uploaded (fmt=0). During the MoE forward, [`crate::concurrent`] dispatches the
//! GPU-resident experts of a layer in one batched `expert_group` call — running
//! on its own thread, concurrently with the io_uring disk lane and the CPU pool.
//!
//! **Devices and vendors.** The tier spans every device
//! [`crate::devices::selected_devices`] returns (`COLI_GPU_DEVICES`, default one
//! CUDA device): residency is *partitioned* across them via an owner map, and a
//! forward dispatches each device's group on its own thread in parallel, folding
//! partials in fixed device order so output never depends on completion order.
//! The kernels themselves are vendor-neutral C ABI — NVIDIA today, AMD/ROCm via
//! hipcc of the same source (see `docs/gpu-vendors.md`).
//!
//! Numerics note: GPU experts compute in **f32** (more accurate than the CPU
//! int4 path), so enabling the tier changes low-order bits versus a pure-CPU run
//! — expected, and why the bit-exact determinism test stays CPU-only. The tier is
//! opt-in (built only when `COLI_GPU` is set) so default behavior is unchanged.
//!
//! The type exists in both builds: the real tier under `cuda`, and a never-built
//! empty stub otherwise, so the scheduler/`Model` stay feature-agnostic.

#[cfg(feature = "cuda")]
pub use real::{upload_lane_counts, GpuDenseTier, GpuTier};
#[cfg(not(feature = "cuda"))]
pub use stub::{GpuDenseTier, GpuTier};

/// `(pinned async, blocking)` expert uploads — always `(0, 0)` with no CUDA,
/// since nothing uploads.
#[cfg(not(feature = "cuda"))]
pub fn upload_lane_counts() -> (usize, usize) {
    (0, 0)
}

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
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, Ordering};

/// Routing-frequency accumulator over `(layer, expert)` — the "heat" that drives
/// dynamic VRAM residency. Bumped once per routed expert per layer during the
/// forward (lock-free), read by `GpuTier::reheat` to keep the hottest experts
/// resident. Present in every build (the `Model` holds one when a GPU tier
/// exists); the ranking is pure, so it is unit-testable without a GPU.
pub struct HeatTable {
    n_experts: usize,
    counts: Vec<AtomicU32>,
    /// Per-slot value of [`Self::clock`] at the slot's most recent routing — the
    /// `last` array [`peregrine_io::pick_lfru`] scores recency from.
    ///
    /// Lives **here** rather than in a second table because heat and recency are
    /// two readings of one event: a separate structure would have its own bump
    /// site, and the two sites would eventually disagree about what counts as a
    /// routing. The cost is one extra relaxed store beside a `fetch_add` that is
    /// already touching the line.
    last: Vec<AtomicU32>,
    /// Monotonic forward counter, advanced once per forward by [`Self::tick`].
    ///
    /// Per **forward**, deliberately, not per layer: `lfru_score` saturates
    /// recency at an age of 255, and a per-layer clock would run 78 ticks a
    /// token on GLM-5.2 — every slot would read as maximally stale within three
    /// tokens and the recency term would stop distinguishing anything. Per
    /// forward gives 255 tokens of resolution against a `reheat` that runs every
    /// 256 steps.
    clock: AtomicU32,
}

impl HeatTable {
    /// A zeroed table for `n_layers × n_experts` routed experts.
    pub fn new(n_layers: usize, n_experts: usize) -> HeatTable {
        HeatTable {
            n_experts,
            counts: (0..n_layers * n_experts).map(|_| AtomicU32::new(0)).collect(),
            last: (0..n_layers * n_experts).map(|_| AtomicU32::new(0)).collect(),
            clock: AtomicU32::new(0),
        }
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
        let idx = layer.checked_mul(self.n_experts).and_then(|b| b.checked_add(expert));
        if let Some(c) = idx.and_then(|i| self.counts.get(i)) {
            c.fetch_add(1, Ordering::Relaxed);
        }
        // Stamped from the same bounds check, so a slot can never have a recency
        // without a count — `pick_lfru` reads the two as one score.
        if let Some(l) = idx.and_then(|i| self.last.get(i)) {
            l.store(self.clock.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    /// Advance the recency clock. Called once per forward; see [`Self::clock`].
    pub fn tick(&self) {
        self.clock.fetch_add(1, Ordering::Relaxed);
    }

    /// The current recency clock, to be passed to `pick_lfru` alongside
    /// [`Self::last_snapshot`].
    pub fn clock(&self) -> u32 {
        self.clock.load(Ordering::Relaxed)
    }

    /// Snapshot of the per-slot last-routed stamps, row-major like
    /// [`Self::snapshot`] and always the same length.
    ///
    /// A fresh process starts every stamp at 0 with the clock at 0, so every
    /// slot scores as maximally *recent* and LFRU degrades to pure frequency
    /// until the clock has advanced. That is the honest cold-start behaviour:
    /// recency the process never observed should not order anything. (`last` is
    /// deliberately **not** persisted in `route_stats.json` — a stamp from a
    /// previous process is measured against a clock that no longer exists.)
    pub fn last_snapshot(&self) -> Vec<u32> {
        self.last.iter().map(|c| c.load(Ordering::Relaxed)).collect()
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

    /// Frequency, recency and the clock they are measured against, read
    /// together so all three describe one instant. Reading them through three
    /// separate calls is what would let a residency generation score a `last`
    /// against a `clock` from a different forward.
    pub fn snapshot_all(&self) -> (Vec<u32>, Vec<u32>, u32) {
        (self.snapshot(), self.last_snapshot(), self.clock())
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

/// One residency generation's view of the routing statistics.
///
/// Bundled rather than passed as three parameters because `pick_lfru` scores
/// frequency and recency **against each other**: a `last` array snapshotted at
/// one instant and a `clock` read at another produce ages that are quietly wrong
/// rather than obviously wrong, and nothing downstream can detect it. Build one
/// with [`HeatTable::snapshot_all`].
/// One residency generation's view of the MTP head's **pin** request.
///
/// Beside [`HeatView`] rather than folded into it because the two are different
/// kinds of claim on the same VRAM. A heat view is a *ranking*: everything in it
/// competes, and `reheat` decides how much of the budget each layer wins. A pin
/// request is a *reservation*: its bytes come off the top, and what is left is
/// what the ranking gets to spend. Merging them would mean expressing "hold this
/// regardless of rank" as a very large count, which is the same mistake as
/// ranking a draft-path layer against a main-stack one — see
/// [`crate::mtp::plan_pins`].
pub struct PinRequest<'a> {
    /// Draft-routing frequency for `layer`'s experts, indexed by expert id.
    pub counts: &'a [u32],
    /// The layer those counts describe — `cfg.n_layers`, the MTP head's.
    pub layer: usize,
    /// VRAM the pinned set may occupy, before the half-the-tier clamp.
    pub budget: usize,
}

impl PinRequest<'_> {
    /// No pins: what every caller without an MTP head, or with the knob unset,
    /// passes. Byte-for-byte the pre-pin behaviour.
    pub fn none() -> PinRequest<'static> {
        PinRequest { counts: &[], layer: 0, budget: 0 }
    }
}

pub struct HeatView<'a> {
    /// Routing frequency, row-major `[layer * n_experts + expert]`.
    pub counts: &'a [u32],
    /// Last-routed stamps, same layout and length as `counts`.
    pub last: &'a [u32],
    /// The forward counter `last` is measured against.
    pub clock: u32,
}

impl<'a> HeatView<'a> {
    /// A frequency-only view, for callers that have counts and no clock.
    ///
    /// [`SwapPolicy::Freq`] works normally against it. [`SwapPolicy::Lfru`]
    /// proposes **nothing** — `pick_lfru` reads an empty `last` as a length
    /// mismatch and answers "no decision", which is the right answer: LFRU
    /// without recency is not LFRU, and quietly running frequency-only under an
    /// `lfru` label would make the two policies indistinguishable in exactly the
    /// measurement meant to tell them apart.
    pub fn frequency_only(counts: &'a [u32]) -> HeatView<'a> {
        HeatView { counts, last: &[], clock: 0 }
    }

    /// This layer's `n_experts`-wide frequency row, or empty if out of range.
    fn heat_row(&self, layer: usize, n_experts: usize) -> &[u32] {
        row_of(self.counts, layer, n_experts)
    }

    /// This layer's recency row. Empty when no recency was supplied, which
    /// `pick_lfru` rejects as a length mismatch — the documented "no decision"
    /// answer rather than a decision made on absent data.
    fn last_row(&self, layer: usize, n_experts: usize) -> &[u32] {
        row_of(self.last, layer, n_experts)
    }
}

/// One layer's row out of a row-major `[layer * n_experts + expert]` table.
fn row_of(table: &[u32], layer: usize, n_experts: usize) -> &[u32] {
    let Some(start) = layer.checked_mul(n_experts) else {
        return &[];
    };
    match start.checked_add(n_experts) {
        Some(end) if end <= table.len() => &table[start..end],
        _ => &[],
    }
}

/// How [`GpuTier::reheat`] chooses the next residency generation.
///
/// The default re-plans the whole set every generation, which is correct but
/// unbounded in churn: any expert whose heat rank moved is a candidate upload.
/// The two incremental policies are `peregrine-io`'s hot-store rules applied to
/// the shape they were written for — **per layer, the resident set is exactly a
/// fixed-size slot array** — and both carry a 25 %-plus-4-count hysteresis, so a
/// marginally hotter expert does not displace a resident at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapPolicy {
    /// Re-rank every candidate and take the hottest that fit (the historical
    /// behaviour, and what an unset `COLI_GPU_TIER_SWAP` selects).
    Replan,
    /// At most one `pick_lfru` swap per layer per generation: frequency first,
    /// recency only as a tiebreak.
    Lfru,
    /// At most one `pick_swap` swap per layer per generation: frequency only.
    Freq,
}

impl SwapPolicy {
    /// Parse `COLI_GPU_TIER_SWAP`. Unrecognized values are **not** silently
    /// treated as off: the caller reports them, because a misspelled policy that
    /// reads as "default" is a knob that stopped being a knob.
    pub fn parse(s: &str) -> Option<SwapPolicy> {
        match s {
            "" | "replan" => Some(SwapPolicy::Replan),
            "lfru" => Some(SwapPolicy::Lfru),
            "freq" => Some(SwapPolicy::Freq),
            _ => None,
        }
    }
}

/// The configured [`SwapPolicy`], read once per `reheat` from
/// `COLI_GPU_TIER_SWAP`. An unparseable value warns and falls back to `Replan`.
pub fn swap_policy() -> SwapPolicy {
    match std::env::var("COLI_GPU_TIER_SWAP") {
        Ok(v) => SwapPolicy::parse(&v).unwrap_or_else(|| {
            peregrine_io::note_advisory_err(
                "COLI_GPU_TIER_SWAP is not a known policy (replan|lfru|freq); using replan",
                &v,
            );
            SwapPolicy::Replan
        }),
        // Both variants named rather than `Err(_)`: unset and non-UTF-8 are the
        // two ways nobody chose a policy, and spelling them out is what stops a
        // future third variant from being swallowed by a wildcard.
        Err(std::env::VarError::NotPresent) | Err(std::env::VarError::NotUnicode(_)) => {
            SwapPolicy::Replan
        }
    }
}

/// One incremental residency generation, as `(evict, admit)` pairs.
///
/// Pure — it decides, it does not upload — so the policy is unit-testable on a
/// host with no GPU, which is the only way the hysteresis behaviour gets tested
/// at all: `mod real`'s tests skip themselves without a device.
///
/// **Per layer, and at most one swap per layer**, because that is the shape
/// `pick_lfru`/`pick_swap` were written for: `pinned` is a slot array and
/// `Swap::slot` indexes it. A layer holding no residents is skipped rather than
/// grown — admitting into an empty set is a *sizing* decision, which is
/// [`SwapPolicy::Replan`]'s job, not a swap.
pub fn plan_swaps(
    heat: &HeatView,
    n_layers: usize,
    first_dense: usize,
    n_experts: usize,
    policy: SwapPolicy,
    resident: impl Fn(usize, usize) -> bool,
) -> Vec<((usize, usize), (usize, usize))> {
    if matches!(policy, SwapPolicy::Replan) || n_experts == 0 || first_dense >= n_layers {
        return Vec::new();
    }
    let mut out = Vec::new();
    for layer in first_dense..n_layers {
        let pinned: Vec<usize> = (0..n_experts).filter(|&e| resident(layer, e)).collect();
        if pinned.is_empty() {
            continue;
        }
        let hr = heat.heat_row(layer, n_experts);
        if hr.is_empty() {
            continue;
        }
        let swap = match policy {
            SwapPolicy::Lfru => {
                peregrine_io::pick_lfru(hr, heat.last_row(layer, n_experts), heat.clock, &pinned)
            }
            SwapPolicy::Freq => peregrine_io::pick_swap(hr, &pinned),
            SwapPolicy::Replan => None,
        };
        // `slot` indexes `pinned`; `eid` is an expert id. Conflating them would
        // evict the wrong expert whenever residency is not a prefix of 0..n.
        if let Some(s) = swap {
            if let Some(&victim) = pinned.get(s.slot) {
                out.push(((layer, victim), (layer, s.eid)));
            }
        }
    }
    out
}

/// Persistent Expert Residency Solver. Greedy knapsack over `(layer, expert)`
/// pairs, maximizing total heat within `budget` bytes. Handles heterogeneous
/// expert sizes (per-layer inter-size variation) by sorting on `heat /
/// bytes_per_expert` — a small expert with modest heat can beat a larger cold
/// one. Falls back to [`plan_residency`] round-robin when the heat table is
/// all-zero. Deterministic ties (equal ratio → lower layer, lower expert).
/// **No production caller, and the reason is worth stating rather than fixing.**
/// A uniform per-expert cost makes this a top-N-by-heat-density, which for equal
/// sizes is just top-N-by-heat — and `rank_by_heat` already does that while
/// preserving `plan_residency`'s round-robin tie-break, which this does not
/// (it breaks ties by ascending layer/expert). On a partially-warm heat table
/// the two disagree on every tie, and the disagreement is pure residency churn.
/// So the callers that look like they want this either want `rank_by_heat`
/// (uniform sizes) or `solve_residency_sized` with a real per-expert closure
/// (mixed sizes — a heat-tiered container). Kept as the documented uniform-cost
/// entry point; wiring it would mean picking the wrong one of those two.
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

/// Split a heat-ordered placement across device budgets.
///
/// Takes `placement` in heat order (the order [`solve_residency_sized`] /
/// [`rank_by_heat`] emit) and assigns each expert to the device index with the
/// most budget left, ties breaking to the lowest index — best-fit-decreasing's
/// opposite on purpose: the hottest experts are also the ones reheat will fight
/// to keep, so they get the headroom, and balancing rather than packing keeps
/// every card's resident set useful when budgets are lopsided. Returns `(key,
/// device_index)` pairs for the prefix that fits; the first expert that fits no
/// device stops the walk, because everything after it is colder and no more
/// valuable (the same truncation rule the single-device upload loop always had).
///
/// Pure and deterministic: the same placement and budgets always produce the
/// same owner map, which is what makes a 1-device vs 2-device A/B a *placement*
/// difference only — the outputs must be identical either way.
///
/// Returns device **indices** into `budgets`, not vendor ordinals; the caller
/// maps them once at upload time. Empty input → empty output; an all-zero
/// budget assigns nothing.
pub fn partition_placement(
    placement: &[(usize, usize)],
    budgets: &[usize],
    bytes_of: impl Fn(&(usize, usize)) -> usize,
) -> Vec<((usize, usize), usize)> {
    let mut remaining = budgets.to_vec();
    let mut out = Vec::with_capacity(placement.len());
    for key in placement {
        let bytes = bytes_of(key);
        let Some(dev) = pick_device(&remaining, bytes) else { break };
        remaining[dev] -= bytes;
        out.push((*key, dev));
    }
    out
}

/// The device index with the most room for `bytes`, ties to the lowest index —
/// the one-expert form of [`partition_placement`]'s rule, so build and reheat
/// target devices identically. `None` when no device fits it.
pub fn pick_device(remaining: &[usize], bytes: usize) -> Option<usize> {
    remaining
        .iter()
        .enumerate()
        .max_by_key(|&(i, &r)| (r, std::cmp::Reverse(i)))
        .filter(|&(_, &r)| r >= bytes)
        .map(|(i, _)| i)
}

/// One line describing an owner map: per-device resident counts in ascending
/// device order (`device 0: 96, device 1: 94`). The startup banner wants "how
/// did the split actually land" as one readable string.
#[cfg(feature = "cuda")]
fn owner_summary(owner: &HashMap<(usize, usize), i32>) -> String {
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for d in owner.values() {
        *counts.entry(*d).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return "nothing resident".to_string();
    }
    counts.iter().map(|(d, n)| format!("device {d}: {n}")).collect::<Vec<_>>().join(", ")
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
    use std::collections::HashSet;

    use crate::gpu_test_lock::gpu_guard;

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
        let after =
            tier.reheat(&st, &cfg, &super::HeatView::frequency_only(&counts), &super::PinRequest::none())?;
        std::fs::remove_dir_all(&dir)?;
        assert!(before > 0, "tiny experts must fit VRAM");
        assert_eq!(after, before, "reheat at full capacity keeps the resident count");
        assert!(tier.has(2, 0), "the hottest expert must remain resident");
        Ok(())
    }

    #[test]
    fn a_swap_policy_with_adaptive_precision_does_not_recurse() -> Result<(), Error> {
        // Regression guard for unbounded mutual recursion between `reheat` and
        // `reheat_incremental`, which was reachable whenever a swap policy and
        // adaptive f32 residency were configured **together**: `reheat`
        // dispatched to `reheat_incremental`, which refuses adaptive precision
        // and "fell back to replan" by calling `reheat` — which re-entered the
        // same dispatch. Each knob was exercised alone and the pair apparently
        // never was.
        //
        // **This test cannot fail politely on the old code.** A stack overflow
        // is SIGSEGV, so the pre-fix behaviour kills the whole test binary
        // rather than reporting one red test. That is the honest shape for it:
        // the assertion is "this returns at all".
        //
        // Both knobs are set here rather than plumbed as parameters because the
        // recursion lived in the dispatch that *reads* them, so a parameterized
        // twin would not exercise the path that was broken. `gpu_guard` is what
        // keeps the process-wide env from reaching a concurrent GPU test.
        let _g = gpu_guard();
        let dir = std::env::temp_dir().join(format!("peregrine_swap_adaptive_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;

        // `COLI_GPU_F32_FRAC` is read by `build_with`, `COLI_GPU_TIER_SWAP` by
        // `reheat` — so both have to be live across the pair, and both are
        // cleared before any assertion can leave the function early.
        std::env::set_var("COLI_GPU_F32_FRAC", "0.5");
        std::env::set_var("COLI_GPU_TIER_SWAP", "lfru");
        let built = GpuTier::build_with(&st, &cfg, 0, true, &[]);
        let out = built.and_then(|tier| match tier {
            None => Ok(None), // no CUDA device on this host → skip
            Some(mut tier) => {
                let n_experts = cfg.n_experts as usize;
                let mut counts = vec![0u32; (cfg.n_layers as usize + 1) * n_experts];
                counts[2 * n_experts] = 99;
                let view = super::HeatView::frequency_only(&counts);
                // The call that used to never return.
                tier.reheat(&st, &cfg, &view, &super::PinRequest::none()).map(Some)
            }
        });
        std::env::remove_var("COLI_GPU_F32_FRAC");
        std::env::remove_var("COLI_GPU_TIER_SWAP");
        std::fs::remove_dir_all(&dir)?;

        match out? {
            None => Ok(()), // skipped: no device
            Some(resident) => {
                // The adaptive path re-plans, so it keeps a resident set rather
                // than emptying the tier — a `0` here would mean it returned by
                // collapsing residency instead of by doing the work.
                assert!(resident > 0, "the replan fallback must keep experts resident");
                Ok(())
            }
        }
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

    #[test]
    fn owner_grouped_compute_matches_the_direct_dispatch() -> Result<(), Error> {
        // The owner-grouped dispatch must be invisible: `compute` scatters
        // per-expert outputs back into job order, so its result has to equal the
        // direct `expert_group` call per expert, bit for bit, on any device
        // count. Run on one device here; the two-device twin below covers the
        // spread.
        let _g = gpu_guard();
        let dir = std::env::temp_dir().join(format!("peregrine_ownergroup_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let tier = GpuTier::build(&st, &cfg, 0, &[])?;
        let Some(tier) = tier else {
            std::fs::remove_dir_all(&dir)?;
            return Ok(()); // no CUDA device on this host → skip
        };
        let hidden = cfg.hidden as usize;
        let layer = cfg.first_dense as usize; // first sparse layer: residents exist here
        let mut r = Lcg(0x07E5);
        let jobs: Vec<(usize, Vec<f32>)> = (0..cfg.n_experts as usize)
            .filter(|&e| tier.has(layer, e))
            .map(|e| {
                // Two gathered rows per expert, so the row-splitting inside the
                // group dispatch is exercised, not just the single-row case.
                (e, (0..2 * hidden).map(|_| r.f()).collect())
            })
            .collect();
        let got = tier.compute(layer, &jobs, hidden);
        std::fs::remove_dir_all(&dir)?;
        let got = got?;
        assert_eq!(got.len(), jobs.len());
        for ((e, xg), out) in jobs.iter().zip(&got) {
            // Direct dispatch of this one expert — the pre-partitioning path.
            let ge =
                tier.expert_for_test(layer, *e).ok_or_else(|| Error::Format("resident vanished".into()))?;
            let rows = (xg.len() / hidden) as i32;
            let want = peregrine_cuda::expert_group(&[ge], &[rows], xg, hidden)?;
            assert_eq!(
                out.len(),
                want.len(),
                "expert {e}: grouped output must keep the direct path's shape"
            );
            assert_eq!(&want, out, "expert {e}: grouping must not change a single bit");
        }
        Ok(())
    }

    #[test]
    fn two_devices_partition_residency_and_agree_bit_for_bit() -> Result<(), Error> {
        // The scale-out-design §1 acceptance test, hardware-gated: across TWO
        // devices the same heat table and budget must produce (a) a placement
        // that actually spans both cards — different from the one-device shape —
        // and (b) outputs identical to that device's own dispatch. Skips on any
        // box with fewer than two usable devices rather than pretending.
        //
        // `gpu_guard` serializes against every other GPU test AND keeps the env
        // write below process-local for their duration.
        let _g = gpu_guard();
        if peregrine_cuda::probe_device_count() < 2 {
            return Ok(());
        }
        let dir = std::env::temp_dir().join(format!("peregrine_twodev_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        build_tiny_model(&dir)?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let prev = std::env::var("COLI_GPU_DEVICES").ok();
        std::env::set_var("COLI_GPU_DEVICES", "cuda:0,cuda:1");
        let built = GpuTier::build(&st, &cfg, 0, &[]);
        match prev {
            Some(v) => std::env::set_var("COLI_GPU_DEVICES", v),
            None => std::env::remove_var("COLI_GPU_DEVICES"),
        }
        let tier = built?;
        std::fs::remove_dir_all(&dir)?;
        let Some(tier) = tier else {
            return Ok(()); // second device listed but not initializable → skip
        };
        let tier = &tier;
        assert!(tier.devices().len() >= 2, "both requested devices must be initialized");
        // (a) the partition is real: residents on both sides of the owner map.
        let owners: HashSet<i32> = (cfg.first_dense as usize..cfg.n_layers as usize)
            .flat_map(|l| (0..cfg.n_experts as usize).filter_map(move |e| tier.owner_of(l, e)))
            .collect();
        assert!(owners.len() >= 2, "residency must span both devices, got {owners:?}");
        // (b) compute through the multi-device gather equals per-expert truth.
        let hidden = cfg.hidden as usize;
        let layer = cfg.first_dense as usize;
        let mut r = Lcg(0x7EC0);
        let jobs: Vec<(usize, Vec<f32>)> = (0..cfg.n_experts as usize)
            .filter(|&e| tier.has(layer, e))
            .map(|e| (e, (0..hidden).map(|_| r.f()).collect()))
            .collect();
        let outs = tier.compute(layer, &jobs, hidden)?;
        for ((e, xg), out) in jobs.iter().zip(&outs) {
            let ge =
                tier.expert_for_test(layer, *e).ok_or_else(|| Error::Format("resident vanished".into()))?;
            let rows = (xg.len() / hidden) as i32;
            let want = peregrine_cuda::expert_group(&[ge], &[rows], xg, hidden)?;
            assert_eq!(&want, out, "two-device gather must be bit-identical to direct dispatch");
        }
        Ok(())
    }
    /// Tiny deterministic RNG for test inputs (same shape as the fixtures in
    /// `peregrine-cuda`'s own GPU tests).
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }
}

#[cfg(test)]
mod placement_tests {
    use super::{partition_placement, pick_device, plan_residency};
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

    /// Tests for the incremental residency policies (`COLI_GPU_TIER_SWAP`).
    ///
    /// Pure, and here rather than in `mod real`, because those tests skip
    /// themselves on a host without a CUDA device — a hysteresis rule tested
    /// only there is a rule with no test on most machines.
    mod swap_policy {
        use super::super::{plan_swaps, HeatView, SwapPolicy};

        const N_LAYERS: usize = 4;
        const FIRST_DENSE: usize = 2;
        const N_EXPERTS: usize = 8;

        /// Row-major heat table with `(layer, expert) -> count` overrides.
        fn table(entries: &[(usize, usize, u32)]) -> Vec<u32> {
            let mut t = vec![0u32; N_LAYERS * N_EXPERTS];
            for &(l, e, c) in entries {
                t[l * N_EXPERTS + e] = c;
            }
            t
        }

        fn plan(
            counts: &[u32],
            last: &[u32],
            clock: u32,
            policy: SwapPolicy,
            resident: &[(usize, usize)],
        ) -> Vec<((usize, usize), (usize, usize))> {
            let view = HeatView { counts, last, clock };
            plan_swaps(&view, N_LAYERS, FIRST_DENSE, N_EXPERTS, policy, |l, e| {
                resident.contains(&(l, e))
            })
        }

        #[test]
        fn replan_proposes_nothing_so_the_default_path_is_untouched() {
            // The knob's off-state must be inert, not merely similar: any swap
            // proposed here would be an upload the historical path never made.
            let counts = table(&[(2, 7, 500), (2, 0, 1)]);
            assert!(plan(&counts, &[], 0, SwapPolicy::Replan, &[(2, 0)]).is_empty());
        }

        #[test]
        fn a_decisively_hotter_expert_swaps_in_at_most_once_per_layer() {
            // Layer 2 holds two cold residents; two non-residents are far hotter.
            // Both qualify, but a generation moves ONE — the bound that makes this
            // policy a brake on PCIe churn rather than a re-plan by another name.
            let counts = table(&[(2, 0, 1), (2, 1, 2), (2, 5, 900), (2, 6, 800)]);
            let swaps = plan(&counts, &[], 0, SwapPolicy::Freq, &[(2, 0), (2, 1)]);
            assert_eq!(swaps.len(), 1, "one swap per layer per generation: {swaps:?}");
            let ((vl, ve), (al, ae)) = swaps[0];
            assert_eq!((vl, al), (2, 2), "a swap never crosses layers");
            assert_eq!(ve, 0, "the coldest resident is the victim");
            assert_eq!(ae, 5, "the hottest non-resident is admitted");
        }

        #[test]
        fn a_marginally_hotter_expert_does_not_evict_a_resident() {
            // 100 -> 104 is +4 counts, exactly the hysteresis floor and under the
            // 25% term. Without the hysteresis this churns every generation, which
            // is the failure mode `pick_swap` was ported to avoid.
            let counts = table(&[(2, 0, 100), (2, 5, 104)]);
            assert!(
                plan(&counts, &[], 0, SwapPolicy::Freq, &[(2, 0)]).is_empty(),
                "a swap inside the hysteresis band is churn with no information behind it"
            );
            // Decisively hotter clears it, so the test above is not passing by
            // the policy being inert.
            let counts = table(&[(2, 0, 100), (2, 5, 400)]);
            assert_eq!(plan(&counts, &[], 0, SwapPolicy::Freq, &[(2, 0)]).len(), 1);
        }

        #[test]
        fn the_victim_is_a_pinned_slot_not_an_expert_id() {
            // Residency {3, 6} is not a prefix of 0..n, so `Swap::slot` (an index
            // into `pinned`) and the expert id differ: slot 0 is expert 3. Reading
            // `slot` as an id would evict expert 0, which is not even resident.
            let counts = table(&[(2, 3, 1), (2, 6, 50), (2, 4, 900)]);
            let swaps = plan(&counts, &[], 0, SwapPolicy::Freq, &[(2, 3), (2, 6)]);
            assert_eq!(swaps, vec![((2, 3), (2, 4))], "coldest pinned slot is expert 3");
        }

        #[test]
        fn dense_layers_and_empty_layers_are_left_alone() {
            // Layer 1 is dense (below `first_dense`) and layer 3 holds nothing.
            // Admitting into an empty layer is a sizing decision, not a swap.
            let counts = table(&[(1, 0, 900), (3, 0, 900)]);
            assert!(plan(&counts, &[], 0, SwapPolicy::Freq, &[(1, 1)]).is_empty());
            assert!(plan(&counts, &[], 0, SwapPolicy::Freq, &[]).is_empty());
        }

        #[test]
        fn lfru_decides_where_frequency_lands_exactly_on_the_hysteresis_edge() {
            // 100 resident vs 129 candidate is `pick_swap`'s boundary to the
            // count: `129 <= 100 + 25 + 4` holds, so frequency-only abstains.
            // LFRU adds a recency term worth at most 255 against one count's 256
            // — enough to carry a candidate that is *already* at the edge, and
            // never enough to carry one that is not. That narrowness is the
            // design (`tier.rs`: "a merely-recent expert cannot displace a
            // genuinely hotter one"), so this test pins the one band where the
            // two policies can differ at all. If it ever widened, LFRU would be
            // promoting on recency alone and both policies would still "work".
            let counts = table(&[(2, 0, 100), (2, 5, 129)]);
            assert!(
                plan(&counts, &[], 0, SwapPolicy::Freq, &[(2, 0)]).is_empty(),
                "frequency-only must abstain exactly on its hysteresis edge"
            );
            let mut last = vec![0u32; N_LAYERS * N_EXPERTS];
            last[2 * N_EXPERTS] = 0; // resident: routed long enough ago to score 0
            last[2 * N_EXPERTS + 5] = 300; // candidate: routed this generation
            let swaps = plan(&counts, &last, 300, SwapPolicy::Lfru, &[(2, 0)]);
            assert_eq!(swaps, vec![((2, 0), (2, 5))], "recency must carry the edge case under lfru");

            // Same counts, candidate equally stale: the recency term cancels and
            // LFRU must land back on frequency's answer. Otherwise the test above
            // would pass for a policy that swaps on nothing.
            let stale = [0u32; N_LAYERS * N_EXPERTS];
            let swaps = plan(&counts, &stale, 300, SwapPolicy::Lfru, &[(2, 0)]);
            assert!(swaps.is_empty(), "with no recency advantage lfru must agree with freq: {swaps:?}");
        }

        #[test]
        fn lfru_without_recency_makes_no_decision() {
            // A `HeatView` carrying no `last` must not quietly degrade to
            // frequency-only: the two policies would then be the same policy
            // under two names, and the A/B measuring them would compare nothing.
            let counts = table(&[(2, 0, 1), (2, 5, 900)]);
            let view = HeatView::frequency_only(&counts);
            let swaps =
                plan_swaps(&view, N_LAYERS, FIRST_DENSE, N_EXPERTS, SwapPolicy::Lfru, |l, e| {
                    (l, e) == (2, 0)
                });
            assert!(swaps.is_empty(), "lfru with no recency data must abstain: {swaps:?}");
        }

        #[test]
        fn an_unknown_policy_name_is_not_silently_off() {
            assert_eq!(SwapPolicy::parse(""), Some(SwapPolicy::Replan));
            assert_eq!(SwapPolicy::parse("replan"), Some(SwapPolicy::Replan));
            assert_eq!(SwapPolicy::parse("lfru"), Some(SwapPolicy::Lfru));
            assert_eq!(SwapPolicy::parse("freq"), Some(SwapPolicy::Freq));
            assert_eq!(SwapPolicy::parse("LFRU"), None, "case is not normalized away");
            assert_eq!(SwapPolicy::parse("1"), None, "a boolean-looking value is not a policy");
        }
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

    #[test]
    fn partition_balances_by_most_room_and_breaks_ties_low() {
        // Two equal budgets: the walk alternates, because after device 0 takes
        // the first expert it has less room and device 1 wins the next pick.
        let placement = [(0, 0), (0, 1), (1, 0), (1, 1)];
        let assigned = partition_placement(&placement, &[300, 300], |_| 100);
        assert_eq!(
            assigned,
            vec![((0, 0), 0), ((0, 1), 1), ((1, 0), 0), ((1, 1), 1)],
            "equal budgets must alternate hottest-first"
        );
        // A bigger first budget keeps winning until its lead is spent.
        let assigned = partition_placement(&placement, &[400, 200], |_| 100);
        assert_eq!(assigned.iter().filter(|(_, d)| *d == 0).count(), 3, "the larger budget absorbs the skew");
    }

    #[test]
    fn partition_truncates_at_the_first_non_fit_and_is_deterministic() {
        // One big card, one small: both first experts land on the big one
        // (most room), then the third needs more than either has left, so the
        // walk stops there — later (colder) candidates must not be considered,
        // which is exactly the single-device truncation rule generalized.
        let placement = [(0, 0), (0, 1), (0, 2), (0, 3)];
        let size = |&(_, e): &(usize, usize)| if e == 2 { 900 } else { 100 };
        let assigned = partition_placement(&placement, &[200, 1000], size);
        assert_eq!(
            assigned,
            vec![((0, 0), 1), ((0, 1), 1)],
            "expert (0,2) fits nowhere at its real size, so the prefix stops at two"
        );
        // Same inputs, same owner map — the property a 1-vs-2-device A/B needs.
        assert_eq!(assigned, partition_placement(&placement, &[200, 1000], size));
        // Degenerate shapes.
        assert!(partition_placement(&[], &[100], |_| 1).is_empty());
        assert!(partition_placement(&placement, &[0, 0], |_| 1).is_empty());
    }

    #[test]
    fn pick_device_takes_most_room_then_lowest_index() {
        assert_eq!(pick_device(&[10, 20, 30], 5), Some(2));
        assert_eq!(pick_device(&[10, 30, 30], 5), Some(1), "ties go to the lowest index");
        assert_eq!(pick_device(&[10, 20, 30], 31), None, "nothing fits → None, never a wrong card");
        assert_eq!(pick_device(&[], 0), None);
        assert_eq!(pick_device(&[7], 7), Some(0), "an exact fit counts as a fit");
    }
}

#[cfg(feature = "cuda")]
mod real {
    use std::collections::{BTreeMap, HashMap, HashSet};

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
    /// VRAM-resident **dense** MLPs, keyed by layer — the GPU tier for models
    /// with no routed experts (Track D). Where [`GpuTier`] holds a *selection*
    /// of a MoE's experts and re-selects as heat moves, this holds whole layers
    /// and never moves them: a dense layer is either resident for the run or it
    /// is not.
    ///
    /// **Layer-bounded and VRAM-probed rather than all-or-nothing.** Residency
    /// stops at whatever the card actually has free when the model loads —
    /// 6 layers next to another process's 10 GB, all 64 on an idle card, some
    /// number in between on a smaller GPU — and the layers that did not fit
    /// compute on the CPU exactly as before. That is what makes partial offload
    /// a working configuration instead of a failure mode.
    ///
    /// **Deterministic by construction.** Layer `L` computes on the same device
    /// for the whole run, so output does not depend on timing. `docs/todo.md`'s
    /// closed "CPU/GPU split GEMM" negative was about splitting *one GEMM's
    /// rows* across devices, which made low-order bits a function of the
    /// scheduler; layer-granular placement is not that and is not covered by
    /// that closure.
    pub struct GpuDenseTier {
        device: i32,
        mlps: HashMap<usize, GpuExpert>,
        bytes: usize,
        /// Layers examined but not uploaded because the budget ran out — the
        /// number that says whether more VRAM would buy anything.
        skipped: usize,
    }

    impl GpuDenseTier {
        /// An empty tier on `device`. Uploading is [`Self::try_add`], one layer
        /// at a time, so the caller keeps the load order and the failure policy.
        pub fn new(device: i32) -> GpuDenseTier {
            GpuDenseTier { device, mlps: HashMap::new(), bytes: 0, skipped: 0 }
        }

        /// Upload layer `li`'s MLP if it fits the device's *current* free VRAM
        /// with `headroom` bytes to spare. Returns whether it landed.
        ///
        /// Non-per-row-int4 weights are refused rather than dequantized: the
        /// f32 path costs 8× the VRAM, which would turn "all 64 layers" into
        /// "eight layers" silently. The container guarantees int4 here (every
        /// MLP tensor verified per-row int4 on the Qwen container), so a
        /// refusal means something upstream changed and the operator should
        /// hear about it rather than watch residency quietly collapse.
        pub fn try_add(
            &mut self,
            li: usize,
            gate: &QtWeight,
            up: &QtWeight,
            down: &QtWeight,
            headroom: usize,
        ) -> Result<bool, Error> {
            use crate::weight::QuantFmt;
            if gate.fmt != QuantFmt::Int4 || up.fmt != QuantFmt::Int4 || down.fmt != QuantFmt::Int4 {
                return Err(Error::Format(format!(
                    "gpu dense tier: layer {li} MLP is not per-row int4 ({:?}/{:?}/{:?}); \
                     uploading it would cost 8x the VRAM as f32",
                    gate.fmt, up.fmt, down.fmt
                )));
            }
            let (gq, gs) = gate.raw();
            let (uq, us) = up.raw();
            let (dq, ds) = down.raw();
            let need = gq.len() + uq.len() + dq.len() + 4 * (gs.len() + us.len() + ds.len());
            let (free, _) = peregrine_cuda::mem_info(self.device)?;
            if free < need.saturating_add(headroom) {
                self.skipped += 1;
                return Ok(false);
            }
            let e = GpuExpert::upload_int4(
                self.device,
                (gq, gs),
                (uq, us),
                (dq, ds),
                gate.i,
                gate.o,
            )?;
            self.mlps.insert(li, e);
            self.bytes += need;
            Ok(true)
        }

        /// Whether layer `li`'s MLP computes on the device.
        pub fn has(&self, li: usize) -> bool {
            self.mlps.contains_key(&li)
        }

        /// Run layer `li`'s SwiGLU on the device. `None` when the layer is not
        /// resident — the caller then takes the CPU path, which is the same
        /// arithmetic by a slower road.
        pub fn mlp(&self, li: usize, x: &[f32], s_n: usize, hidden: usize) -> Option<Result<Vec<f32>, Error>> {
            let e = self.mlps.get(&li)?;
            Some(peregrine_cuda::dense_mlp_w4a16(e, x, s_n, hidden))
        }

        /// `(layers resident, bytes held, layers skipped for budget)`.
        pub fn stats(&self) -> (usize, usize, usize) {
            (self.mlps.len(), self.bytes, self.skipped)
        }
    }

    pub struct GpuTier {
        /// Every device this process initialized for the tier, in the order
        /// [`super::devices::selected_devices`] listed them. One device is the
        /// historical shape and behaves exactly as before; more than one turns
        /// the tier into a partitioned residency (see `owner`).
        devices: Vec<i32>,
        experts: HashMap<(usize, usize), GpuExpert>,
        /// Which device holds each resident — the owner map from
        /// `docs/scale-out-design.md` §1. The key set is exactly
        /// [`Self::experts`]'s (they are inserted and removed together), so
        /// residency questions go to `experts` and *placement* questions to this.
        owner: HashMap<(usize, usize), i32>,
        capacity: usize,
        /// VRAM each device's resident set may occupy, same order as
        /// [`Self::devices`]. Per device, because VRAM is per device: an f32
        /// promotion (~8× its int4 size) on card 0 cannot be paid out of card 1.
        /// The sum is the aggregate a placement is solved against; the split is
        /// what keeps it honest.
        budgets: Vec<usize>,
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
        /// Experts held resident regardless of rank — the MTP head's, chosen by
        /// [`super::PinRequest`]. Their bytes come off the budget before the
        /// heat plan is solved, and they are added to every generation's wanted
        /// set so the eviction pass cannot take them.
        ///
        /// Kept as state rather than recomputed inline because eviction needs to
        /// know what *was* pinned: an expert that cools out of the plan has to
        /// lose its slot, and `self.experts` alone cannot say whether a resident
        /// at layer `n_layers` is there because it was pinned or because it was
        /// pinned last generation and should not be any more.
        pinned: HashSet<(usize, usize)>,
        /// MoE intermediate dim — the `I` of every expert's `KernelShape`. Held
        /// on the tier because `compute` has no `Cfg` and a shape keyed on the
        /// wrong `I` would merge two genuinely different kernels into one row.
        inter: usize,
        /// Online WMMA tile autotuner (`COLI_CUDA_AUTOTUNE=1`), `None` when off.
        /// `Mutex` because `compute` takes `&self` — one GPU-lane thread issues
        /// dispatch, so it is never contended; it is here to hold the borrow
        /// checker's line, not a race.
        tuner: Option<parking_lot::Mutex<crate::wmma_tune::WmmaTuner>>,
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
    /// Host buffers an async upload's DMA is still reading out of.
    ///
    /// The GPU keeps reading these *after* `upload_int4_async` has returned, so
    /// freeing one before the stream drains hands the DMA engine freed memory.
    /// `Drop` therefore drains as a backstop: [`drain_uploads`] is the fast path
    /// that syncs a whole batch at once and marks these done, but any other way
    /// out of the loop — an early return, an error, a `break` — still cannot get
    /// this wrong.
    struct HostStaging {
        device: i32,
        drained: bool,
        _gate: QtWeight,
        _up: QtWeight,
        _down: QtWeight,
    }

    impl Drop for HostStaging {
        fn drop(&mut self) {
            if !self.drained {
                if let Err(e) = peregrine_cuda::stream_sync(self.device) {
                    peregrine_io::note_advisory_err("gpu upload drain on drop", &e);
                }
            }
        }
    }

    /// Experts uploaded through the pinned async lane, and through the blocking
    /// one, since process start.
    ///
    /// These exist because every other symptom of a dead lane is invisible. A
    /// tier that pins nothing still loads, still computes the right answer, and
    /// still prints the same boot line — it is only slower, and only against a
    /// baseline nobody has. Reporting the split turns "is the lane on?" into a
    /// number instead of an inference.
    static UPLOADS_PINNED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static UPLOADS_BLOCKING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// `(pinned async, blocking)` expert uploads so far.
    pub fn upload_lane_counts() -> (usize, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        (UPLOADS_PINNED.load(Relaxed), UPLOADS_BLOCKING.load(Relaxed))
    }

    /// How many experts' uploads may be in flight before the lane drains
    /// (`COLI_GPU_UPLOAD_DEPTH`, default 4).
    ///
    /// This is where the overlap actually comes from: expert N's H2D DMA runs
    /// while expert N+1's weights are still being read off disk. Depth 1 is
    /// queue-then-immediately-wait — the blocking path with extra steps. The
    /// cost is host memory: each in-flight expert holds ~18.9 MB of pinned
    /// landing buffer until its copy completes.
    fn upload_depth() -> usize {
        static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("COLI_GPU_UPLOAD_DEPTH")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .map_or(4, |d| d.clamp(1, 64))
        })
    }

    /// Wait for every queued upload on `device`, then release the host buffers
    /// they were reading. The order is the point: the sync must return before
    /// the staging is dropped.
    fn drain_uploads(device: i32, pending: &mut Vec<HostStaging>) -> Result<(), Error> {
        if pending.is_empty() {
            return Ok(());
        }
        let res = peregrine_cuda::stream_sync(device);
        // Mark drained whether or not the sync reported success: if it failed
        // there is nothing a per-buffer drop can do better, and re-syncing once
        // per buffer on the way out would just repeat the same failure.
        for p in pending.iter_mut() {
            p.drained = true;
        }
        pending.clear();
        res
    }

    fn upload_expert(
        st: &SafeTensors,
        cfg: &Cfg,
        layer: usize,
        e: usize,
        device: i32,
        int4: bool,
    ) -> Result<(GpuExpert, bool, Option<HostStaging>), Error> {
        let hidden = cfg.hidden as usize;
        let inter = cfg.moe_inter as usize;
        let pe = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
        // Whether these bytes reach VRAM verbatim decides the landing buffer,
        // and it has to be decided *before* the load. The int4 path uploads the
        // payload as-is, so it wants a page-aligned buffer the pin hook has
        // registered with CUDA — io_uring then DMAs disk bytes straight into the
        // H2D source. The f32 path dequantizes on the host, so the bytes the GPU
        // receives are computed rather than read and an aligned buffer buys
        // nothing. `expert_is_per_row_int4` answers this from the index, without
        // reading any payload.
        let raw_to_vram =
            int4 && crate::pinned::enabled() && expert_is_per_row_int4(st, cfg, layer, e);
        let load = |name: String, o: usize, i: usize| {
            if raw_to_vram {
                QtWeight::load_aligned(st, &name, o, i)
            } else {
                QtWeight::load(st, &name, o, i)
            }
        };
        let gate = load(pe("gate_proj.weight"), inter, hidden)?;
        let up = load(pe("up_proj.weight"), inter, hidden)?;
        let down = load(pe("down_proj.weight"), hidden, inter)?;
        use crate::weight::QuantFmt;
        let all_int4 = gate.fmt == QuantFmt::Int4 && up.fmt == QuantFmt::Int4 && down.fmt == QuantFmt::Int4;
        if int4 && all_int4 {
            // Async only when the payload really landed in pinned memory. An
            // async copy out of pageable memory is legal but the driver bounces
            // it through its own staging buffer, so it serializes anyway — and
            // it would still owe the sync. Not worth taking on the obligation
            // for none of the benefit.
            // Ask whether the payload really is pinned, not whether its pointer
            // happens to be page-aligned — a plain `Vec` can be that by luck.
            // Async out of pageable memory is legal but the driver bounces it, so
            // it serializes anyway while still owing the caller a sync: all of
            // the obligation, none of the benefit.
            let pinned = raw_to_vram && gate.is_pinned() && up.is_pinned() && down.is_pinned();
            if pinned {
                let ex = GpuExpert::upload_int4_async(device, gate.raw(), up.raw(), down.raw(), hidden, inter)?;
                UPLOADS_PINNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok((ex, true, Some(HostStaging { device, drained: false, _gate: gate, _up: up, _down: down })));
            }
            UPLOADS_BLOCKING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok((GpuExpert::upload_int4(device, gate.raw(), up.raw(), down.raw(), hidden, inter)?, true, None))
        } else {
            UPLOADS_BLOCKING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok((
                GpuExpert::upload(device, &gate.dequant(), &up.dequant(), &down.dequant(), hidden, inter)?,
                false,
                None,
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
        // Every sparse layer, including the MTP head at `cfg.n_layers`, and not
        // just one expert of one layer. A single probe of (first_dense, 0) answers
        // for a uniform container but reports "all int4" on a per-layer-tiered one
        // — the GLM-5.2 checkpoint stores the MTP layer's 256 experts at int8 and
        // the rest at int4, so the probe said yes and the operator never saw the
        // "not per-row int4" notice that explains an 8x-larger VRAM plan. Expert 0
        // still stands in for its layer: precision is chosen per layer by
        // `peregrine-requantize`, and `solve_residency_sized` asks per expert
        // anyway when it actually sizes the tier.
        ((cfg.first_dense as usize)..=(cfg.n_layers as usize))
            .all(|layer| expert_is_per_row_int4(st, cfg, layer, 0))
    }

    /// Whether **this** expert stores all three projections as per-row int4, i.e.
    /// whether it will upload raw (int4-resident) or dequantized (f32, ~8×).
    ///
    /// The per-expert form exists because the whole-container probe above answers
    /// one question — "can the entire tier be planned as int4?" — and a tiered
    /// container makes that question the wrong one. On a mix, the probe correctly
    /// says "no" and the planner then sizes *every* expert at f32, which is safe
    /// but ~8× too pessimistic for the int4 majority; the tier ends up a fraction
    /// of the VRAM it was given. Asking per expert costs one index lookup per
    /// candidate at load time and gets both halves right.
    fn expert_is_per_row_int4(st: &SafeTensors, cfg: &Cfg, layer: usize, e: usize) -> bool {
        use peregrine_core::QtInfo;
        let (hidden, inter) = (cfg.hidden, cfg.moe_inter);
        let pe = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
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
        /// placed by the heat/bytes knapsack over `counts`, **partitioned across
        /// every device** [`super::devices::selected_devices`] returns (one device
        /// is the historical shape and behaves exactly as before). `int4` uploads
        /// per-row int4 weights directly (~8× denser), falling back to dequantized
        /// f32 per expert for sources that aren't per-row int4. `Ok(None)` when no
        /// requested device can be initialized or nothing fits. Takes `int4` and
        /// `counts` explicitly so it is testable without racing process env.
        pub fn build_with(
            st: &SafeTensors,
            cfg: &Cfg,
            headroom_bytes: usize,
            int4: bool,
            counts: &[u32],
        ) -> Result<Option<GpuTier>, Error> {
            let devices = crate::devices::selected_devices();
            if devices.is_empty() {
                eprintln!(
                    "peregrine: COLI_GPU_DEVICES names no usable GPU — building without a VRAM tier \
                     (see docs/gpu-vendors.md)"
                );
                return Ok(None);
            }
            // All-or-nothing on the C side (`coli_cuda_init` resets to zero
            // contexts if any listed device fails), so success here covers the
            // whole list.
            if peregrine_cuda::init(&devices) != 1 {
                return Ok(None);
            }
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
            // Contexts this process actually initialized, as opposed to the
            // driver-visible count the startup banner reports. They differ
            // whenever init partially fails, and that difference is exactly what
            // a "the GPU tier is smaller than I asked for" report needs.
            eprintln!(
                "peregrine: CUDA contexts initialized: {} (devices {devices:?})",
                peregrine_cuda::device_count()
            );
            if int4 && !raw_int4 {
                eprintln!(
                    "peregrine: COLI_GPU_INT4 set, but this container's experts are not per-row \
                     int4 — they upload dequantized to f32 (8x), so the VRAM plan is sized for f32"
                );
            }
            // Headroom is PER DEVICE: each card keeps its own slice for
            // activations/context, so two cards reserve twice in absolute terms —
            // which is the honest reading of what the knob always meant.
            let mut budgets = Vec::with_capacity(devices.len());
            for &d in &devices {
                let (free, _total) = peregrine_cuda::mem_info(d)?;
                budgets.push(free.saturating_sub(headroom_bytes));
            }
            let budget: usize = budgets.iter().sum();

            // Heat-density knapsack over the persisted routing counts. On a cold
            // table (no `route_stats.json`, or a fingerprint mismatch) this falls
            // back internally to the same round-robin `plan_residency` placement,
            // so a first run is byte-for-byte what it always was. Solved over the
            // AGGREGATE budget; the device split happens next, in
            // `partition_placement`.
            let placement = super::solve_residency_sized(
                counts,
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                budget,
                // Per-expert, not one scalar. `solve_residency_sized` takes this
                // closure precisely so a tier holding differently-sized residents
                // still fits its budget, and passing `|_, _| constant` gave that
                // back for free. On a uniform container every call returns the
                // same number and the plan is unchanged; on a tiered one the int4
                // experts stop being budgeted as if they were f32.
                |layer, e| {
                    if int4 && expert_is_per_row_int4(st, cfg, layer, e) {
                        int4_bytes
                    } else {
                        f32_bytes
                    }
                },
            );
            // Split the heat-ordered placement across devices: hottest first,
            // each to the device with the most room left. Pure and deterministic
            // (ties break to the lowest index), so the same heat table and the
            // same budgets always produce the same owner map — and when something
            // stops fitting, truncation matches the single-device loop it
            // replaces: everything after the first non-fit is colder anyway.
            let assigned = super::partition_placement(&placement, &budgets, |(layer, e)| {
                if int4 && expert_is_per_row_int4(st, cfg, *layer, *e) {
                    int4_bytes
                } else {
                    f32_bytes
                }
            });
            let mut capacity = assigned.len(); // expert-count view of the budget
            let mut experts = HashMap::new();
            let mut precision: HashMap<(usize, usize), bool> = HashMap::new();
            let mut forced_f32: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
            let mut owner: HashMap<(usize, usize), i32> = HashMap::with_capacity(capacity);
            // Uploads whose DMA is still running, kept PER DEVICE —
            // `stream_sync` drains one card, so a shared queue would wait on (and
            // mark drained) the wrong one. Nothing here evicts, so each lane can
            // run `upload_depth()` experts deep: expert N's copy to VRAM overlaps
            // expert N+1's read off disk.
            let mut pending: Vec<Vec<HostStaging>> = (0..devices.len()).map(|_| Vec::new()).collect();
            let depth = upload_depth();
            for ((layer, e), dev_idx) in assigned {
                let device = devices[dev_idx];
                match upload_expert(st, cfg, layer, e, device, int4) {
                    Ok((ge, landed_int4, staging)) => {
                        if let Some(sg) = staging {
                            pending[dev_idx].push(sg);
                            if pending[dev_idx].len() >= depth {
                                if let Err(e_s) = drain_uploads(device, &mut pending[dev_idx]) {
                                    peregrine_io::note_advisory_err("gpu upload drain (tier truncated)", &e_s);
                                    capacity = experts.len();
                                    break;
                                }
                            }
                        }
                        experts.insert((layer, e), ge);
                        owner.insert((layer, e), device);
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
            // Before anything reads these weights or drops their host buffers.
            // `HostStaging::drop` would cover the second on its own; doing it
            // here covers the first as well, and does it once per device instead
            // of once per buffer.
            for (idx, q) in pending.iter_mut().enumerate() {
                if let Err(e_s) = drain_uploads(devices[idx], q) {
                    peregrine_io::note_advisory_err("gpu upload drain (final)", &e_s);
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
                eprintln!(
                    "peregrine: GPU tier partitioned across {} device(s): {}",
                    devices.len(),
                    super::owner_summary(&owner)
                );
                Ok(Some(GpuTier {
                    devices,
                    experts,
                    owner,
                    capacity,
                    budgets,
                    expert_bytes: (int4_bytes, f32_bytes),
                    int4,
                    adaptive_f32_frac,
                    precision,
                    forced_f32,
                    // Empty at build, and not for want of trying: the pin
                    // counter is fed by drafting, so at load it is all zeros and
                    // `plan_pins` on a cold table is empty by design. The tier's
                    // first pins arrive at the first `reheat` after the first
                    // draft rounds — which is also when their ranking first
                    // means anything.
                    pinned: HashSet::new(),
                    inter: cfg.moe_inter as usize,
                    // A *second* opt-in on top of `COLI_CUDA_TC_W4A16`: the tile
                    // only reaches that arm, so tuning without it would record
                    // measurements of a kernel the tile never touched and then
                    // "select" between them.
                    tuner: (std::env::var("COLI_CUDA_AUTOTUNE").as_deref() == Ok("1"))
                        .then(|| parking_lot::Mutex::new(crate::wmma_tune::WmmaTuner::new())),
                }))
            }
        }

        /// The tuner's table, for persistence beside `route_stats.json`.
        /// `None` when autotuning is off — so an absent knob writes no file
        /// rather than an empty one that looks like a measured result.
        pub fn tuning_json(&self) -> Option<serde_json::Value> {
            self.tuner.as_ref().map(|t| t.lock().to_json())
        }

        /// Restore a table written by [`Self::tuning_json`]. Ignored when
        /// autotuning is off.
        pub fn restore_tuning(&self, v: &serde_json::Value) {
            if let Some(t) = self.tuner.as_ref() {
                *t.lock() = crate::wmma_tune::WmmaTuner::from_json(v);
            }
        }

        /// Re-select the VRAM-resident set as the `capacity` hottest experts by
        /// `counts` (routing frequency), evicting experts that cooled and uploading
        /// newly-hot ones. Reuses [`Self::build`]'s dequantize+upload path. Called
        /// between forwards with `&mut self`, so residency adapts to the workload
        /// without a rewrite. Returns the resident count after re-selection.
        pub fn reheat(
            &mut self,
            st: &SafeTensors,
            cfg: &Cfg,
            heat: &super::HeatView,
            pins: &super::PinRequest,
        ) -> Result<usize, Error> {
            // Pins first, and outside the policy branch: they are a reservation,
            // not a rank, so every swap policy has to see the same reserved set.
            // It also has to happen before the budget is read — `sync_pins` may
            // upload, and the re-plan must solve against the VRAM that is left
            // afterwards, not the VRAM that existed before.
            let pinned_bytes = self.sync_pins(st, cfg, heat, pins);
            // Incremental policies run instead of the re-plan, not before it:
            // both decide the same thing (which experts are resident next
            // generation) and running one after the other would let the re-plan
            // immediately undo every swap.
            match super::swap_policy() {
                super::SwapPolicy::Replan => {}
                // `plan_swaps` enumerates candidates and victims over
                // `first_dense..n_layers`, exclusive, so a pin at layer
                // `n_layers` is outside its universe entirely — it can neither
                // be swapped in nor chosen as a victim. Nothing further to do
                // for pins here beyond the sync above.
                policy => return self.reheat_incremental(st, cfg, heat, policy, pinned_bytes),
            }
            self.reheat_replan(st, cfg, heat, pinned_bytes)
        }

        /// The re-plan half of [`Self::reheat`]: re-rank every candidate against
        /// the budget the reservation left and migrate residency to match.
        ///
        /// Split out of `reheat` so it has a name the incremental path can call.
        /// It used to fall back by calling `reheat` itself, which re-entered the
        /// policy branch that had just dispatched *to* it — `COLI_GPU_TIER_SWAP`
        /// set to a swap policy together with `COLI_GPU_F32_FRAC` was unbounded
        /// mutual recursion, i.e. a stack overflow on the first residency
        /// generation. The two knobs are individually exercised and the
        /// combination apparently never was.
        fn reheat_replan(
            &mut self,
            st: &SafeTensors,
            cfg: &Cfg,
            heat: &super::HeatView,
            pinned_bytes: usize,
        ) -> Result<usize, Error> {
            let counts = heat.counts;
            // Re-read free VRAM each generation, per device: another process may
            // have taken some since load, and the budget must reflect what is
            // available now (plus what this tier already holds there, which it is
            // free to reuse). Summing per-device budgets is the aggregate the
            // ranking is solved against; each device's own figure still governs
            // where an upload lands (see the targeting below).
            let mut budget = 0usize;
            for &d in &self.devices {
                let held = self.held_on(d);
                let cap = self.budget_of(d);
                budget += match peregrine_cuda::mem_info(d) {
                    Ok((free, _)) => cap.min(free.saturating_add(held)),
                    // Query failure: keep that device's load-time budget rather
                    // than guessing.
                    Err(e) => {
                        peregrine_io::note_advisory_err("gpu mem_info during reheat", &e);
                        cap
                    }
                };
            }
            // What the ranking may spend: the tier's budget less the reservation.
            // `held` above already counts the pinned residents, so this is the
            // one place their bytes are taken out of the ranking's reach.
            let budget = budget.saturating_sub(pinned_bytes);
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
                    // Uniform format: every resident is the same size, so a
                    // byte-budgeted knapsack over a constant cost is the whole
                    // decision.
                    //
                    // This was hand-rolled as `rank_by_heat(.., capacity)` then
                    // `take(budget / uniform)` — which is what
                    // `solve_residency_greedy` does, except that the hand-rolled
                    // version **lost the cold-start fallback**: on an all-zero
                    // heat table `solve_residency_sized` reproduces
                    // `plan_residency`'s round-robin spread, while ranking by
                    // heat and truncating gives whatever order the sort happened
                    // to leave. So `build` and `reheat` disagreed about placement
                    // on a cold table, and the first reheat of a fresh process
                    // reshuffled residency for no reason.
                    None => {
                        // `rank_by_heat`, not `solve_residency_greedy`. They differ
                        // on **equal heat**: `rank_by_heat` keeps candidates in
                        // round-robin order, which is `plan_residency`'s static
                        // placement, while `solve_residency_sized` breaks ties by
                        // ascending (layer, expert). A partially-warm table has many
                        // ties, so swapping them silently reshuffles residency for
                        // experts the heat table cannot distinguish — churn with no
                        // information behind it, and every moved expert is a PCIe
                        // upload. The byte budget is applied after ranking because
                        // in this branch every resident is the same size, which is
                        // what makes the count-based rank sufficient.
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
            let mut want_set: HashSet<(usize, usize)> = want.iter().copied().collect();
            // The reservation joins the *set* but not the upload list: `sync_pins`
            // has already uploaded them, and this is what stops the eviction pass
            // below from taking a pin that no heat ranking ever proposed.
            want_set.extend(self.pinned.iter().copied());
            // evict experts that cooled off — their `Drop` frees the VRAM slot
            self.experts.retain(|k, _| want_set.contains(k));
            self.owner.retain(|k, _| want_set.contains(k));
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

            // At most one admission in flight here: each iteration frees the
            // victim's VRAM, and that must not race a copy still running. The
            // in-flight queue is per device for the same reason as at build.
            let mut pending: Vec<Vec<HostStaging>> = (0..self.devices.len()).map(|_| Vec::new()).collect();
            // Where each device's remaining room stood after the eviction pass:
            // the upload target for every admission this generation, hottest
            // expert first, each to the card with the most space left (the same
            // rule `partition_placement` used at build, so residency migrates
            // toward the split a cold rebuild would have chosen rather than
            // drifting away from it).
            let mut remaining: Vec<usize> = self.devices.iter().map(|&d| self.free_on(d)).collect();
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
                let bytes = if want_int4 { int4_bytes } else { f32_bytes };
                // Most-room-first targeting; ties to the lowest device index.
                let Some(dev_idx) = super::pick_device(&remaining, bytes) else {
                    // Nowhere left at this expert's real size. Same stop rule as
                    // the byte budget above: later candidates are colder.
                    break;
                };
                let device = self.devices[dev_idx];
                upload_quota -= 1;
                // Re-upload on a format change (remove first so the old tensor's
                // Drop frees its VRAM before the new allocation).
                // Drain the previous admission before freeing any VRAM: the
                // eviction below must not race a copy that is still running, and
                // this is also what lets that copy overlap this iteration's
                // planning work.
                if let Err(e_s) = drain_uploads(device, &mut pending[dev_idx]) {
                    peregrine_io::note_advisory_err("gpu reheat upload drain", &e_s);
                }
                self.experts.remove(&key);
                match upload_expert(st, cfg, layer, e, device, want_int4) {
                    // Record the format that actually landed, not the one asked for.
                    Ok((ge, landed_int4, staging)) => {
                        pending[dev_idx].extend(staging);
                        self.experts.insert(key, ge);
                        self.owner.insert(key, device);
                        self.precision.insert(key, landed_int4);
                        if want_int4 && !landed_int4 {
                            self.forced_f32.insert(key);
                        }
                        remaining[dev_idx] = remaining[dev_idx].saturating_sub(bytes);
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
            for (idx, q) in pending.iter_mut().enumerate() {
                if let Err(e_s) = drain_uploads(self.devices[idx], q) {
                    peregrine_io::note_advisory_err("gpu reheat upload drain (final)", &e_s);
                }
            }
            Ok(self.experts.len())
        }

        /// Bring VRAM residency of the MTP head's experts in line with `req`,
        /// and return the bytes the pinned set now occupies.
        ///
        /// **Why the pinned set needs its own pass at all.** Every residency
        /// planner in this module — `plan_residency`, `rank_by_heat`,
        /// `solve_residency_sized`, `plan_precision_fitted`, `plan_swaps` —
        /// enumerates candidates over `first_dense..n_layers`, **exclusive**. The
        /// MTP head is layer `n_layers`. It has therefore never been a candidate
        /// in any of them, no matter what its heat row said, which is why
        /// `COLI_MTP_HEAT` could fill that row and change nothing about VRAM: the
        /// row was read by a ranking that never enumerated it.
        ///
        /// Failures here are advisory. A pin that cannot upload is a residency
        /// loss, not a correctness one — the expert streams from the CPU lane,
        /// producing the same bytes — so a bad generation leaves the tier
        /// consistent and the next one retries.
        fn sync_pins(
            &mut self,
            st: &SafeTensors,
            cfg: &Cfg,
            heat: &super::HeatView,
            req: &super::PinRequest,
        ) -> usize {
            static CLAMPED: std::sync::Once = std::sync::Once::new();
            let budget = crate::mtp::granted_pin_budget(
                "COLI_MTP_PIN_VRAM_MB",
                "GPU tier",
                req.budget,
                self.total_budget(),
                &CLAMPED,
            );
            let (int4_bytes, f32_bytes) = self.expert_bytes;
            // Sized per expert exactly as `build` sizes its placement: an int8 or
            // grouped-int4 source cannot be int4-resident and uploads dequantized
            // at ~8x. Sizing the plan from the *request* instead would plan N
            // experts and upload 8N worth — the same defect the `raw_int4` split
            // in `build_with` exists to prevent, and the MTP layer is precisely
            // the layer a GLM-5.2 container leaves at the other rung.
            let bytes_of = |e: usize| {
                if self.int4 && expert_is_per_row_int4(st, cfg, req.layer, e) {
                    int4_bytes
                } else {
                    f32_bytes
                }
            };
            let want: HashSet<(usize, usize)> = crate::mtp::plan_pins(req.counts, budget, bytes_of)
                .into_iter()
                .map(|e| (req.layer, e))
                .collect();
            // Unpin what cooled out of the plan. Dropping the tensor frees the
            // VRAM before anything below allocates, which is the same ordering
            // the swap path documents.
            for k in std::mem::take(&mut self.pinned) {
                if !want.contains(&k) {
                    self.experts.remove(&k);
                    self.owner.remove(&k);
                    self.precision.remove(&k);
                    self.forced_f32.remove(&k);
                }
            }
            // Ascending expert id is ascending offset within a layer, so the
            // uploads read the container forwards.
            let mut order: Vec<(usize, usize)> = want.iter().copied().collect();
            order.sort_unstable();
            // Make room before uploading, and make it in *heat order*.
            //
            // Without this the reservation could never establish itself on a
            // tier that is doing its job: `build` fills the budget, so the first
            // generation with a pin budget set finds no free VRAM, every pin
            // upload fails, and — since each generation retries identically —
            // the tier stays exactly as it was, forever, while the log fills
            // with advisories. Evicting here rather than leaving it to the
            // re-plan below is what makes the pins work under the incremental
            // swap policies too, which hold the resident *set size* fixed and so
            // can never free room for a reservation on their own.
            //
            // The victims are the coldest non-pinned residents, which is the same
            // order the re-plan would drop them in, so this is not extra churn —
            // it is the same eviction, moved early enough to be useful.
            let need: usize = order
                .iter()
                .filter(|k| !self.experts.contains_key(k))
                .map(|&(_, e)| bytes_of(e))
                .sum();
            // Pins land on ONE device — the emptiest — rather than spreading:
            // a reservation is a handful of experts, and keeping them together
            // keeps the eviction that makes room on one card's ledger instead of
            // spraying victims across every device for the same few bytes.
            let target_idx = self
                .devices
                .iter()
                .enumerate()
                .max_by_key(|&(i, &d)| (self.free_on(d), std::cmp::Reverse(i)))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let target = self.devices[target_idx];
            self.evict_coldest_for(need, target, heat, cfg.n_experts as usize);
            let mut bytes = 0usize;
            let mut pending: Vec<HostStaging> = Vec::new();
            for key in order {
                if self.experts.contains_key(&key) {
                    // Already resident, in whatever format it landed in. A pin is
                    // not a precision request, so a resident pin is never
                    // re-uploaded to chase a format.
                    bytes += if self.precision.get(&key).copied().unwrap_or(self.int4) { int4_bytes } else { f32_bytes };
                    self.pinned.insert(key);
                    continue;
                }
                match upload_expert(st, cfg, key.0, key.1, target, self.int4) {
                    Ok((ge, landed_int4, staging)) => {
                        pending.extend(staging);
                        self.experts.insert(key, ge);
                        self.owner.insert(key, target);
                        self.precision.insert(key, landed_int4);
                        if self.int4 && !landed_int4 {
                            self.forced_f32.insert(key);
                        }
                        self.pinned.insert(key);
                        bytes += if landed_int4 { int4_bytes } else { f32_bytes };
                    }
                    Err(e_up) => {
                        peregrine_io::note_advisory_err("gpu mtp pin upload (expert left streaming)", &e_up);
                        self.precision.remove(&key);
                        break;
                    }
                }
            }
            if let Err(e_s) = drain_uploads(target, &mut pending) {
                peregrine_io::note_advisory_err("gpu mtp pin upload drain", &e_s);
            }
            bytes
        }

        /// Evict the coldest **non-pinned** residents **on `device`** until `need`
        /// more bytes fit inside that device's budget. Returns nothing: a tier that
        /// cannot make the room simply ends up with fewer pins, which the upload
        /// loop already degrades to correctly.
        ///
        /// Victims come only from the device that must absorb the bytes: freeing
        /// VRAM elsewhere does not help the upload that triggered this. Ranked by
        /// the same heat view the re-plan ranks by, and tie-broken on the key, so
        /// a victim here is a victim there — the two passes cannot disagree about
        /// which expert is worth least and swap it back and forth between
        /// generations.
        fn evict_coldest_for(&mut self, need: usize, device: i32, heat: &super::HeatView, n_experts: usize) {
            if need == 0 {
                return;
            }
            let size_of = |k: &(usize, usize), me: &Self| {
                if me.precision.get(k).copied().unwrap_or(me.int4) { me.expert_bytes.0 } else { me.expert_bytes.1 }
            };
            let held: usize =
                self.experts.keys().filter(|k| self.owner.get(*k) == Some(&device)).map(|k| size_of(k, self)).sum();
            let mut over = (held + need).saturating_sub(self.budget_of(device));
            if over == 0 {
                return;
            }
            // Coldest first, deterministic on ties.
            let mut victims: Vec<(usize, usize)> = self
                .experts
                .keys()
                .copied()
                .filter(|k| !self.pinned.contains(k) && self.owner.get(k) == Some(&device))
                .collect();
            let heat_of = |&(l, e): &(usize, usize)| {
                heat.counts.get(l.saturating_mul(n_experts).saturating_add(e)).copied().unwrap_or(0)
            };
            victims.sort_by_key(|k| (heat_of(k), *k));
            for k in victims {
                if over == 0 {
                    break;
                }
                let freed = size_of(&k, self);
                self.experts.remove(&k);
                self.owner.remove(&k);
                self.precision.remove(&k);
                self.forced_f32.remove(&k);
                over = over.saturating_sub(freed);
            }
        }

        /// [`Self::reheat`] under an incremental [`SwapPolicy`]: hold the
        /// resident *set size* fixed and move at most one expert per layer.
        ///
        /// **Why this exists beside the re-plan rather than replacing it.** The
        /// re-plan re-ranks every candidate, so any expert whose heat rank moved
        /// is an upload — on a churny generation that is gigabytes into the PCIe
        /// lane the tier is meant to be feeding, which is why
        /// `COLI_PCIE_BUDGET_MB` had to exist to truncate it. These policies
        /// brake at the source: `pick_lfru`/`pick_swap` refuse a swap unless the
        /// candidate beats the victim by 25 % **plus** four routing counts, so a
        /// generation where nothing meaningfully changed uploads nothing at all.
        ///
        /// The re-plan remains the default because it is the only one that can
        /// *resize* the resident set (VRAM freed by another process, a layer
        /// holding nothing) — a swap has no opinion about how many experts
        /// should be resident, only about which.
        ///
        /// **Adaptive precision is not supported here and falls back rather than
        /// approximating.** `plan_precision_fitted` decides residency and format
        /// together; a one-in-one-out swap that assumed the tier's uniform
        /// format would silently undo the f32 promotions the knob was set for.
        fn reheat_incremental(
            &mut self,
            st: &SafeTensors,
            cfg: &Cfg,
            heat: &super::HeatView,
            policy: super::SwapPolicy,
            pinned_bytes: usize,
        ) -> Result<usize, Error> {
            if self.adaptive_f32_frac.is_some() {
                peregrine_io::note_advisory_err(
                    "COLI_GPU_TIER_SWAP is ignored while adaptive f32 residency is on \
                     (the two decide the same thing); using replan",
                    &format!("{policy:?}"),
                );
                // The re-plan body directly, not `reheat` — `reheat` is what
                // dispatched here, and calling it back is the recursion
                // `reheat_replan`'s doc describes. The pins are already synced.
                return self.reheat_replan(st, cfg, heat, pinned_bytes);
            }
            let swaps = super::plan_swaps(
                heat,
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                policy,
                |layer, e| self.experts.contains_key(&(layer, e)),
            );
            // Same PCIe brake as the re-plan path, over the same cost basis. A
            // swap admits exactly one expert in the tier's uniform format.
            let unit = if self.int4 { self.expert_bytes.0 } else { self.expert_bytes.1 };
            let costs: Vec<usize> = swaps.iter().map(|_| unit).collect();
            let mut quota = super::admit_uploads(&costs, super::pcie_budget_bytes());
            // One admission in flight, for the same reason as the re-plan above:
            // each swap frees the victim's VRAM and must not race a live copy.
            // Single queue plus a "which device it belongs to" tag: swaps are
            // one-in-one-out and strictly ordered, so per-device queues would
            // only ever hold one entry each.
            let mut pending: Vec<HostStaging> = Vec::new();
            let mut pending_device: Option<i32> = None;
            // Remaining room per device, kept current across the loop: a victim's
            // eviction refunds its card before the admission picks its target.
            let mut remaining: Vec<usize> = self.devices.iter().map(|&d| self.free_on(d)).collect();
            for (victim, admit) in swaps {
                if quota == 0 {
                    break;
                }
                quota -= 1;
                // Evict FIRST so the victim's `Drop` returns its VRAM before the
                // admission allocates. Reversed, a tier sized to fill the budget
                // would need one expert's worth of headroom it does not have,
                // and every swap would fail on the last free byte.
                // The previous admission's copy must finish before this
                // eviction frees VRAM — see the ordering note above, which the
                // async lane must not weaken. Drained on whichever device that
                // admission targeted.
                if let Some(d) = pending_device.take() {
                    if let Err(e_s) = drain_uploads(d, &mut pending) {
                        peregrine_io::note_advisory_err("gpu swap upload drain", &e_s);
                    }
                }
                let victim_dev = self.owner.get(&victim).copied();
                let evicted = self.experts.remove(&victim);
                self.owner.remove(&victim);
                let victim_int4 = self.precision.remove(&victim).unwrap_or(self.int4);
                self.forced_f32.remove(&victim);
                drop(evicted);
                // The victim's card earns its bytes back before the targeting
                // step below sees `remaining`.
                if let Some(i) = victim_dev.and_then(|d| self.devices.iter().position(|&x| x == d)) {
                    remaining[i] += unit;
                }
                // The admission lands on whichever device has the most room
                // now (the victim just freed its card), not necessarily where
                // the victim lived.
                let Some(dev_idx) = super::pick_device(&remaining, unit) else {
                    // No device fits even after the eviction — leave the slot
                    // empty; the next reheat retries from a consistent state.
                    break;
                };
                let device = self.devices[dev_idx];
                match upload_expert(st, cfg, admit.0, admit.1, device, self.int4) {
                    Ok((ge, landed_int4, staging)) => {
                        pending.extend(staging);
                        pending_device = Some(device);
                        self.experts.insert(admit, ge);
                        self.owner.insert(admit, device);
                        self.precision.insert(admit, landed_int4);
                        if self.int4 && !landed_int4 {
                            self.forced_f32.insert(admit);
                        }
                        remaining[dev_idx] = remaining[dev_idx].saturating_sub(unit);
                    }
                    // The victim is already gone and its bytes are already freed,
                    // so the tier is one expert short until the next generation
                    // — which is a residency loss, not a correctness one: a
                    // non-resident expert streams from the CPU lane. Do not try
                    // to re-upload the victim; that is the same allocation that
                    // just failed, and failing it twice would leave the maps
                    // describing a tier that does not exist.
                    Err(e_up) => {
                        peregrine_io::note_advisory_err(
                            &format!(
                                "gpu tier swap upload for ({}, {}) — slot left empty, \
                                 evicted ({}, {}) [int4={}] is not restored",
                                admit.0, admit.1, victim.0, victim.1, victim_int4
                            ),
                            &e_up,
                        );
                        break;
                    }
                }
            }
            if let Err(e_s) = pending_device.take().map_or(Ok(()), |d| drain_uploads(d, &mut pending)) {
                peregrine_io::note_advisory_err("gpu swap upload drain (final)", &e_s);
            }
            Ok(self.experts.len())
        }

        /// How many MTP-head experts the tier currently holds as a reservation
        /// (`COLI_MTP_PIN_VRAM_MB`). `0` when the knob is unset.
        pub fn pinned_count(&self) -> usize {
            self.pinned.len()
        }

        /// Whether expert `e` of `layer` is resident in VRAM.
        pub fn has(&self, layer: usize, e: usize) -> bool {
            self.experts.contains_key(&(layer, e))
        }

        /// Test-only handle on one resident's device tensors, so equivalence
        /// tests can dispatch an expert directly and compare against the
        /// grouped path.
        #[cfg(test)]
        pub fn expert_for_test(&self, layer: usize, e: usize) -> Option<&GpuExpert> {
            self.experts.get(&(layer, e))
        }

        /// Number of experts resident (for logging).
        pub fn len(&self) -> usize {
            self.experts.len()
        }

        /// Whether the tier holds no experts.
        pub fn is_empty(&self) -> bool {
            self.experts.is_empty()
        }

        /// The first device this tier was built across — where the dense-MLP
        /// tier and any single-device reporting live. Multi-device tiers use
        /// [`Self::devices`] / the owner map.
        pub fn device(&self) -> i32 {
            self.devices.first().copied().unwrap_or(0)
        }

        /// Every device the tier spans, in init order.
        pub fn devices(&self) -> &[i32] {
            &self.devices
        }

        /// Which device holds `(layer, expert)` — `None` when it is not resident.
        pub fn owner_of(&self, layer: usize, e: usize) -> Option<i32> {
            self.owner.get(&(layer, e)).copied()
        }

        /// The build-time planning budget of `device`.
        fn budget_of(&self, device: i32) -> usize {
            let idx = self.devices.iter().position(|&d| d == device).unwrap_or(0);
            self.budgets.get(idx).copied().unwrap_or(0)
        }

        /// Bytes `device`'s residents occupy right now (per its owner map).
        fn held_on(&self, device: i32) -> usize {
            self.experts
                .keys()
                .filter(|k| self.owner.get(*k) == Some(&device))
                .map(|k| {
                    if self.precision.get(k).copied().unwrap_or(self.int4) {
                        self.expert_bytes.0
                    } else {
                        self.expert_bytes.1
                    }
                })
                .sum()
        }

        /// Budget minus held for `device` — what an upload there may spend.
        fn free_on(&self, device: i32) -> usize {
            self.budget_of(device).saturating_sub(self.held_on(device))
        }

        /// Sum of every device's budget — what a residency *ranking* competes
        /// for. Per-device figures govern only where an upload lands.
        fn total_budget(&self) -> usize {
            self.budgets.iter().sum()
        }

        /// Compute a batch of this layer's GPU-resident experts. `jobs[k]` is
        /// `(expert index, gathered input rows [nr*hidden])`; returns each
        /// expert's SwiGLU output `[nr*hidden]`, in the same order.
        ///
        /// Jobs are grouped by **owner device** and the groups run in parallel —
        /// one host thread per device, each touching only its own card, which is
        /// what makes the concurrency safe (per-device contexts and a thread-local
        /// current-device cache on the C side). One device ⇒ one group ⇒ the
        /// historical single-call path with no threads at all. The results are
        /// scattered back into job order, so no timing-dependent ordering can
        /// reach `concurrent.rs`'s reduce: per-expert outputs are independent
        /// buffers, never summed across devices here.
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
            let groups = self.group_by_owner(layer, jobs);
            let mut parts: Vec<Result<Vec<Vec<f32>>, Error>> = Vec::with_capacity(groups.len());
            if groups.len() == 1 {
                parts.push(self.compute_group(layer, jobs, &groups[0].1, hidden));
            } else {
                std::thread::scope(|s| {
                    let handles: Vec<_> =
                        groups.iter().map(|(_, idxs)| s.spawn(move || self.compute_group(layer, jobs, idxs, hidden))).collect();
                    for h in handles {
                        match h.join() {
                            Ok(r) => parts.push(r),
                            // A worker panicked; resume its unwind rather than
                            // inventing an error code for it.
                            Err(e) => std::panic::resume_unwind(e),
                        }
                    }
                });
            }
            // Assemble in job order. Parts were collected in ascending-device
            // order, so even error selection is deterministic (first failing
            // device wins).
            let mut out = Vec::with_capacity(jobs.len());
            for _ in jobs {
                out.push(Vec::new());
            }
            for ((_, idxs), part) in groups.iter().zip(parts) {
                let part = part?;
                if part.len() != idxs.len() {
                    return Err(Error::Format("gpu compute: group returned wrong arity".into()));
                }
                for (&job_i, res) in idxs.iter().zip(part) {
                    out[job_i] = res;
                }
            }
            Ok(out)
        }

        /// `jobs` grouped by owning device as `(device, job indices)` pairs in
        /// ASCENDING device order — the order dispatch threads are spawned in and
        /// partials are folded in, so nothing downstream can observe completion
        /// order.
        fn group_by_owner(&self, layer: usize, jobs: &[(usize, Vec<f32>)]) -> Vec<(i32, Vec<usize>)> {
            let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
            for (i, (e, _)) in jobs.iter().enumerate() {
                groups.entry(self.owner_of(layer, *e).unwrap_or_else(|| self.device())).or_default().push(i);
            }
            groups.into_iter().collect()
        }

        /// One device's slice of [`Self::compute`]: format-partitioned dispatch
        /// over `idxs` (indices into `jobs`), returning outputs aligned to those
        /// indices.
        fn compute_group(
            &self,
            layer: usize,
            jobs: &[(usize, Vec<f32>)],
            idxs: &[usize],
            hidden: usize,
        ) -> Result<Vec<Vec<f32>>, Error> {
            if idxs.is_empty() {
                return Ok(Vec::new());
            }
            // One call per residency format: a single f32 resident in the group
            // would otherwise drop every expert in the call off the int4 fast
            // paths. A homogeneous group still makes exactly one call.
            let fmt: Vec<bool> = idxs
                .iter()
                .map(|&i| self.precision.get(&(layer, jobs[i].0)).copied().unwrap_or(self.int4))
                .collect();
            let classes = super::partition_by_format(&fmt);

            let mut per_class = Vec::with_capacity(classes.len());
            for cls in &classes {
                let mut refs = Vec::with_capacity(cls.len());
                let mut rows = Vec::with_capacity(cls.len());
                let mut x = Vec::new();
                for &i in cls {
                    let (e, xg) = &jobs[idxs[i]];
                    let ge = self
                        .experts
                        .get(&(layer, *e))
                        .ok_or_else(|| Error::Format(format!("gpu expert ({layer},{e}) not resident")))?;
                    refs.push(ge);
                    rows.push((xg.len() / hidden) as i32);
                    x.extend_from_slice(xg);
                }
                let y = self.dispatch_tuned(&refs, &rows, &x, hidden)?;
                let mut outs = Vec::with_capacity(cls.len());
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
            // Back to job order within the group: the caller scatters by group
            // index afterwards, so the permutation must not escape this function.
            super::scatter_by_index(idxs.len(), &classes, per_class)
                .ok_or_else(|| Error::Format("gpu compute: format partition is not a bijection".into()))
        }

        /// One `expert_group` dispatch, with the WMMA tile chosen by the online
        /// tuner when `COLI_CUDA_AUTOTUNE=1` and left at the default otherwise.
        ///
        /// **Timed with a wall clock, not `COLI_CUDA_PROFILE`'s events, and the
        /// reason is not convenience**: enabling profiling disables the graph
        /// cache (event records are not part of the work being replayed), so a
        /// tuner driven by kernel-ms would only ever measure the un-cached path
        /// and then pick a tile for the cached one. `expert_group` synchronizes
        /// before returning, so the wall clock is a real end-to-end measure; the
        /// H2D/D2H it also contains are tile-independent, so differences between
        /// tiles still attribute to the kernel.
        fn dispatch_tuned(
            &self,
            refs: &[&GpuExpert],
            rows: &[i32],
            x: &[f32],
            hidden: usize,
        ) -> Result<Vec<f32>, Error> {
            let Some(tuner) = self.tuner.as_ref() else {
                return peregrine_cuda::expert_group(refs, rows, x, hidden);
            };
            let shape = crate::wmma_tune::KernelShape {
                d: hidden as u32,
                i: self.inter as u32,
                count: rows.len().min(u16::MAX as usize) as u16,
                max_rows: rows.iter().copied().max().unwrap_or(0).clamp(0, u16::MAX as i32) as u16,
            };
            let tile = tuner.lock().select(shape);
            let t0 = std::time::Instant::now();
            let (y, arm) = peregrine_cuda::expert_group_tiled(refs, rows, x, hidden, tile.w4a16_dims())?;
            let us = t0.elapsed().as_micros() as f32;
            // Record under the arm that ACTUALLY ran, and only when the timing
            // means something about a tile.
            //
            // `COLI_CUDA_TC_W4A16` gates on compute capability and a row-count
            // floor, so a group that missed the floor silently ran the scalar
            // kernel — and crediting that time to the tile the tuner selected is
            // how a table fills with measurements of a kernel the tile never
            // touched. The int4 arm gets its one legal fragment recorded rather
            // than discarded: it is a real measurement at this shape, just not a
            // *choice*, and tagging it keeps it from sharing a row with the
            // fp16 numbers (where the faster arm would read as the faster tile).
            match arm {
                peregrine_cuda::GroupArm::W4A16 => tuner.lock().observe(shape, tile, us),
                peregrine_cuda::GroupArm::Int4Tc => {
                    tuner.lock().observe(shape, crate::wmma_tune::TileConfig::default_int4tc(), us)
                }
                // Tile-insensitive arms: timing them would be noise in the table.
                peregrine_cuda::GroupArm::PackedW4 | peregrine_cuda::GroupArm::Generic => {}
            }
            Ok(y)
        }

        /// [`Self::compute`] with the layer-level gate-weighted accumulation
        /// fused onto the device (`COLI_CUDA_FUSED_REDUCE`): returns one
        /// `[s_n, hidden]` partial instead of a per-expert output each.
        ///
        /// `dst[k]`/`weights[k]` describe job `k`'s rows in the same flattened
        /// order `compute` builds `x` in — job by job, rows within a job in the
        /// job's own order.
        ///
        /// **What changes numerically, stated rather than discovered.** The GPU
        /// experts now sum among themselves before meeting the CPU lane's
        /// contributions, instead of interleaving with them in batch-union
        /// order. Where a residency generation spans two formats,
        /// `partition_by_format` splits it again and the class partials are
        /// added in class order. And where it spans **devices**, each card
        /// produces its own partial and the fold runs in ascending device order —
        /// never completion order, which is the one way multi-GPU could make the
        /// low-order bits a function of which card finished first (the exact
        /// defect `docs/scale-out-design.md` §1 warns about). All three orders
        /// are fixed, so repeats are bit-stable; none is the host reduce's order,
        /// which is why this is a knob.
        pub fn compute_reduced(
            &self,
            layer: usize,
            jobs: &[(usize, Vec<f32>)],
            hidden: usize,
            dst: &[usize],
            weights: &[f32],
            s_n: usize,
        ) -> Result<Vec<f32>, Error> {
            let mut acc = vec![0f32; s_n * hidden];
            if jobs.is_empty() {
                return Ok(acc);
            }
            if hidden == 0 || s_n == 0 {
                return Err(Error::Format("gpu compute_reduced: empty hidden or batch".into()));
            }
            let total: usize = jobs.iter().map(|(_, xg)| xg.len() / hidden.max(1)).sum();
            if dst.len() != total || weights.len() != total {
                return Err(Error::Format("gpu compute_reduced: dst/weights length != total rows".into()));
            }
            for (e, xg) in jobs {
                if !self.experts.contains_key(&(layer, *e)) {
                    return Err(Error::Format(format!("gpu expert ({layer},{e}) not resident")));
                }
                if !xg.len().is_multiple_of(hidden) {
                    return Err(Error::Format("gpu compute_reduced: ragged gathered rows".into()));
                }
            }
            // Where each job's rows start in the flattened `dst`/`weights`, so a
            // format class can slice out exactly its own.
            let mut job_at = Vec::with_capacity(jobs.len() + 1);
            let mut running = 0usize;
            for (_, xg) in jobs {
                job_at.push(running);
                running += xg.len() / hidden;
            }
            job_at.push(running);

            let groups = self.group_by_owner(layer, jobs);
            let mut parts: Vec<Result<Vec<f32>, Error>> = Vec::with_capacity(groups.len());
            if groups.len() == 1 {
                parts.push(self.reduce_group(layer, jobs, &groups[0].1, hidden, dst, weights, &job_at, s_n));
            } else {
                std::thread::scope(|s| {
                    let handles: Vec<_> = groups
                        .iter()
                        .map(|(_, idxs)| {
                            let job_at = job_at.clone();
                            s.spawn(move || self.reduce_group(layer, jobs, idxs, hidden, dst, weights, &job_at, s_n))
                        })
                        .collect();
                    for h in handles {
                        match h.join() {
                            Ok(r) => parts.push(r),
                            Err(e) => std::panic::resume_unwind(e),
                        }
                    }
                });
            }
            // Fold in ASCENDING DEVICE order (`groups` is sorted by device), not
            // in whichever order the cards finished. See the doc above.
            for part in parts {
                let part = part?;
                if part.len() != acc.len() {
                    return Err(Error::Format("gpu compute_reduced: short partial".into()));
                }
                for (a, p) in acc.iter_mut().zip(&part) {
                    *a += p;
                }
            }
            Ok(acc)
        }

        /// One device's slice of [`Self::compute_reduced`]: format-class partials
        /// summed within the group, returning that group's `[s_n, hidden]` share.
        #[allow(clippy::too_many_arguments)]
        fn reduce_group(
            &self,
            layer: usize,
            jobs: &[(usize, Vec<f32>)],
            idxs: &[usize],
            hidden: usize,
            dst: &[usize],
            weights: &[f32],
            job_at: &[usize],
            s_n: usize,
        ) -> Result<Vec<f32>, Error> {
            let mut acc = vec![0f32; s_n * hidden];
            let fmt: Vec<bool> =
                idxs.iter().map(|&i| self.precision.get(&(layer, jobs[i].0)).copied().unwrap_or(self.int4)).collect();
            for cls in &super::partition_by_format(&fmt) {
                let mut refs = Vec::with_capacity(cls.len());
                let mut rows = Vec::with_capacity(cls.len());
                let mut x = Vec::new();
                let mut cdst = Vec::new();
                let mut crw = Vec::new();
                for &i in cls {
                    let job_i = idxs[i];
                    let (e, xg) = &jobs[job_i];
                    let ge = self
                        .experts
                        .get(&(layer, *e))
                        .ok_or_else(|| Error::Format(format!("gpu expert ({layer},{e}) not resident")))?;
                    refs.push(ge);
                    rows.push((xg.len() / hidden) as i32);
                    x.extend_from_slice(xg);
                    cdst.extend_from_slice(&dst[job_at[job_i]..job_at[job_i + 1]]);
                    crw.extend_from_slice(&weights[job_at[job_i]..job_at[job_i + 1]]);
                }
                let layout = peregrine_cuda::ReduceLayout::build(&cdst, s_n)
                    .ok_or_else(|| Error::Format("gpu compute_reduced: row destination out of range".into()))?;
                let part = peregrine_cuda::expert_group_reduce(&refs, &rows, &x, hidden, &layout, &crw, s_n)?;
                if part.len() != acc.len() {
                    return Err(Error::Format("gpu compute_reduced: short partial".into()));
                }
                for (a, p) in acc.iter_mut().zip(&part) {
                    *a += p;
                }
            }
            Ok(acc)
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

    /// Empty dense tier for non-`cuda` builds: never constructed, always says
    /// "not resident", so every caller takes the CPU MLP path unchanged.
    pub struct GpuDenseTier {
        _never: (),
    }

    impl GpuDenseTier {
        /// Never actually constructed in a non-`cuda` build: `Model` gates the
        /// whole tier on `COLI_GPU_DENSE`, and `try_add` here never accepts a
        /// layer, so the tier is always dropped as empty. The constructor
        /// exists so the caller needs no `cfg` of its own.
        pub fn new(_device: i32) -> GpuDenseTier {
            GpuDenseTier { _never: () }
        }

        pub fn try_add(
            &mut self,
            _li: usize,
            _gate: &crate::weight::QtWeight,
            _up: &crate::weight::QtWeight,
            _down: &crate::weight::QtWeight,
            _headroom: usize,
        ) -> Result<bool, Error> {
            Ok(false)
        }

        pub fn has(&self, _li: usize) -> bool {
            false
        }

        pub fn mlp(
            &self,
            _li: usize,
            _x: &[f32],
            _s_n: usize,
            _hidden: usize,
        ) -> Option<Result<Vec<f32>, Error>> {
            None
        }

        pub fn stats(&self) -> (usize, usize, usize) {
            (0, 0, 0)
        }
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
        pub fn pinned_count(&self) -> usize {
            0
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
        pub fn compute_reduced(
            &self,
            _layer: usize,
            _jobs: &[(usize, Vec<f32>)],
            _hidden: usize,
            _dst: &[usize],
            _weights: &[f32],
            _s_n: usize,
        ) -> Result<Vec<f32>, Error> {
            Err(Error::Format("gpu tier not built (no cuda feature)".into()))
        }
        pub fn reheat(
            &mut self,
            _st: &SafeTensors,
            _cfg: &Cfg,
            _heat: &super::HeatView,
            _pins: &super::PinRequest,
        ) -> Result<usize, Error> {
            Ok(0)
        }
        pub fn tuning_json(&self) -> Option<serde_json::Value> {
            None
        }
        pub fn restore_tuning(&self, _v: &serde_json::Value) {}
    }
}

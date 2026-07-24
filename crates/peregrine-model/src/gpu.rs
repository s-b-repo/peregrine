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
        if let Some(c) = self.counts.get(layer * self.n_experts + expert) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A plain snapshot of the counts, row-major `[layer * n_experts + expert]`.
    pub fn snapshot(&self) -> Vec<u32> {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }
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
    /// per-row int4 (8× denser) vs dequantized f32.
    pub struct GpuTier {
        device: i32,
        experts: HashMap<(usize, usize), GpuExpert>,
        capacity: usize,
        int4: bool,
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
            let bytes_per_expert = if int4 { weights / 2 + (2 * inter + hidden) * 4 } else { weights * 4 };
            let budget = free.saturating_sub(headroom_bytes);

            let placement = super::plan_residency(
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                bytes_per_expert,
                budget,
            );
            let capacity = placement.len(); // fixed VRAM budget in experts, for reheat
            let mut experts = HashMap::new();
            for (layer, e) in placement {
                experts.insert((layer, e), upload_expert(st, cfg, layer, e, device, int4)?);
            }

            if experts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(GpuTier { device, experts, capacity, int4 }))
            }
        }

        /// Re-select the VRAM-resident set as the `capacity` hottest experts by
        /// `counts` (routing frequency), evicting experts that cooled and uploading
        /// newly-hot ones. Reuses [`Self::build`]'s dequantize+upload path. Called
        /// between forwards with `&mut self`, so residency adapts to the workload
        /// without a rewrite. Returns the resident count after re-selection.
        pub fn reheat(&mut self, st: &SafeTensors, cfg: &Cfg, counts: &[u32]) -> Result<usize, Error> {
            let want = super::rank_by_heat(
                counts,
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                self.capacity,
            );
            let want_set: HashSet<(usize, usize)> = want.iter().copied().collect();
            // evict experts that cooled off — their `Drop` frees the VRAM slot
            self.experts.retain(|k, _| want_set.contains(k));
            // upload newly-hot experts not already resident (same format as build)
            for (layer, e) in want {
                if self.experts.contains_key(&(layer, e)) {
                    continue;
                }
                let ge = upload_expert(st, cfg, layer, e, self.device, self.int4)?;
                self.experts.insert((layer, e), ge);
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

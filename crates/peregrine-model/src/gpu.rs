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

#[cfg(all(test, feature = "cuda"))]
mod gpu_residency_tests {
    use super::real::GpuTier;
    use crate::testkit::build_tiny_model;
    use peregrine_core::{Cfg, Error, SafeTensors};

    #[test]
    fn tier_spans_multiple_sparse_layers() -> Result<(), Error> {
        // On a real GPU, the round-robin placement must put residents in BOTH of
        // the tiny model's sparse layers (1 and 2) — the old greedy fill would have
        // packed layer 1 first. Skips gracefully when no GPU is available.
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
}

#[cfg(feature = "cuda")]
mod real {
    use std::collections::HashMap;

    use peregrine_core::{Cfg, Error, SafeTensors};
    use peregrine_cuda::GpuExpert;

    use crate::weight::QtWeight;

    /// VRAM-resident f32 experts, keyed by `(layer, expert index)`.
    pub struct GpuTier {
        device: i32,
        experts: HashMap<(usize, usize), GpuExpert>,
    }

    impl GpuTier {
        /// Build the tier by dequantizing routed experts to f32 and uploading as
        /// many as fit within `free VRAM - headroom`, iterating sparse layers and
        /// experts in order. `Ok(None)` when CUDA is unavailable or nothing fits.
        pub fn build(st: &SafeTensors, cfg: &Cfg, headroom_bytes: usize) -> Result<Option<GpuTier>, Error> {
            if peregrine_cuda::init(&[0]) < 1 {
                return Ok(None);
            }
            let device = 0;
            let (free, _total) = peregrine_cuda::mem_info(device)?;
            let hidden = cfg.hidden as usize;
            let inter = cfg.moe_inter as usize;
            // gate + up ([inter,hidden]) + down ([hidden,inter]), all f32
            let bytes_per_expert = (2 * inter * hidden + hidden * inter) * 4;
            let budget = free.saturating_sub(headroom_bytes);

            // Spread residency across ALL sparse layers (round-robin) instead of
            // greedily filling the first layers — so every layer gets a VRAM share.
            let placement = super::plan_residency(
                cfg.n_layers as usize,
                cfg.first_dense as usize,
                cfg.n_experts as usize,
                bytes_per_expert,
                budget,
            );
            let mut experts = HashMap::new();
            for (layer, e) in placement {
                let pe = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
                let gate = QtWeight::load(st, &pe("gate_proj.weight"), inter, hidden)?.dequant();
                let up = QtWeight::load(st, &pe("up_proj.weight"), inter, hidden)?.dequant();
                let down = QtWeight::load(st, &pe("down_proj.weight"), hidden, inter)?.dequant();
                let ge = GpuExpert::upload(device, &gate, &up, &down, hidden, inter)?;
                experts.insert((layer, e), ge);
            }

            if experts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(GpuTier { device, experts }))
            }
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
    }
}

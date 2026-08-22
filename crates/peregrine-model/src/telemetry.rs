//! Runtime telemetry snapshot & plan optimizer.
//!
//! Assembles per-forward observables from the other adaptive tuners
//! ([`crate::iotune::IoTuner`], [`crate::lane::BubbleTuner`], warm-cache
//! counters, prefetch tuner) into a single [`RuntimeTelemetry`] readable from
//! the `/metrics` endpoint. [`PlanOptimizer`] is the between-forward tick that
//! nudges the tunable knobs from measurements.

use crate::iotune::{IoTuner, IowqCap};
use crate::lane::{Bias, BubbleTuner, LaneTimings};

/// One snapshot of runtime state, safe to expose on `/metrics`.
#[derive(Clone, Debug, Default)]
pub struct RuntimeTelemetry {
    pub lane: LaneTimings,
    pub bias: Bias,
    pub io_ewma_us: u64,
    pub io_sq_full: u64,
    pub iowq: Option<IowqCap>,
    /// Prefetch effectiveness ratio (used / (used + wasted)); `None` when the
    /// warm cache is off.
    pub prefetch_accuracy: Option<f32>,
    /// Warm cache hit rate; `None` when the warm cache is off.
    pub cache_hit_rate: Option<f32>,
    /// Cumulative GPU expert-group transfer counters. All zero on a build
    /// without the `cuda` feature, and the millisecond fields are zero unless
    /// `COLI_CUDA_PROFILE` is also set (the CUDA event timing is opt-in because
    /// it costs four events and a sync per call).
    pub gpu: GpuTransferStats,
    /// EWMA of the normalized Shannon entropy of the routed distribution:
    /// ~0 when routing is repetitive (a few experts dominate), ~1 when it is
    /// dispersed. Drives `COLI_ENTROPY_ADAPT`'s prefetch-breadth decision.
    ///
    /// Added here because `Model::routing_entropy_ewma` documented itself "for
    /// telemetry scrapes" while `RuntimeTelemetry` — the actual telemetry
    /// snapshot — had no field for it, so the value was computed every forward
    /// and reachable by nothing. The underlying EWMA was always live; only the
    /// way out was missing.
    pub entropy_ewma: f32,
    /// MTP-head experts currently pinned in the warm cache (`COLI_MTP_PIN_MB`)
    /// and in VRAM (`COLI_MTP_PIN_VRAM_MB`). Both `0` when the knobs are unset,
    /// and both `0` before the first refresh generation.
    ///
    /// Reported because a pin set that never forms is indistinguishable from an
    /// inert mechanism otherwise: the layer still computes, still returns the
    /// same tokens, and still prints the same boot line — it is only slower, and
    /// only against a baseline nobody has.
    pub mtp_pinned_cache: usize,
    pub mtp_pinned_vram: usize,
}

/// PCIe-side observables for the GPU expert lane, mirrored out of
/// `peregrine_cuda::GroupStats` so this module stays compilable without the
/// optional `cuda` dependency.
///
/// These counters existed but were read by nothing outside one unit test —
/// the same "measured but never reported" gap the warm cache's compression
/// counters had. A transfer-bound GPU lane is invisible without them.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTransferStats {
    pub calls: u64,
    pub experts: u64,
    pub rows: u64,
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    /// Expert-group graph cache (`COLI_CUDA_GRAPH=1`), mirrored the same way.
    ///
    /// Reported because the cache's failure mode is silent: a launch shape that
    /// churns captures on every call and is *slower* than the eager path, with
    /// every output still correct. `replays` near zero while `captures` tracks
    /// `calls` is what that looks like from outside, and nothing else shows it.
    pub graph_captures: u64,
    pub graph_replays: u64,
    pub graph_invalidations: u64,
    pub graph_uncacheable: u64,
}

impl GpuTransferStats {
    /// Read the process-global counters. Zeros without the `cuda` feature.
    pub fn read() -> GpuTransferStats {
        #[cfg(feature = "cuda")]
        {
            let g = peregrine_cuda::group_stats();
            let gc = peregrine_cuda::graph_cache_stats();
            GpuTransferStats {
                calls: g.calls,
                experts: g.experts,
                rows: g.rows,
                h2d_ms: g.h2d_ms,
                kernel_ms: g.kernel_ms,
                d2h_ms: g.d2h_ms,
                graph_captures: gc.captures,
                graph_replays: gc.replays,
                graph_invalidations: gc.invalidations,
                graph_uncacheable: gc.uncacheable,
            }
        }
        #[cfg(not(feature = "cuda"))]
        GpuTransferStats::default()
    }

    /// Share of measured GPU-lane time spent moving bytes rather than computing.
    /// `None` until `COLI_CUDA_PROFILE` has produced timings. A high value is the
    /// signal a PCIe budget (`COLI_PCIE_BUDGET_MB`) is meant to act on.
    pub fn transfer_fraction(&self) -> Option<f64> {
        let total = self.h2d_ms + self.kernel_ms + self.d2h_ms;
        if total <= 0.0 {
            return None;
        }
        Some((self.h2d_ms + self.d2h_ms) / total)
    }
}

/// Coordinator that reads all the tuners each forward and nudges the knobs.
/// Owns nothing — just references + a tiny bit of derived state. Designed to be
/// called from the batch engine between decode ticks.
pub struct PlanOptimizer {
    ticks: u64,
    /// How often (in forwards) to update the IoTuner's iowq recommendation.
    /// Cheap enough to run every forward, but this gives space to accumulate an
    /// EWMA sample before adjusting.
    io_tune_period: u64,
    last_sq_full: u64,
    /// Whether `last_sq_full` has been initialized from the live counter.
    seeded: bool,
}

impl PlanOptimizer {
    pub fn new() -> PlanOptimizer {
        // `last_sq_full` is seeded on the first tick, not at 0: starting from 0
        // would make the first delta the *cumulative* count since process start
        // (including model load and warm-up) and halve both caps on the very
        // first adjustment.
        PlanOptimizer { ticks: 0, io_tune_period: 16, last_sq_full: 0, seeded: false }
    }

    /// Called once per forward: folds `lane` into the bubble tuner and steps the
    /// I/O tuner on its period. Mutating — [`Self::snapshot`] is the read-only
    /// view for a `/metrics` scrape, which must not double-feed the EWMA.
    ///
    /// `cache` carries the warm-cache observables (`(hits, misses, prefetch_used,
    /// prefetch_wasted)`) when a warm cache exists, so the returned telemetry
    /// reports real hit/accuracy rates instead of a hardcoded `None`.
    pub fn tick(
        &mut self,
        bubble: &mut BubbleTuner,
        io: &IoTuner,
        lane: LaneTimings,
        latency_target_us: u64,
        cache: Option<CacheCounters>,
    ) -> RuntimeTelemetry {
        self.ticks = self.ticks.saturating_add(1);
        let bias = bubble.observe(lane);
        if !self.seeded {
            self.last_sq_full = io.sq_full();
            self.seeded = true;
        }
        if self.ticks.is_multiple_of(self.io_tune_period) {
            io.step(latency_target_us, self.last_sq_full);
            self.last_sq_full = io.sq_full();
        }
        self.snapshot(bias, io, lane, cache)
    }

    /// A read-only telemetry view — safe to call from a `/metrics` handler.
    pub fn snapshot(
        &self,
        bias: Bias,
        io: &IoTuner,
        lane: LaneTimings,
        cache: Option<CacheCounters>,
    ) -> RuntimeTelemetry {
        let ratio = |num: u64, den: u64| if den == 0 { None } else { Some(num as f32 / den as f32) };
        RuntimeTelemetry {
            lane,
            bias,
            io_ewma_us: io.ewma_us(),
            io_sq_full: io.sq_full(),
            iowq: io.recommend(),
            prefetch_accuracy: cache.and_then(|c| ratio(c.prefetch_used, c.prefetch_used + c.prefetch_wasted)),
            cache_hit_rate: cache.and_then(|c| ratio(c.hits, c.hits + c.misses)),
            gpu: GpuTransferStats::read(),
            // The optimizer has no view of routing; `Model::publish_lane_timings`
            // fills this from `routing_entropy_ewma()` after the snapshot.
            entropy_ewma: 0.0,
            // Likewise: residency is the model's, not the optimizer's.
            mtp_pinned_cache: 0,
            mtp_pinned_vram: 0,
        }
    }
}

/// Warm-cache observables the optimizer folds into a telemetry snapshot.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheCounters {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
}

impl Default for PlanOptimizer {
    fn default() -> PlanOptimizer {
        PlanOptimizer::new()
    }
}

/// Open an LLC-miss hardware counter for the calling thread, gated on
/// `COLI_PERF_COUNTERS=1` (and the kernel granting `perf_event_open` —
/// `perf_event_paranoid <= 2`, a real PMU, no seccomp filter). `None`
/// otherwise.
///
/// The caller hands the result to [`crate::Model::attach_perf_counter`], which
/// reports the total at shutdown and — only under the separate
/// `COLI_PERF_PREFETCH_FEEDBACK=1` — feeds per-forward deltas to the prefetch
/// tuner: rising misses widen the distance.
///
/// **This sentence used to describe that consumer in the present tense while it
/// did not exist**, which is the `[R]` failure mode in `docs/BAD_PATTERNS.md`
/// wearing a doc comment. It exists now; what is still unestablished is whether
/// the *direction* is right, since the counter follows the decode thread and
/// therefore misses the io_uring and `peregrine-par` workers that actually move
/// expert bytes. `Model::llc_trend` carries the argument and the test.
pub fn open_l3_miss_counter() -> Option<peregrine_io::PerfCounter> {
    if !matches!(std::env::var("COLI_PERF_COUNTERS").as_deref(), Ok("1") | Ok("true")) {
        return None;
    }
    peregrine_io::PerfCounter::open_cache_misses()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_fraction_is_none_until_profiling_produces_timings() {
        // The counters tick without COLI_CUDA_PROFILE, but the millisecond fields
        // stay zero — reporting 0% transfer then would read as "compute-bound"
        // when the truth is "not measured".
        let counted = GpuTransferStats { calls: 12, experts: 300, rows: 900, ..Default::default() };
        assert!(counted.transfer_fraction().is_none());
        assert!(GpuTransferStats::default().transfer_fraction().is_none());
    }

    #[test]
    fn transfer_fraction_splits_movement_from_compute() {
        let g = GpuTransferStats { h2d_ms: 3.0, kernel_ms: 2.0, d2h_ms: 5.0, ..Default::default() };
        // 8 ms moving bytes out of 10 ms total → PCIe-bound.
        assert_eq!(g.transfer_fraction(), Some(0.8));
        let compute_bound = GpuTransferStats { h2d_ms: 0.0, kernel_ms: 4.0, d2h_ms: 0.0, ..Default::default() };
        assert_eq!(compute_bound.transfer_fraction(), Some(0.0));
    }

    #[test]
    fn snapshot_carries_gpu_stats() {
        // Zeros on a CPU-only build, but the field must be populated rather than
        // silently absent — that is the gap this closes.
        let opt = PlanOptimizer::new();
        let io = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 1, 16);
        let t = opt.snapshot(Bias::Balanced, &io, LaneTimings::default(), None);
        assert_eq!(t.gpu.calls, 0);
        assert!(t.gpu.transfer_fraction().is_none());
    }

    #[test]
    fn optimizer_returns_telemetry_each_tick() {
        let mut b = BubbleTuner::new(0.3, 1.5, 3);
        let io = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 1, 16);
        let mut o = PlanOptimizer::new();
        let t = o.tick(&mut b, &io, LaneTimings { io_us: 100, cpu_us: 50, gpu_us: 50, reduce_us: 5, cpu_bytes: 0, lane_wall_us: 0, cache_wait_us: 0 }, 1000, None);
        assert_eq!(t.lane.io_us, 100);
    }

    #[test]
    fn cache_counters_populate_the_snapshot() {
        // These two fields were hardcoded `None`, so a scrape always read
        // "warm cache off" no matter what the cache was doing.
        let b = BubbleTuner::new(0.3, 1.5, 3);
        let io = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 1, 16);
        let o = PlanOptimizer::new();
        let c = CacheCounters { hits: 3, misses: 1, prefetch_used: 1, prefetch_wasted: 3 };
        let t = o.snapshot(b.bias(), &io, LaneTimings::default(), Some(c));
        assert_eq!(t.cache_hit_rate, Some(0.75));
        assert_eq!(t.prefetch_accuracy, Some(0.25));
        // No cache → the documented `None`.
        let t2 = o.snapshot(b.bias(), &io, LaneTimings::default(), None);
        assert_eq!(t2.cache_hit_rate, None);
    }

    #[test]
    fn first_tick_does_not_charge_startup_sq_full_to_one_forward() {
        // `last_sq_full` starting at 0 made the first delta the whole cumulative
        // count since process start, which reads as sudden queue pressure.
        let mut b = BubbleTuner::new(0.3, 1.5, 3);
        let io = IoTuner::new(IowqCap { bounded: 8, unbounded: 8 }, 1, 16);
        io.note_read(100, 50); // 50 rejections accumulated during warm-up
        let mut o = PlanOptimizer::new();
        for _ in 0..16 {
            o.tick(&mut b, &io, LaneTimings { io_us: 100, cpu_us: 1, gpu_us: 1, reduce_us: 0, cpu_bytes: 0, lane_wall_us: 0, cache_wait_us: 0 }, 1000, None);
        }
        if let Some(rec) = io.recommend() {
            assert_eq!(rec.bounded, 8, "warm-up rejections must not halve the cap on the first adjustment");
        }
    }
}

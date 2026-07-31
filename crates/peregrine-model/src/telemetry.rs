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
/// otherwise. Consumers feed [`peregrine_io::PerfCounter::read`] deltas into
/// the prefetch tuner: rising misses → widen prefetch distance.
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
    fn optimizer_returns_telemetry_each_tick() {
        let mut b = BubbleTuner::new(0.3, 1.5, 3);
        let io = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 1, 16);
        let mut o = PlanOptimizer::new();
        let t = o.tick(&mut b, &io, LaneTimings { io_us: 100, cpu_us: 50, gpu_us: 50, reduce_us: 5, cpu_bytes: 0 }, 1000, None);
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
            o.tick(&mut b, &io, LaneTimings { io_us: 100, cpu_us: 1, gpu_us: 1, reduce_us: 0, cpu_bytes: 0 }, 1000, None);
        }
        if let Some(rec) = io.recommend() {
            assert_eq!(rec.bounded, 8, "warm-up rejections must not halve the cap on the first adjustment");
        }
    }
}

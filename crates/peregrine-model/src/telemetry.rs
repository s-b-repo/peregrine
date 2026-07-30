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
}

impl PlanOptimizer {
    pub fn new() -> PlanOptimizer {
        PlanOptimizer { ticks: 0, io_tune_period: 16, last_sq_full: 0 }
    }

    /// Called once per forward. Returns the updated telemetry the caller can
    /// scrape for `/metrics`. `latency_target_us` is the SLA the IoTuner uses
    /// to decide whether to grow workers.
    pub fn tick(
        &mut self,
        bubble: &mut BubbleTuner,
        io: &IoTuner,
        lane: LaneTimings,
        latency_target_us: u64,
    ) -> RuntimeTelemetry {
        self.ticks = self.ticks.saturating_add(1);
        let bias = bubble.observe(lane);
        if self.ticks.is_multiple_of(self.io_tune_period) {
            io.step(latency_target_us, self.last_sq_full);
            self.last_sq_full = io.sq_full();
        }
        RuntimeTelemetry {
            lane,
            bias,
            io_ewma_us: io.ewma_us(),
            io_sq_full: io.sq_full(),
            iowq: io.recommend(),
            prefetch_accuracy: None,
            cache_hit_rate: None,
        }
    }
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
        let t = o.tick(&mut b, &io, LaneTimings { io_us: 100, cpu_us: 50, gpu_us: 50, reduce_us: 5, cpu_bytes: 0 }, 1000);
        assert_eq!(t.lane.io_us, 100);
    }
}

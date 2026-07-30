//! Per-lane telemetry and adaptive scheduling substrate.
//!
//! The 3-lane MoE executor ([`crate::concurrent::moe_forward_concurrent`]) runs
//! I/O, CPU compute, and GPU compute concurrently. This module tracks per-lane
//! wall time and exposes it as [`LaneTimings`]; the [`BubbleTuner`] maintains an
//! EWMA and reports a bias signal when one lane starts dominating (a pipeline
//! bubble). [`LaneBalancer`] turns that signal into a per-expert placement
//! recommendation.
//!
//! Everything here is correctness-neutral — timings only reorder or offload
//! work; per-expert numerics are unchanged.

use std::sync::atomic::{AtomicU64, Ordering};

/// One forward's per-lane wall time, microseconds. Populated by the executor
/// after each layer's reduce and reset between forwards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneTimings {
    pub io_us: u64,
    pub cpu_us: u64,
    pub gpu_us: u64,
    pub reduce_us: u64,
}

/// Atomic accumulator so several scoped threads can bump without a lock. The
/// executor calls [`Self::add_io`] / `add_cpu` / `add_gpu` / `add_reduce` and,
/// at the end of the forward, [`Self::snapshot_and_reset`] to hand the summed
/// timings to the tuner.
pub struct LaneTimingsAccum {
    io_us: AtomicU64,
    cpu_us: AtomicU64,
    gpu_us: AtomicU64,
    reduce_us: AtomicU64,
}

impl Default for LaneTimingsAccum {
    fn default() -> LaneTimingsAccum {
        LaneTimingsAccum {
            io_us: AtomicU64::new(0),
            cpu_us: AtomicU64::new(0),
            gpu_us: AtomicU64::new(0),
            reduce_us: AtomicU64::new(0),
        }
    }
}

impl LaneTimingsAccum {
    pub fn new() -> LaneTimingsAccum {
        LaneTimingsAccum::default()
    }

    pub fn add_io(&self, us: u64) {
        self.io_us.fetch_add(us, Ordering::Relaxed);
    }

    pub fn add_cpu(&self, us: u64) {
        self.cpu_us.fetch_add(us, Ordering::Relaxed);
    }

    pub fn add_gpu(&self, us: u64) {
        self.gpu_us.fetch_add(us, Ordering::Relaxed);
    }

    pub fn add_reduce(&self, us: u64) {
        self.reduce_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Take the current sums and reset the accumulator. Called once per forward.
    pub fn snapshot_and_reset(&self) -> LaneTimings {
        LaneTimings {
            io_us: self.io_us.swap(0, Ordering::Relaxed),
            cpu_us: self.cpu_us.swap(0, Ordering::Relaxed),
            gpu_us: self.gpu_us.swap(0, Ordering::Relaxed),
            reduce_us: self.reduce_us.swap(0, Ordering::Relaxed),
        }
    }

    /// Non-destructive peek. Useful for `/metrics` scrapes.
    pub fn snapshot(&self) -> LaneTimings {
        LaneTimings {
            io_us: self.io_us.load(Ordering::Relaxed),
            cpu_us: self.cpu_us.load(Ordering::Relaxed),
            gpu_us: self.gpu_us.load(Ordering::Relaxed),
            reduce_us: self.reduce_us.load(Ordering::Relaxed),
        }
    }
}

/// Which lane is currently the pipeline bottleneck. `None` when balanced within
/// the tuner's tolerance (default: no lane exceeds `1.5 × max_of_others`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bias {
    #[default]
    Balanced,
    TowardIo,
    TowardCpu,
    TowardGpu,
}

/// EWMA + hysteresis tuner over [`LaneTimings`]. Reports a [`Bias`] once the
/// slow lane has consistently dominated for `k_consecutive` forwards. Purely
/// advisory; consumers decide whether to act on the signal.
pub struct BubbleTuner {
    ewma_io: f32,
    ewma_cpu: f32,
    ewma_gpu: f32,
    alpha: f32,
    dominance: f32,
    k_consecutive: u32,
    // Which lane has dominated recently, and for how many consecutive forwards.
    streak_kind: Bias,
    streak: u32,
    current: Bias,
}

impl BubbleTuner {
    /// `alpha`: EWMA weight (0.0–1.0, higher = more reactive).
    /// `dominance`: multiplier by which the top lane must exceed the max of the
    /// other two to be considered dominant (e.g. 1.5).
    /// `k_consecutive`: forwards a dominance must persist before the tuner
    /// publishes the bias (hysteresis against one-off spikes).
    pub fn new(alpha: f32, dominance: f32, k_consecutive: u32) -> BubbleTuner {
        BubbleTuner {
            ewma_io: 0.0,
            ewma_cpu: 0.0,
            ewma_gpu: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            dominance: dominance.max(1.0),
            k_consecutive: k_consecutive.max(1),
            streak_kind: Bias::Balanced,
            streak: 0,
            current: Bias::Balanced,
        }
    }

    /// Fold a new forward's timings into the EWMA and update the bias.
    pub fn observe(&mut self, t: LaneTimings) -> Bias {
        let a = self.alpha;
        self.ewma_io = (1.0 - a) * self.ewma_io + a * t.io_us as f32;
        self.ewma_cpu = (1.0 - a) * self.ewma_cpu + a * t.cpu_us as f32;
        self.ewma_gpu = (1.0 - a) * self.ewma_gpu + a * t.gpu_us as f32;
        let (io, cpu, gpu) = (self.ewma_io, self.ewma_cpu, self.ewma_gpu);
        // pick the max lane and the max-of-others
        let (mx, others) = if io >= cpu && io >= gpu {
            (io, cpu.max(gpu))
        } else if cpu >= gpu {
            (cpu, io.max(gpu))
        } else {
            (gpu, io.max(cpu))
        };
        let dominant = if others > 0.0 && mx / others >= self.dominance {
            if mx == io {
                Bias::TowardIo
            } else if mx == cpu {
                Bias::TowardCpu
            } else {
                Bias::TowardGpu
            }
        } else {
            Bias::Balanced
        };
        // hysteresis: publish the dominance only after k consecutive forwards agree
        if dominant == self.streak_kind {
            self.streak = self.streak.saturating_add(1);
        } else {
            self.streak_kind = dominant;
            self.streak = 1;
        }
        if self.streak >= self.k_consecutive {
            self.current = self.streak_kind;
        }
        self.current
    }

    /// Last published bias.
    pub fn bias(&self) -> Bias {
        self.current
    }

    /// Current EWMA snapshot (used for /metrics).
    pub fn ewma_snapshot(&self) -> LaneTimings {
        LaneTimings {
            io_us: self.ewma_io as u64,
            cpu_us: self.ewma_cpu as u64,
            gpu_us: self.ewma_gpu as u64,
            reduce_us: 0,
        }
    }
}

/// Per-expert placement recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Prefer the CPU + streaming lane.
    Cpu,
    /// Prefer the GPU lane (already resident, no upload needed).
    Gpu,
    /// Prefer the GPU lane with an on-demand upload for this forward (spill).
    GpuSpill,
}

/// Turn a bias signal into per-expert placement. Simple policy:
/// - CPU-bound + expert is hot enough → upgrade to a GPU spill (frees the CPU
///   lane for cold experts).
/// - GPU-bound + expert is cold → downgrade a resident expert to stream.
/// - Otherwise: honor the residency (Gpu if resident, Cpu if streamed).
pub struct LaneBalancer {
    bias: Bias,
    /// Minimum heat score for a non-resident expert to be worth spilling to GPU.
    spill_threshold: u32,
}

impl LaneBalancer {
    pub fn new(bias: Bias, spill_threshold: u32) -> LaneBalancer {
        LaneBalancer { bias, spill_threshold }
    }

    /// Decide where to compute `expert` given whether it is GPU-resident and its
    /// routing heat. Correctness-neutral: the numerics on either lane are already
    /// bit-identical; only *where* the compute lands changes.
    pub fn choose(&self, gpu_resident: bool, heat: u32) -> Placement {
        match self.bias {
            Bias::TowardCpu if !gpu_resident && heat >= self.spill_threshold => Placement::GpuSpill,
            Bias::TowardGpu if gpu_resident => Placement::Cpu, // spill a resident back to CPU
            _ => {
                if gpu_resident {
                    Placement::Gpu
                } else {
                    Placement::Cpu
                }
            }
        }
    }
}

/// Whether adaptive CPU/GPU balance is enabled (`COLI_LANE_BALANCE=1`).
pub fn lane_balance_enabled() -> bool {
    matches!(std::env::var("COLI_LANE_BALANCE").as_deref(), Ok("1") | Ok("true"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accum_swap_resets_to_zero() {
        let a = LaneTimingsAccum::new();
        a.add_io(1000);
        a.add_cpu(500);
        a.add_gpu(300);
        a.add_reduce(50);
        let s = a.snapshot_and_reset();
        assert_eq!(s, LaneTimings { io_us: 1000, cpu_us: 500, gpu_us: 300, reduce_us: 50 });
        let z = a.snapshot();
        assert_eq!(z, LaneTimings::default(), "reset zeros the accumulator");
    }

    #[test]
    fn tuner_reports_cpu_bias_after_k_consecutive() {
        let mut t = BubbleTuner::new(0.5, 1.5, 3);
        for _ in 0..4 {
            let _ = t.observe(LaneTimings { io_us: 200, cpu_us: 1000, gpu_us: 100, reduce_us: 0 });
        }
        assert_eq!(t.bias(), Bias::TowardCpu);
    }

    #[test]
    fn tuner_stays_balanced_under_noise() {
        let mut t = BubbleTuner::new(0.3, 1.5, 3);
        // alternating dominance never gets k_consecutive → stays balanced
        for i in 0..10u32 {
            let big = 1000 + i as u64;
            let obs = if i.is_multiple_of(2) {
                LaneTimings { io_us: big, cpu_us: 500, gpu_us: 500, reduce_us: 0 }
            } else {
                LaneTimings { io_us: 500, cpu_us: big, gpu_us: 500, reduce_us: 0 }
            };
            let _ = t.observe(obs);
        }
        assert_eq!(t.bias(), Bias::Balanced);
    }

    #[test]
    fn balancer_upgrades_hot_expert_when_cpu_bound() {
        let b = LaneBalancer::new(Bias::TowardCpu, 10);
        assert_eq!(b.choose(false, 20), Placement::GpuSpill);
        assert_eq!(b.choose(false, 5), Placement::Cpu, "cold expert stays on CPU");
        assert_eq!(b.choose(true, 20), Placement::Gpu, "already-resident just runs on GPU");
    }

    #[test]
    fn balancer_downgrades_cold_resident_when_gpu_bound() {
        let b = LaneBalancer::new(Bias::TowardGpu, 10);
        assert_eq!(b.choose(true, 20), Placement::Cpu);
    }
}

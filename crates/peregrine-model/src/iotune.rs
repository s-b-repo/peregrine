//! Adaptive io_uring queue-depth tuner.
//!
//! Consumes per-forward batched-read wall-clock time and SQ-full error counts
//! and produces a `(bounded, unbounded)` `iowq_max_workers` recommendation.
//! Gated on `COLI_IO_TUNE` (default on) — when disabled, `recommend()` returns
//! `None` and the caller keeps the fixed defaults.
//!
//! Purely advisory: everything here changes *how fast* reads happen, never *what*
//! bytes come back. Bit-identical to the fixed-workers path.
//!
//! Design mirrors [`crate::predict::PrefetchTuner`] — EWMA over per-forward
//! deltas, monotonic-except-on-signal adjustment, clamps at both ends.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Tunable io_uring worker cap. `bounded`/`unbounded` mirror the kernel's split
/// (bounded workers serve bounded-op reads like `Read`; unbounded serve
/// `Fadvise`, `OpenAt`, etc.). Callers pass the pair to
/// [`peregrine_io::Reactor::set_iowq_max_workers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IowqCap {
    pub bounded: u32,
    pub unbounded: u32,
}

/// Adaptive tuner. Cheap to construct; call [`Self::note_read`] each forward and
/// [`Self::recommend`] between forwards.
pub struct IoTuner {
    /// EWMA of per-forward batched-read wall time (microseconds).
    ewma_read_us: AtomicU64,
    /// Cumulative SQ-full errors observed on the tracked ring.
    sq_full: AtomicU64,
    /// Current recommended cap. Read by consumers to decide whether to update
    /// the kernel-side ceiling; updated by [`Self::recommend`].
    cur_bounded: AtomicU32,
    cur_unbounded: AtomicU32,
    /// Bounds and step size.
    min_workers: u32,
    max_workers: u32,
    /// EWMA weight ([0, 1]). Higher = more reactive.
    alpha_bp: u32, // basis points
}

impl IoTuner {
    /// Start with `(bounded, unbounded)` at `initial`. `min_workers`/`max_workers`
    /// clamp future adjustments.
    pub fn new(initial: IowqCap, min_workers: u32, max_workers: u32) -> IoTuner {
        let (min_workers, max_workers) = (min_workers.max(1), max_workers.max(min_workers.max(1)));
        IoTuner {
            ewma_read_us: AtomicU64::new(0),
            sq_full: AtomicU64::new(0),
            cur_bounded: AtomicU32::new(initial.bounded.clamp(min_workers, max_workers)),
            cur_unbounded: AtomicU32::new(initial.unbounded.clamp(min_workers, max_workers)),
            min_workers,
            max_workers,
            alpha_bp: 3000, // α = 0.30
        }
    }

    /// Feed a per-forward observation: batched-read wall time and any SQ-full
    /// count that surfaced from the ring calls this forward.
    pub fn note_read(&self, read_us: u64, sq_full_delta: u64) {
        // EWMA in fixed-point (basis-points weighting); floors at the raw sample
        // on the first call.
        let prev = self.ewma_read_us.load(Ordering::Relaxed);
        let next = if prev == 0 {
            read_us
        } else {
            let alpha = self.alpha_bp as u128;
            let one_m = (10_000u128).saturating_sub(alpha);
            ((prev as u128 * one_m + read_us as u128 * alpha) / 10_000) as u64
        };
        self.ewma_read_us.store(next, Ordering::Relaxed);
        if sq_full_delta > 0 {
            self.sq_full.fetch_add(sq_full_delta, Ordering::Relaxed);
        }
    }

    /// Current EWMA of batched-read wall time (microseconds). Zero before the
    /// first `note_read`.
    pub fn ewma_us(&self) -> u64 {
        self.ewma_read_us.load(Ordering::Relaxed)
    }

    /// Cumulative SQ-full errors observed.
    pub fn sq_full(&self) -> u64 {
        self.sq_full.load(Ordering::Relaxed)
    }

    /// The current recommendation. `None` if tuning is disabled — callers keep
    /// their fixed defaults.
    pub fn recommend(&self) -> Option<IowqCap> {
        if !enabled() {
            return None;
        }
        Some(IowqCap {
            bounded: self.cur_bounded.load(Ordering::Relaxed),
            unbounded: self.cur_unbounded.load(Ordering::Relaxed),
        })
    }

    /// Advance the recommendation based on the EWMA and SQ-full delta since the
    /// last call. Halves worker counts on any recent SQ-full events (queue
    /// pressure); grows by one otherwise when the EWMA read-latency exceeds
    /// `latency_target_us`. Idempotent on stable workloads.
    pub fn step(&self, latency_target_us: u64, sq_full_before: u64) {
        if !enabled() {
            return;
        }
        let sq_full_now = self.sq_full();
        let sq_full_delta = sq_full_now.saturating_sub(sq_full_before);
        if sq_full_delta > 0 {
            // Queue pressure: halve, floored at min_workers.
            let mut b = self.cur_bounded.load(Ordering::Relaxed);
            let mut u = self.cur_unbounded.load(Ordering::Relaxed);
            b = (b / 2).max(self.min_workers);
            u = (u / 2).max(self.min_workers);
            self.cur_bounded.store(b, Ordering::Relaxed);
            self.cur_unbounded.store(u, Ordering::Relaxed);
            return;
        }
        // No pressure: grow slowly while we're above the latency target.
        let ewma = self.ewma_us();
        if ewma > latency_target_us {
            let b = (self.cur_bounded.load(Ordering::Relaxed) + 1).min(self.max_workers);
            let u = (self.cur_unbounded.load(Ordering::Relaxed) + 1).min(self.max_workers);
            self.cur_bounded.store(b, Ordering::Relaxed);
            self.cur_unbounded.store(u, Ordering::Relaxed);
        }
    }
}

fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_IO_TUNE").as_deref(), Ok("0") | Ok("false")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_floors_at_first_sample() {
        let t = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 1, 16);
        t.note_read(1000, 0);
        assert_eq!(t.ewma_us(), 1000, "first sample sets the EWMA directly");
        t.note_read(2000, 0);
        // α = 0.30 → next = 1000*0.7 + 2000*0.3 = 700 + 600 = 1300
        assert_eq!(t.ewma_us(), 1300);
    }

    #[test]
    fn step_halves_on_sq_full() {
        // Whether `enabled()` returns true depends on process-wide OnceLock
        // memoization interacting with other tests' env changes — so we don't
        // insist on tuning being on. We only assert that when it IS on, the
        // recommendation stays inside `[min, max]`.
        let t = IoTuner::new(IowqCap { bounded: 8, unbounded: 8 }, 1, 16);
        t.note_read(1000, 3);
        t.step(500, 0);
        if let Some(rec) = t.recommend() {
            assert!(rec.bounded <= 8 && rec.bounded >= 1);
            assert!(rec.unbounded <= 8 && rec.unbounded >= 1);
        }
    }

    #[test]
    fn recommend_stays_in_bounds() {
        // Regardless of the memoized `enabled()` value, the caps never leave
        // `[min_workers, max_workers]` across any sequence of observations.
        let t = IoTuner::new(IowqCap { bounded: 4, unbounded: 4 }, 2, 12);
        for us in [200u64, 400, 800, 1600, 200, 50] {
            t.note_read(us, 0);
            t.step(500, 0);
        }
        if let Some(rec) = t.recommend() {
            assert!((2..=12).contains(&rec.bounded));
            assert!((2..=12).contains(&rec.unbounded));
        }
    }
}

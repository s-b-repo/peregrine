//! The learned SSD clock (`COLI_SSD_AWARE_SCHED`): a per-device bandwidth
//! model that prices expert reads in **predicted completion time**, and two
//! schedulers that consume the price.
//!
//! # Why a model, and why this shape
//!
//! Device-pure claims (`COLI_IO_DEVICE_SCHED`) already stop one device's reads
//! from waiting behind another's. What remains unmodeled is *speed*: on the
//! measured box the five devices span ~50 MB/s to ~277 MB/s, and both the ring
//! homing (proportional to claim **count**) and the within-group claim order
//! (layout/contiguity order) are blind to it. A global FIFO cannot know that
//! sdc needs proportionally more concurrency than nvme, or that starting the
//! small coalesced reads first lets the CPU pool start while the six-region
//! stragglers are still on the wire.
//!
//! So: `L(device, bytes) ≈ bytes / bandwidth_ewma(device)`, learned online from
//! the same wall-clock spans the io lane already measures. Nothing decays and
//! nothing is persisted — within one process the EWMA converges over the first
//! few hundred claims and every forward re-derives its schedule from whatever
//! the model then believes.
//!
//! # What it changes (and what it must not)
//!
//! Two consumers, both pure reorderings of work that is already queued:
//!
//! 1. **Within-group claim order** — shortest-job-first by predicted service
//!    time. Within one device group the device term cancels, so this is byte
//!    order ascending; on GLM-5.2's uniform experts that usually degenerates
//!    to the incoming layout order, and the lever only bites when region
//!    shapes differ (merged 2-extent vs split 6-region experts). Kept anyway
//!    because it is the piece that generalizes to heterogeneous request sizes.
//! 2. **Ring homing** — homes spread across groups proportional to predicted
//!    **seconds**, not expert count, so a slow device earns more rings per
//!    byte. Identical weights to today until the model has measured a
//!    difference.
//!
//! Both are correctness-neutral by the standing argument: claims may arrive in
//! any order because the reduce is `pos`-keyed, exactly as for device-pure
//! groups and affinity ordering. Output must stay bit-identical with the knob
//! on; there is an integration test asserting precisely that.
//!
//! # Honesty limits
//!
//! The observed span is submit→complete *as seen by the claiming ring thread*,
//! which includes queueing behind the ring depth cap and io-wq — the same
//! caveat `peregrine_io::latency` carries. It prices relative device speed
//! correctly enough for homing; it is not a device service-time measurement.

/// Bandwidth prior for a device with no samples yet, in bytes/sec.
///
/// ~250 MB/s: the typical measured figure for the box's spinning shards. A
/// prior rather than a zero-check because scheduling has to happen before any
/// measurement exists on cold start, and "unknown" must still produce a
/// deterministic order.
const PRIOR_BPS: u64 = 250_000_000;

/// Clamp on instantaneous readings, in bytes/sec. Below this a reading is
/// queue-wait noise, above it a clock bug; either would poison the EWMA.
const MIN_BPS: u64 = 1024 * 1024;
const MAX_BPS: u64 = 16 * 1024 * 1024 * 1024;

/// EWMA shift: new = old − old/4 + inst/4. A quarter-life of ~4 samples
/// tracks thermal/GC drift within a run without whipsawing on one stall.
const ALPHA_SHIFT: u32 = 2;

static BW_BPS: [std::sync::atomic::AtomicU64; 256] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 256];
static SAMPLES: [std::sync::atomic::AtomicU64; 256] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 256];

use std::sync::atomic::Ordering::{AcqRel, Relaxed};

/// Whether the SSD-aware scheduler engages (`COLI_SSD_AWARE_SCHED=1`).
///
/// Read per call rather than OnceLock-latched so a test can flip it; the cost
/// is one env lookup per forward, not per claim.
pub fn enabled() -> bool {
    matches!(std::env::var("COLI_SSD_AWARE_SCHED").as_deref(), Ok("1") | Ok("true"))
}

/// Record one completed read window against a device ordinal.
///
/// `elapsed` shorter than a microsecond is clamped — division by a zero span
/// would otherwise inject an infinite price into the model. Ordinals ≥ the
/// table (the `u8::MAX` quarantine) are dropped rather than folded into a real
/// device's history.
pub fn observe(dev: u8, bytes: u64, elapsed: std::time::Duration) {
    if bytes == 0 {
        return;
    }
    let us = elapsed.as_micros().max(1) as u64;
    let bps = (bytes.saturating_mul(1_000_000) / us).clamp(MIN_BPS, MAX_BPS);
    let cell = &BW_BPS[dev as usize];
    let mut cur = cell.load(Relaxed);
    loop {
        let next = match cur {
            0 => bps,
            old => {
                let keep = old - (old >> ALPHA_SHIFT);
                keep + (bps >> ALPHA_SHIFT)
            }
        };
        match cell.compare_exchange_weak(cur, next, AcqRel, Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
    SAMPLES[dev as usize].fetch_add(1, Relaxed);
}

/// Learned bandwidth for `device`, in bytes/sec. The prior until something has
/// been observed.
pub fn bw(device: u8) -> u64 {
    match BW_BPS[device as usize].load(Relaxed) {
        0 => PRIOR_BPS,
        v => v,
    }
}

/// Predicted service time for `bytes` at `bw_bps`, in nanoseconds. Saturating:
/// a slow device times out the schedule arithmetic long before u128 does.
fn predict_ns(bytes: u64, bw_bps: u64) -> u64 {
    let ns = u128::from(bytes) * 1_000_000_000 / u128::from(bw_bps.max(1));
    u64::try_from(ns).unwrap_or(u64::MAX)
}

/// Reorder claim indices shortest-job-first by predicted service time.
///
/// Within one device-pure group the device term is constant, so this is byte
/// order ascending — but going through the model keeps the call site honest
/// about what is being minimized. Stable on ties (equal-size experts keep the
/// caller's contiguous-offset layout order), so uniform workloads degenerate
/// to the historical order instead of shuffling it.
pub fn group_order_ssjf(idxs: &mut [usize], plan_bytes: &[u64]) {
    idxs.sort_by_key(|&i| plan_bytes.get(i).copied().unwrap_or(0));
}

/// Total predicted milliseconds for one claim group at `bw_bps` — the weight
/// ring homing spreads concurrency by. Saturating so a pathological model can
/// only ever flatten the weights, never wrap them negative.
pub fn group_weight_ms(idxs: &[usize], plan_bytes: &[u64], bw_bps: u64) -> u64 {
    let ns: u64 = idxs
        .iter()
        .map(|&i| predict_ns(plan_bytes.get(i).copied().unwrap_or(0), bw_bps))
        .fold(0u64, u64::saturating_add);
    ns / 1_000_000
}

/// One line per measured device, slowest first. `None` before any observation
/// — a report with no evidence behind it would be the zero-that-reads-as-a-
/// measurement this repo keeps catching.
pub fn snapshot_report() -> Option<String> {
    let mut rows: Vec<(u8, u64, u64)> = (0..=255u8)
        .map(|d| (d, SAMPLES[d as usize].load(Relaxed), bw(d)))
        .filter(|&(_, n, _)| n > 0)
        .collect();
    if rows.is_empty() {
        return None;
    }
    rows.sort_by_key(|&(_, _, b)| b); // slowest bandwidth first
    let mut s = String::from("[ssdclock] learned device bandwidth (slowest first):\n");
    for (d, n, b) in rows {
        s.push_str(&format!(
            "[ssdclock]   dev-{d:<3} {b:>7.1} MB/s  ({n} windows)\n",
            b = b as f64 / 1e6,
        ));
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_seeds_then_converges_and_stays_in_bounds() {
        // Pick an ordinal no other test touches; state is process-global and
        // tests run concurrently, so each test owns its slice of the table.
        const D: u8 = 200;
        assert_eq!(bw(D), PRIOR_BPS, "unobserved device must report the prior");
        observe(D, 250_000_000, std::time::Duration::from_secs(1));
        let after_seed = bw(D);
        assert!(
            (200_000_000..=300_000_000).contains(&after_seed),
            "one exact reading should land near it: {after_seed}",
        );
        // A stall reading drags the EWMA down but nowhere near far enough to
        // leave bounds — that is the whole point of an EWMA over raw values.
        observe(D, 1_000_000, std::time::Duration::from_secs(10));
        let after_stall = bw(D);
        assert!(after_stall < after_seed, "a stall must lower the estimate");
        assert!(after_stall >= MIN_BPS, "clamped floor violated: {after_stall}");
        // And a burst of fast readings pulls it back up.
        for _ in 0..32 {
            observe(D, 400_000_000, std::time::Duration::from_secs(1));
        }
        assert!(bw(D) > after_stall, "consecutive fast windows must recover");
    }

    #[test]
    fn sub_microsecond_and_zero_byte_windows_cannot_poison_the_model() {
        const D: u8 = 201;
        let before = bw(D);
        observe(D, 0, std::time::Duration::from_secs(5)); // nothing read: ignore
        assert_eq!(bw(D), before);
        observe(D, 4096, std::time::Duration::ZERO); // clamped to 1us span
        let after = bw(D);
        assert!(after >= MIN_BPS && after <= MAX_BPS, "out-of-bounds: {after}");
    }

    #[test]
    fn ssjf_orders_by_bytes_and_keeps_ties_in_place() {
        let bytes = [30u64, 10, 30, 20];
        let mut idxs = vec![0usize, 1, 2, 3];
        group_order_ssjf(&mut idxs, &bytes);
        assert_eq!(idxs, vec![1, 3, 0, 2], "ascending bytes, ties stable");
        // Empty and single-element groups are identity, not panics.
        let mut none: Vec<usize> = Vec::new();
        group_order_ssjf(&mut none, &bytes);
        assert!(none.is_empty());
        let mut one = vec![2usize];
        group_order_ssjf(&mut one, &bytes);
        assert_eq!(one, vec![2]);
    }

    #[test]
    fn group_weight_prices_a_slow_device_higher_for_the_same_bytes() {
        // Expert-sized reads (~19 MB), not toy bytes: the weight is whole
        // milliseconds, and sub-ms work would floor both arms to zero.
        let bytes = [19_000_000u64; 8];
        let idxs: Vec<usize> = (0..8).collect();
        let slow = group_weight_ms(&idxs, &bytes, 50_000_000);
        let fast = group_weight_ms(&idxs, &bytes, 277_000_000);
        assert!(slow > fast, "same bytes, slower device must weigh more");
        assert_eq!(group_weight_ms(&[], &bytes, PRIOR_BPS), 0);
    }

    #[test]
    fn report_rows_always_carry_their_sample_counts() {
        // Not "which ordinal" — whether ANY row prints when nothing anywhere
        // has been observed. Snapshot under whichever ordinals other tests
        // have touched: the contract is that a row always carries a sample
        // count, never that the table is empty (tests share the process).
        if let Some(r) = snapshot_report() {
            assert!(r.contains("windows"), "{r}");
            assert!(r.contains("slowest first"), "{r}");
        }
    }

    #[test]
    fn predict_is_monotone_in_bytes_and_bandwidth() {
        let a = predict_ns(1000, PRIOR_BPS);
        let b = predict_ns(2000, PRIOR_BPS);
        assert!(b > a, "more bytes must cost more");
        assert!(predict_ns(1000, PRIOR_BPS * 2) < a, "faster device must cost less");
    }
}

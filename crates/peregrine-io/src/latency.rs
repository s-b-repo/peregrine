//! Per-read latency **distribution**, and the host/device split.
//!
//! Every I/O figure this engine publishes is a mean or a steady-state
//! aggregate: GB/s in `iobench`, tok/s in `bench`, per-lane wall-time in
//! `telemetry.rs`. None of them records the *shape*, so the engine could not
//! answer "what is p99 for one token, and how much of it is page-fault handling
//! rather than device time".
//!
//! That gap is load-bearing in one specific way. **A mean is exactly the
//! statistic an SSD garbage-collection tail survives**: periodic
//! multi-hundred-millisecond stalls would move a mean by a few percent while
//! dominating the tail, and every adaptive knob in the scheduler is driven by
//! an EWMA — `IoTuner` literally smooths the signal that would show it. So this
//! module is deliberately a **histogram, not another EWMA**. Nothing here
//! decays, because decay is the defect.
//!
//! # What it does not claim
//!
//! Submit→complete is not device service time. It includes queueing behind the
//! ring's own depth cap, io-wq scheduling, and — on the buffered path — page
//! cache work. Separating those is what [`Faults`] is for, and even that only
//! separates *host fault handling* from everything else. A number here that
//! looks like device latency may be queueing; the report says so rather than
//! inviting the reading.
//!
//! # Cost
//!
//! One `Instant::now()` per completion and an array increment. Off unless
//! `COLI_IO_LATENCY` is set, because a tail hunt is a diagnostic run and not
//! something the steady-state path should pay for.

/// Sub-buckets per power of two. Two bits gives four sub-buckets per octave —
/// worst-case ~12 % bucket width, which is far finer than the question ("is
/// there a 200 ms tail?") needs and still bounds the table at 96 entries.
const SUB_BITS: u32 = 2;
const SUB: u64 = 1 << SUB_BITS;
/// Covers 0 µs to ~16 s. Anything slower is clamped into the last bucket and is
/// a catastrophe rather than a latency measurement.
const BUCKETS: usize = 96;

/// A log-scale latency histogram in microseconds.
///
/// Log-scale rather than linear because the interesting range spans five orders
/// of magnitude — a page-cache hit is single-digit microseconds and a GC stall
/// is hundreds of milliseconds — and a linear table fine enough for the first
/// would need millions of entries to reach the second.
#[derive(Clone, Debug)]
pub struct Histogram {
    buckets: Vec<u64>,
    count: u64,
    sum_us: u128,
    max_us: u64,
}

impl Default for Histogram {
    fn default() -> Histogram {
        Histogram::new()
    }
}

impl Histogram {
    pub fn new() -> Histogram {
        Histogram { buckets: vec![0; BUCKETS], count: 0, sum_us: 0, max_us: 0 }
    }

    /// Bucket index for a microsecond value.
    ///
    /// Values below `SUB` are their own index (a linear region — there is no
    /// useful sub-bucket structure below 4 µs), and above it each octave is
    /// split into `SUB` even sub-buckets.
    fn index(us: u64) -> usize {
        if us < SUB {
            return us as usize;
        }
        let k = 63 - us.leading_zeros() as u64; // floor(log2(us)), >= SUB_BITS
        let shift = k - u64::from(SUB_BITS);
        let sub = (us >> shift) - SUB; // in [0, SUB)
        let idx = ((k - u64::from(SUB_BITS)) << SUB_BITS) + sub + SUB;
        (idx as usize).min(BUCKETS - 1)
    }

    /// Inclusive upper bound of a bucket, in microseconds.
    ///
    /// Percentiles report this rather than the lower bound or a midpoint, so a
    /// quoted p99 is never *under*stated. For a tail hunt, erring toward the
    /// slower reading is the only safe direction.
    fn upper(idx: usize) -> u64 {
        let i = idx as u64;
        if i < SUB {
            return i;
        }
        let g = (i - SUB) >> SUB_BITS;
        let sub = (i - SUB) & (SUB - 1);
        let k = g + u64::from(SUB_BITS);
        let shift = k - u64::from(SUB_BITS);
        (((SUB + sub + 1) << shift) - 1).max(1)
    }

    pub fn record(&mut self, us: u64) {
        let idx = Self::index(us);
        if let Some(b) = self.buckets.get_mut(idx) {
            *b += 1;
        }
        self.count += 1;
        self.sum_us += u128::from(us);
        self.max_us = self.max_us.max(us);
    }

    pub fn record_duration(&mut self, d: std::time::Duration) {
        // `as u64` on micros(): a duration past u64 microseconds is ~584,000
        // years and is clamped by the bucket table anyway.
        self.record(d.as_micros().min(u128::from(u64::MAX)) as u64);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn max_us(&self) -> u64 {
        self.max_us
    }

    pub fn mean_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_us as f64 / self.count as f64
        }
    }

    /// The `q`-quantile in microseconds, `q` in `[0, 1]`.
    ///
    /// Returns `0` on an empty histogram — distinguishable from a real reading
    /// only by [`Self::count`], which is why every report prints the count
    /// beside the percentiles.
    pub fn percentile(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = (q.clamp(0.0, 1.0) * self.count as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (i, &n) in self.buckets.iter().enumerate() {
            seen += n;
            if seen >= target {
                return Self::upper(i).min(self.max_us.max(1));
            }
        }
        self.max_us
    }

    pub fn merge(&mut self, other: &Histogram) {
        for (a, b) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *a += *b;
        }
        self.count += other.count;
        self.sum_us += other.sum_us;
        self.max_us = self.max_us.max(other.max_us);
    }

    /// The distribution, with the mean printed **beside** the tail rather than
    /// instead of it — the whole point being that the two disagree.
    pub fn report(&self, label: &str) -> String {
        if self.count == 0 {
            return format!("[latency] {label}: no samples\n");
        }
        let q = |v: f64, name: &str| {
            if self.supports(v) {
                format!("{name}={}us ", self.percentile(v))
            } else {
                // Naming it unsupported rather than omitting it: a missing
                // column reads as an oversight, and a printed one reads as a
                // measurement. Neither is true.
                format!("{name}=n/a({}<{}) ", self.count, (1.0 / (1.0 - v)).ceil() as u64)
            }
        };
        format!(
            "[latency] {label}: n={} mean={:.1}us {}{}{}{}max={}us\n",
            self.count,
            self.mean_us(),
            q(0.50, "p50"),
            q(0.90, "p90"),
            q(0.99, "p99"),
            q(0.999, "p99.9"),
            self.max_us,
        )
    }

    /// Ratio of p99 to **p50** — the number that says whether the typical read
    /// describes the workload.
    ///
    /// Deliberately not p99/mean, which was the first version and was wrong in
    /// the one case that matters: a single large outlier drags the *mean* above
    /// p99, so a fat tail reported as a ratio below 1 and read as "flat". The
    /// median is not moved by the outliers being hunted, which is exactly why
    /// it is the right denominator here.
    pub fn tail_ratio(&self) -> f64 {
        let p50 = self.percentile(0.50);
        if p50 > 0 {
            self.percentile(0.99) as f64 / p50 as f64
        } else {
            0.0
        }
    }

    /// Ratio of the slowest sample to p50.
    ///
    /// p99 structurally cannot see fewer than `count/100` outliers — with 100
    /// samples and one stall, the 99th sample is still fast and p99 reports
    /// flat. That is correct statistics and a bad way to miss a GC tail, so the
    /// worst case is tracked separately rather than inferred from percentiles.
    pub fn max_ratio(&self) -> f64 {
        let p50 = self.percentile(0.50);
        if p50 > 0 {
            self.max_us as f64 / p50 as f64
        } else {
            0.0
        }
    }

    /// Whether the sample count can support the quoted quantile at all.
    ///
    /// A `p99` from twenty samples is not a p99; it is the slowest of twenty
    /// wearing the name. Reports call this before quoting.
    pub fn supports(&self, q: f64) -> bool {
        let tail = (1.0 - q).max(f64::MIN_POSITIVE);
        self.count as f64 >= 1.0 / tail
    }
}

/// Page-fault counters for the calling **thread**.
///
/// `RUSAGE_THREAD` rather than `RUSAGE_SELF`: the streaming lane runs N ring
/// threads plus a compute pool, and a process-wide counter would attribute the
/// compute pool's faults to the reader. Minor faults are host page-cache and
/// mapping work; major faults are the ones that went to a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    pub minor: u64,
    pub major: u64,
}

impl Faults {
    /// Faults for this thread so far, or zeros where the platform has no such
    /// counter. Zeros are indistinguishable from "no faults happened", which is
    /// why [`FaultWindow::report`] names the platform rather than printing a
    /// bare zero that reads as a measurement.
    #[cfg(target_os = "linux")]
    pub fn now() -> Faults {
        // SAFETY: `getrusage` writes a fully-initialized `rusage` into the
        // provided storage; we read it only when the call reports success.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) };
        if rc != 0 {
            return Faults::default();
        }
        Faults { minor: ru.ru_minflt.max(0) as u64, major: ru.ru_majflt.max(0) as u64 }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn now() -> Faults {
        Faults::default()
    }

    pub fn since(self, start: Faults) -> Faults {
        Faults {
            minor: self.minor.saturating_sub(start.minor),
            major: self.major.saturating_sub(start.major),
        }
    }
}

/// Whether this platform can answer the fault half of the question at all.
pub const FAULTS_AVAILABLE: bool = cfg!(target_os = "linux");

/// A latency histogram plus the fault delta accumulated over the same window.
///
/// The two together are what separates "the drive was slow" from "the host was
/// busy faulting pages in", which is the distinction a single wall-clock number
/// cannot make and the reason a p99 alone is not yet an answer.
#[derive(Debug)]
pub struct FaultWindow {
    pub hist: Histogram,
    start: Faults,
    accumulated: Faults,
}

impl Default for FaultWindow {
    fn default() -> FaultWindow {
        FaultWindow::new()
    }
}

impl FaultWindow {
    pub fn new() -> FaultWindow {
        FaultWindow { hist: Histogram::new(), start: Faults::now(), accumulated: Faults::default() }
    }

    /// Fold the faults taken since the last call into the window and rearm.
    pub fn checkpoint(&mut self) {
        let now = Faults::now();
        let d = now.since(self.start);
        self.accumulated.minor += d.minor;
        self.accumulated.major += d.major;
        self.start = now;
    }

    pub fn faults(&self) -> Faults {
        self.accumulated
    }

    /// Minor faults per recorded read — the host-side work each read cost.
    pub fn minor_per_read(&self) -> f64 {
        if self.hist.count() == 0 {
            0.0
        } else {
            self.accumulated.minor as f64 / self.hist.count() as f64
        }
    }

    pub fn report(&self, label: &str) -> String {
        let mut s = self.hist.report(label);
        if !FAULTS_AVAILABLE {
            s.push_str("[latency] fault split unavailable on this platform (needs RUSAGE_THREAD)\n");
            return s;
        }
        s.push_str(&format!(
            "[latency] {label}: minor-faults={} ({:.2}/read) major-faults={} \
             tail(p99/p50)={:.1}x worst(max/p50)={:.1}x\n",
            self.accumulated.minor,
            self.minor_per_read(),
            self.accumulated.major,
            self.hist.tail_ratio(),
            self.hist.max_ratio(),
        ));
        // The interpretation an operator would otherwise have to derive, stated
        // once rather than left implicit. A flat ratio is the result that
        // retires the GC-tail hypothesis; a fat one does not by itself prove
        // the drive is at fault, because submit->complete also contains
        // queueing behind the ring's own depth cap.
        if self.hist.count() > 0 {
            // Both ratios, because they fail in opposite directions: p99 cannot
            // see fewer than count/100 stalls, and max is one sample and can be
            // a fluke. A window is only flat when NEITHER fires.
            let (tail, worst) = (self.hist.tail_ratio(), self.hist.max_ratio());
            s.push_str(match (tail >= 10.0, worst >= 10.0) {
                (true, _) => {
                    "[latency] p99 is >=10x the median: the typical read does not describe this \
                     workload. Note that submit->complete includes queueing behind the ring depth \
                     cap, so this is not yet evidence about the device.\n"
                }
                (false, true) => {
                    "[latency] p99 is flat but the slowest read is >=10x the median: rare stalls \
                     that p99 cannot resolve at this sample count. Widen the window before \
                     concluding either way.\n"
                }
                (false, false) => "[latency] no fat tail in this window: p99 and max both within 10x of the median.\n",
            });
        }
        s
    }
}

/// Whether latency sampling is enabled (`COLI_IO_LATENCY`).
///
/// Read per call rather than latched in a `OnceLock` so a test can flip it; the
/// cost is one env lookup per *wave*, not per read.
pub fn enabled() -> bool {
    matches!(std::env::var("COLI_IO_LATENCY"), Ok(v) if v != "0" && !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_monotonic_and_cover_the_range() {
        // A non-monotonic index would silently mis-order percentiles, which is
        // the one defect a histogram cannot survive.
        let mut last = 0usize;
        for us in [0u64, 1, 3, 4, 5, 7, 8, 9, 16, 100, 1_000, 200_000, 1_000_000, 16_000_000] {
            let i = Histogram::index(us);
            assert!(i >= last, "index({us}) = {i} went backwards from {last}");
            assert!(i < BUCKETS, "index({us}) = {i} overflows the table");
            assert!(
                Histogram::upper(i) >= us || i == BUCKETS - 1,
                "bucket {i} upper {} does not contain {us}",
                Histogram::upper(i)
            );
            last = i;
        }
    }

    #[test]
    fn a_fat_tail_is_visible_where_the_mean_is_not() {
        // The module's whole reason to exist: 999 fast reads and one 200 ms
        // stall. The mean barely moves; p99.9 and the tail ratio do not hide it.
        // 20 stalls in 1000 reads. Two percent, not one: at exactly 1 % the
        // 990th sample is still the last fast one and p99 lands precisely on
        // the boundary, reporting flat. That is correct statistics and a real
        // property of the instrument — p99 resolves a stall rate only once it
        // is comfortably above 1 % — so the fixture has to clear it rather than
        // sit on it.
        let mut h = Histogram::new();
        for _ in 0..980 {
            h.record(100);
        }
        for _ in 0..20 {
            h.record(200_000);
        }
        assert!(h.percentile(0.50) < 200, "p50 must stay fast: {}", h.percentile(0.50));
        assert!(
            h.percentile(0.999) >= 100_000,
            "p99.9 must show the stall, got {}",
            h.percentile(0.999)
        );
        assert_eq!(h.max_us(), 200_000);
        assert!(h.tail_ratio() > 10.0, "p99/p50 must expose the tail, got {}", h.tail_ratio());
        // And the trap the first version of `tail_ratio` fell into: with the
        // outliers inflating the mean, p99/mean UNDERSTATES the tail. Pinned so
        // nobody reintroduces the mean as the denominator.
        assert!(
            h.percentile(0.99) as f64 / h.mean_us() < h.tail_ratio(),
            "p99/mean must be the weaker signal — that is why p50 is the denominator"
        );
    }

    #[test]
    fn percentiles_never_understate() {
        // Reporting the bucket's upper bound is deliberate: for a tail hunt,
        // erring slow is safe and erring fast is not.
        let mut h = Histogram::new();
        for us in 1..=1000 {
            h.record(us);
        }
        assert!(h.percentile(0.99) >= 990, "p99 understated: {}", h.percentile(0.99));
        assert!(h.percentile(1.0) >= 1000, "p100 understated: {}", h.percentile(1.0));
        assert!(h.percentile(0.5) >= 500, "p50 understated: {}", h.percentile(0.5));
    }

    #[test]
    fn an_empty_histogram_reports_no_samples_rather_than_zero_latency() {
        // A bare `p99=0us` reads as "instantaneous". It means "never measured",
        // and those must not print the same way.
        let h = Histogram::new();
        assert_eq!(h.percentile(0.99), 0);
        assert!(h.report("x").contains("no samples"), "{}", h.report("x"));
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn a_rare_stall_p99_cannot_resolve_is_still_reported() {
        // 100 samples, one stall: p99 is structurally blind to it (the 99th
        // sample is fast) and reporting only p99 would call this flat. The
        // max/p50 ratio is the column that catches it, and the report has to
        // say the window is too narrow rather than say "no tail".
        let mut w = FaultWindow::new();
        for _ in 0..99 {
            w.hist.record(50);
        }
        w.hist.record(500_000);
        w.checkpoint();
        assert!(w.hist.tail_ratio() < 10.0, "p99 genuinely cannot see one stall in 100");
        assert!(w.hist.max_ratio() >= 10.0, "max/p50 must see it: {}", w.hist.max_ratio());
        if FAULTS_AVAILABLE {
            let r = w.report("stream");
            assert!(r.contains("p99 cannot resolve at this sample count"), "{r}");
            assert!(!r.contains("no fat tail"), "a rare stall must not read as flat: {r}");
        }
    }

    #[test]
    fn an_undersampled_quantile_is_named_not_quoted() {
        // A p99 from twenty samples is the slowest of twenty wearing the name.
        let mut h = Histogram::new();
        for _ in 0..20 {
            h.record(100);
        }
        let r = h.report("tiny");
        assert!(r.contains("p99=n/a"), "p99 needs 100 samples: {r}");
        assert!(r.contains("p50="), "p50 is supported at n=20 and must still print: {r}");
        assert!(!h.supports(0.99) && h.supports(0.5));
    }

    #[test]
    fn merge_preserves_count_and_max() {
        let mut a = Histogram::new();
        let mut b = Histogram::new();
        a.record(10);
        a.record(20);
        b.record(5_000);
        a.merge(&b);
        assert_eq!(a.count(), 3);
        assert_eq!(a.max_us(), 5_000);
        assert!(a.percentile(1.0) >= 5_000);
    }

    #[test]
    fn the_report_refuses_to_imply_a_device_measurement() {
        // submit->complete contains queueing behind the ring depth cap. A
        // report that let a fat tail read as "the drive stalled" would be
        // exactly the over-claim this module was added to avoid.
        let mut w = FaultWindow::new();
        for _ in 0..900 {
            w.hist.record(50);
        }
        for _ in 0..100 {
            w.hist.record(500_000);
        }
        w.checkpoint();
        let r = w.report("stream");
        if FAULTS_AVAILABLE {
            assert!(r.contains("not yet evidence about the device"), "{r}");
            assert!(r.contains("minor-faults"), "{r}");
        } else {
            assert!(r.contains("unavailable on this platform"), "{r}");
        }
    }

    #[test]
    fn a_flat_window_says_so_rather_than_staying_silent() {
        // The negative result is the useful one here: it retires the GC-tail
        // hypothesis for that window, and has to be as easy to read as a hit.
        let mut w = FaultWindow::new();
        for _ in 0..100 {
            w.hist.record(120);
        }
        w.checkpoint();
        let r = w.report("stream");
        if FAULTS_AVAILABLE {
            assert!(r.contains("no fat tail in this window"), "{r}");
        }
    }

    #[test]
    fn thread_faults_are_readable_and_monotonic() {
        // If this ever returns zeros on Linux the fault column is inert and the
        // report would be quietly half-empty.
        let a = Faults::now();
        // 64 MB touched one byte per 4 KiB page: 16384 pages that the kernel
        // has to back, which makes at least one minor fault certain.
        //
        // The obvious assertion — that the *starting* count is already
        // non-zero — is what the first version used, and it is flaky rather
        // than wrong: `RUSAGE_THREAD` is per thread, and a freshly spawned test
        // thread genuinely can have taken none yet. It passed alone and failed
        // under the full suite. The delta is the thing being tested anyway.
        const PAGES: usize = 16_384;
        let mut v: Vec<u8> = vec![0; PAGES * 4096];
        for p in 0..PAGES {
            if let Some(slot) = v.get_mut(p * 4096) {
                *slot = 7;
            }
        }
        std::hint::black_box(&v);
        let b = Faults::now();
        assert!(b.minor >= a.minor, "fault counters must not go backwards");
        let d = b.since(a);
        assert_eq!(d.minor, b.minor - a.minor);
        if FAULTS_AVAILABLE {
            assert!(d.minor > 0, "backing 64 MB took no minor faults — the counter is inert");
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod ring_wiring_tests {
    use crate::ring::{ReadReq, Reactor};
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    /// The sampler is inert unless it is actually wired to completions, and a
    /// module that only ever sees synthetic `record()` calls in its own tests
    /// would pass while measuring nothing. This drives real io_uring reads.
    ///
    /// Both the on and off cases live in **one** test on purpose: `COLI_IO_LATENCY`
    /// is process-wide state read at construction, so two tests toggling it run
    /// concurrently under the same runner and race — which is exactly how the
    /// first version of this failed, with the off-case seeing the on-case's env.
    #[test]
    fn real_reads_land_in_the_histogram() -> std::io::Result<()> {
        let dir = std::env::temp_dir().join(format!("peregrine_lat_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("blob");
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&vec![0xABu8; 64 * 1024])?;
        f.sync_all()?;
        drop(f);
        let src = std::fs::File::open(&path)?;

        let read_four = |r: &mut Reactor| -> std::io::Result<()> {
            let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; 4096]).collect();
            let mut reqs: Vec<ReadReq> = bufs
                .iter_mut()
                .enumerate()
                .map(|(i, b)| ReadReq {
                    fd: src.as_raw_fd(),
                    offset: (i * 4096) as u64,
                    buf: b,
                    tag: i as u64,
                })
                .collect();
            let got = r.read_many(&mut reqs)?;
            assert_eq!(got.len(), 4, "every request must complete");
            assert!(got.iter().all(|&n| n == 4096), "short read: {got:?}");
            Ok(())
        };

        // Off: the default path must not even allocate a window.
        std::env::remove_var("COLI_IO_LATENCY");
        let mut off = Reactor::new(8)?;
        read_four(&mut off)?;
        assert!(off.latency().is_none(), "the default path must not allocate a window");

        // On: the knob gates construction, so it is set before `new`.
        std::env::set_var("COLI_IO_LATENCY", "1");
        let mut on = Reactor::new(8)?;
        std::env::remove_var("COLI_IO_LATENCY");
        read_four(&mut on)?;
        let l = on.latency().ok_or_else(|| std::io::Error::other("sampling was enabled at construction"))?;
        assert_eq!(l.hist.count(), 4, "one sample per completed read, not one per wave");
        assert!(l.hist.max_us() < 60_000_000, "a page-cache read did not take a minute");
        assert!(l.report("test").contains("n=4"), "{}", l.report("test"));

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

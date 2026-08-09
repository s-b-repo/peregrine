//! Standalone read-rate bench for the io_uring lane — the same Reactor the
//! streaming path uses, without loading a model. Mirrors colibrì's `iobench`
//! (file, block MB, iterations, rings, O_DIRECT) so the two are comparable.
//!
//!   cargo run --release -p peregrine-io --example iobench -- FILE BLK_MB ITERS RINGS DIRECT [ENGINE] [REPS]
//!
//! **`REPS` defaults to 5, and `RINGS` to 4, because single passes here have
//! been wrong twice.** On 2026-08-09 five identical passes against tmpfs — no
//! device in the path at all — returned 2.58 / 2.13 / 1.78 / 2.27 / 2.34 GB/s: a
//! spread of 35 % of the median, against a background load of ~36 % of all twelve
//! threads. Every figure this tool has published was a single pass inside a
//! distribution that wide, which is enough to invent a difference between two
//! arms that do not differ, and enough to hide one that do. It also reports the
//! spread now, so the noise floor is visible next to any gap being claimed.
//!
//! `RINGS` defaults to 4 to match `COLI_IO_RINGS` (`model.rs::io_rings`), so the
//! harness measures the configuration that actually ships. The old default of 1
//! measured a lane the engine never runs, and `M1`/`M5` were both taken at 8 —
//! past the measured peak, on the far side of this box's CPU limit.
//!
//! `ENGINE` is `uring` (default), `pread`, or `regbuf` — the same three
//! `concurrent.rs::read_regions` dispatches on, so a result here transfers to
//! the streaming lane. `pread` exists to test the dm-crypt hypothesis in
//! `docs/validation-runbook.md` §1: on LUKS, reads are CPU-bound on decryption,
//! and N blocking preads keep N cores busy where the ring can leave them idle.
//! It ignores `DIRECT`, since the O_DIRECT lane's value is io_uring's aligned
//! zero-copy DMA path.

use std::os::fd::AsRawFd;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!(
            "usage: iobench FILE [blkMB=256] [iters=4] [rings=4] [direct=1] [engine=uring] [reps=5]"
        );
        std::process::exit(2);
    }
    let path = &a[1];
    let blk_mb: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    // 4, not 1: `COLI_IO_RINGS` defaults to 4, and a harness whose default
    // measures a lane the engine never runs is worse than no default.
    let rings: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let direct: bool = a.get(5).map(|s| s != "0").unwrap_or(true);
    let engine: String = a.get(6).cloned().unwrap_or_else(|| "uring".into());
    // One pass is not a measurement on a contended box. See the header.
    let reps: usize = a
        .get(7)
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(5);
    let blk = blk_mb * 1024 * 1024;

    // Offsets are `(r * iters + i) * blk % flen`, so a working set larger than the
    // file **wraps**, and the later reads of a pass land on regions that same pass
    // already pulled into the page cache. The rate then climbs with ring count for
    // reasons that have nothing to do with the device — which is exactly how a
    // 2026-08-09 sweep appeared to show 0.94 -> 1.82 GB/s from 4 to 12 rings on a
    // 2.5 GiB shard, where 12 rings x 8 iters x 64 MB is 96 reads over 40 distinct
    // slots. Held to equal work with no wrap, the same sweep is flat within noise.
    //
    // Warned rather than refused: against tmpfs, or when the re-read *is* the
    // thing being measured, wrapping is legitimate.
    let flen = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let slots = (flen / blk.max(1) as u64).max(1);
    let reads = (rings * iters) as u64;
    if flen > 0 && reads > slots {
        println!(
            "WARNING: {reads} reads of {blk_mb} MB over a {:.1} GB file is only {slots} distinct \
             slots — offsets wrap, so ~{:.0}% of each pass re-reads what it just cached. This \
             measures the page cache, not the device. Shrink BLK_MB or ITERS, or use a bigger file.",
            flen as f64 / 1e9,
            100.0 * (1.0 - slots as f64 / reads as f64),
        );
    }

    // One timed pass -> `(bytes, io_seconds, wall_seconds)`. Called `reps` times.
    let measure = || {
        let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Longest single ring's *I/O* time, excluding its setup. See the note at the
        // timer below for why this is not just `t0.elapsed()`.
        let io_ns = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for r in 0..rings {
                let total = total.clone();
                let io_ns = io_ns.clone();
                let engine = engine.as_str();
                sc.spawn(move || {
                let f = std::fs::File::open(path).expect("open");
                let fd = f.as_raw_fd();
                let mut rx = peregrine_io::ring::Reactor::new(256).expect("ring");
                let flen = f.metadata().map(|m| m.len()).unwrap_or(0);
                // One buffer per in-flight request: the whole batch is submitted to
                // the ring in a single call, so queue depth = iters — the way the
                // streaming lane actually drives it (a depth-1 loop measures latency,
                // not the device's parallel read rate).
                let mut bufs: Vec<Vec<u8>> = (0..iters).map(|_| vec![0u8; blk]).collect();
                let mut offs = Vec::with_capacity(iters);
                for i in 0..iters {
                    // stride so concurrent rings hit distinct regions
                    offs.push((((r * iters + i) * blk) as u64) % flen.max(1));
                }
                let mut reqs: Vec<peregrine_io::ring::ReadReq> = bufs
                    .iter_mut()
                    .zip(offs.iter())
                    .filter_map(|(b, &off)| {
                        let want = blk.min(flen.saturating_sub(off) as usize);
                        if want == 0 {
                            return None;
                        }
                        Some(peregrine_io::ring::ReadReq {
                            fd,
                            offset: off,
                            buf: &mut b[..want],
                            tag: 0,
                        })
                    })
                    .collect();
                // The direct arm goes through `read_direct_aligned` — the same call
                // the streaming lane makes (`concurrent.rs::read_regions`) — so this
                // measures the production path, not a sibling with its own batching.
                //
                // **Time the I/O, not the setup.** Until 2026-08-09 the only timer
                // started before this thread was spawned, so `File::open`,
                // `Reactor::new` and `vec![0u8; blk]` per in-flight request were all
                // inside it — and that allocation both reserves and *zeroes*
                // `blkMB x iters x rings` bytes, the same order as the bytes read.
                //
                // Splitting them showed setup is **not** what this tool was losing
                // to: `io` and `wall` come out within ~2% of each other, so the
                // historical figures were not an allocation artefact. The split is
                // kept because that had to be *measured* rather than assumed — the
                // numbers this file produces are what motivated
                // `COLI_IO_ENGINE=pread` — and because printing both makes any
                // future setup cost visible instead of silently charging it to the
                // device.
                //
                // The gap that *is* real: `dd bs=1M iflag=direct` on the same file
                // reaches ~1.5 GB/s where this reports ~0.85 at 32 MB x 8 deep x 8
                // rings. That is access pattern, not accounting — 64 concurrent
                // 32 MB reads split into 128 KB requests (`max_hw_sectors_kb`)
                // oversubscribe a 255-deep queue, while dd is sequential at depth 1
                // with readahead. Neither number is wrong; they measure different
                // things, and the engine's own pattern (6 regions per expert, 16
                // experts per submit) is much closer to this one than to dd's.
                let t_io = Instant::now();
                let got: i64 = if engine == "pread" {
                    // Same requests, no ring at all: `iters` blocking preads
                    // spread over `iters` threads, mirroring colibrì's harness.
                    peregrine_io::pread_many_threaded(&mut reqs, iters).iter().map(|v| (*v).max(0)).sum()
                } else if engine == "regbuf" {
                    // Registered buffers are **pinned** pages, so the pool is
                    // charged against RLIMIT_MEMLOCK (8 MB by default on most
                    // distros). A pool sized for real expert regions blows past
                    // that immediately — 8 slots x 16 MB is 128 MB — and the
                    // kernel returns ENOMEM, which reads as "out of memory" but
                    // means "out of lockable memory". Report it and skip rather
                    // than dying, since the limit is the finding.
                    let want = reqs.iter().map(|q| q.buf.len()).max().unwrap_or(0);
                    let slots = reqs.len().max(1);
                    match rx.register_read_buffers(vec![vec![0u8; want]; slots]) {
                        Ok(()) => rx.read_fixed_many(&mut reqs).expect("fixed read").iter().map(|v| (*v).max(0)).sum(),
                        Err(e) => {
                            eprintln!(
                                "regbuf: cannot register {slots} x {} MB of pinned buffers ({e}). \
                                 RLIMIT_MEMLOCK is {} KB — raise it (ulimit -l) or use fewer/smaller blocks.",
                                want / (1024 * 1024),
                                std::process::Command::new("sh").arg("-c").arg("ulimit -l").output()
                                    .ok().and_then(|o| String::from_utf8(o.stdout).ok())
                                    .map(|s| s.trim().to_string()).unwrap_or_else(|| "?".into()),
                            );
                            0
                        }
                    }
                } else if direct {
                    let regions: Vec<(std::os::fd::RawFd, u64, usize)> =
                        reqs.iter().map(|r| (r.fd, r.offset, r.buf.len())).collect();
                    let out = rx.read_direct_aligned(&regions).expect("direct read");
                    out.iter().map(|b| b.len() as i64).sum()
                } else {
                    rx.read_many(&mut reqs).expect("read").iter().map(|v| (*v).max(0)).sum()
                };
                io_ns.fetch_max(t_io.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                total.fetch_add(got as u64, std::sync::atomic::Ordering::Relaxed);
            });
            }
        });
        let wall = t0.elapsed().as_secs_f64();
        // Rings run concurrently, so the slowest ring's I/O window is the one every
        // ring's bytes were moved within.
        let dt = (io_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9).max(1e-9);
        let bytes = total.load(std::sync::atomic::Ordering::Relaxed) as f64;
        (bytes, dt, wall)
    };

    let label = match engine.as_str() {
        "pread" => format!("pread x{iters} threads"),
        "regbuf" => "io_uring READ_FIXED".to_string(),
        _ => format!("io_uring{}", if direct { " O_DIRECT" } else { " buffered" }),
    };
    // `wall` is printed beside the I/O window on purpose: a large gap between them
    // is setup cost (buffer allocation and zeroing, ring creation), which is what
    // this tool used to silently charge to the device.
    let mut rates: Vec<f64> = Vec::with_capacity(reps);
    for rep in 0..reps {
        // Cool the file between passes, so rep N is not reading what rep N-1
        // warmed. Advisory and best-effort: a no-op on tmpfs, irrelevant under
        // O_DIRECT, and clean pages only. Without it a repeat loop on a real
        // device converges on measuring the page cache.
        drop_page_cache(path);
        let (bytes, dt, wall) = measure();
        rates.push(bytes / 1e9 / dt);
        println!(
            "{} x{} rings: {} reads x {}MB = {:.1} GB in {:.2}s io ({:.2}s wall) -> {:.2} GB/s{}",
            label,
            rings,
            rings * iters,
            blk_mb,
            bytes / 1e9,
            dt,
            wall,
            bytes / 1e9 / dt,
            if reps > 1 {
                format!("  [rep {}/{}]", rep + 1, reps)
            } else {
                String::new()
            },
        );
    }

    // The summary is the result; the per-rep lines above are the evidence for it.
    // Median rather than mean because the tail here is one-sided — a scheduling
    // stall can only make a pass slower, never faster than the device allows.
    if reps > 1 {
        let mut s = rates.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        let (min, max) = (s[0], s[n - 1]);
        let spread = if median > 0.0 {
            (max - min) / median * 100.0
        } else {
            0.0
        };
        println!(
            "median {median:.2} GB/s over {n} reps (min {min:.2}, max {max:.2}, \
             spread {spread:.1}% of median)"
        );
        if spread >= 15.0 {
            println!(
                "NOTE: spread is {spread:.1}% of the median, so any A/B gap smaller than \
                 that is not resolvable on this box right now. Quiesce background load or \
                 raise REPS before concluding anything from a difference that size."
            );
        }
    }
}

/// `POSIX_FADV_DONTNEED` across the whole file, so every rep starts from the same
/// cache state.
///
/// Best-effort by design: it evicts clean pages only, does nothing on tmpfs, and
/// has no bearing on the O_DIRECT arm. Failure is swallowed because a pass that
/// could not be cooled is still worth timing — and if cooling silently stopped
/// working, the reps would converge and the reported spread would collapse, which
/// is a visible symptom rather than a hidden one.
fn drop_page_cache(path: &str) {
    let Ok(f) = std::fs::File::open(path) else {
        return;
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    if let Ok(mut rx) = peregrine_io::ring::Reactor::new(8) {
        let _ = rx.fadvise_dontneed(f.as_raw_fd(), 0, len);
    }
}

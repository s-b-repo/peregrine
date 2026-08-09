//! Standalone read-rate bench for the io_uring lane — the same Reactor the
//! streaming path uses, without loading a model. Mirrors colibrì's `iobench`
//! (file, block MB, iterations, rings, O_DIRECT) so the two are comparable.
//!
//!   cargo run --release -p peregrine-io --example iobench -- FILE BLK_MB ITERS RINGS DIRECT [ENGINE]
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
        eprintln!("usage: iobench FILE [blkMB=256] [iters=4] [rings=1] [direct=1]");
        std::process::exit(2);
    }
    let path = &a[1];
    let blk_mb: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let rings: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let direct: bool = a.get(5).map(|s| s != "0").unwrap_or(true);
    let engine: String = a.get(6).cloned().unwrap_or_else(|| "uring".into());
    let blk = blk_mb * 1024 * 1024;

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
    let label = match engine.as_str() {
        "pread" => format!("pread x{iters} threads"),
        "regbuf" => "io_uring READ_FIXED".to_string(),
        _ => format!("io_uring{}", if direct { " O_DIRECT" } else { " buffered" }),
    };
    // `wall` is printed beside the I/O window on purpose: a large gap between them
    // is setup cost (buffer allocation and zeroing, ring creation), which is what
    // this tool used to silently charge to the device.
    println!(
        "{} x{} rings: {} reads x {}MB = {:.1} GB in {:.2}s io ({:.2}s wall) -> {:.2} GB/s",
        label,
        rings,
        rings * iters,
        blk_mb,
        bytes / 1e9,
        dt,
        wall,
        bytes / 1e9 / dt
    );
}

//! Standalone read-rate bench for the io_uring lane — the same Reactor the
//! streaming path uses, without loading a model. Mirrors colibrì's `iobench`
//! (file, block MB, iterations, rings, O_DIRECT) so the two are comparable.
//!
//!   cargo run --release -p peregrine-io --example iobench -- FILE BLK_MB ITERS RINGS DIRECT

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
    let blk = blk_mb * 1024 * 1024;

    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for r in 0..rings {
            let total = total.clone();
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
                let n = if direct {
                    rx.read_direct_many(&mut reqs).expect("direct read")
                } else {
                    rx.read_many(&mut reqs).expect("read")
                };
                let got: i64 = n.iter().map(|v| (*v).max(0)).sum();
                total.fetch_add(got as u64, std::sync::atomic::Ordering::Relaxed);
            });
        }
    });
    let dt = t0.elapsed().as_secs_f64();
    let bytes = total.load(std::sync::atomic::Ordering::Relaxed) as f64;
    println!(
        "io_uring{} x{} rings: {} reads x {}MB = {:.1} GB in {:.2}s -> {:.2} GB/s",
        if direct { " O_DIRECT" } else { " buffered" },
        rings,
        rings * iters,
        blk_mb,
        bytes / 1e9,
        dt,
        bytes / 1e9 / dt
    );
}

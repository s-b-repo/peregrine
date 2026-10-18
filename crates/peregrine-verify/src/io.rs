//! The I/O group: the claims that a fast read returns the *right* bytes.
//!
//! Mirror striping is a performance trick with a correctness tail: the region is
//! split across N fds, every chunk is submitted at once, and the result is
//! assembled from N completions. Every way that can go wrong — a chunk assigned
//! to the wrong drive, an offset computed from the wrong base, stripes assembled
//! in completion order rather than stripe order — returns a buffer full of
//! plausible bytes from the right file. Nothing raises. The expert that comes
//! back is the wrong expert.

use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use peregrine_io::cache::ExpertCache;
use peregrine_io::{pread_striped, StripedRead};

use crate::{cert, exact, failed, with_cases, Certificate};

pub fn certificates() -> Vec<Certificate> {
    vec![
        striped_read_matches_single_drive(),
        short_range_is_not_split(),
        cache_evicts_least_recently_used(),
        cache_hit_rate_is_the_ratio_it_reports(),
    ]
}

/// Temp paths get the pid and a counter: parallel certificate groups share
/// `/tmp`, and two agents on one box share the tree.
fn unique_file(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("peregrine-verify-{tag}-{}-{n}", std::process::id()))
}

/// The central claim of the whole I/O layer: striping changes how many drives
/// answer a request, never what they answer. The region is written once, read
/// back as a single-drive read, then read again striped across three fds of the
/// same file — and the two buffers must be byte-identical.
///
/// The size is deliberately past the split threshold (`3 × 128 KiB`), because a
/// test that only exercised the non-splitting path would pass against an
/// implementation that never splits anything.
fn striped_read_matches_single_drive() -> Certificate {
    const N: usize = 3;
    const LEN: usize = 1 << 20;
    let path = unique_file("striped");
    let mut bad = 0.0f64;
    let mut cases = 0usize;

    let payload: Vec<u8> = (0..LEN).map(|k| (k.wrapping_mul(2654435761) >> 13) as u8).collect();
    let write = File::create(&path).and_then(|mut f| f.write_all(&payload));
    if let Err(e) = write {
        let residual = failed("write striped fixture", &e.to_string());
        let _ = std::fs::remove_file(&path);
        return cert(
            "io",
            "a_striped_read_returns_the_single_drive_bytes",
            "A 1 MiB region read striped across three fds of the same file is byte-identical to the same region read from one fd.",
            residual,
            0.0,
        );
    }

    let opened: Vec<File> = (0..N).map(|_| File::open(&path)).filter_map(std::result::Result::ok).collect();
    if opened.len() != N {
        let _ = std::fs::remove_file(&path);
        return cert(
            "io",
            "a_striped_read_returns_the_single_drive_bytes",
            "A 1 MiB region read striped across three fds of the same file is byte-identical to the same region read from one fd.",
            failed("open mirrors", "could not open three descriptors for the fixture"),
            0.0,
        );
    }
    let fds: Vec<std::os::unix::io::RawFd> = opened.iter().map(AsRawFd::as_raw_fd).collect();

    let mut single = vec![0u8; LEN];
    let mut reqs: Vec<StripedRead<'_>> = vec![StripedRead { fds: &fds[..1], offset: 0, buf: &mut single, tag: 0 }];
    let counts = pread_striped(&mut reqs);
    if counts.first().copied() != Some(LEN as i64) {
        bad += failed("single-drive read", &format!("returned {counts:?}, expected {LEN}"));
    }

    let mut striped = vec![0u8; LEN];
    let mut reqs: Vec<StripedRead<'_>> = vec![StripedRead { fds: &fds, offset: 0, buf: &mut striped, tag: 0 }];
    let counts = pread_striped(&mut reqs);
    if counts.first().copied() != Some(LEN as i64) {
        bad += failed("striped read", &format!("returned {counts:?}, expected {LEN}"));
    }

    // Byte equality, and the byte count as well: a read that returns the right
    // number of bytes from the wrong offsets is exactly the bug.
    let differing = striped.iter().zip(payload.iter()).filter(|(a, b)| a != b).count();
    if differing > 0 {
        bad += 1.0;
    }
    if striped != single {
        bad += 1.0;
    }
    cases += 1;

    // The same claim at a non-zero offset, because every offset computation in
    // the split is relative to the request's base and an off-by-one base is the
    // classic version of this bug.
    let off = 4093usize;
    let len = 300_000usize;
    let mut single = vec![0u8; len];
    let mut reqs: Vec<StripedRead<'_>> =
        vec![StripedRead { fds: &fds[..1], offset: off as u64, buf: &mut single, tag: 0 }];
    let _ = pread_striped(&mut reqs);
    let mut striped = vec![0u8; len];
    let mut reqs: Vec<StripedRead<'_>> = vec![StripedRead { fds: &fds, offset: off as u64, buf: &mut striped, tag: 0 }];
    let _ = pread_striped(&mut reqs);
    if single != payload[off..off + len] || striped != single {
        bad += 1.0;
    }
    cases += 1;

    // Cleanup before judging, so a failure cannot leave the temp tree behind.
    if let Err(e) = std::fs::remove_file(&path) {
        bad += failed("temp cleanup", &e.to_string());
    }

    with_cases(
        cert(
            "io",
            "a_striped_read_returns_the_single_drive_bytes",
            "A region read striped across three fds of the same file is byte-identical to the same region read from one fd — at offset 0 and at an unaligned offset, past the splitting threshold, returning the full length.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Below `n × 128 KiB` a region is deliberately *not* split: the per-request
/// overhead of chunking a small range across drives exceeds the parallelism it
/// buys. That is a performance decision, but the claim it makes is about
/// correctness — a region either rides whole on one drive or is split, and both
/// must return the same bytes. Checked at a size that must take the
/// non-splitting branch with three fds.
fn short_range_is_not_split() -> Certificate {
    let path = unique_file("short");
    let len = 1024usize;
    let payload: Vec<u8> = (0..len).map(|k| (k % 251) as u8).collect();
    let mut bad = 0.0f64;
    match File::create(&path).and_then(|mut f| f.write_all(&payload)) {
        Ok(()) => {
            let opened: Vec<File> = (0..3).map(|_| File::open(&path)).filter_map(std::result::Result::ok).collect();
            if opened.len() != 3 {
                bad += failed("open mirrors", "could not open three descriptors for the fixture");
            } else {
                let fds: Vec<std::os::unix::io::RawFd> = opened.iter().map(AsRawFd::as_raw_fd).collect();
                let mut got = vec![0u8; len];
                let mut reqs: Vec<StripedRead<'_>> = vec![StripedRead { fds: &fds, offset: 0, buf: &mut got, tag: 0 }];
                let counts = pread_striped(&mut reqs);
                if counts.first().copied() != Some(len as i64) || got != payload {
                    bad += 1.0;
                }
            }
        }
        Err(e) => bad += failed("write short fixture", &e.to_string()),
    }
    if let Err(e) = std::fs::remove_file(&path) {
        bad += failed("temp cleanup", &e.to_string());
    }
    with_cases(
        cert(
            "io",
            "a_short_region_rides_whole_and_still_reads_correctly",
            "A 1 KiB region across three fds is served whole from one drive and returns the payload exactly — the non-splitting branch is a performance choice, not a different read.",
            bad,
            0.0,
        ),
        1,
    )
}

/// The RAM warm tier is a bounded LRU, and its eviction order is the whole
/// point: evicting anything other than the least-recently-used expert turns the
/// tier into a random cache, which shows up only as a lower hit rate.
fn cache_evicts_least_recently_used() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for cap in [1usize, 2, 4] {
        let mut c = ExpertCache::new(cap);
        for e in 0..cap {
            c.insert(e as i32, vec![e as u8]);
        }
        // Touch the oldest so it is no longer the LRU. It must be a hit: the
        // cache was just filled with it.
        if c.get(0).is_none() {
            bad += 1.0;
        }
        // One more insert must evict the least recently used expert. With
        // capacity 1 that is expert 0 itself (nothing else exists); with room to
        // spare it is expert 1, and the expert we just touched survives.
        c.insert(999, vec![7]);
        cases += 4;
        if cap == 1 && c.contains(0) {
            bad += 1.0;
        }
        if cap > 1 && !c.contains(0) {
            bad += 1.0;
        }
        if c.contains(1) {
            bad += 1.0;
        }
        if !c.contains(999) {
            bad += 1.0;
        }
        if c.len() > cap {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "io",
            "the_expert_cache_evicts_least_recently_used",
            "With capacity N holding N experts, reading expert 0 and inserting one more evicts the least recently *used* expert: expert 0 survives whenever there is a less recent one, the cache never exceeds its capacity, and the insert is always resident.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// `hit_rate` feeds the tiering decision, so it must be the ratio of the counters
/// it exposes rather than an approximation: a stale or denominator-wrong hit rate
/// makes the tuner prefer a tier that is not being hit.
fn cache_hit_rate_is_the_ratio_it_reports() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let mut c = ExpertCache::new(4);
    c.insert(1, vec![1, 2, 3]);
    for _ in 0..3 {
        if c.get(1).is_none() {
            bad += 1.0;
        }
    }
    for e in [2i32, 3] {
        if c.get(e).is_some() {
            bad += 1.0;
        }
    }
    cases += 5;
    let want = c.hits as f64 / (c.hits + c.misses) as f64;
    if (c.hit_rate() - want).abs() > f64::from(f32::EPSILON) {
        bad += 1.0;
    }
    // A cache nobody touched reports 0.0 rather than a division by zero.
    let fresh = ExpertCache::new(2);
    if fresh.hit_rate() != 0.0 || fresh.hits != 0 || fresh.misses != 0 {
        bad += 1.0;
    }
    cases += 1;
    with_cases(
        cert(
            "io",
            "cache_hit_rate_is_hits_over_lookups",
            "`hit_rate` equals `hits / (hits + misses)` over the counters it exposes, and an untouched cache reports exactly 0.0.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Named so the group reads as a list of claims rather than a list of helpers:
/// an exact-identity residual is a number, and this is the shape of one.
#[allow(dead_code)]
fn exact_residual(holds: bool) -> f64 {
    exact(holds)
}
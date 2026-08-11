//! colibrì expert streaming (M2): the I/O lane.
//!
//! - [`ring`] — batched positioned reads. On Linux, io_uring drives a whole ring
//!   of expert-slab reads with one enter syscall (the "reduce syscalls" goal);
//!   [`ring::pread_many`] is the portable fallback and correctness oracle.
//! - [`cache`] — the per-layer LRU expert cache (RAM warm tier).
//! - [`warmcache`] — the byte-budgeted `(layer,expert)` warm cache the concurrent
//!   scheduler consults before streaming (holds raw quantized bytes → hits are
//!   bit-identical to a fresh stream).
//! - [`tier`] — the LFRU hot-store eviction/promotion policy (`c/tier.h`).
//!
//! The concurrent hand-off between this lane and the CPU/GPU compute lanes is
//! the M4 scheduler.

// tier.h is ported with explicit expert-index loops for line-by-line parity.
// Quality gates: io_uring submission is the only (irreducible) `unsafe` here; no
// panicking error handling in library code.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

/// Report a failed best-effort operation (advisory kernel hints, cleanup,
/// shutdown signalling). Such failures are correctness-neutral by design —
/// they must never abort inference — but they should not vanish either:
/// set `COLI_DEBUG=1` to surface them on stderr.
pub fn note_advisory_err(op: &str, err: &dyn std::fmt::Display) {
    if std::env::var_os("COLI_DEBUG").is_some() {
        eprintln!("[peregrine advisory] {op}: {err}");
    }
}

pub mod cache;
pub mod mem;
pub mod perf;
pub mod ring;
pub mod sensors;
pub mod slab;
pub mod tier;
pub mod topo;
pub mod warmcache;

pub use cache::ExpertCache;
pub use mem::{
    advise_dontneed, advise_dontneed_slice, advise_hugepages, advise_hugepages_slice,
    bind_local_if_enabled, current_numa_node, hugepage_disabled, mbind_to_node,
    pin_current_thread, wire_resident, Wired,
};
pub use perf::PerfCounter;
pub use sensors::{energy_uj, max_temp_c, EnergyMeter};
pub use ring::{
    pread_many, pread_many_threaded, probe_direct, read_file, OwnedReadReq, Reactor, ReadReq,
    RegionDone,
};
pub use slab::{align_down, align_up, AlignedBuf, Bytes, SlabHandle, SlabPool, ALIGN};
pub use tier::{decay, lfru_score, pick_lfru, pick_swap, Swap};
pub use warmcache::{CacheHit, CompressedSlab, ExpertSlab, PreparedInsert, ResidencyHint, WarmCache};

/// Cap glibc's per-thread malloc arenas (`M_ARENA_MAX=2`) so the many worker
/// threads streaming ~18.9 MB expert buffers don't inflate RSS through arena
/// fragmentation — the reason the engine otherwise needs `MALLOC_ARENA_MAX=2` in
/// the environment. Call **once at process start**, before spawning the I/O and
/// compute worker pools. Returns whether the cap was applied; a no-op (`false`)
/// when the user already set `MALLOC_ARENA_MAX`, set `COLI_NO_ARENA_CAP`, or off
/// glibc. This is the pragmatic form of B2's goal; a reusable landing-buffer pool
/// (bounding the allocations themselves) is blocked by expert bytes being consumed
/// into `QtWeight`, and buys little on top of this arena cap.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn cap_malloc_arenas() -> bool {
    if std::env::var_os("MALLOC_ARENA_MAX").is_some() || std::env::var_os("COLI_NO_ARENA_CAP").is_some() {
        return false; // respect an explicit user setting / opt-out
    }
    // SAFETY: `mallopt` is a thread-safe glibc tuning call; `M_ARENA_MAX` bounds the
    // number of allocation arenas. Returns 1 on success.
    unsafe { libc::mallopt(libc::M_ARENA_MAX, 2) == 1 }
}

/// Non-glibc: no per-thread arenas to cap.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn cap_malloc_arenas() -> bool {
    false
}

#[cfg(test)]
mod arena_tests {
    #[test]
    fn cap_malloc_arenas_runs() {
        // Must link and run without panicking; the boolean result depends on the
        // platform and environment (so it is not asserted).
        super::cap_malloc_arenas();
    }
}

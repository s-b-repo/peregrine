//! colibrì expert streaming (M2): the I/O lane.
//!
//! - [`ring`] — batched positioned reads. On Linux, io_uring drives a whole ring
//!   of expert-slab reads with one enter syscall (the "reduce syscalls" goal);
//!   [`ring::pread_many`] is the portable fallback and correctness oracle.
//! - [`cache`] — the per-layer LRU expert cache (RAM warm tier).
//! - [`tier`] — the LFRU hot-store eviction/promotion policy (`c/tier.h`).
//!
//! The concurrent hand-off between this lane and the CPU/GPU compute lanes is
//! the M4 scheduler.

// tier.h is ported with explicit expert-index loops for line-by-line parity.
#![allow(clippy::needless_range_loop)]

pub mod cache;
pub mod ring;
pub mod tier;

pub use cache::ExpertCache;
pub use ring::{pread_many, ReadReq};
pub use tier::{decay, lfru_score, pick_lfru, pick_swap, Swap};

#[cfg(target_os = "linux")]
pub use ring::Reactor;

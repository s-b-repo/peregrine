//! Stable identifiers and a logical clock.
//!
//! Everything addressable gets a content-derived id, so the same material
//! ingested twice lands on the same span and the same fact instead of silently
//! duplicating. Time is a logical counter rather than wall-clock: node
//! timestamps only need a total order for staleness comparisons, and a logical
//! clock keeps runs reproducible.

use std::cell::Cell;
use std::fmt::Write as _;

/// FNV-1a, run twice with different offset bases to widen the digest.
///
/// Not a cryptographic hash and not trying to be — these ids address content
/// inside one process, they do not authenticate it.
fn fnv1a(bytes: &[u8], offset: u64) -> u64 {
    let mut hash = offset;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

const OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
const OFFSET_B: u64 = 0x9e37_79b9_7f4a_7c15;

/// Hex digest of the parts, separated so `("ab", "c")` and `("a", "bc")` differ.
pub fn digest(parts: &[&str], length: usize) -> String {
    let mut buf = Vec::new();
    for part in parts {
        buf.extend_from_slice(part.as_bytes());
        buf.push(0x1f);
    }
    let (a, b) = (fnv1a(&buf, OFFSET_A), fnv1a(&buf, OFFSET_B));
    let mut out = String::with_capacity(32);
    let _ = write!(out, "{a:016x}{b:016x}");
    out.truncate(length);
    out
}

pub fn make_id(prefix: &str, parts: &[&str]) -> String {
    format!("{prefix}_{}", digest(parts, 12))
}

/// Monotonic logical clock. `tick()` is the timestamp stamped on nodes.
#[derive(Debug, Default)]
pub struct Clock {
    now: Cell<u64>,
}

impl Clock {
    pub fn new() -> Self {
        Self { now: Cell::new(0) }
    }

    pub fn tick(&self) -> u64 {
        let next = self.now.get() + 1;
        self.now.set(next);
        next
    }

    pub fn now(&self) -> u64 {
        self.now.get()
    }

    /// Resume a clock after loading a persisted store.
    pub fn restore(&self, value: u64) {
        self.now.set(value);
    }
}

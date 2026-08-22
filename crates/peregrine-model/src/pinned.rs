//! The host end of the disk → GPU lane.
//!
//! Without this module a weight byte destined for VRAM makes this trip:
//!
//! ```text
//! NVMe ──(io_uring)──▶ pageable Vec ──(blocking cudaMemcpy)──▶ VRAM
//! ```
//!
//! The second hop is worse than it looks. CUDA cannot DMA out of pageable
//! memory, so the driver stages it through an internal pinned bounce buffer —
//! an extra copy of every byte — and the blocking form means the caller waits
//! out the whole multi-megabyte transfer with nothing overlapping it.
//!
//! With the hook installed the trip becomes two DMAs and no copy:
//!
//! ```text
//! NVMe ──(io_uring, O_DIRECT)──▶ pinned host page ──(cudaMemcpyAsync)──▶ VRAM
//! ```
//!
//! The mechanism is deliberately small. [`peregrine_io::AlignedBuf`] already
//! allocates page-aligned (`ALIGN = 4096`) and already owns its allocation's
//! lifetime, so `cudaHostRegister` pins it **in place**: no second allocator, no
//! change to the read path, and an exactly-matched unpin in the buffer's own
//! `Drop`. `peregrine-io` holds a function-pointer pair rather than knowing
//! anything about CUDA; this module is what fills it in.
//!
//! What this does **not** do: io_uring cannot DMA from NVMe directly into VRAM.
//! That is GPUDirect Storage (`nvidia-fs` + `libcufile`), a different API
//! entirely, not an io_uring opcode. The chain above is the io_uring-native
//! optimum — zero userspace copies, one hop through pinned host pages.

/// Live pinning state, re-exported from the crate that owns the FFI.
#[cfg(feature = "cuda")]
pub use peregrine_cuda::PinStats;

/// Live pinning state on a build with no CUDA: always empty.
#[cfg(not(feature = "cuda"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinStats {
    pub buffers: usize,
    pub bytes: u64,
    pub declined: usize,
    pub ever: usize,
}

/// Current pinning counters — how many aligned buffers are registered with
/// CUDA, how many bytes that is, and how many registrations the driver refused.
/// A large `declined` with `buffers` near zero is the `RLIMIT_MEMLOCK` story:
/// the lane is nominally on and doing nothing.
#[cfg(feature = "cuda")]
pub fn stats() -> PinStats {
    peregrine_cuda::pin_stats()
}

#[cfg(not(feature = "cuda"))]
pub fn stats() -> PinStats {
    PinStats::default()
}

/// Whether the pinned lane is wanted (`COLI_GPU_PINNED`, default on).
///
/// Off restores the historical path exactly: pageable host buffers and a
/// blocking `cudaMemcpy`. It is the A/B control, and the escape hatch for a host
/// where `cudaHostRegister` misbehaves.
pub fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("COLI_GPU_PINNED").as_deref(), Ok("0") | Ok("false")))
}

/// Install the CUDA pin hook into `peregrine-io`'s aligned allocator.
///
/// Idempotent and cheap to call more than once — the hook is set-once. Returns
/// whether the pinned lane is now active. Call it **before** any aligned buffer
/// is allocated: a buffer that already exists is not retroactively pinned, so
/// installing late silently gives a lane that only half works.
#[cfg(feature = "cuda")]
pub fn install() -> bool {
    if !enabled() || !peregrine_cuda::is_available() {
        return false;
    }
    // `set_pin_hook` returns false if one is already installed, which on a
    // second call is this same hook — so the lane is active either way.
    peregrine_io::set_pin_hook(peregrine_cuda::pin_host, peregrine_cuda::unpin_host);
    peregrine_io::pin_hook_installed()
}

/// No CUDA in this build: there is nothing to pin against, and the aligned
/// allocator keeps its historical behaviour untouched.
#[cfg(not(feature = "cuda"))]
pub fn install() -> bool {
    false
}

#[cfg(test)]
mod tests {
    /// The knob must have exactly one reading, and `install` must be safe to
    /// call on a CPU-only host (where it declines rather than erroring).
    #[test]
    fn install_is_safe_without_a_gpu() {
        // On a CPU-only build this returns false without touching the
        // allocator; on a CUDA host it installs the hook. Either way it must not
        // panic and must agree with `enabled()`.
        let active = super::install();
        assert!(!active || super::enabled(), "the lane cannot be active while the knob is off");
    }

    #[test]
    fn stats_start_empty_or_consistent() {
        let s = super::stats();
        // bytes and buffers move together: no buffers means no bytes.
        assert!(s.buffers > 0 || s.bytes == 0, "byte count without any pinned buffer: {s:?}");
    }

    /// The chain the whole lane rests on: install the hook, allocate an aligned
    /// buffer through `peregrine-io`, and the buffer comes back **pinned**.
    ///
    /// Worth its own test because every link is invisible from the outside. The
    /// hook is a function pointer in another crate, pinning happens inside
    /// `AlignedBuf::with_capacity`, and the only observable is `is_pinned()`. A
    /// break anywhere in that chain does not fail a build or a load — it just
    /// silently leaves every upload on the pageable bounce path.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_aligned_buffer_over_the_floor_comes_back_pinned() {
        if !super::install() {
            eprintln!("skipping: no CUDA device, so there is nothing to pin against");
            return;
        }
        // Over MIN_PIN_BYTES, so it is not declined by policy.
        const LEN: usize = 4 << 20;
        let Some(buf) = peregrine_io::AlignedBuf::with_capacity(LEN) else {
            eprintln!("skipping: {LEN}-byte aligned allocation failed");
            return;
        };
        let s = super::stats();
        if buf.is_pinned() {
            assert!(s.buffers > 0, "a pinned buffer must show in the stats: {s:?}");
            assert!(s.bytes >= LEN as u64, "its bytes must show in the stats: {s:?}");
        } else {
            // A driver refusal is a legitimate outcome; a silently-missing hook
            // is not. Distinguish them.
            assert!(
                s.declined > 0,
                "the buffer is not pinned and nothing was declined — the hook did not run at all: {s:?}"
            );
        }
    }

    /// A buffer under the floor must be declined *by policy*, without reaching
    /// the driver and without being counted as a driver refusal — otherwise the
    /// `declined` counter, which is how an operator diagnoses a dead lane, would
    /// read as alarming on a perfectly healthy run.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_small_buffer_is_declined_without_alarming_the_counters() {
        if !super::install() {
            return;
        }
        let before = super::stats();
        let Some(buf) = peregrine_io::AlignedBuf::with_capacity(4096) else {
            return;
        };
        assert!(!buf.is_pinned(), "a 4 KiB buffer is under the pin floor");
        assert_eq!(
            super::stats().declined,
            before.declined,
            "a policy decline must not be counted as a driver refusal"
        );
    }
}

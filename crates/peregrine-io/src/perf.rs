//! Hardware performance-counter access via `perf_event_open(2)`.
//!
//! One counter type for now: LLC (last-level cache) misses, the signal the
//! prefetch-distance tuner wants ("rising L3 misses → widen prefetch"). The
//! syscall needs `kernel.perf_event_paranoid <= 2` (self-monitoring); on
//! refusal every constructor returns `None` and consumers silently skip the
//! feedback — the counter is an optimization input, never a dependency.
//!
//! Linux-only; the non-Linux build exposes the same API with `None`-only
//! constructors so consumers stay `cfg`-free.

/// `struct perf_event_attr`, first-version layout (`PERF_ATTR_SIZE_VER0` =
/// 64 bytes) — a stable kernel ABI since 2.6.32, hand-declared because the
/// pinned `libc` doesn't export it for gnu targets. The kernel accepts any
/// declared `size` it knows; VER0 carries everything an enabled counting
/// event needs.
#[cfg(target_os = "linux")]
#[repr(C)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period: u64, // union with sample_freq
    sample_type: u64,
    read_format: u64,
    flags: u64, // bitfield: bit0 disabled, bit5 exclude_kernel, bit6 exclude_hv, …
    wakeup_events: u32, // union with wakeup_watermark
    bp_type: u32,
    bp_addr: u64, // union with config1
}

#[cfg(target_os = "linux")]
const PERF_TYPE_HARDWARE: u32 = 0;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_MISSES: u64 = 3;
#[cfg(target_os = "linux")]
const FLAG_EXCLUDE_KERNEL: u64 = 1 << 5;
#[cfg(target_os = "linux")]
const FLAG_EXCLUDE_HV: u64 = 1 << 6;

/// An open per-thread hardware counter. Reads are cumulative since open (or
/// the last [`Self::reset`]). Closes its fd on drop.
pub struct PerfCounter {
    #[cfg(target_os = "linux")]
    fd: i32,
}

impl PerfCounter {
    /// Open an LLC-miss counter following the *calling thread* (pid = 0,
    /// cpu = -1). `None` when the kernel refuses (paranoid level, seccomp,
    /// missing PMU — e.g. most VMs) or on non-Linux.
    pub fn open_cache_misses() -> Option<PerfCounter> {
        #[cfg(target_os = "linux")]
        {
            let attr = PerfEventAttr {
                type_: PERF_TYPE_HARDWARE,
                size: std::mem::size_of::<PerfEventAttr>() as u32, // 64 = VER0
                config: PERF_COUNT_HW_CACHE_MISSES,
                sample_period: 0,
                sample_type: 0,
                read_format: 0,
                // start enabled; user-space only (lower paranoid bar)
                flags: FLAG_EXCLUDE_KERNEL | FLAG_EXCLUDE_HV,
                wakeup_events: 0,
                bp_type: 0,
                bp_addr: 0,
            };
            debug_assert_eq!(std::mem::size_of::<PerfEventAttr>(), 64, "VER0 layout");
            // SAFETY: `attr` is a fully-initialized VER0 attr the kernel copies
            // during the syscall; nothing is retained past the call.
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_perf_event_open,
                    &attr as *const PerfEventAttr,
                    0i32,  // pid: calling thread
                    -1i32, // cpu: any
                    -1i32, // group_fd: none
                    0u64,  // flags
                ) as i32
            };
            if fd < 0 {
                return None;
            }
            Some(PerfCounter { fd })
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// The cumulative count since open / last reset. `None` on a short or
    /// failed read (counter multiplexed away, fd revoked).
    pub fn read(&self) -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            let mut buf = [0u8; 8];
            // SAFETY: `fd` is a live perf fd we own; reading 8 bytes into a
            // stack buffer of exactly 8 bytes.
            let n = unsafe { libc::read(self.fd, buf.as_mut_ptr() as *mut libc::c_void, 8) };
            if n != 8 {
                return None;
            }
            Some(u64::from_ne_bytes(buf))
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Zero the counter (ioctl `PERF_EVENT_IOC_RESET`). Best-effort.
    pub fn reset(&self) {
        #[cfg(target_os = "linux")]
        {
            // PERF_EVENT_IOC_RESET = _IO('$', 3)
            const PERF_EVENT_IOC_RESET: libc::c_ulong = 0x2403;
            // SAFETY: valid owned fd; the ioctl takes no pointer argument.
            unsafe {
                let _ = libc::ioctl(self.fd, PERF_EVENT_IOC_RESET, 0);
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for PerfCounter {
    fn drop(&mut self) {
        // SAFETY: closing the fd we opened; double-close impossible (unique owner).
        unsafe {
            let _ = libc::close(self.fd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_read_reset_never_panic() {
        // The kernel may refuse (CI containers, paranoid ≥ 3, no PMU) — all we
        // assert is the no-panic contract and that a granted counter yields a
        // monotonic read.
        let Some(c) = PerfCounter::open_cache_misses() else {
            return; // refused → nothing further to check on this box
        };
        // Touch some memory to generate misses (best effort).
        let v = vec![1u8; 1 << 20];
        let sum: u64 = v.iter().map(|&b| b as u64).sum();
        assert!(sum > 0);
        let a = c.read();
        assert!(a.is_some(), "granted counter must be readable");
        c.reset();
        let b = c.read();
        assert!(b.is_some());
    }
}

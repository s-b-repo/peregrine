//! Memory hints for hot buffers.
//!
//! [`advise_hugepages`] asks the kernel to back a virtual-address range with
//! transparent huge pages (2 MB on x86_64). It is a soft hint — the kernel may
//! decline, and the mapping stays valid either way, so a failure returns `false`
//! without surfacing an error. On non-Linux it is a no-op that returns `false`.
//!
//! Gated on `COLI_HUGEPAGE` at the call sites (default on). Callers should skip
//! ranges smaller than the huge-page size — a 2 MB hint on a 4 KB buffer buys
//! nothing and just spends a syscall.

/// Advise the kernel to back `[ptr, ptr+len)` with transparent huge pages
/// (`MADV_HUGEPAGE`). No-op on non-Linux or when disabled by env. Returns
/// `true` on kernel-accepted advice, `false` on rejection / disabled / non-Linux.
///
/// # Safety
///
/// `ptr` must be a live allocation of at least `len` bytes, owned by the caller
/// for the duration of the call. The advice does not mutate contents.
pub unsafe fn advise_hugepages(ptr: *mut u8, len: usize) -> bool {
    if len == 0 || ptr.is_null() {
        return false;
    }
    if hugepage_disabled() {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // MADV_HUGEPAGE requires a page-aligned start; if it isn't, EINVAL and
        // we return false. Do NOT widen backwards — the pages before `ptr` may
        // belong to a different allocation, and MADV_HUGEPAGE / _DONTNEED
        // silently apply to those too (with _DONTNEED being actively
        // destructive). So narrow the range to whole pages *inside*
        // `[ptr, ptr+len)`.
        let (base, adj_len) = narrow_to_full_pages(ptr, len);
        if adj_len == 0 {
            return false;
        }
        // SAFETY: `[base, base+adj_len)` is a subset of the caller-owned range.
        let rc = unsafe { libc::madvise(base as *mut libc::c_void, adj_len, libc::MADV_HUGEPAGE) };
        rc == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        false
    }
}

/// Narrow `[ptr, ptr+len)` to the largest subrange whose start and end are
/// both aligned to the 4 KB page. Returns `(base, len_in_bytes)`; `len_in_bytes`
/// is 0 when there is no full page inside the range (which is when the caller
/// should skip the advice entirely).
#[cfg(target_os = "linux")]
fn narrow_to_full_pages(ptr: *mut u8, len: usize) -> (usize, usize) {
    const PAGE: usize = 4096;
    let start = ptr as usize;
    let end = start.saturating_add(len);
    let aligned_start = (start + PAGE - 1) & !(PAGE - 1);
    let aligned_end = end & !(PAGE - 1);
    if aligned_start >= aligned_end {
        (aligned_start, 0)
    } else {
        (aligned_start, aligned_end - aligned_start)
    }
}

/// Advise the kernel that `[ptr, ptr+len)` will not be needed soon
/// (`MADV_DONTNEED`) — releases anonymous pages back to the OS, and drops
/// page-cache references for file-backed pages. Called on the tail of streamed
/// buffers to keep RSS flat under long-running workloads. Same failure/no-op
/// semantics as [`advise_hugepages`].
///
/// # Safety
///
/// Same constraints as [`advise_hugepages`]. Note that on file-backed mappings
/// `MADV_DONTNEED` is destructive to dirty writes; only call on read-only or
/// throw-away buffers.
pub unsafe fn advise_dontneed(ptr: *mut u8, len: usize) -> bool {
    if len == 0 || ptr.is_null() {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // Narrow to whole pages inside the range — see the caveat on
        // [`advise_hugepages`]. DONTNEED is destructive on anonymous pages, so
        // widening to the enclosing pages (which may belong to another
        // allocation) would silently corrupt it.
        let (base, adj_len) = narrow_to_full_pages(ptr, len);
        if adj_len == 0 {
            return false;
        }
        // SAFETY: `[base, base+adj_len)` is a subset of the caller-owned range.
        let rc = unsafe { libc::madvise(base as *mut libc::c_void, adj_len, libc::MADV_DONTNEED) };
        rc == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        false
    }
}

/// Whether the hugepage hint is disabled by env. Cheap to call; not memoized
/// because the env may change between processes and callers only ask on rare
/// paths (allocation, ring registration).
pub fn hugepage_disabled() -> bool {
    std::env::var_os("COLI_HUGEPAGE").is_some_and(|v| v == "0")
}

/// Safe slice wrapper for [`advise_hugepages`]. The slice's lifetime bounds
/// the call, so no aliasing/UAF concern. Callers should only bother for slices
/// at least one huge page long (2 MB on x86_64) — smaller ones don't benefit.
pub fn advise_hugepages_slice(slice: &mut [u8]) -> bool {
    let len = slice.len();
    if len == 0 {
        return false;
    }
    // SAFETY: the mutable slice is live for the duration of the call and points
    // into an allocation the caller owns; advice is read-only from our viewpoint.
    unsafe { advise_hugepages(slice.as_mut_ptr(), len) }
}

/// Safe slice wrapper for [`advise_dontneed`]. Same constraints; only call on
/// throw-away or read-only buffers (see the raw function's caveat).
pub fn advise_dontneed_slice(slice: &mut [u8]) -> bool {
    let len = slice.len();
    if len == 0 {
        return false;
    }
    // SAFETY: as above.
    unsafe { advise_dontneed(slice.as_mut_ptr(), len) }
}

/// Pin the *current* thread to a single logical CPU. Best-effort: returns
/// `true` on kernel-accepted binding, `false` on rejection / non-Linux. No-op
/// when `COLI_NUMA_PIN=0`.
pub fn pin_current_thread(cpu: u32) -> bool {
    if matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("0") | Ok("false")) {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `cpu_set_t` is initialized via `CPU_ZERO`; we then set exactly
        // one bit and call the syscall which reads (not owns) the set. The syscall
        // is documented thread-safe.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu as usize, &mut set);
            let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            rc == 0
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        false
    }
}

/// Bind `[ptr, ptr+len)` to a specific NUMA node (`MPOL_BIND`). Best-effort;
/// returns `false` on rejection / non-Linux / disabled.
///
/// # Safety
///
/// `ptr` must be a live allocation of at least `len` bytes owned by the caller
/// for the duration of the call. Only bind ranges you own — a shared mapping
/// gets its policy stomped.
pub unsafe fn mbind_to_node(ptr: *mut u8, len: usize, node: u32) -> bool {
    if len == 0 || ptr.is_null() {
        return false;
    }
    if matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("0") | Ok("false")) {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // MPOL_BIND == 2. `mbind` isn't in libc — call it via `syscall`.
        const MPOL_BIND: libc::c_ulong = 2;
        // The nodemask is a bit-mask indexed by NUMA node id. One u64 covers
        // 64 nodes — plenty for any single machine.
        let mut mask: u64 = 0;
        if node < 64 {
            mask = 1u64 << node;
        }
        // SAFETY: caller guarantees the range; the syscall reads `mask` as a
        // 64-bit unsigned; passing a bit width of 64 tells the kernel we only
        // supply one u64.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                ptr as usize,
                len,
                MPOL_BIND,
                (&mask as *const u64) as usize,
                64u64, // maxnode: 64 bits of mask
                0u64,  // flags
            )
        };
        rc == 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len, node);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advise_on_a_regular_vec_does_not_crash() {
        let mut v = vec![0u8; 1 << 20]; // 1 MB — smaller than a hugepage but valid
        // SAFETY: the vec is live for the duration of the call.
        let _ = unsafe { advise_hugepages(v.as_mut_ptr(), v.len()) };
        let _ = unsafe { advise_dontneed(v.as_mut_ptr(), v.len()) };
    }

    #[test]
    fn null_and_zero_len_are_noops() {
        // SAFETY: null / zero-len are the documented no-op cases.
        assert!(!unsafe { advise_hugepages(std::ptr::null_mut(), 0) });
        assert!(!unsafe { advise_dontneed(std::ptr::null_mut(), 100) });
    }

    #[test]
    fn env_disable_honored() {
        // Not asserting the exact return — only that setting the env is respected
        // and does not panic.
        std::env::set_var("COLI_HUGEPAGE", "0");
        assert!(hugepage_disabled());
        std::env::remove_var("COLI_HUGEPAGE");
        assert!(!hugepage_disabled());
    }
}

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
        let (_ptr, _len) = (ptr, len); // hint doesn't exist off-Linux; documented no-op
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
        let (_ptr, _len) = (ptr, len); // hint doesn't exist off-Linux; documented no-op
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

/// What [`wire_resident`] did, so a caller can report it honestly rather than
/// claiming a guarantee the kernel did not give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Wired {
    /// Not asked for (`COLI_MLOCK` unset), or not Linux.
    Skipped,
    /// The resident set is wired. `bytes` is `VmLck` read back from
    /// `/proc/self/status` — the kernel's number, not ours.
    Locked { bytes: u64 },
    /// The kernel refused. `limit` is the `RLIMIT_MEMLOCK` soft limit, because
    /// that is almost always the reason and the message is useless without it.
    Refused { errno: i32, limit: u64 },
}

/// Wire the process's *current* resident pages into RAM (`mlockall(MCL_CURRENT)`),
/// so the kernel cannot page out the model trunk under later memory pressure.
///
/// **This does not raise the memory ceiling; it removes variance.** A streaming MoE
/// engine spends its life with a large resident trunk and a cache sized to whatever
/// is left, which is precisely the shape that invites the kernel to reclaim trunk
/// pages to grow the page cache — and every reclaimed trunk page is re-read on the
/// next token, at disk speed, in the middle of the critical path. Wiring the trunk
/// makes that impossible. It cannot make a model that does not fit, fit; a machine
/// that was going to swap will now fail honestly instead, which is the better of the
/// two outcomes.
///
/// **`MCL_CURRENT` and deliberately not `MCL_FUTURE`.** Called after the resident
/// weights are loaded and before the warm cache fills, it wires the trunk and leaves
/// every later allocation — cache slabs, KV, streaming buffers — ordinary reclaimable
/// memory. That asymmetry is the entire point: the cache is *supposed* to be the part
/// the kernel can take back. `MCL_FUTURE` would wire the cache too and convert a
/// gentle slowdown into an allocation failure.
///
/// Opt-in via `COLI_MLOCK=1`, because it needs `RLIMIT_MEMLOCK` headroom that most
/// desktop defaults do not grant. Best-effort: a refusal is reported, never fatal.
pub fn wire_resident() -> Wired {
    if !matches!(std::env::var("COLI_MLOCK").as_deref(), Ok("1") | Ok("true")) {
        return Wired::Skipped;
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `mlockall` takes no pointers and mutates no process memory; it
        // only changes the paging policy of the existing address space.
        let rc = unsafe { libc::mlockall(libc::MCL_CURRENT) };
        if rc == 0 {
            return Wired::Locked { bytes: vm_locked_bytes().unwrap_or(0) };
        }
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        Wired::Refused { errno, limit: memlock_limit() }
    }
    #[cfg(not(target_os = "linux"))]
    {
        Wired::Skipped
    }
}

/// `VmLck` from `/proc/self/status`, in bytes — how much the kernel says is wired.
/// Read back rather than assumed, because `mlockall` succeeding does not by itself
/// say how much it locked.
#[cfg(target_os = "linux")]
fn vm_locked_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmLck:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// The `RLIMIT_MEMLOCK` soft limit in bytes, `u64::MAX` for unlimited.
#[cfg(target_os = "linux")]
fn memlock_limit() -> u64 {
    // SAFETY: `getrlimit` writes into the provided, fully-owned struct.
    unsafe {
        let mut rl: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) == 0 {
            rl.rlim_cur as u64
        } else {
            0
        }
    }
}

/// Pin the *current* thread to a single logical CPU. Best-effort: returns
/// `true` on kernel-accepted binding, `false` on rejection / non-Linux. No-op
/// when `COLI_NUMA_PIN=0`.
#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpu: u32) -> bool {
    if matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("0") | Ok("false")) {
        return false;
    }
    // `CPU_SET` indexes the fixed-size `cpu_set_t` bit array without any bounds
    // check of its own, so a cpu id at or past its capacity would be an
    // out-of-bounds write. Reachable on very large hosts and from sparse sysfs
    // cpu ids, and this function's contract is "best effort, false when it
    // cannot pin".
    const CPU_SETSIZE_BITS: u32 = (8 * std::mem::size_of::<libc::cpu_set_t>()) as u32;
    if cpu >= CPU_SETSIZE_BITS {
        return false;
    }
    // SAFETY: `cpu_set_t` is initialized via `CPU_ZERO`; `cpu` is bounds-checked
    // above, so we set exactly one in-range bit, and the syscall reads (not
    // owns) the set. The syscall is documented thread-safe.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu as usize, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        rc == 0
    }
}

/// No affinity syscall exists off Linux, so nothing is ever pinned. The
/// signature is cfg-split (rather than binding the argument away in the body)
/// so the unused parameter is declared as such.
#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread(_cpu: u32) -> bool {
    false
}

/// The NUMA node the calling thread is currently running on, from
/// `sched_getcpu(3)` + the topology probe. `None` on non-Linux, on syscall
/// failure, or when the CPU isn't in any discovered node (shouldn't happen on
/// a well-formed sysfs).
pub fn current_numa_node() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sched_getcpu takes no pointers; returns the current CPU or -1.
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            return None;
        }
        let cpu = cpu as u32;
        crate::topo::snapshot().numa.iter().find(|n| n.cpus.contains(&cpu)).map(|n| n.id)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// NUMA-bind a freshly-allocated large buffer to the calling thread's current
/// node — the "first-touch made explicit" policy for the streaming landing
/// buffers, so a ring thread's DMA targets live on its own node. Opt-in
/// (`COLI_NUMA_PIN=1`), multi-node only, and only worth it for buffers ≥ 2 MB.
/// Best-effort: any failure leaves the default first-touch policy in place.
///
/// # Safety
///
/// Same contract as [`mbind_to_node`]: `[ptr, ptr+len)` must be a live,
/// caller-owned allocation.
pub unsafe fn bind_local_if_enabled(ptr: *mut u8, len: usize) -> bool {
    if len < 2 * 1024 * 1024 {
        return false;
    }
    if !matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("1") | Ok("true")) {
        return false;
    }
    if !crate::topo::snapshot().multi_numa() {
        return false; // single node — binding is a no-op with syscall cost
    }
    let Some(node) = current_numa_node() else { return false };
    // SAFETY: forwarded caller contract.
    unsafe { mbind_to_node(ptr, len, node) }
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
        let (_ptr, _len, _node) = (ptr, len, node); // mbind doesn't exist off-Linux; documented no-op
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advise_on_a_regular_vec_does_not_crash() {
        let mut v = vec![0u8; 1 << 20]; // 1 MB — smaller than a hugepage but valid
        // SAFETY: the vec is live for the duration of the call. The bool results
        // are informational (kernel may decline the hints) — not asserted.
        unsafe { advise_hugepages(v.as_mut_ptr(), v.len()) };
        unsafe { advise_dontneed(v.as_mut_ptr(), v.len()) };
    }

    #[test]
    fn null_and_zero_len_are_noops() {
        // SAFETY: null / zero-len are the documented no-op cases.
        assert!(!unsafe { advise_hugepages(std::ptr::null_mut(), 0) });
        assert!(!unsafe { advise_dontneed(std::ptr::null_mut(), 100) });
    }

    #[test]
    fn wiring_is_opt_in_and_never_fatal() {
        // Unset (the default) must be `Skipped`, not an attempt: `mlockall` on a
        // desktop `RLIMIT_MEMLOCK` typically fails, and a failure on a path nobody
        // asked for would be noise at best.
        std::env::remove_var("COLI_MLOCK");
        assert_eq!(wire_resident(), Wired::Skipped);
        // Asked for: whatever the kernel says, the call returns rather than aborts.
        // Both outcomes are legitimate here — CI containers usually refuse — so the
        // assertion is on the shape, which is the contract callers depend on.
        std::env::set_var("COLI_MLOCK", "1");
        let w = wire_resident();
        std::env::remove_var("COLI_MLOCK");
        #[cfg(target_os = "linux")]
        assert!(
            matches!(w, Wired::Locked { .. } | Wired::Refused { .. }),
            "COLI_MLOCK=1 on Linux must attempt the lock, got {w:?}"
        );
        #[cfg(not(target_os = "linux"))]
        assert_eq!(w, Wired::Skipped, "there is no mlockall to attempt off Linux");
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

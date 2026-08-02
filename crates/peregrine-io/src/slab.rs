//! Page-aligned buffers for O_DIRECT streaming.
//!
//! O_DIRECT reads require the offset, length, **and** buffer address to be
//! aligned to the device's logical block size. [`AlignedBuf`] provides a
//! 4096-aligned heap buffer (a safe superset of 512/4096-byte NVMe sectors and
//! dm-crypt/LUKS logical sectors), and [`SlabPool`] hands them out from a bounded
//! free-list so the streaming path never allocates per read — which also keeps
//! peak RAM flat on memory-contended machines.
//!
//! This is the only module that performs aligned allocation; the `unsafe` is
//! isolated behind the safe [`AlignedBuf`] API. Pure Rust, no io_uring — it
//! compiles on every target (the O_DIRECT *reads* that use it are Linux-only).

use std::alloc::Layout;
use std::ptr::NonNull;

/// Alignment for O_DIRECT: offset, length, and buffer must all be multiples of
/// this. 4096 covers 512- and 4096-byte NVMe/dm-crypt sectors and the page size.
pub const ALIGN: usize = 4096;

/// Round `x` down to a multiple of `a` (which must be a power of two).
pub const fn align_down(x: u64, a: u64) -> u64 {
    x & !(a - 1)
}

/// Round `x` up to a multiple of `a` (which must be a power of two). Assumes no
/// overflow — file offsets are far below `u64::MAX - a`.
pub const fn align_up(x: u64, a: u64) -> u64 {
    (x + (a - 1)) & !(a - 1)
}

/// `usize` variant of [`align_up`] for buffer-capacity math.
pub const fn align_up_usize(x: usize, a: usize) -> usize {
    (x + (a - 1)) & !(a - 1)
}

/// A heap buffer whose base address and length are both multiples of [`ALIGN`].
/// RAII: frees its allocation on drop.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize, // multiple of ALIGN
    layout: Layout,
}

// SAFETY: `AlignedBuf` uniquely owns its heap allocation (no interior aliasing,
// no thread-affinity), so moving it between threads is sound — exactly like
// `Box<[u8]>`. The streaming reactor lives behind a `Mutex` and hands these
// buffers around within one locked call, so it must be `Send`.
unsafe impl Send for AlignedBuf {}

// SAFETY: a shared `&AlignedBuf` only exposes `as_slice()` (immutable bytes) — the
// uniquely-owned allocation has no interior mutability, so concurrent shared reads
// from multiple threads are sound, exactly like `Box<[u8]>: Sync`. Needed so a
// streamed `QtWeight` holding an aligned `Bytes` stays `Sync` for the data-parallel
// matmul pool (`peregrine-par`).
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Allocate an [`ALIGN`]-aligned buffer of at least `cap` bytes (rounded up to
    /// a multiple of `ALIGN`). Returns `None` on a bad layout or allocation
    /// failure — never panics (honoring the crate's no-panic gate).
    pub fn with_capacity(cap: usize) -> Option<AlignedBuf> {
        let len = align_up_usize(cap.max(1), ALIGN);
        let layout = Layout::from_size_align(len, ALIGN).ok()?;
        // SAFETY: `layout` has non-zero size (>= ALIGN) and a valid power-of-two
        // alignment. `alloc_zeroed` returns a fresh, uniquely-owned allocation or
        // null; null → `None` below (no `handle_alloc_error` panic).
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw)?;
        debug_assert_eq!(ptr.as_ptr() as usize % ALIGN, 0, "allocator honored the layout alignment");
        // Ask the kernel to back the range with a transparent huge page when the
        // buffer is at least one huge page (2 MB) — the streaming path allocates
        // per-expert buffers that comfortably exceed that. No-op on non-Linux and
        // when the user set `COLI_HUGEPAGE=0`.
        if len >= 2 * 1024 * 1024 {
            // SAFETY: allocation just returned, unique, and lives for exactly `len`
            // bytes; the advice is read-only from the caller's viewpoint.
            unsafe {
                crate::mem::advise_hugepages(ptr.as_ptr(), len);
                // NUMA-bind to the allocating thread's node while the pages are
                // still untouched (binding before first touch is the correct
                // moment — MPOL_BIND controls where faults place pages). Opt-in
                // via COLI_NUMA_PIN=1 and multi-node boxes only.
                crate::mem::bind_local_if_enabled(ptr.as_ptr(), len);
            }
        }
        Some(AlignedBuf { ptr, len, layout })
    }

    /// The usable capacity in bytes (a multiple of [`ALIGN`]).
    pub fn capacity(&self) -> usize {
        self.len
    }

    /// Mutable view of the whole buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is a live allocation of exactly `len` bytes, uniquely
        // borrowed for the duration of `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Shared view of the whole buffer.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as above, shared borrow tied to `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with exactly `self.layout` and is
        // freed exactly once here.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Owned read-landing bytes that dereference to `[u8]`. Two shapes:
///
/// - [`Bytes::Vec`] — a plain heap `Vec` (buffered reads, resident weight loads,
///   and the warm cache's clones).
/// - [`Bytes::Aligned`] — an O_DIRECT [`AlignedBuf`] that exposes only its interior
///   region `[head, head+len)`. The kernel DMAs the 4096-aligned superset straight
///   into this buffer and the consumer reads the sub-slice through `Deref`, so the
///   O_DIRECT path needs **no realignment copy** — the bytes the matmul reads are
///   the bytes the disk wrote. `head` is the wanted region's offset inside the
///   aligned window (`0 <= head < ALIGN`); `head + len <= buf.capacity()`.
///
/// `Bytes` backs both a streamed `QtWeight` and an `ExpertSlab` region, so it must
/// be `Send` (it crosses the MoE worker channel) — both variants already are.
pub enum Bytes {
    Vec(Vec<u8>),
    Aligned { buf: AlignedBuf, head: usize, len: usize },
    /// Refcounted bytes shared between holders. The warm cache stores slabs in
    /// this form so a hit hands out a refcount bump instead of copying the whole
    /// (~19 MB) expert — the copy used to happen while the cache lock was held,
    /// serializing every I/O lane behind it.
    Shared(std::sync::Arc<[u8]>),
}

impl Bytes {
    /// The exposed byte length (the logical region, not the aligned window).
    pub fn len(&self) -> usize {
        match self {
            Bytes::Vec(v) => v.len(),
            Bytes::Aligned { len, .. } => *len,
            Bytes::Shared(a) => a.len(),
        }
    }

    /// Resident footprint, which for an O_DIRECT region is the whole aligned
    /// window — up to a block larger than the exposed length. The cache budgets
    /// on this so it accounts for the memory it actually holds.
    pub fn footprint(&self) -> usize {
        match self {
            Bytes::Vec(v) => v.capacity(),
            Bytes::Aligned { buf, .. } => buf.capacity(),
            Bytes::Shared(a) => a.len(),
        }
    }

    /// The exposed bytes as a mutable slice, or `None` for a [`Bytes::Shared`]
    /// region (other holders may be reading it concurrently).
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            Bytes::Vec(v) => Some(v),
            Bytes::Aligned { buf, head, len } => Some(&mut buf.as_mut_slice()[*head..*head + *len]),
            Bytes::Shared(_) => None,
        }
    }

    /// Convert into refcounted shared bytes (copies once, from any variant).
    /// Cloning the result is a refcount bump.
    pub fn into_shared(self) -> Bytes {
        match self {
            Bytes::Shared(a) => Bytes::Shared(a),
            other => Bytes::Shared(std::sync::Arc::from(other.as_slice())),
        }
    }

    /// Whether the exposed region is empty.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// The exposed bytes as a shared slice (identical to `&self[..]`).
    pub fn as_slice(&self) -> &[u8] {
        self
    }
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Vec(v) => v,
            // head+len <= capacity by construction (the aligned window covers the region)
            Bytes::Aligned { buf, head, len } => &buf.as_slice()[*head..*head + *len],
            Bytes::Shared(a) => a,
        }
    }
}

impl Clone for Bytes {
    /// [`Bytes::Shared`] clones by refcount; the owned variants copy their
    /// exposed region into a fresh [`Bytes::Vec`] (this keeps [`AlignedBuf`]
    /// non-cloneable while staying byte-identical to the original).
    fn clone(&self) -> Self {
        match self {
            Bytes::Shared(a) => Bytes::Shared(std::sync::Arc::clone(a)),
            other => Bytes::Vec(other.to_vec()),
        }
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Bytes::Vec(v)
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bytes({} bytes)", self.len())
    }
}

/// A checked-out slab plus its generation tag — bumped every time the pool
/// takes the buffer back, so a stale [`SlabHandle`] can't accidentally be used
/// after a subsequent checkout hands the same allocation to someone else. The
/// generation is checked by [`SlabPool::checkin_tagged`] — **not** by
/// [`SlabPool::checkin`], which validates capacity and free-list length only.
/// Neither tagged variant has a caller (including in tests), so this protection
/// is written and not in force; the live path is the untagged pair. Note also
/// that `checkin_tagged`'s assertion cannot detect the double-return its own doc
/// advertises: any previously-issued handle satisfies `gen < self.gen`. Fix that
/// before wiring it, or the wiring installs a check that never fires.
pub struct SlabHandle {
    pub buf: AlignedBuf,
    pub gen: u32,
}

/// A bounded, reusable pool of [`AlignedBuf`]s of a fixed capacity. Checkout takes
/// a free buffer (or lazily allocates one up to `max_bufs`); checkin returns it.
/// Never grows past `max_bufs`, so total RAM is at most `max_bufs × buf_cap`.
///
/// Each slot carries a `gen` counter bumped on every `checkin` — see
/// [`SlabHandle`] for the misuse-detection story. The `Vec<AlignedBuf>` free-list
/// is intentionally simple; contention is not the bottleneck (the outer
/// `Mutex<Reactor>` already serializes access) and a lock-free swap here would
/// be dead weight until that outer mutex splits.
pub struct SlabPool {
    free: Vec<AlignedBuf>,
    buf_cap: usize,
    max_bufs: usize,
    allocated: usize,
    /// Next generation to stamp on a checkout — monotonically increasing across
    /// checkouts (never reused for the same buffer). Wraps every 2³² checkouts;
    /// aliasing across a full wrap is astronomically unlikely in a decode session.
    gen: u32,
}

impl SlabPool {
    /// A pool of buffers each `buf_cap` bytes (rounded up to [`ALIGN`]), capped at
    /// `max_bufs` total. Allocates nothing until the first [`Self::checkout`].
    pub fn new(buf_cap: usize, max_bufs: usize) -> SlabPool {
        SlabPool {
            free: Vec::new(),
            buf_cap: align_up_usize(buf_cap.max(ALIGN), ALIGN),
            max_bufs: max_bufs.max(1),
            allocated: 0,
            gen: 0,
        }
    }

    /// Borrow a buffer able to hold `needed` bytes. Returns `None` if `needed`
    /// exceeds `buf_cap` (caller must allocate a one-off) or if the pool is at
    /// `max_bufs` with none free (caller must return a buffer first).
    pub fn checkout(&mut self, needed: usize) -> Option<AlignedBuf> {
        if needed > self.buf_cap {
            return None;
        }
        if let Some(b) = self.free.pop() {
            return Some(b);
        }
        if self.allocated < self.max_bufs {
            let b = AlignedBuf::with_capacity(self.buf_cap)?;
            self.allocated += 1;
            return Some(b);
        }
        None
    }

    /// Generation-tagged checkout. Prefer this over [`Self::checkout`] when the
    /// caller wants to be able to catch a "returned twice" or "returned to the
    /// wrong pool" mistake at checkin time (debug builds only).
    pub fn checkout_tagged(&mut self, needed: usize) -> Option<SlabHandle> {
        let buf = self.checkout(needed)?;
        let g = self.gen;
        self.gen = self.gen.wrapping_add(1);
        Some(SlabHandle { buf, gen: g })
    }

    /// Return a buffer to the free-list for reuse.
    pub fn checkin(&mut self, buf: AlignedBuf) {
        // Only buffers this pool handed out may come back: a foreign or
        // wrong-sized buffer would make `free.len()` exceed `allocated` (so
        // `in_use()` underflows) and could later be checked out for a request it
        // cannot hold, slicing past its end.
        if buf.capacity() != self.buf_cap || self.free.len() >= self.allocated {
            crate::note_advisory_err(
                "slab checkin rejected (foreign or duplicate buffer)",
                &format!("capacity {} vs pool {}", buf.capacity(), self.buf_cap),
            );
            return; // dropping it frees the allocation; the pool stays consistent
        }
        self.free.push(buf);
    }

    /// Generation-checked checkin: asserts (in debug) that the caller returned
    /// the same handle they got. Same runtime semantics as [`Self::checkin`] in
    /// release builds.
    pub fn checkin_tagged(&mut self, handle: SlabHandle) {
        // In release the check is compiled out; misuse degrades to an unnecessary
        // check-in, not UB. The purpose is diagnostic — real memory safety comes
        // from `SlabHandle` being a fresh owned move-only type.
        debug_assert!(handle.gen < self.gen || self.gen == 0, "handle generation appears fresh");
        self.checkin(handle.buf);
    }

    /// Capacity of each buffer in this pool (a multiple of [`ALIGN`]).
    pub fn buf_cap(&self) -> usize {
        self.buf_cap
    }

    /// Buffers currently checked out (allocated but not in the free-list).
    pub fn in_use(&self) -> usize {
        self.allocated.saturating_sub(self.free.len())
    }

    /// Total buffers this pool has ever allocated (≤ `max_bufs`).
    pub fn allocated(&self) -> usize {
        self.allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_arith() {
        assert_eq!(align_down(0, 4096), 0);
        assert_eq!(align_down(4095, 4096), 0);
        assert_eq!(align_down(4096, 4096), 4096);
        assert_eq!(align_down(4097, 4096), 4096);
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up_usize(19_000_000, 4096), 19_001_344);
    }

    #[test]
    fn aligned_buf_is_aligned_and_round_trips() -> Result<(), &'static str> {
        for cap in [1usize, 4095, 4096, 4097, 100_000] {
            let mut b = AlignedBuf::with_capacity(cap).ok_or("alloc failed")?;
            assert_eq!(b.as_slice().as_ptr() as usize % ALIGN, 0, "base 4096-aligned");
            assert_eq!(b.capacity(), align_up_usize(cap.max(1), ALIGN));
            // write a pattern through the mutable view, read it back
            for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
                *x = (i % 251) as u8;
            }
            assert!(b.as_slice().iter().enumerate().all(|(i, &x)| x == (i % 251) as u8));
        }
        Ok(())
    }

    #[test]
    fn pool_reuses_same_allocation_and_bounds() -> Result<(), &'static str> {
        let mut p = SlabPool::new(8192, 2);
        assert_eq!(p.buf_cap(), 8192);
        let mut a = p.checkout(4096).ok_or("checkout a")?;
        let b = p.checkout(8192).ok_or("checkout b")?;
        assert_eq!(p.in_use(), 2);
        // at the cap with none free → None
        assert!(p.checkout(4096).is_none(), "pool bounded at max_bufs");
        // tag `a`, return it, check the next checkout hands back the SAME allocation
        let addr_a = a.as_slice().as_ptr() as usize;
        a.as_mut_slice()[0] = 0xAB;
        p.checkin(a);
        let a2 = p.checkout(4096).ok_or("checkout a2")?;
        assert_eq!(a2.as_slice().as_ptr() as usize, addr_a, "reused, not re-allocated");
        assert_eq!(a2.as_slice()[0], 0xAB, "same buffer contents survived checkin/checkout");
        assert_eq!(p.allocated(), 2, "no growth past max_bufs");
        p.checkin(a2);
        p.checkin(b);
        // a read larger than buf_cap cannot be served by this pool
        assert!(p.checkout(9000).is_none());
        Ok(())
    }
}

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

/// A bounded, reusable pool of [`AlignedBuf`]s of a fixed capacity. Checkout takes
/// a free buffer (or lazily allocates one up to `max_bufs`); checkin returns it.
/// Never grows past `max_bufs`, so total RAM is at most `max_bufs × buf_cap`.
pub struct SlabPool {
    free: Vec<AlignedBuf>,
    buf_cap: usize,
    max_bufs: usize,
    allocated: usize,
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

    /// Return a buffer to the free-list for reuse.
    pub fn checkin(&mut self, buf: AlignedBuf) {
        self.free.push(buf);
    }

    /// Capacity of each buffer in this pool (a multiple of [`ALIGN`]).
    pub fn buf_cap(&self) -> usize {
        self.buf_cap
    }

    /// Buffers currently checked out (allocated but not in the free-list).
    pub fn in_use(&self) -> usize {
        self.allocated - self.free.len()
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

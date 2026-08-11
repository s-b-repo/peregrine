//! The I/O lane: batched positioned reads. On Linux the [`Reactor`] uses
//! io_uring — one `submit_and_wait` drives up to a full ring of reads through
//! io-wq (forced `IOSQE_ASYNC`, matching `c/uring.h`), so N expert-slab reads
//! cost one enter syscall instead of N `pread`s. [`pread_many`] is the portable
//! fallback (and the correctness oracle for the ring).

use crate::slab::Bytes;
use std::os::unix::io::RawFd;

/// One positioned read into a caller-owned buffer.
pub struct ReadReq<'a> {
    pub fd: RawFd,
    pub offset: u64,
    pub buf: &'a mut [u8],
    /// caller tag echoed back (e.g. an expert id); not used by the reader
    pub tag: u64,
}

/// One positioned read whose landing buffer the [`Reactor`] owns until its
/// completion is reaped (the owned-completion lane, [`Reactor::submit_owned`]).
/// Unlike [`ReadReq`] there is no borrowed buffer: the reactor allocates the
/// landing storage and hands it back as [`RegionDone::bytes`] — which is what
/// lets completions be delivered one at a time instead of as a whole wave.
pub struct OwnedReadReq {
    pub fd: RawFd,
    pub offset: u64,
    pub len: usize,
    /// caller tag echoed back in [`RegionDone::tag`] (the streaming lane passes
    /// the expert-local index); not used by the reader
    pub tag: u64,
}

/// A completed owned read: exactly `[offset, offset+len)` of its request,
/// byte-identical to a buffered positioned read of the same range.
pub struct RegionDone {
    /// The request's position in its `submit_owned` batch — the stable key,
    /// independent of arrival order (which io-wq deliberately randomizes).
    pub idx: usize,
    pub tag: u64,
    pub bytes: Bytes,
}

/// Portable fallback: one `pread` per request. Returns per-request byte counts
/// (or a negative errno). Always available; used to validate the io_uring path.
pub fn pread_many(reqs: &mut [ReadReq]) -> Vec<i64> {
    use std::mem::ManuallyDrop;
    use std::os::unix::fs::FileExt;
    use std::os::unix::io::FromRawFd;
    reqs.iter_mut()
        .map(|r| {
            // Borrow the caller's fd for a positioned read without taking
            // ownership: `ManuallyDrop` stops `File`'s Drop from closing a
            // descriptor we don't own (clearer and safer than `mem::forget`,
            // which risks a use-after-forget).
            // SAFETY: `r.fd` is a live descriptor the caller keeps open for the
            // duration of this call; we only read from it.
            let file = ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(r.fd) });
            match file.read_at(r.buf, r.offset) {
                Ok(n) => n as i64,
                Err(e) => -(e.raw_os_error().unwrap_or(5) as i64),
            }
        })
        .collect()
}

/// [`pread_many`] spread across `threads` OS threads, each taking a contiguous
/// chunk of `reqs`. Returns per-request byte counts in the caller's order.
///
/// **Why a blocking-`pread` engine exists next to a working io_uring lane.**
/// On the 7 GB laptop in `docs/benchmarks.md` §Second box, peregrine's io_uring
/// O_DIRECT lane measured **0.84 GB/s against colibrì's 2.02 GB/s** from eight
/// OpenMP threads of blocking `pread` — on the same drive, same shard sizes.
/// Batching the O_DIRECT submission (2026-08-02) recovered 1.2–1.3× and left a
/// large gap. The standing hypothesis is dm-crypt: on a LUKS volume reads are
/// CPU-bound on decryption, so *N* blocking preads keep *N* cores decrypting
/// where the ring's completion model can leave cores idle. Independently, the
/// CHEOPS storage-stack study measures plain io_uring at queue depth 1 as
/// *slower* than synchronous pread (81.3 vs 94.6 KIOPS), and puts the Linux
/// block layer at an irreducible ~35% of the CPU budget for every in-kernel
/// stack.
///
/// This makes the hypothesis testable end to end instead of only in a
/// microbenchmark: `read_regions` can pick this engine for the real streaming
/// path. It is deliberately the dumbest possible implementation — a static
/// split, no work stealing — because its whole purpose is to be the same shape
/// as the C harness it is being compared against.
///
/// Threads are capped at the request count (no empty workers) and at 64 (a
/// spawn per region on a large batch would cost more than it saves). One thread
/// runs inline, with no spawn at all.
pub fn pread_many_threaded(reqs: &mut [ReadReq], threads: usize) -> Vec<i64> {
    let n = reqs.len();
    let t = threads.clamp(1, n.max(1)).min(64);
    if t <= 1 || n <= 1 {
        return pread_many(reqs);
    }
    let chunk = n.div_ceil(t);
    let mut out: Vec<i64> = vec![0; n];
    // Scoped threads so `reqs`' borrowed buffers need no 'static bound and no
    // Arc/Mutex: each worker owns a disjoint sub-slice for the scope's lifetime.
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(t);
        for (i, part) in reqs.chunks_mut(chunk).enumerate() {
            handles.push((i, s.spawn(move || pread_many(part))));
        }
        for (i, h) in handles {
            // A worker panic must not poison the whole batch silently: report
            // its chunk as a hard error (EIO) so the caller's existing
            // negative-result handling retries or fails loudly — *and* say so,
            // because a bare -5 is indistinguishable from a real device error.
            let res = match h.join() {
                Ok(v) => v,
                Err(panic) => {
                    crate::note_advisory_err("pread worker panicked; chunk reported as EIO", &format!("{panic:?}"));
                    vec![-5i64; chunk]
                }
            };
            let base = i * chunk;
            for (k, v) in res.into_iter().enumerate() {
                if let Some(slot) = out.get_mut(base + k) {
                    *slot = v;
                }
            }
        }
    });
    out
}

#[cfg(target_os = "linux")]
mod uring {
    use super::{OwnedReadReq, ReadReq, RegionDone};
    use crate::slab::{align_down, align_up, AlignedBuf, Bytes, SlabHandle, SlabPool, ALIGN};
    use io_uring::{opcode, squeue, types, IoUring};
    use std::collections::{HashMap, VecDeque};
    use std::io;
    use std::os::unix::io::RawFd;

    /// Landing storage for one owned read: a plain heap `Vec` for buffered
    /// reads, an [`AlignedBuf`] over the 4096-aligned superset for O_DIRECT.
    enum LandingBuf {
        Plain(Vec<u8>),
        Aligned(AlignedBuf),
    }

    /// One owned read in flight (or queued behind the ring cap). The aligned
    /// window fields mirror `read_direct_aligned`'s `Span`: `need = head + want`
    /// is what must land before the region is complete — the tail padding past
    /// it may legally EOF-short and is never delivered.
    struct InFlight {
        fd: RawFd,
        /// submission offset (already aligned down for O_DIRECT).
        a_off: u64,
        /// the wanted region's offset inside the landing buffer (0 if buffered).
        head: usize,
        /// exposed bytes the caller asked for.
        want: usize,
        /// head + want.
        need: usize,
        /// landing-buffer length actually submitted for reading.
        a_len: usize,
        /// bytes landed so far (continuations resume here).
        done: usize,
        tag: u64,
        idx: usize,
        buf: LandingBuf,
    }

    impl InFlight {
        /// The completed region as owned [`Bytes`] — no realignment copy: the
        /// O_DIRECT variant exposes `[head, head+want)` of the DMA target.
        fn finish(self) -> RegionDone {
            let bytes = match self.buf {
                LandingBuf::Plain(v) => Bytes::Vec(v),
                LandingBuf::Aligned(buf) => Bytes::Aligned { buf, head: self.head, len: self.want },
            };
            RegionDone { idx: self.idx, tag: self.tag, bytes }
        }
    }

    /// Ring construction options (env-free — [`Reactor::new_streaming`] fills
    /// them from `COLI_SQPOLL*`).
    pub(crate) struct RingOpts {
        pub(crate) sqpoll: Option<SqpollOpts>,
    }

    pub(crate) struct SqpollOpts {
        /// ms the poll kthread keeps spinning on an empty SQ before sleeping.
        pub(crate) idle_ms: u32,
        /// optional CPU id to pin the kthread to (`IORING_SETUP_SQ_AFF`).
        pub(crate) cpu: Option<u32>,
    }

    /// io_uring-backed batched reader (the I/O lane owner thread holds one).
    pub struct Reactor {
        ring: IoUring,
        cap: usize,
        force_async: bool,
        /// kernel-side submission polling active (the `COLI_SQPOLL` opt-in).
        sqpoll: bool,
        /// Times a submission-queue push was rejected (queue full) since the
        /// last [`Reactor::take_sq_full`]. Queue-pressure signal for the
        /// adaptive io-wq tuner.
        sq_full: u64,
        /// fds registered with the kernel (index = fixed-file slot). A read whose
        /// fd is here uses `IOSQE_FIXED_FILE`, skipping per-op fd lookup/refcount.
        registered: Vec<RawFd>,
        /// pre-registered landing buffers for `ReadFixed` (index = fixed-buffer
        /// slot). Owned here so the pinned pages stay valid and never reallocate
        /// while registered. Empty until [`Reactor::register_read_buffers`].
        registered_bufs: Vec<Vec<u8>>,
        /// reusable aligned landing buffers for O_DIRECT reads ([`Reactor::read_direct_many`]);
        /// sized once via [`Reactor::configure_slab`]. Idle (no allocation) until used.
        slab: SlabPool,
        /// owned-completion lane: reads in flight, keyed by their `user_data`
        /// (`OWNED_TAG | next_ud`). The reactor owns each landing buffer until
        /// the completion is reaped, so nothing here dangles.
        pending: HashMap<u64, InFlight>,
        /// owned reads accepted but not yet pushed (ring-cap backpressure).
        queued: VecDeque<InFlight>,
        /// queued advisory (fadvise) entries, drained into SQ slots left over
        /// after reads — reads always outrank hints. Bounded drop-oldest.
        advisory: VecDeque<squeue::Entry>,
        /// advisory ops pushed to the kernel and not yet reaped.
        advisory_inflight: usize,
        /// zero-length owned requests, completed at submit time.
        ready: Vec<RegionDone>,
        /// next owned `user_data` low bits (wraps below `OWNED_TAG`).
        next_ud: u64,
    }

    impl Drop for Reactor {
        /// Unregister the kernel-pinned buffers and files before the ring goes
        /// away. Correctness here previously rested on field declaration order
        /// (the ring closing first happens to unpin them); stating it in code
        /// means a future field reshuffle cannot silently turn this into a
        /// use-after-free of pinned pages.
        fn drop(&mut self) {
            // Owned reads still in flight point at buffers this Reactor is about
            // to free — wait their completions out before anything is dropped.
            if !self.pending.is_empty() || self.advisory_inflight > 0 {
                self.abort_owned();
            }
            if !self.registered_bufs.is_empty() {
                if let Err(e) = self.ring.submitter().unregister_buffers() {
                    crate::note_advisory_err("io_uring unregister_buffers at shutdown", &e);
                }
            }
            if !self.registered.is_empty() {
                if let Err(e) = self.ring.submitter().unregister_files() {
                    crate::note_advisory_err("io_uring unregister_files at shutdown", &e);
                }
            }
        }
    }

    impl Reactor {
        /// `entries` = submission-queue depth (rounded up to a power of two by the
        /// kernel). Cold NVMe streaming wants this ≥ the per-layer expert count.
        ///
        /// The ring is set up with `COOP_TASKRUN` (completion task work runs
        /// cooperatively at `io_uring_enter` instead of via IPIs → less overhead).
        /// We deliberately do **not** set `SINGLE_ISSUER`: the concurrent
        /// scheduler's lane threads are scoped and respawned per layer, so the
        /// submitting task identity changes every layer and single-issuer would
        /// reject the second one with `-EEXIST`. If a kernel rejects the flag we
        /// fall back to a plain ring. `SQPOLL` is the `COLI_SQPOLL` opt-in on
        /// [`Reactor::new_streaming`] — unprivileged since kernel 5.13, and
        /// thread-agnostic (the poll kthread issues the ops), which is exactly
        /// why it composes with the per-layer lane threads where single-issuer
        /// cannot.
        pub fn new(entries: u32) -> io::Result<Reactor> {
            let mut r = Self::new_with(entries, RingOpts { sqpoll: None })?;
            r.apply_force_async_env();
            Ok(r)
        }

        /// Streaming-lane ring: [`Reactor::new`] plus the `COLI_SQPOLL` opt-in.
        ///
        /// - `COLI_SQPOLL=1` — ask for kernel-side submission polling: a kernel
        ///   thread drains the SQ, so a hot submit costs no `io_uring_enter` at
        ///   all. Off by default — the kthread busy-polls a core while awake,
        ///   the wrong trade on a CPU-contended box unless measured otherwise.
        /// - `COLI_SQPOLL_IDLE_MS` (default 2000) — how long the kthread polls
        ///   an empty SQ before sleeping. 2 s spans every intra-token gap (per
        ///   layer compute runs tens-to-hundreds of ms) so no hot submit pays a
        ///   wakeup, while between requests the thread sleeps; wakeup after
        ///   sleep is handled by `submit()` itself, so a larger value buys
        ///   nothing but burnt cycles.
        /// - `COLI_SQPOLL_CPU` — optionally pin the kthread to one CPU id,
        ///   confining the polling burn (and its cache pressure) to one core.
        ///
        /// A kernel that refuses the flags falls back to the plain
        /// `COOP_TASKRUN` ring with an advisory note — same bytes either way.
        pub fn new_streaming(entries: u32) -> io::Result<Reactor> {
            let sqpoll = if matches!(std::env::var("COLI_SQPOLL").as_deref(), Ok("1") | Ok("true")) {
                let idle_ms = std::env::var("COLI_SQPOLL_IDLE_MS")
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(2000);
                let cpu = std::env::var("COLI_SQPOLL_CPU").ok().and_then(|s| s.trim().parse::<u32>().ok());
                Some(SqpollOpts { idle_ms, cpu })
            } else {
                None
            };
            let mut r = Self::new_with(entries, RingOpts { sqpoll })?;
            r.apply_force_async_env();
            Ok(r)
        }

        /// Env-free construction from explicit options (testable without
        /// touching the process environment).
        pub(crate) fn new_with(entries: u32, opts: RingOpts) -> io::Result<Reactor> {
            let mut sqpoll_active = false;
            let ring = match &opts.sqpoll {
                Some(sp) => {
                    // No COOP_TASKRUN here: task work runs on the poll kthread
                    // under SQPOLL, and kernels reject the combination.
                    let mut b = IoUring::builder();
                    b.setup_sqpoll(sp.idle_ms);
                    if let Some(cpu) = sp.cpu {
                        b.setup_sqpoll_cpu(cpu);
                    }
                    match b.build(entries) {
                        Ok(r) => {
                            sqpoll_active = true;
                            Ok(r)
                        }
                        Err(e) => {
                            crate::note_advisory_err(
                                "io_uring SQPOLL setup (falling back to COOP_TASKRUN ring)",
                                &e,
                            );
                            Self::plain_ring(entries)
                        }
                    }
                }
                None => Self::plain_ring(entries),
            }?;
            Ok(Reactor {
                ring,
                cap: entries as usize,
                // Forced IOSQE_ASYNC exists so a cold read completing inline
                // cannot serialize the submitting thread. Under SQPOLL the
                // submitter never issues inline at all — the kthread does, with
                // nonblocking semantics, punting cold reads to io-wq itself —
                // so the flag's only remaining effect would be to turn
                // warm-page-cache reads into a pointless io-wq bounce: off.
                force_async: !sqpoll_active,
                sqpoll: sqpoll_active,
                sq_full: 0,
                registered: Vec::new(),
                registered_bufs: Vec::new(),
                slab: SlabPool::new(ALIGN, 1),
                pending: HashMap::new(),
                queued: VecDeque::new(),
                advisory: VecDeque::new(),
                advisory_inflight: 0,
                ready: Vec::new(),
                next_ud: 0,
            })
        }

        /// The default ring: `COOP_TASKRUN`, falling back to a plain ring on
        /// kernels that reject the flag.
        fn plain_ring(entries: u32) -> io::Result<IoUring> {
            IoUring::builder()
                .setup_coop_taskrun()
                .build(entries)
                .or_else(|e| {
                    crate::note_advisory_err("io_uring COOP_TASKRUN setup (using plain ring)", &e);
                    IoUring::new(entries)
                })
        }

        /// `COLI_FORCE_ASYNC` override on top of the ring-dependent default
        /// (`1` forces the io-wq hand-off, `0` runs reads inline). Default is
        /// on for a plain ring — a cold read that completes inline serializes
        /// the submitter — and off under SQPOLL (see `new_with`); on a device
        /// where completion is fast enough (a warm page cache, a very
        /// low-latency NVMe) the io-wq bounce is pure overhead, and this knob
        /// is how you find out.
        fn apply_force_async_env(&mut self) {
            match std::env::var("COLI_FORCE_ASYNC").as_deref() {
                Ok("0") | Ok("false") => self.set_force_async(false),
                Ok("1") | Ok("true") => self.set_force_async(true),
                _ => {}
            }
        }

        /// Whether this ring runs with kernel-side submission polling.
        pub fn is_sqpoll(&self) -> bool {
            self.sqpoll
        }

        /// Submission-queue-full rejections since the last call (swap-reset).
        /// Consumed by the adaptive io-wq tuner as its queue-pressure signal.
        pub fn take_sq_full(&mut self) -> u64 {
            std::mem::take(&mut self.sq_full)
        }

        /// Push one submission entry, counting a queue-full rejection into
        /// [`Self::take_sq_full`]'s counter before surfacing the error.
        ///
        /// # Safety
        ///
        /// Same contract as `submission().push`: any buffer the entry points at
        /// must stay live until its completion is reaped (every caller here
        /// waits on the completion before reusing buffers).
        /// Turn a negative io_uring result into an `io::Error`. Uses checked
        /// negation: `-(i64::MIN)` overflows (a panic in debug, and in release it
        /// wraps back to `i64::MIN`, truncating to errno 0 — an error whose
        /// message reads "Success").
        fn errno_from_result(n: i64) -> io::Error {
            match n.checked_neg().and_then(|e| i32::try_from(e).ok()) {
                Some(e) => io::Error::from_raw_os_error(e),
                None => io::Error::other(format!("io_uring returned an unrepresentable result code {n}")),
            }
        }

        /// High bit of `user_data`, set on advisory (non-read) submissions.
        /// Read completions live in the low range, so a stray advisory
        /// completion can never be mistaken for a read result and scribble into
        /// another batch's results vector.
        const ADVISORY_TAG: u64 = 1 << 63;

        /// Second-highest bit of `user_data`, set on owned-completion reads
        /// ([`Reactor::submit_owned`]). Three disjoint namespaces: legacy wave
        /// reads (low range), owned reads (bit 62), advisories (bit 63) — so no
        /// lane can ever mistake another lane's completion for its own.
        const OWNED_TAG: u64 = 1 << 62;

        unsafe fn push_counted(&mut self, e: &squeue::Entry) -> io::Result<()> {
            // SAFETY: forwarded caller contract (see above).
            if unsafe { self.ring.submission().push(e) }.is_err() {
                self.sq_full += 1;
                return Err(io::Error::other("submission queue full"));
            }
            Ok(())
        }

        /// Register `fds` as fixed files. Subsequent reads whose fd is in this set
        /// use `IOSQE_FIXED_FILE`, so the kernel skips the per-op fd table lookup
        /// and refcount — worthwhile when the same shard fds are read every token.
        /// Replaces any previous registration. Errors are non-fatal to the caller:
        /// on failure, reads simply fall back to the plain-fd path.
        pub fn register_files(&mut self, fds: &[RawFd]) -> io::Result<()> {
            if !self.registered.is_empty() {
                self.ring.submitter().unregister_files()?;
                self.registered.clear();
            }
            self.ring.submitter().register_files(fds)?;
            self.registered = fds.to_vec();
            Ok(())
        }

        /// Read exactly `buf.len()` bytes at `off` from `fd`, looping to complete
        /// a short completion (a positioned read may legally return fewer bytes).
        /// Errors on a negative completion code or a premature EOF — never a
        /// partial success, never a fallback.
        pub fn read_exact(&mut self, fd: RawFd, off: u64, buf: &mut [u8]) -> io::Result<()> {
            let total = buf.len();
            let mut done = 0usize;
            while done < total {
                let n = {
                    let mut reqs = [ReadReq { fd, offset: off + done as u64, buf: &mut buf[done..], tag: 0 }];
                    self.read_many(&mut reqs)?[0]
                };
                if n < 0 {
                    return Err(Self::errno_from_result(n));
                }
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("io_uring read hit EOF after {done} of {total} bytes"),
                    ));
                }
                done += n as usize;
            }
            Ok(())
        }

        /// The fixed-file slot for `fd`, if it was registered.
        pub fn is_registered(&self, fd: RawFd) -> bool {
            self.registered.contains(&fd)
        }

        /// Bound the io-wq worker pool (like `IORING_REGISTER_IOWQ_MAX_WORKERS`).
        /// `[bounded, unbounded]`.
        pub fn set_iowq_max_workers(&mut self, bounded: u32, unbounded: u32) -> io::Result<()> {
            let mut vals = [bounded, unbounded];
            self.ring.submitter().register_iowq_max_workers(&mut vals)
        }

        /// Toggle forced `IOSQE_ASYNC` (default on: cold buffered reads run on
        /// io-wq instead of inline, so the submitter never serializes).
        ///
        /// Driven by `COLI_FORCE_ASYNC=0` at [`Reactor::new`]. The escape hatch
        /// matters on a device fast enough that inline completion beats an io-wq
        /// hand-off — an NVMe with a warm page cache can be exactly that, and
        /// forcing async then adds a thread bounce per read for nothing.
        ///
        /// Advertised in the env tables, deliberately. The audit's own headline
        /// example is `COLI_REGBUF`: a knob documented as live that no code read.
        /// A setter with no reader is the same defect facing the other way, and
        /// the fix for both is that the variable and the code agree.
        pub fn set_force_async(&mut self, on: bool) {
            self.force_async = on;
        }

        /// Register a pool of landing buffers for `ReadFixed` (C2). Once
        /// registered, [`Reactor::read_fixed`] reads into slot `i` and the kernel
        /// skips per-op page pinning. Takes ownership of the buffers so their pages
        /// stay pinned and never reallocate while registered; replaces any prior
        /// registration. Errors (e.g. kernel without buffer registration) are
        /// non-fatal — the caller can fall back to the plain [`Reactor::read_many`].
        pub fn register_read_buffers(&mut self, bufs: Vec<Vec<u8>>) -> io::Result<()> {
            if !self.registered_bufs.is_empty() {
                self.ring.submitter().unregister_buffers()?;
                self.registered_bufs.clear();
            }
            let iovecs: Vec<libc::iovec> = bufs
                .iter()
                .map(|b| libc::iovec { iov_base: b.as_ptr() as *mut libc::c_void, iov_len: b.len() })
                .collect();
            // SAFETY: each iovec points into a buffer in `bufs`, which we move into
            // `self.registered_bufs` and keep alive (never reallocated) for as long
            // as it is registered; unregistered in `Drop`/on the next registration.
            unsafe { self.ring.submitter().register_buffers(&iovecs)? };
            self.registered_bufs = bufs;
            // Ask the kernel to back each registered buffer with transparent huge
            // pages when it clears the 2 MB threshold. The pages are pinned for the
            // lifetime of the registration, so the syscall is amortized. No-op on
            // non-Linux and when `COLI_HUGEPAGE=0`.
            for b in self.registered_bufs.iter_mut() {
                if b.len() >= 2 * 1024 * 1024 {
                    // SAFETY: the vec is owned by `self.registered_bufs`; the advice is
                    // read-only from the caller's viewpoint.
                    unsafe { crate::mem::advise_hugepages(b.as_mut_ptr(), b.len()) };
                }
            }
            Ok(())
        }

        /// The number of registered fixed buffers.
        pub fn fixed_buffer_count(&self) -> usize {
            self.registered_bufs.len()
        }

        /// Bytes in the *smallest* registered buffer — the largest read
        /// `read_fixed_many` can serve. The minimum, not the maximum: one
        /// undersized slot in the pool caps every request, so reporting the max
        /// would let a caller size a batch that then fails mid-wave.
        pub fn fixed_buffer_capacity(&self) -> usize {
            self.registered_bufs.iter().map(|b| b.len()).min().unwrap_or(0)
        }

        /// Read exactly `out.len()` bytes at `off` from `fd` **through the
        /// registered fixed buffer** `buf_index` (C2), then copy them into `out`.
        /// Loops to complete a short read. Byte-identical to [`Reactor::read_exact`];
        /// the copy-out is the trade-off of the owned-buffer hand-off (which is why
        /// the streaming path keeps the plain read by default — see `COLI_REGBUF`).
        pub fn read_fixed(&mut self, fd: RawFd, off: u64, buf_index: u16, out: &mut [u8]) -> io::Result<()> {
            if !self.owned_idle() {
                return Err(Self::err_owned_busy("read_fixed"));
            }
            let bi = buf_index as usize;
            let cap = self.registered_bufs.get(bi).map(|b| b.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "read_fixed: buffer index not registered")
            })?;
            if out.len() > cap {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "read_fixed: out exceeds registered buffer"));
            }
            let total = out.len();
            let mut done = 0usize;
            while done < total {
                let base = self.registered_bufs[bi].as_mut_ptr();
                // SAFETY: `base` is the registered buffer `bi`; `done < total <= cap`
                // so `base+done` and length `total-done` stay within it. The op is
                // waited on below before the buffer is read, so it outlives the read.
                let e = opcode::ReadFixed::new(types::Fd(fd), unsafe { base.add(done) }, (total - done) as u32, buf_index)
                    .offset(off + done as u64)
                    .build()
                    .user_data(0);
                // SAFETY: buffer outlives the op — completion reaped below.
                unsafe { self.push_counted(&e) }?;
                self.ring.submit_and_wait(1)?;
                let mut n: Option<i64> = None;
                for cqe in self.ring.completion() {
                    n = Some(cqe.result() as i64);
                }
                // No completion at all is its own failure, not a negative result:
                // the old `i64::MIN` sentinel negated to itself and reported
                // "errno 0 = Success" in release builds.
                let Some(n) = n else {
                    return Err(io::Error::other("io_uring read_fixed completion missing"));
                };
                if n < 0 {
                    return Err(Self::errno_from_result(n));
                }
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_fixed hit EOF"));
                }
                done += n as usize;
            }
            out.copy_from_slice(&self.registered_bufs[bi][..total]);
            Ok(())
        }

        /// Batched [`Reactor::read_fixed`]: submit up to one op per registered
        /// buffer in a single `submit_and_wait`, then copy each landing buffer
        /// out to its request. Returns per-request byte counts in caller order.
        ///
        /// **This exists because `read_fixed` is depth-1.** It loops
        /// `submit_and_wait(1)` per region — the exact defect that crippled the
        /// O_DIRECT lane until 2026-08-02, where `read_direct_aligned` submitted
        /// one region at a time and consequently measured *slower* than buffered
        /// reads. Wiring `COLI_REGBUF` to the single-region call would have
        /// reintroduced it, so the batched form has to land first.
        ///
        /// **Read the trade before enabling it on the streaming path.** Fixed
        /// buffers remove the per-I/O page-locking and DMA-mapping that scales
        /// with transfer size (FAST '24), but this path *copies out* of the
        /// registered buffer, where `read_many` has the kernel write the
        /// caller's buffer directly. At the ~6 MB regions peregrine streams, that
        /// memcpy plausibly costs more than the pinning it saves — the published
        /// gains are at 4–64 KB. It is offered as a measurable engine, not as a
        /// recommended default, and `COLI_IO_ENGINE=regbuf` is how you find out
        /// on your own hardware.
        ///
        /// Requests longer than a registered buffer are an error rather than a
        /// silent short read.
        pub fn read_fixed_many(&mut self, reqs: &mut [ReadReq]) -> io::Result<Vec<i64>> {
            if !self.owned_idle() {
                return Err(Self::err_owned_busy("read_fixed_many"));
            }
            let nbufs = self.registered_bufs.len();
            if nbufs == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "read_fixed_many: no registered buffers (call register_read_buffers first)",
                ));
            }
            let cap = self.registered_bufs.iter().map(|b| b.len()).min().unwrap_or(0);
            if let Some(bad) = reqs.iter().position(|r| r.buf.len() > cap) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "read_fixed_many: request {bad} is {} bytes, larger than the {cap}-byte registered buffer",
                        reqs[bad].buf.len()
                    ),
                ));
            }
            let mut results: Vec<Option<i64>> = vec![None; reqs.len()];
            // One in-flight op per registered buffer: two ops sharing a buffer
            // would interleave their writes into it.
            let width = self.cap.min(nbufs);
            let mut i = 0usize;
            while i < reqs.len() {
                let end = (i + width).min(reqs.len());
                let mut pushed = 0usize;
                let mut push_err = None;
                for (bi, req) in reqs[i..end].iter().enumerate() {
                    let j = i + bi; // absolute request index; `bi` is the buffer slot
                    let Ok(len32) = u32::try_from(req.buf.len()) else {
                        push_err = Some(io::Error::other(format!("read request {j} exceeds u32 bytes")));
                        break;
                    };
                    let base = self.registered_bufs[bi].as_mut_ptr();
                    let e = opcode::ReadFixed::new(types::Fd(req.fd), base, len32, bi as u16)
                        .offset(req.offset)
                        .build()
                        .user_data(j as u64);
                    // SAFETY: `base` is registered buffer `bi`, owned by `self`
                    // and not reallocated while registered; the op is reaped
                    // below (including on the error paths) before it is reused.
                    match unsafe { self.push_counted(&e) } {
                        Ok(()) => pushed += 1,
                        Err(e) => {
                            push_err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(err) = push_err {
                    if let Err(drain_err) = self.drain_pushed(pushed) {
                        crate::note_advisory_err("io_uring drain after failed fixed submit", &drain_err);
                    }
                    return Err(err);
                }
                self.ring.submit_and_wait(pushed)?;
                let mut got = 0usize;
                for cqe in self.ring.completion() {
                    let ud = cqe.user_data();
                    if ud & Self::ADVISORY_TAG != 0 {
                        continue;
                    }
                    if let Some(slot) = results.get_mut(ud as usize) {
                        *slot = Some(cqe.result() as i64);
                    }
                    got += 1;
                }
                if got < pushed {
                    return Err(io::Error::other("io_uring read_fixed_many: missing completions"));
                }
                // Copy out only after every op in the wave has completed — a
                // buffer read while its op is still in flight is a torn read.
                for (bi, req) in reqs[i..end].iter_mut().enumerate() {
                    let n = results.get(i + bi).copied().flatten().unwrap_or(-5);
                    if n > 0 {
                        let n = (n as usize).min(req.buf.len());
                        req.buf[..n].copy_from_slice(&self.registered_bufs[bi][..n]);
                    }
                }
                i = end;
            }
            Ok(results.into_iter().map(|r| r.unwrap_or(-5)).collect())
        }

        /// Advise the kernel to read `[off, off+len)` of `fd` into the page cache
        /// ahead of a real read (`IORING_OP_FADVISE`, `POSIX_FADV_WILLNEED`) — C3
        /// far-ahead warming. Purely advisory: it moves no bytes and cannot affect
        /// output, so a soft failure is harmless (a hard submit error is surfaced).
        pub fn fadvise_willneed(&mut self, fd: RawFd, off: u64, len: usize) -> io::Result<()> {
            const POSIX_FADV_WILLNEED: i32 = 3;
            self.fadvise(fd, off, len, POSIX_FADV_WILLNEED)
        }

        /// Advise the kernel to drop page-cache references for `[off, off+len)` of
        /// `fd` (`POSIX_FADV_DONTNEED`). Called after the last consumer of a
        /// streamed expert to keep long-running RSS flat under buffered reads.
        /// Purely advisory: soft failure is harmless.
        pub fn fadvise_dontneed(&mut self, fd: RawFd, off: u64, len: usize) -> io::Result<()> {
            const POSIX_FADV_DONTNEED: i32 = 4;
            self.fadvise(fd, off, len, POSIX_FADV_DONTNEED)
        }

        fn fadvise(&mut self, fd: RawFd, off: u64, len: usize, advice: i32) -> io::Result<()> {
            if !self.owned_idle() {
                return Err(Self::err_owned_busy("fadvise"));
            }
            let e = opcode::Fadvise::new(types::Fd(fd), len as libc::off_t, advice)
                .offset(off)
                .build()
                .user_data(0);
            // SAFETY: Fadvise carries no buffer — the op only hints the page cache.
            // SAFETY: Fadvise carries no buffer.
            unsafe { self.push_counted(&e) }?;
            self.ring.submit_and_wait(1)?;
            let mut n = 0i64;
            for cqe in self.ring.completion() {
                n = cqe.result() as i64;
            }
            if n < 0 {
                return Err(Self::errno_from_result(n));
            }
            Ok(())
        }

        /// Retry `read_exact` up to `max_retries` times on transient failures
        /// (`EIO`, `EAGAIN`, `EINTR`). Sleeps between retries with linear backoff
        /// (10 ms × attempt) — enough to survive a queued NVMe error retry
        /// without hanging the forward. Non-transient errors and EOF-short reads
        /// propagate immediately. Returns the last error on exhaustion.
        pub fn read_exact_retry(
            &mut self,
            fd: RawFd,
            off: u64,
            buf: &mut [u8],
            max_retries: u32,
        ) -> io::Result<()> {
            let mut last: io::Error = io::Error::other("no attempt made");
            for attempt in 0..=max_retries {
                match self.read_exact(fd, off, buf) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        let transient = matches!(
                            e.raw_os_error(),
                            Some(libc::EIO) | Some(libc::EAGAIN) | Some(libc::EINTR)
                        );
                        if !transient || attempt == max_retries {
                            return Err(e);
                        }
                        last = e;
                        std::thread::sleep(std::time::Duration::from_millis(10u64 * (attempt as u64 + 1)));
                    }
                }
            }
            Err(last)
        }

        /// Batched `POSIX_FADV_WILLNEED` — one submit for all N regions. Purely
        /// advisory: individual completion codes are ignored (a soft failure is
        /// harmless), only a hard submit error surfaces. Reads submitted through
        /// [`Self::read_many`] afterwards run against pages the kernel already
        /// started warming.
        pub fn fadvise_willneed_many(&mut self, regions: &[(RawFd, u64, usize)]) -> io::Result<()> {
            if !self.owned_idle() {
                return Err(Self::err_owned_busy("fadvise_willneed_many"));
            }
            const POSIX_FADV_WILLNEED: i32 = 3;
            let mut i = 0;
            while i < regions.len() {
                let end = (i + self.cap).min(regions.len());
                let mut pushed = 0usize;
                let mut push_err = None;
                for (j, &(fd, off, len)) in regions.iter().enumerate().take(end).skip(i) {
                    let e = opcode::Fadvise::new(types::Fd(fd), len as libc::off_t, POSIX_FADV_WILLNEED)
                        .offset(off)
                        .build()
                        // Advisory ops live in their own user_data namespace so a
                        // stray completion can never be mistaken for a read's.
                        .user_data(Self::ADVISORY_TAG | j as u64);
                    // SAFETY: Fadvise entries carry no buffer.
                    match unsafe { self.push_counted(&e) } {
                        Ok(()) => pushed += 1,
                        Err(e) => {
                            push_err = Some(e);
                            break;
                        }
                    }
                }
                // Whatever was pushed is already visible to the kernel: it must be
                // submitted and reaped before returning, or it would surface as a
                // stray completion during the caller's next read.
                let submitted = self.drain_pushed(pushed);
                if let Some(e) = push_err {
                    return Err(e);
                }
                submitted?;
                i = end;
            }
            Ok(())
        }

        /// Submit `pushed` queued entries and reap exactly that many completions,
        /// discarding their results. Used to return the ring to a clean state on
        /// an error path — an abandoned SQE is visible to the kernel, so leaving
        /// one queued means the kernel may still write into a buffer the caller
        /// is about to free, and its completion would corrupt the next batch's
        /// results.
        fn drain_pushed(&mut self, pushed: usize) -> io::Result<()> {
            if pushed == 0 {
                return Ok(());
            }
            let r = self.ring.submit_and_wait(pushed);
            let mut seen = 0usize;
            for _ in self.ring.completion() {
                seen += 1;
            }
            // A short reap means completions are still in flight; wait them out so
            // the ring is genuinely empty before the caller reuses it.
            while r.is_ok() && seen < pushed {
                self.ring.submit_and_wait(pushed - seen)?;
                let mut progressed = false;
                for _ in self.ring.completion() {
                    seen += 1;
                    progressed = true;
                }
                if !progressed {
                    break;
                }
            }
            r.map(|_| ())
        }

        /// Submit all `reqs` (chunked to the ring depth) and wait for every
        /// completion. Returns per-request result codes in `reqs` order. The
        /// buffers are filled directly by the kernel.
        pub fn read_many(&mut self, reqs: &mut [ReadReq]) -> io::Result<Vec<i64>> {
            // The wave's `submit_and_wait(pushed)` counts *any* completion, so
            // owned-lane traffic in flight would satisfy its wait in place of a
            // wave read. The two lanes never interleave on one ring.
            if !self.owned_idle() {
                return Err(Self::err_owned_busy("read_many"));
            }
            // `None` = no completion seen yet. A sentinel value would be
            // indistinguishable from a real result: the old `i64::MIN` flowed
            // into `-n` and reported "errno 0 = Success" in release builds.
            let mut results: Vec<Option<i64>> = vec![None; reqs.len()];
            let mut i = 0;
            while i < reqs.len() {
                let end = (i + self.cap).min(reqs.len());
                let mut pushed = 0usize;
                let mut push_err = None;
                for (j, req) in reqs.iter_mut().enumerate().take(end).skip(i) {
                    // A single op cannot carry more than u32 bytes; truncating
                    // would silently issue a short (or zero-length) read.
                    let Ok(len32) = u32::try_from(req.buf.len()) else {
                        push_err = Some(io::Error::other(format!(
                            "read request {j} is {} bytes; io_uring reads are limited to {} per op",
                            req.buf.len(),
                            u32::MAX
                        )));
                        break;
                    };
                    let ptr = req.buf.as_mut_ptr();
                    let off = req.offset;
                    // registered fd → fixed-file read (skips per-op fd lookup)
                    let fixed = self.registered.iter().position(|&f| f == req.fd);
                    let mut e = match fixed {
                        Some(idx) => opcode::Read::new(types::Fixed(idx as u32), ptr, len32).offset(off).build(),
                        None => opcode::Read::new(types::Fd(req.fd), ptr, len32).offset(off).build(),
                    }
                    .user_data(j as u64);
                    if self.force_async {
                        e = e.flags(squeue::Flags::ASYNC);
                    }
                    // SAFETY: buf outlives the op — read_many blocks until every
                    // completion for this chunk is reaped (including on the error
                    // paths below, which drain before returning).
                    match unsafe { self.push_counted(&e) } {
                        Ok(()) => pushed += 1,
                        Err(e) => {
                            push_err = Some(e);
                            break;
                        }
                    }
                }
                if let Some(err) = push_err {
                    // Entries already pushed are visible to the kernel and point
                    // at buffers the caller is about to drop — drain before
                    // surfacing the error. A drain failure is reported but does
                    // not replace the original error, which is the real cause.
                    if let Err(drain_err) = self.drain_pushed(pushed) {
                        crate::note_advisory_err("io_uring drain after failed submit", &drain_err);
                    }
                    return Err(err);
                }
                self.ring.submit_and_wait(pushed)?;
                let mut got = 0;
                while got < pushed {
                    let mut progressed = false;
                    for cqe in self.ring.completion() {
                        let ud = cqe.user_data();
                        if ud & Self::ADVISORY_TAG != 0 {
                            // stray advisory completion — not a read result
                            self.advisory_inflight = self.advisory_inflight.saturating_sub(1);
                            progressed = true;
                            continue;
                        }
                        match usize::try_from(ud).ok().and_then(|k| results.get_mut(k)) {
                            Some(slot) => {
                                *slot = Some(cqe.result() as i64);
                                got += 1;
                                progressed = true;
                            }
                            None => {
                                return Err(io::Error::other(format!(
                                    "io_uring completion for unknown request {ud}"
                                )))
                            }
                        }
                    }
                    if got < pushed {
                        // An advisory completion can satisfy part of the wait
                        // count; wait for the remainder rather than reporting a
                        // lost read. No progress at all is a real loss.
                        if !progressed {
                            return Err(io::Error::other(format!(
                                "io_uring returned {got} completions for {pushed} submitted reads"
                            )));
                        }
                        self.ring.submit_and_wait(pushed - got)?;
                    }
                }
                i = end;
            }
            let mut out = Vec::with_capacity(results.len());
            for (j, r) in results.into_iter().enumerate() {
                match r {
                    Some(v) => out.push(v),
                    None => {
                        return Err(io::Error::other(format!("io_uring completion missing for request {j}")))
                    }
                }
            }
            Ok(out)
        }

        /// Size the internal aligned-buffer pool used by [`Reactor::read_direct_many`].
        /// `buf_cap` should be the largest region streamed directly; `max_bufs`
        /// bounds total pool RAM. Call once before enabling the O_DIRECT path.
        pub fn configure_slab(&mut self, buf_cap: usize, max_bufs: usize) {
            self.slab = SlabPool::new(buf_cap, max_bufs);
        }

        /// Like [`Reactor::read_many`] but for **O_DIRECT** fds: for each request it
        /// reads the 4096-aligned superset `[align_down(off), align_up(off+len))`
        /// into a pooled aligned buffer, then copies out exactly `[off, off+len)`.
        /// The delivered bytes are byte-for-byte identical to a buffered read (only
        /// the page cache is bypassed). Each region is completed internally
        /// (including a legal EOF-short final aligned read on the last region of a
        /// shard), so `results[j] == reqs[j].buf.len()` on success. Requires the fd
        /// to be opened `O_DIRECT`; the buffer/offset/len alignment is handled here.
        pub fn read_direct_many(&mut self, reqs: &mut [ReadReq]) -> io::Result<Vec<i64>> {
            // Every entry is written before this returns (each region is completed
            // in the loop below or the whole call errors), so 0 is a safe start
            // rather than a sentinel that could escape as a bogus result code.
            let mut results = vec![0i64; reqs.len()];
            for (j, r) in reqs.iter_mut().enumerate() {
                let want = r.buf.len();
                if want == 0 {
                    results[j] = 0;
                    continue;
                }
                let a = ALIGN as u64;
                let a_off = align_down(r.offset, a);
                let head = (r.offset - a_off) as usize;
                let a_len = (align_up(r.offset + want as u64, a) - a_off) as usize;
                let need = head + want; // bytes we must have read before we can copy out

                // an aligned landing buffer: pooled (normal) or a one-off if the pool
                // buffer is too small (misconfigured) / momentarily exhausted.
                //
                // Generation-tagged: the pool records this checkout as outstanding
                // and rejects a second return of the same tag, which is the
                // protection `docs/io-and-storage.md` described as active while
                // the untagged pair was the only thing on this path. The handle is
                // split into `(buf, gen)` rather than held whole only because the
                // read body below borrows the buffer directly; that is sound here
                // because the buffer cannot escape this loop iteration — it is
                // returned unconditionally before `outcome?` can propagate.
                let (mut ab, gen) = match self.slab.checkout_tagged(a_len) {
                    Some(h) => (h.buf, Some(h.gen)),
                    None => (
                        AlignedBuf::with_capacity(a_len)
                            .ok_or_else(|| io::Error::other("aligned alloc failed"))?,
                        None,
                    ),
                };

                // Read the aligned window to completion, then copy the exact slice.
                // Captured as a Result so the buffer is always returned to the pool.
                let outcome: io::Result<()> = (|| {
                    let mut done = 0usize;
                    while done < need {
                        let n = {
                            let mut sub = [ReadReq {
                                fd: r.fd,
                                offset: a_off + done as u64,
                                buf: &mut ab.as_mut_slice()[done..a_len],
                                tag: 0,
                            }];
                            self.read_many(&mut sub)?[0]
                        };
                        if n < 0 {
                            return Err(Self::errno_from_result(n));
                        }
                        if n == 0 {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "O_DIRECT read hit EOF before covering the region",
                            ));
                        }
                        done += n as usize;
                        // O_DIRECT requires block-aligned offsets and buffers. The
                        // kernel only returns an unaligned count at EOF, so a short
                        // read that leaves `done` unaligned cannot be continued —
                        // retrying would fail with a bare EINVAL that says nothing
                        // about the real cause.
                        if done < need && !done.is_multiple_of(ALIGN) {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                format!(
                                    "O_DIRECT short read at unaligned boundary ({done} of {need} bytes) —                                      the region extends past the end of the file"
                                ),
                            ));
                        }
                    }
                    r.buf.copy_from_slice(&ab.as_slice()[head..need]);
                    Ok(())
                })();

                if let Some(g) = gen {
                    self.slab.checkin_tagged(SlabHandle { buf: ab, gen: g });
                }
                outcome?;
                results[j] = want as i64;
            }
            Ok(results)
        }

        /// **Zero-copy** O_DIRECT reads. For each `(fd, off, len)` it DMAs the
        /// 4096-aligned superset `[align_down(off), align_up(off+len))` into a freshly
        /// allocated [`AlignedBuf`] and returns it as [`Bytes::Aligned`] exposing
        /// exactly `[off, off+len)`. Unlike [`Reactor::read_direct_many`] there is
        /// **no realignment copy-out** — the consumer (a streamed `QtWeight`) reads
        /// straight out of the DMA target, so an O_DIRECT stream costs zero userspace
        /// copies of the bulk weight bytes (matching the buffered path, which already
        /// has the kernel fill the caller's buffer directly). The delivered bytes are
        /// byte-for-byte identical to a buffered read. Requires each `fd` opened
        /// `O_DIRECT`; the buffer/offset/length alignment is handled here. Each buffer
        /// is owned by its returned `Bytes` (it outlives the read into the weight), so
        /// this path allocates per region rather than using the reusable slab pool.
        pub fn read_direct_aligned(&mut self, reqs: &[(RawFd, u64, usize)]) -> io::Result<Vec<Bytes>> {
            // Every region's aligned window is submitted in ONE `read_many` call, so
            // the ring runs at the caller's batch depth (a 16-expert batch is 96
            // regions) instead of one region per `submit_and_wait`. The per-region
            // loop this replaced left the whole O_DIRECT lane at queue depth 1 no
            // matter what `COLI_IO_BATCH`/`COLI_IO_RINGS` said, which is why direct
            // reads measured *slower* than buffered ones. Peak memory is unchanged:
            // each region already owns its landing buffer for the returned `Bytes`.
            struct Span {
                orig: usize,
                fd: RawFd,
                a_off: u64,
                a_len: usize,
                head: usize,
                need: usize,
                want: usize,
                done: usize,
            }
            let mut slots: Vec<Option<Bytes>> = (0..reqs.len()).map(|_| None).collect();
            let mut spans: Vec<Span> = Vec::with_capacity(reqs.len());
            let mut bufs: Vec<AlignedBuf> = Vec::with_capacity(reqs.len());
            for (orig, &(fd, off, want)) in reqs.iter().enumerate() {
                if want == 0 {
                    slots[orig] = Some(Bytes::Vec(Vec::new()));
                    continue;
                }
                let a = ALIGN as u64;
                let a_off = align_down(off, a);
                let head = (off - a_off) as usize;
                let a_len = (align_up(off + want as u64, a) - a_off) as usize;
                bufs.push(
                    AlignedBuf::with_capacity(a_len).ok_or_else(|| io::Error::other("aligned alloc failed"))?,
                );
                // `need` is what must land before the region is complete; the tail
                // padding past it may legally EOF-short and is never read.
                spans.push(Span { orig, fd, a_off, a_len, head, need: head + want, want, done: 0 });
            }
            // Re-submit only the regions still short. A positioned read may return
            // short, so this converges rather than assuming one pass suffices.
            while spans.iter().any(|s| s.done < s.need) {
                let mut idx: Vec<usize> = Vec::new();
                let results = {
                    let mut batch: Vec<ReadReq> = Vec::new();
                    for (i, buf) in bufs.iter_mut().enumerate() {
                        let s = &spans[i];
                        if s.done >= s.need {
                            continue;
                        }
                        idx.push(i);
                        batch.push(ReadReq {
                            fd: s.fd,
                            offset: s.a_off + s.done as u64,
                            buf: &mut buf.as_mut_slice()[s.done..s.a_len],
                            tag: 0,
                        });
                    }
                    self.read_many(&mut batch)?
                };
                for (k, &i) in idx.iter().enumerate() {
                    let n = results.get(k).copied().ok_or_else(|| {
                        io::Error::other("io_uring returned fewer results than submitted direct reads")
                    })?;
                    if n < 0 {
                        return Err(Self::errno_from_result(n));
                    }
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "O_DIRECT read hit EOF before covering the region",
                        ));
                    }
                    spans[i].done += n as usize;
                }
            }
            for (s, buf) in spans.into_iter().zip(bufs) {
                slots[s.orig] = Some(Bytes::Aligned { buf, head: s.head, len: s.want });
            }
            slots
                .into_iter()
                .map(|b| b.ok_or_else(|| io::Error::other("direct read left a region unfilled")))
                .collect()
        }

        // ---- owned-completion lane ----------------------------------------

        /// True when the owned lane has nothing queued or in flight. The legacy
        /// wave calls require this: their `submit_and_wait(n)` counts *any*
        /// completion toward `n`, so an owned or advisory CQE arriving mid-wave
        /// would satisfy the wait in place of a wave read.
        fn owned_idle(&self) -> bool {
            self.pending.is_empty()
                && self.queued.is_empty()
                && self.advisory.is_empty()
                && self.advisory_inflight == 0
        }

        fn err_owned_busy(what: &str) -> io::Error {
            io::Error::other(format!(
                "{what}: owned-completion reads are in flight on this ring; reap them first"
            ))
        }

        /// Owned requests not yet handed back as [`RegionDone`]s: in flight,
        /// queued behind the ring cap, or ready (zero-length placeholders).
        pub fn pending_owned(&self) -> usize {
            self.pending.len() + self.queued.len() + self.ready.len()
        }

        /// Queue positioned reads whose landing buffers this Reactor owns until
        /// completion, then start as many as fit the ring. Completions are
        /// reaped incrementally with [`Reactor::reap_some`] — unlike
        /// [`Reactor::read_many`] there is no all-or-nothing wave, so the caller
        /// can hand each finished region onward while the rest are still in
        /// flight. With `direct` the fd must be opened `O_DIRECT`; the read
        /// covers the 4096-aligned superset and [`RegionDone::bytes`] exposes
        /// exactly the requested range, zero-copy, as `read_direct_aligned`.
        ///
        /// [`RegionDone::idx`] is the request's position in this call's `reqs`;
        /// submit one batch per claim so the mapping stays unambiguous.
        pub fn submit_owned(&mut self, reqs: Vec<OwnedReadReq>, direct: bool) -> io::Result<()> {
            // Validate and allocate for the whole batch before accepting any of
            // it: one op cannot carry more than u32 bytes, and failing after
            // half the batch was queued would leave orphaned reads to fire
            // during a later caller's reap.
            let mut ready: Vec<RegionDone> = Vec::new();
            let mut accepted: Vec<InFlight> = Vec::with_capacity(reqs.len());
            for (idx, r) in reqs.into_iter().enumerate() {
                if r.len == 0 {
                    ready.push(RegionDone { idx, tag: r.tag, bytes: Bytes::Vec(Vec::new()) });
                    continue;
                }
                let inf = if direct {
                    let a = ALIGN as u64;
                    let a_off = align_down(r.offset, a);
                    let head = (r.offset - a_off) as usize;
                    let a_len = (align_up(r.offset + r.len as u64, a) - a_off) as usize;
                    let buf = AlignedBuf::with_capacity(a_len)
                        .ok_or_else(|| io::Error::other("aligned alloc failed"))?;
                    InFlight {
                        fd: r.fd,
                        a_off,
                        head,
                        want: r.len,
                        need: head + r.len,
                        a_len,
                        done: 0,
                        tag: r.tag,
                        idx,
                        buf: LandingBuf::Aligned(buf),
                    }
                } else {
                    InFlight {
                        fd: r.fd,
                        a_off: r.offset,
                        head: 0,
                        want: r.len,
                        need: r.len,
                        a_len: r.len,
                        done: 0,
                        tag: r.tag,
                        idx,
                        buf: LandingBuf::Plain(vec![0u8; r.len]),
                    }
                };
                if u32::try_from(inf.a_len).is_err() {
                    return Err(io::Error::other(format!(
                        "owned read {idx} needs a {}-byte submission; io_uring reads are limited to {} per op",
                        inf.a_len,
                        u32::MAX
                    )));
                }
                accepted.push(inf);
            }
            self.ready.append(&mut ready);
            self.queued.extend(accepted);
            // Start I/O immediately: the kernel reads while the caller returns
            // to its reap loop (or keeps preparing work).
            self.pump_and_flush()
        }

        /// Build the submission entry for `inf`'s next (continuation) read.
        fn owned_sqe(&self, inf: &mut InFlight, ud: u64) -> io::Result<squeue::Entry> {
            let remaining = inf.a_len - inf.done;
            let Ok(len32) = u32::try_from(remaining) else {
                return Err(io::Error::other(format!(
                    "owned read continuation is {remaining} bytes; io_uring reads are limited to {} per op",
                    u32::MAX
                )));
            };
            let base = match &mut inf.buf {
                LandingBuf::Plain(v) => v.as_mut_ptr(),
                LandingBuf::Aligned(b) => b.as_mut_slice().as_mut_ptr(),
            };
            // SAFETY: `done < a_len` (remaining is the non-empty tail of the
            // same buffer), so `base + done` stays inside the allocation.
            let ptr = unsafe { base.add(inf.done) };
            let off = inf.a_off + inf.done as u64;
            // registered fd → fixed-file read (skips per-op fd lookup)
            let fixed = self.registered.iter().position(|&f| f == inf.fd);
            let mut e = match fixed {
                Some(i) => opcode::Read::new(types::Fixed(i as u32), ptr, len32).offset(off).build(),
                None => opcode::Read::new(types::Fd(inf.fd), ptr, len32).offset(off).build(),
            }
            .user_data(ud);
            if self.force_async {
                e = e.flags(squeue::Flags::ASYNC);
            }
            Ok(e)
        }

        /// Fill the ring from the owned queue (reads first), then let queued
        /// advisory entries ride whatever slots are left — reads always outrank
        /// hints. Returns whether anything was pushed (the caller then submits).
        fn pump(&mut self) -> io::Result<bool> {
            let mut pushed_any = false;
            while self.pending.len() < self.cap {
                let Some(mut inf) = self.queued.pop_front() else { break };
                let ud = Self::OWNED_TAG | self.next_ud;
                let e = match self.owned_sqe(&mut inf, ud) {
                    Ok(e) => e,
                    Err(err) => {
                        self.queued.push_front(inf);
                        return Err(err);
                    }
                };
                // SAFETY: the landing buffer is owned by `inf`, which moves into
                // `self.pending` below and stays there until this completion is
                // reaped (or waited out by `abort_owned`), so the kernel never
                // writes into freed memory.
                match unsafe { self.push_counted(&e) } {
                    Ok(()) => {
                        self.next_ud = (self.next_ud + 1) & (Self::OWNED_TAG - 1);
                        self.pending.insert(ud, inf);
                        pushed_any = true;
                    }
                    Err(e) => {
                        // SQ out of slots before the cap (already counted for
                        // the tuner): flush to the kernel; the next pump retries.
                        self.queued.push_front(inf);
                        crate::note_advisory_err("owned pump push rejected; flushing SQ", &e);
                        self.ring.submit()?;
                        pushed_any = false;
                        break;
                    }
                }
            }
            if self.queued.is_empty() {
                while self.pending.len() + self.advisory_inflight < self.cap {
                    let Some(e) = self.advisory.pop_front() else { break };
                    // SAFETY: advisory entries (fadvise) carry no buffer.
                    match unsafe { self.push_counted(&e) } {
                        Ok(()) => {
                            self.advisory_inflight += 1;
                            pushed_any = true;
                        }
                        Err(err) => {
                            self.advisory.push_front(e);
                            crate::note_advisory_err("advisory pump push rejected", &err);
                            break;
                        }
                    }
                }
            }
            Ok(pushed_any)
        }

        /// Pump, then hand anything pushed to the kernel without waiting.
        fn pump_and_flush(&mut self) -> io::Result<()> {
            if self.pump()? {
                self.ring.submit()?;
            }
            Ok(())
        }

        /// Reap whatever owned completions are available, blocking for at least
        /// one when any read is in flight. Appends finished regions to `out` and
        /// returns how many were appended (which is legitimately 0 after a wave
        /// of advisory or short completions — poll [`Reactor::pending_owned`],
        /// not this count, for lane emptiness). Short reads are resubmitted
        /// ahead of the backlog. On an error CQE the whole owned lane is drained
        /// (dropping queued work) before the error returns, so the ring is
        /// clean for whatever the caller does next.
        pub fn reap_some(&mut self, out: &mut Vec<RegionDone>) -> io::Result<usize> {
            if !self.ready.is_empty() {
                let n = self.ready.len();
                out.append(&mut self.ready);
                return Ok(n);
            }
            self.pump_and_flush()?;
            if self.pending.is_empty() {
                // Nothing to wait for. Absorb any advisory completions already
                // arrived so they never satisfy a later wave's wait.
                for cqe in self.ring.completion() {
                    if cqe.user_data() & Self::ADVISORY_TAG != 0 {
                        self.advisory_inflight = self.advisory_inflight.saturating_sub(1);
                    }
                }
                return Ok(0);
            }
            self.ring.submit_and_wait(1)?;
            let mut reaped = 0usize;
            let mut first_err: Option<io::Error> = None;
            for cqe in self.ring.completion() {
                let ud = cqe.user_data();
                if ud & Self::ADVISORY_TAG != 0 {
                    self.advisory_inflight = self.advisory_inflight.saturating_sub(1);
                    continue;
                }
                let n = cqe.result() as i64;
                let Some(mut inf) = self.pending.remove(&ud) else {
                    if first_err.is_none() {
                        first_err =
                            Some(io::Error::other(format!("io_uring completion for unknown owned request {ud}")));
                    }
                    continue;
                };
                if n < 0 {
                    if first_err.is_none() {
                        first_err = Some(Self::errno_from_result(n));
                    }
                    continue;
                }
                if n == 0 {
                    if first_err.is_none() {
                        first_err = Some(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("owned read hit EOF after {} of {} bytes", inf.done, inf.need),
                        ));
                    }
                    continue;
                }
                inf.done += n as usize;
                if inf.done >= inf.need {
                    out.push(inf.finish());
                    reaped += 1;
                } else if matches!(inf.buf, LandingBuf::Aligned(_)) && !inf.done.is_multiple_of(ALIGN) {
                    // O_DIRECT only returns an unaligned count at EOF; a
                    // continuation from an unaligned offset would EINVAL with no
                    // hint of the real cause (same rule as the wave lane).
                    if first_err.is_none() {
                        first_err = Some(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "O_DIRECT short read at unaligned boundary ({} of {} bytes) — \
                                 the region extends past the end of the file",
                                inf.done, inf.need
                            ),
                        ));
                    }
                } else {
                    // a nearly-done region resumes ahead of the backlog
                    self.queued.push_front(inf);
                }
            }
            if let Some(err) = first_err {
                self.abort_owned();
                return Err(err);
            }
            // Refill the slots this reap freed and start those reads now.
            self.pump_and_flush()?;
            Ok(reaped)
        }

        /// Queue `POSIX_FADV_WILLNEED` hints for `regions`, to ride SQ slots
        /// left over after owned reads. Purely advisory and droppable: the
        /// backlog is bounded (oldest dropped first) and submission failures are
        /// harmless — reads never wait on hints, hints never displace reads.
        pub fn queue_willneed(&mut self, regions: &[(RawFd, u64, usize)]) {
            const POSIX_FADV_WILLNEED: i32 = 3;
            self.queue_advisory(regions, POSIX_FADV_WILLNEED);
        }

        /// Queue `POSIX_FADV_DONTNEED` (drop page-cache references after the
        /// last consumer), same contract as [`Reactor::queue_willneed`].
        pub fn queue_dontneed(&mut self, regions: &[(RawFd, u64, usize)]) {
            const POSIX_FADV_DONTNEED: i32 = 4;
            self.queue_advisory(regions, POSIX_FADV_DONTNEED);
        }

        fn queue_advisory(&mut self, regions: &[(RawFd, u64, usize)], advice: i32) {
            for &(fd, off, len) in regions {
                let e = opcode::Fadvise::new(types::Fd(fd), len as libc::off_t, advice)
                    .offset(off)
                    .build()
                    .user_data(Self::ADVISORY_TAG);
                self.advisory.push_back(e);
            }
            // Hints are droppable by definition: bound the backlog so a caller
            // queuing faster than slots free can never grow it without limit.
            let cap = self.cap.saturating_mul(2).max(8);
            if self.advisory.len() > cap {
                let excess = self.advisory.len() - cap;
                self.advisory.drain(..excess);
            }
            // With reads in flight the hints ride the next pump; with the lane
            // idle there is no later pump, so push and flush now.
            if self.pending.is_empty() && self.queued.is_empty() {
                if let Err(e) = self.pump_and_flush() {
                    crate::note_advisory_err("advisory flush", &e);
                }
            }
        }

        /// Block until the owned lane is fully idle: every queued read pushed
        /// and reaped (appended to `out` exactly as [`Reactor::reap_some`]
        /// would), every advisory completion absorbed, leftover advisory hints
        /// dropped (they are droppable by definition). After this returns `Ok`,
        /// the wave-lane calls are usable again on this ring.
        pub fn quiesce_owned(&mut self, out: &mut Vec<RegionDone>) -> io::Result<()> {
            while self.pending_owned() > 0 {
                self.reap_some(out)?;
            }
            while self.advisory_inflight > 0 {
                self.ring.submit_and_wait(1)?;
                let mut progressed = false;
                for cqe in self.ring.completion() {
                    progressed = true;
                    if cqe.user_data() & Self::ADVISORY_TAG != 0 {
                        self.advisory_inflight = self.advisory_inflight.saturating_sub(1);
                    }
                }
                if !progressed {
                    return Err(io::Error::other("owned quiesce made no progress"));
                }
            }
            self.advisory.clear();
            Ok(())
        }

        /// Return the owned lane to a clean state after an error (or at drop):
        /// drop queued work and ready results, then wait out every in-flight
        /// completion — an abandoned CQE would satisfy a later wave's wait in
        /// place of a real completion, and an abandoned in-flight read points at
        /// a buffer about to be freed (`drain_pushed`'s contract, owned lane).
        fn abort_owned(&mut self) {
            self.queued.clear();
            self.advisory.clear();
            self.ready.clear();
            while !self.pending.is_empty() || self.advisory_inflight > 0 {
                if let Err(e) = self.ring.submit_and_wait(1) {
                    crate::note_advisory_err("owned-lane drain: submit_and_wait", &e);
                    break;
                }
                let mut progressed = false;
                for cqe in self.ring.completion() {
                    progressed = true;
                    let ud = cqe.user_data();
                    if ud & Self::ADVISORY_TAG != 0 {
                        self.advisory_inflight = self.advisory_inflight.saturating_sub(1);
                    } else if self.pending.remove(&ud).is_none() {
                        crate::note_advisory_err("owned-lane drain: stray completion", &ud);
                    }
                }
                if !progressed {
                    break;
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use uring::Reactor;

/// Probe whether O_DIRECT reads actually work on `fd` (some filesystems accept the
/// `O_DIRECT` open but reject aligned reads with `EINVAL` — overlayfs, tmpfs, some
/// network FS). Does a single 4096-aligned positioned read at offset 0 into an
/// aligned buffer; `true` iff it succeeds. The caller falls back to buffered I/O
/// on `false`.
#[cfg(unix)]
pub fn probe_direct(fd: RawFd) -> bool {
    use crate::slab::{AlignedBuf, ALIGN};
    use std::mem::ManuallyDrop;
    use std::os::unix::fs::FileExt;
    use std::os::unix::io::FromRawFd;
    let Some(mut buf) = AlignedBuf::with_capacity(ALIGN) else {
        return false;
    };
    // SAFETY: `fd` is a live descriptor the caller keeps open; `ManuallyDrop` stops
    // `File`'s Drop from closing a descriptor we don't own. We only read.
    let file = ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
    file.read_at(buf.as_mut_slice(), 0).is_ok()
}

#[cfg(not(unix))]
pub fn probe_direct(_fd: RawFd) -> bool {
    false
}

/// Non-Linux placeholder so dependents compile without `cfg`. Every method
/// errors — this engine's disk path is io_uring, with no pread fallback, so a
/// non-Linux build fails loudly at first use rather than silently degrading.
#[cfg(not(target_os = "linux"))]
pub struct Reactor;

#[cfg(not(target_os = "linux"))]
impl Reactor {
    fn unsupported<T>() -> std::io::Result<T> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "io_uring requires Linux"))
    }
    pub fn new(_entries: u32) -> std::io::Result<Reactor> {
        Self::unsupported()
    }
    pub fn new_streaming(_entries: u32) -> std::io::Result<Reactor> {
        Self::unsupported()
    }
    pub fn is_sqpoll(&self) -> bool {
        false
    }
    pub fn register_files(&mut self, _fds: &[RawFd]) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn is_registered(&self, _fd: RawFd) -> bool {
        false
    }
    pub fn read_many(&mut self, _reqs: &mut [ReadReq]) -> std::io::Result<Vec<i64>> {
        Self::unsupported()
    }
    pub fn read_exact(&mut self, _fd: RawFd, _off: u64, _buf: &mut [u8]) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn register_read_buffers(&mut self, _bufs: Vec<Vec<u8>>) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn fixed_buffer_count(&self) -> usize {
        0
    }
    pub fn fixed_buffer_capacity(&self) -> usize {
        0
    }
    pub fn read_fixed(&mut self, _fd: RawFd, _off: u64, _buf_index: u16, _out: &mut [u8]) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn read_fixed_many(&mut self, _reqs: &mut [ReadReq]) -> std::io::Result<Vec<i64>> {
        Self::unsupported().map(|_| Vec::new())
    }
    pub fn fadvise_willneed(&mut self, _fd: RawFd, _off: u64, _len: usize) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn fadvise_dontneed(&mut self, _fd: RawFd, _off: u64, _len: usize) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn fadvise_willneed_many(&mut self, _regions: &[(RawFd, u64, usize)]) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn read_exact_retry(&mut self, _fd: RawFd, _off: u64, _buf: &mut [u8], _max_retries: u32) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn take_sq_full(&mut self) -> u64 {
        0
    }
    pub fn configure_slab(&mut self, _buf_cap: usize, _max_bufs: usize) {}
    pub fn read_direct_many(&mut self, _reqs: &mut [ReadReq]) -> std::io::Result<Vec<i64>> {
        Self::unsupported()
    }
    pub fn read_direct_aligned(&mut self, _reqs: &[(RawFd, u64, usize)]) -> std::io::Result<Vec<crate::Bytes>> {
        Self::unsupported()
    }
    pub fn submit_owned(&mut self, _reqs: Vec<OwnedReadReq>, _direct: bool) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn pending_owned(&self) -> usize {
        0
    }
    pub fn reap_some(&mut self, _out: &mut Vec<RegionDone>) -> std::io::Result<usize> {
        Self::unsupported()
    }
    pub fn quiesce_owned(&mut self, _out: &mut Vec<RegionDone>) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn queue_willneed(&mut self, _regions: &[(RawFd, u64, usize)]) {}
    pub fn queue_dontneed(&mut self, _regions: &[(RawFd, u64, usize)]) {}
}

/// Read a whole file through io_uring (open → size → one ring-backed exact read).
/// For the small metadata files (`config.json`) and any full-file load; bulk
/// tensor reads use a persistent [`Reactor`] instead of a per-call ring.
pub fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path)?;
    let len = f.metadata()?.len() as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        let mut reactor = Reactor::new(1)?;
        reactor.read_exact(f.as_raw_fd(), 0, &mut buf)?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file_with(
        pattern: &[u8],
        n: usize,
    ) -> std::io::Result<(std::fs::File, std::path::PathBuf, Vec<u8>)> {
        let path = std::env::temp_dir().join(format!("peregrine_io_{}_{}", std::process::id(), n));
        let mut data = Vec::new();
        while data.len() < n {
            data.extend_from_slice(pattern);
        }
        data.truncate(n);
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&data)?;
        f.sync_all()?;
        let rf = std::fs::File::open(&path)?;
        Ok((rf, path, data))
    }

    #[test]
    fn pread_many_reads_offsets() -> std::io::Result<()> {
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"0123456789", 1000)?;
        let fd = f.as_raw_fd();
        let mut b0 = [0u8; 10];
        let mut b1 = [0u8; 16];
        let mut b2 = [0u8; 8];
        let mut reqs = vec![
            ReadReq { fd, offset: 0, buf: &mut b0, tag: 0 },
            ReadReq { fd, offset: 100, buf: &mut b1, tag: 1 },
            ReadReq { fd, offset: 500, buf: &mut b2, tag: 2 },
        ];
        let res = pread_many(&mut reqs);
        assert_eq!(res, vec![10, 16, 8]);
        assert_eq!(&b0, &data[0..10]);
        assert_eq!(&b1, &data[100..116]);
        assert_eq!(&b2, &data[500..508]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn threaded_pread_matches_serial_byte_for_byte() -> std::io::Result<()> {
        // The engine choice must be a performance decision and nothing else, so
        // the threaded split has to return exactly what the serial reader does —
        // same bytes, same per-request counts, same order — for any thread count
        // and any request count, including the awkward ones where the chunking
        // does not divide evenly.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"abcdefghij", 2000)?;
        let fd = f.as_raw_fd();
        for n_reqs in [1usize, 2, 3, 7, 16] {
            for threads in [1usize, 2, 3, 8, 64, 1000] {
                let mut want: Vec<Vec<u8>> = (0..n_reqs).map(|k| vec![0u8; 32 + k]).collect();
                let mut got: Vec<Vec<u8>> = (0..n_reqs).map(|k| vec![0u8; 32 + k]).collect();
                let off = |k: usize| (k * 101) as u64;

                let mut wr: Vec<ReadReq> = want
                    .iter_mut()
                    .enumerate()
                    .map(|(k, b)| ReadReq { fd, offset: off(k), buf: b.as_mut_slice(), tag: k as u64 })
                    .collect();
                let rs = pread_many(&mut wr);

                let mut gr: Vec<ReadReq> = got
                    .iter_mut()
                    .enumerate()
                    .map(|(k, b)| ReadReq { fd, offset: off(k), buf: b.as_mut_slice(), tag: k as u64 })
                    .collect();
                let rt = pread_many_threaded(&mut gr, threads);

                assert_eq!(rs, rt, "n={n_reqs} t={threads}: byte counts must match the serial reader");
                assert_eq!(want, got, "n={n_reqs} t={threads}: contents must match the serial reader");
                // …and match the file, so a shared bug in both readers still fails.
                for (k, b) in got.iter().enumerate() {
                    let s = off(k) as usize;
                    assert_eq!(&b[..], &data[s..s + b.len()], "n={n_reqs} t={threads} req {k}");
                }
            }
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_matches_pread() -> std::io::Result<()> {
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"abcdefghijklmnop", 8192)?;
        let fd = f.as_raw_fd();
        // 20 reads > ring depth 8 → exercises chunking
        let mut bufs: Vec<Vec<u8>> = (0..20).map(|k| vec![0u8; 64 + k]).collect();
        let offs: Vec<u64> = (0..20).map(|k| (k as u64 * 97) % 4000).collect();

        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable ({e}); skipping");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        // worker-cap tuning is a best-effort optimization
        if let Err(e) = reactor.set_iowq_max_workers(4, 4) {
            crate::note_advisory_err("iowq worker-cap tuning", &e);
        }
        let mut reqs: Vec<ReadReq> = bufs
            .iter_mut()
            .enumerate()
            .map(|(k, b)| ReadReq { fd, offset: offs[k], buf: b.as_mut_slice(), tag: k as u64 })
            .collect();
        let res = reactor.read_many(&mut reqs)?;

        for k in 0..20 {
            let len = 64 + k;
            assert_eq!(res[k], len as i64, "read {k} short");
            let off = offs[k] as usize;
            assert_eq!(&bufs[k][..], &data[off..off + len], "read {k} data mismatch");
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_registered_files_read() -> std::io::Result<()> {
        // reads through IOSQE_FIXED_FILE (registered fd) must return the same
        // bytes as a plain read.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"registered-file-payload", 4096)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        if reactor.register_files(&[fd]).is_err() {
            std::fs::remove_file(&path)?; // kernel without fixed-files → skip
            return Ok(());
        }
        assert!(reactor.is_registered(fd));
        let mut b0 = vec![0u8; 32];
        let mut b1 = vec![0u8; 40];
        let mut reqs = vec![
            ReadReq { fd, offset: 10, buf: &mut b0, tag: 0 },
            ReadReq { fd, offset: 100, buf: &mut b1, tag: 1 },
        ];
        let res = reactor.read_many(&mut reqs)?;
        assert_eq!(res, vec![32, 40]);
        assert_eq!(&b0[..], &data[10..42]);
        assert_eq!(&b1[..], &data[100..140]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn read_fixed_many_matches_pread_and_batches() -> std::io::Result<()> {
        // The batched fixed-buffer reader must agree with the pread oracle for
        // every region, across waves — the interesting case is more requests
        // than registered buffers, where the reader has to complete one wave
        // fully before reusing a buffer. Copying out mid-flight would tear.
        use std::os::unix::io::AsRawFd;
        // Size must be unique across this module: `temp_file_with` keys the
        // temp path on it, so a duplicate races another test's file.
        let (f, path, data) = temp_file_with(b"fixed-many-payload", 7000)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        // 3 buffers, 7 requests → 3 waves, the last one partial.
        if reactor.register_read_buffers(vec![vec![0u8; 512]; 3]).is_err() {
            std::fs::remove_file(&path)?; // kernel without buffer registration → skip
            return Ok(());
        }
        let n = 7usize;
        let off = |k: usize| (k * 313) as u64;
        let mut bufs: Vec<Vec<u8>> = (0..n).map(|k| vec![0u8; 64 + k * 10]).collect();
        let counts = {
            let mut reqs: Vec<ReadReq> = bufs
                .iter_mut()
                .enumerate()
                .map(|(k, b)| ReadReq { fd, offset: off(k), buf: b.as_mut_slice(), tag: k as u64 })
                .collect();
            reactor.read_fixed_many(&mut reqs)?
        };
        for (k, b) in bufs.iter().enumerate() {
            assert_eq!(counts[k], b.len() as i64, "request {k} must read fully");
            let s = off(k) as usize;
            assert_eq!(&b[..], &data[s..s + b.len()], "request {k} bytes must match the file");
        }

        // A request larger than the registered buffer is refused, not silently
        // truncated — a short read here would be indistinguishable from EOF.
        let mut big = vec![0u8; 513];
        let mut one = vec![ReadReq { fd, offset: 0, buf: big.as_mut_slice(), tag: 0 }];
        assert!(reactor.read_fixed_many(&mut one).is_err(), "oversized request must be refused");

        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn read_fixed_matches_pread() -> std::io::Result<()> {
        // A read through a registered fixed buffer (IORING_OP_READ_FIXED) must
        // return the same bytes as a plain positioned read.
        use std::os::unix::io::AsRawFd;
        // unique size ⇒ unique temp path (temp_file_with keys the path on size),
        // so this doesn't collide with other tests' files under parallel runs.
        let (f, path, data) = temp_file_with(b"registered-buffer-payload", 5000)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        if reactor.register_read_buffers(vec![vec![0u8; 256]]).is_err() {
            std::fs::remove_file(&path)?; // kernel without buffer registration → skip
            return Ok(());
        }
        assert_eq!(reactor.fixed_buffer_count(), 1);
        // read two different spans through the same registered slot
        let mut out = vec![0u8; 100];
        reactor.read_fixed(fd, 20, 0, &mut out)?;
        assert_eq!(&out[..], &data[20..120]);
        let mut out2 = vec![0u8; 200];
        reactor.read_fixed(fd, 0, 0, &mut out2)?;
        assert_eq!(&out2[..], &data[0..200]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fadvise_willneed_then_read() -> std::io::Result<()> {
        // FADVISE is advisory: a subsequent read must still return the right bytes.
        use std::os::unix::io::AsRawFd;
        // unique size ⇒ unique temp path (avoids colliding with other tests).
        let (f, path, data) = temp_file_with(b"willneed-warm-payload", 6000)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        if let Err(e) = reactor.fadvise_willneed(fd, 0, 8192) {
            crate::note_advisory_err("fadvise willneed hint", &e); // hint only; the read below still decides the test
        }
        let mut out = vec![0u8; 64];
        reactor.read_exact(fd, 100, &mut out)?;
        assert_eq!(&out[..], &data[100..164]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_read_matches_pread() -> std::io::Result<()> {
        // O_DIRECT reads of deliberately unaligned regions (incl. block-spanning
        // and an EOF-short tail — the file is NOT a multiple of 4096) must return
        // the same bytes as a plain positioned read. Skips gracefully when the temp
        // filesystem rejects O_DIRECT (overlayfs/tmpfs in CI containers).
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"O_DIRECT-alignment-payload!", 40000)?;
        drop(f); // buffered fd not needed — reads go through the O_DIRECT fd below
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(e) => {
                eprintln!("skipping: filesystem rejects O_DIRECT open: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(()); // open ok but aligned reads rejected (EINVAL) → skip
        }
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        reactor.configure_slab(4096 * 12, 2);
        // (offset, len): unaligned head, block-spanning, exact block, big span, EOF tail
        let cases = [(100u64, 7usize), (4090, 20), (0, 4096), (8191, 4098), (39997, 3)];
        for &(off, len) in &cases {
            let mut got = vec![0u8; len];
            let mut reqs = [ReadReq { fd: dfd, offset: off, buf: &mut got, tag: 0 }];
            match reactor.read_direct_many(&mut reqs) {
                Ok(res) => assert_eq!(res[0], len as i64, "off={off} len={len}"),
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    std::fs::remove_file(&path)?;
                    return Ok(()); // late EINVAL → skip
                }
                Err(e) => return Err(e),
            }
            let o = off as usize;
            assert_eq!(&got[..], &data[o..o + len], "direct off={off} len={len} bytes mismatch");
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_aligned_matches_pread() -> std::io::Result<()> {
        // Zero-copy O_DIRECT: `read_direct_aligned` must expose exactly [off,off+len)
        // of each region (unaligned head, block-spanning, exact block, big span, EOF
        // tail), byte-identical to the source — proving the aligned DMA buffer + its
        // head/len view need no realignment copy. Skips when the temp FS rejects
        // O_DIRECT (overlayfs/tmpfs in CI containers).
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;
        // unique size ⇒ unique temp path (avoids colliding with other tests).
        let (f, path, data) = temp_file_with(b"zero-copy-aligned-DMA-payload!", 41000)?;
        drop(f); // buffered fd not needed — reads go through the O_DIRECT fd below
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(e) => {
                eprintln!("skipping: filesystem rejects O_DIRECT open: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(()); // open ok but aligned reads rejected (EINVAL) → skip
        }
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        // (offset, len): unaligned head, block-spanning, exact block, big span, EOF tail
        let cases = [(100u64, 7usize), (4090, 20), (0, 4096), (8191, 4098), (40997, 3)];
        let reqs: Vec<_> = cases.iter().map(|&(off, len)| (dfd, off, len)).collect();
        let got = match reactor.read_direct_aligned(&reqs) {
            Ok(g) => g,
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                std::fs::remove_file(&path)?;
                return Ok(()); // late EINVAL → skip
            }
            Err(e) => return Err(e),
        };
        for (k, &(off, len)) in cases.iter().enumerate() {
            assert_eq!(got[k].len(), len, "region {k} wrong len");
            let o = off as usize;
            assert_eq!(&got[k][..], &data[o..o + len], "aligned off={off} len={len} bytes mismatch");
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_aligned_batch_exceeding_ring_capacity_matches_serial() -> std::io::Result<()> {
        // `read_direct_aligned` submits every region in one `read_many`, which
        // chunks internally to the ring's entry count. Submitting more regions
        // than the ring holds must therefore still deliver each region's exact
        // bytes, in order — the case a per-region loop could never get wrong and
        // batched submission can. Serial single-region calls are the oracle.
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"batched-direct-submission-oracle", 43000)?;
        drop(f);
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(e) => {
                eprintln!("skipping: filesystem rejects O_DIRECT open: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(());
        }
        // 40 regions through an 8-entry ring: 5 submission chunks.
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let cases: Vec<(RawFd, u64, usize)> =
            (0..40u64).map(|i| (dfd, i * 1013 + 7, 100 + (i as usize % 37))).collect();
        let batched = match reactor.read_direct_aligned(&cases) {
            Ok(g) => g,
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                std::fs::remove_file(&path)?;
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        assert_eq!(batched.len(), cases.len(), "one result per region, in order");
        for (k, &(_, off, len)) in cases.iter().enumerate() {
            let serial = reactor.read_direct_aligned(&[(dfd, off, len)])?;
            let o = off as usize;
            assert_eq!(&batched[k][..], &data[o..o + len], "batched region {k} bytes mismatch");
            assert_eq!(&batched[k][..], &serial[0][..], "batched region {k} differs from serial read");
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    // ---- owned-completion lane ----------------------------------------------

    /// Drive `cases` through the owned lane and assert every region's bytes
    /// match `data`, keyed by `idx`, with tags echoed. Shared body for the
    /// owned-lane byte-identity tests.
    #[cfg(target_os = "linux")]
    fn owned_roundtrip_asserts(
        reactor: &mut Reactor,
        fd: RawFd,
        cases: &[(u64, usize)],
        data: &[u8],
    ) -> std::io::Result<()> {
        let reqs: Vec<OwnedReadReq> = cases
            .iter()
            .enumerate()
            .map(|(k, &(off, len))| OwnedReadReq { fd, offset: off, len, tag: k as u64 * 7 })
            .collect();
        reactor.submit_owned(reqs, false)?;
        // Reap incrementally (the production shape), then quiesce the tail.
        let mut done: Vec<RegionDone> = Vec::new();
        while reactor.pending_owned() > 0 {
            reactor.reap_some(&mut done)?;
        }
        reactor.quiesce_owned(&mut done)?;
        assert_eq!(done.len(), cases.len(), "every region delivered exactly once");
        done.sort_by_key(|d| d.idx);
        for (k, d) in done.iter().enumerate() {
            assert_eq!(d.idx, k, "regions keyed by submission index");
            assert_eq!(d.tag, k as u64 * 7, "caller tag echoed");
            let (off, len) = cases[k];
            let o = off as usize;
            assert_eq!(d.bytes.len(), len, "region {k} wrong length");
            assert_eq!(&d.bytes[..], &data[o..o + len], "region {k} bytes mismatch");
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_completion_matches_pread_oracle() -> std::io::Result<()> {
        // The owned lane must deliver exactly [offset, offset+len) per request,
        // byte-identical to the file (the same oracle the wave lane pins),
        // including a zero-length region, which completes at submit time.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"owned-lane-oracle-payload", 50021)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let cases: Vec<(u64, usize)> =
            vec![(0, 10), (100, 0), (999, 1), (4096, 4096), (12345, 678), (50011, 10)];
        owned_roundtrip_asserts(&mut reactor, fd, &cases, &data)?;
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_completion_out_of_order_arrival() -> std::io::Result<()> {
        // Forced IOSQE_ASYNC (the default) hands every read to io-wq, whose
        // workers finish in whatever order they please — delivery is keyed by
        // `idx`, so arrival order must be unobservable in the results.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"out-of-order-owned-arrival", 64007)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let cases: Vec<(u64, usize)> =
            (0..48u64).map(|i| ((i * 1327) % 60000, 17 + (i as usize % 900))).collect();
        owned_roundtrip_asserts(&mut reactor, fd, &cases, &data)?;
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_batch_exceeds_ring_cap() -> std::io::Result<()> {
        // 64 regions through an 8-entry ring: the internal queue must feed the
        // ring as completions free slots, and still deliver every region once.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"owned-cap-backpressure", 33013)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let cases: Vec<(u64, usize)> =
            (0..64u64).map(|i| ((i * 499) % 32000, 13 + (i as usize % 400))).collect();
        owned_roundtrip_asserts(&mut reactor, fd, &cases, &data)?;
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_short_eof_errors_and_ring_stays_clean() -> std::io::Result<()> {
        // A region extending past EOF must surface as UnexpectedEof, and the
        // failure must leave the ring clean enough that a fresh owned batch on
        // the same Reactor succeeds (abort_owned's contract).
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"eof-short-owned", 5003)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let reqs = vec![
            OwnedReadReq { fd, offset: 0, len: 100, tag: 0 },
            OwnedReadReq { fd, offset: 4900, len: 500, tag: 1 }, // ends 397 bytes past EOF
        ];
        reactor.submit_owned(reqs, false)?;
        let mut done: Vec<RegionDone> = Vec::new();
        let mut saw_err = false;
        while reactor.pending_owned() > 0 {
            match reactor.reap_some(&mut done) {
                Ok(_) => {}
                Err(e) => {
                    assert_eq!(
                        e.kind(),
                        std::io::ErrorKind::UnexpectedEof,
                        "EOF must surface as UnexpectedEof, got: {e}"
                    );
                    saw_err = true;
                    break;
                }
            }
        }
        assert!(saw_err, "a region past EOF must error");
        assert_eq!(reactor.pending_owned(), 0, "the error drain leaves the lane empty");
        let reqs = vec![OwnedReadReq { fd, offset: 10, len: 64, tag: 9 }];
        reactor.submit_owned(reqs, false)?;
        let mut done2: Vec<RegionDone> = Vec::new();
        reactor.quiesce_owned(&mut done2)?;
        assert_eq!(done2.len(), 1, "fresh batch after the error must complete");
        assert_eq!(&done2[0].bytes[..], &data[10..74]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_direct_matches_pread() -> std::io::Result<()> {
        // Owned O_DIRECT reads: unaligned head, block-spanning, exact block,
        // big span, EOF tail — each delivered zero-copy as the exact requested
        // range, byte-identical to the file. Skips when the temp FS rejects
        // O_DIRECT (overlayfs/tmpfs in CI containers).
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"owned-direct-DMA-payload", 41999)?;
        drop(f); // buffered fd not needed — reads go through the O_DIRECT fd below
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(e) => {
                eprintln!("skipping: filesystem rejects O_DIRECT open: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(()); // open ok but aligned reads rejected (EINVAL) → skip
        }
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let cases = [(100u64, 7usize), (4090, 20), (0, 4096), (8191, 4098), (41996, 3)];
        let reqs: Vec<OwnedReadReq> = cases
            .iter()
            .enumerate()
            .map(|(k, &(off, len))| OwnedReadReq { fd: dfd, offset: off, len, tag: k as u64 })
            .collect();
        reactor.submit_owned(reqs, true)?;
        let mut done: Vec<RegionDone> = Vec::new();
        match reactor.quiesce_owned(&mut done) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                std::fs::remove_file(&path)?;
                return Ok(()); // late EINVAL → skip
            }
            Err(e) => return Err(e),
        }
        assert_eq!(done.len(), cases.len());
        done.sort_by_key(|d| d.idx);
        for (k, d) in done.iter().enumerate() {
            let (off, len) = cases[k];
            let o = off as usize;
            assert_eq!(d.bytes.len(), len, "direct region {k} wrong length");
            assert_eq!(&d.bytes[..], &data[o..o + len], "direct region {k} bytes mismatch");
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owned_with_queued_advisory_unaffected() -> std::io::Result<()> {
        // Advisory hints ride spare SQ slots without perturbing read results;
        // the wave lane refuses to interleave with the owned lane and works
        // again once the lane is quiesced.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"advisory-riding-owned", 27011)?;
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        // hints queued while idle flush immediately (and harmlessly)
        reactor.queue_willneed(&[(fd, 0, 8192)]);
        let cases: Vec<(u64, usize)> =
            (0..24u64).map(|i| ((i * 997) % 26000, 33 + (i as usize % 250))).collect();
        let reqs: Vec<OwnedReadReq> = cases
            .iter()
            .enumerate()
            .map(|(k, &(off, len))| OwnedReadReq { fd, offset: off, len, tag: k as u64 })
            .collect();
        reactor.submit_owned(reqs, false)?;
        // more hints while reads are in flight: they wait behind reads and must
        // never surface in the read results
        reactor.queue_willneed(&[(fd, 8192, 8192), (fd, 16384, 8192)]);
        // the wave lane must refuse to interleave
        let mut probe = [0u8; 8];
        let mut wave = [ReadReq { fd, offset: 0, buf: &mut probe, tag: 0 }];
        assert!(
            reactor.read_many(&mut wave).is_err(),
            "wave read must refuse while owned reads are in flight"
        );
        let mut done: Vec<RegionDone> = Vec::new();
        reactor.quiesce_owned(&mut done)?;
        assert_eq!(done.len(), cases.len());
        done.sort_by_key(|d| d.idx);
        for (k, d) in done.iter().enumerate() {
            let (off, len) = cases[k];
            let o = off as usize;
            assert_eq!(&d.bytes[..], &data[o..o + len], "region {k} bytes mismatch");
        }
        // dontneed on an idle lane flushes immediately; then the wave lane works
        reactor.queue_dontneed(&[(fd, 0, 8192)]);
        let mut done2: Vec<RegionDone> = Vec::new();
        reactor.quiesce_owned(&mut done2)?;
        let mut after = vec![0u8; 40];
        let mut wave2 = [ReadReq { fd, offset: 5, buf: &mut after, tag: 0 }];
        let res = reactor.read_many(&mut wave2)?;
        assert_eq!(res, vec![40], "wave lane usable after quiesce");
        assert_eq!(&after[..], &data[5..45]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sqpoll_ring_matches_plain() -> std::io::Result<()> {
        // A SQPOLL ring must return the same bytes as a plain ring, on both the
        // wave lane and the owned lane. Built via env-free `new_with` (no env
        // mutation — parallel-test-safe); skips when the kernel refuses SQPOLL
        // (pre-5.13 unprivileged, io_uring restricted, etc).
        use super::uring::{RingOpts, SqpollOpts};
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"sqpoll-vs-plain-payload", 29009)?;
        let fd = f.as_raw_fd();
        let mut sq = match Reactor::new_with(8, RingOpts { sqpoll: Some(SqpollOpts { idle_ms: 100, cpu: None }) }) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        if !sq.is_sqpoll() {
            eprintln!("skipping: kernel refused SQPOLL (fell back to plain ring)");
            std::fs::remove_file(&path)?;
            return Ok(());
        }
        // wave lane
        let mut b0 = vec![0u8; 100];
        let mut b1 = vec![0u8; 999];
        let mut reqs = vec![
            ReadReq { fd, offset: 11, buf: &mut b0, tag: 0 },
            ReadReq { fd, offset: 20000, buf: &mut b1, tag: 1 },
        ];
        let res = sq.read_many(&mut reqs)?;
        assert_eq!(res, vec![100, 999]);
        assert_eq!(&b0[..], &data[11..111]);
        assert_eq!(&b1[..], &data[20000..20999]);
        // owned lane
        let cases: Vec<(u64, usize)> =
            (0..20u64).map(|i| ((i * 1409) % 28000, 21 + (i as usize % 300))).collect();
        owned_roundtrip_asserts(&mut sq, fd, &cases, &data)?;
        std::fs::remove_file(&path)?;
        Ok(())
    }
}

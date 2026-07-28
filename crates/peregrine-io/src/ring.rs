//! The I/O lane: batched positioned reads. On Linux the [`Reactor`] uses
//! io_uring — one `submit_and_wait` drives up to a full ring of reads through
//! io-wq (forced `IOSQE_ASYNC`, matching `c/uring.h`), so N expert-slab reads
//! cost one enter syscall instead of N `pread`s. [`pread_many`] is the portable
//! fallback (and the correctness oracle for the ring).

use std::os::unix::io::RawFd;

/// One positioned read into a caller-owned buffer.
pub struct ReadReq<'a> {
    pub fd: RawFd,
    pub offset: u64,
    pub buf: &'a mut [u8],
    /// caller tag echoed back (e.g. an expert id); not used by the reader
    pub tag: u64,
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

#[cfg(target_os = "linux")]
mod uring {
    use super::ReadReq;
    use crate::slab::{align_down, align_up, AlignedBuf, Bytes, SlabPool, ALIGN};
    use io_uring::{opcode, squeue, types, IoUring};
    use std::io;
    use std::os::unix::io::RawFd;

    /// io_uring-backed batched reader (the I/O lane owner thread holds one).
    pub struct Reactor {
        ring: IoUring,
        cap: usize,
        force_async: bool,
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
    }

    impl Reactor {
        /// `entries` = submission-queue depth (rounded up to a power of two by the
        /// kernel). Cold NVMe streaming wants this ≥ the per-layer expert count.
        ///
        /// The ring is set up with `COOP_TASKRUN` (completion task work runs
        /// cooperatively at `io_uring_enter` instead of via IPIs → less overhead).
        /// We deliberately do **not** set `SINGLE_ISSUER`: the streaming scheduler
        /// reuses one persistent `Reactor` across `moe_streamed` calls that submit
        /// from different (scoped) worker threads, and single-issuer would reject
        /// a second submitting task with `-EEXIST`. If a kernel rejects the flag we
        /// fall back to a plain ring. (`SQPOLL` needs privileges → future opt-in.)
        pub fn new(entries: u32) -> io::Result<Reactor> {
            let ring = IoUring::builder()
                .setup_coop_taskrun()
                .build(entries)
                .or_else(|_| IoUring::new(entries))?;
            Ok(Reactor {
                ring,
                cap: entries as usize,
                force_async: true,
                registered: Vec::new(),
                registered_bufs: Vec::new(),
                slab: SlabPool::new(ALIGN, 1),
            })
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
                    return Err(io::Error::from_raw_os_error((-n) as i32));
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
            Ok(())
        }

        /// The number of registered fixed buffers.
        pub fn fixed_buffer_count(&self) -> usize {
            self.registered_bufs.len()
        }

        /// Read exactly `out.len()` bytes at `off` from `fd` **through the
        /// registered fixed buffer** `buf_index` (C2), then copy them into `out`.
        /// Loops to complete a short read. Byte-identical to [`Reactor::read_exact`];
        /// the copy-out is the trade-off of the owned-buffer hand-off (which is why
        /// the streaming path keeps the plain read by default — see `COLI_REGBUF`).
        pub fn read_fixed(&mut self, fd: RawFd, off: u64, buf_index: u16, out: &mut [u8]) -> io::Result<()> {
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
                unsafe {
                    self.ring.submission().push(&e).map_err(|_| io::Error::other("submission queue full"))?;
                }
                self.ring.submit_and_wait(1)?;
                let mut n = i64::MIN;
                for cqe in self.ring.completion() {
                    n = cqe.result() as i64;
                }
                if n < 0 {
                    return Err(io::Error::from_raw_os_error((-n) as i32));
                }
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_fixed hit EOF"));
                }
                done += n as usize;
            }
            out.copy_from_slice(&self.registered_bufs[bi][..total]);
            Ok(())
        }

        /// Advise the kernel to read `[off, off+len)` of `fd` into the page cache
        /// ahead of a real read (`IORING_OP_FADVISE`, `POSIX_FADV_WILLNEED`) — C3
        /// far-ahead warming. Purely advisory: it moves no bytes and cannot affect
        /// output, so a soft failure is harmless (a hard submit error is surfaced).
        pub fn fadvise_willneed(&mut self, fd: RawFd, off: u64, len: usize) -> io::Result<()> {
            const POSIX_FADV_WILLNEED: i32 = 3;
            let e = opcode::Fadvise::new(types::Fd(fd), len as libc::off_t, POSIX_FADV_WILLNEED)
                .offset(off)
                .build()
                .user_data(0);
            // SAFETY: Fadvise carries no buffer — the op only hints the page cache.
            unsafe {
                self.ring.submission().push(&e).map_err(|_| io::Error::other("submission queue full"))?;
            }
            self.ring.submit_and_wait(1)?;
            let mut n = 0i64;
            for cqe in self.ring.completion() {
                n = cqe.result() as i64;
            }
            if n < 0 {
                return Err(io::Error::from_raw_os_error((-n) as i32));
            }
            Ok(())
        }

        /// Submit all `reqs` (chunked to the ring depth) and wait for every
        /// completion. Returns per-request result codes in `reqs` order. The
        /// buffers are filled directly by the kernel.
        pub fn read_many(&mut self, reqs: &mut [ReadReq]) -> io::Result<Vec<i64>> {
            let mut results = vec![i64::MIN; reqs.len()];
            let mut i = 0;
            while i < reqs.len() {
                let end = (i + self.cap).min(reqs.len());
                for j in i..end {
                    let (ptr, len) = (reqs[j].buf.as_mut_ptr(), reqs[j].buf.len() as u32);
                    let off = reqs[j].offset;
                    // registered fd → fixed-file read (skips per-op fd lookup)
                    let fixed = self.registered.iter().position(|&f| f == reqs[j].fd);
                    let mut e = match fixed {
                        Some(idx) => opcode::Read::new(types::Fixed(idx as u32), ptr, len).offset(off).build(),
                        None => opcode::Read::new(types::Fd(reqs[j].fd), ptr, len).offset(off).build(),
                    }
                    .user_data(j as u64);
                    if self.force_async {
                        e = e.flags(squeue::Flags::ASYNC);
                    }
                    // SAFETY: buf outlives the op — read_many blocks until every
                    // completion for this chunk is reaped below.
                    unsafe {
                        self.ring
                            .submission()
                            .push(&e)
                            .map_err(|_| io::Error::other("submission queue full"))?;
                    }
                }
                self.ring.submit_and_wait(end - i)?;
                let mut got = 0;
                for cqe in self.ring.completion() {
                    results[cqe.user_data() as usize] = cqe.result() as i64;
                    got += 1;
                }
                debug_assert_eq!(got, end - i);
                i = end;
            }
            Ok(results)
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
            let mut results = vec![i64::MIN; reqs.len()];
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
                let (mut ab, pooled) = match self.slab.checkout(a_len) {
                    Some(b) => (b, true),
                    None => (
                        AlignedBuf::with_capacity(a_len)
                            .ok_or_else(|| io::Error::other("aligned alloc failed"))?,
                        false,
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
                            return Err(io::Error::from_raw_os_error((-n) as i32));
                        }
                        if n == 0 {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "O_DIRECT read hit EOF before covering the region",
                            ));
                        }
                        done += n as usize;
                    }
                    r.buf.copy_from_slice(&ab.as_slice()[head..need]);
                    Ok(())
                })();

                if pooled {
                    self.slab.checkin(ab);
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
            let mut out: Vec<Bytes> = Vec::with_capacity(reqs.len());
            for &(fd, off, want) in reqs {
                if want == 0 {
                    out.push(Bytes::Vec(Vec::new()));
                    continue;
                }
                let a = ALIGN as u64;
                let a_off = align_down(off, a);
                let head = (off - a_off) as usize;
                let a_len = (align_up(off + want as u64, a) - a_off) as usize;
                let need = head + want; // bytes present before the region is complete
                let mut buf =
                    AlignedBuf::with_capacity(a_len).ok_or_else(|| io::Error::other("aligned alloc failed"))?;
                // Read the aligned window to completion. A positioned read may return
                // short; the final aligned read of a shard may legally EOF-short once
                // `need` is covered (the tail padding past the file end is never read).
                let mut done = 0usize;
                while done < need {
                    let n = {
                        let mut sub =
                            [ReadReq { fd, offset: a_off + done as u64, buf: &mut buf.as_mut_slice()[done..a_len], tag: 0 }];
                        self.read_many(&mut sub)?[0]
                    };
                    if n < 0 {
                        return Err(io::Error::from_raw_os_error((-n) as i32));
                    }
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "O_DIRECT read hit EOF before covering the region",
                        ));
                    }
                    done += n as usize;
                }
                out.push(Bytes::Aligned { buf, head, len: want });
            }
            Ok(out)
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
    pub fn read_fixed(&mut self, _fd: RawFd, _off: u64, _buf_index: u16, _out: &mut [u8]) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn fadvise_willneed(&mut self, _fd: RawFd, _off: u64, _len: usize) -> std::io::Result<()> {
        Self::unsupported()
    }
    pub fn configure_slab(&mut self, _buf_cap: usize, _max_bufs: usize) {}
    pub fn read_direct_many(&mut self, _reqs: &mut [ReadReq]) -> std::io::Result<Vec<i64>> {
        Self::unsupported()
    }
    pub fn read_direct_aligned(&mut self, _reqs: &[(RawFd, u64, usize)]) -> std::io::Result<Vec<crate::Bytes>> {
        Self::unsupported()
    }
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
        // worker-cap tuning is a best-effort optimization; ignore if unsupported
        let _ = reactor.set_iowq_max_workers(4, 4);
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
            Err(_) => {
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
            Err(_) => {
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
            Err(_) => {
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let _ = reactor.fadvise_willneed(fd, 0, 8192); // hint; ignore if unsupported
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
        let _ = f; // buffered fd unused; we read through the O_DIRECT fd below
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(_) => {
                std::fs::remove_file(&path)?;
                return Ok(()); // filesystem rejects the O_DIRECT open → skip
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(()); // open ok but aligned reads rejected (EINVAL) → skip
        }
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(_) => {
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
        let _ = f; // buffered fd unused; we read through the O_DIRECT fd below
        let df = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path) {
            Ok(df) => df,
            Err(_) => {
                std::fs::remove_file(&path)?;
                return Ok(()); // filesystem rejects the O_DIRECT open → skip
            }
        };
        let dfd = df.as_raw_fd();
        if !super::probe_direct(dfd) {
            std::fs::remove_file(&path)?;
            return Ok(()); // open ok but aligned reads rejected (EINVAL) → skip
        }
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(_) => {
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
}

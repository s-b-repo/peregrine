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
    }

    impl Reactor {
        /// `entries` = submission-queue depth (rounded up to a power of two by the
        /// kernel). Cold NVMe streaming wants this ≥ the per-layer expert count.
        ///
        /// The ring is set up with `SINGLE_ISSUER` (all submissions come from the
        /// one owner thread → the kernel skips per-op submitter locking) and
        /// `COOP_TASKRUN` (completion task work is run cooperatively at
        /// `io_uring_enter` instead of via IPIs → less overhead). Both match our
        /// one-thread-drains-the-ring model and reduce per-batch cost. If a kernel
        /// rejects the flags we fall back to a plain ring. (`SQPOLL` — submission
        /// with no `enter` syscall at all — is a further win but needs privileges,
        /// so it's left opt-in for future work.)
        pub fn new(entries: u32) -> io::Result<Reactor> {
            let ring = IoUring::builder()
                .setup_single_issuer()
                .setup_coop_taskrun()
                .build(entries)
                .or_else(|_| IoUring::new(entries))?;
            Ok(Reactor { ring, cap: entries as usize, force_async: true, registered: Vec::new() })
        }

        /// Register `fds` as fixed files. Subsequent reads whose fd is in this set
        /// use `IOSQE_FIXED_FILE`, so the kernel skips the per-op fd table lookup
        /// and refcount — worthwhile when the same shard fds are read every token.
        /// Replaces any previous registration. Errors are non-fatal to the caller:
        /// on failure, reads simply fall back to the plain-fd path.
        pub fn register_files(&mut self, fds: &[RawFd]) -> io::Result<()> {
            if !self.registered.is_empty() {
                let _ = self.ring.submitter().unregister_files();
                self.registered.clear();
            }
            self.ring.submitter().register_files(fds)?;
            self.registered = fds.to_vec();
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
    }
}

#[cfg(target_os = "linux")]
pub use uring::Reactor;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file_with(pattern: &[u8], n: usize) -> (std::fs::File, std::path::PathBuf, Vec<u8>) {
        let path = std::env::temp_dir().join(format!("peregrine_io_{}_{}", std::process::id(), n));
        let mut data = Vec::new();
        while data.len() < n {
            data.extend_from_slice(pattern);
        }
        data.truncate(n);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        f.sync_all().unwrap();
        let rf = std::fs::File::open(&path).unwrap();
        (rf, path, data)
    }

    #[test]
    fn pread_many_reads_offsets() {
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"0123456789", 1000);
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
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_matches_pread() {
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"abcdefghijklmnop", 8192);
        let fd = f.as_raw_fd();
        // 20 reads > ring depth 8 → exercises chunking
        let mut bufs: Vec<Vec<u8>> = (0..20).map(|k| vec![0u8; 64 + k]).collect();
        let offs: Vec<u64> = (0..20).map(|k| (k as u64 * 97) % 4000).collect();

        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable ({e}); skipping");
                let _ = std::fs::remove_file(&path);
                return;
            }
        };
        let _ = reactor.set_iowq_max_workers(4, 4);
        let mut reqs: Vec<ReadReq> = bufs
            .iter_mut()
            .enumerate()
            .map(|(k, b)| ReadReq { fd, offset: offs[k], buf: b.as_mut_slice(), tag: k as u64 })
            .collect();
        let res = reactor.read_many(&mut reqs).unwrap();

        for k in 0..20 {
            let len = 64 + k;
            assert_eq!(res[k], len as i64, "read {k} short");
            let off = offs[k] as usize;
            assert_eq!(&bufs[k][..], &data[off..off + len], "read {k} data mismatch");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_registered_files_read() {
        // reads through IOSQE_FIXED_FILE (registered fd) must return the same
        // bytes as a plain read.
        use std::os::unix::io::AsRawFd;
        let (f, path, data) = temp_file_with(b"registered-file-payload", 4096);
        let fd = f.as_raw_fd();
        let mut reactor = match Reactor::new(8) {
            Ok(r) => r,
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                return;
            }
        };
        if reactor.register_files(&[fd]).is_err() {
            let _ = std::fs::remove_file(&path); // kernel without fixed-files → skip
            return;
        }
        assert!(reactor.is_registered(fd));
        let mut b0 = vec![0u8; 32];
        let mut b1 = vec![0u8; 40];
        let mut reqs = vec![
            ReadReq { fd, offset: 10, buf: &mut b0, tag: 0 },
            ReadReq { fd, offset: 100, buf: &mut b1, tag: 1 },
        ];
        let res = reactor.read_many(&mut reqs).unwrap();
        assert_eq!(res, vec![32, 40]);
        assert_eq!(&b0[..], &data[10..42]);
        assert_eq!(&b1[..], &data[100..140]);
        let _ = std::fs::remove_file(&path);
    }
}

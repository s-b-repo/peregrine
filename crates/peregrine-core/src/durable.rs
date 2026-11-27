//! Crash-durable file replacement.
//!
//! Every artifact this engine persists — `route_stats.json`, `automaton.json`,
//! `macrostates.json`, `schedule.json`, `tiers.json`, `plan.json`, and the
//! rewritten checkpoint — replaces a file that a later run reads back. A plain
//! `fs::write` opens with `O_TRUNC` and then streams, so a crash, `SIGKILL`, or
//! `ENOSPC` mid-write leaves a truncated file. The reader cannot tell a torn
//! file from an absent one, so the session's accumulated state is silently
//! discarded (or, for the checkpoint rewrite, the only copy of the weights is
//! lost).
//!
//! [`write_atomic`] gives the all-or-nothing swap those artifacts need: write a
//! sibling temp file, `fsync` it, `rename` it over the target (atomic within a
//! filesystem), then `fsync` the directory so the rename itself is durable.

use crate::{Context, Error};
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Replace `path` with `bytes`, atomically. Either the old contents or the new
/// ones survive a crash — never a partial write.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = AtomicFile::new(path)?;
    peregrine_io::ring::write_open_file(tmp.file(), bytes)
        .ctx(|| format!("write {}", path.display()))?;
    tmp.commit()
}

/// An exclusively created sibling file. Concurrent writers own independent
/// inodes, and readers see a complete file once its writer renames it.
pub struct AtomicFile {
    target: PathBuf,
    temp: PathBuf,
    file: File,
    committed: bool,
}

impl AtomicFile {
    pub fn new(target: &Path) -> Result<Self, Error> {
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        let prefix = temp_sibling(target)?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        loop {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let mut name = prefix.as_os_str().to_os_string();
            name.push(format!(".{}.{serial}", std::process::id()));
            let temp = PathBuf::from(name);
            match options.open(&temp) {
                Ok(file) => return Ok(Self { target: target.to_path_buf(), temp, file, committed: false }),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error).ctx(|| format!("create {}", temp.display())),
            }
        }
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    /// Replace the destination after the caller has synchronized the contents.
    pub fn commit(mut self) -> Result<(), Error> {
        finish_swap(&self.temp, &self.target)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if !self.committed {
            if let Err(error) = std::fs::remove_file(&self.temp) {
                if error.kind() != ErrorKind::NotFound {
                    crate::note_advisory_err("remove incomplete atomic file", &error);
                }
            }
        }
    }
}

/// Legacy fixed sibling name for callers that manage their own staging file.
pub fn temp_sibling(path: &Path) -> Result<std::path::PathBuf, Error> {
    let name = path
        .file_name()
        .ok_or_else(|| Error::Format(format!("{} has no file name", path.display())))?;
    let mut tmp = name.to_os_string();
    tmp.push(".tmp");
    Ok(path.with_file_name(tmp))
}

/// Commit a fully-written staging file over `path`. The
/// caller is responsible for `sync_all`-ing the temp file's contents first;
/// this fsyncs the directory so the rename survives a crash.
pub fn commit_atomic(tmp: &Path, path: &Path) -> Result<(), Error> {
    finish_swap(tmp, path)
}

fn finish_swap(tmp: &Path, path: &Path) -> Result<(), Error> {
    match std::fs::rename(tmp, path) {
        Ok(()) => {}
        Err(e) => {
            // Leaving the temp file behind would shadow the next attempt's
            // create; drop it, but report the rename failure, not the cleanup.
            if let Err(rm) = std::fs::remove_file(tmp) {
                if rm.kind() != std::io::ErrorKind::NotFound {
                    crate::note_advisory_err("remove temp file after failed rename", &rm);
                }
            }
            return Err(e).ctx(|| format!("rename {} -> {}", tmp.display(), path.display()));
        }
    }
    sync_parent_dir(path);
    Ok(())
}

/// `fsync` the directory holding `path` so the rename is durable. Advisory:
/// some filesystems reject directory fsync, and the rename itself has already
/// succeeded, so a failure here costs durability of the *link*, not the data.
fn sync_parent_dir(path: &Path) {
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        // A bare file name means the current directory.
        _ => Path::new("."),
    };
    match std::fs::File::open(dir) {
        Ok(f) => {
            use std::os::unix::io::AsRawFd;
            let synced = match peregrine_io::Reactor::new(1) {
                Ok(mut r) => r.fsync(f.as_raw_fd()),
                Err(e) => {
                    crate::note_advisory_err("io_uring unavailable for directory fsync (using fsync)", &e);
                    f.sync_all()
                }
            };
            if let Err(e) = synced {
                crate::note_advisory_err("fsync parent directory after rename", &e);
            }
        }
        Err(e) => crate::note_advisory_err("open parent directory for fsync", &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The library no longer writes through `std::io::Write` (its writes go
    // through the ring); the tests still build fixtures with it.
    use std::io::Write;

    fn tmpdir(tag: &str) -> Result<std::path::PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_durable_{}_{}", std::process::id(), tag));
        if let Err(e) = std::fs::remove_dir_all(&d) {
            if e.kind() != std::io::ErrorKind::NotFound {
                return Err(e.into());
            }
        }
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    #[test]
    fn replaces_contents_and_leaves_no_temp() -> Result<(), Error> {
        let d = tmpdir("replace")?;
        let p = d.join("state.json");
        write_atomic(&p, b"{\"v\":1}")?;
        assert_eq!(std::fs::read(&p)?, b"{\"v\":1}");
        // Overwriting an existing file keeps the new contents whole...
        write_atomic(&p, b"{\"v\":2,\"more\":true}")?;
        assert_eq!(std::fs::read(&p)?, b"{\"v\":2,\"more\":true}");
        // ...and never leaves the temp file behind for the next reader to find.
        assert!(!temp_sibling(&p)?.exists(), "temp file must be renamed away");
        let entries: Vec<_> = std::fs::read_dir(&d)?.filter_map(|e| e.ok()).collect();
        assert_eq!(entries.len(), 1, "only the target file remains");
        std::fs::remove_dir_all(&d)?;
        Ok(())
    }

    #[test]
    fn old_contents_survive_a_failed_write() -> Result<(), Error> {
        // The pre-swap failure mode: the temp write fails, so the original file
        // must still hold its previous (complete) contents.
        let d = tmpdir("failed")?;
        let p = d.join("state.json");
        write_atomic(&p, b"original")?;
        let tmp = AtomicFile::new(&p)?;
        tmp.file().write_all(b"incomplete replacement")?;
        drop(tmp);
        assert_eq!(std::fs::read(&p)?, b"original", "the old file is intact");
        assert_eq!(std::fs::read_dir(&d)?.count(), 1);
        std::fs::remove_dir_all(&d)?;
        Ok(())
    }

    #[test]
    fn overlapping_writers_commit_independent_files() -> Result<(), Error> {
        let dir = tmpdir("writers")?;
        let target = dir.join("state.json");
        let first = AtomicFile::new(&target)?;
        let second = AtomicFile::new(&target)?;
        peregrine_io::ring::write_open_file(first.file(), b"first value")?;
        peregrine_io::ring::write_open_file(second.file(), b"second")?;
        first.commit()?;
        assert_eq!(std::fs::read(&target)?, b"first value");
        second.commit()?;
        assert_eq!(std::fs::read(&target)?, b"second");
        assert_eq!(std::fs::read_dir(&dir)?.count(), 1);
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn streaming_commit_swaps_in_place() -> Result<(), Error> {
        let d = tmpdir("stream")?;
        let p = d.join("model.bin");
        std::fs::write(&p, b"old-weights")?;
        let tmp = temp_sibling(&p)?;
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(b"new-weights")?;
            f.sync_all()?;
        }
        commit_atomic(&tmp, &p)?;
        assert_eq!(std::fs::read(&p)?, b"new-weights");
        assert!(!tmp.exists());
        std::fs::remove_dir_all(&d)?;
        Ok(())
    }
}

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
use std::io::Write;
use std::path::Path;

/// Replace `path` with `bytes`, atomically. Either the old contents or the new
/// ones survive a crash — never a partial write.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = temp_sibling(path)?;
    // Scoped so the file is closed (after fsync) before the rename.
    {
        let mut f = std::fs::File::create(&tmp).ctx(|| format!("create {}", tmp.display()))?;
        f.write_all(bytes).ctx(|| format!("write {}", tmp.display()))?;
        f.sync_all().ctx(|| format!("fsync {}", tmp.display()))?;
    }
    finish_swap(&tmp, path)
}

/// The temp path [`write_atomic`] would use for `path`, for callers that must
/// stream their payload instead of holding it in memory (the checkpoint
/// rewrite). Pair it with [`commit_atomic`].
pub fn temp_sibling(path: &Path) -> Result<std::path::PathBuf, Error> {
    let name = path
        .file_name()
        .ok_or_else(|| Error::Format(format!("{} has no file name", path.display())))?;
    let mut tmp = name.to_os_string();
    tmp.push(".tmp");
    Ok(path.with_file_name(tmp))
}

/// Commit a fully-written temp file (from [`temp_sibling`]) over `path`. The
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
            if let Err(e) = f.sync_all() {
                crate::note_advisory_err("fsync parent directory after rename", &e);
            }
        }
        Err(e) => crate::note_advisory_err("open parent directory for fsync", &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // A directory at the temp path makes `File::create` fail.
        let tmp = temp_sibling(&p)?;
        std::fs::create_dir(&tmp)?;
        assert!(write_atomic(&p, b"replacement").is_err());
        assert_eq!(std::fs::read(&p)?, b"original", "the old file is intact");
        std::fs::remove_dir_all(&d)?;
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

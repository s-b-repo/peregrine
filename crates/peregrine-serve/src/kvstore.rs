//! Disk-persisted KV sessions (`COLI_KV_STORE_DIR`): the prefix cache extended
//! across process restarts — a port of ds4/DwarfStar's `ds4_kvstore` design.
//!
//! Why it pays on this engine in particular: a restored prefix of `n` tokens
//! skips `n` positions of prefill, and every prefill position streams its
//! routed-expert union from disk — ~10.85 GB per token measured. One stored
//! row is ~176 KiB (f32 at GLM-5.2 shapes), so the trade is roughly 60 000
//! bytes of expert reads saved per byte of checkpoint read back.
//!
//! Safety over cleverness, in the order the failure would bite:
//! - **Identity.** Every file records a fingerprint of the model container
//!   (config bytes + each shard's header page, which carries the requantize
//!   provenance stamp). A checkpoint from a different container — another
//!   quantization scheme, another model — is never even indexed.
//! - **Integrity.** An FNV-1a-64 trailer over the whole file; a flipped byte
//!   skips the file with an advisory, and the request cold-prefills instead.
//! - **Match by tokens, never by hash.** The full token ids live in the file
//!   and are compared against the prompt before use, preserving the prefix
//!   cache's documented no-collision invariant (`batch.rs`). The hash in the
//!   *filename* is only a dedup key.
//! - **Dtype.** The engine's `COLI_KV_DTYPE` must equal the file's; a mismatch
//!   is a skip, not a conversion — f16 rows restored into an f32 engine would
//!   build a cache no cold prefill of that engine could produce.
//!
//! Writes are asynchronous as of 2026-08-15 — the follow-up the previous
//! paragraph here named ("a background writer … if that ever shows") is now
//! the implementation. The engine thread pays only the prefix copy
//! (`SeqKv::export_prefix`, a memcpy that must borrow the live KV); the
//! serialize + hash + fsync — the 0.3–2 s a 2k-token/~350 MB entry costs —
//! happens on a dedicated writer thread behind a **depth-1 queue**. A writer
//! still busy with the previous checkpoint costs the new one (`dropped_busy`),
//! never a decode stall: a checkpoint is an optimization, and the one thing it
//! must never do is hold up the batch it exists to speed up. Files still
//! commit temp + atomic rename, so a crash mid-write leaves no torn entry.

use parking_lot::Mutex;
use peregrine_core::config::Cfg;
use peregrine_core::{durable, note_advisory_err, Context, Error};
use peregrine_model::{KvDtype, KvExport, KvLayerExport, Model, SeqKv};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAGIC: [u8; 4] = *b"PGKV";
const VERSION: u32 = 1;

/// Prompts shorter than this never touch the disk store. Deliberately higher
/// than the in-memory `PREFIX_CACHE_MIN_TOKENS`: a disk entry costs a write, a
/// directory scan on every boot, and cap budget, so it must save a prefill
/// long enough to notice ("after long prefill" in ds4's trigger list).
const KV_STORE_MIN_TOKENS: usize = 256;

/// FNV-1a-64 — inline because `peregrine-serve` adds no dependency for eight
/// lines, and because this is a corruption tripwire and a dedup key, not
/// cryptography: nothing here defends against an adversary editing files.
fn fnv1a64(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// ds4's save canonicalization: trim a short tail (their BPE-boundary hedge —
/// the last few tokens of a stopped stream re-tokenize differently more often
/// than the body), then align down so re-saves of a growing session land on
/// shared boundaries and dedup instead of accumulating off-by-a-few variants.
fn save_len(n: usize, trim: usize, align: usize) -> usize {
    let n = n.saturating_sub(trim);
    if align > 1 {
        n - n % align
    } else {
        n
    }
}

/// Longest shared prefix of two token streams.
fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Identity of a model container, from its artifacts alone: `config.json`
/// bytes, then each `*.safetensors` shard's name, size, and first 4 KiB — the
/// header page, which carries tensor shapes/offsets *and* the
/// `peregrine.requantize.scheme` provenance stamp. Two containers that differ
/// in scheme, layout, or shard set therefore fingerprint differently without
/// reading any payload. (Two containers with byte-identical headers but
/// different payloads — e.g. the same scheme converted twice from different
/// sources — would collide; the header stamp records the source dir precisely
/// to keep that theoretical.)
fn container_fingerprint(dir: &Path) -> Result<u64, Error> {
    let cfg_path = dir.join("config.json");
    let cfg_bytes = peregrine_io::read_file(&cfg_path).ctx(|| format!("read {}", cfg_path.display()))?;
    let mut h = fnv1a64(FNV_OFFSET, &cfg_bytes);
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .ctx(|| format!("scan {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    for shard in shards {
        if let Some(name) = shard.file_name().and_then(|n| n.to_str()) {
            h = fnv1a64(h, name.as_bytes());
        }
        let meta = std::fs::metadata(&shard).ctx(|| format!("stat {}", shard.display()))?;
        h = fnv1a64(h, &meta.len().to_le_bytes());
        // A shard is multi-gigabyte, so this reads the header page only —
        // through the ring, and short for a shard smaller than a page.
        let head = peregrine_io::read_region(&shard, 0, 4096)
            .ctx(|| format!("read header of {}", shard.display()))?;
        h = fnv1a64(h, &head);
    }
    Ok(h)
}

fn dtype_tag(dt: KvDtype) -> u32 {
    match dt {
        KvDtype::F32 => 0,
        KvDtype::F16 => 1,
    }
}

fn dtype_from_tag(t: u32) -> Option<KvDtype> {
    match t {
        0 => Some(KvDtype::F32),
        1 => Some(KvDtype::F16),
        _ => None,
    }
}

/// A `Write` shim that folds everything written through it into a running
/// FNV-1a-64, so a checkpoint streams to disk without ever assembling the
/// ~350 MB payload in memory just to hash it.
struct HashingWriter<W: Write> {
    w: W,
    h: u64,
}

/// A `std::io::Write` sink whose bytes reach the file through io_uring.
///
/// Two things `Write` does not give us and this has to carry itself. Positioned
/// writes need an explicit offset, so the sink owns its cursor. And a checkpoint
/// is emitted as a long run of small `write_all`s (one per header field, one per
/// payload element), which is why the old code wrapped a `BufWriter` — so the
/// sink keeps that coalescing and issues one `Reactor::write_all` per full
/// chunk.
///
/// Bounded staging rather than one whole-frame write is deliberate: a
/// 2k-token checkpoint is ~350 MB, and materializing it to submit a single
/// write would double the writer thread's peak RSS on a box that already needs
/// an RSS guard.
struct RingSink {
    reactor: peregrine_io::Reactor,
    fd: std::os::unix::io::RawFd,
    /// Next file offset to write at — `Write` carries no position of its own.
    off: u64,
    buf: Vec<u8>,
}

impl RingSink {
    /// Staging size. Large enough that the per-op cost disappears against the
    /// payload, small enough to be invisible next to the checkpoint itself.
    const CHUNK: usize = 8 << 20;

    fn new(reactor: peregrine_io::Reactor, fd: std::os::unix::io::RawFd) -> RingSink {
        RingSink { reactor, fd, off: 0, buf: Vec::with_capacity(Self::CHUNK) }
    }

    fn drain(&mut self) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.reactor.write_all(self.fd, self.off, &self.buf)?;
        self.off += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }

    /// Flush anything staged, then force it to the device. Ordering is safe only
    /// because `write_all` reaps every write completion before this submits the
    /// sync — io_uring does not order a write against a following fsync.
    fn finish(&mut self) -> std::io::Result<()> {
        self.drain()?;
        self.reactor.fsync(self.fd)
    }
}

impl Write for RingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(buf);
        if self.buf.len() >= Self::CHUNK {
            self.drain()?;
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.drain()
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.w.write(buf)?;
        self.h = fnv1a64(self.h, &buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// One indexed checkpoint. Tokens are held in memory (a few KiB per entry) so
/// a lookup is a pure compare; the payload stays on disk until a hit.
struct IndexEntry {
    path: PathBuf,
    tokens: Vec<i32>,
    bytes: u64,
}

/// The store: a directory of `.pgkv` checkpoints plus an in-memory token
/// index, LRU-ordered (scan order = mtime at boot, touched entries move to the
/// back, eviction pops the front). The index is shared with the writer thread
/// — a checkpoint appears in it only once its file has committed — and every
/// lock hold on it is index bookkeeping, never file I/O.
pub struct KvSessionStore {
    dir: PathBuf,
    trim: usize,
    align: usize,
    fingerprint: u64,
    dtype: KvDtype,
    cfg: Cfg,
    entries: Arc<Mutex<Vec<IndexEntry>>>,
    /// The write path's inputs, kept for the synchronous mode; the writer
    /// thread owns a clone sharing the same `entries`.
    ctx: WriterCtx,
    /// `COLI_KV_STORE_SYNC=1`: serialize + fsync on the calling (engine) thread
    /// — the pre-2026-08-15 behaviour. Exists as the A/B control arm for the
    /// async writer and as an operator fallback; resolved once at open, never
    /// through a process-global.
    sync_writes: bool,
    /// Depth-1 channel to the writer thread; `None` if the spawn failed at open
    /// (checkpoints disabled by advisory, loads unaffected) or in sync mode.
    writer_tx: Option<std::sync::mpsc::SyncSender<WriterMsg>>,
    writer_join: Option<std::thread::JoinHandle<()>>,
    /// Checkpoints accepted for write. The write itself is asynchronous, so a
    /// downstream failure is an advisory, not a decrement — this counts what
    /// serving decided to persist.
    pub saved: u64,
    pub loaded: u64,
    pub tokens_restored: u64,
    /// Checkpoints declined because the writer was still busy with the previous
    /// one — each was an optimization forgone, never an error.
    pub dropped_busy: u64,
}

/// Work for the writer thread.
enum WriterMsg {
    /// Serialize + fsync one checkpoint, then index it.
    Write { tokens: Vec<i32>, export: KvExport },
    /// Barrier: reply once every earlier write has committed or failed. Tests
    /// and orderly shutdown use it; serving never does.
    Sync(std::sync::mpsc::Sender<()>),
}

/// Everything `write_entry` and the index bookkeeping need, snapshotted at
/// open so the writer owns its inputs outright. `Clone` because the store keeps
/// one copy for the synchronous path (`COLI_KV_STORE_SYNC`) and hands one to
/// the writer thread — the `entries` Arc makes both views the same index.
#[derive(Clone)]
struct WriterCtx {
    dir: PathBuf,
    fingerprint: u64,
    dtype: KvDtype,
    cfg: Cfg,
    cap_bytes: u64,
    entries: Arc<Mutex<Vec<IndexEntry>>>,
}

impl KvSessionStore {
    /// Build from the environment: `None` when `COLI_KV_STORE_DIR` is unset —
    /// the historical no-disk behavior — or when the store cannot open (an
    /// advisory, never a fatal: a broken store must not take serving down).
    /// `COLI_KV_STORE_MB` caps the directory (default 16384); ds4's suffix
    /// trim is `COLI_KV_STORE_TRIM` (default 32 tokens).
    pub fn from_env(model: &Model, align: usize) -> Option<KvSessionStore> {
        let dir = std::env::var("COLI_KV_STORE_DIR").ok()?;
        let dir = PathBuf::from(dir.trim());
        if dir.as_os_str().is_empty() {
            return None;
        }
        let cap_mb = std::env::var("COLI_KV_STORE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(16384);
        let trim = std::env::var("COLI_KV_STORE_TRIM")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(32);
        let sync = matches!(std::env::var("COLI_KV_STORE_SYNC").ok().as_deref(), Some("1") | Some("true"));
        match KvSessionStore::open(&dir, model, cap_mb.saturating_mul(1024 * 1024), trim, align, sync) {
            Ok(s) => Some(s),
            Err(e) => {
                note_advisory_err("kvstore open (disk KV disabled)", &e);
                None
            }
        }
    }

    fn open(dir: &Path, model: &Model, cap_bytes: u64, trim: usize, align: usize, sync: bool) -> Result<KvSessionStore, Error> {
        std::fs::create_dir_all(dir).ctx(|| format!("create {}", dir.display()))?;
        let ctx = WriterCtx {
            dir: dir.to_path_buf(),
            fingerprint: container_fingerprint(model.checkpoint_dir())?,
            dtype: peregrine_model::kv_dtype(),
            cfg: model.cfg.clone(),
            cap_bytes,
            entries: Arc::new(Mutex::new(Vec::new())),
        };
        let mut s = KvSessionStore {
            dir: dir.to_path_buf(),
            trim,
            align: align.max(1),
            fingerprint: ctx.fingerprint,
            dtype: ctx.dtype,
            cfg: model.cfg.clone(),
            entries: Arc::clone(&ctx.entries),
            ctx,
            sync_writes: sync,
            writer_tx: None,
            writer_join: None,
            saved: 0,
            loaded: 0,
            tokens_restored: 0,
            dropped_busy: 0,
        };
        s.scan()?;
        if !sync {
            let ctx = s.ctx.clone();
            let (tx, rx) = std::sync::mpsc::sync_channel::<WriterMsg>(1);
            match std::thread::Builder::new().name("peregrine-kvstore-writer".into()).spawn(move || writer_loop(ctx, rx))
            {
                Ok(j) => {
                    s.writer_tx = Some(tx);
                    s.writer_join = Some(j);
                }
                // No writer means no new checkpoints; restores still work, and a
                // store that cannot spawn a thread has bigger problems than this.
                Err(e) => note_advisory_err("kvstore writer spawn (checkpoints disabled)", &e),
            }
        }
        Ok(s)
    }

    /// Index every readable checkpoint for this container, oldest first.
    /// Wrong-fingerprint or wrong-dtype files are skipped without noise — they
    /// are valid checkpoints for some other configuration sharing the dir.
    fn scan(&mut self) -> Result<(), Error> {
        let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&self.dir)
            .ctx(|| format!("scan {}", self.dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "pgkv"))
            .filter_map(|p| std::fs::metadata(&p).and_then(|m| m.modified()).ok().map(|t| (t, p)))
            .collect();
        found.sort();
        for (_, path) in found {
            match self.read_header(&path) {
                Ok(Some((tokens, bytes))) => self.entries.lock().push(IndexEntry { path, tokens, bytes }),
                Ok(None) => {} // another container's checkpoint; leave it be
                Err(e) => note_advisory_err("kvstore index (file skipped)", &e),
            }
        }
        Ok(())
    }

    /// Parse just the fixed header and token list; `Ok(None)` = a well-formed
    /// file that belongs to a different container or dtype.
    fn read_header(&self, path: &Path) -> Result<Option<(Vec<i32>, u64)>, Error> {
        let bytes = std::fs::metadata(path).ctx(|| format!("stat {}", path.display()))?.len();
        // Two positioned reads rather than one batched submit: the token list's
        // length is parsed out of the fixed header, so the second read cannot be
        // sized until the first has landed.
        const FIXED: usize = 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4;
        let fixed = peregrine_io::read_region(path, 0, FIXED)
            .ctx(|| format!("read header of {}", path.display()))?;
        if fixed.len() < FIXED {
            return Err(Error::Format(format!("{}: truncated header", path.display())));
        }
        let (magic, rest) = fixed.split_at(4);
        if magic != MAGIC {
            return Err(Error::Format(format!("{}: not a PGKV file", path.display())));
        }
        let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let version = u32_at(rest, 0);
        if version != VERSION {
            return Err(Error::Format(format!("{}: PGKV version {version}, this build reads {VERSION}", path.display())));
        }
        let fp = u64::from_le_bytes([rest[4], rest[5], rest[6], rest[7], rest[8], rest[9], rest[10], rest[11]]);
        let dt = u32_at(rest, 12);
        if fp != self.fingerprint || dtype_from_tag(dt) != Some(self.dtype) {
            return Ok(None);
        }
        let n_layers = u32_at(rest, 16) as usize;
        let n_tokens = u32_at(rest, 28) as usize;
        if n_layers != self.cfg.n_layers as usize {
            return Ok(None);
        }
        let want = n_tokens.saturating_mul(4);
        let tok_bytes = peregrine_io::read_region(path, FIXED as u64, want)
            .ctx(|| format!("read tokens of {}", path.display()))?;
        if tok_bytes.len() < want {
            return Err(Error::Format(format!("{}: truncated token list", path.display())));
        }
        let tokens = tok_bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Some((tokens, bytes)))
    }

    /// Longest match the index *could* serve for `prompt`, without any file
    /// I/O — the cheap probe `PrefixStore` uses to decide whether reading a
    /// checkpoint can beat what memory already has.
    pub fn best_match_len(&self, prompt: &[i32]) -> usize {
        let cap = prompt.len().saturating_sub(1);
        self.entries.lock().iter().map(|e| common_prefix(&e.tokens, prompt).min(cap)).max().unwrap_or(0)
    }

    /// Restore the longest stored prefix of `prompt`, capped (like the
    /// in-memory cache) at `prompt.len() - 1` so prefill still runs for at
    /// least one position. Every failure is an advisory and a `None` — the
    /// request then cold-prefills, which is always correct.
    pub fn load_longest(&mut self, prompt: &[i32]) -> Option<(SeqKv, usize)> {
        let cap = prompt.len().checked_sub(1)?;
        // Pick the candidate under the lock, read its file outside it: the writer
        // thread indexes and evicts under the same lock, and a multi-hundred-MB
        // file read must not hold it. The index can shift while the file is read,
        // so the touch/remove below re-finds by path and shrugs if it's gone.
        let (path, n) = {
            let entries = self.entries.lock();
            let (idx, n) = entries
                .iter()
                .enumerate()
                .map(|(i, e)| (i, common_prefix(&e.tokens, prompt).min(cap)))
                .max_by_key(|&(_, n)| n)?;
            (entries[idx].path.clone(), n)
        };
        if n < self.align.max(2) {
            return None;
        }
        match self.read_entry(&path, prompt, cap) {
            Ok(hit) => {
                if hit.is_some() {
                    // LRU touch: survivors of eviction are the ones being used.
                    let mut entries = self.entries.lock();
                    if let Some(idx) = entries.iter().position(|e| e.path == path) {
                        let e = entries.remove(idx);
                        entries.push(e);
                    }
                    self.loaded += 1;
                    if let Some((_, n)) = hit {
                        self.tokens_restored += n as u64;
                    }
                }
                hit
            }
            Err(e) => {
                note_advisory_err("kvstore load (cold prefill instead)", &e);
                // Whatever is wrong with the file will be wrong next time too.
                let mut entries = self.entries.lock();
                if let Some(idx) = entries.iter().position(|e| e.path == path) {
                    entries.remove(idx);
                }
                None
            }
        }
    }

    /// Full read of one checkpoint: checksum the whole file, re-verify the
    /// prompt match against the *file's* tokens (the index is a cache of
    /// them), rebuild the KV, and narrow it to the matched prefix.
    fn read_entry(&self, path: &Path, prompt: &[i32], cap: usize) -> Result<Option<(SeqKv, usize)>, Error> {
        let bytes = peregrine_io::read_file(path).ctx(|| format!("read {}", path.display()))?;
        if bytes.len() < 8 {
            return Err(Error::Format(format!("{}: truncated", path.display())));
        }
        let (body, trailer) = bytes.split_at(bytes.len() - 8);
        let want = u64::from_le_bytes([
            trailer[0], trailer[1], trailer[2], trailer[3], trailer[4], trailer[5], trailer[6], trailer[7],
        ]);
        if fnv1a64(FNV_OFFSET, body) != want {
            return Err(Error::Format(format!("{}: checksum mismatch (corrupt or torn)", path.display())));
        }
        let mut r = Reader { b: body, at: 0, path };
        let magic = r.take(4)?;
        if magic != MAGIC {
            return Err(Error::Format(format!("{}: not a PGKV file", path.display())));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(Error::Format(format!("{}: PGKV version {version}", path.display())));
        }
        let fp = r.u64()?;
        let dt = dtype_from_tag(r.u32()?);
        let n_layers = r.u32()? as usize;
        let kv_lora = r.u32()? as usize;
        let qk_rope = r.u32()? as usize;
        let n_tokens = r.u32()? as usize;
        if fp != self.fingerprint || dt != Some(self.dtype) {
            return Err(Error::Format(format!("{}: container/dtype changed under the index", path.display())));
        }
        if n_layers != self.cfg.n_layers as usize
            || kv_lora != self.cfg.kv_lora as usize
            || qk_rope != self.cfg.qk_rope as usize
        {
            return Err(Error::Format(format!("{}: dims do not match this model", path.display())));
        }
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            tokens.push(r.u32()? as i32);
        }
        // The decisive match: full token compare against the file, never the
        // index and never a hash.
        let n = common_prefix(&tokens, prompt).min(cap);
        if n == 0 {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let ix_width = r.u32()? as usize;
            let lc_len = r.u64()? as usize;
            let rc_len = r.u64()? as usize;
            let ix_len = r.u64()? as usize;
            let lc = r.f32s(lc_len)?;
            let rc = r.f32s(rc_len)?;
            let ix = r.f32s(ix_len)?;
            layers.push(KvLayerExport { lc, rc, ix, ix_width });
        }
        let export = KvExport { n: n_tokens, layers };
        let mut kv = SeqKv::import(&self.cfg, self.dtype, &export)?;
        kv.truncate(n);
        Ok(Some((kv, n)))
    }

    /// Persist `tokens`' completed KV if it clears the floor and no
    /// equal-or-longer entry already covers it. Errors are advisories: a full
    /// disk must cost the checkpoint, never the request.
    ///
    /// Asynchronous: this pays only the prefix copy (`export_prefix` borrows the
    /// live KV, so it cannot move off this thread) and hands the serialize +
    /// hash + fsync to the writer. The entry appears in the index when its file
    /// commits, not when this returns.
    pub fn save(&mut self, tokens: &[i32], kv: &SeqKv) {
        let n = save_len(tokens.len().min(kv.len()), self.trim, self.align);
        if n < KV_STORE_MIN_TOKENS {
            return;
        }
        let tokens = &tokens[..n];
        if self.entries.lock().iter().any(|e| e.tokens.len() >= n && e.tokens.starts_with(tokens)) {
            return;
        }
        if self.sync_writes {
            // The historical path, kept as the A/B control arm and operator
            // fallback: everything on the calling thread, durable on return.
            let export = kv.export_prefix(n);
            match Self::write_entry(&self.ctx, tokens, &export) {
                Ok(entry) => {
                    self.saved += 1;
                    index_committed(&self.ctx, entry, tokens);
                }
                Err(e) => note_advisory_err("kvstore save", &e),
            }
            return;
        }
        let Some(tx) = self.writer_tx.clone() else {
            return; // spawn failed at open; the advisory already fired there
        };
        let export = kv.export_prefix(n);
        match tx.try_send(WriterMsg::Write { tokens: tokens.to_vec(), export }) {
            Ok(()) => self.saved += 1,
            // Depth-1 queue: a writer still mid-checkpoint costs this one. The
            // dedupe above means a dropped save of a still-growing session gets
            // another chance at its next retirement.
            Err(std::sync::mpsc::TrySendError::Full(_)) => self.dropped_busy += 1,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                note_advisory_err("kvstore save", &"writer thread exited");
            }
        }
    }

    /// Block until every accepted checkpoint has committed or failed. Serving
    /// never calls this — tests do, and so does anything that wants the index
    /// to reflect all prior [`Self::save`] calls before reading it.
    pub fn flush(&self) {
        let Some(tx) = self.writer_tx.as_ref() else { return };
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        if tx.send(WriterMsg::Sync(ack_tx)).is_ok() && ack_rx.recv().is_err() {
            note_advisory_err("kvstore flush", &"writer exited before acking");
        }
    }

    /// Emit the checkpoint frame and its FNV-1a-64 trailer. Generic over the
    /// sink so the ring path and the no-ring fallback produce byte-identical
    /// files — the on-disk format is a compatibility surface (a checkpoint
    /// written by one build must load in the other), so there is exactly one
    /// place that decides its bytes.
    fn serialize_entry<W: Write>(
        w: &mut HashingWriter<W>,
        ctx: &WriterCtx,
        tokens: &[i32],
        export: &KvExport,
    ) -> Result<(), Error> {
        let n = tokens.len();
        w.write_all(&MAGIC).ctx(|| "kvstore header".to_string())?;
        w.write_all(&VERSION.to_le_bytes()).ctx(|| "kvstore header".to_string())?;
        w.write_all(&ctx.fingerprint.to_le_bytes()).ctx(|| "kvstore header".to_string())?;
        w.write_all(&dtype_tag(ctx.dtype).to_le_bytes()).ctx(|| "kvstore header".to_string())?;
        for dim in [ctx.cfg.n_layers as u32, ctx.cfg.kv_lora as u32, ctx.cfg.qk_rope as u32, n as u32] {
            w.write_all(&dim.to_le_bytes()).ctx(|| "kvstore header".to_string())?;
        }
        for &t in tokens {
            w.write_all(&(t as u32).to_le_bytes()).ctx(|| "kvstore tokens".to_string())?;
        }
        for le in &export.layers {
            w.write_all(&(le.ix_width as u32).to_le_bytes()).ctx(|| "kvstore layer".to_string())?;
            for len in [le.lc.len() as u64, le.rc.len() as u64, le.ix.len() as u64] {
                w.write_all(&len.to_le_bytes()).ctx(|| "kvstore layer".to_string())?;
            }
            for stream in [&le.lc, &le.rc, &le.ix] {
                for v in stream.iter() {
                    w.write_all(&v.to_le_bytes()).ctx(|| "kvstore payload".to_string())?;
                }
            }
        }
        let trailer = w.h;
        w.write_all(&trailer.to_le_bytes()).ctx(|| "kvstore trailer".to_string())?;
        Ok(())
    }

    fn write_entry(ctx: &WriterCtx, tokens: &[i32], export: &KvExport) -> Result<IndexEntry, Error> {
        let n = tokens.len();
        let mut name_hash = FNV_OFFSET;
        for &t in tokens {
            name_hash = fnv1a64(name_hash, &t.to_le_bytes());
        }
        let name = format!("{name_hash:016x}-{n}.pgkv");
        let path = ctx.dir.join(name);
        let tmp = durable::temp_sibling(&path)?;
        {
            let f = std::fs::File::create(&tmp).ctx(|| format!("create {}", tmp.display()))?;
            match peregrine_io::Reactor::new(8) {
                Ok(reactor) => {
                    use std::os::unix::io::AsRawFd;
                    let mut w = HashingWriter { w: RingSink::new(reactor, f.as_raw_fd()), h: FNV_OFFSET };
                    Self::serialize_entry(&mut w, ctx, tokens, export)?;
                    w.w.finish().ctx(|| format!("fsync {}", tmp.display()))?;
                }
                // A checkpoint is an optimization, so a host without io_uring
                // must still be able to save one — the same rule the whole-file
                // helpers in `peregrine-io` follow. Both sinks emit the identical
                // frame; only the syscall shape differs.
                Err(e) => {
                    note_advisory_err("io_uring unavailable for checkpoint write (using buffered write)", &e);
                    let mut w = HashingWriter { w: std::io::BufWriter::new(f), h: FNV_OFFSET };
                    Self::serialize_entry(&mut w, ctx, tokens, export)?;
                    w.flush().ctx(|| "kvstore flush".to_string())?;
                    let f = w.w.into_inner().map_err(|e| Error::Format(format!("kvstore buffered write: {e}")))?;
                    f.sync_all().ctx(|| format!("fsync {}", tmp.display()))?;
                }
            }
        }
        durable::commit_atomic(&tmp, &path)?;
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Ok(IndexEntry { path, tokens: tokens.to_vec(), bytes })
    }

    /// Bytes currently indexed on disk.
    pub fn resident_bytes(&self) -> u64 {
        self.entries.lock().iter().map(|e| e.bytes).sum()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.lock().len()
    }
}

impl Drop for KvSessionStore {
    fn drop(&mut self) {
        // Closing the channel is the writer's shutdown signal; the join bounds
        // the wait to the one checkpoint possibly in flight. (A process killed
        // harder than a drop still tears nothing — files commit temp + rename.)
        self.writer_tx = None;
        if let Some(j) = self.writer_join.take() {
            if j.join().is_err() {
                note_advisory_err("kvstore writer join", &"writer thread panicked");
            }
        }
    }
}

/// The writer thread: serialize + fsync checkpoints off the engine thread, then
/// index them. One job at a time, FIFO; the depth-1 channel in front of it is
/// the whole backpressure story.
fn writer_loop(ctx: WriterCtx, rx: std::sync::mpsc::Receiver<WriterMsg>) {
    loop {
        // `recv`'s only error is `Disconnected` — the store dropped its sender,
        // which is this thread's normal shutdown signal (same shape as
        // `prefetch_worker` in peregrine-model, and spelled out so the strict
        // audit can see the error path is handled, not swallowed).
        let msg = match rx.recv() {
            Ok(msg) => msg,
            Err(std::sync::mpsc::RecvError) => break,
        };
        match msg {
            WriterMsg::Write { tokens, export } => match KvSessionStore::write_entry(&ctx, &tokens, &export) {
                Ok(entry) => index_committed(&ctx, entry, &tokens),
                Err(e) => note_advisory_err("kvstore save", &e),
            },
            WriterMsg::Sync(ack) => {
                if ack.send(()).is_err() {
                    note_advisory_err("kvstore flush ack", &"flush requester gone");
                }
            }
        }
    }
}

/// Post-commit index bookkeeping, writer-side: replace any same-path row (a
/// racing save of the identical prefix commits to the same filename), drop
/// entries the new one strictly covers, append, and evict LRU-first to the cap
/// — the same rules `save` applied when it was synchronous. Victim files are
/// unlinked *outside* the lock: the engine thread probes this index on its hot
/// path, and unlink latency is the disk's business, not the index's.
fn index_committed(ctx: &WriterCtx, entry: IndexEntry, tokens: &[i32]) {
    let n = tokens.len();
    let mut victims: Vec<PathBuf> = Vec::new();
    {
        let mut entries = ctx.entries.lock();
        entries.retain(|e| e.path != entry.path);
        let mut i = 0;
        while i < entries.len() {
            if entries[i].tokens.len() < n && tokens.starts_with(&entries[i].tokens) {
                victims.push(entries.remove(i).path);
            } else {
                i += 1;
            }
        }
        entries.push(entry);
        let mut total: u64 = entries.iter().map(|e| e.bytes).sum();
        while total > ctx.cap_bytes && entries.len() > 1 {
            let victim = entries.remove(0);
            total = total.saturating_sub(victim.bytes);
            victims.push(victim.path);
        }
    }
    for path in victims {
        if let Err(e) = std::fs::remove_file(&path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                note_advisory_err("kvstore evict", &e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dir(tag: &str) -> Result<PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_kvstore_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        peregrine_model::testkit::build_tiny_model(&d)?;
        Ok(d)
    }

    fn store_dir(tag: &str) -> Result<PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_kvdir_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        Ok(d)
    }

    /// A prompt long enough to clear the 256-token save floor after trim+align.
    fn long_prompt() -> Vec<i32> {
        (0..400).map(|k| (k * 3 + 1) % 32).collect()
    }

    #[test]
    fn save_len_trims_then_aligns() {
        // ds4's canonicalization: 400 - 32 = 368, aligned down to 64 → 320.
        assert_eq!(save_len(400, 32, 64), 320);
        assert_eq!(save_len(64, 32, 64), 0, "a short session rounds to nothing");
        assert_eq!(save_len(100, 0, 1), 100, "no trim, no align = identity");
        assert_eq!(save_len(10, 32, 64), 0, "trim never underflows");
    }

    #[test]
    fn a_restart_restores_the_prefix_and_decodes_identically() -> Result<(), Error> {
        // The whole point: save from one store instance, load from a *fresh*
        // one (the restart), and the restored cache must continue decoding
        // exactly where the live one would. Bit-identical logits are the
        // decisive check, same idiom as the prefix-cache seeding test.
        let mdir = tiny_dir("roundtrip")?;
        let sdir = store_dir("roundtrip")?;
        let model = peregrine_model::Model::load(&mdir)?;
        let prompt = long_prompt();
        let mut live = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&prompt, &mut live, 0)?;

        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        store.save(&prompt, &live);
        store.flush(); // the write is asynchronous; the index fills on commit
        assert_eq!(store.entry_count(), 1, "one checkpoint written");
        assert_eq!(store.saved, 1);

        // The restart: a fresh store over the same directory.
        let mut fresh = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        assert_eq!(fresh.entry_count(), 1, "the checkpoint is re-indexed after restart");
        let (mut restored, n) = fresh
            .load_longest(&prompt)
            .ok_or_else(|| Error::Format("expected a disk hit".into()))?;
        assert_eq!(n, 320, "trimmed and aligned length is what was stored");
        assert_eq!(restored.len(), n);
        assert_eq!(fresh.tokens_restored, n as u64);

        // Continue both caches over the identical remaining positions.
        let mut live_cut = live.clone_prefix(n);
        let want = model.forward_prefill_seq(&prompt[n..], &mut live_cut, n)?;
        let got = model.forward_prefill_seq(&prompt[n..], &mut restored, n)?;
        assert!(
            want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
            "decode from the restored KV must be bit-identical to the live path"
        );
        std::fs::remove_dir_all(&mdir)?;
        std::fs::remove_dir_all(&sdir)?;
        Ok(())
    }

    #[test]
    fn a_flipped_byte_is_skipped_not_served() -> Result<(), Error> {
        let mdir = tiny_dir("corrupt")?;
        let sdir = store_dir("corrupt")?;
        let model = peregrine_model::Model::load(&mdir)?;
        let prompt = long_prompt();
        let mut live = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&prompt, &mut live, 0)?;
        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        store.save(&prompt, &live);
        store.flush();

        // Flip one payload byte in the checkpoint on disk.
        let path = store.entries.lock()[0].path.clone();
        let mut bytes = std::fs::read(&path)?;
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x40;
        std::fs::write(&path, &bytes)?;

        let mut fresh = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        assert!(
            fresh.load_longest(&prompt).is_none(),
            "a corrupt checkpoint must be skipped so the request cold-prefills"
        );
        assert_eq!(fresh.loaded, 0);
        std::fs::remove_dir_all(&mdir)?;
        std::fs::remove_dir_all(&sdir)?;
        Ok(())
    }

    #[test]
    fn another_containers_checkpoint_is_never_indexed() -> Result<(), Error> {
        let mdir = tiny_dir("fpr")?;
        let sdir = store_dir("fpr")?;
        let model = peregrine_model::Model::load(&mdir)?;
        let prompt = long_prompt();
        let mut live = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&prompt, &mut live, 0)?;
        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        store.save(&prompt, &live);
        store.flush();

        // "Another container": same weights, but its config.json differs — the
        // fingerprint hashes config bytes, so this is a different identity, as
        // it must be (dims drive how the KV is rebuilt).
        let mut cfg_bytes = std::fs::read(mdir.join("config.json"))?;
        cfg_bytes.push(b' ');
        std::fs::write(mdir.join("config.json"), &cfg_bytes)?;
        let other = peregrine_model::Model::load(&mdir)?;
        let fresh = KvSessionStore::open(&sdir, &other, 1 << 30, 32, 64, false)?;
        assert_eq!(fresh.entry_count(), 0, "a foreign checkpoint must not be indexed");
        assert_eq!(fresh.best_match_len(&prompt), 0);
        std::fs::remove_dir_all(&mdir)?;
        std::fs::remove_dir_all(&sdir)?;
        Ok(())
    }

    #[test]
    fn the_size_cap_evicts_oldest_first_and_dedup_skips_covered_saves() -> Result<(), Error> {
        let mdir = tiny_dir("evict")?;
        let sdir = store_dir("evict")?;
        let model = peregrine_model::Model::load(&mdir)?;
        let a = long_prompt();
        let mut b = long_prompt();
        b[10] = 0; // a[10] is 31, so the two prompts diverge at position 10
        let mut kv_a = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&a, &mut kv_a, 0)?;
        let mut kv_b = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&b, &mut kv_b, 0)?;

        // Cap below two entries: saving the second evicts the first (LRU).
        let mut store = KvSessionStore::open(&sdir, &model, 1, 32, 64, false)?;
        store.save(&a, &kv_a);
        store.flush();
        assert_eq!(store.entry_count(), 1);
        store.save(&b, &kv_b);
        store.flush();
        assert_eq!(store.entry_count(), 1, "the cap holds one entry, oldest evicted");
        // The survivor is the newer entry: its stored (trimmed+aligned) 320
        // tokens all match `b`, while `a` would diverge at position 10.
        assert_eq!(store.best_match_len(&b), 320, "the survivor is the newer entry");
        assert!(store.best_match_len(&a) < 320, "the evicted entry's prompt no longer matches deeply");

        // A save already covered by an equal-or-longer entry writes nothing.
        let before = store.saved;
        store.save(&b[..320], &kv_b.clone_prefix(320));
        assert_eq!(store.saved, before, "a covered prefix must dedup, not rewrite");
        std::fs::remove_dir_all(&mdir)?;
        std::fs::remove_dir_all(&sdir)?;
        Ok(())
    }

    #[test]
    fn sync_mode_commits_on_the_calling_thread_and_interoperates() -> Result<(), Error> {
        // COLI_KV_STORE_SYNC: the historical write path, kept as the A/B
        // control arm. Durable and indexed on return, no writer thread, and a
        // checkpoint it writes is indistinguishable to an async-mode store.
        let mdir = tiny_dir("syncmode")?;
        let sdir = store_dir("syncmode")?;
        let model = peregrine_model::Model::load(&mdir)?;
        let prompt = long_prompt();
        let mut live = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&prompt, &mut live, 0)?;
        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, true)?;
        assert!(store.writer_join.is_none(), "sync mode must not spawn a writer");
        store.save(&prompt, &live);
        // Deliberately no flush: sync mode's contract is durable-on-return.
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.saved, 1);
        let mut fresh = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64, false)?;
        assert!(
            fresh.load_longest(&prompt).is_some(),
            "an async-mode store must read a sync-mode checkpoint (same format, same index rules)"
        );
        std::fs::remove_dir_all(&mdir)?;
        std::fs::remove_dir_all(&sdir)?;
        Ok(())
    }
}

/// Bounds-checked little-endian reader over a checksummed body.
struct Reader<'a> {
    b: &'a [u8],
    at: usize,
    path: &'a Path,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| {
            Error::Format(format!("{}: truncated at byte {}", self.path.display(), self.at))
        })?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, Error> {
        let b = self.take(n.checked_mul(4).ok_or_else(|| Error::Format("kvstore length overflow".into()))?)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

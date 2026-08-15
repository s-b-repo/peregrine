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
//! Writes are synchronous on the engine thread, temp + atomic rename like
//! every other artifact this project persists: a 2k-token entry is ~350 MB ≈
//! 0.3–2 s against a 16 s/token decode. A background writer over the
//! refcounted prefix snapshot is the named follow-up if that ever shows.

use peregrine_core::config::Cfg;
use peregrine_core::{durable, note_advisory_err, Context, Error};
use peregrine_model::{KvDtype, KvExport, KvLayerExport, Model, SeqKv};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

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
    let cfg_bytes = std::fs::read(&cfg_path).ctx(|| format!("read {}", cfg_path.display()))?;
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
        let mut head = vec![0u8; 4096];
        let mut f = std::fs::File::open(&shard).ctx(|| format!("open {}", shard.display()))?;
        let got = f.read(&mut head).ctx(|| format!("read header of {}", shard.display()))?;
        h = fnv1a64(h, &head[..got]);
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
/// back, eviction pops the front).
pub struct KvSessionStore {
    dir: PathBuf,
    cap_bytes: u64,
    trim: usize,
    align: usize,
    fingerprint: u64,
    dtype: KvDtype,
    cfg: Cfg,
    entries: Vec<IndexEntry>,
    pub saved: u64,
    pub loaded: u64,
    pub tokens_restored: u64,
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
        match KvSessionStore::open(&dir, model, cap_mb.saturating_mul(1024 * 1024), trim, align) {
            Ok(s) => Some(s),
            Err(e) => {
                note_advisory_err("kvstore open (disk KV disabled)", &e);
                None
            }
        }
    }

    fn open(dir: &Path, model: &Model, cap_bytes: u64, trim: usize, align: usize) -> Result<KvSessionStore, Error> {
        std::fs::create_dir_all(dir).ctx(|| format!("create {}", dir.display()))?;
        let mut s = KvSessionStore {
            dir: dir.to_path_buf(),
            cap_bytes,
            trim,
            align: align.max(1),
            fingerprint: container_fingerprint(model.checkpoint_dir())?,
            dtype: peregrine_model::kv_dtype(),
            cfg: model.cfg.clone(),
            entries: Vec::new(),
            saved: 0,
            loaded: 0,
            tokens_restored: 0,
        };
        s.scan()?;
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
                Ok(Some((tokens, bytes))) => self.entries.push(IndexEntry { path, tokens, bytes }),
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
        let mut f = std::fs::File::open(path).ctx(|| format!("open {}", path.display()))?;
        let mut fixed = [0u8; 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4];
        f.read_exact(&mut fixed).ctx(|| format!("read header of {}", path.display()))?;
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
        let mut tok_bytes = vec![0u8; n_tokens.saturating_mul(4)];
        f.read_exact(&mut tok_bytes).ctx(|| format!("read tokens of {}", path.display()))?;
        let tokens = tok_bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Some((tokens, bytes)))
    }

    /// Longest match the index *could* serve for `prompt`, without any file
    /// I/O — the cheap probe `PrefixStore` uses to decide whether reading a
    /// checkpoint can beat what memory already has.
    pub fn best_match_len(&self, prompt: &[i32]) -> usize {
        let cap = prompt.len().saturating_sub(1);
        self.entries.iter().map(|e| common_prefix(&e.tokens, prompt).min(cap)).max().unwrap_or(0)
    }

    /// Restore the longest stored prefix of `prompt`, capped (like the
    /// in-memory cache) at `prompt.len() - 1` so prefill still runs for at
    /// least one position. Every failure is an advisory and a `None` — the
    /// request then cold-prefills, which is always correct.
    pub fn load_longest(&mut self, prompt: &[i32]) -> Option<(SeqKv, usize)> {
        let cap = prompt.len().checked_sub(1)?;
        let (idx, n) = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt).min(cap)))
            .max_by_key(|&(_, n)| n)?;
        if n < self.align.max(2) {
            return None;
        }
        match self.read_entry(idx, prompt, cap) {
            Ok(hit) => {
                if hit.is_some() {
                    // LRU touch: survivors of eviction are the ones being used.
                    let e = self.entries.remove(idx);
                    self.entries.push(e);
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
                self.entries.remove(idx);
                None
            }
        }
    }

    /// Full read of one checkpoint: checksum the whole file, re-verify the
    /// prompt match against the *file's* tokens (the index is a cache of
    /// them), rebuild the KV, and narrow it to the matched prefix.
    fn read_entry(&self, idx: usize, prompt: &[i32], cap: usize) -> Result<Option<(SeqKv, usize)>, Error> {
        let entry = self
            .entries
            .get(idx)
            .ok_or_else(|| Error::Format("kvstore index out of range".into()))?;
        let path = &entry.path;
        let bytes = std::fs::read(path).ctx(|| format!("read {}", path.display()))?;
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
    pub fn save(&mut self, tokens: &[i32], kv: &SeqKv) {
        let n = save_len(tokens.len().min(kv.len()), self.trim, self.align);
        if n < KV_STORE_MIN_TOKENS {
            return;
        }
        let tokens = &tokens[..n];
        if self.entries.iter().any(|e| e.tokens.len() >= n && e.tokens.starts_with(tokens)) {
            return;
        }
        match self.write_entry(tokens, kv) {
            Ok(entry) => {
                self.saved += 1;
                // Entries this one strictly covers are redundant, same rule as
                // the in-memory cache — drop their files with them.
                let covered: Vec<usize> = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.tokens.len() < n && tokens.starts_with(&e.tokens))
                    .map(|(i, _)| i)
                    .collect();
                for i in covered.into_iter().rev() {
                    let old = self.entries.remove(i);
                    if let Err(e) = std::fs::remove_file(&old.path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            note_advisory_err("kvstore remove covered entry", &e);
                        }
                    }
                }
                self.entries.push(entry);
                self.evict_to_cap();
            }
            Err(e) => note_advisory_err("kvstore save", &e),
        }
    }

    fn write_entry(&self, tokens: &[i32], kv: &SeqKv) -> Result<IndexEntry, Error> {
        let n = tokens.len();
        let export = kv.export_prefix(n);
        let mut name_hash = FNV_OFFSET;
        for &t in tokens {
            name_hash = fnv1a64(name_hash, &t.to_le_bytes());
        }
        let name = format!("{name_hash:016x}-{n}.pgkv");
        let path = self.dir.join(name);
        let tmp = durable::temp_sibling(&path)?;
        {
            let f = std::fs::File::create(&tmp).ctx(|| format!("create {}", tmp.display()))?;
            let mut w = HashingWriter { w: std::io::BufWriter::new(f), h: FNV_OFFSET };
            w.write_all(&MAGIC).ctx(|| "kvstore header".to_string())?;
            w.write_all(&VERSION.to_le_bytes()).ctx(|| "kvstore header".to_string())?;
            w.write_all(&self.fingerprint.to_le_bytes()).ctx(|| "kvstore header".to_string())?;
            w.write_all(&dtype_tag(self.dtype).to_le_bytes()).ctx(|| "kvstore header".to_string())?;
            for dim in [self.cfg.n_layers as u32, self.cfg.kv_lora as u32, self.cfg.qk_rope as u32, n as u32] {
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
            w.flush().ctx(|| "kvstore flush".to_string())?;
            let f = w.w.into_inner().map_err(|e| Error::Format(format!("kvstore buffered write: {e}")))?;
            f.sync_all().ctx(|| format!("fsync {}", tmp.display()))?;
        }
        durable::commit_atomic(&tmp, &path)?;
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Ok(IndexEntry { path, tokens: tokens.to_vec(), bytes })
    }

    /// Drop least-recently-used checkpoints (index front) until the directory
    /// fits `COLI_KV_STORE_MB`.
    fn evict_to_cap(&mut self) {
        let mut total: u64 = self.entries.iter().map(|e| e.bytes).sum();
        while total > self.cap_bytes && self.entries.len() > 1 {
            let victim = self.entries.remove(0);
            total = total.saturating_sub(victim.bytes);
            if let Err(e) = std::fs::remove_file(&victim.path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    note_advisory_err("kvstore evict", &e);
                }
            }
        }
    }

    /// Bytes currently indexed on disk.
    pub fn resident_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
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

        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64)?;
        store.save(&prompt, &live);
        assert_eq!(store.entry_count(), 1, "one checkpoint written");
        assert_eq!(store.saved, 1);

        // The restart: a fresh store over the same directory.
        let mut fresh = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64)?;
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
        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64)?;
        store.save(&prompt, &live);

        // Flip one payload byte in the checkpoint on disk.
        let path = store.entries[0].path.clone();
        let mut bytes = std::fs::read(&path)?;
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x40;
        std::fs::write(&path, &bytes)?;

        let mut fresh = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64)?;
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
        let mut store = KvSessionStore::open(&sdir, &model, 1 << 30, 32, 64)?;
        store.save(&prompt, &live);

        // "Another container": same weights, but its config.json differs — the
        // fingerprint hashes config bytes, so this is a different identity, as
        // it must be (dims drive how the KV is rebuilt).
        let mut cfg_bytes = std::fs::read(mdir.join("config.json"))?;
        cfg_bytes.push(b' ');
        std::fs::write(mdir.join("config.json"), &cfg_bytes)?;
        let other = peregrine_model::Model::load(&mdir)?;
        let fresh = KvSessionStore::open(&sdir, &other, 1 << 30, 32, 64)?;
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
        let mut store = KvSessionStore::open(&sdir, &model, 1, 32, 64)?;
        store.save(&a, &kv_a);
        assert_eq!(store.entry_count(), 1);
        store.save(&b, &kv_b);
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

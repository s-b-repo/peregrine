//! Byte-budgeted RAM warm cache for streamed expert weights — the warm tier the
//! concurrent MoE scheduler consults before hitting the disk.
//!
//! Unlike [`crate::cache::ExpertCache`] (a slot-count LRU keyed by a single
//! expert id), this cache is bounded by **bytes** and keyed by `(layer, expert)`,
//! because an expert id is only unique within its layer and expert slabs vary in
//! size. It stores the raw streamed *quantized* bytes verbatim (weight + scale for
//! gate/up/down), so a hit reconstructs a **bit-identical** `QtWeight` — the cache
//! only changes load timing, never the numeric output. Holding quantized (not
//! dequantized) bytes also keeps the RAM footprint small (todo.txt "quantized RAM
//! cache").

use std::collections::HashMap;

use crate::Bytes;

/// One expert's streamed bytes: `(weight, scale)` for each of gate, up, down —
/// exactly what the I/O lane reads and hands to `rebuild`. Each region is a
/// [`Bytes`], so the O_DIRECT lane can hand over its aligned DMA buffer directly
/// (zero-copy) while the buffered lane and cache clones use a plain `Vec`.
pub type ExpertSlab = [(Bytes, Bytes); 3];

/// Resident bytes a slab occupies (all six weight/scale regions). Uses each
/// region's real footprint, not its exposed length: an O_DIRECT region holds a
/// block-aligned window that can be up to 8 KiB larger than the bytes it
/// exposes, and budgeting on the smaller number over-admits.
fn slab_bytes(s: &ExpertSlab) -> usize {
    s.iter().map(|(w, sc)| w.footprint() + sc.footprint()).sum()
}

/// Convert every region to refcounted shared bytes. Done once at admission so
/// each later hit is six refcount bumps instead of a full ~19 MB copy.
fn share_slab(s: ExpertSlab) -> ExpertSlab {
    s.map(|(w, sc)| (w.into_shared(), sc.into_shared()))
}

/// Compress each of a slab's six regions into a `SlotBytes::Compressed`
/// variant. Returns `None` on a codec failure (the caller then stores the raw
/// slab). The Compressed variant also carries `orig_lens` so decompression
/// can pre-size and validate.
fn encode_slab(s: &ExpertSlab) -> Option<(SlotBytes, usize)> {
    // 6 regions in gate/up/down × (weight, scale) order — the layout the reader
    // and the `pack_slab` builder both use, so the six-Vec array preserves it.
    let mut six: [Vec<u8>; 6] = Default::default();
    let mut orig: [usize; 6] = [0; 6];
    let mut idx = 0usize;
    for (w, sc) in s.iter() {
        let ew = zstd::stream::encode_all(&w[..], 3).ok()?;
        orig[idx] = w.len();
        six[idx] = ew;
        idx += 1;
        let es = zstd::stream::encode_all(&sc[..], 3).ok()?;
        orig[idx] = sc.len();
        six[idx] = es;
        idx += 1;
    }
    let compressed_bytes: usize = six.iter().map(|v| v.len()).sum();
    Some((SlotBytes::Compressed { six, orig_lens: orig }, compressed_bytes))
}

/// Materialize an `ExpertSlab` from a stored `SlotBytes`. For `Raw` this clones
/// six refcounts (the regions are shared). For `Compressed` it decodes each
/// region into a fresh `Vec` and assembles the three (weight, scale) pairs.
///
/// Returns `None` when any region fails to decode or comes back the wrong
/// length. That must NOT be papered over with empty bytes: the caller counts
/// the lookup as a hit and feeds the slab straight into `QtWeight`, which reads
/// it as an `[o, i]` weight — a zero-length region then indexes out of bounds
/// (an abort, under `panic = "abort"`) or silently contributes garbage. A miss
/// is the correct degradation: the expert is simply re-streamed from disk.
fn materialize(sb: &SlotBytes) -> Option<ExpertSlab> {
    match sb {
        SlotBytes::Raw(s) => Some(s.clone()),
        SlotBytes::Compressed { six, orig_lens } => {
            let decode = |i: usize| -> Option<Bytes> {
                match zstd::stream::decode_all(&six[i][..]) {
                    Ok(v) if v.len() == orig_lens[i] => Some(Bytes::from(v)),
                    _ => None,
                }
            };
            Some([
                (decode(0)?, decode(1)?),
                (decode(2)?, decode(3)?),
                (decode(4)?, decode(5)?),
            ])
        }
    }
}

/// A cached expert slab, held either verbatim or transparently zstd-compressed.
/// `Compressed` shrinks resident footprint at the cost of one zstd decode per
/// hit; the choice per slot is fixed at admission time by `COLI_CACHE_COMPRESS`.
/// Callers never observe the difference — [`WarmCache::get`] materializes a
/// byte-identical [`ExpertSlab`] either way.
enum SlotBytes {
    Raw(ExpertSlab),
    Compressed { six: [Vec<u8>; 6], orig_lens: [usize; 6] },
}

struct Slot {
    used: u64,
    bytes: usize,
    data: SlotBytes,
    /// This slot was populated by the prefetch lane (`insert_prefetched`), not a
    /// critical-path miss. Drives the `prefetch_used`/`prefetch_wasted` accounting.
    from_prefetch: bool,
    /// This slot has served at least one cache hit. Lets us count a prefetched slot
    /// as "used" exactly once, and as "wasted" only if evicted before any hit.
    ever_hit: bool,
    /// Eviction protection score set by the model's predictor (0 = unprotected).
    /// Victims are chosen by `(prio, used)`, so a higher `prio` is evicted only
    /// after every lower-priority slot. Priority is *orthogonal to recency* — it
    /// never bumps `used`/`clock`, so with all priorities equal the policy is
    /// byte-for-byte the original LRU.
    prio: u32,
}

/// Small Bloom filter over currently-resident `(layer, expert)` keys. Two hash
/// positions per key, 2048 bits (256 bytes). Optimizes the miss path — if the
/// bloom returns "definitely not present" we skip the HashMap probe. False
/// positives fall through to the HashMap so correctness is unchanged.
///
/// Rebuilt after each eviction batch (via [`Bloom::rebuild`]) so we don't have
/// to track per-slot removal (a counting Bloom would double the space).
struct Bloom {
    bits: [u64; 32],
}

impl Bloom {
    const N_BITS: u32 = 32 * 64; // 2048

    fn new() -> Bloom {
        Bloom { bits: [0; 32] }
    }

    fn hashes(key: (u32, u32)) -> (u32, u32) {
        // Two independent, cheap non-cryptographic hashes derived by FNV-like
        // mixing. Sufficient for a hint at this scale.
        let mut a = 0x811c9dc5u32;
        a ^= key.0;
        a = a.wrapping_mul(0x01000193);
        a ^= key.1;
        a = a.wrapping_mul(0x01000193);

        let mut b = 0xdeadbeefu32;
        b ^= key.1.rotate_left(13);
        b = b.wrapping_mul(0x9e3779b1);
        b ^= key.0.rotate_left(7);
        b = b.wrapping_mul(0x9e3779b1);

        (a % Self::N_BITS, b % Self::N_BITS)
    }

    fn add(&mut self, key: (u32, u32)) {
        let (h1, h2) = Self::hashes(key);
        self.bits[(h1 / 64) as usize] |= 1u64 << (h1 % 64);
        self.bits[(h2 / 64) as usize] |= 1u64 << (h2 % 64);
    }

    /// `false` ⇒ definitely absent; `true` ⇒ probably present (fall through to
    /// the exact map).
    fn probably_contains(&self, key: (u32, u32)) -> bool {
        let (h1, h2) = Self::hashes(key);
        let b1 = self.bits[(h1 / 64) as usize] & (1u64 << (h1 % 64)) != 0;
        let b2 = self.bits[(h2 / 64) as usize] & (1u64 << (h2 % 64)) != 0;
        b1 && b2
    }

    fn clear(&mut self) {
        self.bits = [0; 32];
    }
}

/// Bounded-by-bytes LRU cache of expert slabs, keyed by `(layer, expert)`.
pub struct WarmCache {
    budget: usize,
    used: usize,
    map: HashMap<(u32, u32), Slot>,
    clock: u64,
    /// Bloom hint over resident keys. Rebuilt on evictions; consulted in the
    /// miss-fast-path of `contains`/`get`.
    bloom: Bloom,
    /// Negative-cache TTL. Slots that haven't been hit in this many `clock`
    /// ticks become eligible for eager eviction ahead of pure LRU order. `0`
    /// disables (default). Set via `COLI_CACHE_NEGATIVE_TTL`.
    negative_ttl: u64,
    pub hits: u64,
    pub misses: u64,
    /// Slots dropped because their stored payload would not decode (corruption
    /// or a codec failure). The lookup degrades to a miss and re-streams.
    decode_failures: u64,
    /// expert reads on the **critical path** (a main-lane miss streamed them).
    pub disk_reads: u64,
    /// expert reads done **ahead of time** by the prefetch lane (off the critical
    /// path). Kept separate so a test can show prefetch moved I/O off the main lane
    /// rather than eliminating it.
    pub prefetch_reads: u64,
    /// disk reads attributed per layer (grows to fit the highest layer seen);
    /// lets the prefetch test isolate the effect on a specific layer.
    disk_reads_by_layer: Vec<u64>,
    /// prefetch-lane reads attributed per layer. With layer look-ahead, this shows
    /// that *early* layers are warmed (their prefetch was emitted mid-forward)
    /// rather than only at the end of the forward.
    prefetch_reads_by_layer: Vec<u64>,
    /// prefetched slabs that were later hit at least once (the prefetch paid off).
    /// Counted once per slab, on its first hit.
    pub prefetch_used: u64,
    /// prefetched slabs evicted before ever being hit (the prefetch was wasted).
    pub prefetch_wasted: u64,
    /// low-confidence experts the prefetch lane merely *hinted* to the page cache
    /// via `fadvise(WILLNEED)` (multi-path tier 2), rather than fully streaming.
    pub fadvise_hints: u64,
    /// speculative reads whose opt-in re-read verification found differing bytes —
    /// always 0 in a correct system; nonzero signals a real I/O bug.
    pub verify_mismatch: u64,
    /// Fast-miss decisions the Bloom filter made (for diagnostics).
    pub bloom_skips: u64,
    /// Whether new admissions compress their slabs (`COLI_CACHE_COMPRESS=1`).
    /// Fixed at construction; changing the env mid-process doesn't retroactively
    /// compress or decompress existing slots.
    compress_on_admit: bool,
    /// Cumulative resident bytes we would have held without compression —
    /// diagnostic to observe the compression win.
    pub uncompressed_bytes_seen: u64,
    /// Cumulative bytes actually admitted (post-compression when enabled).
    pub compressed_bytes_seen: u64,
}

impl WarmCache {
    /// A cache bounded to `budget_bytes` of resident expert slabs.
    pub fn new(budget_bytes: usize) -> WarmCache {
        let ttl: u64 = std::env::var("COLI_CACHE_NEGATIVE_TTL")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let compress = matches!(std::env::var("COLI_CACHE_COMPRESS").as_deref(), Ok("1") | Ok("true"));
        WarmCache {
            budget: budget_bytes,
            used: 0,
            map: HashMap::new(),
            clock: 0,
            bloom: Bloom::new(),
            negative_ttl: ttl,
            hits: 0,
            misses: 0,
            decode_failures: 0,
            disk_reads: 0,
            prefetch_reads: 0,
            disk_reads_by_layer: Vec::new(),
            prefetch_reads_by_layer: Vec::new(),
            prefetch_used: 0,
            prefetch_wasted: 0,
            fadvise_hints: 0,
            verify_mismatch: 0,
            bloom_skips: 0,
            compress_on_admit: compress,
            uncompressed_bytes_seen: 0,
            compressed_bytes_seen: 0,
        }
    }

    /// Chainable override of the compress-on-admit flag. Useful in tests where
    /// mutating `COLI_CACHE_COMPRESS` at runtime would race with parallel test
    /// threads sharing the process env.
    pub fn with_compression(mut self, on: bool) -> WarmCache {
        self.compress_on_admit = on;
        self
    }

    /// The compression ratio actually achieved across every admission so far
    /// (uncompressed ÷ admitted), or `None` when nothing has been admitted with
    /// compression enabled.
    ///
    /// Worth reporting rather than assuming: the payload is packed int4 nibbles,
    /// whose only compressible structure is nibble-value skew, so the realistic
    /// ratio is ~1.2x. A value near 1.0 means compression is paying a decode on
    /// every hit to save nothing, and `COLI_CACHE_COMPRESS` should be turned off
    /// for that checkpoint.
    pub fn compression_ratio(&self) -> Option<f64> {
        // Both counters are bumped on every admission, compressed or not, so with
        // the feature off they are equal and the ratio is a meaningless 1.00x.
        // Report nothing rather than a number that looks like a measurement.
        if !self.compress_on_admit || self.compressed_bytes_seen == 0 || self.uncompressed_bytes_seen == 0 {
            return None;
        }
        Some(self.uncompressed_bytes_seen as f64 / self.compressed_bytes_seen as f64)
    }

    /// Compress the coldest still-raw slot in place, freeing the byte
    /// difference from the budget accounting. Designed for idle ticks — the
    /// engine calls this between forwards when no requests are pending, so a
    /// cache admitted uncompressed (fast admits on the hot path) gradually
    /// densifies in the background. Returns the bytes saved (`0` when nothing
    /// was recompressed: every slot already compressed, empty cache, or the
    /// coldest raw slot didn't shrink). One slot per call keeps the pause
    /// bounded; call repeatedly while idle.
    pub fn recompress_one_cold(&mut self) -> usize {
        // Coldest = smallest `used` clock among Raw slots. Priority is ignored:
        // recompression keeps the bytes resident, so protection is unaffected.
        let victim = self
            .map
            .iter()
            .filter(|(_, s)| matches!(s.data, SlotBytes::Raw(_)))
            .min_by_key(|(_, s)| s.used)
            .map(|(k, _)| *k);
        let Some(key) = victim else { return 0 };
        let Some(slot) = self.map.get_mut(&key) else { return 0 };
        let SlotBytes::Raw(slab) = &slot.data else { return 0 };
        let Some((compressed, new_bytes)) = encode_slab(slab) else { return 0 };
        if new_bytes >= slot.bytes {
            return 0; // incompressible — leave raw (decode cost for no gain)
        }
        let saved = slot.bytes - new_bytes;
        self.used -= saved;
        slot.bytes = new_bytes;
        slot.data = compressed;
        saved
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn contains(&self, key: (u32, u32)) -> bool {
        // Bloom fast-path: a "definitely absent" answer skips the HashMap probe.
        // A "probably present" answer falls through to the exact map — false
        // positives are safe (the HashMap gives the true answer).
        if !self.bloom.probably_contains(key) {
            return false;
        }
        self.map.contains_key(&key)
    }

    /// Look up an expert slab. On a hit, bumps its recency and returns a
    /// (possibly decompressed) [`ExpertSlab`] clone; on a miss returns `None`
    /// (the caller streams from disk, then [`Self::insert`]s). Returns owned
    /// because compressed slots materialize their bytes fresh per hit.
    pub fn get(&mut self, key: (u32, u32)) -> Option<ExpertSlab> {
        self.clock += 1;
        // Bloom fast-miss: skip the HashMap probe entirely on "definitely absent".
        if !self.bloom.probably_contains(key) {
            self.misses += 1;
            self.bloom_skips += 1;
            return None;
        }
        let now = self.clock;
        // One probe: bump recency and materialize from the same borrow.
        let (slab, first_prefetch_hit) = match self.map.get_mut(&key) {
            Some(slot) => {
                slot.used = now;
                let first = slot.from_prefetch && !slot.ever_hit;
                if first {
                    slot.ever_hit = true;
                }
                (materialize(&slot.data), first)
            }
            None => {
                self.misses += 1;
                return None;
            }
        };
        match slab {
            Some(slab) => {
                self.hits += 1;
                if first_prefetch_hit {
                    self.prefetch_used += 1;
                }
                Some(slab)
            }
            None => {
                // A resident slot whose payload will not decode is worse than no
                // slot at all: drop it so the next lookup re-streams clean bytes,
                // and count the lookup as the miss it effectively is.
                crate::note_advisory_err("warm cache: undecodable slot dropped", &"zstd decode failed or wrong length");
                self.decode_failures += 1;
                if let Some(s) = self.map.remove(&key) {
                    self.used = self.used.saturating_sub(s.bytes);
                }
                self.rebuild_bloom();
                self.misses += 1;
                None
            }
        }
    }

    /// Slots dropped because their stored payload failed to decode. Non-zero
    /// means real data corruption (or a codec bug) — the affected experts were
    /// re-streamed, so output stays correct, but it is worth surfacing.
    pub fn decode_failures(&self) -> u64 {
        self.decode_failures
    }

    /// Corrupt a resident compressed slot's payload, to exercise the
    /// decode-failure path (bit rot in a multi-GB resident cache is exactly the
    /// scenario this guards). No-op for a raw slot.
    #[cfg(test)]
    fn corrupt_slot_for_test(&mut self, key: (u32, u32)) {
        if let Some(slot) = self.map.get_mut(&key) {
            if let SlotBytes::Compressed { six, .. } = &mut slot.data {
                six[0] = vec![0xFF; 8]; // not a valid zstd frame
            }
        }
    }

    /// Rebuild the Bloom hint from the current resident set. Called after any
    /// eviction so the hint stays tight (deletes-invalidate the flat bit array).
    fn rebuild_bloom(&mut self) {
        self.bloom.clear();
        for &key in self.map.keys() {
            self.bloom.add(key);
        }
    }

    /// The negative-cache TTL (clock ticks since last hit). `0` disables.
    pub fn negative_ttl(&self) -> u64 {
        self.negative_ttl
    }

    /// Record that a miss streamed one expert from disk at `layer`. Kept separate
    /// from `misses` so a test can distinguish "not resident" from "bytes fetched".
    pub fn note_disk_read(&mut self, layer: u32) {
        self.disk_reads += 1;
        let li = layer as usize;
        if li >= self.disk_reads_by_layer.len() {
            self.disk_reads_by_layer.resize(li + 1, 0);
        }
        self.disk_reads_by_layer[li] += 1;
    }

    /// Disk reads attributed to `layer` so far.
    pub fn disk_reads_for_layer(&self, layer: u32) -> u64 {
        self.disk_reads_by_layer.get(layer as usize).copied().unwrap_or(0)
    }

    /// Record one expert read done by the prefetch lane (off the critical path),
    /// attributed to `layer` so a test can see which layers the look-ahead warmed.
    pub fn note_prefetch_read(&mut self, layer: u32) {
        self.prefetch_reads += 1;
        let li = layer as usize;
        if li >= self.prefetch_reads_by_layer.len() {
            self.prefetch_reads_by_layer.resize(li + 1, 0);
        }
        self.prefetch_reads_by_layer[li] += 1;
    }

    /// Prefetch reads attributed to `layer` so far.
    pub fn prefetch_reads_for_layer(&self, layer: u32) -> u64 {
        self.prefetch_reads_by_layer.get(layer as usize).copied().unwrap_or(0)
    }

    /// Record one low-confidence expert hinted to the page cache via `fadvise`.
    pub fn note_fadvise(&mut self) {
        self.fadvise_hints += 1;
    }

    /// Record one speculative read whose verification re-read differed (an I/O bug).
    pub fn note_verify_mismatch(&mut self) {
        self.verify_mismatch += 1;
    }

    /// Drop all resident slabs and zero the counters. Used by tests to force a
    /// cold cache so the prefetch lane's contribution is observable in isolation.
    pub fn clear(&mut self) {
        self.map.clear();
        self.used = 0;
        self.bloom.clear();
        self.hits = 0;
        self.misses = 0;
        self.disk_reads = 0;
        self.prefetch_reads = 0;
        self.disk_reads_by_layer.clear();
        self.prefetch_reads_by_layer.clear();
        self.prefetch_used = 0;
        self.prefetch_wasted = 0;
        self.fadvise_hints = 0;
        self.verify_mismatch = 0;
        self.bloom_skips = 0;
    }

    /// Insert (or refresh) an expert slab streamed on the **critical path**,
    /// evicting least-recently-used slots until the total fits the byte budget. A
    /// single slab larger than the whole budget is still held (as the sole
    /// resident) so streaming never stalls for cache room.
    pub fn insert(&mut self, key: (u32, u32), data: ExpertSlab) {
        self.insert_inner(key, data, false);
    }

    /// Insert a slab warmed **ahead of time** by the prefetch lane. Identical to
    /// [`Self::insert`] but tags the slot so effectiveness accounting can tell
    /// whether the speculative read was later used (`prefetch_used`) or evicted
    /// unused (`prefetch_wasted`).
    pub fn insert_prefetched(&mut self, key: (u32, u32), data: ExpertSlab) {
        self.insert_inner(key, data, true);
    }

    fn insert_inner(&mut self, key: (u32, u32), data: ExpertSlab, from_prefetch: bool) {
        self.clock += 1;
        let now = self.clock;
        // Share once on admission: every later hit is a refcount bump instead of
        // a full copy of the slab while the cache lock is held.
        let data = share_slab(data);
        let raw_bytes = slab_bytes(&data);
        // Encode the payload into whichever SlotBytes representation is active.
        // `incoming` counts the resident footprint the eviction policy sees.
        // Codec failure silently degrades to `Raw` (correctness-neutral).
        let (slot_data, incoming) = if self.compress_on_admit {
            match encode_slab(&data) {
                Some((sb, compressed_bytes)) => (sb, compressed_bytes),
                None => (SlotBytes::Raw(data), raw_bytes),
            }
        } else {
            (SlotBytes::Raw(data), raw_bytes)
        };
        self.uncompressed_bytes_seen = self.uncompressed_bytes_seen.saturating_add(raw_bytes as u64);
        self.compressed_bytes_seen = self.compressed_bytes_seen.saturating_add(incoming as u64);
        let was_new;
        match self.map.get_mut(&key) {
            Some(slot) => {
                self.used = self.used - slot.bytes + incoming;
                slot.bytes = incoming;
                slot.data = slot_data;
                slot.used = now;
                // refreshing overwrites provenance: this is now a fresh fetch, so
                // re-arm the used/wasted tracking from the new source.
                slot.from_prefetch = from_prefetch;
                slot.ever_hit = false;
                was_new = false;
            }
            None => {
                self.used += incoming;
                self.map
                    .insert(key, Slot { used: now, bytes: incoming, data: slot_data, from_prefetch, ever_hit: false, prio: 0 });
                was_new = true;
            }
        }
        if was_new {
            self.bloom.add(key);
        }
        let evicted = self.evict_to_budget(Some(key));
        if evicted {
            // Slot removals invalidate flat Bloom bits — rebuild from the current
            // resident set. Cheap: ≤ 32 words × slot count.
            self.rebuild_bloom();
        }
    }

    /// Set one resident slot's eviction-protection score (no-op if not resident).
    /// **Does not touch recency** (`used`/`clock`), so protecting an expert never
    /// perturbs the LRU victim order among equal-priority slots.
    pub fn set_priority(&mut self, key: (u32, u32), prio: u32) {
        if let Some(slot) = self.map.get_mut(&key) {
            slot.prio = prio;
        }
    }

    /// Bulk [`Self::set_priority`]: protect every resident key in `keys` at `prio`.
    /// Keys not currently resident are ignored (a miss re-streams them anyway).
    pub fn set_protected(&mut self, keys: &[(u32, u32)], prio: u32) {
        for &k in keys {
            self.set_priority(k, prio);
        }
    }

    /// Reset every slot's protection score to 0 (called at sequence reset so a new
    /// sequence starts from pure LRU until its predictor re-protects experts).
    pub fn clear_priorities(&mut self) {
        for slot in self.map.values_mut() {
            slot.prio = 0;
        }
    }

    /// A resident slot's current protection score (0 if unprotected or not resident).
    /// For introspection/tests.
    pub fn priority(&self, key: (u32, u32)) -> u32 {
        self.map.get(&key).map(|s| s.prio).unwrap_or(0)
    }

    /// Evict the least-recently-used slots until `used <= budget`, always keeping
    /// at least one resident (the just-touched newcomer, which has the max clock,
    /// is never the LRU victim). Returns whether at least one slot was removed
    /// (so the caller can decide to rebuild the Bloom hint).
    fn evict_to_budget(&mut self, keep: Option<(u32, u32)>) -> bool {
        let mut removed = false;
        // Negative-cache pass first: eagerly drop slots that haven't been hit
        // within `negative_ttl` clock ticks, regardless of priority. Prevents a
        // resident-but-cold slab from squatting on cache room a genuinely-hot
        // stream would use. Disabled at TTL = 0.
        if self.negative_ttl > 0 && self.map.len() > 1 {
            let cutoff = self.clock.saturating_sub(self.negative_ttl);
            let stale: Vec<(u32, u32)> = self
                .map
                .iter()
                .filter(|(k, s)| s.used < cutoff && s.prio == 0 && Some(**k) != keep)
                .map(|(k, _)| *k)
                .collect();
            for k in stale {
                if self.map.len() <= 1 {
                    break;
                }
                if let Some(s) = self.map.remove(&k) {
                    self.used -= s.bytes;
                    if s.from_prefetch && !s.ever_hit {
                        self.prefetch_wasted += 1;
                    }
                    removed = true;
                }
            }
        }
        while self.used > self.budget && self.map.len() > 1 {
            // lowest (priority, recency) is the victim: unprotected slabs go first,
            // and within a priority the least-recently-used, exactly as before.
            // The slot just inserted is never its own victim. New slots start at
            // `prio: 0` while any predictor-protected slot is >= 1, so ordering
            // by `(prio, used)` would otherwise pick the newcomer first and make
            // every admission a silent no-op once anything is protected.
            let victim = self
                .map
                .iter()
                .filter(|(k, _)| Some(**k) != keep)
                .min_by_key(|(_, s)| (s.prio, s.used))
                .map(|(k, _)| *k);
            let Some(vk) = victim else { break };
            if let Some(s) = self.map.remove(&vk) {
                self.used -= s.bytes;
                if s.from_prefetch && !s.ever_hit {
                    self.prefetch_wasted += 1;
                }
                removed = true;
            }
        }
        removed
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab(w: usize, s: usize) -> ExpertSlab {
        let region = || (Bytes::from(vec![0u8; w]), Bytes::from(vec![0u8; s]));
        [region(), region(), region()]
    }

    /// A slab whose bytes are deterministic and non-uniform, so zstd actually
    /// finds redundancy (all-zero payloads compress to ~4 bytes each and don't
    /// stress the round-trip machinery interestingly).
    ///
    /// Deliberately NOT representative of a real payload: these periodic ramps
    /// compress ~1900x better than the packed int4 nibbles a real expert holds
    /// (~1.2x — see `peregrine_core::compress` tests). Tests over this fixture
    /// may only assert *that* compression round-trips, never how much it saves.
    fn patterned_slab(w: usize, s: usize) -> ExpertSlab {
        let mkw: Vec<u8> = (0..w).map(|k| (k % 251) as u8).collect();
        let mks: Vec<u8> = (0..s).map(|k| (k.wrapping_add(17) % 253) as u8).collect();
        let region = || (Bytes::from(mkw.clone()), Bytes::from(mks.clone()));
        [region(), region(), region()]
    }

    #[test]
    fn admission_lands_even_when_every_other_slot_is_protected() {
        // Regression: victims were chosen by `(prio, used)`, and a new slot
        // always starts at prio 0 while protected slots are >= 1 — so the
        // just-inserted slab was picked as its own victim and every admission
        // silently did nothing once the predictor protected anything.
        let mut c = WarmCache::new(3 * (1024 + 16)); // room for ~1 slab
        c.insert((0, 1), slab(1024, 16));
        c.set_priority((0, 1), 5); // predictor protects the resident
        assert!(c.contains((0, 1)));

        c.insert((0, 2), slab(1024, 16)); // over budget → something must go
        assert!(c.contains((0, 2)), "the newly admitted slab must be resident");
        assert!(c.get((0, 2)).is_some(), "and must be retrievable");
    }

    #[test]
    fn undecodable_slot_is_a_miss_not_an_empty_hit() {
        // Regression: a zstd region that failed to decode was returned as empty
        // `Bytes` on a path already counted as a HIT. `QtWeight` does no length
        // validation, so an empty weight region reached the matmul and indexed
        // out of bounds — an abort in release. The correct degradation is a
        // miss: the caller simply re-streams the expert.
        let mut c = WarmCache::new(1 << 20).with_compression(true);
        c.insert((0, 9), patterned_slab(4096, 128));
        assert!(c.get((0, 9)).is_some(), "sanity: a healthy slot hits");
        let hits_before = c.hits;

        c.corrupt_slot_for_test((0, 9));
        let got = c.get((0, 9));
        assert!(got.is_none(), "a corrupt slot must read as a miss, never as empty bytes");
        assert_eq!(c.hits, hits_before, "a failed decode must not count as a hit");
        assert_eq!(c.decode_failures(), 1, "the corruption is surfaced in a counter");
        assert!(!c.contains((0, 9)), "the poisoned slot is dropped, so the retry re-streams");
        // The cache stays usable afterwards.
        c.insert((0, 9), patterned_slab(4096, 128));
        assert!(c.get((0, 9)).is_some());
    }

    #[test]
    fn hits_share_bytes_instead_of_copying() {
        // Admission converts the slab to refcounted bytes, so a hit is six
        // refcount bumps rather than a ~19 MB copy taken while the cache lock is
        // held (which serialized every I/O lane behind one memcpy).
        let mut c = WarmCache::new(1 << 20);
        c.insert((1, 3), patterned_slab(8192, 64));
        let (a, b) = (c.get((1, 3)), c.get((1, 3)));
        assert!(a.is_some() && b.is_some(), "both lookups must hit");
        if let (Some(a), Some(b)) = (a, b) {
            for i in 0..3 {
                assert_eq!(&a[i].0[..], &b[i].0[..], "region {i} bytes match");
                assert!(
                    std::ptr::eq(&a[i].0[0], &b[i].0[0]),
                    "region {i} must be shared, not copied per hit"
                );
            }
        }
    }

    #[test]
    fn recompress_one_cold_shrinks_and_round_trips() {
        // Admit raw (compression off), recompress in the background path, and
        // confirm: bytes accounting shrank, and a subsequent hit still returns
        // byte-identical data.
        let mut c = WarmCache::new(1 << 20); // compress_on_admit off by default env
        let expected = patterned_slab(4096, 128);
        let expected_bytes: Vec<(Vec<u8>, Vec<u8>)> =
            expected.iter().map(|(w, s)| (w.to_vec(), s.to_vec())).collect();
        c.insert((0, 7), expected);
        let before = c.used();
        let saved = c.recompress_one_cold();
        assert!(saved > 0, "patterned slab must compress");
        assert_eq!(c.used(), before - saved, "budget accounting tracks the shrink");
        // Second call: nothing raw left → no-op.
        assert_eq!(c.recompress_one_cold(), 0);
        // Hit still round-trips byte-identically.
        let got = c.get((0, 7));
        assert!(got.is_some());
        if let Some(got) = got {
            for (i, (w, s)) in got.iter().enumerate() {
                assert_eq!(&w[..], &expected_bytes[i].0[..], "region {i} weight bytes");
                assert_eq!(&s[..], &expected_bytes[i].1[..], "region {i} scale bytes");
            }
        }
    }

    #[test]
    fn compressed_admit_round_trips_bytes() {
        // With compression enabled, admissions are zstd-encoded on the way in
        // and decoded on the way out. Hits must be byte-identical to the
        // originally-inserted slab (transparent compression == correctness-
        // neutral). We opt in via the constructor override to keep this test
        // from racing other threads' `COLI_CACHE_COMPRESS` reads.
        let mut c = WarmCache::new(1 << 20).with_compression(true);
        let expected = patterned_slab(4096, 128);
        let expected_bytes: Vec<(Vec<u8>, Vec<u8>)> =
            expected.iter().map(|(w, s)| (w.to_vec(), s.to_vec())).collect();
        c.insert((0, 42), expected);
        let got = c.get((0, 42));
        assert!(got.is_some(), "must hit");
        let got = match got {
            Some(g) => g,
            None => return,
        };
        for (i, (w, s)) in got.iter().enumerate() {
            assert_eq!(&w[..], &expected_bytes[i].0[..], "region {i} weight bytes");
            assert_eq!(&s[..], &expected_bytes[i].1[..], "region {i} scale bytes");
        }
        // Both counters must have moved, so `compression_ratio()` has something to
        // report. Deliberately NOT asserting how much was saved: this fixture is a
        // periodic ramp, and any magnitude measured on it says nothing about the
        // real payload (see `patterned_slab`). The honest figure is pinned by
        // `peregrine_core::compress`'s `quantized_nibbles_compress_by_entropy_only`.
        assert!(c.uncompressed_bytes_seen > 0, "raw footprint must be counted");
        assert!(c.compressed_bytes_seen > 0, "admitted footprint must be counted");
        assert!(c.compression_ratio().is_some(), "ratio must be reportable after a compressed admit");
    }

    #[test]
    fn compression_ratio_is_none_until_a_compressed_admit_lands() {
        // Nothing admitted yet.
        let mut c = WarmCache::new(1 << 20).with_compression(true);
        assert!(c.compression_ratio().is_none(), "no admissions yet");
        // Compression off: the counters stay untouched, so there is no ratio to
        // report rather than a misleading 1.00x.
        let mut off = WarmCache::new(1 << 20).with_compression(false);
        off.insert((0, 1), patterned_slab(4096, 128));
        assert!(off.compression_ratio().is_none(), "compression disabled reports nothing");
        // Compression on: a ratio becomes available.
        c.insert((0, 1), patterned_slab(4096, 128));
        assert!(c.compression_ratio().is_some_and(|r| r > 1.0), "compressed admit reports a ratio");
    }

    #[test]
    fn hit_and_miss() {
        let mut c = WarmCache::new(1 << 20);
        assert!(c.get((0, 1)).is_none()); // miss
        c.insert((0, 1), slab(4, 2));
        assert!(c.get((0, 1)).is_some()); // hit
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
        // same expert id in a different layer is a distinct key
        assert!(c.get((1, 1)).is_none());
    }

    #[test]
    fn byte_budget_evicts_lru() {
        // each slab = 3*(10+2) = 36 bytes; budget fits two.
        let mut c = WarmCache::new(80);
        c.insert((0, 0), slab(10, 2));
        c.insert((0, 1), slab(10, 2));
        assert!(c.get((0, 0)).is_some()); // touch 0 → (0,1) becomes LRU
        c.insert((0, 2), slab(10, 2)); // evicts (0,1)
        assert!(c.contains((0, 0)));
        assert!(c.contains((0, 2)));
        assert!(!c.contains((0, 1)));
        assert!(c.used() <= c.budget(), "used {} must stay within budget {}", c.used(), c.budget());
    }

    #[test]
    fn refresh_updates_bytes_not_count() {
        let mut c = WarmCache::new(1 << 20);
        c.insert((2, 5), slab(4, 1));
        let u1 = c.used();
        c.insert((2, 5), slab(8, 1)); // same key, larger → refresh, not a new slot
        assert_eq!(c.len(), 1);
        assert!(c.used() > u1, "byte accounting must track the larger payload");
    }

    #[test]
    fn oversized_slab_stays_resident() {
        // a slab bigger than the whole budget is kept as the sole resident.
        let mut c = WarmCache::new(8);
        c.insert((0, 0), slab(100, 4));
        assert_eq!(c.len(), 1);
        assert!(c.contains((0, 0)));
    }

    #[test]
    fn prefetch_used_counts_once_on_first_hit() {
        let mut c = WarmCache::new(1 << 20);
        c.insert_prefetched((0, 1), slab(4, 2));
        assert_eq!(c.prefetch_used, 0); // warmed, not yet used
        assert!(c.get((0, 1)).is_some());
        assert!(c.get((0, 1)).is_some()); // second hit must not double-count
        assert_eq!(c.prefetch_used, 1);
        assert_eq!(c.prefetch_wasted, 0);
    }

    #[test]
    fn prefetch_wasted_counts_eviction_before_use() {
        // budget fits two 36-byte slabs; a third insert evicts the LRU.
        let mut c = WarmCache::new(80);
        c.insert_prefetched((0, 0), slab(10, 2)); // prefetched, never hit → LRU victim
        c.insert((0, 1), slab(10, 2));
        c.insert((0, 2), slab(10, 2)); // evicts (0,0) unused
        assert!(!c.contains((0, 0)));
        assert_eq!(c.prefetch_wasted, 1);
        assert_eq!(c.prefetch_used, 0);
    }

    #[test]
    fn prefetch_used_then_evicted_is_not_wasted() {
        let mut c = WarmCache::new(80);
        c.insert_prefetched((0, 0), slab(10, 2));
        assert!(c.get((0, 0)).is_some()); // use it, then let it fall out
        c.insert((0, 1), slab(10, 2));
        c.insert((0, 2), slab(10, 2)); // (0,0) is now LRU → evicted, but it was used
        assert!(!c.contains((0, 0)));
        assert_eq!(c.prefetch_used, 1);
        assert_eq!(c.prefetch_wasted, 0);
    }

    #[test]
    fn priority_protects_older_slab_from_eviction() {
        // budget fits two 36-byte slabs. Protect the OLDER one; a third insert must
        // evict the unprotected (younger-but-lower-priority) slab instead of the LRU.
        let mut c = WarmCache::new(80);
        c.insert((0, 0), slab(10, 2)); // oldest
        c.insert((0, 1), slab(10, 2));
        c.set_priority((0, 0), 1); // protect the LRU
        c.insert((0, 2), slab(10, 2)); // over budget → evict lowest (prio, used) = (0,1)
        assert!(c.contains((0, 0)), "protected slab must survive despite being LRU");
        assert!(!c.contains((0, 1)), "unprotected slab is the victim");
        assert!(c.contains((0, 2)));
    }

    #[test]
    fn all_equal_priority_is_pure_lru() {
        // With no priorities set (all 0), eviction must match the original LRU order.
        let mut c = WarmCache::new(80);
        c.insert((0, 0), slab(10, 2));
        c.insert((0, 1), slab(10, 2));
        assert!(c.get((0, 0)).is_some()); // touch 0 → (0,1) is LRU
        c.clear_priorities(); // still all 0
        c.insert((0, 2), slab(10, 2));
        assert!(c.contains((0, 0)));
        assert!(!c.contains((0, 1))); // LRU victim, exactly as byte_budget_evicts_lru
        assert!(c.contains((0, 2)));
    }

    #[test]
    fn set_priority_does_not_bump_recency() {
        // Protecting then un-protecting a slab must leave LRU order unchanged, i.e.
        // priority is orthogonal to recency.
        let mut c = WarmCache::new(80);
        c.insert((0, 0), slab(10, 2)); // LRU
        c.insert((0, 1), slab(10, 2));
        c.set_priority((0, 0), 5);
        c.set_priority((0, 0), 0); // back to unprotected; must not have touched `used`
        c.insert((0, 2), slab(10, 2));
        assert!(!c.contains((0, 0)), "(0,0) is still the LRU victim");
        assert!(c.contains((0, 1)));
        assert!(c.contains((0, 2)));
    }

    #[test]
    fn per_layer_disk_reads() {
        let mut c = WarmCache::new(1 << 20);
        c.note_disk_read(0);
        c.note_disk_read(2);
        c.note_disk_read(2);
        assert_eq!(c.disk_reads, 3);
        assert_eq!(c.disk_reads_for_layer(0), 1);
        assert_eq!(c.disk_reads_for_layer(2), 2);
        assert_eq!(c.disk_reads_for_layer(1), 0);
    }
}

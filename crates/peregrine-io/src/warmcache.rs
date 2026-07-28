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

/// Total bytes a slab occupies (all six weight/scale regions).
fn slab_bytes(s: &ExpertSlab) -> usize {
    s.iter().map(|(w, sc)| w.len() + sc.len()).sum()
}

struct Slot {
    used: u64,
    bytes: usize,
    data: ExpertSlab,
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

/// Bounded-by-bytes LRU cache of expert slabs, keyed by `(layer, expert)`.
pub struct WarmCache {
    budget: usize,
    used: usize,
    map: HashMap<(u32, u32), Slot>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
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
}

impl WarmCache {
    /// A cache bounded to `budget_bytes` of resident expert slabs.
    pub fn new(budget_bytes: usize) -> WarmCache {
        WarmCache {
            budget: budget_bytes,
            used: 0,
            map: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            disk_reads: 0,
            prefetch_reads: 0,
            disk_reads_by_layer: Vec::new(),
            prefetch_reads_by_layer: Vec::new(),
            prefetch_used: 0,
            prefetch_wasted: 0,
            fadvise_hints: 0,
            verify_mismatch: 0,
        }
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
        self.map.contains_key(&key)
    }

    /// Look up an expert slab. On a hit, bumps its recency and returns the bytes;
    /// on a miss returns `None` (the caller streams from disk, then [`Self::insert`]s).
    pub fn get(&mut self, key: (u32, u32)) -> Option<&ExpertSlab> {
        self.clock += 1;
        let now = self.clock;
        let mut first_prefetch_hit = false;
        let hit = match self.map.get_mut(&key) {
            Some(slot) => {
                slot.used = now;
                if slot.from_prefetch && !slot.ever_hit {
                    slot.ever_hit = true;
                    first_prefetch_hit = true;
                }
                true
            }
            None => false,
        };
        if hit {
            self.hits += 1;
            if first_prefetch_hit {
                self.prefetch_used += 1;
            }
            self.map.get(&key).map(|s| &s.data)
        } else {
            self.misses += 1;
            None
        }
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
        let incoming = slab_bytes(&data);
        match self.map.get_mut(&key) {
            Some(slot) => {
                self.used = self.used - slot.bytes + incoming;
                slot.bytes = incoming;
                slot.data = data;
                slot.used = now;
                // refreshing overwrites provenance: this is now a fresh fetch, so
                // re-arm the used/wasted tracking from the new source.
                slot.from_prefetch = from_prefetch;
                slot.ever_hit = false;
            }
            None => {
                self.used += incoming;
                self.map
                    .insert(key, Slot { used: now, bytes: incoming, data, from_prefetch, ever_hit: false, prio: 0 });
            }
        }
        self.evict_to_budget();
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
    /// is never the LRU victim).
    fn evict_to_budget(&mut self) {
        while self.used > self.budget && self.map.len() > 1 {
            // lowest (priority, recency) is the victim: unprotected slabs go first,
            // and within a priority the least-recently-used, exactly as before.
            let victim = self.map.iter().min_by_key(|(_, s)| (s.prio, s.used)).map(|(k, _)| *k);
            let Some(vk) = victim else { break };
            if let Some(s) = self.map.remove(&vk) {
                self.used -= s.bytes;
                if s.from_prefetch && !s.ever_hit {
                    self.prefetch_wasted += 1;
                }
            }
        }
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

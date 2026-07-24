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

/// One expert's streamed bytes: `(weight, scale)` for each of gate, up, down —
/// exactly what the I/O lane reads and hands to `rebuild`.
pub type ExpertSlab = [(Vec<u8>, Vec<u8>); 3];

/// Total bytes a slab occupies (all six weight/scale regions).
fn slab_bytes(s: &ExpertSlab) -> usize {
    s.iter().map(|(w, sc)| w.len() + sc.len()).sum()
}

struct Slot {
    used: u64,
    bytes: usize,
    data: ExpertSlab,
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
        let hit = match self.map.get_mut(&key) {
            Some(slot) => {
                slot.used = now;
                true
            }
            None => false,
        };
        if hit {
            self.hits += 1;
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

    /// Record one expert read done by the prefetch lane (off the critical path).
    pub fn note_prefetch_read(&mut self) {
        self.prefetch_reads += 1;
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
    }

    /// Insert (or refresh) an expert slab, evicting least-recently-used slots until
    /// the total fits the byte budget. A single slab larger than the whole budget
    /// is still held (as the sole resident) so streaming never stalls for cache room.
    pub fn insert(&mut self, key: (u32, u32), data: ExpertSlab) {
        self.clock += 1;
        let now = self.clock;
        let incoming = slab_bytes(&data);
        match self.map.get_mut(&key) {
            Some(slot) => {
                self.used = self.used - slot.bytes + incoming;
                slot.bytes = incoming;
                slot.data = data;
                slot.used = now;
            }
            None => {
                self.used += incoming;
                self.map.insert(key, Slot { used: now, bytes: incoming, data });
            }
        }
        self.evict_to_budget();
    }

    /// Evict the least-recently-used slots until `used <= budget`, always keeping
    /// at least one resident (the just-touched newcomer, which has the max clock,
    /// is never the LRU victim).
    fn evict_to_budget(&mut self) {
        while self.used > self.budget && self.map.len() > 1 {
            let victim = self.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k);
            let Some(vk) = victim else { break };
            if let Some(s) = self.map.remove(&vk) {
                self.used -= s.bytes;
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
        [(vec![0u8; w], vec![0u8; s]), (vec![0u8; w], vec![0u8; s]), (vec![0u8; w], vec![0u8; s])]
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

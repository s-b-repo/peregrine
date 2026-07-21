//! Per-layer LRU expert cache — the RAM warm tier. Models `ecache`/`ESlot.used`
//! from `c/glm.c`: a bounded set of expert slabs keyed by expert id, evicting
//! the least-recently-used slot on a miss. The pinned hot-store (never evicted)
//! and live re-tiering are driven by [`crate::tier`] and land in the M4 scheduler.

use std::collections::HashMap;

struct Slot {
    eid: i32,
    used: u64,
    data: Vec<u8>,
}

/// Bounded LRU cache of expert slabs.
pub struct ExpertCache {
    cap: usize,
    slots: Vec<Slot>,
    map: HashMap<i32, usize>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl ExpertCache {
    pub fn new(cap: usize) -> ExpertCache {
        ExpertCache { cap: cap.max(1), slots: Vec::new(), map: HashMap::new(), clock: 0, hits: 0, misses: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn contains(&self, eid: i32) -> bool {
        self.map.contains_key(&eid)
    }

    /// Look up an expert. On a hit, bumps its recency and returns the slab; on a
    /// miss returns `None` (the caller streams it, then [`Self::insert`]s it).
    pub fn get(&mut self, eid: i32) -> Option<&[u8]> {
        match self.map.get(&eid).copied() {
            Some(idx) => {
                self.clock += 1;
                self.slots[idx].used = self.clock;
                self.hits += 1;
                Some(&self.slots[idx].data)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert (or refresh) an expert slab, evicting the LRU slot when full.
    pub fn insert(&mut self, eid: i32, data: Vec<u8>) {
        self.clock += 1;
        if let Some(&idx) = self.map.get(&eid) {
            self.slots[idx].data = data;
            self.slots[idx].used = self.clock;
            return;
        }
        if self.slots.len() < self.cap {
            let idx = self.slots.len();
            self.slots.push(Slot { eid, used: self.clock, data });
            self.map.insert(eid, idx);
        } else {
            let mut victim = 0;
            for z in 1..self.slots.len() {
                if self.slots[z].used < self.slots[victim].used {
                    victim = z;
                }
            }
            self.map.remove(&self.slots[victim].eid);
            self.slots[victim] = Slot { eid, used: self.clock, data };
            self.map.insert(eid, victim);
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

    #[test]
    fn hit_and_miss() {
        let mut c = ExpertCache::new(2);
        assert!(c.get(1).is_none()); // miss
        c.insert(1, vec![10, 11]);
        assert_eq!(c.get(1), Some(&[10, 11][..])); // hit
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
    }

    #[test]
    fn evicts_least_recently_used() {
        let mut c = ExpertCache::new(2);
        c.insert(1, vec![1]);
        c.insert(2, vec![2]);
        // touch 1 → 2 becomes LRU
        assert!(c.get(1).is_some());
        c.insert(3, vec![3]); // evicts 2
        assert!(c.contains(1));
        assert!(c.contains(3));
        assert!(!c.contains(2));
    }

    #[test]
    fn refresh_does_not_grow() {
        let mut c = ExpertCache::new(2);
        c.insert(5, vec![1]);
        c.insert(5, vec![2]); // same eid → update, not a second slot
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(5), Some(&[2][..]));
    }
}

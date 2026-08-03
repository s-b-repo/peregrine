//! Bounded exact response memo.
//!
//! An OpenAI-compatible server is re-asked the same question constantly — a health
//! probe, a retried request, a client that re-sends a conversation unchanged, an eval
//! harness replaying a fixture. On an engine where a single token costs a pass over
//! gigabytes of streamed experts, serving one of those from memory instead of from
//! the model is worth more than it would be almost anywhere else.
//!
//! Three rules keep it from becoming a correctness hazard.
//!
//! **It keeps token ids, not a rendered response.** A hit rebuilds the transport
//! framing — completion id, `created`, SSE chunking — so a memoized reply is a fresh
//! response with the same content, and a streaming request can be served from a
//! non-streaming entry and vice versa. Storing the wire bytes would leak one
//! request's identifiers into another's response.
//!
//! **A hit never enters the model.** It cannot stage or publish KV state, and it does
//! not touch the prefix cache. Treating a response-cache hit as a new provider
//! boundary is how a cache stops being an optimization and starts being a source of
//! state, and that is the one thing this must not do.
//!
//! **Only deterministic requests are eligible.** `temperature > 0` samples against a
//! clock-derived seed; replaying a stored answer for it would quietly convert a
//! sampling endpoint into a deterministic one — a user asking twice for variety would
//! get the same text and no indication why. Greedy decoding is reproducible by
//! contract, so that is where memoization is honest. See [`MemoKey::eligible`].
//!
//! Bounded by entry count *and* bytes, both settable to zero to disable.

use std::collections::VecDeque;

/// Everything about a request that can change its output. Two requests share a memo
/// entry only if every one of these agrees.
///
/// Compared field-by-field rather than by hash, matching the prefix cache's rule and
/// for the same reason: a hash collision would serve one user another user's answer,
/// silently and with no bound on how wrong it is. The keys are a few hundred bytes
/// and the table holds tens of them, so there is nothing to buy by hashing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoKey {
    /// The tokenized prompt — the rendered chat, so a change of role, ordering or a
    /// single character of content is a different key.
    pub ids: Vec<u32>,
    pub max_new: usize,
    /// `top_p` by its exact bits. A float key compared by bit pattern is deliberate:
    /// `0.95` from two requests is the same answer only when it is the same number.
    pub top_p_bits: u32,
    pub model: String,
}

impl MemoKey {
    /// Whether a request with this sampling configuration may be memoized at all.
    ///
    /// Greedy only. The temperature is not part of the key because a non-zero one
    /// disqualifies the request outright rather than selecting a different entry —
    /// keying on it would build a table of answers that are each one sample from a
    /// distribution and then serve them as if they were *the* answer.
    pub fn eligible(temperature: f32) -> bool {
        temperature == 0.0
    }

    fn bytes(&self) -> usize {
        self.ids.len() * std::mem::size_of::<u32>() + self.model.len() + std::mem::size_of::<MemoKey>()
    }
}

struct Entry {
    key: MemoKey,
    /// The generated token ids, exactly as the engine emitted them.
    out: Vec<u32>,
    bytes: usize,
}

/// A bounded LRU of certified responses.
pub struct ResponseMemo {
    entries: VecDeque<Entry>,
    max_entries: usize,
    max_bytes: usize,
    used: usize,
    hits: u64,
    misses: u64,
}

impl ResponseMemo {
    pub fn new(max_entries: usize, max_bytes: usize) -> ResponseMemo {
        ResponseMemo { entries: VecDeque::new(), max_entries, max_bytes, used: 0, hits: 0, misses: 0 }
    }

    /// Build from the environment: `COLI_MEMO_ENTRIES` (default 32) and
    /// `COLI_MEMO_MB` (default 64). Either at `0` disables the memo entirely.
    pub fn from_env() -> ResponseMemo {
        let entries = env_usize("COLI_MEMO_ENTRIES", 32);
        let mb = env_usize("COLI_MEMO_MB", 64);
        ResponseMemo::new(entries, mb.saturating_mul(1024 * 1024))
    }

    pub fn enabled(&self) -> bool {
        self.max_entries > 0 && self.max_bytes > 0
    }

    /// The stored completion for `key`, if there is one. Refreshes its recency.
    pub fn get(&mut self, key: &MemoKey) -> Option<Vec<u32>> {
        if !self.enabled() {
            return None;
        }
        let Some(i) = self.entries.iter().position(|e| &e.key == key) else {
            self.misses += 1;
            return None;
        };
        self.hits += 1;
        // Move to the back = most recently used.
        let e = self.entries.remove(i)?;
        let out = e.out.clone();
        self.entries.push_back(e);
        Some(out)
    }

    /// Store a completed response. Evicts least-recently-used entries until both
    /// bounds hold. An entry too large for the whole budget is not stored at all
    /// rather than emptying the table to make room for it.
    pub fn insert(&mut self, key: MemoKey, out: Vec<u32>) {
        if !self.enabled() || out.is_empty() {
            return;
        }
        let bytes = key.bytes() + out.len() * std::mem::size_of::<u32>();
        if bytes > self.max_bytes {
            return;
        }
        // Replacing an existing key keeps the table a set, not a multiset — otherwise
        // a repeated request grows the table with duplicates of one answer.
        if let Some(i) = self.entries.iter().position(|e| e.key == key) {
            if let Some(old) = self.entries.remove(i) {
                self.used = self.used.saturating_sub(old.bytes);
            }
        }
        self.entries.push_back(Entry { key, out, bytes });
        self.used = self.used.saturating_add(bytes);
        while self.entries.len() > self.max_entries || self.used > self.max_bytes {
            match self.entries.pop_front() {
                Some(old) => self.used = self.used.saturating_sub(old.bytes),
                None => break,
            }
        }
    }

    /// `(hits, misses, entries, bytes)` — for the `/health` view and shutdown logs.
    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (self.hits, self.misses, self.entries.len(), self.used)
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ids: &[u32], max_new: usize) -> MemoKey {
        MemoKey { ids: ids.to_vec(), max_new, top_p_bits: 0.95f32.to_bits(), model: "m".into() }
    }

    #[test]
    fn a_repeated_request_is_served_from_the_memo() {
        let mut m = ResponseMemo::new(4, 1 << 20);
        assert_eq!(m.get(&key(&[1, 2, 3], 16)), None);
        m.insert(key(&[1, 2, 3], 16), vec![9, 8, 7]);
        assert_eq!(m.get(&key(&[1, 2, 3], 16)), Some(vec![9, 8, 7]));
        let (hits, misses, entries, _) = m.stats();
        assert_eq!((hits, misses, entries), (1, 1, 1));
    }

    #[test]
    fn one_token_or_one_option_of_difference_misses() {
        // The memo's whole safety argument is that the key is the complete request
        // semantics. Each of these is a different question and must get a different
        // answer, not a near-match.
        let mut m = ResponseMemo::new(8, 1 << 20);
        m.insert(key(&[1, 2, 3], 16), vec![9]);
        assert_eq!(m.get(&key(&[1, 2, 4], 16)), None, "a changed token");
        assert_eq!(m.get(&key(&[1, 2], 16)), None, "a truncated prompt");
        assert_eq!(m.get(&key(&[1, 2, 3, 4], 16)), None, "an extended prompt");
        assert_eq!(m.get(&key(&[1, 2, 3], 17)), None, "a changed max_tokens");
        let mut other_top_p = key(&[1, 2, 3], 16);
        other_top_p.top_p_bits = 0.9f32.to_bits();
        assert_eq!(m.get(&other_top_p), None, "a changed top_p");
        let mut other_model = key(&[1, 2, 3], 16);
        other_model.model = "n".into();
        assert_eq!(m.get(&other_model), None, "a different model id");
        // ...and the original is still there, untouched by all those misses.
        assert_eq!(m.get(&key(&[1, 2, 3], 16)), Some(vec![9]));
    }

    #[test]
    fn only_greedy_requests_are_eligible() {
        // Memoizing a sampled request would turn variety into repetition with no
        // indication to the caller that it had happened.
        assert!(MemoKey::eligible(0.0));
        assert!(!MemoKey::eligible(0.7));
        assert!(!MemoKey::eligible(f32::MIN_POSITIVE));
    }

    #[test]
    fn both_bounds_are_enforced_and_zero_disables() {
        // Entry count.
        let mut m = ResponseMemo::new(2, 1 << 20);
        for i in 0..3u32 {
            m.insert(key(&[i], 8), vec![i]);
        }
        assert_eq!(m.stats().2, 2, "capped at max_entries");
        assert_eq!(m.get(&key(&[0], 8)), None, "the oldest was evicted");
        assert_eq!(m.get(&key(&[2], 8)), Some(vec![2]), "the newest survives");

        // Byte budget: a tiny one holds at most one entry.
        let small = std::mem::size_of::<MemoKey>() + 64;
        let mut m = ResponseMemo::new(100, small);
        m.insert(key(&[1], 8), vec![1; 4]);
        m.insert(key(&[2], 8), vec![2; 4]);
        assert!(m.stats().3 <= small, "byte budget respected: {} > {small}", m.stats().3);

        // Either bound at zero is off.
        let mut off = ResponseMemo::new(0, 1 << 20);
        off.insert(key(&[1], 8), vec![1]);
        assert!(!off.enabled());
        assert_eq!(off.get(&key(&[1], 8)), None);
        let mut off = ResponseMemo::new(8, 0);
        off.insert(key(&[1], 8), vec![1]);
        assert_eq!(off.get(&key(&[1], 8)), None);
    }

    #[test]
    fn a_response_too_large_for_the_budget_is_skipped_not_ruinous() {
        // Storing it would evict everything and then not fit either.
        let mut m = ResponseMemo::new(8, std::mem::size_of::<MemoKey>() + 128);
        m.insert(key(&[1], 8), vec![1, 2]);
        let before = m.stats().2;
        m.insert(key(&[2], 8), vec![0u32; 10_000]);
        assert_eq!(m.stats().2, before, "the oversized entry did not displace the table");
        assert_eq!(m.get(&key(&[1], 8)), Some(vec![1, 2]));
    }

    #[test]
    fn reinserting_a_key_replaces_rather_than_duplicates() {
        let mut m = ResponseMemo::new(8, 1 << 20);
        m.insert(key(&[1], 8), vec![1]);
        let used_once = m.stats().3;
        m.insert(key(&[1], 8), vec![2]);
        assert_eq!(m.stats().2, 1, "one key, one entry");
        assert_eq!(m.stats().3, used_once, "and its bytes counted once");
        assert_eq!(m.get(&key(&[1], 8)), Some(vec![2]), "the newer answer wins");
    }

    #[test]
    fn an_empty_completion_is_not_stored() {
        // Nothing to serve, and storing it would let a failed generation shadow a
        // later successful one for the same request.
        let mut m = ResponseMemo::new(8, 1 << 20);
        m.insert(key(&[1], 8), vec![]);
        assert_eq!(m.stats().2, 0);
    }

    #[test]
    fn recency_is_refreshed_by_a_hit() {
        let mut m = ResponseMemo::new(2, 1 << 20);
        m.insert(key(&[1], 8), vec![1]);
        m.insert(key(&[2], 8), vec![2]);
        assert_eq!(m.get(&key(&[1], 8)), Some(vec![1])); // 1 is now the newest
        m.insert(key(&[3], 8), vec![3]); // evicts the least recent, which is 2
        assert_eq!(m.get(&key(&[2], 8)), None);
        assert_eq!(m.get(&key(&[1], 8)), Some(vec![1]));
    }
}

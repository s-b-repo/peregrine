//! Context plane for chat prompts (`COLI_CTX`), backed by `peregrine-ctx`
//! (the ported subnext/DCR runtime).
//!
//! When on, the prompt the model sees is the runtime's planned working set:
//! message turns ingested role-tagged, relevance-planned for the last user
//! turn under the token budget, rendered as context. The plane is a context
//! source, not a compression fallback — it assembles what the model should
//! attend to on every request, not only over-budget ones. With the knob off
//! the raw rendered prompt serves exactly as before (bit-identical).
//!
//! Stateless per call — a fresh runtime per request, no cross-request memory
//! (the server is stateless per request), deterministic for identical turns.
//! An ingestion failure or empty assembly falls back to the raw prompt, never
//! a partial context: a turn that fails to ingest aborts the whole assembly.

use std::sync::OnceLock;

use parking_lot::Mutex;
use peregrine_ctx::Dcr;

struct CacheEntry {
    turns: Vec<(String, String)>,
    query: String,
    budget: usize,
    output: String,
    bytes: usize,
    stamp: u64,
}

struct AssemblyCache {
    entries: Vec<CacheEntry>,
    max_entries: usize,
    max_bytes: usize,
    used: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl AssemblyCache {
    fn new() -> Self {
        let max_entries = std::env::var("COLI_CTX_CACHE_ENTRIES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32);
        let max_mb = std::env::var("COLI_CTX_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16);
        Self::with_limits(max_entries, max_mb.saturating_mul(1024 * 1024))
    }

    fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            max_bytes,
            used: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, turns: &[(String, String)], query: &str, budget: usize) -> Option<String> {
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        let index = self.entries.iter().position(|entry| {
            entry.budget == budget && entry.query == query && entry.turns == turns
        })?;
        self.entries[index].stamp = stamp;
        self.hits = self.hits.saturating_add(1);
        Some(self.entries[index].output.clone())
    }

    fn insert(&mut self, turns: &[(String, String)], query: &str, budget: usize, output: String) {
        if self.max_entries == 0 || self.max_bytes == 0 {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        let bytes = cache_entry_bytes(turns, query, budget, &output);
        if bytes > self.max_bytes {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.budget == budget && entry.query == query && entry.turns == turns)
        {
            self.used = self.used.saturating_sub(self.entries[index].bytes);
            self.entries[index] = CacheEntry {
                turns: turns.to_vec(),
                query: query.to_string(),
                budget,
                output,
                bytes,
                stamp,
            };
            self.used = self.used.saturating_add(bytes);
            self.evict();
            return;
        }
        self.entries.push(CacheEntry {
            turns: turns.to_vec(),
            query: query.to_string(),
            budget,
            output,
            bytes,
            stamp,
        });
        self.used = self.used.saturating_add(bytes);
        self.evict();
    }

    fn evict(&mut self) {
        while self.entries.len() > self.max_entries || self.used > self.max_bytes {
            let Some(index) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.stamp)
                .map(|(index, _)| index)
            else {
                break;
            };
            let entry = self.entries.swap_remove(index);
            self.used = self.used.saturating_sub(entry.bytes);
        }
    }
}

fn cache_entry_bytes(turns: &[(String, String)], query: &str, budget: usize, output: &str) -> usize {
    turns
        .iter()
        .map(|(role, text)| role.len().saturating_add(text.len()).saturating_add(16))
        .chain(std::iter::once(query.len().saturating_add(budget)))
        .fold(output.len().saturating_add(64), usize::saturating_add)
}

fn assembly_cache() -> &'static Mutex<AssemblyCache> {
    static CACHE: OnceLock<Mutex<AssemblyCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(AssemblyCache::new()))
}

/// Context-plane master switch (`COLI_CTX=1`). Off by default: engagement
/// changes tokens, so it must be asked for, like every other knob that can
/// move a token.
pub fn ctx_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_CTX").as_deref(), Ok("1") | Ok("true")))
}

/// Assemble `turns` (`(role, text)` oldest-first) into a working set for
/// `query` under `budget` tokens. Returns `None` when the plane is off, when
/// nothing ingested, or when assembly yields no text — every `None` means
/// "serve the raw prompt", never "serve a partial context".
pub fn assemble_context(turns: &[(String, String)], query: &str, budget: usize) -> Option<String> {
    if !ctx_enabled() {
        return None;
    }
    if let Some(output) = assembly_cache().lock().get(turns, query, budget) {
        return Some(output);
    }
    let mut cache = assembly_cache().lock();
    cache.misses = cache.misses.saturating_add(1);
    drop(cache);
    let output = assemble_context_enabled(turns, query, budget);
    if let Some(output) = &output {
        assembly_cache().lock().insert(turns, query, budget, output.clone());
    }
    output
}

/// [`assemble_context`] without the env gate — the testable core. Same
/// contract: `None` on empty input, failed ingest, or empty assembly.
pub fn assemble_context_enabled(
    turns: &[(String, String)],
    query: &str,
    budget: usize,
) -> Option<String> {
    if turns.is_empty() || budget == 0 {
        return None;
    }
    let mut rt = Dcr::new(budget);
    let mut ingested = 0usize;
    for (role, text) in turns {
        if text.trim().is_empty() {
            continue;
        }
        let tagged = format!("[{role}]\n{text}");
        if let Err(e) = rt.ingest(&tagged, None) {
            peregrine_core::note_advisory_err("ctx ingest (falling back to raw prompt)", &e);
            return None;
        }
        ingested += 1;
    }
    if ingested == 0 {
        return None;
    }
    let context = rt.plan(query, Some(budget));
    let assembled = rt.prompt_for(&context);
    if assembled.trim().is_empty() {
        return None;
    }
    Some(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_cache_is_exact_and_bounded() {
        let turns = vec![("user".to_string(), "hello".to_string())];
        let mut cache = AssemblyCache::with_limits(2, 1024);
        cache.insert(&turns, "hello", 128, "one".to_string());
        assert_eq!(cache.get(&turns, "hello", 128).as_deref(), Some("one"));
        assert!(cache.get(&turns, "different", 128).is_none());
        assert!(cache.get(&turns, "hello", 129).is_none());
        cache.insert(&turns, "second", 128, "two".to_string());
        cache.insert(&turns, "third", 128, "three".to_string());
        assert!(cache.entries.len() <= 2);
        assert!(cache.used <= cache.max_bytes);
    }

    #[test]
    fn oversized_assembly_is_not_cached() {
        let turns = vec![("user".to_string(), "hello".to_string())];
        let mut cache = AssemblyCache::with_limits(2, 8);
        cache.insert(&turns, "hello", 128, "a".repeat(64));
        assert!(cache.entries.is_empty());
        assert_eq!(cache.used, 0);
    }

    #[test]
    fn empty_input_never_assembles() {
        assert!(assemble_context_enabled(&[], "q", 100).is_none());
        assert!(assemble_context_enabled(&[("user".to_string(), "hi".to_string())], "q", 0).is_none());
        // Whitespace-only turns ingest nothing: no partial context.
        assert!(assemble_context_enabled(&[("user".to_string(), "   ".to_string())], "q", 100).is_none());
    }

    #[test]
    fn assembly_covers_the_query_terms() -> Result<(), String> {
        let turns = vec![
            ("system".to_string(), "You are a network assistant.".to_string()),
            ("user".to_string(), "The server ip is 10.0.4.12.".to_string()),
            ("user".to_string(), "The deploy window is 02:00.".to_string()),
            ("user".to_string(), "What is the server ip?".to_string()),
        ];
        let out = assemble_context_enabled(&turns, "what is the server ip?", 200)
            .ok_or_else(|| "a small multi-turn history must assemble".to_string())?;
        if !out.contains("10.0.4.12") {
            return Err(format!("working set must carry the queried fact: {out}"));
        }
        if !out.contains("QUESTION") {
            return Err(format!("assembled prompt keeps the question frame: {out}"));
        }
        Ok(())
    }
}

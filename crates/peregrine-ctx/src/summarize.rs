//! L1 — chunk summaries.
//!
//! Deliberately extractive. An abstractive summary is a second place a
//! hallucination can enter, and the ladder's whole safety story is that L1 is a
//! lossy *view* of L0 that can always be escalated back to exact bytes.
//! Extractive summaries keep that property: every character of L1 exists
//! verbatim in L0.

use std::collections::HashMap;

use crate::text::{content_tokens, sentence_ranges};

#[derive(Debug, Clone)]
pub struct ExtractiveSummarizer {
    pub max_chars: usize,
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self { max_chars: 200 }
    }
}

impl ExtractiveSummarizer {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }

    /// Pick the highest term-weight sentences, capped at `max_chars`.
    pub fn summarize(&self, text: &str) -> String {
        let text = text.trim();
        if text.chars().count() <= self.max_chars {
            return text.to_string();
        }
        let sentences: Vec<&str> = sentence_ranges(text)
            .into_iter()
            .map(|(s, e)| text[s..e].trim())
            .filter(|s| !s.is_empty())
            .collect();
        if sentences.is_empty() {
            return text.chars().take(self.max_chars).collect();
        }

        let mut freq: HashMap<String, u32> = HashMap::new();
        for token in content_tokens(text) {
            *freq.entry(token).or_insert(0) += 1;
        }

        let mut scored: Vec<(f32, usize, &str)> = Vec::new();
        for (i, sentence) in sentences.iter().enumerate() {
            let tokens = content_tokens(sentence);
            if tokens.is_empty() {
                continue;
            }
            let weight: u32 = tokens
                .iter()
                .map(|t| freq.get(t).copied().unwrap_or(0))
                .sum();
            let mut score = weight as f32 / (tokens.len() as f32).sqrt();
            // Leading sentences carry disproportionate signal in logs and prose.
            if i == 0 {
                score *= 1.3;
            }
            scored.push((score, i, sentence));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut chosen: Vec<(usize, &str)> = Vec::new();
        let mut used = 0usize;
        for (_score, i, sentence) in scored {
            let len = sentence.chars().count();
            if used + len > self.max_chars && !chosen.is_empty() {
                break;
            }
            chosen.push((i, sentence));
            used += len + 1;
        }
        chosen.sort_by_key(|(i, _)| *i);
        let joined = chosen
            .into_iter()
            .map(|(_, s)| s)
            .collect::<Vec<_>>()
            .join(" ");
        joined
            .chars()
            .take(self.max_chars)
            .collect::<String>()
            .trim_end()
            .to_string()
    }
}

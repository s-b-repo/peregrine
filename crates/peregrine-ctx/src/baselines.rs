//! Context assemblies to lose to.
//!
//! The bundled benchmark compares DCR against full context and a sliding
//! window. Both are weak in the same direction — they are *truncations* — so
//! beating them says little about whether the machinery earns its complexity.
//! The audit's demand was a benchmark capable of proving DCR loses, and that
//! needs baselines that also retrieve:
//!
//! | baseline | what it is | what it costs DCR to beat |
//! |---|---|---|
//! | `Rag` | top-k raw chunks by the same hybrid index | the ladder and the graph must beat plain similarity |
//! | `SummarizeAll` | L1 of everything, truncated to budget | compression alone must be insufficient |
//! | `Recursive` | chunk → per-chunk answer → reduce | the RLM shape must be beatable at equal budget |
//!
//! Every baseline is fed the same corpus and read by the same deterministic
//! reasoner as DCR, so the comparison remains one between *context assemblies*
//! rather than between models. They are deliberately competent: `Rag` uses the
//! crate's real index rather than a strawman, and `Recursive` gets to see every
//! chunk, which is more raw material than DCR ever admits.
//!
//! What these still cannot show: latency against a real model, or quality under
//! a reasoner that can paraphrase. A line-matcher rewards exact overlap, which
//! flatters retrieval baselines on lookup probes and punishes them on
//! multi-hop ones.

use crate::embed::embed;
use crate::index::{HybridIndex, Namespace};
use crate::llm::Reasoner;
use crate::summarize::ExtractiveSummarizer;
use crate::text::content_tokens;
use crate::tokens::estimate_tokens;

/// Split a corpus into retrievable chunks — one per document, which is the
/// unit the corpus generator writes and the most favourable split for RAG.
pub fn chunk(docs: &[(String, String)]) -> Vec<String> {
    docs.iter().map(|(_, text)| text.clone()).collect()
}

/// Top-k chunk retrieval with no graph, no ladder and no budget solver.
///
/// This is ordinary RAG: embed the query, take the best chunks, concatenate
/// until the budget is spent. It uses the same [`HybridIndex`] DCR uses, so any
/// difference in the results is the *architecture* rather than the retriever.
pub struct Rag {
    index: HybridIndex,
    chunks: Vec<String>,
    budget: usize,
}

impl Rag {
    pub fn new(docs: &[(String, String)], budget: usize) -> Rag {
        let chunks = chunk(docs);
        let mut index = HybridIndex::default();
        for (idx, text) in chunks.iter().enumerate() {
            index.add_span(idx, text);
            // Eager vectors: a RAG baseline that skipped them would be a
            // strawman, since eager embedding is exactly what RAG does.
            index.add_span_vector(idx, text);
        }
        Rag {
            index,
            chunks,
            budget,
        }
    }

    /// The assembled context for a query, and its token cost.
    pub fn assemble(&self, query: &str) -> (String, usize) {
        let query_vec = embed(query);
        let hits = self
            .index
            .search(Namespace::Span, query, &query_vec, 32);
        let mut kept: Vec<&str> = Vec::new();
        let mut used = 0usize;
        for (idx, _) in hits {
            let Some(text) = self.chunks.get(idx) else {
                continue;
            };
            let cost = estimate_tokens(text);
            if used + cost > self.budget {
                continue;
            }
            kept.push(text);
            used += cost;
        }
        (kept.join("\n\n"), used)
    }
}

/// Summarise everything, then truncate to the budget.
///
/// The honest form of "just compress it": every document reduced to its most
/// informative sentences, newest first so a correction survives the cut. It has
/// the same information *coverage* as full context and the same token cost as
/// DCR, which makes it the sharpest test of whether a ladder plus a planner
/// beats uniform compression.
pub struct SummarizeAll {
    summaries: Vec<String>,
    budget: usize,
}

impl SummarizeAll {
    pub fn new(docs: &[(String, String)], budget: usize) -> SummarizeAll {
        // The same extractive summariser the ladder uses at L1, so this is
        // DCR's own compression applied uniformly instead of selectively.
        let summarizer = ExtractiveSummarizer::new(120);
        let summaries = docs
            .iter()
            .map(|(_, text)| summarizer.summarize(text))
            .filter(|s| !s.trim().is_empty())
            .collect();
        SummarizeAll { summaries, budget }
    }

    pub fn assemble(&self, _query: &str) -> (String, usize) {
        let mut kept: Vec<&str> = Vec::new();
        let mut used = 0usize;
        // Newest first: under a budget, the last thing said about a fact is
        // the one worth keeping.
        for summary in self.summaries.iter().rev() {
            let cost = estimate_tokens(summary);
            if used + cost > self.budget {
                break;
            }
            kept.push(summary);
            used += cost;
        }
        kept.reverse();
        (kept.join("\n"), used)
    }
}

/// The RLM shape: map the question over every chunk, then reduce.
///
/// Each chunk is read separately by the same reasoner, the non-empty answers
/// are collected, and the reduction step answers from that digest. It sees
/// *all* the history — no truncation anywhere — which is the property RLMs
/// claim and the reason it is worth measuring against.
///
/// Its cost is charged honestly: every chunk read counts, so the token column
/// reflects what recursion actually spends rather than only the final prompt.
pub struct Recursive {
    chunks: Vec<String>,
    /// Cap on the per-chunk answers carried into the reduction.
    fanin: usize,
}

impl Recursive {
    pub fn new(docs: &[(String, String)], fanin: usize) -> Recursive {
        Recursive {
            chunks: chunk(docs),
            fanin,
        }
    }

    /// Returns the answer and the total tokens read across every level.
    pub fn answer(&self, query: &str, reasoner: &mut dyn Reasoner) -> (String, usize) {
        let mut spent = 0usize;
        let mut found: Vec<String> = Vec::new();

        for text in &self.chunks {
            let prompt = format!("{text}\n\nQUESTION: {query}\n");
            spent += estimate_tokens(&prompt);
            let answer = reasoner.complete(&prompt, "");
            if answer.trim().is_empty() || answer.starts_with("I don't have") {
                continue;
            }
            found.push(answer);
        }

        // Keep the most recent hits: later material corrects earlier material,
        // and an unbounded reduction prompt would defeat the point.
        if found.len() > self.fanin {
            found = found.split_off(found.len() - self.fanin);
        }
        let digest = found.join("\n");
        let prompt = format!("{digest}\n\nQUESTION: {query}\n");
        spent += estimate_tokens(&prompt);
        (reasoner.complete(&prompt, ""), spent)
    }
}

/// Does an answer contain what was expected?
///
/// Retained for callers scoring a bare recall check. The benchmark itself goes
/// through [`crate::bench::Probe::scores`], which also handles refusal probes —
/// a probe passed by *not* answering cannot be scored by containment alone.
pub fn hit(answer: &str, expected: &str) -> bool {
    answer.to_lowercase().contains(&expected.to_lowercase())
}

/// Overlap between a query and a candidate line — exposed so a baseline can be
/// checked against the reasoner's own notion of relevance.
pub fn overlap(query: &str, text: &str) -> f32 {
    let q: Vec<String> = content_tokens(query);
    if q.is_empty() {
        return 0.0;
    }
    let t: Vec<String> = content_tokens(text);
    let matched = q.iter().filter(|token| t.contains(token)).count();
    matched as f32 / q.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{BaselineReasoner, build_corpus};

    #[test]
    fn rag_retrieves_within_its_budget() {
        let corpus = build_corpus(60);
        let rag = Rag::new(&corpus.docs, 400);
        let (context, tokens) = rag.assemble("what is the server ip?");
        assert!(tokens <= 400, "rag exceeded its budget: {tokens}");
        assert!(!context.is_empty());
    }

    #[test]
    fn summarize_all_stays_within_its_budget() {
        let corpus = build_corpus(60);
        let summarizer = SummarizeAll::new(&corpus.docs, 400);
        let (_, tokens) = summarizer.assemble("what is the server ip?");
        assert!(tokens <= 400, "summarize-all exceeded its budget: {tokens}");
    }

    /// The recursive baseline reads everything, so its cost must exceed the
    /// bounded ones by a wide margin — that is the trade it makes.
    #[test]
    fn recursive_charges_for_every_chunk_it_reads() {
        let corpus = build_corpus(60);
        let recursive = Recursive::new(&corpus.docs, 8);
        let mut reasoner = BaselineReasoner;
        let (_, spent) = recursive.answer("what is the server ip?", &mut reasoner);
        assert!(
            spent > estimate_tokens(&corpus.text()),
            "recursion must charge for reading the whole corpus"
        );
    }
}

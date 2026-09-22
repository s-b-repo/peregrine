//! L0 — the immutable raw store.
//!
//! Every byte that enters the runtime is addressed by a stable span id before
//! anything else happens. Spans are the ground truth every claim, calculation
//! and decision must eventually point back at; nothing here is ever mutated or
//! deleted. Corrections arrive as new spans plus a `supersedes` edge in the
//! graph, which is what keeps the transcript a valid audit log.

use std::collections::HashMap;

use crate::ids::{Clock, make_id};
use crate::text::{paragraph_ranges, sentence_ranges};

pub type SpanIdx = usize;

/// An immutable, addressable byte range in a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub id: String,
    pub doc_id: String,
    pub start: usize,
    pub end: usize,
    pub seq: usize,
    pub ts: u64,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub ts: u64,
    pub source: Option<String>,
}

#[derive(Debug)]
pub enum StoreError {
    /// L0 is immutable: a document id cannot be rebound to different content.
    Immutable(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Immutable(id) => write!(
                f,
                "document {id} already exists with different content; L0 is immutable \
                 — ingest a new document and supersede"
            ),
        }
    }
}

impl std::error::Error for StoreError {}

/// Append-only store of documents and the spans addressing them.
#[derive(Debug)]
pub struct RawStore {
    pub max_span_chars: usize,
    documents: Vec<Document>,
    doc_index: HashMap<String, usize>,
    spans: Vec<Span>,
    span_index: HashMap<String, SpanIdx>,
    doc_spans: Vec<Vec<SpanIdx>>,
}

impl Default for RawStore {
    fn default() -> Self {
        Self::new(600)
    }
}

impl RawStore {
    pub fn new(max_span_chars: usize) -> Self {
        Self {
            max_span_chars,
            documents: Vec::new(),
            doc_index: HashMap::new(),
            spans: Vec::new(),
            span_index: HashMap::new(),
            doc_spans: Vec::new(),
        }
    }

    // -- ingestion ---------------------------------------------------------
    pub fn add_document(
        &mut self,
        text: &str,
        doc_id: Option<&str>,
        source: Option<String>,
        clock: &Clock,
    ) -> Result<Vec<SpanIdx>, StoreError> {
        let doc_id = match doc_id {
            Some(id) => id.to_string(),
            None => make_id("doc", &[text]),
        };
        if let Some(&existing) = self.doc_index.get(&doc_id) {
            // Append-only: re-ingesting identical material is a no-op, and a
            // *different* body under a used id is a programming error rather
            // than an edit.
            if self.documents[existing].text != text {
                return Err(StoreError::Immutable(doc_id));
            }
            return Ok(self.doc_spans[existing].clone());
        }

        let ts = clock.tick();
        let doc_pos = self.documents.len();
        self.documents.push(Document {
            id: doc_id.clone(),
            text: text.to_string(),
            ts,
            source,
        });
        self.doc_index.insert(doc_id.clone(), doc_pos);

        let mut ids = Vec::new();
        for (seq, (start, end)) in self.chunk(text).into_iter().enumerate() {
            let span_id = make_id("s", &[&doc_id, &start.to_string(), &end.to_string()]);
            let idx = self.spans.len();
            self.spans.push(Span {
                id: span_id.clone(),
                doc_id: doc_id.clone(),
                start,
                end,
                seq,
                ts,
            });
            self.span_index.insert(span_id, idx);
            ids.push(idx);
        }
        self.doc_spans.push(ids.clone());
        Ok(ids)
    }

    /// Paragraph-ish spans, hard-wrapped at sentence boundaries.
    fn chunk(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (para_start, para_end) in paragraph_ranges(text) {
            let block = &text[para_start..para_end];
            if block.chars().count() <= self.max_span_chars {
                if !block.trim().is_empty() {
                    out.push((para_start, para_end));
                }
                continue;
            }
            let mut buf_start = 0usize;
            let mut buf_len = 0usize;
            for (s, e) in sentence_ranges(block) {
                let piece_len = block[s..e].chars().count();
                if buf_len + piece_len > self.max_span_chars && s > buf_start {
                    out.push((para_start + buf_start, para_start + s));
                    buf_start = s;
                    buf_len = 0;
                }
                buf_len += piece_len + 1;
            }
            if buf_start < block.len() {
                out.push((para_start + buf_start, para_end));
            }
        }
        out
    }

    // -- reads -------------------------------------------------------------
    pub fn span(&self, idx: SpanIdx) -> &Span {
        &self.spans[idx]
    }

    pub fn span_by_id(&self, span_id: &str) -> Option<&Span> {
        self.span_index.get(span_id).map(|&i| &self.spans[i])
    }

    pub fn has_span(&self, span_id: &str) -> bool {
        self.span_index.contains_key(span_id)
    }

    pub fn span_idx(&self, span_id: &str) -> Option<SpanIdx> {
        self.span_index.get(span_id).copied()
    }

    pub fn text_of(&self, span_id: &str) -> &str {
        match self.span_by_id(span_id) {
            Some(span) => {
                self.documents[self.doc_index[&span.doc_id]].text[span.start..span.end].trim()
            }
            None => "",
        }
    }

    /// Concatenated text of several spans, newline-joined.
    pub fn text_of_many(&self, span_ids: &[String]) -> String {
        span_ids
            .iter()
            .map(|id| self.text_of(id))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Adjacent spans — used when an exact quote needs its surroundings.
    pub fn neighbours(&self, span_id: &str, radius: usize) -> Vec<&Span> {
        let Some(span) = self.span_by_id(span_id) else {
            return Vec::new();
        };
        let siblings = &self.doc_spans[self.doc_index[&span.doc_id]];
        let lo = span.seq.saturating_sub(radius);
        let hi = (span.seq + radius + 1).min(siblings.len());
        siblings[lo..hi].iter().map(|&i| &self.spans[i]).collect()
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }

    pub fn spans_for(&self, doc_id: &str) -> Vec<&Span> {
        match self.doc_index.get(doc_id) {
            Some(&i) => self.doc_spans[i].iter().map(|&s| &self.spans[s]).collect(),
            None => Vec::new(),
        }
    }

    pub fn total_chars(&self) -> usize {
        self.documents.iter().map(|d| d.text.len()).sum()
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Rebuild from persisted state without re-chunking.
    pub fn restore(&mut self, documents: Vec<Document>, spans: Vec<Span>) {
        self.documents = documents;
        self.doc_index = self
            .documents
            .iter()
            .enumerate()
            .map(|(i, d)| (d.id.clone(), i))
            .collect();
        self.doc_spans = vec![Vec::new(); self.documents.len()];
        self.spans = spans;
        self.span_index = self
            .spans
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.clone(), i))
            .collect();
        let mut ordered: Vec<(usize, usize, usize)> = self
            .spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.doc_index.get(&s.doc_id).map(|&d| (d, s.seq, i)))
            .collect();
        ordered.sort_by_key(|(d, seq, _)| (*d, *seq));
        for (doc, _seq, idx) in ordered {
            self.doc_spans[doc].push(idx);
        }
    }
}

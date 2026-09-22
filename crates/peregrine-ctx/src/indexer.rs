//! State Indexer — the entry point.
//!
//! ```text
//! index(event) -> { spans, nodes, stale }
//! ```
//!
//! Ingestion is where the runtime earns the right to throw the transcript away.
//! Three things happen, in this order:
//!
//! 1. **Span addressing (L0), eagerly.** Everything else is lazy.
//! 2. **Node extraction.** Candidate claims/decisions/goals/constraints, each
//!    carrying the spans it came from.
//! 3. **Contradiction detection.** A new claim that disagrees with a live claim
//!    on the same key is *not* silently written over the old one — both are
//!    kept, linked by `contradicts`, and (normally) the newer supersedes the
//!    older, which marks dependents stale.
//!
//! Step 3 is the part that actually fixes context rot rather than hiding it:
//! the stale set is what stops a corrected fact from living on in the cache.

use std::collections::HashMap;

use crate::graph::{DcrError, MemoryGraph};
use crate::ids::Clock;
use crate::index::HybridIndex;
use crate::ladder::Ladder;
use crate::nodes::{EdgeType, Kind, NewNode, Node, NodeIdx, NodeMeta, Status};
use crate::spans::{RawStore, SpanIdx};
use crate::text::{contains_any, is_word_char, strip_quotes};

/// A node the extractor proposes. Nothing here is admitted until it has been
/// grounded against a span.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub kind: Kind,
    pub value: String,
    pub key: Option<String>,
    pub confidence: f32,
    pub corrective: bool,
}

#[derive(Debug, Default)]
pub struct IndexResult {
    pub spans: Vec<SpanIdx>,
    pub nodes: Vec<NodeIdx>,
    pub stale: Vec<NodeIdx>,
    pub contradictions: Vec<(String, String)>,
    pub superseded: Vec<String>,
}

impl IndexResult {
    pub fn summary(&self) -> String {
        format!(
            "{} spans, {} nodes, {} contradictions, {} stale",
            self.spans.len(),
            self.nodes.len(),
            self.contradictions.len(),
            self.stale.len()
        )
    }
}

const NOISE_KEYS: &[&str] = &[
    "it",
    "this",
    "that",
    "there",
    "here",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "the",
    "a",
    "an",
    "note",
    "reply",
    "answer",
    "question",
    "thing",
    "something",
    "anything",
    "everyone",
    "someone",
    "i",
    "we",
    "you",
    "they",
    "he",
    "she",
    "and",
    "but",
    "so",
    "then",
    "goal",
    "objective",
    "aim",
    "target",
    "decision",
    "decided",
    "action",
    "resolved",
    "correction",
    "update",
    "constraint",
    "requirement",
    "summary",
    "recap",
    "tldr",
];

const CORRECTION_MARKERS: &[&str] = &[
    "correction",
    "actually",
    "update",
    "revised",
    "no longer",
    "instead",
    "supersedes",
    "scratch that",
];

const GOAL_PREFIXES: &[&str] = &["goal", "objective", "aim", "target"];
const DECISION_PREFIXES: &[&str] = &[
    "decision",
    "decided",
    "we will",
    "we decided to",
    "action",
    "resolved",
];
const CONSTRAINT_PREFIXES: &[&str] = &[
    "constraint",
    "requirement",
    "must",
    "never",
    "do not",
    "don't",
    "limit",
];

/// Left-boundary words: an assignment may start after one of these.
const BOUNDARY_WORDS: &[&str] = &["the", "actually", "now", "current"];
/// Words that end a value clause.
const CLAUSE_WORDS: &[&str] = &["and", "but"];
/// Copulas that may sit inside a key phrase ("the error was: …").
const COPULA_SEPARATORS: &[&str] = &["is", "was", "are", "were"];

/// Multi-word separators that assign a new value to a subject the same way a
/// copula does: `the datastore has moved to postgres-15`.
///
/// Corrections in the wild carry verbs, and a copula-only extractor silently
/// drops most of them — it read "has moved to" as prose and produced no fact at
/// all. Each phrase here has an unambiguous subject before it and its value
/// after it, which is the same shape `=` and `is` already have. Phrasings whose
/// subject follows the verb ("we switched X to Y") are deliberately NOT here:
/// they need a different rule, and guessing at them would over-extract.
const TRANSITION_SEPARATORS: &[&str] = &[
    "has moved to",
    "have moved to",
    "has changed to",
    "have changed to",
    "was replaced by",
    "were replaced by",
    "is replaced by",
    "was changed to",
    "moved to",
    "changed to",
    "should be",
];

/// Deterministic, model-free node extraction.
///
/// Deliberately conservative: it proposes state only for shapes that carry an
/// unambiguous subject (`key = value`, `Decision: …`). Over-extraction is not a
/// neutral failure — a wrongly cached claim is worse than a long prompt — so
/// anything it is unsure about stays as raw L0 and gets found by search.
#[derive(Debug, Clone)]
pub struct HeuristicExtractor {
    pub min_confidence: f32,
    pub max_value_chars: usize,
}

impl Default for HeuristicExtractor {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            max_value_chars: 120,
        }
    }
}

impl HeuristicExtractor {
    pub fn extract(&self, span_text: &str) -> Vec<Proposal> {
        let corrective = contains_any(&span_text.to_lowercase(), CORRECTION_MARKERS);
        let mut found: Vec<Proposal> = Vec::new();

        for line in span_text.lines() {
            let line = strip_bullet(line);
            if let Some(value) = prefixed_value(line, GOAL_PREFIXES) {
                found.push(self.make(Kind::Goal, value, None, 0.9, corrective));
            }
            if let Some(value) = prefixed_value(line, CONSTRAINT_PREFIXES) {
                found.push(self.make(Kind::Constraint, value, None, 0.85, corrective));
            }
            if let Some(value) = prefixed_value(line, DECISION_PREFIXES) {
                found.push(self.make(Kind::Decision, value, None, 0.85, corrective));
            }
            let trimmed = line.trim();
            let chars = trimmed.chars().count();
            if trimmed.ends_with('?') && (8..=160).contains(&chars) {
                found.push(self.make(Kind::OpenQuestion, trimmed, None, 0.7, corrective));
            }
        }

        let mut seen: Vec<(String, String)> = Vec::new();
        for (key, value) in self.assignments(span_text, 0) {
            if seen.contains(&(key.clone(), value.clone())) {
                continue;
            }
            seen.push((key.clone(), value.clone()));
            let confidence = if corrective { 0.9 } else { 0.8 };
            found.push(self.make(Kind::Claim, &value, Some(key), confidence, corrective));
        }

        found.retain(|p| p.confidence >= self.min_confidence && !p.value.is_empty());
        found
    }

    fn make(
        &self,
        kind: Kind,
        value: &str,
        key: Option<String>,
        confidence: f32,
        corrective: bool,
    ) -> Proposal {
        Proposal {
            kind,
            value: value.trim().trim_end_matches('.').trim().to_string(),
            key,
            confidence,
            corrective,
        }
    }

    /// Key/value pairs, descending once into structural prefixes.
    ///
    /// "Correction: the server ip is 10.0.9.7" parses first as
    /// `correction = the server ip is 10.0.9.7`. That key is noise, but its
    /// *value* holds the real assignment — so noise matches are re-scanned
    /// instead of discarded, which is how corrections get extracted at all.
    fn assignments(&self, text: &str, depth: usize) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut cursor = 0usize;
        while let Some((sep_start, sep_end)) = next_separator(text, cursor) {
            cursor = sep_end.max(sep_start + 1);
            // A separator must be followed by whitespace, then a value.
            let after_ws = skip_ws(text, sep_end);
            if after_ws == sep_end || after_ws >= text.len() {
                continue;
            }
            let Some((key_start, key_raw)) = key_before(text, sep_start) else {
                continue;
            };
            if !is_left_boundary(text, key_start) {
                continue;
            }
            let Some(value_end) = clause_end(text, after_ws, self.max_value_chars) else {
                continue;
            };
            let raw_value = text[after_ws..value_end].trim_end();
            if raw_value.is_empty() {
                continue;
            }
            let key = normalise_key(&key_raw);
            let mut value = clean_value(raw_value).to_string();
            let mut consumed = value_end;
            // Applies at every depth: corrections reach here at depth 1, after
            // descending through the "Correction:" prefix, which is exactly the
            // case this rule exists for.
            if let Some((restated, end)) =
                restatement_after(text, value_end, self.max_value_chars)
            {
                value = restated;
                consumed = end;
            }
            if key.is_empty() || value.is_empty() {
                continue;
            }
            if NOISE_KEYS.contains(&key.as_str()) {
                if depth == 0 {
                    // `clause_end` stops at the conjunction, so a structural
                    // prefix ("Correction: ...") would hand the inner parse a
                    // slice truncated before a coordinated restatement — it
                    // would see "was migrated" and never "is now postgres-15".
                    // Widen the slice to whatever the restatement consumes.
                    let end = restatement_after(text, value_end, self.max_value_chars)
                        .map(|(_, e)| e)
                        .unwrap_or(value_end);
                    out.extend(self.assignments(text[after_ws..end].trim_end(), depth + 1));
                }
                continue;
            }
            out.push((key, value));
            cursor = consumed.max(cursor);
        }
        out
    }
}

fn strip_bullet(line: &str) -> &str {
    let trimmed = line.trim_start();
    for bullet in ["- ", "* ", "\u{2022} ", "-", "*", "\u{2022}"] {
        if let Some(rest) = trimmed.strip_prefix(bullet) {
            return rest.trim_start();
        }
    }
    trimmed
}

/// `Goal: restore checkout` → `restore checkout`, for any of `prefixes`.
fn prefixed_value<'a>(line: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    let lower = line.to_lowercase();
    for prefix in prefixes {
        if !lower.starts_with(prefix) {
            continue;
        }
        let rest = &line[prefix.len()..];
        let boundary_ok = rest
            .chars()
            .next()
            .is_none_or(|c| c == ':' || c.is_whitespace());
        if !boundary_ok {
            continue;
        }
        let value = rest.trim_start_matches([':', ' ', '\t']).trim();
        let chars = value.chars().count();
        if (3..=160).contains(&chars) {
            return Some(value);
        }
    }
    None
}

/// Next `=`, `:` or copula that could act as a key/value separator.
fn next_separator(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b'=' | b':' => return Some((i, i + 1)),
            b if b.is_ascii_alphabetic() => {
                let start = i;
                let mut end = i;
                while end < bytes.len() && is_word_char(bytes[end] as char) {
                    end += 1;
                }
                let word = text[start..end].to_lowercase();
                let boundary = start == 0 || !is_word_char(bytes[start - 1] as char);
                if boundary {
                    // Longest match first, so "was replaced by" is not read as
                    // the copula "was" with "replaced by ..." as its value.
                    for phrase in TRANSITION_SEPARATORS {
                        let stop = start + phrase.len();
                        if stop <= text.len()
                            && text.is_char_boundary(stop)
                            && text[start..stop].eq_ignore_ascii_case(phrase)
                            && text
                                .as_bytes()
                                .get(stop)
                                .is_none_or(|b| !is_word_char(*b as char))
                        {
                            return Some((start, stop));
                        }
                    }
                    if COPULA_SEPARATORS.contains(&word.as_str()) {
                        return Some((start, end));
                    }
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn skip_ws(text: &str, from: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() && bytes[i] != b'\n' {
        i += 1;
    }
    i
}

/// Up to three words immediately left of `sep_start`, shortest first.
///
/// Mirrors the non-greedy key of the original pattern: prefer the tightest
/// subject that still sits behind a valid left boundary.
fn key_before(text: &str, sep_start: usize) -> Option<(usize, String)> {
    let bytes = text.as_bytes();
    let mut end = sep_start;
    while end > 0 && (bytes[end - 1] as char).is_whitespace() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut starts: Vec<usize> = Vec::new();
    let mut cursor = end;
    for _ in 0..3 {
        let word_end = cursor;
        while cursor > 0 && (is_word_char(bytes[cursor - 1] as char) || bytes[cursor - 1] == b'-') {
            cursor -= 1;
        }
        if cursor == word_end {
            break;
        }
        if !(bytes[cursor] as char).is_ascii_alphabetic() {
            break;
        }
        starts.push(cursor);
        // Words in a key phrase are joined by a single space or a dot.
        if cursor == 0 || !matches!(bytes[cursor - 1], b' ' | b'.') {
            break;
        }
        cursor -= 1;
    }
    for start in starts {
        if is_left_boundary(text, start) {
            return Some((start, text[start..end].to_string()));
        }
    }
    None
}

/// Start of text/line, an opening delimiter, or one of the boundary words.
fn is_left_boundary(text: &str, key_start: usize) -> bool {
    if key_start == 0 {
        return true;
    }
    let head = &text[..key_start];
    let trimmed = head.trim_end_matches([' ', '\t']);
    if trimmed.is_empty() {
        return true;
    }
    match trimmed.chars().next_back() {
        Some('\n') | Some(',') | Some(';') | Some('(') => return true,
        _ => {}
    }
    if head.len() == trimmed.len() {
        // No whitespace between the previous word and the key.
        return false;
    }
    let last_word: String = trimmed
        .chars()
        .rev()
        .take_while(|c| is_word_char(*c))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    BOUNDARY_WORDS.contains(&last_word.to_lowercase().as_str())
}

/// End of the value clause: `,` `;` newline, ` and `/` but `, `. ` or EOF.
///
/// A value ends at a clause boundary, not at the first period — otherwise every
/// dotted identifier (10.0.4.12, api.v2.stats) is truncated.
fn clause_end(text: &str, from: usize, max_chars: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = from;
    let mut chars = 0usize;
    while i < bytes.len() {
        if chars > max_chars {
            return Some(i);
        }
        match bytes[i] {
            b',' | b';' | b'\n' => return Some(i),
            b'.' if i + 1 >= bytes.len() || (bytes[i + 1] as char).is_whitespace() => {
                return Some(i);
            }
            b' ' | b'\t' => {
                let word_start = i + 1;
                let mut word_end = word_start;
                while word_end < bytes.len() && is_word_char(bytes[word_end] as char) {
                    word_end += 1;
                }
                let word = text[word_start..word_end].to_lowercase();
                if CLAUSE_WORDS.contains(&word.as_str()) {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
        chars += 1;
    }
    Some(text.len())
}

/// Temporal adverbs that front a value without being part of it:
/// `is now postgres-15` carries the value `postgres-15`.
const LEADING_ADVERBS: &[&str] = &["now", "currently", "presently", "already"];

fn strip_leading_adverb(value: &str) -> &str {
    let trimmed = value.trim_start();
    for adv in LEADING_ADVERBS {
        if trimmed.len() > adv.len()
            && trimmed.is_char_boundary(adv.len())
            && trimmed[..adv.len()].eq_ignore_ascii_case(adv)
            && trimmed.as_bytes()[adv.len()].is_ascii_whitespace()
        {
            let rest = trimmed[adv.len()..].trim_start();
            if !rest.is_empty() {
                return rest;
            }
        }
    }
    trimmed
}

fn matches_word(text: &str, at: usize, word: &str) -> bool {
    let end = at + word.len();
    if end > text.len() || !text.is_char_boundary(at) || !text.is_char_boundary(end) {
        return false;
    }
    if !text[at..end].eq_ignore_ascii_case(word) {
        return false;
    }
    text.as_bytes()
        .get(end)
        .is_none_or(|b| !is_word_char(*b as char))
}

/// A coordinated clause that restates the subject's *current* value:
/// `the datastore was migrated and is now postgres-15`.
///
/// The subject carries across the conjunction, so the later value is the live
/// one and the earlier participle ("migrated") is not a value at all. This is
/// deliberately tight — the copula must be present tense *and* followed by
/// `now` — because "X is A and is B" is a real ambiguity and over-extraction is
/// not a neutral failure here.
fn restatement_after(text: &str, from: usize, max_chars: usize) -> Option<(String, usize)> {
    let mut i = skip_ws(text, from);
    if text.as_bytes().get(i).is_some_and(|b| *b == b',' || *b == b';') {
        i = skip_ws(text, i + 1);
    }
    for conj in ["and", "but", "then", "so"] {
        if matches_word(text, i, conj) {
            i = skip_ws(text, i + conj.len());
            break;
        }
    }
    if matches_word(text, i, "it") {
        i = skip_ws(text, i + 2);
    }
    let copula = ["is", "are"].into_iter().find(|c| matches_word(text, i, c))?;
    i = skip_ws(text, i + copula.len());
    if !matches_word(text, i, "now") {
        return None;
    }
    i = skip_ws(text, i + 3);
    let end = clause_end(text, i, max_chars)?;
    let value = clean_value(text[i..end].trim_end()).to_string();
    (!value.is_empty()).then_some((value, end))
}

fn normalise_key(key: &str) -> String {
    let mut key = key.trim().to_lowercase();
    key = key.split_whitespace().collect::<Vec<_>>().join(".");
    for article in [
        "the.", "a.", "an.", "our.", "its.", "his.", "her.", "their.",
    ] {
        if let Some(rest) = key.strip_prefix(article) {
            key = rest.to_string();
            break;
        }
    }
    // "the error was:" parses with the copula inside the key phrase; drop it so
    // `error.was` and `error` are not two different facts.
    for copula in [".was", ".is", ".are", ".were", ".will.be"] {
        if let Some(rest) = key.strip_suffix(copula) {
            key = rest.to_string();
            break;
        }
    }
    key.trim_matches('.').to_string()
}

/// A quoted value is the value: `error was "connection refused" on 8080`.
fn clean_value(value: &str) -> &str {
    let value = value.trim();
    for (open, close) in [('"', '"'), ('\u{201c}', '\u{201d}')] {
        if value.starts_with(open) {
            let rest = &value[open.len_utf8()..];
            if let Some(end) = rest.find(close) {
                let inner = rest[..end].trim();
                if !inner.is_empty() && inner.chars().count() <= 120 {
                    return inner;
                }
            }
        }
    }
    strip_leading_adverb(strip_quotes(value))
}

/// Everything the indexer needs to write into, borrowed for one ingest.
pub struct IngestCtx<'a> {
    pub raw: &'a mut RawStore,
    pub graph: &'a mut MemoryGraph,
    pub index: &'a mut HybridIndex,
    pub ladder: &'a Ladder,
    pub clock: &'a Clock,
}

#[derive(Debug)]
pub struct StateIndexer {
    pub extractor: HeuristicExtractor,
    pub supersede_on_conflict: bool,
    /// Link new nodes to the live facts they mention by value. Off, the graph
    /// is a star and `explain()` stops one hop down.
    pub reference_linking: bool,
    pub eager_span_vectors: bool,
    /// span id → evidence node id. Evidence nodes are created lazily, one per
    /// referenced span.
    pub evidence_by_span: HashMap<String, String>,
}

impl Default for StateIndexer {
    fn default() -> Self {
        Self {
            extractor: HeuristicExtractor::default(),
            supersede_on_conflict: true,
            reference_linking: true,
            eager_span_vectors: false,
            evidence_by_span: HashMap::new(),
        }
    }
}

impl StateIndexer {
    pub fn ingest(
        &mut self,
        ctx: &mut IngestCtx<'_>,
        text: &str,
        doc_id: Option<&str>,
        source: Option<String>,
    ) -> Result<IndexResult, DcrError> {
        let mut result = IndexResult::default();
        let spans = ctx.raw.add_document(text, doc_id, source, ctx.clock)?;
        result.spans = spans.clone();
        for span_idx in spans {
            let span_id = ctx.raw.span(span_idx).id.clone();
            let span_text = ctx.raw.text_of(&span_id).to_string();
            if span_text.trim().is_empty() {
                continue;
            }
            ctx.index.add_span(span_idx, &span_text);
            if self.eager_span_vectors {
                ctx.index.add_span_vector(span_idx, &span_text);
            }
            for proposal in self.extractor.extract(&span_text) {
                if let Some(idx) = self.admit(ctx, proposal, &span_id, &span_text, &mut result)? {
                    result.nodes.push(idx);
                }
            }
        }
        Ok(result)
    }

    fn admit(
        &mut self,
        ctx: &mut IngestCtx<'_>,
        proposal: Proposal,
        span_id: &str,
        span_text: &str,
        result: &mut IndexResult,
    ) -> Result<Option<NodeIdx>, DcrError> {
        let evidence = self.evidence_node(ctx, span_id, span_text)?;
        let evidence_id = ctx.graph.node(evidence).id.clone();
        // Ingest cost, honestly: after the live_by_key change `duplicate` /
        // `conflicts` are O(live) and no longer the dominant term. `reference_links`
        // (and `backfill_references` below) still scan every node per admitted fact.
        // A load-independent count settles the shape the noisy wall-clock could not:
        // node-visits across these two scans grow quadratically — x3.95 for a 2x
        // corpus and x8.57 for 3x, per-fact visits scaling ~linearly with N (measured
        // at 1k/3k/6k standard turns). So ingest WORK here is O(N^2) even though the
        // measured *time* exponent is not robust under concurrent load. This is the
        // larger, unsolved cost; a prior SET-NARROWING inverted-index attempt on it
        // was reverted as behaviour-changing (this index is set-preserving, which is
        // the difference — see git history). Not addressed here.
        let links = match self.reference_linking {
            true => self.reference_links(ctx.graph, &proposal.value, span_id),
            false => Vec::new(),
        };
        let mut deps = vec![evidence_id];
        deps.extend(links);
        let node = NewNode::new(proposal.kind, proposal.value.clone())
            .spans([span_id.to_string()])
            .deps(deps)
            .confidence(proposal.confidence)
            .maybe_key(proposal.key.clone())
            .meta(NodeMeta {
                corrective: proposal.corrective,
                ..Default::default()
            })
            .build();

        if ctx.graph.idx_of(&node.id).is_some() {
            // Same claim, same span — already known. Bump nothing.
            return Ok(None);
        }
        // The same fact restated elsewhere is corroboration, not a second fact.
        // Merging keeps the working set free of near-identical lines and makes
        // repetition raise confidence, which is what repetition actually means.
        if let Some(twin) = self.duplicate(ctx.graph, &node) {
            self.reinforce(ctx, twin, span_id, evidence);
            return Ok(None);
        }

        let conflicts = self.conflicts(ctx.graph, &node);
        let idx = ctx.graph.upsert(node, ctx.raw, ctx.clock)?;
        self.reindex(ctx, idx);
        if self.reference_linking {
            self.backfill_references(ctx, idx);
        }

        for other in conflicts {
            ctx.graph.contradict(idx, other);
            result.contradictions.push((
                ctx.graph.node(other).id.clone(),
                ctx.graph.node(idx).id.clone(),
            ));
            if self.supersede_on_conflict
                && may_supersede(ctx.graph.node(other), ctx.graph.node(idx))
            {
                let version = ctx.graph.version;
                ctx.graph.supersede(other, idx);
                result.superseded.push(ctx.graph.node(other).id.clone());
                result.stale.extend(ctx.graph.invalidated_since(version));
                ctx.index.remove_node(other);
            }
        }
        Ok(Some(idx))
    }

    /// A live node of the same kind asserting the same thing.
    fn duplicate(&self, graph: &MemoryGraph, node: &Node) -> Option<NodeIdx> {
        let pool = match &node.key {
            // Same non-superseded pool `by_key(key, true)` returns, but read
            // from the maintained live index — no per-call bucket copy + sort.
            // `.find` is order-independent, so no sort is needed here.
            Some(key) => graph.live_by_key(key).to_vec(),
            None => graph.by_kind(node.kind, false),
        };
        pool.into_iter().find(|&idx| {
            let other = graph.node(idx);
            other.id != node.id
                && other.status == Status::Fresh
                && other.kind == node.kind
                && other.key == node.key
                && same_value(&other.value, &node.value)
        })
    }

    fn reinforce(
        &mut self,
        ctx: &mut IngestCtx<'_>,
        idx: NodeIdx,
        span_id: &str,
        evidence: NodeIdx,
    ) {
        let evidence_id = ctx.graph.node(evidence).id.clone();
        let mut link_evidence = false;
        {
            let node = ctx.graph.node_mut(idx);
            if !node.source_spans.iter().any(|s| s == span_id) {
                node.source_spans.push(span_id.to_string());
            }
            if !node.dependencies.contains(&evidence_id) {
                node.dependencies.push(evidence_id);
                link_evidence = true;
            }
            node.confidence = (node.confidence + 0.05).min(0.97);
            node.meta.corroborations += 1;
            let mut cache = node.level_cache.borrow_mut();
            cache.l1 = None;
            cache.l2 = None;
        }
        if link_evidence {
            ctx.graph
                .add_edge(idx, evidence, crate::nodes::EdgeType::Supports);
        }
        // Deliberately not routed through `upsert`: the value did not change,
        // so dependents must not be invalidated.
        self.reindex(ctx, idx);
    }

    /// Live claims on the same key that disagree on value.
    fn conflicts(&self, graph: &MemoryGraph, node: &Node) -> Vec<NodeIdx> {
        let Some(key) = &node.key else {
            return Vec::new();
        };
        if node.kind != Kind::Claim {
            return Vec::new();
        }
        graph
            // Timestamp-sorted view: byte-identical order to by_key(key, true),
            // so which node supersedes which is unchanged — over O(live), not
            // the whole append-only bucket.
            .live_by_key_sorted(key)
            .into_iter()
            .filter(|&idx| {
                let other = graph.node(idx);
                other.id != node.id
                    && other.status != Status::Superseded
                    && other.kind == Kind::Claim
                    && !same_value(&other.value, &node.value)
            })
            .collect()
    }

    /// Link *existing* facts to a newly arrived one they already mention.
    ///
    /// [`Self::reference_links`] only looks backwards, at nodes that exist when
    /// a node arrives. A chain written the way people write one — "failures were
    /// traced to alpha-checkout", and only later "alpha-checkout is owned by
    /// team-payments" — puts the mention *before* the thing mentioned, so the
    /// backward pass can never see it and the chain is never joined. Without
    /// this the graph stays a star however good the planner is, and a probe
    /// needing two hops fails with graph expansion switched on or off, which is
    /// exactly what the multi-hop set showed.
    fn backfill_references(&self, ctx: &mut IngestCtx<'_>, new_idx: NodeIdx) {
        let new_node = ctx.graph.node(new_idx);
        if new_node.kind != Kind::Claim {
            return;
        }
        let new_id = new_node.id.clone();
        let Some(key) = new_node.key.clone() else {
            return;
        };
        let key = key.to_lowercase();
        let key_spaced = key.replace('.', " ");
        let mut needles: Vec<&str> = vec![key.as_str(), key_spaced.as_str()];
        needles.retain(|n| n.chars().count() >= 4);
        if needles.is_empty() {
            return;
        }
        let mentioners: Vec<NodeIdx> = ctx
            .graph
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, other)| {
                other.kind == Kind::Claim
                    && other.status == Status::Fresh
                    && other.id != new_id
                    && !other.dependencies.contains(&new_id)
                    && {
                        let hay = other.value.to_lowercase();
                        needles.iter().any(|n| hay.contains(n))
                    }
            })
            .map(|(i, _)| NodeIdx::from(i))
            .collect();
        for m in mentioners {
            ctx.graph.add_edge(m, new_idx, EdgeType::DerivedFrom);
            ctx.graph.node_mut(m).dependencies.push(new_id.clone());
        }
    }

    /// Link a new node to the live facts it mentions by value.
    ///
    /// Without this the graph is a star — every node hanging off its own
    /// evidence span — and `explain()` on a decision stops one hop down.
    /// Matching on the *value* of an existing claim is conservative but real.
    fn reference_links(&self, graph: &MemoryGraph, value: &str, span_id: &str) -> Vec<String> {
        let haystack = value.to_lowercase();
        let mut matches: Vec<(usize, String)> = Vec::new();
        for node in graph.nodes() {
            if node.kind != Kind::Claim || node.status != Status::Fresh || node.key.is_none() {
                continue;
            }
            if node.source_spans.iter().any(|s| s == span_id) {
                continue;
            }
            // Match the referenced node's value *or* its subject key. Only the
            // value was tried before, which meant a chain written the way people
            // actually write one — "failures were traced to alpha-checkout",
            // then "alpha-checkout is owned by team-payments" — was never
            // linked: the mention is of the second node's key, not its value.
            // That is why the ablation reported reference linking as having no
            // effect. It was not weak, it was unreachable.
            let value = node.value.to_lowercase();
            let key = node.key.clone().unwrap_or_default().to_lowercase();
            let key_spaced = key.replace('.', " ");
            let needle = [value.trim(), key.trim(), key_spaced.trim()]
                .into_iter()
                .filter(|n| n.chars().count() >= 4 && haystack.contains(n))
                // Longest wins: prefer the most specific mention.
                .max_by_key(|n| n.len());
            let Some(needle) = needle else {
                continue;
            };
            matches.push((needle.len(), node.id.clone()));
        }
        matches.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        matches.into_iter().take(3).map(|(_, id)| id).collect()
    }

    pub fn evidence_node(
        &mut self,
        ctx: &mut IngestCtx<'_>,
        span_id: &str,
        span_text: &str,
    ) -> Result<NodeIdx, DcrError> {
        if let Some(existing) = self.evidence_by_span.get(span_id) {
            if let Some(idx) = ctx.graph.idx_of(existing) {
                return Ok(idx);
            }
        }
        let node = NewNode::new(Kind::Evidence, span_text)
            .spans([span_id.to_string()])
            .build();
        let id = node.id.clone();
        let idx = match ctx.graph.idx_of(&id) {
            Some(idx) => idx,
            None => {
                let idx = ctx.graph.upsert(node, ctx.raw, ctx.clock)?;
                self.reindex(ctx, idx);
                idx
            }
        };
        self.evidence_by_span.insert(span_id.to_string(), id);
        Ok(idx)
    }

    /// Index the dotted key *and* its parts: `server.ip` is a single token to
    /// the tokenizer, so a query for "server ip" would otherwise miss it.
    pub fn reindex(&self, ctx: &mut IngestCtx<'_>, idx: NodeIdx) {
        let node = ctx.graph.node(idx);
        let key = node.key.clone().unwrap_or_default();
        let text = [
            key.clone(),
            key.replace('.', " "),
            node.value.clone(),
            node.kind.as_str().into(),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
        let vector = ctx.ladder.vector(ctx.raw, node);
        ctx.index.add_node(idx, &text, vector);
    }
}

/// Should the newer claim replace the older one, or should both stand?
///
/// Ingest order is event order, so last-writer-wins is the default. The
/// exception is an explicitly corrective statement ("Correction: …",
/// "actually …"): plain material arriving after one must not silently revert
/// it, because ingest order is only *approximately* chronology — files read
/// from a directory, transcripts merged from two sources. In that case both
/// sides stay live, linked by `contradicts`, and the model adjudicates with the
/// evidence in front of it.
pub fn may_supersede(existing: &Node, incoming: &Node) -> bool {
    !existing.meta.corrective || incoming.meta.corrective
}

fn same_value(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    norm(a) == norm(b)
}

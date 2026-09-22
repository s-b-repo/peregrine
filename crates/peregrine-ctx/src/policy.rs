//! Runtime decision policy — what to do with each piece of context.
//!
//! The wiki's decision table, made executable. The hard part it leaves open is
//! `sufficient(L, q)`: there is no clean definition of the cheapest level that
//! can answer a query. This implements the practical proxy the spec recommends
//! — route by query type, then *measure* escalation and let the escalation rate
//! be the telemetry that catches a bad router — plus a de-escalation rule so a
//! node that keeps being read at L0 without yielding anything new collapses
//! into a cached fact.

use std::collections::HashSet;

use crate::ladder::Level;
use crate::nodes::{Kind, Node, NodeIdx, Status};
use crate::planner::ActiveContext;
use crate::text::{contains_any, content_tokens};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    QuoteExact,
    ValueLookup,
    Justify,
    Recompute,
    Summarize,
    Open,
}

impl QueryType {
    pub fn as_str(self) -> &'static str {
        match self {
            QueryType::QuoteExact => "quote_exact",
            QueryType::ValueLookup => "value_lookup",
            QueryType::Justify => "justify",
            QueryType::Recompute => "recompute",
            QueryType::Summarize => "summarize",
            QueryType::Open => "open",
        }
    }

    /// Preferred admission levels, cheapest sufficient first.
    pub fn routing(self) -> [Level; 3] {
        match self {
            QueryType::QuoteExact => [Level::L0, Level::L1, Level::L2],
            QueryType::ValueLookup => [Level::L2, Level::L1, Level::L0],
            QueryType::Justify => [Level::L2, Level::L1, Level::L0],
            QueryType::Recompute => [Level::L3, Level::L2, Level::L0],
            QueryType::Summarize => [Level::L1, Level::L2, Level::L0],
            QueryType::Open => [Level::L2, Level::L1, Level::L0],
        }
    }
}

const QUOTE_MARKERS: &[&str] = &[
    "quote",
    "verbatim",
    "exact",
    "exactly",
    "word for word",
    "word-for-word",
    "literal",
    "raw text",
    "exact wording",
    "full text",
    "transcript",
    "as written",
    "stack trace",
    "exact error",
    "error message",
    "log line",
];

const RECOMPUTE_MARKERS: &[&str] = &[
    "recompute",
    "recalculate",
    "rerun",
    "re-run",
    "compute",
    "calculate",
    "total of",
    "total for",
    "sum of",
    "sum for",
    "average of",
    "cost of",
    "cost for",
    "what's the cost",
    "what is the cost",
    "what's the total",
    "what is the total",
    "what's the result",
    "what is the result",
];

const JUSTIFY_MARKERS: &[&str] = &[
    "why",
    "justify",
    "justification",
    "explain",
    "reason",
    "because",
    "on what basis",
    "how did you",
    "how did we",
    "what led to",
    "audit",
    "evidence for",
    "prove",
    "back up",
    "backing up",
];

const SUMMARIZE_MARKERS: &[&str] = &[
    "summary",
    "summarise",
    "summarize",
    "recap",
    "overview",
    "tldr",
    "tl;dr",
    "gist",
];

const VALUE_MARKERS: &[&str] = &[
    "what is",
    "what was",
    "what are",
    "what were",
    "which",
    "who",
    "when",
    "where",
    "current",
    "status of",
    "value of",
    "how much",
    "how many",
];

/// Lexical overlap between a question and one rendered context line.
///
/// Shared deliberately. [`LocalReasoner`](crate::llm::LocalReasoner) scores
/// context lines with this to decide whether to emit `#ESCALATE`, and
/// [`DecisionPolicy::needs_escalation`] scores the *assembled window* with the
/// same function to reach the same decision without being told. That is the
/// whole basis for serving a harness that cannot signal: the escalation
/// judgement never depended on anything private to the model, only on the
/// query and the bytes the model was handed. If these two ever diverge, the
/// runtime-side check is no longer a stand-in for the model-side one, so there
/// is one function rather than two that agree today.
pub fn overlap(query: &str, body: &str) -> f32 {
    let q: HashSet<String> = content_tokens(query).into_iter().collect();
    if q.is_empty() {
        return 0.0;
    }
    // `server.ip = 10.0.9.7` should match "server ip", so split dotted keys.
    let expanded = format!("{} {}", body.replace('.', " "), body);
    let tokens: HashSet<String> = content_tokens(&expanded).into_iter().collect();
    let mut score = q.intersection(&tokens).count() as f32 / q.len() as f32;
    // A question is about the *subject*, so key matches count double.
    let head = body.split('=').next().unwrap_or("").replace('.', " ");
    let head_tokens: HashSet<String> = content_tokens(&head).into_iter().collect();
    let head_hits = q.intersection(&head_tokens).count();
    if head_hits > 0 {
        score += 0.5 * head_hits as f32 / q.len() as f32;
    }
    score
}

/// Kinds that are cheap, near-always relevant, and pinned into every turn.
pub const ALWAYS_ADMIT: [Kind; 2] = [Kind::Goal, Kind::Constraint];

/// What the runtime decided about one node this turn — for telemetry and for
/// `explain_plan()`.
#[derive(Debug, Clone)]
pub struct Decision {
    pub level: Option<Level>,
    pub admitted: bool,
    pub recompute: bool,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub struct DecisionPolicy {
    pub deescalate_after: u32,
    /// Below this a match is unusable. Mirrors `LocalReasoner::threshold`.
    pub answer_floor: f32,
    /// Between `answer_floor` and this, the best match is plausible but thin.
    /// Mirrors `LocalReasoner::escalate_below`.
    pub escalate_below: f32,
    /// Below `answer_floor` but at or above this, there is enough overlap that
    /// the raw span is worth a look rather than answering "I don't have that".
    pub escalate_from: f32,
}

impl Default for DecisionPolicy {
    fn default() -> Self {
        Self {
            deescalate_after: 3,
            answer_floor: 0.25,
            escalate_below: 0.6,
            escalate_from: 0.1,
        }
    }
}

impl DecisionPolicy {
    // -- routing -----------------------------------------------------------
    pub fn classify(&self, query: &str) -> QueryType {
        let q = query.to_lowercase();
        if contains_any(&q, QUOTE_MARKERS) {
            return QueryType::QuoteExact;
        }
        // Deliberately narrow: a bare "how many" is a lookup far more often
        // than it is a computation, and mis-routing a lookup to L3 wastes an
        // execution and returns nothing.
        if contains_any(&q, RECOMPUTE_MARKERS)
            || (contains_any(&q, &["how much"]) && contains_any(&q, &["cost"]))
        {
            return QueryType::Recompute;
        }
        if contains_any(&q, JUSTIFY_MARKERS) {
            return QueryType::Justify;
        }
        if contains_any(&q, SUMMARIZE_MARKERS) {
            return QueryType::Summarize;
        }
        if contains_any(&q, VALUE_MARKERS) {
            return QueryType::ValueLookup;
        }
        QueryType::Open
    }

    /// Cheapest level in the routing order that this node actually has.
    pub fn preferred_level(&self, node: &Node, qtype: QueryType, available: &[Level]) -> Level {
        for level in qtype.routing() {
            if available.contains(&level) {
                if level == Level::L0 && self.prefers_compact(node, qtype) {
                    continue;
                }
                return level;
            }
        }
        available.first().copied().unwrap_or(Level::L2)
    }

    /// Never promote a settled, well-grounded claim to raw bytes just because
    /// the router said L0 — level over-promotion is the failure mode that
    /// destroys the cost model.
    fn prefers_compact(&self, node: &Node, qtype: QueryType) -> bool {
        qtype != QueryType::QuoteExact && node.kind != Kind::Evidence && node.confidence >= 0.9
    }

    pub fn sufficient(&self, level: Level, qtype: QueryType) -> bool {
        if qtype == QueryType::QuoteExact {
            level == Level::L0
        } else {
            qtype.routing()[..2].contains(&level)
        }
    }

    /// Next richer level to try when the model reports insufficiency.
    pub fn escalation_target(&self, current: Level, available: &[Level]) -> Option<Level> {
        let mut target = match current {
            Level::L2 => Some(Level::L1),
            Level::L1 | Level::L3 => Some(Level::L0),
            Level::L0 => None,
        };
        while let Some(level) = target {
            if available.contains(&level) {
                return Some(level);
            }
            target = match level {
                Level::L2 => Some(Level::L1),
                Level::L1 | Level::L3 => Some(Level::L0),
                Level::L0 => None,
            };
        }
        None
    }

    /// A node read at L0 repeatedly with nothing new extracted should be
    /// collapsed into a cached fact instead of re-quoted every turn.
    pub fn should_deescalate(&self, node: &Node) -> bool {
        node.meta.l0_reads.get() >= self.deescalate_after && node.meta.l0_yield.get() == 0
    }

    // -- the decision table ------------------------------------------------
    pub fn decide(
        &self,
        node: &Node,
        qtype: QueryType,
        available: &[Level],
        in_plan: bool,
    ) -> Decision {
        if node.status == Status::Stale {
            return Decision {
                level: None,
                admitted: false,
                recompute: true,
                reason: "stale — recompute or re-ground before admitting",
            };
        }
        if node.status == Status::Superseded {
            return Decision {
                level: None,
                admitted: false,
                recompute: false,
                reason: "superseded",
            };
        }
        if !in_plan {
            return Decision {
                level: None,
                admitted: false,
                recompute: false,
                reason: "not referenced by current plan",
            };
        }
        let level = self.preferred_level(node, qtype, available);
        Decision {
            level: Some(level),
            admitted: true,
            recompute: false,
            reason: match level {
                Level::L0 => "query needs exact wording",
                Level::L1 => "referenced for gist only",
                Level::L2 => "compact state object is sufficient",
                Level::L3 => "answer is a function of known inputs",
            },
        }
    }

    /// The node this window is too compressed to answer from, if any —
    /// computed by the runtime instead of waiting for the model to say so.
    ///
    /// The documented poor fit was "reasoners that cannot signal": a harness
    /// with no way to emit `#ESCALATE` loses the mechanism that carries the
    /// exact-quote and buried-detail probes. But the model-side trigger is a
    /// function of the query and the rendered window and nothing else, so the
    /// runtime can evaluate it directly. See [`overlap`].
    ///
    /// This is a *proxy*, and the distinction matters for what may be claimed
    /// from it: a real model's sense of insufficiency is not this function.
    /// What it establishes is that the signal is reconstructible from what the
    /// runtime already holds, not that escalation no longer needs a model.
    pub fn needs_escalation(
        &self,
        query: &str,
        context: &ActiveContext,
        already: &HashSet<NodeIdx>,
    ) -> Option<NodeIdx> {
        let mut best: Option<(f32, &crate::planner::Admission)> = None;
        for entry in &context.entries {
            let score = overlap(query, &entry.rendered);
            if best.is_none_or(|(top, _)| score > top) {
                best = Some((score, entry));
            }
        }
        let (score, entry) = best?;
        if entry.level == Level::L0 || already.contains(&entry.node) {
            return None;
        }
        let wants_quote = self.classify(query) == QueryType::QuoteExact;
        // Below the floor with *some* overlap the detail may be in the raw
        // span; between the floor and `escalate_below` the compact form is
        // plausible but thin, which is the case escalation exists for.
        let thin = score < self.escalate_below && score >= self.answer_floor;
        let unusable = score < self.answer_floor && score >= self.escalate_from;
        if wants_quote || thin || unusable {
            Some(entry.node)
        } else {
            None
        }
    }

    pub fn pinned_kinds(&self, qtype: QueryType) -> Vec<Kind> {
        let mut kinds = ALWAYS_ADMIT.to_vec();
        match qtype {
            QueryType::Justify => kinds.push(Kind::Decision),
            QueryType::Recompute => kinds.push(Kind::Calculation),
            QueryType::Open => kinds.push(Kind::OpenQuestion),
            _ => {}
        }
        kinds
    }
}

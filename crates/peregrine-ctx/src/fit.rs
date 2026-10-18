//! The fit gate — detecting the workloads DCR is a poor fit for, and getting
//! out of the way.
//!
//! Planning is rent paid on history: indexing, scoring and the knapsack buy a
//! small, sufficient window out of a store too big to show whole. When the
//! entire store *fits* in the window untouched, that rent buys nothing — the
//! planner can only remove material the budget could have held. The gate
//! detects that shape and bypasses the machinery rather than pretending the
//! overhead is free.
//!
//! Two deliberate limits:
//!
//! * A store with skills registered never bypasses — admission is what loads
//!   them, and skipping the planner would leave them dormant.
//! * The bypass sends raw history, so it is only as good as the reasoner's
//!   ability to read raw text. `LocalReasoner` is a line-matcher over the
//!   assembled format and answers nothing from it; the gate is for real
//!   reasoners, which is exactly the deployment it exists for.

use crate::graph::MemoryGraph;
use crate::nodes::Kind;

/// What the gate decided. Kept as an enum rather than a bool so the reason is
/// part of the type a caller matches on — a bypass is a shortcut that must
/// stay auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Every ingested token fits in `B_attention`: show it all, skip planning.
    Bypass,
    /// History exceeds the window, or capabilities need the planner: the
    /// machinery is earning its rent.
    Plan,
}

/// The decision, from the two numbers that define it plus the skills check.
pub fn assess(graph: &MemoryGraph, history_tokens: usize, budget: usize) -> Fit {
    if history_tokens <= budget && !graph.nodes().iter().any(|n| n.kind == Kind::Skill) {
        Fit::Bypass
    } else {
        Fit::Plan
    }
}

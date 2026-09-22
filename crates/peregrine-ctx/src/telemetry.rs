//! Telemetry — the metrics the evaluation design asks for.
//!
//! Phase 5 of the roadmap names the numbers that could falsify the `O(k + r)`
//! claim, so the runtime records them rather than asserting them:
//!
//! * **escalation rate** — how often the cheap level was insufficient;
//! * **stale-fact read rate** — how often a stale node reached the model;
//! * **prefetch hit rate** — whether speculation pays for its waste;
//! * **tokens per resolved query** — `k + r`, measured;
//! * **audit-path completeness** — the fraction of `explain()` walks that reach
//!   evidence without truncating.

use crate::policy::QueryType;

#[derive(Debug, Clone)]
pub struct Turn {
    pub query: String,
    pub qtype: QueryType,
    pub tokens: usize,  // audit-allow: LM token count, not a credential
    pub budget: usize,
    pub admitted: usize,
    pub considered: usize,
    pub escalations: u32,
    pub stale_reads: u32,
    pub demotions: usize,
    pub overflow: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub turns: Vec<Turn>,
    pub escalations: u32,
    pub stale_reads: u32,
    pub explain_calls: u32,
    pub explain_complete: u32,
    /// Prefetch outcomes, copied in from the speculator at report time.
    ///
    /// The evaluation design lists prefetch hit rate among the metrics that
    /// decide whether speculation pays for itself, so it belongs in the report
    /// rather than only in a subsystem's own `stats()` string.
    pub prefetch_issued: u32,
    pub prefetch_hits: u32,
    pub wasted_builds: u32,
    /// What the naive baseline would have paid: every token ever ingested.
    pub history_tokens: usize,  // audit-allow: LM token count, not a credential
}

impl Telemetry {
    pub fn record_turn(&mut self, turn: Turn) {
        self.escalations += turn.escalations;
        self.stale_reads += turn.stale_reads;
        self.turns.push(turn);
    }

    /// Copy the speculator's counters in before reporting. Speculation owns
    /// them; telemetry only publishes them.
    pub fn record_prefetch(&mut self, issued: u32, hits: u32, wasted: u32) {
        self.prefetch_issued = issued;
        self.prefetch_hits = hits;
        self.wasted_builds = wasted;
    }

    pub fn record_explain(&mut self, complete: bool) {
        self.explain_calls += 1;
        self.explain_complete += u32::from(complete);
    }

    pub fn report(&self) -> Report {
        let turns = self.turns.len();
        let tokens: Vec<usize> = self.turns.iter().map(|t| t.tokens).collect();
        let mean = if tokens.is_empty() {
            0.0
        } else {
            tokens.iter().sum::<usize>() as f64 / tokens.len() as f64
        };
        Report {
            turns,
            escalation_rate: ratio(self.escalations as f64, turns as f64),
            stale_fact_read_rate: ratio(self.stale_reads as f64, turns as f64),
            tokens_per_query_mean: mean,
            tokens_per_query_max: tokens.iter().copied().max().unwrap_or(0),
            budget_overflows: self.turns.iter().filter(|t| t.overflow).count(),
            demotions: self.turns.iter().map(|t| t.demotions).sum(),
            audit_path_completeness: ratio(self.explain_complete as f64, self.explain_calls as f64),
            history_tokens: self.history_tokens,
            compression_ratio: if mean > 0.0 {
                Some(self.history_tokens as f64 / mean)
            } else {
                None
            },
            prefetch_hit_rate: ratio(self.prefetch_hits as f64, self.prefetch_issued as f64),
            wasted_builds: self.wasted_builds,
        }
    }
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

#[derive(Debug, Clone)]
pub struct Report {
    pub turns: usize,
    pub escalation_rate: Option<f64>,
    pub stale_fact_read_rate: Option<f64>,
    pub tokens_per_query_mean: f64,  // audit-allow: LM token count, not a credential
    pub tokens_per_query_max: usize,  // audit-allow: LM token count, not a credential
    pub budget_overflows: usize,
    pub demotions: usize,
    pub audit_path_completeness: Option<f64>,
    pub history_tokens: usize,  // audit-allow: LM token count, not a credential
    pub compression_ratio: Option<f64>,
    /// `None` until speculation has issued a prefetch: a rate over zero
    /// attempts is not zero, it is undefined, and printing 0.000 would read as
    /// "speculation is failing" rather than "speculation has not run".
    pub prefetch_hit_rate: Option<f64>,
    pub wasted_builds: u32,
}

fn opt(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) => format!("{v:.decimals$}"),
        None => "n/a".to_string(),
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: [(&str, String); 12] = [
            ("turns", self.turns.to_string()),
            ("escalation_rate", opt(self.escalation_rate, 3)),
            ("stale_fact_read_rate", opt(self.stale_fact_read_rate, 3)),
            (
                "tokens_per_query_mean",
                format!("{:.1}", self.tokens_per_query_mean),
            ),
            (
                "tokens_per_query_max",
                self.tokens_per_query_max.to_string(),
            ),
            ("budget_overflows", self.budget_overflows.to_string()),
            ("demotions", self.demotions.to_string()),
            (
                "audit_path_completeness",
                opt(self.audit_path_completeness, 3),
            ),
            ("history_tokens", self.history_tokens.to_string()),
            ("compression_ratio", opt(self.compression_ratio, 1)),
            ("prefetch_hit_rate", opt(self.prefetch_hit_rate, 3)),
            ("wasted_builds", self.wasted_builds.to_string()),
        ];
        let width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (key, value) in rows {
            writeln!(f, "{key:<width$} : {value}")?;
        }
        Ok(())
    }
}

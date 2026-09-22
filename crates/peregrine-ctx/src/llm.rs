//! Reasoner adapters — "Solution A".
//!
//! Two implementations, for two different purposes:
//!
//! * [`LocalReasoner`] is deterministic, offline and deliberately literal. It
//!   can only answer from the text of the active context it is handed, and it
//!   signals insufficiency with `#ESCALATE <node_id>` instead of guessing. That
//!   makes it the right instrument for evaluating the *runtime*: if the answer
//!   is wrong, the planner put the wrong thing in the window, and no model
//!   quality can be blamed for it.
//!
//! * [`CommandReasoner`] shells out to a program that talks to a real model.
//!   Rust has no official Anthropic SDK, so rather than hand-rolling TLS this
//!   pipes the prompt to a command of your choosing — see the module docs below
//!   for a ready-made `curl` invocation against the Messages API.
//!
//! The asymmetry the wiki argues for shows up here: the Memory Runtime
//! (indexing, addressing, traversal, memoisation, invalidation) needs no model
//! at all, and the Reasoner never sees history — only `k`.

use std::collections::HashSet;
use std::io::Write as _;
use std::process::{Command, Stdio};

use crate::text::contains_any;

pub const DEFAULT_MODEL: &str = "claude-opus-5";

pub const REASONER_SYSTEM: &str = r#"You are the Reasoner in a Dynamic Context Runtime.

You never see the full history. You see a small ACTIVE CONTEXT assembled by a
memory runtime: cached facts at L2, summaries at L1, exact source spans at L0,
and computed results at L3. Every line starts with the node id it came from.

Rules:
- Answer only from the active context. Do not invent facts.
- Cite the node ids you used, in the form [node_id].
- If the context is insufficient — you need the exact wording of something you
  were given only as a summary or a fact — reply with exactly:
      #ESCALATE <node_id>
  and nothing else. The runtime will re-plan and give you that node at L0.
- A fact marked status=stale must not be used; escalate instead.
"#;

pub trait Reasoner {
    fn complete(&mut self, prompt: &str, system: &str) -> String;
}

/// `#ESCALATE <node_id>` anywhere in a reply.
pub fn parse_escalation(text: &str) -> Option<String> {
    let start = text.find("#ESCALATE")? + "#ESCALATE".len();
    let rest = text[start..].trim_start();
    let id: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if id.is_empty() { None } else { Some(id) }
}

struct ContextLine {
    node: String,
    level: String,
    body: String,
}

/// Offline, deterministic reasoner over the rendered active context.
#[derive(Debug, Clone)]
pub struct LocalReasoner {
    pub threshold: f32,
    /// Between `threshold` and this, the best match is plausible but thin — the
    /// compact form probably dropped the detail the question turns on. That is
    /// precisely the "model signals insufficiency" case escalation exists for.
    pub escalate_below: f32,
    /// Whether this reasoner can emit `#ESCALATE` at all.
    ///
    /// `false` models the documented poor fit — a harness with no way to signal
    /// insufficiency. It is a property of the *harness*, so it is set here and
    /// not on the runtime; `Dcr::auto_escalate` is the runtime's answer to it.
    pub signal: bool,
    escalated: HashSet<String>,
    current_query: Option<String>,
}

impl Default for LocalReasoner {
    fn default() -> Self {
        Self {
            threshold: 0.25,
            escalate_below: 0.6,
            signal: true,
            escalated: HashSet::new(),
            current_query: None,
        }
    }
}

impl LocalReasoner {
    pub fn new() -> Self {
        Self::default()
    }

    fn query_of(prompt: &str) -> String {
        for line in prompt.lines() {
            if let Some(rest) = line.strip_prefix("QUESTION:") {
                return rest.trim().to_string();
            }
        }
        prompt.trim().lines().next_back().unwrap_or("").to_string()
    }

    /// Parse `[node_id L2] body` lines out of the rendered context.
    fn lines(prompt: &str) -> Vec<ContextLine> {
        let mut out = Vec::new();
        for raw in prompt.lines() {
            let raw = raw.trim();
            let Some(rest) = raw.strip_prefix('[') else {
                continue;
            };
            let Some(close) = rest.find(']') else {
                continue;
            };
            let head = &rest[..close];
            let body = rest[close + 1..].trim().to_string();
            let mut parts = head.split_whitespace();
            let (Some(node), Some(level)) = (parts.next(), parts.next()) else {
                continue;
            };
            if !matches!(level, "L0" | "L1" | "L2" | "L3") {
                continue;
            }
            out.push(ContextLine {
                node: node.to_string(),
                level: level.to_string(),
                body,
            });
        }
        out
    }

    /// Delegates to [`crate::policy::overlap`], which the runtime also uses to
    /// reconstruct this reasoner's escalation decision without being told it.
    fn score(query: &str, line: &ContextLine) -> f32 {
        crate::policy::overlap(query, &line.body)
    }
}

impl Reasoner for LocalReasoner {
    fn complete(&mut self, prompt: &str, _system: &str) -> String {
        let query = Self::query_of(prompt);
        if self.current_query.as_deref() != Some(query.as_str()) {
            // Escalation memory is per question, not per session: refusing to
            // escalate a node twice across unrelated questions would silently
            // answer a later one from a summary that already proved too thin.
            self.current_query = Some(query.clone());
            self.escalated.clear();
        }
        let lines = Self::lines(prompt);
        if lines.is_empty() {
            return "I don't have that in the active context.".to_string();
        }
        let wants_quote = contains_any(
            &query.to_lowercase(),
            &["quote", "exact", "exactly", "verbatim", "word for word"],
        );
        let mut scored: Vec<(f32, &ContextLine)> =
            lines.iter().map(|l| (Self::score(&query, l), l)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        // `scored` is built from `lines`, which the guard above proved
        // non-empty, so `first()` is always `Some`; use it anyway so the method
        // carries no index that could panic if that guard ever moves.
        let Some((best_score, best)) = scored.first() else {
            return "I don't have that in the active context.".to_string();
        };
        let escalatable =
            self.signal && best.level != "L0" && !self.escalated.contains(&best.node);

        if *best_score < self.threshold {
            // Nothing usable. If the best candidate is a compressed form with
            // *some* overlap, the detail may be in its raw span — escalate
            // rather than answer "I don't know" from a summary.
            if *best_score >= 0.1 && escalatable {
                self.escalated.insert(best.node.clone());
                return format!("#ESCALATE {}", best.node);
            }
            return "I don't have that in the active context.".to_string();
        }
        let weak = *best_score < self.escalate_below;
        if (wants_quote || weak) && escalatable {
            self.escalated.insert(best.node.clone());
            return format!("#ESCALATE {}", best.node);
        }
        if best.level == "L0" {
            let body = best.body.trim().trim_matches('"');
            return format!("{body} [{}]", best.node);
        }
        let mut value = best
            .body
            .split('\u{b7}')
            .next()
            .unwrap_or(&best.body)
            .trim()
            .to_string();
        if value.contains('=') && !wants_quote {
            value = value
                .split_once('=')
                .map(|(_, v)| v.trim().to_string())
                .unwrap_or(value);
        }
        format!("{value} [{}]", best.node)
    }
}

/// Reasoner backed by an external command.
///
/// The command receives the prompt on stdin and must print the answer on
/// stdout. A working Anthropic Messages API bridge is three lines of shell:
///
/// ```bash
/// #!/usr/bin/env bash
/// # save as ask-claude.sh, chmod +x
/// jq -Rs '{model:"claude-opus-5", max_tokens:16000, thinking:{type:"adaptive"},
///          system:$ENV.DCR_SYSTEM, messages:[{role:"user", content:.}]}' \
///   | curl -s https://api.anthropic.com/v1/messages \
///       -H "content-type: application/json" \
///       -H "x-api-key: $ANTHROPIC_API_KEY" \
///       -H "anthropic-version: 2023-06-01" -d @- \
///   | jq -r '.content[] | select(.type=="text") | .text'
/// ```
///
/// then `CommandReasoner::new("./ask-claude.sh", &[])`. The system prompt is
/// passed in the `DCR_SYSTEM` environment variable.
#[derive(Debug, Clone)]
pub struct CommandReasoner {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandReasoner {
    pub fn new(program: impl Into<String>, args: &[&str]) -> Self {
        Self {
            program: program.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Reasoner for CommandReasoner {
    fn complete(&mut self, prompt: &str, system: &str) -> String {
        let child = Command::new(&self.program)
            .args(&self.args)
            .env("DCR_SYSTEM", system)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(e) => return format!("[reasoner {} failed to start: {e}]", self.program),
        };
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(prompt.as_bytes());
        }
        match child.wait_with_output() {
            Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
            Err(e) => format!("[reasoner {} failed: {e}]", self.program),
        }
    }
}

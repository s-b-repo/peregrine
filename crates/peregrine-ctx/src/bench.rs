//! Benchmark: DCR vs full context vs a sliding window.
//!
//! The roadmap's Phase 5 asks for a benchmark that could *falsify* the
//! `O(k + r)` claim, so this measures the things that would show it failing:
//! tokens per resolved query as history grows, and whether the answers stay
//! correct while the window stays small.
//!
//! The reasoner is the same deterministic line-matcher for every system, so the
//! comparison is between *context assemblies*, not between models. Baselines
//! are given every advantage that is honest: full context sees the entire
//! transcript, the sliding window sees the most recent `window` tokens, and on
//! a scoring tie both prefer the *latest* matching line, so they get
//! corrections right whenever the correction is inside their window.

use std::collections::HashSet;
use std::time::Instant;

use crate::baselines::{Rag, Recursive, SummarizeAll};
use crate::context_store::ContextStore;
use crate::graph::{DcrError, MemoryGraph};
use crate::index::Namespace;
use crate::nodes::{Kind, NodeIdx};
use crate::llm::{LocalReasoner, Reasoner};
use crate::runtime::Dcr;
use crate::text::content_tokens;
use crate::tokens::estimate_tokens;

#[derive(Clone)]
pub struct Probe {
    pub query: &'static str,
    pub expected: &'static str,
    pub label: &'static str,
    /// When true the probe passes if `expected` is **absent** from the answer.
    ///
    /// A recall probe asks "can you find it". A refusal probe asks "can you
    /// decline to serve something you should not" — a stale derived figure, or
    /// a fact nobody ever stated. A benchmark made only of recall probes
    /// rewards confidently answering everything, which is the failure mode the
    /// whole design is aimed at.
    pub absent: bool,
    /// Score the assembled **context** rather than the answer.
    ///
    /// For probes that need a join the harness's reasoner cannot perform — it
    /// is a line-matcher, so it cannot chain A→B→C — scoring the answer
    /// measures the stand-in model, not the assembly. Scoring the window keeps
    /// the comparison on what this benchmark is actually about. Applied
    /// uniformly: full context trivially passes, because it genuinely does
    /// contain the material.
    pub on_context: bool,
}

impl Probe {
    /// Passes when the answer contains `expected`.
    pub const fn recall(query: &'static str, expected: &'static str, label: &'static str) -> Probe {
        Probe {
            query,
            expected,
            label,
            absent: false,
            on_context: false,
        }
    }

    /// Passes when the assembled context contains `expected`, whatever the
    /// reasoner then made of it.
    pub const fn assembled(
        query: &'static str,
        expected: &'static str,
        label: &'static str,
    ) -> Probe {
        Probe {
            query,
            expected,
            label,
            absent: false,
            on_context: true,
        }
    }

    /// Passes when the answer does **not** contain `expected`.
    pub const fn refuse(query: &'static str, expected: &'static str, label: &'static str) -> Probe {
        Probe {
            query,
            expected,
            label,
            absent: true,
            on_context: false,
        }
    }

    /// Did this answer pass?
    pub fn scores(&self, answer: &str) -> bool {
        let found = answer.to_lowercase().contains(&self.expected.to_lowercase());
        found != self.absent
    }
}

pub struct Corpus {
    pub docs: Vec<(String, String)>,
    pub probes: Vec<Probe>,
}

impl Corpus {
    pub fn text(&self) -> String {
        self.docs
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

const NOISE: &[&str] = &[
    "Standup notes for slot {n}: dashboards were refreshed after the overnight batch, {n} alerts \
     acknowledged and closed without action, and the on-call rotation stays as published. Nobody \
     raised a blocker. The mobile team asked about the shared staging cluster again; the answer is \
     unchanged, they should book a slot in the calendar rather than grabbing it. No follow-up \
     needed from this thread.",
    "Triage sweep {n}: the queue is at {n} items, mostly duplicate reports of the same flaky \
     integration test. Two were closed as cannot-reproduce, one was merged into an existing \
     thread, and the rest are waiting on logs from the reporter. Nothing here touches the checkout \
     path. The label taxonomy is still a mess and someone should clean it up when there is time, \
     which there is not.",
    "Chat log {n}: the coffee machine is broken again, facilities ticket {n} is filed, and the \
     descaling kit is apparently on back-order until next month. Somebody suggested a kettle. \
     Somebody else suggested going outside. The thread then drifted into a long argument about \
     tabs versus spaces which nobody won and which is reproduced here in full for completeness of \
     the audit log.",
    "Metrics digest {n}: p99 latency held steady at {n}ms across all regions, error budget \
     consumption is flat, and the saturation alarms did not fire. Cache hit rate drifted down by a \
     fraction of a percent, which is within noise. The synthetic probes from the eu-west region \
     were briefly red during a network maintenance window that was announced two weeks ago and is \
     not an incident.",
    "Retro prep {n}: the agenda doc has {n} comments, most of them about process rather than the \
     outage itself. Recurring themes are alert fatigue, unclear ownership of the shared queues, \
     and the fact that runbooks are out of date. Someone will volunteer to own runbook cleanup and \
     then not do it, as is traditional. Meeting is Thursday, same room, coffee situation \
     permitting.",
    "Deploy bot {n}: build {n} passed CI in eleven minutes, artifacts uploaded, images signed, and \
     the staging smoke suite is green. No action needed. The flaky browser test failed once and \
     passed on retry, as it does roughly one run in nine. The changelog is auto-generated and \
     contains nothing but dependency bumps and a typo fix in a comment nobody will ever read.",
    "Capacity note {n}: disk usage on runner-{n} sits at 61 percent, below the threshold, and the \
     cleanup cron is doing its job. Memory headroom is comfortable. The build cache could be \
     pruned more aggressively but the savings do not justify the churn. Nothing about this affects \
     the checkout service or any of its dependencies, and no action is required from anyone \
     reading this.",
    "Handover {n}: on-call handover completed, {n} open threads carried over, none of them \
     customer-facing. The incoming engineer has the runbook links and the escalation ladder. One \
     long-running investigation into intermittent DNS timeouts continues with no conclusion; it \
     has been open for weeks and remains a mystery that everyone has quietly agreed to live with \
     for now.",
];

/// A long ops transcript: a few facts, three corrections, a lot of noise.
pub fn build_corpus(turns: usize) -> Corpus {
    let mut docs: Vec<(String, String)> = Vec::new();
    let mut add = |id: &str, text: &str| docs.push((id.to_string(), text.to_string()));

    add(
        "t000",
        "Goal: restore checkout by 09:00 UTC.\n\n\
         Constraint: never restart the payment service during business hours.",
    );
    add(
        "t001",
        "The service is alpha-checkout and the owner is team-payments.",
    );
    add(
        "t002",
        "The error was \"connection refused\" when talking to the inventory host.",
    );
    add("t003", "The server ip is 10.0.4.12 and the port is 8080.");
    add("t004", "The deploy window is 23:00-01:00 UTC.");
    add(
        "t005",
        "The blocker is firewall rule 37, which drops traffic to the checkout subnet.",
    );
    add(
        "t006",
        "Decision: roll back to build 4471 because the blocker is firewall rule 37.",
    );
    add(
        "t007",
        "The engineer count is 3 and the incident hours are 4.",
    );
    add("t008", "The hourly rate is 180 USD.");
    add(
        "t009",
        "Deploy log for build 4471: the rollout started at 04:03 and moved through canary, then 10 \
         percent, then 50 percent of the fleet without incident. At 04:11 the checkout pods began \
         failing readiness probes against the inventory host, the load balancer drained them, and \
         the rollout controller entered a retry loop. The retry budget was exhausted after 7 \
         attempts and the final failure code was ERR_CONN_REFUSED_37, at which point the \
         controller stopped and paged on-call.",
    );

    let fixed = docs.len();
    let corrections: [(usize, &str); 3] = [
        (
            (turns as f64 * 0.35) as usize,
            "Correction: actually the server ip is 10.0.9.7, we misread the dashboard.",
        ),
        (
            (turns as f64 * 0.80) as usize,
            "Update: the deploy window is 02:00-04:00 UTC after the change-freeze review.",
        ),
        (
            (turns as f64 * 0.90) as usize,
            "Correction: the hourly rate is 210 USD, finance updated the figure.",
        ),
    ];
    for i in fixed..turns.max(fixed + 1) {
        let id = format!("t{i:03}");
        match corrections.iter().find(|(at, _)| *at == i) {
            Some((_, text)) => docs.push((id, text.to_string())),
            None => docs.push((id, NOISE[i % NOISE.len()].replace("{n}", &i.to_string()))),
        }
    }

    let probes = vec![
        Probe::recall(
            "what is the server ip?",
            "10.0.9.7",
            "corrected fact (mid-history)",
        ),
        Probe::recall(
            "what is the deploy window?",
            "02:00-04:00",
            "corrected fact (late)",
        ),
        Probe::recall(
            "who is the owner of the checkout service?",
            "team-payments",
            "old fact, never repeated",
        ),
        Probe::recall(
            "quote the exact error message",
            "connection refused",
            "exact quote",
        ),
        Probe::recall(
            "why did we roll back?",
            "firewall rule 37",
            "justification / multi-hop",
        ),
        Probe::recall(
            "how many retry attempts were made before the failure?",
            "7 attempts",
            "detail buried in a long span",
        ),
        Probe::recall(
            "what is the hourly rate?",
            "210",
            "corrected fact (very late)",
        ),
    ];
    Corpus { docs, probes }
}

/// Line-matcher over raw text — the same scoring the DCR reasoner uses.
///
/// On a tie it prefers the *latest* line, which is the strongest honest
/// baseline: it gets a correction right whenever the correction is in view.
#[derive(Debug, Default)]
pub struct BaselineReasoner;

impl Reasoner for BaselineReasoner {
    fn complete(&mut self, prompt: &str, _system: &str) -> String {
        let (query, body) = match prompt.rfind("\nQUESTION:") {
            Some(at) => (prompt[at + "\nQUESTION:".len()..].trim(), &prompt[..at]),
            None => ("", prompt),
        };
        let q: HashSet<String> = content_tokens(query).into_iter().collect();
        if q.is_empty() {
            return String::new();
        }
        let mut best = "";
        let mut best_score = 0.0f32;
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let expanded = format!("{} {line}", line.replace('.', " "));
            let tokens: HashSet<String> = content_tokens(&expanded).into_iter().collect();
            let score = q.intersection(&tokens).count() as f32 / q.len() as f32;
            if score >= best_score {
                best_score = score;
                best = line;
            }
        }
        if best_score >= 0.2 {
            best.to_string()
        } else {
            "I don't have that in the context.".to_string()
        }
    }
}

/// Keep the last `budget` tokens — the sliding-window baseline.
fn truncate_tokens(text: &str, budget: usize) -> String {
    if estimate_tokens(text) <= budget {
        return text.to_string();
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut used = 0usize;
    for line in text.lines().rev() {
        let cost = estimate_tokens(line);
        if used + cost > budget {
            break;
        }
        kept.push(line);
        used += cost;
    }
    kept.reverse();
    kept.join("\n")
}

struct Row {
    label: &'static str,
    query: &'static str,
    correct_full: bool,
    correct_window: bool,
    correct_dcr: bool,
    dcr_tokens: usize,
    dcr_answer: String,
}

pub fn run_benchmark(turns: usize, budget: usize, window: usize) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    let full_text = corpus.text();
    let full_tokens = estimate_tokens(&full_text);
    let windowed = truncate_tokens(&full_text, window);
    let windowed_tokens = estimate_tokens(&windowed);

    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id))?;
    }

    let mut baseline = BaselineReasoner;
    let mut reasoner = LocalReasoner::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut ungrounded: Vec<&str> = Vec::new();

    for probe in &corpus.probes {
        let full = baseline.complete(&format!("{full_text}\n\nQUESTION: {}", probe.query), "");
        let win = baseline.complete(&format!("{windowed}\n\nQUESTION: {}", probe.query), "");
        let answer = runtime.ask_with(probe.query, None, &mut reasoner);
        // Auditability is part of the deliverable, not a bonus: every DCR
        // answer must be walkable back to raw spans.
        for node_id in &answer.cited {
            let _ = runtime.explain(node_id);
        }
        // Substring containment is not enough on its own. A reviewer measured
        // 34 of 41 fallback turns in their own corpus scoring "correct" with no
        // grounding at all, because the scorer checked completion rather than
        // grounding — and containment has the identical hole: a fluent answer
        // assembled from the wrong context passes it exactly as cleanly as a
        // right one. DCR already records whether each cited node walks back to
        // raw spans and had never been allowed to fail a probe. Now it can.
        // On an honest run this changes nothing, which is the point.
        let grounded = !answer.cited.is_empty()
            && answer.cited.iter().all(|id| {
                runtime
                    .graph
                    .idx_of(id)
                    .is_some_and(|idx| runtime.graph.explain(idx, None).complete)
            });
        let hit = |text: &str| text.to_lowercase().contains(&probe.expected.to_lowercase());
        if hit(&answer.text) && !grounded {
            ungrounded.push(probe.label);
        }
        rows.push(Row {
            label: probe.label,
            query: probe.query,
            correct_full: hit(&full),
            correct_window: hit(&win),
            correct_dcr: hit(&answer.text) && grounded,
            dcr_tokens: answer.tokens,
            dcr_answer: answer.text,
        });
    }

    let line = "-".repeat(88);
    println!("{line}");
    println!(
        "CONTEXT ROT BENCHMARK — {turns} turns, {} documents, {full_tokens} tokens of history",
        corpus.docs.len()
    );
    println!("B_attention = {budget} tokens \u{b7} sliding window = {window} tokens");
    println!("{line}");
    let header = format!(
        "{:<38} {:>6} {:>7} {:>6}   {:>10}",
        "probe", "full", "window", "DCR", "DCR tokens"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    let mark = |ok: bool| if ok { "  ok  " } else { " MISS " };
    for row in &rows {
        println!(
            "{:<38}{} {} {} {:>10}",
            row.label,
            mark(row.correct_full),
            mark(row.correct_window),
            mark(row.correct_dcr),
            row.dcr_tokens
        );
    }
    println!("{}", "-".repeat(header.len()));
    let total = |f: fn(&Row) -> bool| rows.iter().filter(|r| f(r)).count();
    println!(
        "{:<38}{:>6} {:>7} {:>6}   of {}",
        "correct",
        total(|r| r.correct_full),
        total(|r| r.correct_window),
        total(|r| r.correct_dcr),
        rows.len()
    );
    let mean_dcr = rows.iter().map(|r| r.dcr_tokens).sum::<usize>() as f64 / rows.len() as f64;
    println!(
        "{:<38}{:>6.1} {:>7.1} {:>6.1}",
        "mean tokens per query", full_tokens as f64, windowed_tokens as f64, mean_dcr
    );
    println!(
        "{:<38}{:>6} {:>7} {:>6}",
        "attention vs full history",
        "1x",
        format!("{:.1}x", full_tokens as f64 / windowed_tokens.max(1) as f64),
        format!("{:.0}x", full_tokens as f64 / mean_dcr.max(1.0))
    );
    println!("{line}");
    println!("per-probe answers (DCR):");
    for row in &rows {
        println!(
            "  [{}] {}",
            if row.correct_dcr { "ok  " } else { "MISS" },
            row.query
        );
        let answer: String = row.dcr_answer.chars().take(110).collect();
        println!("          -> {answer}");
    }
    println!("{line}");
    println!("DCR telemetry");
    print!("{}", runtime.telemetry.report());
    println!("{line}");
    println!("one-time indexing cost (amortised over all queries, not per query):");
    let stats = runtime.graph.stats();
    println!(
        "  state nodes: {}   edges: {}   L0 spans: {}",
        stats.nodes,
        stats.edges,
        runtime.raw.len()
    );
    println!(
        "  ladder builds: {}   superseded: {}   stale: {}",
        runtime.ladder.builds(),
        stats.superseded,
        stats.stale
    );
    println!("  note: storage stays O(N); the claim is bounded *attention*, not bounded storage.");
    println!("{line}");
    if ungrounded.is_empty() {
        println!(
            "grounding gate: 0 of {} answers matched the expected value without a complete\n\
             audit path to raw spans. Correctness above is grounded correctness, not containment.",
            corpus.probes.len()
        );
    } else {
        println!(
            "grounding gate: {} answer(s) matched the expected value but could not be walked back\n\
             to raw spans, and are scored as failures: {}",
            ungrounded.len(),
            ungrounded.join("; ")
        );
    }
    println!("{line}");
    println!("how to read this");
    println!("  * The reasoner is a deterministic line-matcher for all three systems, so this");
    println!("    compares context assemblies, not models. Full-context accuracy here is a");
    println!("    FLOOR, not a ceiling: a real model reading the whole transcript would do");
    println!("    better on the retrieval probes. Do not read these columns as 'DCR is more");
    println!("    accurate than a long-context model'.");
    println!("  * The load-bearing results are (a) the token counts, which no model quality");
    println!("    changes, and (b) the sliding window's misses, which are structural — a fact");
    println!("    outside the window is unrecoverable at any model quality.");
    println!("  * Escalations are counted and charged: a probe that needed L0 costs more, and");
    println!("    that cost is in the DCR token column, not hidden.");
    Ok(())
}

/// Which mechanism is load-bearing for which question?
///
/// Each row disables exactly one thing and re-runs the whole probe set, so a
/// drop in the `correct` column names the mechanism that was carrying it. A
/// design whose parts can all be removed without changing the result does not
/// need those parts.
pub fn run_ablation(turns: usize, budget: usize) -> Result<(), DcrError> {
    // `apply` configures the runtime *and* the harness, because two of these
    // rows ablate the caller rather than the runtime: a model with no way to
    // emit `#ESCALATE` is a property of the harness, and it is the documented
    // poor fit the last two rows exist to price.
    struct Ablation {
        name: &'static str,
        apply: fn(&mut Dcr, &mut LocalReasoner),
    }
    let ablations = [
        Ablation {
            name: "full runtime",
            apply: |_, _| {},
        },
        Ablation {
            name: "no supersession",
            apply: |rt, _| rt.indexer.supersede_on_conflict = false,
        },
        Ablation {
            name: "no reference linking",
            apply: |rt, _| rt.indexer.reference_linking = false,
        },
        Ablation {
            name: "no escalation",
            apply: |rt, _| rt.max_escalations = 0,
        },
        Ablation {
            name: "no seed floor",
            apply: |rt, _| rt.planner.seed_min_ratio = 0.0,
        },
        Ablation {
            name: "no graph expansion",
            apply: |rt, _| rt.planner.max_depth = 0,
        },
        Ablation {
            name: "L2 only (no ladder)",
            apply: |rt, _| {
                rt.max_escalations = 0;
                rt.ladder.flatten_to_l2 = true;
            },
        },
        // The mechanism is still enabled; the harness simply cannot ask for it.
        // This is the row the "reasoners that cannot signal" poor fit describes,
        // and it should reproduce the "no escalation" result exactly. If it does
        // not, the two are not measuring the same loss and the doc is wrong.
        Ablation {
            name: "harness cannot signal",
            apply: |_, r| r.signal = false,
        },
        // Same harness, with the runtime inferring the request instead.
        Ablation {
            name: "  + runtime infers it",
            apply: |rt, r| {
                r.signal = false;
                rt.auto_escalate = true;
            },
        },
    ];

    let corpus = build_corpus(turns);
    println!(
        "ABLATION - {turns} turns, {} tokens of history, B_attention = {budget}",
        estimate_tokens(&corpus.text())
    );
    let header = format!(
        "{:<24} {:>8} {:>9} {:>7} {:>7}   {}",
        "variant", "correct", "mean k", "esc.", "nodes", "probes that fail"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for ablation in ablations {
        let mut runtime = Dcr::new(budget);
        let mut reasoner = LocalReasoner::new();
        (ablation.apply)(&mut runtime, &mut reasoner);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut correct = 0usize;
        let mut failed: Vec<&str> = Vec::new();
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            if answer
                .text
                .to_lowercase()
                .contains(&probe.expected.to_lowercase())
            {
                correct += 1;
            } else {
                failed.push(probe.label);
            }
        }
        let report = runtime.telemetry.report();
        println!(
            "{:<24} {:>8} {:>9.1} {:>7.2} {:>7}   {}",
            ablation.name,
            format!("{correct}/{}", corpus.probes.len()),
            report.tokens_per_query_mean,
            report.escalation_rate.unwrap_or(0.0),
            runtime.graph.len(),
            if failed.is_empty() {
                "-".to_string()
            } else {
                failed.join("; ")
            }
        );
    }
    println!("{}", "-".repeat(header.len()));
    Ok(())
}

/// How does the answer change as `B_attention` changes?
///
/// The budget is the one knob an operator actually turns, so the curve of
/// correctness against it is the practical question: how small can the window
/// be before answers start to go missing?
pub fn run_sweep(turns: usize, budgets: &[usize]) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    println!(
        "BUDGET SWEEP - {turns} turns, {} tokens of history",
        estimate_tokens(&corpus.text())
    );
    println!(
        "{:>11} {:>9} {:>8} {:>8} {:>10} {:>10}",
        "B_attention", "correct", "mean k", "max k", "escal.", "demotions"
    );
    println!("{}", "-".repeat(60));
    for &budget in budgets {
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let mut correct = 0usize;
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            if answer
                .text
                .to_lowercase()
                .contains(&probe.expected.to_lowercase())
            {
                correct += 1;
            }
        }
        let report = runtime.telemetry.report();
        println!(
            "{:>11} {:>9} {:>8.1} {:>8} {:>10.2} {:>10}",
            budget,
            format!("{correct}/{}", corpus.probes.len()),
            report.tokens_per_query_mean,
            report.tokens_per_query_max,
            report.escalation_rate.unwrap_or(0.0),
            report.demotions
        );
    }
    println!("{}", "-".repeat(60));
    Ok(())
}

/// Positive control for `stale_fact_read_rate`.
///
/// A zero is only evidence if the run could have produced nonzero. This builds
/// a run that *can*: a derived value is computed, its input is then corrected
/// so the derivation goes stale, and a probe asks for exactly that stale value.
/// The expected outcomes are written down here, before execution, per the
/// critique that motivated this control:
///
///   guard ON  (production) : stale node skipped -> stale_fact_read_rate = 0.0
///   guard OFF (bypassed)   : stale node admitted -> stale_fact_read_rate = 1.0
///
/// If the OFF run does not reach 1.0, the metric is dead and the ON zero means
/// nothing. If ON is not 0.0, the guard leaks. Only ON=0 with OFF=1 licenses
/// reading the production zero as a fired guard.
pub fn run_poison(budget: usize) -> Result<(), DcrError> {
    fn scenario(budget: usize, admit_stale: bool) -> Result<f64, DcrError> {
        let mut rt = Dcr::new(budget);
        rt.planner.admit_stale = admit_stale;
        rt.register("incident_cost", |inputs| {
            let get = |n: &str| {
                inputs
                    .iter()
                    .find(|(k, _)| k == n)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0)
            };
            get("rate") * get("hours")
        });
        rt.ingest("The hourly rate is 180 USD.", Some("t1"))?;
        rt.ingest("The incident lasted 4 hours.", Some("t2"))?;
        let rate = *rt.graph.by_key("hourly.rate", true).last().unwrap();
        let deps = vec![rt.graph.node(rate).id.clone()];
        rt.compute(
            "incident_cost",
            vec![("rate".into(), 180.0), ("hours".into(), 4.0)],
            deps,
            Some("incident.cost"),
        )?;
        // Poison: correct the input. The derivation now depends on a superseded
        // fact, so it is marked stale.
        rt.ingest("Correction: the hourly rate is 210 USD.", Some("t3"))?;
        let cost = rt.graph.by_key("incident.cost", true)[0];
        assert_eq!(
            rt.graph.node(cost).status.as_str(),
            "stale",
            "the derivation must be stale for this control to mean anything"
        );
        let mut reasoner = LocalReasoner::new();
        rt.ask_with("what is the incident cost?", None, &mut reasoner);
        Ok(rt.telemetry.report().stale_fact_read_rate.unwrap_or(0.0))
    }

    let line = "-".repeat(68);
    println!("{line}");
    println!("STALE-FACT POSITIVE CONTROL");
    println!("{line}");
    println!("scenario: a derived value goes stale after its input is corrected,");
    println!("          then a probe asks for exactly that value.\n");
    println!(
        "  {:<26} {:>10} {:>10} {:>8}",
        "run", "expected", "measured", "verdict"
    );
    println!("  {}", "-".repeat(58));
    let on = scenario(budget, false)?;
    let off = scenario(budget, true)?;
    let row = |name: &str, expected: f64, measured: f64| {
        let ok = (expected - measured).abs() < 1e-9;
        println!(
            "  {:<26} {:>10.1} {:>10.1} {:>8}",
            name,
            expected,
            measured,
            if ok { "PASS" } else { "FAIL" }
        );
        ok
    };
    let a = row("guard ON (production)", 0.0, on);
    let b = row("guard OFF (bypassed)", 1.0, off);
    println!("{line}");
    if a && b {
        println!("Both controls pass: the metric CAN return nonzero, and the guard drove");
        println!("it to zero. The production 0.0 is a fired guard, not an unexercised one.");
    } else {
        println!("A control failed: do not read the production zero as evidence.");
    }
    println!("{line}");
    Ok(())
}

/// Read coverage across a growing history — the offline, unconditioned metric.
///
/// Answers the question the probe-based table cannot: as `N` grows, does the
/// set of spans ever assembled grow with it, or stay flat while the unread
/// region grows linearly? Coverage is measured after replaying the fixed probe
/// set at each size.
pub fn run_coverage(sizes: &[usize], budget: usize) -> Result<(), DcrError> {
    println!(
        "{:>7} {:>9} {:>9} {:>10} {:>12} {:>10} {:>9}",
        "turns", "spans N", "assembled", "coverage", "never seen", "dep pairs", "co-seen"
    );
    println!("{}", "-".repeat(74));
    let mut first: Option<(usize, usize)> = None;
    let mut last = (0usize, 0usize, 0usize);
    for &turns in sizes {
        let corpus = build_corpus(turns);
        let mut rt = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            rt.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        for probe in &corpus.probes {
            rt.ask_with(probe.query, None, &mut reasoner);
        }
        let cov = rt.coverage();
        let pc = rt.pair_coverage();
        println!(
            "{turns:>7} {:>9} {:>9} {:>9.1}% {:>12} {:>10} {:>8.1}%",
            cov.total_spans,
            cov.assembled_spans,
            cov.fraction * 100.0,
            cov.total_spans - cov.assembled_spans,
            pc.linked_pairs,
            pc.fraction * 100.0
        );
        if first.is_none() {
            first = Some((cov.total_spans, cov.assembled_spans));
        }
        last = (
            cov.total_spans,
            cov.assembled_spans,
            cov.total_spans - cov.assembled_spans,
        );
    }
    println!("{}", "-".repeat(74));
    if let Some((n0, a0)) = first {
        let n_growth = last.0 as f64 / n0.max(1) as f64;
        let a_growth = last.1 as f64 / a0.max(1) as f64;
        println!("history (N) grew {n_growth:.0}x; spans ever shown at L0 grew {a_growth:.1}x.",);
        println!("Coverage counts L0 only: the sole level that renders a span's actual bytes.");
        println!(
            "'dep pairs' are span pairs joined by a dependency edge — where co-occurrence is\n\
             load-bearing — and 'co-seen' is the fraction ever rendered in the *same* window.\n\
             It collapses faster than single-span coverage and its denominator grows faster than\n\
             N, so the one-dimensional number is the optimistic view: a span can be covered while\n\
             every pairing that made it matter never once co-occurred."
        );
        println!("A fixed probe set surfaces a roughly constant handful of spans, so the count");
        println!(
            "stays flat while the unread region ({} spans here) grows with the history and",
            last.2
        );
        println!("the covered fraction collapses toward zero. This is the dual cost of bounded");
        println!("attention that the probe-based table cannot show: the runtime answers from");
        println!("compact facts without ever surfacing most of the history's actual content, so");
        println!("a span whose specifics silently governed an answer can never appear in a probe.");
    }
    Ok(())
}

/// Does `k` stay flat while `N` grows? That is the whole claim.
pub fn run_scaling(sizes: &[usize], budget: usize) -> Result<(), DcrError> {
    println!(
        "{:>7} {:>9} {:>7} {:>8} {:>7} {:>8} {:>8} {:>9} {:>9} {:>8}",
        "turns", "history", "nodes", "mean k", "max k", "correct", "ingest",
        "query", "ann query", "ann k"
    );
    println!("{}", "-".repeat(88));
    let mut first: Option<(usize, f64)> = None;
    let mut last: (usize, f64) = (0, 0.0);
    for &turns in sizes {
        let corpus = build_corpus(turns);
        let started = Instant::now();
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let ingest_s = started.elapsed().as_secs_f64();

        let started = Instant::now();
        // Fresh per size: node ids are content-derived, so a shared reasoner
        // would carry escalation memory across runs.
        let mut reasoner = LocalReasoner::new();
        let mut correct = 0usize;
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            if answer
                .text
                .to_lowercase()
                .contains(&probe.expected.to_lowercase())
            {
                correct += 1;
            }
        }
        let query_ms = started.elapsed().as_secs_f64() * 1000.0 / corpus.probes.len() as f64;
        let report = runtime.telemetry.report();

        // Same corpus, approximate retrieval. Reported beside the exact path
        // rather than replacing it: LSH buys latency and costs a little
        // attention, and the trade is the reader's to judge.
        let mut ann = Dcr::new(budget);
        ann.index.set_exact(false);
        for (doc_id, text) in &corpus.docs {
            ann.ingest(text, Some(doc_id))?;
        }
        let mut ann_reasoner = LocalReasoner::new();
        let ann_started = Instant::now();
        let mut ann_correct = 0usize;
        for probe in &corpus.probes {
            let a = ann.ask_with(probe.query, None, &mut ann_reasoner);
            if a.text
                .to_lowercase()
                .contains(&probe.expected.to_lowercase())
            {
                ann_correct += 1;
            }
        }
        let ann_ms = ann_started.elapsed().as_secs_f64() * 1000.0 / corpus.probes.len() as f64;
        let ann_report = ann.telemetry.report();
        // Not debug_assert: every benchmark here runs --release, where
        // debug assertions are compiled out, so this check would have been a
        // control that cannot fire sitting inside the table that reports the
        // result. If the approximate path ever diverges, the row is wrong and
        // should not be printed.
        assert_eq!(
            ann_correct, correct,
            "approximate retrieval changed correctness at {turns} turns \
             ({ann_correct} vs {correct}); Table 4's equality claim is void"
        );
        let history = estimate_tokens(&corpus.text());
        println!(
            "{turns:>7} {history:>9} {:>7} {:>8.1} {:>7} {:>8} {:>7.2}s {:>7.1}ms {:>7.1}ms {:>8.1}",
            runtime.graph.len(),
            report.tokens_per_query_mean,
            report.tokens_per_query_max,
            format!("{correct}/{}", corpus.probes.len()),
            ingest_s,
            query_ms,
            ann_ms,
            ann_report.tokens_per_query_mean
        );
        if first.is_none() {
            first = Some((history, report.tokens_per_query_mean));
        }
        last = (history, report.tokens_per_query_mean);
    }
    println!("{}", "-".repeat(88));
    if let Some((first_history, first_k)) = first {
        println!(
            "history grew {:.0}x; active context grew {:.2}x  <- the O(k + r) claim",
            last.0 as f64 / first_history.max(1) as f64,
            last.1 / first_k.max(1.0)
        );
    }
    println!(
        "query latency is still not flat. 'query' scores every vector; 'ann query' prunes with\n\
         random-hyperplane LSH, which is roughly an order of magnitude faster at the largest size\n\
         with identical correctness, at the cost of a slightly larger working set — approximate\n\
         seeding admits a slightly different candidate set. Profiling the remainder shows the\n\
         retrieval step is no longer what grows: with ~96% of vectors pruned the index call is\n\
         a small fraction of a query and the cost has moved into planning, so 'the vector search\n\
         is a linear scan' is no longer the reason latency scales."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Mutation-and-correction probe
// ---------------------------------------------------------------------------

/// One mutation case: a fact planted early, referenced repeatedly so it
/// accumulates dependents, then superseded late.
///
/// The point of the repeated references is that a correction arriving at 85% of
/// the way through has to overcome whatever weight the original accumulated —
/// graph proximity, read-through, and the dependents built on top of it. A
/// correction to an isolated leaf is the easy case and proves little.
pub struct Mutation {
    pub label: &'static str,
    pub query: &'static str,
    /// Ground truth: the value that must no longer be served.
    pub stale: &'static str,
    /// Ground truth: the value that must be served instead.
    pub live: &'static str,
    pub establish: &'static str,
    pub references: &'static [&'static str],
    pub correction: &'static str,
}

pub const MUTATIONS: &[Mutation] = &[
    Mutation {
        label: "datastore (4 dependents)",
        query: "what is the primary datastore?",
        stale: "postgres-11",
        live: "postgres-15",
        establish: "The primary datastore is postgres-11 on host db-alpha.",
        references: &[
            "Nightly backup job targets the primary datastore postgres-11 on db-alpha; retention is 14 days.",
            "The read replica lags the primary datastore postgres-11 by under 200ms during business hours.",
            "Schema migration 0042 was applied against postgres-11 and verified on the replica.",
            "Capacity note: postgres-11 on db-alpha is at 61 percent disk with no growth alarm set.",
        ],
        correction: "Correction: the primary datastore was migrated and is now postgres-15 on host db-omega.",
    },
    Mutation {
        label: "autoscaler threshold (3 dependents)",
        query: "what is the autoscaler threshold?",
        stale: "65 percent",
        live: "80 percent",
        establish: "The autoscaler threshold is 65 percent CPU.",
        references: &[
            "Load test 19 held the fleet just under the autoscaler threshold of 65 percent CPU for forty minutes.",
            "The scale-out event at 03:12 fired because sustained CPU crossed 65 percent.",
            "Cost review flagged that a 65 percent trigger keeps roughly two extra nodes warm overnight.",
        ],
        correction: "Update: the autoscaler threshold is 80 percent CPU after the cost review.",
    },
    Mutation {
        label: "failover region (3 dependents)",
        query: "what is the failover region?",
        stale: "eu-west-2",
        live: "eu-central-1",
        establish: "The failover region is eu-west-2.",
        references: &[
            "The disaster recovery runbook fails traffic over to eu-west-2 and expects a 12 minute RTO.",
            "Cross-region replication to eu-west-2 was re-enabled after the maintenance window.",
            "The last failover drill exercised eu-west-2 and completed inside the RTO budget.",
        ],
        correction: "Correction: the failover region is eu-central-1, eu-west-2 is being decommissioned.",
    },
    Mutation {
        label: "escalation extension (2 dependents)",
        query: "what is the escalation extension?",
        stale: "4412",
        live: "4419",
        establish: "The escalation extension is 4412 for the platform on-call rota.",
        references: &[
            "Page at 02:40 was routed to extension 4412 and acknowledged in ninety seconds.",
            "The incident bridge dials extension 4412 before opening a severity review.",
        ],
        correction: "Update: the escalation extension is 4419, the rota moved to a new bridge.",
    },
];

/// The adversarial set: same four facts, worded so the *superseded* value is
/// the more attractive retrieval target.
///
/// Proposed by a reviewer who pointed out that [`MUTATIONS`] does not
/// distinguish two hypotheses. There the correction is phrased much like the
/// original, so both score alike on overlap with the query and a clean result is
/// equally consistent with "the planner respected the supersession edge" and
/// "the planner picked up the newer text because nothing pulled the other way".
///
/// Here the stale line repeats the query's own terms and the correction states
/// the key once and paraphrases the rest, so lexical attraction points at the
/// wrong answer. The subject key still matches — supersession has to be able to
/// fire — but nothing else helps. A pass means the edge won something.
pub const ADVERSARIAL: &[Mutation] = &[
    Mutation {
        label: "datastore (stale is lexically closer)",
        query: "what is the primary datastore host serving checkout reads?",
        stale: "postgres-11",
        live: "postgres-15",
        establish: "The primary datastore is postgres-11 on host db-alpha, serving all checkout reads.",
        references: &[
            "Backups target the primary datastore on host db-alpha serving checkout reads nightly.",
            "The primary datastore host db-alpha serves checkout reads at 61 percent disk.",
            "Replica lag against the primary datastore host serving checkout reads is under 200ms.",
            "Schema migration 0042 ran on the primary datastore host serving checkout reads.",
        ],
        correction: "Correction: the primary datastore is postgres-15.",
    },
    Mutation {
        label: "threshold (stale is lexically closer)",
        query: "what is the autoscaler threshold for sustained cpu scale-out?",
        stale: "65 percent",
        live: "80 percent",
        establish: "The autoscaler threshold is 65 percent for sustained cpu scale-out events.",
        references: &[
            "Sustained cpu crossing the autoscaler threshold triggers a scale-out event.",
            "Load test 19 held sustained cpu just under the autoscaler threshold scale-out point.",
            "Cost review flagged the autoscaler threshold for sustained cpu scale-out as expensive.",
        ],
        correction: "Update: the autoscaler threshold is 80 percent.",
    },
    Mutation {
        label: "region (stale is lexically closer)",
        query: "what is the failover region in the disaster recovery runbook?",
        stale: "eu-west-2",
        live: "eu-central-1",
        establish: "The failover region is eu-west-2 in the disaster recovery runbook.",
        references: &[
            "The disaster recovery runbook fails over to the failover region with a 12 minute RTO.",
            "Cross-region replication to the failover region in the disaster recovery runbook resumed.",
            "The last drill exercised the failover region named in the disaster recovery runbook.",
        ],
        correction: "Correction: the failover region is eu-central-1.",
    },
    Mutation {
        label: "extension (stale is lexically closer)",
        query: "what is the escalation extension for the platform oncall rota bridge?",
        stale: "4412",
        live: "4419",
        establish: "The escalation extension is 4412 for the platform oncall rota bridge.",
        references: &[
            "Paging routes to the escalation extension for the platform oncall rota bridge.",
            "The incident bridge dials the escalation extension of the platform oncall rota.",
        ],
        correction: "Update: the escalation extension is 4419.",
    },
];

/// Interleave the mutation cases into a long transcript: each fact is
/// established early, referenced across the first 60%, and corrected at 85%.
pub fn build_mutation_corpus(turns: usize) -> Corpus {
    build_mutation_corpus_from(turns, MUTATIONS)
}

/// As [`build_mutation_corpus`], for any mutation set.
pub fn build_mutation_corpus_from(turns: usize, set: &'static [Mutation]) -> Corpus {
    let mut docs: Vec<(String, String)> = Vec::new();
    for (m_idx, m) in set.iter().enumerate() {
        docs.push((format!("m{m_idx:02}e"), m.establish.to_string()));
    }

    let established = docs.len();
    let correct_at = (turns as f64 * 0.85) as usize;
    // References are spread across the first 60% so the originals are still
    // accumulating dependents well after they were planted.
    let ref_span_end = (turns as f64 * 0.60) as usize;
    let mut refs: Vec<(usize, &str)> = Vec::new();
    let total_refs: usize = set.iter().map(|m| m.references.len()).sum();
    let mut slot = 0usize;
    for m in set {
        for r in m.references {
            let at = established
                + ((ref_span_end.saturating_sub(established)) * slot) / total_refs.max(1);
            refs.push((at.max(established), r));
            slot += 1;
        }
    }

    for i in established..turns.max(established + 1) {
        let id = format!("t{i:03}");
        // corrections land in consecutive slots starting at 85%
        let corr = set
            .iter()
            .enumerate()
            .find(|(k, _)| correct_at + k == i)
            .map(|(_, m)| m.correction);
        if let Some(text) = corr {
            docs.push((id, text.to_string()));
            continue;
        }
        let planted: Vec<&str> = refs
            .iter()
            .filter(|(at, _)| *at == i)
            .map(|(_, r)| *r)
            .collect();
        if !planted.is_empty() {
            docs.push((id, planted.join("\n\n")));
            continue;
        }
        docs.push((id, NOISE[i % NOISE.len()].replace("{n}", &i.to_string())));
    }

    Corpus {
        docs,
        probes: Vec::new(),
    }
}

/// Measure what the main benchmark cannot: whether a correction is actually
/// served once the original has dependents, and whether the runtime can show
/// the supersession edge that justifies it.
///
/// Both measurements are made against ground truth held by the corpus, not
/// against the runtime's own `Status::Stale` marking. That distinction is the
/// whole point. `stale_fact_read_rate` counts entries whose node the runtime
/// has *marked* stale, so it cannot fire in the configuration where marking is
/// switched off — a control that cannot fire and a control that passes are
/// indistinguishable. Checking the answer text for a value the corpus knows was
/// superseded fires in every configuration, so the `no supersession` row below
/// is a live negative control rather than a decorative one.
pub fn run_mutation_probe(turns: usize, budget: usize) -> Result<(), DcrError> {
    run_mutation_set(turns, budget, MUTATIONS, "MUTATION AND CORRECTION")?;
    println!();
    run_mutation_set(
        turns,
        budget,
        ADVERSARIAL,
        "ADVERSARIAL - the superseded value is the lexically closer match",
    )
}

fn run_mutation_set(
    turns: usize,
    budget: usize,
    set: &'static [Mutation],
    heading: &str,
) -> Result<(), DcrError> {
    struct Variant {
        name: &'static str,
        apply: fn(&mut Dcr),
    }
    let variants = [
        Variant {
            name: "full runtime",
            apply: |_| {},
        },
        Variant {
            name: "no supersession",
            apply: |rt| rt.indexer.supersede_on_conflict = false,
        },
        Variant {
            name: "no reference linking",
            apply: |rt| rt.indexer.reference_linking = false,
        },
        Variant {
            name: "no graph expansion",
            apply: |rt| rt.planner.max_depth = 0,
        },
    ];

    let corpus = build_mutation_corpus_from(turns, set);
    let n = set.len();
    println!(
        "{heading} - {turns} turns, {} tokens of history, B_attention = {budget}",
        estimate_tokens(&corpus.text())
    );
    println!(
        "{n} facts established early, referenced {} times in total, superseded at 85% of history",
        set.iter().map(|m| m.references.len()).sum::<usize>()
    );
    let header = format!(
        "{:<24} {:>10} {:>13} {:>11} {:>11} {:>8}   {}",
        "variant", "corrected", "stale served", "edge shown", "guard fired", "stale k", "notes"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut control_fired = false;
    let mut served: Vec<(&str, String, usize)> = Vec::new();
    for variant in variants {
        let mut runtime = Dcr::new(budget);
        (variant.apply)(&mut runtime);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let (mut corrected, mut stale, mut edges, mut guarded) = (0usize, 0usize, 0usize, 0usize);
        let mut stale_cases: Vec<&str> = Vec::new();
        let mut missed: Vec<&str> = Vec::new();
        let mut no_edge: Vec<&str> = Vec::new();

        for m in set {
            let answer = runtime.ask_with(m.query, None, &mut reasoner);
            let text = answer.text.to_lowercase();
            let live_v = m.live.to_lowercase();
            let stale_v = m.stale.to_lowercase();

            if text.contains(&live_v) {
                corrected += 1;
            } else if !text.contains(&stale_v) {
                // Neither value: the planner served something unrelated, which
                // is a different failure from serving the superseded value and
                // is worth separating.
                missed.push(m.label);
            }
            if text.contains(&stale_v) {
                stale += 1;
                stale_cases.push(m.label);
            }
            // Can the runtime show *why* the live value wins? A cited node must
            // supersede a node that still carries the stale value.
            // Was the stale node retrieved and then rejected *by supersession*,
            // or did it simply never surface? A reviewer's point: a pass can be
            // correct through an unobserved shortcut and look identical to the
            // mechanism under test. `stale_seen` is what the planner skipped for
            // staleness, so it distinguishes "the guard fired" from "the guard
            // was never reached".
            let rejected_by_guard = answer.context.stale_seen.iter().any(|idx| {
                runtime
                    .graph
                    .nodes()
                    .get(usize::from(*idx))
                    .is_some_and(|n| n.value.to_lowercase().contains(&stale_v))
            });
            if rejected_by_guard {
                guarded += 1;
            }
            let shown = answer.cited.iter().any(|id| {
                runtime.graph.get(id).is_some_and(|node| {
                    node.meta.supersedes.iter().any(|old| {
                        runtime
                            .graph
                            .get(old)
                            .is_some_and(|o| o.value.to_lowercase().contains(&stale_v))
                    })
                })
            });
            if shown {
                edges += 1;
            } else if text.contains(&live_v) {
                // Served the right value but cannot point at the edge that
                // justifies it — a provenance gap, not a retrieval one.
                no_edge.push(m.label);
            }
            if variant.name == "full runtime" {
                served.push((m.label, answer.text.clone(), answer.tokens));
            }
        }

        let report = runtime.telemetry.report();
        let marked = match report.stale_fact_read_rate {
            Some(v) => format!("{v:.2}"),
            None => "n/a".to_string(),
        };
        if variant.name == "no supersession" && stale > 0 {
            control_fired = true;
        }
        println!(
            "{:<24} {:>8}/{n} {:>11}/{n} {:>9}/{n} {:>9}/{n} {:>8}   {}",
            variant.name,
            corrected,
            stale,
            edges,
            guarded,
            marked,
            {
                let mut notes = Vec::new();
                for c in &stale_cases {
                    notes.push(format!("STALE {c}"));
                }
                for c in &missed {
                    notes.push(format!("no answer {c}"));
                }
                for c in &no_edge {
                    notes.push(format!("no edge {c}"));
                }
                if notes.is_empty() {
                    "-".to_string()
                } else {
                    notes.join("; ")
                }
            }
        );
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "policy digest: {}  — the weights, seed floor, depth/fan-out/candidate caps and control\n\
         flags that decided these rejections. A pass is attributable to a named version of the\n\
         rule rather than to whatever the policy happens to be when you read this.",
        Dcr::new(budget).planner.policy_digest()
    );
    println!(
        "'stale k' is the runtime's own stale_fact_read_rate, shown for comparison: it stays 0\n\
         even on the row where superseded values are provably served, because disabling\n\
         supersession means nothing is ever marked. The 'stale served' column is ground truth."
    );
    println!("\nwhat the full runtime served:");
    for (label, text, tokens) in &served {
        let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let shown: String = one_line.chars().take(96).collect();
        println!("  {label:<32} {tokens:>4} tok  {shown}");
    }
    println!();
    if control_fired {
        println!("negative control FIRED: the instrument can distinguish a pass from a no-op.");
    } else {
        println!(
            "negative control DID NOT FIRE: 'no supersession' served no stale value, so this\n\
             run does not establish that the probe can detect the failure it tests for."
        );
    }
    Ok(())
}

// -- the wider baseline set ------------------------------------------------

/// DCR against the retrieval baselines, not only the truncation ones.
///
/// The existing table compares against full context and a sliding window: both
/// lose material by *position*, which is the failure DCR was built for, so
/// winning there proves less than it looks. This adds three assemblies that
/// also retrieve — plain RAG, uniform summarisation, and the recursive
/// map-reduce shape — at the same budget and under the same reasoner.
///
/// The table is capable of showing DCR losing. If a flat top-k retriever
/// answers these probes at this budget, the graph and the ladder are not
/// paying for themselves, and that is what the numbers are for.
pub fn run_baselines(turns: usize, budget: usize) -> Result<(), DcrError> {
    println!("STANDARD CORPUS — the probes the headline table uses");
    baseline_table(&build_corpus(turns), turns, budget)?;
    println!();
    println!("DISCRIMINATING CORPUS — similarity is misleading, and refusing is sometimes correct");
    baseline_table(&build_adversarial_corpus(turns), turns, budget)?;
    println!(
        "  Two probes are passed by *declining*: a stale derivation whose inputs were\n  \
         corrected, and a fact nobody ever stated. DCR fails both, and the reasons are\n  \
         worth more than the score. The stale derivation is served because invalidation\n  \
         only tracks derivations the runtime actually computed — a figure stated as text\n  \
         is never recomputed. The absent fact is served because a window pre-filled with\n  \
         selected facts makes near-miss material easy to reach for.\n  \
         The contested probe fails for a *documented* reason: two disagreeing claims\n  \
         where neither is phrased as a correction are resolved by ingest order, not\n  \
         marked as a dispute (`may_supersede`). Last-writer-wins is the right default\n  \
         for a chronological transcript — the cost is that two sources disagreeing are\n  \
         silently settled by arrival time, and that cost is what this row prices."
    );
    Ok(())
}

fn baseline_table(corpus: &Corpus, turns: usize, budget: usize) -> Result<(), DcrError> {
    let full_text = corpus.text();

    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id))?;
    }

    let rag = Rag::new(&corpus.docs, budget);
    let summarizer = SummarizeAll::new(&corpus.docs, budget);
    let recursive = Recursive::new(&corpus.docs, 8);

    let mut baseline = BaselineReasoner;
    let mut reasoner = LocalReasoner::new();

    struct Score {
        correct: usize,
        tokens: usize,
    }
    let mut scores: Vec<(&str, Score)> = ["full context", "RAG (top-k)", "summarize-all", "recursive", "DCR"]
        .into_iter()
        .map(|name| (name, Score { correct: 0, tokens: 0 }))
        .collect();

    let line = "-".repeat(88);
    println!("{line}");
    println!(
        "{:<32}{:>10}{:>10}{:>14}{:>10}{:>10}",
        "probe", "full", "RAG", "summarize", "recurse", "DCR"
    );
    println!("{line}");

    for probe in &corpus.probes {
        let full = baseline.complete(&format!("{full_text}\n\nQUESTION: {}", probe.query), "");
        let (rag_ctx, rag_tokens) = rag.assemble(probe.query);
        let rag_answer = baseline.complete(&format!("{rag_ctx}\n\nQUESTION: {}", probe.query), "");
        let (sum_ctx, sum_tokens) = summarizer.assemble(probe.query);
        let sum_answer = baseline.complete(&format!("{sum_ctx}\n\nQUESTION: {}", probe.query), "");
        let (rec_answer, rec_tokens) = recursive.answer(probe.query, &mut baseline);
        let dcr = runtime.ask_with(probe.query, None, &mut reasoner);

        // `Probe::scores` handles both directions: a refusal probe passes when
        // the value is absent, so a system that answers everything confidently
        // is penalised rather than rewarded. A context-scored probe is judged
        // on what each system assembled, uniformly — the harness's line-matcher
        // cannot perform a join, and scoring its output would measure the
        // stand-in reasoner instead of the assembly.
        let dcr_context = dcr.context.render_with_header(false);
        let judged = |answer: &str, context: &str| {
            probe.scores(if probe.on_context { context } else { answer })
        };
        let outcomes = [
            (
                judged(&full, &full_text),
                estimate_tokens(&full_text),
            ),
            (judged(&rag_answer, &rag_ctx), rag_tokens),
            (judged(&sum_answer, &sum_ctx), sum_tokens),
            // Recursion reads every chunk, so its "context" is the whole corpus.
            (judged(&rec_answer, &full_text), rec_tokens),
            (judged(&dcr.text, &dcr_context), dcr.tokens),
        ];
        for ((_, score), (correct, tokens)) in scores.iter_mut().zip(outcomes.iter()) {
            score.correct += usize::from(*correct);
            score.tokens += tokens;
        }
        let mark = |ok: bool| if ok { "ok" } else { "MISS" };
        println!(
            "{:<32}{:>10}{:>10}{:>14}{:>10}{:>10}",
            probe.label,
            mark(outcomes[0].0),
            mark(outcomes[1].0),
            mark(outcomes[2].0),
            mark(outcomes[3].0),
            mark(outcomes[4].0)
        );
    }

    println!("{line}");
    let probes = corpus.probes.len().max(1);
    print!("{:<32}", "correct");
    for (_, score) in &scores {
        print!("{:>10}", format!("{}/{}", score.correct, probes));
    }
    println!();
    print!("{:<32}", "mean tokens per query");
    for (_, score) in &scores {
        print!("{:>10}", format!("{:.0}", score.tokens as f64 / probes as f64));
    }
    println!();
    println!("{line}");
    println!(
        "{} turns \u{b7} {} tokens of history \u{b7} B_attention = {budget}",
        turns,
        estimate_tokens(&full_text)
    );
    println!(
        "  Same deterministic reasoner across every column, so this compares context\n  \
         assemblies rather than models. RAG and summarize-all get the same budget DCR\n  \
         does; recursive is charged for every chunk it reads, which is the cost its\n  \
         unlimited coverage actually has."
    );
    Ok(())
}

/// Is "destroy and rebuild the workspace" cheap enough to be a real guarantee?
///
/// The specification claims the working set can be discarded and reconstructed
/// from memory at any time. That is only a useful invariant if rebuilding is
/// cheaper than never having discarded it, so this measures the rebuild rather
/// than asserting it: cold assembly, then the same assembly with the ladder's
/// caches warm.
pub fn run_rebuild(turns: usize, budget: usize) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id))?;
    }

    let line = "-".repeat(88);
    println!("{line}");
    println!(
        "{:<34}{:>12}{:>12}{:>12}{:>14}",
        "probe", "cold (ms)", "warm (ms)", "nodes", "L1/L2/L3"
    );
    println!("{line}");

    let mut cold_total = 0f64;
    let mut warm_total = 0f64;
    for probe in &corpus.probes {
        // Cold is the whole destroy-and-rebuild: caches dropped *and* the
        // working set reassembled. Timing a plan after `rebuild_workspace`
        // would time a warm one, since that call has already replanned.
        let started = Instant::now();
        let report = runtime.rebuild_workspace(probe.query, None);
        let cold = started.elapsed().as_secs_f64() * 1000.0;

        // Warm: the same query again, with every representation now cached.
        let started = Instant::now();
        let context = runtime.plan(probe.query, None);
        let warm = started.elapsed().as_secs_f64() * 1000.0;

        cold_total += cold;
        warm_total += warm;
        println!(
            "{:<34}{:>12.2}{:>12.2}{:>12}{:>14}",
            probe.label,
            cold,
            warm,
            context.entries.len(),
            format!(
                "{}/{}/{}",
                report.rebuilt_l1, report.rebuilt_l2, report.rebuilt_l3
            )
        );
    }

    println!("{line}");
    let probes = corpus.probes.len().max(1) as f64;
    println!(
        "mean cold rebuild {:.2}ms \u{b7} mean warm assembly {:.2}ms \u{b7} {} nodes, {} spans",
        cold_total / probes,
        warm_total / probes,
        runtime.graph.len(),
        runtime.raw.len()
    );
    println!(
        "  Cold is the real cost of the guarantee: every cached representation dropped,\n  \
         then the working set reassembled from L0 alone. Warm is the same query with\n  \
         the caches populated. The gap is what the level cache buys, and it scales with\n  \
         how much L1 has to be rebuilt — the probes that admit only cached facts rebuild\n  \
         nothing and cost the same either way, while the one that pulls long spans pays\n  \
         the most. Rebuild is bounded by the working set, not by history: k nodes, not N."
    );
    Ok(())
}

/// The security claims, made falsifiable.
///
/// Each case corrupts a container in a specific way and asserts the runtime
/// notices. A claim that cannot fail in a test is not a claim, and "tamper
/// evident" is exactly the sort of property that reads as true until someone
/// checks.
pub fn run_tamper(budget: usize) -> Result<(), DcrError> {
    let dir = std::env::temp_dir().join(format!("dcr-tamper-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut runtime = Dcr::new(budget);
    runtime.ingest(
        "The server ip is 10.0.4.12 and the port is 8080.\n\n\
         Decision: roll back to build 4471 because the blocker is firewall rule 37.",
        Some("t1"),
    )?;
    runtime.ingest("Correction: the server ip is 10.0.9.7.", Some("t2"))?;
    runtime.save_context(&dir, None)?;

    let line = "-".repeat(88);
    println!("{line}");
    println!("{:<44}{:>12}   {:<26}", "attack", "detected", "how");
    println!("{line}");

    let report = |attack: &str, detected: bool, how: &str| {
        println!(
            "{:<44}{:>12}   {:<26}",
            attack,
            if detected { "yes" } else { "NO" },
            how
        );
        detected
    };
    let mut all = true;

    // 1 — flip a bit in a stored object.
    {
        let store = ContextStore::open(&dir).map_err(|e| DcrError::Parse(e.to_string()))?;
        let victim = store.object_ids().next().cloned().unwrap_or_default();
        let path = store.object_path(&victim);
        let original = std::fs::read(&path).map_err(|e| DcrError::Io(e.to_string()))?;
        let mut bytes = original.clone();
        let at = bytes.len() / 2;
        bytes[at] ^= 0x01;
        std::fs::write(&path, &bytes).map_err(|e| DcrError::Io(e.to_string()))?;

        all &= report(
            "flip one bit in an object",
            store.verify(None).objects_failed.contains(&victim),
            "content address",
        );
        all &= report(
            "…and the runtime refuses to load it",
            Dcr::open_context(&dir, budget).is_err(),
            "gateway",
        );
        std::fs::write(&path, &original).map_err(|e| DcrError::Io(e.to_string()))?;
    }

    // 2 — rewrite history: edit an old checkpoint and leave the rest alone.
    {
        let path = dir.join("checkpoints").join("000001");
        let original = std::fs::read_to_string(&path).map_err(|e| DcrError::Io(e.to_string()))?;
        let forged = original.replace("\"object_count\":", "\"object_count\": 1, \"ignored\":");
        std::fs::write(&path, &forged).map_err(|e| DcrError::Io(e.to_string()))?;
        all &= report(
            "rewrite a historical checkpoint",
            ContextStore::open(&dir).is_err(),
            "hash chain",
        );
        std::fs::write(&path, &original).map_err(|e| DcrError::Io(e.to_string()))?;
    }

    // 3 — roll back to an older, internally consistent state.
    {
        let mut runtime = Dcr::open_context(&dir, budget)?;
        runtime.ingest("The retry budget is 5 attempts per minute.", Some("t3"))?;
        let sealed = runtime.save_context(&dir, None)?;
        let newest = dir
            .join("checkpoints")
            .join(format!("{:06}", sealed.generation));
        let manifest_path = dir.join("manifest");
        let manifest = std::fs::read_to_string(&manifest_path)
            .map_err(|e| DcrError::Io(e.to_string()))?;
        let rolled = manifest.replace(
            &format!("\"highest_generation\":{}", sealed.generation),
            &format!("\"highest_generation\":{}", sealed.generation - 1),
        );
        std::fs::remove_file(&newest).map_err(|e| DcrError::Io(e.to_string()))?;
        std::fs::write(&manifest_path, &rolled).map_err(|e| DcrError::Io(e.to_string()))?;

        all &= report(
            "roll back to an older signed state",
            matches!(
                ContextStore::open(&dir),
                Err(crate::context_store::ContextError::Rollback { .. })
            ),
            "generation high-water mark",
        );
    }

    println!("{line}");
    println!(
        "  What this does NOT show: resistance to an attacker who rewrites the objects,\n  \
         the chain, the manifest and the high-water mark together. Hashes make tampering\n  \
         evident, not impossible — signatures and an out-of-band high-water mark are what\n  \
         raise that bar, and neither is bundled."
    );
    let _ = std::fs::remove_dir_all(&dir);
    if !all {
        return Err(DcrError::Parse(
            "a tamper case went undetected".to_string(),
        ));
    }
    Ok(())
}

// -- the discriminating corpus --------------------------------------------

/// A corpus built to separate DCR from a good retriever.
///
/// On the standard corpus, plain top-k RAG ties full context at 5/7 — it
/// answers most probes because most probes are findable by similarity. That
/// makes the headline number look better than the evidence supports, and it
/// tells you almost nothing about whether the graph and the ladder are earning
/// their complexity.
///
/// These five probes are chosen so that similarity is actively *misleading*, or
/// so that answering at all is the wrong behaviour:
///
/// | probe | what it attacks |
/// |---|---|
/// | lexical decoy | the superseded value repeats the query's words; the correction does not |
/// | three-hop | the answer shares no vocabulary with the question |
/// | stale derivation | the tempting figure is arithmetically dead |
/// | absent fact | nothing in history answers it |
/// | contested fact | two claims disagree and neither is corrective |
///
/// Two of them (stale derivation, absent fact) are **refusal** probes: they are
/// passed by declining, and DCR can fail them exactly as easily as a baseline
/// can. One (contested fact) no truncation or top-k baseline can pass by
/// construction, and the table says so rather than quietly banking the point.
///
/// **Vocabulary hygiene matters more than it looks.** An earlier draft of this
/// corpus used "cache hit rate", "build" and "runbook" — all of which appear in
/// `NOISE`. The probes then measured a collision with the noise generator
/// rather than the property they named, and every column failed for the same
/// uninteresting reason. Subjects here are chosen to appear nowhere else in the
/// corpus.
pub fn build_adversarial_corpus(turns: usize) -> Corpus {
    let mut docs: Vec<(String, String)> = Vec::new();
    let mut add = |id: &str, text: &str| docs.push((id.to_string(), text.to_string()));

    add("a000", "Goal: bring checkout p99 latency under 250ms.");

    // 1. Lexical decoy. The stale document repeats the query's words four
    //    times; the correction states the same key once and moves on. Both
    //    carry the *same* key, so supersession can fire — which is the point.
    //    If the runtime is riding on lexical attraction rather than the
    //    supersession edge, it serves 91.
    add(
        "a001",
        "The shard rebalance ratio is 91 percent. The shard rebalance ratio was measured across \
         every shard rebalance window, and the shard rebalance ratio reported by the shard \
         rebalance controller agrees with the shard rebalance ratio in the ledger.",
    );

    // 2. Three-hop chain. "primary datastore" appears only in the first hop;
    //    the rack only in the third. Nothing shares vocabulary end to end.
    add("a002", "The primary datastore is postgres-11.");
    add("a003", "postgres-11 runs on host db-a-14.");
    add("a004", "Host db-a-14 is installed in rack R-19.");

    // 3. A derived figure that names its inputs, one of which is corrected
    //    later. After the correction 2160 is arithmetically dead, and serving
    //    it is wrong even though it is stated verbatim in history.
    add(
        "a005",
        "The engineer count is 3, the incident hours are 4, and the hourly rate is 180 USD.",
    );
    add(
        "a006",
        "The incident cost estimate is 2160 USD, computed from the engineer count, the incident \
         hours and the hourly rate.",
    );

    // 4. Adjacent material for the absent-fact probe: an escalation policy
    //    exists, a phone number does not.
    add(
        "a007",
        "The severity-1 paging ladder is defined in policy document 12, which covers escalation \
         order.",
    );

    // 5. Two live claims on the same key that disagree, neither phrased as a
    //    correction, so supersession must NOT fire and both stay live.
    // Phrased in the `subject is value` shape the scanner handles, so this
    // probe tests contradiction *marking* rather than extraction coverage.
    // The extractor's blind spots are a separate finding, recorded in TODO.md.
    add("a008", "The rollout outcome is reverted after readiness failures.");
    add("a009", "The rollout outcome is completed successfully across the fleet.");

    let fixed = docs.len();
    let corrections: [(usize, &str); 2] = [
        (
            (turns as f64 * 0.75) as usize,
            "Correction: the shard rebalance ratio is 62 percent.",
        ),
        (
            (turns as f64 * 0.85) as usize,
            "Correction: the hourly rate is 210 USD, finance updated the figure.",
        ),
    ];
    for i in fixed..turns.max(fixed + 1) {
        let id = format!("a{i:03}");
        match corrections.iter().find(|(at, _)| *at == i) {
            Some((_, text)) => docs.push((id, text.to_string())),
            None => docs.push((id, NOISE[i % NOISE.len()].replace("{n}", &i.to_string()))),
        }
    }

    let probes = vec![
        Probe::recall(
            "what is the shard rebalance ratio?",
            "62",
            "lexical decoy on a stale fact",
        ),
        Probe::assembled(
            "which rack is the primary datastore in?",
            "R-19",
            "three-hop dependency (context)",
        ),
        Probe::refuse(
            "what is the incident cost estimate?",
            "2160",
            "stale derivation (must refuse)",
        ),
        Probe::refuse(
            "what is the severity-1 pager phone number?",
            "policy document 12",
            "absent fact (must refuse)",
        ),
        Probe::assembled(
            "what is the rollout outcome?",
            "CONTRADICTS",
            "contested fact (order decides)",
        ),
    ];
    Corpus { docs, probes }
}

// ---------------------------------------------------------------------------
// Multi-hop probes
// ---------------------------------------------------------------------------

/// A probe whose answer is never stated beside the thing the query names.
///
/// The ablation shows graph expansion costs 46% of the working set and loses
/// nothing on the standard corpus, and reference linking has no effect at all.
/// Two readings fit that: the mechanisms are not load-bearing, or the corpus
/// never asks for a join. The standard probes cannot separate them, because
/// every answer sits in a document that also names the subject.
///
/// These do not. Each answer requires following a dependency edge from the term
/// the query uses to a term it never mentions, so a planner that cannot expand
/// along edges has to miss them.
pub const MULTI_HOP: &[Probe] = &[
    // Scored on the assembled context, not the answer. The harness reasoner is
    // a line-matcher and cannot chain A->B->C by construction, so scoring its
    // output would measure the stand-in model rather than whether the planner
    // brought the joining material into the window — which is the question.
    //
    // The construction matters more than the count. In a first attempt every
    // query named terms from both ends of its chain, so both facts were reachable
    // by direct lexical hit and no join was ever required — the probes passed
    // with graph expansion switched off, which proved nothing. Here the
    // second-hop fact shares no content token with the query at all
    // (`the_second_hop_is_lexically_unreachable` pins that), so the only route
    // to it is the edge from the first.
    Probe::assembled(
        "who should be paged for the readiness probe failures?",
        "team-payments",
        "owner via failing service (2 hops)",
    ),
    Probe::assembled(
        "is there capacity risk for checkout reads?",
        "61 percent",
        "disk via host via service (2 hops)",
    ),
    Probe::assembled(
        "what blocks build 4471?",
        "rule 37",
        "rule via subnet via build (2 hops)",
    ),
];

/// Facts for [`MULTI_HOP`], each stating exactly one link in a chain. The
/// first fact of each chain is lexically reachable from its query; the second
/// is not, and is reachable only by following the link.
const HOP_FACTS: &[&str] = &[
    "The readiness probe failures were traced to alpha-checkout.",
    "alpha-checkout is owned by team-payments.",
    "Checkout reads are served by host db-alpha.",
    "db-alpha disk headroom is 61 percent.",
    "Build 4471 serves the checkout subnet.",
    "Firewall rule 37 drops traffic to the checkout subnet.",
];

pub fn build_multi_hop_corpus(turns: usize) -> Corpus {
    let mut docs: Vec<(String, String)> = Vec::new();
    for (i, fact) in HOP_FACTS.iter().enumerate() {
        docs.push((format!("h{i:02}"), (*fact).to_string()));
    }
    let fixed = docs.len();
    for i in fixed..turns.max(fixed + 1) {
        docs.push((
            format!("t{i:03}"),
            NOISE[i % NOISE.len()].replace("{n}", &i.to_string()),
        ));
    }
    Corpus {
        docs,
        probes: MULTI_HOP.to_vec(),
    }
}

/// Does graph expansion buy anything when the answer needs a join?
pub fn run_multi_hop(turns: usize, budget: usize) -> Result<(), DcrError> {
    struct Variant {
        name: &'static str,
        apply: fn(&mut Dcr),
    }
    let variants = [
        Variant {
            name: "full runtime",
            apply: |_| {},
        },
        Variant {
            name: "no graph expansion",
            apply: |rt| rt.planner.max_depth = 0,
        },
        Variant {
            name: "no reference linking",
            apply: |rt| rt.indexer.reference_linking = false,
        },
    ];
    let corpus = build_multi_hop_corpus(turns);
    println!(
        "MULTI-HOP - {turns} turns, {} tokens of history, B_attention = {budget}",
        estimate_tokens(&corpus.text())
    );
    println!(
        "{} probes whose answer is never stated beside the term the query uses",
        corpus.probes.len()
    );
    let header = format!(
        "{:<24} {:>9} {:>9}   {}",
        "variant", "correct", "mean k", "probes that fail"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for variant in variants {
        let mut runtime = Dcr::new(budget);
        (variant.apply)(&mut runtime);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let (mut correct, mut failed) = (0usize, Vec::new());
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            let scored_on = if probe.on_context {
                answer.context.render()
            } else {
                answer.text.clone()
            };
            if probe.scores(&scored_on) {
                correct += 1;
            } else {
                failed.push(probe.label);
            }
        }
        println!(
            "{:<24} {:>7}/{} {:>9.1}   {}",
            variant.name,
            correct,
            corpus.probes.len(),
            runtime.telemetry.report().tokens_per_query_mean,
            if failed.is_empty() {
                "-".to_string()
            } else {
                failed.join("; ")
            }
        );
    }
    println!("{}", "-".repeat(header.len()));
    Ok(())
}

/// Does pruning the candidate set by recency buy latency without costing
/// correctness?
///
/// Suggested by a reader who measured a halving of latency at no accuracy cost
/// on their own 1.5k-node graph. It reproduces here as a latency win and does
/// not reproduce as free: moderate cutoffs break exactly the probes that reach
/// furthest back, which is the trade a single aggregate latency number hides.
///
/// The non-monotonicity is the part worth reading carefully. Correctness does
/// not fall as the cutoff rises — it falls, then recovers. The recovery is not
/// the filter succeeding: an aggressive cutoff starves the seed list, and a
/// starved seed list falls back to the raw lexical index, which finds the span
/// directly. So the best-looking row is the one where the mechanism under test
/// has been bypassed, and its latency belongs to the fallback rather than to
/// the filter.
pub fn run_decay(turns: usize, budget: usize) -> Result<(), DcrError> {
    let cutoffs = [0.0f32, 0.25, 0.5, 0.75];
    println!("RECENCY PREFILTER - {turns} turns, B_attention = {budget}");
    let header = format!(
        "{:>7} {:>9} {:>9} {:>10} {:>8}   {}",
        "cutoff", "correct", "mean k", "ms/query", "seeds", "probes that fail"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for cutoff in cutoffs {
        let corpus = build_corpus(turns);
        let mut runtime = Dcr::new(budget);
        runtime.planner.recency_cutoff = cutoff;
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let started = Instant::now();
        let (mut correct, mut failed, mut considered) = (0usize, Vec::new(), 0usize);
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            considered += answer.context.considered;
            let scored = if probe.on_context {
                answer.context.render()
            } else {
                answer.text.clone()
            };
            if probe.scores(&scored) {
                correct += 1;
            } else {
                failed.push(probe.label);
            }
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0 / corpus.probes.len() as f64;
        println!(
            "{cutoff:>7.2} {:>7}/{} {:>9.1} {:>8.2}ms {:>8.1}   {}",
            correct,
            corpus.probes.len(),
            runtime.telemetry.report().tokens_per_query_mean,
            ms,
            considered as f64 / corpus.probes.len() as f64,
            if failed.is_empty() {
                "-".to_string()
            } else {
                failed.join("; ")
            }
        );
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "Correctness does not fall monotonically with the cutoff — it falls, then recovers.\n\
         The recovery is the seed fallback firing, not the filter succeeding: an aggressive\n\
         cutoff starves the seed list, and a starved list falls back to the raw lexical index.\n\
         Default is 0.0 (off): on this corpus the filter is a correctness knob wearing a\n\
         latency costume."
    );
    Ok(())
}

/// Correctness and cost when the store is written to *while* a turn is in
/// flight.
///
/// Every other table here comes from a single-threaded read path against a
/// static store. The design calls its answer to concurrent mutation snapshot
/// isolation with an interrupt: a plan records the store version it was built
/// from, and if background consolidation invalidated anything in the working
/// set during the model call, the runtime rebuilds rather than answering from a
/// workspace that no longer holds. That path has tests but no numbers, and
/// `replanned` is recorded on every answer and never reported.
///
/// This does not make the runtime concurrent — it is still one thread — but it
/// does exercise the interrupt on the real probe set and price it.
pub fn run_consolidation(turns: usize, budget: usize) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    println!(
        "CONCURRENT CONSOLIDATION - {turns} turns, {} tokens of history, B_attention = {budget}",
        estimate_tokens(&corpus.text())
    );
    println!(
        "a consolidation pass invalidates part of the working set mid-turn, after the plan is\n\
         built and before the answer is returned"
    );
    let header = format!(
        "{:<26} {:>9} {:>9} {:>10} {:>9}   {}",
        "pressure", "correct", "mean k", "replanned", "esc.", "probes that fail"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for (label, invalidate_every) in [("none (baseline)", 0usize), ("every turn", 1)] {
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let (mut correct, mut replans, mut failed) = (0usize, 0usize, Vec::new());
        for (turn, probe) in corpus.probes.iter().enumerate() {
            let hit = invalidate_every > 0 && turn % invalidate_every == 0;
            let mut fired = false;
            let answer = runtime.ask_with_consolidation(
                probe.query,
                None,
                &mut reasoner,
                &mut |graph: &mut MemoryGraph| {
                    if !hit || fired {
                        return;
                    }
                    fired = true;
                    // Invalidate the most recently touched live claim: the
                    // adversarial case is consolidating exactly what the turn
                    // in flight is standing on.
                    let victim = graph
                        .nodes()
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| n.kind == Kind::Claim && n.status.is_live())
                        .max_by_key(|(_, n)| n.timestamp)
                        .map(|(i, _)| NodeIdx::from(i));
                    if let Some(v) = victim {
                        graph.invalidate(v, true);
                    }
                },
            );
            if answer.replanned {
                replans += 1;
            }
            let scored = if probe.on_context {
                answer.context.render()
            } else {
                answer.text.clone()
            };
            if probe.scores(&scored) {
                correct += 1;
            } else {
                failed.push(probe.label);
            }
        }
        let report = runtime.telemetry.report();
        println!(
            "{label:<26} {:>7}/{} {:>9.1} {:>9} {:>9.2}   {}",
            correct,
            corpus.probes.len(),
            report.tokens_per_query_mean,
            format!("{replans}/{}", corpus.probes.len()),
            report.escalation_rate.unwrap_or(0.0),
            if failed.is_empty() {
                "-".to_string()
            } else {
                failed.join("; ")
            }
        );
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "Still single-threaded: this prices the interrupt, it does not show the runtime is\n\
         safe under real concurrency. Nothing here runs two turns at once, and no lock is\n\
         exercised, so contention and torn reads remain unmeasured."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// A lexically diverse corpus
// ---------------------------------------------------------------------------

/// Vocabulary for [`build_corpus_diverse`]. Combined multiplicatively, these
/// yield far more distinct sentences than the corpus has documents, so the
/// distractor set stops being eight templates wearing different integers.
const SUBJECTS: &[&str] = &[
    "the ingest worker", "the replica set", "the edge cache", "the billing job",
    "the search shard", "the webhook relay", "the auth broker", "the export queue",
    "the metrics rollup", "the image resizer", "the audit sink", "the rate limiter",
    "the session store", "the config poller", "the schema registry", "the batch loader",
];
const VERBS: &[&str] = &[
    "was restarted after", "drained cleanly during", "reported elevated latency in",
    "held steady through", "backed off from", "was rescheduled around",
    "logged a warning about", "recovered without help from", "was drained ahead of",
    "paged nobody during", "absorbed the spike in", "fell behind on",
];
const OBJECTS: &[&str] = &[
    "the overnight compaction", "a partial network partition", "the weekly index rebuild",
    "an upstream certificate rotation", "the regional failover drill", "a noisy neighbour",
    "the quarterly retention sweep", "an operator typo", "the canary rollout",
    "a stuck leader election", "the cold-start stampede", "a leaked file handle",
];
const CLOSERS: &[&str] = &[
    "No action was taken.", "The runbook was not consulted.", "It resolved on its own.",
    "A ticket was filed and closed.", "The on-call slept through it.",
    "Nobody noticed until the digest.", "The graph looked fine afterwards.",
    "It has not recurred since.",
];

/// The same probes and corrections as [`build_corpus`], with distractors drawn
/// from a combinatorial vocabulary instead of eight repeated templates.
///
/// The standard corpus scales in *length* but not in *variety*: at 45,000 turns
/// its noise is still eight sentences with an integer substituted, which is why
/// pair coverage reads 0.0% and why "four million tokens of history" overstates
/// what is being asked of retrieval. This generator makes each distractor
/// lexically distinct, so a large run tests search against genuinely varied
/// content rather than against a handful of memorised shapes.
///
/// Kept alongside the original rather than replacing it: every published figure
/// comes from `build_corpus`, and silently changing the corpus under those
/// tables would invalidate them.
pub fn build_corpus_diverse(turns: usize) -> Corpus {
    let base = build_corpus(turns);
    // Which documents carry signal: the fixed opening facts, and the three
    // corrections wherever the generator placed them. Detecting noise by prefix
    // does not work — the templates hold an unsubstituted `{n}`, so comparing a
    // rendered document against the raw template never matches and every
    // distractor survives untouched. That is what left the first version at
    // 2,324 distinct documents after the radix fix.
    let keep: std::collections::HashSet<String> = base
        .docs
        .iter()
        .enumerate()
        .filter(|(i, (_, text))| {
            *i < 10 || text.starts_with("Correction:") || text.starts_with("Update:")
        })
        .map(|(_, (id, _))| id.clone())
        .collect();

    let mut docs: Vec<(String, String)> = Vec::with_capacity(base.docs.len());
    for (i, (id, text)) in base.docs.iter().enumerate() {
        if keep.contains(id) {
            docs.push((id.clone(), text.clone()));
            continue;
        }
        // Mixed-radix decomposition of the index, so every combination is
        // reachable and the sequence repeats only after their product.
        //
        // A first version used strides — `i * 7 % 16`, `i * 5 % 12` and so on —
        // with a comment claiming they would not repeat until the product was
        // exhausted. That is false: independent strides repeat at the *lowest
        // common multiple* of their periods, which here is 48. It produced 26
        // distinct documents out of 30,000 and would have shipped as a
        // "diverse" corpus, because the claim was in a comment rather than in a
        // test. `corpus_diversity_is_measured_not_asserted` now measures it.
        let mut n = i;
        let s = SUBJECTS[n % SUBJECTS.len()];
        n /= SUBJECTS.len();
        let v = VERBS[n % VERBS.len()];
        n /= VERBS.len();
        let o = OBJECTS[n % OBJECTS.len()];
        n /= OBJECTS.len();
        let c = CLOSERS[n % CLOSERS.len()];
        docs.push((
            id.clone(),
            format!(
                "Shift note {i}: {s} {v} {o}. {c} Window {i} closed with no follow-up, \
                 and the {s} owner acknowledged at slot {i}.",
            ),
        ));
    }
    Corpus {
        docs,
        probes: base.probes,
    }
}

/// Scaling on the lexically diverse corpus, out to millions of tokens.
///
/// The standard generator emits 21 distinct documents at any size, so its large
/// runs grow in length at constant variety and say more about the generator than
/// about the system — including its cost, which is dominated by comparing each
/// new document against thousands of near-duplicates. This runs the same probes
/// against [`build_corpus_diverse`], and reports the distinct-document count in
/// the table so the difference is visible rather than asserted.
pub fn run_scaling_diverse(sizes: &[usize], budget: usize) -> Result<(), DcrError> {
    println!(
        "{:>8} {:>11} {:>10} {:>8} {:>9} {:>7} {:>9} {:>8}",
        "turns", "history", "distinct", "nodes", "mean k", "correct", "ingest", "query"
    );
    println!("{}", "-".repeat(76));
    let mut first: Option<(usize, f64)> = None;
    let mut last = (0usize, 0.0f64);
    for &turns in sizes {
        let corpus = build_corpus_diverse(turns);
        let started = Instant::now();
        let mut runtime = Dcr::new(budget);
        runtime.index.set_exact(false);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let ingest_s = started.elapsed().as_secs_f64();
        let mut reasoner = LocalReasoner::new();
        let started = Instant::now();
        let mut correct = 0usize;
        for probe in &corpus.probes {
            let a = runtime.ask_with(probe.query, None, &mut reasoner);
            let scored = if probe.on_context {
                a.context.render()
            } else {
                a.text.clone()
            };
            if probe.scores(&scored) {
                correct += 1;
            }
        }
        let query_ms = started.elapsed().as_secs_f64() * 1000.0 / corpus.probes.len() as f64;
        let distinct: HashSet<String> = corpus
            .docs
            .iter()
            .map(|(_, t)| t.chars().filter(|c| !c.is_ascii_digit()).collect())
            .collect();
        let history = estimate_tokens(&corpus.text());
        let report = runtime.telemetry.report();
        println!(
            "{turns:>8} {history:>11} {:>10} {:>8} {:>9.1} {:>7} {:>8.0}s {:>6.0}ms",
            distinct.len(),
            runtime.graph.len(),
            report.tokens_per_query_mean,
            format!("{correct}/{}", corpus.probes.len()),
            ingest_s,
            query_ms
        );
        if first.is_none() {
            first = Some((history, report.tokens_per_query_mean));
        }
        last = (history, report.tokens_per_query_mean);
    }
    println!("{}", "-".repeat(76));
    if let Some((h0, k0)) = first {
        println!(
            "history grew {:.0}x; active context grew {:.2}x",
            last.0 as f64 / h0.max(1) as f64,
            last.1 / k0.max(1.0)
        );
    }
    println!(
        "The generator exhausts its vocabulary at 18,432 combinations, so past roughly 18,000\n\
         turns the distinct count plateaus and documents begin to repeat. Diversity is far\n\
         higher than the standard corpus at every size and is not unbounded."
    );
    Ok(())
}

/// Per-stage planner clocks and rejection counts across a scaling sweep.
///
/// Proposed by [@cwahq](https://www.moltbook.com/post/78237a57-17ef-4c78-b05f-8c1e5a944196):
/// *"split candidate generation, scoring, and graph expansion into separate
/// clocks, then publish the rejected-candidate count at each stage."*
///
/// Every latency number this repo published before this existed was taken at
/// the outer edge of a turn, so "planning" was the residual left after
/// retrieval rather than a measured quantity. That is the reason the previous
/// diagnosis ("the vector index is a linear scan") was confidently wrong: the
/// cost was in a stage with no clock on it.
pub fn run_stages(sizes: &[usize], budget: usize) -> Result<(), DcrError> {
    println!("PLANNER STAGES - standard corpus, B_attention = {budget}, per query");
    println!();
    let header = format!(
        "{:>7} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}",
        "turns", "nodes", "plan us", "seed", "expand", "pin", "score", "knap", "admit", "resid",
        "spans/q", "L0 builds/q"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut rows: Vec<(usize, usize, crate::planner::StageProfile, u64)> = Vec::new();
    let mut last_memoised_spans = 0.0f64;
    for &turns in sizes {
        let corpus = build_corpus(turns);
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        // Planning only: the accumulator is reset after ingest so the table
        // prices the read path, not the writes that built the store.
        runtime.planning = crate::planner::StageProfile::default();
        runtime.plans = 0;
        let mut reasoner = LocalReasoner::new();
        for probe in &corpus.probes {
            let _ = runtime.ask_with(probe.query, None, &mut reasoner);
        }
        let plans = runtime.plans.max(1);
        let p = runtime.planning;
        let l0_builds = runtime.ladder.l0_builds();
        let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / plans as f64;
        println!(
            "{turns:>7} {:>7} {:>10.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10.0} {:>12.2}",
            runtime.graph.len(),
            // `measured_time` is the independent clock around the whole call.
            // Reporting `total()` here would make the residual identically zero
            // by construction, which is the defect this column exists to expose.
            per(p.measured_time),
            per(p.seed_time),
            per(p.expand_time),
            per(p.pin_time),
            per(p.score_time),
            per(p.knapsack_time),
            per(p.admit_time),
            per(p.residual()),
            p.score_spans_priced as f64 / plans as f64,
            l0_builds as f64 / plans as f64,
        );
        last_memoised_spans = runtime.ladder.l0_build_spans() as f64 / plans as f64;
        rows.push((turns, runtime.graph.len(), p, plans));
    }

    println!();
    println!("Candidates rejected per query, by the stage that rejected them:");
    println!();
    let rej_header = format!(
        "{:>7} {:>11} {:>13} {:>12} {:>13} {:>10} {:>16} {:>11} {:>16}",
        "turns", "seed:floor", "seed:status", "seed:kept", "pin:added", "score:cap",
        "score:superseded", "knap:drop", "expand cap bound"
    );
    println!("{rej_header}");
    println!("{}", "-".repeat(rej_header.len()));
    for (turns, _, p, plans) in &rows {
        let per = |v: usize| v as f64 / *plans as f64;
        println!(
            "{turns:>7} {:>11.1} {:>13.1} {:>12.1} {:>13.1} {:>10.1} {:>16.1} {:>11.1} {:>16}",
            per(p.seed_dropped_floor),
            per(p.seed_dropped_status),
            per(p.seeds_kept),
            per(p.pinned_added),
            per(p.score_dropped_cap),
            per(p.score_dropped_superseded),
            per(p.knapsack_dropped),
            if p.expand_capped { "yes" } else { "no" },
        );
    }

    println!();
    if let (Some(first), Some(last)) = (rows.first(), rows.last()) {
        let growth = |f: fn(&crate::planner::StageProfile) -> std::time::Duration| {
            let a = f(&first.2).as_secs_f64() / first.3 as f64;
            let b = f(&last.2).as_secs_f64() / last.3 as f64;
            if a > 0.0 { b / a } else { f64::NAN }
        };
        println!(
            "Growth {}x turns ({} -> {} nodes): total {:.1}x, seed {:.1}x, expand {:.1}x, \
             pin {:.1}x, score {:.1}x, knapsack {:.1}x",
            last.0 / first.0.max(1),
            first.1,
            last.1,
            growth(|p| p.total()),
            growth(|p| p.seed_time),
            growth(|p| p.expand_time),
            growth(|p| p.pin_time),
            growth(|p| p.score_time),
            growth(|p| p.knapsack_time),
        );
        println!(
            "Nodes grew {:.1}x; nodes visited by the pin stage per query grew {:.1}x.",
            last.1 as f64 / first.1.max(1) as f64,
            (last.2.pin_scanned as f64 / last.3 as f64)
                / (first.2.pin_scanned as f64 / first.3.max(1) as f64).max(1.0),
        );
    }
    // The memo off, at the largest size. A speed-up nobody can make disappear
    // on purpose is not a measured speed-up, and on a loaded machine the clocks
    // above cannot show it — so the comparison is in spans concatenated, which
    // is deterministic.
    if let Some(&largest) = sizes.last() {
        let corpus = build_corpus(largest);
        let mut off = Dcr::new(budget);
        off.ladder.memoise_l0 = false;
        for (doc_id, text) in &corpus.docs {
            off.ingest(text, Some(doc_id))?;
        }
        off.plans = 0;
        let before = off.ladder.l0_build_spans();
        let mut reasoner = LocalReasoner::new();
        for probe in &corpus.probes {
            let _ = off.ask_with(probe.query, None, &mut reasoner);
        }
        let plans = off.plans.max(1);
        let spans_off = (off.ladder.l0_build_spans() - before) as f64 / plans as f64;
        println!();
        println!(
            "Control, {largest} turns: with the L0 memo disabled the planner concatenates\n\
             {spans_off:.0} source spans per query; with it enabled, {:.0}. The memo is what\n\
             stops a candidate set capped at 120 from doing work linear in history.",
            last_memoised_spans,
        );
    }

    println!();
    println!(
        "Timings are single-run on one machine and carry the spread every other timing\n\
         here carries; reproduce the ordering between stages, not the microseconds."
    );
    Ok(())
}

/// Spearman rank correlation between two rankings of the same candidate set.
///
/// Absent items are given the worst rank rather than dropped, because "the
/// vector channel never surfaced this at all" is disagreement, not missing data.
fn rank_correlation(a: &[usize], b: &[usize]) -> Option<f64> {
    let mut union: Vec<usize> = a.to_vec();
    for &id in b {
        if !union.contains(&id) {
            union.push(id);
        }
    }
    let n = union.len();
    if n < 3 {
        return None;
    }
    let worst = n as f64;
    let rank_in = |list: &[usize], id: usize| -> f64 {
        list.iter().position(|&x| x == id).map_or(worst, |p| p as f64)
    };
    let ra: Vec<f64> = union.iter().map(|&id| rank_in(a, id)).collect();
    let rb: Vec<f64> = union.iter().map(|&id| rank_in(b, id)).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(&ra), mean(&rb));
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let (x, y) = (ra[i] - ma, rb[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return None;
    }
    Some(num / (da.sqrt() * db.sqrt()))
}

/// A deterministic pseudo-random ranking of `k` ids drawn from `0..n`.
///
/// This is the negative control, and its *structure* matters more than its
/// randomness. The first version of this control shuffled the lexical channel's
/// own ids, which gave it 100% overlap with the thing it was controlling for
/// while the real comparison had 11% — so it reported what the statistic does
/// to a reordering, not what it does to two mostly-disjoint lists, and the two
/// numbers were not comparable. Drawing a fresh subset reproduces the
/// disjointness as well as the disagreement.
fn random_ranking(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out: Vec<usize> = Vec::new();
    while out.len() < k.min(n) {
        let pick = (next() % n as u64) as usize;
        if !out.contains(&pick) {
            out.push(pick);
        }
    }
    out
}

#[allow(dead_code)]
fn shuffled(ids: &[usize], seed: u64) -> Vec<usize> {
    let mut out = ids.to_vec();
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..out.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// How independent are the lexical and vector channels?
///
/// `docs/use-cases.md` states that the bundled 256-dimensional hashing
/// embedding "finds material sharing vocabulary, and the lexical and vector
/// channels are therefore correlated rather than independent evidence". That
/// was asserted, never measured, and it is the stated reason paraphrase search
/// is a poor fit — so it should carry a number.
///
/// The statistic is Spearman rank correlation between the two channels' top-k
/// over the same query, with absent items ranked worst. A negative control
/// correlates the lexical channel against a deterministic shuffle of the same
/// ids: if the measure cannot report independence when the rankings *are*
/// independent, it cannot support a claim about correlation either.
pub fn run_channels(turns: usize, budget: usize) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id))?;
    }
    const K: usize = 20;

    println!(
        "CHANNEL INDEPENDENCE - {turns} turns, {} nodes, top-{K} per channel",
        runtime.graph.len()
    );
    println!();
    let header = format!(
        "{:<46} {:>8} {:>9} {:>9} {:>10}",
        "probe", "rho", "lex only", "vec only", "shared"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut rhos: Vec<f64> = Vec::new();
    let mut controls: Vec<f64> = Vec::new();
    let mut only_one = 0usize;
    let mut shared_total = 0usize;
    let mut control_shared = 0usize;
    for probe in &corpus.probes {
        let qv = runtime.ladder.query_vector(probe.query);
        let (lex, vec) = runtime
            .index
            .channels(Namespace::Node, probe.query, &qv, K);
        let lex_ids: Vec<usize> = lex.iter().map(|(i, _)| *i).collect();
        let vec_ids: Vec<usize> = vec.iter().map(|(i, _)| *i).collect();
        let shared = lex_ids.iter().filter(|i| vec_ids.contains(i)).count();
        let lex_only = lex_ids.len() - shared;
        let vec_only = vec_ids.len() - shared;
        only_one += lex_only + vec_only;
        shared_total += shared;
        let rho = rank_correlation(&lex_ids, &vec_ids);
        if let Some(r) = rho {
            rhos.push(r);
        }
        // Negative control with the same shape: a fresh pseudo-random draw of
        // K ids from the same node population, so it carries the same
        // disjointness as the real comparison rather than only the same length.
        let seed = 0x5DEE_CE66_D1CE_F00D ^ (rhos.len() as u64);
        let control_ids = random_ranking(runtime.graph.len(), K, seed);
        if let Some(c) = rank_correlation(&lex_ids, &control_ids) {
            controls.push(c);
        }
        control_shared += lex_ids.iter().filter(|i| control_ids.contains(i)).count();
        let label: String = probe.label.chars().take(45).collect();
        match rho {
            Some(r) => println!(
                "{label:<46} {r:>8.3} {lex_only:>9} {vec_only:>9} {shared:>10}"
            ),
            None => println!(
                "{label:<46} {:>8} {lex_only:>9} {vec_only:>9} {shared:>10}",
                "n/a"
            ),
        }
    }
    println!("{}", "-".repeat(header.len()));

    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let rho_mean = mean(&rhos);
    let control_mean = mean(&controls);
    println!(
        "{:<46} {rho_mean:>8.3} {:>9} {:>9} {shared_total:>10}",
        "mean", "", ""
    );
    println!(
        "{:<46} {control_mean:>8.3} {:>9} {:>9} {control_shared:>10}   <- control",
        "lexical vs random draw", "", ""
    );
    println!();

    // Overlap is the statistic that survives the convention. rho above ranks
    // absent items worst, so two mostly-disjoint lists correlate negatively
    // whatever order they are in — which is why the control is drawn the same
    // way rather than compared against zero.
    let probes = corpus.probes.len().max(1);
    let expected = (K * K) as f64 / runtime.graph.len().max(1) as f64;
    let observed = shared_total as f64 / probes as f64;
    println!("Agreement, per probe, top-{K} of {} nodes:", runtime.graph.len());
    println!("  shared by both channels     : {observed:.1}");
    println!("  expected if independent     : {expected:.1}  (K^2 / N)");
    println!("  control, measured           : {:.1}", control_shared as f64 / probes as f64);
    println!(
        "  ratio, observed to expected : {:.1}x",
        observed / expected.max(f64::EPSILON)
    );
    println!();
    println!(
        "How to read this. rho is reported against the control and not against zero:\n\
         ranking absent items worst makes any two mostly-disjoint lists correlate\n\
         negatively regardless of order, so the number carries the convention as much\n\
         as the data. The load-bearing figure is the overlap ratio.\n\n\
         The channels agree more than chance and much less than the documentation\n\
         implies. 'Correlated rather than independent evidence' is directionally\n\
         right and overstated as written: {only_one} of {} ranked positions across the\n\
         probe set were surfaced by exactly one channel.\n\n\
         What this does not measure: paraphrase. The standard seven probes were\n\
         written to be answerable by vocabulary overlap, which is the condition\n\
         under which the two channels are most likely to agree. A paraphrase probe\n\
         set is the measurement still missing, and it is the one the poor-fit entry\n\
         actually rests on.",
        only_one + shared_total
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Subject-identification control
// ---------------------------------------------------------------------------

/// The standard corpus with several documents per subject, only one of which
/// carries the answer.
///
/// Proposed by a reader: *when one document equals one domain, domain is not
/// identified.* [`build_corpus`] states each fact once, so "found the right
/// document" and "found the right subject" are the same event and no probe can
/// separate them. Every retrieval result published from that corpus therefore
/// has subject identification confounded with document identification.
///
/// Here each subject appears in three documents — one stating the value, two
/// discussing the subject without it, matched for length and position. A runtime
/// keying on *does this document mention the subject* now has a one-in-three
/// chance; one that identifies the fact is unaffected.
///
/// Added alongside the standard corpus, never replacing it: the published
/// figures come from that one, and swapping it underneath them would invalidate
/// the tables silently.
pub fn build_corpus_permuted(turns: usize) -> Corpus {
    let base = build_corpus(turns);
    // Two decoys per fixed fact: same subject, no value, comparable length.
    const DECOYS: &[(&str, &str)] = &[
        ("the server ip", "was reviewed during the network audit and nobody raised a concern about it"),
        ("the deploy window", "is discussed every fortnight at the change board and rarely moves"),
        ("the service owner", "was confirmed unchanged in the last two on-call handovers"),
        ("the error message", "appears in the archive from an unrelated incident eighteen months ago"),
        ("the blocker", "was carried over from the previous review with no new detail attached"),
        ("the hourly rate", "is set annually by finance and was not revisited this quarter"),
        ("the engineer count", "is tracked in the staffing sheet and did not change this cycle"),
        ("the retry budget", "is documented in the rollout policy and has never been amended"),
        ("the checkout subnet", "was included in the quarterly inventory with no findings"),
        ("the build", "was signed off by two reviewers before the freeze began"),
    ];

    let mut docs: Vec<(String, String)> = Vec::new();
    let mut decoy = 0usize;
    for (i, (id, text)) in base.docs.iter().enumerate() {
        docs.push((id.clone(), text.clone()));
        // Two decoys after each of the fixed opening documents, so the subject
        // is present three times and the answering document is not positionally
        // distinguished by being the only mention.
        if i < 10 {
            for k in 0..2 {
                let (subject, tail) = DECOYS[(decoy + k) % DECOYS.len()];
                docs.push((
                    format!("d{i:02}{k}"),
                    format!("Review note {i}{k}: {subject} {tail}."),
                ));
            }
            decoy += 2;
        }
    }
    Corpus {
        docs,
        probes: base.probes,
    }
}

/// Does the runtime identify a subject, or the document that mentions it?
pub fn run_subject_control(turns: usize, budget: usize) -> Result<(), DcrError> {
    println!("SUBJECT IDENTIFICATION CONTROL - B_attention = {budget}");
    println!(
        "The standard corpus states each fact once, so 'found the document' and 'found the\n\
         subject' are the same event. This adds two decoy documents per subject that mention\n\
         it without carrying its value."
    );
    let header = format!(
        "{:<28} {:>7} {:>9} {:>9}   {}",
        "corpus", "docs", "correct", "mean k", "probes that fail"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for (label, corpus) in [
        ("standard (1 doc/subject)", build_corpus(turns)),
        ("with decoys (3 docs/subject)", build_corpus_permuted(turns)),
    ] {
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let (mut correct, mut failed) = (0usize, Vec::new());
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            let scored = if probe.on_context {
                answer.context.render()
            } else {
                answer.text.clone()
            };
            if probe.scores(&scored) {
                correct += 1;
            } else {
                failed.push(probe.label);
            }
        }
        println!(
            "{label:<28} {:>7} {:>7}/{} {:>9.1}   {}",
            corpus.docs.len(),
            correct,
            corpus.probes.len(),
            runtime.telemetry.report().tokens_per_query_mean,
            if failed.is_empty() { "-".to_string() } else { failed.join("; ") }
        );
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "A drop here means the previous results measured document identification with subject\n\
         confounded into it. Holding means subject identification is doing the work. Either\n\
         way the standard corpus alone could not distinguish them."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Prompt-cache friendliness
// ---------------------------------------------------------------------------

/// How much of each turn's assembled context is a byte-identical prefix of the
/// previous turn's.
///
/// External work on production context assembly treats cache-aware layout as a
/// first-order concern: providers bill cached prefix tokens at a large discount,
/// so a context whose stable material comes first is materially cheaper than one
/// that is re-derived each turn, at the same token count.
///
/// This runtime optimises the opposite quantity. It re-solves the knapsack every
/// turn to assemble the cheapest *sufficient* context, which is by construction
/// a different context. Nothing in the design tries to keep a stable prefix, and
/// nothing in the reported cost model accounts for the discount that stability
/// would earn. So "457 tokens per query" and "457 billable tokens per query" are
/// not the same claim, and this measures the gap rather than assuming it away.
pub fn run_cache_layout(turns: usize, budget: usize) -> Result<(), DcrError> {
    let corpus = build_corpus(turns);
    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id))?;
    }
    let mut reasoner = LocalReasoner::new();

    println!("PROMPT-CACHE LAYOUT - {turns} turns, B_attention = {budget}");
    println!(
        "Shared prefix between consecutive turns, in the order the context is actually rendered."
    );
    let header = format!(
        "{:<38} {:>9} {:>14} {:>10}",
        "probe", "tokens", "shared prefix", "cacheable"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut previous: Option<String> = None;
    let (mut total_tokens, mut total_shared) = (0usize, 0usize);
    for probe in &corpus.probes {
        let answer = runtime.ask_with(probe.query, None, &mut reasoner);
        let rendered = answer.context.render();
        let tokens = estimate_tokens(&rendered);
        let shared_chars = previous
            .as_ref()
            .map(|prev| {
                prev.as_bytes()
                    .iter()
                    .zip(rendered.as_bytes())
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .unwrap_or(0);
        // Price the shared run in tokens, so the number is comparable to k.
        let shared_tokens = estimate_tokens(&rendered[..shared_chars.min(rendered.len())]);
        total_tokens += tokens;
        total_shared += shared_tokens;
        println!(
            "{:<38} {tokens:>9} {shared_tokens:>14} {:>9.1}%",
            probe.label,
            shared_tokens as f64 / tokens.max(1) as f64 * 100.0
        );
        previous = Some(rendered);
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "overall cacheable prefix: {:.1}% of assembled tokens ({total_shared} of {total_tokens})",
        total_shared as f64 / total_tokens.max(1) as f64 * 100.0
    );
    println!(
        "A low number is not a bug — it is the cost of re-planning. It does mean the token\n\
         counts elsewhere in this report are *assembled* tokens rather than *billable* ones,\n\
         and that a cache-friendly assembler sending more tokens could still be cheaper per\n\
         turn. Nothing here has measured that comparison."
    );
    Ok(())
}

/// Reciprocal rank fusion against the linear blend — and the seed floor, which
/// is how a real defect was found.
///
/// External writing on production retrieval prefers RRF because two channels'
/// scores are not on a common scale, and this index normalises each channel by
/// its own top hit — which makes a channel's influence depend on the shape of
/// its score distribution rather than on how well it ranked anything. The
/// argument is sound and the swap is implemented. It buys nothing here, which
/// the channel measurement predicted: 195 of 220 ranked positions on this
/// corpus are surfaced by exactly one channel, and RRF rewards agreement.
///
/// The floor sweep is the part that mattered. RRF flattens the score list, which
/// stops `seed_min_ratio` binding, which changed which nodes seeded, which
/// changed which expansions ran — and that exposed a guard with a hole in it.
/// Seeding excludes evidence whose every live dependent has been superseded;
/// `expand` did not, and re-admitted exactly those nodes through a dependency
/// edge. Raising the floor made the second door easy to walk through, so a
/// configuration 3x cheaper on the standard probes served the superseded value
/// on 3 of 4 adversarial queries.
///
/// `expand` now applies the same rule. Both tables below are what that looks
/// like: correctness no longer moves with the floor, so the floor is a cost
/// knob rather than a correctness knob, and the cheap configuration is the
/// default.
pub fn run_fusion(turns: usize, budget: usize) -> Result<(), DcrError> {
    println!("FUSION AND SEED FLOOR - rank fusion against the linear blend, across seed floors");
    println!("corpus: standard, {turns} turns, B_attention = {budget}\n");

    let corpus = build_corpus(turns);
    let modes: [(&str, crate::index::Fusion, f32); 6] = [
        ("linear (old default)", crate::index::Fusion::Linear, 0.3),
        ("linear (default)", crate::index::Fusion::Linear, 0.5),
        ("linear", crate::index::Fusion::Linear, 0.7),
        ("linear", crate::index::Fusion::Linear, 0.85),
        ("rrf k0=60", crate::index::Fusion::Rrf { k0: 60.0 }, 0.3),
        ("rrf k0=60", crate::index::Fusion::Rrf { k0: 60.0 }, 0.95),
    ];

    let header = format!(
        "{:<18} {:>7} {:>8} {:>7} {:>9} {:>11} {:>11} {:>9}",
        "fusion", "floor", "mean k", "max k", "correct", "seed:kept", "seed:floor", "query"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut rankings: Vec<(String, Vec<Vec<usize>>)> = Vec::new();
    for (label, fusion, floor) in modes {
        let mut runtime = Dcr::new(budget);
        runtime.index.fusion = fusion;
        runtime.planner.seed_min_ratio = floor;
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let mut correct = 0usize;
        let started = Instant::now();
        for probe in &corpus.probes {
            let answer = runtime.ask_with(probe.query, None, &mut reasoner);
            if answer
                .text
                .to_lowercase()
                .contains(&probe.expected.to_lowercase())
            {
                correct += 1;
            }
        }
        let query_ms = started.elapsed().as_secs_f64() * 1000.0 / corpus.probes.len() as f64;
        let report = runtime.telemetry.report();
        let plans = runtime.plans.max(1) as f64;
        println!(
            "{label:<18} {floor:>7.2} {:>8.1} {:>7} {:>9} {:>11.1} {:>11.1} {:>8.1}ms",
            report.tokens_per_query_mean,
            report.tokens_per_query_max,
            format!("{correct}/{}", corpus.probes.len()),
            runtime.planning.seeds_kept as f64 / plans,
            runtime.planning.seed_dropped_floor as f64 / plans,
            query_ms,
        );
        // Captured after the probes so the index is in the state the run used,
        // and per probe rather than pooled, so a shift on one query cannot be
        // averaged away by six that did not move.
        let ranked = corpus
            .probes
            .iter()
            .map(|probe| {
                let qv = crate::embed::hashing_embed(probe.query, crate::embed::DIM);
                runtime
                    .index
                    .search(crate::index::Namespace::Node, probe.query, &qv, 12)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            })
            .collect();
        rankings.push((format!("{label} @{floor:.2}"), ranked));
    }
    println!("{}", "-".repeat(header.len()));

    let base = &rankings[0].1;
    for (label, other) in rankings.iter().skip(1) {
        let (mut shared, mut total, mut same_top) = (0usize, 0usize, 0usize);
        for (a, b) in base.iter().zip(other.iter()) {
            let set: HashSet<usize> = b.iter().copied().collect();
            shared += a.iter().filter(|i| set.contains(i)).count();
            total += a.len();
            if a.first() == b.first() {
                same_top += 1;
            }
        }
        println!(
            "{label:<22} vs linear @0.30: top-12 overlap {:>5.1}%, identical top-1 {same_top}/{}",
            shared as f64 / total.max(1) as f64 * 100.0,
            base.len()
        );
    }

    // The table above is 7/7 in every row across a 3x spread of working-set
    // size, which does not mean every row is equally good — it means these
    // probes cannot tell them apart. The adversarial mutation set can: each of
    // its queries has a superseded value that is the *closer* lexical match, so
    // a configuration that admits too little context, or the wrong context,
    // answers with the stale value rather than failing to answer.
    println!("\nSame floors against the adversarial mutation set - the superseded value is the");
    println!("closer lexical match, so serving it is a distinguishable failure:\n");
    let mut_corpus = build_mutation_corpus_from(turns, ADVERSARIAL);
    let mut_header = format!(
        "{:<18} {:>7} {:>11} {:>14} {:>10}",
        "fusion", "floor", "corrected", "stale served", "mean k"
    );
    println!("{}", "-".repeat(mut_header.len()));
    println!("{mut_header}");
    println!("{}", "-".repeat(mut_header.len()));
    for (label, fusion, floor) in modes {
        let mut runtime = Dcr::new(budget);
        runtime.index.fusion = fusion;
        runtime.planner.seed_min_ratio = floor;
        for (doc_id, text) in &mut_corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let mut reasoner = LocalReasoner::new();
        let (mut corrected, mut stale) = (0usize, 0usize);
        for m in ADVERSARIAL {
            let text = runtime
                .ask_with(m.query, None, &mut reasoner)
                .text
                .to_lowercase();
            if text.contains(&m.live.to_lowercase()) {
                corrected += 1;
            }
            if text.contains(&m.stale.to_lowercase()) {
                stale += 1;
            }
        }
        println!(
            "{label:<18} {floor:>7.2} {:>11} {:>14} {:>10.1}",
            format!("{corrected}/{}", ADVERSARIAL.len()),
            format!("{stale}/{}", ADVERSARIAL.len()),
            runtime.telemetry.report().tokens_per_query_mean,
        );
    }
    println!("{}", "-".repeat(mut_header.len()));
    println!(
        "\nBoth tables used to disagree, and the disagreement was the finding. Raising the floor\n\
         cut the working set 3x with the standard probes still at 7/7, while correction-following\n\
         fell from 4/4 to 1/4 -- so the floor looked like a free saving and was silently buying\n\
         correctness. It was not. The floor changed which nodes seeded, which changed which\n\
         expansions ran, and `expand` was re-admitting superseded evidence that `seed` had\n\
         already excluded. A guard on one entrance of a room with two doors.\n\
         \n\
         With the same rule applied in both places, the corrected column holds at every floor\n\
         and the token column still falls. That is the shape a real fix makes: the cheap\n\
         configuration stops being dangerous rather than the expensive one being justified.\n\
         \n\
         The instrument lesson survives the fix and is the more transferable half. Seven probes\n\
         reading 7/7 across a 3x range of working-set size were not agreeing that the settings\n\
         were equivalent -- they were unable to disagree. A second probe set built for an\n\
         unrelated purpose is the only reason the hole was found, and it was found by tuning a\n\
         threshold that had nothing to do with supersession."
    );
    Ok(())
}

/// Overlap between the approximate index's top-k and the exact scan's top-k.
///
/// The report currently defends LSH with "correctness is identical on both
/// paths", which is seven probes agreeing on an answer. External guidance on
/// approximate indices quotes recall — the fraction of the true top-k the
/// approximate path actually returns — and that is the stronger statement,
/// because two paths can agree on every answer while retrieving different
/// material and diverge on the eighth question nobody asked.
pub fn run_recall(sizes: &[usize], budget: usize) -> Result<(), DcrError> {
    println!("APPROXIMATE RETRIEVAL RECALL - top-k overlap against the exact scan");
    let header = format!(
        "{:<7} {:>7} {:>10} {:>12} {:>12}",
        "turns", "nodes", "probes", "recall@12", "identical top-1"
    );
    println!("{}", "-".repeat(header.len()));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for &turns in sizes {
        let corpus = build_corpus(turns);
        let mut runtime = Dcr::new(budget);
        for (doc_id, text) in &corpus.docs {
            runtime.ingest(text, Some(doc_id))?;
        }
        let (mut hits, mut total, mut top1) = (0usize, 0usize, 0usize);
        for probe in &corpus.probes {
            let qv = crate::embed::hashing_embed(probe.query, crate::embed::DIM);
            runtime.index.set_exact(true);
            let exact = runtime
                .index
                .search(crate::index::Namespace::Node, probe.query, &qv, 12);
            runtime.index.set_exact(false);
            let ann = runtime
                .index
                .search(crate::index::Namespace::Node, probe.query, &qv, 12);
            let ann_ids: HashSet<usize> = ann.iter().map(|(i, _)| *i).collect();
            hits += exact.iter().filter(|(i, _)| ann_ids.contains(i)).count();
            total += exact.len();
            if exact.first().map(|(i, _)| *i) == ann.first().map(|(i, _)| *i) {
                top1 += 1;
            }
        }
        runtime.index.set_exact(true);
        println!(
            "{turns:<7} {:>7} {:>10} {:>11.1}% {:>11}",
            runtime.graph.len(),
            corpus.probes.len(),
            hits as f64 / total.max(1) as f64 * 100.0,
            format!("{top1}/{}", corpus.probes.len())
        );
    }
    println!("{}", "-".repeat(header.len()));
    println!(
        "Recall below 100% with correctness unchanged means the two paths retrieve different\n\
         material and the probe set does not distinguish them. That is a weaker guarantee than\n\
         'identical correctness' sounds, and it is the honest one."
    );
    Ok(())
}

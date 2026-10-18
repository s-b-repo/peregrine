//! The fit gate: when the whole store fits in the window, the runtime must
//! get out of the way — and when it does not, the gate must not fire.

use peregrine_ctx::runtime::Dcr;
use peregrine_ctx::{Fit, Reasoner};

/// A reasoner that reads raw text — what the bypass is actually for. The
/// offline line-matcher cannot, which is why the gate is opt-in.
struct RawReader;
impl Reasoner for RawReader {
    fn complete(&mut self, prompt: &str, _system: &str) -> String {
        if prompt.contains("10.0.4.12") {
            "10.0.4.12 [raw]".to_string()
        } else {
            "not in the history".to_string()
        }
    }
}

#[test]
fn small_store_bypasses_the_planner_when_the_gate_is_on() {
    let mut rt = Dcr::new(800);
    rt.fit_gate = true;
    rt.ingest("The server ip is 10.0.4.12.", Some("ops")).unwrap();
    assert_eq!(rt.fit(), Fit::Bypass);

    let answer = rt.ask_with("what is the server ip?", None, &mut RawReader);
    assert!(answer.bypassed, "the gate did not fire");
    assert!(answer.text.contains("10.0.4.12"), "{:?}", answer.text);
    assert_eq!(rt.plans, 0, "a bypassed turn must not plan");
    assert_eq!(rt.telemetry.bypassed, 1);
    assert_eq!(rt.telemetry.report().bypass_rate, Some(1.0));
}

#[test]
fn the_gate_stays_closed_when_history_exceeds_the_window() {
    let mut rt = Dcr::new(40);
    rt.fit_gate = true;
    rt.ingest(
        "The server ip is 10.0.4.12 and the port is 8080, and this sentence \
         keeps going and going, with more operational detail than any small \
         window could hold: rack numbers, vlan tags, failover notes, ticket \
         references, and the rest of the runbook appended for good measure.",
        Some("ops"),
    )
    .unwrap();
    assert_eq!(rt.fit(), Fit::Plan);

    let answer = rt.ask_with("what is the server ip?", None, &mut RawReader);
    assert!(!answer.bypassed, "the gate fired over a too-big store");
    assert!(rt.plans > 0, "a planned turn must leave planner telemetry");
}

#[test]
fn the_gate_is_off_by_default() {
    let mut rt = Dcr::new(800);
    rt.ingest("The server ip is 10.0.4.12.", Some("ops")).unwrap();
    let answer = rt.ask_with("what is the server ip?", None, &mut RawReader);
    assert!(!answer.bypassed);
    assert!(rt.plans > 0);
    assert_eq!(rt.telemetry.bypassed, 0);
}

#[test]
fn a_store_with_skills_never_bypasses() {
    let mut rt = Dcr::new(800);
    rt.fit_gate = true;
    rt.add_skill("code-review", "reviews code diffs", "sha256:abc").unwrap();
    // The store is otherwise empty and tiny — but admission is what loads a
    // skill, and bypassing the planner would leave it dormant.
    assert_eq!(rt.fit(), Fit::Plan);
}

/// The gate compares `history + prompt wrapper + query` against the budget,
/// not history alone: certifying a prompt that then overflows the window
/// would be the exact failure the gate exists to avoid.
#[test]
fn the_gate_counts_the_prompt_wrapper_against_the_budget() {
    // 340 bytes = 85 heuristic tokens, under a budget of 100 — but the query
    // (~16 tokens) plus the fixed wrapper (32) push the actual prompt over.
    let mut rt = Dcr::new(100);
    rt.fit_gate = true;
    let doc = "operational detail, ticket references, runbook notes. ".repeat(8);
    rt.ingest(&doc[..340], Some("ops")).unwrap();
    assert_eq!(rt.telemetry.history_tokens, 85);

    let query = "summarise every ticket reference and runbook note above please";
    let answer = rt.ask_with(query, None, &mut RawReader);
    assert!(
        !answer.bypassed,
        "history fits but the full prompt does not — the gate must not fire"
    );
}

/// A bypassed turn is still a turn: it must feed the prefix-reuse metric
/// (otherwise reuse would be compared *across* bypasses) and leave no stale
/// membership for the stickiness weight to pay toward.
#[test]
fn a_bypassed_turn_keeps_cross_turn_state_honest() {
    let mut rt = Dcr::new(800);
    rt.fit_gate = true;
    rt.ingest("The server ip is 10.0.4.12.", Some("ops")).unwrap();
    rt.ask_with("what is the server ip?", None, &mut RawReader);
    assert_eq!(rt.telemetry.report().prefix_reuse_mean, None);
    rt.ask_with("what is the server ip?", None, &mut RawReader);
    // Identical bypass prompts: full reuse, recorded rather than skipped.
    assert_eq!(rt.telemetry.report().prefix_reuse_mean, Some(1.0));
    assert!(
        rt.last_assembly.is_empty(),
        "a bypassed turn must clear the planned membership it replaces"
    );
}

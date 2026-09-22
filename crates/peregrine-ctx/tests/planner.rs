//! Routing, budget pressure and what is allowed into the window.

use peregrine_ctx::ladder::Level;
use peregrine_ctx::nodes::{Kind, NewNode, Status};
use peregrine_ctx::policy::{DecisionPolicy, QueryType};
use peregrine_ctx::runtime::Dcr;

const TRANSCRIPT: &[(&str, &str)] = &[
    (
        "t1",
        "Goal: restore checkout by 09:00 UTC.\n\n\
         Constraint: never restart the payment service during business hours.",
    ),
    (
        "t2",
        "The error was \"connection refused\" when talking to the inventory host.",
    ),
    ("t3", "The server ip is 10.0.4.12 and the port is 8080."),
    (
        "t4",
        "The blocker is firewall rule 37, which drops checkout traffic.",
    ),
    (
        "t5",
        "Decision: roll back to build 4471 because the blocker is firewall rule 37.",
    ),
];

fn runtime() -> Dcr {
    let mut rt = Dcr::new(800);
    for (doc_id, text) in TRANSCRIPT {
        rt.ingest(text, Some(doc_id)).unwrap();
    }
    for i in 0..40 {
        rt.ingest(
            &format!("Chatter {i}: nothing relevant here, queue at {i} items."),
            Some(&format!("n{i}")),
        )
        .unwrap();
    }
    rt
}

// -- policy ---------------------------------------------------------------

#[test]
fn query_classification() {
    let policy = DecisionPolicy::default();
    let cases = [
        ("quote the exact error", QueryType::QuoteExact),
        ("what was the server ip?", QueryType::ValueLookup),
        ("how many retries were there?", QueryType::ValueLookup),
        ("why did we roll back?", QueryType::Justify),
        ("recompute the incident cost", QueryType::Recompute),
        ("summarize the incident", QueryType::Summarize),
    ];
    for (query, expected) in cases {
        assert_eq!(policy.classify(query), expected, "{query}");
    }
}

#[test]
fn stale_nodes_are_never_admitted_by_the_policy() {
    let policy = DecisionPolicy::default();
    let mut node = NewNode::new(Kind::Claim, "x")
        .spans(["s".to_string()])
        .key("k")
        .build();
    node.status = Status::Stale;
    let decision = policy.decide(&node, QueryType::ValueLookup, &[Level::L2], true);
    assert!(!decision.admitted);
    assert!(decision.recompute);
}

#[test]
fn deescalation_after_repeated_unproductive_l0_reads() {
    let policy = DecisionPolicy::default();
    let node = NewNode::new(Kind::Claim, "x")
        .spans(["s".to_string()])
        .key("k")
        .build();
    node.meta.l0_reads.set(3);
    node.meta.l0_yield.set(0);
    assert!(policy.should_deescalate(&node));
    node.meta.l0_yield.set(2);
    assert!(!policy.should_deescalate(&node));
}

// -- planner --------------------------------------------------------------

#[test]
fn value_query_uses_compact_state() {
    let mut rt = runtime();
    let context = rt.plan("what is the server ip?", None);
    assert!(context.entries.iter().all(|e| e.level == Level::L2));
}

#[test]
fn quote_query_promotes_to_raw() {
    let mut rt = runtime();
    let context = rt.plan("quote the exact error message", None);
    assert!(context.entries.iter().any(|e| e.level == Level::L0));
}

#[test]
fn budget_is_never_exceeded() {
    let mut rt = runtime();
    for budget in [30, 60, 120, 400, 2000] {
        let context = rt.plan("what is the server ip?", Some(budget));
        assert!(context.tokens <= budget, "budget {budget}");
    }
}

#[test]
fn budget_pressure_demotes_before_dropping() {
    let mut rt = runtime();
    let roomy = rt.plan("quote the exact error message", Some(800));
    let tight = rt.plan("quote the exact error message", Some(120));
    assert!(roomy.entries.iter().any(|e| e.level == Level::L0));
    assert!(
        !tight.entries.is_empty(),
        "something should survive at a cheaper level"
    );
    assert!(tight.entries.iter().all(|e| e.cost <= 120));
}

#[test]
fn goals_and_constraints_are_pinned() {
    let mut rt = runtime();
    let context = rt.plan("what is the server ip?", None);
    let kinds: Vec<Kind> = context.entries.iter().map(|e| e.kind).collect();
    assert!(kinds.contains(&Kind::Goal));
    assert!(kinds.contains(&Kind::Constraint));
}

#[test]
fn justification_pulls_the_dependency_path() {
    let mut rt = runtime();
    let context = rt.plan("why did we roll back?", None);
    let values: String = context
        .entries
        .iter()
        .map(|e| rt.graph.node(e.node).value.clone())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(values.contains("firewall rule 37"), "got {values}");
    assert!(!context.explain_paths.is_empty());
}

#[test]
fn noise_is_not_admitted() {
    let mut rt = runtime();
    let context = rt.plan("what is the server ip?", None);
    assert!(
        !context
            .entries
            .iter()
            .any(|e| rt.graph.node(e.node).value.contains("Chatter")),
        "irrelevant chatter must not reach the window"
    );
}

#[test]
fn stale_nodes_are_skipped_and_reported() {
    let mut rt = runtime();
    let claim = rt.graph.by_key("server.ip", true)[0];
    rt.graph.invalidate(claim, true);
    let context = rt.plan("what is the server ip?", None);
    assert!(!context.nodes().contains(&claim));
    assert!(context.stale_seen.contains(&claim));
}

#[test]
fn plan_records_a_snapshot_version() {
    let mut rt = runtime();
    let context = rt.plan("what is the server ip?", None);
    assert_eq!(context.snapshot_version, rt.graph.version);
}

#[test]
fn explain_plan_shows_the_arithmetic() {
    let mut rt = runtime();
    let context = rt.plan("what is the server ip?", None);
    let text = rt.explain_plan(&context);
    assert!(text.contains("ADMIT"));
    assert!(text.contains("similarity="));
}

/// The L0 memo must be a pure speed-up: same costs, same levels, same plan.
///
/// `Ladder::memoise_l0 = false` exists so the saving can be made to disappear
/// on purpose, but a control nobody runs is a control that never fires — the
/// same shape as the metric that could not report a failure and the ablation
/// row that could not pass. This runs it.
#[test]
fn the_l0_memo_changes_costs_for_no_node() {
    let mut memoised = runtime();
    let mut direct = runtime();
    direct.ladder.memoise_l0 = false;

    for query in [
        "what is the server ip?",
        "quote the exact error message",
        "why was the payment service not restarted?",
        "what is the deploy window?",
    ] {
        let a = memoised.plan(query, None);
        let b = direct.plan(query, None);
        assert_eq!(
            a.tokens, b.tokens,
            "memo changed the plan cost for {query:?}"
        );
        assert_eq!(
            a.node_ids(),
            b.node_ids(),
            "memo changed which nodes were admitted for {query:?}"
        );
        let levels: Vec<Level> = a.entries.iter().map(|e| e.level).collect();
        let direct_levels: Vec<Level> = b.entries.iter().map(|e| e.level).collect();
        assert_eq!(levels, direct_levels, "memo changed levels for {query:?}");
    }
}

/// And the flag has to actually do something, or the test above is vacuous.
#[test]
fn disabling_the_l0_memo_restores_the_repeated_work() {
    let mut rt = runtime();
    rt.ladder.memoise_l0 = false;
    let before = rt.ladder.l0_builds();
    rt.plan("what is the server ip?", None);
    let uncached = rt.ladder.l0_builds() - before;

    let mut rt = runtime();
    let before = rt.ladder.l0_builds();
    rt.plan("what is the server ip?", None);
    rt.plan("what is the server ip?", None);
    let cached = rt.ladder.l0_builds() - before;

    assert!(
        cached < uncached,
        "two memoised plans ({cached} builds) should cost fewer concatenations \
         than one unmemoised plan ({uncached} builds); if they do not, the memo \
         is not doing anything and neither is the flag"
    );
}

/// The embedder is a seam, not a hard-coded call.
///
/// The docs claimed "the interface accepts any `str -> Vec<f32>`" long before
/// it did — `Ladder` called the hashing embedder directly. This asserts the
/// claim rather than restating it: a replacement embedder must actually be the
/// one producing the vectors.
#[test]
fn the_vector_channel_is_swappable() {
    #[derive(Debug)]
    struct Constant;
    impl peregrine_ctx::embed::Embedder for Constant {
        fn embed(&self, _text: &str) -> Vec<f32> {
            vec![1.0, 0.0, 0.0, 0.0]
        }
        fn dim(&self) -> usize {
            4
        }
    }

    let mut rt = runtime();
    rt.ladder.embedder = Box::new(Constant);
    assert_eq!(rt.ladder.embedder.dim(), 4);
    assert_eq!(rt.ladder.query_vector("anything at all"), vec![1.0, 0.0, 0.0, 0.0]);

    // And the runtime still plans through it rather than falling back.
    let context = rt.plan("what is the server ip?", None);
    assert!(
        !context.entries.is_empty(),
        "a replaced embedder must still produce a plan"
    );
}

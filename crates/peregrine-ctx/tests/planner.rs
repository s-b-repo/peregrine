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

// -- prefix-stable layout ---------------------------------------------------

use peregrine_ctx::planner::{ActiveContext, Admission, UtilityTerms};

fn admission(node_id: &str, kind: Kind, utility: f32) -> Admission {
    Admission {
        node: 0,
        node_id: node_id.to_string(),
        kind,
        label: node_id.to_string(),
        level: Level::L2,
        cost: 10,
        utility,
        reason: "test",
        pinned: false,
        rendered: format!("[{node_id} L2] {node_id} = value"),
        terms: UtilityTerms::default(),
    }
}

fn context(tokens: usize, qtype: QueryType, entries: Vec<Admission>) -> ActiveContext {
    ActiveContext {
        query: "q".to_string(),
        qtype: Some(qtype),
        budget: 800,
        entries,
        tokens,
        considered: 42,
        ..Default::default()
    }
}

/// Same membership, different per-turn stats: the stable layout must be
/// byte-identical up to the trailer, while the ranked layout differs from
/// the first line — that first line is what kills the cached prefix today.
#[test]
fn prefix_stable_keeps_per_turn_data_out_of_the_prefix() {
    let entries = vec![
        admission("clai_b", Kind::Claim, 0.9),
        admission("clai_a", Kind::Claim, 0.1),
        admission("goal_1", Kind::Goal, 0.5),
    ];
    let turn1 = context(300, QueryType::ValueLookup, entries.clone());
    let turn2 = context(650, QueryType::Justify, entries);

    let stable1 = turn1.render_prefix_stable();
    let stable2 = turn2.render_prefix_stable();
    let trailer = "\n\n---\n";
    let cut1 = stable1.find(trailer).expect("trailer");
    let cut2 = stable2.find(trailer).expect("trailer");
    assert_eq!(stable1[..cut1], stable2[..cut2]);
    assert!(stable1[cut1..].contains("k=300/800 tokens"));
    assert!(stable2[cut2..].contains("k=650/800 tokens"));
    assert!(stable1[cut1..].contains("query type: value_lookup"));
    assert!(stable2[cut2..].contains("query type: justify"));

    assert_ne!(
        turn1.render().lines().next(),
        turn2.render().lines().next(),
        "the ranked header is per-turn data at byte zero"
    );
}

/// Within a kind, order follows the content address, never the per-turn
/// utility — a node whose score moved keeps its place if its text did not.
#[test]
fn prefix_stable_orders_by_content_not_utility() {
    let ctx = context(
        100,
        QueryType::Open,
        vec![
            admission("clai_b", Kind::Claim, 0.9),
            admission("clai_a", Kind::Claim, 0.1),
        ],
    );
    let stable = ctx.render_prefix_stable();
    assert!(
        stable.find("clai_a").unwrap() < stable.find("clai_b").unwrap(),
        "content order, not utility order: {stable}"
    );
    let ranked = ctx.render();
    assert!(
        ranked.find("clai_b").unwrap() < ranked.find("clai_a").unwrap(),
        "the ranked layout keeps utility order: {ranked}"
    );
    // Kind grouping is unchanged: goals still precede facts.
    let ctx = context(
        100,
        QueryType::Open,
        vec![
            admission("clai_a", Kind::Claim, 0.9),
            admission("goal_1", Kind::Goal, 0.1),
        ],
    );
    let stable = ctx.render_prefix_stable();
    assert!(stable.find("## GOALS").unwrap() < stable.find("clai_a").unwrap());
}

/// The runtime switch: `prefix_stable` changes what the reasoner is sent.
#[test]
fn prompt_uses_prefix_stable_layout_when_enabled() {
    let mut rt = runtime();
    rt.prefix_stable = true;
    let context = rt.plan("what was the server ip?", None);
    let prompt = rt.prompt_for(&context);
    assert!(prompt.starts_with("# ACTIVE CONTEXT\n"), "{prompt}");
    assert!(prompt.contains("\n\n---\nk="), "{prompt}");

    rt.prefix_stable = false;
    let context = rt.plan("what was the server ip?", None);
    let prompt = rt.prompt_for(&context);
    assert!(prompt.starts_with("# ACTIVE CONTEXT  (k="), "{prompt}");
}

// -- stickiness -------------------------------------------------------------

/// The retention bonus keeps last turn's membership from oscillating: with
/// stickiness on, a node admitted last turn wins a tie it would otherwise
/// lose. Measured on the runtime so `last_assembly` tracking is exercised,
/// not just the weight.
#[test]
fn stickiness_retains_the_previous_window() {
    fn membership(stickiness: f32, queries: &[&str]) -> Vec<Vec<String>> {
        let mut rt = runtime();
        rt.planner.weights.stickiness = stickiness;
        queries
            .iter()
            .map(|q| {
                rt.plan(q, None)
                    .node_ids()
                    .iter()
                    .map(|id| id.to_string())
                    .collect()
            })
            .collect()
    }

    // Two near-identical queries in a row: the second plan should overlap the
    // first at least as much with the bonus on as with it off.
    let queries = ["what was the server ip?", "what was the server ip address?"];
    let off = membership(0.0, &queries);
    let on = membership(0.9, &queries);
    let overlap = |a: &[String], b: &[String]| a.iter().filter(|id| b.contains(id)).count();
    assert!(
        overlap(&on[0], &on[1]) >= overlap(&off[0], &off[1]),
        "stickiness reduced retention: off={off:?} on={on:?}"
    );
}

/// `prefix_reuse` is recorded per turn from the prompts actually sent, and is
/// undefined — not zero — before two turns have run.
#[test]
fn prefix_reuse_is_measured_per_turn() {
    let mut rt = runtime();
    assert_eq!(rt.telemetry.report().prefix_reuse_mean, None);
    rt.ask("what was the server ip?", None);
    assert_eq!(rt.telemetry.report().prefix_reuse_mean, None);
    rt.ask("what was the server ip address?", None);
    let mean = rt.telemetry.report().prefix_reuse_mean;
    assert!(mean.is_some(), "two turns in, the metric must exist");
    let mean = mean.unwrap();
    assert!((0.0..=1.0).contains(&mean), "not a fraction: {mean}");
    // The stable layout must not reuse *less* than the ranked one on the same
    // near-identical queries.
    let mut stable = runtime();
    stable.prefix_stable = true;
    stable.ask("what was the server ip?", None);
    stable.ask("what was the server ip address?", None);
    let stable_mean = stable.telemetry.report().prefix_reuse_mean.unwrap();
    assert!(
        stable_mean >= mean,
        "stable layout reused less ({stable_mean}) than ranked ({mean})"
    );
}

/// The other half of the stickiness trade: the bonus is paid for with
/// membership decisions, so correctness with the weight on must be asserted,
/// not assumed. Same probes, same expectations, stickiness at the bench's
/// operating point.
#[test]
fn stickiness_does_not_cost_probe_correctness() {
    let corpus = peregrine_ctx::bench::build_corpus(300);
    let mut rt = Dcr::new(1200);
    rt.planner.weights.stickiness = 0.5;
    for (id, text) in &corpus.docs {
        rt.ingest(text, Some(id)).expect("ingest");
    }
    let mut reasoner = peregrine_ctx::llm::LocalReasoner::new();
    for probe in &corpus.probes {
        let answer = rt.ask_with(probe.query, None, &mut reasoner);
        assert!(
            answer.text.to_lowercase().contains(&probe.expected.to_lowercase()),
            "stickiness lost probe {:?}: {:?}",
            probe.label,
            answer.text
        );
    }
}

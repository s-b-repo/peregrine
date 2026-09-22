//! Provenance, invalidation and the audit path.

use peregrine_ctx::graph::{DcrError, MemoryGraph};
use peregrine_ctx::ids::Clock;
use peregrine_ctx::nodes::{EdgeType, Kind, NewNode, NodeIdx, Status};
use peregrine_ctx::spans::RawStore;

struct Fixture {
    clock: Clock,
    raw: RawStore,
    graph: MemoryGraph,
    span: String,
    evidence: NodeIdx,
}

fn fixture() -> Fixture {
    let clock = Clock::new();
    let mut raw = RawStore::default();
    let mut graph = MemoryGraph::new();
    let spans = raw
        .add_document(
            "The server ip is 10.0.4.12.\n\nFirewall rule 37 blocks 8080.\n\n\
             We rolled back to build 4471.",
            Some("d"),
            None,
            &clock,
        )
        .unwrap();
    let span = raw.span(spans[0]).id.clone();
    let text = raw.text_of(&span).to_string();
    let evidence = graph
        .upsert(
            NewNode::new(Kind::Evidence, text)
                .spans([span.clone()])
                .build(),
            &raw,
            &clock,
        )
        .unwrap();
    Fixture {
        clock,
        raw,
        graph,
        span,
        evidence,
    }
}

impl Fixture {
    fn claim(&mut self, value: &str, key: &str, deps: Option<Vec<String>>) -> NodeIdx {
        let deps = deps.unwrap_or_else(|| vec![self.graph.node(self.evidence).id.clone()]);
        self.graph
            .upsert(
                NewNode::new(Kind::Claim, value).deps(deps).key(key).build(),
                &self.raw,
                &self.clock,
            )
            .unwrap()
    }

    fn derive(&mut self, kind: Kind, value: &str, dep: NodeIdx) -> NodeIdx {
        let dep_id = self.graph.node(dep).id.clone();
        self.graph
            .upsert(
                NewNode::new(kind, value).deps([dep_id]).build(),
                &self.raw,
                &self.clock,
            )
            .unwrap()
    }
}

// -- provenance -----------------------------------------------------------

#[test]
fn unsourced_fact_is_rejected() {
    let f = fixture();
    let node = NewNode::new(Kind::Claim, "invented").key("nowhere").build();
    let mut graph = f.graph;
    assert!(matches!(
        graph.upsert(node, &f.raw, &f.clock),
        Err(DcrError::Provenance(_))
    ));
}

#[test]
fn evidence_without_spans_is_rejected() {
    let f = fixture();
    let mut graph = f.graph;
    let node = NewNode::new(Kind::Evidence, "no span here").build();
    assert!(matches!(
        graph.upsert(node, &f.raw, &f.clock),
        Err(DcrError::Provenance(_))
    ));
}

#[test]
fn unknown_span_is_rejected() {
    let f = fixture();
    let mut graph = f.graph;
    let node = NewNode::new(Kind::Evidence, "text")
        .spans(["s_missing".to_string()])
        .build();
    assert!(matches!(
        graph.upsert(node, &f.raw, &f.clock),
        Err(DcrError::Provenance(_))
    ));
}

#[test]
fn every_non_evidence_node_reaches_evidence() {
    let mut f = fixture();
    let claim = f.claim("10.0.4.12", "server.ip", None);
    let decision = f.derive(Kind::Decision, "roll back", claim);
    let path = f.graph.explain(decision, None);
    assert!(path.complete);
    assert!(path.nodes.contains(&f.evidence));
    assert!(path.spans.contains(&f.span));
}

#[test]
fn explain_reports_incompleteness_when_truncated() {
    let mut f = fixture();
    let mut node = f.claim("10.0.4.12", "server.ip", None);
    for i in 0..6 {
        node = f.derive(Kind::Calculation, &format!("step {i}"), node);
    }
    let path = f.graph.explain(node, Some(2));
    assert!(!path.complete);
    assert!(!path.truncated_at.is_empty());
}

// -- invalidation ---------------------------------------------------------

#[test]
fn invalidation_cascades_to_dependents() {
    let mut f = fixture();
    let claim = f.claim("10.0.4.12", "server.ip", None);
    let calc = f.derive(Kind::Calculation, "reachable", claim);
    let decision = f.derive(Kind::Decision, "roll back", calc);
    f.graph.invalidate(claim, true);
    assert_eq!(f.graph.node(calc).status, Status::Stale);
    assert_eq!(f.graph.node(decision).status, Status::Stale);
}

#[test]
fn cascade_is_bounded_and_defers_the_tail() {
    let mut f = fixture();
    f.graph.max_cascade = 3;
    let root = f.claim("10.0.4.12", "server.ip", None);
    let mut node = root;
    for i in 0..10 {
        node = f.derive(Kind::Calculation, &format!("step {i}"), node);
    }
    let marked = f.graph.invalidate(root, true);
    assert!(marked.len() <= 3);
    assert!(
        f.graph.pending_count() > 0,
        "tail must be deferred, not lost"
    );
    assert!(!f.graph.drain_pending(32).is_empty());
}

#[test]
fn evidence_is_never_marked_stale() {
    let mut f = fixture();
    let claim = f.claim("10.0.4.12", "server.ip", None);
    f.graph.invalidate(claim, true);
    assert_eq!(f.graph.node(f.evidence).status, Status::Fresh);
}

// -- revision -------------------------------------------------------------

#[test]
fn supersede_keeps_history() {
    let mut f = fixture();
    let old = f.claim("10.0.4.12", "server.ip", None);
    let new = f.claim("10.0.9.7", "server.ip", None);
    f.graph.supersede(old, new);
    assert_eq!(f.graph.node(old).status, Status::Superseded);
    assert_eq!(
        f.graph.node(new).meta.supersedes,
        vec![f.graph.node(old).id.clone()],
        "nothing is ever deleted"
    );
    let live: Vec<&str> = f
        .graph
        .by_key("server.ip", true)
        .into_iter()
        .map(|i| f.graph.node(i).value.as_str())
        .collect();
    assert_eq!(live, vec!["10.0.9.7"]);
}

#[test]
fn contradiction_keeps_both_sides() {
    let mut f = fixture();
    let a = f.claim("10.0.4.12", "server.ip", None);
    let b = f.claim("10.0.9.7", "server.ip", None);
    f.graph.contradict(a, b);
    assert_eq!(
        f.graph.neighbors(a, Some(EdgeType::Contradicts), true),
        vec![b]
    );
    // Both sides stay live and both say so: a node in an open disagreement
    // reports it in its own status rather than leaving a reader to notice the
    // edge.
    assert_eq!(f.graph.node(a).status, Status::Contradicted);
    assert_eq!(f.graph.node(b).status, Status::Contradicted);
    assert!(f.graph.node(a).status.is_live());
}

#[test]
fn resolving_a_contradiction_settles_the_survivor() {
    let mut f = fixture();
    let old = f.claim("10.0.4.12", "server.ip", None);
    let new = f.claim("10.0.9.7", "server.ip", None);
    f.graph.contradict(old, new);
    assert_eq!(f.graph.node(new).status, Status::Contradicted);

    // Once the correction supersedes the old value nothing live disagrees with
    // the survivor, so it must stop reading as contested — otherwise every
    // correction leaves a permanent open dispute behind it.
    f.graph.supersede(old, new);
    assert_eq!(f.graph.node(old).status, Status::Superseded);
    assert_eq!(f.graph.node(new).status, Status::Fresh);
    assert!(f.graph.node(new).meta.contradicts.is_empty());
}

#[test]
fn version_bumps_on_mutation() {
    let mut f = fixture();
    let before = f.graph.version;
    f.claim("10.0.4.12", "server.ip", None);
    assert!(f.graph.version > before);
}

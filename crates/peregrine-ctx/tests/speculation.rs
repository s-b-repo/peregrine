//! Speculative prefetch: budget isolation and learning from feedback.

use std::collections::HashSet;

use peregrine_ctx::ids::Clock;
use peregrine_ctx::ladder::Ladder;
use peregrine_ctx::nodes::{Kind, NewNode, Node};
use peregrine_ctx::spans::RawStore;
use peregrine_ctx::speculation::{Features, Predictor, Speculator};

fn features(sim: f32, proximity: f32, read_rate: f32, recency: f32, stale: f32) -> Features {
    Features {
        bias: 1.0,
        query_sim: sim,
        proximity,
        read_rate,
        recency,
        co_used: 0.3,
        stale,
    }
}

#[test]
fn learns_from_hit_and_miss_feedback() {
    let mut predictor = Predictor::default();
    let useful = features(0.9, 0.9, 0.9, 0.9, 0.0);
    let useless = features(0.05, 0.1, 0.0, 0.1, 0.0);
    let before = predictor.score(&useless);
    for _ in 0..50 {
        predictor.observe(&useful, true);
        predictor.observe(&useless, false);
    }
    assert!(predictor.score(&useless) < before);
    assert!(predictor.score(&useful) > predictor.score(&useless));
}

#[test]
fn stale_nodes_are_predicted_unlikely() {
    let predictor = Predictor::default();
    assert!(
        predictor.score(&features(0.8, 0.8, 0.5, 0.8, 0.0))
            > predictor.score(&features(0.8, 0.8, 0.5, 0.8, 1.0))
    );
}

struct Fixture {
    raw: RawStore,
    ladder: Ladder,
    nodes: Vec<Node>,
}

fn fixture() -> Fixture {
    let clock = Clock::new();
    let mut raw = RawStore::default();
    let text = (0..12)
        .map(|i| {
            format!(
                "Paragraph {i} with enough words to be worth summarising, repeatedly, so that L1 \
                 has something to do."
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let spans = raw.add_document(&text, Some("d"), None, &clock).unwrap();
    let nodes: Vec<Node> = spans
        .iter()
        .enumerate()
        .map(|(i, &span)| {
            let mut node = NewNode::new(Kind::Claim, format!("value {i}"))
                .spans([raw.span(span).id.clone()])
                .key(format!("k{i}"))
                .build();
            node.timestamp = clock.tick();
            node
        })
        .collect();
    Fixture {
        raw,
        ladder: Ladder::default(),
        nodes,
    }
}

#[test]
fn prefetch_respects_its_own_budget_and_not_the_attention_budget() {
    let f = fixture();
    let mut spec = Speculator::new(0.0, 4);
    let scored: Vec<(usize, &Node, f32, f32)> = f
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (i, n, 0.9, 1.0))
        .collect();
    let predictions = spec.predict(&scored, 100);
    let issued = spec.prefetch(&predictions, &f.nodes, &f.ladder, &f.raw);
    let builds = f.ladder.builds();
    assert!(builds.l1 + builds.l2 <= 4 + issued.len());
    assert!(
        issued.len() < f.nodes.len(),
        "budget must cap materialisation"
    );
}

#[test]
fn tau_gates_materialisation() {
    let f = fixture();
    let mut spec = Speculator::new(0.99, 10);
    let scored: Vec<(usize, &Node, f32, f32)> = f
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (i, n, 0.01, 0.01))
        .collect();
    let predictions = spec.predict(&scored, 100);
    assert!(
        spec.prefetch(&predictions, &f.nodes, &f.ladder, &f.raw)
            .is_empty()
    );
}

#[test]
fn feedback_records_hit_rate() {
    let f = fixture();
    let mut spec = Speculator::new(0.0, 3);
    let scored: Vec<(usize, &Node, f32, f32)> = f
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (i, n, 0.9, 1.0))
        .collect();
    let predictions = spec.predict(&scored, 100);
    let issued = spec.prefetch(&predictions, &f.nodes, &f.ladder, &f.raw);
    let used: HashSet<usize> = issued.first().map(|p| p.node).into_iter().collect();
    let (_issued, hits) = spec.feedback(&used);
    assert_eq!(hits, used.len());
    assert_eq!(spec.hits as usize, used.len());
    assert!(spec.predictor.updates > 0);
}

#[test]
fn co_use_is_remembered() {
    let mut spec = Speculator::default();
    let used: HashSet<usize> = [0, 1, 2].into_iter().collect();
    spec.feedback(&used);
    assert!(spec.co_use_row(0).is_some_and(|row| !row.is_empty()));
}

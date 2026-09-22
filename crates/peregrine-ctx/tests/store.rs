//! L0 store, ladder and budget solver.

use peregrine_ctx::budget::{Candidate, Choice, solve};
use peregrine_ctx::ids::Clock;
use peregrine_ctx::ladder::{Ladder, Level};
use peregrine_ctx::nodes::{Kind, NewNode};
use peregrine_ctx::spans::RawStore;
use std::collections::HashMap;

fn store() -> (Clock, RawStore) {
    (Clock::new(), RawStore::default())
}

#[test]
fn span_ids_are_stable_across_stores() {
    let text = "alpha beta\n\ngamma delta";
    let (c1, mut s1) = store();
    let (c2, mut s2) = store();
    let a = s1.add_document(text, Some("d"), None, &c1).unwrap();
    let b = s2.add_document(text, Some("d"), None, &c2).unwrap();
    let ids = |store: &RawStore, idx: &[usize]| -> Vec<String> {
        idx.iter().map(|&i| store.span(i).id.clone()).collect()
    };
    assert_eq!(ids(&s1, &a), ids(&s2, &b));
}

#[test]
fn reingesting_identical_content_is_a_noop() {
    let (clock, mut raw) = store();
    let first = raw
        .add_document("hello world", Some("d"), None, &clock)
        .unwrap();
    let again = raw
        .add_document("hello world", Some("d"), None, &clock)
        .unwrap();
    assert_eq!(first, again);
    assert_eq!(raw.documents().len(), 1);
}

#[test]
fn l0_is_immutable() {
    let (clock, mut raw) = store();
    raw.add_document("original", Some("d"), None, &clock)
        .unwrap();
    assert!(raw.add_document("edited", Some("d"), None, &clock).is_err());
}

#[test]
fn long_documents_split_into_addressable_spans() {
    let (clock, mut raw) = store();
    let text: String = (0..120)
        .map(|i| format!("Sentence number {i} about the incident. "))
        .collect();
    let spans = raw.add_document(&text, Some("d"), None, &clock).unwrap();
    assert!(spans.len() > 1);
    let rebuilt: String = spans
        .iter()
        .map(|&i| {
            let span = raw.span(i);
            text[span.start..span.end].to_string()
        })
        .collect();
    assert_eq!(rebuilt, text);
}

#[test]
fn neighbours_returns_surrounding_spans() {
    let (clock, mut raw) = store();
    let spans = raw
        .add_document("a\n\nb\n\nc\n\nd", Some("d"), None, &clock)
        .unwrap();
    let middle = raw.span(spans[2]).id.clone();
    let around: Vec<usize> = raw.neighbours(&middle, 1).iter().map(|s| s.seq).collect();
    assert_eq!(around, vec![1, 2, 3]);
}

// -- ladder ---------------------------------------------------------------

const LONG: &str = "The deploy failed at 04:12 with connection refused against the inventory \
host. The controller retried seven times before giving up and paging the on-call engineer, who \
rolled the fleet back to the previous build.";

fn ladder_fixture() -> (RawStore, Ladder, peregrine_ctx::nodes::Node) {
    let (clock, mut raw) = store();
    let spans = raw.add_document(LONG, Some("d"), None, &clock).unwrap();
    let span_id = raw.span(spans[0]).id.clone();
    let node = NewNode::new(Kind::Claim, "connection refused")
        .spans([span_id])
        .key("error")
        .confidence(0.8)
        .build();
    (raw, Ladder::default(), node)
}

#[test]
fn cost_ordering_l2_cheapest_l0_dearest() {
    let (raw, ladder, node) = ladder_fixture();
    let cost = |level| ladder.cost(&raw, &node, level);
    assert!(cost(Level::L2) < cost(Level::L1));
    assert!(cost(Level::L1) < cost(Level::L0));
    assert!(Level::L2.cheaper_than(Level::L0));
}

#[test]
fn higher_levels_are_built_lazily_and_cached() {
    let (raw, ladder, node) = ladder_fixture();
    assert_eq!(ladder.builds().l1, 0);
    ladder.summary(&raw, &node);
    ladder.summary(&raw, &node);
    assert_eq!(ladder.builds().l1, 1);
    assert!(node.level_cache.borrow().l1.is_some());
}

#[test]
fn l0_is_verbatim() {
    let (raw, ladder, node) = ladder_fixture();
    let rendered = ladder.render(&raw, &node, Level::L0);
    assert!(rendered.contains(raw.text_of(&node.source_spans[0])));
}

#[test]
fn l1_is_extractive() {
    let (raw, ladder, node) = ladder_fixture();
    let source = raw.text_of(&node.source_spans[0]).to_string();
    for sentence in ladder.summary(&raw, &node).split(". ") {
        let sentence = sentence.trim_matches(['.', ' ']);
        assert!(
            source.contains(sentence),
            "L1 introduced text not in L0: {sentence:?}"
        );
    }
}

// -- knapsack -------------------------------------------------------------

fn brute_force(candidates: &[Candidate], budget: usize) -> Option<f32> {
    fn walk(
        candidates: &[Candidate],
        i: usize,
        cost: usize,
        value: f32,
        budget: usize,
        best: &mut Option<f32>,
    ) {
        if cost > budget {
            return;
        }
        if i == candidates.len() {
            *best = Some(best.map_or(value, |b: f32| b.max(value)));
            return;
        }
        if !candidates[i].pinned {
            walk(candidates, i + 1, cost, value, budget, best);
        }
        for option in &candidates[i].options {
            walk(
                candidates,
                i + 1,
                cost + option.cost,
                value + option.utility,
                budget,
                best,
            );
        }
    }
    let mut best = None;
    walk(candidates, 0, 0, 0.0, budget, &mut best);
    best
}

/// Deterministic pseudo-random numbers: the test must not depend on a crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1000) as f32 / 1000.0
    }
}

#[test]
fn matches_brute_force_on_small_instances() {
    let mut rng = Lcg(7);
    for trial in 0..25 {
        let candidates: Vec<Candidate> = (0..6)
            .map(|i| {
                let levels = [Level::L2, Level::L1, Level::L0];
                let count = rng.range(1, 3);
                let options = levels[..count]
                    .iter()
                    .map(|&l| Choice::new(l, rng.range(3, 40), 0.1 + rng.unit() * 2.9))
                    .collect();
                Candidate::new(i, options, i == 0 && trial % 3 == 0)
            })
            .collect();
        let budget = rng.range(20, 120);
        let got = solve(&candidates, budget, Some(1), &HashMap::new());
        assert!(got.tokens <= budget);
        match brute_force(&candidates, budget) {
            // No feasible set — the pinned item cannot fit.
            None => assert!(got.overflow),
            Some(expected) => assert!(
                (got.utility - expected).abs() < 1e-4,
                "trial {trial}: got {} want {expected}",
                got.utility
            ),
        }
    }
}

#[test]
fn never_exceeds_the_budget() {
    let mut rng = Lcg(3);
    let candidates: Vec<Candidate> = (0..60)
        .map(|i| {
            Candidate::new(
                i,
                vec![
                    Choice::new(Level::L2, rng.range(5, 30), rng.unit()),
                    Choice::new(Level::L0, rng.range(40, 200), rng.unit() + 1.0),
                ],
                false,
            )
        })
        .collect();
    for budget in [0, 17, 250, 4000] {
        assert!(solve(&candidates, budget, None, &HashMap::new()).tokens <= budget);
    }
}

#[test]
fn demotes_before_evicting() {
    let candidates: Vec<Candidate> = (0..2)
        .map(|i| {
            Candidate::new(
                i,
                vec![
                    Choice::new(Level::L2, 10, 1.0),
                    Choice::new(Level::L0, 90, 1.4),
                ],
                false,
            )
        })
        .collect();
    let preferred = HashMap::from([(0, Level::L0), (1, Level::L0)]);
    let allocation = solve(&candidates, 40, None, &preferred);
    assert_eq!(
        allocation.chosen.len(),
        2,
        "both nodes should survive, demoted"
    );
    assert!(allocation.chosen.values().all(|o| o.level == Level::L2));
    assert_eq!(allocation.demoted.len(), 2);
    assert!(allocation.dropped.is_empty());
}

#[test]
fn pinned_candidates_are_admitted_first() {
    let candidates = vec![
        Candidate::new(0, vec![Choice::new(Level::L2, 30, 0.05)], true),
        Candidate::new(1, vec![Choice::new(Level::L2, 30, 5.0)], false),
    ];
    let allocation = solve(&candidates, 40, None, &HashMap::new());
    assert!(allocation.chosen.contains_key(&0));
    assert!(!allocation.chosen.contains_key(&1));
}

#[test]
fn overflow_is_reported_not_silently_exceeded() {
    let candidates = vec![Candidate::new(
        0,
        vec![Choice::new(Level::L0, 500, 1.0)],
        true,
    )];
    let allocation = solve(&candidates, 50, None, &HashMap::new());
    assert!(allocation.overflow);
    assert!(allocation.tokens <= 50);
}

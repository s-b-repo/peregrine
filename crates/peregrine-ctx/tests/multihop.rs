//! The multi-hop probes, and the property that makes them worth running.

use peregrine_ctx::bench::{build_multi_hop_corpus, MULTI_HOP};
use peregrine_ctx::text::content_tokens;

/// If a query's answer is lexically reachable on its own, the probe measures
/// search, not the join, and passes with graph expansion switched off. A first
/// version of this set did exactly that.
#[test]
fn the_second_hop_is_lexically_unreachable() {
    let corpus = build_multi_hop_corpus(50);
    for probe in MULTI_HOP {
        let q: Vec<String> = content_tokens(probe.query);
        let carrier = corpus
            .docs
            .iter()
            .find(|(_, text)| text.to_lowercase().contains(&probe.expected.to_lowercase()))
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("{}: no document carries the answer", probe.label));
        let shared: Vec<&String> = q
            .iter()
            .filter(|w| content_tokens(&carrier).contains(w))
            .collect();
        assert!(
            shared.is_empty(),
            "{}: the answer's document shares {:?} with the query, so it is reachable \
             without following an edge and the probe does not test the join",
            probe.label,
            shared
        );
    }
}

/// A chain written in natural order — the mention before the thing mentioned —
/// must still be joined. Reference linking only looked backwards, so it never
/// fired on this shape and the graph stayed a star.
#[test]
fn a_forward_mention_is_linked_when_its_target_arrives() {
    use peregrine_ctx::runtime::Dcr;
    let mut rt = Dcr::new(1200);
    rt.ingest("The readiness probe failures were traced to alpha-checkout.", Some("a"))
        .unwrap();
    rt.ingest("alpha-checkout is owned by team-payments.", Some("b"))
        .unwrap();

    let mentioner = rt
        .graph
        .nodes()
        .iter()
        .find(|n| n.value.to_lowercase().contains("alpha-checkout") && n.key.is_some())
        .expect("the mentioning claim");
    let target = rt
        .graph
        .nodes()
        .iter()
        .find(|n| n.key.as_deref() == Some("alpha-checkout"))
        .expect("the mentioned claim");
    assert!(
        mentioner.dependencies.contains(&target.id),
        "the earlier claim does not depend on the later one it names: deps={:?}",
        mentioner.dependencies
    );
}

/// Graph expansion and reference linking must both be load-bearing when the
/// answer genuinely needs a join. If this stops separating, either the fix
/// regressed or the probes stopped requiring the hop.
#[test]
fn expansion_and_linking_are_load_bearing_on_a_join() {
    use peregrine_ctx::bench::build_multi_hop_corpus;
    use peregrine_ctx::llm::LocalReasoner;
    use peregrine_ctx::runtime::Dcr;

    let solved = |apply: fn(&mut Dcr)| {
        let corpus = build_multi_hop_corpus(300);
        let mut rt = Dcr::new(1200);
        apply(&mut rt);
        for (id, t) in &corpus.docs {
            rt.ingest(t, Some(id)).expect("ingest");
        }
        let mut r = LocalReasoner::new();
        corpus
            .probes
            .iter()
            .filter(|p| {
                let a = rt.ask_with(p.query, None, &mut r);
                p.scores(&a.context.render())
            })
            .count()
    };
    let full = solved(|_| {});
    assert!(full > 0, "the full runtime solved no multi-hop probe");
    assert_eq!(solved(|rt| rt.planner.max_depth = 0), 0, "expansion off still solved one");
    assert_eq!(
        solved(|rt| rt.indexer.reference_linking = false),
        0,
        "linking off still solved one"
    );
}

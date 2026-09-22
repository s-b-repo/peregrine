//! Approximate retrieval must be a pruning step, never a source of silent
//! misses. These pin the two properties that make that true.

use peregrine_ctx::bench::build_corpus;
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;

fn answers(exact: bool, turns: usize) -> Vec<String> {
    let corpus = build_corpus(turns);
    let mut rt = Dcr::new(1200);
    rt.index.set_exact(exact);
    for (id, t) in &corpus.docs {
        rt.ingest(t, Some(id)).expect("ingest");
    }
    let mut r = LocalReasoner::new();
    corpus
        .probes
        .iter()
        .map(|p| rt.ask_with(p.query, None, &mut r).text)
        .collect()
}

#[test]
fn approximate_retrieval_does_not_change_correctness() {
    let corpus = build_corpus(3000);
    for exact in [true, false] {
        let got = answers(exact, 3000);
        let correct = corpus
            .probes
            .iter()
            .zip(&got)
            .filter(|(p, a)| a.to_lowercase().contains(&p.expected.to_lowercase()))
            .count();
        assert_eq!(
            correct,
            corpus.probes.len(),
            "exact={exact} lost a probe: {got:?}"
        );
    }
}

#[test]
fn the_index_is_deterministic() {
    // Hyperplanes come from a fixed seed. Two runs of one configuration that
    // disagreed would make any ablation built on top unattributable.
    assert_eq!(answers(false, 300), answers(false, 300));
}

#[test]
fn a_small_store_bypasses_the_structure() {
    // Below the threshold a full scan is already cheap and bucketing only costs
    // recall, so the approximate path must be identical to the exact one there.
    assert_eq!(answers(false, 100), answers(true, 100));
}

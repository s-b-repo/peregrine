//! Correction-following must not depend on how much context the planner admits.
//!
//! It used to. Seeding excluded evidence whose every live dependent had been
//! superseded — it still carries the old value verbatim, and a matcher that
//! ignores notes answers from it — but `expand` re-admitted exactly those nodes
//! through a dependency edge. A guard on one entrance of a room with two doors.
//!
//! It surfaced by accident. Raising `seed_min_ratio` from 0.3 to 0.5 cut the
//! working set from 461.9 tokens to 145.1 with all seven standard probes still
//! passing, which looked like a free 3x saving; on the adversarial mutation set
//! the same change served the superseded value on 3 of 4 queries. The threshold
//! was never the cause — it changed which nodes seeded, which changed which
//! expansions ran, and let the excluded node in through the second door.
//!
//! `expand` now applies the same rule `seed` does, and the tests below pin the
//! property and its control separately.

use peregrine_ctx::bench::{ADVERSARIAL, build_mutation_corpus_from};
use peregrine_ctx::index::{Fusion, Namespace};
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;

/// (corrected, stale served) over the adversarial mutation set.
fn adversarial(floor: f32, fusion: Fusion) -> (usize, usize) {
    score(floor, fusion, false)
}

/// The same, with the staleness exclusion disabled.
fn adversarial_admitting_stale(floor: f32) -> (usize, usize) {
    score(floor, Fusion::Linear, true)
}

fn score(floor: f32, fusion: Fusion, admit_stale: bool) -> (usize, usize) {
    let corpus = build_mutation_corpus_from(300, ADVERSARIAL);
    let mut rt = Dcr::new(1200);
    rt.planner.seed_min_ratio = floor;
    rt.planner.admit_stale = admit_stale;
    rt.index.fusion = fusion;
    for (id, t) in &corpus.docs {
        rt.ingest(t, Some(id)).expect("ingest");
    }
    let mut reasoner = LocalReasoner::new();
    let (mut corrected, mut stale) = (0, 0);
    for m in ADVERSARIAL {
        let text = rt.ask_with(m.query, None, &mut reasoner).text.to_lowercase();
        if text.contains(&m.live.to_lowercase()) {
            corrected += 1;
        }
        if text.contains(&m.stale.to_lowercase()) {
            stale += 1;
        }
    }
    (corrected, stale)
}

#[test]
fn corrections_survive_every_seed_floor() {
    // The property. How much context the planner admits is a cost decision; it
    // must not be a correctness decision. Before the expansion guard this
    // failed at 0.5 and above.
    for floor in [0.3, 0.5, 0.7, 0.85] {
        let (corrected, stale) = adversarial(floor, Fusion::Linear);
        assert_eq!(
            corrected,
            ADVERSARIAL.len(),
            "floor {floor} lost corrections: {corrected}/{}",
            ADVERSARIAL.len()
        );
        assert_eq!(stale, 0, "floor {floor} served {stale} superseded values");
    }
}

#[test]
fn admitting_stale_evidence_breaks_correction_following() {
    // The control for the test above, and the reason it is not a check that
    // cannot fail. `admit_stale` disables the exclusion that both `seed` and
    // `expand` apply. With it off the corpus is answered correctly at every
    // floor; with it on, every query serves the superseded value at every
    // floor. So the guard is what carries corrections, demonstrated rather
    // than asserted.
    for floor in [0.3, 0.5, 0.7, 0.85] {
        let (corrected, stale) = adversarial_admitting_stale(floor);
        assert_eq!(
            corrected, 0,
            "floor {floor}: admitting stale evidence should lose every correction"
        );
        assert_eq!(
            stale,
            ADVERSARIAL.len(),
            "floor {floor}: admitting stale evidence should serve every superseded value"
        );
    }
}

#[test]
fn rank_fusion_is_off_by_default() {
    // Every published figure was measured on the linear blend.
    assert_eq!(Dcr::new(1200).index.fusion, Fusion::Linear);
}

#[test]
fn rank_fusion_changes_the_ranking_it_is_offered_to_change() {
    // An option that cannot be observed to do anything is not an option, and
    // `bench --fusion` would be comparing a configuration against itself.
    let corpus = build_mutation_corpus_from(300, ADVERSARIAL);
    let mut rt = Dcr::new(1200);
    for (id, t) in &corpus.docs {
        rt.ingest(t, Some(id)).expect("ingest");
    }
    let query = ADVERSARIAL[0].query;
    let qv = peregrine_ctx::embed::hashing_embed(query, peregrine_ctx::embed::DIM);
    let linear = rt.index.search(Namespace::Node, query, &qv, 12);
    rt.index.fusion = Fusion::Rrf { k0: 60.0 };
    let rrf = rt.index.search(Namespace::Node, query, &qv, 12);
    assert_ne!(
        linear.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        rrf.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        "rank fusion returned the same order as the linear blend"
    );
}

#[test]
fn rank_fusion_scores_stay_on_the_linear_scale() {
    // RRF's raw scores are ~1/60 of the linear path's. Anything downstream that
    // reads a score as a magnitude — the seed floor is one — would change
    // behaviour for a reason unrelated to ranking, so the fused list is
    // rescaled to top = 1.0. This pins that.
    let corpus = build_mutation_corpus_from(300, ADVERSARIAL);
    let mut rt = Dcr::new(1200);
    rt.index.fusion = Fusion::Rrf { k0: 60.0 };
    for (id, t) in &corpus.docs {
        rt.ingest(t, Some(id)).expect("ingest");
    }
    let query = ADVERSARIAL[0].query;
    let qv = peregrine_ctx::embed::hashing_embed(query, peregrine_ctx::embed::DIM);
    let top = rt.index.search(Namespace::Node, query, &qv, 12);
    let top = top.first().expect("a hit").1;
    assert!(
        (top - 1.0).abs() < 1e-5,
        "fused top score should be rescaled to 1.0, got {top}"
    );
}

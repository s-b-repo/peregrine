//! The mutation-and-correction probe, and the property that makes it worth
//! having: its negative control can actually fail.
//!
//! `stale_fact_read_rate` counts context entries whose node carries
//! `Status::Stale`, and `supersede_on_conflict` gates whether that status is
//! ever set — so the metric reads 0.00 both when the runtime correctly excludes
//! stale nodes and when it never marked any. These tests pin the ground-truth
//! measurement that distinguishes those two cases.

use peregrine_ctx::bench::{MUTATIONS, build_mutation_corpus};
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;

/// Ingest the mutation corpus and return, per mutation, whether the answer
/// carried the live value and whether it carried the superseded one.
fn serve(supersede: bool) -> Vec<(bool, bool)> {
    let corpus = build_mutation_corpus(300);
    let mut runtime = Dcr::new(1200);
    runtime.indexer.supersede_on_conflict = supersede;
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id)).expect("ingest");
    }
    let mut reasoner = LocalReasoner::new();
    MUTATIONS
        .iter()
        .map(|m| {
            let text = runtime
                .ask_with(m.query, None, &mut reasoner)
                .text
                .to_lowercase();
            (
                text.contains(&m.live.to_lowercase()),
                text.contains(&m.stale.to_lowercase()),
            )
        })
        .collect()
}

#[test]
fn corrections_are_served_once_the_original_has_dependents() {
    let results = serve(true);
    let stale: Vec<&str> = results
        .iter()
        .zip(MUTATIONS)
        .filter(|((_, served_stale), _)| *served_stale)
        .map(|(_, m)| m.label)
        .collect();
    assert!(
        stale.is_empty(),
        "the full runtime served superseded values for: {stale:?}"
    );
}

#[test]
fn the_negative_control_can_fail() {
    // The whole point of the probe. If disabling supersession does not produce
    // a single stale read, the probe is not measuring what it claims to and the
    // "0 stale reads" result in the full runtime is unfalsifiable.
    let without = serve(false);
    let stale_reads = without.iter().filter(|(_, s)| *s).count();
    assert!(
        stale_reads > 0,
        "disabling supersession served no superseded value; the control cannot \
         fire, so a passing run proves nothing"
    );
}

#[test]
fn supersession_strictly_improves_correction_delivery() {
    let with = serve(true).iter().filter(|(live, _)| *live).count();
    let without = serve(false).iter().filter(|(live, _)| *live).count();
    assert!(
        with > without,
        "supersession did not improve correction delivery ({with} vs {without}); \
         either the mechanism is not load-bearing on this corpus or the probe \
         is not exercising it"
    );
}

#[test]
fn ground_truth_values_are_distinguishable() {
    // A case whose stale and live values are substrings of one another would
    // make every measurement above meaningless.
    for m in MUTATIONS {
        let (s, l) = (m.stale.to_lowercase(), m.live.to_lowercase());
        assert!(
            !s.contains(&l) && !l.contains(&s),
            "{}: stale {:?} and live {:?} are not distinguishable by substring",
            m.label,
            m.stale,
            m.live
        );
    }
}

#[test]
fn a_coordinated_restatement_carries_the_subject() {
    use peregrine_ctx::indexer::HeuristicExtractor;
    let ex = HeuristicExtractor::default();
    // "was <participle> and is now <value>" — the subject carries across the
    // conjunction, so the live value is the one after "is now", not the
    // participle. Both the bare and the "Correction:"-prefixed forms must
    // agree, because the prefix sends the parse down a different path.
    for text in [
        "The primary datastore was migrated and is now postgres-15 on host db-omega.",
        "Correction: the primary datastore was migrated and is now postgres-15 on host db-omega.",
    ] {
        let claims: Vec<_> = ex
            .extract(text)
            .into_iter()
            .filter(|p| p.key.as_deref() == Some("primary.datastore"))
            .collect();
        assert_eq!(
            claims.len(),
            1,
            "{text:?} produced {} claims on one key: {:?}",
            claims.len(),
            claims.iter().map(|c| &c.value).collect::<Vec<_>>()
        );
        assert!(
            claims[0].value.contains("postgres-15"),
            "{text:?} extracted {:?}, expected the value after \"is now\"",
            claims[0].value
        );
    }
}

#[test]
fn a_fronted_adverb_is_not_part_of_the_value() {
    use peregrine_ctx::indexer::HeuristicExtractor;
    let ex = HeuristicExtractor::default();
    let claims = ex.extract("The failover region is now eu-central-1.");
    let c = claims
        .iter()
        .find(|p| p.key.as_deref() == Some("failover.region"))
        .expect("claim on failover.region");
    assert_eq!(c.value, "eu-central-1", "the adverb leaked into the value");
}

/// Corrections in the wild carry verbs. A copula-only extractor drops most of
/// them silently — no fact at all, rather than a wrong one — which is the
/// quieter half of the `migrated` bug.
#[test]
fn corrections_phrased_with_verbs_are_extracted() {
    use peregrine_ctx::indexer::HeuristicExtractor;
    let ex = HeuristicExtractor::default();
    for text in [
        "The primary datastore has moved to postgres-15.",
        "The primary datastore was replaced by postgres-15.",
        "The primary datastore changed to postgres-15.",
        "The primary datastore should be postgres-15.",
        "The primary datastore is now postgres-15.",
        "The primary datastore was migrated and is now postgres-15.",
    ] {
        let values: Vec<String> = ex
            .extract(text)
            .into_iter()
            .filter(|p| p.key.as_deref() == Some("primary.datastore"))
            .map(|p| p.value)
            .collect();
        assert!(
            values.iter().any(|v| v.contains("postgres-15")),
            "{text:?} extracted {values:?}, expected the value after the transition"
        );
        assert!(
            !values.iter().any(|v| v.contains("postgres-11")),
            "{text:?} also extracted a stale value: {values:?}"
        );
    }
}

/// The point of the adversarial set: when the superseded value is the better
/// lexical match, only supersession can produce the right answer. If this ever
/// passes with supersession disabled, the set has stopped being adversarial.
#[test]
fn supersession_beats_lexical_attraction() {
    use peregrine_ctx::bench::{build_mutation_corpus_from, ADVERSARIAL};
    use peregrine_ctx::llm::LocalReasoner;
    use peregrine_ctx::runtime::Dcr;

    let run = |supersede: bool| {
        let corpus = build_mutation_corpus_from(300, ADVERSARIAL);
        let mut rt = Dcr::new(1200);
        rt.indexer.supersede_on_conflict = supersede;
        for (id, t) in &corpus.docs {
            rt.ingest(t, Some(id)).expect("ingest");
        }
        let mut r = LocalReasoner::new();
        ADVERSARIAL
            .iter()
            .filter(|m| {
                rt.ask_with(m.query, None, &mut r)
                    .text
                    .to_lowercase()
                    .contains(&m.stale.to_lowercase())
            })
            .count()
    };
    assert_eq!(run(true), 0, "the full runtime served a superseded value");
    assert_eq!(
        run(false),
        ADVERSARIAL.len(),
        "with supersession off the stale value should win every case — if it \
         does not, the stale node is not actually the more attractive target \
         and the set proves nothing"
    );
}

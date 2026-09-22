//! Controls proposed by readers, and the properties that make them worth running.

use peregrine_ctx::bench::{build_corpus, build_corpus_permuted};
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;

/// The residual must be measured, not defined.
///
/// `StageProfile::total()` sums the clocked stages, so reporting it as "planner
/// time" makes the residual identically zero and an unclocked stage invisible.
/// `measured_time` is an independent clock around the whole call; this pins that
/// the two are genuinely different numbers.
#[test]
fn the_plan_clock_is_independent_of_its_stages() {
    let corpus = build_corpus(300);
    let mut rt = Dcr::new(1200);
    for (id, t) in &corpus.docs {
        rt.ingest(t, Some(id)).expect("ingest");
    }
    let mut r = LocalReasoner::new();
    for p in &corpus.probes {
        rt.ask_with(p.query, None, &mut r);
    }
    let prof = rt.planning;
    assert!(
        prof.measured_time > std::time::Duration::ZERO,
        "the independent plan clock never ran"
    );
    assert!(
        prof.measured_time >= prof.total(),
        "the whole call measured less than the sum of its stages ({:?} < {:?}) — \
         the outer clock is not bracketing them",
        prof.measured_time,
        prof.total()
    );
}

/// The digest has to change when the policy changes, or binding it to a receipt
/// records nothing. Deliberately break it: alter one knob, expect a new digest.
#[test]
fn the_policy_digest_moves_when_the_policy_does() {
    let base = Dcr::new(1200);
    let before = base.planner.policy_digest();

    let mut changed = Dcr::new(1200);
    changed.planner.max_depth += 1;
    assert_ne!(
        before,
        changed.planner.policy_digest(),
        "changing max_depth left the digest identical — it is not covering the caps"
    );

    let mut weighted = Dcr::new(1200);
    weighted.planner.weights.similarity += 0.5;
    assert_ne!(
        before,
        weighted.planner.policy_digest(),
        "changing a utility weight left the digest identical"
    );

    // Same configuration, same digest — otherwise it cannot identify anything.
    assert_eq!(before, Dcr::new(1200).planner.policy_digest());
}

/// The decoy corpus must actually contain decoys: every subject mentioned by
/// more than one document, or "found the document" and "found the subject"
/// remain the same event and the control proves nothing.
#[test]
fn the_subject_control_actually_adds_decoys() {
    let plain = build_corpus(300);
    let decoyed = build_corpus_permuted(300);
    assert!(
        decoyed.docs.len() > plain.docs.len(),
        "the decoy corpus added no documents"
    );
    let mentions = |c: &peregrine_ctx::bench::Corpus, subject: &str| {
        c.docs
            .iter()
            .filter(|(_, t)| t.to_lowercase().contains(subject))
            .count()
    };
    for subject in ["the server ip", "the deploy window", "the hourly rate"] {
        assert!(
            mentions(&decoyed, subject) > mentions(&plain, subject),
            "{subject:?} is mentioned no more often than in the standard corpus"
        );
    }
}

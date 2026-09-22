//! The diverse corpus has to actually be diverse, and that has to be measured.

use peregrine_ctx::bench::{build_corpus, build_corpus_diverse};
use std::collections::HashSet;

/// Distinct documents, ignoring digits — two sentences differing only by an
/// index are the same sentence for a retriever.
fn shapes(corpus: &peregrine_ctx::bench::Corpus) -> usize {
    corpus
        .docs
        .iter()
        .map(|(_, t)| t.chars().filter(|c| !c.is_ascii_digit()).collect::<String>())
        .collect::<HashSet<_>>()
        .len()
}

#[test]
fn corpus_diversity_is_measured_not_asserted() {
    let n = 30_000;
    let standard = shapes(&build_corpus(n));
    let diverse = shapes(&build_corpus_diverse(n));
    // The standard corpus is eight templates plus the signal documents.
    assert!(standard < 40, "standard corpus unexpectedly varied: {standard}");
    // The point of the diverse one. A first attempt used independent strides
    // and produced 26 — repeating at the lcm of their periods rather than the
    // product — while its comment claimed the opposite.
    assert!(
        diverse > 5_000,
        "diverse corpus produced only {diverse} distinct documents in {n}; \
         it is not diverse and must not be described as such"
    );
}

//! The load-bearing claim: `k` stays bounded while `N` grows.

use peregrine_ctx::bench::build_corpus;
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;
use peregrine_ctx::tokens::estimate_tokens;

struct Measurement {
    history: usize,
    mean_k: f64,
    max_k: usize,
    correct: usize,
    probes: usize,
    stale_reads: Option<f64>,
}

fn measure(turns: usize, budget: usize) -> Measurement {
    let corpus = build_corpus(turns);
    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in &corpus.docs {
        runtime.ingest(text, Some(doc_id)).unwrap();
    }
    let mut reasoner = LocalReasoner::new();
    let mut correct = 0;
    for probe in &corpus.probes {
        let answer = runtime.ask_with(probe.query, None, &mut reasoner);
        if answer
            .text
            .to_lowercase()
            .contains(&probe.expected.to_lowercase())
        {
            correct += 1;
        }
    }
    let report = runtime.telemetry.report();
    Measurement {
        history: estimate_tokens(&corpus.text()),
        mean_k: report.tokens_per_query_mean,
        max_k: report.tokens_per_query_max,
        correct,
        probes: corpus.probes.len(),
        stale_reads: report.stale_fact_read_rate,
    }
}

#[test]
fn active_context_stays_bounded_while_history_grows() {
    let small = measure(60, 800);
    let large = measure(600, 800);

    assert!(
        large.history as f64 / small.history as f64 > 5.0,
        "the test corpus must actually grow"
    );
    assert!(
        large.mean_k / small.mean_k < 1.5,
        "k must not track N: {} -> {}",
        small.mean_k,
        large.mean_k
    );
    assert!(large.max_k <= 800);

    for case in [&small, &large] {
        assert_eq!(case.correct, case.probes, "answers must stay correct");
        assert_eq!(
            case.stale_reads,
            Some(0.0),
            "no stale fact may reach the model"
        );
    }
}

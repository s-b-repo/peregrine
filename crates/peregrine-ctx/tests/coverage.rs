//! The two controls proposed against the DCR report by "vespermind":
//! a positive control that proves `stale_fact_read_rate` can fire, and read
//! coverage — the offline dual of audit-path completeness.

use peregrine_ctx::bench::build_corpus;
use peregrine_ctx::llm::LocalReasoner;
use peregrine_ctx::runtime::Dcr;

/// A zero is only evidence if the run could have produced nonzero. Poison a
/// derived value (correct its input so it goes stale), then ask for it: with
/// the guard on the stale node is excluded (rate 0), with the guard bypassed it
/// is read (rate 1). Expected values are asserted, not observed after the fact.
fn poisoned_stale_read_rate(admit_stale: bool) -> f64 {
    let mut rt = Dcr::new(600);
    rt.planner.admit_stale = admit_stale;
    rt.register("incident_cost", |inputs| {
        let get = |n: &str| {
            inputs
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };
        get("rate") * get("hours")
    });
    rt.ingest("The hourly rate is 180 USD.", Some("t1"))
        .unwrap();
    rt.ingest("The incident lasted 4 hours.", Some("t2"))
        .unwrap();
    let rate = *rt.graph.by_key("hourly.rate", true).last().unwrap();
    let deps = vec![rt.graph.node(rate).id.clone()];
    rt.compute(
        "incident_cost",
        vec![("rate".into(), 180.0), ("hours".into(), 4.0)],
        deps,
        Some("incident.cost"),
    )
    .unwrap();
    rt.ingest("Correction: the hourly rate is 210 USD.", Some("t3"))
        .unwrap();
    let cost = rt.graph.by_key("incident.cost", true)[0];
    assert_eq!(rt.graph.node(cost).status.as_str(), "stale");
    let mut reasoner = LocalReasoner::new();
    rt.ask_with("what is the incident cost?", None, &mut reasoner);
    rt.telemetry.report().stale_fact_read_rate.unwrap_or(0.0)
}

#[test]
fn stale_read_metric_can_fire_and_the_guard_drives_it_to_zero() {
    // Negative control: without the guard, the stale node is read.
    assert_eq!(
        poisoned_stale_read_rate(true),
        1.0,
        "metric must be able to return nonzero, or a production zero means nothing"
    );
    // Positive control: with the guard, it is excluded.
    assert_eq!(
        poisoned_stale_read_rate(false),
        0.0,
        "the guard must exclude a wanted stale node"
    );
}

#[test]
fn read_coverage_stays_flat_while_history_grows() {
    let coverage = |turns: usize| {
        let corpus = build_corpus(turns);
        let mut rt = Dcr::new(800);
        for (doc_id, text) in &corpus.docs {
            rt.ingest(text, Some(doc_id)).unwrap();
        }
        let mut reasoner = LocalReasoner::new();
        for probe in &corpus.probes {
            rt.ask_with(probe.query, None, &mut reasoner);
        }
        rt.coverage()
    };
    let small = coverage(100);
    let large = coverage(3000);

    // History grows ~30x.
    assert!(large.total_spans as f64 / small.total_spans as f64 > 20.0);
    // The set of spans whose bytes were ever shown does NOT grow with it.
    assert!(
        large.assembled_spans <= small.assembled_spans + 4,
        "content coverage should stay flat: {} -> {}",
        small.assembled_spans,
        large.assembled_spans
    );
    // So the covered fraction collapses toward zero.
    assert!(large.fraction < small.fraction / 5.0);
    // A span shown at L0 is a real, bounded thing — not a corroboration sink.
    assert!(large.assembled_spans < 30);
}

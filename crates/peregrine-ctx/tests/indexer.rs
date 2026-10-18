//! Extraction, contradiction handling and corroboration.

use peregrine_ctx::indexer::HeuristicExtractor;
use peregrine_ctx::nodes::{Kind, Status};
use peregrine_ctx::runtime::Dcr;

fn keys(text: &str) -> Vec<(Option<String>, String)> {
    HeuristicExtractor::default()
        .extract(text)
        .into_iter()
        .filter(|p| p.kind == Kind::Claim)
        .map(|p| (p.key, p.value))
        .collect()
}

fn value_for(text: &str, key: &str) -> Option<String> {
    keys(text)
        .into_iter()
        .find(|(k, _)| k.as_deref() == Some(key))
        .map(|(_, v)| v)
}

#[test]
fn dotted_identifiers_survive() {
    assert_eq!(
        value_for("The server ip is 10.0.4.12.", "server.ip").as_deref(),
        Some("10.0.4.12")
    );
}

#[test]
fn two_facts_in_one_sentence() {
    let text = "The server ip is 10.0.4.12 and the port is 8080.";
    assert_eq!(value_for(text, "server.ip").as_deref(), Some("10.0.4.12"));
    assert_eq!(value_for(text, "port").as_deref(), Some("8080"));
}

#[test]
fn structural_prefix_does_not_hide_the_fact() {
    let text = "Correction: actually the server ip is 10.0.9.7, we misread it.";
    assert_eq!(value_for(text, "server.ip").as_deref(), Some("10.0.9.7"));
}

#[test]
fn quoted_value() {
    let text = "The error was \"connection refused\" on port 8080.";
    assert_eq!(
        value_for(text, "error").as_deref(),
        Some("connection refused")
    );
}

#[test]
fn kinds_are_recognised() {
    let extractor = HeuristicExtractor::default();
    let mut kinds: Vec<Kind> = [
        "Goal: restore checkout by 09:00.",
        "Constraint: never restart payments during business hours.",
        "Decision: roll back to build 4471.",
        "Should we page the on-call engineer?",
    ]
    .iter()
    .flat_map(|t| extractor.extract(t))
    .map(|p| p.kind)
    .collect();
    kinds.sort();
    kinds.dedup();
    for expected in [
        Kind::Goal,
        Kind::Constraint,
        Kind::Decision,
        Kind::OpenQuestion,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}

#[test]
fn unstructured_chatter_yields_nothing() {
    let text = "just some chatter about lunch, nobody said anything actionable";
    assert!(HeuristicExtractor::default().extract(text).is_empty());
}

// -- contradictions -------------------------------------------------------

fn seeded() -> Dcr {
    let mut rt = Dcr::new(400);
    rt.ingest("The server ip is 10.0.4.12.", Some("t1"))
        .unwrap();
    rt
}

#[test]
fn conflicting_value_supersedes_and_keeps_both() {
    let mut rt = seeded();
    let result = rt
        .ingest("Correction: the server ip is 10.0.9.7.", Some("t2"))
        .unwrap();
    assert_eq!(result.contradictions.len(), 1);
    let history: Vec<(String, Status)> = rt
        .graph
        .by_key("server.ip", false)
        .into_iter()
        .map(|i| (rt.graph.node(i).value.clone(), rt.graph.node(i).status))
        .collect();
    assert_eq!(
        history,
        vec![
            ("10.0.4.12".to_string(), Status::Superseded),
            ("10.0.9.7".to_string(), Status::Fresh),
        ]
    );
}

#[test]
fn superseded_fact_never_reaches_the_model() {
    let mut rt = seeded();
    rt.ingest("Correction: the server ip is 10.0.9.7.", Some("t2"))
        .unwrap();
    let rendered = rt.plan("what is the server ip?", None).render();
    assert!(rendered.contains("10.0.9.7"));
    assert!(!rendered.contains("server.ip = 10.0.4.12"));
}

#[test]
fn restating_a_fact_is_corroboration_not_duplication() {
    let mut rt = seeded();
    let before = rt.graph.by_key("server.ip", true);
    let confidence_before = rt.graph.node(before[0]).confidence;
    rt.ingest("As noted, the server ip is 10.0.4.12.", Some("t3"))
        .unwrap();
    let after = rt.graph.by_key("server.ip", true);
    assert_eq!(after.len(), before.len());
    let node = rt.graph.node(after[0]);
    assert!(node.confidence > confidence_before);
    assert_eq!(node.source_spans.len(), 2);
}

#[test]
fn reference_links_build_a_multi_hop_path() {
    let mut rt = Dcr::new(400);
    rt.ingest("The blocker is firewall rule 37.", Some("a"))
        .unwrap();
    rt.ingest(
        "Decision: roll back to build 4471 because the blocker is firewall rule 37.",
        Some("b"),
    )
    .unwrap();
    let decision = rt.graph.by_kind(Kind::Decision, true)[0];
    let path = rt.graph.explain(decision, None);
    let values: Vec<&str> = path
        .nodes
        .iter()
        .map(|&i| rt.graph.node(i).value.as_str())
        .collect();
    assert!(path.complete);
    assert!(values.contains(&"firewall rule 37"), "got {values:?}");
}

#[test]
fn out_of_order_material_cannot_revert_a_correction() {
    let mut rt = Dcr::new(400);
    rt.ingest("Correction: actually the server ip is 10.0.9.7.", Some("b"))
        .unwrap();
    rt.ingest("The server ip is 10.0.4.12.", Some("a")).unwrap();
    assert_eq!(
        rt.graph.by_key("server.ip", true).len(),
        2,
        "both sides stay live for adjudication"
    );
    assert!(
        rt.plan("what is the server ip?", None)
            .render()
            .contains("CONTRADICTS=")
    );
}

#[test]
fn a_resolved_contradiction_leaves_no_trace_of_the_old_value() {
    let mut rt = seeded();
    rt.ingest("Correction: the server ip is 10.0.9.7.", Some("t2"))
        .unwrap();
    let rendered = rt.plan("what is the server ip?", None).render();
    // Resolved, so the adjudication warning goes away.
    assert!(!rendered.contains("CONTRADICTS="));
    // And the superseded value does not reach the window at all. It used to,
    // annotated "source of a superseded fact; do not treat as current" — which
    // is a guard only against a model that reads notes. On the adversarial
    // mutation set the superseded sentence out-scores the correction on lexical
    // overlap, so a matcher that ignores the note answered from it every time.
    assert!(
        !rendered.contains("10.0.4.12"),
        "superseded value reached the window:\n{rendered}"
    );
    assert!(
        rendered.contains("10.0.9.7"),
        "corrected value missing:\n{rendered}"
    );
}

#[test]
fn ingest_is_idempotent() {
    let mut rt = seeded();
    let before = rt.graph.len();
    rt.ingest("The server ip is 10.0.4.12.", Some("t1"))
        .unwrap();
    assert_eq!(rt.graph.len(), before);
}

/// The separator scanner walks bytes but widened each byte to a char for the
/// word check; a lead byte like 0xE2 widens to 'â' (alphanumeric), so the
/// word scan ran into the middle of a multibyte character and sliced a
/// non-boundary. Ingesting any document with box-drawing or accented text
/// panicked. Regression test: unicode content must ingest cleanly.
#[test]
fn ingest_does_not_panic_on_multibyte_text() {
    let mut rt = peregrine_ctx::runtime::Dcr::new(800);
    rt.ingest(
        "┌──────────┐\n│ architecture diagram │\n└──────────┘\n\
         The server ip is 10.0.4.12. Ümläuts and café au lait.",
        Some("unicode"),
    )
    .expect("unicode ingest must not fail");
    let answer = rt.ask("what is the server ip?", None);
    assert!(answer.text.contains("10.0.4.12"), "{:?}", answer.text);
}

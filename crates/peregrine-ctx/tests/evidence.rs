//! The evidence hierarchy: origin, and the live states a node can report.
//!
//! The invariant under test is the one the whole classification exists for —
//! **the reasoner must never receive a derived hypothesis as an observed
//! fact**. Every assertion here is either "derived material is labelled" or
//! "labelled material reaches the window with its label intact".

use peregrine_ctx::ladder::Level;
use peregrine_ctx::nodes::{Kind, NewNode, Origin, Status};
use peregrine_ctx::runtime::Dcr;

fn runtime() -> Dcr {
    let mut rt = Dcr::new(800);
    rt.ingest(
        "The server ip is 10.0.9.7 and the port is 8080.\n\n\
         The retry budget is 5 attempts per minute.",
        Some("t1"),
    )
    .expect("ingest");
    rt
}

/// Register a derivation and run it over a real ingested fact.
///
/// The dependency is not decoration: an ungrounded calculation is refused at
/// `upsert`, which is the provenance rule doing its job.
fn compute_doubled(rt: &mut Dcr) -> peregrine_ctx::nodes::NodeIdx {
    rt.execution
        .register("double", |inputs: &Vec<(String, f64)>| {
            inputs.first().map(|(_, v)| v * 2.0).unwrap_or_default()
        });
    let source = *rt
        .graph
        .by_key("retry.budget", true)
        .last()
        .expect("the retry budget was extracted");
    let deps = vec![rt.graph.node(source).id.clone()];
    rt.compute(
        "double",
        vec![("n".to_string(), 5.0)],
        deps,
        Some("retry.budget.doubled"),
    )
    .expect("compute")
}

#[test]
fn origins_round_trip_through_their_names() {
    for origin in Origin::ALL {
        assert_eq!(Origin::parse(origin.as_str()), Some(origin));
    }
    assert_eq!(Origin::parse("guessed"), None);
    assert_eq!(Origin::default(), Origin::Observed);
}

#[test]
fn only_read_or_computed_material_counts_as_grounds() {
    assert!(Origin::Observed.is_grounds());
    assert!(Origin::ExternallySourced.is_grounds());
    assert!(Origin::Computed.is_grounds());
    // A conclusion is not its own evidence.
    assert!(!Origin::Inferred.is_grounds());
    assert!(!Origin::Hypothetical.is_grounds());
}

#[test]
fn ingested_material_is_observed() {
    let rt = runtime();
    assert!(rt.graph.len() > 0);
    for node in rt.graph.nodes() {
        assert_eq!(
            node.origin,
            Origin::Observed,
            "node {} came straight out of ingest",
            node.id
        );
    }
}

#[test]
fn a_computed_value_is_labelled_computed() {
    let mut rt = runtime();
    let idx = compute_doubled(&mut rt);

    let node = rt.graph.node(idx);
    assert_eq!(node.kind, Kind::Calculation);
    assert_eq!(node.origin, Origin::Computed);
    // …and it says so where the model can see it.
    let rendered = rt.ladder.render(&rt.raw, node, Level::L2);
    assert!(
        rendered.contains("origin=computed"),
        "computed value must carry its origin: {rendered}"
    );
}

#[test]
fn a_hypothesis_never_renders_like_an_observation() {
    let rt = runtime();
    let observed = NewNode::new(Kind::Claim, "server.ip = 10.0.9.7")
        .spans(["s_1".to_string()])
        .key("server.ip")
        .build();
    let guessed = NewNode::new(Kind::Claim, "server.ip = 10.0.9.7")
        .spans(["s_1".to_string()])
        .key("server.ip")
        .origin(Origin::Hypothetical)
        .build();

    let a = rt.ladder.render(&rt.raw, &observed, Level::L2);
    let b = rt.ladder.render(&rt.raw, &guessed, Level::L2);

    // Same kind, same value, same spans — and they must not read alike.
    assert!(!a.contains("origin="), "the ordinary case stays uncluttered: {a}");
    assert!(b.contains("origin=hypothetical"), "{b}");
    assert_ne!(a, b);
}

#[test]
fn origin_survives_a_save_and_load() {
    let dir = std::env::temp_dir().join(format!("dcr-origin-{}", std::process::id()));
    let path = dir.join("store.json");

    let mut rt = runtime();
    let idx = compute_doubled(&mut rt);
    let id = rt.graph.node(idx).id.clone();
    rt.save(&path).expect("save");

    let restored = Dcr::load(&path, 800).expect("load");
    let node = restored.graph.get(&id).expect("node survived");
    assert_eq!(node.origin, Origin::Computed);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_open_question_starts_unresolved_and_stays_live() {
    let node = NewNode::new(Kind::OpenQuestion, "why did the retries stop?")
        .spans(["s_1".to_string()])
        .build();
    assert_eq!(node.status, Status::Unresolved);
    assert!(node.status.is_live());

    // Everything else still starts fresh.
    let claim = NewNode::new(Kind::Claim, "x").spans(["s_1".to_string()]).build();
    assert_eq!(claim.status, Status::Fresh);
}

#[test]
fn live_states_are_the_ones_that_can_be_planned() {
    assert!(Status::Fresh.is_live());
    assert!(Status::Contradicted.is_live());
    assert!(Status::Unresolved.is_live());
    // Stale needs re-grounding first; superseded is history.
    assert!(!Status::Stale.is_live());
    assert!(!Status::Superseded.is_live());
}

#[test]
fn statuses_round_trip_through_their_names() {
    for status in [
        Status::Fresh,
        Status::Stale,
        Status::Superseded,
        Status::Contradicted,
        Status::Unresolved,
    ] {
        assert_eq!(Status::parse(status.as_str()), Some(status));
    }
}

/// A contradicted node is live, so it reaches the window — and when it does it
/// must arrive carrying the warning, not looking settled.
#[test]
fn a_contradicted_node_reaches_the_window_with_its_marker() {
    let mut rt = Dcr::new(800);
    rt.ingest("The server ip is 10.0.4.12.", Some("t1")).expect("ingest");
    rt.ingest("The server ip is 10.0.9.7.", Some("t2")).expect("ingest");

    let contested: Vec<&peregrine_ctx::nodes::Node> = rt
        .graph
        .nodes()
        .iter()
        .filter(|n| !n.meta.contradicts.is_empty() && n.status.is_live())
        .collect();

    for node in &contested {
        let rendered = rt.ladder.render(&rt.raw, node, Level::L2);
        assert!(
            rendered.contains("CONTRADICTS="),
            "a live contested node must carry its warning: {rendered}"
        );
    }
}

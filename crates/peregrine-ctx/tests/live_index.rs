//! `live_by_key` must stay in lockstep with `by_key(key, true)`: the same
//! elements in the same order, on the ingest path AND after a disk round-trip.
//! A desync would silently drop a live node from the planner's view — a wrong
//! answer that looks exactly like a correct one, which is the failure this
//! system exists to make visible. So the invariant is checked here, and it is
//! made to fire (see the deliberate-break note in the fix's verification).

use peregrine_ctx::nodes::Status;
use peregrine_ctx::runtime::Dcr;

/// For every indexed key, the maintained live index must equal what
/// `by_key(key, true)` computes from scratch — positionally, not as a set — and
/// no superseded node may leak into the live set.
fn assert_live_index_consistent(rt: &Dcr) {
    for key in rt.graph.keys() {
        assert_eq!(
            rt.graph.live_by_key_sorted(key),
            rt.graph.by_key(key, true),
            "live_by_key diverged from by_key(key, true) for key {key:?}"
        );
        for &idx in rt.graph.live_by_key(key) {
            assert_ne!(
                rt.graph.node(idx).status,
                Status::Superseded,
                "a superseded node leaked into live_by_key[{key:?}]"
            );
        }
    }
}

#[test]
fn live_index_survives_supersession_and_a_round_trip() {
    let mut rt = Dcr::new(600);
    // A fact, an unrelated fact (so more than one bucket exists), then a
    // correction that supersedes the first (old -> Superseded, new -> Fresh).
    rt.ingest("The server ip is 10.0.4.12.", Some("t1")).unwrap();
    rt.ingest("The firewall rule is 37.", Some("t2")).unwrap();
    let r = rt
        .ingest("Correction: the server ip is 10.0.9.7.", Some("t3"))
        .unwrap();
    assert_eq!(r.contradictions.len(), 1, "the correction must supersede");

    // In memory: server.ip has one live node (10.0.9.7) and two in history.
    assert_eq!(rt.graph.by_key("server.ip", true).len(), 1);
    assert_eq!(rt.graph.by_key("server.ip", false).len(), 2);
    assert_eq!(rt.graph.live_by_key("server.ip").len(), 1);
    assert_live_index_consistent(&rt);

    // After a JSON round-trip, `insert_restored` rebuilds the index from disk;
    // it must land in the same state, not merely be internally plausible. This
    // is the path fresh-run benchmarks never exercise.
    let dir = std::env::temp_dir().join(format!("dcr-live-index-{}", std::process::id()));
    let path = dir.join("store.json");
    rt.save(&path).unwrap();
    let restored = Dcr::load(&path, 600).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(restored.graph.by_key("server.ip", true).len(), 1);
    assert_eq!(restored.graph.by_key("server.ip", false).len(), 2);
    assert_eq!(restored.graph.live_by_key("server.ip").len(), 1);
    assert_live_index_consistent(&restored);
}

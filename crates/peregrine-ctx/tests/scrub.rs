//! Bit-rot: detection, repair from a verified replica, and the refusal to
//! repair from an unverified one.

use std::path::PathBuf;

use peregrine_ctx::context_store::{ContextStore, ObjectKind, ObjectRecord};
use peregrine_ctx::json::Json;
use peregrine_ctx::scrub::{AuditEvent, ScrubOptions, scrub};
use peregrine_ctx::trust::InsecureDevSigner;

fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let base = std::env::temp_dir().join(format!("dcr-scrub-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    (base.join("primary"), base.join("replica"))
}

fn span(id: &str, text: &str) -> ObjectRecord {
    ObjectRecord::new(
        ObjectKind::Span,
        id,
        Json::obj(vec![("text", Json::str(text))]),
    )
}

/// A store with `n` objects, optionally mirrored to a replica directory.
fn build(primary: &PathBuf, replica: Option<&PathBuf>, n: usize) -> (ContextStore, Vec<String>) {
    let mut store = ContextStore::create(primary, "agent_7").expect("create");
    if let Some(replica) = replica {
        store.add_replica(replica);
    }
    let ids = (0..n)
        .map(|i| {
            store
                .put(span(&format!("s_{i}"), &format!("fact number {i}")))
                .expect("put")
        })
        .collect();
    store.commit(1, None).expect("commit");
    (store, ids)
}

/// Flip a bit in the object's stored bytes — a disk error, not an edit.
fn rot(store: &ContextStore, id: &str) {
    let path = store.object_path(id);
    let mut bytes = std::fs::read(&path).expect("read");
    let position = bytes.len() / 2;
    bytes[position] ^= 0x01;
    std::fs::write(&path, &bytes).expect("write");
}

#[test]
fn a_clean_store_scrubs_clean() {
    let (primary, _) = scratch("clean");
    let (mut store, _) = build(&primary, None, 6);

    let report = scrub(&mut store, &ScrubOptions::default(), None).expect("scrub");
    assert_eq!(report.checked, 6);
    assert_eq!(report.healthy, 6);
    assert!(report.corrupt.is_empty());
    assert!(report.clean());

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn corruption_is_detected_without_repairing_anything() {
    let (primary, _) = scratch("detect");
    let (mut store, ids) = build(&primary, None, 5);
    rot(&store, &ids[2]);

    // The default options are read-only: detect, report, change nothing.
    let report = scrub(&mut store, &ScrubOptions::default(), None).expect("scrub");
    assert_eq!(report.corrupt, vec![ids[2].clone()]);
    assert_eq!(report.healthy, 4);
    assert!(report.repaired.is_empty());
    assert!(report.quarantined.is_empty());
    assert_eq!(report.replicas_configured, 0);
    assert!(store.object_path(&ids[2]).exists(), "nothing was moved");

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn a_verified_replica_repairs_the_primary() {
    let (primary, replica) = scratch("repair");
    let (mut store, ids) = build(&primary, Some(&replica), 5);
    let damaged = ids[3].clone();
    let original = std::fs::read(store.object_path(&damaged)).expect("read");
    rot(&store, &damaged);

    let report = scrub(&mut store, &ScrubOptions::repairing(2), None).expect("scrub");
    assert_eq!(report.corrupt, vec![damaged.clone()]);
    assert_eq!(report.repaired, vec![damaged.clone()]);
    assert!(report.quarantined.is_empty());
    assert!(report.clean());

    // The bytes are back, and the object reads through the normal path again.
    assert_eq!(std::fs::read(store.object_path(&damaged)).expect("read"), original);
    assert!(store.get(&damaged).is_ok());
    assert!(store.verify(None).ok());
    // A pure repair changes no object, so no new generation was needed.
    assert_eq!(report.sealed_generation, None);
    assert_eq!(store.generation(), 1);

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn an_unverified_replica_is_never_used_to_repair() {
    let (primary, replica) = scratch("bad-replica");
    let (mut store, ids) = build(&primary, Some(&replica), 4);
    let damaged = ids[1].clone();

    // Corrupt the primary *and* the replica. The replica is the shape of a
    // trap: it exists, it is readable, and it is wrong.
    rot(&store, &damaged);
    let replica_path = replica
        .join("objects")
        .join(&damaged[..2])
        .join(&damaged[2..]);
    std::fs::write(&replica_path, b"{\"kind\":\"span\",\"forged\":true}").expect("write");

    let report = scrub(&mut store, &ScrubOptions::repairing(2), None).expect("scrub");
    assert!(report.repaired.is_empty(), "a bad replica must not repair");
    assert_eq!(report.unrepairable, vec![damaged.clone()]);
    assert_eq!(report.quarantined, vec![damaged.clone()]);
    assert!(
        report.events.iter().any(|e| matches!(
            e,
            AuditEvent::ReplicaRejected { object, .. } if object == &damaged
        )),
        "the rejected replica should be in the audit trail"
    );

    // The forged bytes never entered the store.
    assert!(!store.object_path(&damaged).exists());
    assert!(store.get(&damaged).is_err());

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn an_unrepairable_object_is_quarantined_and_the_loss_is_recorded() {
    let (primary, _) = scratch("loss");
    let (mut store, ids) = build(&primary, None, 4);
    let lost = ids[0].clone();
    rot(&store, &lost);

    let signer = InsecureDevSigner::new("secret");
    let report = scrub(&mut store, &ScrubOptions::repairing(2), Some(&signer)).expect("scrub");
    assert_eq!(report.quarantined, vec![lost.clone()]);
    assert_eq!(report.sealed_generation, Some(2));

    // Sealing is what keeps the store honest: the committed root now describes
    // the three objects that actually survive, and it verifies again.
    assert_eq!(store.len(), 3);
    let report = store.verify(Some(&signer));
    assert!(report.root_matches, "the sealed root must describe the store");
    assert!(report.ok());

    // …and the loss is not silent: the quarantine record says what went.
    let quarantined = store.quarantined();
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].0, lost);

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn a_missing_object_is_treated_as_corruption() {
    let (primary, replica) = scratch("missing");
    let (mut store, ids) = build(&primary, Some(&replica), 4);
    let gone = ids[2].clone();
    std::fs::remove_file(store.object_path(&gone)).expect("remove");

    let report = scrub(&mut store, &ScrubOptions::repairing(2), None).expect("scrub");
    assert_eq!(report.corrupt, vec![gone.clone()]);
    assert_eq!(report.repaired, vec![gone.clone()]);
    assert!(store.get(&gone).is_ok());

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

#[test]
fn repair_refuses_bytes_that_do_not_match_the_address() {
    let (primary, _) = scratch("restore-guard");
    let (mut store, ids) = build(&primary, None, 3);

    // The last line of defence: even a direct restore call cannot install
    // content that does not hash to the address it is being written to.
    assert!(store.restore_object(&ids[0], b"not the original bytes").is_err());
    assert!(store.get(&ids[0]).is_ok(), "the original must be untouched");

    let _ = std::fs::remove_dir_all(primary.parent().expect("parent"));
}

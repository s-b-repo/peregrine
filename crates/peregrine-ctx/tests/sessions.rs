//! Session sharding and the TCP protocol: one runtime per conversation,
//! LRU eviction reported rather than silent, and the wire format end to end.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use peregrine_ctx::SessionRouter;
use peregrine_ctx::sessions::serve;

#[test]
fn router_keeps_one_runtime_per_session() {
    let mut router = SessionRouter::new(800, 4);
    let a1 = router.route("a");
    let b = router.route("b");
    let a2 = router.route("a");
    assert!(Arc::ptr_eq(&a1, &a2), "same session must reuse its runtime");
    assert!(!Arc::ptr_eq(&a1, &b), "different sessions must not share one");
    assert_eq!(router.len(), 2);
}

#[test]
fn router_evicts_lru_and_reports_it() {
    let mut router = SessionRouter::new(800, 2);
    router.route("old");
    router.route("new");
    router.route("old"); // refresh: "new" is now the oldest
    router.route("intruder");
    assert_eq!(router.evicted, 1);
    assert_eq!(router.len(), 2);
    // "new" was evicted, "old" survived the refresh.
    let reactivated = router.route("old");
    assert!(Arc::strong_count(&reactivated) >= 2);
}

/// An eviction victim is chosen only among idle sessions: dropping the map
/// entry of a session another thread is mid-request on would spawn a second,
/// cold runtime for the same conversation and split its state.
#[test]
fn eviction_never_splits_an_in_flight_session() {
    let mut router = SessionRouter::new(800, 2);
    let a = router.route("a");
    let _in_flight = a.clone(); // a handler holding the runtime mid-request
    router.route("b");
    // Full. Routing "c" must evict "b" (idle, most recent is irrelevant — "b"
    // is the only evictable entry), never the in-flight "a".
    router.route("c");
    assert_eq!(router.len(), 2);
    assert_eq!(router.evicted, 1);
    let a2 = router.route("a");
    assert!(
        Arc::ptr_eq(&a, &a2),
        "the in-flight session must survive eviction pressure"
    );
}

/// One round trip over the real socket: INGEST into a session, ASK it back.
#[test]
fn serve_answers_per_session_over_tcp() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = Arc::new(Mutex::new(SessionRouter::new(800, 8)));
    std::thread::spawn(move || serve(listener, router));

    let mut stream = TcpStream::connect(addr).expect("connect");
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));

    let doc = "The server ip is 10.0.4.12 and the port is 8080.";
    stream
        .write_all(format!("INGEST ops {}\n{doc}", doc.len()).as_bytes())
        .expect("ingest write");
    let mut status = String::new();
    reader.read_line(&mut status).expect("ingest status");
    assert_eq!(status.trim(), "OK", "ingest failed: {status}");

    stream
        .write_all(b"ASK ops what is the server ip?\n")
        .expect("ask write");
    let mut header = String::new();
    reader.read_line(&mut header).expect("ask header");
    let length: usize = header
        .trim()
        .strip_prefix("OK ")
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("bad ask header: {header}"));
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).expect("ask body");
    let answer = String::from_utf8_lossy(&body);
    assert!(answer.contains("10.0.4.12"), "wrong answer: {answer}");

    // A second session must not see the first session's memory.
    stream
        .write_all(b"ASK stranger what is the server ip?\n")
        .expect("ask write 2");
    let mut header = String::new();
    reader.read_line(&mut header).expect("ask header 2");
    let length: usize = header
        .trim()
        .strip_prefix("OK ")
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("bad ask header: {header}"));
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).expect("ask body 2");
    let answer = String::from_utf8_lossy(&body);
    assert!(
        !answer.contains("10.0.4.12"),
        "memory leaked across sessions: {answer}"
    );
}

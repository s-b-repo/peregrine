//! Session sharding — concurrency across conversations, not within one.
//!
//! `Dcr` is a single-writer facade: one conversation, one runtime, no interior
//! locking. Conversations are naturally independent, so the honest concurrency
//! model is to shard by session and lock at the shard boundary, which gives
//! continuous-batching-style throughput with zero changes to the runtime's
//! internals. `serve` puts that behind a dependency-free TCP protocol so N
//! clients can share one process.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::runtime::Dcr;

/// One conversation = one runtime. Routing is by session id; eviction is LRU.
///
/// Eviction drops the runtime's whole state — session memory lives only here,
/// so `capacity` is a memory bound, and an evicted conversation restarts cold.
/// That is reported (`evicted`) rather than hidden, because a silently
/// forgotten conversation is the same failure as a stale fact. One exception
/// to the bound: a session mid-request is never evicted (that would split a
/// live conversation across two runtimes), so capacity is exceeded while
/// every slot is in flight.
pub struct SessionRouter {
    budget: usize,
    capacity: usize,
    sessions: HashMap<String, Arc<Mutex<Dcr>>>,
    /// Oldest first, refreshed on touch.
    lru: VecDeque<String>,
    /// Sessions dropped to make room. A rising counter means capacity is too
    /// small for the working set, not that the router is misbehaving.
    pub evicted: u64,
}

impl SessionRouter {
    pub fn new(budget: usize, capacity: usize) -> Self {
        Self {
            budget,
            capacity: capacity.max(1),
            sessions: HashMap::new(),
            lru: VecDeque::new(),
            evicted: 0,
        }
    }

    pub fn route(&mut self, session: &str) -> Arc<Mutex<Dcr>> {
        if !self.sessions.contains_key(session) {
            if self.sessions.len() >= self.capacity {
                // Evict the least-recently-used session that is not in flight.
                // An entry another thread is mid-request on (strong_count > 1)
                // must not be dropped from the map: the next request for that
                // id would spawn a second, cold runtime and split the
                // conversation's state across two runtimes.
                let victim = self
                    .lru
                    .iter()
                    .position(|id| {
                        id != session
                            && self
                                .sessions
                                .get(id)
                                .is_some_and(|d| Arc::strong_count(d) == 1)
                    })
                    .map(|pos| self.lru.remove(pos).unwrap());
                if let Some(oldest) = victim {
                    self.sessions.remove(&oldest);
                    self.evicted += 1;
                }
                // If every session is in flight, capacity is exceeded rather
                // than splitting a live conversation — a soft, reported-only
                // bound beats corrupting state.
            }
            self.sessions.insert(
                session.to_string(),
                Arc::new(Mutex::new(Dcr::new(self.budget))),
            );
        }
        self.lru.retain(|id| id != session);
        self.lru.push_back(session.to_string());
        self.sessions[session].clone()
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

/// A poisoned lock is a panicked turn, not corrupt state — the runtime is
/// single-writer, so the last completed write is intact. Recover rather than
/// kill the connection.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}

/// Serve sessions over TCP until the listener errors. One thread per
/// connection; connections are cheap because the runtimes do the work.
///
/// Protocol, line-based commands with length-prefixed bodies where needed:
///
/// ```text
/// C: INGEST <session> <byte-len>\n<body>
/// S: OK\n
/// C: ASK <session> <query...>\n
/// S: OK <byte-len>\n<answer>
/// S: ERR <message>\n
/// ```
pub fn serve(listener: TcpListener, router: Arc<Mutex<SessionRouter>>) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let router = Arc::clone(&router);
                std::thread::spawn(move || handle(stream, router));
            }
            Err(e) => eprintln!("accept failed: {e}"),
        }
    }
}

/// Largest single INGEST body, in bytes. The body is allocated up front from
/// the client-supplied length, so an uncapped length is a one-line memory
/// exhaustion against the server.
const MAX_INGEST_BODY: usize = 16 * 1024 * 1024;

/// Idle-connection deadline. Threads are spawned per connection and never
/// reaped while a client holds one open silently, so a connection that stops
/// talking must eventually cost nothing.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

fn respond(stream: &mut TcpStream, text: &str) {
    let _ = stream.write_all(text.as_bytes());
}

fn handle(mut stream: TcpStream, router: Arc<Mutex<SessionRouter>>) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    if read_half.set_read_timeout(Some(READ_TIMEOUT)).is_err() {
        return;
    }
    let mut reader = BufReader::new(read_half);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock
                    && e.kind() != std::io::ErrorKind::TimedOut
                {
                    eprintln!("session read failed: {e}");
                }
                return;
            }
            Ok(_) => {}
        }
        let mut parts = line.trim_end_matches(['\r', '\n']).splitn(3, ' ');
        match (parts.next(), parts.next(), parts.next()) {
            (Some("ASK"), Some(session), Some(query)) if !session.is_empty() => {
                let dcr = lock(&router).route(session);
                let answer = lock(&dcr).ask(query, None);
                respond(&mut stream, &format!("OK {}\n{}", answer.text.len(), answer.text));
            }
            (Some("INGEST"), Some(session), Some(rest)) if !session.is_empty() => {
                let length = rest.trim().parse::<usize>().ok();
                match length {
                    Some(length) if length > MAX_INGEST_BODY => {
                        respond(&mut stream, "ERR ingest body exceeds 16 MiB cap\n");
                    }
                    Some(length) => {
                        let mut body = vec![0u8; length];
                        if reader.read_exact(&mut body).is_err() {
                            respond(&mut stream, "ERR truncated ingest body\n");
                            return;
                        }
                        let text = String::from_utf8_lossy(&body).to_string();
                        let dcr = lock(&router).route(session);
                        match lock(&dcr).ingest(&text, None) {
                            Ok(_) => respond(&mut stream, "OK\n"),
                            Err(e) => respond(&mut stream, &format!("ERR {e}\n")),
                        }
                    }
                    None => respond(&mut stream, "ERR usage: INGEST <session> <byte-len>\n"),
                }
            }
            _ => respond(
                &mut stream,
                "ERR usage: ASK <session> <query> | INGEST <session> <byte-len>\n",
            ),
        }
    }
}

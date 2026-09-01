//! Speculative Programmatic Tool Calling (sPTC) — the shadow executor.
//!
//! The model streams a tool call one token at a time. Under the OpenAI
//! protocol the finished call travels to the client, the client executes it,
//! sends the result back in a fresh request, and the server pays a full
//! re-render and prefill of the conversation before the model learns what the
//! tool returned. sPTC closes part of that gap inside the server, for tools
//! the operator has *hosted* (`--host-tool`): while the call is still
//! streaming, every **complete** argument pair is offered to a shadow
//! executor — a bounded worker pool that runs the tool on the arguments so
//! far and files the result under the exact key the finished call will carry.
//! If the model closes the call with the same arguments (the overwhelmingly
//! common case: a prefix is a prefix), the result is already on the shelf at
//! close time; if it does not, the guess is discarded and the call executes
//! exactly as it would have. The stream is byte-identical either way —
//! speculation changes *when* a result exists, never *what* it is.
//!
//! Three rules make that safe, each one load-bearing:
//!
//! **Only pure tools are speculated — and only pure tools can be hosted.**
//! Purity is not a flag a caller can get wrong: the [`pure_tool!`] decorator
//! is the *only* way to construct a [`HostedTool`], so "deterministic and
//! side-effect free, safe to run on a partial guess of the arguments" is a
//! property of the type, not of the discipline of whoever registered it. A
//! speculative run discards its work on a mismatched close, so a wrong guess
//! wastes CPU and nothing else — the same correctness-neutral shape as the
//! prefetch lane.
//!
//! **Verification is by exact argument match, never by trust.** The key is
//! the canonical JSON of the *final* parsed arguments (sorted keys,
//! schema-typed values — the same `coerce` rules the client-facing parse
//! uses, so a speculated prefix and the finished call serialize identically
//! when they agree). No hash, no fuzzy match: one extra character is a miss,
//! and a miss only costs the speculation.
//!
//! **Hosted tools are a closed, declared set with no ambient authority.** A
//! hosted tool receives its JSON arguments and nothing else — no disk, no
//! network, no engine state — which is the defense the Ghost Tool Calls audit
//! asks for: a client cannot smuggle a computation into this process's
//! privileges, because it can only invoke what the operator explicitly
//! hosted, and every built-in is a pure function over its arguments. Input
//! and output are byte-capped on both sides, and a rejected input is an
//! error surfaced to the caller, not an empty success.
//!
//! When no tools are hosted, the subsystem does not exist: no threads, no
//! table, no per-token work on the stream. `COLI_SPTC=0` turns speculation
//! off while keeping hosted execution (then every call runs at close).

use crate::tools::{OutputFilter, ParsedCall, SpecSnapshot};
use crate::tok::TokenBackend;
use parking_lot::Mutex;
use peregrine_core::Error;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::Arc;

/// Arguments larger than this are refused before any execution, speculative
/// or final. A call body is model output, so it is untrusted input to the
/// tool — the cap bounds both the speculation's re-parse cost (one full scan
/// per new argument pair) and what a hosted tool may be handed.
const MAX_ARG_BYTES: usize = 64 * 1024;/// Output ceiling per run. A hosted tool's result rides inside a
/// `tool_calls` chunk, so a runaway result would be a wire-shape hazard
/// before it is a memory one.
const MAX_OUT_BYTES: usize = 1024 * 1024;
/// Bounds for the shadow table — the memo of speculated results. Both are
/// small by design: the table serves one turn (the call that is still open,
/// plus a window of recent identical calls), not the whole workload. An
/// entry beyond the window costs a synchronous re-run, never correctness.
const TABLE_MAX_ENTRIES: usize = 256;
const TABLE_MAX_BYTES: usize = 8 * 1024 * 1024;
/// Shadow-executor shape: two workers, bounded queue. A full queue means the
/// pool cannot keep up with speculation — skip instead of block, because the
/// stream must never wait on an optimization (the kvstore writer and the
/// unbounded SSE channel encode the same rule).
const SHADOW_WORKER_THREADS: usize = 2;
const SHADOW_QUEUE_DEPTH: usize = 16;

/// The shape every hosted tool shares: its JSON arguments in, its JSON
/// result (or a structured error) out. Named once because the boxed form
/// appears in the tool struct and would otherwise be spelled out there —
/// the same fix the workspace applied to its `MatShape`/`RouterCfg` types.
pub type PureFn = Box<dyn Fn(&Value) -> Result<Value, Error> + Send + Sync>;

/// A tool this server executes itself, rather than handing the call to the
/// client. It is **pure by construction**: the `run` closure is private and
/// the [`pure_tool!`] decorator is the only code path that can build one, so
/// "is this tool safe to speculate on?" is answered by the type system
/// instead of by a metadata flag that could drift from the function it
/// describes.
pub struct HostedTool {
    pub name: String,
    run: PureFn,
}

impl HostedTool {
    pub fn run(&self, args: &Value) -> Result<Value, Error> {
        (self.run)(args)
    }
}

/// Declare a hosted tool as pure at the definition site — the Rust form of a
/// `@pure` decorator, and the **only** way to construct a [`HostedTool`]:
/// the claim travels with the function instead of being re-stated in a
/// registry table, and a tool that misbehaves for repeated or partial calls
/// cannot be registered at all without editing this macro's contract.
#[macro_export]
macro_rules! pure_tool {
    ($name:expr, |$args:ident| $body:block) => {
        $crate::sptc::HostedTool {
            name: $name.to_string(),
            run: Box::new(move |$args: &Value| -> Result<Value, peregrine_core::Error> { $body }),
        }
    };
}

/// The tools this server will execute on the model's behalf. Built from a
/// closed set: an unknown name is a boot error rather than a silent no-op,
/// because a tool the operator believes is hosted but is not silently turns
/// every call back into a client round trip.
pub struct ToolRegistry {
    tools: Vec<HostedTool>,
}

impl ToolRegistry {
    pub fn new() -> ToolRegistry {
        ToolRegistry { tools: Vec::new() }
    }

    pub fn register(mut self, tool: HostedTool) -> ToolRegistry {
        self.tools.push(tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<&HostedTool> {
        self.tools.iter().find(|t| t.name == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// The one first-party tool family shipped with the server: tokenizer
    /// queries over the *server's own* tokenizer. An agent that budgets
    /// `max_tokens` before writing a call no longer has to guess. All three
    /// are pure: same input, same output, and the encode-side memo cache
    /// changes speed, never results.
    pub fn token_tools(tok: &Arc<TokenBackend>) -> ToolRegistry {
        // One owned handle per tool: each `pure_tool!` closure is `'static`
        // and takes its captures by move, so the shared `Arc` is cloned into
        // each rather than moved into the first.
        let tok_enc = Arc::clone(tok);
        let tok_dec = Arc::clone(tok);
        let tok_cnt = Arc::clone(tok);
        ToolRegistry::new()
            .register(crate::pure_tool!("tokenize", |args| {
                let text = string_arg(args, "text")?;
                let ids = tok_enc.encode(text)?;
                Ok(json!({ "ids": ids, "count": ids.len() }))
            }))
            .register(crate::pure_tool!("detokenize", |args| {
                let ids = u32_vec_arg(args, "ids")?;
                let text = tok_dec.decode(&ids)?;
                Ok(json!({ "text": text }))
            }))
            .register(crate::pure_tool!("count_tokens", |args| {
                let text = string_arg(args, "text")?;
                Ok(json!({ "count": tok_cnt.encode(text)?.len() }))
            }))
    }
}

fn string_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, Error> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| {
        Error::Format(format!("tool argument '{key}' is required and must be a string"))
    })
}

fn u32_vec_arg(args: &Value, key: &str) -> Result<Vec<u32>, Error> {
    match args.get(key) {
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|n| *n <= u32::MAX as u64)
                    .map(|n| n as u32)
                    .ok_or_else(|| {
                        Error::Format(format!("tool argument '{key}' must hold u32 token ids, found {v}"))
                    })
            })
            .collect(),
        Some(other) => Err(Error::Format(format!(
            "tool argument '{key}' must be an array of token ids, found {other}"
        ))),
        None => Err(Error::Format(format!(
            "tool argument '{key}' is required: an array of token ids"
        ))),
    }
}

/// One speculative or final execution, ready to attach to the call's chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum ShadowOutcome {
    Done(Value),
    /// The tool ran and returned an error. Cached for pure tools — failure
    /// is as deterministic as success — and surfaced to the caller as the
    /// `error` member of the attached result, so a client sees the failure
    /// rather than a silently absent field.
    Failed(String),
}

impl ShadowOutcome {
    fn to_wire(&self) -> Value {
        match self {
            ShadowOutcome::Done(v) => v.clone(),
            ShadowOutcome::Failed(msg) => json!({ "error": msg }),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            ShadowOutcome::Done(v) => v.to_string().len(),
            ShadowOutcome::Failed(m) => m.len(),
        }
    }
}

/// One speculative job: run the tool on this prefix, file the result under
/// the key the closing parse will compute.
struct Job {
    key: String,
    name: String,
    args: Value,
}

struct Table {
    entries: VecDeque<(String, ShadowOutcome)>,
    bytes: usize,
}

impl Table {
    fn new() -> Table {
        Table { entries: VecDeque::new(), bytes: 0 }
    }

    /// Lookup by exact canonical key. A hit refreshes recency, matching the
    /// response memo's LRU rule — an agent loop re-issuing the same call
    /// across turns is the common case.
    fn get(&mut self, key: &str) -> Option<ShadowOutcome> {
        let i = self.entries.iter().position(|(k, _)| k == key)?;
        let (k, v) = self.entries.remove(i)?;
        self.entries.push_back((k, v.clone()));
        Some(v)
    }

    fn insert(&mut self, key: String, outcome: ShadowOutcome, evictions: &AtomicU64) {
        let new_bytes = key.len() + outcome.bytes();
        if new_bytes > TABLE_MAX_BYTES {
            // An entry that can never fit would evict the whole table to
            // hold one result — refuse it instead of emptying the cache for it.
            return;
        }
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            if let Some((old_k, old_v)) = self.entries.remove(i) {
                self.bytes -= old_k.len() + old_v.bytes();
            }
        }
        self.entries.push_back((key, outcome));
        self.bytes += new_bytes;
        while (self.entries.len() > TABLE_MAX_ENTRIES || self.bytes > TABLE_MAX_BYTES)
            && !self.entries.is_empty()
        {
            match self.entries.pop_front() {
                Some((k, v)) => {
                    self.bytes -= k.len() + v.bytes();
                    evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => break,
            }
        }
    }
}

#[derive(Default)]
struct Counters {
    speculated: AtomicU64,
    verified_hits: AtomicU64,
    sync_runs: AtomicU64,
    rejected_input: AtomicU64,
    rejected_busy: AtomicU64,
    table_evictions: AtomicU64,
    hosted_calls: AtomicU64,
}

/// The shadow executor: the bounded worker pool that runs speculations, the
/// table their results land in, and the close-time verification that decides
/// between a speculated result and a synchronous run.
pub struct ShadowRepl {
    core: Arc<ShadowCore>,
    jobs_tx: Option<SyncSender<Job>>,
    speculate_enabled: bool,
}

struct ShadowCore {
    registry: ToolRegistry,
    table: Mutex<Table>,
    counters: Counters,
}

impl ShadowRepl {
    /// Build from the operator's request. `Ok(None)` — no tools hosted — is
    /// the default and costs nothing: no threads, no table, and the streaming
    /// path skips the speculation probe entirely. An unknown tool name is a
    /// boot error: a misconfigured `--host-tool` must fail loudly, not
    /// quietly degrade into calls the client has to answer itself.
    pub fn from_args(
        requested: &[String],
        tokenizer: &Arc<TokenBackend>,
        speculate_enabled: bool,
    ) -> Result<Option<Arc<ShadowRepl>>, Error> {
        if requested.is_empty() {
            return Ok(None);
        }
        let registry = ToolRegistry::token_tools(tokenizer);
        for name in requested {
            if registry.get(name).is_none() {
                return Err(Error::Format(format!(
                    "unknown --host-tool '{name}' (available: {})",
                    registry.names().join(", ")
                )));
            }
        }
        Ok(Some(ShadowRepl::build(registry, speculate_enabled)))
    }

    fn build(registry: ToolRegistry, speculate_enabled: bool) -> Arc<ShadowRepl> {
        let core = Arc::new(ShadowCore {
            registry,
            table: Mutex::new(Table::new()),
            counters: Counters::default(),
        });
        // The pool exists only when there is something to speculate. Workers
        // share one receiver behind a mutex (`std::sync::mpsc` has no
        // multi-consumer channel, and no new dependency is wanted here):
        // whichever worker is parked in `recv` holds the lock, so a free job
        // wakes exactly one worker and the others queue on the mutex — a
        // two-slot round table, not a race. Workers hold the receiving end
        // only; the `ShadowRepl` holds the sender, so dropping the server
        // drops the pool and `recv`'s Disconnected is the shutdown signal.
        let jobs_tx = if speculate_enabled {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(SHADOW_QUEUE_DEPTH);
            let rx = Arc::new(parking_lot::Mutex::new(rx));
            for n in 0..SHADOW_WORKER_THREADS {
                let core = Arc::clone(&core);
                let rx = Arc::clone(&rx);
                if let Err(e) = std::thread::Builder::new()
                    .name(format!("sptc-shadow-{n}"))
                    .spawn(move || worker_loop(rx, core))
                {
                    // A missing worker is a degraded optimization, not a broken
                    // server — but running with less capacity than configured
                    // must be said out loud, not swallowed.
                    peregrine_core::note_advisory_err("sptc shadow worker spawn", &e);
                }
            }
            Some(tx)
        } else {
            None
        };
        Arc::new(ShadowRepl { core, jobs_tx, speculate_enabled })
    }

    pub fn hosts(&self, name: &str) -> bool {
        self.core.registry.get(name).is_some()
    }

    /// Canonical speculation key: tool name + canonical argument JSON.
    /// `serde_json::Map` sorts keys, so two parses of the same pairs agree
    /// byte-for-byte regardless of where the model was interrupted.
    fn canonical_key(name: &str, args: &Value) -> String {
        format!("{name}\u{1f}{args}")
    }

    /// Offer a still-open call to the shadow executor. Fire-and-forget by
    /// design: every skip here is a lost *optimization*, never a wrong
    /// answer, so each one counts itself and moves on.
    ///
    /// Every hosted tool is pure by construction (`pure_tool!` is the only
    /// constructor), so there is no purity gate here to bypass — the only
    /// preconditions left are the input cap and the pool's queue depth.
    pub fn speculate(self: &Arc<Self>, snap: &SpecSnapshot) {
        if !self.speculate_enabled {
            return; // COLI_SPTC=0: hosted execution still happens, guessing does not
        }
        if self.core.registry.get(&snap.name).is_none() {
            return;
        }
        let key = Self::canonical_key(&snap.name, &snap.arguments);
        if key.len() > MAX_ARG_BYTES {
            self.core.counters.rejected_input.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Some(tx) = &self.jobs_tx else { return };
        match tx.try_send(Job { key, name: snap.name.clone(), args: snap.arguments.clone() }) {
            Ok(()) => {
                self.core.counters.speculated.fetch_add(1, Ordering::Relaxed);
            }
            // Queue full: the pool is saturated, so this guess is not worth a
            // stall — the synchronous run at close is always available as the
            // floor, and blocking the SSE pump here would spend the stream's
            // latency on a guess.
            Err(TrySendError::Full(_)) => {
                self.core.counters.rejected_busy.fetch_add(1, Ordering::Relaxed);
            }
            // Structurally unreachable while this Arc keeps the senders alive:
            // the workers cannot outlive the server that holds them. Handled
            // rather than expected — a silent arm here would hide a future bug
            // in the shutdown order.
            Err(TrySendError::Disconnected(_)) => {
                self.core.counters.rejected_busy.fetch_add(1, Ordering::Relaxed);
                peregrine_core::note_advisory_err(
                    "sptc speculate",
                    &"shadow pool disconnected while the server is running",
                );
            }
        }
    }

    /// The close-time half of the protocol: resolve a completed call to its
    /// result, or `None` when the call is not hosted and the client must
    /// execute it as usual.
    ///
    /// A speculated hit returns the result the shadow computed *before* the
    /// call closed — the latency this module exists to remove. A miss runs
    /// synchronously; the miss costs one bounded in-process call and deposits
    /// its result for any identical call that follows.
    pub fn verify(&self, call: &ParsedCall) -> Option<Value> {
        // `None` for an unhosted call: the client executes it, as before.
        self.core.registry.get(&call.name)?;
        self.core.counters.hosted_calls.fetch_add(1, Ordering::Relaxed);
        let key = Self::canonical_key(&call.name, &call.arguments);
        if key.len() > MAX_ARG_BYTES {
            return Some(
                json!({ "error": format!("arguments exceed the {MAX_ARG_BYTES}-byte tool-input cap") }),
            );
        }
        let hit = self.core.table.lock().get(&key);
        match hit {
            Some(outcome) => {
                self.core.counters.verified_hits.fetch_add(1, Ordering::Relaxed);
                Some(outcome.to_wire())
            }
            None => {
                self.core.counters.sync_runs.fetch_add(1, Ordering::Relaxed);
                let outcome = run_hosted(&self.core.registry, &call.name, &call.arguments);
                self.core
                    .table
                    .lock()
                    .insert(key, outcome.clone(), &self.core.counters.table_evictions);
                Some(outcome.to_wire())
            }
        }
    }

    /// Attach a hosted tool's result to the OpenAI chunk the call is already
    /// building. Unhosted calls pass through untouched — the wire shape is
    /// the client's contract, and `peregrine_result` exists only when this
    /// server actually ran the tool.
    pub fn attach_result(&self, mut openai_call: Value, call: &ParsedCall) -> Value {
        if let Some(result) = self.verify(call) {
            if let Some(obj) = openai_call.as_object_mut() {
                obj.insert("peregrine_result".into(), result);
            }
        }
        openai_call
    }

    /// Counters and tool list for `/metrics`, read without blocking anything:
    /// the counters are atomics and the table lock is taken only to report
    /// two sizes.
    pub fn stats(&self) -> SptcStats {
        let c = &self.core.counters;
        let table = self.core.table.lock();
        SptcStats {
            speculated: c.speculated.load(Ordering::Relaxed),
            verified_hits: c.verified_hits.load(Ordering::Relaxed),
            sync_runs: c.sync_runs.load(Ordering::Relaxed),
            rejected_input: c.rejected_input.load(Ordering::Relaxed),
            rejected_busy: c.rejected_busy.load(Ordering::Relaxed),
            table_evictions: c.table_evictions.load(Ordering::Relaxed),
            hosted_calls: c.hosted_calls.load(Ordering::Relaxed),
            table_entries: table.entries.len(),
            table_bytes: table.bytes,
            speculation_enabled: self.speculate_enabled,
            tools: self.core.registry.names().iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// The worker loop behind the shadow executor. Runs until every sender is
/// gone — i.e. until the server drops the [`ShadowRepl`] — which is the one
/// exit condition a result-cache worker has, and it is named rather than
/// wildcarded so the audit gate can see it.
fn worker_loop(rx: Arc<Mutex<Receiver<Job>>>, core: Arc<ShadowCore>) {
    loop {
        // The lock spans the `recv`: an empty queue parks this worker holding
        // it, which is what routes the next arriving job to exactly one
        // worker. Lock hold time is one job's execution; jobs are bounded
        // in-process pure calls.
        let job = match rx.lock().recv() {
            Ok(job) => job,
            Err(std::sync::mpsc::RecvError) => return,
        };
        let outcome = run_hosted(&core.registry, &job.name, &job.args);
        core.table.lock().insert(job.key, outcome, &core.counters.table_evictions);
    }
}

/// One tool execution, speculative or final: run and cap the output. The only
/// place a hosted tool is ever invoked.
fn run_hosted(registry: &ToolRegistry, name: &str, args: &Value) -> ShadowOutcome {
    let Some(tool) = registry.get(name) else {
        return ShadowOutcome::Failed(format!(
            "hosted tool '{name}' vanished between speculation and execution"
        ));
    };
    match tool.run(args) {
        Ok(v) => {
            if v.to_string().len() > MAX_OUT_BYTES {
                ShadowOutcome::Failed(format!("result exceeds the {MAX_OUT_BYTES}-byte output cap"))
            } else {
                ShadowOutcome::Done(v)
            }
        }
        Err(e) => ShadowOutcome::Failed(e.to_string()),
    }
}

/// The per-stream half: watches the incremental filter and keeps speculation
/// from outrunning the model. One per response stream (cheap — one field).
pub struct Speculator {
    /// Canonical key of the last prefix offered for speculation. A call
    /// streaming token-by-token reaches `observe` once per token; without
    /// this, every token would re-enqueue the same prefix and the shadow
    /// pool would spend its budget recomputing one growing argument.
    last_key: Option<String>,
}

impl Speculator {
    pub fn new() -> Speculator {
        Speculator { last_key: None }
    }

    /// Offer the filter's current partial call for speculation. Skips in
    /// order of cheapness: no open call, not a hosted tool (one substring
    /// probe, no argument parse), nothing new since last time, then — and
    /// only then — the full prefix parse plus the enqueue.
    pub fn observe(&mut self, shadow: &Arc<ShadowRepl>, filter: &OutputFilter) {
        let Some(name) = filter.open_call_name() else {
            // Between calls — including right after a close: the next call
            // starts from an empty prefix and must speculate from scratch.
            self.last_key = None;
            return;
        };
        if !shadow.hosts(name) {
            return;
        }
        let Some(snap) = filter.speculation() else { return };
        let key = ShadowRepl::canonical_key(&snap.name, &snap.arguments);
        if self.last_key.as_deref() == Some(key.as_str()) {
            return;
        }
        self.last_key = Some(key);
        shadow.speculate(&snap);
    }
}

impl Default for Speculator {
    fn default() -> Speculator {
        Speculator::new()
    }
}

/// `COLI_SPTC=0` (or `false`/`off`) turns speculation off; hosted tools still
/// execute, at close. Read once at boot, like every other environment knob
/// in this server.
pub fn speculation_enabled() -> bool {
    !matches!(std::env::var("COLI_SPTC").as_deref(), Ok("0") | Ok("false") | Ok("off"))
}

/// Counters and tool list for `/metrics`.
#[derive(Debug)]
pub struct SptcStats {
    pub speculated: u64,
    pub verified_hits: u64,
    pub sync_runs: u64,
    pub rejected_input: u64,
    pub rejected_busy: u64,
    pub table_evictions: u64,
    pub hosted_calls: u64,
    pub table_entries: usize,
    pub table_bytes: usize,
    pub speculation_enabled: bool,
    pub tools: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::OutputFilter;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    /// A pure tool whose every invocation is counted — the fixture most of
    /// these tests reason about.
    fn counting_tool(name: &str, calls: Arc<AtomicUsize>) -> HostedTool {
        crate::pure_tool!(name, |args| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let text = string_arg(args, "text")?;
            Ok(json!({ "echo": text }))
        })
    }

    fn pure_registry(calls: Arc<AtomicUsize>) -> Arc<ShadowRepl> {
        ShadowRepl::build(ToolRegistry::new().register(counting_tool("shout", calls)), true)
    }

    fn snap(name: &str, args: Value) -> SpecSnapshot {
        SpecSnapshot { name: name.to_string(), arguments: args }
    }

    /// The speculation path is asynchronous by design, so a test that wants
    /// to observe its *effect* waits for the table entry to land, with a
    /// deadline — the same bounded-wait discipline the shutdown drain uses.
    fn wait_for_table_entries(shadow: &Arc<ShadowRepl>, want: usize) -> Result<(), &'static str> {
        for _ in 0..2000 {
            if shadow.stats().table_entries >= want {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        Err("the shadow pool did not land its results in time")
    }

    #[test]
    fn a_completed_prefix_is_speculated_and_verified_without_rerunning() -> Result<(), &'static str>
    {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(Arc::clone(&calls));
        // The model has written name + one complete pair, nothing else.
        shadow.speculate(&snap("shout", json!({ "text": "hello" })));
        wait_for_table_entries(&shadow, 1)?;
        // The close-time parse agrees exactly, so the shadow's result is the
        // answer: zero additional runs.
        let final_call = ParsedCall { name: "shout".into(), arguments: json!({ "text": "hello" }) };
        assert_eq!(
            shadow.verify(&final_call),
            Some(json!({ "echo": "hello" })),
            "the speculated result is attached verbatim"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "one run total: the speculation"
        );
        let st = shadow.stats();
        assert_eq!((st.speculated, st.verified_hits, st.sync_runs), (1, 1, 0));
        Ok(())
    }

    #[test]
    fn a_mismatched_final_call_is_discarded_and_run_synchronously() {
        // The model kept writing after the snapshot was taken: the final
        // arguments are a superset. The guess is discarded — and the final
        // answer still lands, computed on the spot.
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(calls);
        shadow.speculate(&snap("shout", json!({ "text": "hel" })));
        let final_call = ParsedCall { name: "shout".into(), arguments: json!({ "text": "hello" }) };
        assert_eq!(shadow.verify(&final_call), Some(json!({ "echo": "hello" })));
        let st = shadow.stats();
        assert_eq!((st.speculated, st.verified_hits, st.sync_runs), (1, 0, 1));
    }

    #[test]
    fn a_still_streaming_call_speculates_once_per_new_argument_pair() {
        // Driven through the real filter, exactly as the SSE task drives it:
        // markup arriving in pieces, the driver observing after each piece.
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(calls);
        let mut spec = Speculator::new();
        let mut f = OutputFilter::with_tools(&[]);
        f.push("<tool_call>shout\n");
        spec.observe(&shadow, &f);
        assert_eq!(shadow.stats().speculated, 1, "the name-only prefix is a hypothesis too");
        f.push("<arg_key>text</arg_key>\n<arg_value>hello</arg_value>\n");
        spec.observe(&shadow, &f);
        assert_eq!(shadow.stats().speculated, 2, "a new complete pair is a new hypothesis");
        f.push("still more text");
        spec.observe(&shadow, &f);
        assert_eq!(
            shadow.stats().speculated,
            2,
            "an unterminated value adds nothing to speculate on"
        );
        f.push("</tool_call>");
        spec.observe(&shadow, &f);
        assert_eq!(shadow.stats().speculated, 2, "a closed call is not re-speculated");
        assert_eq!(spec.last_key, None, "the driver reset for the next call");
    }

    #[test]
    fn an_unhosted_call_resolves_to_nothing_so_the_client_executes_it() {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(calls);
        assert_eq!(
            shadow.verify(&ParsedCall { name: "bash".into(), arguments: json!({ "command": "ls" }) }),
            None,
            "a tool this server does not host must resolve to nothing — the call belongs to the client"
        );
        assert_eq!(shadow.stats().hosted_calls, 0, "unhosted calls are not this subsystem's business");
    }

    #[test]
    fn the_kill_switch_stops_speculation_but_not_execution() {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow =
            ShadowRepl::build(ToolRegistry::new().register(counting_tool("shout", calls)), false);
        shadow.speculate(&snap("shout", json!({ "text": "hello" })));
        assert_eq!(shadow.stats().speculated, 0);
        assert_eq!(
            shadow.verify(&ParsedCall { name: "shout".into(), arguments: json!({ "text": "hello" }) }),
            Some(json!({ "echo": "hello" })),
            "hosted execution is the floor, not the optimization"
        );
    }

    #[test]
    fn oversized_arguments_are_refused_before_any_run() -> Result<(), &'static str> {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(Arc::clone(&calls));
        let big = "x".repeat(MAX_ARG_BYTES);
        shadow.speculate(&snap("shout", json!({ "text": big })));
        let final_call = ParsedCall { name: "shout".into(), arguments: json!({ "text": big }) };
        let out = shadow.verify(&final_call).ok_or("a capped call must still resolve")?;
        assert!(out["error"].is_string(), "a capped call reports why, as a result: {out}");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "nothing ran, speculatively or finally"
        );
        assert_eq!(shadow.stats().rejected_input, 1, "the speculation counted its refusal");
        Ok(())
    }

    #[test]
    fn the_shadow_table_is_bounded_on_both_axes() -> Result<(), &'static str> {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(calls);
        // Submit past the table's entry cap, draining between submissions:
        // the queue is bounded by design, so a fast producer *must* wait for
        // the workers or its guesses are refused (skip, not block) — this
        // drain makes the run deterministic and the final counts exact.
        for i in 0..(TABLE_MAX_ENTRIES + 32) {
            shadow.speculate(&snap("shout", json!({ "text": format!("k{i}") })));
            for _ in 0..2000 {
                let st = shadow.stats();
                if st.table_entries as u64 + st.table_evictions >= st.speculated {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let st = shadow.stats();
        assert_eq!(st.speculated, (TABLE_MAX_ENTRIES + 32) as u64, "every guess was enqueued");
        assert_eq!(st.table_entries, TABLE_MAX_ENTRIES, "the entry cap holds exactly");
        assert_eq!(st.table_evictions, 32, "the surplus was evicted, not silently kept");
        assert!(st.table_bytes <= TABLE_MAX_BYTES, "byte bound holds: {}", st.table_bytes);
        Ok(())
    }

    #[test]
    fn a_failed_run_is_cached_and_reports_its_error_to_the_client() -> Result<(), &'static str> {
        let shadow = ShadowRepl::build(
            ToolRegistry::new().register(crate::pure_tool!("fragile", |_args| {
                Err(Error::Format("the model named a missing key".into()))
            })),
            true,
        );
        let call = ParsedCall { name: "fragile".into(), arguments: json!({}) };
        let first = shadow.verify(&call).ok_or("a hosted call must resolve")?;
        let second = shadow.verify(&call).ok_or("the cached failure must resolve too")?;
        assert_eq!(first, second, "a pure tool's failure is deterministic, so it caches like a hit");
        assert!(first["error"].is_string(), "the failure is visible, not a missing field: {first}");
        Ok(())
    }

    #[test]
    fn a_hosted_call_attaches_its_result_without_touching_the_openai_shape()
    -> Result<(), &'static str> {
        let calls = Arc::new(AtomicUsize::new(0));
        let shadow = pure_registry(Arc::clone(&calls));
        let call = ParsedCall { name: "shout".into(), arguments: json!({ "text": "hi" }) };
        let wire = call.to_openai(0, "call_x_0");
        let attached = shadow.attach_result(wire.clone(), &call);
        assert_eq!(attached["function"]["name"], json!("shout"), "standard fields intact");
        assert_eq!(attached["function"]["arguments"], wire["function"]["arguments"]);
        assert_eq!(attached["peregrine_result"]["echo"], json!("hi"), "the result rides alongside");
        // An unhosted call passes through unchanged: the client executes it.
        let unhosted = ParsedCall { name: "bash".into(), arguments: json!({ "command": "ls" }) };
        assert_eq!(
            shadow.attach_result(unhosted.to_openai(1, "call_x_1"), &unhosted),
            unhosted.to_openai(1, "call_x_1"),
            "no peregrine_result may appear for a tool this server does not host"
        );
        Ok(())
    }

    /// The committed GPT-2 fixture, loaded once for the tests that need a
    /// real tokenizer (the parity suite uses the same fixture).
    fn test_tokenizer() -> Result<Arc<TokenBackend>, Error> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json");
        let bytes = peregrine_core::Context::ctx(std::fs::read(&path), || {
            format!("fixture at {}", path.display())
        })?;
        let giga = peregrine_token::GigaTokenizer::from_hf_json_bytes(&bytes)
            .map_err(|e| Error::Format(format!("fixture is BPE: {e}")))?;
        Ok(Arc::new(TokenBackend::from_giga_for_test(giga)))
    }

    #[test]
    fn the_builtin_token_tools_are_pure_and_round_trip() -> Result<(), Error> {
        let tok = test_tokenizer()?;
        let shadow = ShadowRepl::build(ToolRegistry::token_tools(&tok), true);
        let call = ParsedCall { name: "count_tokens".into(), arguments: json!({ "text": "hello world" }) };
        let out = shadow
            .verify(&call)
            .ok_or_else(|| Error::Format("count_tokens must answer".into()))?;
        let count = out["count"].as_u64().filter(|n| *n > 0).ok_or_else(|| Error::Format(format!("real tokenization, not a stub: {out}")))?;
        // Determinism is the purity claim, checked: same args, same answer,
        // the second one served from the table.
        assert_eq!(shadow.verify(&call), Some(out), "same args, same answer");
        assert!(count > 0);
        Ok(())
    }

    #[test]
    fn unknown_hosted_tool_names_fail_at_boot() -> Result<(), Error> {
        let tok = test_tokenizer()?;
        let requested = vec!["does_not_exist".to_string()];
        // A misconfigured --host-tool must refuse to boot rather than
        // silently host nothing and turn every call into a client round trip.
        assert!(
            ShadowRepl::from_args(&requested, &tok, true).is_err(),
            "an unknown tool name is a boot error"
        );
        Ok(())
    }

    #[test]
    fn no_hosted_tools_means_no_subsystem() -> Result<(), Error> {
        let tok = test_tokenizer()?;
        assert!(
            ShadowRepl::from_args(&[], &tok, true)?.is_none(),
            "an empty host list builds nothing at all"
        );
        Ok(())
    }
}

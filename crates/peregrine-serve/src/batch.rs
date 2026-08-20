//! Continuous-batching engine: one dedicated OS thread owns the [`Model`] and
//! decodes many concurrent requests in lockstep batched steps.
//!
//! This replaces the one-request-at-a-time `Mutex<Model>` path. Each HTTP handler
//! submits an [`EngineRequest`] and streams sampled token ids back on its own
//! channel; the engine thread prefills each new sequence
//! ([`Model::forward_prefill_seq`]) and then, every step, feeds one decode token
//! per active sequence through a single [`Model::forward_step_batched`]. Because
//! the MoE lane reads each routed expert once and serves every row routing to it,
//! B concurrent sequences share one set of expert reads — the batching
//! amortization that lifts aggregate tokens/sec.
//!
//! Decoding text stays in the async handler (the engine emits token ids only), so
//! the engine has no tokenizer dependency and one code path serves streaming and
//! non-streaming requests.

use std::collections::VecDeque;
use std::thread::JoinHandle;

use parking_lot::Mutex;
use peregrine_core::Error;
use peregrine_model::{Model, RouteHistory, Sampler, SeqKv};
use tokio::sync::mpsc;

/// Batched decode steps between heat-ranked VRAM re-selections ([`Model::reheat`]).
/// A no-op without a GPU tier, so it is harmless in CPU-only deployments.
const REHEAT_EVERY: usize = 256;

/// Ticks between package-energy samples. Matches `GovernorState::SENSOR_PERIOD`
/// and exists for the same reason: a sysfs read per forward is overhead nobody
/// asked for. Totals are unaffected — see the sample site.
const ENERGY_SAMPLE_EVERY: usize = 16;

/// Ticket dispenser for [`SeqState::seq_id`] — the value that picks a sequence's
/// prefetch lane, and the only thing about a sequence that must not move.
///
/// It has to be *stable*, which the obvious candidate is not: `active` is compacted
/// with `retain` every tick, so a sequence's index into it slides down whenever an
/// earlier one retires. Keying the lane on that index (as this did until 2026-08-08)
/// migrated live streams between lanes mid-flight, leaving a sequence's queued reads
/// split across two io_uring rings with no ordering between them.
///
/// The trade-off is deliberate and worth stating: the positional index spread the
/// *currently active* sequences perfectly across lanes, and a monotonic ticket does
/// not — ids 0 and 4 collide on lane 0 under four lanes even if lane 2 is idle. That
/// is the cheaper mistake (a shared warm cache absorbs it), and occupancy-aware
/// assignment is only worth building if the lane-count measurement says lanes pay.
/// Wrapping is fine: the pool takes this modulo its width.
static NEXT_SEQ_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Prompt tokens prefilled per engine step for an admitting sequence. Bounding the
/// chunk lets active sequences keep decoding while a new long prompt prefills, so a
/// big admission doesn't stall the batch for its whole prefill. Also the floor for
/// [`prefill_chunk`].
const PREFILL_CHUNK: usize = 64;

/// Divisor for the adaptive prefill chunk (`COLI_PREFILL_CHUNK_DIV`).
/// Default **4** (2026-08-13): geometric chunk boundaries keep the total
/// dense-path reconstruction linear in prompt length instead of quadratic —
/// see [`prefill_chunk`], whose math is what this default buys. `0` restores
/// the historical fixed [`PREFILL_CHUNK`] exactly.
fn prefill_chunk_div() -> usize {
    std::env::var("COLI_PREFILL_CHUNK_DIV").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(4)
}

/// How many prompt tokens to prefill in one step, given how many are already cached.
///
/// A fixed chunk makes prefill **quadratic in prompt length**. Attention's dense
/// path rebuilds `[k_nope|v]` for *every cached position* on every call
/// (`attend_dense`: `kv_b.apply_vec(&cache.lc[..tk * kvl], tk)`), so a prompt of
/// `N` tokens in fixed chunks of `C` reconstructs `Σ cC ≈ N²/(2C)` rows instead of
/// `N` — at `N = 8192, C = 64` that is ~64× redundant work, per layer, across every
/// layer. Growing the chunk with `pos` makes the chunk boundaries geometric, so the
/// total reconstruction is linear in `N` again.
///
/// **Chunk size cannot change the output** — each token still attends exactly its
/// causal prefix, which is what `engine_chunked_prefill_matches_reference` and
/// `prefill_seq_matches_forward_step` already assert. The only thing traded is how
/// long one prefill step blocks the decode batch — which is why the divisor is a
/// knob (`COLI_PREFILL_CHUNK_DIV=0` restores the historical fixed chunk exactly)
/// even though the default is now the geometric schedule.
/// Pure so the schedule is unit-testable — `iotune.rs` documents why a
/// process-wide `OnceLock` for an enable flag makes a feature untestable.
fn prefill_chunk(pos: usize, div: usize) -> usize {
    match div {
        0 => PREFILL_CHUNK,
        d => PREFILL_CHUNK.max(pos / d),
    }
}

/// Byte budget for the cross-request prefix cache (`COLI_PREFIX_CACHE_MB`).
/// Default **2048 MB** (2026-08-13): prefix reuse is bit-identical by
/// construction (refcounted rows, same math), so the only trade is RAM, and
/// 2 GB buys multi-turn conversations their assistant turns back on an engine
/// where every re-prefilled token streams ~11 GB of experts. `0` disables it —
/// the historical behaviour: every request prefills its whole prompt from
/// scratch.
fn prefix_cache_budget() -> usize {
    std::env::var("COLI_PREFIX_CACHE_MB")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2048)
        .saturating_mul(1024 * 1024)
}

/// Prompts shorter than this are not worth caching: the snapshot copy would cost
/// more than the prefill it could save.
const PREFIX_CACHE_MIN_TOKENS: usize = 64;

/// One cached prompt and the KV it produced.
struct PrefixEntry {
    tokens: Vec<i32>,
    kv: SeqKv,
    bytes: usize,
    used: u64,
}

/// Cross-request KV prefix cache.
///
/// Every request built a fresh `SeqKv` and prefilled from position 0, so N
/// requests sharing a system prompt each paid its full prefill — the dominant
/// cost on a disk-bound engine, since every prompt token routes its own experts.
///
/// Sharing is sound because each position attends only its causal prefix: two
/// prompts agreeing on their first `n` tokens produce bit-identical KV for those
/// positions, so seeding from a cached prefix gives exactly what prefilling them
/// would have.
///
/// Entries are matched by comparing **tokens**, not a hash. A hash-only key would
/// let a collision serve another prompt's KV, which is silent, unbounded
/// corruption of the output — the one failure mode this must not have.
struct PrefixCache {
    entries: Vec<PrefixEntry>,
    budget: usize,
    used: usize,
    clock: u64,
    hits: u64,
    tokens_saved: u64,
}

impl PrefixCache {
    fn new(budget: usize) -> PrefixCache {
        PrefixCache { entries: Vec::new(), budget, used: 0, clock: 0, hits: 0, tokens_saved: 0 }
    }

    fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// Index and match length of the best entry for `prompt`, touching no
    /// counters — shared by [`Self::lookup`] and [`PrefixStore`]'s probe of
    /// whether the disk index can do better before it reads a checkpoint.
    fn best_match(&self, prompt: &[i32]) -> Option<(usize, usize)> {
        if !self.enabled() || prompt.len() < 2 {
            return None;
        }
        let cap = prompt.len() - 1;
        let mut best: Option<(usize, usize)> = None; // (entry index, prefix len)
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(cap).min(e.kv.len());
            if n > 0 && best.is_none_or(|(_, b)| n > b) {
                best = Some((i, n));
            }
        }
        best
    }

    /// Longest cached prefix of `prompt`, as a seeded cache and its length.
    ///
    /// Never returns the whole prompt: prefill must still run for at least one
    /// position, since that forward is what produces the logits the first token
    /// is sampled from.
    fn lookup(&mut self, prompt: &[i32]) -> Option<(SeqKv, usize)> {
        let (i, n) = self.best_match(prompt)?;
        self.clock += 1;
        let clock = self.clock;
        let e = self.entries.get_mut(i)?;
        e.used = clock;
        self.hits += 1;
        self.tokens_saved = self.tokens_saved.saturating_add(n as u64);
        Some((e.kv.clone_prefix(n), n))
    }

    /// Cache `prompt`'s completed KV, evicting least-recently-used entries to
    /// stay within budget. A prompt already covered by an equal-or-longer entry
    /// is not stored twice.
    fn insert(&mut self, prompt: &[i32], kv: &SeqKv) {
        if !self.enabled() || prompt.len() < PREFIX_CACHE_MIN_TOKENS || kv.len() < prompt.len() {
            return;
        }
        if self.entries.iter().any(|e| e.tokens.len() >= prompt.len() && e.tokens.starts_with(prompt)) {
            return;
        }
        // Entries the new one strictly covers are redundant — any lookup they
        // could serve, the longer entry serves at least as well. Dropping them
        // here keeps a conversation's turn-by-turn retires from accumulating
        // one entry per turn (each a prefix of the next).
        let used = &mut self.used;
        self.entries.retain(|e| {
            let covered = e.tokens.len() < prompt.len() && prompt.starts_with(&e.tokens);
            if covered {
                *used = used.saturating_sub(e.bytes);
            }
            !covered
        });
        let snapshot = kv.clone_prefix(prompt.len());
        let bytes = snapshot.bytes();
        if bytes > self.budget {
            return; // a single sequence larger than the whole budget
        }
        self.clock += 1;
        self.entries.push(PrefixEntry { tokens: prompt.to_vec(), kv: snapshot, bytes, used: self.clock });
        self.used += bytes;
        while self.used > self.budget && self.entries.len() > 1 {
            let Some(victim) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
            else {
                break;
            };
            self.used -= self.entries.swap_remove(victim).bytes;
        }
    }
}

/// The in-memory prefix cache plus its optional disk extension
/// (`COLI_KV_STORE_DIR`, see [`crate::kvstore`]). One type, one `&mut`
/// parameter, everywhere the engine used to pass the cache — wrapping instead
/// of widening signatures, since `finish_prefill_chunk` already sits at the
/// strict audit's argument limit.
struct PrefixStore {
    mem: PrefixCache,
    disk: Option<crate::kvstore::KvSessionStore>,
}

impl PrefixStore {
    fn new(mem_budget: usize, disk: Option<crate::kvstore::KvSessionStore>) -> PrefixStore {
        PrefixStore { mem: PrefixCache::new(mem_budget), disk }
    }

    /// Memory first. The disk is consulted only when its index says it can
    /// beat memory's best match — a pure token compare, no file I/O — which is
    /// the after-restart case and the conversation-continued-across-restart
    /// case. A disk hit is promoted into memory, so each checkpoint is read at
    /// most once per process and every later request takes the cheap path.
    fn lookup(&mut self, prompt: &[i32]) -> Option<(SeqKv, usize)> {
        let mem_n = self.mem.best_match(prompt).map_or(0, |(_, n)| n);
        if let Some(disk) = &mut self.disk {
            if disk.best_match_len(prompt) > mem_n {
                if let Some((kv, n)) = disk.load_longest(prompt) {
                    if n > mem_n {
                        self.mem.insert(&prompt[..n], &kv);
                        return Some((kv, n));
                    }
                }
            }
        }
        self.mem.lookup(prompt)
    }

    /// Every insert is offered to the disk first (its own floor, trim, and
    /// dedup decide whether anything is written), then cached in memory.
    /// Because qualifying entries reach the disk at insert time, memory
    /// eviction never loses anything the disk wanted and shutdown needs no
    /// separate persist pass — the ds4 trigger list (long prefill, eviction,
    /// shutdown) collapses to this one hook.
    fn insert(&mut self, prompt: &[i32], kv: &SeqKv) {
        if let Some(disk) = &mut self.disk {
            disk.save(prompt, kv);
        }
        self.mem.insert(prompt, kv);
    }
}

/// A sequence being prefilled incrementally (chunked) before it joins `active`.
/// Chunked prefill is bit-identical to a whole-prompt prefill — the KV is built up
/// the same way, each token attending its causal prefix.
struct Prefilling {
    seq: SeqKv,
    prompt: Vec<i32>,
    pos: usize, // next prompt position to prefill
    sampler: Sampler,
    out: mpsc::UnboundedSender<EngineOut>,
    max_new: usize,
    /// This stream's routing history, created at admission rather than at
    /// promotion so a fused prefill row has somewhere to record. Moves into the
    /// `SeqState` when the prefill completes, so the sequence keeps one history
    /// across its whole life instead of starting blank at its first decode.
    hist: Mutex<RouteHistory>,
    /// Stable prefetch-lane key, from [`NEXT_SEQ_ID`]. Assigned at admission and
    /// carried into the `SeqState`, for the same reason `hist` is: a fused prefill
    /// row prefetches too, and it must use the lane the sequence will keep.
    seq_id: usize,
}

/// Request priority. Higher priority requests are admitted and drained before
/// normal ones — the batching engine drains its `high` channel first each tick,
/// then normal. Correctness-neutral: priority only reorders admission, never the
/// final token stream for any individual request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Priority {
    #[default]
    Normal,
    High,
}

/// A generation request handed to the engine. The engine prefills `prompt`, then
/// emits sampled token ids on `out` until a stop id, `max_new` tokens, or the
/// client drops `out`. `sampler` carries this request's temperature/nucleus/seed,
/// so each sequence draws from its own RNG stream.
pub struct EngineRequest {
    pub prompt: Vec<i32>,
    pub max_new: usize,
    pub sampler: Sampler,
    pub out: mpsc::UnboundedSender<EngineOut>,
    /// Admission priority. Defaults to `Normal`; the HTTP handler maps an
    /// optional `X-Peregrine-Priority: high` header to `High`.
    #[doc(hidden)]
    pub priority: Priority,
    /// Workload class inferred from the prompt tail by the HTTP handler (the
    /// engine is tokenizer-free, so classification happens where the text is).
    /// Selects per-class prefetch-breadth overrides on the model
    /// (`COLI_PREFETCH_WARM_PATHS_<CLASS>`). `TokenClass::Prose` == base policy.
    pub class: peregrine_model::TokenClass,
}

/// One engine → handler message. The token stream ends when the channel closes
/// (stop id, `max_new` reached, or a fatal error already reported via `Error`).
pub enum EngineOut {
    Token(u32),
    Error(String),
}

/// What the engine thread publishes each tick for `GET /metrics` to read.
///
/// The engine thread **owns** the `Model` exclusively — the HTTP handlers hold
/// only channels — so a scrape cannot call `Model::telemetry()` itself. This is
/// the seam: the engine writes a snapshot once per tick, the handler clones it.
/// Cheap enough to publish unconditionally (a struct copy under an uncontended
/// lock, once per decode step, against a step that reads gigabytes).
#[derive(Clone, Debug, Default)]
pub struct EngineTelemetry {
    pub runtime: peregrine_model::RuntimeTelemetry,
    /// Most recent forward's per-lane wall time.
    pub lane_last: peregrine_model::LaneTimings,
    /// The bubble tuner's smoothed per-lane times — what the balancer acts on.
    /// Distinct from `lane_last`: one token's cost versus which lane dominates.
    pub lane_ewma: peregrine_model::LaneTimings,
    /// Sequences decoding right now, and prompts waiting to be prefilled.
    pub active: usize,
    pub pending: usize,
    /// Decode steps the engine has run since start.
    pub steps: u64,
    /// Warm-cache `(hits, misses, disk_reads)` and speculative reads, cumulative.
    ///
    /// Published because these are the only *byte-convertible* counters the engine
    /// has — `disk_reads × bytes_per_expert` is the disk rate, and the two lane
    /// timing fields cannot give it (they are thread-sums, so they need both
    /// thread counts to interpret and still exclude the prefetch lane entirely,
    /// which is untimed). Until 2026-08-10 these reached the shutdown log and
    /// nothing else, so a run's throughput could only be computed after it ended.
    /// `None` without a warm cache.
    pub ecache: Option<(u64, u64, u64)>,
    pub prefetch_reads: u64,
    /// MTP speculation, cumulative: drafts proposed and drafts the model's
    /// accept rule kept. The ratio is the only number that says whether
    /// `COLI_DRAFT`'s depth is earning its rows — each proposed draft is a
    /// verify row in the batched forward, and a rejected one is a row of
    /// wasted expert reads. Both zero when speculation is off.
    pub spec_proposed: u64,
    pub spec_accepted: u64,
    /// Drafts the `COLI_SPEC_CONF` floor cut short of their requested depth,
    /// cumulative. Zero when the floor is off (the default).
    pub spec_conf_stops: u64,
    /// The cost side of `COLI_SPEC_GDN`, cumulative, all three zero unless it
    /// is on: bytes copied into pre-verify recurrent snapshots, ticks where at
    /// least one sequence overshot and needed a re-advance, and rows that
    /// re-advance redid. A snapshot is ~151 MB per drafting sequence at
    /// Qwen3.5-27B dims, so `gdn_snapshot_bytes` against the tokens the same
    /// window emitted is the break-even question stated directly; `gdn_replays`
    /// says how often the free path (full acceptance) was missed, which is what
    /// the `COLI_SPEC_CONF` floor exists to raise.
    pub gdn_snapshot_bytes: u64,
    pub gdn_replays: u64,
    pub gdn_replay_rows: u64,
    /// Prompt-lookup drafting (`COLI_DRAFT_NGRAM`), cumulative and **separate
    /// from `spec_*`**: the two sources cost so differently — a memcmp against
    /// a sparse-MoE layer per draft step — that one pooled accept rate would
    /// average them into a number that decides nothing. Keeping them apart is
    /// what lets an operator see that the free source is carrying the run.
    pub ngram_proposed: u64,
    pub ngram_accepted: u64,
    /// Tokens actually sent to clients, and decode rows actually forwarded,
    /// both cumulative. **These are the numerators the engine was missing.**
    ///
    /// `/metrics` already published the denominator — `ecache.hits + misses` is
    /// every routed-expert entry the streaming lane resolved, and
    /// `ecache.disk_reads` is the subset that reached the device — but nothing
    /// published what those reads bought. Delta both across two scrapes and the
    /// quantity that decides every speculative technique on the streaming track
    /// falls out directly:
    ///
    /// ```text
    /// tokens per expert read  = Δtokens_emitted / Δ(hits + misses)
    /// tokens per disk read    = Δtokens_emitted / Δdisk_reads
    /// rows per token          = Δdecode_rows    / Δtokens_emitted
    /// ```
    ///
    /// The first two are `docs/speculative-decoding-alternatives.md`'s stated
    /// metric — SSD bytes per *accepted* token — made observable instead of
    /// derived. The third is speculation's row overhead, and it is what an
    /// expert-union budget would be spent against. Wall clock cannot separate
    /// these, which is why the runbook keeps saying to measure fusion and
    /// speculation with counters instead.
    ///
    /// Read `rows per token` with one correction: the **first** token of each
    /// request is sampled from the prefill's last position and costs no decode
    /// row at all, so the ratio sits just *below* 1.0 unspeculated and
    /// approaches 1.0 as requests lengthen. Above 1.0 is speculation's
    /// overhead; a short-request workload that reads 0.8 is not a win.
    pub tokens_emitted: u64,
    pub decode_rows: u64,
    /// Admission latency: total and count (mean = total / count) plus the worst
    /// single wait, all cumulative, in microseconds.
    ///
    /// The one span no existing instrument covers.
    /// `bench-serve-lanes.py` times whole requests and `bench-serve-gaps.py`
    /// times inter-token gaps; both start counting once a request is *already
    /// being served*, so queue time hides inside "the server was slow" and
    /// cannot be told apart from decode being slow. With `COLI_QUEUE_DEPTH`
    /// shedding at the door, knowing whether admitted requests also *waited* is
    /// what separates "at capacity" from "over capacity".
    pub queue_wait_us: u64,
    pub queue_admits: u64,
    pub queue_wait_max_us: u64,
    /// Times `COLI_SPEC_UNION_MAX` cut a draft below the depth the row cap and
    /// token budget would otherwise have allowed, cumulative. Zero when the
    /// ceiling is unset (the default) or never binds.
    ///
    /// The cost-side counterpart of `spec_conf_stops`: that one says how often
    /// the *acceptance* floor bit, this one how often the *byte* ceiling did.
    /// Together they say which of the two terms in
    /// `(1 + accepted) / union_growth` is actually limiting a run.
    pub spec_union_stops: u64,
    /// Ticks on which a sequence's two draft sources disagreed enough to be
    /// worth verifying as a **tree** (`COLI_DRAFT_TREE`), and — of those — how
    /// often the accepted path left the first branch.
    ///
    /// `trees` counts the hedge being taken at all; `branch_wins` counts it
    /// paying, i.e. the model following the MTP chain where prompt-lookup would
    /// have been committed alone. A `trees` that climbs with `branch_wins` flat
    /// means the extra rows are buying nothing and the knob should go back off.
    pub spec_trees: u64,
    pub spec_tree_branch_wins: u64,
    /// Cumulative CPU-package energy in microjoules, or `None` when the host
    /// will not give it up.
    ///
    /// With `tokens_emitted` this is **joules per token**, which the project has
    /// never measured. Two caveats it must be read with, both real:
    ///
    /// - **RAPL covers the CPU package, not the machine.** On this box the
    ///   domains are `package-0` and `core` — no DRAM domain, and certainly no
    ///   SSD. On an engine whose bottleneck is 10.85 GB of expert reads per
    ///   token, the component doing the most work is the one RAPL cannot see,
    ///   so this figure is a floor on system energy, not an estimate of it.
    /// - **`energy_uj` is root-only on most current kernels** (the PLATYPUS
    ///   mitigation), so this reads `None` for an unprivileged server. To
    ///   enable it, grant the counter rather than running the server as root:
    ///   `SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod g+r
    ///   /sys/class/powercap/%k/energy_uj"` plus a group the server belongs to.
    ///
    /// `None` is therefore the expected reading on a stock host, and is
    /// deliberately not `0` — zero energy and no permission are different facts.
    pub energy_uj: Option<u64>,
    /// RLM recursive refinement `(passes_emitted, tokens_recursed)`,
    /// cumulative — `(0, 0)` unless `COLI_RLM=1`.
    pub rlm: (u64, u64),
    /// Disk-persisted KV sessions `(saved, loaded, tokens_restored)`,
    /// cumulative — `None` unless `COLI_KV_STORE_DIR` is set.
    pub kvstore: Option<(u64, u64, u64)>,
    /// O_DIRECT slab buffers currently checked out across the streaming rings
    /// (`None` when experts are resident). Stuck at the pool cap = reads are
    /// serializing on buffer availability.
    pub io_slab_in_use: Option<usize>,
}

/// Handle for submitting requests to the engine thread. Cheap to clone and
/// `Send + Sync` (a tokio unbounded sender), so it lives in shared server state.
#[derive(Clone)]
pub struct EngineHandle {
    tx_normal: mpsc::UnboundedSender<Queued>,
    tx_high: mpsc::UnboundedSender<Queued>,
    /// Published by the engine thread each tick; read by `/metrics`.
    telemetry: std::sync::Arc<parking_lot::Mutex<EngineTelemetry>>,
    /// Requests sent but not yet drained by the engine (see [`queue_depth_cap`]).
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Resolved [`queue_depth_cap`]; 0 = unbounded.
    queue_cap: usize,
}

/// Why [`EngineHandle::submit`] refused a request.
#[derive(Debug)]
pub enum SubmitRefused {
    /// `COLI_QUEUE_DEPTH` backpressure — the backlog is at its cap; the client
    /// should retry (HTTP 503), and nothing about the request was wrong.
    Full,
    /// The engine thread is gone (shutdown or crash).
    Down(String),
}

impl From<SubmitRefused> for Error {
    fn from(r: SubmitRefused) -> Error {
        match r {
            SubmitRefused::Full => Error::Format("engine queue is full (COLI_QUEUE_DEPTH)".into()),
            SubmitRefused::Down(m) => Error::Format(format!("batch engine is not running: {m}")),
        }
    }
}

impl EngineHandle {
    /// Submit a request at its `priority` — the engine drains high-priority
    /// requests before normal ones each tick. Refuses with
    /// [`SubmitRefused::Full`] against a backlog at `COLI_QUEUE_DEPTH`, or
    /// [`SubmitRefused::Down`] once the engine thread has shut down.
    pub fn submit(&self, req: EngineRequest) -> Result<(), SubmitRefused> {
        use std::sync::atomic::Ordering::Relaxed;
        // Approximate on purpose: two racing submits can both pass a cap-1
        // check. The cap is overload shedding, not an admission invariant —
        // off-by-a-few under race is fine, silently unbounded is not.
        if self.queue_cap > 0 && self.queued.load(Relaxed) >= self.queue_cap {
            return Err(SubmitRefused::Full);
        }
        let ch = match req.priority {
            Priority::High => &self.tx_high,
            Priority::Normal => &self.tx_normal,
        };
        ch.send(Queued { req, at: std::time::Instant::now() })
            .map_err(|send_err| SubmitRefused::Down(send_err.to_string()))?;
        self.queued.fetch_add(1, Relaxed);
        Ok(())
    }

    /// The engine's most recent published telemetry. All-zero before the first
    /// tick, which is the honest answer to "what is it doing" at that point.
    pub fn telemetry(&self) -> EngineTelemetry {
        self.telemetry.lock().clone()
    }
}

/// A submitted request plus the instant it entered the queue.
///
/// Internal, and deliberately not a field on [`EngineRequest`]: a caller should
/// not have to stamp its own clock to be measured, and a caller that stamped it
/// wrong would bias the number silently. The engine takes the timestamp at the
/// one place every request passes through — [`EngineHandle::submit`].
struct Queued {
    req: EngineRequest,
    at: std::time::Instant,
}

/// State shared between the engine thread and its [`EngineHandle`], bundled
/// so `run` stays within the argument count its own comment defends.
struct EngineShared {
    telemetry: std::sync::Arc<parking_lot::Mutex<EngineTelemetry>>,
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// Every engine-loop knob, resolved from the environment **once at spawn** and
/// threaded through — never a process-wide `OnceLock`. Two reasons, both
/// already paid for in this repo: an `OnceLock` latch voided the
/// route-min-share A/B (both arms in one process read the first arm's value —
/// `todo.md` §6), and the `spawn_fused`/`spawn_tuned`/`spawn_spec` ladder below
/// existed purely so tests could dodge the latch knob-by-knob. Tests now
/// override fields on this struct instead; the parser functions keep their
/// documentation and defaults and are called exactly once, from [`Self::from_env`].
#[derive(Clone, Copy)]
struct EngineKnobs {
    /// `COLI_FUSE_PREFILL` — see [`fuse_prefill`].
    fuse_prefill: bool,
    /// `COLI_PREFILL_CHUNK_DIV` — see [`prefill_chunk_div`].
    prefill_chunk_div: usize,
    /// `COLI_PREFIX_CACHE_MB` in bytes — see [`prefix_cache_budget`].
    prefix_cache_budget: usize,
    /// `COLI_BATCH_SLA_MS` — see [`batch_sla_ms`].
    batch_sla_ms: Option<u64>,
    /// `COLI_DRAFT` — see [`draft_depth`].
    draft_depth: usize,
    /// `COLI_DRAFT_SAMPLED` — see [`draft_sampled`].
    draft_sampled: bool,
    /// `COLI_SPEC_CONF` — see [`spec_conf`].
    spec_conf: f32,
    /// `COLI_SPEC_GDN` — see [`spec_gdn`].
    spec_gdn: bool,
    /// `COLI_SPEC_GDN_MAX_B` — see [`spec_gdn_max_b`].
    spec_gdn_max_b: usize,
    /// `COLI_DRAFT_NGRAM` — see [`draft_ngram`].
    draft_ngram: usize,
    /// `COLI_SPEC_UNION_MAX` — see [`spec_union_max`].
    spec_union_max: u64,
    /// `COLI_DRAFT_TREE` — see [`draft_tree`].
    draft_tree: bool,
    /// `COLI_MAX_BATCH_ROWS` — see [`max_batch_rows`].
    max_batch_rows: usize,
    /// `COLI_QUEUE_DEPTH` — see [`queue_depth_cap`].
    queue_depth_cap: usize,
    /// `COLI_ADAPTIVE_WINDOW` — see [`adaptive_window_ratio`].
    adaptive_window_ratio: u64,
    /// `COLI_KV_BUDGET_MB` in bytes — see [`kv_budget_bytes`].
    kv_budget_bytes: usize,
}

impl EngineKnobs {
    /// The one place the engine reads its environment.
    fn from_env() -> EngineKnobs {
        EngineKnobs {
            fuse_prefill: fuse_prefill(),
            prefill_chunk_div: prefill_chunk_div(),
            prefix_cache_budget: prefix_cache_budget(),
            batch_sla_ms: batch_sla_ms(),
            draft_depth: draft_depth(),
            draft_sampled: draft_sampled(),
            spec_conf: spec_conf(),
            spec_gdn: spec_gdn(),
            spec_gdn_max_b: spec_gdn_max_b(),
            draft_ngram: draft_ngram(),
            spec_union_max: spec_union_max(),
            draft_tree: draft_tree(),
            max_batch_rows: max_batch_rows(),
            queue_depth_cap: queue_depth_cap(),
            adaptive_window_ratio: adaptive_window_ratio(),
            kv_budget_bytes: kv_budget_bytes(),
        }
    }
}

/// Spawn the engine on a dedicated OS thread that owns `model`, batching up to
/// `max_batch` sequences per step. Returns a submit handle and the thread's join
/// handle (the thread exits once every [`EngineHandle`] is dropped and all active
/// sequences finish).
pub fn spawn(model: Model, max_batch: usize) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    spawn_with_knobs(model, max_batch, EngineKnobs::from_env())
}

/// [`spawn`] with the knobs supplied by the caller — production hands it
/// [`EngineKnobs::from_env`]; tests hand it a struct with the fields under test
/// overridden, so no test ever mutates the process environment.
fn spawn_with_knobs(
    model: Model,
    max_batch: usize,
    knobs: EngineKnobs,
) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    let (tx_normal, rx_normal) = mpsc::unbounded_channel::<Queued>();
    let (tx_high, rx_high) = mpsc::unbounded_channel::<Queued>();
    let cap = max_batch.max(1);
    let telemetry = std::sync::Arc::new(parking_lot::Mutex::new(EngineTelemetry::default()));
    let queued = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let shared = EngineShared { telemetry: telemetry.clone(), queued: queued.clone() };
    let join = std::thread::Builder::new()
        .name("peregrine-batch".to_string())
        .spawn(move || run(model, rx_normal, rx_high, cap, knobs, shared))
        .map_err(|e| Error::Format(format!("spawn batch engine thread: {e}")))?;
    Ok((EngineHandle { tx_normal, tx_high, telemetry, queued, queue_cap: knobs.queue_depth_cap }, join))
}

/// [`spawn`] with the prefill/decode fusion forced on or off (test knob override).
#[cfg(test)]
fn spawn_fused(model: Model, max_batch: usize, fuse: bool) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    spawn_tuned(model, max_batch, fuse, None)
}

/// Override for the speculation knobs, `None` meaning "the environment's value".
/// Kept as the test ladder's argument shape; it folds into [`EngineKnobs`] in
/// [`spawn_spec`]. Bundled rather than passed as parameters because the two are
/// one decision: sampled speculation without a depth is not a configuration.
#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct SpecOverride {
    /// `COLI_DRAFT`.
    depth: Option<usize>,
    /// `COLI_DRAFT_SAMPLED`.
    sampled: Option<bool>,
    /// `COLI_SPEC_CONF`.
    conf: Option<f32>,
    /// `COLI_SPEC_GDN`.
    gdn: Option<bool>,
    /// `COLI_DRAFT_NGRAM`.
    ngram: Option<usize>,
    /// `COLI_SPEC_UNION_MAX`.
    union_max: Option<u64>,
    /// `COLI_DRAFT_TREE`.
    tree: Option<bool>,
}

/// [`spawn`] with the fusion and the speculation depth forced (test knob override).
#[cfg(test)]
fn spawn_tuned(
    model: Model,
    max_batch: usize,
    fuse: bool,
    depth: Option<usize>,
) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    spawn_spec(model, max_batch, fuse, SpecOverride { depth, ..SpecOverride::default() })
}

/// [`spawn_tuned`] with the sampled-speculation knob forced too.
#[cfg(test)]
fn spawn_spec(
    model: Model,
    max_batch: usize,
    fuse: bool,
    spec: SpecOverride,
) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    let mut knobs = EngineKnobs::from_env();
    knobs.fuse_prefill = fuse;
    if let Some(d) = spec.depth {
        knobs.draft_depth = d;
    }
    if let Some(s) = spec.sampled {
        knobs.draft_sampled = s;
    }
    if let Some(c) = spec.conf {
        knobs.spec_conf = c;
    }
    if let Some(g) = spec.gdn {
        knobs.spec_gdn = g;
    }
    if let Some(n) = spec.ngram {
        knobs.draft_ngram = n;
    }
    if let Some(u) = spec.union_max {
        knobs.spec_union_max = u;
    }
    if let Some(t) = spec.tree {
        knobs.draft_tree = t;
    }
    spawn_with_knobs(model, max_batch, knobs)
}

/// Latency SLA target for adaptive batching, in milliseconds. When set
/// (`COLI_BATCH_SLA_MS=<n>`, or an [`EngineKnobs`] override in tests), the engine
/// shrinks the working batch cap on p95-latency overrun and grows it back when
/// slack appears. Unset → static `max_batch` (the historical default).
fn batch_sla_ms() -> Option<u64> {
    std::env::var("COLI_BATCH_SLA_MS").ok().and_then(|s| s.trim().parse::<u64>().ok()).filter(|&n| n > 0)
}

/// Prompt-lookup draft depth in *suffix length* (`COLI_DRAFT_NGRAM=<max_n>`,
/// typical 3). `0`/unset is off, and anything below the module's 2-token floor
/// reads as off.
///
/// A second draft source for the same verify path, costing a backward scan of
/// the token history instead of a model call. It proposes whatever followed the
/// last occurrence of the current suffix, so it is right exactly when the
/// output repeats something already in context — quoted code, an edited file, a
/// list being walked.
///
/// It takes priority over the MTP head whenever it matches, because it is
/// strictly cheaper: an MTP draft step is a full sparse-MoE layer at `s_n = 1`
/// (~300 MB of SSD per step on the streaming container, with no batch-union
/// amortization), and this is a memcmp. When it does not match, the MTP head
/// drafts as before — the two are alternatives per tick, not a chain, because
/// `mtp_draft` continues from a hidden state that assumes its own prefix.
///
/// Greedy requests only, and not because of a policy choice: an n-gram draft is
/// not drawn from any distribution, so `accept_run_sampled` would have no `q`
/// to score it against and the distribution-preserving guarantee would be void.
/// `COLI_DRAFT_SAMPLED` does not extend to it.
///
/// Needs `COLI_DRAFT` non-zero for the depth, but **not** an MTP head: a
/// checkpoint converted without `--mtp` can speculate through this alone.
fn draft_ngram() -> usize {
    std::env::var("COLI_DRAFT_NGRAM").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
}

/// Verify both draft sources as a **token tree** instead of choosing one
/// (`COLI_DRAFT_TREE=1`). Default **off**.
///
/// Today prompt-lookup and the MTP head are alternatives: when the n-gram
/// matches it wins and the head's chain is discarded unseen. They are often
/// right about different continuations, and one forward can check both — root
/// = the pending token, one branch per source. Whichever the model's own argmax
/// follows is the one that commits, by the same greedy-identity rule a chain
/// uses, so the served stream is unchanged either way.
///
/// **MLA only, and that is the constraint that decides where this can pay.** A
/// recurrent layer advances one delta-rule state row by row, so siblings would
/// chain instead of branching; the batched GQA path takes no key set at all.
/// `Model::forward_tree_rows_hidden` refuses both rather than silently
/// linearizing a tree. So this runs on the *streaming* track, where an extra
/// verify row costs disk bytes — and not on the resident track, where extra
/// rows would be nearly free. That is backwards from where the value is, and it
/// is why tree width must be spent against [`spec_union_max`] rather than set
/// to a constant.
///
/// Two costs to weigh before enabling it, both real:
///
/// - **Every branch row is a full row of the routed-expert union.** A two-branch
///   tree roughly doubles a sequence's draft rows, and on the streaming
///   container that is bytes, not FLOPs.
/// - **A tree row's key set is explicit, so it is O(context) to build and walks
///   an index list where a dense row runs a tight loop.** At a 4 k context and
///   five nodes that is ~160 KB per forward; at 100 k it is megabytes. Trees are
///   therefore cheapest at *short* context. The fix is a compact
///   `prefix + extras` key-set representation so the prefix part stays a range;
///   until that exists, this knob is for short-context workloads.
///
/// Greedy requests only: a sampled request needs `accept_run_sampled` and the
/// `q` each draft was drawn from, and there is no tree analogue of that rule.
fn draft_tree() -> bool {
    matches!(std::env::var("COLI_DRAFT_TREE").ok().as_deref(), Some("1") | Some("true"))
}

/// Ceiling on the routed-expert union a single tick may cost, in expert-read
/// requests (`COLI_SPEC_UNION_MAX`). `0`/unset is off.
///
/// The cost-side twin of [`spec_conf`]. `COLI_SPEC_CONF` prunes drafts by
/// expected **acceptance** and inverted the `COLI_DRAFT=4` regression into
/// +37 % — but nothing pruned them by expected **cost**, which is the term the
/// 2.63× union growth at γ=4 actually lives in. Speculation's whole economics
/// on the streaming track is
///
/// ```text
/// speedup = (1 + accepted) / union_growth
/// ```
///
/// and until this knob the engine could only act on the numerator.
///
/// The projection is deliberately **conservative**: expected union entries are
/// estimated as `rows × (entries per row)`, where the per-row figure is an EWMA
/// of what recent ticks actually cost (`ecache` hits + misses, which is exactly
/// the union entries the warm tier resolved). A real union is *sublinear* in
/// rows — that sublinearity is the entire batching win — so a linear projection
/// overestimates, and the gate cuts depth sooner than strictly necessary. For a
/// budget that is the safe direction, and it is stated here rather than
/// discovered later from a disappointing sweep.
///
/// Depth-only, exactly like the confidence floor: it changes how many rows are
/// proposed and never which token is emitted, so a greedy stream stays
/// bit-identical. Inert on a resident model, where `ecache_stats` is `None`
/// because no expert is ever read.
///
/// **Unset by default and deliberately untuned.** The number that should set it
/// is `decode.tokens_emitted` against `ecache`, measured on the real container;
/// picking a ceiling before that exists would be tuning against a quantity
/// nobody has measured, which is the failure `docs/measurement.md` opens with.
fn spec_union_max() -> u64 {
    std::env::var("COLI_SPEC_UNION_MAX").ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0)
}

/// Speculation on a **recurrent** architecture (`COLI_SPEC_GDN`). Default
/// **off**.
///
/// A hybrid (GDN) sequence's context is a point state, not per-position rows:
/// the verify forward advances it by `1 + γ` tokens and `truncate` cannot undo
/// that. The rollback is `SeqKv::gdn_snapshot` before the forward and
/// `gdn_restore` + a re-advance over the accepted rows after a partial
/// acceptance — see the tick in [`run`]. This knob is what says that rollback
/// is wired; with it off, `spec_reject_is_kv_only()` alone decides and hybrid
/// models decode one token at a time exactly as before.
///
/// It is a knob rather than an unconditional enable because the snapshot is not
/// free: ~3.1 MB per linear layer (≈151 MB per sequence at Qwen3.5-27B's 48
/// linear layers), taken every tick a sequence drafts. Output is unaffected
/// either way — `accept_run` still pins the greedy stream — so this is a
/// throughput decision, and it has to be measured per batch width.
fn spec_gdn() -> bool {
    matches!(std::env::var("COLI_SPEC_GDN").ok().as_deref(), Some("1") | Some("true"))
}

/// Batch width above which recurrent speculation switches itself off
/// (`COLI_SPEC_GDN_MAX_B`). `0`/unset is uncapped.
///
/// The snapshot cost is per *sequence*, so it scales with the batch while the
/// resident weight read that dominates a forward does not. There is therefore a
/// width past which the copies cost more than the accepted tokens repay. Rather
/// than let that regress a busy server silently, the operator caps it — and the
/// `gdn_snapshot_bytes` counter says where the crossover actually is on this
/// box, instead of leaving it to be guessed.
fn spec_gdn_max_b() -> usize {
    std::env::var("COLI_SPEC_GDN_MAX_B").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
}

/// Speculative draft depth for the batched engine (`COLI_DRAFT`). `0`/unset is
/// the historical one-token-per-sequence decode.
///
/// Use 4-6, never 2: the only published "MTP barely helps" figure for this
/// model class came from a depth-2 fork where 2.46 accepted was already 82% of
/// that configuration's ceiling of 3.
fn draft_depth() -> usize {
    std::env::var("COLI_DRAFT").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
}

/// Confidence floor for the MTP draft (`COLI_SPEC_CONF`, clamped to `[0, 1)`).
/// `0`/unset drafts the full `COLI_DRAFT` depth — the historical behavior and
/// the default until an A/B licenses otherwise.
///
/// The ds4/DSpark observation this ports: draft yield is bimodal — predictable
/// continuations accept nearly everything, uncertain ones reject nearly
/// everything — and the MTP head's own top-token probability separates the two
/// before any verify row is spent. Every drafted token becomes a verify row in
/// the batched forward, and a rejected row is a row of wasted expert reads, so
/// stopping a low-confidence draft attacks bytes/accepted-token directly.
/// Depth-only by design: `accept_run`'s greedy identity is untouched, so this
/// knob can never change emitted tokens (ds4's version gates acceptance too and
/// documents output drift; peregrine's invariant forbids that trade).
/// ds4 ships 0.6 (Metal) / 0.7 (CUDA) as defaults — 0.65 is the A/B arm.
fn spec_conf() -> f32 {
    std::env::var("COLI_SPEC_CONF")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 0.999_999))
        .unwrap_or(0.0)
}

/// Extend speculation to temperature > 0 requests via rejection sampling
/// (`COLI_DRAFT_SAMPLED`). Default **off**, and it is not a performance knob.
///
/// [`peregrine_model::speculative_sample`] emits exactly the target
/// distribution, so this is *distributionally* invisible — but not
/// *sequence*-invisible: the tokens a seeded request produces change, because
/// rejection sampling draws two uniforms per draft where plain decode draws one
/// per token. A caller pinning outputs by seed would see different text with the
/// same seed and no error anywhere.
///
/// It also rests on a claim this engine cannot check: that the MTP head's
/// distribution really is the `q` the drafts were drawn from. `mtp_draft_sampled`
/// makes that true by construction *for the head as implemented*; whether that
/// head is a good proposal distribution at temperature is a modelling question,
/// and a bad one costs acceptance rate rather than correctness.
fn draft_sampled() -> bool {
    matches!(std::env::var("COLI_DRAFT_SAMPLED").ok().as_deref(), Some("1") | Some("true"))
}

/// How deep this sequence may speculate.
///
/// **Zero for a sampled request unless `sampled` is set.** Accepting a draft
/// only where it matches the model's argmax makes speculation
/// *sequence*-identical to greedy decoding; at temperature > 0 the best
/// available guarantee is distribution-preservation, and silently changing which
/// tokens a sampled request emits is not a speedup, it is a different answer.
/// `sampled` (from `COLI_DRAFT_SAMPLED`) is the operator taking that trade
/// deliberately. A sequence with depth 0 contributes exactly one row, which is
/// the historical path — so drafting and non-drafting requests share a batch
/// with no special casing anywhere else.
///
/// Pure so the policy is testable without a model.
fn draft_depth_for(global: usize, has_mtp: bool, temp: f32, budget_left: usize, sampled: bool) -> usize {
    if global == 0 || !has_mtp || (temp > 0.0 && !sampled) {
        return 0;
    }
    // Never draft past what the request can still emit: a draft accepted beyond
    // `max_new` is work done to produce a token that is thrown away.
    global.min(budget_left.saturating_sub(1))
}

/// Build the two-branch hedge: `root` with the prompt-lookup run as one child
/// chain and the MTP chain as the other.
///
/// Returns the tree and its non-root tokens in DFS order — `SeqState::draft`
/// keeps holding those, so the row-assembly loop counts `1 + draft.len()` rows
/// exactly as it does for a chain and needs no branch of its own.
///
/// `None` when hedging would buy nothing: if the two sources agree on their
/// first token the branches are not alternatives, and a tree over them would
/// pay two rows at depth 1 to verify the same candidate twice. Falling back to
/// the longer chain is strictly better there.
fn hedge_tree(root: i32, ngram: &[i32], head: &[i32]) -> (Option<peregrine_model::CandidateTree>, Vec<i32>) {
    let (Some(&a0), Some(&b0)) = (ngram.first(), head.first()) else {
        return (None, if ngram.is_empty() { head.to_vec() } else { ngram.to_vec() });
    };
    if a0 == b0 {
        let longer = if ngram.len() >= head.len() { ngram } else { head };
        return (None, longer.to_vec());
    }
    // DFS: root, then all of branch A, then all of branch B. Parents precede
    // children, which is what lets the whole thing be ordinary ascending cache
    // slots.
    let mut tokens = Vec::with_capacity(1 + ngram.len() + head.len());
    let mut parent = Vec::with_capacity(tokens.capacity());
    tokens.push(root);
    parent.push(0);
    for (j, &t) in ngram.iter().enumerate() {
        tokens.push(t);
        parent.push(j); // 0 for the first (child of root), then the previous node
    }
    let b_base = 1 + ngram.len();
    for (j, &t) in head.iter().enumerate() {
        tokens.push(t);
        parent.push(if j == 0 { 0 } else { b_base + j - 1 });
    }
    let flat = tokens[1..].to_vec();
    match peregrine_model::CandidateTree::new(tokens, parent) {
        Ok(t) => (Some(t), flat),
        // A malformed tree is a bug in the two lines above, not a request
        // failure: fall back to the n-gram chain, which is what would have been
        // used with the knob off.
        Err(e) => {
            peregrine_core::note_advisory_err("hedge tree", &e);
            (None, ngram.to_vec())
        }
    }
}

/// Max draft depth that keeps a tick's *projected* routed-expert union under
/// `cap` entries, given `per_row` measured entries per row and `n_active`
/// decoding sequences. `usize::MAX` means "no opinion" — the ceiling is off, or
/// nothing has been measured yet.
///
/// Each sequence contributes `1 + g` rows, and the committed row is not
/// optional: a budget too small for even one row per sequence still yields
/// depth 0 rather than refusing to decode. A ceiling is allowed to stop
/// speculation and is never allowed to stop progress.
///
/// The projection is linear in rows and a real union is **sublinear** — that
/// sublinearity is the entire batching win — so this overestimates and cuts
/// sooner than strictly necessary. Conservative is the right direction for a
/// budget, and saying so here is cheaper than rediscovering it from a
/// disappointing sweep.
///
/// Pure so the policy is testable without a model, for the same reason
/// [`draft_depth_for`] and `prefill_chunk` are.
fn union_depth_cap(cap: u64, per_row: f64, n_active: usize) -> usize {
    // `is_finite` first so a NaN or infinite EWMA — which an empty or
    // pathological measurement window could produce — reads as "no opinion"
    // rather than as a cap of zero.
    if cap == 0 || n_active == 0 || !per_row.is_finite() || per_row <= 0.0 {
        return usize::MAX;
    }
    let rows_afforded = cap as f64 / per_row / n_active as f64;
    // `as usize` on a non-finite or huge float saturates rather than wrapping,
    // but clamp explicitly so the intent does not depend on that.
    let rows = if rows_afforded >= usize::MAX as f64 { usize::MAX } else { rows_afforded as usize };
    rows.max(1) - 1
}

/// Fuse a prefill chunk into the same forward as the decode batch
/// (`COLI_FUSE_PREFILL`). Default **on** (2026-08-13); `=0` restores the
/// historical two-forward tick.
///
/// On a tick with both, the unfused engine runs `forward_prefill_seq` *and*
/// `forward_step_batched`: two disjoint forwards, each streaming its own
/// routed-expert union off disk, ~11.3 GB per token apiece at GLM-5.2 shapes.
/// The MoE lane is row-batch-union'd and does not care which sequence a row
/// belongs to, so the two share one set of expert reads instead.
///
/// **Output-neutral, and proven so rather than argued**:
/// `a_fused_chunk_is_indistinguishable_from_two_separate_forwards` (model) and
/// `fused_prefill_emits_the_same_tokens_as_the_two_forward_tick` (here). The
/// byte win that justified flipping the default is a union-share measurement
/// (`COLI_UNION_STATS`); the serve-path A/B recipe is in
/// `docs/validation-runbook.md`.
fn fuse_prefill() -> bool {
    !matches!(std::env::var("COLI_FUSE_PREFILL").ok().as_deref(), Some("0") | Some("false"))
}

/// Ceiling on rows in one fused forward (`COLI_MAX_BATCH_ROWS`); `0`/unset =
/// uncapped, the historical behaviour. A fused tick's row count is
/// `Σ(1 + drafts)` over the decode batch plus the prefill chunk, and nothing
/// bounded the total: a geometric prefill chunk riding a full speculative
/// batch could assemble an arbitrarily large forward. The cap shrinks the
/// fused chunk first (always leaving one token of prefill progress) and
/// bounds next-tick draft depth so the decode block itself fits. Purely a
/// scheduling bound — which rows run when — never which tokens come out.
fn max_batch_rows() -> usize {
    std::env::var("COLI_MAX_BATCH_ROWS").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
}

/// Admission-queue depth cap (`COLI_QUEUE_DEPTH`); `0`/unset = unbounded, the
/// historical behaviour. When set, a submit against a backlog this deep is
/// refused ([`SubmitRefused::Full`] → HTTP 503) instead of queued forever —
/// overload becomes visible backpressure rather than unbounded memory and a
/// client timeout long after the fact.
fn queue_depth_cap() -> usize {
    std::env::var("COLI_QUEUE_DEPTH").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
}

/// Number of decode ticks per prefill tick when the adaptive window is on.
/// `COLI_ADAPTIVE_WINDOW=<n>` (default `1` = the historical every-tick prefill
/// interleave). Larger values let decode run further before yielding to prefill,
/// trading admission latency for decode throughput when the workload is decode-
/// heavy. Purely a scheduling knob — correctness-neutral.
fn adaptive_window_ratio() -> u64 {
    std::env::var("COLI_ADAPTIVE_WINDOW")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

/// Resident KV budget in bytes (`COLI_KV_BUDGET_MB`); 0 = off, the historical
/// count-only admission.
///
/// **Why this exists.** Admission is capped by a *count* (`--max-batch`,
/// default 32) and nothing ever reads bytes, so at GLM-5.2 shapes
/// (175.5 KiB/token of MLA KV) the default flags admit a worst case of ~53 GB
/// of KV with no accounting — `todo.md` §12 flags the unbounded case and tracks
/// it nowhere else. It is also the link that breaks the whole KV-saving chain:
/// halving KV bytes cannot raise concurrency while concurrency is a count, so
/// every downstream KV optimization is worth exactly zero extra batch slots
/// until this exists.
fn kv_budget_bytes() -> usize {
    std::env::var("COLI_KV_BUDGET_MB")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|mb| mb.saturating_mul(1 << 20))
        .unwrap_or(0)
}

/// Whether the engine may admit another sequence given the KV already resident.
///
/// Pure so the policy is testable without a model, following `ram.rs`'s
/// precedent (`project_load` / `ram_verdict`) of keeping sizing decisions out of
/// the machine they size.
///
/// This is a **high-water gate, not a predictive one**: it stops admitting once
/// resident KV crosses the budget, so the true peak overshoots by at most one
/// sequence's KV. Predicting the next request's cost would need its prompt
/// length, which is not known until after `try_recv` has already consumed it —
/// and dropping a consumed request is worse than a bounded overshoot.
///
/// `in_flight == 0` always admits. Without that, a single request whose KV
/// exceeds the budget would be refused forever and the engine would hang
/// instead of running it — the same "keep at least one" guard the warm cache's
/// eviction uses.
fn kv_admits(resident: usize, budget: usize, in_flight: usize) -> bool {
    budget == 0 || in_flight == 0 || resident < budget
}

/// Bytes of KV currently held by every in-flight sequence, decoding and
/// prefilling alike.
///
/// Each sequence's private tail, plus **one** charge per distinct shared prefix
/// however many sequences view it. That is what a refcounted prefix actually
/// costs: `SeqKv::clone_prefix` hands out `Arc` views of a single allocation, so
/// summing logical `bytes()` would charge a 2 k-token system prompt once per
/// concurrent request and choke admission on RAM that was never allocated. The
/// identity comes from the allocation itself, so two sequences sharing a prefix
/// at different depths still count it once.
///
/// A prefix whose cache entry has since been evicted is still charged here,
/// because a live sequence still holds it. What is *not* counted is a cache
/// entry no in-flight sequence is using — that lives under the separate
/// `COLI_PREFIX_CACHE_MB` budget, as it did before sharing existed.
fn resident_kv(active: &[SeqState], pending: &VecDeque<Prefilling>) -> usize {
    resident_kv_of(active.iter().map(|s| &s.seq).chain(pending.iter().map(|p| &p.seq)))
}

/// [`resident_kv`] over any set of sequences.
fn resident_kv_of<'a>(seqs: impl Iterator<Item = &'a SeqKv>) -> usize {
    dedup_kv_bytes(seqs.map(|s| (s.owned_bytes(), s.shared_prefix())))
}

/// Private bytes plus one charge per distinct shared allocation, over
/// `(owned_bytes, shared_prefix)` pairs.
///
/// Pure in the pairs rather than in `SeqKv` so the dedup — the part that can
/// actually be wrong — is testable without a loaded model to fill a cache with.
fn dedup_kv_bytes(seqs: impl Iterator<Item = (usize, Option<(usize, usize)>)>) -> usize {
    let mut total = 0usize;
    // A linear scan, not a set: an in-flight batch is tens of sequences over a
    // handful of distinct prefixes, so this never leaves L1.
    let mut seen: Vec<usize> = Vec::new();
    for (owned, shared) in seqs {
        total += owned;
        if let Some((id, bytes)) = shared {
            if !seen.contains(&id) {
                seen.push(id);
                total += bytes;
            }
        }
    }
    total
}

/// The admission-loop form of [`kv_admits`]. Ordered so that the default-off
/// path returns before walking every sequence's layer list — the sum is
/// re-evaluated on each admission attempt, so it must cost nothing when the
/// budget is unset.
fn kv_headroom(active: &[SeqState], pending: &VecDeque<Prefilling>, budget: usize) -> bool {
    if budget == 0 {
        return true;
    }
    let in_flight = active.len() + pending.len();
    in_flight == 0 || kv_admits(resident_kv(active, pending), budget, in_flight)
}

/// One in-flight sequence. Invariant at the top of a decode step: `seq.len() ==
/// pos`, and `next_tok` is an already-emitted token to be fed at `pos` next.
struct SeqState {
    seq: SeqKv,
    /// This stream's own routing history — the per-sequence prefetch predictor, so
    /// each concurrent stream prefetches from its own routing (not the cross-sequence
    /// union). Wrapped in a `Mutex` because the batched forward records into it.
    hist: Mutex<RouteHistory>,
    /// Stable prefetch-lane key — see [`NEXT_SEQ_ID`]. Not this sequence's index in
    /// `active`, which slides when an earlier sequence retires.
    seq_id: usize,
    pos: usize,
    next_tok: i32,
    sampler: Sampler,
    out: mpsc::UnboundedSender<EngineOut>,
    produced: usize,
    max_new: usize,
    /// Tokens this sequence speculated for the *next* forward, drafted from
    /// [`Self::hlast`] by the MTP head. Empty when speculation is off, when the
    /// checkpoint has no MTP head, or when this request samples at
    /// temperature > 0 without `COLI_DRAFT_SAMPLED` — see [`draft_depth_for`].
    draft: Vec<i32>,
    /// The distribution each entry of [`Self::draft`] was drawn from, under
    /// `COLI_DRAFT_SAMPLED`; empty on the greedy path, which needs no `q`.
    ///
    /// Carried beside `draft` rather than derived at verify time because
    /// [`peregrine_model::accept_run_sampled`]'s correctness rests on `q` being
    /// the distribution the draft *was* sampled from — re-deriving it from the
    /// draft head at verify time would recompute a forward, and recomputing it
    /// from anything else would be a different distribution wearing its name.
    draft_q: Vec<Vec<f32>>,
    /// Pre-final-norm hidden at this sequence's last committed position: what
    /// the next draft continues from. Empty until the first verify produces it.
    hlast: Vec<f32>,
    /// Whether this tick's `draft` came from prompt-lookup rather than the MTP
    /// head. Only for accounting — the accept rule is the same either way.
    draft_from_ngram: bool,
    /// The candidate **tree** this tick's rows form, when `COLI_DRAFT_TREE` is
    /// on and both draft sources produced something.
    ///
    /// `draft` still holds the tree's non-root tokens in DFS order, so the row
    /// count is `1 + draft.len()` exactly as for a chain and the assembly loop
    /// needs no special case. What the tree adds is the *shape*: which node each
    /// row's parent is, which decides the row's RoPE depth, its key set, and
    /// which subset of slots survives acceptance.
    tree: Option<peregrine_model::CandidateTree>,
    /// The fed-token log, row-aligned with `seq`: `toks[i]` is the token whose
    /// feed produced KV row `i` (the prompt, then each committed decode/draft
    /// token). Kept so a retiring sequence can hand `prompt + output` to the
    /// prefix cache — a multi-turn client resends exactly that as the next
    /// prompt's head, and without this entry it re-prefills the assistant turn
    /// it just received. Invariant: `toks.len() == pos == seq.len()` at the
    /// top of a decode step.
    toks: Vec<i32>,
}

/// The engine loop: admit + prefill new requests, then decode all active
/// sequences one batched step at a time until each hits a stop id, its token
/// budget, or a dropped client.
fn run(
    mut model: Model,
    mut rx_normal: mpsc::UnboundedReceiver<Queued>,
    mut rx_high: mpsc::UnboundedReceiver<Queued>,
    max_batch: usize,
    knobs: EngineKnobs,
    shared: EngineShared,
) {
    let EngineShared { telemetry, queued } = shared;
    let fuse = knobs.fuse_prefill;
    let vocab = model.cfg.vocab as usize;
    let stop_ids = model.cfg.stop_ids.clone();
    let mut active: Vec<SeqState> = Vec::new();
    let mut pending: VecDeque<Prefilling> = VecDeque::new();
    let mut steps = 0usize;
    // MTP acceptance accounting (see `EngineTelemetry::spec_proposed`).
    let mut spec_proposed: u64 = 0;
    let mut spec_accepted: u64 = 0;
    let mut spec_conf_stops: u64 = 0;
    // Recurrent-rollback accounting (see `EngineTelemetry::gdn_snapshot_bytes`).
    // These three are the whole cost side of `COLI_SPEC_GDN`: what the copies
    // cost, how often the cheap path missed, and how many rows the miss redid.
    let mut gdn_snapshot_bytes: u64 = 0;
    let mut gdn_replays: u64 = 0;
    let mut gdn_replay_rows: u64 = 0;
    // Prompt-lookup accounting, split out from `spec_*` because the two draft
    // sources have completely different costs: an n-gram draft is a memcmp, an
    // MTP draft is a sparse-MoE layer. A single accept rate over both would
    // average a free source with an expensive one and mean nothing.
    let mut ngram_proposed: u64 = 0;
    let mut ngram_accepted: u64 = 0;
    // The decode-economics numerators — see `EngineTelemetry::tokens_emitted`.
    let mut tokens_emitted: u64 = 0;
    let mut decode_rows: u64 = 0;
    // Admission latency — the span between `submit` and the request becoming a
    // `Prefilling`. Sum/count/max rather than a histogram: percentiles for the
    // *serving* side already live in `bench-serve-gaps.py`, and this is the
    // queueing side, where the question is "is anything waiting at all" before
    // it is "how long is the tail".
    let mut queue_wait_us: u64 = 0;
    let mut queue_admits: u64 = 0;
    let mut queue_wait_max_us: u64 = 0;
    // Union-cost accounting for `COLI_SPEC_UNION_MAX`. `entries` is the warm
    // tier's hit+miss count, i.e. exactly the routed-expert union entries a
    // forward resolved; the EWMA is per row, so a draft depth can be priced
    // before it is proposed.
    let mut union_entries_last: u64 = 0;
    let mut union_per_row_ewma: f64 = 0.0;
    let mut spec_union_stops: u64 = 0;
    let mut spec_trees: u64 = 0;
    let mut spec_tree_branch_wins: u64 = 0;
    let note_wait = |w: u64, total: &mut u64, n: &mut u64, max: &mut u64| {
        *total = total.saturating_add(w);
        *n += 1;
        *max = (*max).max(w);
    };
    // Package energy, accumulated so `Δenergy / Δtokens` is joules per token.
    // Sampled on the same period the sensor governors use, for the same reason
    // — one sysfs read per forward is overhead nobody asked for — and totals
    // stay exact regardless, because `delta_uj` reports everything since the
    // *last call*, not since the last tick.
    let mut energy = peregrine_io::EnergyMeter::new();
    let mut energy_uj: Option<u64> = None;
    // Adaptive-batching state. `working_cap` is the current admission ceiling
    // (starts at `max_batch`, shrinks under SLA overrun, grows on slack). EWMA
    // over per-forward wall time drives the adjustment.
    let sla_ms = knobs.batch_sla_ms;
    // Prefill chunking is a latency/work trade, not a per-tick decision.
    let chunk_div = knobs.prefill_chunk_div;
    let depth = knobs.draft_depth;
    let sampled_spec = knobs.draft_sampled;
    let conf_floor = knobs.spec_conf;
    // Speculation additionally requires that rejecting a draft can be undone.
    // On a KV-only arch that is `truncate` and nothing else. A recurrent arch's
    // verify forward advances its `GdnState` in place and truncation cannot
    // reach it, so it needs the snapshot/restore rollback — which is wired
    // below, behind `COLI_SPEC_GDN` because the snapshot costs ~151 MB per
    // drafting sequence per tick at 27B dims and that price has to be measured
    // per batch width, not assumed.
    let ngram = peregrine_model::NgramDrafter::new(knobs.draft_ngram);
    let gdn_rollback = !model.spec_reject_is_kv_only() && knobs.spec_gdn;
    let has_mtp = model.has_mtp() && (model.spec_reject_is_kv_only() || gdn_rollback);
    if model.has_mtp() && !model.spec_reject_is_kv_only() && !gdn_rollback {
        eprintln!(
            "peregrine: [spec] MTP head present but speculation is off — this arch rewinds \
             a recurrent state, which costs a per-tick snapshot; set COLI_SPEC_GDN=1 to enable it"
        );
    }
    // A GDN sequence's context is a point state, not per-position rows: a
    // prefix hit or a disk checkpoint would need a state snapshot taken exactly
    // at the boundary. Until that trade is measured, hybrid models skip both
    // (Track C phase 2a) — budget 0 is the prefix cache's documented off state.
    let cacheable = model.prefix_cachable();
    let mut prefix = PrefixStore::new(
        if cacheable { knobs.prefix_cache_budget } else { 0 },
        if cacheable {
            crate::kvstore::KvSessionStore::from_env(&model, PREFIX_CACHE_MIN_TOKENS)
        } else {
            None
        },
    );
    // The KV byte ceiling admission respects alongside the count.
    let kv_budget = knobs.kv_budget_bytes;
    // The fused-forward row ceiling (0 = uncapped).
    let max_rows = knobs.max_batch_rows;
    // Decode ticks per prefill tick (COLI_ADAPTIVE_WINDOW). Hoisted with the
    // rest: it used to be re-read every tick through its process-wide latch,
    // which made it look per-tick-tunable when it never was.
    let win = knobs.adaptive_window_ratio;
    let mut working_cap = max_batch;
    let mut ewma_decode_us: u64 = 0;
    // Small current-thread runtime just for the priority-aware blocking recv.
    // Owned by this thread — it never crosses `spawn` boundaries. Without it
    // the loop degrades to the spin-recv path below, so build failure is
    // advisory, not fatal.
    let idle_rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => Some(rt),
        Err(e) => {
            peregrine_core::note_advisory_err("batch: idle-recv runtime build", &e);
            None
        }
    };

    loop {
        // Drain queued requests into the prefill queue: HIGH first (so they
        // beat normal admissions to the batch), NORMAL second — both capped by
        // the *working* cap so an SLA-shrunk engine backpressures both queues,
        // and by the KV byte budget so a long-context workload backpressures on
        // memory rather than only on sequence count. Re-evaluated each iteration
        // because every admission grows the resident set.
        while active.len() + pending.len() < working_cap && kv_headroom(&active, &pending, kv_budget) {
            match rx_high.try_recv() {
                Ok(q) => note_wait(
                    admit_pending(&model, &mut pending, q, &mut prefix, &queued),
                    &mut queue_wait_us,
                    &mut queue_admits,
                    &mut queue_wait_max_us,
                ),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        while active.len() + pending.len() < working_cap && kv_headroom(&active, &pending, kv_budget) {
            match rx_normal.try_recv() {
                Ok(q) => note_wait(
                    admit_pending(&model, &mut pending, q, &mut prefix, &queued),
                    &mut queue_wait_us,
                    &mut queue_admits,
                    &mut queue_wait_max_us,
                ),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break, // drain, then exit
            }
        }
        // Nothing in flight → idle. Run one bounded maintenance step (background
        // warm-cache recompression, COLI_CACHE_COMPRESS_IDLE) and keep going while
        // it makes progress AND no request has arrived — then block for the next
        // request (high preferred). Exit when both senders are gone.
        if active.is_empty() && pending.is_empty() {
            while model.idle_maintenance() > 0 {
                // A request arriving interrupts the sweep immediately. An empty
                // or disconnected queue just continues the sweep — disconnection
                // is handled as shutdown by `recv_priority` below.
                match rx_high.try_recv() {
                    Ok(q) => {
                        let w = admit_pending(&model, &mut pending, q, &mut prefix, &queued);
                        note_wait(w, &mut queue_wait_us, &mut queue_admits, &mut queue_wait_max_us);
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
                }
                match rx_normal.try_recv() {
                    Ok(q) => {
                        let w = admit_pending(&model, &mut pending, q, &mut prefix, &queued);
                        note_wait(w, &mut queue_wait_us, &mut queue_admits, &mut queue_wait_max_us);
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
                }
            }
            if !pending.is_empty() {
                continue;
            }
            match recv_priority(idle_rt.as_ref(), &mut rx_high, &mut rx_normal) {
                Some(q) => {
                    let w = admit_pending(&model, &mut pending, q, &mut prefix, &queued);
                    note_wait(w, &mut queue_wait_us, &mut queue_admits, &mut queue_wait_max_us);
                    continue;
                }
                None => break,
            }
        }

        // Advance one prefill chunk (a finished prefill joins `active`), interleaved
        // with decode so a long admission never stalls the batch for its whole prefill.
        // Adaptive window: when `COLI_ADAPTIVE_WINDOW=N > 1`, only run prefill every
        // Nth engine tick so decode gets more consecutive time before yielding.
        // If no decodes are active yet, always run prefill (else the engine stalls
        // waiting for a prefill that never fires).
        let do_prefill = win == 1 || active.is_empty() || steps.is_multiple_of(win as usize);
        // Fusion needs something to fuse *with*: a live decode batch and a
        // pending prefill. Otherwise there is only one forward to run anyway, so
        // take the historical path and leave the row assembly out of it.
        let mut fused: Option<(Prefilling, usize)> = None;
        if do_prefill {
            let fusable = fuse && !active.is_empty() && !pending.is_empty();
            match fusable.then(|| pending.pop_front()).flatten() {
                Some(p) => {
                    let mut end = chunk_end(p.pos, p.prompt.len(), chunk_div);
                    // COLI_MAX_BATCH_ROWS: the fused chunk yields first, down
                    // to one token so the prefill always makes progress.
                    if max_rows > 0 {
                        let dec_rows: usize = active.iter().map(|s| 1 + s.draft.len()).sum();
                        let room = max_rows.saturating_sub(dec_rows).max(1);
                        end = end.min(p.pos + room);
                    }
                    fused = Some((p, end));
                }
                None => {
                    // A promoted sequence emits its first token here, and it
                    // counts: the ratio this feeds is per *served* token.
                    tokens_emitted +=
                        prefill_step(&model, &mut pending, &mut active, vocab, &stop_ids, chunk_div, &mut prefix)
                            as u64;
                }
            }
        }
        if active.is_empty() {
            continue; // only prefilling so far (or everything retired) — no decode yet
        }

        // One forward for the whole tick: each active sequence's pending decode
        // token, plus — when fusing — every position of one prefill chunk. The
        // MoE lane unions routed experts across *rows*, so those rows share one
        // set of expert reads instead of streaming two disjoint unions.
        let n_dec = active.len();
        // Each active sequence contributes its pending token plus whatever it
        // speculated last tick — `1 + g_s` rows, and `g_s` differs per sequence
        // because a sampled request drafts nothing and a nearly-finished one
        // drafts less. `first_row[s]` is where sequence `s`'s block starts.
        let mut tokens: Vec<i32> = Vec::with_capacity(n_dec);
        let mut pos_of: Vec<usize> = Vec::with_capacity(n_dec);
        let mut owner: Vec<usize> = Vec::with_capacity(n_dec);
        let mut first_row: Vec<usize> = Vec::with_capacity(n_dec);
        let mut rows_of: Vec<usize> = Vec::with_capacity(n_dec);
        // Tree rows need two extra vectors: a RoPE position that is the node's
        // *depth* rather than its cache slot, and an explicit key set naming its
        // ancestors. Both must cover every row, because the attention cores
        // index them positionally — but an entry of `None` means dense, so
        // ordinary chains and the fused prefill chunk stay on the untouched
        // loops even when one sequence is speculating on a tree.
        let mut rope_of: Vec<usize> = Vec::with_capacity(n_dec);
        let mut sel: Vec<Option<Vec<usize>>> = Vec::with_capacity(n_dec);
        let mut any_tree = false;
        for (i, st) in active.iter().enumerate() {
            first_row.push(tokens.len());
            rows_of.push(1 + st.draft.len());
            tokens.push(st.next_tok);
            pos_of.push(st.pos);
            owner.push(i);
            match &st.tree {
                Some(t) => {
                    any_tree = true;
                    let rp = t.rope_positions(st.pos);
                    let ks = t.key_sets(st.pos);
                    rope_of.push(rp.first().copied().unwrap_or(st.pos));
                    sel.push(ks.first().cloned());
                    for (j, &tok) in st.draft.iter().enumerate() {
                        tokens.push(tok);
                        // The *slot* stays consecutive — `LayerKv::append`
                        // requires it — while the RoPE position is the depth.
                        pos_of.push(st.pos + 1 + j);
                        owner.push(i);
                        rope_of.push(rp.get(j + 1).copied().unwrap_or(st.pos + 1 + j));
                        sel.push(ks.get(j + 1).cloned());
                    }
                }
                None => {
                    rope_of.push(st.pos);
                    sel.push(None);
                    for (j, &t) in st.draft.iter().enumerate() {
                        tokens.push(t);
                        pos_of.push(st.pos + 1 + j);
                        owner.push(i);
                        rope_of.push(st.pos + 1 + j);
                        sel.push(None);
                    }
                }
            }
        }
        let n_dec_rows = tokens.len();
        // Decode rows only: a fused prefill chunk's rows are appended after
        // this and are not what speculation's overhead is measured against.
        decode_rows += n_dec_rows as u64;
        if let Some((p, end)) = &fused {
            for (j, &t) in p.prompt[p.pos..*end].iter().enumerate() {
                tokens.push(t);
                pos_of.push(p.pos + j);
                owner.push(n_dec); // the prefilling sequence is appended after the decoders (owners are per *sequence*, not per row)
                // A prefill chunk is ordinary causal rows: slot == position, and
                // dense attention over the whole prefix.
                rope_of.push(p.pos + j);
                sel.push(None);
            }
        }
        // A recurrent arch cannot rewind by truncation. This forward is about
        // to advance every drafting sequence's `GdnState` by `1 + g` tokens,
        // and only the accepted prefix of that is real, so the pre-forward
        // context is saved first. On full acceptance the snapshot is dropped
        // untouched; otherwise it is restored and the committed rows are
        // re-advanced below. Taken for any sequence carrying a draft — never
        // gated on the batch width, because a cap that changed between the
        // draft and its verify would silently skip a snapshot that was needed.
        let gdn_snaps: Vec<Option<Vec<(usize, peregrine_model::gdn::GdnSnapshot)>>> = if gdn_rollback {
            active.iter().map(|s| if s.draft.is_empty() { None } else { s.seq.gdn_snapshot() }).collect()
        } else {
            Vec::new()
        };
        for snaps in gdn_snaps.iter().flatten() {
            gdn_snapshot_bytes += snaps.iter().map(|(_, sn)| sn.bytes() as u64).sum::<u64>();
        }
        // Dropped at the end of the tick: speculated rows record here so a
        // rejected draft never reaches the prefetch predictor.
        let scratch_hist = Mutex::new(model.new_route_history());
        let t_decode = std::time::Instant::now();
        let (logits, hidden) = {
            // Split each SeqState into (KV, history) disjoint borrows so the batched
            // forward records each sequence's *own* routed set into its own history.
            let (mut refs, per_seq): (Vec<&mut SeqKv>, Vec<&Mutex<RouteHistory>>) = active
                .iter_mut()
                .map(|s| {
                    let SeqState { seq, hist, .. } = s;
                    (seq, &*hist)
                })
                .unzip();
            // Histories are per **row**, so a sequence with `g` drafts needs
            // `1 + g` entries. Only its first row — the already-confirmed token
            // — records into the sequence's own history; the speculated rows
            // record into a scratch that is dropped at the end of the tick.
            //
            // A rejected draft's routing is a plausible-but-wrong future, and
            // feeding it to the prefetch predictor would have it warm experts
            // for a token that never existed. `generate_speculative` refuses
            // the same thing for the same reason (`route_log: None` on drafts).
            let mut hists: Vec<&Mutex<RouteHistory>> = Vec::with_capacity(n_dec_rows);
            for (i, h) in per_seq.iter().enumerate() {
                hists.push(h);
                for _ in 1..rows_of.get(i).copied().unwrap_or(1) {
                    hists.push(&scratch_hist);
                }
            }
            // `hists` is per *row*, so a chunk contributes one entry per position —
            // all pointing at the prefilling sequence's own history, which is why
            // `Prefilling` carries one from admission rather than getting one at
            // promotion.
            if let Some((p, end)) = fused.as_mut() {
                let rows = *end - p.pos;
                let Prefilling { seq, hist, .. } = p;
                refs.push(seq);
                for _ in 0..rows {
                    hists.push(&*hist);
                }
            }
            let out = if any_tree {
                let rows = peregrine_model::TreeRows { rope_pos: &rope_of, sel: &sel };
                model.forward_tree_rows_hidden(&tokens, &owner, &mut refs, &pos_of, rows, Some(&hists))
            } else {
                model.forward_rows_batched_hidden(&tokens, &owner, &mut refs, &pos_of, Some(&hists))
            };
            match out {
                Ok(l) => l,
                Err(e) => {
                    for s in &active {
                        if s.out.send(EngineOut::Error(e.to_string())).is_err() {
                            peregrine_core::note_advisory_err("batch error forward", &"client already disconnected");
                        }
                    }
                    active.clear();
                    if let Some((p, _)) = fused {
                        if p.out.send(EngineOut::Error(e.to_string())).is_err() {
                            peregrine_core::note_advisory_err("fused prefill error", &"client already disconnected");
                        }
                    }
                    continue;
                }
            }
        };
        // What this tick's forward actually cost in union entries, per row.
        // Folded before the draft step below, so the depth it prices is priced
        // against the most recent evidence rather than last tick's.
        if let Some((h, m, _)) = model.ecache_stats() {
            let now = h.saturating_add(m);
            let delta = now.saturating_sub(union_entries_last);
            union_entries_last = now;
            let rows = tokens.len().max(1) as f64;
            if delta > 0 {
                let per_row = delta as f64 / rows;
                // α = 0.3, the same smoothing the decode-time EWMA uses.
                union_per_row_ewma =
                    if union_per_row_ewma == 0.0 { per_row } else { union_per_row_ewma * 0.7 + per_row * 0.3 };
            }
        }
        let decode_us = t_decode.elapsed().as_micros() as u64;
        // EWMA of decode wall time (α = 0.3). Feeds the SLA-driven cap adjustment.
        ewma_decode_us = if ewma_decode_us == 0 {
            decode_us
        } else {
            (ewma_decode_us as u128 * 7 / 10 + decode_us as u128 * 3 / 10) as u64
        };
        if let Some(sla_ms) = sla_ms {
            let sla_us = sla_ms * 1000;
            if ewma_decode_us > sla_us && working_cap > 1 {
                working_cap -= 1; // shrink one at a time — smooth backpressure
            } else if ewma_decode_us * 2 < sla_us && working_cap < max_batch {
                working_cap += 1; // grow slowly when there's slack
            }
        }
        // Per-sequence, parallel-async prefetch: warm each stream's predicted next
        // experts onto its assigned lane while sampling proceeds. Keyed on the
        // sequence's own id, not its index here — see [`NEXT_SEQ_ID`].
        for s in active.iter() {
            model.enqueue_seq_prefetch(&s.hist, s.seq_id);
        }

        // Emit each sequence's confirmed run and decide who continues.
        //
        // Without speculation every sequence's block is one row and this is the
        // historical sample-one-token loop. With drafts, `accept_run` says how
        // many of them the model's own argmax agrees with, and the whole
        // accepted run is emitted — which is exactly what greedy decoding would
        // have produced, one forward at a time.
        let d_hidden = model.cfg.hidden as usize;
        let mut keep: Vec<bool> = Vec::with_capacity(active.len());
        // `(active index, first committed position, one past the last)` for the
        // sequences whose recurrent state overshot and has to be re-advanced.
        let mut gdn_replay: Vec<(usize, usize, usize)> = Vec::new();
        // A restore that fails leaves the state part-rewound, which is a wiring
        // bug and not something to serve through: those sequences are retired
        // after the loop, where `keep` can still be corrected.
        let mut gdn_failed: Vec<(usize, String)> = Vec::new();
        for (i, s) in active.iter_mut().enumerate() {
            let base = first_row.get(i).copied().unwrap_or(i);
            let g = s.draft.len();
            // The position this sequence's block starts at — where the rewind
            // goes back to, captured before `s.pos` advances below.
            let pos0 = s.pos;
            // A greedy drafting sequence takes `accept_run`, whose argmax rule
            // and `sampler.pick` agree by construction. A sampled one (only
            // reachable under `COLI_DRAFT_SAMPLED`) takes the rejection-sampling
            // rule, which needs the `q` each draft was drawn from. A
            // non-drafting sequence keeps its own sampler, temperature and all.
            // The accepted node path, for a tree — its entries are indices into
            // the tree's DFS order, which are *not* `1..=k` when a branch was
            // taken. Empty for a chain, where the path is implicit.
            let mut path: Vec<usize> = Vec::new();
            let (k, spec_next) = if let Some(t) = s.tree.as_ref().filter(|_| g > 0) {
                let rows = logits.get(base * vocab..(base + g + 1) * vocab).unwrap_or(&[]);
                let (p, next) = peregrine_model::accept_tree(rows, vocab, t);
                path = p;
                (path.len(), next)
            } else if g > 0 {
                let rows = logits.get(base * vocab..(base + g + 1) * vocab).unwrap_or(&[]);
                if s.sampler.temp > 0.0 {
                    // `take`, not borrow: a `q` must never be scored against a
                    // different forward's rows, so a tick that somehow skipped
                    // drafting finds none rather than last tick's.
                    let q = std::mem::take(&mut s.draft_q);
                    peregrine_model::accept_run_sampled(rows, vocab, &s.draft, &q, &mut s.sampler)
                } else {
                    peregrine_model::accept_run(rows, vocab, &s.draft)
                }
            } else {
                (0, 0)
            };
            if g > 0 {
                // Same accept rule, separate ledgers — see `ngram_proposed`.
                if s.draft_from_ngram {
                    ngram_accepted += k.min(g) as u64;
                } else {
                    spec_proposed += g as u64;
                    spec_accepted += k.min(g) as u64;
                }
            }
            // **`s.next_tok` was already emitted** — by the prefill that
            // promoted this sequence, or by the previous round. It is the token
            // being *fed* now, not one to send. What this round emits is
            // whatever comes after it: every accepted draft, then the model's
            // prediction past them.
            //
            // With no drafts that is a single token from row 0, which is the
            // historical decode step exactly.
            //
            // RLM composition (COLI_RLM, structural no-op when unset): refine
            // the *decision* row before it becomes `final_next`, mirroring the
            // model-resident composition in `generate_speculative` — accept
            // decisions always use the raw logits; only the post-acceptance
            // contested row (or the sole row of a non-drafting step) refines;
            // and a sampled speculative run is skipped, because refining after
            // `accept_run_sampled`'s residual draw would break its
            // distribution-preserving guarantee.
            let mut refined_hidden: Option<Vec<f32>> = None;
            let final_next = if g > 0 {
                // Already chosen by whichever accept rule ran: `accept_run`'s
                // argmax for a greedy request, `accept_run_sampled`'s residual
                // or bonus draw for a sampled one. Re-picking here would take a
                // second sample from the same row and emit a token the accept
                // rule never verified against. (An RLM pass on the contested
                // row recomputes the argmax from *refined* logits — a different
                // footing, not a second sample from the same one.)
                if s.sampler.temp <= 0.0 {
                    let rows = ForwardRows { logits: &logits, hidden: &hidden, vocab, d_hidden };
                    let decision_row = match s.tree.as_ref() {
                        Some(_) => base + path.last().copied().unwrap_or(0),
                        None => base + k,
                    };
                    match rlm_refine_row(&model, &s.seq, s.pos + k, &rows, decision_row, 0.0) {
                        Some((lg, h)) => {
                            refined_hidden = Some(h);
                            peregrine_model::argmax(&lg) as i32
                        }
                        None => spec_next,
                    }
                } else {
                    spec_next
                }
            } else {
                let lo = base * vocab;
                match logits.get(lo..lo + vocab) {
                    Some(r) => {
                        let rows = ForwardRows { logits: &logits, hidden: &hidden, vocab, d_hidden };
                        match rlm_refine_row(&model, &s.seq, s.pos, &rows, base, s.sampler.temp) {
                            Some((lg, h)) => {
                                refined_hidden = Some(h);
                                s.sampler.pick(&lg, -1) as i32
                            }
                            None => s.sampler.pick(r, -1) as i32,
                        }
                    }
                    None => s.next_tok,
                }
            };
            // The hedge paid when the accepted path is not branch A's prefix —
            // branch A occupies nodes `1..=len(A)` in DFS order, so any accepted
            // node beyond that is the head's chain winning a row prompt-lookup
            // would otherwise have committed alone.
            if s.tree.is_some() && path.iter().enumerate().any(|(d, &nd)| nd != d + 1) {
                spec_tree_branch_wins += 1;
            }
            let mut run: Vec<i32> = Vec::with_capacity(k + 1);
            match s.tree.as_ref() {
                // A tree's accepted tokens are the nodes along the path, which
                // is not a prefix of `draft` once a branch was taken.
                Some(t) => run.extend(path.iter().filter_map(|&nd| t.tokens().get(nd).copied())),
                None => run.extend_from_slice(&s.draft[..k.min(g)]),
            }
            run.push(final_next);

            let mut alive = true;
            let mut drafts_emitted = 0usize;
            for (j, &tok) in run.iter().enumerate() {
                if stop_ids.contains(&tok) {
                    alive = false; // a stop token is not emitted, and ends the run
                    break;
                }
                if s.out.send(EngineOut::Token(tok as u32)).is_err() {
                    alive = false;
                    break;
                }
                s.produced += 1;
                tokens_emitted += 1;
                if j < k {
                    drafts_emitted += 1; // the final entry is not a cached row
                }
                if s.produced >= s.max_new {
                    alive = false;
                    break;
                }
            }
            // Commit the fed token plus every draft that was actually emitted.
            // A run cut short by a stop token or the budget must not leave its
            // speculated tail cached — the next round would append after rows
            // the client never received.
            s.pos += 1 + drafts_emitted;
            // The KV half of the rewind — and, on a recurrent arch that did not
            // accept its whole draft, only the second half of it. `GdnState`'s
            // protocol is restore-then-truncate-then-re-advance in that order,
            // so the state goes back to the pre-forward snapshot, the KV goes
            // back to the same boundary, and the replay forward below carries
            // both to `s.pos` over exactly the rows the client was sent.
            let overshot = alive && g > 0 && drafts_emitted < g;
            match gdn_snaps.get(i).and_then(Option::as_ref).filter(|_| overshot) {
                Some(snap) => match s.seq.gdn_restore(snap) {
                    Ok(()) => {
                        s.seq.truncate(pos0);
                        gdn_replay.push((i, pos0, s.pos));
                    }
                    Err(e) => {
                        s.seq.truncate(s.pos);
                        gdn_failed.push((i, e.to_string()));
                    }
                },
                // A **tree** cannot rewind by truncation either, for a different
                // reason than a recurrent state: its accepted path is a
                // non-contiguous subset of the block's slots, with the rejected
                // siblings interleaved. The kept rows are gathered down instead,
                // which is a pure move — each was roped at its tree depth, which
                // is exactly the position it lands on. Trees are MLA-only, so
                // this and the recurrent arm can never both apply.
                None if s.tree.is_some() && g > 0 => {
                    let keep: Vec<usize> = std::iter::once(pos0)
                        .chain(path.iter().take(drafts_emitted).map(|&nd| pos0 + nd))
                        .collect();
                    if let Err(e) = s.seq.retain_tail(pos0, &keep) {
                        // The cache is now neither shape. Retire rather than
                        // serve on a history nothing else can detect is wrong.
                        peregrine_core::note_advisory_err("tree commit", &e);
                        alive = false;
                    }
                }
                None => s.seq.truncate(s.pos),
            }
            // The fed-token log commits the same rows: the token fed this tick
            // (the *old* `next_tok`, so this must precede the overwrite below)
            // plus the accepted drafts, keeping `toks` row-aligned with `seq`.
            s.toks.push(s.next_tok);
            s.toks.extend_from_slice(&s.draft[..drafts_emitted]);
            // `final_next` has been emitted but not yet fed, which is exactly
            // the pending-token invariant. Only meaningful when the sequence
            // survives; a retiring one is dropped below.
            s.next_tok = final_next;
            // Carry the hidden at the row that produced `final_next`, so the
            // next draft continues from this forward rather than re-running
            // the stack.
            // For a tree the decision row is the last accepted *node*, whose
            // offset within the block is its DFS index — not `drafts_emitted`,
            // which is a count and not a slot once a branch was taken.
            let last_node = match s.tree.as_ref() {
                Some(_) => path.get(drafts_emitted.wrapping_sub(1)).copied().unwrap_or(0),
                None => drafts_emitted,
            };
            let hrow = (base + last_node) * d_hidden;
            // An absent row means the forward returned less than it was asked
            // for — clear the hidden rather than default it, so the next tick
            // skips drafting instead of drafting from zeros.
            //
            // A refined hidden replaces the raw row (the next MTP draft then
            // continues from the refined footing, as the model composition
            // does), but only when the emitted run wasn't cut short — the
            // refinement sits at the decision row, which is `hrow` exactly
            // when `drafts_emitted == k` (a cut-short run retires below, so
            // nothing downstream reads a mismatched hidden either way).
            s.hlast = match refined_hidden {
                Some(h) if drafts_emitted == k.min(g) => h,
                _ => match hidden.get(hrow..hrow + d_hidden) {
                    Some(h) => h.to_vec(),
                    None => Vec::new(),
                },
            };
            // A retiring sequence's KV is about to drop; freeze prompt+output
            // first. This is what turns a multi-turn conversation's next
            // request — the same ids plus a new user turn — into a refcount
            // bump instead of a full re-prefill of the assistant turn. The
            // 64-token floor and the LRU budget in `insert` apply unchanged,
            // and exact-id matching means a client whose retokenization
            // differs simply degrades to the prompt-prefix match it gets today.
            if !alive {
                prefix.insert(&s.toks[..s.pos.min(s.toks.len())], &s.seq);
            }
            keep.push(alive);
        }
        // A restore that failed left a part-rewound recurrent state. Nothing
        // downstream can tell, which is exactly why the request ends here
        // rather than continuing on a context that is quietly wrong.
        for (i, e) in &gdn_failed {
            if let (Some(k), Some(st)) = (keep.get_mut(*i), active.get(*i)) {
                *k = false;
                if st.out.send(EngineOut::Error(format!("recurrent state rollback failed: {e}"))).is_err() {
                    peregrine_core::note_advisory_err("gdn restore", &"client already disconnected");
                }
            }
        }

        // Re-advance every overshooting sequence's recurrent state over the
        // rows it actually committed. One forward for all of them, not one
        // each: on a resident model the cost of a forward is the weight read,
        // which these rows share exactly as the decode batch does.
        //
        // The logits are discarded — this pass exists for its side effect on
        // `GdnState` and the KV. `s.hlast` keeps the value the verify forward
        // produced, which is the same value this pass recomputes: same tokens,
        // same positions, same restored state, and rows of different sequences
        // do not see each other.
        if !gdn_replay.is_empty() {
            let mut rtok: Vec<i32> = Vec::new();
            let mut rpos: Vec<usize> = Vec::new();
            let mut rowner: Vec<usize> = Vec::new();
            let mut rrefs: Vec<&mut SeqKv> = Vec::new();
            for (i, st) in active.iter_mut().enumerate() {
                let Some(&(_, pos0, pos_end)) = gdn_replay.iter().find(|&&(r, _, _)| r == i) else {
                    continue;
                };
                let slot = rrefs.len();
                for p in pos0..pos_end {
                    // `toks` is row-aligned with the cache by construction, so
                    // a miss here is a bookkeeping bug; skip the row rather
                    // than index-panic an engine thread that runs under
                    // `panic = "abort"` and would take every sequence with it.
                    let Some(&t) = st.toks.get(p) else { continue };
                    rtok.push(t);
                    rpos.push(p);
                    rowner.push(slot);
                }
                rrefs.push(&mut st.seq);
            }
            gdn_replays += 1;
            gdn_replay_rows += rtok.len() as u64;
            if let Err(e) = model.forward_rows_batched(&rtok, &rowner, &mut rrefs, &rpos, None) {
                // The states are now neither pre- nor post-forward. Same
                // reasoning as a failed restore: retire rather than serve on.
                for &(i, _, _) in &gdn_replay {
                    if let (Some(k), Some(st)) = (keep.get_mut(i), active.get(i)) {
                        *k = false;
                        if st.out.send(EngineOut::Error(e.to_string())).is_err() {
                            peregrine_core::note_advisory_err("gdn replay", &"client already disconnected");
                        }
                    }
                }
            }
        }

        let mut idx = 0usize;
        active.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });

        // Draft for the next tick. Done after retirement so a finished sequence
        // is never drafted for, and from the hidden this forward produced —
        // `mtp_draft` takes `&self`, so the drafts do not serialise on a borrow.
        //
        // A draft failure is not a request failure: speculation is a
        // wall-clock optimisation, so an error here drops back to plain decode
        // for that sequence rather than dropping the client.
        if depth > 0 {
            let sampled = sampled_spec;
            // COLI_MAX_BATCH_ROWS bounds next tick's decode block too: B
            // sequences each drafting g assemble B*(1+g) rows before any
            // fused chunk is added.
            let depth_cap = if max_rows > 0 && !active.is_empty() {
                (max_rows / active.len()).saturating_sub(1)
            } else {
                usize::MAX
            };
            // `COLI_SPEC_UNION_MAX`: the same ceiling expressed in *bytes* rather
            // than rows. Rows are what the engine controls; union entries are
            // what they cost, and the two are related by a per-row figure only
            // this tick's measurement can supply. Conservative by construction —
            // a real union is sublinear in rows, so a linear projection cuts
            // sooner than strictly necessary.
            let union_cap = union_depth_cap(knobs.spec_union_max, union_per_row_ewma, active.len());
            // The recurrent snapshot is charged per *sequence* while the
            // resident weight read a forward is made of is shared across the
            // batch, so past some width the copies cost more than the accepted
            // tokens repay. `COLI_SPEC_GDN_MAX_B` is where the operator says
            // that width is; drafting simply stops above it, which is the same
            // state the engine is in with speculation off.
            let gdn_width_ok =
                !gdn_rollback || knobs.spec_gdn_max_b == 0 || active.len() <= knobs.spec_gdn_max_b;
            let may_draft = has_mtp && gdn_width_ok;
            // Prompt-lookup needs the same rollback guarantee the MTP head does
            // — any draft row advances a recurrent state — but it does *not*
            // need an MTP head in the checkpoint, so the two sources gate
            // separately. A container converted without `--mtp` can speculate
            // through this one alone.
            let rollback_ok = model.spec_reject_is_kv_only() || (gdn_rollback && gdn_width_ok);
            let may_ngram = ngram.is_enabled() && rollback_ok;
            // Trees are MLA-only — a recurrent layer would chain siblings and
            // the batched GQA path ignores a key set — so the arch check is a
            // gate here rather than an error at the forward.
            let may_tree = knobs.draft_tree && may_draft && may_ngram && model.supports_token_trees();
            // Greedy sequences that fall through to the head draft *together*
            // in one call below: the MTP layer is sparse, so a per-sequence
            // loop streams one routed-expert union per sequence per step where
            // a batched call streams one per step for all of them. Collected
            // here (index, depth) rather than drafted here.
            let mut head_batch: Vec<(usize, usize)> = Vec::new();
            for (i, s) in active.iter_mut().enumerate() {
                let budget_left = s.max_new - s.produced.min(s.max_new);
                let want = draft_depth_for(depth, may_draft || may_ngram, s.sampler.temp, budget_left, sampled)
                    .min(depth_cap);
                let g = want.min(union_cap);
                if g < want {
                    // The byte budget bit, not the row cap or the token budget —
                    // the number that says whether the ceiling is pruning waste
                    // or starving depth, which is the same question
                    // `spec_conf_stops` answers for the acceptance side.
                    spec_union_stops += 1;
                }
                s.draft.clear();
                s.draft_q.clear();
                s.draft_from_ngram = false;
                s.tree = None;
                if g == 0 {
                    continue;
                }
                // Prompt-lookup first, because when it matches it is strictly
                // the cheaper source: a backward scan of `toks` against a full
                // sparse-MoE layer per draft step. Greedy only — an n-gram
                // draft comes from no distribution, so `accept_run_sampled`
                // would have no `q` to score it against.
                if may_ngram && s.sampler.temp <= 0.0 {
                    let d = ngram.draft(&s.toks, s.next_tok, g);
                    if !d.is_empty() {
                        ngram_proposed += d.len() as u64;
                        s.draft_from_ngram = true;
                        s.draft = d;
                        // With `COLI_DRAFT_TREE` the head still drafts, and both
                        // branches are verified below instead of the n-gram
                        // simply winning. Recorded here; the tree is assembled
                        // after the batched head call, which is the only place
                        // both chains exist at once.
                        if may_tree {
                            head_batch.push((i, g));
                            continue;
                        }
                        continue;
                    }
                }
                // No match (or a sampled request): fall back to the head. The
                // two are alternatives per tick rather than a chain, because
                // `mtp_draft` continues from a hidden that assumes its own
                // prefix and cannot be seeded with someone else's tokens.
                if !may_draft || s.hlast.is_empty() {
                    continue;
                }
                // The greedy majority defers to the batched call; only a
                // sampled request drafts inline, because it needs its own
                // `Sampler` and the `q` it drew from.
                if s.sampler.temp <= 0.0 {
                    head_batch.push((i, g));
                    continue;
                }
                // A sampled request needs its drafts drawn from a distribution
                // it can hand to the verifier; a greedy one needs argmax, and
                // handing it a sampled draft would break `accept_run`'s
                // sequence-identity with greedy decoding.
                let drafted = model.mtp_draft_sampled(s.next_tok, g, &s.hlast, conf_floor, &mut s.sampler).map(
                    |(d, q)| {
                        s.draft_q = q;
                        d
                    },
                );
                match drafted {
                    Ok(d) => {
                        // A draft shorter than requested under an active floor
                        // is the gate firing — the number that says whether
                        // 0.65 is pruning wasted verify rows or starving depth.
                        if conf_floor > 0.0 && d.len() < g {
                            spec_conf_stops += 1;
                        }
                        s.draft = d;
                    }
                    // A partial `draft_q` from a failed draft must not survive:
                    // the next verify would score this tick's rows against it.
                    Err(e) => {
                        s.draft_q.clear();
                        peregrine_core::note_advisory_err("mtp draft", &e);
                    }
                }
            }

            // One forward per draft *step* for every greedy sequence, instead
            // of one per sequence per step. On the streaming container the MTP
            // layer is sparse and stored int8 — ~2× a normal expert's bytes —
            // so a per-sequence loop was paying `B` disjoint routed-expert
            // unions for rows that share one here. Depths are per sequence, so
            // a nearly-finished request leaves the batch at its own depth
            // rather than being drafted deep and truncated.
            if !head_batch.is_empty() {
                let nexts: Vec<i32> = head_batch.iter().map(|&(i, _)| active[i].next_tok).collect();
                let hids: Vec<&[f32]> = head_batch.iter().map(|&(i, _)| active[i].hlast.as_slice()).collect();
                let depths: Vec<usize> = head_batch.iter().map(|&(_, g)| g).collect();
                match model.mtp_draft_batched(&nexts, &hids, &depths, conf_floor) {
                    Ok(drafts) => {
                        for ((i, g), d) in head_batch.iter().copied().zip(drafts) {
                            // A draft shorter than requested under an active
                            // floor is the gate firing — the number that says
                            // whether 0.65 prunes wasted verify rows or starves
                            // depth.
                            if conf_floor > 0.0 && d.len() < g {
                                spec_conf_stops += 1;
                            }
                            if let Some(s) = active.get_mut(i) {
                                match (may_tree && s.draft_from_ngram, s.draft.is_empty(), d.is_empty()) {
                                    // Both sources spoke: verify both. Branch A
                                    // is the n-gram run already in `s.draft`,
                                    // branch B the head's chain — two children
                                    // of the pending token, each a chain below
                                    // its own root child.
                                    (true, false, false) => {
                                        let (tree, flat) = hedge_tree(s.next_tok, &s.draft, &d);
                                        s.draft = flat;
                                        if tree.is_some() {
                                            spec_trees += 1;
                                        }
                                        s.tree = tree;
                                    }
                                    // Only the head spoke (no n-gram match), or
                                    // the head came up empty — a chain either
                                    // way, which needs no tree at all.
                                    (_, true, _) => s.draft = d,
                                    _ => {}
                                }
                            }
                        }
                    }
                    // Speculation is a wall-clock optimisation: a failed draft
                    // drops the whole batch back to plain decode for this tick
                    // rather than dropping any client.
                    Err(e) => peregrine_core::note_advisory_err("mtp draft (batched)", &e),
                }
            }
        }

        // The chunk's logits are the rows after the decoders. Finishing it here,
        // through the same `finish_prefill_chunk` the unfused path uses, is what
        // keeps the two ticks observationally identical.
        if let Some((p, end)) = fused {
            let out_cfg = OutputCfg { vocab, stop_ids: &stop_ids };
            tokens_emitted +=
                finish_prefill_chunk(p, end, &logits[n_dec_rows * vocab..], &mut pending, &mut active, out_cfg, &mut prefix)
                    as u64;
        }

        // Periodically migrate the hottest experts into VRAM (heat-ranked
        // residency). Between steps, so it holds the exclusive borrow reheat needs;
        // a no-op without a GPU tier.
        steps += 1;
        // Package energy on the sensor-governor period. Sampling coarsely does
        // not lose energy: `delta_uj` reports everything since the previous
        // call, so the running total is exact whatever the period.
        if steps.is_multiple_of(ENERGY_SAMPLE_EVERY) {
            if let Some(uj) = energy.delta_uj() {
                energy_uj = Some(energy_uj.unwrap_or(0).saturating_add(uj));
            }
        }
        if steps.is_multiple_of(REHEAT_EVERY) {
            if let Err(e) = model.reheat() {
                eprintln!("peregrine: reheat failed: {e}");
            }
        }
        // Publish this tick's view for `GET /metrics`. The engine thread owns
        // the model exclusively, so this is the only place these numbers can be
        // read from; a handler cannot reach `Model` at all.
        *telemetry.lock() = EngineTelemetry {
            runtime: model.telemetry(),
            lane_last: model.last_lane_timings(),
            lane_ewma: model.lane_ewma(),
            active: active.len(),
            pending: pending.len(),
            steps: steps as u64,
            ecache: model.ecache_stats(),
            prefetch_reads: model.ecache_prefetch_reads().unwrap_or(0),
            spec_proposed,
            spec_accepted,
            spec_conf_stops,
            gdn_snapshot_bytes,
            gdn_replays,
            gdn_replay_rows,
            ngram_proposed,
            ngram_accepted,
            tokens_emitted,
            decode_rows,
            queue_wait_us,
            queue_admits,
            queue_wait_max_us,
            spec_union_stops,
            spec_trees,
            spec_tree_branch_wins,
            energy_uj,
            rlm: model.rlm_stats(),
            kvstore: prefix.disk.as_ref().map(|d| (d.saved, d.loaded, d.tokens_restored)),
            io_slab_in_use: model.io_slab_in_use(),
        };
    }
    // Shutdown: report what the prefix cache absorbed. Silent when it is off, so
    // a default run's output is unchanged.
    if prefix.mem.enabled() {
        eprintln!(
            "[prefix-cache] hits={} tokens_reused={} entries={} resident={:.1} MiB",
            prefix.mem.hits,
            prefix.mem.tokens_saved,
            prefix.mem.entries.len(),
            prefix.mem.used as f64 / (1024.0 * 1024.0)
        );
    }
    // Disk-persisted sessions, silent unless COLI_KV_STORE_DIR enabled them.
    if let Some(d) = &prefix.disk {
        // The write path is asynchronous (kvstore.rs); draining it first makes
        // entries/resident reflect every accepted checkpoint, and dropped_busy
        // says how many the depth-1 writer queue declined.
        d.flush();
        eprintln!(
            "[kvstore] saved={} loaded={} tokens_restored={} entries={} resident={:.1} MiB dropped_busy={}",
            d.saved,
            d.loaded,
            d.tokens_restored,
            d.entry_count(),
            d.resident_bytes() as f64 / (1024.0 * 1024.0),
            d.dropped_busy
        );
    }
    // Topic-routing profiles learn in-process; without this a long-running
    // server relearns its workload mix every boot — the sidecar's other writer
    // is the stdio engine's route-stats path, which this server never takes.
    // Unconditional by design: the call is a documented no-op when
    // COLI_TOPIC_ROUTING is off.
    if let Err(e) = model.save_topic_profiles_here() {
        peregrine_core::note_advisory_err("topic profiles persist", &e);
    }
    // Speculation accounting, silent when speculation never ran. The accept
    // rate is what says whether COLI_DRAFT's depth pays for its verify rows.
    if spec_proposed > 0 {
        eprintln!(
            "[spec] proposed={spec_proposed} accepted={spec_accepted} conf_stops={spec_conf_stops} accept_rate={:.1}%",
            spec_accepted as f64 / spec_proposed as f64 * 100.0
        );
    }
    // RLM refinement accounting, same shape ((0,0) prints nothing).
    let (rlm_passes, rlm_tokens) = model.rlm_stats();
    if rlm_passes > 0 {
        eprintln!("[rlm] passes={rlm_passes} tokens_recursed={rlm_tokens}");
    }
    // Warm-tier and prefetch effectiveness. This has to happen *here* — the engine
    // thread owns the `Model`, so `main` cannot ask it anything after the server
    // stops, and the model is dropped the moment this function returns.
    //
    // The stdio binary has printed these since the feature landed and the HTTP
    // server never did, so on the serving path — the only path with per-sequence
    // prefetch, and therefore the only one where prefetch lanes mean anything —
    // there was no way to see whether prefetch was earning its keep. Deliberately
    // just the two aggregate lines: the per-layer breakdown, gate stats, look-ahead
    // and predictor scoreboard the stdio binary also prints are a larger reporting
    // surface than this needs, and belong in a change of their own.
    //
    // Silent without a warm cache (`ecache_stats` is `None`), so a resident-mode
    // run's output is unchanged.
    if let Some((resolved, mergeable)) = model.expert_map_stats() {
        let share = 100.0 * mergeable as f64 / resolved.max(1) as f64;
        let reads = 6 * (resolved - mergeable) + 2 * mergeable;
        eprintln!(
            "[expertmap] indexed={resolved} coalescing={mergeable} ({share:.1}%) \
             -> {reads} reads per full sweep vs {} unmerged",
            6 * resolved
        );
    }
    // The hit rate below is not interpretable without this: under one token's
    // working set a layer sweep drives any recency policy to zero, so a low
    // number is the budget talking, not the policy.
    if let Some((per_token, protect)) = model.expert_working_set() {
        eprintln!(
            "[workingset] {:.2} GB per token; prefetch-protect {}",
            per_token as f64 / (1u64 << 30) as f64,
            if protect { "on (budget cannot hold a pass)" } else { "off (budget holds a pass)" }
        );
    }
    // Where the time actually went. The four lane counters have always been
    // collected — `moe_forward_concurrent` bumps them per layer — but the only
    // consumer was the bubble tuner, and `snapshot_and_reset` wipes them every
    // forward, so no operator could ever ask "was that run I/O-bound?".
    {
        let (t, forwards) = model.lane_totals();
        if forwards > 0 {
            let s = |us: u64| us as f64 / 1e6;
            let (io, cpu, gpu, red) = (s(t.io_us), s(t.cpu_us), s(t.gpu_us), s(t.reduce_us));
            let sum = (io + cpu + gpu + red).max(1e-9);
            let pct = |v: f64| 100.0 * v / sum;
            eprintln!(
                "[lane] {forwards} forwards: io {io:.1}s ({:.0}%) cpu {cpu:.1}s ({:.0}%) \
                 gpu {gpu:.1}s ({:.0}%) reduce {red:.1}s ({:.0}%)",
                pct(io),
                pct(cpu),
                pct(gpu),
                pct(red)
            );
            // Per forward is the number that maps onto a decode token, and the
            // caveat is load-bearing: these are summed over lanes that run at the
            // same time, so `sum / wall` is the overlap achieved, not overhead.
            // A sum close to wall clock means the lanes are serialising.
            eprintln!(
                "[lane] per forward: io {:.2}s cpu {:.2}s (lane-summed, so compare \
                 sum/wall for the overlap achieved)",
                io / forwards as f64,
                cpu / forwards as f64,
            );
            // The duty cycles. `io_us` is summed over one thread per ring, so the
            // I/O lane's occupancy is `io_us / (rings x lane_wall)`. Below ~1.0 the
            // rings are idle inside the MoE call; and `lane_wall` against the
            // token's own wall clock is how much of a token is in the MoE lane at
            // all, the rest being attention, the router and the reduce.
            let wall = s(t.lane_wall_us);
            if wall > 0.0 {
                // The engine's *resolved* ring count, not a second reading of
                // the environment. `io_rings()` clamps to 16; this line used to
                // re-parse `COLI_IO_RINGS` raw, so `=64` ran 16 rings and
                // divided the duty by 64 — understating the engine's headline
                // occupancy number by 4x, in the direction that makes the I/O
                // lane look idle when it is saturated.
                let rings = peregrine_model::io_rings() as f64;
                eprintln!(
                    "[lane] moe wall {wall:.1}s over {forwards} forwards ({:.2}s each); \
                     io duty {:.0}% of {rings:.0} rings, cpu {:.1} workers busy",
                    wall / forwards as f64,
                    100.0 * io / (rings * wall),
                    cpu / wall,
                );
            }
            if t.cpu_us > 0 && t.cpu_bytes > 0 {
                eprintln!(
                    "[lane] cpu-lane bandwidth {:.2} GB/s over {:.1} GB of expert slabs",
                    t.cpu_bytes as f64 / 1e9 / s(t.cpu_us),
                    t.cpu_bytes as f64 / 1e9,
                );
            }
            // Cache-lock contention, thread-summed like io_us. Read it against
            // io thread time: a few percent is bookkeeping, tens of percent is
            // rings queueing on the mutex — the evidence a sharded cache needs.
            if t.cache_wait_us > 0 && t.io_us > 0 {
                eprintln!(
                    "[lane] cache-lock wait {:.2}s ({:.1}% of io thread time)",
                    s(t.cache_wait_us),
                    100.0 * t.cache_wait_us as f64 / t.io_us as f64,
                );
            }
        }
    }
    if let Some((h, m, d)) = model.ecache_stats() {
        let hr = 100.0 * h as f64 / (h + m).max(1) as f64;
        let pf = model.ecache_prefetch_reads().unwrap_or(0);
        eprintln!("[ecache] hits={h} misses={m} disk_reads={d} prefetch_reads={pf} hit_rate={hr:.1}%");
        // Occupancy, because a ~0% hit rate has two opposite causes and the line
        // above cannot distinguish them: a full cache is evicting the working set
        // before it can be reused; an empty one is not admitting in the first place.
        if let Some((slots, used, budget)) = model.ecache_occupancy() {
            let fill = if budget > 0 { 100.0 * used as f64 / budget as f64 } else { 0.0 };
            eprintln!(
                "[ecache] resident: {slots} slots, {:.2} GB of {:.2} GB budget ({fill:.1}% full)",
                used as f64 / 1e9,
                budget as f64 / 1e9
            );
        }
        let (used, wasted) = model.ecache_prefetch_effectiveness().unwrap_or((0, 0));
        let acc = 100.0 * model.prefetch_accuracy().unwrap_or(0.0);
        let fadv = model.ecache_fadvise_hints().unwrap_or(0);
        let vm = model.ecache_verify_mismatch().unwrap_or(0);
        // `accuracy` is `used/(used+wasted)`, and **`wasted` is only incremented on
        // eviction** (`warmcache.rs`): a prefetched slab still resident at shutdown
        // is neither used nor wasted, so it is not in that denominator at all. On the
        // first serving-path run 433 reads were issued and only 64 were classified,
        // which makes a bare `accuracy` a survivorship-biased view of a much smaller
        // effective yield — 21.9% of the classified, 3.2% of the issued. Both are
        // printed, with the unclassified remainder, so neither can be quoted alone.
        //
        // Worth knowing when reading it: `PrefetchTuner` (`COLI_PREFETCH_TUNE`) EWMAs
        // the same used/wasted pair, so it is steering on the classified slice only.
        // Whether that biases the tuner is a real question and not answered here.
        let unclassified = pf.saturating_sub(used + wasted);
        let yield_pct = if pf > 0 { 100.0 * used as f64 / pf as f64 } else { 0.0 };
        // `stale_dropped` is *not* part of `issued`: those items never reached a
        // read, so they are disk bandwidth the gate returned to the demand lane.
        let sd = model.ecache_prefetch_stale_dropped().unwrap_or(0);
        eprintln!(
            "[prefetch] used={used} wasted={wasted} unclassified={unclassified} \
             accuracy={acc:.1}% (of {} classified) yield={yield_pct:.1}% (of {pf} issued) \
             fadvise={fadv} verify_mismatch={vm} stale_dropped={sd}",
            used + wasted
        );
        // How much of the cache speculation is *holding* — the unclassified slabs
        // above, as bytes. This is the quantity that competes with demand data for
        // the budget, and no counter reported it before 2026-08-09.
        if let Some((bytes, slots, budget)) = model.ecache_speculative_resident() {
            let share = if budget > 0 { 100.0 * bytes as f64 / budget as f64 } else { 0.0 };
            eprintln!(
                "[prefetch] resident-unused: {slots} slots, {:.2} GB of {:.2} GB budget ({share:.1}%)",
                bytes as f64 / 1e9,
                budget as f64 / 1e9
            );
        }
    }
}

/// Blocking-wait for a request across the two priority channels, biased toward
/// high-priority. Returns `None` when both senders are dropped (shutdown).
fn recv_priority(
    rt: Option<&tokio::runtime::Runtime>,
    rx_high: &mut mpsc::UnboundedReceiver<Queued>,
    rx_normal: &mut mpsc::UnboundedReceiver<Queued>,
) -> Option<Queued> {
    // Fast path: something already queued (fold both into one blocking wait if
    // not; empty/disconnected queues fall through — the slow path treats
    // both-senders-gone as the shutdown signal).
    match rx_high.try_recv() {
        Ok(r) => return Some(r),
        Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
    }
    match rx_normal.try_recv() {
        Ok(r) => return Some(r),
        Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
    }
    // Slow path: park until whichever arrives first, biased toward high. A
    // `None` from one channel means only *that* sender is gone — the other may
    // still deliver work, so keep waiting on it rather than reading one closed
    // channel as engine shutdown.
    match rt {
        Some(rt) => rt.block_on(async {
            let (mut high_open, mut normal_open) = (true, true);
            loop {
                if !high_open && !normal_open {
                    return None; // both senders dropped → shutdown
                }
                tokio::select! {
                    biased;
                    r = rx_high.recv(), if high_open => match r {
                        Some(req) => return Some(req),
                        None => high_open = false,
                    },
                    r = rx_normal.recv(), if normal_open => match r {
                        Some(req) => return Some(req),
                        None => normal_open = false,
                    },
                }
            }
        }),
        // Fallback if the mini-runtime failed to build (e.g. resource-starved test
        // process): busy-wait poll with a short sleep. Correctness preserved.
        None => {
            loop {
                let high = match rx_high.try_recv() {
                    Ok(r) => return Some(r),
                    Err(state) => state,
                };
                let norm = match rx_normal.try_recv() {
                    Ok(r) => return Some(r),
                    Err(state) => state,
                };
                // Both senders dropped and both queues empty → shutdown signal.
                let high_dead = matches!(high, mpsc::error::TryRecvError::Disconnected);
                let norm_dead = matches!(norm, mpsc::error::TryRecvError::Disconnected);
                if high_dead && norm_dead {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

/// Queue a request for chunked prefill. Validates it (empty prompt / zero budget is
/// a clean no-op); the forward happens in [`prefill_step`], interleaved with decode.
/// The most recent admission's workload class becomes the model's active class —
/// a pragmatic "latest wins" policy for a mixed batch (per-sequence classes would
/// need per-sequence prefetch policies; the breadth knob is batch-global today).
/// Returns how long this request waited between `submit` and admission, in
/// microseconds — the one span no existing instrument covers.
/// `bench-serve-lanes.py` measures whole-request wall time and
/// `bench-serve-gaps.py` measures inter-token gaps; both start counting once a
/// request is already being served, so queue time hides inside "the server was
/// slow" and cannot be told apart from decode being slow.
fn admit_pending(
    model: &Model,
    pending: &mut VecDeque<Prefilling>,
    q: Queued,
    prefix: &mut PrefixStore,
    queued: &std::sync::atomic::AtomicUsize,
) -> u64 {
    let waited = q.at.elapsed().as_micros().min(u64::MAX as u128) as u64;
    let req = q.req;
    // The request has left the channel: it no longer counts against
    // `COLI_QUEUE_DEPTH`. Saturating (a plain `fetch_sub` would wrap) because
    // tests drive this function directly with a counter no submit incremented.
    let mut cur = queued.load(std::sync::atomic::Ordering::Relaxed);
    while cur > 0 {
        match queued.compare_exchange_weak(
            cur,
            cur - 1,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(now) => cur = now,
        }
    }
    if req.prompt.is_empty() || req.max_new == 0 {
        return waited; // nothing to generate; dropping req.out closes the stream cleanly
    }
    model.set_workload_class(req.class);
    // Seed from the longest cached prefix of this prompt, so the shared head of a
    // chat template (system prompt, few-shot examples) is prefilled once rather
    // than once per request. `None` — including whenever the cache is off —
    // reproduces the historical cold start exactly.
    let (seq, pos) = match prefix.lookup(&req.prompt) {
        Some((kv, n)) => (kv, n),
        None => (SeqKv::new(&model.cfg), 0),
    };
    pending.push_back(Prefilling {
        seq,
        prompt: req.prompt,
        pos,
        sampler: req.sampler,
        out: req.out,
        max_new: req.max_new,
        hist: Mutex::new(model.new_route_history()),
        seq_id: NEXT_SEQ_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    });
    waited
}

/// Advance the front prefilling sequence by up to `PREFILL_CHUNK` tokens. When its
/// prompt is fully prefilled, sample the first token and move it to `active` (or
/// retire it). Round-robins the queue so no one prefill monopolizes the engine.
fn prefill_step(
    model: &Model,
    pending: &mut VecDeque<Prefilling>,
    active: &mut Vec<SeqState>,
    vocab: usize,
    stop_ids: &[i32],
    chunk_div: usize,
    prefix: &mut PrefixStore,
) -> usize {
    let Some(mut p) = pending.pop_front() else {
        return 0;
    };
    let end = chunk_end(p.pos, p.prompt.len(), chunk_div);
    // The chunk is a plain slice of the prompt; `seq` is a disjoint field, so no
    // per-step copy is needed to satisfy the borrow checker.
    let Prefilling { seq, prompt, pos, out, .. } = &mut p;
    let logits = match model.forward_prefill_seq(&prompt[*pos..end], seq, *pos) {
        Ok(l) => l,
        Err(e) => {
            if out.send(EngineOut::Error(e.to_string())).is_err() {
                peregrine_core::note_advisory_err("prefill error forward", &"client already disconnected");
            }
            return 0; // drop this sequence
        }
    };
    finish_prefill_chunk(p, end, &logits, pending, active, OutputCfg { vocab, stop_ids }, prefix)
}

/// The two things sampling needs, together — they are read as a pair
/// everywhere and bundling them keeps [`finish_prefill_chunk`] under the
/// argument limit without an `#[allow]`, which the strict audit rejects.
#[derive(Clone, Copy)]
struct OutputCfg<'a> {
    vocab: usize,
    stop_ids: &'a [i32],
}

/// Where this sequence's next prefill chunk ends.
fn chunk_end(pos: usize, prompt_len: usize, chunk_div: usize) -> usize {
    (pos + prefill_chunk(pos, chunk_div)).min(prompt_len)
}

/// Consume one prefill chunk's logits: either queue the next chunk, or complete
/// the prefill — snapshot it for the prefix cache, sample the first token, and
/// promote the sequence into the decode batch.
///
/// Split out of [`prefill_step`] so the fused tick, which produces these logits
/// inside the *decode* forward, finishes a chunk by exactly the same rules. The
/// two paths differing here is how a fusion silently changes served output.
/// Borrowed row-indexed view of one batched forward's outputs (logits and
/// pre-final-norm hidden, with their row strides).
struct ForwardRows<'a> {
    logits: &'a [f32],
    hidden: &'a [f32],
    vocab: usize,
    d_hidden: usize,
}

/// RLM refinement of one decision row of a batched forward (COLI_RLM).
///
/// Copies the row's logits and pre-final-norm hidden only after the cheap
/// enabled check, hands them to [`Model::rlm_refine_external`] (which loops
/// the uncertainty policy over throwaway KV replays of `seq`'s causal prefix
/// at `pos`), and returns the refined pair — `None` when RLM is off, the row
/// is out of range, no pass triggered, or the replay failed (advisory, like a
/// draft failure: refinement is a quality optimisation, never worth dropping
/// a client over).
fn rlm_refine_row(
    model: &Model,
    seq: &SeqKv,
    pos: usize,
    rows: &ForwardRows<'_>,
    row: usize,
    temp: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if !peregrine_model::rlm::rlm_enabled() {
        return None;
    }
    let r = rows.logits.get(row * rows.vocab..(row + 1) * rows.vocab)?;
    let hsrc = rows.hidden.get(row * rows.d_hidden..(row + 1) * rows.d_hidden)?;
    let mut lg = r.to_vec();
    let mut h = hsrc.to_vec();
    match model.rlm_refine_external(seq, pos, &mut h, &mut lg, temp) {
        Ok(true) => Some((lg, h)),
        Ok(false) => None,
        Err(e) => {
            peregrine_core::note_advisory_err("rlm refine", &e);
            None
        }
    }
}

fn finish_prefill_chunk(
    p: Prefilling,
    end: usize,
    logits: &[f32],
    pending: &mut VecDeque<Prefilling>,
    active: &mut Vec<SeqState>,
    out_cfg: OutputCfg,
    prefix: &mut PrefixStore,
) -> usize {
    let OutputCfg { vocab, stop_ids } = out_cfg;
    let Prefilling { seq, prompt, pos, mut sampler, out, max_new, hist, seq_id } = p;
    let chunk_len = end - pos;
    if end < prompt.len() {
        // more chunks to go — round-robin with the others
        pending.push_back(Prefilling { seq, prompt, pos: end, sampler, out, max_new, hist, seq_id });
        return 0;
    }
    // Prefill complete. Snapshot it before the KV moves into the active set, so
    // the next request sharing this prompt's head starts from here. No-op when
    // the cache is disabled or the prompt is too short to be worth copying.
    prefix.insert(&prompt, &seq);
    // Sample the first token from the last prompt position. An empty chunk would
    // mean an empty prompt, which `admit_pending` rejects.
    let Some(last) = chunk_len.checked_sub(1).map(|c| c * vocab) else {
        return 0;
    };
    let t0 = sampler.pick(&logits[last..last + vocab], -1) as i32;
    if stop_ids.contains(&t0) {
        return 0; // first token is a stop → emit nothing
    }
    if out.send(EngineOut::Token(t0 as u32)).is_err() {
        return 0; // client already gone — the token reached nobody
    }
    // Sent, so it counts from here whatever becomes of the sequence.
    if max_new <= 1 {
        return 1; // only one token requested
    }
    active.push(SeqState {
        seq,
        hist,
        seq_id,
        pos: prompt.len(),
        next_tok: t0,
        sampler,
        out,
        produced: 1,
        max_new,
        // No draft yet: the first verify produces the hidden a draft needs.
        draft: Vec::new(),
        draft_q: Vec::new(),
        draft_from_ngram: false,
        tree: None,
        hlast: Vec::new(),
        // The fed-token log starts as the prompt (row-aligned with `seq`,
        // whose rows so far are exactly the prompt); `t0` joins it when fed.
        toks: prompt,
    });
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_model::argmax;

    #[test]
    fn kv_budget_off_is_exactly_the_historical_admission() {
        // The default must be indistinguishable from the count-only engine that
        // shipped: no byte can refuse an admission when the budget is 0.
        for resident in [0usize, 1, usize::MAX] {
            for in_flight in [0usize, 1, 32] {
                assert!(kv_admits(resident, 0, in_flight), "budget 0 must never refuse");
            }
        }
    }

    #[test]
    fn a_shared_prefix_is_charged_once_not_once_per_sequence() {
        // Byte-aware admission and refcounted prefix sharing have to agree or
        // the first cancels the second: charging a shared system prompt to every
        // concurrent request would refuse admissions over RAM that was never
        // allocated. Four sequences, three of them sharing one 900-byte prefix.
        let shared = Some((0xA11CE, 900));
        let seqs = [(10, shared), (20, shared), (30, shared), (40, Some((0xB0B, 500)))];
        assert_eq!(dedup_kv_bytes(seqs.into_iter()), 10 + 20 + 30 + 40 + 900 + 500);
        // Sequences owning their KV privately are unaffected — the historical case.
        assert_eq!(dedup_kv_bytes([(10, None), (20, None)].into_iter()), 30);
        assert_eq!(dedup_kv_bytes(std::iter::empty()), 0);
        // Distinct allocations are never merged, even at equal sizes.
        assert_eq!(dedup_kv_bytes([(0, Some((1, 700))), (0, Some((2, 700)))].into_iter()), 1400);
    }

    #[test]
    fn kv_budget_refuses_once_resident_crosses_it() {
        let budget = 1000;
        assert!(kv_admits(999, budget, 4), "under budget admits");
        assert!(!kv_admits(1000, budget, 4), "at budget refuses — this is a high-water gate");
        assert!(!kv_admits(5000, budget, 4), "over budget refuses");
    }

    #[test]
    fn an_empty_engine_always_admits_even_over_budget() {
        // Without this the engine hangs instead of running a request whose KV
        // exceeds the whole budget: it would be refused forever, and refusing
        // forever is a worse failure than overshooting once. Same "keep at least
        // one" shape as the warm cache's eviction guard.
        assert!(kv_admits(usize::MAX, 1, 0), "nothing in flight must still admit");
        assert!(!kv_admits(usize::MAX, 1, 1), "…but only while nothing is in flight");
    }

    fn tiny_dir(tag: &str) -> Result<std::path::PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_batch_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        peregrine_model::testkit::build_tiny_model(&d)?;
        Ok(d)
    }

    /// A tiny Qwen3.5-hybrid checkpoint *with* an MTP head — the recurrent arch
    /// that could not speculate until `COLI_SPEC_GDN` wired the state rollback.
    fn hybrid_mtp_dir(tag: &str) -> Result<std::path::PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_hyb_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        peregrine_model::testkit::build_tiny_hybrid_model_with_mtp(&d, 0x5EC)?;
        Ok(d)
    }

    /// Reference greedy decode of one sequence alone, via the same absorb path the
    /// engine uses (prefill, then B==1 batched steps) — the ground truth the
    /// batched engine must reproduce for each concurrent request.
    fn ref_decode(model: &Model, prompt: &[i32], n: usize) -> Result<Vec<u32>, Error> {
        let vocab = model.cfg.vocab as usize;
        let stop = model.cfg.stop_ids.clone();
        let mut seq = SeqKv::new(&model.cfg);
        let logits = model.forward_prefill_seq(prompt, &mut seq, 0)?;
        let last = (prompt.len() - 1) * vocab;
        let mut tok = argmax(&logits[last..last + vocab]) as i32;
        let mut out = Vec::new();
        let mut pos = prompt.len();
        while out.len() < n {
            if stop.contains(&tok) {
                break;
            }
            out.push(tok as u32);
            if out.len() >= n {
                break;
            }
            let mut one: [&mut SeqKv; 1] = [&mut seq];
            let lg = model.forward_step_batched(&[tok], &mut one, &[pos], None)?;
            pos += 1;
            tok = argmax(&lg[..vocab]) as i32;
        }
        Ok(out)
    }

    #[test]
    fn queue_cap_refuses_at_depth_and_recovers_on_drain() {
        // The COLI_QUEUE_DEPTH contract, driven directly (the knob itself is a
        // process-wide OnceLock, so the cap is injected here the same way the
        // spec/fuse overrides are): at the cap a submit refuses Full — nothing
        // wrong with the request — and draining makes room again.
        let (tx_normal, _rx_n) = mpsc::unbounded_channel::<Queued>();
        let (tx_high, _rx_h) = mpsc::unbounded_channel::<Queued>();
        let queued = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handle = EngineHandle {
            tx_normal,
            tx_high,
            telemetry: std::sync::Arc::new(parking_lot::Mutex::new(EngineTelemetry::default())),
            queued: queued.clone(),
            queue_cap: 2,
        };
        let req = || {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            (
                EngineRequest {
                    prompt: vec![1, 2, 3],
                    max_new: 1,
                    sampler: Sampler::new(0.0, 0.9, 1),
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                },
                rx,
            )
        };
        let (r1, _k1) = req();
        let (r2, _k2) = req();
        let (r3, _k3) = req();
        assert!(handle.submit(r1).is_ok(), "under the cap admits");
        assert!(handle.submit(r2).is_ok(), "at cap-1 admits");
        assert!(matches!(handle.submit(r3), Err(SubmitRefused::Full)), "at the cap refuses Full");
        // The engine draining one request makes room for exactly one more.
        queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        let (r4, _k4) = req();
        let (r5, _k5) = req();
        assert!(handle.submit(r4).is_ok(), "drain restores one admission");
        assert!(matches!(handle.submit(r5), Err(SubmitRefused::Full)), "and only one");
    }

    #[test]
    fn retiring_output_extends_the_prefix_cache_past_the_prompt() -> Result<(), Error> {
        // The multi-turn shape: a client's next request resends prompt+output
        // as its new prompt's head. Before the retire-time insert, the cache
        // held prompt-only entries, so the assistant turn was re-prefilled
        // every round trip. This drives the exact sequence the engine performs
        // — prompt-only insert at prefill completion, fed-token log through
        // decode, prompt+output insert at retire — and asserts the follow-up
        // request matches past the original prompt, with the covered
        // prompt-only entry dropped rather than accumulated.
        let dir = tiny_dir("prefixgen")?;
        let model = Model::load(&dir)?;
        let vocab = model.cfg.vocab as usize;
        let mut prefix = PrefixCache::new(1 << 20);

        // A prompt over the 64-token floor, prefilled the way the engine does.
        let prompt: Vec<i32> = (0..96).map(|i| (i * 5 % vocab.min(11)) as i32).collect();
        let mut seq = SeqKv::new(&model.cfg);
        let logits = model.forward_prefill_seq(&prompt, &mut seq, 0)?;
        prefix.insert(&prompt, &seq); // finish_prefill_chunk's prompt-only entry
        assert_eq!(prefix.entries.len(), 1, "prompt entry cached");

        // Decode a few tokens, keeping the fed-token log the accept loop keeps.
        let last = (prompt.len() - 1) * vocab;
        let mut tok = argmax(&logits[last..last + vocab]) as i32;
        let mut toks = prompt.clone();
        let mut pos = prompt.len();
        for _ in 0..4 {
            let mut one: [&mut SeqKv; 1] = [&mut seq];
            let lg = model.forward_step_batched(&[tok], &mut one, &[pos], None)?;
            toks.push(tok);
            pos += 1;
            tok = argmax(&lg[..vocab]) as i32;
        }
        assert_eq!(toks.len(), seq.len(), "fed-token log stays row-aligned");

        // Retire: insert prompt+output. The prompt-only entry is now covered.
        prefix.insert(&toks, &seq);
        assert_eq!(prefix.entries.len(), 1, "covered prompt-only entry dropped, not accumulated");
        assert_eq!(prefix.entries[0].tokens, toks, "surviving entry is prompt+output");
        let bytes_sum: usize = prefix.entries.iter().map(|e| e.bytes).sum();
        assert_eq!(prefix.used, bytes_sum, "used-bytes accounting survives the cleanup");

        // The next turn resends prompt+output plus new user tokens: the match
        // must run past the original prompt — that is the whole point.
        let mut next_turn = toks.clone();
        next_turn.extend_from_slice(&[1, 2, 3]);
        let (seeded, n) = prefix.lookup(&next_turn).ok_or(Error::Format("lookup must hit".into()))?;
        assert_eq!(n, toks.len(), "match depth covers the generated tokens, not just the prompt");
        assert!(n > prompt.len(), "deeper than any prompt-only entry could reach");
        assert_eq!(seeded.len(), n, "seeded KV length equals the match");
        Ok(())
    }

    #[test]
    fn a_retiring_sequence_does_not_renumber_the_streams_behind_it() -> Result<(), Error> {
        // Until 2026-08-08 the prefetch lane was keyed on a sequence's index in
        // `active`, which the decode loop compacts with `retain` every tick. When a
        // middle stream retired, every stream behind it slid down one index and so
        // changed lane mid-flight, leaving its queued reads split across two
        // io_uring rings with no ordering between them. Prefetch is
        // correctness-neutral, so nothing failed — the lane key just stopped
        // meaning what the design said it meant.
        let dir = tiny_dir("seqid")?;
        let model = Model::load(&dir)?;
        let mut pending: VecDeque<Prefilling> = VecDeque::new();
        let mut prefix = PrefixStore::new(0, None);
        let mut keepalive = Vec::new();
        for _ in 0..3 {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            keepalive.push(rx); // hold the receivers so no send sees a dropped client
            admit_pending(
                &model,
                &mut pending,
                Queued {
                    at: std::time::Instant::now(),
                    req: EngineRequest {
                        prompt: vec![3i32, 7],
                        max_new: 4,
                        sampler: Sampler::new(0.0, 0.9, 1),
                        out: tx,
                        priority: Priority::Normal,
                        class: peregrine_model::TokenClass::Prose,
                    },
                },
                &mut prefix,
                &std::sync::atomic::AtomicUsize::new(0),
            );
        }
        let ids: Vec<usize> = pending.iter().map(|p| p.seq_id).collect();
        assert_eq!(ids.len(), 3, "three admissions");
        assert!(ids[0] < ids[1] && ids[1] < ids[2], "admission order gives increasing ids, got {ids:?}");

        // Promote them the way `finish_prefill_chunk` does — the point is that the
        // id set at admission is what reaches `active`, not a fresh one.
        let mut active: Vec<SeqState> = pending
            .into_iter()
            .map(|p| SeqState {
                seq: p.seq,
                hist: p.hist,
                seq_id: p.seq_id,
                pos: 0,
                next_tok: 1,
                sampler: p.sampler,
                out: p.out,
                produced: 1,
                max_new: p.max_new,
                draft: Vec::new(),
                draft_q: Vec::new(),
                draft_from_ngram: false,
        tree: None,
                hlast: Vec::new(),
                toks: p.prompt,
            })
            .collect();
        assert_eq!(active.iter().map(|s| s.seq_id).collect::<Vec<_>>(), ids, "promotion preserves the id");

        // The middle stream retires, exactly as the decode loop's `retain` does.
        let keep = [true, false, true];
        let mut k = keep.iter();
        active.retain(|_| *k.next().unwrap_or(&true));

        assert_eq!(active.len(), 2, "one retired");
        assert_eq!(active[0].seq_id, ids[0], "the leader is untouched");
        // The survivor *did* slide down an index — that is the compaction the old
        // scheme keyed on — but its lane key is still the one it was admitted with.
        assert_eq!(active.iter().position(|s| s.seq_id == ids[2]), Some(1), "it slid to index 1");
        assert_eq!(active[1].seq_id, ids[2], "…and kept its lane across the retire");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn hybrid_engine_batches_and_matches_reference() -> Result<(), Error> {
        // The Track C phase-2a end-to-end: the batch engine serves a hybrid
        // (GDN + gated-GQA) model, three concurrent greedy requests each
        // matching the standalone decode — with the prefix cache and disk KV
        // sessions auto-disabled for the recurrent architecture.
        let d = std::env::temp_dir().join(format!("peregrine_batch_hybrid_{}", std::process::id()));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        peregrine_model::testkit::build_tiny_hybrid_model(&d, 45)?;
        let prompt = vec![3i32, 7, 1, 4];
        let n = 6usize;
        let want = {
            let m = Model::load(&d)?;
            assert!(!m.prefix_cachable(), "hybrid must opt out of prefix caching in 2a");
            ref_decode(&m, &prompt, n)?
        };
        assert!(!want.is_empty(), "reference must produce tokens");

        let (handle, join) = spawn(Model::load(&d)?, 8)?;
        let mut rxs = Vec::new();
        for _ in 0..3 {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: prompt.clone(),
                max_new: n,
                sampler: Sampler::new(0.0, 0.9, 1),
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            rxs.push(rx);
        }
        let mut outs = Vec::new();
        for mut rx in rxs {
            let mut toks = Vec::new();
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    EngineOut::Token(t) => toks.push(t),
                    EngineOut::Error(e) => return Err(Error::Format(e)),
                }
            }
            outs.push(toks);
        }
        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        for (i, o) in outs.iter().enumerate() {
            assert_eq!(o, &want, "hybrid batched request {i} must match the reference decode");
        }
        std::fs::remove_dir_all(&d)?;
        Ok(())
    }

    #[test]
    fn engine_batches_and_matches_reference() -> Result<(), Error> {
        // Three identical greedy requests submitted concurrently must each decode
        // to the same tokens as a standalone single-sequence decode — proving the
        // batched multiplexing (admit/pack/retire) is correct, not just the math.
        let dir = tiny_dir("match")?;
        let prompt = vec![3i32, 7, 1, 4];
        let n = 6usize;
        let want = {
            let m = Model::load(&dir)?;
            ref_decode(&m, &prompt, n)?
        };
        assert!(!want.is_empty(), "reference must produce tokens");

        let (handle, join) = spawn(Model::load(&dir)?, 8)?;
        let mut rxs = Vec::new();
        for _ in 0..3 {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: prompt.clone(),
                max_new: n,
                sampler: Sampler::new(0.0, 0.9, 1),
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            rxs.push(rx);
        }

        let mut outs = Vec::new();
        for mut rx in rxs {
            let mut toks = Vec::new();
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    EngineOut::Token(t) => toks.push(t),
                    EngineOut::Error(e) => return Err(Error::Format(e)),
                }
            }
            outs.push(toks);
        }
        drop(handle); // let the engine thread observe shutdown and exit
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }

        for (i, o) in outs.iter().enumerate() {
            assert_eq!(o, &want, "batched request {i} must match the reference decode");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefill_chunk_schedule_defaults_to_the_historical_fixed_size() {
        // div 0 (unset) must reproduce the constant exactly, at every position.
        for pos in [0usize, 1, 63, 64, 1000, 8192, 100_000] {
            assert_eq!(prefill_chunk(pos, 0), PREFILL_CHUNK, "pos={pos}");
        }
    }

    #[test]
    fn adaptive_prefill_chunk_grows_with_position_and_holds_the_floor() {
        // Below the floor the chunk stays at PREFILL_CHUNK, so early chunks are
        // unchanged and a short prompt behaves exactly as before.
        assert_eq!(prefill_chunk(0, 4), PREFILL_CHUNK);
        assert_eq!(prefill_chunk(255, 4), PREFILL_CHUNK, "255/4 = 63 < floor");
        // Past it the chunk scales with how much cache each step has to re-derive.
        assert_eq!(prefill_chunk(1024, 4), 256);
        assert_eq!(prefill_chunk(8192, 4), 2048);
        // Monotone, so the schedule can never regress mid-prompt.
        let mut prev = 0;
        for pos in (0..9000).step_by(97) {
            let c = prefill_chunk(pos, 8);
            assert!(c >= prev, "chunk shrank at pos={pos}");
            prev = c;
        }
    }

    #[test]
    fn adaptive_chunking_turns_quadratic_reconstruction_linear() {
        // The reason the knob exists. `attend_dense` re-derives every cached
        // position on each call, so total reconstruction is Σ pos over the chunk
        // boundaries. Fixed chunks make that ~N²/2C; growing chunks make it ~N.
        let total_reconstructed = |n: usize, div: usize| {
            let (mut pos, mut acc) = (0usize, 0usize);
            while pos < n {
                acc += pos; // this call re-derives the `pos` rows already cached
                pos += prefill_chunk(pos, div).min(n - pos);
            }
            acc
        };
        let n = 8192;
        let fixed = total_reconstructed(n, 0);
        let adaptive = total_reconstructed(n, 4);
        // Fixed chunking is the quadratic: ~N²/(2·64) = ~524k rows for N=8192.
        assert!(fixed > 500_000, "fixed-chunk reconstruction should be quadratic, got {fixed}");
        // Adaptive keeps it within a small constant factor of N.
        assert!(adaptive < 8 * n, "adaptive reconstruction should be ~linear, got {adaptive}");
        assert!(fixed / adaptive.max(1) > 8, "expected a large reduction, got {fixed} vs {adaptive}");
    }

    #[test]
    fn prefix_cache_is_inert_until_given_a_budget() -> Result<(), Error> {
        // Default (unset) must be exactly the historical cold start.
        let dir = tiny_dir("prefixoff")?;
        let model = Model::load(&dir)?;
        let prompt: Vec<i32> = (0..80).map(|k| (k * 3 + 1) % 32).collect();
        let mut off = PrefixCache::new(0);
        let mut seq = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&prompt, &mut seq, 0)?;
        off.insert(&prompt, &seq);
        assert!(off.entries.is_empty(), "a disabled cache must store nothing");
        assert!(off.lookup(&prompt).is_none(), "a disabled cache must never hit");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefix_cache_seeded_prefill_matches_cold_prefill() -> Result<(), Error> {
        // The correctness claim: seeding from a cached prefix must produce exactly
        // what prefilling those tokens would have. Two prompts share an 80-token
        // head and diverge after it — the classic shared-system-prompt shape.
        let dir = tiny_dir("prefixhit")?;
        let model = Model::load(&dir)?;
        let shared: Vec<i32> = (0..80).map(|k| (k * 3 + 1) % 32).collect();
        let mut first = shared.clone();
        first.extend_from_slice(&[5, 6, 7]);
        let mut second = shared.clone();
        second.extend_from_slice(&[9, 1, 2]);

        // Cold reference for `second`.
        let want = {
            let mut s = SeqKv::new(&model.cfg);
            model.forward_prefill_seq(&second, &mut s, 0)?
        };

        // Warm `first`, then serve `second` from the shared head.
        let mut cache = PrefixCache::new(64 * 1024 * 1024);
        let mut s1 = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&first, &mut s1, 0)?;
        cache.insert(&first, &s1);

        let (mut s2, pos) = cache.lookup(&second).ok_or_else(|| Error::Format("expected a hit".into()))?;
        assert_eq!(pos, shared.len(), "the shared head is the reusable prefix");
        assert_eq!(s2.len(), pos, "seeded cache holds exactly the shared positions");
        let got = model.forward_prefill_seq(&second[pos..], &mut s2, pos)?;

        // Compare the prompt's final position — the row both paths share.
        let vocab = model.cfg.vocab as usize;
        let (g, w) = (&got[got.len() - vocab..], &want[want.len() - vocab..]);
        for (i, (a, b)) in g.iter().zip(w).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "logit {i} differs after a prefix-cache hit");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefix_cache_never_serves_a_prompt_it_does_not_match() -> Result<(), Error> {
        // Entries are matched by tokens, not a hash, so an unrelated prompt can
        // never be seeded from another's KV — the failure mode that would corrupt
        // output silently.
        let dir = tiny_dir("prefixmiss")?;
        let model = Model::load(&dir)?;
        let a: Vec<i32> = (0..80).map(|k| (k * 3 + 1) % 32).collect();
        let b: Vec<i32> = (0..80).map(|k| (k * 7 + 5) % 32 + 1).collect();
        let mut cache = PrefixCache::new(64 * 1024 * 1024);
        let mut s = SeqKv::new(&model.cfg);
        model.forward_prefill_seq(&a, &mut s, 0)?;
        cache.insert(&a, &s);
        // `b` shares no leading token with `a`, so there is nothing to reuse.
        assert!(cache.lookup(&b).is_none(), "a non-matching prompt must miss");
        // And a hit never consumes the whole prompt — prefill must still run at
        // least one position to produce the logits the first token comes from.
        let (_, n) = cache.lookup(&a).ok_or_else(|| Error::Format("self-lookup should hit".into()))?;
        assert_eq!(n, a.len() - 1, "a hit always leaves at least one position to prefill");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefix_cache_evicts_to_stay_within_budget() -> Result<(), Error> {
        // The budget is the whole point: KV is ~175 KiB/token at real shapes, so
        // an unbounded cache would be an OOM. Insert past a tiny budget and the
        // least-recently-used entries must go.
        let dir = tiny_dir("prefixevict")?;
        let model = Model::load(&dir)?;
        let mk = |seed: i32| -> Vec<i32> { (0..80).map(|k| ((k * 3 + seed) % 31) + 1).collect() };
        let one_bytes = {
            let p = mk(1);
            let mut s = SeqKv::new(&model.cfg);
            model.forward_prefill_seq(&p, &mut s, 0)?;
            s.bytes()
        };
        // Room for about two entries.
        let mut cache = PrefixCache::new(one_bytes * 2 + one_bytes / 2);
        for seed in [1, 5, 9, 13] {
            let p = mk(seed);
            let mut s = SeqKv::new(&model.cfg);
            model.forward_prefill_seq(&p, &mut s, 0)?;
            cache.insert(&p, &s);
        }
        assert!(cache.used <= cache.budget, "used {} over budget {}", cache.used, cache.budget);
        assert!(!cache.entries.is_empty(), "eviction must not empty the cache");
        assert!(cache.entries.len() < 4, "some entries must have been evicted");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn every_chunk_schedule_produces_identical_logits() -> Result<(), Error> {
        // The guarantee the whole optimization rests on: chunk size is a pure
        // work/latency trade and cannot move a single bit of the output. Each
        // token attends exactly its causal prefix regardless of where the chunk
        // boundaries fall, so prefilling the same prompt under different
        // schedules must give bit-identical logits.
        let dir = tiny_dir("chunksched")?;
        let model = Model::load(&dir)?;
        let prompt: Vec<i32> = (0..200).map(|k| (k * 7 + 3) % 32).collect();

        let prefill_with = |div: usize| -> Result<Vec<f32>, Error> {
            let mut seq = SeqKv::new(&model.cfg);
            let mut pos = 0usize;
            let mut last = Vec::new();
            while pos < prompt.len() {
                let end = (pos + prefill_chunk(pos, div)).min(prompt.len());
                last = model.forward_prefill_seq(&prompt[pos..end], &mut seq, pos)?;
                pos = end;
            }
            Ok(last)
        };

        // Whole-prompt in one shot is the reference.
        let want = {
            let mut seq = SeqKv::new(&model.cfg);
            model.forward_prefill_seq(&prompt, &mut seq, 0)?
        };
        let vocab = model.cfg.vocab as usize;
        for div in [0usize, 2, 4, 8, 16] {
            let got = prefill_with(div)?;
            // `forward_prefill_seq` returns logits for every position in the call,
            // so the buffer width tracks the *last chunk's* length and legitimately
            // differs between schedules. The prompt's final position is the row all
            // schedules share, and it must be bit-exact.
            let (g, w) = (&got[got.len() - vocab..], &want[want.len() - vocab..]);
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "div={div}: logit {i} differs");
            }
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The engine must actually publish telemetry, not just own a struct.
    ///
    /// `PlanOptimizer::snapshot` and `BubbleTuner::ewma_snapshot` both documented
    /// themselves as existing "for /metrics" while no such handler and no
    /// publishing path existed. This asserts the seam end to end: after real
    /// decode steps, the handle-side snapshot carries this engine's work.
    #[test]
    fn engine_publishes_telemetry_for_metrics() -> Result<(), Error> {
        let dir = tiny_dir("metrics_pub")?;
        let (handle, join) = spawn_tuned(Model::load_streaming_ecache(&dir, true, 8 << 20)?, 4, false, Some(0))?;
        // Before any tick, the snapshot is all zeros — the honest answer.
        assert_eq!(handle.telemetry().steps, 0, "no steps published before the first tick");

        let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest {
            prompt: vec![1i32, 5, 9, 2],
            max_new: 6,
            sampler: Sampler::new(0.0, 0.9, 1),
            out: tx,
            priority: Priority::Normal,
            class: peregrine_model::TokenClass::Prose,
        })?;
        let mut rx = rx;
        let mut produced = 0usize;
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                EngineOut::Token(_) => produced += 1,
                EngineOut::Error(e) => return Err(Error::Format(e)),
            }
        }
        let t = handle.telemetry();
        drop(handle);
        // The engine thread's join result carries no value and a panic there
        // would already have failed the test; ignore it explicitly by matching
        // rather than with `let _ =`, which the audit's [B] section flags.
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        std::fs::remove_dir_all(&dir)?;

        assert!(produced > 0, "sanity: the request decoded something");
        assert!(t.steps > 0, "the engine must publish its step count; got {}", t.steps);
        // Lane timings come from the model, so a non-zero value proves the
        // publish path reaches `Model`, not just that a counter incremented.
        assert!(
            t.lane_last.reduce_us + t.lane_last.io_us + t.lane_last.cpu_us > 0,
            "published lane timings must be the model's, not a default: {:?}",
            t.lane_last
        );
        Ok(())
    }

    #[test]
    fn a_hedged_tree_changes_no_token_and_actually_branches() -> Result<(), Error> {
        // `COLI_DRAFT_TREE` stops the two draft sources being alternatives:
        // instead of prompt-lookup winning and the head's chain being discarded
        // unseen, both are verified in one forward and whichever the model's own
        // argmax follows commits. The greedy-identity rule is unchanged, so the
        // served stream must be unchanged — and this is the path where that is
        // least obvious, because an accepted branch leaves a *non-contiguous*
        // set of cache slots that has to be gathered rather than truncated.
        //
        // The prompt repeats a run so prompt-lookup fires; the head is a random
        // tiny fixture, so the two disagree and the hedge is genuinely taken.
        let dir = tiny_dir("hedge_tree")?;
        let prompts = [vec![1i32, 5, 9, 2, 7, 1, 5, 9, 2], vec![3i32, 8, 4, 3, 8, 4]];
        let n = 32usize;

        let run = |spec: SpecOverride| -> Result<(Vec<Vec<u32>>, EngineTelemetry), Error> {
            let (handle, join) = spawn_spec(Model::load(&dir)?, 8, false, spec)?;
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt: p.clone(),
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1),
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            let t = handle.telemetry();
            drop(handle);
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok((out, t))
        };

        // An independent decode, not the engine with the knob off.
        let want: Vec<Vec<u32>> = {
            let m = Model::load(&dir)?;
            prompts.iter().map(|p| ref_decode(&m, p, n)).collect::<Result<_, Error>>()?
        };

        let base = SpecOverride { depth: Some(4), ngram: Some(3), ..SpecOverride::default() };
        let (off, t_off) = run(SpecOverride { tree: Some(false), ..base })?;
        assert_eq!(off, want, "the un-hedged engine already disagrees with a plain greedy decode");
        assert_eq!(t_off.spec_trees, 0, "no trees may be built with the knob off");

        let (on, t_on) = run(SpecOverride { tree: Some(true), ..base })?;
        assert!(on.iter().all(|s| !s.is_empty()), "COLI_DRAFT_TREE=1: nothing generated");
        assert_eq!(on, want, "COLI_DRAFT_TREE=1 changed the served token stream");
        // Without this the test would pass on a run where the hedge never fired
        // and no tree row was ever built — which is most of what could go wrong.
        assert!(t_on.spec_trees > 0, "the two sources must actually have disagreed; trees=0");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn union_depth_cap_prices_rows_and_never_stops_progress() {
        // The engine-level test above can only prove the ceiling changes no
        // token — on a resident fixture `ecache_stats` is `None`, so the gate is
        // inert and every cap passes trivially. This is where the policy itself
        // is exercised, the same way `draft_depth_for` is.
        //
        // "No opinion" cases: off, nothing decoding, nothing measured yet.
        assert_eq!(union_depth_cap(0, 8.0, 4), usize::MAX, "cap 0 is off");
        assert_eq!(union_depth_cap(1000, 8.0, 0), usize::MAX, "no sequences, no opinion");
        assert_eq!(union_depth_cap(1000, 0.0, 4), usize::MAX, "no measurement yet, no opinion");
        assert_eq!(union_depth_cap(1000, f64::NAN, 4), usize::MAX, "a NaN EWMA must not become a cap");

        // 800 entries at 8 per row over 4 sequences = 25 rows each → depth 24
        // (one of those rows is the committed token, which is not a draft).
        assert_eq!(union_depth_cap(800, 8.0, 4), 24);
        // Halving the budget: 12.5 rows each, truncated to 12 → depth 11. Rows
        // are whole, so the answer is floor-then-subtract and not half of 24.
        assert_eq!(union_depth_cap(400, 8.0, 4), 11);
        // Doubling the batch cuts it again: the budget is per *tick*, so more
        // sequences means each may speculate less. 6.25 rows → depth 5.
        assert_eq!(union_depth_cap(400, 8.0, 8), 5);
        // Costlier rows buy fewer of them — same 12.5 rows by another route.
        assert_eq!(union_depth_cap(800, 16.0, 4), 11);

        // The floor, which is the case that matters: a budget affording exactly
        // one row per sequence permits depth 0 — speculation stops, decoding
        // does not.
        assert_eq!(union_depth_cap(32, 8.0, 4), 0, "one row each: no drafts, still decodes");
        assert_eq!(union_depth_cap(8, 8.0, 4), 0, "a quarter of a row each must not underflow");
        assert_eq!(union_depth_cap(1, 1e9, 64), 0, "an absurd ceiling still yields a legal depth");
    }

    #[test]
    fn the_union_ceiling_is_depth_only_and_changes_no_token() -> Result<(), Error> {
        // `COLI_SPEC_UNION_MAX` is the cost-side twin of `COLI_SPEC_CONF`, and
        // it inherits the same hard promise: it may change how many rows are
        // *proposed* and never which token is *emitted*. A ceiling that moved a
        // token would be a correctness bug wearing a performance knob's name.
        //
        // Swept from "unbound" to "so tight nothing may draft", because the
        // interesting failure is at the tight end — a cap that pushed depth
        // negative, or that refused to decode at all rather than merely
        // refusing to speculate.
        let dir = tiny_dir("union_cap")?;
        let prompts = [vec![1i32, 5, 9, 2], vec![3i32, 8, 4]];
        let n = 10usize;

        let run = |cap: u64| -> Result<(Vec<Vec<u32>>, EngineTelemetry), Error> {
            let (handle, join) = spawn_spec(
                Model::load(&dir)?,
                8,
                false,
                SpecOverride { depth: Some(4), union_max: Some(cap), ..SpecOverride::default() },
            )?;
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt: p.clone(),
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1),
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            let t = handle.telemetry();
            drop(handle);
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok((out, t))
        };

        // An independent decode, not the engine with the knob off.
        let want: Vec<Vec<u32>> = {
            let m = Model::load(&dir)?;
            prompts.iter().map(|p| ref_decode(&m, p, n)).collect::<Result<_, Error>>()?
        };
        for cap in [0u64, 1, 8, 64, 100_000] {
            let (got, t) = run(cap)?;
            assert!(got.iter().all(|s| !s.is_empty()), "cap {cap}: nothing generated");
            assert_eq!(got, want, "COLI_SPEC_UNION_MAX={cap} changed the served token stream");
            if cap == 0 {
                assert_eq!(t.spec_union_stops, 0, "the ceiling is off at 0 and must never bite");
            }
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn queue_wait_is_measured_and_only_counts_admitted_requests() -> Result<(), Error> {
        // The gap this closes: `bench-serve-lanes.py` times whole requests and
        // `bench-serve-gaps.py` times inter-token gaps, and both start counting
        // once a request is *already being served*. Time spent waiting to be
        // admitted was therefore indistinguishable from slow decode — which, on
        // a server that sheds at `COLI_QUEUE_DEPTH`, is exactly the distinction
        // between "at capacity" and "over capacity".
        //
        // `max_batch = 1` forces the queueing: the second and third requests
        // cannot be admitted until the first retires.
        let dir = tiny_dir("queue_wait")?;
        let (handle, join) = spawn_tuned(Model::load(&dir)?, 1, false, Some(0))?;
        let mut rxs = Vec::new();
        for p in [vec![1i32, 5, 9, 2], vec![3i32, 8, 4], vec![7i32, 1, 6]] {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: p,
                max_new: 6,
                sampler: Sampler::new(0.0, 0.9, 1),
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            rxs.push(rx);
        }
        for mut rx in rxs {
            while let Some(msg) = rx.blocking_recv() {
                if let EngineOut::Error(e) = msg {
                    return Err(Error::Format(e));
                }
            }
        }
        let t = handle.telemetry();
        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        // Counted per admission, not per submit: a request refused at the door
        // never waited for anything, and averaging a refusal in as a zero would
        // flatter the mean exactly when the server is most overloaded.
        assert_eq!(t.queue_admits, 3, "every admitted request must be counted once");
        assert!(
            t.queue_wait_max_us >= t.queue_wait_us / t.queue_admits,
            "the max must bound the mean: max={} mean={}",
            t.queue_wait_max_us,
            t.queue_wait_us / t.queue_admits
        );
        // With `max_batch = 1` at least one request demonstrably waited behind
        // another, so a counter stuck at zero is a wiring failure, not a fast
        // machine.
        assert!(t.queue_wait_max_us > 0, "serialized admissions must show a non-zero wait");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn decode_counters_are_the_numerator_the_ratio_needs() -> Result<(), Error> {
        // `/metrics` has always published the denominator — `ecache.hits +
        // misses` is every routed-expert entry the streaming lane resolved —
        // and never what those reads bought. These two are that numerator, so
        // what they have to be is *exact*: `tokens_emitted` must equal the
        // tokens the clients actually received, not the tokens the engine
        // decoded, or every ratio taken against it is quietly optimistic about
        // work thrown away at a stop token or a budget edge.
        //
        // And `rows / tokens` must read 1.0 with speculation off, since that is
        // the baseline the speculated figure is compared against.
        let dir = tiny_dir("decode_counters")?;
        let prompts = [vec![1i32, 5, 9, 2], vec![3i32, 8, 4]];
        let n = 6usize;
        let (handle, join) = spawn_tuned(Model::load(&dir)?, 8, false, Some(0))?;
        let mut rxs = Vec::new();
        for p in &prompts {
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: p.clone(),
                max_new: n,
                sampler: Sampler::new(0.0, 0.9, 1),
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            rxs.push(rx);
        }
        let mut received = 0u64;
        for mut rx in rxs {
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    EngineOut::Token(_) => received += 1,
                    EngineOut::Error(e) => return Err(Error::Format(e)),
                }
            }
        }
        let t = handle.telemetry();
        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        assert!(received > 0, "sanity: something decoded");
        // Not `== received`: telemetry publishes at the *end* of a tick, and a
        // client can be holding the last tick's tokens before that publish
        // lands, so an equality here is a race that passes alone and fails in a
        // loaded suite. What is true at every tick boundary is asserted
        // instead — an upper bound (no token is counted that no client got)
        // and the exact relation below.
        assert!(t.tokens_emitted > 0, "the counter must move at all");
        assert!(
            t.tokens_emitted <= received,
            "counted {} tokens but clients got {received} — the counter is inventing work",
            t.tokens_emitted
        );
        // Unspeculated, every token costs exactly one decode row *except* the
        // first of each request, which is sampled from the prefill's last
        // position and rides rows that were going to be forwarded anyway. So
        // `rows / tokens` sits slightly below 1.0 on short requests and
        // approaches 1.0 on long ones — worth knowing before reading it as a
        // speculation figure, where the same ratio rises above 1.
        assert_eq!(
            t.decode_rows + prompts.len() as u64,
            t.tokens_emitted,
            "unspeculated: one decode row per token, minus the free first token of each request"
        );
        // This is the assertion that would have caught the counter as first
        // written: it missed the token each request emits at promotion, which
        // reads as a 20% understatement of tokens-per-expert-read on short
        // requests — a bias in the flattering direction.
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prompt_lookup_drafts_do_not_change_the_served_token_stream() -> Result<(), Error> {
        // Same contract as every other draft source, and the reason this one is
        // worth having: it proposes tokens without a model call at all, so on
        // the streaming track it is the only source that does not first pay
        // ~300 MB of MTP expert reads per draft step. It has to be exactly as
        // output-neutral as the head it bypasses.
        //
        // The prompt deliberately repeats a run, so the suffix match fires:
        // without an actual match the drafter is silent and this test would be
        // measuring nothing, which is what the `ngram_proposed > 0` assertion
        // below refuses to allow.
        let dir = tiny_dir("ngram_engine")?;
        let prompts = [vec![1i32, 5, 9, 2, 7, 1, 5, 9, 2], vec![3i32, 8, 4, 3, 8, 4]];
        // Long enough that suffix repeats have somewhere to land: `draft_depth_for`
        // clamps to `budget_left - 1`, so the last couple of positions never
        // draft at all and a short run can spend its whole match window there.
        let n = 32usize;

        let run = |spec: SpecOverride| -> Result<(Vec<Vec<u32>>, EngineTelemetry), Error> {
            let (handle, join) = spawn_spec(Model::load(&dir)?, 8, false, spec)?;
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt: p.clone(),
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1),
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            let t = handle.telemetry();
            drop(handle);
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok((out, t))
        };

        // An independent decode, not the engine with the knob off.
        let want: Vec<Vec<u32>> = {
            let m = Model::load(&dir)?;
            prompts.iter().map(|p| ref_decode(&m, p, n)).collect::<Result<_, Error>>()?
        };

        let (got, t) = run(SpecOverride { depth: Some(4), ngram: Some(3), ..SpecOverride::default() })?;
        assert!(got.iter().all(|t| !t.is_empty()), "COLI_DRAFT_NGRAM=3: nothing generated");
        assert_eq!(got, want, "COLI_DRAFT_NGRAM=3 changed the served token stream");
        assert!(t.ngram_proposed > 0, "the repeated prompt must produce suffix matches; proposed=0");

        // Off is the historical path, and must land in the same place.
        let (off, t_off) = run(SpecOverride { depth: Some(4), ngram: Some(0), ..SpecOverride::default() })?;
        assert_eq!(off, want, "the un-lookahead engine already disagrees with a plain greedy decode");
        assert_eq!(t_off.ngram_proposed, 0, "COLI_DRAFT_NGRAM=0 must propose nothing");
        Ok(())
    }

    #[test]
    fn prompt_lookup_speculates_without_an_mtp_head() -> Result<(), Error> {
        // The capability the n-gram source adds beyond cost: it needs no head
        // in the checkpoint. A hybrid container converted without `--mtp` could
        // not speculate at all before — `draft_depth_for` returns 0 without one
        // — and now it can, provided the recurrent rollback is enabled, because
        // an n-gram draft row advances the GDN state exactly like any other.
        let dir = std::env::temp_dir().join(format!("peregrine_hyb_nomtp_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        peregrine_model::testkit::build_tiny_hybrid_model(&dir, 0x5EC)?; // no MTP head
        let prompt = vec![1i32, 5, 9, 2, 7, 1, 5, 9, 2];
        let n = 8usize;

        let m = Model::load(&dir)?;
        assert!(!m.has_mtp(), "this fixture must have no head, or the test proves nothing");
        let want = ref_decode(&m, &prompt, n)?;
        drop(m);

        let (handle, join) = spawn_spec(
            Model::load(&dir)?,
            4,
            false,
            SpecOverride { depth: Some(4), ngram: Some(3), gdn: Some(true), ..SpecOverride::default() },
        )?;
        let (tx, mut rx) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest {
            prompt: prompt.clone(),
            max_new: n,
            sampler: Sampler::new(0.0, 0.9, 1),
            out: tx,
            priority: Priority::Normal,
            class: peregrine_model::TokenClass::Prose,
        })?;
        let mut got = Vec::new();
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                EngineOut::Token(t) => got.push(t),
                EngineOut::Error(e) => return Err(Error::Format(e)),
            }
        }
        let t = handle.telemetry();
        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        assert_eq!(got, want, "prompt-lookup changed the stream on a headless checkpoint");
        assert!(t.ngram_proposed > 0, "a headless checkpoint must still draft through prompt-lookup");
        assert_eq!(t.spec_proposed, 0, "there is no MTP head, so nothing may be drafted by one");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn recurrent_speculation_does_not_change_the_served_token_stream() -> Result<(), Error> {
        // The same contract as `speculation_does_not_change_the_served_token_stream`,
        // on the arch that could not honour it until now. A hybrid's context is
        // a point state: the verify forward folds every draft into it and
        // `truncate` cannot take them back out, which is why `COLI_SPEC_GDN`
        // exists and why this test is the thing that licenses it.
        //
        // Two ragged prompts, so per-sequence draft blocks differ in width and
        // the rollback runs on some sequences and not others in the same tick.
        let dir = hybrid_mtp_dir("spec")?;
        let prompts = [vec![1i32, 5, 9, 2], vec![3i32, 8, 4]];
        let n = 8usize;

        let run_engine = |spec: SpecOverride| -> Result<(Vec<Vec<u32>>, EngineTelemetry), Error> {
            let (handle, join) = spawn_spec(Model::load(&dir)?, 8, false, spec)?;
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt: p.clone(),
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1), // greedy: the case speculation is exact for
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            let t = handle.telemetry();
            drop(handle);
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok((out, t))
        };

        // An independent decode, not the engine with the knob off — the same
        // reasoning the KV-arch twin records: a reference sharing the path it
        // checks agrees with it even when both are wrong.
        let want: Vec<Vec<u32>> = {
            let m = Model::load(&dir)?;
            prompts.iter().map(|p| ref_decode(&m, p, n)).collect::<Result<_, Error>>()?
        };

        // Off: the historical behaviour on this arch, and the proof that the
        // fixture and the reference agree before speculation is involved.
        let (off, t_off) = run_engine(SpecOverride { depth: Some(4), gdn: Some(false), ..SpecOverride::default() })?;
        assert_eq!(off, want, "the hybrid engine disagrees with a plain greedy decode before speculation");
        assert_eq!(t_off.spec_proposed, 0, "COLI_SPEC_GDN off must leave a recurrent arch un-speculated");
        assert_eq!(t_off.gdn_snapshot_bytes, 0, "nothing should be copied when speculation is off");

        // On: same stream, and the mechanism has to have actually run.
        let (on, t_on) = run_engine(SpecOverride { depth: Some(4), gdn: Some(true), ..SpecOverride::default() })?;
        assert!(on.iter().all(|t| !t.is_empty()), "COLI_SPEC_GDN=1: nothing generated");
        assert_eq!(on, want, "COLI_SPEC_GDN=1 changed the served token stream");
        // A speculation test that passes because nothing speculated is not a
        // test — the same rule the DSA arm is held to.
        assert!(t_on.spec_proposed > 0, "COLI_SPEC_GDN=1 must actually draft; proposed=0");
        assert!(t_on.gdn_snapshot_bytes > 0, "drafting on a recurrent arch must snapshot its state");

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn speculation_does_not_change_the_served_token_stream() -> Result<(), Error> {
        // **The contract.** Speculation buys wall clock, never different
        // output: a draft is accepted only where it matches the model's own
        // argmax, so a greedy request must emit exactly what it emitted
        // without speculation — same tokens, same count, same order.
        //
        // Two requests of different lengths so the batch is ragged and the
        // per-sequence draft blocks are not all the same width.
        let dir = tiny_dir("spec_engine")?;
        let prompts = [vec![1i32, 5, 9, 2], vec![3i32, 8, 4]];
        let n = 8usize;

        let run_engine = |depth: usize| -> Result<Vec<Vec<u32>>, Error> {
            let (handle, join) = spawn_tuned(Model::load(&dir)?, 8, false, Some(depth))?;
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt: p.clone(),
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1), // greedy: the case speculation is exact for
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            drop(handle);
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok(out)
        };

        // The reference is an *independent* decode, not the engine with the knob
        // off. Comparing on-vs-off only proves the two agree — and when this
        // loop was first written both were equally wrong (a duplicated first
        // token, because `next_tok` has already been emitted and was re-sent),
        // which on-vs-off passed and `engine_batches_and_matches_reference`
        // caught. A speculation test whose reference shares the code path it is
        // testing is not a test.
        let want: Vec<Vec<u32>> = {
            let m = Model::load(&dir)?;
            prompts.iter().map(|p| ref_decode(&m, p, n)).collect::<Result<_, Error>>()?
        };
        for depth in [0usize, 1, 4] {
            let got = run_engine(depth)?;
            assert!(got.iter().all(|t| !t.is_empty()), "COLI_DRAFT={depth}: nothing generated");
            assert_eq!(got, want, "COLI_DRAFT={depth} changed the served token stream");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The engine half of `COLI_DRAFT_SAMPLED`: a temperature > 0 request must
    /// actually draft, actually verify, and still serve a complete stream.
    ///
    /// Deliberately **not** an equality assertion against unspeculated decode —
    /// there is no such equality to assert, and writing one would either fail or
    /// (worse) pass by the drafts never being taken. Rejection sampling draws
    /// two uniforms per draft where plain decode draws one per token, so the RNG
    /// stream diverges the moment a draft exists. What must hold is that the
    /// stream is complete, in-vocabulary, and reproducible from its seed.
    #[test]
    fn sampled_speculation_serves_a_complete_reproducible_stream() -> Result<(), Error> {
        let dir = tiny_dir("spec_sampled_engine")?;
        let prompt = vec![1i32, 5, 9, 2];
        let n = 8usize;

        let run_engine = |spec: SpecOverride| -> Result<Vec<u32>, Error> {
            let (handle, join) = spawn_spec(Model::load(&dir)?, 4, false, spec)?;
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: prompt.clone(),
                max_new: n,
                // Temperature > 0: the case the default path refuses to draft for.
                sampler: Sampler::new(0.8, 0.95, 4242),
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            drop(handle);
            let mut toks = Vec::new();
            let mut rx = rx;
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    EngineOut::Token(t) => toks.push(t),
                    EngineOut::Error(e) => return Err(Error::Format(e)),
                }
            }
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok(toks)
        };

        let on = SpecOverride { depth: Some(4), sampled: Some(true), conf: None, gdn: None, ngram: None, union_max: None, tree: None };
        let got = run_engine(on)?;
        assert!(!got.is_empty(), "sampled speculation generated nothing");
        assert!(got.len() <= n, "served {} tokens for max_new {n}", got.len());
        let vocab = Model::load(&dir)?.cfg.vocab as u32;
        assert!(got.iter().all(|&t| t < vocab), "a served token is out of vocabulary: {got:?}");

        // Same seed, same knobs, same stream: the accept path must consume the
        // RNG deterministically, or a seeded request is not reproducible.
        assert_eq!(run_engine(on)?, got, "sampled speculation is not reproducible from its seed");

        // And the knob is a knob: with it off, the same request takes the
        // historical one-row path, whose stream differs precisely because the
        // RNG is consumed differently.
        let off = run_engine(SpecOverride { depth: Some(4), sampled: Some(false), conf: None, gdn: None, ngram: None, union_max: None, tree: None })?;
        assert!(!off.is_empty(), "the unspeculated path generated nothing");
        assert_ne!(
            off, got,
            "sampled speculation must consume the RNG differently from plain decode — \
             identical streams mean no draft was ever taken and the test proves nothing"
        );

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The confidence floor's one hard promise: whatever `COLI_SPEC_CONF` is
    /// set to, a greedy request serves exactly the tokens it would serve with
    /// the floor off. The floor decides how *deep* a draft goes; `accept_run`
    /// alone decides what is emitted, so any divergence here means the gate
    /// leaked past depth into acceptance.
    #[test]
    fn the_confidence_floor_never_changes_a_greedy_stream() -> Result<(), Error> {
        let dir = tiny_dir("spec_conf_engine")?;
        let prompt = vec![1i32, 5, 9, 2];
        let n = 8usize;

        let run_engine = |spec: SpecOverride| -> Result<Vec<u32>, Error> {
            let (handle, join) = spawn_spec(Model::load(&dir)?, 4, false, spec)?;
            let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
            handle.submit(EngineRequest {
                prompt: prompt.clone(),
                max_new: n,
                sampler: Sampler::new(0.0, 0.9, 1), // greedy: sequence-identity is the contract
                out: tx,
                priority: Priority::Normal,
                class: peregrine_model::TokenClass::Prose,
            })?;
            drop(handle);
            let mut toks = Vec::new();
            let mut rx = rx;
            while let Some(msg) = rx.blocking_recv() {
                match msg {
                    EngineOut::Token(t) => toks.push(t),
                    EngineOut::Error(e) => return Err(Error::Format(e)),
                }
            }
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok(toks)
        };

        let base = run_engine(SpecOverride { depth: Some(4), sampled: None, conf: Some(0.0), gdn: None, ngram: None, union_max: None, tree: None })?;
        assert!(!base.is_empty(), "the baseline generated nothing");
        for floor in [0.65f32, 0.999] {
            let gated = run_engine(SpecOverride { depth: Some(4), sampled: None, conf: Some(floor), gdn: None, ngram: None, union_max: None, tree: None })?;
            assert_eq!(gated, base, "a {floor} confidence floor changed a greedy stream");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_sampled_request_never_speculates_unless_asked_to() -> Result<(), Error> {
        // Accepting on argmax makes speculation *sequence*-identical to greedy
        // decoding. At temperature > 0 it is only distribution-preserving, and
        // quietly changing which tokens a sampled request emits is not a
        // speedup — it is a different answer. So by default a sampled request
        // drafts nothing and takes the historical one-row path, in the same
        // batch as greedy ones, with no special casing anywhere else.
        assert_eq!(draft_depth_for(4, true, 0.7, 100, false), 0, "temperature > 0 must not speculate");
        assert_eq!(draft_depth_for(4, true, 0.0, 100, false), 4, "greedy may");
        assert_eq!(draft_depth_for(4, false, 0.0, 100, false), 0, "…but not without an MTP head");
        assert_eq!(draft_depth_for(0, true, 0.0, 100, false), 0, "…nor with the knob off");
        // Never draft past what the request can still emit: a draft accepted
        // beyond `max_new` is work done to produce a token that is discarded.
        assert_eq!(draft_depth_for(4, true, 0.0, 3, false), 2, "budget of 3 leaves room for 2 drafts");
        assert_eq!(draft_depth_for(4, true, 0.0, 1, false), 0, "the last token needs no draft");
        assert_eq!(draft_depth_for(4, true, 0.0, 0, false), 0);

        // `COLI_DRAFT_SAMPLED` opens the temperature > 0 path — and only that
        // one. Every other reason to refuse still refuses, so the knob cannot
        // become a way to draft without an MTP head or past the token budget.
        assert_eq!(draft_depth_for(4, true, 0.7, 100, true), 4, "sampled speculation is opt-in, not impossible");
        assert_eq!(draft_depth_for(4, false, 0.7, 100, true), 0, "…still needs an MTP head");
        assert_eq!(draft_depth_for(0, true, 0.7, 100, true), 0, "…still needs COLI_DRAFT");
        assert_eq!(draft_depth_for(4, true, 0.7, 3, true), 2, "…still respects the budget");
        Ok(())
    }

    #[test]
    fn fused_prefill_emits_the_same_tokens_as_the_two_forward_tick() -> Result<(), Error> {
        // The fusion's whole justification is that it is a *byte* win and not a
        // behaviour change: a prefill chunk and the decode batch go through one
        // forward instead of two, sharing one routed-expert union.
        //
        // So the observable output must not move. Two requests, one long enough
        // to prefill in chunks while the other decodes — which is exactly the
        // mixed tick fusion targets — and the emitted token streams must be
        // identical with the fusion on and off.
        let dir = tiny_dir("fused")?;
        let long: Vec<i32> = (0..80).map(|k| (k * 3 + 1) % 32).collect(); // > PREFILL_CHUNK
        let short: Vec<i32> = vec![1, 5, 9, 2];
        let n = 6usize;

        let run_engine = |fuse: bool| -> Result<Vec<Vec<u32>>, Error> {
            let (handle, join) = spawn_fused(Model::load(&dir)?, 8, fuse)?;
            let mut rxs = Vec::new();
            for prompt in [short.clone(), long.clone()] {
                let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
                handle.submit(EngineRequest {
                    prompt,
                    max_new: n,
                    sampler: Sampler::new(0.0, 0.9, 1),
                    out: tx,
                    priority: Priority::Normal,
                    class: peregrine_model::TokenClass::Prose,
                })?;
                rxs.push(rx);
            }
            drop(handle);
            let mut out = Vec::new();
            for mut rx in rxs {
                let mut toks = Vec::new();
                while let Some(msg) = rx.blocking_recv() {
                    match msg {
                        EngineOut::Token(t) => toks.push(t),
                        EngineOut::Error(e) => return Err(Error::Format(e)),
                    }
                }
                out.push(toks);
            }
            if join.join().is_err() {
                return Err(Error::Format("engine thread panicked".into()));
            }
            Ok(out)
        };

        let plain = run_engine(false)?;
        let fused = run_engine(true)?;
        assert!(plain.iter().all(|t| !t.is_empty()), "the reference run must actually generate");
        assert_eq!(fused, plain, "fusing a prefill chunk into the decode forward changed the token stream");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn engine_chunked_prefill_matches_reference() -> Result<(), Error> {
        // A prompt longer than PREFILL_CHUNK is prefilled in chunks interleaved with
        // decode; the output must still equal the whole-prompt reference decode
        // (chunked prefill is bit-identical — the KV is built the same way).
        let dir = tiny_dir("chunked")?;
        let prompt: Vec<i32> = (0..80).map(|k| (k * 3 + 1) % 32).collect(); // 80 > PREFILL_CHUNK (64)
        let n = 6usize;
        let want = {
            let m = Model::load(&dir)?;
            ref_decode(&m, &prompt, n)?
        };

        let (handle, join) = spawn(Model::load(&dir)?, 8)?;
        let (tx, mut rx) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest { prompt: prompt.clone(), max_new: n, sampler: Sampler::new(0.0, 0.9, 1), out: tx, priority: Priority::Normal, class: peregrine_model::TokenClass::Prose })?;
        let mut got = Vec::new();
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                EngineOut::Token(t) => got.push(t),
                EngineOut::Error(e) => return Err(Error::Format(e)),
            }
        }
        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        assert_eq!(got, want, "chunked-prefill engine output must match whole-prefill reference");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn slow_client_does_not_stall_other_streams() -> Result<(), Error> {
        // Head-of-line regression: one client that never reads its channel must
        // not block the engine thread and freeze every other sequence. With a
        // bounded per-request channel the engine blocked in `send` once the slow
        // reader's queue filled, wedging the whole batch.
        let dir = tiny_dir("slowclient")?;
        let n = 64usize; // far more than the old 64-slot bound could hold unread
        let (handle, join) = spawn(Model::load(&dir)?, 8)?;

        // Submitted first, never drained until the very end.
        let (slow_tx, mut slow_rx) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest {
            prompt: vec![3, 7, 1, 4],
            max_new: n,
            sampler: Sampler::new(0.0, 0.9, 1),
            out: slow_tx,
            priority: Priority::Normal,
            class: peregrine_model::TokenClass::Prose,
        })?;

        // A second stream submitted behind it must still run to completion.
        let (fast_tx, mut fast_rx) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest {
            prompt: vec![3, 7, 1, 4],
            max_new: n,
            sampler: Sampler::new(0.0, 0.9, 1),
            out: fast_tx,
            priority: Priority::Normal,
            class: peregrine_model::TokenClass::Prose,
        })?;
        let mut fast = Vec::new();
        while let Some(msg) = fast_rx.blocking_recv() {
            match msg {
                EngineOut::Token(t) => fast.push(t),
                EngineOut::Error(e) => return Err(Error::Format(e)),
            }
        }
        assert_eq!(fast.len(), n, "the reading client completes while the other never drains");

        // The slow client's tokens were queued all along, not dropped.
        let mut slow = Vec::new();
        while let Some(msg) = slow_rx.blocking_recv() {
            match msg {
                EngineOut::Token(t) => slow.push(t),
                EngineOut::Error(e) => return Err(Error::Format(e)),
            }
        }
        assert_eq!(slow, fast, "the un-drained stream is buffered intact");

        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn engine_respects_max_new_and_empty_prompt() -> Result<(), Error> {
        // max_new caps the stream length; an empty prompt yields zero tokens and a
        // cleanly closed channel (no hang).
        let dir = tiny_dir("caps")?;
        let (handle, join) = spawn(Model::load(&dir)?, 4)?;

        let (tx1, mut rx1) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest { prompt: vec![2, 5, 1], max_new: 3, sampler: Sampler::new(0.0, 0.9, 1), out: tx1, priority: Priority::Normal, class: peregrine_model::TokenClass::Prose })?;
        let mut n1 = 0;
        while let Some(msg) = rx1.blocking_recv() {
            if let EngineOut::Token(_) = msg {
                n1 += 1;
            }
        }
        assert_eq!(n1, 3, "max_new must cap emitted tokens");

        let (tx2, mut rx2) = mpsc::unbounded_channel::<EngineOut>();
        handle.submit(EngineRequest { prompt: vec![], max_new: 5, sampler: Sampler::new(0.0, 0.9, 1), out: tx2, priority: Priority::Normal, class: peregrine_model::TokenClass::Prose })?;
        let mut n2 = 0;
        while let Some(msg) = rx2.blocking_recv() {
            if let EngineOut::Token(_) = msg {
                n2 += 1;
            }
        }
        assert_eq!(n2, 0, "empty prompt must produce no tokens and close the stream");

        drop(handle);
        if join.join().is_err() {
            return Err(Error::Format("engine thread panicked".into()));
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

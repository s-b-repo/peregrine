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

/// Prompt tokens prefilled per engine step for an admitting sequence. Bounding the
/// chunk lets active sequences keep decoding while a new long prompt prefills, so a
/// big admission doesn't stall the batch for its whole prefill. Also the floor for
/// [`prefill_chunk`].
const PREFILL_CHUNK: usize = 64;

/// Divisor for the adaptive prefill chunk (`COLI_PREFILL_CHUNK_DIV`). `0`/unset
/// keeps the historical fixed [`PREFILL_CHUNK`].
fn prefill_chunk_div() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_PREFILL_CHUNK_DIV").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
    })
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
/// long one prefill step blocks the decode batch, so this stays opt-in and the
/// default reproduces the historical fixed chunk exactly.
/// Pure so the schedule is unit-testable — `iotune.rs` documents why a
/// process-wide `OnceLock` for an enable flag makes a feature untestable.
fn prefill_chunk(pos: usize, div: usize) -> usize {
    match div {
        0 => PREFILL_CHUNK,
        d => PREFILL_CHUNK.max(pos / d),
    }
}

/// Byte budget for the cross-request prefix cache (`COLI_PREFIX_CACHE_MB`).
/// `0`/unset disables it, which is the historical behaviour: every request
/// prefills its whole prompt from scratch.
fn prefix_cache_budget() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_PREFIX_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(0)
    })
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

    /// Longest cached prefix of `prompt`, as a seeded cache and its length.
    ///
    /// Never returns the whole prompt: prefill must still run for at least one
    /// position, since that forward is what produces the logits the first token
    /// is sampled from.
    fn lookup(&mut self, prompt: &[i32]) -> Option<(SeqKv, usize)> {
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
        let (i, n) = best?;
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

/// Handle for submitting requests to the engine thread. Cheap to clone and
/// `Send + Sync` (a tokio unbounded sender), so it lives in shared server state.
#[derive(Clone)]
pub struct EngineHandle {
    tx_normal: mpsc::UnboundedSender<EngineRequest>,
    tx_high: mpsc::UnboundedSender<EngineRequest>,
}

impl EngineHandle {
    /// Submit a request at its `priority` — the engine drains high-priority
    /// requests before normal ones each tick. Errors only if the engine thread
    /// has already shut down.
    pub fn submit(&self, req: EngineRequest) -> Result<(), Error> {
        let ch = match req.priority {
            Priority::High => &self.tx_high,
            Priority::Normal => &self.tx_normal,
        };
        ch.send(req).map_err(|send_err| Error::Format(format!("batch engine is not running: {send_err}")))
    }
}

/// Spawn the engine on a dedicated OS thread that owns `model`, batching up to
/// `max_batch` sequences per step. Returns a submit handle and the thread's join
/// handle (the thread exits once every [`EngineHandle`] is dropped and all active
/// sequences finish).
pub fn spawn(model: Model, max_batch: usize) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    let (tx_normal, rx_normal) = mpsc::unbounded_channel::<EngineRequest>();
    let (tx_high, rx_high) = mpsc::unbounded_channel::<EngineRequest>();
    let cap = max_batch.max(1);
    let join = std::thread::Builder::new()
        .name("peregrine-batch".to_string())
        .spawn(move || run(model, rx_normal, rx_high, cap))
        .map_err(|e| Error::Format(format!("spawn batch engine thread: {e}")))?;
    Ok((EngineHandle { tx_normal, tx_high }, join))
}

/// Latency SLA target for adaptive batching, in milliseconds. When set
/// (`COLI_BATCH_SLA_MS=<n>` or via [`spawn_with_sla`] callers), the engine
/// shrinks the working batch cap on p95-latency overrun and grows it back when
/// slack appears. Unset → static `max_batch` (the historical default).
fn batch_sla_ms() -> Option<u64> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<u64>> = OnceLock::new();
    *V.get_or_init(|| std::env::var("COLI_BATCH_SLA_MS").ok().and_then(|s| s.trim().parse::<u64>().ok()).filter(|&n| n > 0))
}

/// Number of decode ticks per prefill tick when the adaptive window is on.
/// `COLI_ADAPTIVE_WINDOW=<n>` (default `1` = the historical every-tick prefill
/// interleave). Larger values let decode run further before yielding to prefill,
/// trading admission latency for decode throughput when the workload is decode-
/// heavy. Purely a scheduling knob — correctness-neutral.
fn adaptive_window_ratio() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_ADAPTIVE_WINDOW")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

/// One in-flight sequence. Invariant at the top of a decode step: `seq.len() ==
/// pos`, and `next_tok` is an already-emitted token to be fed at `pos` next.
struct SeqState {
    seq: SeqKv,
    /// This stream's own routing history — the per-sequence prefetch predictor, so
    /// each concurrent stream prefetches from its own routing (not the cross-sequence
    /// union). Wrapped in a `Mutex` because the batched forward records into it.
    hist: Mutex<RouteHistory>,
    pos: usize,
    next_tok: i32,
    sampler: Sampler,
    out: mpsc::UnboundedSender<EngineOut>,
    produced: usize,
    max_new: usize,
}

/// The engine loop: admit + prefill new requests, then decode all active
/// sequences one batched step at a time until each hits a stop id, its token
/// budget, or a dropped client.
fn run(
    mut model: Model,
    mut rx_normal: mpsc::UnboundedReceiver<EngineRequest>,
    mut rx_high: mpsc::UnboundedReceiver<EngineRequest>,
    max_batch: usize,
) {
    let vocab = model.cfg.vocab as usize;
    let stop_ids = model.cfg.stop_ids.clone();
    let mut active: Vec<SeqState> = Vec::new();
    let mut pending: VecDeque<Prefilling> = VecDeque::new();
    let mut steps = 0usize;
    // Adaptive-batching state. `working_cap` is the current admission ceiling
    // (starts at `max_batch`, shrinks under SLA overrun, grows on slack). EWMA
    // over per-forward wall time drives the adjustment.
    let sla_ms = batch_sla_ms();
    // Resolved once: prefill chunking is a latency/work trade, not a per-tick decision.
    let chunk_div = prefill_chunk_div();
    let mut prefix = PrefixCache::new(prefix_cache_budget());
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
        // the *working* cap so an SLA-shrunk engine backpressures both queues.
        while active.len() + pending.len() < working_cap {
            match rx_high.try_recv() {
                Ok(req) => admit_pending(&model, &mut pending, req, &mut prefix),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        while active.len() + pending.len() < working_cap {
            match rx_normal.try_recv() {
                Ok(req) => admit_pending(&model, &mut pending, req, &mut prefix),
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
                    Ok(req) => {
                        admit_pending(&model, &mut pending, req, &mut prefix);
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
                }
                match rx_normal.try_recv() {
                    Ok(req) => {
                        admit_pending(&model, &mut pending, req, &mut prefix);
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
                }
            }
            if !pending.is_empty() {
                continue;
            }
            let req = recv_priority(idle_rt.as_ref(), &mut rx_high, &mut rx_normal);
            match req {
                Some(req) => {
                    admit_pending(&model, &mut pending, req, &mut prefix);
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
        let win = adaptive_window_ratio();
        if win == 1 || active.is_empty() || steps.is_multiple_of(win as usize) {
            prefill_step(&model, &mut pending, &mut active, vocab, &stop_ids, chunk_div, &mut prefix);
        }
        if active.is_empty() {
            continue; // only prefilling so far (or everything retired) — no decode yet
        }

        // One batched decode step: feed each active sequence's pending token.
        let tokens: Vec<i32> = active.iter().map(|s| s.next_tok).collect();
        let pos_of: Vec<usize> = active.iter().map(|s| s.pos).collect();
        let t_decode = std::time::Instant::now();
        let logits = {
            // Split each SeqState into (KV, history) disjoint borrows so the batched
            // forward records each sequence's *own* routed set into its own history.
            let (mut refs, hists): (Vec<&mut SeqKv>, Vec<&Mutex<RouteHistory>>) = active
                .iter_mut()
                .map(|s| {
                    let SeqState { seq, hist, .. } = s;
                    (seq, &*hist)
                })
                .unzip();
            match model.forward_step_batched(&tokens, &mut refs, &pos_of, Some(&hists)) {
                Ok(l) => l,
                Err(e) => {
                    for s in &active {
                        if s.out.send(EngineOut::Error(e.to_string())).is_err() {
                            peregrine_core::note_advisory_err("batch error forward", &"client already disconnected");
                        }
                    }
                    active.clear();
                    continue;
                }
            }
        };
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
        // experts onto its assigned lane (round-robin) while sampling proceeds.
        for (i, s) in active.iter().enumerate() {
            model.enqueue_seq_prefetch(&s.hist, i);
        }

        // Sample the next token per sequence, emit it, and decide who continues.
        let mut keep: Vec<bool> = Vec::with_capacity(active.len());
        for (i, s) in active.iter_mut().enumerate() {
            let tok = s.sampler.pick(&logits[i * vocab..i * vocab + vocab], -1) as i32;
            s.pos += 1; // the token just fed now occupies its slot
            if stop_ids.contains(&tok) {
                keep.push(false); // stop token is not emitted
                continue;
            }
            let delivered = s.out.send(EngineOut::Token(tok as u32)).is_ok();
            s.produced += 1;
            s.next_tok = tok;
            keep.push(delivered && s.produced < s.max_new);
        }
        let mut idx = 0usize;
        active.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });

        // Periodically migrate the hottest experts into VRAM (heat-ranked
        // residency). Between steps, so it holds the exclusive borrow reheat needs;
        // a no-op without a GPU tier.
        steps += 1;
        if steps.is_multiple_of(REHEAT_EVERY) {
            if let Err(e) = model.reheat() {
                eprintln!("peregrine: reheat failed: {e}");
            }
        }
    }
    // Shutdown: report what the prefix cache absorbed. Silent when it is off, so
    // a default run's output is unchanged.
    if prefix.enabled() {
        eprintln!(
            "[prefix-cache] hits={} tokens_reused={} entries={} resident={:.1} MiB",
            prefix.hits,
            prefix.tokens_saved,
            prefix.entries.len(),
            prefix.used as f64 / (1024.0 * 1024.0)
        );
    }
}

/// Blocking-wait for a request across the two priority channels, biased toward
/// high-priority. Returns `None` when both senders are dropped (shutdown).
fn recv_priority(
    rt: Option<&tokio::runtime::Runtime>,
    rx_high: &mut mpsc::UnboundedReceiver<EngineRequest>,
    rx_normal: &mut mpsc::UnboundedReceiver<EngineRequest>,
) -> Option<EngineRequest> {
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
fn admit_pending(model: &Model, pending: &mut VecDeque<Prefilling>, req: EngineRequest, prefix: &mut PrefixCache) {
    if req.prompt.is_empty() || req.max_new == 0 {
        return; // nothing to generate; dropping req.out closes the stream cleanly
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
    });
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
    prefix: &mut PrefixCache,
) {
    let Some(p) = pending.pop_front() else {
        return;
    };
    // Destructure so the prompt and the KV cache are disjoint borrows — the
    // chunk is then a plain slice instead of a per-step copy.
    let Prefilling { mut seq, prompt, pos, mut sampler, out, max_new } = p;
    let end = (pos + prefill_chunk(pos, chunk_div)).min(prompt.len());
    let chunk = &prompt[pos..end];
    let logits = match model.forward_prefill_seq(chunk, &mut seq, pos) {
        Ok(l) => l,
        Err(e) => {
            if out.send(EngineOut::Error(e.to_string())).is_err() {
                peregrine_core::note_advisory_err("prefill error forward", &"client already disconnected");
            }
            return; // drop this sequence
        }
    };
    let chunk_len = chunk.len();
    if end < prompt.len() {
        // more chunks to go — round-robin with the others
        pending.push_back(Prefilling { seq, prompt, pos: end, sampler, out, max_new });
        return;
    }
    // Prefill complete. Snapshot it before the KV moves into the active set, so
    // the next request sharing this prompt's head starts from here. No-op when
    // the cache is disabled or the prompt is too short to be worth copying.
    prefix.insert(&prompt, &seq);
    // Sample the first token from the last prompt position. An empty chunk would
    // mean an empty prompt, which `admit_pending` rejects.
    let Some(last) = chunk_len.checked_sub(1).map(|c| c * vocab) else {
        return;
    };
    let t0 = sampler.pick(&logits[last..last + vocab], -1) as i32;
    if stop_ids.contains(&t0) {
        return; // first token is a stop → emit nothing
    }
    if out.send(EngineOut::Token(t0 as u32)).is_err() {
        return; // client already gone
    }
    if max_new <= 1 {
        return; // only one token requested
    }
    active.push(SeqState {
        seq,
        hist: Mutex::new(model.new_route_history()),
        pos: prompt.len(),
        next_tok: t0,
        sampler,
        out,
        produced: 1,
        max_new,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_model::argmax;

    fn tiny_dir(tag: &str) -> Result<std::path::PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_batch_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        peregrine_model::testkit::build_tiny_model(&d)?;
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

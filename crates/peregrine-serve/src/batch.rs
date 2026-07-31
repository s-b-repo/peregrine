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
/// big admission doesn't stall the batch for its whole prefill.
const PREFILL_CHUNK: usize = 64;

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
                Ok(req) => admit_pending(&model, &mut pending, req),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        while active.len() + pending.len() < working_cap {
            match rx_normal.try_recv() {
                Ok(req) => admit_pending(&model, &mut pending, req),
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
                        admit_pending(&model, &mut pending, req);
                        break;
                    }
                    Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {}
                }
                match rx_normal.try_recv() {
                    Ok(req) => {
                        admit_pending(&model, &mut pending, req);
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
                    admit_pending(&model, &mut pending, req);
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
            prefill_step(&model, &mut pending, &mut active, vocab, &stop_ids);
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
fn admit_pending(model: &Model, pending: &mut VecDeque<Prefilling>, req: EngineRequest) {
    if req.prompt.is_empty() || req.max_new == 0 {
        return; // nothing to generate; dropping req.out closes the stream cleanly
    }
    model.set_workload_class(req.class);
    pending.push_back(Prefilling {
        seq: SeqKv::new(&model.cfg),
        prompt: req.prompt,
        pos: 0,
        sampler: req.sampler,
        out: req.out,
        max_new: req.max_new,
    });
}

/// Advance the front prefilling sequence by up to `PREFILL_CHUNK` tokens. When its
/// prompt is fully prefilled, sample the first token and move it to `active` (or
/// retire it). Round-robins the queue so no one prefill monopolizes the engine.
fn prefill_step(model: &Model, pending: &mut VecDeque<Prefilling>, active: &mut Vec<SeqState>, vocab: usize, stop_ids: &[i32]) {
    let Some(p) = pending.pop_front() else {
        return;
    };
    // Destructure so the prompt and the KV cache are disjoint borrows — the
    // chunk is then a plain slice instead of a per-step copy.
    let Prefilling { mut seq, prompt, pos, mut sampler, out, max_new } = p;
    let end = (pos + PREFILL_CHUNK).min(prompt.len());
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
    // Prefill complete: sample the first token from the last prompt position.
    // An empty chunk would mean an empty prompt, which `admit_pending` rejects.
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

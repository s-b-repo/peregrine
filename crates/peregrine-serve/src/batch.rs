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

use std::thread::JoinHandle;

use peregrine_core::Error;
use peregrine_model::{Model, Sampler, SeqKv};
use tokio::sync::mpsc;

/// Batched decode steps between heat-ranked VRAM re-selections ([`Model::reheat`]).
/// A no-op without a GPU tier, so it is harmless in CPU-only deployments.
const REHEAT_EVERY: usize = 256;

/// A generation request handed to the engine. The engine prefills `prompt`, then
/// emits sampled token ids on `out` until a stop id, `max_new` tokens, or the
/// client drops `out`. `sampler` carries this request's temperature/nucleus/seed,
/// so each sequence draws from its own RNG stream.
pub struct EngineRequest {
    pub prompt: Vec<i32>,
    pub max_new: usize,
    pub sampler: Sampler,
    pub out: mpsc::Sender<EngineOut>,
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
    tx: mpsc::UnboundedSender<EngineRequest>,
}

impl EngineHandle {
    /// Submit a request; the engine streams tokens on `req.out`. Errors only if
    /// the engine thread has already shut down.
    pub fn submit(&self, req: EngineRequest) -> Result<(), Error> {
        self.tx.send(req).map_err(|_| Error::Format("batch engine is not running".into()))
    }
}

/// Spawn the engine on a dedicated OS thread that owns `model`, batching up to
/// `max_batch` sequences per step. Returns a submit handle and the thread's join
/// handle (the thread exits once every [`EngineHandle`] is dropped and all active
/// sequences finish).
pub fn spawn(model: Model, max_batch: usize) -> Result<(EngineHandle, JoinHandle<()>), Error> {
    let (tx, rx) = mpsc::unbounded_channel::<EngineRequest>();
    let cap = max_batch.max(1);
    let join = std::thread::Builder::new()
        .name("peregrine-batch".to_string())
        .spawn(move || run(model, rx, cap))
        .map_err(|e| Error::Format(format!("spawn batch engine thread: {e}")))?;
    Ok((EngineHandle { tx }, join))
}

/// One in-flight sequence. Invariant at the top of a decode step: `seq.len() ==
/// pos`, and `next_tok` is an already-emitted token to be fed at `pos` next.
struct SeqState {
    seq: SeqKv,
    pos: usize,
    next_tok: i32,
    sampler: Sampler,
    out: mpsc::Sender<EngineOut>,
    produced: usize,
    max_new: usize,
}

/// The engine loop: admit + prefill new requests, then decode all active
/// sequences one batched step at a time until each hits a stop id, its token
/// budget, or a dropped client.
fn run(mut model: Model, mut rx: mpsc::UnboundedReceiver<EngineRequest>, max_batch: usize) {
    let vocab = model.cfg.vocab as usize;
    let stop_ids = model.cfg.stop_ids.clone();
    let mut active: Vec<SeqState> = Vec::new();
    let mut steps = 0usize;

    loop {
        // Idle: block for the next request (or exit when every handle is dropped).
        if active.is_empty() {
            match rx.blocking_recv() {
                Some(req) => admit(&model, &mut active, req, vocab, &stop_ids),
                None => break, // all senders dropped and nothing in flight → shutdown
            }
        }
        // Fill the batch with any already-queued requests (non-blocking).
        while active.len() < max_batch {
            match rx.try_recv() {
                Ok(req) => admit(&model, &mut active, req, vocab, &stop_ids),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break, // drain active, then exit
            }
        }
        if active.is_empty() {
            continue; // admission may have retired everything (immediate stop / one-token request)
        }

        // One batched decode step: feed each active sequence's pending token.
        let tokens: Vec<i32> = active.iter().map(|s| s.next_tok).collect();
        let pos_of: Vec<usize> = active.iter().map(|s| s.pos).collect();
        let logits = {
            let mut refs: Vec<&mut SeqKv> = active.iter_mut().map(|s| &mut s.seq).collect();
            match model.forward_step_batched(&tokens, &mut refs, &pos_of) {
                Ok(l) => l,
                Err(e) => {
                    for s in &active {
                        let _ = s.out.blocking_send(EngineOut::Error(e.to_string()));
                    }
                    active.clear();
                    continue;
                }
            }
        };

        // Sample the next token per sequence, emit it, and decide who continues.
        let mut keep: Vec<bool> = Vec::with_capacity(active.len());
        for (i, s) in active.iter_mut().enumerate() {
            let tok = s.sampler.pick(&logits[i * vocab..i * vocab + vocab], -1) as i32;
            s.pos += 1; // the token just fed now occupies its slot
            if stop_ids.contains(&tok) {
                keep.push(false); // stop token is not emitted
                continue;
            }
            let delivered = s.out.blocking_send(EngineOut::Token(tok as u32)).is_ok();
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

/// Prefill one request's prompt into a fresh per-sequence KV, emit its first
/// sampled token, and (unless it already finished) push it onto `active`.
fn admit(model: &Model, active: &mut Vec<SeqState>, req: EngineRequest, vocab: usize, stop_ids: &[i32]) {
    if req.prompt.is_empty() || req.max_new == 0 {
        return; // nothing to generate; dropping req.out closes the stream cleanly
    }
    let mut seq = SeqKv::new(&model.cfg);
    let logits = match model.forward_prefill_seq(&req.prompt, &mut seq, 0) {
        Ok(l) => l,
        Err(e) => {
            let _ = req.out.blocking_send(EngineOut::Error(e.to_string()));
            return;
        }
    };
    let mut sampler = req.sampler;
    let last = (req.prompt.len() - 1) * vocab;
    let t0 = sampler.pick(&logits[last..last + vocab], -1) as i32;
    if stop_ids.contains(&t0) {
        return; // first token is a stop → emit nothing
    }
    if req.out.blocking_send(EngineOut::Token(t0 as u32)).is_err() {
        return; // client already gone
    }
    if req.max_new <= 1 {
        return; // only one token requested
    }
    active.push(SeqState {
        seq,
        pos: req.prompt.len(),
        next_tok: t0,
        sampler,
        out: req.out,
        produced: 1,
        max_new: req.max_new,
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
            let lg = model.forward_step_batched(&[tok], &mut one, &[pos])?;
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
            let (tx, rx) = mpsc::channel::<EngineOut>(64);
            handle.submit(EngineRequest {
                prompt: prompt.clone(),
                max_new: n,
                sampler: Sampler::new(0.0, 0.9, 1),
                out: tx,
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
        let _ = join.join();

        for (i, o) in outs.iter().enumerate() {
            assert_eq!(o, &want, "batched request {i} must match the reference decode");
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

        let (tx1, mut rx1) = mpsc::channel::<EngineOut>(16);
        handle.submit(EngineRequest { prompt: vec![2, 5, 1], max_new: 3, sampler: Sampler::new(0.0, 0.9, 1), out: tx1 })?;
        let mut n1 = 0;
        while let Some(msg) = rx1.blocking_recv() {
            if let EngineOut::Token(_) = msg {
                n1 += 1;
            }
        }
        assert_eq!(n1, 3, "max_new must cap emitted tokens");

        let (tx2, mut rx2) = mpsc::channel::<EngineOut>(16);
        handle.submit(EngineRequest { prompt: vec![], max_new: 5, sampler: Sampler::new(0.0, 0.9, 1), out: tx2 })?;
        let mut n2 = 0;
        while let Some(msg) = rx2.blocking_recv() {
            if let EngineOut::Token(_) = msg {
                n2 += 1;
            }
        }
        assert_eq!(n2, 0, "empty prompt must produce no tokens and close the stream");

        drop(handle);
        let _ = join.join();
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

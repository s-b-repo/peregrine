//! Top-level GLM-5.2 model: weight loading by the container naming scheme, the
//! per-layer forward loop, and the generate loop. Ports the structure of
//! `model_load` (`c/glm.c:1425-1469`) and `layer_forward_rows` (`c/glm.c:3629`).
//!
//! Experts are held resident (fine for the tiny/oracle model); disk streaming
//! for the 744B model is M2. Absorption/DSA are M5 — attention runs the dense
//! reconstruction path.

use std::sync::Arc;

use parking_lot::Mutex;
use peregrine_core::{Cfg, Context, Error, SafeTensors};
use peregrine_io::{Reactor, WarmCache};

use crate::attention::{mla_attention, mla_attention_batched, AttnWeights, LayerKv};
use crate::concurrent::{default_workers, moe_forward_concurrent, ForwardCtx, EXPERTS_PER_BATCH};
use crate::gpu::{GpuTier, HeatTable};
use crate::math::rmsnorm;
use crate::mlp::{moe_forward, Mlp};
use crate::sample::Sampler;
use crate::weight::QtWeight;

/// Per-layer weights.
struct LayerW {
    in_ln: Vec<f32>,
    post_ln: Vec<f32>,
    q_a: QtWeight,
    q_a_ln: Vec<f32>,
    q_b: QtWeight,
    kv_a: QtWeight,
    kv_a_ln: Vec<f32>,
    kv_b: QtWeight,
    o: QtWeight,
    sparse: bool,
    dense: Option<Mlp>,          // dense layers (i < first_dense)
    router: Vec<f32>,            // [E, hidden] (sparse only)
    router_bias: Vec<f32>,       // [E]
    shared: Option<Mlp>,
    experts: Vec<Mlp>,
}

impl LayerW {
    fn attn(&self) -> AttnWeights<'_> {
        AttnWeights {
            q_a: &self.q_a,
            q_a_ln: &self.q_a_ln,
            q_b: &self.q_b,
            kv_a: &self.kv_a,
            kv_a_ln: &self.kv_a_ln,
            kv_b: &self.kv_b,
            o: &self.o,
        }
    }
}

/// The MTP (multi-token-prediction) head: a one-layer draft model for speculative
/// decode. Present only in checkpoints converted with `--mtp`. The head has no
/// persistent KV — each draft runs a fresh local KV (the hidden state carries the
/// context), and every draft is verified against the main model, so the emitted
/// sequence is identical to greedy decoding.
struct MtpHead {
    layer: LayerW,      // the MTP transformer layer (index n_layers)
    eh_proj: QtWeight,  // [hidden, 2*hidden]: projects concat(embed_norm, hidden_norm)
    enorm: Vec<f32>,    // RMSNorm for the token embedding
    hnorm: Vec<f32>,    // RMSNorm for the incoming hidden state
    mtp_norm: Vec<f32>, // final RMSNorm before lm_head (shared_head.norm)
}

/// A loaded model plus its per-layer KV cache.
pub struct Model {
    pub cfg: Cfg,
    embed: Vec<f32>, // [vocab, hidden], dequantized
    layers: Vec<LayerW>,
    final_norm: Vec<f32>,
    lm_head: QtWeight,
    kv: Vec<LayerKv>,
    /// When set, routed experts are streamed from `st` per layer on demand
    /// instead of held resident — required to run models that exceed RAM
    /// (e.g. the 744B GLM-5.2). `LayerW::experts` is empty in this mode.
    stream_experts: bool,
    /// Stream expert reads via O_DIRECT (bypass the page cache). `true` only when
    /// `COLI_DIRECT` is set, streaming is on, and the shards opened O_DIRECT fds.
    direct: bool,
    /// Retained safetensors index (keeps shard fds open) for streaming reads.
    st: SafeTensors,
    /// The concurrent MoE lane's **pool of io_uring rings** (streaming mode only) —
    /// one per I/O worker thread so N expert reads run in parallel. Separate from
    /// `st`'s ring. Empty when experts are resident. Size via `COLI_IO_RINGS`.
    io_reactors: Vec<Mutex<Reactor>>,
    /// CPU-lane worker count for the concurrent MoE.
    workers: usize,
    /// RAM warm tier: the byte-budgeted `(layer,expert)` cache the I/O lane
    /// consults before streaming, so hot experts stop re-hitting the SSD every
    /// token. `Arc` so the prefetch lane shares it. `Some` only in streaming mode
    /// with a non-zero budget; persists across `reset` (a warm tier is
    /// per-process, not per-sequence).
    ecache: Option<Arc<Mutex<WarmCache>>>,
    /// Per-layer routed-expert history from the last main forward — the prefetch
    /// predictor ("next token routes like this one"). `Some` alongside `prefetch`.
    route_hist: Option<Mutex<Vec<Vec<i32>>>>,
    /// Background prefetch lane: warms the next token's predicted experts into
    /// `ecache` on its own ring, off the critical path. `Some` alongside `ecache`.
    prefetch: Option<PrefetchHandle>,
    /// Optional GPU VRAM expert tier (the 3rd lane). Built only when `COLI_GPU`
    /// is set and the `cuda` backend is available; `None` otherwise.
    gpu: Option<GpuTier>,
    /// Optional MTP head for speculative decode; `None` unless the checkpoint has
    /// the `model.layers.{n_layers}.eh_proj` tensors.
    mtp: Option<MtpHead>,
    /// Routing-frequency accumulator driving heat-ranked VRAM residency; `Some`
    /// only when a GPU tier exists (bumped during the forward, read by `reheat`).
    heat: Option<HeatTable>,
}

/// Per-sequence KV cache: one [`LayerKv`] per layer. Owned by the batching
/// scheduler rather than the [`Model`], so a single resident model can decode
/// many independent sequences concurrently via [`Model::forward_step_batched`].
/// (A paged/block-pooled variant that bounds many-sequence RAM is a follow-up;
/// this per-sequence layout is the correct, bit-identical foundation.)
pub struct SeqKv {
    layers: Vec<LayerKv>,
}

impl SeqKv {
    /// A fresh, empty cache sized for a model with `cfg`'s dimensions.
    pub fn new(cfg: &Cfg) -> SeqKv {
        let (kvl, qkr) = (cfg.kv_lora as usize, cfg.qk_rope as usize);
        SeqKv { layers: (0..cfg.n_layers).map(|_| LayerKv::new(kvl, qkr)).collect() }
    }

    /// Positions cached so far (the sequence length); all layers share it.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |k| k.len)
    }

    /// Whether no positions are cached yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rewind every layer to `new_len` (speculative-decode reject cleanup).
    pub fn truncate(&mut self, new_len: usize) {
        for k in &mut self.layers {
            k.truncate(new_len);
        }
    }
}

/// `MemAvailable` from `/proc/meminfo`, in bytes (0 if unreadable).
fn mem_available_bytes() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return 0 };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

/// Warm-cache byte budget from the environment: `COLI_ECACHE_GB` (GiB float) if
/// set, else 10% of `MemAvailable` capped at 2 GiB. `0` (or an unparseable value)
/// disables the cache. Kept independent of the streaming-vs-resident RAM
/// heuristic so the two knobs don't interfere.
fn ecache_budget_bytes() -> usize {
    const GIB: f64 = (1u64 << 30) as f64;
    if let Ok(v) = std::env::var("COLI_ECACHE_GB") {
        let g: f64 = v.trim().parse().unwrap_or(0.0);
        return (g.max(0.0) * GIB) as usize;
    }
    let avail = mem_available_bytes() as f64;
    (0.10 * avail).min(2.0 * GIB) as usize
}

/// Conservative peak transient RAM the streaming lanes hold at once: up to
/// `io_rings × EXPERTS_PER_BATCH` experts have landing buffers in flight, plus
/// `workers` being reconstructed/computed, each roughly one full expert
/// (`per_expert_bytes`). Used to keep the warm cache from claiming RAM the
/// streaming path needs, so batched prefill/decode can't OOM.
fn stream_transient_reserve(io_rings: usize, workers: usize, per_expert_bytes: usize) -> usize {
    io_rings
        .saturating_mul(EXPERTS_PER_BATCH)
        .saturating_add(workers)
        .saturating_mul(per_expert_bytes)
}

/// Cap a requested warm-cache budget so the cache + the streaming lanes' transient
/// buffers + a safety margin fit in `mem_available` (already net of the resident
/// model, since it is read after load). Returns `min(requested, headroom)`, or `0`
/// when RAM is too tight (cache disabled rather than risking OOM). Pure arithmetic
/// (inputs injected) so the policy is unit-testable without a specific machine.
fn cap_ecache_budget(requested: usize, mem_available: usize, transient_reserve: usize, safety: usize) -> usize {
    let headroom = mem_available.saturating_sub(transient_reserve.saturating_add(safety));
    requested.min(headroom)
}

/// Whether O_DIRECT streaming is requested via `COLI_DIRECT`. Default **off** —
/// direct I/O regresses on page-cache-warm runs and O_DIRECT-unfriendly filesystems.
fn direct_enabled() -> bool {
    matches!(std::env::var("COLI_DIRECT").ok().as_deref(), Some("1") | Some("true"))
}

/// Number of parallel io_uring rings for the streaming I/O lane (`COLI_IO_RINGS`,
/// default 4). More rings = more concurrent expert reads (and parallel dm-crypt on
/// encrypted volumes); `1` restores single-ring behavior. Capped at 16.
fn io_rings() -> usize {
    std::env::var("COLI_IO_RINGS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4)
        .min(16)
}

/// Capacity for the O_DIRECT aligned slab pool: the largest routed-expert region on
/// disk plus alignment slack (the 4096-aligned superset of a region can exceed it by
/// up to two blocks). Scans the safetensors index; a safe default if none found.
fn max_expert_region_bytes(st: &SafeTensors) -> usize {
    let max = st
        .tensors()
        .iter()
        .filter(|t| t.name.contains(".mlp.experts."))
        .map(|t| t.nbytes as usize)
        .max()
        .unwrap_or(32 << 20);
    max + 2 * peregrine_io::ALIGN
}

/// Messages to the background prefetch lane.
enum PrefetchMsg {
    /// Warm these experts into the shared cache (skipping ones already resident).
    Warm(Vec<crate::concurrent::PrefetchItem>),
    /// Barrier: reply once every earlier message has been processed (tests).
    Sync(crossbeam_channel::Sender<()>),
    /// Drain and exit.
    Stop,
}

/// Owns the prefetch lane's channel + thread; joined on `Model` drop.
struct PrefetchHandle {
    tx: crossbeam_channel::Sender<PrefetchMsg>,
    join: Option<std::thread::JoinHandle<()>>,
}

/// The prefetch lane: stream predicted experts into the shared warm cache on this
/// lane's *own* ring (no contention with the critical I/O lane). Best-effort — a
/// failed speculative read is dropped (the real forward will stream it normally).
fn prefetch_worker(mut reactor: Reactor, cache: Arc<Mutex<WarmCache>>, rx: crossbeam_channel::Receiver<PrefetchMsg>, direct: bool) {
    while let Ok(msg) = rx.recv() {
        match msg {
            PrefetchMsg::Warm(items) => {
                for item in items {
                    let key = item.key();
                    if cache.lock().contains(key) {
                        continue; // already warm — don't re-read
                    }
                    if let Ok(slab) = crate::concurrent::prefetch_read(&mut reactor, &item, direct) {
                        let mut c = cache.lock();
                        c.note_prefetch_read();
                        c.insert(key, slab);
                    }
                }
            }
            PrefetchMsg::Sync(reply) => {
                let _ = reply.send(());
            }
            PrefetchMsg::Stop => break,
        }
    }
}

fn load_f32(st: &SafeTensors, name: &str, n: usize) -> Result<Vec<f32>, Error> {
    let mut v = vec![0f32; n];
    st.read_f32(name, &mut v)?;
    Ok(v)
}

/// Load one transformer layer (`model.layers.{i}.*`). In streaming mode the
/// routed experts are left on disk (presence-checked only); otherwise resident.
/// Reused for both the main stack and the MTP head layer.
fn load_layer(st: &SafeTensors, i: usize, cfg: &Cfg, stream_experts: bool) -> Result<LayerW, Error> {
    let d = cfg.hidden as usize;
    let h = cfg.n_heads as usize;
    let (qkh, vh) = (cfg.qk_head as usize, cfg.v_head as usize);
    let (ql, kvl, qkr, qkn) = (cfg.q_lora as usize, cfg.kv_lora as usize, cfg.qk_rope as usize, cfg.qk_nope as usize);
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let sparse = i >= cfg.first_dense as usize;

    let (mut dense, mut router, mut router_bias, mut shared, mut experts) =
        (None, Vec::new(), Vec::new(), None, Vec::new());
    if !sparse {
        let di = cfg.dense_inter as usize;
        dense = Some(Mlp {
            gate: QtWeight::load(st, &p("mlp.gate_proj.weight"), di, d)?,
            up: QtWeight::load(st, &p("mlp.up_proj.weight"), di, d)?,
            down: QtWeight::load(st, &p("mlp.down_proj.weight"), d, di)?,
        });
    } else {
        let (e_n, mi, si) = (cfg.n_experts as usize, cfg.moe_inter as usize, (cfg.moe_inter * cfg.n_shared) as usize);
        router = load_f32(st, &p("mlp.gate.weight"), e_n * d)?;
        router_bias = load_f32(st, &p("mlp.gate.e_score_correction_bias"), e_n)?;
        shared = Some(Mlp {
            gate: QtWeight::load(st, &p("mlp.shared_experts.gate_proj.weight"), si, d)?,
            up: QtWeight::load(st, &p("mlp.shared_experts.up_proj.weight"), si, d)?,
            down: QtWeight::load(st, &p("mlp.shared_experts.down_proj.weight"), d, si)?,
        });
        for e in 0..e_n {
            let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
            if stream_experts {
                // don't hold experts resident; verify presence so a malformed
                // checkpoint fails at load, not mid-decode.
                for t in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                    let name = pe(t);
                    if !st.has(&name) {
                        return Err(Error::Format(format!("missing expert tensor: {name}")));
                    }
                }
            } else {
                experts.push(Mlp {
                    gate: QtWeight::load(st, &pe("gate_proj.weight"), mi, d)?,
                    up: QtWeight::load(st, &pe("up_proj.weight"), mi, d)?,
                    down: QtWeight::load(st, &pe("down_proj.weight"), d, mi)?,
                });
            }
        }
    }

    Ok(LayerW {
        in_ln: load_f32(st, &p("input_layernorm.weight"), d)?,
        post_ln: load_f32(st, &p("post_attention_layernorm.weight"), d)?,
        q_a: QtWeight::load(st, &p("self_attn.q_a_proj.weight"), ql, d)?,
        q_a_ln: load_f32(st, &p("self_attn.q_a_layernorm.weight"), ql)?,
        q_b: QtWeight::load(st, &p("self_attn.q_b_proj.weight"), h * qkh, ql)?,
        kv_a: QtWeight::load(st, &p("self_attn.kv_a_proj_with_mqa.weight"), kvl + qkr, d)?,
        kv_a_ln: load_f32(st, &p("self_attn.kv_a_layernorm.weight"), kvl)?,
        kv_b: QtWeight::load(st, &p("self_attn.kv_b_proj.weight"), h * (qkn + vh), kvl)?,
        o: QtWeight::load(st, &p("self_attn.o_proj.weight"), d, h * vh)?,
        sparse,
        dense,
        router,
        router_bias,
        shared,
        experts,
    })
}

/// Row-wise RMSNorm of `x[s_n, d]` with weight `w`, into a fresh buffer.
fn rmsnorm_rows(x: &[f32], w: &[f32], s_n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; s_n * d];
    for s in 0..s_n {
        let src = x[s * d..s * d + d].to_vec();
        rmsnorm(&mut out[s * d..s * d + d], &src, w, eps);
    }
    out
}

/// Forward one transformer layer in place: `x += attn(norm(x)); x += ffn(norm(x))`.
/// Shared by the main stack and the MTP head; the sparse-MoE streaming/GPU lanes
/// apply exactly as in the main loop. Compute state travels in [`ForwardCtx`].
fn forward_layer(
    l: &LayerW,
    li: usize,
    kv: &mut LayerKv,
    ctx: &ForwardCtx,
    x: &mut [f32],
    s_n: usize,
    pos_base: usize,
) -> Result<(), Error> {
    let cfg = ctx.cfg;
    let d = cfg.hidden as usize;
    let eps = cfg.eps;
    let nrm = rmsnorm_rows(x, &l.in_ln, s_n, d, eps);
    let attn = mla_attention(&l.attn(), &nrm, s_n, pos_base, kv, cfg);
    for z in 0..s_n * d {
        x[z] += attn[z];
    }
    let nrm2 = rmsnorm_rows(x, &l.post_ln, s_n, d, eps);
    let ffn: Vec<f32> = if l.sparse {
        if ctx.stream_experts {
            moe_forward_concurrent(ctx, li, &nrm2, &l.router, &l.router_bias, l.shared.as_ref(), s_n)?
        } else {
            moe_forward(
                &nrm2,
                &l.router,
                &l.router_bias,
                &l.experts,
                l.shared.as_ref(),
                s_n,
                d,
                cfg.topk as usize,
                cfg.norm_topk,
                cfg.routed_scale,
            )
        }
    } else {
        let dense = l
            .dense
            .as_ref()
            .ok_or_else(|| Error::Format(format!("layer {li}: dense MLP weights missing")))?;
        dense.swiglu(&nrm2, s_n)
    };
    for z in 0..s_n * d {
        x[z] += ffn[z];
    }
    Ok(())
}

/// Forward one transformer layer over B **independent sequences** (one new token
/// each): batched MLA decode attention into each row's own `caches[s]`, then the
/// row-agnostic MoE (resident or the concurrent streaming lane) over all B rows.
/// Mirrors [`forward_layer`]; the MoE half is byte-for-byte the same call — it
/// already batch-unions experts across rows regardless of which sequence a row
/// belongs to, so B sequences share one set of expert reads (the amortization).
fn forward_layer_batched(
    l: &LayerW,
    li: usize,
    caches: &mut [&mut LayerKv],
    ctx: &ForwardCtx,
    x: &mut [f32],
    s_n: usize,
    pos_of: &[usize],
) -> Result<(), Error> {
    let cfg = ctx.cfg;
    let d = cfg.hidden as usize;
    let eps = cfg.eps;
    let nrm = rmsnorm_rows(x, &l.in_ln, s_n, d, eps);
    let attn = mla_attention_batched(&l.attn(), &nrm, pos_of, caches, cfg);
    for z in 0..s_n * d {
        x[z] += attn[z];
    }
    let nrm2 = rmsnorm_rows(x, &l.post_ln, s_n, d, eps);
    let ffn: Vec<f32> = if l.sparse {
        if ctx.stream_experts {
            moe_forward_concurrent(ctx, li, &nrm2, &l.router, &l.router_bias, l.shared.as_ref(), s_n)?
        } else {
            moe_forward(
                &nrm2,
                &l.router,
                &l.router_bias,
                &l.experts,
                l.shared.as_ref(),
                s_n,
                d,
                cfg.topk as usize,
                cfg.norm_topk,
                cfg.routed_scale,
            )
        }
    } else {
        let dense = l
            .dense
            .as_ref()
            .ok_or_else(|| Error::Format(format!("layer {li}: dense MLP weights missing")))?;
        dense.swiglu(&nrm2, s_n)
    };
    for z in 0..s_n * d {
        x[z] += ffn[z];
    }
    Ok(())
}

impl Model {
    /// Load a model directory (config.json + `*.safetensors` in the int4/int8
    /// container format).
    /// Load a model directory, auto-deciding whether to stream routed experts
    /// from disk (large models) or hold them resident (small models). The
    /// `COLI_STREAM=1|0` env var overrides the decision.
    pub fn load(dir: &std::path::Path) -> Result<Model, Error> {
        Self::load_inner(dir, None, None, None)
    }

    /// Load, forcing routed-expert streaming on (`true`) or off (`false`).
    /// Bypasses the RAM-budget heuristic — used to run a >RAM model explicitly
    /// and to test that the streamed path matches the resident one.
    pub fn load_streaming(dir: &std::path::Path, stream: bool) -> Result<Model, Error> {
        Self::load_inner(dir, Some(stream), None, None)
    }

    /// Load with streaming forced and an explicit warm-cache byte budget
    /// (`0` disables the cache). Bypasses `COLI_ECACHE_GB` so tests can toggle the
    /// cache deterministically without touching process env (which races under
    /// parallel test execution).
    pub fn load_streaming_ecache(dir: &std::path::Path, stream: bool, ecache_budget: usize) -> Result<Model, Error> {
        Self::load_inner(dir, Some(stream), Some(ecache_budget), None)
    }

    /// Load with streaming forced and O_DIRECT explicitly on/off (cache disabled).
    /// Bypasses `COLI_DIRECT` so tests can compare the direct-streamed path against
    /// the resident one without touching process env (parallel-test-safe).
    pub fn load_streaming_direct(dir: &std::path::Path, stream: bool, direct: bool) -> Result<Model, Error> {
        Self::load_inner(dir, Some(stream), Some(0), Some(direct))
    }

    fn load_inner(
        dir: &std::path::Path,
        force_stream: Option<bool>,
        force_ecache: Option<usize>,
        force_direct: Option<bool>,
    ) -> Result<Model, Error> {
        let cfg = Cfg::load(dir)?;
        let st = SafeTensors::open(dir)?;

        // Decide whether routed experts must be streamed from disk: sum their
        // on-disk payload and compare to available RAM (leaving headroom for
        // activations/KV). An explicit override or `COLI_STREAM=1|0` wins.
        let routed_bytes: u64 = st
            .tensors()
            .iter()
            .filter(|t| t.name.contains(".mlp.experts."))
            .map(|t| t.nbytes as u64)
            .sum();
        let stream_experts = force_stream.unwrap_or_else(|| {
            match std::env::var("COLI_STREAM").ok().as_deref() {
                Some("1") | Some("true") => true,
                Some("0") | Some("false") => false,
                _ => {
                    let avail = mem_available_bytes();
                    avail > 0 && routed_bytes as f64 > 0.6 * avail as f64
                }
            }
        });

        let d = cfg.hidden as usize;
        let (kvl, qkr) = (cfg.kv_lora as usize, cfg.qk_rope as usize);
        let vocab = cfg.vocab as usize;

        let embed = QtWeight::load(&st, "model.embed_tokens.weight", vocab, d)?.dequant();
        let lm_head = QtWeight::load(&st, "lm_head.weight", vocab, d)?;
        let final_norm = load_f32(&st, "model.norm.weight", d)?;

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            layers.push(load_layer(&st, i, &cfg, stream_experts)?);
        }

        let kv = (0..cfg.n_layers).map(|_| LayerKv::new(kvl, qkr)).collect();
        // The concurrent MoE lane needs its own ring, set up once, so a layer's
        // experts stream while the CPU pool computes. Only in streaming mode.
        let io_reactors: Vec<Mutex<Reactor>> = if stream_experts {
            let n = io_rings();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(Mutex::new(Reactor::new(256).ctx(|| "concurrent MoE io_uring reactor init".to_string())?));
            }
            v
        } else {
            Vec::new()
        };
        let workers = default_workers();
        // O_DIRECT streaming (opt-in via `COLI_DIRECT`): bypass the page cache for
        // the 0.6%-reuse expert reads. Only when streaming AND the shards actually
        // opened O_DIRECT fds. Size each reactor's aligned slab pool to the largest
        // expert region (Strategy A: 2 buffers in flight ≈ 2×19 MB).
        let want_direct = force_direct.unwrap_or_else(direct_enabled) && stream_experts;
        let direct = want_direct && st.has_any_direct();
        if want_direct {
            eprintln!(
                "peregrine: O_DIRECT streaming {}",
                if direct { "enabled" } else { "requested but unavailable — buffered fallback" }
            );
        }
        if direct {
            let cap = max_expert_region_bytes(&st);
            for r in &io_reactors {
                r.lock().configure_slab(cap, 2);
            }
        }
        // Warm tier: bound the RAM cache to an explicit budget (tests) or, when
        // unset, `COLI_ECACHE_GB` / an auto fraction of MemAvailable. Only built
        // when experts stream from disk and the budget is non-zero.
        let ecache = {
            let requested = force_ecache.unwrap_or_else(ecache_budget_bytes);
            // Cap the warm cache so it + the streaming lanes' transient landing/compute
            // buffers + a safety margin fit in available RAM (already net of the
            // resident model), so batched prefill/decode can't OOM. This is the peak-RAM
            // auto-budget; bounding the per-read allocation churn itself is the slab arena.
            let budget = if stream_experts && requested > 0 {
                let per_expert = 4 * max_expert_region_bytes(&st);
                let reserve = stream_transient_reserve(io_reactors.len(), workers, per_expert);
                let capped = cap_ecache_budget(requested, mem_available_bytes() as usize, reserve, 1usize << 30);
                if capped < requested {
                    eprintln!(
                        "peregrine: warm cache capped {} MiB -> {} MiB to keep RAM headroom for streaming buffers",
                        requested >> 20,
                        capped >> 20
                    );
                }
                capped
            } else {
                requested
            };
            if stream_experts && budget > 0 {
                Some(Arc::new(Mutex::new(WarmCache::new(budget))))
            } else {
                None
            }
        };
        // Prefetch lane: a background worker warming the next token's predicted
        // experts into the shared cache via its own ring. Spawned only when the
        // cache exists (streaming mode). `route_hist` is the predictor's state.
        let (route_hist, prefetch) = match &ecache {
            Some(cache) => {
                let mut reactor = Reactor::new(64).ctx(|| "prefetch io_uring reactor init".to_string())?;
                if direct {
                    reactor.configure_slab(max_expert_region_bytes(&st), 2);
                }
                let cache = Arc::clone(cache);
                let (tx, rx) = crossbeam_channel::unbounded::<PrefetchMsg>();
                let join = std::thread::Builder::new()
                    .name("peregrine-prefetch".to_string())
                    .spawn(move || prefetch_worker(reactor, cache, rx, direct))
                    .map_err(|e| Error::Format(format!("spawn prefetch thread: {e}")))?;
                (
                    Some(Mutex::new(vec![Vec::new(); cfg.n_layers as usize])),
                    Some(PrefetchHandle { tx, join: Some(join) }),
                )
            }
            None => (None, None),
        };
        // Optional GPU VRAM tier (opt-in via COLI_GPU): dequantize as many experts
        // as fit to f32 and upload. Reserve 2 GB headroom for activations/context.
        let gpu = if std::env::var("COLI_GPU").is_ok() {
            let tier = GpuTier::build(&st, &cfg, 2 * 1024 * 1024 * 1024)?;
            if let Some(t) = &tier {
                eprintln!("peregrine: GPU tier holds {} experts in VRAM", t.len());
            }
            tier
        } else {
            None
        };
        // Heat accumulator for dynamic VRAM residency — only useful (and only built)
        // when there is a GPU tier to migrate hot experts into.
        let heat = gpu.as_ref().map(|_| HeatTable::new(cfg.n_layers as usize, cfg.n_experts as usize));

        // Optional MTP head (checkpoints converted with --mtp): a full layer at
        // index n_layers plus the embed/hidden projection and norms.
        let n = cfg.n_layers as usize;
        let mtp = if st.has(&format!("model.layers.{n}.eh_proj.weight")) {
            Some(MtpHead {
                layer: load_layer(&st, n, &cfg, stream_experts)?,
                eh_proj: QtWeight::load(&st, &format!("model.layers.{n}.eh_proj.weight"), d, 2 * d)?,
                enorm: load_f32(&st, &format!("model.layers.{n}.enorm.weight"), d)?,
                hnorm: load_f32(&st, &format!("model.layers.{n}.hnorm.weight"), d)?,
                mtp_norm: load_f32(&st, &format!("model.layers.{n}.shared_head.norm.weight"), d)?,
            })
        } else {
            None
        };

        Ok(Model {
            cfg,
            embed,
            layers,
            final_norm,
            lm_head,
            kv,
            stream_experts,
            direct,
            st,
            io_reactors,
            workers,
            ecache,
            route_hist,
            prefetch,
            gpu,
            mtp,
            heat,
        })
    }

    /// Clear the KV cache to start a fresh sequence. Also clears the prefetch
    /// predictor's history (a new sequence has no useful routing history); the
    /// warm cache itself persists (a warm expert is warm regardless of sequence).
    pub fn reset(&mut self) {
        let (kvl, qkr) = (self.cfg.kv_lora as usize, self.cfg.qk_rope as usize);
        for k in &mut self.kv {
            *k = LayerKv::new(kvl, qkr);
        }
        if let Some(h) = &self.route_hist {
            for v in h.lock().iter_mut() {
                v.clear();
            }
        }
    }

    /// Enqueue a speculative prefetch of the next token's likely experts: predict
    /// each sparse layer's experts as the ones this forward just routed, and warm
    /// the not-yet-resident, non-GPU ones into the cache on the background lane.
    /// A no-op without a prefetch lane, or on the first forward (empty history).
    fn enqueue_prefetch(&self) {
        let (Some(prefetch), Some(hist), Some(cache)) = (&self.prefetch, &self.route_hist, &self.ecache) else {
            return;
        };
        let first_dense = self.cfg.first_dense as usize;
        let mut items = Vec::new();
        {
            let hist = hist.lock();
            let cache = cache.lock();
            for (layer, experts) in hist.iter().enumerate() {
                if layer < first_dense {
                    continue; // dense layer — no routed experts
                }
                for &e in experts {
                    let key = (layer as u32, e as u32);
                    if cache.contains(key) {
                        continue; // already warm
                    }
                    if self.gpu.as_ref().is_some_and(|g| g.has(layer, e as usize)) {
                        continue; // computed on the GPU lane, never streamed
                    }
                    if let Ok(item) = crate::concurrent::prefetch_item(&self.st, &self.cfg, layer, e as usize) {
                        items.push(item);
                    }
                }
            }
        }
        if !items.is_empty() {
            let _ = prefetch.tx.send(PrefetchMsg::Warm(items));
        }
    }

    /// Rewind every layer's KV cache to `new_len` (speculative-decode: drop the
    /// KV of rejected draft positions so the next forward appends in order).
    fn truncate_kv(&mut self, new_len: usize) {
        for k in &mut self.kv {
            k.truncate(new_len);
        }
    }

    /// Whether this model has an MTP head available for speculative decode.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// `(hits, misses, disk_reads)` from the warm tier, or `None` when not
    /// streaming with a cache. For introspection/tests — the cache never affects
    /// output, only how many expert reads actually hit the disk.
    pub fn ecache_stats(&self) -> Option<(u64, u64, u64)> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            (c.hits, c.misses, c.disk_reads)
        })
    }

    /// Disk reads the warm tier has attributed to `layer` so far (for the
    /// prefetch test, which isolates the effect on one layer).
    pub fn ecache_disk_reads_for_layer(&self, layer: usize) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().disk_reads_for_layer(layer as u32))
    }

    /// Experts the prefetch lane has streamed ahead of time (off the critical path).
    pub fn ecache_prefetch_reads(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().prefetch_reads)
    }

    /// Drop all warm-tier entries and zero its counters (forces a cold cache;
    /// used by the prefetch test to isolate the prefetch lane's contribution).
    pub fn ecache_clear(&self) {
        if let Some(c) = &self.ecache {
            c.lock().clear();
        }
    }

    /// Block until the prefetch lane has processed every message queued so far
    /// (FIFO barrier). Lets a test observe the effect of a prefetch deterministically
    /// without racing the background thread. A no-op without a prefetch lane.
    pub fn prefetch_barrier(&self) {
        if let Some(p) = &self.prefetch {
            let (tx, rx) = crossbeam_channel::bounded(1);
            if p.tx.send(PrefetchMsg::Sync(tx)).is_ok() {
                let _ = rx.recv();
            }
        }
    }

    /// Trigger the next-token prefetch on demand (same path `forward_hidden` uses).
    /// Exposed for tests that warm a deliberately-cleared cache from history.
    pub fn prefetch_from_history(&self) {
        self.enqueue_prefetch();
    }

    /// Run `tokens` (new positions from `pos_base`) through all layers, appending
    /// to the KV cache, and return the **pre-final-norm** hidden state `[S,
    /// hidden]`. The MTP head reuses the last position's hidden as its draft seed.
    pub fn forward_hidden(&mut self, tokens: &[i32], pos_base: usize) -> Result<Vec<f32>, Error> {
        let s_n = tokens.len();
        let d = self.cfg.hidden as usize;

        // embedding lookup — clamp out-of-range ids (a malformed prompt token, or
        // a negative id) into `0..vocab` so a bad input can't index out of bounds.
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }

        // Run the stack in a block so the split borrows of `self` end before we
        // re-borrow `self` to enqueue prefetch.
        {
            // split disjoint fields so attention can borrow layers (imm) + kv (mut)
            let Model { cfg, layers, kv, st, stream_experts, direct, io_reactors, workers, ecache, route_hist, gpu, heat, .. } =
                self;
            let ctx = ForwardCtx {
                st,
                reactors: io_reactors,
                gpu: gpu.as_ref(),
                workers: *workers,
                cfg,
                stream_experts: *stream_experts,
                ecache: ecache.as_deref(),
                route_log: route_hist.as_ref(),
                direct: *direct,
                heat: heat.as_ref(),
            };
            for (li, l) in layers.iter().enumerate() {
                forward_layer(l, li, &mut kv[li], &ctx, &mut x, s_n, pos_base)?;
            }
        }
        // Predict + prefetch the next token's experts (main forward only).
        self.enqueue_prefetch();
        Ok(x)
    }

    /// Run `tokens` through all layers and return logits `[S, vocab]`.
    pub fn forward_step(&mut self, tokens: &[i32], pos_base: usize) -> Result<Vec<f32>, Error> {
        let s_n = tokens.len();
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let x = self.forward_hidden(tokens, pos_base)?;
        let xf = rmsnorm_rows(&x, &self.final_norm, s_n, d, eps);
        Ok(self.lm_head.apply_vec(&xf, s_n))
    }

    /// Build the per-forward compute context from the resident model state with
    /// prefetch/route-logging **disabled** — the shape the external-KV batched and
    /// prefill paths use (the B-way expert union is not a useful next-token
    /// predictor, so prefetch is gated off under batching).
    fn forward_ctx(&self) -> ForwardCtx<'_> {
        ForwardCtx {
            st: &self.st,
            reactors: &self.io_reactors,
            gpu: self.gpu.as_ref(),
            workers: self.workers,
            cfg: &self.cfg,
            stream_experts: self.stream_experts,
            ecache: self.ecache.as_deref(),
            route_log: None,
            direct: self.direct,
            heat: self.heat.as_ref(),
        }
    }

    /// Prefill one sequence's prompt into an **external** per-sequence KV (`seq`),
    /// returning logits `[S, vocab]`. Same dense path as [`Self::forward_step`] but
    /// writing to caller-owned KV via `&self`, so the batching scheduler can
    /// prefill each new sequence before batching its decode steps. Bit-identical to
    /// `forward_step` on a fresh model (external vs internal cache is the only diff).
    pub fn forward_prefill_seq(&self, tokens: &[i32], seq: &mut SeqKv, pos_base: usize) -> Result<Vec<f32>, Error> {
        let s_n = tokens.len();
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }
        let ctx = self.forward_ctx();
        for (li, l) in self.layers.iter().enumerate() {
            forward_layer(l, li, &mut seq.layers[li], &ctx, &mut x, s_n, pos_base)?;
        }
        let xf = rmsnorm_rows(&x, &self.final_norm, s_n, d, eps);
        Ok(self.lm_head.apply_vec(&xf, s_n))
    }

    /// Batched decode step over B **independent sequences** (one new token each):
    /// `tokens[s]` is sequence `s`'s next token at absolute position `pos_of[s]`,
    /// and `seqs[s]` is that sequence's KV (the new latent is appended in place).
    /// Returns logits `[B, vocab]`. The MoE lane reads each routed expert once and
    /// serves every row routing to it, so B sequences share one set of expert reads
    /// — the batching amortization. Row `s`'s logits are identical to decoding
    /// sequence `s` alone (guarded by `batched_decode_matches_per_sequence`).
    ///
    /// `&self`: per-sequence KV is caller-owned, so one resident model drives many
    /// concurrent sequences from a single scheduler thread. Prefetch/route-logging
    /// are off ([`Self::forward_ctx`]); MTP speculation stays a B==1 path.
    pub fn forward_step_batched(&self, tokens: &[i32], seqs: &mut [&mut SeqKv], pos_of: &[usize]) -> Result<Vec<f32>, Error> {
        let s_n = tokens.len();
        if seqs.len() != s_n || pos_of.len() != s_n {
            return Err(Error::Format(format!(
                "forward_step_batched: {s_n} tokens but {} seqs / {} positions",
                seqs.len(),
                pos_of.len()
            )));
        }
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }
        let ctx = self.forward_ctx();
        for (li, l) in self.layers.iter().enumerate() {
            let mut caches: Vec<&mut LayerKv> = seqs.iter_mut().map(|sk| &mut sk.layers[li]).collect();
            forward_layer_batched(l, li, &mut caches, &ctx, &mut x, s_n, pos_of)?;
        }
        let xf = rmsnorm_rows(&x, &self.final_norm, s_n, d, eps);
        Ok(self.lm_head.apply_vec(&xf, s_n))
    }

    /// Re-select the GPU tier's resident experts as the current hottest set (by
    /// accumulated routing frequency), migrating cooled experts out of VRAM and hot
    /// ones in. A no-op without a GPU tier (or the `cuda` feature). Call between
    /// forwards (`&mut self`); the batch engine invokes it periodically so residency
    /// adapts to the workload without a rewrite.
    pub fn reheat(&mut self) -> Result<(), Error> {
        let Some(counts) = self.heat.as_ref().map(|h| h.snapshot()) else {
            return Ok(());
        };
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.reheat(&self.st, &self.cfg, &counts)?;
        }
        Ok(())
    }

    /// Greedy/sampled generation: prefill `prompt`, then decode `n_new` tokens.
    /// Resets the KV cache first. Returns the newly generated token ids.
    pub fn generate(&mut self, prompt: &[i32], n_new: usize, sampler: &mut Sampler) -> Result<Vec<i32>, Error> {
        // an empty prompt has no last-position logits to sample from, and no
        // requested tokens is a no-op — both would otherwise underflow below.
        if prompt.is_empty() || n_new == 0 {
            return Ok(Vec::new());
        }
        self.reset();
        let vocab = self.cfg.vocab as usize;
        let logits = self.forward_step(prompt, 0)?;
        let mut next = sampler.pick(&logits[(prompt.len() - 1) * vocab..prompt.len() * vocab], -1) as i32;
        let mut out = vec![next];
        for step in 1..n_new {
            let pos = prompt.len() + step - 1; // first decode attends at prompt.len()
            let lg = self.forward_step(&[next], pos)?;
            next = sampler.pick(&lg[..vocab], -1) as i32;
            out.push(next);
        }
        Ok(out)
    }

    /// Teacher-forcing predictions: greedy argmax at each position of a single
    /// full forward over `tokens`. The shape the oracle gate compares against.
    pub fn teacher_forcing(&mut self, tokens: &[i32]) -> Result<Vec<i32>, Error> {
        self.reset();
        let vocab = self.cfg.vocab as usize;
        let logits = self.forward_step(tokens, 0)?;
        Ok((0..tokens.len())
            .map(|s| crate::sample::argmax(&logits[s * vocab..s * vocab + vocab]) as i32)
            .collect())
    }

    /// Draft `g_draft` tokens with the MTP head, seeded by token `next_tok` and
    /// the pre-final-norm hidden `hlast`. Uses a fresh local KV at relative
    /// positions (context flows through the hidden state), so it never touches the
    /// main KV. The drafts are only *guesses* — the caller verifies them.
    fn mtp_draft(&mut self, next_tok: i32, g_draft: usize, hlast: &[f32]) -> Result<Vec<i32>, Error> {
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let n_layers = self.cfg.n_layers as usize;
        let (kvl, qkr) = (self.cfg.kv_lora as usize, self.cfg.qk_rope as usize);

        let Model { mtp, st, embed, lm_head, final_norm, io_reactors, workers, ecache, direct, gpu, stream_experts, cfg, .. } =
            self;
        let mtp = mtp.as_ref().ok_or_else(|| Error::Format("mtp_draft without an MTP head".into()))?;
        let ctx = ForwardCtx {
            st,
            reactors: io_reactors,
            gpu: gpu.as_ref(),
            workers: *workers,
            cfg,
            stream_experts: *stream_experts,
            ecache: ecache.as_deref(),
            route_log: None, // drafts must not overwrite the main-stream prediction
            direct: *direct,
            heat: None, // speculative drafts must not skew residency heat
        };
        let mut kv = LayerKv::new(kvl, qkr);
        let mut h = hlast.to_vec(); // pre-final-norm hidden
        let mut tok = next_tok;
        let mut draft = Vec::with_capacity(g_draft);
        for g in 0..g_draft {
            // norm(embed(tok))
            let tid = (tok.max(0) as usize).min(vocab.saturating_sub(1));
            let mut e = vec![0f32; d];
            rmsnorm(&mut e, &embed[tid * d..tid * d + d], &mtp.enorm, eps);
            // the incoming hidden: g==0 is the main model's hidden → apply final_norm
            // first; afterwards it is a prior MTP-layer output, used directly.
            if g == 0 {
                let hc = h.clone();
                rmsnorm(&mut h, &hc, final_norm, eps);
            }
            let mut hn = vec![0f32; d];
            rmsnorm(&mut hn, &h, &mtp.hnorm, eps);
            // hx = eh_proj([e | hn])
            let mut cat = e;
            cat.extend_from_slice(&hn);
            let mut hx = mtp.eh_proj.apply_vec(&cat, 1);
            // one MTP transformer layer at relative position g (fresh local KV)
            forward_layer(&mtp.layer, n_layers, &mut kv, &ctx, &mut hx, 1, g)?;
            let row = rmsnorm_rows(&hx, &mtp.mtp_norm, 1, d, eps);
            let logit = lm_head.apply_vec(&row, 1);
            let t2 = crate::sample::argmax(&logit) as i32;
            draft.push(t2);
            tok = t2;
            h = hx; // next hidden = this MTP layer's output
        }
        Ok(draft)
    }

    /// Greedy speculative decode with the MTP head: draft `g_draft` tokens, verify
    /// them in one batched forward, accept the matching prefix, and rewind the KV
    /// for any rejects. The emitted sequence is **identical** to plain greedy
    /// [`Self::generate`] (each token is the model's argmax), just with fewer full
    /// forwards. Falls back to greedy when no MTP head is present.
    pub fn generate_speculative(&mut self, prompt: &[i32], n_new: usize, g_draft: usize) -> Result<Vec<i32>, Error> {
        if prompt.is_empty() || n_new == 0 {
            return Ok(Vec::new());
        }
        if self.mtp.is_none() || g_draft == 0 {
            let mut greedy = Sampler::new(0.0, 0.9, 1);
            return self.generate(prompt, n_new, &mut greedy);
        }
        self.reset();
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let plen = prompt.len();

        // prefill
        let x = self.forward_hidden(prompt, 0)?;
        let mut hlast = x[(plen - 1) * d..plen * d].to_vec();
        let logits = {
            let xf = rmsnorm_rows(&x, &self.final_norm, plen, d, eps);
            self.lm_head.apply_vec(&xf, plen)
        };
        let mut next = crate::sample::argmax(&logits[(plen - 1) * vocab..plen * vocab]) as i32;

        let mut out: Vec<i32> = Vec::new();
        let mut pos = plen;
        while out.len() < n_new {
            let budget = n_new - out.len();
            // draft at most budget-1 (we always emit `next` this round)
            let g_want = g_draft.min(budget.saturating_sub(1));
            let draft = if g_want > 0 { self.mtp_draft(next, g_want, &hlast)? } else { Vec::new() };
            let g = draft.len();

            // verify [next, draft...] in one forward
            let mut batch = Vec::with_capacity(1 + g);
            batch.push(next);
            batch.extend_from_slice(&draft);
            let s = batch.len();
            let xb = self.forward_hidden(&batch, pos)?;
            let logits_b = {
                let xbf = rmsnorm_rows(&xb, &self.final_norm, s, d, eps);
                self.lm_head.apply_vec(&xbf, s)
            };

            // `next` is confirmed (it was the model's argmax); emit it
            out.push(next);
            // accept drafts while they match the model's greedy prediction
            let mut k = 0usize;
            while k < g && out.len() < n_new {
                let pred = crate::sample::argmax(&logits_b[k * vocab..(k + 1) * vocab]) as i32;
                if pred == draft[k] {
                    out.push(draft[k]);
                    k += 1;
                } else {
                    break;
                }
            }
            // the model's prediction at position k is the next token to process
            next = crate::sample::argmax(&logits_b[k * vocab..(k + 1) * vocab]) as i32;
            hlast = xb[k * d..(k + 1) * d].to_vec();
            // committed this round: `next` (already emitted) + k accepted drafts
            let committed = 1 + k;
            self.truncate_kv(pos + committed);
            pos += committed;
            if out.last().is_some_and(|t| self.cfg.stop_ids.contains(t)) {
                break;
            }
        }
        out.truncate(n_new);
        Ok(out)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Stop and join the prefetch lane before `st` (its shard fds) is dropped.
        if let Some(mut p) = self.prefetch.take() {
            let _ = p.tx.send(PrefetchMsg::Stop);
            if let Some(j) = p.join.take() {
                let _ = j.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::build_tiny_model;
    use std::path::PathBuf;

    fn tmp_model_dir(tag: &str) -> Result<PathBuf, peregrine_core::Error> {
        let d = std::env::temp_dir().join(format!("peregrine_model_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        build_tiny_model(&d)?;
        Ok(d)
    }

    #[test]
    fn loads_and_runs_forward() -> Result<(), peregrine_core::Error> {
        let dir = tmp_model_dir("fwd")?;
        let mut m = Model::load(&dir)?;
        let logits = m.forward_step(&[1, 5, 9, 2], 0)?;
        assert_eq!(logits.len(), 4 * m.cfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn streamed_experts_match_resident() -> Result<(), peregrine_core::Error> {
        // The on-demand streamed expert path must produce identical logits to
        // the resident path — same bytes, same kernels, only load timing differs.
        let dir = tmp_model_dir("stream")?;
        let mut resident = Model::load_streaming(&dir, false)?;
        let mut streamed = Model::load_streaming(&dir, true)?;
        assert!(streamed.stream_experts && !resident.stream_experts);
        let toks = [1, 5, 9, 2, 7];
        let lr = resident.forward_step(&toks, 0)?;
        let ls = streamed.forward_step(&toks, 0)?;
        assert_eq!(lr, ls, "streamed logits must equal resident logits");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn streamed_direct_matches_resident() -> Result<(), peregrine_core::Error> {
        // O_DIRECT-streamed experts must produce identical logits to the resident
        // path (same bytes — only the page-cache behavior differs). If the temp
        // filesystem rejects O_DIRECT, `direct` falls back to buffered and this is
        // still a valid streamed==resident check.
        let dir = tmp_model_dir("direct")?;
        let mut resident = Model::load_streaming(&dir, false)?;
        let mut direct = Model::load_streaming_direct(&dir, true, true)?;
        assert!(direct.stream_experts);
        let toks = [1, 5, 9, 2, 7];
        let lr = resident.forward_step(&toks, 0)?;
        let ld = direct.forward_step(&toks, 0)?;
        assert_eq!(lr, ld, "O_DIRECT-streamed logits must equal resident");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn ecache_bit_identical() -> Result<(), peregrine_core::Error> {
        // The warm cache must not change numerics: streamed-with-cache,
        // streamed-without-cache, and resident must all produce identical logits
        // (same bytes → same rebuild → same swiglu).
        let dir = tmp_model_dir("ecache_id")?;
        let mut resident = Model::load_streaming(&dir, false)?;
        let mut no_cache = Model::load_streaming_ecache(&dir, true, 0)?; // streaming, cache off
        let mut cached = Model::load_streaming_ecache(&dir, true, 8 << 20)?; // 8 MiB cache
        assert!(cached.ecache_stats().is_some() && no_cache.ecache_stats().is_none());
        let toks = [1, 5, 9, 2, 7];
        let lr = resident.forward_step(&toks, 0)?;
        let ln = no_cache.forward_step(&toks, 0)?;
        let lc = cached.forward_step(&toks, 0)?;
        assert_eq!(lr, ln, "streamed (no cache) must equal resident");
        assert_eq!(lr, lc, "streamed (cached) must equal resident");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn ecache_second_pass_hits() -> Result<(), peregrine_core::Error> {
        // Re-running an identical forward (KV reset, cache retained) must serve
        // every expert from the warm tier — hits rise, disk reads do not.
        let dir = tmp_model_dir("ecache_hits")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        let _ = m.forward_step(&toks, 0)?;
        let (h1, _, d1) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(d1 > 0, "first pass must stream experts from disk");
        m.reset(); // clear KV so the second forward routes identically; cache persists
        let _ = m.forward_step(&toks, 0)?;
        let (h2, _, d2) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(h2 > h1, "second pass must register cache hits (got {h1} → {h2})");
        assert_eq!(d2, d1, "second pass must not re-read any expert from disk");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefetch_output_identical() -> Result<(), peregrine_core::Error> {
        // The prefetch lane only warms the cache — it must never change output.
        let dir = tmp_model_dir("prefetch_id")?;
        let mut with_pf = Model::load_streaming_ecache(&dir, true, 8 << 20)?; // cache + prefetch on
        let mut without = Model::load_streaming_ecache(&dir, true, 0)?; // no cache, no prefetch
        let prompt = [3, 7, 1, 4, 2];
        let mut s1 = Sampler::new(0.0, 0.9, 1);
        let mut s2 = Sampler::new(0.0, 0.9, 1);
        let a = with_pf.generate(&prompt, 8, &mut s1)?;
        let b = without.generate(&prompt, 8, &mut s2)?;
        assert_eq!(a, b, "prefetch/cache must not change generated tokens");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefetch_warms_cold_cache() -> Result<(), peregrine_core::Error> {
        // The prefetch lane must warm the predicted experts off the critical path:
        // after a cold clear + prefetch, the next identical forward does zero
        // critical-path disk reads.
        let dir = tmp_model_dir("prefetch_cold")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        let _ = m.forward_step(&toks, 0)?; // pass 1: fills routing history (and warms the cache)
        m.prefetch_barrier(); // drain the auto-enqueued prefetch
        m.ecache_clear(); // cold cache + zeroed counters (routing history retained)
        m.prefetch_from_history(); // stream predicted (= pass-1) experts on the background lane
        m.prefetch_barrier(); // wait for it to finish
        let pf = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let (_, _, d_pref) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(pf > 0, "prefetch lane must have streamed experts (got {pf})");
        assert_eq!(d_pref, 0, "prefetch must not count as critical-path disk reads");
        m.reset(); // KV reset (history cleared); the warm cache persists
        let _ = m.forward_step(&toks, 0)?; // pass 2: identical routing → served from the warm cache
        let (h2, _, d2) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(h2 > 0, "pass 2 must hit the warm tier");
        assert_eq!(d2, 0, "pass 2 must do no critical-path disk reads (prefetch pre-warmed)");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn generate_is_deterministic_greedy() -> Result<(), peregrine_core::Error> {
        let dir = tmp_model_dir("gen")?;
        let mut m = Model::load(&dir)?;
        let prompt = [3, 7, 1, 4];
        let mut s1 = Sampler::new(0.0, 0.9, 1); // greedy
        let a = m.generate(&prompt, 8, &mut s1)?;
        let mut s2 = Sampler::new(0.0, 0.9, 1);
        let b = m.generate(&prompt, 8, &mut s2)?;
        assert_eq!(a, b, "greedy generation must be deterministic");
        assert!(a.iter().all(|&t| (t as usize) < m.cfg.vocab as usize));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn decode_matches_teacher_forcing_prefix() -> Result<(), peregrine_core::Error> {
        // greedy decode's first token == teacher-forcing argmax at the last
        // prompt position (both are argmax of the same prefill logits).
        let dir = tmp_model_dir("tf")?;
        let mut m = Model::load(&dir)?;
        let prompt = [2, 6, 3, 8, 1];
        let tf = m.teacher_forcing(&prompt)?;
        let mut s = Sampler::new(0.0, 0.9, 1);
        let gen = m.generate(&prompt, 1, &mut s)?;
        assert_eq!(gen[0], tf[prompt.len() - 1]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn speculative_matches_greedy() -> Result<(), peregrine_core::Error> {
        // MTP speculative decode must emit exactly the same tokens as plain greedy
        // (every token is verified against the model's argmax) — only faster.
        let dir = tmp_model_dir("spec")?;
        let mut m = Model::load(&dir)?;
        assert!(m.has_mtp(), "tiny model should carry an MTP head");
        let prompt = [3, 7, 1, 4];
        let spec = m.generate_speculative(&prompt, 8, 3)?;
        let mut greedy = Sampler::new(0.0, 0.9, 1);
        let base = m.generate(&prompt, 8, &mut greedy)?;
        assert_eq!(spec, base, "speculative output must equal greedy");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn handles_empty_prompt_and_out_of_range_tokens() -> Result<(), peregrine_core::Error> {
        // regression: empty prompt / zero n_new must not underflow, and out-of-
        // range or negative token ids must be clamped, not index out of bounds.
        let dir = tmp_model_dir("edge")?;
        let mut m = Model::load(&dir)?;
        let mut s = Sampler::new(0.0, 0.9, 1);
        assert!(m.generate(&[], 4, &mut s)?.is_empty());
        assert!(m.generate(&[1, 2], 0, &mut s)?.is_empty());
        let logits = m.forward_step(&[9999, -3, 0], 0)?;
        assert_eq!(logits.len(), 3 * m.cfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn cap_ecache_budget_leaves_ram_headroom() {
        // plenty of RAM → the requested budget is unchanged
        assert_eq!(cap_ecache_budget(4 << 30, 32 << 30, 2 << 30, 1 << 30), 4 << 30);
        // tight RAM → capped to (available - transient reserve - safety)
        assert_eq!(cap_ecache_budget(10 << 30, 12 << 30, 2 << 30, 1 << 30), 9 << 30);
        // no headroom left → 0 disables the cache rather than risking OOM
        assert_eq!(cap_ecache_budget(4 << 30, 2 << 30, 2 << 30, 1 << 30), 0);
    }

    #[test]
    fn stream_transient_reserve_scales_with_lanes() {
        assert_eq!(stream_transient_reserve(4, 8, 1000), (4 * EXPERTS_PER_BATCH + 8) * 1000);
        assert_eq!(stream_transient_reserve(0, 0, 1000), 0); // no lanes → no reserve
    }

    #[test]
    fn prefill_seq_matches_forward_step() -> Result<(), peregrine_core::Error> {
        // forward_prefill_seq into an external SeqKv must equal the internal
        // forward_step (same dense path; external vs internal cache is the only
        // difference), so the scheduler's prefill produces identical state.
        let dir = tmp_model_dir("prefill_seq")?;
        let mut m = Model::load(&dir)?;
        let toks = [1, 5, 9, 2, 7];
        let internal = m.forward_step(&toks, 0)?;
        let mut seq = SeqKv::new(&m.cfg);
        let external = m.forward_prefill_seq(&toks, &mut seq, 0)?;
        assert_eq!(internal, external, "external-KV prefill must equal internal forward_step");
        assert_eq!(seq.len(), toks.len());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn batched_decode_matches_per_sequence() -> Result<(), peregrine_core::Error> {
        // Three sequences prefilled to different lengths, then one batched decode
        // step. Row s of the batched step must equal decoding sequence s alone
        // (B==1) — both via the absorb core — so batching many sequences through
        // one forward is bit-identical to decoding them separately.
        let dir = tmp_model_dir("batched_decode")?;
        let m = Model::load(&dir)?;
        let vocab = m.cfg.vocab as usize;
        let prompts: [&[i32]; 3] = [&[1, 5, 9], &[2, 7], &[3, 8, 4, 6]];
        let newtok = [4i32, 6, 2];

        // reference: prefill each sequence, then decode it alone (B==1)
        let mut ref_logits = vec![0f32; 3 * vocab];
        for (s, p) in prompts.iter().enumerate() {
            let mut sk = SeqKv::new(&m.cfg);
            let _ = m.forward_prefill_seq(p, &mut sk, 0)?;
            let mut one: [&mut SeqKv; 1] = [&mut sk];
            let pos = [p.len()];
            let lg = m.forward_step_batched(&[newtok[s]], &mut one, &pos)?;
            ref_logits[s * vocab..s * vocab + vocab].copy_from_slice(&lg);
        }

        // batched: prefill all three into fresh caches, then ONE batched decode
        let mut seqs: Vec<SeqKv> = Vec::new();
        for p in prompts.iter() {
            let mut sk = SeqKv::new(&m.cfg);
            let _ = m.forward_prefill_seq(p, &mut sk, 0)?;
            seqs.push(sk);
        }
        let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
        let toks: Vec<i32> = newtok.to_vec();
        let pos_of: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let bat = m.forward_step_batched(&toks, &mut refs, &pos_of)?;

        for z in 0..3 * vocab {
            assert!((ref_logits[z] - bat[z]).abs() < 1e-4, "z={z} ref={} bat={}", ref_logits[z], bat[z]);
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

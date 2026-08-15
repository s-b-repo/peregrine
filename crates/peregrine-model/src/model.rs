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

use crate::attention::{
    mla_attention, mla_attention_absorb, mla_attention_dsa_indexed, mla_attention_rows, AttnWeights, KvDtype,
    LayerKv, RowLayout,
};
use crate::dsa::IndexerWeights;
use crate::concurrent::{default_workers, experts_per_batch, moe_forward_dispatch, ForwardCtx};
use crate::gpu::{GpuTier, HeatTable};
use crate::math::rmsnorm;
use crate::mlp::{moe_forward, Mlp, MoeCfg};
use crate::predict::{Momentum, PredictSource, PrefetchTuner, RouteHistory, TransitionTable};
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
    /// DSA lightning-indexer weights, when the checkpoint carries them. Weights
    /// only — the key cache is per-sequence and lives in `LayerKv`.
    indexer: Option<IndexerWeights>,
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
/// Every artifact the one-pass "galactic" preprocessing produces: the expert
/// transition automaton, the macro-state table (temporal routing compression),
/// and the raw per-forward routing trace the layout tools consume.
pub type OfflineArtifacts = (TransitionTable, crate::predict::MacroTable, Vec<Vec<Vec<i32>>>);

pub struct Model {
    pub cfg: Cfg,
    /// `[vocab, hidden]`, kept **packed**. Only one row per token is ever read
    /// (`dequant_row_into`), so dequantizing the whole table cost 3.81 GB of f32
    /// against 0.95 GB packed at GLM-5.2 shapes — 2.85 GB of resident set for
    /// rows that are never touched. Row `r` is bit-identical either way:
    /// `dequant()` is defined as a loop over `dequant_row`.
    embed: QtWeight, // [vocab, hidden], packed
    /// `COLI_MLA_ABSORB`, resolved once at load (see `absorb_enabled`).
    absorb: bool,
    dsa: bool,
    /// RSS ceiling the guard enforces, in bytes; 0 disables it.
    /// `COLI_RSS_GUARD_GB`, else the projected peak recorded at load.
    rss_limit_bytes: u64,
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
    /// Load-time `(layer, expert)` → tensor-plan/extent map for the streaming
    /// path. `Some` only when experts stream; a resident model never reads it.
    /// Removes the per-request re-derivation of both an expert's on-disk
    /// location and its quantized format.
    expert_index: Option<crate::concurrent::ExpertIndex>,
    /// fd → device-ordinal table for device-pure io claims
    /// (`COLI_IO_DEVICE_SCHED`, read once here at build — not OnceLock-latched,
    /// for the same A/B-aliasing reason as `SweepClock`). `Some` only when
    /// experts stream through >1 ring AND the shards actually span >1 device;
    /// everywhere else `None` keeps the concurrent lane's historical blind
    /// cursor, which is behavior-identical on a single device.
    fd_device_table: Option<std::collections::HashMap<std::os::unix::io::RawFd, u8>>,
    /// Bytes one token's routing touches (0 when experts are resident). The
    /// threshold that decides whether capacity or policy binds this deployment.
    expert_per_token_bytes: u64,
    /// Whether predictive eviction protection is active, resolved once at load
    /// from the budget against `expert_per_token_bytes` — the mechanism helps
    /// below that threshold and hurts above it. See `prefetch_protect_default`.
    prefetch_protect: bool,
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
    /// Per-layer routed-expert history (K-deep) from recent main forwards — the
    /// prefetch predictor's substrate. `Some` alongside `prefetch`.
    route_hist: Option<Mutex<RouteHistory>>,
    /// Set by a forward that logs into [`Self::route_hist`], consumed (and
    /// cleared) by `publish_lane_timings`. Distinguishes "this forward produced
    /// a new routing frame" from the batched path, which records per-sequence
    /// histories and leaves `route_hist` frozen.
    route_hist_epoch: std::sync::atomic::AtomicBool,
    /// Strategy that turns [`Self::route_hist`] into a ranked list of experts to
    /// prefetch for the next forward. Defaults to recency-weighted momentum.
    predictor: PredictSource,
    /// Predictor scoreboard (`COLI_PREDICT_EVAL=1`). Scores the router look-ahead,
    /// the statistical predictor and a previous-token baseline against the routing
    /// that actually happened, so the choice between them is measured rather than
    /// argued. `None` — and free — unless asked for.
    predict_eval: Option<Mutex<crate::predeval::PredictEval>>,
    /// Multi-path tiering: how many ranked candidates per layer to fully stream vs.
    /// merely page-cache-hint.
    prefetch_policy: PrefetchPolicy,
    /// Optional adaptive controller: when present, it overrides the warm-tier breadth
    /// each forward from observed prefetch used/wasted rates. `None` = static policy.
    prefetch_tuner: Option<PrefetchTuner>,
    /// LLC-miss counter for the decode thread, handed over by the binary that
    /// opened it (`perf_event_open` follows the *calling* thread, so the model
    /// cannot open its own — it is constructed on whichever thread loaded it).
    /// `None` unless `COLI_PERF_COUNTERS=1` and the kernel allowed it.
    perf_llc: Option<peregrine_io::PerfCounter>,
    /// Previous cumulative reading, so each forward sees a delta. Primed at
    /// attach time: without that the first delta is the whole counter and reads
    /// as an enormous spike.
    perf_llc_last: u64,
    /// EWMA (α = 0.3, matching [`PrefetchTuner`]) of the per-forward miss delta.
    /// `0.0` means "no sample yet", which is why the first observation seeds
    /// rather than blends.
    perf_llc_ewma: f32,
    /// Background prefetch lane: warms the next token's predicted experts into
    /// `ecache` on its own ring, so its *submissions* never queue behind the
    /// streaming lane's. Not "off the critical path" in any stronger sense — see
    /// [`prefetch_worker`] for what the separate ring does and does not buy.
    /// `Some` alongside `ecache`.
    prefetch: Option<PrefetchPool>,
    /// Layer-step clock + staleness policy shared with the prefetch workers, so a
    /// speculative warm whose layer window has passed is dropped before it costs a
    /// disk read (`COLI_PREFETCH_STALE_DROP`; see [`SweepClock`]). Always present —
    /// ticking it is two instructions, and gating its existence on the knob would
    /// put an `Option` probe in every layer of every forward.
    sweep: Arc<SweepClock>,
    /// Optional GPU VRAM expert tier (the 3rd lane). Built only when `COLI_GPU`
    /// is set and the `cuda` backend is available; `None` otherwise.
    gpu: Option<GpuTier>,
    /// Optional MTP head for speculative decode; `None` unless the checkpoint has
    /// the `model.layers.{n_layers}.eh_proj` tensors.
    mtp: Option<MtpHead>,
    /// RLM controller (always present; inert when `COLI_RLM` is unset, in which
    /// case [`crate::rlm::RLMController::should_recurse`] always returns `false`
    /// and the emitted token stream is byte-identical to the non-RLM path).
    /// Decides when a token's forward needs recursive refinement passes; the
    /// recursive pass itself is [`Self::forward_hidden_recursive`].
    rlm: crate::rlm::RLMController,
    /// Routing-frequency accumulator driving heat-ranked VRAM residency; `Some`
    /// only when a GPU tier exists (bumped during the forward, read by `reheat`).
    heat: Option<HeatTable>,
    /// Deferred-spill log (`COLI_GPU_SPILL`): `(layer, expert)` pairs the lane
    /// balancer verdicted [`crate::lane::Placement::GpuSpill`] mid-forward.
    /// Acting on the verdict needs `&mut GpuTier`, which a forward holds by `&`,
    /// so the pairs queue here and [`Self::reheat`] drains them between forwards
    /// into the heat snapshot ahead of the ranking. `Some` only when the knob is
    /// on and a GPU tier exists.
    spill_log: Option<Mutex<Vec<(usize, usize)>>>,
    /// Per-lane wall-time accumulator. `moe_forward_concurrent` bumps this each
    /// layer; the model reads and resets it between forwards to feed the bubble
    /// tuner. Always present (cheap — four atomic counters).
    lane_timings: Arc<crate::lane::LaneTimingsAccum>,
    /// Adaptive pipeline-bubble tuner. Reads per-forward [`LaneTimings`] snapshots
    /// and publishes a [`crate::lane::Bias`] the CPU/GPU balancer consumes.
    bubble: Mutex<crate::lane::BubbleTuner>,
    /// The per-forward tick that folds lane timings into the tuners and yields a
    /// telemetry snapshot (see [`crate::telemetry::PlanOptimizer`]).
    plan_optimizer: Mutex<crate::telemetry::PlanOptimizer>,
    /// Most recent telemetry snapshot, for `/metrics`-style readers.
    last_telemetry: Mutex<crate::telemetry::RuntimeTelemetry>,
    /// Last snapshot of per-forward per-lane wall time, for `/metrics` scrapes.
    last_lane: Mutex<crate::lane::LaneTimings>,
    /// Run-lifetime lane totals, and the number of forwards folded into them.
    ///
    /// Separate from `lane_timings` because that one is *reset* every forward by
    /// [`Self::publish_lane_timings`] — it exists to feed the tuner a per-forward
    /// sample, so by construction it can never answer "where did this run's time
    /// go". The four lane counters were being collected and consumed only by an
    /// adaptive controller, with no way out to an operator; this is the way out.
    /// Cheap: four atomics bumped once per forward, not per layer.
    lane_totals: Arc<crate::lane::LaneTimingsAccum>,
    lane_forwards: std::sync::atomic::AtomicU64,
    /// Adaptive io_uring worker-cap tuner. Consumes per-forward `io_us` from
    /// [`Self::publish_lane_timings`] and — when `COLI_IO_TUNE` is on — applies
    /// the recommended `(bounded, unbounded)` cap to every reactor between
    /// forwards. Correctness-neutral: only worker parallelism changes.
    io_tuner: crate::iotune::IoTuner,
    /// Tracks the last-applied cap so we don't re-issue the `register_iowq_max_workers`
    /// syscall when nothing has changed. Also lets tests observe the applied value.
    last_iowq: Mutex<Option<crate::iotune::IowqCap>>,
    /// Current workload class (from the serving layer's prompt classifier) —
    /// selects per-class prefetch-breadth overrides. Defaults to `Prose`
    /// (== base policy unless the operator set per-class envs).
    workload_class: Mutex<crate::workload::TokenClass>,
    /// CPU-lane worker count the governors may adjust at runtime, clamped to
    /// `[2, workers]`. Read once per forward when building `ForwardCtx`; the
    /// thermal / power / bandwidth governors write it between forwards.
    effective_workers: std::sync::atomic::AtomicUsize,
    /// Sensor-governor state (tick counter, RAPL meter, bandwidth EWMA).
    governor: Mutex<GovernorState>,
    /// EWMA of normalized routing entropy (0 = fully repetitive routing,
    /// 1 = maximally dispersed). Drives the entropy-adaptive prefetch breadth.
    entropy_ewma: Mutex<f32>,
    /// Long-term expert co-activation counts (runtime + cross-session fusion
    /// substrate). Fed from the routing history each forward; persisted in
    /// `route_stats.json`.
    coactivation: Mutex<crate::predict::CoActivation>,
    /// Current affinity ordering hints (fused pairs + hyperedge components),
    /// rebuilt from `coactivation` every 64 forwards. Arc-swapped so forwards
    /// borrow a stable snapshot without holding the lock.
    affinity: Mutex<Arc<crate::concurrent::AffinityHints>>,
    /// Online learned scheduler (bandit / Q-learning over knob configs), `None`
    /// unless `COLI_LEARN_SCHED=1` / `COLI_RL_SCHED=1`. Policy persisted in
    /// `route_stats.json`.
    learner: Mutex<Option<crate::learn::Learner>>,
    /// Prefetch-distance target the learner chose (0 = none); applied to the
    /// tuner in `forward_hidden` where `&mut` is available.
    learned_prefetch: std::sync::atomic::AtomicUsize,
    /// Wall-clock of the previous `publish_lane_timings` — the inter-forward
    /// interval is the decode-latency reward signal for the learners.
    last_forward_at: Mutex<Option<std::time::Instant>>,
    /// The checkpoint directory this model was loaded from — kept so cross-session
    /// artifacts (route stats, layout hints, kernel tuning) can be re-persisted
    /// without threading the path through the caller each time.
    checkpoint_dir: std::path::PathBuf,
    /// Calibration capture (`COLI_CALIB_CAPTURE=<out.json>`, ideas #7): the
    /// output path and the running per-layer channel sums. `None` in serving —
    /// the env is read once at load (no OnceLock; `enable_calib_capture` is
    /// the test/tool seam), and a `None` here costs one branch per sparse
    /// layer per forward.
    calib: Option<(std::path::PathBuf, Mutex<CalibAccum>)>,
    /// Optional expert-order hint per layer, loaded from `<dir>/schedule.json`
    /// (emitted by `peregrine-layout-reorg`). When present, the concurrent
    /// scheduler sorts each layer's streamed-expert plan by this order so the
    /// batched io_uring submit issues contiguous-offset reads first — the
    /// disk-queue coalescing win the tool's community detection is aiming for.
    /// Correctness-neutral: reorder ≠ different bytes.
    layout_schedule: Option<Vec<Vec<u32>>>,
}

/// Per-sequence KV cache: one [`LayerKv`] per layer. Owned by the batching
/// scheduler rather than the [`Model`], so a single resident model can decode
/// many independent sequences concurrently via [`Model::forward_step_batched`].
///
/// Sequences seeded from the same prompt prefix **share** its storage by
/// refcount and own only their own tail, so admitting a request against a
/// cached system prompt costs a pointer rather than a copy of it. A
/// block-pooled variant that also reclaims growth fragmentation is a follow-up;
/// what makes either safe is that readers go through `KvSpan` and cannot tell a
/// split cache from a contiguous one.
pub struct SeqKv {
    layers: Vec<LayerKv>,
}

/// `COLI_KV_DTYPE`: the element type every KV cache in this process is built
/// with. Default `f32` — the historical, output-neutral layout.
///
/// An unrecognised value is **reported and ignored**, not silently coerced: a
/// typo'd dtype that quietly halved precision would change token values with
/// nothing in the output saying so.
/// `COLI_GPU_SPILL=1`: act on the lane balancer's [`crate::lane::Placement::GpuSpill`]
/// verdicts by queueing the spilled `(layer, expert)` pairs for the next
/// residency generation (see `Model::spill_log`). Off by default — the verdict
/// stays advisory, the historical behavior.
fn gpu_spill_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("COLI_GPU_SPILL").as_deref(), Ok("1") | Ok("true")))
}

/// Fold drained spill verdicts into a heat snapshot ahead of the residency
/// ranking: each spill event bumps its expert's count by one, so an expert the
/// balancer kept wanting resident outranks an equally-routed one it never
/// asked for — proportional to how often it was asked, rather than jumping the
/// whole queue on a single verdict. Pure so the arithmetic is testable without
/// a GPU.
fn merge_spills(counts: &mut [u32], spills: &[(usize, usize)], n_experts: usize) {
    for &(layer, e) in spills {
        if let Some(c) = counts.get_mut(layer * n_experts + e) {
            *c = c.saturating_add(1);
        }
    }
}

pub fn kv_dtype() -> KvDtype {
    static V: std::sync::OnceLock<KvDtype> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let Some(s) = std::env::var("COLI_KV_DTYPE").ok() else { return KvDtype::F32 };
        match KvDtype::parse(&s) {
            Some(dt) => dt,
            None => {
                peregrine_core::note_advisory_err(
                    "COLI_KV_DTYPE",
                    &format!("unrecognised value {s:?}; using f32 (accepted: f32, f16)"),
                );
                KvDtype::F32
            }
        }
    })
}

impl SeqKv {
    /// A fresh, empty cache sized for a model with `cfg`'s dimensions, in the
    /// process-wide [`kv_dtype`].
    pub fn new(cfg: &Cfg) -> SeqKv {
        SeqKv::with_dtype(cfg, kv_dtype())
    }

    /// [`Self::new`] at an explicit element type, so a test can exercise the
    /// narrow path without a process-wide environment variable.
    pub fn with_dtype(cfg: &Cfg, dt: KvDtype) -> SeqKv {
        let (kvl, qkr) = (cfg.kv_lora as usize, cfg.qk_rope as usize);
        SeqKv { layers: (0..cfg.n_layers).map(|_| LayerKv::with_dtype(kvl, qkr, dt)).collect() }
    }

    /// Positions cached so far (the sequence length); all layers share it.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |k| k.len())
    }

    /// Whether no positions are cached yet.
    pub fn is_empty(&self) -> bool {
        self.layers.first().is_none_or(|k| k.is_empty())
    }

    /// Rewind every layer to `new_len` (speculative-decode reject cleanup).
    pub fn truncate(&mut self, new_len: usize) {
        for k in &mut self.layers {
            k.truncate(new_len);
        }
    }

    /// Logical bytes across every layer: what this sequence would cost holding
    /// every position privately. Sharing a prefix does not shrink it, so a
    /// budget built on it is never an under-estimate — see
    /// [`Self::owned_bytes`] + [`Self::shared_prefix`] for the exact split.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|k| k.bytes()).sum()
    }

    /// DSA indexer keys cached so far, across the shared prefix and this
    /// sequence's own tail. Equals [`Self::len`] once the indexer is running.
    #[cfg(test)]
    fn index_len(&self) -> usize {
        self.layers.first().map_or(0, |k| k.index_len())
    }

    /// Bytes held privately by this sequence, excluding any shared prefix.
    pub fn owned_bytes(&self) -> usize {
        self.layers.iter().map(|k| k.owned_bytes()).sum()
    }

    /// The shared prefix this sequence was seeded from: a stable identity and
    /// the bytes of the whole shared allocation. Sequences seeded from the same
    /// prefix-cache entry return the same identity, which is what lets a budget
    /// over many sequences charge one shared prefix once instead of once each.
    ///
    /// Layer 0 supplies the identity: a `SeqKv`'s layers are always cloned
    /// together from one parent, so if layer 0's prefix matches, all of them do.
    pub fn shared_prefix(&self) -> Option<(usize, usize)> {
        let id = self.layers.first()?.shared_id()?;
        Some((id, self.layers.iter().map(|k| k.shared_bytes()).sum()))
    }

    /// A copy holding only the first `n` positions of every layer.
    ///
    /// This is what makes a prompt prefix shareable: two prompts agreeing on
    /// their first `n` tokens produce identical KV for those positions, because
    /// each position attends only its causal prefix. So a cached sequence can
    /// seed any prompt it is a prefix of, and the result is the same as having
    /// prefilled those tokens directly.
    pub fn clone_prefix(&self, n: usize) -> SeqKv {
        SeqKv { layers: self.layers.iter().map(|k| k.clone_prefix(n)).collect() }
    }

    /// The first `n` positions of every layer as plain `f32` vectors — the
    /// disk-persistence seam (`COLI_KV_STORE_DIR`). Reads go through the same
    /// [`KvSpan`](crate::attention::KvSpan)s attention itself uses, so the
    /// export sees exactly the values a forward would.
    ///
    /// Round-trip exactness: an `f16` cache exports through the widening
    /// `f16 → f32` (exact), and [`Self::import`] re-narrows values that were
    /// `f16` back to the identical bits — so export/import is lossless for
    /// both dtypes *when the importing cache uses the same dtype*, which the
    /// disk store enforces via its header.
    pub fn export_prefix(&self, n: usize) -> KvExport {
        let n = n.min(self.len());
        let layers = self
            .layers
            .iter()
            .map(|k| {
                // The indexer stream is exported only when it is row-aligned
                // with the latents from position 0. Mid-sequence DSA enablement
                // leaves `ix` shorter than `len` with its rows belonging to
                // *later* positions; exporting those as rows 0..ix_rows would
                // rebuild a silently misaligned cache. Dropping `ix` is safe —
                // it degrades DSA selection for the restored prefix, never
                // correctness of the latents.
                let ixw = if k.index_len() == k.len() { k.ix_width() } else { 0 };
                let mut lc = Vec::new();
                let mut rc = Vec::new();
                let mut ix = Vec::new();
                k.lc_span(n).extend_f32(n, k.kv_lora_width(), &mut lc);
                k.rc_span(n).extend_f32(n, k.qk_rope_width(), &mut rc);
                if ixw > 0 {
                    k.ix_span(n).extend_f32(n, ixw, &mut ix);
                }
                KvLayerExport { lc, rc, ix, ix_width: ixw }
            })
            .collect();
        KvExport { n, layers }
    }

    /// Rebuild a cache from an export — the inverse of [`Self::export_prefix`].
    /// Widths are validated against `cfg` before any row lands, and every row
    /// goes through the same order-checked [`LayerKv::append`] path prefill
    /// uses, so an import can never construct a cache prefill could not.
    pub fn import(cfg: &Cfg, dt: KvDtype, ex: &KvExport) -> Result<SeqKv, Error> {
        let (kvl, qkr) = (cfg.kv_lora as usize, cfg.qk_rope as usize);
        if ex.layers.len() != cfg.n_layers as usize {
            return Err(Error::Format(format!(
                "KV import: {} layers in the export, model has {}",
                ex.layers.len(),
                cfg.n_layers
            )));
        }
        let mut kv = SeqKv::with_dtype(cfg, dt);
        for (k, le) in kv.layers.iter_mut().zip(&ex.layers) {
            if le.lc.len() != ex.n * kvl || le.rc.len() != ex.n * qkr {
                return Err(Error::Format(format!(
                    "KV import: layer stream lengths ({}, {}) do not match {} rows of ({kvl}, {qkr})",
                    le.lc.len(),
                    le.rc.len(),
                    ex.n
                )));
            }
            if le.ix_width > 0 && le.ix.len() != ex.n * le.ix_width {
                return Err(Error::Format(format!(
                    "KV import: indexer stream is {} elements, expected {} rows of {}",
                    le.ix.len(),
                    ex.n,
                    le.ix_width
                )));
            }
            for r in 0..ex.n {
                k.append(r, &le.lc[r * kvl..(r + 1) * kvl], &le.rc[r * qkr..(r + 1) * qkr])?;
            }
            if le.ix_width > 0 {
                for row in le.ix.chunks_exact(le.ix_width) {
                    k.append_index_key(row);
                }
            }
        }
        Ok(kv)
    }
}

/// One layer's exported KV streams (see [`SeqKv::export_prefix`]): flat
/// row-major `f32`, `n` rows each of the layer's own widths.
pub struct KvLayerExport {
    pub lc: Vec<f32>,
    pub rc: Vec<f32>,
    /// DSA indexer keys; empty (with `ix_width == 0`) when the layer has none
    /// or they were not aligned enough to export.
    pub ix: Vec<f32>,
    pub ix_width: usize,
}

/// A `SeqKv` prefix as plain vectors — the unit the serve-side disk store
/// (`COLI_KV_STORE_DIR`) serializes and restores.
pub struct KvExport {
    /// Positions exported.
    pub n: usize,
    pub layers: Vec<KvLayerExport>,
}

/// Memory this process may actually use, in bytes (0 if unreadable): the smaller of
/// the host's `MemAvailable` and whatever its cgroup permits.
///
/// `/proc/meminfo` is not namespaced, so inside a container it describes the host.
/// Sizing a cache against it there is how a small container on a large machine gets
/// OOM-killed while every projection in the engine reports room to spare — see
/// [`crate::ram::effective_available`].
fn mem_available_bytes() -> u64 {
    crate::ram::effective_available(host_mem_available_bytes(), cgroup_available_bytes())
}

/// Accept a draft run against the model's own greedy predictions.
///
/// `rows` is `[1 + drafts.len(), vocab]` — row `k` predicts the token *after*
/// draft `k-1`, so row 0 judges the first draft. Returns how many drafts were
/// accepted and the model's prediction at the first rejected position, which is
/// the next round's token and is already paid for by this forward.
///
/// **This is the definition of "greedy-identical" and there is exactly one of
/// it.** The model-level verify and the serving engine both call it; two copies
/// of this rule would be two chances for speculation to start emitting
/// something greedy decoding would not.
pub fn accept_run(rows: &[f32], vocab: usize, drafts: &[i32]) -> (usize, i32) {
    let row = |k: usize| rows.get(k * vocab..(k + 1) * vocab);
    let mut k = 0usize;
    while k < drafts.len() {
        let Some(r) = row(k) else { break };
        if crate::sample::argmax(r) as i32 != drafts[k] {
            break;
        }
        k += 1;
    }
    let next = row(k).map_or(0, |r| crate::sample::argmax(r) as i32);
    (k, next)
}

/// Accept a draft run against a **sampled** target distribution — the
/// temperature > 0 twin of [`accept_run`], and the production caller of
/// [`crate::speculative_sample`].
///
/// `rows` is `[1 + drafts.len(), vocab]` exactly as in [`accept_run`], and
/// `draft_q[k]` is the distribution draft `k` was actually drawn from
/// ([`Model::mtp_draft_sampled`]). Returns how many drafts were accepted and the
/// token to emit after them.
///
/// **The emitted sequence is not the one an unspeculated sampled request would
/// have produced, and cannot be.** `accept_run`'s guarantee is *sequence*
/// identity with greedy decoding; this one's is only *distributional* identity
/// with sampling at the request's temperature — the tokens differ, the
/// distribution does not. Rejection sampling also draws two uniforms per draft
/// where plain decode draws one per token, so the RNG stream advances
/// differently: a seeded request is reproducible against itself, not against the
/// same seed with `COLI_DRAFT` unset. That is why this path is opt-in.
///
/// On rejection the resampled token **ends the run**: rows past a rejected
/// position were computed conditioned on a token that is no longer being
/// emitted, so nothing there is valid to accept.
pub fn accept_run_sampled(
    rows: &[f32],
    vocab: usize,
    drafts: &[i32],
    draft_q: &[Vec<f32>],
    sampler: &mut Sampler,
) -> (usize, i32) {
    let row = |k: usize| rows.get(k * vocab..(k + 1) * vocab);
    let mut k = 0usize;
    while k < drafts.len() {
        // A missing row or a missing/short `q` is a shape fault, not a
        // rejection: fall through to sampling row `k` normally, which is what
        // this round would have emitted with no speculation at all. Guessing a
        // uniform `q` instead would feed `speculative_sample` a ratio computed
        // against a distribution nothing was drawn from.
        let (Some(r), Some(q)) = (row(k), draft_q.get(k)) else { break };
        let drafted = match usize::try_from(drafts[k]).ok().filter(|&d| d < vocab && d < q.len()) {
            Some(d) => d,
            None => break, // out-of-vocab draft: cannot be scored, so reject it
        };
        let p = sampler.distribution(r).to_vec();
        let (u_accept, u_resample) = (sampler.uniform(), sampler.uniform());
        let emitted = crate::speculative_sample(&p, q, drafted, u_accept, u_resample);
        if emitted != drafted {
            // Rejected: the residual sample replaces the draft and terminates
            // the run. (`speculative_sample` only resamples when `p/q < 1`, and
            // the residual `(p-q)+` is then zero at `drafted` — so this compare
            // is a faithful "was it accepted", not an approximation of one.)
            return (k, emitted as i32);
        }
        k += 1;
    }
    // Every draft accepted (or the run stopped on a shape fault): the bonus
    // token comes from the row past the accepted run, sampled normally.
    let next = row(k).map_or(0, |r| sampler.pick(r, -1) as i32);
    (k, next)
}

/// One sequence's speculative round: what was confirmed, and what comes next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    /// Tokens confirmed this round — `next` followed by every accepted draft.
    /// Exactly what greedy decoding would have emitted, which is the property
    /// speculation has to preserve to be worth anything.
    pub tokens: Vec<i32>,
    /// The model's prediction after the accepted run: the next round's `next`,
    /// already paid for by this forward.
    pub next: i32,
    /// Drafts accepted this round (`tokens.len() - 1`), for an acceptance-rate
    /// counter. A draft depth whose acceptance sits at its own ceiling is the
    /// signal that the depth, not the mechanism, is the limit.
    pub accepted: usize,
}

/// `MemAvailable` from `/proc/meminfo`, in bytes (0 if unreadable). The host's view.
fn host_mem_available_bytes() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return 0 };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

/// The `0::<path>` line of a `/proc/self/cgroup` dump — the process's own v2
/// cgroup, relative to the hierarchy root. Pure so the parse is testable.
fn self_cgroup_v2_rel(contents: &str) -> Option<&str> {
    contents.lines().find_map(|l| l.strip_prefix("0::")).map(str::trim)
}

/// Every directory from the process's cgroup up to the hierarchy root, leaf
/// first. The kernel enforces the *tightest* limit on this path, and a
/// transient `systemd-run --scope -p MemoryMax=` puts its limit several levels
/// below the root — the 2026-08-15 stage-5 arm OOM'd exactly there: auto
/// ecache read the root's `max`, sized 26 GB inside a 34G scope that already
/// held the weights, and the kernel ended the arm at boot. Pure for the test.
fn cgroup_walk_dirs(rel: &str) -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut dir = root.join(rel.trim_start_matches('/'));
    let mut dirs = Vec::new();
    loop {
        dirs.push(dir.clone());
        if dir == root || !dir.pop() {
            break;
        }
    }
    dirs
}

/// Bytes left inside this process's cgroup memory controller, `None` when there is no
/// limit or no controller. Tries v2's unified hierarchy first, then v1.
///
/// v2 is probed twice, and the tighter answer wins: once walking
/// `/proc/self/cgroup` up to the root (the transient-scope case above), and
/// once at the cgroup *root* paths — because in a containerized process the
/// container's cgroup is mounted as its root and `/proc/self/cgroup` may name
/// a path its mount namespace does not expose.
fn cgroup_available_bytes() -> Option<u64> {
    let mut best: Option<u64> = None;
    let mut consider = |max: Option<String>, cur: Option<String>| {
        if let (Some(max), Some(cur)) = (max, cur) {
            if let Some(v) = crate::ram::cgroup_v2_available(&max, &cur) {
                best = Some(best.map_or(v, |b| b.min(v)));
            }
        }
    };
    if let Some(rel) = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .as_deref()
        .and_then(self_cgroup_v2_rel)
    {
        for dir in cgroup_walk_dirs(rel) {
            consider(
                read_cgroup_file(&dir.join("memory.max").to_string_lossy()),
                read_cgroup_file(&dir.join("memory.current").to_string_lossy()),
            );
        }
    }
    consider(
        read_cgroup_file("/sys/fs/cgroup/memory.max"),
        read_cgroup_file("/sys/fs/cgroup/memory.current"),
    );
    if best.is_some() {
        return best;
    }
    let limit = read_cgroup_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")?;
    // An unreadable usage file means "unknown", which the parser reads as zero
    // used. That reports the whole limit as available, which is the same
    // direction as having no cgroup information at all — the projection is then
    // no worse than it was before this probe existed.
    let usage = match read_cgroup_file("/sys/fs/cgroup/memory/memory.usage_in_bytes") {
        Some(u) => u,
        None => String::from("0"),
    };
    crate::ram::cgroup_v1_available(&limit, &usage)
}

/// One cgroup control file, `None` when it is not there.
///
/// Absence is the ordinary case — no memory controller, or simply not running in
/// a container — so it is not reported. A *genuine* read failure is advisory
/// (`COLI_DEBUG=1` surfaces it): the host's own `MemAvailable` still stands, so
/// the run continues with the figure it would have used anyway.
fn read_cgroup_file(path: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(s) => Some(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            peregrine_io::note_advisory_err("cgroup memory limit read", &e);
            None
        }
    }
}

/// How `COLI_ECACHE_GB` was spelled: an explicit byte budget, or the ds4-style
/// `auto` fraction of `MemAvailable`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EcacheSpec {
    /// Explicit GiB (`0` = disabled) — the historical spelling, byte-for-byte.
    Fixed(usize),
    /// `auto`: this fraction of `MemAvailable` at resolution time.
    AutoFrac(f64),
}

/// Pure parse of the `COLI_ECACHE_GB` spelling (`frac` is
/// `COLI_ECACHE_AUTO_FRAC`, consulted only for `auto`). `None` = unparseable;
/// the caller notes the advisory and disables the cache, the historical
/// behavior for garbage input.
///
/// The default fraction is ds4/DwarfStar's budget rule — 80% of the backend's
/// recommended working set — which lands here as 80% of post-load
/// `MemAvailable`: read after the resident weights are mapped, so they are
/// already netted out, exactly the subtraction ds4 does explicitly. Clamped to
/// (0, 0.95]: a fraction of 1.0 would hand the cache every byte the box has
/// left, and `cap_ecache_budget`'s reserve-and-safety cut still applies on top.
fn parse_ecache_spec(v: &str, frac: Option<&str>) -> Option<EcacheSpec> {
    const GIB: f64 = (1u64 << 30) as f64;
    let t = v.trim();
    if t.eq_ignore_ascii_case("auto") {
        let f = frac
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite() && *f > 0.0)
            .map(|f| f.min(0.95))
            .unwrap_or(0.80);
        return Some(EcacheSpec::AutoFrac(f));
    }
    t.parse::<f64>().ok().map(|g| EcacheSpec::Fixed((g.max(0.0) * GIB) as usize))
}

/// Warm-cache byte budget from the environment: `COLI_ECACHE_GB` (GiB float,
/// or `auto` for a `COLI_ECACHE_AUTO_FRAC` share of `MemAvailable`) if set,
/// else 10% of `MemAvailable` capped at 2 GiB. `0` (or an unparseable value)
/// disables the cache. Kept independent of the streaming-vs-resident RAM
/// heuristic so the two knobs don't interfere.
fn ecache_budget_bytes() -> usize {
    const GIB: f64 = (1u64 << 30) as f64;
    match std::env::var("COLI_ECACHE_GB") {
        Ok(v) => {
            let frac_env = match std::env::var("COLI_ECACHE_AUTO_FRAC") {
                Ok(f) => Some(f),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => {
                    peregrine_io::note_advisory_err("COLI_ECACHE_AUTO_FRAC read", &e);
                    None
                }
            };
            return match parse_ecache_spec(&v, frac_env.as_deref()) {
                Some(EcacheSpec::Fixed(b)) => b,
                Some(EcacheSpec::AutoFrac(f)) => (f * mem_available_bytes() as f64) as usize,
                None => {
                    let e = std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("expected GiB or 'auto', got {v:?}"),
                    );
                    peregrine_io::note_advisory_err("COLI_ECACHE_GB parse (cache disabled)", &e);
                    0
                }
            };
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(e) => peregrine_io::note_advisory_err("COLI_ECACHE_GB read", &e),
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
        .saturating_mul(experts_per_batch())
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

/// Tokens between RSS-guard checks. Frequent enough to catch growth before the
/// kernel does, rare enough that one small `/proc` read costs nothing. colibrì
/// checks on the same cadence.
const RSS_GUARD_EVERY: usize = 16;

/// Whether MLA weight absorption runs on the shared attention path
/// (`COLI_MLA_ABSORB`). Default **off**: absorb is algebraically equal to the
/// dense reconstruction but not numerically identical, since the dense path
/// pushes the cached latent back through the quantized `kv_b` and absorb folds
/// that weight into the query instead. It therefore changes token values, which
/// puts it in the same class as `COLI_ROUTE_MIN_SHARE` — size the cost with
/// `Model::prediction_flip_rate` before turning it on.
/// Whether the DSA lightning indexer runs (`COLI_DSA`). Default **off**: the
/// indexer selects a subset of cached keys, so it changes token values, and the
/// laptop-converted container skipped the indexer tensors entirely — with no
/// indexer in the checkpoint the flag is inert.
fn dsa_enabled() -> bool {
    matches!(std::env::var("COLI_DSA").ok().as_deref(), Some("1") | Some("true"))
}

fn absorb_enabled() -> bool {
    matches!(std::env::var("COLI_MLA_ABSORB").ok().as_deref(), Some("1") | Some("true"))
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

/// Capacity for the O_DIRECT aligned slab pool: the largest single **read** the
/// streaming lane will issue for a routed expert, plus alignment slack (the
/// 4096-aligned superset of a region can exceed it by up to two blocks).
///
/// That is the largest *extent*, not the largest tensor. When an expert's three
/// weight regions are contiguous — which they are for ~99.5 % of this container —
/// the lane coalesces them into one read, so sizing this per tensor under-counts
/// threefold (18.9 MB read against a 6.3 MB pool at int4, and 37.7 MB against
/// 12.6 MB on the int8 MTP layer). A too-small pool is not a correctness problem
/// on the O_DIRECT path, which falls back to a one-off allocation, but it does
/// feed `stream_transient_reserve` → `ram::project_load`, and under-projecting
/// the streaming reserve is how a run gets OOM-killed with no warning.
fn max_expert_region_bytes(st: &SafeTensors) -> usize {
    use std::collections::HashMap;
    // Per (expert, is_scale): the summed bytes of that group, which is the extent
    // length whenever the group is contiguous. Experts differ in size across
    // layers on a precision-tiered container, so this maxes over all of them.
    let mut runs: HashMap<(&str, bool), usize> = HashMap::new();
    for t in st.tensors() {
        let Some((head, rest)) = t.name.split_once(".mlp.experts.") else { continue };
        let Some((eid, _)) = rest.split_once('.') else { continue };
        let key_end = head.len() + ".mlp.experts.".len() + eid.len();
        let Some(key) = t.name.get(..key_end) else { continue };
        *runs.entry((key, t.name.ends_with(".qs"))).or_insert(0) += t.nbytes.max(0) as usize;
    }
    let max = runs.into_values().max().unwrap_or(32 << 20);
    max + 2 * peregrine_io::ALIGN
}

/// The layer-step clock the forward loops advance and the prefetch workers read.
///
/// Exists because a speculative warm is only worth its disk read *inside the layer
/// boundary it was emitted for*. The 2026-08-13 defaults run measured the failure
/// mode this closes: at B=16 the rings run at 93 % duty, the unbounded prefetch
/// queue backlogs by minutes, and by the time the lane services an item the demand
/// path has long since streamed — and the cache long since evicted — that expert.
/// 40 352 of 41 159 speculative reads (98.6 %) were classified wasted, ~12.6 % of
/// all disk reads on an engine whose wall clock *is* its disk time. A late
/// speculation is not a cheap miss; it is a demand read's bandwidth spent on a
/// guess about a token that already happened.
///
/// `step` advances once per layer the forward sweep executes (decode, batched
/// decode, and external-KV prefill — the paths whose demand reads churn the cache).
/// A `Warm` batch is stamped with `step` at emit; the worker drops items once
/// `step` has moved more than `slack` past their stamp, *before* paying the read.
/// `slack` is in layer-steps: an item emitted during step `s` targets the layer
/// executing at `s + 1`, so the default slack of 1 keeps exactly the window the
/// emitter designed for. `slack == u64::MAX` disables the gate — the historical
/// behaviour, and the default until the A/B says otherwise
/// (`COLI_PREFETCH_STALE_DROP=1`, slack via `COLI_PREFETCH_STALE_SLACK`).
///
/// Read from env at model build, not through a process-global `OnceLock`: the
/// route-min-share measurement was voided by exactly that latch (both A/B arms in
/// one process saw the first arm's value), and this knob exists to be A/B'd.
struct SweepClock {
    /// Layer-step ordinal, monotonically increasing across the model's lifetime.
    step: std::sync::atomic::AtomicU64,
    /// Layer-steps past its stamp a warm item stays serviceable; `u64::MAX` = gate off.
    slack: std::sync::atomic::AtomicU64,
}

impl SweepClock {
    fn from_env() -> SweepClock {
        let on = matches!(std::env::var("COLI_PREFETCH_STALE_DROP").as_deref(), Ok("1") | Ok("true"));
        let slack = if on { env_usize("COLI_PREFETCH_STALE_SLACK", 1) as u64 } else { u64::MAX };
        SweepClock {
            step: std::sync::atomic::AtomicU64::new(0),
            slack: std::sync::atomic::AtomicU64::new(slack),
        }
    }

    /// The current layer-step, for stamping a `Warm` batch at emit time.
    fn now(&self) -> u64 {
        self.step.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Advance one layer-step. Called by the forward loops as each layer executes.
    fn tick(&self) {
        self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether a warm item stamped at `stamp` is past its service window at `now`.
///
/// Pure, so the boundary cases are testable without a model or a lane thread. A
/// `u64::MAX` stamp (the bulk warms: `tiers.json` seed, expert replicas — deliberate
/// transfers, not layer-boundary speculation) saturates to a zero age and is never
/// stale; likewise `slack == u64::MAX` (gate off) admits any age.
fn warm_item_is_stale(now: u64, stamp: u64, slack: u64) -> bool {
    now.saturating_sub(stamp) > slack
}

/// Messages to the background prefetch lane.
enum PrefetchMsg {
    /// Warm these experts into the shared cache (skipping ones already resident).
    /// The `u64` is the [`SweepClock`] stamp at emit; `u64::MAX` marks a deliberate
    /// (non-speculative) warm the stale gate must never drop.
    Warm(Vec<crate::concurrent::PrefetchItem>, u64),
    /// Page-cache-hint these low-confidence experts via `fadvise(WILLNEED)` — no
    /// streaming, no cache insert (multi-path tier 2).
    Hint(Vec<crate::concurrent::HintItem>),
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

/// A pool of background prefetch lanes, each with its own io_uring ring, so per-stream
/// prefetch proceeds in parallel. Lane 0 serves the single-stream path; batched serving
/// spreads sequences across lanes by `seq_id % lanes` for parallel-async prefetch.
struct PrefetchPool {
    lanes: Vec<PrefetchHandle>,
}

impl PrefetchPool {
    /// The lane assigned to work item `i` (round-robin). Never empty.
    ///
    /// **Only [`Model::enqueue_seq_prefetch`] passes anything but 0**, and that is a
    /// policy rather than an oversight. The per-layer emitter and the router
    /// look-ahead are *staggered ahead of the compute cursor*: layer L's warm should
    /// land before layer L+1's, and a lane is FIFO only with respect to itself, so
    /// spreading one stream's emissions across lanes would let a later layer's read
    /// overtake an earlier one's — the opposite of what look-ahead is for. One
    /// logical stream therefore belongs on one lane.
    ///
    /// The genuinely spreadable callers are the bulk, order-free ones — the
    /// `tiers.json` seed and `enqueue_expert_replicas` — which enqueue a *set*. They
    /// stay on lane 0 until the lane-count measurement says lanes pay for themselves
    /// at all; parallelising a once-per-load warm on the strength of a plausible
    /// story is how the rest of this file grew knobs nobody measured.
    fn lane(&self, i: usize) -> &PrefetchHandle {
        &self.lanes[i % self.lanes.len()]
    }

    /// Block until every lane has drained its queue (FIFO barrier across the pool).
    fn barrier(&self) {
        for l in &self.lanes {
            let (tx, rx) = crossbeam_channel::bounded(1);
            if l.tx.send(PrefetchMsg::Sync(tx)).is_ok() && rx.recv().is_err() {
                peregrine_io::note_advisory_err("prefetch lane sync", &"lane exited before acking the barrier");
            }
        }
    }

    /// Drain and join every lane (called on `Model` drop).
    fn stop(&mut self) {
        for l in &mut self.lanes {
            if l.tx.send(PrefetchMsg::Stop).is_err() {
                peregrine_io::note_advisory_err("prefetch lane stop", &"lane already exited");
            }
            if let Some(j) = l.join.take() {
                if j.join().is_err() {
                    peregrine_io::note_advisory_err("prefetch lane join", &"lane thread panicked");
                }
            }
        }
    }
}

/// Number of parallel prefetch lanes. `COLI_PREFETCH_LANES` (default 1, floored at 1).
fn prefetch_lanes() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_usize("COLI_PREFETCH_LANES", 1).max(1))
}

/// Spawn `lanes` prefetch workers sharing `cache`, each on its own ring. Read
/// `COLI_PREFETCH_VERIFY` fresh here (not a process-global) so it can be toggled at
/// load time: when set, each worker re-reads and byte-compares its speculative loads.
fn spawn_prefetch_pool(
    cache: &Arc<Mutex<WarmCache>>,
    st: &SafeTensors,
    direct: bool,
    lanes: usize,
    sweep: &Arc<SweepClock>,
) -> Result<PrefetchPool, Error> {
    let lanes = lanes.max(1);
    let verify = matches!(std::env::var("COLI_PREFETCH_VERIFY").as_deref(), Ok("1") | Ok("true"));
    let mut handles = Vec::with_capacity(lanes);
    for i in 0..lanes {
        let mut reactor = Reactor::new(64).ctx(|| "prefetch io_uring reactor init".to_string())?;
        if direct {
            reactor.configure_slab(max_expert_region_bytes(st), 2);
        }
        let cache = Arc::clone(cache);
        let sweep = Arc::clone(sweep);
        let (tx, rx) = crossbeam_channel::unbounded::<PrefetchMsg>();
        let join = std::thread::Builder::new()
            .name(format!("peregrine-prefetch-{i}"))
            .spawn(move || {
                numa_pin_worker(i); // opt-in NUMA affinity (COLI_NUMA_PIN=1)
                prefetch_worker(reactor, cache, rx, direct, verify, sweep)
            })
            .map_err(|e| Error::Format(format!("spawn prefetch thread: {e}")))?;
        handles.push(PrefetchHandle { tx, join: Some(join) });
    }
    Ok(PrefetchPool { lanes: handles })
}

/// Pin the calling worker thread to a CPU chosen round-robin from the discovered
/// NUMA topology. CPUs are enumerated node-grouped (all of node 0, then node 1,
/// …), so consecutive worker indices land on the same node first — keeping a
/// pool's memory traffic node-local before spilling to the next socket.
/// **Opt-in**: no-op unless `COLI_NUMA_PIN=1` (pinning on a single-node desktop
/// usually just fights the OS scheduler). Also installed as the `peregrine-par`
/// worker-startup hook (see [`install_numa_pin_hook`]).
fn numa_pin_worker(worker: usize) {
    if !matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("1") | Ok("true")) {
        return;
    }
    let topo = peregrine_io::topo::snapshot();
    let cpus: Vec<u32> = topo.numa.iter().flat_map(|n| n.cpus.iter().copied()).collect();
    if cpus.is_empty() {
        return;
    }
    peregrine_io::pin_current_thread(cpus[worker % cpus.len()]);
}

/// Install [`numa_pin_worker`] as the `peregrine-par` pool's worker-startup
/// hook, and — when NUMA pinning is on and the box is multi-node — the
/// worker→node group map that switches the pool to hierarchical (two-level)
/// dispatch: contiguous per-node blocks, then per-worker chunks, so a node's
/// workers touch a node-local slice of every parallel range. Idempotent; must
/// run before the global pool's first `par_*` call to affect its workers (the
/// pool is built lazily), which is why `Model::load` calls this first thing.
fn install_numa_pin_hook() {
    peregrine_par::set_worker_start_hook(numa_pin_worker);
    if !matches!(std::env::var("COLI_NUMA_PIN").as_deref(), Ok("1") | Ok("true")) {
        return;
    }
    let topo = peregrine_io::topo::snapshot();
    if !topo.multi_numa() {
        return; // single node — hierarchical == flat, skip the map
    }
    // Mirror `peregrine_par::global()`'s sizing so the map length matches the
    // pool that will be built (a mismatched map is ignored by the dispatcher).
    let workers = std::env::var("COLI_PAR_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get().min(16)).unwrap_or(4));
    // Worker w pins to cpus[w % len] (node-grouped enumeration in
    // `numa_pin_worker`); its group is that CPU's node.
    let cpus: Vec<(u32, usize)> = topo
        .numa
        .iter()
        .enumerate()
        .flat_map(|(node_idx, n)| n.cpus.iter().map(move |&c| (c, node_idx)))
        .collect();
    if cpus.is_empty() {
        return;
    }
    let groups: Vec<usize> = (0..workers).map(|w| cpus[w % cpus.len()].1).collect();
    peregrine_par::set_worker_groups(groups);
}

/// Depth of the per-layer routing history (how many recent routed sets the momentum
/// predictor votes over). Tunable via `COLI_ROUTE_HIST_DEPTH` (default 4, floored at
/// 1); read once. Tests construct [`RouteHistory`] directly with an explicit depth,
/// so they never depend on this process-global.
fn route_hist_depth() -> usize {
    use std::sync::OnceLock;
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("COLI_ROUTE_HIST_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d >= 1)
            .unwrap_or(4)
    })
}

/// Jaccard distance above which [`PredictSource::PhaseAware`] declares a routing phase
/// shift, in basis points. Tunable via `COLI_PHASE_THRESHOLD` as a fraction in
/// `[0.0, 1.0]` (default 0.6 → 6000 bp); read once.
///
/// The knob is spelled as a fraction because that is what it meant when only
/// `PhaseTracker` read it — and until 2026-08-08 that struct was the *only*
/// reader, while having no production caller, so the documented default governed
/// nothing. The live predictor used a hardcoded `6000`. This converts at the boundary
/// so one env var drives both and the units stay honest on each side.
/// (`PhaseTracker` itself was deleted 2026-08-14 on the `COLI_PREDICT_EVAL`
/// scoreboard — see `workload.rs`'s module doc; the fraction spelling stays.)
fn phase_threshold_bp() -> u32 {
    use std::sync::OnceLock;
    static BP: OnceLock<u32> = OnceLock::new();
    *BP.get_or_init(|| {
        let frac = std::env::var("COLI_PHASE_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(0.6);
        (frac * 10_000.0).round() as u32
    })
}

/// Whether the prefetch lane emits per-layer *during* the forward (look-ahead) so a
/// layer's read overlaps later layers' compute, vs. one bulk enqueue at the end of
/// the forward. On by default; disable with `COLI_PREFETCH_LOOKAHEAD=0`.
fn prefetch_lookahead() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_PREFETCH_LOOKAHEAD").as_deref(), Ok("0") | Ok("false")))
}

/// Whether to run the **router look-ahead**: at the end of layer `L`, apply layer
/// `L+1`'s post-attention norm and router to layer `L`'s output and prefetch the
/// experts that ranking names. On by default; disable with `COLI_ROUTER_LOOKAHEAD=0`.
///
/// This is a different predictor from everything in [`crate::predict`], and the
/// difference is the point. `PredictSource` is a *statistic over the router's past
/// answers* — momentum, a transition automaton, macro-states, co-activation. The
/// look-ahead asks the router itself. Measured on WASTE's K3 container (their
/// `LEARNED.md` §29/§34, 1092 layer transitions), recall@16 of the next layer's
/// actual set: 29.0 % for held-out co-occurrence, 29.5 % for "the previous token's
/// set" — and **59.0 %** for the next layer's router run on this layer's hidden
/// state. Those are their numbers on their model; `COLI_PREDICT_EVAL` is how to get
/// ours (see [`LookaheadEval`]).
///
/// It cannot change a token. The authoritative router still runs at layer `L+1` and
/// still decides; this only starts I/O early, and a wrong guess costs a read the
/// cache will evict unused. That is what makes it safe to leave on.
fn router_lookahead() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_ROUTER_LOOKAHEAD").as_deref(), Ok("0") | Ok("false")))
}

/// How many of the next layer's ranked experts the look-ahead actually streams
/// (`COLI_ROUTER_LOOKAHEAD_N`, default 6).
///
/// **This is a window, not a tuning constant.** The right value is however many
/// reads fit in the layer boundary — the stretch of attention and non-expert MoE
/// work during which the readers would otherwise have nothing queued — and that is
/// a property of the disk and the model, not a number to carry between machines.
/// Six is where WASTE landed on an M5 Pro (~5.9 ms boundary, ~0.92 ms a read); they
/// measured 3, 4 and 10 all worse, and at 10 total bytes read rose 4 %. Their
/// rank-precision profile is the reason a small number wins: 92.2 % at rank 1,
/// 81.4 % cumulative at 6, 59.0 % at 16 — so widening past the boundary buys
/// steadily worse guesses that displace reads the engine actually needs.
fn router_lookahead_width() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_usize("COLI_ROUTER_LOOKAHEAD_N", 6))
}

/// Whether the router look-ahead is allowed to fire on a **batched** decode step
/// (B > 1), in addition to the historical B == 1 case. `COLI_ROUTER_LOOKAHEAD_BATCH`
/// (default `1` = on). The look-ahead issues advisory prefetch reads during the
/// inter-layer attention window; on a batched step the potential union grows with
/// `s_n`, so the window's `width` is a budget — never `width × s_n` — capped by
/// [`LookaheadCtx::rank`] dedupe. Off ⇒ the historical behaviour (decode-only,
/// B == 1). The total `width` (not `width × s_n`) bounds the prefetch stream's IO
/// depth, matched against the disk's idle window — the same constraint WASTE
/// measured at B == 1, applied to one cross-row union.
fn router_lookahead_batch() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_ROUTER_LOOKAHEAD_BATCH").as_deref(), Ok("0") | Ok("false")))
}

/// The arms the predictor scoreboard compares, in the order the forward loop stashes
/// them. See [`predict_eval_init`].
const PREDICT_EVAL_ARMS: [&str; 3] = ["router-lookahead", "predictor", "prev-token"];

/// Build the predictor scoreboard when `COLI_PREDICT_EVAL=1`
/// (`COLI_PREDICT_EVAL_N` candidates per arm, default: the model's top-k, so recall
/// is directly comparable with a routing decision's own width).
///
/// Three arms, chosen so the comparison is the one that matters:
///
/// - **`router-lookahead`** — layer `L+1`'s router applied to layer `L`'s output.
/// - **`predictor`** — whatever [`PredictSource`] is configured: momentum, the
///   transition automaton, macro-states. A statistic over the router's past answers.
/// - **`prev-token`** — the previous token's routed set at that layer, verbatim. The
///   baseline that matters, because it is the one the expert cache already exploits
///   for free: a predictor that does not beat it has bought nothing, whatever its
///   recall looks like in isolation.
fn predict_eval_init(topk: usize) -> Option<Mutex<crate::predeval::PredictEval>> {
    if !matches!(std::env::var("COLI_PREDICT_EVAL").as_deref(), Ok("1") | Ok("true")) {
        return None;
    }
    let width = env_usize("COLI_PREDICT_EVAL_N", topk).max(1);
    Some(Mutex::new(crate::predeval::PredictEval::new(width, &PREDICT_EVAL_ARMS)))
}

/// Multi-path tiering: per layer, the top `warm_paths` ranked candidates are fully
/// streamed and the next `hint_paths` get a page-cache `fadvise` hint. The default
/// (`warm_paths = MAX`, `hint_paths = 0`) reproduces "warm everything predicted".
/// Env overrides: `COLI_PREFETCH_WARM_PATHS`, `COLI_PREFETCH_HINT_PATHS`.
#[derive(Clone, Copy)]
struct PrefetchPolicy {
    warm_paths: usize,
    hint_paths: usize,
}

impl PrefetchPolicy {
    fn from_env() -> PrefetchPolicy {
        PrefetchPolicy {
            warm_paths: env_usize("COLI_PREFETCH_WARM_PATHS", usize::MAX),
            hint_paths: env_usize("COLI_PREFETCH_HINT_PATHS", 0),
        }
    }

    /// This policy with per-workload-class overrides applied. Each class can
    /// override the warm/hint breadth via `COLI_PREFETCH_WARM_PATHS_<CLASS>` /
    /// `COLI_PREFETCH_HINT_PATHS_<CLASS>` (CLASS ∈ CODE, JSON, MATH, PROSE,
    /// MIXED); unset falls back to the base value. Lets an operator give code
    /// workloads (diverse routing) wider prefetch than prose (stable routing)
    /// without touching the base knobs.
    fn for_class(&self, class: crate::workload::TokenClass) -> PrefetchPolicy {
        let suffix = match class {
            crate::workload::TokenClass::Code => "CODE",
            crate::workload::TokenClass::Json => "JSON",
            crate::workload::TokenClass::Math => "MATH",
            crate::workload::TokenClass::Prose => "PROSE",
            crate::workload::TokenClass::Mixed => "MIXED",
        };
        PrefetchPolicy {
            warm_paths: env_usize(&format!("COLI_PREFETCH_WARM_PATHS_{suffix}"), self.warm_paths),
            hint_paths: env_usize(&format!("COLI_PREFETCH_HINT_PATHS_{suffix}"), self.hint_paths),
        }
    }
}

/// Parse a `usize` env knob, falling back to `default` when unset or unparseable.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

/// Build the adaptive prefetch-distance controller when `COLI_PREFETCH_TUNE=1`.
/// Initial/max distance from `COLI_PREFETCH_DIST` (default 4) / `COLI_PREFETCH_DIST_MAX`
/// (default 16). `None` (the default) leaves the static [`PrefetchPolicy`] in charge.
fn prefetch_tuner_init() -> Option<PrefetchTuner> {
    if !matches!(std::env::var("COLI_PREFETCH_TUNE").as_deref(), Ok("1") | Ok("true")) {
        return None;
    }
    Some(PrefetchTuner::new(env_usize("COLI_PREFETCH_DIST", 4), env_usize("COLI_PREFETCH_DIST_MAX", 16)))
}

/// Whether the LLC-miss counter additionally *steers* the prefetch tuner, rather
/// than only being reported (`COLI_PERF_PREFETCH_FEEDBACK=1`).
///
/// **Deliberately a second gate, not folded into `COLI_PERF_COUNTERS`.** The
/// counter is a measurement; this is a control loop driven by it, and the two
/// deserve separate consent. `todo.md` §10 argues the case against ever wiring
/// this — "what a miss rate *should* change is unmeasured, and wiring a governor
/// to an unvalidated signal is how a knob becomes load-bearing by accident" —
/// while the shortlist carries "hardware-counter-driven scheduler feedback" as an
/// open item and `telemetry.rs` documents the consumer as if it existed. Both are
/// now true statements: the consumer exists, and it is off unless asked for twice.
///
/// **The direction is a hypothesis and is not measured.** `telemetry.rs` specifies
/// "rising misses → widen prefetch distance", which is what this implements, but
/// the counter follows the decode thread — attention, the router matmul and the
/// deterministic reduce — and *not* the io_uring workers or the `peregrine-par`
/// pool that stream and compute experts. A rising miss rate there most plausibly
/// tracks a growing KV cache or batch, which widening the prefetch breadth does
/// nothing about and may worsen. Validating it means showing that enabling this
/// moves `[prefetch] used/wasted/accuracy` favourably at constant disk reads; if
/// that cannot be shown, the honest end state is to delete the loop and keep the
/// reporting, not to leave a knob nobody can justify.
fn perf_prefetch_feedback() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_PERF_PREFETCH_FEEDBACK").as_deref(), Ok("1") | Ok("true")))
}

/// One LLC-miss observation's verdict on the prefetch distance:
/// `1` widen, `-1` narrow, `0` hold.
///
/// Pure, so the dead band and the seeding rule are testable **without a PMU**.
/// `perf_event_open` is refused on most VMs, in containers, and at
/// `perf_event_paranoid >= 3`, so a test that needed a live counter would
/// silently pass by not running — which is how an untested controller ends up
/// looking tested.
///
/// `prev_ewma <= 0.0` is the seeding observation (no trend yet) and holds. The
/// ±10% band exists because a thread-following counter on a contended desktop
/// moves several percent between identical forwards; nudging on every wobble
/// makes the distance a random walk rather than a controller.
fn llc_trend(prev_ewma: f32, delta: f32) -> i8 {
    if prev_ewma <= 0.0 {
        return 0;
    }
    if delta > prev_ewma * 1.1 {
        1
    } else if delta < prev_ewma * 0.9 {
        -1
    } else {
        0
    }
}

/// Whether predictive eviction is active: after each forward, resident experts the
/// predictor expects to be reused are protected from eviction. It only reorders
/// eviction victims, never output.
///
/// **The default is decided by the cache budget, because this mechanism has
/// opposite signs either side of one token's working set.** Measured at 8 decode
/// tokens on the GLM-5.2 container:
///
/// | budget | slots vs one pass | protect on | pure LRU |
/// |---|---|---|---|
/// | 4.29 GB | 227 (38 %) | **193** | 0 |
/// | 12.88 GB | 681 (113 %) | 564 | **945** |
///
/// Below the threshold a front-to-back layer sweep drives plain recency to
/// *exactly zero* hits — the minimum-`used` resident slot at the end of a pass is
/// by construction the earliest-layer expert, which is what the next token asks
/// for first — and feeding `pack_prio` into the victim key is the only thing
/// keeping the count off the floor. Above it the cache can hold a pass unaided
/// and the same priority ordering costs 40 % of the hits and 381 extra reads.
///
/// So: on when the budget cannot clear a pass, off when it can. `COLI_PREFETCH_PROTECT`
/// still forces either way; this only picks the default, which was unconditionally
/// on until 2026-08-09 with nothing tracking which side a deployment sat on.
fn prefetch_protect_default(budget_bytes: usize, per_token_bytes: u64) -> bool {
    match std::env::var("COLI_PREFETCH_PROTECT").as_deref() {
        Ok("0") | Ok("false") => false,
        Ok("1") | Ok("true") => true,
        // An unknown value, or no working-set figure (resident mode, or an
        // unindexed container), keeps the historical always-on behaviour.
        _ => per_token_bytes == 0 || (budget_bytes as u64) < per_token_bytes,
    }
}

/// Whether speculative reads are still earning their bandwidth.
///
/// Measured at 8 decode tokens on the GLM-5.2 container: **4034 speculative reads
/// bought 3 hits** — 0.3 % — for **+41 % wall time**. At that yield the prefetch
/// lane is pure contention for a disk the demand path is already saturating, and
/// it was the demand path that produced 183 of the 196 hits. Prefetch is worth
/// having when the predictor is right often enough to fill an idle window; it is
/// not worth having unconditionally.
///
/// Once `COLI_PREFETCH_MIN_READS` speculative reads have been issued (default
/// 512 — a decode token's worth), keep issuing only if at least
/// `COLI_PREFETCH_MIN_YIELD` percent of them were used (default 2). Setting the
/// yield to 0 disables the guard and restores the unconditional behaviour.
///
/// **One-way**: once the guard trips, `issued` stops growing, so the ratio is
/// frozen and the lane stays off for the process. That is deliberate — a lane
/// that re-enables itself would re-pay the wall-time cost to re-learn the same
/// answer — but it does mean the sample has to be big enough to trust, which is
/// what `min_reads` is for.
///
/// Correctness-neutral, like everything else in this subsystem: not issuing a
/// speculative read only means the demand path streams that expert itself.
fn prefetch_pays(issued: u64, used: u64) -> bool {
    use std::sync::OnceLock;
    static MIN_READS: OnceLock<u64> = OnceLock::new();
    static MIN_YIELD: OnceLock<u64> = OnceLock::new();
    let min_reads = *MIN_READS
        .get_or_init(|| std::env::var("COLI_PREFETCH_MIN_READS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(512));
    let min_yield = *MIN_YIELD
        .get_or_init(|| std::env::var("COLI_PREFETCH_MIN_YIELD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2));
    if min_yield == 0 || issued < min_reads {
        return true;
    }
    // Integer form of `100 * used / issued >= min_yield`, so no float rounding
    // decides whether the lane lives.
    used.saturating_mul(100) >= issued.saturating_mul(min_yield)
}

/// Pack an eviction-protection score: predictor likelihood in the high bits, routing
/// heat as a low-bits tiebreak, `+1` so any predicted expert outranks an unprotected
/// slot (priority 0). Saturating — never wraps back to 0.
fn pack_prio(score: u32, heat: u32) -> u32 {
    ((score.min(0xFFFF) << 16) | heat.min(0xFFFF)).saturating_add(1)
}

/// A coarse fingerprint of the model's shape, stamped into a built automaton so an
/// artifact from a different checkpoint is ignored on load.
fn config_tag(cfg: &Cfg) -> String {
    format!(
        "L{}E{}H{}I{}D{}V{}",
        cfg.n_layers, cfg.n_experts, cfg.hidden, cfg.moe_inter, cfg.first_dense, cfg.vocab
    )
}

/// Write a built automaton to `path` as JSON — the `automaton.json` artifact a model
/// auto-loads from its checkpoint directory.
pub fn save_automaton(table: &TransitionTable, path: &std::path::Path) -> Result<(), Error> {
    let json = serde_json::to_vec(&table.to_json()).map_err(|e| Error::Format(format!("serialize automaton: {e}")))?;
    peregrine_core::write_atomic(path, &json)?;
    Ok(())
}

/// Write a built macro-state table to `path` — the `macrostates.json` artifact a
/// model auto-loads (and blends into the predictor) from its checkpoint dir.
pub fn save_macrostates(table: &crate::predict::MacroTable, path: &std::path::Path) -> Result<(), Error> {
    let json = serde_json::to_vec(&table.to_json()).map_err(|e| Error::Format(format!("serialize macrostates: {e}")))?;
    peregrine_core::write_atomic(path, &json)?;
    Ok(())
}

/// Whether cross-session routing-history persistence is on. Default on when the
/// model dir is writable; disable with `COLI_ROUTE_STATS_PERSIST=0`. Correctness-
/// neutral (heat and route history only affect prefetch/eviction, not output).
fn route_stats_persist_enabled() -> bool {
    !matches!(std::env::var("COLI_ROUTE_STATS_PERSIST").as_deref(), Ok("0") | Ok("false"))
}

/// Read just the heat snapshot out of `<dir>/route_stats.json`, before a
/// [`Model`] exists.
///
/// Initial VRAM residency is a knapsack over routing heat, but the heat table is
/// only restored (in `try_load_route_stats`) well after the tier is built — and
/// it is itself only created when a tier exists. That circularity is why initial
/// placement had no heat to rank by. Peeking the file first breaks it: last
/// session's routing decides what lands in VRAM this session, and `reheat` then
/// refines it live.
///
/// Returns an empty vec whenever anything is off — persistence disabled, no
/// file, unparseable, or a config fingerprint mismatch — which
/// [`gpu::solve_residency_sized`] treats as a cold table and answers with the
/// deterministic round-robin placement. Correctness-neutral either way.
/// Read and parse an optional sidecar artifact (`plan.json`, `tiers.json`,
/// `automaton.json`, `macrostates.json`, `route_stats.json`).
///
/// **Absent** returns `None` in silence — every one of these is optional, and a
/// model directory without them is the normal case, not a problem.
///
/// **Present but unreadable or malformed** also returns `None`, but says so
/// through `note_advisory_err`. That distinction is the point: these loaders
/// used to treat a corrupt artifact exactly like a missing one, so a
/// syntax error in `plan.json` left the engine running default behavior with
/// nothing anywhere to explain why the file the operator had just written had no
/// effect. Correctness is unaffected either way — that is what makes it
/// advisory rather than fatal — but `COLI_DEBUG=1` now names the file and the
/// reason.
fn read_optional_artifact(dir: &std::path::Path, file: &str) -> Option<serde_json::Value> {
    let path = dir.join(file);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            peregrine_io::note_advisory_err(&format!("read optional artifact {file}"), &e);
            return None;
        }
    };
    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            peregrine_io::note_advisory_err(&format!("parse optional artifact {file} (ignored)"), &e);
            None
        }
    }
}

fn peek_persisted_heat(dir: &std::path::Path, cfg: &Cfg) -> Vec<u32> {
    if !route_stats_persist_enabled() {
        return Vec::new();
    }
    let Some(v) = read_optional_artifact(dir, "route_stats.json") else { return Vec::new(); };
    // Same fingerprint gate as `try_load_route_stats`: heat is a flat
    // `[layer * n_experts + expert]` array, so restoring it under a different
    // expert count maps every count onto the wrong pair.
    if v.get("tag").and_then(|t| t.as_str()) != Some(config_tag(cfg).as_str()) {
        return Vec::new();
    }
    let Some(arr) = v.get("heat").and_then(|h| h.as_array()) else {
        return Vec::new();
    };
    let snap: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
    // A short/long array would silently misalign layers; take it only at the
    // exact expected shape.
    if snap.len() != (cfg.n_layers as usize) * (cfg.n_experts as usize) {
        return Vec::new();
    }
    snap
}

/// Entropy-adaptive prefetch breadth: when on (`COLI_ENTROPY_ADAPT=1`),
/// dispersed routing (high entropy) widens the prefetch-distance tuner and
/// repetitive routing narrows it. Requires the tuner (`COLI_PREFETCH_TUNE=1`)
/// to have an effect. Correctness-neutral.
fn entropy_adapt_enabled() -> bool {
    matches!(std::env::var("COLI_ENTROPY_ADAPT").as_deref(), Ok("1") | Ok("true"))
}

/// Shared state for the sensor-driven governors (thermal / power / memory
/// bandwidth). All three adjust the same knob — the effective CPU-lane worker
/// count — with "any shrink wins over a grow" arbitration, sampled every
/// [`GovernorState::SENSOR_PERIOD`] forwards to bound sysfs-read overhead.
struct GovernorState {
    tick: u64,
    base_workers: usize,
    energy: peregrine_io::EnergyMeter,
    last_sample: Option<std::time::Instant>,
    ewma_gbps: f32,
    /// Tick of the last bandwidth +1 probe (periodic regrow attempt).
    last_probe: u64,
}

impl GovernorState {
    const SENSOR_PERIOD: u64 = 16;

    fn new(base_workers: usize) -> GovernorState {
        GovernorState {
            tick: 0,
            base_workers,
            energy: peregrine_io::EnergyMeter::new(),
            last_sample: None,
            ewma_gbps: 0.0,
            last_probe: 0,
        }
    }

    /// One governor step. Returns the worker-count delta to apply (−1, 0, +1).
    /// Heuristics, all opt-in:
    /// - `COLI_THERMAL_LIMIT_C=<c>`: above the limit → shrink; 8 °C below → regrow.
    /// - `COLI_POWER_CAP_W=<w>`: RAPL watts above cap → shrink; below 80 % → regrow.
    /// - `COLI_BW_GOVERNOR=1`: CPU-lane GB/s EWMA plateau → shrink (bandwidth-bound,
    ///   extra workers just contend); a periodic +1 probe (every 256 forwards)
    ///   rediscovers headroom after the workload shifts.
    fn step(&mut self, sample: crate::lane::LaneTimings, current: usize) -> i32 {
        self.tick += 1;
        let thermal: Option<i32> = std::env::var("COLI_THERMAL_LIMIT_C").ok().and_then(|v| v.trim().parse().ok());
        let power: Option<f32> = std::env::var("COLI_POWER_CAP_W").ok().and_then(|v| v.trim().parse().ok());
        let bw = matches!(std::env::var("COLI_BW_GOVERNOR").as_deref(), Ok("1") | Ok("true"));
        if thermal.is_none() && power.is_none() && !bw {
            return 0;
        }
        let mut shrink = false;
        let mut grow = false;
        // Bandwidth plateau check runs every forward (numbers already in hand).
        if bw && sample.cpu_us > 0 && sample.cpu_bytes > 0 {
            let gbps = sample.cpu_bytes as f32 / sample.cpu_us as f32 / 1000.0; // bytes/µs → GB/s
            if self.ewma_gbps > 0.0 && gbps < 0.98 * self.ewma_gbps && current > 2 {
                shrink = true; // more workers stopped buying bandwidth
            }
            self.ewma_gbps = if self.ewma_gbps == 0.0 { gbps } else { 0.7 * self.ewma_gbps + 0.3 * gbps };
            // Reset the probe clock whenever the interval elapses, not only when
            // a probe actually fires: otherwise, once 256 ticks had passed at
            // full worker count, the very next shrink was undone by an immediate
            // "periodic" probe on the following forward.
            if self.tick.saturating_sub(self.last_probe) >= 256 {
                if current < self.base_workers {
                    grow = true; // periodic probe upward
                }
                self.last_probe = self.tick;
            }
        }
        // Sensor reads only every SENSOR_PERIOD forwards.
        if self.tick.is_multiple_of(Self::SENSOR_PERIOD) {
            if let Some(limit) = thermal {
                if let Some(t) = peregrine_io::max_temp_c() {
                    if t > limit {
                        shrink = true;
                    } else if t < limit - 8 && current < self.base_workers {
                        grow = true;
                    }
                }
            }
            if let Some(cap) = power {
                let now = std::time::Instant::now();
                let dt = self.last_sample.map(|p| now.duration_since(p).as_secs_f32()).unwrap_or(0.0);
                self.last_sample = Some(now);
                if let Some(uj) = self.energy.delta_uj() {
                    if dt > 0.0 {
                        let watts = (uj as f32 / 1e6) / dt;
                        if watts > cap {
                            shrink = true;
                        } else if watts < 0.8 * cap && current < self.base_workers {
                            grow = true;
                        }
                    }
                }
            }
        }
        // Any shrink wins over any grow — thermal/power safety beats throughput.
        if shrink {
            -1
        } else if grow {
            1
        } else {
            0
        }
    }
}

/// Load `<dir>/schedule.json` if present — the per-layer expert-order hint
/// emitted by `peregrine-layout-reorg`. Silently returns `None` on a missing,
/// malformed, or wrong-version file; the model then falls back to natural
/// expert-id order. Correctness-neutral (order changes only affect batched-read
/// coalescing, never the reduced values).
fn load_layout_schedule(dir: &std::path::Path) -> Option<Vec<Vec<u32>>> {
    if matches!(std::env::var("COLI_LAYOUT_SCHEDULE").as_deref(), Ok("0") | Ok("false")) {
        return None;
    }
    let bytes = std::fs::read(dir.join("schedule.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    parse_schedule_value(&v)
}

/// Parse a schedule document (the `schedule.json` shape) — shared by the
/// standalone file loader and the compiled-plan consumer.
fn parse_schedule_value(v: &serde_json::Value) -> Option<Vec<Vec<u32>>> {
    // Only version 1 for now.
    if v.get("version").and_then(|x| x.as_u64())? != 1 {
        return None;
    }
    let order_arr = v.get("order")?.as_array()?;
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(order_arr.len());
    for layer in order_arr {
        let ids = layer.as_array()?;
        let mut row: Vec<u32> = Vec::with_capacity(ids.len());
        for id in ids {
            let n = id.as_i64()?;
            if n < 0 {
                continue;
            }
            row.push(n as u32);
        }
        out.push(row);
    }
    Some(out)
}

/// The borrowed state the prefetch emitter needs, bundled so [`PrefetchCtx::emit_layer`]
/// takes a single receiver (and so `forward_hidden` can build one from its destructured
/// field borrows to emit mid-forward). Holds shared borrows only.
struct PrefetchCtx<'a> {
    prefetch: &'a PrefetchHandle,
    predictor: &'a PredictSource,
    hist: &'a Mutex<RouteHistory>,
    cache: &'a Mutex<WarmCache>,
    gpu: Option<&'a GpuTier>,
    st: &'a SafeTensors,
    cfg: &'a Cfg,
    /// The model's load-time expert map, so a speculative read resolves its
    /// regions exactly the way the demand path will.
    expert_index: Option<&'a crate::concurrent::ExpertIndex>,
    /// Multi-path tiering: the top `warm_paths` ranked candidates per layer are fully
    /// streamed (tier 1); the next `hint_paths` get a page-cache `fadvise` hint (tier 2).
    warm_paths: usize,
    hint_paths: usize,
    /// Under O_DIRECT the page cache is bypassed, so tier-2 hints are pointless and
    /// suppressed.
    direct: bool,
    /// Stamps each `Warm` batch with the layer-step it was emitted in, so the lane
    /// can drop it unread once its window has passed ([`SweepClock`]).
    sweep: &'a SweepClock,
}

impl PrefetchCtx<'_> {
    /// Emit prefetch work for one layer's predicted next-token experts. The predictor
    /// ranks candidates; among the not-already-warm, non-GPU ones, the top
    /// `warm_paths` are fully streamed (tier 1) and the next `hint_paths` get a cheap
    /// page-cache `fadvise` hint (tier 2, suppressed under O_DIRECT) — this is the
    /// multi-path bandwidth allocation. Locks history then cache in turn (never
    /// nested), so it can't invert lock order against the I/O lane. A dense layer, or
    /// one with no fresh candidates, is a no-op.
    fn emit_layer(&self, layer: usize) {
        if layer < self.cfg.first_dense as usize {
            return; // dense layer — no routed experts
        }
        // Stop speculating once the measured yield says it is not paying for
        // itself. Checked before the predictor runs, so a dead lane costs a lock
        // and two loads rather than a ranking pass.
        {
            let c = self.cache.lock();
            if !prefetch_pays(c.prefetch_reads, c.prefetch_used) {
                return;
            }
        }
        let candidates = {
            let hist = self.hist.lock();
            self.predictor.predict_layer(layer, &hist)
        };
        if candidates.is_empty() {
            return;
        }
        let hint_cutoff = self.warm_paths.saturating_add(self.hint_paths);
        let mut warms = Vec::new();
        let mut hints = Vec::new();
        {
            let cache = self.cache.lock();
            let mut rank = 0usize; // rank among fresh (not-warm, non-GPU) candidates
            for (e, _score) in candidates {
                let key = (layer as u32, e);
                if cache.contains(key) {
                    continue; // already warm
                }
                if self.gpu.is_some_and(|g| g.has(layer, e as usize)) {
                    continue; // computed on the GPU lane, never streamed
                }
                if rank < self.warm_paths {
                    match crate::concurrent::prefetch_item(self.expert_index, self.st, self.cfg, layer, e as usize) {
                        Ok(item) => warms.push(item),
                        // speculative: the real forward will stream this expert normally
                        Err(e) => peregrine_io::note_advisory_err("prefetch item resolve", &e),
                    }
                } else if rank < hint_cutoff && !self.direct {
                    match crate::concurrent::prefetch_hint_item(self.expert_index, self.st, self.cfg, layer, e as usize) {
                        Ok(item) => hints.push(item),
                        Err(e) => peregrine_io::note_advisory_err("prefetch hint resolve", &e),
                    }
                } else {
                    break; // beyond both tiers — lower-ranked candidates ignored
                }
                rank += 1;
            }
        }
        if !warms.is_empty() && self.prefetch.tx.send(PrefetchMsg::Warm(warms, self.sweep.now())).is_err() {
            peregrine_io::note_advisory_err("prefetch warm dispatch", &"prefetch lane is down");
        }
        if !hints.is_empty() && self.prefetch.tx.send(PrefetchMsg::Hint(hints)).is_err() {
            peregrine_io::note_advisory_err("prefetch hint dispatch", &"prefetch lane is down");
        }
    }

}

/// The state the router look-ahead needs — which is [`PrefetchCtx`] minus the
/// routing history and minus the predictor.
///
/// That absence is the substantive difference between the two emitters, not an
/// implementation detail. Everything in [`crate::predict`] is a function of what the
/// router has answered before, so it needs a history and needs it to be warm. The
/// look-ahead needs neither: it reads the answer off the next layer's own weights,
/// which are already resident. It is therefore right on the first token of a cold
/// process, where every history-based predictor is still empty.
#[derive(Clone, Copy)]
struct LookaheadCtx<'a> {
    prefetch: &'a PrefetchHandle,
    cache: &'a Mutex<WarmCache>,
    gpu: Option<&'a GpuTier>,
    st: &'a SafeTensors,
    cfg: &'a Cfg,
    /// Same load-time expert map [`PrefetchCtx`] carries; `None` falls back to
    /// re-deriving plans per request, which is what this path did before.
    expert_index: Option<&'a crate::concurrent::ExpertIndex>,
    /// Stamps each `Warm` batch with the layer-step it was emitted in, so the lane
    /// can drop it unread once its window has passed ([`SweepClock`]).
    sweep: &'a SweepClock,
}

impl LookaheadCtx<'_> {
    /// Ask layer `next`'s own router which experts it is about to want, and start
    /// those reads now — during the boundary the disk would otherwise spend idle.
    ///
    /// `x` is layer `next - 1`'s output, which is exactly layer `next`'s input. The
    /// authoritative router at layer `next` will see `rmsnorm(x + attn_next,
    /// post_ln_next)`; this sees `rmsnorm(x, post_ln_next)`. The one term missing is
    /// that layer's own attention delta, and the residual stream dominates it — which
    /// is why a ranking taken from the router beats every statistic over the router's
    /// history by roughly 2× (see [`router_lookahead`]).
    ///
    /// Ordering matches [`PrefetchCtx::emit_layer`]: rank by the router, then drop the
    /// candidates that need no read (already warm, or GPU-resident and never
    /// streamed), and spend the window on the top `width` of what is left. Filtering
    /// after ranking rather than before is what keeps the window full — a prediction
    /// that is right and already cached is not a read, so it should not consume one
    /// of the six slots.
    ///
    /// Advisory throughout: a resolve failure or a dead lane costs nothing but the
    /// speculation, because the real forward streams the expert through the ordinary
    /// path regardless.
    fn emit(&self, layers: &[LayerW], next: usize, x: &[f32], width: usize) {
        let d = self.cfg.hidden as usize;
        if width == 0 || next >= layers.len() || next < self.cfg.first_dense as usize || x.len() < d {
            return;
        }
        let l = &layers[next];
        if !l.sparse {
            return;
        }
        let mut warms = Vec::new();
        for e in self.rank(l, next, x, width) {
            let Ok(eu) = usize::try_from(e) else { continue };
            match crate::concurrent::prefetch_item(self.expert_index, self.st, self.cfg, next, eu) {
                Ok(item) => warms.push(item),
                Err(err) => peregrine_io::note_advisory_err("lookahead item resolve", &err),
            }
        }
        if warms.is_empty() {
            return;
        }
        LOOKAHEAD_ISSUED.fetch_add(warms.len() as u64, std::sync::atomic::Ordering::Relaxed);
        if self.prefetch.tx.send(PrefetchMsg::Warm(warms, self.sweep.now())).is_err() {
            peregrine_io::note_advisory_err("lookahead dispatch", &"prefetch lane is down");
        }
    }

    /// The experts this look-ahead will actually stream for layer `next`: the router's
    /// ranking, minus the candidates that would cost no read, truncated to `width`.
    ///
    /// Dropping the already-warm and the GPU-resident *after* ranking rather than
    /// before is what keeps the window full: a prediction that is right and already
    /// cached is not a read, so it should not consume one of the `width` slots.
    fn rank(&self, l: &LayerW, next: usize, x: &[f32], width: usize) -> Vec<i32> {
        // Scan the router's top-`k` — the width of a real routing decision — and take
        // the first `width` that would cost a read. Past rank `k` the ranking's
        // precision has fallen far enough that the candidates are not worth the scan.
        let scan = width.max(self.cfg.topk.max(0) as usize);
        let d = self.cfg.hidden as usize;
        // s_n inferred from `x.len()`: a single-row path emits the historical narrow
        // ranking; a multi-row path takes the **union** of per-row rankings. The union
        // is the right call for a batched-decode look-ahead because every row whose
        // routing wants an expert that is not yet warm will pay a disk miss — so a
        // union-bounded `width` is exactly the set the prefetch lane has to keep warm
        // to clear the next layer's misses in the next step's gap.
        let s_n = (x.len() / d).max(1);
        let ranks = if s_n == 1 {
            router_ranks_for(l, self.cfg, x, scan)
        } else {
            router_ranks_for_batch(l, self.cfg, x, s_n, scan)
        };
        let mut out = Vec::with_capacity(width.min(ranks.len()));
        let cache = self.cache.lock();
        for e in ranks {
            if out.len() >= width {
                break;
            }
            let Ok(eu) = usize::try_from(e) else { continue };
            if cache.contains((next as u32, e as u32)) {
                continue; // already warm — the look-ahead's job is done for it
            }
            if self.gpu.is_some_and(|g| g.has(next, eu)) {
                continue; // computed on the GPU lane, never streamed
            }
            out.push(e);
        }
        out
    }
}

/// Layer `l`'s router's own top-`n` ranking of the hidden state `x` — the router
/// look-ahead's *prediction*, before any fetch policy is applied to it.
///
/// A free function rather than a method on [`LookaheadCtx`] because the two callers
/// need different things from it and only one of them has a prefetch lane: the
/// look-ahead ranks in order to fetch, and [`score_and_stash`] ranks in order to
/// measure. Keeping measurement independent of the fetch machinery means a predictor
/// can be evaluated on a model with no warm cache at all.
fn router_ranks_for(l: &LayerW, cfg: &Cfg, x: &[f32], n: usize) -> Vec<i32> {
    let d = cfg.hidden as usize;
    if x.len() < d {
        return Vec::new();
    }
    // One row: this is a decode-only path, so there is exactly one position to rank
    // and no union to take.
    let nrm = rmsnorm_rows(x, &l.post_ln, 1, d, cfg.eps);
    crate::router::route_ranks(&nrm, &l.router, &l.router_bias, d, cfg.n_experts as usize, n)
}

/// Like [`router_ranks_for`] but ranks *every row* of an `[s_n, d]` hidden batch and
/// returns the deduplicated union in row-major order (row 0's picks, then row 1's
/// new picks, …). Past-the-batch prefetch is the only multi-row caller: the
/// authoritative router still runs at layer `L+1` and decides, so the cross-row
/// union's job is to capture whichever row's routing is about to cost a disk read.
///
/// `per_row` bounds each row's rank list (router top-`k` is the natural value); the
/// caller truncates the final union to its own `width` budget. Bounded dedupe with a
/// linear scan keeps order deterministic — never a hash — so the prefetch stream is
/// reproducible and the [R] reachability audit and `COLI_PREDICT_EVAL` see a fixed
/// issuance order.
fn router_ranks_for_batch(l: &LayerW, cfg: &Cfg, x: &[f32], s_n: usize, per_row: usize) -> Vec<i32> {
    let d = cfg.hidden as usize;
    if s_n == 0 || x.len() < s_n * d || per_row == 0 {
        return Vec::new();
    }
    let nrm = rmsnorm_rows(x, &l.post_ln, s_n, d, cfg.eps);
    let mut out: Vec<i32> = Vec::with_capacity(s_n * per_row.min(cfg.n_experts as usize));
    for s in 0..s_n {
        let row = &nrm[s * d..s * d + d];
        // Per-row route_ranks redirects into the same kernel path the single-row
        // look-ahead uses, so the rankings are bit-for-bit the router's own order for
        // that row — no side table, no approximation. A row that degenerates to
        // all-NaN contributes nothing, exactly as the single-row path does.
        let ranks = crate::router::route_ranks(row, &l.router, &l.router_bias, d, cfg.n_experts as usize, per_row);
        for e in ranks {
            if e < 0 {
                continue;
            }
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
    out
}

/// Settle layer `li`'s outstanding prediction against what it actually routed, then
/// predict layer `li + 1` with every arm. Called once per layer per decode step when
/// `COLI_PREDICT_EVAL=1`; costs nothing otherwise, because the caller holds `None`.
///
/// The order matters and is the reason this is one function rather than two. At this
/// instant `forward_layer` has just published layer `li`'s authoritative routed set,
/// so `latest(li)` is the answer to the prediction stashed a layer ago — and
/// `latest(li + 1)` has *not* been overwritten yet, so it is still the previous
/// token's set there, which is precisely the `prev-token` baseline. Predicting before
/// scoring, or scoring after the next layer runs, would quietly change what both
/// numbers mean.
fn score_and_stash(
    eval: &Mutex<crate::predeval::PredictEval>,
    rh: &Mutex<RouteHistory>,
    predictor: &PredictSource,
    layers: &[LayerW],
    cfg: &Cfg,
    li: usize,
    x: &[f32],
) {
    let width = eval.lock().width();
    let next = li + 1;
    let predict_next = next < layers.len() && layers[next].sparse && next >= cfg.first_dense as usize;
    // One history lock for everything that reads it, so the baseline and the
    // statistical arm see the same frame.
    let (actual, prev, statistical) = {
        let hist = rh.lock();
        // An absent frame means this layer has not routed yet — the normal
        // state on the first decode step, not an error, and scoring against an
        // empty set is exactly right there.
        let actual = match hist.latest(li) {
            Some(set) => set.clone(),
            None => Vec::new(),
        };
        let (prev, statistical) = if predict_next {
            let prev = match hist.latest(next) {
                Some(set) => set.clone(),
                None => Vec::new(),
            };
            let stat = predictor
                .predict_layer(next, &hist)
                .into_iter()
                .take(width)
                .map(|(e, _score)| e as i32)
                .collect();
            (prev, stat)
        } else {
            (Vec::new(), Vec::new())
        };
        (actual, prev, statistical)
    };
    let lookahead = if predict_next { router_ranks_for(&layers[next], cfg, x, width) } else { Vec::new() };
    let mut ev = eval.lock();
    ev.score(li, &actual);
    if predict_next {
        // Arm order must match `PREDICT_EVAL_ARMS`.
        ev.stash(next, vec![lookahead, statistical, prev]);
    }
}

/// Experts the router look-ahead has speculatively streamed. Monotonic, lock-free,
/// diagnostic — read at shutdown beside the warm cache's `prefetch_used` /
/// `prefetch_wasted`, which already tell whether a speculative read was later hit.
///
/// Deliberately **not** folded into the cache's `misses`: a speculative read is not a
/// demand access, and counting it as one would make a look-ahead that guessed wrong
/// look like a cache that performed badly — and would quietly change what every
/// hit-rate number in the engine's telemetry means.
static LOOKAHEAD_ISSUED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Experts the router look-ahead has speculatively streamed so far this process.
pub fn lookahead_issued() -> u64 {
    LOOKAHEAD_ISSUED.load(std::sync::atomic::Ordering::Relaxed)
}

impl Model {
    /// Per-predictor recall and precision-by-rank (`COLI_PREDICT_EVAL=1`), with the
    /// number of layer transitions they were scored over. `None` unless the
    /// scoreboard is on and something has actually been scored — an evaluation with
    /// no evidence reports nothing rather than a row of zeroes that reads like a
    /// result.
    pub fn predict_eval_report(&self) -> Option<(Vec<crate::predeval::ArmReport>, u64)> {
        let ev = self.predict_eval.as_ref()?.lock();
        let report = ev.report();
        (!report.is_empty()).then(|| (report, ev.scored_layers()))
    }
}

/// The prefetch lane: stream predicted experts into the shared warm cache on this
/// lane's *own* ring. Best-effort — a failed speculative read is dropped (the real
/// forward will stream it normally).
///
/// **This said "no contention with the critical I/O lane" until 2026-08-09, and that
/// was only true of io_uring submission.** A separate ring means a speculative read
/// never queues behind a demand read *in the submission queue*; both rings then feed
/// one block device, one dm-crypt worker pool and one NVMe queue. On a device whose
/// reads are CPU-bound on LUKS decryption — this repo's own standing hypothesis — a
/// speculative read consumes exactly the decrypt cycles a demand read needs. There is
/// no prioritisation, throttle, backpressure or in-flight cap between the two: the
/// channel is unbounded, nothing drains or cancels a queued speculation, and this
/// worker submits **one expert (6 regions) per `submit_and_wait`**, so a backlog of N
/// items is N sequential round-trips that a demand read cannot jump.
///
/// Two smaller couplings, both real: this holds the `WarmCache` mutex across a full
/// insert-with-eviction (and, under `COLI_CACHE_COMPRESS`, a ~19 MB zstd encode), and
/// the demand lane probes that same lock on its critical path.
fn prefetch_worker(
    mut reactor: Reactor,
    cache: Arc<Mutex<WarmCache>>,
    rx: crossbeam_channel::Receiver<PrefetchMsg>,
    direct: bool,
    verify: bool,
    sweep: Arc<SweepClock>,
) {
    loop {
        // `recv`'s only error is `Disconnected` — the model dropped its sender,
        // which is this thread's shutdown signal.
        let msg = match rx.recv() {
            Ok(msg) => msg,
            Err(crossbeam_channel::RecvError) => break,
        };
        match msg {
            PrefetchMsg::Warm(items, stamp) => {
                for item in items {
                    // Checked per item, not per batch: a batch can take long enough
                    // to service that it goes stale midway, and the whole point is
                    // to stop *before* the read. Two relaxed loads — cheaper than
                    // the cache probe below.
                    let slack = sweep.slack.load(std::sync::atomic::Ordering::Relaxed);
                    if warm_item_is_stale(sweep.now(), stamp, slack) {
                        cache.lock().note_prefetch_stale_dropped(1);
                        continue;
                    }
                    let key = item.key();
                    if cache.lock().contains(key) {
                        continue; // already warm — don't re-read
                    }
                    let slab = match crate::concurrent::prefetch_read(&mut reactor, &item, direct) {
                        Ok(slab) => slab,
                        // failed speculative read: dropped by design — the real
                        // forward will stream the expert through the normal path
                        Err(e) => {
                            peregrine_io::note_advisory_err("speculative prefetch read", &e);
                            continue;
                        }
                    };
                    if verify {
                        // re-read and byte-compare — a mismatch means a real I/O bug,
                        // recorded as a counter (never a panic — lint-forbidden).
                        match crate::concurrent::prefetch_read(&mut reactor, &item, direct) {
                            Ok(check) => {
                                if check != slab {
                                    cache.lock().note_verify_mismatch();
                                }
                            }
                            Err(e) => peregrine_io::note_advisory_err("prefetch verify re-read", &e),
                        }
                    }
                    let mut c = cache.lock();
                    c.note_prefetch_read(key.0);
                    c.insert_prefetched(key, slab);
                }
            }
            PrefetchMsg::Hint(items) => {
                for item in items {
                    for &(fd, off, len) in item.regions() {
                        // advisory only — moves no bytes, can't affect output
                        if let Err(e) = reactor.fadvise_willneed(fd, off, len) {
                            peregrine_io::note_advisory_err("fadvise willneed hint", &e);
                        }
                    }
                    cache.lock().note_fadvise();
                }
            }
            PrefetchMsg::Sync(reply) => {
                if reply.send(()).is_err() {
                    peregrine_io::note_advisory_err("prefetch sync ack", &"barrier requester gone");
                }
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
        indexer: IndexerWeights::load(st, i, cfg)?,
    })
}

/// Row-wise RMSNorm of `x[s_n, d]` with weight `w`, into a fresh buffer. Rows are
/// independent, so they run on the persistent compute pool above `PAR_ROWS_MIN`
/// (serial below it); the per-row math is untouched, so the result is bit-identical
/// to the serial loop (guarded by `rmsnorm_rows_parallel_matches_serial`). This is
/// the highest-frequency hotspot — 2× per layer + the final norm.
fn rmsnorm_rows(x: &[f32], w: &[f32], s_n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; s_n * d];
    // Only parallelize when the row is wide enough that per-row work covers the pool
    // dispatch; narrow rows (the tiny test model, d=16) stay serial. Real GLM-5.2
    // (d=6144) clears this comfortably.
    let gate = if d >= 256 { peregrine_par::PAR_ROWS_MIN } else { usize::MAX };
    peregrine_par::par_rows_mut(&mut out, d, s_n, gate, |s, row| {
        // `out` is freshly allocated here and can never alias `x`, so the row can
        // be read straight from the source: the copy existed only to satisfy
        // `rmsnorm`'s non-aliasing signature and cost an allocation per row per
        // norm — two per layer, on every token.
        rmsnorm(row, &x[s * d..s * d + d], w, eps);
    });
    out
}

/// Forward one transformer layer in place: `x += attn(norm(x)); x += ffn(norm(x))`.
/// Shared by the main stack and the MTP head; the sparse-MoE streaming/GPU lanes
/// apply exactly as in the main loop. Compute state travels in [`ForwardCtx`].
/// Accumulator for the calibration capture pass (`COLI_CALIB_CAPTURE`, ideas
/// #7): per expert-bearing layer, `Σ|x|` per hidden channel over every
/// main-stream row that reached the MoE branch, plus that layer's row count.
/// Pooled per layer on purpose — every expert in a layer sees the same
/// pre-gating hidden distribution, which is what makes ~16 routed
/// samples/expert/layer of per-expert statistics unnecessary.
///
/// Rows are `n_layers + 1` in the `HeatTable` convention. The MTP row exists
/// but never accumulates: capture runs draft nothing, and draft forwards
/// deliberately carry `calib: None`. Its sidecar row is therefore empty and
/// the converter falls back to data-free rounding for the MTP experts —
/// combine `--calib` with `--keep-last-layers` when they should be protected
/// instead.
pub struct CalibAccum {
    sums: Vec<Vec<f64>>,
    rows: Vec<u64>,
    hidden: usize,
}

impl CalibAccum {
    fn new(n_layers: usize, hidden: usize) -> CalibAccum {
        CalibAccum { sums: vec![Vec::new(); n_layers + 1], rows: vec![0; n_layers + 1], hidden }
    }

    /// Fold `s_n` rows of a layer's MoE input into the running per-channel
    /// `Σ|x|`. The row width is trusted to be `hidden` — the caller passes the
    /// same `nrm2` it hands the router.
    fn accumulate(&mut self, li: usize, x: &[f32], s_n: usize) {
        let d = self.hidden;
        let (Some(sum), Some(rows)) = (self.sums.get_mut(li), self.rows.get_mut(li)) else {
            return;
        };
        if sum.is_empty() {
            sum.resize(d, 0.0);
        }
        for r in 0..s_n {
            let Some(row) = x.get(r * d..(r + 1) * d) else { break };
            for (acc, &v) in sum.iter_mut().zip(row) {
                *acc += f64::from(v.abs());
            }
        }
        *rows += s_n as u64;
    }

    /// The sidecar value: mean `|x|` per channel per layer, empty rows for
    /// layers that never accumulated (dense layers, the MTP row).
    fn to_sidecar_json(&self) -> serde_json::Value {
        let layers: Vec<serde_json::Value> = self
            .sums
            .iter()
            .zip(&self.rows)
            .map(|(sum, &n)| {
                if sum.is_empty() || n == 0 {
                    serde_json::json!([])
                } else {
                    let means: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
                    serde_json::json!(means)
                }
            })
            .collect();
        serde_json::json!({
            "version": 1,
            "stat": "mean_abs",
            "hidden": self.hidden,
            "positions": self.rows.iter().copied().max().unwrap_or(0),
            "layers": layers,
        })
    }
}

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
    // Weight absorption is the decode-shaped form of the same algebra: it works
    // in the 512-wide latent space instead of reconstructing `[k_nope|v]` for
    // every cached position on every step, which is what makes the dense path
    // cost grow with context. It is **not** bit-identical — `absorb_approximates_dense`
    // holds it to a 10% relative bound, because the dense path quantizes the
    // reconstruction through `kv_b` and absorb never materializes it. So this is
    // opt-in and off by default, like every other knob that can move a token.
    // DSA first: it is the only path that maintains the indexer key cache, and
    // that cache has to be built from position 0 or a later selection has no
    // keys for the early positions. Absorb has no sparse form, so a checkpoint
    // with an indexer and `COLI_DSA=1` takes the dense-sparse path even when
    // `COLI_MLA_ABSORB` is also set — stated here rather than left to
    // whichever branch happened to come first.
    let attn = match (ctx.dsa.then_some(()).and(l.indexer.as_ref()), ctx.absorb) {
        (Some(ix), _) => mla_attention_dsa_indexed(&l.attn(), ix, &nrm, s_n, pos_base, kv, cfg)?,
        (None, true) => mla_attention_absorb(&l.attn(), &nrm, s_n, pos_base, kv, cfg)?,
        (None, false) => mla_attention(&l.attn(), &nrm, s_n, pos_base, kv, cfg)?,
    };
    for z in 0..s_n * d {
        x[z] += attn[z];
    }
    let nrm2 = rmsnorm_rows(x, &l.post_ln, s_n, d, eps);
    let ffn: Vec<f32> = if l.sparse {
        // Calibration capture sees exactly what the router is about to see —
        // the one place "the MoE input's channel magnitudes" is unambiguous.
        if let Some(cal) = ctx.calib {
            cal.lock().accumulate(li, &nrm2, s_n);
        }
        if ctx.stream_experts {
            moe_forward_dispatch(ctx, li, &nrm2, &l.router, &l.router_bias, l.shared.as_ref(), s_n)?
        } else {
            moe_forward(&nrm2, &l.router, &l.router_bias, &l.experts, l.shared.as_ref(), MoeCfg { s_n, hidden: d, k: cfg.topk as usize, norm_topk: cfg.norm_topk, routed_scale: cfg.routed_scale })
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
    rows_at: RowLayout,
    ctx: &ForwardCtx,
    x: &mut [f32],
) -> Result<(), Error> {
    let cfg = ctx.cfg;
    let d = cfg.hidden as usize;
    let eps = cfg.eps;
    let s_n = rows_at.len();
    let nrm = rmsnorm_rows(x, &l.in_ln, s_n, d, eps);
    let attn = mla_attention_rows(&l.attn(), &nrm, rows_at, caches, cfg, ctx.absorb)?;
    for z in 0..s_n * d {
        x[z] += attn[z];
    }
    let nrm2 = rmsnorm_rows(x, &l.post_ln, s_n, d, eps);
    let ffn: Vec<f32> = if l.sparse {
        // Same capture point as `forward_layer` — the batched rows are main
        // stream too (draft rows never reach here with `calib` set).
        if let Some(cal) = ctx.calib {
            cal.lock().accumulate(li, &nrm2, s_n);
        }
        if ctx.stream_experts {
            moe_forward_dispatch(ctx, li, &nrm2, &l.router, &l.router_bias, l.shared.as_ref(), s_n)?
        } else {
            moe_forward(&nrm2, &l.router, &l.router_bias, &l.experts, l.shared.as_ref(), MoeCfg { s_n, hidden: d, k: cfg.topk as usize, norm_topk: cfg.norm_topk, routed_scale: cfg.routed_scale })
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
        // Before any par_* call can lazily build the global pool: install the
        // NUMA-pinning worker hook (no-op unless COLI_NUMA_PIN=1).
        install_numa_pin_hook();
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
        // Compressed checkpoints require the decompressing read path in
        // `read_raw` / `read_f32`; the streaming lane hands raw on-disk bytes
        // straight to the kernels, which would produce garbage. Force resident
        // mode when any tensor is compressed. Emit a note so a user forcing
        // streaming knows why it was overridden.
        //
        // This override MUST precede the RAM preflight below. It used to sit
        // after it, so on a compressed checkpoint the projection was computed for
        // streaming — charging zero bytes for the routed experts — and the engine
        // then loaded all of them resident. The guard passed and the kernel killed
        // the process anyway, which is the exact failure the preflight exists to
        // prevent.
        let stream_experts = if stream_experts && st.has_compressed_tensors() {
            eprintln!("[peregrine] compressed checkpoint detected — disabling expert streaming (compressed reads decompress on read)");
            false
        } else {
            stream_experts
        };

        // Preflight: can this machine hold what is about to be loaded? Every byte
        // needed is already in the headers, so the verdict costs no extra I/O and
        // lands before the first allocation. Without it an over-large model is a
        // silent SIGKILL minutes into loading, with nothing in the log to read.
        let rss_limit_bytes: u64 = {
            // `uncompressed_nbytes`, not `nbytes`: the latter is the *on-disk*
            // length, which for a zstd tensor is the compressed payload, while
            // what the process actually holds is the decompressed size. A 2.5:1
            // container would have projected at 40% of its real footprint —
            // under-reporting in precisely the direction that gets a run killed.
            let total_bytes: u64 = st.tensors().iter().map(|t| t.uncompressed_nbytes.max(0) as u64).sum();
            let proj = crate::ram::project_load(&crate::ram::ProjectInputs {
                dense_disk: total_bytes.saturating_sub(routed_bytes),
                expert_disk: routed_bytes,
                stream_experts,
                ecache: ecache_budget_bytes() as u64,
                stream_transient: stream_transient_reserve(
                    io_rings(),
                    default_workers(),
                    4 * max_expert_region_bytes(&st),
                ) as u64,
                kv_pool: crate::ram::kv_pool_bytes(
                    cfg.kv_lora as u64,
                    cfg.qk_rope as u64,
                    cfg.n_layers as u64,
                    crate::ram::DEFAULT_PROJECTED_CTX,
                ),
                buffered_reads: !direct_enabled(),
            });
            let avail = mem_available_bytes();
            eprintln!("{}", crate::ram::summary(&proj, avail));
            let overcommit =
                matches!(std::env::var("COLI_RAM_OVERCOMMIT").ok().as_deref(), Some("1") | Some("true"));
            if let Err(msg) = crate::ram::ram_verdict(&proj, avail, overcommit) {
                return Err(Error::Format(msg));
            }
            // The guard's ceiling: an explicit override, else the peak we just
            // projected. Using the projection means the guard corrects the
            // estimate against reality rather than needing a second one.
            match std::env::var("COLI_RSS_GUARD_GB").ok().as_deref().map(str::trim) {
                Some(v) if !v.is_empty() => match v.parse::<f64>() {
                    Ok(g) if g >= 0.0 => (g * 1e9) as u64,
                    _ => {
                        eprintln!("peregrine: COLI_RSS_GUARD_GB is not a number — using the projected peak");
                        proj.peak
                    }
                },
                _ => proj.peak,
            }
        };

        // Compressed checkpoints require the decompressing read path in
        // `read_raw` / `read_f32`; the streaming lane hands raw on-disk bytes
        // straight to the kernels, which would produce garbage. Force resident
        // mode when any tensor is compressed. Emit a note so a user forcing
        // streaming knows why it was overridden.
        let stream_experts = if stream_experts && st.has_compressed_tensors() {
            eprintln!("[peregrine] compressed checkpoint detected — disabling expert streaming (compressed reads decompress on read)");
            false
        } else {
            stream_experts
        };

        let d = cfg.hidden as usize;
        let (kvl, qkr) = (cfg.kv_lora as usize, cfg.qk_rope as usize);
        let vocab = cfg.vocab as usize;

        let embed = QtWeight::load(&st, "model.embed_tokens.weight", vocab, d)?;
        let lm_head = QtWeight::load(&st, "lm_head.weight", vocab, d)?;
        let final_norm = load_f32(&st, "model.norm.weight", d)?;

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            layers.push(load_layer(&st, i, &cfg, stream_experts)?);
        }

        let kv = (0..cfg.n_layers).map(|_| LayerKv::with_dtype(kvl, qkr, kv_dtype())).collect();
        // The concurrent MoE lane needs its own ring, set up once, so a layer's
        // experts stream while the CPU pool computes. Only in streaming mode.
        // `new_streaming` honours the `COLI_SQPOLL` opt-in; other Reactors in
        // the process (prefetch lanes, the SafeTensors loader) keep plain rings
        // — each SQPOLL ring costs a polling kthread, worth it only on the
        // per-token critical path, and only when measured to win.
        let io_reactors: Vec<Mutex<Reactor>> = if stream_experts {
            let n = io_rings();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mut r =
                    Reactor::new_streaming(256).ctx(|| "concurrent MoE io_uring reactor init".to_string())?;
                // Fixed-file registration: reads whose fd is registered skip the
                // per-op fd table lookup/refcount — the same shard fds are read
                // every token. Non-fatal: on failure reads use the plain-fd path.
                if let Err(e) = r.register_files(&st.shard_fds()) {
                    peregrine_io::note_advisory_err("register shard fds with io_uring (plain-fd reads)", &e);
                }
                v.push(Mutex::new(r));
            }
            // One boot line reporting what the rings actually run with, read
            // back from the rings themselves rather than echoed from env: the
            // sqpoll-on bench arm ran for a day with nothing in any log saying
            // whether `COLI_SQPOLL` took, and `COLI_REGBUF` has a history of
            // being inert — a knob whose effect no output can confirm is a
            // knob that silently dies. `is_registered` verifies the first
            // shard fd's fixed-file slot survived registration.
            if let Some(first) = v.first() {
                let r = first.lock();
                let fixed = st.shard_fds().first().is_some_and(|&fd| r.is_registered(fd));
                eprintln!(
                    "peregrine: [io] rings={} sqpoll={} fixed_files={}",
                    v.len(),
                    if r.is_sqpoll() { "on" } else { "off" },
                    if fixed { "registered" } else { "plain-fd" },
                );
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
        // The granted budget escapes the block as well as the cache: whether it
        // clears one token's working set is what decides the `prefetch_protect`
        // default below.
        let (ecache, granted_ecache_budget) = {
            let requested = force_ecache.unwrap_or_else(ecache_budget_bytes);
            // Cap the warm cache so it + the streaming lanes' transient landing/compute
            // buffers + a safety margin fit in available RAM (already net of the
            // resident model), so batched prefill/decode can't OOM. This is the peak-RAM
            // auto-budget; bounding the per-read allocation churn itself is the slab arena.
            let budget = if stream_experts && requested > 0 {
                let per_expert = 4 * max_expert_region_bytes(&st);
                let reserve = stream_transient_reserve(io_reactors.len(), workers, per_expert);
                let avail = mem_available_bytes();
                let capped = cap_ecache_budget(requested, avail as usize, reserve, 1usize << 30);
                // One line covering both hazards: a request that had to be cut, and a
                // request granted in full that is nonetheless large enough to make its
                // own hits into page faults. The second is the counter-intuitive one —
                // it improves every number the cache itself reports while collapsing
                // throughput — so it is worth saying out loud rather than leaving to a
                // benchmark nobody runs.
                if let Some(w) = crate::ram::cache_cliff_warning(requested as u64, capped as u64, avail) {
                    eprintln!("peregrine: {w}");
                }
                capped
            } else {
                requested
            };
            let cache =
                if stream_experts && budget > 0 { Some(Arc::new(Mutex::new(WarmCache::new(budget)))) } else { None };
            (cache, budget)
        };
        // Prefetch lane: a background worker warming the next token's predicted
        // experts into the shared cache via its own ring. Spawned only when the
        // cache exists (streaming mode). `route_hist` is the predictor's state.
        let sweep = Arc::new(SweepClock::from_env());
        let (route_hist, prefetch) = match &ecache {
            Some(cache) => (
                Some(Mutex::new(RouteHistory::new(cfg.n_layers as usize, route_hist_depth()))),
                Some(spawn_prefetch_pool(cache, &st, direct, prefetch_lanes(), &sweep)?),
            ),
            None => (None, None),
        };
        // Optional GPU VRAM tier (opt-in via COLI_GPU): dequantize as many experts
        // as fit to f32 and upload. Reserve 2 GB headroom for activations/context.
        let gpu = if std::env::var("COLI_GPU").is_ok() {
            // Last session's routing heat, if any, so the residency knapsack has
            // something to rank by (empty → deterministic round-robin placement).
            let warm_heat = peek_persisted_heat(dir, &cfg);
            let tier = GpuTier::build(&st, &cfg, 2 * 1024 * 1024 * 1024, &warm_heat)?;
            if let Some(t) = &tier {
                let src = if warm_heat.is_empty() { "cold" } else { "heat-ranked" };
                eprintln!("peregrine: GPU tier holds {} experts in VRAM ({src})", t.len());
            }
            tier
        } else {
            None
        };
        // Heat accumulator for dynamic VRAM residency — only useful (and only built)
        // when there is a GPU tier to migrate hot experts into.
        // `n_layers + 1` rows: the MTP head sits at layer index `cfg.n_layers` and
        // routes a full set of experts. Sized `n_layers` until 2026-08-09, and
        // `bump` drops out-of-range silently, so that layer's experts could never
        // accumulate heat — the LFRU eviction score and the VRAM reheat ranking
        // both read this table, and both were blind to one layer's worth.
        let heat = gpu.as_ref().map(|_| HeatTable::new(cfg.n_layers as usize + 1, cfg.n_experts as usize));

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

        let model_n_layers = cfg.n_layers as usize; // read before `cfg` moves into the struct
        let model_topk = cfg.topk.max(1) as usize; // likewise
        // Resolve every routed expert's location and quantized format once, while
        // `st` and `cfg` are still borrowable. Only worth it when experts stream:
        // a resident model never consults it. See `ExpertIndex` for why this is
        // not done lazily per request.
        let expert_index =
            if stream_experts { Some(crate::concurrent::ExpertIndex::build(&st, &cfg)) } else { None };
        // One token's routed working set, and therefore which of capacity or
        // policy binds on this deployment. Reported because an operator cannot
        // otherwise tell which side of the threshold they are on, and the two
        // sides want opposite settings.
        let expert_per_token_bytes = expert_index.as_ref().map_or(0, |ix| ix.per_token_bytes(&cfg));
        let prefetch_protect = prefetch_protect_default(granted_ecache_budget, expert_per_token_bytes);
        if expert_per_token_bytes > 0 {
            let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
            let ratio = 100.0 * granted_ecache_budget as f64 / expert_per_token_bytes as f64;
            let holds = (granted_ecache_budget as u64) >= expert_per_token_bytes;
            eprintln!(
                "peregrine: [workingset] one token routes {:.2} GB of experts against a {:.2} GB cache \
                 ({ratio:.0}% of a pass) — {}; prefetch-protect {}",
                gb(expert_per_token_bytes),
                gb(granted_ecache_budget as u64),
                if holds { "recency alone can hold a pass" } else { "no eviction order can hold a pass" },
                if prefetch_protect { "on" } else { "off" },
            );
        }
        // Device-pure io claims (`COLI_IO_DEVICE_SCHED=1`): built only when the
        // claim grouping can differ from the blind cursor — streaming, >1 ring,
        // and shards genuinely on >1 device. The env read happens here at build
        // so two A/B arms in one process can never alias through a latch.
        let fd_device_table = if stream_experts
            && io_reactors.len() > 1
            && std::env::var("COLI_IO_DEVICE_SCHED").is_ok_and(|v| v.trim() == "1")
        {
            let table: std::collections::HashMap<std::os::unix::io::RawFd, u8> =
                st.fd_devices().into_iter().collect();
            let devices = table.values().collect::<std::collections::BTreeSet<_>>().len();
            (devices > 1).then(|| {
                eprintln!(
                    "peregrine: [io] device-pure claims on ({devices} devices, {} shard fds)",
                    table.len()
                );
                table
            })
        } else {
            None
        };
        let mut model = Model {
            route_hist_epoch: std::sync::atomic::AtomicBool::new(false),
            cfg,
            embed,
            absorb: absorb_enabled(),
            dsa: dsa_enabled(),
            rss_limit_bytes,
            layers,
            final_norm,
            lm_head,
            kv,
            stream_experts,
            direct,
            st,
            expert_index,
            fd_device_table,
            expert_per_token_bytes,
            prefetch_protect,
            io_reactors,
            workers,
            ecache,
            route_hist,
            predictor: PredictSource::default(),
            predict_eval: predict_eval_init(model_topk),
            prefetch_policy: PrefetchPolicy::from_env(),
            prefetch_tuner: prefetch_tuner_init(),
            perf_llc: None,
            perf_llc_last: 0,
            perf_llc_ewma: 0.0,
            prefetch,
            sweep,
            gpu,
            mtp,
            // `heat` is `Some` exactly when a GPU tier exists, which is also
            // the only world where a spill verdict can mean anything.
            spill_log: (gpu_spill_enabled() && heat.is_some()).then(|| Mutex::new(Vec::new())),
            heat,
            lane_timings: Arc::new(crate::lane::LaneTimingsAccum::new()),
            lane_totals: Arc::new(crate::lane::LaneTimingsAccum::new()),
            lane_forwards: std::sync::atomic::AtomicU64::new(0),
            bubble: Mutex::new(crate::lane::BubbleTuner::new(0.3, 1.5, 3)),
            plan_optimizer: Mutex::new(crate::telemetry::PlanOptimizer::new()),
            last_telemetry: Mutex::new(crate::telemetry::RuntimeTelemetry::default()),
            last_lane: Mutex::new(crate::lane::LaneTimings::default()),
            io_tuner: crate::iotune::IoTuner::new(
                crate::iotune::IowqCap { bounded: 4, unbounded: 4 },
                1,
                32,
            ),
            last_iowq: Mutex::new(None),
            workload_class: Mutex::new(crate::workload::TokenClass::Prose),
            effective_workers: std::sync::atomic::AtomicUsize::new(workers),
            governor: Mutex::new(GovernorState::new(workers)),
            entropy_ewma: Mutex::new(0.5),
            coactivation: Mutex::new(crate::predict::CoActivation::new(model_n_layers)),
            affinity: Mutex::new(Arc::new(crate::concurrent::AffinityHints::default())),
            learner: Mutex::new(crate::learn::Learner::from_env(workers)),
            learned_prefetch: std::sync::atomic::AtomicUsize::new(0),
            last_forward_at: Mutex::new(None),
            checkpoint_dir: dir.to_path_buf(),
            calib: None,
            layout_schedule: load_layout_schedule(dir),
            rlm: crate::rlm::RLMController::new(),
        };
        // Calibration capture (ideas #7): read per load, not through a
        // OnceLock latch — the seam `enable_calib_capture` is what tests and
        // the calib-capture subcommand use, this env spelling is for co-runs
        // (e.g. alongside COLI_PREDICT_EVAL in one instrumented pass).
        match std::env::var("COLI_CALIB_CAPTURE") {
            Ok(p) if !p.trim().is_empty() => model.enable_calib_capture(std::path::PathBuf::from(p.trim())),
            Ok(_) => {}
            Err(std::env::VarError::NotPresent) => {}
            Err(e) => peregrine_io::note_advisory_err("COLI_CALIB_CAPTURE read", &e),
        }
        // Upgrade the predictor to the offline transition automaton if a matching
        // `automaton.json` sits next to the checkpoint (else stay on momentum),
        // then blend in the macro-state table if one is present.
        model.try_attach_automaton(dir);
        model.try_attach_macrostates(dir);
        // Load any persisted routing history / heat snapshot so a fresh process
        // starts warm on the last session's routing patterns. Correctness-neutral;
        // missing/stale files are silently ignored.
        model.try_load_route_stats(dir);
        // This host's WMMA tuning table, if a previous run on this machine wrote
        // one. A no-op unless `COLI_CUDA_AUTOTUNE=1`; a table from another GPU
        // is not rejected because it cannot be detected, which is precisely why
        // it re-explores every shape before trusting a restored winner
        // (`WmmaTuner::select`).
        model.try_load_kernel_tuning(dir);
        // Storage-tier seed: prefetch-warm the offline-planned RAM tier so the
        // co-firing communities the planner placed in RAM are resident before
        // the first token. Best-effort; bounded; `COLI_TIER_SEED=0` disables.
        model.try_seed_tiers(dir);
        // Compiled execution plan (plan.json from `compile-plan`) — a single
        // profile-guided artifact bundling all of the above; applied last so a
        // plan wins over the standalone files.
        model.try_load_plan(dir);
        // Wire the trunk (`COLI_MLOCK=1`). Here specifically, and the position is the
        // feature: every resident weight is loaded by now, and the warm cache has not
        // filled yet, so `mlockall(MCL_CURRENT)` pins exactly the part that must never
        // be paged out and leaves the part that is *supposed* to be reclaimable alone.
        // Reported rather than silent — a refusal is the normal outcome on a desktop
        // `RLIMIT_MEMLOCK`, and an operator who asked for wiring needs to know they
        // did not get it.
        match peregrine_io::wire_resident() {
            peregrine_io::Wired::Skipped => {}
            peregrine_io::Wired::Locked { bytes } => {
                eprintln!("peregrine: [mlock] trunk wired, {:.1} GB resident and unswappable", bytes as f64 / 1e9);
            }
            peregrine_io::Wired::Refused { errno, limit } => {
                let lim = if limit == u64::MAX { "unlimited".to_string() } else { format!("{:.1} MB", limit as f64 / 1e6) };
                eprintln!(
                    "peregrine: [mlock] COLI_MLOCK=1 but the kernel refused (errno {errno}, RLIMIT_MEMLOCK {lim}) \
                     — running unwired. Raise it with `ulimit -l unlimited` or grant CAP_IPC_LOCK; \
                     nothing else changes, the trunk is just page-out-eligible again."
                );
            }
        }
        Ok(model)
    }

    /// Consume a compiled execution plan (`<dir>/plan.json`, from the engine's
    /// `compile-plan` subcommand): one config-tagged artifact bundling the
    /// automaton, macro-states, layout schedule, tier placement, and learned
    /// knob policy — every input a recorded profile, applied in one shot
    /// ("profile-guided execution planning"). Parts are individually optional;
    /// each goes through the same validation as its standalone file. Applied
    /// after the standalone artifacts, so a plan wins where both exist.
    fn try_load_plan(&mut self, dir: &std::path::Path) {
        let Some(v) = read_optional_artifact(dir, "plan.json") else { return };
        let tag = config_tag(&self.cfg);
        if let Some(av) = v.get("automaton") {
            if let Some(table) = TransitionTable::from_json(av) {
                if table.tag() == tag {
                    self.predictor =
                        PredictSource::Automaton { table: Arc::new(table), fallback: Momentum::default() };
                }
            }
        }
        if let Some(mv) = v.get("macrostates") {
            if let Some(table) = crate::predict::MacroTable::from_json(mv) {
                if table.tag() == tag {
                    let inner = std::mem::take(&mut self.predictor);
                    self.predictor =
                        PredictSource::WithMacro { table: Arc::new(table), inner: Box::new(inner) };
                }
            }
        }
        if let Some(sv) = v.get("schedule") {
            if let Some(sched) = parse_schedule_value(sv) {
                self.layout_schedule = Some(sched);
            }
        }
        if let Some(lv) = v.get("learn") {
            if !lv.is_null() {
                if let Some(l) = self.learner.lock().as_mut() {
                    l.restore(lv);
                }
            }
        }
        if let Some(tv) = v.get("tiers") {
            self.seed_tiers_from_value(tv);
        }
    }

    /// Read `<dir>/tiers.json` (from the galactic pass) and enqueue prefetch
    /// warms for the RAM-tier experts — bounded at 256 entries so a huge plan
    /// can't stall load. No-op without a prefetch pool / warm cache.
    fn try_seed_tiers(&self, dir: &std::path::Path) {
        let Some(v) = read_optional_artifact(dir, "tiers.json") else { return };
        self.seed_tiers_from_value(&v);
    }

    /// Shared tier-seed application (standalone tiers.json and plan.json both
    /// land here).
    fn seed_tiers_from_value(&self, v: &serde_json::Value) {
        if matches!(std::env::var("COLI_TIER_SEED").as_deref(), Ok("0") | Ok("false")) {
            return;
        }
        let (Some(pool), Some(_)) = (&self.prefetch, &self.ecache) else { return };
        let Some(ram) = v.get("ram").and_then(|r| r.as_array()) else { return };
        let mut items = Vec::new();
        for pair in ram.iter().take(256) {
            let Some(a) = pair.as_array() else { continue };
            let (Some(l), Some(e)) = (a.first().and_then(|x| x.as_u64()), a.get(1).and_then(|x| x.as_u64()))
            else {
                continue;
            };
            match crate::concurrent::prefetch_item(self.expert_index.as_ref(), &self.st, &self.cfg, l as usize, e as usize) {
                Ok(item) => items.push(item),
                // seed warming is speculative; a bad tier entry is skipped
                Err(e) => peregrine_io::note_advisory_err("tier-seed prefetch resolve", &e),
            }
        }
        if !items.is_empty() && pool.lane(0).tx.send(PrefetchMsg::Warm(items, u64::MAX)).is_err() {
            peregrine_io::note_advisory_err("prefetch warm dispatch", &"prefetch lane is down");
        }
    }

    /// Clear the KV cache to start a fresh sequence. Also clears the prefetch
    /// predictor's history (a new sequence has no useful routing history); the
    /// warm cache itself persists (a warm expert is warm regardless of sequence).
    pub fn reset(&mut self) {
        let (kvl, qkr) = (self.cfg.kv_lora as usize, self.cfg.qk_rope as usize);
        for k in &mut self.kv {
            *k = LayerKv::with_dtype(kvl, qkr, kv_dtype());
        }
        if let Some(h) = &self.route_hist {
            h.lock().clear();
        }
        // A new sequence starts from pure LRU until its predictor re-protects experts.
        if let Some(c) = &self.ecache {
            c.lock().clear_priorities();
        }
        // RLM: reset per-token recursion state so the new sequence starts clean.
        self.rlm.reset();
    }

    /// Set the current workload class (from the serving layer's prompt
    /// classifier). Subsequent prefetch enqueues use that class's breadth
    /// overrides ([`PrefetchPolicy::for_class`]). `&self` — interior mutability
    /// so the batch engine can call it while holding a shared model borrow.
    pub fn set_workload_class(&self, class: crate::workload::TokenClass) {
        *self.workload_class.lock() = class;
    }

    /// The currently-active per-class prefetch policy.
    fn class_policy(&self) -> PrefetchPolicy {
        let mut policy = self.prefetch_policy.for_class(*self.workload_class.lock());
        // The adaptive controller's cap belongs here, not at one call site: it
        // used to be applied only in `forward_hidden`, so `prefetch_ctx()` (the
        // COLI_PREFETCH_LOOKAHEAD=0 path) and per-sequence batched prefetch kept
        // the raw per-class breadth. The tuner then reported a converged
        // distance that nothing ever applied.
        if let Some(t) = &self.prefetch_tuner {
            policy.warm_paths = t.distance();
        }
        policy
    }

    /// Borrow `self`'s prefetch lane, predictor, history, cache and tensor handles as
    /// a [`PrefetchCtx`]. `None` unless both the prefetch lane and warm cache exist.
    fn prefetch_ctx(&self) -> Option<PrefetchCtx<'_>> {
        let policy = self.class_policy();
        match (&self.prefetch, &self.route_hist, &self.ecache) {
            (Some(pool), Some(hist), Some(cache)) => Some(PrefetchCtx {
                prefetch: pool.lane(0),
                predictor: &self.predictor,
                hist,
                cache,
                gpu: self.gpu.as_ref(),
                st: &self.st,
                cfg: &self.cfg,
                expert_index: self.expert_index.as_ref(),
                warm_paths: policy.warm_paths,
                hint_paths: policy.hint_paths,
                direct: self.direct,
                sweep: self.sweep.as_ref(),
            }),
            _ => None,
        }
    }

    /// Borrow `self`'s prefetch lane, cache and tensor handles as a [`LookaheadCtx`].
    /// `None` unless both the prefetch lane and the warm cache exist — with nowhere to
    /// put a speculative slab, there is nothing for the look-ahead to do.
    ///
    /// Note what is *not* required: a [`RouteHistory`]. The look-ahead reads the next
    /// layer's routing off that layer's own weights, so it works on a cold process
    /// where every history-based predictor is still empty.
    fn lookahead_ctx(&self) -> Option<LookaheadCtx<'_>> {
        match (&self.prefetch, &self.ecache) {
            (Some(pool), Some(cache)) => Some(LookaheadCtx {
                prefetch: pool.lane(0),
                cache,
                gpu: self.gpu.as_ref(),
                st: &self.st,
                cfg: &self.cfg,
                expert_index: self.expert_index.as_ref(),
                sweep: self.sweep.as_ref(),
            }),
            _ => None,
        }
    }

    /// Predictive eviction: protect the experts the predictor expects to be reused
    /// next. For each sparse layer, set a priority on the *resident* predicted experts
    /// (higher predictor score → higher priority, routing heat breaking ties). Experts
    /// not predicted keep priority 0 and are evicted first. Correctness-neutral —
    /// priority only reorders eviction victims, never what `get`/`insert` return. A
    /// no-op without both history and cache.
    fn update_cache_protection(&self) {
        if let Some(hist) = &self.route_hist {
            self.protect_from(hist);
        }
    }

    /// Protect the experts predicted from `hist` (a single stream's routing) in the
    /// shared cache. Shared by the single-stream path and per-sequence batched prefetch.
    fn protect_from(&self, hist: &Mutex<RouteHistory>) {
        let Some(cache) = &self.ecache else {
            return;
        };
        let heat = self.heat.as_ref().map(|h| h.snapshot());
        let n_experts = self.cfg.n_experts as usize;
        let first_dense = self.cfg.first_dense as usize;
        let n_layers = self.cfg.n_layers as usize;
        // Build the whole protection set under the *history* lock only, then
        // apply it in one short cache-lock hold. Previously both locks were held
        // across all 78 layers × K predictions; the cache lock is contended by
        // the prefetch lane and by every streamed read, so that window cost more
        // than the work inside it.
        let entries: Vec<((u32, u32), u32)> = {
            let hist = hist.lock();
            let mut v = Vec::new();
            for layer in first_dense..n_layers {
                for (e, score) in self.predictor.predict_layer(layer, &hist) {
                    let h = heat.as_ref().and_then(|c| c.get(layer * n_experts + e as usize).copied()).unwrap_or(0);
                    v.push(((layer as u32, e), pack_prio(score, h)));
                }
            }
            v
        };
        cache.lock().set_protected(&entries);
    }

    /// Whole-forward next-token prefetch: emit every sparse layer's prediction in one
    /// pass at the end of the forward. Used by tests, and as the fallback when layer
    /// look-ahead is disabled (`COLI_PREFETCH_LOOKAHEAD=0`).
    fn enqueue_prefetch(&self) {
        let Some(ctx) = self.prefetch_ctx() else {
            return;
        };
        for layer in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
            ctx.emit_layer(layer);
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
    /// Run-lifetime per-lane totals and the forward count they cover.
    ///
    /// **The sums are across concurrent lanes, so they exceed wall clock when the
    /// pipeline is working.** That is the point: `(io+cpu+gpu+reduce) / wall` is
    /// the overlap actually achieved, and a ratio near 1.0 means the lanes are
    /// running one after another — the thing a concurrent scheduler exists to
    /// prevent, and which nothing in this engine could previously report.
    pub fn lane_totals(&self) -> (crate::lane::LaneTimings, u64) {
        (
            self.lane_totals.snapshot(),
            self.lane_forwards.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn ecache_stats(&self) -> Option<(u64, u64, u64)> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            // total_misses, not the raw field: misses the I/O lane resolved
            // lock-free against the residency filter live in a separate atomic.
            (c.hits, c.total_misses(), c.disk_reads)
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

    /// O_DIRECT landing buffers checked out across the streaming rings right
    /// now (`None` when experts are resident — no rings). Stuck at the pool
    /// cap means reads are serializing on buffer availability.
    pub fn io_slab_in_use(&self) -> Option<usize> {
        if self.io_reactors.is_empty() {
            return None;
        }
        Some(self.io_reactors.iter().map(|r| r.lock().slab_in_use()).sum())
    }

    /// Prefetch-lane reads the warm tier has attributed to `layer` (lets the
    /// look-ahead test confirm early layers were warmed mid-forward).
    pub fn ecache_prefetch_reads_for_layer(&self, layer: usize) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().prefetch_reads_for_layer(layer as u32))
    }

    /// Low-confidence experts the prefetch lane hinted to the page cache via
    /// `fadvise` (multi-path tier 2).
    /// `(resolved, mergeable)` from the load-time expert map: how many routed
    /// experts were indexed, and how many of those read as two coalesced extents
    /// rather than six regions. `None` when experts are resident (no map built).
    ///
    /// Reported at shutdown because the coalescing win is entirely a property of
    /// *this* container's layout — a rewritten or straddling checkpoint can drop
    /// the mergeable count without anything else changing.
    pub fn expert_map_stats(&self) -> Option<(usize, usize)> {
        self.expert_index.as_ref().map(|ix| (ix.resolved(), ix.mergeable()))
    }

    /// Bytes one token's routing touches, and whether predictive eviction
    /// protection ended up on. `None` when experts are resident.
    ///
    /// Read this next to the hit rate: below one token's working set a sweep
    /// drives plain recency to zero hits and the number says nothing about the
    /// policy; above it, recency works unaided. The two regimes want opposite
    /// settings, so a hit rate quoted without this is not interpretable.
    pub fn expert_working_set(&self) -> Option<(u64, bool)> {
        (self.expert_per_token_bytes > 0).then_some((self.expert_per_token_bytes, self.prefetch_protect))
    }

    pub fn ecache_fadvise_hints(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().fadvise_hints)
    }

    /// Speculative reads whose opt-in verification re-read differed (always 0 in a
    /// correct system; nonzero signals an I/O bug).
    pub fn ecache_verify_mismatch(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().verify_mismatch)
    }

    /// Speculative warm items the lane discarded unread because their layer window
    /// had passed (`COLI_PREFETCH_STALE_DROP`). Each is a disk read not spent.
    pub fn ecache_prefetch_stale_dropped(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().prefetch_stale_dropped)
    }

    /// The warm cache's achieved compression ratio (uncompressed ÷ admitted).
    /// `None` without a cache or when `COLI_CACHE_COMPRESS` is off. Around 1.2x
    /// is the expected figure for packed int4 experts; near 1.0 means the decode
    /// on every hit is buying nothing.
    pub fn ecache_compression_ratio(&self) -> Option<f64> {
        self.ecache.as_ref().and_then(|c| c.lock().compression_ratio())
    }

    /// Prefetch accuracy = `used / (used + wasted)` — the share of speculative reads
    /// that paid off. `None` without a cache; `0.0` before any prefetch settled.
    pub fn prefetch_accuracy(&self) -> Option<f64> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            let total = c.prefetch_used + c.prefetch_wasted;
            if total == 0 {
                0.0
            } else {
                c.prefetch_used as f64 / total as f64
            }
        })
    }

    /// Prefetch effectiveness: `(used, wasted)` — slabs warmed by the prefetch lane
    /// that were later hit vs. evicted before any use. The signal the distance tuner
    /// (`PrefetchTuner`) and shutdown accuracy log read.
    pub fn ecache_prefetch_effectiveness(&self) -> Option<(u64, u64)> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            (c.prefetch_used, c.prefetch_wasted)
        })
    }

    /// `(bytes, slots)` currently held by prefetched slabs nothing has hit, and the
    /// cache budget for scale. See [`WarmCache::speculative_resident`] — this is how
    /// much of the cache speculation is *holding*, which `used`/`wasted` cannot say.
    pub fn ecache_speculative_resident(&self) -> Option<(usize, u64, usize)> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            let (bytes, slots) = c.speculative_resident();
            (bytes, slots, c.budget())
        })
    }

    /// `(slots, used_bytes, budget_bytes)` resident at this moment — how full the
    /// warm cache actually is.
    ///
    /// Reported because a near-zero hit rate has two very different causes that the
    /// hit/miss counters cannot tell apart: a **full** cache means the working set
    /// is being evicted before reuse, an **empty** one means admission is not
    /// happening at all. Guessing which was costing whole measurement runs.
    pub fn ecache_occupancy(&self) -> Option<(usize, usize, usize)> {
        self.ecache.as_ref().map(|c| {
            let c = c.lock();
            (c.len(), c.used_bytes(), c.budget())
        })
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
            p.barrier();
        }
    }

    /// Trigger the next-token prefetch on demand (same path `forward_hidden` uses).
    /// Exposed for tests that warm a deliberately-cleared cache from history.
    pub fn prefetch_from_history(&self) {
        self.enqueue_prefetch();
    }

    /// Trigger the per-layer look-ahead prefetch for a single `layer` (the mid-forward
    /// path in isolation). Exposed for tests that observe one layer's warming.
    pub fn prefetch_layer_from_history(&self, layer: usize) {
        if let Some(ctx) = self.prefetch_ctx() {
            ctx.emit_layer(layer);
        }
    }

    /// Enqueue per-sequence prefetch for the batched serving engine: warm the experts
    /// predicted from one sequence's own routing history onto prefetch lane `lane`
    /// (round-robin across the pool → parallel-async streaming), and protect them from
    /// eviction. Correctness-neutral. No-op without a prefetch pool + cache.
    pub fn enqueue_seq_prefetch(&self, hist: &Mutex<RouteHistory>, lane: usize) {
        let (Some(pool), Some(cache)) = (&self.prefetch, &self.ecache) else {
            return;
        };
        let policy = self.class_policy();
        let ctx = PrefetchCtx {
            prefetch: pool.lane(lane),
            predictor: &self.predictor,
            hist,
            cache,
            gpu: self.gpu.as_ref(),
            st: &self.st,
            cfg: &self.cfg,
            expert_index: self.expert_index.as_ref(),
            warm_paths: policy.warm_paths,
            hint_paths: policy.hint_paths,
            direct: self.direct,
            sweep: self.sweep.as_ref(),
        };
        for layer in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
            ctx.emit_layer(layer);
        }
        if self.prefetch_protect {
            self.protect_from(hist);
        }
    }

    /// A fresh per-sequence routing history sized to this model (for the batched
    /// engine to give each stream its own predictor state).
    pub fn new_route_history(&self) -> RouteHistory {
        RouteHistory::new(self.cfg.n_layers as usize, route_hist_depth())
    }

    /// Runtime expert replication: for the top `k` hottest GPU-resident experts,
    /// also enqueue prefetch reads so their bytes land in the warm cache. When
    /// the lane balancer later downgrades a resident expert back to CPU (phase
    /// shift, GPU overload), the CPU lane serves it out of RAM instead of
    /// hitting the disk. Correctness-neutral (WarmCache holds the same bytes
    /// the disk lane would have streamed); the cost is one prefetch per
    /// replicated expert per invocation.
    ///
    /// Enabled by `COLI_REPLICATE_K` (default 0 = off). Called from
    /// [`Self::reheat`] once the residency set is settled. No-op without a GPU
    /// tier, without the prefetch lane, or without the warm cache.
    pub fn enqueue_expert_replicas(&self, k: usize) {
        if k == 0 {
            return;
        }
        let (Some(gpu), Some(pool), Some(cache)) = (self.gpu.as_ref(), &self.prefetch, &self.ecache) else {
            return;
        };
        let Some(heat) = &self.heat else { return };
        let counts = heat.snapshot();
        let n_experts = self.cfg.n_experts as usize;
        // Rank the resident (layer, expert) pairs by heat and take the hottest k.
        let mut resident_hot: Vec<((usize, usize), u32)> = Vec::new();
        for layer in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
            for e in 0..n_experts {
                if gpu.has(layer, e) {
                    let h = counts.get(layer * n_experts + e).copied().unwrap_or(0);
                    resident_hot.push(((layer, e), h));
                }
            }
        }
        resident_hot.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        resident_hot.truncate(k);
        // Build one PrefetchItem per replica; skip those already warm (would be
        // a redundant read and would pollute the warm/wasted accounting).
        let mut items = Vec::new();
        {
            let c = cache.lock();
            for ((layer, e), _) in resident_hot {
                let key = (layer as u32, e as u32);
                if c.contains(key) {
                    continue;
                }
                match crate::concurrent::prefetch_item(self.expert_index.as_ref(), &self.st, &self.cfg, layer, e) {
                    Ok(item) => items.push(item),
                    Err(e) => peregrine_io::note_advisory_err("warm-list prefetch resolve", &e),
                }
            }
        }
        if !items.is_empty() && pool.lane(0).tx.send(PrefetchMsg::Warm(items, u64::MAX)).is_err() {
            peregrine_io::note_advisory_err("prefetch warm dispatch", &"prefetch lane is down");
        }
    }

    /// Whether runtime expert replication is enabled and the target replica set
    /// size from env (`COLI_REPLICATE_K`). Default `0` (off).
    fn replica_k() -> usize {
        std::env::var("COLI_REPLICATE_K").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
    }

    /// Load a matching `automaton.json` next to the checkpoint and, if its tag matches
    /// this model's config, switch the predictor to the transition automaton (with a
    /// momentum fallback). A missing, malformed, or stale artifact is silently ignored
    /// (the model stays on momentum). Correctness-neutral.
    fn try_attach_automaton(&mut self, dir: &std::path::Path) {
        let Some(v) = read_optional_artifact(dir, "automaton.json") else { return; };
        let Some(table) = TransitionTable::from_json(&v) else {
            return;
        };
        if table.tag() == config_tag(&self.cfg) {
            self.predictor = PredictSource::Automaton { table: Arc::new(table), fallback: Momentum::default() };
        }
    }

    /// Load a matching `macrostates.json` next to the checkpoint and, if its tag
    /// matches, wrap the current predictor with the macro-state blend. Called
    /// after [`Self::try_attach_automaton`] so it composes over whichever base
    /// source won. Missing/stale artifacts are silently ignored.
    fn try_attach_macrostates(&mut self, dir: &std::path::Path) {
        let Some(v) = read_optional_artifact(dir, "macrostates.json") else { return; };
        let Some(table) = crate::predict::MacroTable::from_json(&v) else {
            return;
        };
        if table.tag() == config_tag(&self.cfg) {
            let inner = std::mem::take(&mut self.predictor);
            self.predictor = PredictSource::WithMacro { table: Arc::new(table), inner: Box::new(inner) };
        }
    }

    /// Load `<dir>/route_stats.json` if present and its config fingerprint matches:
    /// restore the routing history and (when a GPU tier exists) the routing-heat
    /// counters. Missing/malformed/stale files are silently ignored — the model
    /// starts from a cold predictor. Correctness-neutral (history and heat only
    /// affect prefetch/eviction/residency, never logits).
    fn try_load_route_stats(&mut self, dir: &std::path::Path) {
        if !route_stats_persist_enabled() {
            return;
        }
        let Some(v) = read_optional_artifact(dir, "route_stats.json") else { return; };
        let tag = config_tag(&self.cfg);
        // The file carries the config fingerprint it was written under, but only
        // the history checked it. Heat is a flat `[layer * n_experts + expert]`
        // array, so a checkpoint re-converted with a different expert count
        // restored every count onto the WRONG (layer, expert) pair — driving
        // VRAM residency and cache admission from silently scrambled data. The
        // co-activation pairs and the learned policy have the same exposure.
        // One gate for the whole document: a fingerprint mismatch means start
        // cold, which is exactly what a fresh checkpoint should do.
        if v.get("tag").and_then(|t| t.as_str()) != Some(tag.as_str()) {
            if v.get("tag").is_some() {
                peregrine_io::note_advisory_err(
                    "route_stats.json ignored (written for a different model config)",
                    &"config fingerprint mismatch",
                );
            }
            return;
        }
        if let Some(hist_v) = v.get("hist") {
            if let (Some(hist), Some(new_hist)) = (self.route_hist.as_ref(), RouteHistory::from_json(hist_v, &tag)) {
                *hist.lock() = new_hist;
            }
        }
        if let Some(heat_v) = v.get("heat") {
            if let (Some(heat), Some(arr)) = (self.heat.as_ref(), heat_v.as_array()) {
                let snap: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
                if !heat.restore(&snap) {
                    peregrine_io::note_advisory_err(
                        "route_stats heat snapshot ignored (shape mismatch)",
                        &format!("{} counts for {} slots", snap.len(), heat.len()),
                    );
                }
            }
        }
        // Cross-session co-activation ("automatic expert fusion from long-term
        // co-activation"): restore the pair counts and immediately rebuild the
        // affinity snapshot so the very first forwards already order fused
        // pairs adjacently.
        if let Some(coact_v) = v.get("coact") {
            if let Some(co) = crate::predict::CoActivation::from_json(coact_v) {
                *self.coactivation.lock() = co;
                self.rebuild_affinity();
            }
        }
        // Learned-scheduler policy: restore into whichever learner mode is
        // active (kind-mismatched policies are ignored inside `restore`).
        if let Some(learn_v) = v.get("learn") {
            if !learn_v.is_null() {
                if let Some(l) = self.learner.lock().as_mut() {
                    l.restore(learn_v);
                }
            }
        }
    }

    /// The checkpoint directory this model was loaded from — the natural home for
    /// cross-session artifacts (route stats, layout hints, kernel tuning).
    /// Total on-disk bytes of one expert's gate/up/down tensors (weights plus
    /// scales), as the checkpoint actually stores them. `None` when the expert
    /// is absent from the index. Offline planners size VRAM/RAM tiers with this
    /// instead of assuming a quantization format.
    pub fn expert_bytes_on_disk(&self, layer: usize, expert: usize) -> Option<u64> {
        let mut total = 0u64;
        for t in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            let name = format!("model.layers.{layer}.mlp.experts.{expert}.{t}");
            let w = self.st.uncompressed_nbytes(&name)?;
            let s = self.st.uncompressed_nbytes(&format!("{name}.qs")).unwrap_or(0);
            total = total.saturating_add(w.max(0) as u64).saturating_add(s.max(0) as u64);
        }
        Some(total)
    }

    pub fn checkpoint_dir(&self) -> &std::path::Path {
        &self.checkpoint_dir
    }

    /// Install the calibration accumulator (ideas #7), directing the sidecar
    /// to `out`. The explicit seam behind `COLI_CALIB_CAPTURE`, so tests and
    /// the `calib-capture` subcommand never touch process env (the same
    /// rationale as `load_streaming_ecache`). Subsequent main-stream forwards
    /// fold every sparse layer's MoE input into per-channel `Σ|x|`; call
    /// [`Self::write_calib_sidecar`] to persist the means.
    pub fn enable_calib_capture(&mut self, out: std::path::PathBuf) {
        let (n_layers, hidden) = (self.cfg.n_layers as usize, self.cfg.hidden as usize);
        self.calib = Some((out, Mutex::new(CalibAccum::new(n_layers, hidden))));
    }

    /// Write the captured calibration means as the sidecar `--calib` reads
    /// (`{"version":1,"stat":"mean_abs",...}`), atomically. Deliberately an
    /// explicit call rather than a `Drop` hook: a crashed or interrupted
    /// capture writes nothing, instead of persisting a partial mean that
    /// would silently calibrate every later conversion. `Ok(None)` when
    /// capture was never enabled.
    pub fn write_calib_sidecar(&self) -> Result<Option<std::path::PathBuf>, Error> {
        let Some((out, acc)) = &self.calib else {
            return Ok(None);
        };
        let v = acc.lock().to_sidecar_json();
        let bytes =
            serde_json::to_vec_pretty(&v).map_err(|e| Error::Format(format!("calibration sidecar: {e}")))?;
        peregrine_core::durable::write_atomic(out, &bytes)?;
        Ok(Some(out.clone()))
    }

    /// Snapshot this forward's per-lane wall time into the bubble tuner and the
    /// `last_lane` cache readers of `lane_timings()` observe. Idempotent —
    /// calling twice in a row returns zeros the second time because the
    /// accumulator is swap-reset. Also folds the I/O sample into the IoTuner
    /// and, when its recommendation changes, applies the new cap to each
    /// reactor (`register_iowq_max_workers`). The set-workers syscall is
    /// best-effort — a soft error there is not surfaced.
    fn publish_lane_timings(&self) {
        let sample = self.lane_timings.snapshot_and_reset();
        *self.last_lane.lock() = sample;
        // Fold into the run-lifetime totals before the sample is handed to the
        // tuner, so the operator-visible report and the controller see the same
        // numbers rather than two independently-derived ones.
        self.lane_totals.add_io(sample.io_us);
        self.lane_totals.add_cpu(sample.cpu_us);
        self.lane_totals.add_gpu(sample.gpu_us);
        self.lane_totals.add_reduce(sample.reduce_us);
        self.lane_totals.add_cpu_bytes(sample.cpu_bytes);
        self.lane_totals.add_lane_wall(sample.lane_wall_us);
        self.lane_totals.add_cache_wait(sample.cache_wait_us);
        self.lane_forwards.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Advance the heat table's recency clock exactly once per forward. This
        // is the per-forward tick every other adaptive structure already hangs
        // off, which is why it lives here rather than beside the per-layer heat
        // bump: ticking there would run 78× a token and saturate `lfru_score`'s
        // 255-step recency window within three tokens.
        if let Some(heat) = &self.heat {
            heat.tick();
        }
        // One implementation of the per-forward tick: `PlanOptimizer` folds the
        // sample into the bubble tuner and steps the I/O tuner on its period.
        // (It was previously a second, divergent copy of this policy that no
        // code path ever constructed.)
        let cache_counters = self.ecache.as_ref().map(|c| {
            let c = c.lock();
            crate::telemetry::CacheCounters {
                hits: c.hits,
                misses: c.total_misses(),
                prefetch_used: c.prefetch_used,
                prefetch_wasted: c.prefetch_wasted,
            }
        });
        let mut telemetry = {
            let mut bubble = self.bubble.lock();
            self.plan_optimizer.lock().tick(&mut bubble, &self.io_tuner, sample, 10_000, cache_counters)
        };
        // Routing entropy is computed per forward and, until now, had no way
        // out of the model: `routing_entropy_ewma()` said "for telemetry
        // scrapes" and no telemetry structure carried it.
        telemetry.entropy_ewma = self.routing_entropy_ewma();
        *self.last_telemetry.lock() = telemetry;
        // Sensor governors: thermal / power / bandwidth, all writing the one
        // effective-worker knob with shrink-wins arbitration.
        let governor_ceiling = {
            use std::sync::atomic::Ordering;
            let current = self.effective_workers.load(Ordering::Relaxed);
            let delta = self.governor.lock().step(sample, current);
            let next = if delta != 0 {
                let n = (current as i64 + delta as i64).clamp(2, self.workers as i64) as usize;
                self.effective_workers.store(n, Ordering::Relaxed);
                n
            } else {
                current
            };
            // A thermal/power shrink is a hardware-safety limit, not a
            // suggestion. Remember it as a ceiling: the learned scheduler below
            // used to overwrite this value outright, so every throttle was undone
            // in the same function call that applied it.
            if delta < 0 {
                next
            } else {
                self.workers
            }
        };
        // Routing-entropy EWMA (drives entropy-adaptive prefetch breadth; the
        // nudge itself happens in forward_hidden where the tuner is &mut).
        // Also updated when a learner is active: the Q learner's `stability`
        // feature reads this EWMA, so leaving it pinned at its initial value
        // collapsed half the Q-table to dead rows whenever COLI_ENTROPY_ADAPT
        // happened to be off.
        if entropy_adapt_enabled() || self.learner.lock().is_some() {
            if let Some(h) = self.routing_entropy() {
                let mut e = self.entropy_ewma.lock();
                *e = 0.7 * *e + 0.3 * h;
            }
        }
        // Learned scheduler: reward the last choice with the inter-forward wall
        // interval (the observable decode latency), then choose the next knob
        // configuration and stage it (workers applied here; prefetch distance
        // staged for forward_hidden where the tuner is &mut).
        {
            let mut learner = self.learner.lock();
            if let Some(l) = learner.as_mut() {
                use std::sync::atomic::Ordering;
                let now = std::time::Instant::now();
                let latency_us = {
                    let mut prev = self.last_forward_at.lock();
                    let d = prev.map(|p| now.duration_since(p).as_micros() as u64);
                    *prev = Some(now);
                    d
                };
                let bias = self.bubble.lock().bias();
                let entropy = *self.entropy_ewma.lock();
                if let Some(us) = latency_us {
                    l.reward(us, bias, entropy);
                }
                // Read the learner's own last choice verbatim. Re-flooring it at
                // 4 here made `PrefetchDown` below 4 a permanent no-op: the
                // learner kept choosing it, kept being rewarded for a knob change
                // that never happened, and its model of `cur` drifted from what
                // the engine actually applied.
                let cur = crate::learn::KnobArm {
                    prefetch_distance: self.learned_prefetch.load(Ordering::Relaxed).max(1),
                    workers: self.effective_workers.load(Ordering::Relaxed),
                };
                let next = l.choose(bias, entropy, cur, self.workers);
                self.learned_prefetch.store(next.prefetch_distance, Ordering::Relaxed);
                // Clamped by the governor's ceiling, so a learned policy can
                // only ever choose *within* the safe envelope.
                let w = next.workers.clamp(2, self.workers.min(governor_ceiling.max(2)));
                self.effective_workers.store(w, Ordering::Relaxed);
            }
        }
        // Co-activation tracking: fold this forward's routed sets (newest frame
        // per layer) into the pair counter; every 64 forwards rebuild the
        // affinity snapshot (fused pairs at the high threshold, hyperedge
        // components at half that). Single-stream mode only — the batched path
        // records per-sequence histories the engine owns, not `route_hist`.
        // Only fold a frame the *current* forward produced. The batched decode
        // path records per-sequence histories instead of `route_hist`, so this
        // used to re-observe one frozen frame on every step: every pair in it
        // reached a co-firing rate of ~1.0, the fusion threshold declared them
        // all fused, and that fabricated snapshot was persisted for the next
        // session to start from.
        let advanced = self.route_hist_epoch.swap(false, std::sync::atomic::Ordering::Relaxed);
        if let Some(hist) = self.route_hist.as_ref().filter(|_| advanced) {
            let frames = {
                let h = hist.lock();
                let mut co = self.coactivation.lock();
                for l in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
                    if let Some(f) = h.latest(l) {
                        co.observe(l, f);
                    }
                }
                co.note_forward();
                co.frames
            };
            if frames.is_multiple_of(64) {
                self.rebuild_affinity();
            }
        }
        // Feed the I/O tuner: sample the batched-read wall time (µs) plus this
        // forward's submission-queue-full rejections (the queue-pressure signal
        // that triggers worker halving), then step it (static SLA target of
        // 10ms per forward — well above the per-forward budget of typical MoE
        // inference). When it recommends a new cap, apply it once across all
        // reactors and remember what we set.
        // Drain the reactors' SQ-full counters every forward, even one served
        // entirely from the warm cache: leaving them to accumulate attributed a
        // whole quiet period's rejections to whichever later forward happened to
        // do I/O, which read as a spike and halved the worker cap.
        // Under `COLI_SQPOLL` the poll kthread drains the SQ continuously, so
        // this delta reads near-zero and the halving trigger goes quiet — the
        // read-µs EWMA is then the tuner's live signal. The io-wq caps the tuner
        // sets still bind: cold reads punt to io-wq under SQPOLL all the same.
        let sq_full_delta: u64 = self.io_reactors.iter().map(|r| r.lock().take_sq_full()).sum();
        if sample.io_us > 0 || sq_full_delta > 0 {
            self.io_tuner.note_read(sample.io_us, sq_full_delta);
        }
        if let Some(rec) = self.io_tuner.recommend() {
            let mut prev = self.last_iowq.lock();
            if *prev != Some(rec) {
                for ring in &self.io_reactors {
                    let mut r = ring.lock();
                    if let Err(e) = r.set_iowq_max_workers(rec.bounded, rec.unbounded) {
                        peregrine_io::note_advisory_err("iowq worker-cap tuning", &e);
                    }
                }
                *prev = Some(rec);
            }
        }
    }

    /// The most recent runtime telemetry snapshot: per-lane timings, the bubble
    /// tuner's bias, io_uring EWMA/SQ-full/worker-cap, and warm-cache hit and
    /// prefetch-accuracy rates. Refreshed once per forward.
    pub fn telemetry(&self) -> crate::telemetry::RuntimeTelemetry {
        self.last_telemetry.lock().clone()
    }

    /// The most recent io_uring worker cap the tuner asked for (or `None`
    /// before the first application). For `/metrics` scrapes and tests.
    pub fn last_iowq(&self) -> Option<crate::iotune::IowqCap> {
        *self.last_iowq.lock()
    }

    /// The governor-adjusted CPU-lane worker count for the next forward.
    pub fn effective_workers(&self) -> usize {
        self.effective_workers.load(std::sync::atomic::Ordering::Relaxed).max(1)
    }

    /// Rebuild the affinity snapshot from the co-activation tracker: fused
    /// pairs at `COLI_FUSE_THRESHOLD` (default 0.9 co-rate) and hyperedge
    /// components (union-find over pairs at half the threshold). Arc-swapped so
    /// in-flight forwards keep their borrowed snapshot.
    fn rebuild_affinity(&self) {
        let threshold: f32 = std::env::var("COLI_FUSE_THRESHOLD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&t: &f32| (0.0..=1.0).contains(&t))
            .unwrap_or(0.9);
        let co = self.coactivation.lock();
        let pairs = co.fused_pairs(threshold);
        let loose = co.fused_pairs(threshold * 0.5);
        drop(co);
        // Union-find per layer over the loose pairs → expert → component id.
        let mut groups: Vec<std::collections::HashMap<u32, u32>> = Vec::with_capacity(loose.len());
        for layer_pairs in &loose {
            let mut parent: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            fn find(parent: &mut std::collections::HashMap<u32, u32>, x: u32) -> u32 {
                let p = *parent.entry(x).or_insert(x);
                if p == x {
                    return x;
                }
                let root = find(parent, p);
                parent.insert(x, root);
                root
            }
            for &(a, b) in layer_pairs {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent.insert(ra.max(rb), ra.min(rb)); // smaller root wins → deterministic
                }
            }
            let keys: Vec<u32> = parent.keys().copied().collect();
            let mut map = std::collections::HashMap::new();
            for k in keys {
                let root = find(&mut parent, k);
                map.insert(k, root);
            }
            // singleton components carry no grouping signal — drop them
            let mut sizes: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &r in map.values() {
                *sizes.entry(r).or_insert(0) += 1;
            }
            map.retain(|_, r| sizes.get(r).copied().unwrap_or(0) >= 2);
            groups.push(map);
        }
        *self.affinity.lock() = Arc::new(crate::concurrent::AffinityHints { pairs, groups });
    }

    /// The current affinity snapshot (cheap Arc clone).
    fn affinity_snapshot(&self) -> Arc<crate::concurrent::AffinityHints> {
        self.affinity.lock().clone()
    }

    /// Normalized routing entropy over the recent history window: per sparse
    /// layer, the Shannon entropy of the expert-selection distribution across
    /// the K-deep frames, normalized by `ln(distinct)` and averaged over layers
    /// with data. `None` before any routing was recorded. 0 → the same experts
    /// every token; 1 → maximally dispersed routing.
    pub fn routing_entropy(&self) -> Option<f32> {
        let hist = self.route_hist.as_ref()?;
        let h = hist.lock();
        let mut sum = 0f32;
        let mut layers = 0u32;
        for l in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
            let mut counts: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
            let mut total = 0u32;
            for frame in h.frames(l) {
                for &e in frame {
                    if e >= 0 {
                        *counts.entry(e).or_insert(0) += 1;
                        total += 1;
                    }
                }
            }
            if total == 0 || counts.len() < 2 {
                continue;
            }
            let mut ent = 0f32;
            for &c in counts.values() {
                let p = c as f32 / total as f32;
                ent -= p * p.ln();
            }
            sum += ent / (counts.len() as f32).ln();
            layers += 1;
        }
        (layers > 0).then(|| sum / layers as f32)
    }

    /// EWMA of [`Self::routing_entropy`] (updated per forward when
    /// `COLI_ENTROPY_ADAPT=1`). For telemetry scrapes.
    pub fn routing_entropy_ewma(&self) -> f32 {
        *self.entropy_ewma.lock()
    }

    /// Idle-tick maintenance: recompress one cold warm-cache slot (background
    /// densification — see [`WarmCache::recompress_one_cold`]). Returns the
    /// bytes saved this call; `0` when there was nothing to do. The batch
    /// engine calls this while no requests are pending, so the pause is off
    /// every request's critical path. Gated on `COLI_CACHE_COMPRESS_IDLE=1`.
    /// Tokens between RSS-guard checks. Frequent enough to catch growth before
    /// the kernel does, rare enough that one small `/proc` read is free.
    /// colibrì checks on the same cadence.
    /// Correct the pre-load projection against **measured** footprint.
    ///
    /// The projection is an estimate, and colibrì's experience is that estimates
    /// drift: it recorded one at 74.4 GB against a real 115.6 GB and took three
    /// kernel kills. So this reads actual RSS every `RSS_GUARD_EVERY` tokens and,
    /// when the process is meaningfully over budget, shrinks the warm cache by
    /// the overshoot — *lowering the budget*, not just evicting, so the cache
    /// cannot refill to the old ceiling on the next token.
    ///
    /// Correctness-neutral by construction: an evicted slab is re-read from disk
    /// on its next hit, producing the same bytes. The budget is the only lever
    /// available — resident weights and the KV cache cannot be handed back — so
    /// this bounds growth rather than guaranteeing a ceiling.
    ///
    /// `COLI_RSS_GUARD_GB` sets the limit; unset uses the projected peak recorded
    /// at load. Zero disables it.
    fn rss_guard(&self) {
        let Some(cache) = self.ecache.as_ref() else {
            return; // resident mode: no cache to give back
        };
        let limit = self.rss_limit_bytes;
        if limit == 0 {
            return;
        }
        let rss = crate::ram::read_rss_bytes();
        // `parking_lot::Mutex` does not poison, so there is no error case here.
        let mut c = cache.lock();
        if let Some(new_budget) = crate::ram::rss_guard_decide(rss, limit, c.budget()) {
            let freed = c.shrink_budget(new_budget);
            eprintln!(
                "peregrine: [ram] RSS {:.1} GB over the {:.1} GB budget — warm cache lowered to                  {:.1} GB, freed {:.1} GB",
                rss as f64 / 1e9,
                limit as f64 / 1e9,
                new_budget as f64 / 1e9,
                freed as f64 / 1e9,
            );
        }
    }

    pub fn idle_maintenance(&self) -> usize {
        if !matches!(std::env::var("COLI_CACHE_COMPRESS_IDLE").as_deref(), Ok("1") | Ok("true")) {
            return 0;
        }
        match &self.ecache {
            Some(c) => c.lock().recompress_one_cold(),
            None => 0,
        }
    }

    /// The most recent per-lane wall-time snapshot (microseconds). Zeros before
    /// the first forward.
    pub fn last_lane_timings(&self) -> crate::lane::LaneTimings {
        *self.last_lane.lock()
    }

    /// The bubble tuner's **smoothed** per-lane times, as opposed to
    /// [`Self::last_lane_timings`]'s single most recent forward.
    ///
    /// Both are worth scraping and they answer different questions: the raw
    /// snapshot shows what the last token cost, the EWMA shows which lane is
    /// structurally dominating — which is the one the balancer acts on. The EWMA
    /// carries no `reduce_us`/`cpu_bytes` (the tuner does not smooth them), so
    /// those read 0 here by construction rather than by accident.
    pub fn lane_ewma(&self) -> crate::lane::LaneTimings {
        self.bubble.lock().ewma_snapshot()
    }

    /// Current published bubble bias — the pipeline lane the tuner thinks is
    /// dominating. `Bias::Balanced` before the tuner has enough samples.
    pub fn lane_bias(&self) -> crate::lane::Bias {
        self.bubble.lock().bias()
    }

    /// Build a fresh [`crate::lane::LaneBalancer`] from the currently-published
    /// bias — or `None` when balancing is off. Called once per forward; the
    /// balancer itself is a Copy of the tuner's state at that instant.
    fn build_balancer(&self) -> Option<crate::lane::LaneBalancer> {
        if !crate::lane::lane_balance_enabled() {
            return None;
        }
        let bias = self.lane_bias();
        if matches!(bias, crate::lane::Bias::Balanced) {
            return None; // no signal → no override
        }
        // Spill threshold: median heat + 1 (a per-forward `median` would be an
        // extra scan; a fixed 1 favors any nonzero heat, which is what
        // `TowardGpu` downgrade wants — the *coldest* residents move first).
        Some(crate::lane::LaneBalancer::new(bias, 1))
    }

    /// Save routing stats into this model's own checkpoint directory. Convenience
    /// wrapper for the common shutdown call site.
    pub fn save_route_stats_here(&self) -> Result<(), Error> {
        let dir = self.checkpoint_dir.clone();
        self.save_route_stats(&dir)
    }

    /// Serialize the current routing history and heat snapshot to
    /// `<dir>/route_stats.json`. Overwrites any existing file. Best-effort — a
    /// missing history or non-writable dir returns `Ok(())` without an error, so
    /// callers can invoke this from shutdown paths without special-casing.
    /// Restore `<dir>/kernel_tuning.json` into the GPU tier's autotuner.
    ///
    /// Best-effort and silent on every failure: this file only ever changes
    /// which of three equally-correct kernel instantiations runs, so a missing,
    /// truncated or foreign-GPU table costs an exploration round and nothing
    /// else. Surfacing it as an error would make a performance hint able to fail
    /// a model load.
    fn try_load_kernel_tuning(&self, dir: &std::path::Path) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Ok(bytes) = std::fs::read(dir.join("kernel_tuning.json")) else { return };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
        gpu.restore_tuning(&v);
    }

    pub fn save_route_stats(&self, dir: &std::path::Path) -> Result<(), Error> {
        if !route_stats_persist_enabled() {
            return Ok(());
        }
        let Some(hist) = &self.route_hist else {
            return Ok(());
        };
        let tag = config_tag(&self.cfg);
        let hist_json = hist.lock().to_json(&tag);
        let heat_json: serde_json::Value = match &self.heat {
            Some(h) => serde_json::json!(h.snapshot()),
            None => serde_json::Value::Null,
        };
        let coact_json = self.coactivation.lock().to_json();
        let learn_json: serde_json::Value = match self.learner.lock().as_ref() {
            Some(l) => l.to_json(),
            None => serde_json::Value::Null,
        };
        let doc = serde_json::json!({
            "tag": tag,
            "hist": hist_json,
            "heat": heat_json,
            "coact": coact_json,
            "learn": learn_json,
        });
        let bytes = serde_json::to_vec(&doc).map_err(|e| Error::Format(format!("serialize route stats: {e}")))?;
        // best-effort: a non-writable checkpoint dir is not an error — the model
        // continues to run with in-memory-only history.
        if let Err(e) = peregrine_core::write_atomic(&dir.join("route_stats.json"), &bytes) {
            peregrine_io::note_advisory_err("persist route_stats.json", &e);
        }
        // The WMMA tuning table is a *separate* file, not a field of the one
        // above: it describes this host's GPU, while `route_stats.json`
        // describes the workload. Copying a checkpoint directory between
        // machines should carry the routing history and leave the kernel timings
        // behind, and one file cannot do both.
        if let Some(tuning) = self.gpu.as_ref().and_then(|g| g.tuning_json()) {
            match serde_json::to_vec(&tuning) {
                Ok(tb) => {
                    if let Err(e) = peregrine_core::write_atomic(&dir.join("kernel_tuning.json"), &tb) {
                        peregrine_io::note_advisory_err("persist kernel_tuning.json", &e);
                    }
                }
                Err(e) => peregrine_io::note_advisory_err("serialize kernel_tuning.json", &e),
            }
        }
        Ok(())
    }

    /// Build a transition automaton by running `corpus` through the model token by
    /// token and accumulating each layer's consecutive routed-set transitions. Requires
    /// streaming mode (the routing history the accumulation reads). Tagged with this
    /// model's config fingerprint. Resets the KV cache first.
    pub fn build_automaton(&mut self, corpus: &[i32]) -> Result<TransitionTable, Error> {
        if self.route_hist.is_none() {
            return Err(Error::Format("build_automaton requires streaming mode (COLI_STREAM=1)".into()));
        }
        let n_layers = self.cfg.n_layers as usize;
        let first_dense = self.cfg.first_dense as usize;
        let mut table = TransitionTable::new(n_layers, config_tag(&self.cfg));
        self.reset();
        let mut prev: Option<Vec<Vec<i32>>> = None;
        for (i, &tok) in corpus.iter().enumerate() {
            self.forward_step(&[tok], i)?;
            let cur = self.route_snapshot(n_layers);
            if let Some(p) = &prev {
                for layer in first_dense..n_layers {
                    table.observe(layer, &p[layer], &cur[layer]);
                }
            }
            prev = Some(cur);
        }
        Ok(table)
    }

    /// One-pass "galactic" preprocessing: run `corpus` once and produce every
    /// offline artifact at once — the transition automaton, the macro-state
    /// table (temporal routing compression), and the raw routing trace (for the
    /// layout tools). Streaming mode only. Resets the KV cache first.
    pub fn build_artifacts(&mut self, corpus: &[i32]) -> Result<OfflineArtifacts, Error> {
        if self.route_hist.is_none() {
            return Err(Error::Format("build_artifacts requires streaming mode (COLI_STREAM=1)".into()));
        }
        let n_layers = self.cfg.n_layers as usize;
        let first_dense = self.cfg.first_dense as usize;
        let tag = config_tag(&self.cfg);
        let mut table = TransitionTable::new(n_layers, tag.clone());
        let mut macros = crate::predict::MacroTable::new(n_layers, tag);
        let mut trace: Vec<Vec<Vec<i32>>> = Vec::with_capacity(corpus.len());
        self.reset();
        let mut prev: Option<Vec<Vec<i32>>> = None;
        for (i, &tok) in corpus.iter().enumerate() {
            self.forward_step(&[tok], i)?;
            let cur = self.route_snapshot(n_layers);
            if let Some(p) = &prev {
                for layer in first_dense..n_layers {
                    table.observe(layer, &p[layer], &cur[layer]);
                    macros.observe(layer, &p[layer], &cur[layer]);
                }
            }
            trace.push(cur.clone());
            prev = Some(cur);
        }
        Ok((table, macros, trace))
    }

    /// Capture the raw per-forward routed sets for `corpus` (each entry is one forward's
    /// per-layer routed experts) — the trace `build-automaton` aggregates. Streaming
    /// mode only. Resets the KV cache first.
    pub fn dump_routes(&mut self, corpus: &[i32]) -> Result<Vec<Vec<Vec<i32>>>, Error> {
        if self.route_hist.is_none() {
            return Err(Error::Format("dump_routes requires streaming mode (COLI_STREAM=1)".into()));
        }
        let n_layers = self.cfg.n_layers as usize;
        let mut trace = Vec::with_capacity(corpus.len());
        self.reset();
        for (i, &tok) in corpus.iter().enumerate() {
            self.forward_step(&[tok], i)?;
            trace.push(self.route_snapshot(n_layers));
        }
        Ok(trace)
    }

    /// Run `dump_routes` and write the trace to `path` as JSON, returning the number
    /// of forwards captured. Keeps trace serialization inside this crate.
    pub fn dump_routes_to(&mut self, corpus: &[i32], path: &std::path::Path) -> Result<usize, Error> {
        let trace = self.dump_routes(corpus)?;
        let json = serde_json::to_vec(&trace).map_err(|e| Error::Format(format!("serialize trace: {e}")))?;
        peregrine_core::write_atomic(path, &json)?;
        Ok(trace.len())
    }

    /// Snapshot the current per-layer routed sets from the routing history (newest
    /// frame per layer). Empty vecs for layers with no history (dense / not yet routed).
    fn route_snapshot(&self, n_layers: usize) -> Vec<Vec<i32>> {
        match &self.route_hist {
            Some(h) => {
                let h = h.lock();
                (0..n_layers).map(|l| h.latest(l).cloned().unwrap_or_else(Vec::new)).collect()
            }
            None => vec![Vec::new(); n_layers],
        }
    }

    /// Replace the prefetch predictor (tests / advanced callers).
    pub fn set_predictor(&mut self, predictor: PredictSource) {
        self.predictor = predictor;
    }

    /// Apply `COLI_PREDICT_SOURCE` if it names a predictor this model can build.
    ///
    /// Load already picks the strongest source the *artifacts* support (automaton
    /// → macro → momentum), so this exists to force a **weaker** one — which is
    /// exactly what an A/B needs. `COLI_PREDICT_EVAL` scores the arms against the
    /// routing that actually happened; without a way to select the arm in
    /// production, that scoreboard could only ever grade a choice nobody could
    /// change.
    ///
    /// - `momentum` — recency-weighted vote, no offline artifact.
    /// - `phase-aware` — wraps the current source with a phase-shift boost.
    ///
    /// Anything else (including `automaton`/`macro`) is left alone: those need
    /// artifacts that may not exist, and silently degrading to momentum while
    /// reporting the requested name is worse than ignoring the request.
    /// Returns what it applied, for the startup report.
    pub fn apply_predictor_override(&mut self) -> Option<&'static str> {
        let var = std::env::var("COLI_PREDICT_SOURCE");
        match var.as_deref() {
            Ok("momentum") => {
                self.set_predictor(PredictSource::Momentum(crate::predict::Momentum::default()));
                Some("momentum")
            }
            Ok("phase-aware") => {
                let inner = std::mem::replace(
                    &mut self.predictor,
                    PredictSource::Momentum(crate::predict::Momentum::default()),
                );
                // `boost` is derived from the history depth, not picked: at the
                // `boost: 2` this shipped with, a newest-frame expert merely *tied*
                // one that had just dropped out (depth-4 momentum scores it 6), so
                // "trust recency on a phase shift" reduced to a tie broken by
                // ascending expert id. The unit tests passed throughout because they
                // build their own source with `boost: 100`. `phase_boost` outranks
                // the whole momentum scale by construction.
                self.set_predictor(PredictSource::PhaseAware {
                    inner: Box::new(inner),
                    threshold_bp: phase_threshold_bp(),
                    boost: crate::predict::phase_boost(route_hist_depth()),
                });
                Some("phase-aware")
            }
            _ => None,
        }
    }

    /// Whether the prefetch predictor is the transition automaton (introspection/tests).
    pub fn predictor_is_automaton(&self) -> bool {
        matches!(self.predictor, PredictSource::Automaton { .. })
    }

    /// Override multi-path tiering: fully stream the top `warm_paths` predicted
    /// experts per layer and page-cache-hint the next `hint_paths` (suppressed under
    /// O_DIRECT). Overrides the `COLI_PREFETCH_*_PATHS` env defaults.
    pub fn set_prefetch_policy(&mut self, warm_paths: usize, hint_paths: usize) {
        self.prefetch_policy = PrefetchPolicy { warm_paths, hint_paths };
    }

    /// Enable the adaptive prefetch-distance controller (bypasses `COLI_PREFETCH_TUNE`).
    /// It re-tunes the warm-tier breadth each forward within `[1, d_max]` from observed
    /// prefetch used/wasted rates, starting at `initial`.
    /// Hand the model the decode thread's LLC-miss counter.
    ///
    /// The binary opens it, not the model: `perf_event_open(2)` counts the thread
    /// that called it (`pid = 0`), and a `Model` is constructed on whichever
    /// thread loaded the checkpoint — which is not necessarily, and for the
    /// batched server definitely not, the thread that decodes. Opening it here
    /// would silently measure a thread that does no inference.
    ///
    /// Priming the baseline matters: `read()` is cumulative since open, so
    /// without this the first per-forward delta is the entire counter and would
    /// register as a miss spike that never happened.
    pub fn attach_perf_counter(&mut self, counter: peregrine_io::PerfCounter) {
        self.perf_llc_last = counter.read().unwrap_or(0);
        self.perf_llc = Some(counter);
    }

    /// Cumulative LLC misses on the decode thread since [`Self::attach_perf_counter`],
    /// or `None` when no counter is attached or the read failed.
    pub fn llc_misses(&self) -> Option<u64> {
        self.perf_llc.as_ref().and_then(|c| c.read())
    }

    /// This forward's LLC-miss delta, advancing the baseline. `None` when no
    /// counter is attached or the kernel read failed.
    fn llc_delta(&mut self) -> Option<u64> {
        let now = self.perf_llc.as_ref().and_then(|c| c.read())?;
        let last = self.perf_llc_last;
        self.perf_llc_last = now;
        // `saturating_sub` rather than a subtraction: a counter can be reset by
        // something outside this process, and a wrapped delta would read as a
        // colossal spike and slam the tuner to its ceiling.
        Some(now.saturating_sub(last))
    }

    pub fn enable_prefetch_tuner(&mut self, initial: usize, d_max: usize) {
        self.prefetch_tuner = Some(PrefetchTuner::new(initial, d_max));
    }

    /// Current adaptive prefetch distance, if the tuner is enabled (introspection/tests).
    pub fn prefetch_distance(&self) -> Option<usize> {
        self.prefetch_tuner.as_ref().map(|t| t.distance())
    }

    /// Recompute eviction protection from the current routing history (same path
    /// `forward_hidden` uses). Exposed for tests.
    pub fn protect_cache_from_history(&self) {
        self.update_cache_protection();
    }

    /// A resident expert's eviction-protection score (0 if unprotected / not resident).
    /// For tests/introspection.
    pub fn ecache_priority(&self, layer: usize, expert: usize) -> Option<u32> {
        self.ecache.as_ref().map(|c| c.lock().priority((layer as u32, expert as u32)))
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
            self.embed.dequant_row_into(tid, &mut x[s * d..s * d + d]);
        }

        // This forward logs into `route_hist` (see the `route_log` field below),
        // so the co-activation fold in `publish_lane_timings` has a fresh frame.
        self.route_hist_epoch.store(true, std::sync::atomic::Ordering::Relaxed);
        let lookahead = prefetch_lookahead();
        // Per-class breadth first (Copy; read before the disjoint destructure),
        // then the adaptive controller's cap wins over both when active.
        let policy = self.class_policy(); // already tuner-capped (see `class_policy`)
        // Run the stack in a block so the split borrows of `self` end before we
        // re-borrow `self` to enqueue prefetch.
        {
            // split disjoint fields so attention can borrow layers (imm) + kv (mut)
            let balancer = self.build_balancer();
            let heat_snapshot = balancer.as_ref().and_then(|_| self.heat.as_ref().map(|h| h.snapshot()));
            let eff_workers = self.effective_workers();
            let aff = self.affinity_snapshot();
            let Model {
                cfg, layers, kv, st, expert_index, fd_device_table, stream_experts, direct, io_reactors, ecache, route_hist, predictor, predict_eval, prefetch, gpu, heat, spill_log, lane_timings, layout_schedule, absorb, dsa, sweep, calib, ..
            } = self;
            let sweep: &SweepClock = sweep;
            let ctx = ForwardCtx {
                st,
                absorb: *absorb,
                dsa: *dsa,
                reactors: io_reactors,
                gpu: gpu.as_ref(),
                workers: eff_workers,
                cfg,
                stream_experts: *stream_experts,
                ecache: ecache.as_deref(),
                route_log: route_hist.as_ref(),
                calib: calib.as_ref().map(|(_, a)| a),
                route_log_multi: None,
                direct: *direct,
                heat: heat.as_ref(),
                spill: spill_log.as_ref(),
                timings: Some(lane_timings.as_ref()),
                balancer: balancer.as_ref(),
                heat_counts: heat_snapshot.as_deref(),
                layout_schedule: layout_schedule.as_deref(),
                affinity: Some(aff.as_ref()),
                expert_index: expert_index.as_ref(),
                fd_devices: fd_device_table.as_ref(),
            };
            // Layer look-ahead: a shared prefetch view over the same field borrows, so
            // each layer's next-token prefetch is emitted the moment that layer
            // finishes (its read then overlaps later layers' compute). `None` when
            // look-ahead is off or the prefetch lane/cache is absent — the bulk
            // enqueue below runs instead. Mutually exclusive, so no double-enqueue.
            let pfc = match (lookahead, prefetch.as_ref(), route_hist.as_ref(), ecache.as_ref()) {
                (true, Some(pool), Some(rh), Some(ec)) => Some(PrefetchCtx {
                    prefetch: pool.lane(0),
                    predictor,
                    hist: rh,
                    cache: ec,
                    gpu: gpu.as_ref(),
                    st,
                    cfg,
                    expert_index: expert_index.as_ref(),
                    warm_paths: policy.warm_paths,
                    hint_paths: policy.hint_paths,
                    direct: *direct,
                    sweep,
                }),
                _ => None,
            };
            // The router look-ahead is **decode-only**, and that is a measured
            // boundary rather than a simplification. A decode layer claims `k` cache
            // slots and leaves a real idle window — its attention — for six
            // speculative reads to land in. A prefill chunk claims the union over
            // every position in the chunk (hundreds of slots), so the speculative
            // records are the freshest unpinned entries in the cache and are exactly
            // what eviction takes first: WASTE built this hook on their chunk path
            // and measured the signature of a prefetch thrown away and re-fetched —
            // demand hit rate tripled, total bytes read rose 6.9 %, wall clock did
            // not move (their `LEARNED.md` §36). They removed it rather than
            // defaulting it off. There is also no window to fill there: a chunk
            // layer's readers are busy continuously, so a speculative read does not
            // move a read into idle time, it moves it in front of another read.
            //
            // Multi-row look-ahead here would mean prefill-chunk prefetch — the WASTE
            // negative. The **batched decode** multi-row path is `forward_rows_inner`
            // (which has `owner` separating per-sequence rows from a prefetch-chunk),
            // and that is the one the new multi-row look-ahead lives in. Keep this gate
            // the historical shape (single-row decode-only) so chunk-prefill stays
            // measured-neutral.
            let la_width = if router_lookahead() && s_n == 1 { router_lookahead_width() } else { 0 };
            // Built independently of `pfc`, not from it: the two are separate
            // features with separate knobs, and one asks the routing history while
            // the other asks the next layer's router. Deriving this from `pfc` would
            // make `COLI_PREFETCH_LOOKAHEAD=0` silently disable the router look-ahead
            // as well, which is not what that knob says it does.
            let la = match (la_width > 0, prefetch.as_ref(), ecache.as_ref()) {
                (true, Some(pool), Some(ec)) => {
                    Some(LookaheadCtx {
                        prefetch: pool.lane(0),
                        cache: ec,
                        gpu: gpu.as_ref(),
                        st,
                        cfg,
                        expert_index: None,
                        sweep,
                    })
                }
                _ => None,
            };
            // The scoreboard rides the same decode-only rule, and on the same
            // reasoning: a prefill chunk's "actual set" is a union over positions, so
            // recall against it would not be the number any of these predictors is
            // trying to hit.
            let eval = (s_n == 1).then_some(predict_eval.as_ref()).flatten();
            let layers: &[LayerW] = layers;
            for (li, l) in layers.iter().enumerate() {
                sweep.tick();
                forward_layer(l, li, &mut kv[li], &ctx, &mut x, s_n, pos_base)?;
                if let Some(pfc) = &pfc {
                    pfc.emit_layer(li);
                }
                // Emitted here, after this layer's own reads have been consumed and
                // before the next layer's attention, because that gap is the whole
                // resource being spent. `x` is this layer's output, which is the next
                // layer's input.
                if let Some(la) = &la {
                    la.emit(layers, li + 1, &x, la_width);
                }
                if let (Some(eval), Some(rh)) = (eval, route_hist.as_ref()) {
                    // Score first, then predict. `forward_layer` has just pushed this
                    // layer's authoritative set, so `latest(li)` is the answer to the
                    // prediction stashed for `li` one layer ago — while `latest(li+1)`
                    // is still the *previous token's* set there, which is exactly the
                    // baseline arm.
                    score_and_stash(eval, rh, predictor, layers, cfg, li, &x);
                }
            }
        }
        // When look-ahead is off, fall back to one bulk next-token enqueue after the
        // forward (main forward only).
        if !lookahead {
            self.enqueue_prefetch();
        }
        // Predictive eviction: protect the experts we expect to reuse next.
        if self.prefetch_protect {
            self.update_cache_protection();
        }
        // Feed the adaptive controller this forward's prefetch effectiveness so it can
        // re-tune the warm-tier breadth for the next forward. Read the counters first
        // (shared borrow), then update the tuner (disjoint mut borrow).
        if self.prefetch_tuner.is_some() {
            let obs = self.ecache.as_ref().map(|c| {
                let c = c.lock();
                (c.prefetch_used, c.prefetch_wasted)
            });
            if let (Some(t), Some((used, wasted))) = (self.prefetch_tuner.as_mut(), obs) {
                t.observe(used, wasted);
            }
        }
        // Hardware-counter feedback (`COLI_PERF_PREFETCH_FEEDBACK=1`): the
        // consumer `telemetry.rs` has documented since the counter landed —
        // "feed `PerfCounter::read` deltas into the prefetch tuner: rising
        // misses → widen prefetch distance". See `perf_prefetch_feedback` for
        // why this is a second opt-in and why the *direction* is a hypothesis
        // rather than a measurement.
        //
        // A ±10% dead band, because the raw per-forward delta is noisy: a
        // counter that follows one thread on a contended desktop moves several
        // percent between identical forwards, and nudging on every wobble makes
        // the distance a random walk rather than a controller.
        if perf_prefetch_feedback() {
            if let Some(delta) = self.llc_delta() {
                let d = delta as f32;
                let prev = self.perf_llc_ewma;
                self.perf_llc_ewma = if prev == 0.0 { d } else { 0.7 * prev + 0.3 * d };
                let step = llc_trend(prev, d);
                if let Some(t) = self.prefetch_tuner.as_mut() {
                    match step {
                        1 => t.nudge_up(),
                        -1 => t.nudge_down(),
                        _ => {}
                    }
                }
            }
        }
        // Entropy-adaptive breadth: dispersed routing widens the prefetch
        // distance (more candidates worth warming), repetitive routing narrows
        // it. Applied here because the tuner needs `&mut`.
        if entropy_adapt_enabled() {
            let e = *self.entropy_ewma.lock();
            if let Some(t) = self.prefetch_tuner.as_mut() {
                if e >= 0.85 {
                    t.nudge_up();
                } else if e <= 0.5 {
                    t.nudge_down();
                }
            }
        }
        // Learned prefetch distance (bandit / Q): apply the staged choice to the
        // tuner. Only meaningful with the tuner on; nudged toward the target one
        // step per forward so the change stays smooth.
        {
            let target = self.learned_prefetch.load(std::sync::atomic::Ordering::Relaxed);
            if target > 0 {
                if let Some(t) = self.prefetch_tuner.as_mut() {
                    match t.distance().cmp(&target) {
                        std::cmp::Ordering::Less => t.nudge_up(),
                        std::cmp::Ordering::Greater => t.nudge_down(),
                        std::cmp::Ordering::Equal => {}
                    }
                }
            }
        }
        // Snapshot this forward's per-lane wall time and fold it into the bubble
        // tuner. `publish_lane_timings` is idempotent — safe to call from every
        // main-forward path (and the batched one below).
        self.publish_lane_timings();
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

    /// **RLM recursive pass** — re-runs the last `K = COLI_RLM_LAYERS` transformer
    /// layers of the main stack on a refined `h`, returning refined logits
    /// `[s_n, vocab]`. Used by [`Self::generate`] and [`Self::generate_speculative`]
    /// when [`crate::rlm::RLMController::should_recurse`] indicates the first-pass
    /// logits are uncertain.
    ///
    /// The recursive pass **never appends to the main KV** — the cross-layer
    /// attention it performs reads the same cache a regular decode at `pos_base`
    /// would, but writes nothing to it. This matches the property used by the MTP
    /// draft path (`mtp_draft_with`), and keeps "one extra pass on this token" from
    /// silently inflating context length or breaking the truncate-rewind contract in
    /// speculative decode.
    ///
    /// Routing is **not logged** into `route_hist` (the `forward_ctx()` context sets
    /// `route_log: None`), for the same reason drafts must not overwrite the
    /// main-stream router history — recursive refinement at one token must not
    /// skew the prefetch predictor against future tokens. The 3-lane scheduler and
    /// warm cache still serve the recursive pass normally — on the second pass for a
    /// contested token, most routed experts are already warm in `ecache`, which is
    /// the cache-amortization win that makes RLM a net speedup rather than a slowdown
    /// in the disk-bound regime (see `README.md` "warm cache 3.58× on repeat forward").
    ///
    /// When `COLI_RLM` is unset, callers never invoke this (the controller's
    /// `should_recurse` returns `false`), so this path has zero effect on bit-identity
    /// gates. Off-by-default correctness is therefore structural, not a runtime
    /// branch in the hot path.
    pub(crate) fn forward_hidden_recursive(
        &self,
        h: &mut [f32],
        s_n: usize,
        pos_base: usize,
    ) -> Result<Vec<f32>, Error> {
        self.forward_hidden_recursive_with(h, s_n, pos_base, &|li| self.kv[li].clone_prefix(pos_base))
    }

    /// [`Self::forward_hidden_recursive`] over an **external** per-sequence KV
    /// (the serve engine's [`SeqKv`]) instead of the model-resident cache. Same
    /// exclusions, same throwaway-tail contract — the replay reads the cached
    /// causal prefix `[0, pos_base)` of `seq` and writes nothing back.
    pub(crate) fn forward_hidden_recursive_seq(
        &self,
        seq: &SeqKv,
        h: &mut [f32],
        s_n: usize,
        pos_base: usize,
    ) -> Result<Vec<f32>, Error> {
        self.forward_hidden_recursive_with(h, s_n, pos_base, &|li| seq.layers[li].clone_prefix(pos_base))
    }

    /// Shared body of the two entries above, parameterized on where a replay
    /// layer's KV prefix comes from (`kv_at` receives the **absolute** layer
    /// index and returns a throwaway clone at `pos_base`).
    fn forward_hidden_recursive_with(
        &self,
        h: &mut [f32],
        s_n: usize,
        pos_base: usize,
        kv_at: &dyn Fn(usize) -> LayerKv,
    ) -> Result<Vec<f32>, Error> {
        let n_layers = self.cfg.n_layers as usize;
        let k = crate::rlm::rlm_layers().min(n_layers);
        let start = n_layers - k;
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        // Fresh local KV per replay layer — **not** `LayerKv::new()`, but
        // `clone_prefix(pos_base)` of the real per-layer cache: `mla_attention`
        // always appends at `pos_base`, so the throwaway must see all `pos_base`
        // positions already populated. `clone_prefix` shares those rows by
        // `Arc` (zero-copy on the shared-prefix path) and gives the replay its
        // own private tail to append position `pos_base` into — the recursive
        // pass's attention reads the real cached past and writes nothing back,
        // exactly like `mtp_draft_with` on a fresh local KV. Bit-identical KV
        // state on the real cache because we never touch its tail.
        let mut kv_local: Vec<LayerKv> = (0..k).map(|i| kv_at(start + i)).collect();
        // Build a route-logging-off ForwardCtx (the prefill/draft shape), reusing
        // the resident scheduler, GPU lane and warm cache. Replays do not touch
        // the lane telemetry (timings = None), so bubble/IO tuners stay on the
        // main forward's signal. Scoped so the `&self` borrow ends before the
        // `lm_head` re-borrow (clippy: a `drop` here would be a `drop_non_drop`
        // noise — the lifetime, not the destructor, is what we want to bound).
        let lg = {
            let ctx = self.forward_ctx();
            for (i, l) in self.layers[start..].iter().enumerate() {
                forward_layer(l, start + i, &mut kv_local[i], &ctx, h, s_n, pos_base)?;
            }
            // Drop `kv_local` here is unnecessary — it owns no shared `&self`
            // borrow; the constraint is just `ctx` going out of scope.
            let xf = rmsnorm_rows(h, &self.final_norm, s_n, d, eps);
            self.lm_head.apply_vec(&xf, s_n)
        };
        Ok(lg)
    }

    /// RLM controller telemetry — recursive passes triggered this run, and the
    /// number of tokens that triggered at least one. `(0, 0)` unless `COLI_RLM=1`.
    /// Intended for the `/metrics` scrape path or the engine-shutdown summary,
    /// same role as `lookahead_issued()` for the router look-ahead.
    pub fn rlm_stats(&self) -> (u64, u64) {
        (self.rlm.passes_emitted(), self.rlm.tokens_recursed())
    }

    /// RLM refinement for an external-KV driver (the serve engine): refine one
    /// row's pre-final-norm hidden `h` and its `logits_row` in place, looping
    /// the same uncertainty policy and depth cap as the model-resident
    /// composition in [`Self::generate_speculative`]. Returns whether any pass
    /// ran; `Ok(false)` immediately — no copies, no KV clones — when
    /// `COLI_RLM` is unset, which is what keeps the serve path bit-identical
    /// with RLM off.
    ///
    /// The depth counter is **local** (the controller's per-token `depth` needs
    /// `&mut`, and the batched accept loop holds the model by `&`); the shared
    /// statistics go through [`crate::rlm::RLMController::note_pass`], so
    /// `/metrics` and the shutdown summary see external passes too.
    pub fn rlm_refine_external(
        &self,
        seq: &SeqKv,
        pos: usize,
        h: &mut [f32],
        logits_row: &mut [f32],
        temp: f32,
    ) -> Result<bool, Error> {
        if !crate::rlm::rlm_enabled() {
            return Ok(false);
        }
        let vocab = self.cfg.vocab as usize;
        let max_depth = crate::rlm::rlm_max_depth();
        let mut depth = 0usize;
        while depth < max_depth && crate::rlm::wants_recursion(&logits_row[..vocab], temp) {
            depth += 1;
            self.rlm.note_pass(depth == 1);
            // The replay returns final_norm + lm_head over the refined hidden —
            // exactly the recompute the model-resident composition does.
            let lg = self.forward_hidden_recursive_seq(seq, h, 1, pos)?;
            logits_row[..vocab].copy_from_slice(&lg[..vocab]);
        }
        Ok(depth > 0)
    }

    /// Build the per-forward compute context from the resident model state with
    /// prefetch/route-logging **disabled** — the shape the external-KV batched and
    /// prefill paths use (the B-way expert union is not a useful next-token
    /// predictor, so prefetch is gated off under batching).
    fn forward_ctx(&self) -> ForwardCtx<'_> {
        ForwardCtx {
            st: &self.st,
            absorb: self.absorb,
            dsa: self.dsa,
            reactors: &self.io_reactors,
            gpu: self.gpu.as_ref(),
            workers: self.effective_workers(),
            cfg: &self.cfg,
            stream_experts: self.stream_experts,
            ecache: self.ecache.as_deref(),
            route_log: None,
            calib: self.calib.as_ref().map(|(_, a)| a),
            route_log_multi: None,
            direct: self.direct,
            heat: self.heat.as_ref(),
            // No balancer in this context, so no spill verdict can occur.
            spill: None,
            timings: None,
            balancer: None,
            heat_counts: None,
            layout_schedule: self.layout_schedule.as_deref(),
            affinity: None,
            expert_index: self.expert_index.as_ref(),
            fd_devices: self.fd_device_table.as_ref(),
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
            self.embed.dequant_row_into(tid, &mut x[s * d..s * d + d]);
        }
        let ctx = self.forward_ctx();
        for (li, l) in self.layers.iter().enumerate() {
            // Advancing the sweep clock here matters even though this path emits no
            // speculation: a prefill's demand reads churn the cache exactly like a
            // decode step's, so a warm item queued just before a long prefill is
            // stale by the end of it — and should read as such.
            self.sweep.tick();
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
    /// concurrent sequences from a single scheduler thread. MTP speculation stays a
    /// B==1 path. When `histories` is `Some`, each sequence's own routed set (position
    /// `s` ↔ sequence `s`) is recorded into `histories[s]` for per-stream prefetch;
    /// `None` disables per-sequence route logging (bit-identical either way).
    pub fn forward_step_batched(
        &self,
        tokens: &[i32],
        seqs: &mut [&mut SeqKv],
        pos_of: &[usize],
        histories: Option<&[&Mutex<RouteHistory>]>,
    ) -> Result<Vec<f32>, Error> {
        let s_n = tokens.len();
        if seqs.len() != s_n || pos_of.len() != s_n {
            return Err(Error::Format(format!(
                "forward_step_batched: {s_n} tokens but {} seqs / {} positions",
                seqs.len(),
                pos_of.len()
            )));
        }
        let owner: Vec<usize> = (0..s_n).collect();
        self.forward_rows_batched(tokens, &owner, seqs, pos_of, histories)
    }

    /// Verify per-sequence speculative drafts in **one** forward.
    ///
    /// Sequence `s` contributes rows `[next_of[s], drafts[s]...]` starting at
    /// `pos_of[s]`, and every sequence's rows go into a single
    /// [`Self::forward_rows_batched`]. That is the whole point: B sequences
    /// speculating `γ` deep is `B·(1+γ)` rows sharing **one** set of expert
    /// reads, where B separate speculative decodes would stream B unions off
    /// disk. Speculation only pays on a disk-bound engine if the verify is
    /// shared, which is why this needed the row plumbing and not just a loop.
    ///
    /// **Greedy-identical by construction.** A draft is accepted only where it
    /// equals the model's own argmax at that position, so the emitted stream is
    /// exactly what one-token-at-a-time greedy decoding produces — speculation
    /// buys wall clock, never different output.
    /// `speculative_rows_emit_exactly_what_greedy_would` asserts it against a
    /// real greedy decode, with deliberately wrong drafts mixed in.
    ///
    /// Returns the per-sequence results and the **pre-final-norm hidden states**
    /// `[B, hidden]` at each sequence's accepted position — everything the next
    /// round needs, from one forward. `mtp_draft` takes `&self`, so those
    /// drafts can then run without serialising on a borrow.
    ///
    /// Each sequence's KV is rewound to its committed length before returning,
    /// so a rejected draft leaves no trace. At temperature > 0 this is
    /// distribution-preserving rather than sequence-identical; that path is
    /// `speculative_sample`, not this.
    pub fn verify_drafts_batched(
        &self,
        next_of: &[i32],
        drafts: &[Vec<i32>],
        seqs: &mut [&mut SeqKv],
        pos_of: &[usize],
    ) -> Result<(Vec<Verified>, Vec<f32>), Error> {
        let b = next_of.len();
        if drafts.len() != b || seqs.len() != b || pos_of.len() != b {
            return Err(Error::Format(format!(
                "verify_drafts_batched: {b} tokens but {} drafts / {} seqs / {} positions",
                drafts.len(),
                seqs.len(),
                pos_of.len()
            )));
        }
        // Row layout: sequence `s` owns `1 + drafts[s].len()` consecutive rows.
        let mut tokens: Vec<i32> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        let mut rows_pos: Vec<usize> = Vec::new();
        let mut first_row: Vec<usize> = Vec::with_capacity(b);
        for s in 0..b {
            first_row.push(tokens.len());
            tokens.push(next_of[s]);
            owner.push(s);
            rows_pos.push(pos_of[s]);
            for (j, &t) in drafts[s].iter().enumerate() {
                tokens.push(t);
                owner.push(s);
                rows_pos.push(pos_of[s] + 1 + j);
            }
        }
        // The hidden form, because the next round's draft starts from the
        // hidden at the accepted position — and this forward already computed
        // it. Recovering it with a second pass would hand back the speedup.
        let (logits, hidden) = self.forward_rows_batched_hidden(&tokens, &owner, seqs, &rows_pos, None)?;
        let d = self.cfg.hidden as usize;
        let mut hlast = Vec::with_capacity(b * d);

        let vocab = self.cfg.vocab as usize;
        let mut out = Vec::with_capacity(b);
        for s in 0..b {
            let base = first_row[s];
            let g = drafts[s].len();
            // `next_of[s]` is already confirmed — it was the model's argmax
            // last round. Everything after it is judged by `accept_run`, the
            // one definition of the greedy-identity rule.
            let rows = logits.get(base * vocab..(base + g + 1) * vocab).unwrap_or(&[]);
            let (k, next) = accept_run(rows, vocab, &drafts[s]);
            let mut confirmed = vec![next_of[s]];
            confirmed.extend_from_slice(&drafts[s][..k.min(g)]);
            // Rewind the speculated tail. Committed is `next_of` plus `k`
            // accepted drafts; anything beyond it was never real.
            seqs[s].truncate(pos_of[s] + 1 + k);
            // Row `base + k` is the one whose logits produced `next`, so its
            // hidden is what the next draft continues from.
            let h = (base + k) * d;
            hlast.extend_from_slice(hidden.get(h..h + d).unwrap_or(&[]));
            out.push(Verified { tokens: confirmed, next, accepted: k });
        }
        Ok((out, hlast))
    }

    /// [`Self::forward_rows_batched`], also returning the **pre-final-norm**
    /// hidden states `[s_n, hidden]`.
    ///
    /// The MTP head drafts from that hidden, not from logits, so a batched
    /// speculative loop cannot be built on the logits-only form without
    /// re-running the stack to recover what this forward already computed.
    /// Pre-final-norm specifically: `mtp_draft` applies `final_norm` itself on
    /// its first step, so handing it a normalised hidden would norm it twice
    /// and quietly degrade every draft.
    pub fn forward_rows_batched_hidden(
        &self,
        tokens: &[i32],
        owner: &[usize],
        seqs: &mut [&mut SeqKv],
        pos_of: &[usize],
        histories: Option<&[&Mutex<RouteHistory>]>,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        self.forward_rows_inner(tokens, owner, seqs, pos_of, histories)
    }

    /// One batched forward over arbitrary **(token, sequence) rows**: `owner[r]`
    /// names which of `seqs` row `r` belongs to, so a sequence may contribute
    /// several rows.
    ///
    /// [`Self::forward_step_batched`] is the one-row-per-sequence case.
    /// Several rows on one owner is what a **prefill chunk** is — consecutive
    /// positions on a single cache — and what a **speculative draft** is: `γ+1`
    /// tokens on one sequence. Both were blocked on a signature that assumed one
    /// token per sequence, which is why they are one change and not two.
    ///
    /// The MoE lane needs nothing: it is already row-batch-union'd over `s_n`
    /// rows regardless of which sequence each belongs to, so fusing a prefill
    /// chunk into a decode batch makes them share one set of expert reads
    /// instead of streaming two disjoint unions off disk.
    ///
    /// `histories`, when given, is **per row**, not per sequence — a chunk's
    /// rows each record their own routed set, exactly as sequential prefill
    /// does. Rows of one owner must be in ascending position order with no gaps;
    /// the cache reports a violation rather than absorbing it.
    pub fn forward_rows_batched(
        &self,
        tokens: &[i32],
        owner: &[usize],
        seqs: &mut [&mut SeqKv],
        pos_of: &[usize],
        histories: Option<&[&Mutex<RouteHistory>]>,
    ) -> Result<Vec<f32>, Error> {
        Ok(self.forward_rows_inner(tokens, owner, seqs, pos_of, histories)?.0)
    }

    /// Shared body of the two forms above: `(logits, pre-final-norm hidden)`.
    fn forward_rows_inner(
        &self,
        tokens: &[i32],
        owner: &[usize],
        seqs: &mut [&mut SeqKv],
        pos_of: &[usize],
        histories: Option<&[&Mutex<RouteHistory>]>,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let s_n = tokens.len();
        if owner.len() != s_n || pos_of.len() != s_n {
            return Err(Error::Format(format!(
                "forward_rows_batched: {s_n} tokens but {} owners / {} positions",
                owner.len(),
                pos_of.len()
            )));
        }
        if let Some(&bad) = owner.iter().find(|&&o| o >= seqs.len()) {
            return Err(Error::Format(format!(
                "forward_rows_batched: owner {bad} is out of range for {} sequences",
                seqs.len()
            )));
        }
        if let Some(h) = histories {
            if h.len() != s_n {
                return Err(Error::Format(format!("forward_rows_batched: {s_n} rows but {} histories", h.len())));
            }
        }
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            self.embed.dequant_row_into(tid, &mut x[s * d..s * d + d]);
        }
        // Built inline (not via `forward_ctx`) so the per-sequence history borrow and
        // the model borrows share one inferred lifetime.
        let balancer = self.build_balancer();
        let heat_snapshot = balancer.as_ref().and_then(|_| self.heat.as_ref().map(|h| h.snapshot()));
        let aff = self.affinity_snapshot();
        let ctx = ForwardCtx {
            st: &self.st,
            absorb: self.absorb,
            dsa: self.dsa,
            reactors: &self.io_reactors,
            gpu: self.gpu.as_ref(),
            workers: self.effective_workers(),
            cfg: &self.cfg,
            stream_experts: self.stream_experts,
            ecache: self.ecache.as_deref(),
            route_log: None,
            calib: self.calib.as_ref().map(|(_, a)| a),
            route_log_multi: histories,
            direct: self.direct,
            heat: self.heat.as_ref(),
            spill: self.spill_log.as_ref(),
            timings: Some(self.lane_timings.as_ref()),
            balancer: balancer.as_ref(),
            heat_counts: heat_snapshot.as_deref(),
            layout_schedule: self.layout_schedule.as_deref(),
            affinity: Some(aff.as_ref()),
            expert_index: self.expert_index.as_ref(),
            fd_devices: self.fd_device_table.as_ref(),
        };
        // Router look-ahead, on the same decode-only rule as `forward_hidden`. B == 1
        // *is* a decode step — the serving engine reaching this path with one live
        // sequence is the ordinary single-stream shape — so it gets the window. B > 1
        // historically did not. The multi-row enablement here fires only for a
        // **true batched decode** step — one row per sequence, no multi-row verify
        // batch (which is a server-side chunk-by-another-name). Detected by
        // `owner.len() == distinct owners`, which is exactly the relationship a
        // decode step has and a verify batch ($1 + \gamma$ rows per sequence) does
        // not. Gated on [`router_lookahead_batch`] so the recall / precision
        // trade-off is measurable (`COLI_PREDICT_EVAL=1`) and a default can be set
        // from numbers rather than conjecture.
        let batched_decode = s_n > 0 && {
            let mut seen = std::collections::HashSet::with_capacity(seqs.len());
            owner.iter().all(|&o| seen.insert(o)) && seen.len() == s_n
        };
        let la_width = if router_lookahead() && (s_n == 1 || (batched_decode && router_lookahead_batch())) {
            router_lookahead_width()
        } else {
            0
        };
        let la = (la_width > 0).then(|| self.lookahead_ctx()).flatten();
        let layers: &[LayerW] = &self.layers;
        for (li, l) in layers.iter().enumerate() {
            self.sweep.tick();
            let mut caches: Vec<&mut LayerKv> = seqs.iter_mut().map(|sk| &mut sk.layers[li]).collect();
            forward_layer_batched(l, li, &mut caches, RowLayout { pos_of, owner }, &ctx, &mut x)?;
            if let Some(la) = &la {
                la.emit(layers, li + 1, &x, la_width);
            }
        }
        self.publish_lane_timings();
        let xf = rmsnorm_rows(&x, &self.final_norm, s_n, d, eps);
        Ok((self.lm_head.apply_vec(&xf, s_n), x))
    }

    /// Re-select the GPU tier's resident experts as the current hottest set (by
    /// accumulated routing frequency), migrating cooled experts out of VRAM and hot
    /// ones in. A no-op without a GPU tier (or the `cuda` feature). Call between
    /// forwards (`&mut self`); the batch engine invokes it periodically so residency
    /// adapts to the workload without a rewrite.
    pub fn reheat(&mut self) -> Result<(), Error> {
        // Frequency, recency and clock read together: `COLI_GPU_TIER_SWAP=lfru`
        // scores the first two against the third, and three separate reads would
        // let a generation age its stamps against a clock from another forward.
        let Some((mut counts, last, clock)) = self.heat.as_ref().map(|h| h.snapshot_all()) else {
            return Ok(());
        };
        // Drain the deferred-spill log (`COLI_GPU_SPILL`) into this generation's
        // snapshot before the ranking: every mid-forward `GpuSpill` verdict since
        // the last reheat bumps its expert, so what the balancer kept asking for
        // finally outranks what it never did. The uploads still go through the
        // ranking's own budgeted path (`admit_uploads`, `COLI_PCIE_BUDGET_MB`) —
        // this changes the order of candidates, not the spend limit.
        if let Some(log) = self.spill_log.as_ref() {
            let drained = std::mem::take(&mut *log.lock());
            merge_spills(&mut counts, &drained, self.cfg.n_experts as usize);
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.reheat(&self.st, &self.cfg, &crate::gpu::HeatView { counts: &counts, last: &last, clock })?;
        }
        // Runtime expert replication: warm the top-K hottest resident experts
        // into the CPU warm cache too, so a bias shift toward CPU never pays
        // the disk-read tax on a hot expert. Gated on `COLI_REPLICATE_K`.
        self.enqueue_expert_replicas(Self::replica_k());
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
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;

        // Inline forward+head without `forward_step` so we keep the pre-final-norm
        // hidden (`h_last`) in hand for RLM recursion. `forward_step` stays
        // unchanged — same algebra, bit-identical.
        let mut x_all = self.forward_hidden(prompt, 0)?;
        let xf = rmsnorm_rows(&x_all, &self.final_norm, prompt.len(), d, eps);
        let logits = self.lm_head.apply_vec(&xf, prompt.len());
        let mut h_last = x_all[(prompt.len() - 1) * d..prompt.len() * d].to_vec();
        let mut lg = logits[(prompt.len() - 1) * vocab..prompt.len() * vocab].to_vec();
        // RLM refinement loop: each pass refines `h_last` and recomputes the head.
        // `should_recurse` returns `false` for the whole call site when `COLI_RLM`
        // is unset, so the block below is the bit-identical pass-0 logits sample.
        while self.rlm.should_recurse(&lg, sampler.temp) {
            self.forward_hidden_recursive(&mut h_last, 1, prompt.len() - 1)?;
            let xf2 = rmsnorm_rows(&h_last, &self.final_norm, 1, d, eps);
            lg = self.lm_head.apply_vec(&xf2, 1);
        }
        let mut next = sampler.pick(&lg, -1) as i32;
        let mut out = vec![next];
        // Stop ids end generation here as they do on the speculative path. They
        // used to be honored only by `generate_speculative`, so the same model
        // and prompt returned different-length sequences depending on whether a
        // draft head was in play.
        if self.cfg.stop_ids.contains(&next) {
            return Ok(out);
        }
        for step in 1..n_new {
            let pos = prompt.len() + step - 1; // first decode attends at prompt.len()
            // RLM: reset per-token recursion at the start of each step.
            self.rlm.reset();
            // Forward to last-layer hidden with the single next token.
            x_all = self.forward_hidden(&[next], pos)?;
            h_last = x_all[..d].to_vec();
            let xf = rmsnorm_rows(&x_all, &self.final_norm, 1, d, eps);
            lg = self.lm_head.apply_vec(&xf, 1);
            // RLM recursion: refine `h_last` and recompute logits while the
            // controller says to. No-op (structurally — see `rlm.rs:114`) when off.
            while self.rlm.should_recurse(&lg, sampler.temp) {
                self.forward_hidden_recursive(&mut h_last, 1, pos)?;
                let xf2 = rmsnorm_rows(&h_last, &self.final_norm, 1, d, eps);
                lg = self.lm_head.apply_vec(&xf2, 1);
            }
            next = sampler.pick(&lg, -1) as i32;
            out.push(next);
            if step.is_multiple_of(RSS_GUARD_EVERY) {
                self.rss_guard();
            }
            if self.cfg.stop_ids.contains(&next) {
                break;
            }
        }
        Ok(out)
    }

    /// Fraction of teacher-forcing positions where two runs predict a different
    /// token — `0.0` means the runs agree everywhere. `None` on a length mismatch
    /// or an empty comparison, so "no data" cannot read as "no change".
    ///
    /// The repo's quality gates are all bit-identity anchors
    /// (`docs/testing-and-quality.md`), which is right for lossless work but leaves
    /// nothing to gate an *approximation* with: a lossy change fails an
    /// `assert_eq!` by construction, so there is no way to say how much it cost.
    /// This is that missing gate in its simplest honest form — turn the
    /// `assert_eq!` in `peregrine-tools/tests/apply_layout.rs` into a bounded flip
    /// rate and any future approximation becomes measurable.
    ///
    /// Top-1 agreement only. [`Self::teacher_forcing`] returns argmax ids, so a
    /// distributional metric (NLL, KL) needs per-position logit capture first —
    /// deliberately not added here, since nothing consumes it yet.
    pub fn prediction_flip_rate(a: &[i32], b: &[i32]) -> Option<f64> {
        if a.len() != b.len() || a.is_empty() {
            return None;
        }
        let flips = a.iter().zip(b).filter(|(x, y)| x != y).count();
        Some(flips as f64 / a.len() as f64)
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
    /// Takes `&self`, not `&mut self`: the body only reads the model and builds
    /// its own local `LayerKv`, so the `&mut` was incidental to the destructure.
    /// That matters — the batching thread holds `&Model` while several
    /// sequences draft, and a `&mut` here would have serialised them behind the
    /// one borrow.
    /// `conf_floor > 0` stops the draft early when the MTP head's top-token
    /// probability drops under it (`0.0` = draft the full depth, the historical
    /// behavior). Depth-only: acceptance is the caller's `accept_run`, so the
    /// floor can never change an emitted token — it trades draft depth for
    /// fewer rejected verify rows, each of which streams expert bytes.
    pub fn mtp_draft(&self, next_tok: i32, g_draft: usize, hlast: &[f32], conf_floor: f32) -> Result<Vec<i32>, Error> {
        self.mtp_draft_with(next_tok, g_draft, hlast, conf_floor, |lo| crate::sample::argmax(lo) as i32)
    }

    /// Draft `g_draft` tokens **sampled from `sampler`'s own distribution**,
    /// returning each draft alongside the distribution `q` it was drawn from.
    ///
    /// The sampled twin of [`Self::mtp_draft`], and the half of `COLI_DRAFT_SAMPLED`
    /// that makes the other half correct. [`crate::speculative_sample`] proves
    /// its output is distributed exactly as the target `p` **given that the
    /// draft was drawn from the `q` it is handed** — so a draft picked by argmax
    /// and described by a softmax would break the guarantee while looking
    /// entirely reasonable. `pick_with_distribution` draws and describes in one
    /// call precisely so the two cannot come apart.
    ///
    /// **Cost, stated because it is not obvious**: `q` is dense over the
    /// vocabulary, so a depth-`g` draft holds `g * vocab` floats per sequence
    /// between ticks — ~2.4 MB per sequence at GLM-5.2's vocab and `g = 4`. The
    /// nucleus zeroes most of it; a sparse form would trade that for a second
    /// representation of the same distribution, and getting *those* out of sync
    /// is the failure this function exists to prevent.
    pub fn mtp_draft_sampled(
        &self,
        next_tok: i32,
        g_draft: usize,
        hlast: &[f32],
        conf_floor: f32,
        sampler: &mut Sampler,
    ) -> Result<(Vec<i32>, Vec<Vec<f32>>), Error> {
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(g_draft);
        let drafts = self.mtp_draft_with(next_tok, g_draft, hlast, conf_floor, |lo| {
            let (t, q) = sampler.pick_with_distribution(lo);
            qs.push(q);
            t as i32
        })?;
        Ok((drafts, qs))
    }

    /// The shared body of [`Self::mtp_draft`] and [`Self::mtp_draft_sampled`],
    /// parameterized only by how a draft token is chosen from its logits.
    ///
    /// One body, because the *rest* of the draft — the MTP layer, the local KV,
    /// the hidden that feeds the next step — must be identical between the two.
    /// Two copies would let the greedy and sampled paths drift, and the drift
    /// would appear as an acceptance rate quietly falling rather than as a test
    /// failing.
    fn mtp_draft_with(
        &self,
        next_tok: i32,
        g_draft: usize,
        hlast: &[f32],
        conf_floor: f32,
        mut pick: impl FnMut(&[f32]) -> i32,
    ) -> Result<Vec<i32>, Error> {
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let n_layers = self.cfg.n_layers as usize;
        let (kvl, qkr) = (self.cfg.kv_lora as usize, self.cfg.qk_rope as usize);

        let Model { mtp, st, embed, lm_head, final_norm, io_reactors, workers, ecache, direct, gpu, stream_experts, cfg, absorb, dsa, .. } =
            self;
        let mtp = mtp.as_ref().ok_or_else(|| Error::Format("mtp_draft without an MTP head".into()))?;
        let ctx = ForwardCtx {
            st,
            absorb: *absorb,
            dsa: *dsa,
            reactors: io_reactors,
            gpu: gpu.as_ref(),
            workers: *workers,
            cfg,
            stream_experts: *stream_experts,
            ecache: ecache.as_deref(),
            route_log: None, // drafts must not overwrite the main-stream prediction
            calib: None, // drafts replay accumulated positions — counting them double-weights
            route_log_multi: None,
            direct: *direct,
            heat: None, // speculative drafts must not skew residency heat
            spill: None, // drafts must not queue uploads for a speculative future
            timings: None, // drafts must not skew the main-stream lane balance
            balancer: None, // drafts run under the plain static residency policy
            heat_counts: None,
            layout_schedule: None, // drafts benefit less from disk-order tuning
            affinity: None,
            expert_index: None,
            fd_devices: None, // drafts keep the blind claim cursor, like the rest of their tuning
        };
        let mut kv = LayerKv::new(kvl, qkr);
        let mut h = hlast.to_vec(); // pre-final-norm hidden
        let mut tok = next_tok;
        let mut draft = Vec::with_capacity(g_draft);
        for g in 0..g_draft {
            // norm(embed(tok))
            let tid = (tok.max(0) as usize).min(vocab.saturating_sub(1));
            let mut e = vec![0f32; d];
            let mut erow = vec![0f32; d];
            embed.dequant_row_into(tid, &mut erow);
            rmsnorm(&mut e, &erow, &mtp.enorm, eps);
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
            // The confidence gate (ds4's DSpark idea): a step whose top token
            // holds less than `conf_floor` of the distribution ends the draft
            // *before* `pick` — the low-confidence token itself is excluded,
            // since it would be the likeliest wasted verify row. Breaking
            // before `pick` also keeps the sampled path's `drafts`/`qs`
            // aligned by construction.
            if conf_floor > 0.0 && crate::sample::top_prob(&logit) < conf_floor {
                break;
            }
            let t2 = pick(&logit);
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
            // No confidence floor here: the serve engine resolves
            // `COLI_SPEC_CONF`; this single-sequence path keeps the historical
            // fixed depth so its output-identity contract stays trivially true.
            let draft = if g_want > 0 { self.mtp_draft(next, g_want, &hlast, 0.0)? } else { Vec::new() };
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
            // A stop token can land anywhere in the accepted run, not just at
            // the end of a round — testing only `out.last()` let generation
            // continue past one and emit it mid-stream.
            let mut stopped = self.cfg.stop_ids.contains(&next);
            // accept drafts while they match the model's greedy prediction
            let mut k = 0usize;
            while k < g && out.len() < n_new && !stopped {
                let pred = crate::sample::argmax(&logits_b[k * vocab..(k + 1) * vocab]) as i32;
                if pred == draft[k] {
                    out.push(draft[k]);
                    stopped = self.cfg.stop_ids.contains(&draft[k]);
                    k += 1;
                } else {
                    break;
                }
            }
            // the model's prediction at position k is the next token to process.
            // (Recomputed after the RLM refinement below if the controller
            // requested extra passes; the logits/route history is held to the
            // same exclusion `mtp_draft_with` enforces on drafts — `route_log`
            // is `None` via `forward_ctx`.)
            hlast = xb[k * d..(k + 1) * d].to_vec();
            // MTP + RLM composition — only the post-acceptance contested position.
            // We just emitted positions `[next, draft[..k]]`. `next` for the next
            // round is set to `logits_b[k]`'s argmax. If that footing was
            // uncertain (greedy top-2 margin < `COLI_RLM_MARGIN`), refine
            // `hlast = xb[k*d..]` with a recursive pass and recompute `next`.
            //
            // Self-consistent bounds:
            // - we never resume from a rejected draft token (it was already
            //   wrong for a reason; one more pass on a wrong-but-now-rejected
            //   position is the born-corrected case)
            // - we never recurse on row 0 — `next` was already committed as
            //   argmax earlier in this same block (`out.push(next)` above)
            //   and is treated as confirmed
            //
            // Bit-identical to plain `generate_speculative` when `COLI_RLM`
            // unset: `should_recurse` returns `false` for the whole loop.
            self.rlm.reset();
            let mut lb_k = logits_b[k * vocab..(k + 1) * vocab].to_vec();
            while self.rlm.should_recurse(&lb_k, 0.0) {
                self.forward_hidden_recursive(&mut hlast, 1, pos + k)?;
                let xf2 = rmsnorm_rows(&hlast, &self.final_norm, 1, d, eps);
                let lg2 = self.lm_head.apply_vec(&xf2, 1);
                lb_k = lg2[..vocab].to_vec();
            }
            next = crate::sample::argmax(&lb_k) as i32;
            // committed this round: `next` (already emitted) + k accepted drafts
            let committed = 1 + k;
            self.truncate_kv(pos + committed);
            pos += committed;
            if stopped {
                break;
            }
        }
        out.truncate(n_new);
        Ok(out)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Stop and join every prefetch lane before `st` (its shard fds) is dropped.
        if let Some(mut p) = self.prefetch.take() {
            p.stop();
        }
        // Persist routing history + heat so the next process starts warm on the
        // last session's routing patterns. Best-effort and correctness-neutral —
        // a failure is reported (COLI_DEBUG) and the model keeps in-memory history.
        if let Err(e) = self.save_route_stats_here() {
            peregrine_io::note_advisory_err("persist route stats on drop", &e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::argmax;
    use crate::testkit::build_tiny_model;
    use std::path::PathBuf;

    #[test]
    fn merge_spills_bumps_by_multiplicity_and_ignores_out_of_range() {
        // 2 layers × 4 experts. Expert (1,2) spilled three times, (0,0) once;
        // a stale pair past the table must be ignored, not panic or wrap.
        let mut counts = vec![10u32, 0, 0, 0, 0, 0, 5, 0];
        merge_spills(&mut counts, &[(1, 2), (0, 0), (1, 2), (1, 2), (7, 3)], 4);
        assert_eq!(counts, vec![11, 0, 0, 0, 0, 0, 8, 0]);
        // Saturation, not overflow, at the ceiling.
        let mut hot = vec![u32::MAX];
        merge_spills(&mut hot, &[(0, 0)], 1);
        assert_eq!(hot, vec![u32::MAX]);
        // Empty drain is a no-op.
        let mut same = vec![3u32, 4];
        merge_spills(&mut same, &[], 2);
        assert_eq!(same, vec![3, 4]);
    }

    /// The protect default has to flip at one token's working set, because the
    /// mechanism measured +193 hits below that line and −381 above it. Pinning
    /// both directions, since a default that is merely "on" was the bug.
    #[test]
    fn protect_default_follows_the_working_set_threshold() {
        // These are the two measured arms: 4.29 GB and 12.88 GB against a pass of
        // ~11.8 GB.
        let pass = 11_800_000_000u64;
        assert!(prefetch_protect_default(4_290_000_000, pass), "below a pass: protect is the only thing off zero");
        assert!(!prefetch_protect_default(12_880_000_000, pass), "above a pass: protect costs 40% of the hits");
        // Exactly at the threshold the cache can hold a pass, so recency suffices.
        assert!(!prefetch_protect_default(pass as usize, pass));
        // No working-set figure (resident mode, unindexed container) keeps the
        // historical always-on behaviour rather than guessing.
        assert!(prefetch_protect_default(4_290_000_000, 0));
    }

    /// The prefetch lane must switch itself off at the yield that was measured to
    /// cost +41 % wall time.
    #[test]
    fn prefetch_guard_trips_on_the_measured_yield() {
        // Under the sample floor, always keep speculating — an early window is
        // not evidence.
        assert!(prefetch_pays(100, 0));
        // The measured case: 4034 speculative reads bought 3 hits.
        assert!(!prefetch_pays(4034, 3), "0.3% yield must stop the lane");
        // A lane that is actually earning stays on.
        assert!(prefetch_pays(1000, 25), "2.5% yield clears the 2% floor");
        // And the boundary is inclusive, so a lane sitting exactly at the floor
        // is not killed by rounding.
        assert!(prefetch_pays(1000, 20));
        assert!(!prefetch_pays(1000, 19));
    }

    fn tmp_model_dir(tag: &str) -> Result<PathBuf, peregrine_core::Error> {
        let d = std::env::temp_dir().join(format!("peregrine_model_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        build_tiny_model(&d)?;
        Ok(d)
    }

    /// `accept_run_sampled` must emit the request's own distribution.
    ///
    /// Model-free on purpose: this is the *rule*, and the rule is what has to be
    /// right. The engine's job is only to hand it `q` from the same draw that
    /// produced the draft, which `mtp_draft_sampled` guarantees structurally.
    ///
    /// Measured on the **first emitted token of the run**, because that is the
    /// token the accept decision produces: `draft[0]` when accepted, the
    /// residual draw when not (`batch.rs` builds `run = draft[..k] ++ next`).
    /// Every verify row carries the same target logits, so the reference
    /// distribution is exactly `Sampler::distribution` over them.
    #[test]
    fn sampled_speculation_emits_the_requests_own_distribution() {
        const VOCAB: usize = 6;
        const G: usize = 3;
        const N: usize = 40_000;
        // Target and proposal are deliberately different distributions — a
        // proposal that already matched the target would make this test pass
        // for a rule that ignored `q` entirely.
        let target = [2.0f32, 0.5, -1.0, 1.25, 0.0, -0.5];
        let proposal = [-1.0f32, 1.5, 2.0, -0.5, 0.75, 0.25];

        let mut reference = Sampler::new(0.9, 1.0, 12345);
        let p_ref: Vec<f32> = reference.distribution(&target).to_vec();
        let q_ref: Vec<f32> = Sampler::new(0.9, 1.0, 1).distribution(&proposal).to_vec();
        let tv: f64 = p_ref.iter().zip(&q_ref).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / 2.0;
        assert!(tv > 0.3, "proposal must actually differ from target, TV = {tv:.3}");

        // `rows` repeats the target: row k judges draft k, and a bonus draw past
        // the accepted run comes from the same distribution, so the emitted
        // first token is p-distributed whichever branch it took.
        let rows: Vec<f32> = (0..G + 1).flat_map(|_| target.iter().copied()).collect();

        let mut sampler = Sampler::new(0.9, 1.0, 0xC0FFEE);
        let mut drafter = Sampler::new(0.9, 1.0, 0xDECAF);
        let mut hist = [0u32; VOCAB];
        let mut accepted_total = 0usize;
        for _ in 0..N {
            let mut drafts = Vec::with_capacity(G);
            let mut qs = Vec::with_capacity(G);
            for _ in 0..G {
                let (t, q) = drafter.pick_with_distribution(&proposal);
                drafts.push(t as i32);
                qs.push(q);
            }
            let (k, next) = accept_run_sampled(&rows, VOCAB, &drafts, &qs, &mut sampler);
            accepted_total += k;
            let first = if k > 0 { drafts[0] } else { next };
            hist[first as usize] += 1;
        }
        for (i, &p) in p_ref.iter().enumerate() {
            let freq = hist[i] as f64 / N as f64;
            assert!(
                (freq - p as f64).abs() < 0.015,
                "token {i}: emitted {freq:.4} vs target {p:.4} — speculation changed the output distribution"
            );
        }
        // A rule that rejected everything would also pass the histogram check
        // (the bonus draw is p-distributed), and would be a speedup of zero.
        // With a mismatched proposal some drafts must still land.
        assert!(accepted_total > 0, "no draft was ever accepted — speculation bought nothing");
    }

    /// A malformed round must fall back to plain sampling, not to a guess.
    #[test]
    fn a_missing_or_out_of_vocab_draft_distribution_stops_the_run() {
        const VOCAB: usize = 4;
        let target = [1.0f32, 0.0, 0.0, 0.0];
        let rows: Vec<f32> = (0..3).flat_map(|_| target.iter().copied()).collect();
        let q: Vec<f32> = Sampler::new(1.0, 1.0, 1).distribution(&target).to_vec();

        // Two drafts, one `q`: the second cannot be scored against anything, so
        // the run must stop rather than invent a distribution for it.
        let mut s = Sampler::new(1.0, 1.0, 7);
        let (k, _) = accept_run_sampled(&rows, VOCAB, &[0, 0], std::slice::from_ref(&q), &mut s);
        assert!(k <= 1, "a draft with no recorded q must not be accepted: k = {k}");

        // An out-of-vocabulary draft id cannot index `p`/`q` — reject, never index.
        let mut s = Sampler::new(1.0, 1.0, 7);
        let (k, next) = accept_run_sampled(&rows, VOCAB, &[99], &[q], &mut s);
        assert_eq!(k, 0, "an out-of-vocab draft is not acceptable");
        assert!((next as usize) < VOCAB, "the replacement token must be in vocabulary");

        // No drafts at all is the historical single-token step.
        let mut s = Sampler::new(1.0, 1.0, 7);
        let (k, next) = accept_run_sampled(&rows, VOCAB, &[], &[], &mut s);
        assert_eq!(k, 0);
        assert!((next as usize) < VOCAB);
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
    fn optional_artifacts_distinguish_absent_from_malformed() -> Result<(), peregrine_core::Error> {
        // The whole point of the change: "you have no plan.json" and "your
        // plan.json is corrupt" used to be the same silent `None`, so a typo in
        // an artifact an operator had just written produced default behavior and
        // no explanation anywhere. Both still return `None` — correctness must
        // not depend on an optional file — but only one of them is silent.
        let dir = std::env::temp_dir().join(format!("peregrine_artifact_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;

        assert!(read_optional_artifact(&dir, "plan.json").is_none(), "absent -> None");

        std::fs::write(dir.join("plan.json"), b"{not json")?;
        assert!(read_optional_artifact(&dir, "plan.json").is_none(), "malformed -> None, reported");

        std::fs::write(dir.join("plan.json"), br#"{"version": 1}"#)?;
        let v = read_optional_artifact(&dir, "plan.json");
        assert!(v.is_some(), "valid -> Some");
        assert_eq!(
            v.and_then(|v| v.get("version").and_then(|n| n.as_i64())),
            Some(1),
            "and the parsed value is handed back intact"
        );

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn mla_absorb_knob_is_inert_when_off_and_live_when_on() -> Result<(), peregrine_core::Error> {
        // Two claims, because a knob that changes token values has to prove both:
        // unset reproduces the historical logits *exactly*, and set actually
        // reaches a different code path (otherwise it is the `COLI_REGBUF` defect
        // — documented, settable, and wired to nothing).
        //
        // The flag is set on the model, not read from the environment per
        // forward, so this test does not mutate process-global state — `cargo
        // test` runs these in parallel threads and an env write here corrupted
        // `ecache_bit_identical` when it was written that way.
        let dir = tmp_model_dir("absorb")?;
        let baseline = Model::load(&dir)?.forward_step(&[1, 5, 9, 2], 0)?;
        let again = Model::load(&dir)?.forward_step(&[1, 5, 9, 2], 0)?;
        for (a, b) in baseline.iter().zip(&again) {
            assert_eq!(a.to_bits(), b.to_bits(), "off must be bit-identical run to run");
        }

        let mut m = Model::load(&dir)?;
        m.absorb = true;
        let absorbed = m.forward_step(&[1, 5, 9, 2], 0)?;

        assert_eq!(absorbed.len(), baseline.len());
        assert!(absorbed.iter().all(|v| v.is_finite()), "absorb path must produce finite logits");
        let differed = baseline.iter().zip(&absorbed).any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(differed, "COLI_MLA_ABSORB=1 changed nothing — the knob is not wired");

        // Deliberately NOT asserting a closeness bound here. `absorb_approximates_dense`
        // bounds one attention call at 10% relative; this is logits after a whole
        // stack, where that per-layer difference compounds — measured max 2.6
        // absolute on this synthetic model's untrained random weights. Whether
        // that matters is a question about *predictions*, not logit magnitudes,
        // and `Model::prediction_flip_rate` on a real checkpoint is the instrument
        // for it. Asserting an invented bound here would be theatre: it would pass
        // without evidence and fail for reasons unrelated to correctness.
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn layout_schedule_is_bit_identical() -> Result<(), peregrine_core::Error> {
        // Adding a `schedule.json` next to the checkpoint must not change any
        // logit: it only reorders the io_uring submit within a layer, and the
        // deterministic reduce uses `pos` (batch-union index) as its scatter
        // key. Correctness invariant.
        let dir = tmp_model_dir("layout_schedule")?;
        // Reference: no schedule → capture output.
        let want = {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            m.generate(&[3, 7, 1, 4], 6, &mut Sampler::new(0.0, 0.9, 1))?
        };
        // Write a reversed-order schedule for every layer.
        let n_experts = 4i32; // tiny model has 4 routed experts
        let n_layers = 2usize; // tiny model has 2 layers total; first is dense
        let order: Vec<Vec<i32>> = (0..n_layers).map(|_| (0..n_experts).rev().collect()).collect();
        let doc = serde_json::json!({"version": 1, "n_layers": n_layers, "order": order});
        peregrine_core::write_atomic(&dir.join("schedule.json"), &serde_json::to_vec(&doc)?)?;
        // Re-load and re-generate: outputs must match exactly.
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        assert!(m.layout_schedule.is_some(), "schedule.json must load");
        let got = m.generate(&[3, 7, 1, 4], 6, &mut Sampler::new(0.0, 0.9, 1))?;
        assert_eq!(got, want, "schedule reordering must be bit-identical");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn coactivation_ordering_is_bit_identical() -> Result<(), peregrine_core::Error> {
        // Run enough forwards for the co-activation snapshot to rebuild (64+
        // forwards), then confirm the generated stream equals a fresh model's —
        // affinity ordering only permutes dispatch, never the reduce.
        let dir = tmp_model_dir("coact")?;
        let prompt = [3i32, 7, 1, 4];
        let want = {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            m.generate(&prompt, 70, &mut Sampler::new(0.0, 0.9, 1))?
        };
        // Second model with a *pre-seeded* co-activation snapshot: run once to
        // persist route_stats (includes coact), reload → affinity active from
        // the first forward.
        let got = {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            m.generate(&prompt, 70, &mut Sampler::new(0.0, 0.9, 1))?
        };
        assert_eq!(got, want, "affinity/fusion ordering must not change tokens");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn tiled_rows_streamed_matches_resident() -> Result<(), peregrine_core::Error> {
        // Cooperative tiled dispatch: a forward whose row count crosses the
        // par-pool gate computes each streamed expert's rows across the pool
        // (Mlp::swiglu → QtWeight::apply_vec → par_rows). Output must equal the
        // resident path bit-for-bit — the tiling is row-disjoint.
        let dir = tmp_model_dir("tiled")?;
        let toks: Vec<i32> = (0..40).map(|k| (k * 5 + 2) % 32).collect(); // 40 rows > PAR_ROWS_MIN
        let mut resident = Model::load_streaming(&dir, false)?;
        let mut streamed = Model::load_streaming(&dir, true)?;
        let lr = resident.forward_step(&toks, 0)?;
        let ls = streamed.forward_step(&toks, 0)?;
        assert_eq!(lr, ls, "tiled streamed compute must equal resident");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn telemetry_snapshot_is_live_after_a_forward() -> Result<(), peregrine_core::Error> {
        // `PlanOptimizer`/`RuntimeTelemetry` were documented as the per-forward
        // tick and the `/metrics` view, but nothing constructed them — the real
        // policy lived in a second, divergent copy. This proves the wired one
        // runs and reports the warm-cache rates it used to hardcode as `None`.
        let dir = tmp_model_dir("telemetry")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        m.forward_step(&[3, 7, 1, 4], 0)?;
        let t = m.telemetry();
        assert!(
            t.lane.reduce_us + t.lane.io_us + t.lane.cpu_us > 0,
            "telemetry carries this forward's lane timings: {:?}",
            t.lane
        );
        assert!(t.cache_hit_rate.is_some(), "a warm cache exists, so its hit rate is reported");
        // Routing entropy must reach the telemetry snapshot, not just the model.
        // Before this field existed, `routing_entropy_ewma()` documented itself
        // "for telemetry scrapes" and no telemetry structure carried it, so the
        // value was computed every forward and readable by nothing.
        assert_eq!(
            t.entropy_ewma,
            m.routing_entropy_ewma(),
            "telemetry must carry the model's routing entropy, not a default"
        );
        assert!(
            t.entropy_ewma > 0.0,
            "a forward that routed experts has non-zero routing entropy; got {}",
            t.entropy_ewma
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn lane_timings_populate_after_streaming_forward() -> Result<(), peregrine_core::Error> {
        // Streaming mode exercises `moe_forward_concurrent`, which brackets its
        // three lanes with `Instant::now()` and bumps the accumulator. After a
        // forward we expect a non-zero snapshot: at minimum the reduce phase
        // (single-threaded, always runs), and typically the I/O and CPU lanes
        // for a routed layer.
        let dir = tmp_model_dir("lane_timings")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        // Before any forward — no samples yet.
        let before = m.last_lane_timings();
        assert_eq!(before.io_us + before.cpu_us + before.gpu_us + before.reduce_us, 0);
        m.forward_step(&[3, 7, 1, 4], 0)?;
        let after = m.last_lane_timings();
        // The reduce phase (deterministic scatter over batch-union) always runs
        // when there is any routed expert; it is the guaranteed-non-zero signal.
        // I/O and CPU are ~always nonzero too on a routed MoE layer, but their
        // wall time under `Instant` on tiny inputs can fall below microsecond
        // resolution — so we assert on the always-nonzero `reduce_us`.
        assert!(
            after.reduce_us + after.io_us + after.cpu_us > 0,
            "some lane must have accrued time: {:?}",
            after
        );
        // Snapshot-and-reset semantics: a second immediate read observes the
        // sample we just published (via `last_lane`), not fresh accumulator data.
        let again = m.last_lane_timings();
        assert_eq!(again, after, "`last_lane_timings` is stable between forwards");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn route_stats_round_trip() -> Result<(), peregrine_core::Error> {
        // The persisted routing history must round-trip: run one model to populate
        // the history, save, drop, reload — the reloaded model's history must
        // reproduce the saved frames. Correctness-neutral (history only steers
        // prefetch), so we assert on the snapshot, not on any logit.
        let dir = tmp_model_dir("route_stats")?;
        // populate: streaming mode wires up the route history
        {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            let mut s = Sampler::new(0.0, 0.9, 1);
            m.generate(&[3, 7, 1, 4], 6, &mut s)?;
            m.save_route_stats_here()?;
        }
        // route_stats.json is present next to the checkpoint
        assert!(dir.join("route_stats.json").exists(), "shutdown saved route_stats.json");
        // reload: the persisted history is restored (best-effort — only the newest
        // frame per layer needs to be non-empty for the check to be meaningful)
        let m2 = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let hist = m2.route_hist.as_ref().ok_or_else(|| {
            peregrine_core::Error::Format("streaming model must attach route history".into())
        })?;
        let any_frame = (0..m2.cfg.n_layers as usize).any(|l| hist.lock().latest(l).is_some());
        assert!(any_frame, "reloaded route history must contain at least one frame");
        drop(m2);
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
        m.forward_step(&toks, 0)?;
        let (h1, _, d1) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(d1 > 0, "first pass must stream experts from disk");
        m.reset(); // clear KV so the second forward routes identically; cache persists
        m.forward_step(&toks, 0)?;
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
        m.forward_step(&toks, 0)?; // pass 1: fills routing history (and warms the cache)
        m.prefetch_barrier(); // drain the auto-enqueued prefetch
        m.ecache_clear(); // cold cache + zeroed counters (routing history retained)
        m.prefetch_from_history(); // stream predicted (= pass-1) experts on the background lane
        m.prefetch_barrier(); // wait for it to finish
        let pf = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let (_, _, d_pref) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(pf > 0, "prefetch lane must have streamed experts (got {pf})");
        assert_eq!(d_pref, 0, "prefetch must not count as critical-path disk reads");
        m.reset(); // KV reset (history cleared); the warm cache persists
        m.forward_step(&toks, 0)?; // pass 2: identical routing → served from the warm cache
        let (h2, _, d2) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(h2 > 0, "pass 2 must hit the warm tier");
        assert_eq!(d2, 0, "pass 2 must do no critical-path disk reads (prefetch pre-warmed)");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn layer_lookahead_warms_named_layer_in_isolation() -> Result<(), peregrine_core::Error> {
        // The per-layer look-ahead path warms exactly the requested sparse layer off
        // the critical path, leaving other layers (and dense layers) untouched — the
        // evidence that prefetch is emitted per layer, not one bulk dump.
        let dir = tmp_model_dir("lookahead_layer")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?; // fill routing history
        m.prefetch_barrier();
        m.ecache_clear(); // cold cache, history retained
        let first_sparse = m.cfg.first_dense as usize;
        let n_layers = m.cfg.n_layers as usize;
        m.prefetch_layer_from_history(first_sparse);
        m.prefetch_barrier();
        let warmed =
            m.ecache_prefetch_reads_for_layer(first_sparse).ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(warmed > 0, "look-ahead must warm the requested sparse layer (got {warmed})");
        // a different sparse layer was not warmed by this single-layer emission
        let other = first_sparse + 1;
        if other < n_layers {
            let untouched =
                m.ecache_prefetch_reads_for_layer(other).ok_or_else(|| Error::Format("no ecache".into()))?;
            assert_eq!(untouched, 0, "one-layer look-ahead must not warm another layer");
        }
        // a dense layer (below first_dense) is always a no-op
        if first_sparse > 0 {
            m.prefetch_layer_from_history(first_sparse - 1);
            m.prefetch_barrier();
            let dense =
                m.ecache_prefetch_reads_for_layer(first_sparse - 1).ok_or_else(|| Error::Format("no ecache".into()))?;
            assert_eq!(dense, 0, "dense-layer look-ahead is a no-op");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn multipath_hint_tier_issues_fadvise() -> Result<(), peregrine_core::Error> {
        // With warm_paths=0, every fresh predicted expert falls into the fadvise hint
        // tier: the lane issues page-cache hints and streams nothing into the cache.
        let dir = tmp_model_dir("multipath_hint")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?; // fill history + warm cache
        m.prefetch_barrier();
        m.ecache_clear();
        m.set_prefetch_policy(0, usize::MAX); // warm nothing, hint everything
        let first_sparse = m.cfg.first_dense as usize;
        m.prefetch_layer_from_history(first_sparse);
        m.prefetch_barrier();
        let hints = m.ecache_fadvise_hints().ok_or_else(|| Error::Format("no ecache".into()))?;
        let streamed = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let (_, _, disk) = m.ecache_stats().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(hints > 0, "hint tier must issue fadvise hints (got {hints})");
        assert_eq!(streamed, 0, "warm_paths=0 must stream nothing into the cache");
        assert_eq!(disk, 0, "hints are off the critical path");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn stale_predicate_draws_the_window_where_the_emitter_designed_it() {
        // An item emitted during step s targets the layer executing at s + 1, so at
        // slack=1 it is fresh through that layer and dead the step after.
        assert!(!warm_item_is_stale(10, 10, 1), "same-step service is fresh");
        assert!(!warm_item_is_stale(11, 10, 1), "the target layer's own step is fresh");
        assert!(warm_item_is_stale(12, 10, 1), "one step past the target layer is stale");
        // The two sentinel values disarm the gate from either side: a MAX stamp
        // (deliberate bulk warms) and a MAX slack (gate off) are never stale.
        assert!(!warm_item_is_stale(u64::MAX, u64::MAX, 0), "bulk warms never go stale");
        assert!(!warm_item_is_stale(u64::MAX, 0, u64::MAX), "gate off admits any age");
        // Unset env means gate off — the historical behaviour is the default until
        // the A/B licenses a flip.
        assert_eq!(
            SweepClock::from_env().slack.load(std::sync::atomic::Ordering::Relaxed),
            u64::MAX,
            "stale-drop must be opt-in"
        );
    }

    #[test]
    fn a_stale_warm_batch_is_dropped_before_it_costs_a_disk_read() -> Result<(), peregrine_core::Error> {
        let dir = tmp_model_dir("stale_drop")?;
        let m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        // Arm the gate and age the clock *before* anything is enqueued, so the
        // drop decision is already fixed when the lane dequeues — no race with the
        // worker thread, unlike advancing the clock after a send.
        m.sweep.slack.store(1, std::sync::atomic::Ordering::Relaxed);
        m.sweep.step.store(10, std::sync::atomic::Ordering::Relaxed);
        let first_sparse = m.cfg.first_dense as usize;
        let pool = m.prefetch.as_ref().ok_or_else(|| Error::Format("no prefetch pool".into()))?;
        // Stamped at step 0, ten steps ago: the layer window this item was emitted
        // for is long gone, which is exactly the 98.6%-wasted shape of the
        // 2026-08-13 B=16 run.
        let item = crate::concurrent::prefetch_item(m.expert_index.as_ref(), &m.st, &m.cfg, first_sparse, 0)?;
        if pool.lane(0).tx.send(PrefetchMsg::Warm(vec![item], 0)).is_err() {
            return Err(Error::Format("prefetch lane is down".into()));
        }
        m.prefetch_barrier();
        let streamed = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let dropped = m.ecache_prefetch_stale_dropped().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert_eq!(streamed, 0, "a stale item must be dropped before the read, not after");
        assert_eq!(dropped, 1, "the drop must be counted, or the [prefetch] line can't show the win");
        // The same item stamped at the current step is inside its window and
        // streams normally — the gate kills lateness, not speculation.
        let item = crate::concurrent::prefetch_item(m.expert_index.as_ref(), &m.st, &m.cfg, first_sparse, 0)?;
        if pool.lane(0).tx.send(PrefetchMsg::Warm(vec![item], 10)).is_err() {
            return Err(Error::Format("prefetch lane is down".into()));
        }
        m.prefetch_barrier();
        let streamed = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert_eq!(streamed, 1, "a fresh item must stream exactly as before the gate existed");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn gate_left_at_default_services_arbitrarily_late_warms() -> Result<(), peregrine_core::Error> {
        // The load-time default (COLI_PREFETCH_STALE_DROP unset) is slack=MAX: an
        // ancient stamp still streams, pinning "off = the historical behaviour".
        let dir = tmp_model_dir("stale_gate_off")?;
        let m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        m.sweep.step.store(1_000_000, std::sync::atomic::Ordering::Relaxed);
        let first_sparse = m.cfg.first_dense as usize;
        let pool = m.prefetch.as_ref().ok_or_else(|| Error::Format("no prefetch pool".into()))?;
        let item = crate::concurrent::prefetch_item(m.expert_index.as_ref(), &m.st, &m.cfg, first_sparse, 0)?;
        if pool.lane(0).tx.send(PrefetchMsg::Warm(vec![item], 0)).is_err() {
            return Err(Error::Format("prefetch lane is down".into()));
        }
        m.prefetch_barrier();
        let streamed = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let dropped = m.ecache_prefetch_stale_dropped().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert_eq!(streamed, 1, "gate off: even a million-step-old warm still streams");
        assert_eq!(dropped, 0, "gate off must count nothing");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn multipath_warm_tier_streams_not_hints() -> Result<(), peregrine_core::Error> {
        // warm-all policy streams predicted experts and issues no fadvise hints.
        let dir = tmp_model_dir("multipath_warm")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?;
        m.prefetch_barrier();
        m.ecache_clear();
        m.set_prefetch_policy(usize::MAX, 0); // warm all, hint none
        let first_sparse = m.cfg.first_dense as usize;
        m.prefetch_layer_from_history(first_sparse);
        m.prefetch_barrier();
        let hints = m.ecache_fadvise_hints().ok_or_else(|| Error::Format("no ecache".into()))?;
        let streamed = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert_eq!(hints, 0, "warm-all policy issues no hints");
        assert!(streamed > 0, "warm tier must stream experts (got {streamed})");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The LLC-miss control law, which cannot be exercised end to end here:
    /// `perf_event_open` is refused on most VMs and CI, so the live-counter path
    /// would pass by never running.
    #[test]
    fn llc_trend_holds_while_seeding_and_inside_the_dead_band() {
        // No trend yet — the first observation records and must not steer, or the
        // counter's whole accumulated history reads as a one-forward spike.
        assert_eq!(llc_trend(0.0, 1_000_000.0), 0, "seeding observation must not steer");
        assert_eq!(llc_trend(-1.0, 1_000_000.0), 0, "a negative EWMA is not a trend either");
        // Inside ±10% is noise on a thread-following counter, not signal.
        assert_eq!(llc_trend(1000.0, 1000.0), 0);
        assert_eq!(llc_trend(1000.0, 1099.0), 0, "just inside the upper band");
        assert_eq!(llc_trend(1000.0, 901.0), 0, "just inside the lower band");
    }

    #[test]
    fn llc_trend_widens_on_rising_misses_and_narrows_on_falling() {
        // The direction `telemetry.rs` documents. Pinned here so the loop cannot
        // be silently inverted by a later edit — it is a hypothesis about the
        // signal, and an untested hypothesis is indistinguishable from a typo.
        assert_eq!(llc_trend(1000.0, 1101.0), 1, "rising misses widen");
        assert_eq!(llc_trend(1000.0, 5000.0), 1);
        assert_eq!(llc_trend(1000.0, 899.0), -1, "falling misses narrow");
        assert_eq!(llc_trend(1000.0, 0.0), -1);
    }

    #[test]
    fn prefetch_tuner_observes_and_stays_in_bounds() -> Result<(), peregrine_core::Error> {
        // Wiring smoke test: the tuner is consulted before and updated after each
        // forward, and stays within [1, d_max] while observing real prefetch activity.
        // (Direction of adaptation is covered by the control-law unit tests.) Uses the
        // cold-cache pattern so a prefetched expert is actually hit — in steady decode
        // the predicted experts are already cached from the current forward.
        let dir = tmp_model_dir("tuner_wire")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        m.enable_prefetch_tuner(2, 6);
        assert_eq!(m.prefetch_distance(), Some(2));
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?; // fill history + warm cache
        m.prefetch_barrier();
        m.ecache_clear(); // cold cache, history retained
        m.prefetch_from_history(); // warm the predicted experts into the empty cache
        m.prefetch_barrier();
        m.reset(); // KV reset; the prefetched slabs persist in the cache
        m.forward_step(&toks, 0)?; // identical routing → hits the prefetched slabs
        m.prefetch_barrier();
        let (used, _wasted) =
            m.ecache_prefetch_effectiveness().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(used > 0, "prefetched experts must be used, so the tuner observes activity (got {used})");
        let dist = m.prefetch_distance().ok_or_else(|| Error::Format("no tuner".into()))?;
        assert!((1..=6).contains(&dist), "distance {dist} must stay within [1, d_max]");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn predictive_eviction_protects_predicted_experts() -> Result<(), peregrine_core::Error> {
        // After a forward, the model must protect the resident experts it predicts
        // will be reused (priority > 0), leaving unpredicted slots at priority 0. The
        // WarmCache unit tests already prove priority reorders eviction; this proves
        // the model populates it. (Priority never affects output — see the bit-identical
        // tests, which run with protection on by default.)
        let dir = tmp_model_dir("protect")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?; // route + cache experts, fill history
        m.prefetch_barrier();
        m.protect_cache_from_history(); // set priorities from the prediction
        let first_sparse = m.cfg.first_dense as usize;
        let n_experts = m.cfg.n_experts as usize;
        let protected = (0..n_experts)
            .filter(|&e| m.ecache_priority(first_sparse, e).unwrap_or(0) > 0)
            .count();
        assert!(protected > 0, "at least one predicted, resident expert must be protected");
        // a non-resident key always reports priority 0 (never protected).
        assert_eq!(m.ecache_priority(first_sparse, n_experts + 100), Some(0), "non-resident key is unprotected");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn per_sequence_prefetch_records_and_warms() -> Result<(), peregrine_core::Error> {
        // Batched decode must record each stream's *own* routed set into its own
        // history (position s ↔ sequence s), and per-sequence prefetch must warm that
        // stream's experts. Two streams with different prompts are recorded separately.
        let dir = tmp_model_dir("per_seq")?;
        let m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let mut s0 = SeqKv::new(&m.cfg);
        let mut s1 = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&[1, 5, 9], &mut s0, 0)?;
        m.forward_prefill_seq(&[2, 6, 3], &mut s1, 0)?;
        let h0 = Mutex::new(m.new_route_history());
        let h1 = Mutex::new(m.new_route_history());
        {
            let hists = [&h0, &h1];
            let mut refs = [&mut s0, &mut s1];
            m.forward_step_batched(&[4, 7], &mut refs, &[3, 3], Some(&hists))?;
        }
        let first_sparse = m.cfg.first_dense as usize;
        assert!(h0.lock().frames(first_sparse).count() > 0, "seq 0 recorded its own routing");
        assert!(h1.lock().frames(first_sparse).count() > 0, "seq 1 recorded its own routing");
        // Per-sequence prefetch into a cold cache streams that stream's predicted experts.
        m.ecache_clear();
        m.enqueue_seq_prefetch(&h0, 0);
        m.prefetch_barrier();
        let pf = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(pf > 0, "per-sequence prefetch must warm the stream's experts (got {pf})");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn automaton_builds_saves_and_reloads() -> Result<(), peregrine_core::Error> {
        // Offline pipeline round-trip: build an automaton from a corpus, save it next
        // to the checkpoint, and confirm the next load auto-attaches it (predictor
        // becomes the automaton) — and that output stays bit-identical either way.
        let dir = tmp_model_dir("automaton")?;
        let corpus: Vec<i32> = (0..40i32).map(|i| (i * 7 + 3) % 32).collect();
        // reference output on a plain (momentum) model
        let want = {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            let mut s = Sampler::new(0.0, 0.9, 1);
            m.generate(&[3, 7, 1, 4], 6, &mut s)?
        };
        // build + save the automaton
        {
            let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
            let table = m.build_automaton(&corpus)?;
            save_automaton(&table, &dir.join("automaton.json"))?;
        }
        // reload: the predictor must now be the automaton, and output unchanged
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        assert!(m.predictor_is_automaton(), "reload must auto-attach the matching automaton");
        let mut s = Sampler::new(0.0, 0.9, 1);
        let got = m.generate(&[3, 7, 1, 4], 6, &mut s)?;
        assert_eq!(got, want, "automaton predictor must not change output (prefetch only)");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn prefetch_verify_reports_no_mismatch() -> Result<(), peregrine_core::Error> {
        // With verification on, the prefetch lane re-reads each speculative slab and
        // byte-compares it; deterministic reads mean zero mismatches. Also confirms the
        // verify path actually ran (prefetch reads happened). Accuracy counters populate.
        std::env::set_var("COLI_PREFETCH_VERIFY", "1");
        let dir = tmp_model_dir("verify")?;
        let load = Model::load_streaming_ecache(&dir, true, 8 << 20); // pool spawns with verify
        std::env::remove_var("COLI_PREFETCH_VERIFY");
        let mut m = load?;
        let toks = [1, 5, 9, 2, 7];
        m.forward_step(&toks, 0)?; // history
        m.prefetch_barrier();
        m.ecache_clear();
        m.prefetch_from_history(); // real speculative reads → each re-read + compared
        m.prefetch_barrier();
        let pf = m.ecache_prefetch_reads().ok_or_else(|| Error::Format("no ecache".into()))?;
        let mm = m.ecache_verify_mismatch().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(pf > 0, "verify path must run on real prefetch reads (got {pf})");
        assert_eq!(mm, 0, "deterministic reads → zero verify mismatches");
        // reset() then a matching forward makes the prefetched slabs count as used.
        m.reset();
        m.forward_step(&toks, 0)?;
        m.prefetch_barrier();
        let (used, _) = m.ecache_prefetch_effectiveness().ok_or_else(|| Error::Format("no ecache".into()))?;
        assert!(used > 0, "accuracy counters must populate (used={used})");
        assert!(m.prefetch_accuracy().unwrap_or(-1.0) >= 0.0, "accuracy is well-defined");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn flip_rate_measures_disagreement_and_refuses_to_guess() {
        // Identical runs: the answer a lossless change must produce.
        assert_eq!(Model::prediction_flip_rate(&[1, 2, 3, 4], &[1, 2, 3, 4]), Some(0.0));
        // One position in four moved.
        assert_eq!(Model::prediction_flip_rate(&[1, 2, 3, 4], &[1, 2, 9, 4]), Some(0.25));
        assert_eq!(Model::prediction_flip_rate(&[1], &[2]), Some(1.0));
        // No data is not the same as no change — both must be `None` rather than
        // a tempting 0.0 that would silently pass a gate.
        assert_eq!(Model::prediction_flip_rate(&[], &[]), None);
        assert_eq!(Model::prediction_flip_rate(&[1, 2], &[1]), None);
    }

    #[test]
    fn flip_rate_is_zero_between_two_identical_forwards() -> Result<(), peregrine_core::Error> {
        // End-to-end sanity: the harness reports agreement on a real (tiny) model,
        // so a nonzero rate later means the change under test, not the harness.
        let dir = tmp_model_dir("fliprate")?;
        let mut m = Model::load(&dir)?;
        let toks: Vec<i32> = (0..12).map(|k| (k * 5 + 1) % 32).collect();
        let a = m.teacher_forcing(&toks)?;
        let b = m.teacher_forcing(&toks)?;
        assert_eq!(Model::prediction_flip_rate(&a, &b), Some(0.0), "a model must agree with itself");
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
        let mut greedy = Sampler::new(0.0, 0.9, 1);
        let base = m.generate(&prompt, 8, &mut greedy)?;
        // Sweep the depths an operator can actually reach through `--draft`.
        // The single depth-3 case this used to test left the recommended 4-6
        // range unguarded — and depth is exactly the axis the "MTP is a net
        // loss" figures were taken on (they used 2, where 2.46 accepted is
        // already 82% of that configuration's ceiling of 3).
        for g in [1usize, 2, 3, 4, 5, 6, 8] {
            let spec = m.generate_speculative(&prompt, 8, g)?;
            assert_eq!(spec, base, "speculative output must equal greedy at draft depth {g}");
        }
        // Depth 0 means "no drafting" and must degrade to the plain path rather
        // than erroring or spinning: `g_draft == 0` is an explicit early return.
        assert_eq!(m.generate_speculative(&prompt, 8, 0)?, base, "draft depth 0 is the plain path");
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
        assert_eq!(stream_transient_reserve(4, 8, 1000), (4 * experts_per_batch() + 8) * 1000);
        assert_eq!(stream_transient_reserve(0, 0, 1000), 0); // no lanes → no reserve
    }

    #[test]
    fn self_cgroup_rel_takes_the_v2_line_and_ignores_v1_noise() {
        // A hybrid dump: v1 controllers above, the v2 line among them.
        let dump = "12:pids:/user.slice\n1:name=systemd:/init.scope\n0::/user.slice/user-1000.slice/run-abc.scope\n";
        assert_eq!(
            self_cgroup_v2_rel(dump),
            Some("/user.slice/user-1000.slice/run-abc.scope")
        );
        assert_eq!(self_cgroup_v2_rel("12:pids:/x\n"), None, "no v2 line, no walk");
    }

    #[test]
    fn cgroup_walk_visits_leaf_to_root_so_the_tightest_limit_wins() {
        // The stage-5 OOM shape: the MemoryMax lives on the transient scope,
        // not the root — every level must be visited or the limit is missed.
        let dirs = cgroup_walk_dirs("/user.slice/run-abc.scope");
        let s: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
        assert_eq!(
            s,
            vec![
                "/sys/fs/cgroup/user.slice/run-abc.scope",
                "/sys/fs/cgroup/user.slice",
                "/sys/fs/cgroup",
            ]
        );
        // Root-relative spelling degenerates to the root probe alone.
        assert_eq!(cgroup_walk_dirs("/"), vec![std::path::PathBuf::from("/sys/fs/cgroup")]);
    }

    #[test]
    fn ecache_spec_parses_numbers_auto_and_rejects_garbage() {
        const GIB: usize = 1 << 30;
        // The historical numeric spelling is untouched, fractions included.
        assert_eq!(parse_ecache_spec("8", None), Some(EcacheSpec::Fixed(8 * GIB)));
        assert_eq!(parse_ecache_spec("0.5", None), Some(EcacheSpec::Fixed(GIB / 2)));
        assert_eq!(parse_ecache_spec("0", None), Some(EcacheSpec::Fixed(0)));
        assert_eq!(parse_ecache_spec("-2", None), Some(EcacheSpec::Fixed(0)), "negative clamps to disabled");
        // `auto` takes ds4's 0.80 unless COLI_ECACHE_AUTO_FRAC narrows it.
        assert_eq!(parse_ecache_spec("auto", None), Some(EcacheSpec::AutoFrac(0.80)));
        assert_eq!(parse_ecache_spec(" AUTO ", None), Some(EcacheSpec::AutoFrac(0.80)), "case/space insensitive");
        assert_eq!(parse_ecache_spec("auto", Some("0.5")), Some(EcacheSpec::AutoFrac(0.5)));
        // The fraction clamps to 0.95 — 1.0 would hand the cache every free
        // byte — and nonsense fractions fall back to the default rather than 0.
        assert_eq!(parse_ecache_spec("auto", Some("1.4")), Some(EcacheSpec::AutoFrac(0.95)));
        assert_eq!(parse_ecache_spec("auto", Some("0")), Some(EcacheSpec::AutoFrac(0.80)));
        assert_eq!(parse_ecache_spec("auto", Some("nan")), Some(EcacheSpec::AutoFrac(0.80)));
        assert_eq!(parse_ecache_spec("auto", Some("gibberish")), Some(EcacheSpec::AutoFrac(0.80)));
        // Garbage spellings surface as None so the caller can disable-with-advisory.
        assert_eq!(parse_ecache_spec("lots", None), None);
        assert_eq!(parse_ecache_spec("", None), None);
    }

    #[test]
    fn rmsnorm_rows_parallel_matches_serial() {
        // rmsnorm_rows runs rows on the compute pool when the row is wide enough
        // (d >= 256); use d=512 so the parallel path engages, and assert it stays
        // bit-identical to a plain serial loop (rows are independent).
        let (s_n, d) = (17usize, 512usize);
        let x: Vec<f32> = (0..s_n * d).map(|k| ((k * 7 + 3) as f32 * 0.01).sin()).collect();
        let w: Vec<f32> = (0..d).map(|j| 0.5 + j as f32 * 0.01).collect();
        let eps = 1e-5;
        let par = rmsnorm_rows(&x, &w, s_n, d, eps);
        let mut serial = vec![0f32; s_n * d];
        for s in 0..s_n {
            let src = x[s * d..s * d + d].to_vec();
            rmsnorm(&mut serial[s * d..s * d + d], &src, &w, eps);
        }
        assert!(
            par.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()),
            "rmsnorm_rows must be bit-identical parallel vs serial"
        );
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
    fn kv_export_import_round_trips_bit_identically() -> Result<(), peregrine_core::Error> {
        // The disk-persistence seam's whole contract: a cache rebuilt from an
        // export must be indistinguishable from the live one — the values it
        // exports again are equal, and (the decisive check) a forward continued
        // from it produces bit-identical logits. Both dtypes, since f16 narrows
        // on append and must re-narrow to the same bits.
        let dir = tmp_model_dir("kvexport")?;
        let m = Model::load(&dir)?;
        let toks = [1i32, 5, 9, 2, 7, 3];
        for dt in [KvDtype::F32, KvDtype::F16] {
            let mut live = SeqKv::with_dtype(&m.cfg, dt);
            m.forward_prefill_seq(&toks, &mut live, 0)?;
            let ex = live.export_prefix(live.len());
            assert_eq!(ex.n, toks.len());
            let restored = SeqKv::import(&m.cfg, dt, &ex)?;
            assert_eq!(restored.len(), live.len());

            let again = restored.export_prefix(restored.len());
            for (a, b) in ex.layers.iter().zip(&again.layers) {
                assert!(a.lc.iter().zip(&b.lc).all(|(x, y)| x.to_bits() == y.to_bits()), "lc drifted ({dt:?})");
                assert!(a.rc.iter().zip(&b.rc).all(|(x, y)| x.to_bits() == y.to_bits()), "rc drifted ({dt:?})");
                assert_eq!(a.lc.len(), b.lc.len());
                assert_eq!(a.rc.len(), b.rc.len());
            }

            let (mut a, mut b) = (live, restored);
            let pos = toks.len();
            let mut one: [&mut SeqKv; 1] = [&mut a];
            let from_live = m.forward_rows_batched(&[7], &[0], &mut one, &[pos], None)?;
            let mut one: [&mut SeqKv; 1] = [&mut b];
            let from_restored = m.forward_rows_batched(&[7], &[0], &mut one, &[pos], None)?;
            assert!(
                from_live.iter().zip(&from_restored).all(|(x, y)| x.to_bits() == y.to_bits()),
                "a forward from the restored cache must be bit-identical ({dt:?})"
            );
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn calib_capture_accumulates_moe_inputs_and_writes_the_sidecar() -> Result<(), peregrine_core::Error> {
        // The ideas-#7 capture hook end to end at model level: teacher-force a
        // corpus with capture enabled, and the sidecar must hold mean-|x|
        // vectors exactly where experts live — dense layer 0 empty, layers 1–2
        // populated at hidden width, the MTP row empty (capture never drafts).
        let dir = tmp_model_dir("calibcap")?;
        let mut m = Model::load(&dir)?;
        assert!(m.write_calib_sidecar()?.is_none(), "no capture enabled → nothing to write");

        let out = dir.join("calib_channels.json");
        m.enable_calib_capture(out.clone());
        let toks = [1i32, 5, 9, 2, 7, 3, 11, 4];
        m.teacher_forcing(&toks)?;
        let p = m
            .write_calib_sidecar()?
            .ok_or_else(|| peregrine_core::Error::Format("sidecar expected".into()))?;
        assert_eq!(p, out);

        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out)?)
            .map_err(|e| peregrine_core::Error::Format(format!("sidecar parse: {e}")))?;
        assert_eq!(v["version"], 1);
        assert_eq!(v["stat"], "mean_abs");
        assert_eq!(v["hidden"], 16);
        assert_eq!(v["positions"], toks.len());
        let layers = v["layers"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        assert_eq!(layers.len(), 4, "3 layers + the MTP row");
        assert!(layers[0].as_array().is_some_and(Vec::is_empty), "dense layer 0 has no MoE input");
        assert!(layers[3].as_array().is_some_and(Vec::is_empty), "the MTP row never accumulates");
        for l in [1, 2] {
            let row = layers[l].as_array().map(Vec::as_slice).unwrap_or(&[]);
            assert_eq!(row.len(), 16, "layer {l} at hidden width");
            let vals: Vec<f64> = row.iter().filter_map(|x| x.as_f64()).collect();
            assert_eq!(vals.len(), 16);
            assert!(vals.iter().all(|x| x.is_finite() && *x >= 0.0), "means are |x| averages");
            assert!(vals.iter().any(|x| *x > 0.0), "a real forward leaves nonzero magnitude");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn kv_import_refuses_mismatched_shapes() -> Result<(), peregrine_core::Error> {
        let dir = tmp_model_dir("kvimportbad")?;
        let m = Model::load(&dir)?;
        let mut live = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&[1, 5, 9, 2], &mut live, 0)?;
        let mut ex = live.export_prefix(live.len());
        // Truncated stream → refused, not silently misaligned.
        if let Some(l0) = ex.layers.first_mut() {
            l0.lc.pop();
        }
        assert!(SeqKv::import(&m.cfg, KvDtype::F32, &ex).is_err(), "a short lc stream must be refused");
        // Wrong layer count → refused.
        let mut ex = live.export_prefix(live.len());
        ex.layers.pop();
        assert!(SeqKv::import(&m.cfg, KvDtype::F32, &ex).is_err(), "a missing layer must be refused");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    fn tmp_indexer_model_dir(tag: &str, topk: i64) -> Result<PathBuf, peregrine_core::Error> {
        let d = std::env::temp_dir().join(format!("peregrine_dsa_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        crate::testkit::build_tiny_model_with_indexer(&d, 0xD5A, topk)?;
        Ok(d)
    }

    #[test]
    fn dsa_is_inert_without_an_indexer_and_below_index_topk() -> Result<(), peregrine_core::Error> {
        // Two ways the flag must change nothing, both bit-exact.
        //
        // (1) No indexer in the checkpoint — the state a laptop-converted
        //     container is actually in — so `COLI_DSA` has nothing to run.
        // (2) An indexer, but a context no longer than `index_topk`: attention
        //     over at most that many keys *is* the selection, so skipping the
        //     scoring pass is exactly output-neutral, not approximately so.
        //     That is the C engine's activation rule, and getting it wrong
        //     would show up as a silent quality regression on short prompts.
        let dir = tmp_model_dir("dsa_inert")?;
        let mut m = Model::load(&dir)?;
        let toks = [1i32, 5, 9, 2];
        let mut off = SeqKv::new(&m.cfg);
        let a = m.forward_prefill_seq(&toks, &mut off, 0)?;
        m.dsa = true;
        let mut on = SeqKv::new(&m.cfg);
        let b = m.forward_prefill_seq(&toks, &mut on, 0)?;
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "no indexer: DSA must be inert");
        std::fs::remove_dir_all(&dir)?;

        let dir = tmp_indexer_model_dir("below", 64)?;
        let mut m = Model::load(&dir)?;
        assert!(m.layers.iter().any(|l| l.indexer.is_some()), "the fixture must carry indexer tensors");
        let mut off = SeqKv::new(&m.cfg);
        let a = m.forward_prefill_seq(&toks, &mut off, 0)?;
        m.dsa = true;
        let mut on = SeqKv::new(&m.cfg);
        let b = m.forward_prefill_seq(&toks, &mut on, 0)?;
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "4 tokens under index_topk=64: selection is the identity, so output must be bit-identical"
        );
        // …and the key cache is built anyway, because a later selection needs
        // keys for every earlier position.
        assert_eq!(on.index_len(), toks.len(), "indexer keys must be cached from position 0");
        assert_eq!(off.index_len(), 0, "…but only when DSA is on");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn dsa_selects_a_subset_once_context_exceeds_index_topk() -> Result<(), peregrine_core::Error> {
        // The path that was shipped, unit-tested and never once constructed.
        // Past `index_topk` the indexer keeps a strict subset, so the output
        // must *differ* from dense — a sparse-attention flag whose test would
        // pass unchanged if selection did nothing is not testing selection.
        let dir = tmp_indexer_model_dir("above", 2)?;
        let mut m = Model::load(&dir)?;
        let toks = [1i32, 5, 9, 2, 6, 3];
        let mut off = SeqKv::new(&m.cfg);
        let dense = m.forward_prefill_seq(&toks, &mut off, 0)?;
        m.dsa = true;
        let mut on = SeqKv::new(&m.cfg);
        let sparse = m.forward_prefill_seq(&toks, &mut on, 0)?;
        assert_eq!(dense.len(), sparse.len());
        assert!(sparse.iter().all(|v| v.is_finite()), "sparse attention must not produce NaN");
        assert!(
            dense.iter().zip(&sparse).any(|(p, q)| p.to_bits() != q.to_bits()),
            "index_topk=2 over 6 positions must attend a strict subset"
        );
        // Decode continues on the sparse cache, and both streams stay aligned.
        let mut one: [&mut SeqKv; 1] = [&mut on];
        m.forward_step_batched(&[7], &mut one, &[toks.len()], None)?;
        assert_eq!(on.len(), toks.len() + 1);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn indexer_keys_ride_the_kv_cache_through_sharing_and_rewind() -> Result<(), peregrine_core::Error> {
        // The indexer key cache is a third stream in `LayerKv` precisely so it
        // inherits the two lifecycle properties the latents already have. If it
        // did not, a shared prefix would serve keys for the wrong positions and
        // a speculative rewind would leave the two streams misaligned — both
        // silent, both producing plausible-looking output.
        let dir = tmp_indexer_model_dir("share", 2)?;
        let mut m = Model::load(&dir)?;
        m.dsa = true;
        let toks = [1i32, 5, 9, 2, 6, 3];
        let mut seq = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&toks, &mut seq, 0)?;
        assert_eq!(seq.index_len(), toks.len());

        let entry = seq.clone_prefix(4);
        assert_eq!(entry.index_len(), 4, "a shared prefix carries its indexer keys");
        assert_eq!(entry.len(), 4);
        let seeded = entry.clone_prefix(4);
        assert_eq!(seeded.index_len(), 4, "…and a refcounted view of it still sees them");

        // Seeding from the shared prefix and prefilling the rest must be
        // bit-identical to a cold run of the whole prompt — under DSA too,
        // since the selection at position 4 scores keys 0..4, which are the
        // same bytes either way. This is prefix sharing and sparse selection
        // composing, which neither feature's own tests can show.
        let mut warm = entry.clone_prefix(4);
        let warm_logits = m.forward_prefill_seq(&toks[4..], &mut warm, 4)?;
        let mut cold = SeqKv::new(&m.cfg);
        let cold_logits = m.forward_prefill_seq(&toks, &mut cold, 0)?;
        assert_eq!(warm.len(), toks.len());
        assert_eq!(warm.index_len(), toks.len(), "the tail's keys append after the shared ones");
        let vocab = m.cfg.vocab as usize;
        let tail = &cold_logits[4 * vocab..];
        assert_eq!(warm_logits.len(), tail.len());
        for (k, (p, q)) in warm_logits.iter().zip(tail).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "logit {k}: seeding from a shared prefix moved a bit");
        }

        // A rewind takes both streams back together.
        let mut r = seq;
        r.truncate(2);
        assert_eq!((r.len(), r.index_len()), (2, 2), "latents and indexer keys rewind together");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn f16_kv_survives_the_whole_path_and_halves_the_sequence_bytes() -> Result<(), peregrine_core::Error> {
        // `COLI_KV_DTYPE=f16` has to hold up across prefill, batched decode and
        // the byte accounting admission reads — not just inside the attention
        // core where it is unit-tested. The saving is exact, so assert it as an
        // equality rather than a bound.
        let dir = tmp_model_dir("kv_f16")?;
        let m = Model::load(&dir)?;
        let prompt = [1i32, 5, 9, 2, 6, 3];
        let mut wide = SeqKv::with_dtype(&m.cfg, KvDtype::F32);
        let mut narrow = SeqKv::with_dtype(&m.cfg, KvDtype::F16);
        let lw = m.forward_prefill_seq(&prompt, &mut wide, 0)?;
        let ln = m.forward_prefill_seq(&prompt, &mut narrow, 0)?;
        assert_eq!(narrow.len(), wide.len());
        assert_eq!(narrow.bytes() * 2, wide.bytes(), "f16 must be exactly half the resident bytes");
        assert_eq!(narrow.owned_bytes() * 2, wide.owned_bytes());
        assert_eq!(lw.len(), ln.len());

        // Decode continues on both, and the narrowed cache tracks the wide one.
        // This is a *lossy* knob, so the assertion is closeness, not equality —
        // `Model::prediction_flip_rate` on a real checkpoint is the real gate.
        let mut a: [&mut SeqKv; 1] = [&mut wide];
        let dw = m.forward_step_batched(&[7], &mut a, &[prompt.len()], None)?;
        let mut b: [&mut SeqKv; 1] = [&mut narrow];
        let dn = m.forward_step_batched(&[7], &mut b, &[prompt.len()], None)?;
        assert_eq!(narrow.len(), wide.len(), "both advanced by one position");
        assert_eq!(narrow.bytes() * 2, wide.bytes(), "…and the halving holds after decode");
        let span = dw.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        for (k, (p, q)) in dw.iter().zip(&dn).enumerate() {
            assert!((p - q).abs() < 5e-2 * span, "logit {k}: {p} vs {q}");
        }

        // Sharing composes with the element type: a narrowed prefix is still
        // frozen once and viewed by refcount.
        let entry = narrow.clone_prefix(4);
        let seeded = entry.clone_prefix(4);
        assert_eq!(entry.shared_prefix(), seeded.shared_prefix(), "narrowing must not defeat sharing");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn sharing_a_prefix_charges_one_allocation_not_one_per_sequence() -> Result<(), peregrine_core::Error> {
        // What the serving engine's byte budget has to see: two admissions
        // seeded from one cached prompt hold *one* allocation between them, not
        // two. Reporting logical `bytes()` twice would refuse admissions over
        // RAM that was never allocated — cancelling the saving it is measuring.
        let dir = tmp_model_dir("kv_share_bytes")?;
        let m = Model::load(&dir)?;
        let mut src = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&[1, 5, 9, 2, 6, 3], &mut src, 0)?;
        assert!(src.owned_bytes() > 0, "a prefilled sequence holds KV");
        assert_eq!(src.shared_prefix(), None, "…and owns all of it privately");
        assert_eq!(src.bytes(), src.owned_bytes(), "logical == private when nothing is shared");

        let entry = src.clone_prefix(4); // what the prefix cache stores
        let a = entry.clone_prefix(4); // two admissions matching it, to different depths
        let b = entry.clone_prefix(3);
        let missing = || Error::Format("a seeded sequence must report its shared prefix".into());
        let (id_a, bytes_a) = a.shared_prefix().ok_or_else(missing)?;
        let (id_b, bytes_b) = b.shared_prefix().ok_or_else(missing)?;
        assert_eq!(id_a, id_b, "both admissions view the same allocation");
        assert_eq!(bytes_a, bytes_b, "…so both report its full size, not their view of it");
        assert_eq!(bytes_a, entry.bytes(), "which is what the snapshot itself costs");
        assert_eq!((a.owned_bytes(), b.owned_bytes()), (0, 0), "neither owns a row yet");
        assert!(b.bytes() < a.bytes(), "the shallower view is still logically smaller");

        // The accounting identity the engine's `resident_kv` relies on: the pair
        // costs one prefix, where summing `bytes()` would have charged two.
        let deduped = a.owned_bytes() + b.owned_bytes() + bytes_a;
        assert_eq!(deduped, entry.bytes());
        assert!(deduped < a.bytes() + b.bytes(), "sharing must show up as a saving, not a wash");

        // Decoding on top adds private bytes only, leaving the shared charge fixed.
        let mut a = a;
        let mut one: [&mut SeqKv; 1] = [&mut a];
        m.forward_step_batched(&[7], &mut one, &[4], None)?;
        assert!(a.owned_bytes() > 0, "the new position went into the private tail");
        assert_eq!(a.shared_prefix().ok_or_else(missing)?.1, bytes_a, "the shared charge is unchanged");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn batched_decode_honours_the_absorb_knob_and_defaults_to_dense() -> Result<(), peregrine_core::Error> {
        // **The knob used to stop at the batched path.** `forward_layer_batched`
        // called an absorb-only core unconditionally, so over HTTP a request ran
        // its prefill on the dense core and every decode token on absorption —
        // two algebraically-equal, numerically-different implementations inside
        // one response — while the docs called absorb opt-in and off by default.
        //
        // The contract this now asserts is the documented one: at the default,
        // batched decode is the *same* core as the single-sequence decode, bit
        // for bit; with the knob set, it is not.
        let dir = tmp_model_dir("batched_absorb")?;
        let mut m = Model::load(&dir)?;
        assert!(!m.absorb, "the default must still be dense");
        let prompt = [1i32, 5, 9];
        let pos = prompt.len();

        // Three identically prefilled sequences, all through the dense path, so
        // the decode step below is the only thing that varies.
        let prefilled = || -> Result<SeqKv, peregrine_core::Error> {
            let mut sk = SeqKv::new(&m.cfg);
            m.forward_prefill_seq(&prompt, &mut sk, 0)?;
            Ok(sk)
        };
        let (mut a, mut b, mut c) = (prefilled()?, prefilled()?, prefilled()?);

        let dense = m.forward_prefill_seq(&[4], &mut c, pos)?; // single-sequence, honours the knob
        let mut one: [&mut SeqKv; 1] = [&mut a];
        let batched_off = m.forward_step_batched(&[4], &mut one, &[pos], None)?;
        m.absorb = true;
        let mut one: [&mut SeqKv; 1] = [&mut b];
        let batched_on = m.forward_step_batched(&[4], &mut one, &[pos], None)?;

        assert_eq!(dense.len(), batched_off.len());
        for (k, (p, q)) in dense.iter().zip(&batched_off).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "logit {k}: batched decode at the default must be the dense core");
        }
        assert!(
            batched_off.iter().zip(&batched_on).any(|(p, q)| p.to_bits() != q.to_bits()),
            "…and setting COLI_MLA_ABSORB must actually reach it"
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn the_batched_forward_hands_back_the_hidden_a_draft_needs() -> Result<(), peregrine_core::Error> {
        // A batched speculative loop needs two things from one forward: the
        // logits to verify against, and the **pre-final-norm** hidden the MTP
        // head drafts from. Without the second it would have to re-run the
        // stack to recover what this forward already computed, which is the
        // speedup.
        //
        // Pre-final-norm matters: `mtp_draft` applies `final_norm` itself on
        // its first step. Handing it a normalised hidden would norm it twice —
        // no error, no crash, just quietly worse drafts and a lower acceptance
        // rate that would read as "MTP does not help here".
        let dir = tmp_model_dir("rows_hidden")?;
        let m = Model::load(&dir)?;
        let (d, vocab) = (m.cfg.hidden as usize, m.cfg.vocab as usize);
        let prompt = [1i32, 5, 9];
        let mut sk = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&prompt, &mut sk, 0)?;
        let tok = 7i32;
        let pos = prompt.len();

        // The logits-only form and the hidden form must agree bit for bit on
        // the logits — one is a wrapper over the other, and a divergence would
        // mean the speculative path verifies against different numbers than
        // the plain decode path.
        let mut a = sk.clone_prefix(pos);
        let mut one: [&mut SeqKv; 1] = [&mut a];
        let plain = m.forward_rows_batched(&[tok], &[0], &mut one, &[pos], None)?;
        let mut b = sk.clone_prefix(pos);
        let mut one: [&mut SeqKv; 1] = [&mut b];
        let (with_h, hidden) = m.forward_rows_batched_hidden(&[tok], &[0], &mut one, &[pos], None)?;
        assert_eq!(plain.len(), vocab);
        assert!(plain.iter().zip(&with_h).all(|(p, q)| p.to_bits() == q.to_bits()), "the two forms must agree");
        assert_eq!(hidden.len(), d, "one row of hidden per row of input");
        assert!(hidden.iter().all(|v| v.is_finite()));

        // It is genuinely pre-norm: applying `final_norm` changes it. If the
        // forward had already normalised, this would be a no-op and the double
        // norm would be invisible.
        let normed = rmsnorm_rows(&hidden, &m.final_norm, 1, d, m.cfg.eps);
        assert!(
            hidden.iter().zip(&normed).any(|(p, q)| p.to_bits() != q.to_bits()),
            "the hidden must be pre-final-norm, or mtp_draft will norm it twice"
        );

        // And it is what the drafter accepts: `mtp_draft` takes `&self`, so
        // several sequences can draft from one `&Model` without serialising.
        if m.has_mtp() {
            let draft = m.mtp_draft(tok, 2, &hidden, 0.0)?;
            assert_eq!(draft.len(), 2, "the head drafts to the requested depth");
            assert!(draft.iter().all(|&t| t >= 0 && (t as usize) < vocab), "drafts must be real token ids");
            // An impossible floor stops the draft at depth 0; the tokens a
            // permissive floor keeps are a prefix of the unfloored draft —
            // the gate may only shorten, never redirect.
            let none = m.mtp_draft(tok, 2, &hidden, 1.1)?;
            assert!(none.is_empty(), "a floor above 1.0 must draft nothing");
            let floored = m.mtp_draft(tok, 2, &hidden, 1e-9)?;
            assert_eq!(floored, draft, "an always-passing floor must not change the draft");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn speculative_rows_emit_exactly_what_greedy_would() -> Result<(), peregrine_core::Error> {
        // **The property speculation exists to preserve.** A draft is accepted
        // only where it equals the model's own argmax, so the emitted stream
        // must be indistinguishable from one-token-at-a-time greedy decoding —
        // speculation buys wall clock, never different output.
        //
        // The drafts here are *deliberately* a mix of right and wrong: a test
        // that only ever drafted correctly would pass with the reject path
        // broken, and a test that only ever drafted wrongly would never
        // exercise acceptance at all.
        let dir = tmp_model_dir("spec_rows")?;
        let m = Model::load(&dir)?;
        let vocab = m.cfg.vocab as usize;
        let prompt = [1i32, 5, 9, 2];
        let n_new = 6usize;

        // Greedy reference: one token at a time through the batched path.
        let mut want = Vec::new();
        {
            let mut sk = SeqKv::new(&m.cfg);
            let lg = m.forward_prefill_seq(&prompt, &mut sk, 0)?;
            let mut next = argmax(&lg[(prompt.len() - 1) * vocab..]) as i32;
            let mut pos = prompt.len();
            while want.len() < n_new {
                want.push(next);
                let mut one: [&mut SeqKv; 1] = [&mut sk];
                let lg = m.forward_step_batched(&[next], &mut one, &[pos], None)?;
                next = argmax(&lg) as i32;
                pos += 1;
            }
        }

        // Speculative: same sequence, drafts that are sometimes right (taken
        // from the reference) and sometimes deliberately wrong.
        let mut got = Vec::new();
        {
            let mut sk = SeqKv::new(&m.cfg);
            let lg = m.forward_prefill_seq(&prompt, &mut sk, 0)?;
            let mut next = argmax(&lg[(prompt.len() - 1) * vocab..]) as i32;
            let mut pos = prompt.len();
            let mut round = 0usize;
            let mut any_accepted = false;
            let mut any_rejected = false;
            while got.len() < n_new {
                // Round 0 drafts correctly (from the reference), round 1 drafts
                // garbage, and so on — so both paths are exercised.
                let take = got.len() + 1;
                let draft: Vec<i32> = if round.is_multiple_of(2) {
                    want.iter().skip(take).take(2).copied().collect()
                } else {
                    vec![(vocab as i32) - 1, 0]
                };
                let mut one: [&mut SeqKv; 1] = [&mut sk];
                let (v, _h) = m.verify_drafts_batched(&[next], std::slice::from_ref(&draft), &mut one, &[pos])?;
                let r = v.first().ok_or_else(|| Error::Format("no verification result".into()))?;
                any_accepted |= r.accepted > 0;
                any_rejected |= r.accepted < draft.len();
                got.extend_from_slice(&r.tokens);
                pos += r.tokens.len();
                next = r.next;
                round += 1;
            }
            got.truncate(n_new);
            assert!(any_accepted, "the accept path was never taken — the test proves nothing about it");
            assert!(any_rejected, "…nor was the reject path");
        }
        assert_eq!(got, want, "speculation changed the token stream");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_rejected_draft_leaves_no_trace_in_the_cache() -> Result<(), peregrine_core::Error> {
        // The rewind. A speculated tail that was not accepted must not remain
        // cached: the next round appends at the committed position, and a stale
        // row there would silently attend a token the model never emitted.
        let dir = tmp_model_dir("spec_rewind")?;
        let m = Model::load(&dir)?;
        let prompt = [1i32, 5, 9];
        let mut sk = SeqKv::new(&m.cfg);
        let lg = m.forward_prefill_seq(&prompt, &mut sk, 0)?;
        let vocab = m.cfg.vocab as usize;
        let next = argmax(&lg[(prompt.len() - 1) * vocab..]) as i32;

        // Four drafts, all deliberately wrong, so none can be accepted.
        let draft = vec![(vocab as i32) - 1, (vocab as i32) - 2, 0, 1];
        let mut one: [&mut SeqKv; 1] = [&mut sk];
        let (v, _h) = m.verify_drafts_batched(&[next], &[draft], &mut one, &[prompt.len()])?;
        let r = v.first().ok_or_else(|| Error::Format("no verification result".into()))?;
        assert_eq!(r.accepted, 0, "a wrong draft must not be accepted");
        assert_eq!(r.tokens, vec![next], "only the already-confirmed token is emitted");
        assert_eq!(sk.len(), prompt.len() + 1, "the cache holds the prompt plus one committed token");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn sequences_speculate_to_different_depths_in_one_forward() -> Result<(), peregrine_core::Error> {
        // The shape that makes speculation pay on a disk-bound engine: B
        // sequences × (1+γ) rows in *one* forward, sharing one routed-expert
        // union. Depths differ per sequence, because a draft that hits a stop
        // token or a budget is shorter — so the row layout cannot assume a
        // rectangle.
        let dir = tmp_model_dir("spec_multi")?;
        let m = Model::load(&dir)?;
        let vocab = m.cfg.vocab as usize;
        let prompts: [&[i32]; 3] = [&[1, 5, 9], &[2, 7], &[3, 8, 4, 6]];

        // Per-sequence reference: verify each one alone.
        let mut want = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let mut sk = SeqKv::new(&m.cfg);
            let lg = m.forward_prefill_seq(p, &mut sk, 0)?;
            let next = argmax(&lg[(p.len() - 1) * vocab..]) as i32;
            let draft: Vec<i32> = (0..i).map(|j| ((j + i) % vocab) as i32).collect();
            let mut one: [&mut SeqKv; 1] = [&mut sk];
            let (v, _h) = m.verify_drafts_batched(&[next], &[draft], &mut one, &[p.len()])?;
            want.push(v.into_iter().next().ok_or_else(|| Error::Format("no result".into()))?);
        }

        // Batched: all three, depths 0, 1 and 2, in one call.
        let mut seqs: Vec<SeqKv> = Vec::new();
        let mut next_of = Vec::new();
        let mut drafts = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let mut sk = SeqKv::new(&m.cfg);
            let lg = m.forward_prefill_seq(p, &mut sk, 0)?;
            next_of.push(argmax(&lg[(p.len() - 1) * vocab..]) as i32);
            drafts.push((0..i).map(|j| ((j + i) % vocab) as i32).collect::<Vec<i32>>());
            seqs.push(sk);
        }
        let pos_of: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
        let (got, hlast) = m.verify_drafts_batched(&next_of, &drafts, &mut refs, &pos_of)?;

        assert_eq!(got.len(), 3);
        // One hidden row per sequence, so the next round can draft from this
        // forward instead of re-running the stack.
        let d = m.cfg.hidden as usize;
        assert_eq!(hlast.len(), 3 * d, "a hidden row per sequence");
        assert!(hlast.iter().all(|v| v.is_finite()));
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_eq!(a, b, "sequence {i}: batched verification differs from verifying it alone");
        }
        for (i, (sk, p)) in seqs.iter().zip(prompts.iter()).enumerate() {
            assert_eq!(sk.len(), p.len() + 1 + got[i].accepted, "sequence {i} cache length");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_fused_chunk_is_indistinguishable_from_two_separate_forwards() -> Result<(), peregrine_core::Error> {
        // What the prefill/decode fusion buys, and the property it has to keep.
        //
        // On a tick with both, the engine runs `forward_prefill_seq` *and*
        // `forward_step_batched` — two disjoint forwards, each streaming its own
        // routed-expert union off disk, ~11.3 GB/token apiece at GLM-5.2 shapes.
        // They can be one call, because the MoE lane is row-batch-union'd and
        // does not care which sequence a row came from.
        //
        // "Can be" only if the fused call gives every row exactly what it would
        // have got alone. That is asserted here bit for bit, because if it does
        // not hold, fusing changes served output.
        let dir = tmp_model_dir("fused_rows")?;
        let m = Model::load(&dir)?;
        let vocab = m.cfg.vocab as usize;
        let (chunk, decode_tok) = ([1i32, 5, 9], 7i32);
        let decode_prompt = [2i32, 6, 3];

        // Reference: prefill sequence A alone, decode sequence B alone.
        let mut ref_a = SeqKv::new(&m.cfg);
        let want_a = m.forward_prefill_seq(&chunk, &mut ref_a, 0)?;
        let mut ref_b = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&decode_prompt, &mut ref_b, 0)?;
        let mut one: [&mut SeqKv; 1] = [&mut ref_b];
        let want_b = m.forward_step_batched(&[decode_tok], &mut one, &[decode_prompt.len()], None)?;

        // Fused: A's three prefill rows and B's one decode row, one forward.
        let mut fa = SeqKv::new(&m.cfg);
        let mut fb = SeqKv::new(&m.cfg);
        m.forward_prefill_seq(&decode_prompt, &mut fb, 0)?;
        let tokens = [chunk[0], chunk[1], chunk[2], decode_tok];
        let owner = [0usize, 0, 0, 1];
        let pos_of = [0usize, 1, 2, decode_prompt.len()];
        let mut refs: Vec<&mut SeqKv> = vec![&mut fa, &mut fb];
        let got = m.forward_rows_batched(&tokens, &owner, &mut refs, &pos_of, None)?;

        assert_eq!(got.len(), 4 * vocab);
        for (k, (p, q)) in want_a.iter().zip(&got[..3 * vocab]).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "prefill logit {k} moved when fused with a decode row");
        }
        for (k, (p, q)) in want_b.iter().zip(&got[3 * vocab..]).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "decode logit {k} moved when fused with a prefill chunk");
        }
        assert_eq!((fa.len(), fb.len()), (3, decode_prompt.len() + 1), "each cache advanced by its own rows");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn forward_rows_batched_reports_a_bad_owner_rather_than_indexing_past_it() -> Result<(), peregrine_core::Error> {
        // The fusion builds `owner` from scheduler state, so a scheduler bug
        // must fail the request, not index out of bounds — the release profile
        // aborts on panic and would take every concurrent sequence with it.
        let dir = tmp_model_dir("rows_guard")?;
        let m = Model::load(&dir)?;
        let mut sk = SeqKv::new(&m.cfg);
        let mut refs: Vec<&mut SeqKv> = vec![&mut sk];
        assert!(m.forward_rows_batched(&[1], &[3], &mut refs, &[0], None).is_err(), "owner past the end");
        assert!(m.forward_rows_batched(&[1, 2], &[0], &mut refs, &[0, 1], None).is_err(), "owner count mismatch");
        // …and the engine is still usable afterwards.
        assert!(m.forward_rows_batched(&[1], &[0], &mut refs, &[0], None).is_ok());
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
            m.forward_prefill_seq(p, &mut sk, 0)?;
            let mut one: [&mut SeqKv; 1] = [&mut sk];
            let pos = [p.len()];
            let lg = m.forward_step_batched(&[newtok[s]], &mut one, &pos, None)?;
            ref_logits[s * vocab..s * vocab + vocab].copy_from_slice(&lg);
        }

        // batched: prefill all three into fresh caches, then ONE batched decode
        let mut seqs: Vec<SeqKv> = Vec::new();
        for p in prompts.iter() {
            let mut sk = SeqKv::new(&m.cfg);
            m.forward_prefill_seq(p, &mut sk, 0)?;
            seqs.push(sk);
        }
        let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
        let toks: Vec<i32> = newtok.to_vec();
        let pos_of: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let bat = m.forward_step_batched(&toks, &mut refs, &pos_of, None)?;

        for z in 0..3 * vocab {
            assert!((ref_logits[z] - bat[z]).abs() < 1e-4, "z={z} ref={} bat={}", ref_logits[z], bat[z]);
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn router_lookahead_cannot_move_a_token() -> Result<(), peregrine_core::Error> {
        // The property that makes the look-ahead shippable on by default: it is a
        // scheduling change and nothing else. Layer L+1's router is run early to
        // decide what to *read*; the authoritative router still runs at L+1 and still
        // decides what is *computed*. So a streamed model — which has the prefetch
        // lane and therefore the look-ahead — must decode bit-identically to a
        // resident one, which has neither.
        let dir = tmp_model_dir("lookahead_exact")?;
        let mut streamed = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let mut resident = Model::load_streaming(&dir, false)?;
        assert!(streamed.lookahead_ctx().is_some(), "the streamed model must have a look-ahead to test");
        assert!(resident.lookahead_ctx().is_none(), "the resident model is the control: no lane, no look-ahead");

        let issued_before = lookahead_issued();
        let prompt = [3i32, 7, 1, 4];
        assert_eq!(streamed.forward_step(&prompt, 0)?, resident.forward_step(&prompt, 0)?, "prefill diverged");
        // Decode steps are what exercise it: the look-ahead is `s_n == 1` only.
        for (i, tok) in [5i32, 2, 9, 6].iter().enumerate() {
            let pos = prompt.len() + i;
            let a = streamed.forward_step(&[*tok], pos)?;
            let b = resident.forward_step(&[*tok], pos)?;
            assert_eq!(a, b, "decode step {i} diverged — the look-ahead moved a token");
        }
        // Smoke signal that the emit path ran at all, so the identity above is not
        // vacuously true of a look-ahead that never fired. The counter is
        // process-global and monotonic, so a parallel test can only inflate it —
        // this can false-pass, never false-fail. `rank` is pinned precisely by the
        // two tests below; this pins that `emit` reaches it from a real forward.
        assert!(
            lookahead_issued() > issued_before,
            "no speculative read was issued across four decode steps of a sparse model"
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn router_lookahead_ranks_by_the_next_layers_router() -> Result<(), peregrine_core::Error> {
        // The look-ahead's candidates must be layer L+1's router's own ranking of the
        // hidden state — not layer L's, which is the thing every history-based
        // predictor is stuck approximating. Asserting against `route` on the same
        // input is what distinguishes "asked the router" from "asked a statistic".
        let dir = tmp_model_dir("lookahead_ranks")?;
        let m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let d = m.cfg.hidden as usize;
        let next = m.cfg.first_dense as usize; // the first sparse layer
        assert!(m.layers[next].sparse, "fixture must have a sparse layer to rank");
        let x: Vec<f32> = (0..d).map(|i| ((i * 13 + 5) as f32 * 0.07).sin()).collect();

        let ranks = router_ranks_for(&m.layers[next], &m.cfg, &x, m.cfg.topk as usize);
        // The oracle: normalize with that layer's post-attention norm, then route.
        let nrm = rmsnorm_rows(&x, &m.layers[next].post_ln, 1, d, m.cfg.eps);
        let routed = crate::router::route(
            &nrm,
            &m.layers[next].router,
            &m.layers[next].router_bias,
            crate::router::RouterCfg {
                s_n: 1,
                d_n: d,
                e_n: m.cfg.n_experts as usize,
                k: m.cfg.topk as usize,
                norm_topk: m.cfg.norm_topk,
                routed_scale: m.cfg.routed_scale,
                min_share: 0.0,
            },
        );
        assert_eq!(ranks, routed.idx, "look-ahead must rank exactly as that layer's router does");
        // A short hidden state is refused rather than read out of bounds — the
        // look-ahead is advisory and must degrade, never panic.
        assert!(router_ranks_for(&m.layers[next], &m.cfg, &x[..2], 4).is_empty());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn router_lookahead_spends_its_window_on_experts_that_need_a_read() -> Result<(), peregrine_core::Error> {
        // Filtering happens *after* ranking, and the window slides rather than
        // shrinks: a candidate that is right but already warm costs no read, so it
        // must not consume one of the `width` slots. Get that backwards and a warm
        // cache silently narrows the look-ahead to nothing exactly when it is working.
        let dir = tmp_model_dir("lookahead_window")?;
        let m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        let la = m.lookahead_ctx().ok_or_else(|| Error::Format("streamed model has a look-ahead".into()))?;
        let cache = m.ecache.as_ref().ok_or_else(|| Error::Format("streamed model has a warm cache".into()))?;
        let d = m.cfg.hidden as usize;
        let next = m.cfg.first_dense as usize;
        let x: Vec<f32> = (0..d).map(|i| ((i * 13 + 5) as f32 * 0.07).sin()).collect();
        let ranked = router_ranks_for(&m.layers[next], &m.cfg, &x, m.cfg.n_experts as usize);
        assert_eq!(la.rank(&m.layers[next], next, &x, 1), vec![ranked[0]], "a cold cache takes the top rank");

        let warm = |e: i32| {
            let region = || (peregrine_io::Bytes::from(vec![0u8; 8]), peregrine_io::Bytes::from(vec![0u8; 4]));
            cache.lock().insert((next as u32, e as u32), [region(), region(), region()]);
        };
        // Warm the top-ranked candidate: the one slot must slide to rank 2, not go
        // unused. The scan reaches `max(width, topk)` ranks, so the slide stays
        // inside the width of a real routing decision.
        warm(ranked[0]);
        assert_eq!(la.rank(&m.layers[next], next, &x, 1), vec![ranked[1]], "the freed slot slides down the ranking");

        // Warm the whole scanned prefix and the look-ahead has nothing left to do.
        // It must issue *nothing* rather than reach past the routing width for a
        // candidate the layer is unlikely to route at all — a wrong speculative read
        // displaces a needed one, so silence is the correct output here.
        for &e in ranked.iter().take(m.cfg.topk as usize) {
            warm(e);
        }
        assert!(la.rank(&m.layers[next], next, &x, 1).is_empty(), "a fully warm prefix issues no read");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn the_router_lookahead_does_not_depend_on_routing_history() -> Result<(), peregrine_core::Error> {
        // The structural claim in `LookaheadCtx`'s doc: it reads the next layer's
        // routing off that layer's own weights, so unlike every predictor in
        // `predict.rs` it needs no history and is useful on the first token of a cold
        // process. A `LookaheadCtx` that could not be built without a `RouteHistory`
        // would quietly reintroduce the dependency it exists to avoid.
        let dir = tmp_model_dir("lookahead_no_history")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        m.route_hist = None;
        let la = m.lookahead_ctx().ok_or_else(|| Error::Format("look-ahead without history".into()))?;
        let d = m.cfg.hidden as usize;
        let next = m.cfg.first_dense as usize;
        let x: Vec<f32> = (0..d).map(|i| ((i * 3 + 1) as f32 * 0.11).cos()).collect();
        assert!(!la.rank(&m.layers[next], next, &x, 2).is_empty(), "a cold process still gets a ranking");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn predict_eval_scores_every_arm_over_a_decode() -> Result<(), peregrine_core::Error> {
        // End-to-end wiring for the scoreboard, set directly rather than through
        // `COLI_PREDICT_EVAL` because the env knob is read once per process and the
        // test suite shares one.
        let dir = tmp_model_dir("predict_eval")?;
        let mut m = Model::load_streaming_ecache(&dir, true, 8 << 20)?;
        m.predict_eval = Some(Mutex::new(crate::predeval::PredictEval::new(2, &PREDICT_EVAL_ARMS)));
        m.forward_step(&[3, 7, 1, 4], 0)?;
        assert!(m.predict_eval_report().is_none(), "prefill is not scored — its actual set is a union");
        for (i, tok) in [5i32, 2, 9].iter().enumerate() {
            m.forward_step(&[*tok], 4 + i)?;
        }
        let (arms, layers) = m
            .predict_eval_report()
            .ok_or_else(|| Error::Format("decode steps score layer transitions".into()))?;
        assert_eq!(arms.len(), PREDICT_EVAL_ARMS.len());
        assert!(layers > 0, "at least one transition scored");
        for (a, name) in arms.iter().zip(PREDICT_EVAL_ARMS) {
            assert_eq!(a.name, name, "arm order must match the stash order");
            assert!((0.0..=1.0).contains(&a.recall), "{name} recall out of range: {}", a.recall);
            assert!((0.0..=1.0).contains(&a.precision), "{name} precision out of range: {}", a.precision);
            assert_eq!(a.precision_at.len(), 2, "one precision figure per scored rank");
        }
        // The look-ahead needs no routing history, so it is never silent — that is
        // the structural advantage over both history-based arms, and it shows up on
        // the very first decode step of a cold process.
        assert_eq!(arms[0].silent, 0, "the router look-ahead always has an answer");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

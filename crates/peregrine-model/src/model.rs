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
use crate::concurrent::{default_workers, experts_per_batch, moe_forward_concurrent, ForwardCtx};
use crate::gpu::{GpuTier, HeatTable};
use crate::math::rmsnorm;
use crate::mlp::{moe_forward, Mlp};
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
    /// Per-layer routed-expert history (K-deep) from recent main forwards — the
    /// prefetch predictor's substrate. `Some` alongside `prefetch`.
    route_hist: Option<Mutex<RouteHistory>>,
    /// Strategy that turns [`Self::route_hist`] into a ranked list of experts to
    /// prefetch for the next forward. Defaults to recency-weighted momentum.
    predictor: PredictSource,
    /// Multi-path tiering: how many ranked candidates per layer to fully stream vs.
    /// merely page-cache-hint.
    prefetch_policy: PrefetchPolicy,
    /// Optional adaptive controller: when present, it overrides the warm-tier breadth
    /// each forward from observed prefetch used/wasted rates. `None` = static policy.
    prefetch_tuner: Option<PrefetchTuner>,
    /// Background prefetch lane: warms the next token's predicted experts into
    /// `ecache` on its own ring, off the critical path. `Some` alongside `ecache`.
    prefetch: Option<PrefetchPool>,
    /// Optional GPU VRAM expert tier (the 3rd lane). Built only when `COLI_GPU`
    /// is set and the `cuda` backend is available; `None` otherwise.
    gpu: Option<GpuTier>,
    /// Optional MTP head for speculative decode; `None` unless the checkpoint has
    /// the `model.layers.{n_layers}.eh_proj` tensors.
    mtp: Option<MtpHead>,
    /// Routing-frequency accumulator driving heat-ranked VRAM residency; `Some`
    /// only when a GPU tier exists (bumped during the forward, read by `reheat`).
    heat: Option<HeatTable>,
    /// Per-lane wall-time accumulator. `moe_forward_concurrent` bumps this each
    /// layer; the model reads and resets it between forwards to feed the bubble
    /// tuner. Always present (cheap — four atomic counters).
    lane_timings: Arc<crate::lane::LaneTimingsAccum>,
    /// Adaptive pipeline-bubble tuner. Reads per-forward [`LaneTimings`] snapshots
    /// and publishes a [`crate::lane::Bias`] the CPU/GPU balancer consumes.
    bubble: Mutex<crate::lane::BubbleTuner>,
    /// Last snapshot of per-forward per-lane wall time, for `/metrics` scrapes.
    last_lane: Mutex<crate::lane::LaneTimings>,
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
        self.layers.first().is_none_or(|k| k.len == 0)
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
    match std::env::var("COLI_ECACHE_GB") {
        Ok(v) => {
            let g: f64 = match v.trim().parse() {
                Ok(g) => g,
                Err(e) => {
                    peregrine_io::note_advisory_err("COLI_ECACHE_GB parse (cache disabled)", &e);
                    0.0
                }
            };
            return (g.max(0.0) * GIB) as usize;
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
fn spawn_prefetch_pool(cache: &Arc<Mutex<WarmCache>>, st: &SafeTensors, direct: bool, lanes: usize) -> Result<PrefetchPool, Error> {
    let lanes = lanes.max(1);
    let verify = matches!(std::env::var("COLI_PREFETCH_VERIFY").as_deref(), Ok("1") | Ok("true"));
    let mut handles = Vec::with_capacity(lanes);
    for i in 0..lanes {
        let mut reactor = Reactor::new(64).ctx(|| "prefetch io_uring reactor init".to_string())?;
        if direct {
            reactor.configure_slab(max_expert_region_bytes(st), 2);
        }
        let cache = Arc::clone(cache);
        let (tx, rx) = crossbeam_channel::unbounded::<PrefetchMsg>();
        let join = std::thread::Builder::new()
            .name(format!("peregrine-prefetch-{i}"))
            .spawn(move || {
                numa_pin_worker(i); // opt-in NUMA affinity (COLI_NUMA_PIN=1)
                prefetch_worker(reactor, cache, rx, direct, verify)
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

/// Whether the prefetch lane emits per-layer *during* the forward (look-ahead) so a
/// layer's read overlaps later layers' compute, vs. one bulk enqueue at the end of
/// the forward. On by default; disable with `COLI_PREFETCH_LOOKAHEAD=0`.
fn prefetch_lookahead() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_PREFETCH_LOOKAHEAD").as_deref(), Ok("0") | Ok("false")))
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

/// Whether predictive eviction is active: after each forward, resident experts the
/// predictor expects to be reused are protected from eviction. On by default (it only
/// reorders eviction victims, never output); disable with `COLI_PREFETCH_PROTECT=0`.
fn prefetch_protect() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_PREFETCH_PROTECT").as_deref(), Ok("0") | Ok("false")))
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
    std::fs::write(path, json)?;
    Ok(())
}

/// Write a built macro-state table to `path` — the `macrostates.json` artifact a
/// model auto-loads (and blends into the predictor) from its checkpoint dir.
pub fn save_macrostates(table: &crate::predict::MacroTable, path: &std::path::Path) -> Result<(), Error> {
    let json = serde_json::to_vec(&table.to_json()).map_err(|e| Error::Format(format!("serialize macrostates: {e}")))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Whether cross-session routing-history persistence is on. Default on when the
/// model dir is writable; disable with `COLI_ROUTE_STATS_PERSIST=0`. Correctness-
/// neutral (heat and route history only affect prefetch/eviction, not output).
fn route_stats_persist_enabled() -> bool {
    !matches!(std::env::var("COLI_ROUTE_STATS_PERSIST").as_deref(), Ok("0") | Ok("false"))
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
            if self.tick.saturating_sub(self.last_probe) >= 256 && current < self.base_workers {
                grow = true; // periodic probe upward
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
    /// Multi-path tiering: the top `warm_paths` ranked candidates per layer are fully
    /// streamed (tier 1); the next `hint_paths` get a page-cache `fadvise` hint (tier 2).
    warm_paths: usize,
    hint_paths: usize,
    /// Under O_DIRECT the page cache is bypassed, so tier-2 hints are pointless and
    /// suppressed.
    direct: bool,
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
                    match crate::concurrent::prefetch_item(self.st, self.cfg, layer, e as usize) {
                        Ok(item) => warms.push(item),
                        // speculative: the real forward will stream this expert normally
                        Err(e) => peregrine_io::note_advisory_err("prefetch item resolve", &e),
                    }
                } else if rank < hint_cutoff && !self.direct {
                    match crate::concurrent::prefetch_hint_item(self.st, self.cfg, layer, e as usize) {
                        Ok(item) => hints.push(item),
                        Err(e) => peregrine_io::note_advisory_err("prefetch hint resolve", &e),
                    }
                } else {
                    break; // beyond both tiers — lower-ranked candidates ignored
                }
                rank += 1;
            }
        }
        if !warms.is_empty() && self.prefetch.tx.send(PrefetchMsg::Warm(warms)).is_err() {
            peregrine_io::note_advisory_err("prefetch warm dispatch", &"prefetch lane is down");
        }
        if !hints.is_empty() && self.prefetch.tx.send(PrefetchMsg::Hint(hints)).is_err() {
            peregrine_io::note_advisory_err("prefetch hint dispatch", &"prefetch lane is down");
        }
    }
}

/// The prefetch lane: stream predicted experts into the shared warm cache on this
/// lane's *own* ring (no contention with the critical I/O lane). Best-effort — a
/// failed speculative read is dropped (the real forward will stream it normally).
fn prefetch_worker(
    mut reactor: Reactor,
    cache: Arc<Mutex<WarmCache>>,
    rx: crossbeam_channel::Receiver<PrefetchMsg>,
    direct: bool,
    verify: bool,
) {
    while let Ok(msg) = rx.recv() { // audit-allow: recv's only Err is Disconnected = clean shutdown
        match msg {
            PrefetchMsg::Warm(items) => {
                for item in items {
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
        let src = x[s * d..s * d + d].to_vec();
        rmsnorm(row, &src, w, eps);
    });
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
        let stream_experts = if stream_experts && st.has_compressed_tensors() {
            eprintln!("[peregrine] compressed checkpoint detected — disabling expert streaming (compressed reads decompress on read)");
            false
        } else {
            stream_experts
        };

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
            Some(cache) => (
                Some(Mutex::new(RouteHistory::new(cfg.n_layers as usize, route_hist_depth()))),
                Some(spawn_prefetch_pool(cache, &st, direct, prefetch_lanes())?),
            ),
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

        let model_n_layers = cfg.n_layers as usize; // read before `cfg` moves into the struct
        let mut model = Model {
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
            predictor: PredictSource::default(),
            prefetch_policy: PrefetchPolicy::from_env(),
            prefetch_tuner: prefetch_tuner_init(),
            prefetch,
            gpu,
            mtp,
            heat,
            lane_timings: Arc::new(crate::lane::LaneTimingsAccum::new()),
            bubble: Mutex::new(crate::lane::BubbleTuner::new(0.3, 1.5, 3)),
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
            layout_schedule: load_layout_schedule(dir),
        };
        // Upgrade the predictor to the offline transition automaton if a matching
        // `automaton.json` sits next to the checkpoint (else stay on momentum),
        // then blend in the macro-state table if one is present.
        model.try_attach_automaton(dir);
        model.try_attach_macrostates(dir);
        // Load any persisted routing history / heat snapshot so a fresh process
        // starts warm on the last session's routing patterns. Correctness-neutral;
        // missing/stale files are silently ignored.
        model.try_load_route_stats(dir);
        // Storage-tier seed: prefetch-warm the offline-planned RAM tier so the
        // co-firing communities the planner placed in RAM are resident before
        // the first token. Best-effort; bounded; `COLI_TIER_SEED=0` disables.
        model.try_seed_tiers(dir);
        // Compiled execution plan (plan.json from `compile-plan`) — a single
        // profile-guided artifact bundling all of the above; applied last so a
        // plan wins over the standalone files.
        model.try_load_plan(dir);
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
        let Ok(bytes) = std::fs::read(dir.join("plan.json")) else { return };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
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
        let Ok(bytes) = std::fs::read(dir.join("tiers.json")) else { return };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
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
            match crate::concurrent::prefetch_item(&self.st, &self.cfg, l as usize, e as usize) {
                Ok(item) => items.push(item),
                // seed warming is speculative; a bad tier entry is skipped
                Err(e) => peregrine_io::note_advisory_err("tier-seed prefetch resolve", &e),
            }
        }
        if !items.is_empty() && pool.lane(0).tx.send(PrefetchMsg::Warm(items)).is_err() {
            peregrine_io::note_advisory_err("prefetch warm dispatch", &"prefetch lane is down");
        }
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
            h.lock().clear();
        }
        // A new sequence starts from pure LRU until its predictor re-protects experts.
        if let Some(c) = &self.ecache {
            c.lock().clear_priorities();
        }
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
        self.prefetch_policy.for_class(*self.workload_class.lock())
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
                warm_paths: policy.warm_paths,
                hint_paths: policy.hint_paths,
                direct: self.direct,
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
        let hist = hist.lock();
        let mut cache = cache.lock();
        for layer in first_dense..n_layers {
            for (e, score) in self.predictor.predict_layer(layer, &hist) {
                let h = heat.as_ref().and_then(|c| c.get(layer * n_experts + e as usize).copied()).unwrap_or(0);
                cache.set_priority((layer as u32, e), pack_prio(score, h));
            }
        }
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

    /// Prefetch-lane reads the warm tier has attributed to `layer` (lets the
    /// look-ahead test confirm early layers were warmed mid-forward).
    pub fn ecache_prefetch_reads_for_layer(&self, layer: usize) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().prefetch_reads_for_layer(layer as u32))
    }

    /// Low-confidence experts the prefetch lane hinted to the page cache via
    /// `fadvise` (multi-path tier 2).
    pub fn ecache_fadvise_hints(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().fadvise_hints)
    }

    /// Speculative reads whose opt-in verification re-read differed (always 0 in a
    /// correct system; nonzero signals an I/O bug).
    pub fn ecache_verify_mismatch(&self) -> Option<u64> {
        self.ecache.as_ref().map(|c| c.lock().verify_mismatch)
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
            warm_paths: policy.warm_paths,
            hint_paths: policy.hint_paths,
            direct: self.direct,
        };
        for layer in (self.cfg.first_dense as usize)..(self.cfg.n_layers as usize) {
            ctx.emit_layer(layer);
        }
        if prefetch_protect() {
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
                match crate::concurrent::prefetch_item(&self.st, &self.cfg, layer, e) {
                    Ok(item) => items.push(item),
                    Err(e) => peregrine_io::note_advisory_err("warm-list prefetch resolve", &e),
                }
            }
        }
        if !items.is_empty() && pool.lane(0).tx.send(PrefetchMsg::Warm(items)).is_err() {
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
        let Ok(bytes) = std::fs::read(dir.join("automaton.json")) else {
            return;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return;
        };
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
        let Ok(bytes) = std::fs::read(dir.join("macrostates.json")) else {
            return;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return;
        };
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
        let Ok(bytes) = std::fs::read(dir.join("route_stats.json")) else {
            return;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return;
        };
        let tag = config_tag(&self.cfg);
        if let Some(hist_v) = v.get("hist") {
            if let (Some(hist), Some(new_hist)) = (self.route_hist.as_ref(), RouteHistory::from_json(hist_v, &tag)) {
                *hist.lock() = new_hist;
            }
        }
        if let Some(heat_v) = v.get("heat") {
            if let (Some(heat), Some(arr)) = (self.heat.as_ref(), heat_v.as_array()) {
                let snap: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
                heat.restore(&snap);
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
    pub fn checkpoint_dir(&self) -> &std::path::Path {
        &self.checkpoint_dir
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
        self.bubble.lock().observe(sample);
        // Sensor governors: thermal / power / bandwidth, all writing the one
        // effective-worker knob with shrink-wins arbitration.
        {
            use std::sync::atomic::Ordering;
            let current = self.effective_workers.load(Ordering::Relaxed);
            let delta = self.governor.lock().step(sample, current);
            if delta != 0 {
                let next = (current as i64 + delta as i64).clamp(2, self.workers as i64) as usize;
                self.effective_workers.store(next, Ordering::Relaxed);
            }
        }
        // Routing-entropy EWMA (drives entropy-adaptive prefetch breadth; the
        // nudge itself happens in forward_hidden where the tuner is &mut).
        if entropy_adapt_enabled() {
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
                let cur = crate::learn::KnobArm {
                    prefetch_distance: self.learned_prefetch.load(Ordering::Relaxed).max(4),
                    workers: self.effective_workers.load(Ordering::Relaxed),
                };
                let next = l.choose(bias, entropy, cur, self.workers);
                self.learned_prefetch.store(next.prefetch_distance, Ordering::Relaxed);
                let w = next.workers.clamp(2, self.workers);
                self.effective_workers.store(w, Ordering::Relaxed);
            }
        }
        // Co-activation tracking: fold this forward's routed sets (newest frame
        // per layer) into the pair counter; every 64 forwards rebuild the
        // affinity snapshot (fused pairs at the high threshold, hyperedge
        // components at half that). Single-stream mode only — the batched path
        // records per-sequence histories the engine owns, not `route_hist`.
        if let Some(hist) = &self.route_hist {
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
        if sample.io_us > 0 {
            let sq_full_delta: u64 = self.io_reactors.iter().map(|r| r.lock().take_sq_full()).sum();
            let before = self.io_tuner.sq_full();
            self.io_tuner.note_read(sample.io_us, sq_full_delta);
            self.io_tuner.step(10_000, before);
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
        if let Err(e) = std::fs::write(dir.join("route_stats.json"), bytes) {
            peregrine_io::note_advisory_err("persist route_stats.json", &e);
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
    #[allow(clippy::type_complexity)]
    pub fn build_artifacts(
        &mut self,
        corpus: &[i32],
    ) -> Result<(TransitionTable, crate::predict::MacroTable, Vec<Vec<Vec<i32>>>), Error> {
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
        std::fs::write(path, json)?;
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
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }

        let lookahead = prefetch_lookahead();
        // Per-class breadth first (Copy; read before the disjoint destructure),
        // then the adaptive controller's cap wins over both when active.
        let mut policy = self.class_policy();
        if let Some(t) = &self.prefetch_tuner {
            policy.warm_paths = t.distance(); // adaptive controller caps the warm tier
        }
        // Run the stack in a block so the split borrows of `self` end before we
        // re-borrow `self` to enqueue prefetch.
        {
            // split disjoint fields so attention can borrow layers (imm) + kv (mut)
            let balancer = self.build_balancer();
            let heat_snapshot = balancer.as_ref().and_then(|_| self.heat.as_ref().map(|h| h.snapshot()));
            let eff_workers = self.effective_workers();
            let aff = self.affinity_snapshot();
            let Model {
                cfg, layers, kv, st, stream_experts, direct, io_reactors, ecache, route_hist, predictor, prefetch, gpu, heat, lane_timings, layout_schedule, ..
            } = self;
            let ctx = ForwardCtx {
                st,
                reactors: io_reactors,
                gpu: gpu.as_ref(),
                workers: eff_workers,
                cfg,
                stream_experts: *stream_experts,
                ecache: ecache.as_deref(),
                route_log: route_hist.as_ref(),
                route_log_multi: None,
                direct: *direct,
                heat: heat.as_ref(),
                timings: Some(lane_timings.as_ref()),
                balancer: balancer.as_ref(),
                heat_counts: heat_snapshot.as_deref(),
                layout_schedule: layout_schedule.as_deref(),
                affinity: Some(aff.as_ref()),
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
                    warm_paths: policy.warm_paths,
                    hint_paths: policy.hint_paths,
                    direct: *direct,
                }),
                _ => None,
            };
            for (li, l) in layers.iter().enumerate() {
                forward_layer(l, li, &mut kv[li], &ctx, &mut x, s_n, pos_base)?;
                if let Some(pfc) = &pfc {
                    pfc.emit_layer(li);
                }
            }
        }
        // When look-ahead is off, fall back to one bulk next-token enqueue after the
        // forward (main forward only).
        if !lookahead {
            self.enqueue_prefetch();
        }
        // Predictive eviction: protect the experts we expect to reuse next.
        if prefetch_protect() {
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

    /// Build the per-forward compute context from the resident model state with
    /// prefetch/route-logging **disabled** — the shape the external-KV batched and
    /// prefill paths use (the B-way expert union is not a useful next-token
    /// predictor, so prefetch is gated off under batching).
    fn forward_ctx(&self) -> ForwardCtx<'_> {
        ForwardCtx {
            st: &self.st,
            reactors: &self.io_reactors,
            gpu: self.gpu.as_ref(),
            workers: self.effective_workers(),
            cfg: &self.cfg,
            stream_experts: self.stream_experts,
            ecache: self.ecache.as_deref(),
            route_log: None,
            route_log_multi: None,
            direct: self.direct,
            heat: self.heat.as_ref(),
            timings: None,
            balancer: None,
            heat_counts: None,
            layout_schedule: self.layout_schedule.as_deref(),
            affinity: None,
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
        if let Some(h) = histories {
            if h.len() != s_n {
                return Err(Error::Format(format!("forward_step_batched: {s_n} tokens but {} histories", h.len())));
            }
        }
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }
        // Built inline (not via `forward_ctx`) so the per-sequence history borrow and
        // the model borrows share one inferred lifetime.
        let balancer = self.build_balancer();
        let heat_snapshot = balancer.as_ref().and_then(|_| self.heat.as_ref().map(|h| h.snapshot()));
        let aff = self.affinity_snapshot();
        let ctx = ForwardCtx {
            st: &self.st,
            reactors: &self.io_reactors,
            gpu: self.gpu.as_ref(),
            workers: self.effective_workers(),
            cfg: &self.cfg,
            stream_experts: self.stream_experts,
            ecache: self.ecache.as_deref(),
            route_log: None,
            route_log_multi: histories,
            direct: self.direct,
            heat: self.heat.as_ref(),
            timings: Some(self.lane_timings.as_ref()),
            balancer: balancer.as_ref(),
            heat_counts: heat_snapshot.as_deref(),
            layout_schedule: self.layout_schedule.as_deref(),
            affinity: Some(aff.as_ref()),
        };
        for (li, l) in self.layers.iter().enumerate() {
            let mut caches: Vec<&mut LayerKv> = seqs.iter_mut().map(|sk| &mut sk.layers[li]).collect();
            forward_layer_batched(l, li, &mut caches, &ctx, &mut x, s_n, pos_of)?;
        }
        self.publish_lane_timings();
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
            route_log_multi: None,
            direct: *direct,
            heat: None, // speculative drafts must not skew residency heat
            timings: None, // drafts must not skew the main-stream lane balance
            balancer: None, // drafts run under the plain static residency policy
            heat_counts: None,
            layout_schedule: None, // drafts benefit less from disk-order tuning
            affinity: None,
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
        std::fs::write(dir.join("schedule.json"), serde_json::to_vec(&doc)?)?;
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
        assert_eq!(stream_transient_reserve(4, 8, 1000), (4 * experts_per_batch() + 8) * 1000);
        assert_eq!(stream_transient_reserve(0, 0, 1000), 0); // no lanes → no reserve
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
}

//! The concurrent MoE lane (M4): the throughput centerpiece.
//!
//! Per sparse layer, the batch-union of routed experts is streamed from NVMe
//! through **io_uring** (the I/O lane) while a **core-count CPU worker pool**
//! computes each expert's SwiGLU as soon as its weights land — so disk reads and
//! matmuls overlap instead of running phased. An [`AtomicUsize`] tracks completion.
//!
//! Determinism is preserved: workers compute per-expert partials independently
//! (no shared-row races), and the final scatter/reduce runs single-threaded in a
//! fixed (batch-union) order — so the concurrent output is **bit-identical** to
//! the sequential path. This is the CPU∥SSD design; the GPU lane composes the
//! same way (a third producer feeding the same reduce).

use std::collections::HashMap;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use peregrine_core::{Cfg, Context, Error, QtInfo, SafeTensors};
use peregrine_io::{Bytes, CacheHit, OwnedReadReq, Reactor, ReadReq, RegionDone, WarmCache};

use crate::gpu::{GpuDenseTier, GpuTier, HeatTable};
use crate::lane::LaneTimingsAccum;
use crate::mlp::Mlp;
use crate::predict::RouteHistory;
use crate::router::{batch_union, route, routed_at, RouterCfg};
use crate::weight::{QtWeight, QuantFmt};

/// Shared per-forward state threaded through the layer/MoE compute: the
/// safetensors index, the streaming io_uring ring, the GPU tier, the CPU-lane
/// width, the config, and whether experts stream from disk. Passed by reference
/// so the layer/MoE entry points stay small (no long argument lists).
pub struct ForwardCtx<'a> {
    pub st: &'a SafeTensors,
    /// Run MLA attention through weight absorption instead of the dense
    /// reconstruction (`COLI_MLA_ABSORB`). Carried on the context rather than
    /// read from the environment per forward, so tests can exercise both paths
    /// without mutating process-global state that parallel tests share.
    pub absorb: bool,
    /// Run the DSA lightning indexer where a layer carries one (`COLI_DSA`).
    /// Carried here for the same reason `absorb` is: so a test can exercise the
    /// sparse path without mutating process-global state parallel tests share.
    pub dsa: bool,
    /// A **pool of io_uring rings** for the I/O lane — one dedicated ring per I/O
    /// worker thread, so N reads proceed in parallel (each ring is locked only by
    /// its owner, so the lock is uncontended). Empty in resident mode.
    pub reactors: &'a [Mutex<Reactor>],
    pub gpu: Option<&'a GpuTier>,
    /// VRAM-resident dense MLPs (Track D). `Some` only for dense/hybrid
    /// architectures with `COLI_GPU_DENSE` on and at least one layer uploaded;
    /// a layer it does not hold computes on the CPU exactly as before.
    pub gpu_dense: Option<&'a GpuDenseTier>,
    pub workers: usize,
    pub cfg: &'a Cfg,
    pub stream_experts: bool,
    /// RAM warm tier consulted by the I/O lane before streaming (streaming mode
    /// only). A hit returns the exact previously-streamed bytes, so output is
    /// bit-identical; a miss streams then inserts. `None` disables caching.
    pub ecache: Option<&'a Mutex<WarmCache>>,
    /// Per-layer routing history: after each layer's reduce, this forward's
    /// batch-union of routed experts is pushed as the newest frame. The prefetch
    /// lane's predictor reads it to guess the next token's experts. `None` on
    /// speculative-draft forwards (so drafts don't pollute the main-stream
    /// prediction) and when prefetch is off.
    pub route_log: Option<&'a Mutex<RouteHistory>>,
    /// Calibration capture (`COLI_CALIB_CAPTURE`, ideas #7): each sparse
    /// layer's MoE-input rows fold into per-channel `Σ|x|` right before the
    /// router runs. `None` in serving (the env is only set on capture runs)
    /// and on draft forwards — drafts replay positions the main stream
    /// already accumulated, so counting them would double-weight whatever the
    /// drafter happened to explore.
    pub calib: Option<&'a Mutex<crate::model::CalibAccum>>,
    /// Per-**row** routing history for batched decode: `route_log_multi[r]` receives
    /// row `r`'s own routed set, so each concurrent stream predicts and prefetches
    /// from its *own* routing rather than the weak cross-sequence union. `None` on the
    /// single-stream path (which uses `route_log`).
    ///
    /// Indexed by row, **not** by sequence — the two differ whenever a sequence
    /// contributes more than one row (speculative drafts, a fused prefill chunk), and
    /// this said "per-sequence" until 2026-08-08 while the write loop below indexed by
    /// row. Mapping sequences onto rows is the caller's job:
    /// `peregrine-serve`'s `batch.rs` expands one entry per sequence into `1 + drafts`
    /// entries, pointing the speculated rows at a scratch history so a rejected draft
    /// never reaches the predictor. `forward_rows_inner` requires `len() == s_n`.
    pub route_log_multi: Option<&'a [&'a Mutex<RouteHistory>]>,
    /// Routed sets **with their gate weights**, for trace capture
    /// (`Model::enable_gate_trace`). `None` on every production path.
    ///
    /// Separate from `route_log` because `RouteHistory` stores `batch_union` —
    /// expert ids with the weights discarded, which is why no artifact the
    /// engine writes has ever carried gate mass, and why `peregrine-prune`'s
    /// Σ-gate-weight saliency silently degrades to counting.
    pub gate_trace: Option<&'a Mutex<crate::model::GateTrace>>,
    /// Stream expert reads via O_DIRECT (bypass the page cache) when the shards
    /// opened O_DIRECT fds. Bytes are identical to the buffered path; only the
    /// cache behavior differs. `false` disables (buffered reads).
    pub direct: bool,
    /// Routing-frequency accumulator for heat-ranked VRAM residency: bumped once
    /// per routed expert per layer so [`crate::gpu::GpuTier::reheat`] can migrate
    /// hot experts into VRAM. `None` disables accumulation (no GPU tier / drafts).
    pub heat: Option<&'a HeatTable>,
    /// Per-lane wall-time accumulator: I/O, CPU, GPU, and reduce phases bump
    /// this from within `moe_forward_concurrent`. The Model reads and resets
    /// it between forwards so the `BubbleTuner` sees per-forward deltas.
    /// `None` disables the (very cheap) bracketing.
    pub timings: Option<&'a LaneTimingsAccum>,
    /// Deferred-spill log (`COLI_GPU_SPILL`): a [`crate::lane::Placement::GpuSpill`]
    /// verdict records its `(layer, expert)` here — the expert still computes on
    /// the CPU lane *this* forward — and [`crate::Model::reheat`] drains the log
    /// between forwards, boosting those pairs in the heat snapshot so the next
    /// residency generation actually uploads what the balancer kept asking for.
    /// Acting between forwards is what keeps this out of the mid-forward
    /// `&mut GpuTier` problem the `GpuSpill` doc describes, at the cost of the
    /// spill paying off next generation instead of this token. `None` = the
    /// verdict stays advisory (the historical behavior).
    pub spill: Option<&'a Mutex<Vec<(usize, usize)>>>,
    /// Optional adaptive CPU/GPU lane balancer. When `Some` (i.e. bias is
    /// non-`Balanced` and `COLI_LANE_BALANCE=1`), the scheduler downgrades cold
    /// GPU-resident experts to the CPU lane when the GPU is the bottleneck.
    /// Correctness-neutral: an expert served on the CPU lane produces the same
    /// bytes (either through the warm cache or a fresh stream) as it would on
    /// the GPU lane, and the reduce order stays fixed.
    pub balancer: Option<&'a crate::lane::LaneBalancer>,
    /// Optional heat snapshot the balancer consults for per-expert decisions.
    /// One flat `[layer * n_experts + expert]` slice; `None` disables balancing.
    pub heat_counts: Option<&'a [u32]>,
    /// Optional per-layer expert-order hint (from `<dir>/schedule.json`, emitted
    /// by `peregrine-layout-reorg`). When present, the streamed `EPlan`s for a
    /// layer are sorted by the schedule's rank of each expert id so the batched
    /// io_uring submit issues contiguous-offset reads first. Bit-identical to
    /// the natural-id order (only the read submission order changes).
    pub layout_schedule: Option<&'a [Vec<u32>]>,
    /// Optional co-activation affinity hints (runtime expert fusion +
    /// hypergraph scheduling): fused pairs kept adjacent in dispatch order,
    /// hyperedge components grouped into the same io-batch claim window.
    /// Bit-identical — only submission/claim order changes.
    pub affinity: Option<&'a AffinityHints>,
    /// Load-time `(layer, expert)` → plans/extents map. When present, expert
    /// lookup is a bounds-checked index instead of re-deriving the tensor
    /// locations and quantized format per request. `None` falls back to
    /// [`tplan`], which is what every path did before the index existed —
    /// same bytes either way.
    pub expert_index: Option<&'a ExpertIndex>,
    /// fd → device-ordinal table for **device-pure io claims**
    /// (`COLI_IO_DEVICE_SCHED`, read at model build — not OnceLock-latched).
    /// When `Some` and more than one ring exists, the claim order is
    /// partitioned by device so no deep submit mixes devices: a mixed batch
    /// reaps behind the slowest device (`submit_and_wait` completes the whole
    /// claim), which is the suspected mechanism behind 0.86 GB/s delivered
    /// against a ~1.42 GB/s predicted aggregate on the 3-device r4 split.
    /// Ordinals are opaque group keys (the Seam 1 contract in the 2026-08-15
    /// coordination file); fds absent from the table group under `u8::MAX`.
    /// `None` = the historical device-blind single cursor. Bit-identical either
    /// way — only claim/submission order changes; the reduce keys on `pos`.
    pub fd_devices: Option<&'a HashMap<RawFd, u8>>,
}

/// Per-layer co-activation ordering hints, rebuilt periodically by the model
/// from the [`crate::predict::CoActivation`] tracker.
#[derive(Default)]
pub struct AffinityHints {
    /// Per-layer fused expert pairs (co-rate ≥ the fusion threshold): the
    /// scheduler keeps each pair adjacent in the dispatch order.
    pub pairs: Vec<Vec<(u32, u32)>>,
    /// Per-layer hyperedge membership (expert → component id) at the lower
    /// hypergraph threshold: members are grouped contiguously so one claim
    /// window covers a whole co-firing group. Consulted only under
    /// `COLI_HYPER_SCHED=1`.
    pub groups: Vec<std::collections::HashMap<u32, u32>>,
}

/// Whether hypergraph (component-grouped) scheduling is on. `COLI_HYPER_SCHED=1`.
fn hyper_sched_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_HYPER_SCHED").as_deref(), Ok("1") | Ok("true")))
}

/// Apply the affinity ordering to a layer's streamed plans: hyperedge grouping
/// first (contiguous components, first-appearance order, non-members after in
/// original order), then fused-pair adjacency (partner moved right after its
/// mate). Stable throughout, so untouched experts keep their relative order.
fn apply_affinity_order(plans: &mut Vec<EPlan>, layer: usize, aff: &AffinityHints) {
    if hyper_sched_enabled() {
        if let Some(gmap) = aff.groups.get(layer) {
            if !gmap.is_empty() {
                // decorate–stable-sort–undecorate; group rank = first appearance.
                let mut rank: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
                let mut next = 0usize;
                let keys: Vec<usize> = plans
                    .iter()
                    .map(|p| match gmap.get(&(p.expert as u32)) {
                        Some(&g) => *rank.entry(g).or_insert_with(|| {
                            let k = next;
                            next += 1;
                            k
                        }),
                        None => usize::MAX,
                    })
                    .collect();
                let mut zipped: Vec<(usize, usize, EPlan)> =
                    plans.drain(..).enumerate().map(|(i, p)| (keys[i], i, p)).collect();
                zipped.sort_by_key(|&(k, i, _)| (k, i));
                plans.extend(zipped.into_iter().map(|(_, _, p)| p));
            }
        }
    }
    if let Some(pairs) = aff.pairs.get(layer) {
        for &(a, b) in pairs {
            let pos_a = plans.iter().position(|p| p.expert as u32 == a);
            let pos_b = plans.iter().position(|p| p.expert as u32 == b);
            if let (Some(ia), Some(ib)) = (pos_a, pos_b) {
                if ib != ia + 1 && ia != ib {
                    let moved = plans.remove(ib);
                    // removing before `ia` shifts it left by one
                    let dst = if ib < ia { ia } else { ia + 1 };
                    plans.insert(dst.min(plans.len()), moved);
                }
            }
        }
    }
}

/// Default CPU-lane width: the machine's parallelism, capped so a huge core
/// count doesn't oversubscribe memory bandwidth on the quantized kernels.
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map(|n| n.get().min(16)).unwrap_or(4)
}

/// One on-disk quantized tensor region + the shape/format to rebuild it.
#[derive(Clone, Copy)]
struct TPlan {
    w_fd: RawFd,
    w_off: u64,
    w_len: usize,
    s_fd: RawFd,
    s_off: u64,
    s_len: usize,
    /// O_DIRECT twin fds for the weight/scale regions (same offsets/lengths), when
    /// available. Used by the direct read path; `None` ⇒ that region reads buffered.
    w_fd_direct: Option<RawFd>,
    s_fd_direct: Option<RawFd>,
    fmt: QuantFmt,
    o: usize,
    i: usize,
    gs: usize,
}

/// One contiguous on-disk span covering several adjacent regions at once.
#[derive(Clone, Copy)]
struct Extent {
    fd: RawFd,
    /// O_DIRECT twin for `fd`, when every merged region had one.
    fd_direct: Option<RawFd>,
    off: u64,
    len: usize,
}

/// One routed expert, fully resolved once at load: its three tensor plans — which
/// carry the on-disk regions *and* the quantized format and group size, the
/// "type" a request needs — plus the merged extents that let six reads become two.
///
/// Measured on the GLM-5.2 container: an expert's three weight regions form one
/// contiguous run (18,874,368 bytes at int4, 37,748,736 on the int8 MTP layer)
/// and its three `.qs` scales form another, for all but the ~0.45 % that straddle
/// a shard boundary. Those keep `None` and read their six regions as before.
#[derive(Clone, Copy)]
struct ExpertEntry {
    /// gate, up, down — in the order the read path expects them, which is **not**
    /// the order they sit on disk (that is alphabetical: down, gate, up). Every
    /// split of a merged extent must therefore be computed from each plan's own
    /// offset, never from position in this array.
    plans: [TPlan; 3],
    w_run: Option<Extent>,
    s_run: Option<Extent>,
}

/// Merge one expert's three weight (or three scale) regions into a single extent
/// when they are adjacent on one fd. Returns `None` the moment the run breaks —
/// a different shard, a gap, or an overlap — so a straddling expert falls back to
/// the unmerged six regions rather than reading the wrong bytes.
fn merge_run(plans: &[TPlan; 3], weights: bool) -> Option<Extent> {
    let part = |t: &TPlan| {
        if weights {
            (t.w_fd, t.w_fd_direct, t.w_off, t.w_len)
        } else {
            (t.s_fd, t.s_fd_direct, t.s_off, t.s_len)
        }
    };
    let mut r: [(RawFd, Option<RawFd>, u64, usize); 3] = [part(&plans[0]), part(&plans[1]), part(&plans[2])];
    r.sort_by_key(|&(fd, _, off, _)| (fd, off));
    let (fd, fd_direct, off, _) = r[0];
    let mut end = off;
    for &(f, fd_d, o, l) in r.iter() {
        // Same shard, exactly abutting, and agreeing about the O_DIRECT twin — a
        // merged read issues against one fd, so a region whose twin differs
        // cannot be folded in.
        if f != fd || o != end || fd_d != fd_direct {
            return None;
        }
        end = o.checked_add(l as u64)?;
    }
    Some(Extent { fd, fd_direct, off, len: usize::try_from(end.checked_sub(off)?).ok()? })
}

/// Whether to coalesce an expert's adjacent regions into one read per run.
/// `COLI_EXPERT_MERGE=0` reverts to six reads per expert without reverting the
/// expert map, so the two can be A/B'd against a bit-identity assertion.
fn expert_merge_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_EXPERT_MERGE").as_deref(), Ok("0")))
}

/// Load-time map from `(layer, expert)` to its resolved tensor plans.
///
/// This replaces re-deriving both on every request. [`tplan`] costs four
/// `format!` allocations, a [`QtInfo::detect`] (which re-infers the quantized
/// format from byte counts, group-size probe loop included) and ~7 hash probes,
/// and was paid per expert, per sparse layer, per forward, at all three call
/// sites — the demand path, [`prefetch_item`] and [`prefetch_hint_item`].
///
/// Dense `Vec` indexed `layer * n_experts + expert` — the same flattening
/// `heat_counts` already uses — so a lookup is a bounds check rather than a hash.
/// Entries are `None` for dense layers and for any expert whose tensors do not
/// resolve; both fall back to the original [`tplan`] path, so an unusual
/// container behaves exactly as it did before this existed.
pub struct ExpertIndex {
    n_experts: usize,
    entries: Vec<Option<ExpertEntry>>,
}

impl ExpertIndex {
    /// Resolve every routed expert once, up front.
    ///
    /// Best-effort by design: an expert whose tensors are missing or unquantized
    /// stores `None` instead of failing the load, because the caller falls back
    /// to [`tplan`] and would raise the identical error there. Building it here
    /// must not turn a container that used to run into one that will not load.
    pub fn build(st: &SafeTensors, cfg: &Cfg) -> ExpertIndex {
        let n_experts = cfg.n_experts.max(0) as usize;
        let n_layers = cfg.n_layers.max(0) as usize;
        let first_dense = cfg.first_dense.clamp(0, cfg.n_layers) as usize;
        let hidden = cfg.hidden as usize;
        let mi = cfg.moe_inter as usize;
        // `n_layers + 1` rows, and the loop is inclusive of `n_layers`: the MTP
        // head sits at layer index `cfg.n_layers` and carries a full set of
        // routed experts. On this container those 256 are stored at **int8**
        // while every other sparse layer is int4, so an exclusive bound would
        // leave the one layer whose format differs unindexed — exactly the case
        // the map exists to get right. (`HeatTable` has the off-by-one this
        // avoids: it is sized `n_layers × n_experts`, so the MTP layer's experts
        // have never accumulated heat.)
        let mut entries = vec![None; (n_layers + 1).saturating_mul(n_experts)];
        for layer in first_dense..=n_layers {
            for e in 0..n_experts {
                let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
                let plans = match (
                    tplan(st, &p("gate_proj.weight"), mi, hidden),
                    tplan(st, &p("up_proj.weight"), mi, hidden),
                    tplan(st, &p("down_proj.weight"), hidden, mi),
                ) {
                    (Ok(g), Ok(u), Ok(d)) => [g, u, d],
                    _ => continue,
                };
                entries[layer * n_experts + e] =
                    Some(ExpertEntry { w_run: merge_run(&plans, true), s_run: merge_run(&plans, false), plans });
            }
        }
        ExpertIndex { n_experts, entries }
    }

    /// The resolved entry for one expert, or `None` if it was not indexed.
    fn get(&self, layer: usize, expert: usize) -> Option<&ExpertEntry> {
        if expert >= self.n_experts {
            return None;
        }
        self.entries.get(layer.checked_mul(self.n_experts)?.checked_add(expert)?)?.as_ref()
    }

    /// How many experts resolved — the denominator for the index-agreement test.
    pub fn resolved(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// How many resolved experts can be read as two extents instead of six
    /// regions. Reported by the census so the layout claim is checkable.
    pub fn mergeable(&self) -> usize {
        self.entries.iter().flatten().filter(|e| e.w_run.is_some() && e.s_run.is_some()).count()
    }

    /// Bytes one token's routing touches: `topk` experts in every sparse layer,
    /// each sized from *its own* layer, since a precision-tiered container does
    /// not have one expert size (this checkpoint stores the MTP layer at int8 and
    /// the rest at int4).
    ///
    /// This is the number that decides which of capacity or policy binds. A
    /// front-to-back layer sweep has no intra-pass reuse, so a budget below one
    /// token's working set cannot hold a pass no matter how it evicts, while a
    /// budget above it makes plain recency work. Measured either side of that
    /// threshold on this engine, the same `COLI_PREFETCH_PROTECT` mechanism is
    /// worth +193 hits below it and −381 above — which is why the engine needs to
    /// know where it sits rather than picking one default for both.
    pub fn per_token_bytes(&self, cfg: &Cfg) -> u64 {
        let n_experts = self.n_experts;
        let topk = cfg.topk.max(0) as u64;
        let mut total = 0u64;
        for layer in (cfg.first_dense.max(0) as usize)..=(cfg.n_layers.max(0) as usize) {
            // One resolved expert stands in for its layer: within a layer every
            // expert has the same shape and format.
            let Some(base) = layer.checked_mul(n_experts) else { continue };
            let Some(e) = self.entries.get(base..base.saturating_add(n_experts)).and_then(|r| r.iter().flatten().next())
            else {
                continue;
            };
            let per_expert: u64 = e.plans.iter().map(|t| t.w_len as u64 + t.s_len as u64).sum();
            total = total.saturating_add(per_expert.saturating_mul(topk));
        }
        total
    }
}

/// Resolve one expert, preferring the load-time index and falling back to
/// deriving it when the index has no entry.
fn entry_for(
    index: Option<&ExpertIndex>,
    st: &SafeTensors,
    cfg: &Cfg,
    layer: usize,
    expert: usize,
) -> Result<ExpertEntry, Error> {
    if let Some(e) = index.and_then(|ix| ix.get(layer, expert)) {
        return Ok(*e);
    }
    let hidden = cfg.hidden as usize;
    let mi = cfg.moe_inter as usize;
    let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{expert}.{t}");
    let plans = [
        tplan(st, &p("gate_proj.weight"), mi, hidden)?,
        tplan(st, &p("up_proj.weight"), mi, hidden)?,
        tplan(st, &p("down_proj.weight"), hidden, mi)?,
    ];
    Ok(ExpertEntry { w_run: merge_run(&plans, true), s_run: merge_run(&plans, false), plans })
}

/// The `(fd, offset, len)` regions to read for one expert, and how many there are.
///
/// Two when both runs merged and merging is on, six otherwise. The caller must
/// pair this with [`pack_slab_from`], which re-splits whatever shape comes back.
fn expert_regions(e: &ExpertEntry, direct: bool) -> Vec<(RawFd, u64, usize)> {
    let pick = |fd: RawFd, twin: Option<RawFd>| if direct { twin.unwrap_or(fd) } else { fd };
    if expert_merge_enabled() {
        if let (Some(w), Some(s)) = (e.w_run, e.s_run) {
            // Scales first: they sit at the front of the shard, so this issues in
            // ascending offset order.
            return vec![
                (pick(s.fd, s.fd_direct), s.off, s.len),
                (pick(w.fd, w.fd_direct), w.off, w.len),
            ];
        }
    }
    let mut out = Vec::with_capacity(6);
    for t in e.plans.iter() {
        out.push((pick(t.w_fd, t.w_fd_direct), t.w_off, t.w_len));
        out.push((pick(t.s_fd, t.s_fd_direct), t.s_off, t.s_len));
    }
    out
}

/// Rebuild an [`peregrine_io::ExpertSlab`] from whatever [`expert_regions`] asked
/// for: either the six unmerged regions in order, or two coalesced extents that
/// get carved into six refcounted windows.
///
/// The carve is computed from each plan's **own** offset relative to the extent
/// base, never from its position in `plans` — on disk the three projections sit
/// in alphabetical order (down, gate, up), and after an `apply_layout` rewrite
/// they can sit in any order at all. All three are same-sized int4 blobs, so a
/// positional split would load gate's bytes into down's matrix and still produce
/// plausible-looking activations.
fn pack_slab_from(e: &ExpertEntry, mut got: Vec<Bytes>) -> Result<peregrine_io::ExpertSlab, Error> {
    if got.len() == 6 {
        return pack_slab(got);
    }
    let (Some(w), Some(s)) = (e.w_run, e.s_run) else {
        return Err(Error::Format(format!("expert read returned {} regions with no merged extents", got.len())));
    };
    if got.len() != 2 {
        return Err(Error::Format(format!("expert read returned wrong region count: {} (want 2 or 6)", got.len())));
    }
    let w_arc = got.pop().ok_or_else(|| Error::Format("merged weight extent missing".into()))?.into_arc();
    let s_arc = got.pop().ok_or_else(|| Error::Format("merged scale extent missing".into()))?.into_arc();
    let carve = |arc: &std::sync::Arc<[u8]>, base: u64, off: u64, len: usize, what: &str| {
        let head = off
            .checked_sub(base)
            .and_then(|h| usize::try_from(h).ok())
            .ok_or_else(|| Error::Format(format!("{what} region lies before its extent")))?;
        Bytes::view(arc, head, len)
            .ok_or_else(|| Error::Format(format!("{what} region [{head}, {head}+{len}) escapes its extent")))
    };
    let mut regions: Vec<Bytes> = Vec::with_capacity(6);
    for t in e.plans.iter() {
        regions.push(carve(&w_arc, w.off, t.w_off, t.w_len, "weight")?);
        regions.push(carve(&s_arc, s.off, t.s_off, t.s_len, "scale")?);
    }
    pack_slab(regions)
}

/// One expert's streaming+compute plan: which rows route to it (+ gate weights),
/// where its gate/up/down tensors live on disk, and its batch-union position
/// (`pos`) for the deterministic ordered reduce (GPU-resident experts take the
/// intervening positions).
struct EPlan {
    pos: usize,
    expert: usize,
    rows: Vec<usize>,
    rw: Vec<f32>,
    /// Where this expert's bytes are and what format they are, resolved once.
    entry: ExpertEntry,
}

/// One GPU-resident expert's plan: its position, routed rows/weights, expert id,
/// and the gathered input rows to feed the batched `expert_group`.
struct GPlan {
    pos: usize,
    e: usize,
    rows: Vec<usize>,
    rw: Vec<f32>,
    xg: Vec<f32>,
}

/// Fuse the layer-level gate-weighted accumulation for GPU-resident experts
/// onto the device (`COLI_CUDA_FUSED_REDUCE`). Default **off**.
///
/// `expert_group` already fuses gate/up/silu/down, but returns `Σrows × hidden`
/// floats for the host to accumulate. Reducing on the device sends `s_n × hidden`
/// instead — at a saturated batch that is the expert-per-row factor, ~5× on the
/// measured GLM-5.2 unions at B=16, and exactly 1× at B=1.
///
/// **Opt-in because it moves the GPU arm's low bits**, not because it is
/// unfinished. GPU experts accumulate among themselves before meeting the CPU
/// lane's contributions rather than interleaving with them in `pos` order, and
/// `f32 +=` is not associative. It stays *stable* — the device reduce is CSR-
/// ordered with no atomics — so a given configuration reproduces itself; it
/// simply is not the same sum the host reduce computes. Every bit-identity
/// anchor in the suite holds with the knob unset.
fn fused_reduce_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("COLI_CUDA_FUSED_REDUCE").ok().as_deref(), Some("1") | Some("true")))
}

/// What a lane hands the collector.
enum LaneMsg {
    /// One expert's output, keyed by its batch-union position.
    Slot(usize, EOut),
    /// The GPU lane's device-reduced `[s_n, hidden]` partial, covering every
    /// GPU-resident expert of this layer at once. At most one per forward.
    GpuPartial(Vec<f32>),
}

/// A computed expert result, tagged with its batch-union position for the
/// deterministic ordered reduce.
struct EOut {
    rows: Vec<usize>,
    rw: Vec<f32>,
    h: Vec<f32>, // [rows.len() * hidden]
}

fn tplan(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<TPlan, Error> {
    let info = QtInfo::detect(st, name, o as i64, i as i64);
    let fmt = QuantFmt::from_qt(info.fmt)
        .ok_or_else(|| Error::Format(format!("{name}: unquantized (F32) has no compute path")))?;
    // Refuse any tensor whose on-disk bytes are not the bytes the kernel expects.
    // `tplan` is the single funnel for *streamed* expert regions, and the
    // streaming path reads raw extents — only `SafeTensors::read_raw` un-permutes
    // a `kblock` tiling or inflates zstd. A `kblock` expert therefore used to be
    // handed to the kernel permuted, with no conversion and no error: wrong
    // numbers that still look like plausible activations. Resident loading is
    // unaffected (it goes through `read_raw`), and `has_compressed_tensors`
    // already forces a compressed container resident, so this is a guard against
    // the combination reaching the streaming lane rather than a new restriction.
    if let Some((kind, gs)) = st.find(name).and_then(|t| t.layout.as_ref()) {
        return Err(Error::Format(format!(
            "{name}: on-disk layout '{kind}' (group {gs}) cannot be streamed — the streaming lane \
             reads raw extents and would not un-permute it; load this container resident instead"
        )));
    }
    if st.compression(name) != peregrine_core::Compression::None {
        return Err(Error::Format(format!(
            "{name}: compressed tensors cannot be streamed — the streaming lane reads raw extents; \
             load this container resident instead"
        )));
    }
    let (w_fd, w_off, w_len) = st.region(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))?;
    let sname = format!("{name}.qs");
    let (s_fd, s_off, s_len) = st.region(&sname).ok_or_else(|| Error::Format(format!("missing tensor {sname}")))?;
    let w_fd_direct = st.region_direct(name).map(|(fd, _, _)| fd);
    let s_fd_direct = st.region_direct(&sname).map(|(fd, _, _)| fd);
    Ok(TPlan { w_fd, w_off, w_len, s_fd, s_off, s_len, w_fd_direct, s_fd_direct, fmt, o, i, gs: info.gs as usize })
}

/// The reactor a ring-backed engine requires.
///
/// `None` can only arrive here if the resolved engine disagrees with what
/// `load_streaming` built rings for, so this is a clean error rather than an
/// unwrap: the engine is env-resolved and a mismatch should name itself.
fn need_ring(r: Option<&mut Reactor>) -> Result<&mut Reactor, Error> {
    r.ok_or_else(|| {
        Error::Format("a ring-backed io engine was selected but no io_uring ring was constructed".into())
    })
}

/// Finish a short positioned read without assuming a ring exists.
///
/// A positioned read may return short on *either* engine, so the completion has
/// to work on both. With a ring it reuses `read_exact`; without one it re-issues
/// plain `pread` until the tail is filled, which is what makes the pread engine
/// usable on a host that has no io_uring to fall back to.
fn complete_short_read(r: Option<&mut Reactor>, fd: RawFd, off: u64, buf: &mut [u8]) -> Result<(), Error> {
    if let Some(r) = r {
        return r.read_exact(fd, off, buf).ctx(|| "io_uring short-read completion".to_string());
    }
    let total = buf.len();
    let mut done = 0usize;
    while done < total {
        let mut req = [ReadReq { fd, offset: off + done as u64, buf: &mut buf[done..], tag: 0 }];
        let res = peregrine_io::pread_many(&mut req);
        let Some(&n) = res.first() else {
            return Err(Error::Format("pread short-read completion returned no result".into()));
        };
        if n < 0 {
            return Err(Error::Io(std::io::Error::from_raw_os_error((-n) as i32)));
        }
        if n == 0 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "pread short-read completion hit EOF",
            )));
        }
        done += n as usize;
    }
    Ok(())
}

/// Read a flat list of `(fd, offset, len)` regions and return one [`Bytes`] per
/// region, in order. Two lanes, both byte-identical to the resident path:
///
/// - **direct** — [`Reactor::read_direct_aligned`] DMAs each region's 4096-aligned
///   superset straight into an owned aligned buffer and returns it as an aligned
///   [`Bytes`] view, so the streamed `QtWeight` reads out of the DMA target with
///   **no realignment copy** (zero-copy O_DIRECT).
/// - **buffered** — one deep `read_many` submit fills plain landing `Vec`s directly
///   (the kernel writes the caller's buffer, so this is already zero userspace copy);
///   any short read is completed per region.
/// - **pread** — `COLI_IO_ENGINE=pread`: N OS threads of blocking `pread`,
///   bypassing io_uring entirely. See [`io_engine`] for why this exists.
fn read_regions(mut r: Option<&mut Reactor>, regions: &[(RawFd, u64, usize)], direct: bool) -> Result<Vec<Bytes>, Error> {
    if direct && io_engine() != IoEngine::Pread {
        return need_ring(r.as_deref_mut())?
            .read_direct_aligned(regions)
            .ctx(|| "io_uring O_DIRECT zero-copy expert read".to_string());
    }
    // The `regbuf` engine is exempt from splitting: its fixed buffers are sized
    // per-slot from the largest request, and it exists as a measurement option —
    // changing its request shape would change what it measures.
    let split = match io_engine() {
        IoEngine::RegBuf => 0,
        _ => io_split_bytes(),
    };
    let mut bufs: Vec<Vec<u8>> = regions.iter().map(|&(_, _, len)| vec![0u8; len]).collect();
    {
        // One request per region, or several sub-requests over disjoint
        // `split_at_mut` slices of the region's landing buffer when splitting
        // is on (`COLI_IO_SPLIT_MB`). `meta` maps each request back to
        // (region index, offset within region, length) for the short-read
        // completion, which must resolve file offsets per sub-request.
        let mut reqs: Vec<ReadReq> = Vec::with_capacity(regions.len());
        let mut meta: Vec<(usize, usize, usize)> = Vec::with_capacity(regions.len());
        for (i, (buf, &(fd, off, _))) in bufs.iter_mut().zip(regions).enumerate() {
            let mut sub = 0usize;
            let mut rest: &mut [u8] = buf.as_mut_slice();
            loop {
                let take = if split > 0 && rest.len() > split { split } else { rest.len() };
                let (head, tail) = rest.split_at_mut(take);
                reqs.push(ReadReq { fd, offset: off + sub as u64, buf: head, tag: 0 });
                meta.push((i, sub, take));
                sub += take;
                rest = tail;
                if rest.is_empty() {
                    break;
                }
            }
        }
        // Same request set either way, so the two engines are directly
        // comparable and produce byte-identical results — only the syscall shape
        // differs. The short-read completion below is shared, because a
        // positioned read may return short on either path.
        let res = match io_engine() {
            IoEngine::Pread => peregrine_io::pread_many_threaded(&mut reqs, pread_threads()),
            IoEngine::RegBuf => {
                // Registered buffers must exist before the first read. Sizing
                // them needs the largest region in the batch, which is only
                // known here, so registration is lazy and grows once. A failure
                // is not fatal: fall back to the plain submit rather than lose
                // the request, since this engine is a measurement option.
                let want = reqs.iter().map(|q| q.buf.len()).max().unwrap_or(0);
                let r = need_ring(r.as_deref_mut())?;
                match ensure_fixed_buffers(r, want) {
                    Ok(()) => r.read_fixed_many(&mut reqs).ctx(|| "io_uring fixed-buffer expert read".to_string())?,
                    Err(e) => {
                        // ENOMEM here almost always means RLIMIT_MEMLOCK, not
                        // RAM: registered buffers are pinned pages, and a pool
                        // sized for ~6 MB expert regions needs far more lockable
                        // memory than the 8 MB most distros default to.
                        peregrine_io::note_advisory_err(
                            "register fixed buffers (raise RLIMIT_MEMLOCK / ulimit -l?); using plain submit",
                            &e,
                        );
                        r.read_many(&mut reqs).ctx(|| "io_uring batched expert read".to_string())?
                    }
                }
            }
            IoEngine::Uring => need_ring(r.as_deref_mut())?
                .read_many(&mut reqs)
                .ctx(|| "io_uring batched expert read".to_string())?,
        };
        for (j, &n) in res.iter().enumerate() {
            if n < 0 {
                return Err(Error::Io(std::io::Error::from_raw_os_error((-n) as i32)));
            }
            let (i, sub, len) = meta[j];
            let done = n as usize;
            if done < len {
                let (fd, off, _) = regions[i];
                complete_short_read(
                    r.as_deref_mut(),
                    fd,
                    off + (sub + done) as u64,
                    &mut bufs[i][sub + done..sub + len],
                )?;
            }
        }
    }
    Ok(bufs.into_iter().map(Bytes::from).collect())
}

/// Which syscall shape the streaming lane uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IoEngine {
    /// io_uring — the historical default, batched submit through the [`Reactor`].
    Uring,
    /// N threads of blocking `pread`.
    Pread,
    /// io_uring through pre-registered fixed buffers (`IORING_OP_READ_FIXED`).
    RegBuf,
}

/// How many registered buffers the `regbuf` engine keeps (`COLI_REGBUF_SLOTS`).
/// One in-flight op per buffer, so this is the engine's queue depth.
const REGBUF_SLOTS_DEFAULT: usize = 16;

/// Register fixed buffers of at least `want` bytes, if not already adequate.
///
/// Grow-only and idempotent: re-registering unregisters the previous set, which
/// is a syscall plus re-pinning every page, so it must not happen per read.
fn ensure_fixed_buffers(r: &mut Reactor, want: usize) -> std::io::Result<()> {
    if want == 0 {
        return Ok(());
    }
    if r.fixed_buffer_count() > 0 && r.fixed_buffer_capacity() >= want {
        return Ok(());
    }
    let slots = std::env::var("COLI_REGBUF_SLOTS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(REGBUF_SLOTS_DEFAULT);
    r.register_read_buffers(vec![vec![0u8; want]; slots])
}

/// The streaming read engine (`COLI_IO_ENGINE=uring|pread`), default `uring`.
///
/// **Why this knob exists.** peregrine's own measurement has its io_uring
/// O_DIRECT lane at **0.84 GB/s against colibrì's 2.02 GB/s** from eight
/// blocking-`pread` threads on the same drive (`docs/benchmarks.md` §Second
/// box). The repo's standing explanation is the dm-crypt tax — on LUKS, reads
/// are CPU-bound on decryption, and N blocking preads keep N cores decrypting
/// where the ring's completion model can leave cores idle. That hypothesis has
/// never been tested on the real streaming path, only in `iobench`.
///
/// Output is byte-identical either way: same regions, same offsets, same
/// destination buffers. Only the syscall shape changes, so this can be A/B'd
/// against a bit-identity assertion rather than eyeballed.
///
/// `pread` also implies **no O_DIRECT**: the direct lane's whole value is the
/// aligned zero-copy DMA path in `read_direct_aligned`, which is io_uring-only.
/// Setting both is not an error — `pread` simply wins, and `read_regions` says
/// so at its branch — because the point of the knob is to compare engines, and
/// silently honouring `COLI_DIRECT` here would compare something else.
/// Can this kernel give us an io_uring at all?
///
/// Probed once, by building a minimal ring and dropping it. Three common Linux
/// configurations say no and none of them are exotic: kernels older than 5.1,
/// hosts hardened with `kernel.io_uring_disabled=2` (increasingly the default in
/// security-conscious distros since the 2023 exploit run), and containers whose
/// seccomp profile blocks `io_uring_setup` — Docker's default profile does.
///
/// The engine used to treat all three as fatal: `Reactor::new_streaming` failed
/// and the model would not load, even though the `pread` engine beside it needs
/// no ring whatsoever. Probing lets the default degrade instead of dying.
fn uring_available() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match Reactor::new(1) {
        Ok(_) => true,
        Err(e) => {
            eprintln!(
                "peregrine: [io] io_uring unavailable ({e}); falling back to the pread engine. \
                 Set COLI_IO_ENGINE=uring to make this fatal instead."
            );
            false
        }
    })
}

/// Does the resolved engine need io_uring rings built at load time?
///
/// `load_streaming` asks before constructing any, so the `pread` engine costs no
/// ring — and, more importantly, cannot fail to load on a host that has none.
pub(crate) fn engine_needs_rings() -> bool {
    !matches!(io_engine(), IoEngine::Pread)
}

/// Can the resolved engine issue O_DIRECT reads?
///
/// O_DIRECT requires the buffer, offset and length to be block-aligned, and the
/// only aligned buffers in this engine are the `Reactor`'s slab
/// (`read_direct_aligned`). The `pread` path lands into plain heap `Vec`s, so
/// handing it the O_DIRECT twin fd would fail every read with `EINVAL`. Kept as
/// its own predicate rather than folded into [`engine_needs_rings`] because the
/// two happen to agree today for different reasons.
pub(crate) fn engine_supports_direct() -> bool {
    !matches!(io_engine(), IoEngine::Pread)
}

fn io_engine() -> IoEngine {
    static V: std::sync::OnceLock<IoEngine> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("COLI_IO_ENGINE").as_deref() {
        Ok("pread") => IoEngine::Pread,
        Ok("regbuf") => IoEngine::RegBuf,
        // Historical spelling: `COLI_REGBUF=1` was documented and benchmarked
        // for a year while being read by no code at all. Honour it here so the
        // knob finally means something, rather than deleting it and silently
        // changing what a published benchmark arm did. It stays *ahead* of the
        // explicit `uring` arm below because that is the precedence it already
        // had, and reordering it would change a published arm just as silently.
        _ if matches!(std::env::var("COLI_REGBUF").as_deref(), Ok("1") | Ok("true")) => IoEngine::RegBuf,
        // An *explicit* `uring` stays strict — no probe, and a missing ring stays
        // fatal. A benchmark arm that asked for io_uring must fail loudly rather
        // than quietly become a pread arm and report its number under the wrong
        // name. Only the unset default falls back.
        Ok("uring") => IoEngine::Uring,
        _ if uring_available() => IoEngine::Uring,
        _ => IoEngine::Pread,
    })
}

/// Worker threads for the `pread` engine (`COLI_IO_THREADS`), defaulting to the
/// same worker count the rest of the streaming lane uses. Eight is colibrì's
/// harness figure and the one the 2.02 GB/s came from.
fn pread_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_IO_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(default_workers)
    })
}

/// Split threshold for large streamed reads (`COLI_IO_SPLIT_MB`, MiB; 0 = off,
/// the default). Regions larger than this are submitted as several sub-reads
/// into disjoint slices of the same landing buffer.
///
/// Why: a decode claim is 2 merged experts ≈ 4 in-flight reads per ring, while
/// `iobench` reaches the device's 1.12 GB/s at 8-deep — the submit depth, not
/// the claim size, is the suspect for the lane's 0.80 GB/s. Splitting a ~9.5 MB
/// merged region into 4 MiB pieces takes a ring's depth to ~10 without touching
/// claim sizing, and on LUKS each in-flight read is also an independent
/// dm-crypt decryption unit. Byte-identical trivially: the same bytes land at
/// the same offsets of the same buffer, in however many pieces.
fn io_split_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_IO_SPLIT_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|n| n << 20)
            .unwrap_or(0)
    })
}

/// Pack six in-order region [`Bytes`] into an [`ExpertSlab`] (gate/up/down ×
/// weight+scale). Errors (rather than panics) if the reader returned the wrong
/// count — keeps the no-unwrap gate satisfied.
fn pack_slab(six: Vec<Bytes>) -> Result<peregrine_io::ExpertSlab, Error> {
    let [gw, gs, uw, us, dw, ds]: [Bytes; 6] = six
        .try_into()
        .map_err(|wrong: Vec<Bytes>| {
            Error::Format(format!("expert read returned wrong region count: {} (want 6)", wrong.len()))
        })?;
    Ok([(gw, gs), (uw, us), (dw, ds)])
}

/// Stream one expert's gate/up/down (six weight+scale regions) through the ring in
/// a **single batched submit**. Zero-copy on the O_DIRECT lane (see [`read_regions`]);
/// byte-identical to six `read_exact`s either way, so the streamed output stays
/// bit-identical to the resident path.
fn read_expert(r: Option<&mut Reactor>, e: &ExpertEntry, direct: bool) -> Result<peregrine_io::ExpertSlab, Error> {
    let regions = expert_regions(e, direct);
    pack_slab_from(e, read_regions(r, &regions, direct)?)
}

/// How many experts' reads to submit to the ring at once. 6 regions/expert, so
/// `16 × 6 = 96` in-flight reads keep the io_uring queue deep (vs. 6 when reading
/// one expert at a time — the colibrì deep-queue model) while bounding the transient
/// landing-buffer memory to ~`16 × 18.9 MB ≈ 300 MB` (this box is RAM-contended, so
/// a bounded batch matters; a reusable slab arena would remove the ceiling entirely).
pub const EXPERTS_PER_BATCH: usize = 16;

/// Runtime I/O queue depth: `COLI_IO_BATCH` experts per ring claim (default
/// [`EXPERTS_PER_BATCH`] = 16 → 96 in-flight reads). Deeper queues can help on
/// faster storage (this LUKS box is already disk-saturated at 2 rings, so it's a
/// marginal lever here). Cached once, so the hot loop pays a single atomic load.
pub fn experts_per_batch() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_IO_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(EXPERTS_PER_BATCH)
    })
}

/// Partition plan indices into device-pure claim groups: one `Vec<usize>` per
/// device ordinal, ascending, each preserving the incoming plan order (which the
/// caller already sorted for contiguous offsets). Fds absent from the table land
/// in a trailing `u8::MAX` group rather than poisoning a real device's claims.
/// Pure so it unit-tests without mounts or env.
fn device_claim_groups(
    fds: impl Iterator<Item = RawFd>,
    table: &HashMap<RawFd, u8>,
) -> Vec<Vec<usize>> {
    let mut by_dev: std::collections::BTreeMap<u8, Vec<usize>> = std::collections::BTreeMap::new();
    for (idx, fd) in fds.enumerate() {
        by_dev.entry(table.get(&fd).copied().unwrap_or(u8::MAX)).or_default().push(idx);
    }
    by_dev.into_values().collect()
}

/// Home-group assignment for the io rings: rings are spread across claim groups
/// proportionally to group size (largest remainder), and every non-empty group
/// gets at least one ring while rings remain — a device with little work must
/// not be orphaned, or its reads wait for a stealing ring to go dry elsewhere.
/// Returns `homes[ring] = group`. Pure for the same testability reason.
fn ring_homes(group_sizes: &[usize], n_rings: usize) -> Vec<usize> {
    let total: usize = group_sizes.iter().sum();
    if total == 0 || group_sizes.is_empty() {
        return vec![0; n_rings];
    }
    // Ideal share per group, floored; then hand leftover rings to the largest
    // remainders, seeding empty-handed non-empty groups first.
    let mut share: Vec<usize> =
        group_sizes.iter().map(|&s| n_rings * s / total).collect();
    let mut assigned: usize = share.iter().sum();
    let mut order: Vec<usize> = (0..group_sizes.len()).collect();
    order.sort_by_key(|&g| {
        // Zero-share non-empty groups first, then by remainder, descending.
        let starved = share[g] == 0 && group_sizes[g] > 0;
        let rem = n_rings * group_sizes[g] % total;
        (std::cmp::Reverse(starved as usize), std::cmp::Reverse(rem), g)
    });
    for &g in order.iter().cycle() {
        if assigned >= n_rings {
            break;
        }
        if group_sizes[g] > 0 {
            share[g] += 1;
            assigned += 1;
        }
    }
    let mut homes = Vec::with_capacity(n_rings);
    for (g, &n) in share.iter().enumerate() {
        homes.extend(std::iter::repeat_n(g, n));
    }
    homes.resize(n_rings, homes.last().copied().unwrap_or(0));
    homes
}

/// Whether to fire `POSIX_FADV_WILLNEED` for the main streamed-expert regions
/// just before the batched read (buffered mode only — the O_DIRECT path bypasses
/// the page cache, so the hint would be ignored). Default on when buffered;
/// turn off with `COLI_FADVISE_MAIN=0`. Advisory, so bit-identical either way.
fn fadvise_main_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_FADVISE_MAIN").as_deref(), Ok("0")))
}

/// Whether to fire `POSIX_FADV_DONTNEED` for the just-consumed regions after
/// the read — releases page-cache pages under long-running loads to keep RSS
/// flat. Off by default (drops the pages a warm cache would otherwise reuse);
/// enable with `COLI_FADVISE_DROP=1` on memory-tight boxes.
fn fadvise_drop_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_FADVISE_DROP").as_deref(), Ok("1")))
}

/// Stream a *batch* of experts' gate/up/down (six weight+scale regions each) through
/// the ring in **one deep `read_many` submit**, so the disk queue stays full across
/// the whole batch instead of draining one expert at a time. Short reads are
/// completed per region, so the returned bytes are identical to [`read_expert`] —
/// the streamed output stays bit-identical to the resident path. Slabs are returned
/// in `plans` order.
fn read_experts_batched(mut r: Option<&mut Reactor>, plans: &[&EPlan], direct: bool) -> Result<Vec<peregrine_io::ExpertSlab>, Error> {
    let n = plans.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // one (fd, offset, len) per region, in gate/up/down × (weight, scale) order. In
    // direct mode use the O_DIRECT twin fd (falling back per-region if a twin is
    // somehow missing); the reader applies the block alignment.
    // Two regions per expert when its runs coalesced, six when they did not, so
    // the flat list is no longer uniformly `6 * n` — `counts` records how many
    // each expert contributed so the results can be re-split.
    let mut regions: Vec<(RawFd, u64, usize)> = Vec::with_capacity(6 * n);
    let mut counts: Vec<usize> = Vec::with_capacity(n);
    for p in plans {
        let r = expert_regions(&p.entry, direct);
        counts.push(r.len());
        regions.extend_from_slice(&r);
    }
    // Buffered path: hint the kernel to start readahead on every region before the
    // batched read. The advice fires from the same ring in one submit, so it costs
    // one extra syscall and can overlap NVMe queue depth with our submit ceremony.
    // Purely advisory — bytes returned are identical either way.
    if !direct && fadvise_main_enabled() {
        // Soft-failure only: readahead failures never affect correctness. With a
        // ring the hints ride one submit; without one they are plain
        // `posix_fadvise` calls, which is the same advice at a syscall each.
        let hinted = match r.as_deref_mut() {
            Some(r) => r.fadvise_willneed_many(&regions),
            None => peregrine_io::fadvise_many(&regions, peregrine_io::FADV_WILLNEED),
        };
        if let Err(e) = hinted {
            peregrine_io::note_advisory_err("fadvise willneed (batched readahead)", &e);
        }
    }
    // one deep submit for all 6·n regions (buffered) or per-region aligned DMA
    // (direct, zero-copy); bytes come back in region order, six per expert.
    let bytes = match read_regions(r.as_deref_mut(), &regions, direct) {
        Ok(b) => b,
        Err(e) if io_recovery_enabled() && !direct => {
            // Retry ladder: on a batched-read failure, re-issue each region as
            // an individual `read_exact_retry` (which handles EIO/EAGAIN/EINTR
            // with backoff). Trades throughput for resilience — used only after
            // the fast path has already failed. O_DIRECT is skipped because the
            // recovery path uses buffered reads.
            eprintln!("[io-recovery] batched read failed ({e}); retrying regions individually");
            read_regions_with_retry(r.as_deref_mut(), &regions)?
        }
        Err(e) => return Err(e),
    };
    let mut bytes = bytes.into_iter();
    let mut slabs: Vec<peregrine_io::ExpertSlab> = Vec::with_capacity(n);
    for (p, &k) in plans.iter().zip(counts.iter()) {
        let got: Vec<Bytes> = bytes.by_ref().take(k).collect();
        slabs.push(pack_slab_from(&p.entry, got)?);
    }
    // Optional page-cache release for long-running RSS-bounded workloads. Only
    // useful when the warm cache is off / cold — a hit would otherwise re-read the
    // pages we just dropped. Purely advisory, so a soft failure is harmless.
    if !direct && fadvise_drop_enabled() {
        let dropped = match r {
            Some(r) => {
                let mut first_err = None;
                for &(fd, off, len) in &regions {
                    if let Err(e) = r.fadvise_dontneed(fd, off, len) {
                        first_err.get_or_insert(e);
                    }
                }
                match first_err {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            }
            None => peregrine_io::fadvise_many(&regions, peregrine_io::FADV_DONTNEED),
        };
        if let Err(e) = dropped {
            peregrine_io::note_advisory_err("fadvise dontneed (page-cache release)", &e);
        }
    }
    Ok(slabs)
}

/// Whether the I/O recovery path (batched read → per-region retry with backoff)
/// is enabled. Default on; disable with `COLI_IO_RECOVERY=0`.
fn io_recovery_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_IO_RECOVERY").as_deref(), Ok("0") | Ok("false")))
}

/// Cache-admission heat threshold: a just-streamed expert is admitted into the
/// warm cache only once its routing heat reaches this count. `0` (default)
/// admits everything — the historical behavior. `1` means "cache from the
/// second routing onward" (heat is bumped after the reduce, so a first-time
/// expert still reads 0 here), filtering one-off experts out of the cache.
/// Correctness-neutral: a skipped admission just re-streams identical bytes.
fn cache_admit_min_heat() -> u32 {
    use std::sync::OnceLock;
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_CACHE_ADMIT_MIN_HEAT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    })
}

/// Per-region fallback for the buffered path: reads each region with
/// `Reactor::read_exact_retry` (transient EIO/EAGAIN/EINTR retried with linear
/// backoff). Slower than the batched submit, but preserves the byte-identical
/// contract when the batched path suffers a transient failure.
fn read_regions_with_retry(mut r: Option<&mut Reactor>, regions: &[(RawFd, u64, usize)]) -> Result<Vec<Bytes>, Error> {
    let mut out: Vec<Bytes> = Vec::with_capacity(regions.len());
    for &(fd, off, len) in regions {
        let mut buf = vec![0u8; len];
        match r.as_deref_mut() {
            Some(r) => r
                .read_exact_retry(fd, off, &mut buf, 3)
                .ctx(|| format!("io_uring per-region retry @ off={off} len={len}"))?,
            // `complete_short_read` already re-issues until the buffer is full,
            // so on the pread path the loop *is* the retry.
            None => complete_short_read(None, fd, off, &mut buf)
                .ctx(|| format!("pread per-region retry @ off={off} len={len}"))?,
        }
        out.push(Bytes::from(buf));
    }
    Ok(out)
}

/// Whether the I/O lane forwards each expert as its own regions complete (the
/// owned-completion lane) instead of waiting for the whole claim's wave.
/// Default on; `COLI_IO_COMPLETION=0` restores the wave path byte-for-byte.
/// `COLI_IO_ENGINE=pread|regbuf` also implies the wave path — those engines are
/// wave-shaped measurement arms, and reshaping their requests would change what
/// they measure.
fn completion_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COLI_IO_COMPLETION").as_deref(), Ok("0") | Ok("false")))
}

/// Stream a claim of experts through the owned-completion lane: submit every
/// region with Reactor-owned buffers, reap completions incrementally, and hand
/// each expert to `forward` the moment its last region lands — no whole-claim
/// barrier, so the CPU pool starts on expert 1 while experts 2..N are still on
/// the wire. Bytes are byte-identical to [`read_experts_batched`]: same
/// regions, same offsets, same [`pack_slab_from`] carve; only delivery timing
/// differs, which the `pos`-keyed reduce is built to absorb.
///
/// `forward(k, slab)` receives the index into `plans` and the packed slab; it
/// returns `false` when the consumer is gone, which stops the stream. Returns
/// `Ok(false)` in that case, `Ok(true)` when every expert was forwarded. On
/// any early exit the lane is quiesced before returning, so the ring is clean
/// for the next claim.
///
/// On a read error with `COLI_IO_RECOVERY` on (buffered only), the experts not
/// yet forwarded are re-read individually with the retry ladder; the ones
/// already forwarded keep their delivered bytes.
fn stream_experts_completion(
    r: &mut Reactor,
    plans: &[&EPlan],
    direct: bool,
    mut forward: impl FnMut(usize, peregrine_io::ExpertSlab) -> bool,
) -> Result<bool, Error> {
    let n = plans.len();
    if n == 0 {
        return Ok(true);
    }
    // One owned request per region; `tag` packs (expert k, region slot j) so a
    // completion routes back without a lookup table. Region counts differ per
    // expert (2 coalesced, 6 not) — `remaining` tracks each expert's countdown.
    let mut reqs: Vec<OwnedReadReq> = Vec::with_capacity(6 * n);
    let mut parts: Vec<Vec<Option<Bytes>>> = Vec::with_capacity(n);
    let mut remaining: Vec<usize> = Vec::with_capacity(n);
    let mut all_regions: Vec<(RawFd, u64, usize)> = Vec::with_capacity(6 * n);
    for (k, p) in plans.iter().enumerate() {
        let regs = expert_regions(&p.entry, direct);
        remaining.push(regs.len());
        parts.push((0..regs.len()).map(|_| None).collect());
        for (j, &(fd, off, len)) in regs.iter().enumerate() {
            reqs.push(OwnedReadReq { fd, offset: off, len, tag: ((k as u64) << 32) | j as u64 });
        }
        all_regions.extend_from_slice(&regs);
    }
    let mut forwarded = vec![false; n];
    let outcome: Result<bool, Error> = (|| {
        r.submit_owned(reqs, direct).ctx(|| "owned-completion expert submit".to_string())?;
        let mut done_buf: Vec<RegionDone> = Vec::new();
        while r.pending_owned() > 0 {
            r.reap_some(&mut done_buf).ctx(|| "owned-completion expert reap".to_string())?;
            for d in done_buf.drain(..) {
                let k = (d.tag >> 32) as usize;
                let j = (d.tag & 0xffff_ffff) as usize;
                let slot = parts
                    .get_mut(k)
                    .and_then(|p| p.get_mut(j))
                    .ok_or_else(|| Error::Format(format!("owned completion for unknown region {k}/{j}")))?;
                if slot.is_some() {
                    return Err(Error::Format(format!("duplicate owned completion for region {k}/{j}")));
                }
                *slot = Some(d.bytes);
                let rem = remaining
                    .get_mut(k)
                    .ok_or_else(|| Error::Format(format!("owned completion for unknown expert {k}")))?;
                *rem -= 1; // safe: the duplicate guard above means each region decrements once
                if *rem == 0 {
                    let got: Option<Vec<Bytes>> = parts[k].iter_mut().map(|o| o.take()).collect();
                    let got =
                        got.ok_or_else(|| Error::Format(format!("expert {k} completed with a region missing")))?;
                    let slab = pack_slab_from(&plans[k].entry, got)?;
                    forwarded[k] = true;
                    if !forward(k, slab) {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    })();
    // Whatever happened, leave the lane empty: owned reads left in flight would
    // poison the next claim on this ring (and, at drop, dangle their buffers).
    if r.pending_owned() > 0 {
        let mut scratch: Vec<RegionDone> = Vec::new();
        if let Err(e) = r.quiesce_owned(&mut scratch) {
            peregrine_io::note_advisory_err("owned-lane quiesce after early exit", &e);
        }
    }
    match outcome {
        Ok(true) => {
            // Optional page-cache release for long-running RSS-bounded loads —
            // advisory, riding spare ring slots (see `fadvise_drop_enabled`).
            if !direct && fadvise_drop_enabled() {
                r.queue_dontneed(&all_regions);
            }
            Ok(true)
        }
        Ok(false) => Ok(false),
        Err(e) if io_recovery_enabled() && !direct => {
            // Retry ladder, scoped to the experts that never made it out: the
            // accounting knows exactly which those are, so a transient failure
            // does not force re-reading (or re-forwarding) the delivered ones.
            eprintln!("[io-recovery] completion-lane read failed ({e}); retrying unfinished experts individually");
            for (k, p) in plans.iter().enumerate() {
                if forwarded[k] {
                    continue;
                }
                let regs = expert_regions(&p.entry, false);
                let bytes = read_regions_with_retry(Some(r), &regs)?;
                let slab = pack_slab_from(&p.entry, bytes)?;
                forwarded[k] = true;
                if !forward(k, slab) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Err(e) => Err(e),
    }
}

/// Re-stream one expert with the blocking oracle reader ([`pread_many`]),
/// looping out short reads. Only for the corruption-recovery path on a CPU
/// worker (which owns no ring): a warm-cache slot failed to decode, the slot
/// has been dropped, and these bytes replace it straight from disk. Rare by
/// construction — non-zero traffic here means real bit rot (see
/// `WarmCache::decode_failures`).
fn restream_expert_blocking(e: &ExpertEntry) -> Result<peregrine_io::ExpertSlab, Error> {
    let regions = expert_regions(e, false);
    let mut bufs: Vec<Vec<u8>> = regions.iter().map(|&(_, _, len)| vec![0u8; len]).collect();
    for (buf, &(fd, off, len)) in bufs.iter_mut().zip(regions.iter()) {
        let mut done = 0usize;
        while done < len {
            let n = {
                let mut req =
                    [ReadReq { fd, offset: off + done as u64, buf: &mut buf[done..], tag: 0 }];
                peregrine_io::pread_many(&mut req).first().copied().unwrap_or(-5)
            };
            if n < 0 {
                return Err(Error::Io(std::io::Error::from_raw_os_error((-n) as i32)));
            }
            if n == 0 {
                return Err(Error::Format(format!(
                    "corrupt-slot re-stream hit EOF at {done} of {len} bytes @ off={off}"
                )));
            }
            done += n as usize;
        }
    }
    pack_slab_from(e, bufs.into_iter().map(Bytes::from).collect())
}

fn rebuild(t: &TPlan, wb: Bytes, sb: Bytes) -> QtWeight {
    // scale bytes → f32 (a copy inherent to the reinterpret; scales are tiny). The
    // weight bytes `wb` move into the QtWeight with no copy — zero-copy end to end
    // when `wb` is an O_DIRECT aligned region.
    let scale: Vec<f32> = sb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    match t.fmt {
        QuantFmt::Int4Grouped => QtWeight::new_grouped(t.o, t.i, wb, scale, t.gs),
        f => QtWeight::new(f, t.o, t.i, wb, scale),
    }
}

/// Concurrent streamed MoE forward: io_uring disk lane ∥ CPU worker pool ∥
/// (optional) GPU VRAM lane, merged by a deterministic fixed-order reduce.
///
/// Without a GPU tier this is bit-identical to the sequential streamed path (only
/// faster). With a GPU tier, GPU-resident experts compute in f32 on the device
/// concurrently — those experts' values differ from the CPU int4 path (higher
/// precision), documented in [`crate::gpu`].
/// A pluggable MoE expert-dispatch implementation.
///
/// **Exists because `peregrine-sched` cannot be called directly.** That crate
/// depends on `peregrine-model` (`peregrine-sched/Cargo.toml`), so a
/// `peregrine-model → peregrine-sched` dependency is a cycle and Cargo rejects
/// it for normal dependencies. The dependency has to be inverted: this crate
/// declares the shape, `peregrine-sched` implements it, and a *binary* — which
/// may depend on both — installs the implementation.
///
/// Selecting an alternative engine is an operator decision with real cost
/// (`peregrine-sched`'s `moe_streamed` is the two-lane ancestor: no GPU lane, no
/// warm cache, no prefetch), which is why nothing installs one by default.
pub trait MoeEngine: Send + Sync {
    /// Same contract as [`moe_forward_concurrent`]: `[s_n, hidden]` output for
    /// this layer's routed experts plus the shared expert.
    fn moe_forward(&self, ctx: &ForwardCtx, call: MoeCall) -> Result<Vec<f32>, Error>;

    /// Short name for reporting, e.g. `"sched"`. Read by [`moe_engine_name`] so
    /// `/metrics` can say which implementation is *dispatching* rather than
    /// which one the environment asked for — the two diverge whenever an
    /// install fails, which is exactly when an operator needs to be told.
    fn name(&self) -> &'static str;
}

/// One layer's MoE inputs, bundled.
///
/// `moe_forward_concurrent` already sits at clippy's seven-argument limit, and a
/// trait method's `&self` puts it one over — so this exists partly for that. It
/// is the better shape regardless: five of the six travel together everywhere
/// and an engine implementation reads them by name instead of by position.
pub struct MoeCall<'a> {
    pub layer: usize,
    pub x: &'a [f32],
    pub router_w: &'a [f32],
    pub router_bias: &'a [f32],
    pub shared: Option<&'a Mlp>,
    pub s_n: usize,
}

static MOE_ENGINE: std::sync::OnceLock<Box<dyn MoeEngine>> = std::sync::OnceLock::new();

/// Install the process-wide MoE engine. First call wins; later calls are
/// rejected (returning `false`) rather than silently swapping the dispatch path
/// out from under a forward already in flight.
///
/// Called by the binaries when `COLI_MOE_ENGINE` selects a non-default engine.
pub fn install_moe_engine(engine: Box<dyn MoeEngine>) -> bool {
    MOE_ENGINE.set(engine).is_ok()
}

/// Whether an alternative engine is installed.
///
/// The binaries call this **after** their install attempt, because
/// [`install_moe_engine`] returning `true` and the dispatch path actually
/// changing are different facts: the `OnceLock` is process-global and first-call-
/// wins, so a second installer (a test harness, a library embedding the engine)
/// silently loses. Reporting the env var instead would print "engine = sched"
/// for a process dispatching through [`moe_forward_concurrent`].
pub fn moe_engine_installed() -> bool {
    MOE_ENGINE.get().is_some()
}

/// The name of the engine [`moe_forward_dispatch`] will actually route through:
/// the installed engine's own [`MoeEngine::name`], or `"concurrent"` for the
/// built-in three-lane path. Reported on `GET /metrics`.
pub fn moe_engine_name() -> &'static str {
    match MOE_ENGINE.get() {
        Some(engine) => engine.name(),
        None => "concurrent",
    }
}

/// The MoE dispatch entry point every forward goes through.
///
/// Routes to an installed [`MoeEngine`] when one exists, else to the default
/// three-lane [`moe_forward_concurrent`]. With nothing installed this is one
/// `OnceLock` load per sparse layer and the path is unchanged.
pub fn moe_forward_dispatch(
    ctx: &ForwardCtx,
    layer: usize,
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    shared: Option<&Mlp>,
    s_n: usize,
) -> Result<Vec<f32>, Error> {
    match MOE_ENGINE.get() {
        Some(engine) => {
            engine.moe_forward(ctx, MoeCall { layer, x, router_w, router_bias, shared, s_n })
        }
        None => moe_forward_concurrent(ctx, layer, x, router_w, router_bias, shared, s_n),
    }
}

pub fn moe_forward_concurrent(
    ctx: &ForwardCtx,
    layer: usize,
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    shared: Option<&Mlp>,
    s_n: usize,
) -> Result<Vec<f32>, Error> {
    let st = ctx.st;
    let gpu = ctx.gpu;
    let workers = ctx.workers;
    let cfg = ctx.cfg;
    let ecache = ctx.ecache; // Copy `Option<&Mutex<WarmCache>>`; captured only by the I/O lane
    // Lock-free residency filter, fetched once per layer call (one brief lock),
    // so each ring can answer "definitely absent → stream it" without queueing
    // on the cache mutex behind every other ring's probe and insert.
    let cache_hint = ecache.map(|c| c.lock().hint());
    let cache_hint = &cache_hint;
    let use_direct = ctx.direct; // O_DIRECT streaming (page-cache-bypassing); Copy bool
    let reactors = ctx.reactors;
    // No rings is a legitimate configuration, not a broken one: the `pread`
    // engine reads through plain positioned reads and `load_streaming` therefore
    // builds no reactors for it. That is what lets this engine run on a host
    // with no io_uring at all — an older kernel, `kernel.io_uring_disabled=2`,
    // or a container whose seccomp profile blocks `io_uring_setup`. It is still
    // fatal for the engines that genuinely need a ring.
    if reactors.is_empty() && engine_needs_rings() {
        return Err(Error::Format("streaming mode without io_uring reactors".into()));
    }
    let hidden = cfg.hidden as usize;
    // `moe_inter` is no longer read here: the expert map (or `plans_for`'s
    // fallback) owns the tensor shapes now.
    let (e_n, k) = (cfg.n_experts as usize, cfg.topk as usize);
    let r = route(x, router_w, router_bias, RouterCfg { s_n, d_n: hidden, e_n, k, norm_topk: cfg.norm_topk, routed_scale: cfg.routed_scale, min_share: crate::router::route_min_share() });

    // Partition the batch-union into GPU-resident (compute on device) and disk
    // (stream + CPU) experts, assigning each a global `pos` in batch-union order
    // so the final reduce stays deterministic regardless of which lane finishes.
    let mut plans: Vec<EPlan> = Vec::new();
    let mut gplans: Vec<GPlan> = Vec::new();
    let mut pos = 0usize;
    let uniq = batch_union(&r, s_n);
    // Number of experts per layer — needed to look up per-expert heat in the
    // flat counts slice (fallback to 0 when the balancer is disabled).
    let n_experts_layer = cfg.n_experts as usize;
    // Bucket every position's routed experts in ONE pass over the routing table.
    // The previous shape rescanned the whole table per unique expert
    // (O(|union| × S × K)); at prefill scale that is millions of comparisons per
    // sparse layer, on the critical path before any lane can be dispatched.
    // Emission order (batch-union) and per-expert row order (ascending `s`,
    // first matching `kk`) are unchanged, so the reduce stays bit-identical.
    let mut rows_of: Vec<(Vec<usize>, Vec<f32>)> = uniq.iter().map(|_| (Vec::new(), Vec::new())).collect();
    let mut slot_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::with_capacity(uniq.len());
    for (i, &e) in uniq.iter().enumerate() {
        // `batch_union` only yields ids the router selected, and the router
        // never emits a negative id — a conversion failure would mean corrupt
        // routing, so skip the entry rather than indexing with a wrapped value.
        let Ok(e) = usize::try_from(e) else { continue };
        slot_of.entry(e).or_insert(i);
    }
    for s in 0..s_n {
        for kk in 0..(r.keff[s].max(0) as usize).min(r.k) {
            let Ok(e) = usize::try_from(r.idx[s * r.k + kk]) else { continue };
            let Some(&i) = slot_of.get(&e) else { continue };
            if rows_of[i].0.last() == Some(&s) {
                continue; // one row per position per expert (the old `break`)
            }
            rows_of[i].0.push(s);
            rows_of[i].1.push(r.w[s * r.k + kk]);
        }
    }
    for (&e, (rows, rw)) in uniq.iter().zip(rows_of) {
        let Ok(e) = usize::try_from(e) else { continue };
        if rows.is_empty() {
            continue;
        }
        let this_pos = pos;
        pos += 1;
        let gpu_resident = gpu.is_some_and(|g| g.has(layer, e));
        // LaneBalancer consultation: if the balancer says the GPU lane is the
        // bottleneck and this expert is cold enough, route it through the CPU
        // lane instead — even though it's GPU-resident. When the resident
        // expert has been replicated (see `Model::enqueue_expert_replicas`) the
        // CPU lane serves it from the warm cache with no disk read.
        let route_to_gpu = match (ctx.balancer, ctx.heat_counts) {
            (Some(bal), Some(counts)) => {
                let heat = counts.get(layer * n_experts_layer + e).copied().unwrap_or(0);
                // Exhaustive on purpose: a *mid-forward* upload path still does
                // not exist, so `GpuSpill` resolves to the CPU lane this
                // forward. What changed (2026-08-13): the verdict is no longer
                // discarded — with `COLI_GPU_SPILL` it records into `ctx.spill`
                // and `Model::reheat` uploads the expert for the *next*
                // generation. Spelling the variant out keeps a future real
                // mid-forward spill a compile error here rather than a verdict
                // that silently does nothing.
                match bal.choose(gpu_resident, heat) {
                    crate::lane::Placement::Gpu => true,
                    crate::lane::Placement::Cpu => false,
                    crate::lane::Placement::GpuSpill => {
                        if let Some(log) = ctx.spill {
                            log.lock().push((layer, e));
                        }
                        false
                    }
                }
            }
            _ => gpu_resident,
        };
        if route_to_gpu {
            let nr = rows.len();
            let mut xg = vec![0f32; nr * hidden];
            for (ri, &s) in rows.iter().enumerate() {
                xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
            }
            gplans.push(GPlan { pos: this_pos, e, rows, rw, xg });
        } else {
            let entry = entry_for(ctx.expert_index, st, cfg, layer, e)?;
            plans.push(EPlan { pos: this_pos, expert: e, rows, rw, entry });
        }
    }
    let n = pos;

    // Order the streamed `EPlan`s so the batched submit issues ascending-offset
    // reads. This changes only the io_uring submit order — the deterministic
    // reduce uses `pos` (batch-union index) as its scatter key, so outputs stay
    // bit-identical either way.
    //
    // Sort on the **real** `(fd, offset)` when the expert map has resolved them.
    // `schedule.json`'s rank is a routing-community order that only coincides with
    // disk order after a `peregrine-layout-reorg --apply` rewrite — and that tool
    // is single-shard only, so it cannot run on a sharded container at all. This
    // sort claimed to issue "contiguous-offset reads first" while sorting by
    // community until 2026-08-09; on the GLM-5.2 checkpoint, whose tensors sit in
    // lexicographic name order (expert 14 between 139 and 140), the two orders are
    // unrelated. The schedule stays as the fallback for the case where no map was
    // built, which is the only case where its proxy is the best available signal.
    if ctx.expert_index.is_some() {
        plans.sort_by_key(|p| {
            let t = &p.entry.plans[0];
            (p.entry.w_run.map_or(t.w_fd, |r| r.fd), p.entry.w_run.map_or(t.w_off, |r| r.off))
        });
    } else if let Some(sched) = ctx.layout_schedule {
        if let Some(row) = sched.get(layer) {
            let rank: std::collections::HashMap<u32, usize> =
                row.iter().enumerate().map(|(i, &e)| (e, i)).collect();
            // Experts absent from the schedule keep their original relative order,
            // appended at the end.
            plans.sort_by_key(|p| rank.get(&(p.expert as u32)).copied().unwrap_or(usize::MAX));
        }
    }
    // Co-activation affinity: hyperedge grouping + fused-pair adjacency on top
    // of (or instead of) the layout order. Reduce keys on `pos`, so any order
    // here is bit-identical.
    if let Some(aff) = ctx.affinity {
        apply_affinity_order(&mut plans, layer, aff);
    }

    // job: work for one CPU worker, keyed by disk-plan index. `Bytes` regions so
    // an O_DIRECT read can hand its aligned DMA buffer over the channel with no
    // copy (`Bytes3` == peregrine_io::ExpertSlab).
    type Bytes3 = [(Bytes, Bytes); 3];
    enum Job {
        /// Freshly streamed from disk: compute, and admit to the warm cache on
        /// the worker — the zstd encode runs here, not on the I/O lane.
        Stream(usize, Bytes3),
        /// Warm hit, raw slot: compute only.
        CacheRaw(usize, Bytes3),
        /// Warm hit, compressed slot: the refcounted frames decode on the
        /// worker — the decode used to run on the I/O lane *inside* the cache
        /// lock, serializing every lane behind one zstd pass.
        CacheZ(usize, std::sync::Arc<peregrine_io::CompressedSlab>),
    }
    let (job_tx, job_rx) = crossbeam_channel::bounded::<Job>(workers.max(1) * 2);
    // result: one computed expert keyed by `pos`, or — under
    // `COLI_CUDA_FUSED_REDUCE` — the GPU lane's single pre-accumulated
    // `[s_n, hidden]` partial standing in for all of its experts at once.
    let (res_tx, res_rx) = crossbeam_channel::bounded::<Result<LaneMsg, Error>>(workers.max(1) * 2);
    // Whether the GPU lane reduces on the device. Resolved once here rather than
    // per lane so the collector's expected message count and the lane's
    // behaviour cannot disagree.
    let fused_reduce = gpu.is_some() && !gplans.is_empty() && fused_reduce_enabled();

    let completed = AtomicUsize::new(0);
    // Claim order for the I/O rings. Device-pure when the fd→device table is
    // present (`COLI_IO_DEVICE_SCHED`) and more than one ring exists: one group
    // of plan indices per device ordinal, so a claim window — and therefore each
    // deep submit — never mixes devices and never reaps behind a slower one.
    // Otherwise a single group spanning all plans, which reproduces the
    // historical shared-cursor behavior claim-for-claim. Each group keeps its
    // own lock-free cursor; rings claim from their home group first and steal
    // device-pure windows from the others when it runs dry, so no expert is
    // read twice and no ring idles while any device still has work.
    let claim_groups: Vec<Vec<usize>> = match ctx.fd_devices {
        Some(table) if reactors.len() > 1 => device_claim_groups(
            plans.iter().map(|p| p.entry.w_run.map_or(p.entry.plans[0].w_fd, |r| r.fd)),
            table,
        ),
        _ => vec![(0..plans.len()).collect()],
    };
    let group_cursors: Vec<AtomicUsize> =
        claim_groups.iter().map(|_| AtomicUsize::new(0)).collect();
    // Home-ring assignment and per-group claim sizes, hoisted out of the thread
    // scope: scoped spawns may only borrow what outlives the scope itself.
    // Lanes, not rings: with `pread` there are no reactors, but the claim
    // geometry (home groups, steal order, batch sizes) must stay identical, so
    // the lane count falls back to what `io_rings()` would have built.
    let n_rings = if reactors.is_empty() { crate::model::io_rings() } else { reactors.len() }.max(1);
    let group_sizes: Vec<usize> = claim_groups.iter().map(|g| g.len()).collect();
    let homes = ring_homes(&group_sizes, n_rings);
    let rings_in: Vec<usize> = (0..claim_groups.len())
        .map(|g| homes.iter().filter(|&&h| h == g).count())
        .collect();
    let batches: Vec<usize> = group_sizes
        .iter()
        .zip(&rings_in)
        .map(|(&glen, &r)| experts_per_batch().min(glen.div_ceil(r.max(1))).max(1))
        .collect();
    let claim_groups_ref = &claim_groups;
    let group_cursors_ref = &group_cursors;
    let batches_ref = &batches;
    let plans_ref = &plans;
    let gplans_ref = &gplans;
    let x_ref = x;
    let completed_ref = &completed;
    // Per-lane wall-time accumulator (or `None` for the no-tracking path). Copied
    // into each scoped thread so the atomic bumps in the accumulator's four counters
    // are the only synchronization the timing incurs.
    let timings_ref = ctx.timings;
    // Cache-admission gate: shared HeatTable ref + threshold, consulted per
    // streamed expert before inserting into the warm cache. Threshold 0 (the
    // default) admits everything.
    let heat_ref = ctx.heat;
    let admit_min_heat = cache_admit_min_heat();
    // Completion lane vs wave, resolved once: pread/regbuf are wave-shaped
    // measurement arms whose request shape *is* what they measure, so only the
    // uring engine streams per-expert. `COLI_IO_COMPLETION=0` is the escape
    // hatch back to the wave on uring too.
    let completion = completion_enabled() && matches!(io_engine(), IoEngine::Uring);
    // Fixed at cache construction; read once here so workers can run the zstd
    // encode (`WarmCache::prepare_insert`) without touching the cache lock.
    let cache_compress = ecache.map(|c| c.lock().compress_on_admit()).unwrap_or(false);

    // `(per-expert slots, the GPU lane's device-reduced partial)`. The partial is
    // `None` unless `COLI_CUDA_FUSED_REDUCE` is on and the layer had GPU experts.
    type LaneResults = (Vec<Option<EOut>>, Option<Vec<f32>>);
    // Wall clock of the 3-lane region. Every other lane counter is summed over
    // *threads*, which cannot distinguish a saturated lane from an idle one; this
    // is the denominator that makes them duty cycles.
    let t_lane = std::time::Instant::now();
    let results: Result<LaneResults, Error> = std::thread::scope(|scope| {
        // ---- I/O lanes: N io_uring rings in PARALLEL, lock-free (atomic) work-stealing ----
        // One thread per ring. Each atomically claims a batch of experts off its
        // home group's cursor (falling back to stealing from other groups),
        // serves warm-tier hits immediately, and streams the misses through *its own*
        // ring in one deep submit — so N reads run concurrently (which also parallelizes
        // dm-crypt decryption on encrypted volumes). The `pos`-ordered reduce is
        // order-independent, so which ring reads which expert never changes the output.
        //
        // Claim size, sized so **every ring gets work**.
        //
        // `experts_per_batch()` (`COLI_IO_BATCH`, default 16) is a submit-depth
        // ceiling chosen for prefill, where a chunk's routed union is ~69 experts
        // per layer. A *decode* token routes 8. With a fixed batch of 16, ring 0's
        // `fetch_add(16)` returns start 0 and claims all 8; rings 1..N get starts
        // 16/32/48, every one `>= n_plans`, and break without issuing a single
        // read. One ring then does the work of four — measured at **24% io duty
        // across 4 rings**, and ~0.6 GB/s where the same device gives 1.12 GB/s at
        // 4 rings under `iobench`.
        //
        // Ceil-divide instead, keeping the configured value as an upper bound:
        // decode gets ceil(8/4) = 2 and all four rings run; prefill gets
        // ceil(69/4) = 18, clamped back to 16, so its deep submits are unchanged.
        // Measured on GLM-5.2: decode 21.8 -> 14.8 s/tok, ttft 157 -> 116 s, io
        // duty 24% -> 90%. Under device-pure groups the same arithmetic runs
        // per group against the rings homed on it, for the same reason (the
        // group/home/batch tables are hoisted above the scope for lifetimes).
        // One thread per ring, or — when the engine needs none — one per lane the
        // rings would have occupied, so both engines steal work the same way.
        for ri in 0..n_rings {
            let ring = reactors.get(ri);
            let job_tx = job_tx.clone();
            let res_tx = res_tx.clone();
            let home = homes.get(ri).copied().unwrap_or(0);
            scope.spawn(move || {
                loop {
                    // Home group first; on dry, steal a device-pure window from
                    // the remaining groups so no ring idles while any device
                    // still has work.
                    let mut claim: Option<(usize, usize, usize)> = None;
                    for gi in std::iter::once(home)
                        .chain((0..claim_groups_ref.len()).filter(|&g| g != home))
                    {
                        let glen = claim_groups_ref[gi].len();
                        if glen == 0 {
                            continue;
                        }
                        let start = group_cursors_ref[gi].fetch_add(batches_ref[gi], Ordering::Relaxed);
                        if start >= glen {
                            continue; // this group is dry
                        }
                        claim = Some((gi, start, (start + batches_ref[gi]).min(glen)));
                        break;
                    }
                    let Some((gi, start, end)) = claim else {
                        break; // every group dry: no work left for this ring
                    };
                    let idxs = &claim_groups_ref[gi][start..end];
                    // split the claimed range into warm-tier hits (dispatch now) and
                    // misses (one deep async submit on this ring)
                    let mut miss: Vec<usize> = Vec::new();
                    for &idx in idxs {
                        let plan = &plans_ref[idx];
                        let key = (layer as u32, plan.expert as u32);
                        // Filter first: on "definitely absent" skip the mutex
                        // entirely and go straight to disk. Races are byte-safe
                        // (see `ResidencyHint`); the wait on the lock we do
                        // take is metered as evidence for/against sharding.
                        let hit = ecache.and_then(|c| {
                            match cache_hint.as_ref() {
                                Some(h) if !h.might_contain(key) => {
                                    h.note_fast_miss();
                                    None
                                }
                                _ => {
                                    let t_lock = std::time::Instant::now();
                                    let mut g = c.lock();
                                    if let Some(t) = timings_ref {
                                        t.add_cache_wait(t_lock.elapsed().as_micros() as u64);
                                    }
                                    g.get_hit(key)
                                }
                            }
                        });
                        match hit {
                            // Raw slot: bytes are ready — straight to a worker.
                            Some(CacheHit::Raw(bytes)) => {
                                if job_tx.send(Job::CacheRaw(idx, *bytes)).is_err() {
                                    return;
                                }
                            }
                            // Compressed slot: ship the refcounted frames; the
                            // zstd decode runs on the worker, not on this lane
                            // (and no longer inside the cache lock).
                            Some(CacheHit::Compressed(frames)) => {
                                if job_tx.send(Job::CacheZ(idx, frames)).is_err() {
                                    return;
                                }
                            }
                            None => miss.push(idx),
                        }
                    }
                    if miss.is_empty() {
                        continue;
                    }
                    let chunk_plans: Vec<&EPlan> = miss.iter().map(|&i| &plans_ref[i]).collect();
                    let t_io = std::time::Instant::now();
                    if completion {
                        // Owned-completion lane: each expert is forwarded the
                        // moment its last region lands, so the CPU pool starts
                        // on expert 1 while the rest are still on the wire.
                        //
                        // Sending on `job_tx` under this ring's lock cannot
                        // deadlock: workers never take a ring mutex, and iotune
                        // locks rings only between forwards. `add_io` spans the
                        // reap loop here, so io duty includes the per-expert
                        // forwarding — accepted semantics shift vs the wave.
                        // `completion` implies `IoEngine::Uring`, which implies
                        // `load_streaming` built rings, so this lane has one.
                        // Ending the lane rather than unwrapping keeps the
                        // impossible case harmless instead of loud.
                        let Some(ring) = ring else { break };
                        let mut r = ring.lock(); // this ring, uncontended (owned by this thread)
                        if !use_direct && fadvise_main_enabled() {
                            // Hint the NEXT claim window of this group while
                            // this one streams — genuinely ahead of its reads,
                            // unlike the wave path's same-wave hint. Best-effort:
                            // the window may be claimed by another ring, which
                            // only means the readahead lands in the page cache
                            // it shares. Same-group by construction, so the hint
                            // stays on this claim's device.
                            let ahead_end =
                                (end + batches_ref[gi]).min(claim_groups_ref[gi].len());
                            let mut ahead: Vec<(RawFd, u64, usize)> = Vec::new();
                            for &ai in &claim_groups_ref[gi][end..ahead_end] {
                                ahead.extend_from_slice(&expert_regions(&plans_ref[ai].entry, false));
                            }
                            if !ahead.is_empty() {
                                r.queue_willneed(&ahead);
                            }
                        }
                        let streamed = stream_experts_completion(&mut r, &chunk_plans, use_direct, |k, slab| {
                            job_tx.send(Job::Stream(miss[k], slab)).is_ok()
                        });
                        if let Some(t) = timings_ref {
                            t.add_io(t_io.elapsed().as_micros() as u64);
                        }
                        match streamed {
                            Ok(true) => continue,
                            // A dropped receiver means the collector saw an
                            // error and the forward is unwinding.
                            Ok(false) => return,
                            Err(e) => {
                                if res_tx.send(Err(e)).is_err() {
                                    peregrine_io::note_advisory_err("io lane error forward", &"collector already gone");
                                }
                                return;
                            }
                        }
                    }
                    // Wave path: the `COLI_IO_COMPLETION=0` escape hatch and the
                    // pread/regbuf measurement arms. One blocking submit for the
                    // whole claim; admission still runs on the worker
                    // (`Job::Stream`), unified with the completion lane.
                    let slabs = {
                        // `None` on the pread engine: `read_experts_batched`
                        // needs no ring there, and holding one would only
                        // serialize lanes that are not sharing anything.
                        let mut guard = ring.map(|rg| rg.lock());
                        read_experts_batched(guard.as_deref_mut(), &chunk_plans, use_direct)
                    };
                    if let Some(t) = timings_ref {
                        t.add_io(t_io.elapsed().as_micros() as u64);
                    }
                    let slabs: Vec<Bytes3> = match slabs {
                        Ok(s) => s,
                        Err(e) => {
                            if res_tx.send(Err(e)).is_err() {
                                peregrine_io::note_advisory_err("io lane error forward", &"collector already gone");
                            }
                            return;
                        }
                    };
                    for (&idx, bytes) in miss.iter().zip(slabs) {
                        if job_tx.send(Job::Stream(idx, bytes)).is_err() {
                            return;
                        }
                    }
                }
                // this ring's senders drop → CPU pool drains once all rings finish
            });
        }

        // ---- GPU lane: one batched expert_group for the layer's VRAM experts ----
        if let Some(g) = gpu {
            if !gplans_ref.is_empty() {
                let res_tx = res_tx.clone();
                scope.spawn(move || {
                    let jobs: Vec<(usize, Vec<f32>)> = gplans_ref.iter().map(|p| (p.e, p.xg.clone())).collect();
                    let t_gpu = std::time::Instant::now();
                    // Fused: the device folds every GPU expert into one
                    // `[s_n, hidden]` partial, so `dst`/`weights` describe the
                    // gathered rows in the same flattened order `jobs` builds
                    // them — plan by plan, rows within a plan in plan order.
                    let fused = if fused_reduce {
                        let dst: Vec<usize> = gplans_ref.iter().flat_map(|p| p.rows.iter().copied()).collect();
                        let rw: Vec<f32> = gplans_ref.iter().flat_map(|p| p.rw.iter().copied()).collect();
                        Some(g.compute_reduced(layer, &jobs, hidden, &dst, &rw, s_n))
                    } else {
                        None
                    };
                    let plain = if fused.is_none() { Some(g.compute(layer, &jobs, hidden)) } else { None };
                    if let Some(t) = timings_ref {
                        t.add_gpu(t_gpu.elapsed().as_micros() as u64);
                    }
                    let sent = match (fused, plain) {
                        (Some(Ok(partial)), _) => {
                            // One message for the whole lane: every GPU expert's
                            // contribution is already inside it.
                            completed_ref.fetch_add(gplans_ref.len(), Ordering::Relaxed);
                            res_tx.send(Ok(LaneMsg::GpuPartial(partial)))
                        }
                        (_, Some(Ok(hs))) => {
                            let mut last = Ok(());
                            for (gp, h) in gplans_ref.iter().zip(hs) {
                                completed_ref.fetch_add(1, Ordering::Relaxed);
                                let out = EOut { rows: gp.rows.clone(), rw: gp.rw.clone(), h };
                                last = res_tx.send(Ok(LaneMsg::Slot(gp.pos, out)));
                                if last.is_err() {
                                    break;
                                }
                            }
                            last
                        }
                        (Some(Err(e)), _) | (_, Some(Err(e))) => res_tx.send(Err(e)),
                        (None, None) => Ok(()), // unreachable: exactly one of the two ran
                    };
                    if sent.is_err() {
                        peregrine_io::note_advisory_err("gpu lane error forward", &"collector already gone");
                    }
                });
            }
        }

        // ---- CPU lane: pool of workers computing SwiGLU per disk expert ----
        for _ in 0..workers.max(1) {
            let job_rx = job_rx.clone();
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                loop {
                    // `recv`'s only error is `Disconnected`, which is this
                    // pool's shutdown signal: every ring has finished and
                    // dropped its sender.
                    let job = match job_rx.recv() {
                        Ok(job) => job,
                        Err(crossbeam_channel::RecvError) => break,
                    };
                    let (idx, bytes) = match job {
                        Job::CacheRaw(idx, bytes) => (idx, bytes),
                        Job::CacheZ(idx, frames) => match frames.materialize() {
                            Some(slab) => (idx, slab),
                            None => {
                                // Undecodable slot: drop it (with its
                                // decode-failure accounting) and re-stream from
                                // disk with the blocking oracle reader — this
                                // worker owns no ring. Rare by construction:
                                // non-zero traffic here means real bit rot.
                                let plan = &plans_ref[idx];
                                if let Some(c) = ecache {
                                    let t_lock = std::time::Instant::now();
                                    let mut g = c.lock();
                                    if let Some(t) = timings_ref {
                                        t.add_cache_wait(t_lock.elapsed().as_micros() as u64);
                                    }
                                    g.remove_corrupt((layer as u32, plan.expert as u32));
                                }
                                match restream_expert_blocking(&plan.entry) {
                                    Ok(slab) => (idx, slab),
                                    Err(e) => {
                                        if res_tx.send(Err(e)).is_err() {
                                            peregrine_io::note_advisory_err(
                                                "cpu lane error forward",
                                                &"collector already gone",
                                            );
                                        }
                                        break;
                                    }
                                }
                            }
                        },
                        Job::Stream(idx, bytes) => match ecache {
                            None => (idx, bytes),
                            Some(c) => {
                                // Admission moved here from the I/O lane: the
                                // zstd encode (`prepare_insert`) runs before the
                                // lock; only map insert + eviction run under it.
                                //
                                // Admission gate: only cache experts with demonstrated
                                // reuse (routing heat ≥ threshold). Heat is bumped after
                                // the reduce, so a first-ever routing reads 0 here —
                                // threshold 1 = "cache from the second routing on".
                                //
                                // **No heat table means the gate cannot be evaluated, so it
                                // does not apply.** It used to be `is_some_and`, i.e. "no
                                // table → admit nothing": `heat` is `Some` only when a GPU
                                // tier exists (`model.rs`), so on any CPU-only run setting
                                // this knob to 1 silently turned the demand path's cache
                                // admission off entirely — while the prefetch lane, which
                                // has no such gate, went on admitting everything. A knob
                                // documented as "filter one-off experts" instead inverted
                                // which lane owned the cache.
                                let expert = plans_ref[idx].expert;
                                let admit = admit_min_heat == 0
                                    || heat_ref.is_none_or(|h| h.get(layer, expert) >= admit_min_heat);
                                if admit {
                                    // Share-convert once: the cache slot and this
                                    // worker's compute alias the same refcounted
                                    // bytes. (The old ring-thread admission
                                    // deep-copied the whole slab instead.)
                                    let shared = bytes.map(|(w, s)| (w.into_shared(), s.into_shared()));
                                    let prepared = WarmCache::prepare_insert(shared.clone(), cache_compress);
                                    let t_lock = std::time::Instant::now();
                                    let mut g = c.lock();
                                    if let Some(t) = timings_ref {
                                        t.add_cache_wait(t_lock.elapsed().as_micros() as u64);
                                    }
                                    g.note_disk_read(layer as u32);
                                    g.insert_prepared((layer as u32, expert as u32), prepared);
                                    (idx, shared)
                                } else {
                                    let t_lock = std::time::Instant::now();
                                    let mut g = c.lock();
                                    if let Some(t) = timings_ref {
                                        t.add_cache_wait(t_lock.elapsed().as_micros() as u64);
                                    }
                                    g.note_disk_read(layer as u32);
                                    (idx, bytes)
                                }
                            }
                        },
                    };
                    let plan = &plans_ref[idx];
                    // Slab bytes consumed by this expert — the bandwidth-governor
                    // numerator (counted before the regions move into the Mlp).
                    if let Some(t) = timings_ref {
                        let nb: usize = bytes.iter().map(|(w, s)| w.len() + s.len()).sum();
                        t.add_cpu_bytes(nb as u64);
                    }
                    let [(gw, gs), (uw, us), (dw, ds)] = bytes;
                    let t_cpu = std::time::Instant::now();
                    let mlp = Mlp {
                        gate: rebuild(&plan.entry.plans[0], gw, gs),
                        up: rebuild(&plan.entry.plans[1], uw, us),
                        down: rebuild(&plan.entry.plans[2], dw, ds),
                    };
                    let nr = plan.rows.len();
                    let mut xg = vec![0f32; nr * hidden];
                    for (ri, &s) in plan.rows.iter().enumerate() {
                        xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x_ref[s * hidden..s * hidden + hidden]);
                    }
                    let h = mlp.swiglu(&xg, nr);
                    if let Some(t) = timings_ref {
                        t.add_cpu(t_cpu.elapsed().as_micros() as u64);
                    }
                    completed_ref.fetch_add(1, Ordering::Relaxed);
                    let out = EOut { rows: plan.rows.clone(), rw: plan.rw.clone(), h };
                    if res_tx.send(Ok(LaneMsg::Slot(plan.pos, out))).is_err() {
                        break;
                    }
                }
            });
        }

        // Drop the main thread's channel handles so the loops terminate once the
        // spawned threads finish; then collect exactly `n` results (or an error).
        drop(job_tx);
        drop(job_rx);
        drop(res_tx);

        let mut slots: Vec<Option<EOut>> = (0..n).map(|_| None).collect();
        let mut got = 0usize;
        // Under the fused reduce the GPU lane sends **one** message covering all
        // of its experts, so the collector expects that many fewer slots plus
        // exactly one partial. Deriving both from the same `fused_reduce` flag
        // the lane read is what keeps "how many messages are coming" from being
        // two independent opinions that can disagree and hang the forward.
        let gpu_slots = if fused_reduce { gplans_ref.len() } else { 0 };
        let want_slots = n - gpu_slots;
        let mut gpu_partial: Option<Vec<f32>> = None;
        let complete = |got: usize, partial: &Option<Vec<f32>>| {
            got == want_slots && (!fused_reduce || partial.is_some())
        };
        // A lane error must NOT return straight out of this closure: `res_rx`
        // lives in the caller's frame, so leaving it undrained lets the still-
        // running lanes fill the bounded result channel and block forever in
        // `send` — `thread::scope`'s join at the end of this closure would then
        // never complete and the whole decode wedges with no error surfaced.
        // Record the failure, stop collecting, and drain until every sender is
        // gone so all lanes can finish and be joined.
        let mut failure: Option<Error> = None;
        loop {
            match res_rx.recv() {
                Ok(Ok(LaneMsg::Slot(pos, eo))) => {
                    // Two results for one position would silently drop an
                    // expert's contribution from the layer output.
                    match slots.get_mut(pos) {
                        Some(slot) if slot.is_none() => *slot = Some(eo),
                        Some(_) => {
                            failure = Some(Error::Format(format!(
                                "concurrent MoE: duplicate result for expert slot {pos}"
                            )));
                            break;
                        }
                        None => {
                            failure = Some(Error::Format(format!(
                                "concurrent MoE: result slot {pos} out of range ({n} experts)"
                            )));
                            break;
                        }
                    }
                    got += 1;
                    if complete(got, &gpu_partial) {
                        break;
                    }
                }
                Ok(Ok(LaneMsg::GpuPartial(p))) => {
                    // Exactly one is expected; a second would double-count every
                    // GPU expert's contribution into the layer output.
                    if gpu_partial.is_some() {
                        failure = Some(Error::Format("concurrent MoE: duplicate GPU reduce partial".into()));
                        break;
                    }
                    if p.len() != s_n * hidden {
                        failure = Some(Error::Format(format!(
                            "concurrent MoE: GPU reduce partial is {} floats, expected {}",
                            p.len(),
                            s_n * hidden
                        )));
                        break;
                    }
                    gpu_partial = Some(p);
                    if complete(got, &gpu_partial) {
                        break;
                    }
                }
                Ok(Err(e)) => {
                    failure = Some(e);
                    break;
                }
                // channel closed: fine only if every expert already arrived
                Err(recv_err) => {
                    if complete(got, &gpu_partial) {
                        break;
                    }
                    // `completed` counts what the lanes finished computing, which
                    // distinguishes "a lane died before computing" from "results
                    // were computed but never delivered".
                    let done = completed_ref.load(Ordering::Relaxed);
                    let missing_partial = if fused_reduce && gpu_partial.is_none() { " (GPU partial missing)" } else { "" };
                    failure = Some(Error::Format(format!(
                        "concurrent MoE: io/cpu lane ended early ({got}/{want_slots} experts collected, \
                         {done} computed){missing_partial}: {recv_err}"
                    )));
                    break;
                }
            }
        }
        if let Some(e) = failure {
            // Unblock every lane still queued on a full channel, then report.
            while res_rx.recv().is_ok() {}
            return Err(e);
        }
        Ok((slots, gpu_partial))
    });
    // Recorded before `?` would return: a layer that failed still consumed wall
    // clock, and dropping it would flatter the duty cycle exactly when something
    // has gone wrong.
    if let Some(t) = timings_ref {
        t.add_lane_wall(t_lane.elapsed().as_micros() as u64);
    }
    let (slots, gpu_partial) = results?;

    // ---- deterministic reduce: scatter in fixed batch-union order ----
    //
    // Per-row parallel scatter, bit-identical to the historical serial loop:
    // `f32 +=` is not associative, so changing the order experts contribute to a
    // shared `out[s * hidden + d]` would change the result bits. The contract
    // (`moe_forward_parallel_matches_serial` in `mlp.rs`) is that **per row**
    // experts accumulate in batch-union (`pos`) order, so each row's sum is
    // bit-identical to the serial loop; rows are independent, so scattering rows
    // in parallel is bit-identical too. That is the only ordering the bit-identity
    // anchors actually pin, and this is the only ordering this code uses.
    //
    // The shape that made this safe to lift: each `EOut` carries its own rows
    // (always ascending `s`, first matching `kk`) so the per-row contribution
    // lists are built by walking `slots` in `pos` order and pushing each expert's
    // contribution into the row it claims. Then the `out[s * hidden..]` slot is
    // written from exactly one `par_rows_mut` worker, with no inter-worker aliasing.
    let t_reduce = std::time::Instant::now();
    let mut out = vec![0f32; s_n * hidden];
    // `row_contribs[s]` = ordered list of `(slice_of_hidden, weight)` in batch-union
    // order. `Box<[&f32]>` would be ideal but we need owned slices to ship across
    // the pool boundary; `&[f32]` inside the closure captures by reference, which
    // is the shape `par_rows_mut` was already designed around.
    let mut row_contribs: Vec<Vec<(&[f32], f32)>> = (0..s_n).map(|_| Vec::new()).collect();
    for eo in slots.iter().flatten() {
        for (ri, (&s, &wgt)) in eo.rows.iter().zip(&eo.rw).enumerate() {
            let src = &eo.h[ri * hidden..ri * hidden + hidden];
            if let Some(slot) = row_contribs.get_mut(s) {
                slot.push((src, wgt));
            }
        }
    }
    // Note: `par_rows_mut` accepts a closure Fn(usize, &mut [f32]) and runs each row
    // on a pool worker. The `+ Sync` bound on the closure is what lets the
    // contribution lists be borrowed across workers safely (read-only access).
    if s_n >= peregrine_par::PAR_ROWS_MIN {
        // borrowed once across workers — the closure reads `row_contribs[s]`
        let contribs: &[Vec<(&[f32], f32)>] = &row_contribs;
        peregrine_par::par_rows_mut(
            &mut out,
            hidden,
            s_n,
            peregrine_par::PAR_ROWS_MIN,
            |s, dst| {
                for &(src, wgt) in contribs[s].iter() {
                    for d in 0..hidden {
                        dst[d] += wgt * src[d];
                    }
                }
            },
        );
    } else {
        for s in 0..s_n {
            let dst = &mut out[s * hidden..s * hidden + hidden];
            for &(src, wgt) in row_contribs[s].iter() {
                for d in 0..hidden {
                    dst[d] += wgt * src[d];
                }
            }
        }
    }
    drop(row_contribs);
    // Drop the actual `EOut` slab ownership (the slices above were references
    // into `eo.h`; we still need `slots` for nothing else, so free it now).
    drop(slots);
    // The device-reduced GPU partial, added in a **fixed position** — after
    // every CPU contribution, in its own pass — rather than wherever the lane
    // happened to finish. Arrival order must not reach the arithmetic: that
    // would make the output depend on machine timing, which is the one thing no
    // adaptive path in this engine is allowed to do.
    if let Some(p) = gpu_partial {
        for (o, v) in out.iter_mut().zip(&p) {
            *o += v;
        }
    }
    if let Some(sh) = shared {
        let hs = sh.swiglu(x, s_n);
        for z in 0..s_n * hidden {
            out[z] += hs[z];
        }
    }
    if let Some(t) = ctx.timings {
        t.add_reduce(t_reduce.elapsed().as_micros() as u64);
    }

    // Accumulate routing frequency so the GPU tier can migrate hot experts into
    // VRAM (heat-ranked residency). Union hotness is the batched-relevant signal;
    // single-threaded here (after the reduce), so the lock-free bumps never race.
    if let Some(heat) = ctx.heat {
        for &e in &uniq {
            heat.bump(layer, e as usize);
        }
    }

    // Record this layer's routed set as the newest history frame so the prefetch
    // lane can predict the next token's experts. Single-threaded here (after the
    // reduce) — exactly one writer, no race.
    if let Some(rl) = ctx.route_log {
        rl.lock().push_layer(layer, uniq);
    }
    // Batched decode: record each sequence's *own* routed set (position s ↔ sequence
    // s) into its per-sequence history, for per-stream prediction.
    if let Some(multi) = ctx.route_log_multi {
        for (s, rh) in multi.iter().enumerate().take(s_n) {
            rh.lock().push_layer(layer, routed_at(&r, s));
        }
    }
    // Per position, ids *and* weights, before `Routed` is dropped. This is the
    // only place both exist together: `batch_union` above has already thrown the
    // weights away, and nothing downstream can recover them.
    if let Some(gt) = ctx.gate_trace {
        let mut gt = gt.lock();
        for s in 0..s_n {
            let keff = r.keff.get(s).copied().unwrap_or(0).max(0) as usize;
            let base = s * r.k;
            let ids: Vec<i32> = (0..keff).filter_map(|kk| r.idx.get(base + kk).copied()).collect();
            let w: Vec<f32> = (0..keff).filter_map(|kk| r.w.get(base + kk).copied()).collect();
            gt.push(layer, ids, w);
        }
    }
    Ok(out)
}

/// One expert queued for speculative prefetch: its `(layer, expert)` cache key and
/// the three tensor plans to stream. Built by [`prefetch_item`], consumed by
/// [`prefetch_read`] on the prefetch lane's own ring.
pub struct PrefetchItem {
    key: (u32, u32),
    entry: ExpertEntry,
}

impl PrefetchItem {
    /// The `(layer, expert)` warm-cache key this item populates.
    pub fn key(&self) -> (u32, u32) {
        self.key
    }
}

/// Build the streaming plan for one routed expert (gate/up/down), for the prefetch
/// lane. Mirrors the disk-plan construction in [`moe_forward_concurrent`], including
/// its use of the load-time [`ExpertIndex`] — the two must agree, or a prefetched
/// slab would not be the one a later demand read expects.
pub fn prefetch_item(
    index: Option<&ExpertIndex>,
    st: &SafeTensors,
    cfg: &Cfg,
    layer: usize,
    expert: usize,
) -> Result<PrefetchItem, Error> {
    let entry = entry_for(index, st, cfg, layer, expert)?;
    Ok(PrefetchItem { key: (layer as u32, expert as u32), entry })
}

/// Stream one prefetch item's six regions through `reactor` into an owned slab
/// (one batched submit) — the exact bytes the I/O lane would read, so a later hit
/// is bit-identical.
pub fn prefetch_read(reactor: &mut Reactor, item: &PrefetchItem, direct: bool) -> Result<peregrine_io::ExpertSlab, Error> {
    read_expert(Some(reactor), &item.entry, direct)
}

/// One expert queued for a page-cache *hint* (`fadvise(WILLNEED)`): its six
/// `(fd, offset, len)` regions to warm without streaming into the cache. Used for
/// low-confidence multi-path predictions, where a full prefetch isn't worth the
/// bandwidth but a cheap page-cache hint may still help a later miss.
pub struct HintItem {
    regions: Vec<(RawFd, u64, usize)>,
}

impl HintItem {
    /// The `(fd, offset, len)` regions to hint: two when the expert's runs
    /// coalesced, six otherwise.
    pub fn regions(&self) -> &[(RawFd, u64, usize)] {
        &self.regions
    }
}

/// Build a [`HintItem`] for one expert's six regions. Uses the **buffered** fds:
/// `fadvise` only populates the page cache, so it's a no-op for O_DIRECT reads and
/// the caller gates hints off under direct I/O.
pub fn prefetch_hint_item(
    index: Option<&ExpertIndex>,
    st: &SafeTensors,
    cfg: &Cfg,
    layer: usize,
    expert: usize,
) -> Result<HintItem, Error> {
    let entry = entry_for(index, st, cfg, layer, expert)?;
    // `direct: false` — this is the buffered-fd list by construction.
    Ok(HintItem { regions: expert_regions(&entry, false) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::QuantFmt;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    /// Device-pure partition: every plan index lands in exactly one group, groups
    /// come out in ascending-ordinal order preserving in-group plan order, and an
    /// fd the table has never seen is quarantined in the trailing `u8::MAX` group
    /// rather than polluting a real device's claim windows.
    #[test]
    fn device_claim_groups_partition_by_ordinal_and_quarantine_unknown_fds() {
        let table: HashMap<RawFd, u8> = [(10, 0u8), (11, 1u8), (12, 0u8)].into_iter().collect();
        // Plans interleaved across devices, one fd (99) unknown.
        let fds = [10, 11, 12, 99, 11, 10];
        let groups = device_claim_groups(fds.iter().copied(), &table);
        assert_eq!(groups, vec![vec![0, 2, 5], vec![1, 4], vec![3]]);
        let mut all: Vec<usize> = groups.concat();
        all.sort_unstable();
        assert_eq!(all, (0..fds.len()).collect::<Vec<_>>());
    }

    /// The blind path's contract, restated for the grouped machinery: a table
    /// mapping every fd to one ordinal yields a single group in plan order —
    /// claim windows are then identical to the historical shared cursor.
    #[test]
    fn device_claim_groups_single_device_is_the_identity_order() {
        let table: HashMap<RawFd, u8> = [(7, 3u8)].into_iter().collect();
        let groups = device_claim_groups([7, 7, 7, 7].into_iter(), &table);
        assert_eq!(groups, vec![vec![0, 1, 2, 3]]);
    }

    /// The whole correctness argument for device-pure claims, asserted end to
    /// end: the same forward, once through the blind cursor and once through a
    /// forced two-group split (shard fds alternately assigned ordinals 0/1 —
    /// the tiny model lives on one real device, so the split is synthetic on
    /// purpose), produces **bit-identical** output. Only claim/submission order
    /// may change; the `pos`-keyed reduce erases it.
    #[test]
    fn device_pure_claims_are_bit_identical_to_the_blind_cursor() -> Result<(), Error> {
        use peregrine_core::Cfg;

        let dir = std::env::temp_dir()
            .join(format!("peregrine_devsched_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(Error::Io)?;
        }
        crate::testkit::build_tiny_model_seeded(&dir, 0xD5C4ED)?;
        let cfg = Cfg::load(&dir)?;
        let st = SafeTensors::open(&dir)?;

        // Two rings: grouping only engages with more than one, and two claim
        // groups against two rings exercises homes + stealing both.
        let reactors = match (Reactor::new(32), Reactor::new(32)) {
            (Ok(a), Ok(b)) => vec![Mutex::new(a), Mutex::new(b)],
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_dir_all(&dir).map_err(Error::Io)?;
                return Ok(());
            }
        };
        let table: HashMap<RawFd, u8> = st
            .fd_devices()
            .into_iter()
            .enumerate()
            .map(|(i, (fd, _))| (fd, (i % 2) as u8))
            .collect();
        assert!(
            table.values().any(|&d| d == 1),
            "fixture must span two synthetic ordinals for the test to bite"
        );

        let (hidden, e_n) = (cfg.hidden as usize, cfg.n_experts as usize);
        let s_n = 3usize;
        let layer = cfg.first_dense as usize;
        // Deterministic inputs, no rand dependency: a small LCG.
        let mut state = 0x00C0_FFEEu64;
        let mut f = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let x: Vec<f32> = (0..s_n * hidden).map(|_| f()).collect();
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| f() * 0.1).collect();

        let ctx_of = |fd_devices| ForwardCtx {
            gate_trace: None,
            st: &st,
            absorb: false,
            dsa: false,
            reactors: &reactors,
            gpu: None,
            gpu_dense: None,
            workers: 2,
            cfg: &cfg,
            stream_experts: true,
            ecache: None,
            route_log: None,
            calib: None,
            route_log_multi: None,
            direct: false,
            heat: None,
            spill: None,
            timings: None,
            balancer: None,
            heat_counts: None,
            layout_schedule: None,
            affinity: None,
            expert_index: None,
            fd_devices,
        };
        let blind =
            moe_forward_concurrent(&ctx_of(None), layer, &x, &router_w, &router_bias, None, s_n)?;
        let grouped = moe_forward_concurrent(
            &ctx_of(Some(&table)),
            layer,
            &x,
            &router_w,
            &router_bias,
            None,
            s_n,
        )?;
        assert_eq!(blind.len(), grouped.len());
        for (i, (a, b)) in blind.iter().zip(&grouped).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "diverged at f32 index {i}");
        }
        std::fs::remove_dir_all(&dir).map_err(Error::Io)?;
        Ok(())
    }

    /// Ring homes are proportional to group size, and a non-empty group is never
    /// orphaned while rings remain — a device with little work must not wait for
    /// a stealing ring to run dry elsewhere before its reads start.
    #[test]
    fn ring_homes_are_proportional_and_never_orphan_a_nonempty_group() {
        assert_eq!(ring_homes(&[80, 20], 4), vec![0, 0, 0, 1]);
        // Fewer rings than groups: valid homes, one per ring, no panic; the
        // unhomed group is reached via stealing.
        let homes = ring_homes(&[10, 10, 10], 2);
        assert_eq!(homes.len(), 2);
        assert!(homes.iter().all(|&h| h < 3));
        // Empty groups draw nothing.
        assert_eq!(ring_homes(&[0, 5], 2), vec![1, 1]);
        // Degenerate inputs hold the shape contract.
        assert_eq!(ring_homes(&[], 3), vec![0, 0, 0]);
        assert_eq!(ring_homes(&[0, 0], 2), vec![0, 0]);
    }

    /// A fresh tiny-model checkpoint on disk. Mirrors `model.rs`'s `tmp_model_dir`:
    /// this crate has no `tempfile` dependency, and the tests are per-process.
    fn tmp_expert_index_dir(tag: &str) -> Result<std::path::PathBuf, Error> {
        let d = std::env::temp_dir().join(format!("peregrine_expert_index_{}_{}", std::process::id(), tag));
        if d.exists() {
            std::fs::remove_dir_all(&d).map_err(Error::Io)?;
        }
        crate::testkit::build_tiny_model(&d)?;
        Ok(d)
    }

    /// The index is only safe if it resolves an expert to exactly what deriving it
    /// per request would have. Field for field, every expert — this is the whole
    /// correctness argument for replacing `tplan` with a lookup, and it is what
    /// catches a transposition slip like `down_proj` being `(hidden, mi)` while
    /// gate/up are `(mi, hidden)`.
    #[test]
    fn expert_index_agrees_with_deriving_per_request() -> Result<(), Error> {
        let dir = tmp_expert_index_dir("agree")?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let index = ExpertIndex::build(&st, &cfg);
        assert!(index.resolved() > 0, "fixture indexed no experts at all");

        let hidden = cfg.hidden as usize;
        let mi = cfg.moe_inter as usize;
        let mut checked = 0usize;
        for layer in (cfg.first_dense as usize)..=(cfg.n_layers as usize) {
            for e in 0..(cfg.n_experts as usize) {
                let Some(entry) = index.get(layer, e) else { continue };
                let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
                let want = [
                    tplan(&st, &p("gate_proj.weight"), mi, hidden)?,
                    tplan(&st, &p("up_proj.weight"), mi, hidden)?,
                    tplan(&st, &p("down_proj.weight"), hidden, mi)?,
                ];
                for (got, want) in entry.plans.iter().zip(want.iter()) {
                    assert_eq!((got.w_fd, got.w_off, got.w_len), (want.w_fd, want.w_off, want.w_len));
                    assert_eq!((got.s_fd, got.s_off, got.s_len), (want.s_fd, want.s_off, want.s_len));
                    assert_eq!((got.w_fd_direct, got.s_fd_direct), (want.w_fd_direct, want.s_fd_direct));
                    // The shape/format half — the "correct type" the request needs.
                    assert_eq!((got.fmt, got.o, got.i, got.gs), (want.fmt, want.o, want.i, want.gs));
                }
                checked += 1;
            }
        }
        assert_eq!(checked, index.resolved(), "walked a different set than the index holds");
        Ok(())
    }

    /// One token's working set is `topk` experts per sparse layer, sized from
    /// each layer's own experts. This is the threshold the protect default and
    /// the capacity-vs-policy reading both hang off, so it has to be derived, not
    /// assumed uniform — a precision-tiered container stores different layers at
    /// different widths.
    #[test]
    fn per_token_bytes_counts_topk_experts_in_every_sparse_layer() -> Result<(), Error> {
        let dir = tmp_expert_index_dir("workingset")?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let index = ExpertIndex::build(&st, &cfg);

        let layers: Vec<usize> = ((cfg.first_dense as usize)..=(cfg.n_layers as usize))
            .filter(|&l| (0..cfg.n_experts as usize).any(|e| index.get(l, e).is_some()))
            .collect();
        assert!(!layers.is_empty(), "fixture has no sparse layers");

        // Independently: sum each sparse layer's own per-expert bytes × topk.
        let mut want = 0u64;
        for &l in &layers {
            let e = (0..cfg.n_experts as usize).find_map(|e| index.get(l, e)).ok_or_else(|| {
                Error::Format("layer reported sparse but resolved no expert".into())
            })?;
            let per: u64 = e.plans.iter().map(|t| t.w_len as u64 + t.s_len as u64).sum();
            want += per * cfg.topk.max(0) as u64;
        }
        assert_eq!(index.per_token_bytes(&cfg), want);
        assert!(want > 0);
        Ok(())
    }

    /// A `None` entry must be indistinguishable from never having had an index:
    /// `plans_for` falls back to deriving, so an unusual container behaves exactly
    /// as it did before the map existed.
    #[test]
    fn absent_index_entry_falls_back_to_deriving() -> Result<(), Error> {
        let dir = tmp_expert_index_dir("fallback")?;
        let st = SafeTensors::open(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let index = ExpertIndex::build(&st, &cfg);
        let layer = cfg.first_dense as usize;

        let with = entry_for(Some(&index), &st, &cfg, layer, 0)?.plans;
        let without = entry_for(None, &st, &cfg, layer, 0)?.plans;
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!((a.w_fd, a.w_off, a.w_len), (b.w_fd, b.w_off, b.w_len));
            assert_eq!((a.s_fd, a.s_off, a.s_len), (b.s_fd, b.s_off, b.s_len));
            assert_eq!((a.fmt, a.o, a.i, a.gs), (b.fmt, b.o, b.i, b.gs));
        }
        // Out of range on either axis resolves to `None` rather than panicking or
        // reading a neighbouring expert's row.
        assert!(index.get(layer, cfg.n_experts as usize).is_none());
        assert!(index.get(cfg.n_layers as usize + 1, 0).is_none());
        Ok(())
    }

    /// The whole point of the pread engine is that it needs no ring. Prove the
    /// ring-free path returns the same bytes, because a host with no io_uring
    /// has nothing to fall back to if this is wrong — and nothing on this box
    /// exercises it, since this box *has* io_uring.
    ///
    /// Deliberately calls the ring-free helpers directly rather than going
    /// through `io_engine()`: that resolves into a `OnceLock` latched by
    /// whichever test ran first, so an env-var-driven test of engine selection
    /// would pass or fail depending on test order.
    #[test]
    fn the_ring_free_read_path_returns_the_same_bytes() -> Result<(), Error> {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("peregrine_noring_{}", std::process::id()));
        let regions_src: Vec<Vec<u8>> = (0..4usize).map(|i| vec![(i as u8).wrapping_mul(37); 1024 + i * 97]).collect();
        {
            let mut f = std::fs::File::create(&path)?;
            for r in &regions_src {
                f.write_all(r)?;
            }
            f.flush()?;
        }
        let rf = std::fs::File::open(&path)?;
        let fd = rf.as_raw_fd();
        let mut off = 0u64;
        let mut regions: Vec<(RawFd, u64, usize)> = Vec::new();
        for r in &regions_src {
            regions.push((fd, off, r.len()));
            off += r.len() as u64;
        }

        // No reactor anywhere in this call.
        let got = read_regions_with_retry(None, &regions)?;
        assert_eq!(got.len(), regions_src.len());
        for (i, (g, want)) in got.iter().zip(regions_src.iter()).enumerate() {
            assert_eq!(&g[..], &want[..], "region {i} differs on the ring-free path");
        }

        // `complete_short_read` is the piece that has to loop to a full buffer
        // when a positioned read comes back short; check it fills exactly.
        let mut buf = vec![0u8; regions_src[2].len()];
        complete_short_read(None, fd, regions[2].1, &mut buf)?;
        assert_eq!(&buf[..], &regions_src[2][..], "short-read completion filled wrong bytes");

        // The hints are advisory, but they must not error on a healthy fd —
        // a failure here would spam the advisory log on every claim.
        assert!(
            peregrine_io::fadvise_many(&regions, peregrine_io::FADV_WILLNEED).is_ok(),
            "willneed on a healthy fd should succeed"
        );

        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn read_expert_batched_bytes_identical() -> Result<(), Error> {
        // Six regions (gate/up/down × weight+scale) laid into one file; the batched
        // read must return exactly those bytes, in order — proving the single-submit
        // path is byte-identical to six separate reads.
        let path = std::env::temp_dir().join(format!("peregrine_read_expert_{}", std::process::id()));
        let regions: Vec<Vec<u8>> = (0..6usize)
            .map(|k| (0..(16 + k * 7)).map(|b| (b as u8).wrapping_add(k as u8 * 31)).collect())
            .collect();
        let mut f = std::fs::File::create(&path)?;
        let mut offs = Vec::new();
        let mut cur = 0u64;
        for r in &regions {
            offs.push(cur);
            f.write_all(r)?;
            cur += r.len() as u64;
        }
        f.sync_all()?;
        let rf = std::fs::File::open(&path)?;
        let fd = rf.as_raw_fd();

        let tp = |wi: usize, si: usize| TPlan {
            w_fd: fd,
            w_off: offs[wi],
            w_len: regions[wi].len(),
            s_fd: fd,
            s_off: offs[si],
            s_len: regions[si].len(),
            w_fd_direct: None,
            s_fd_direct: None,
            fmt: QuantFmt::Int4,
            o: 1,
            i: 1,
            gs: 0,
        };
        let plans = [tp(0, 1), tp(2, 3), tp(4, 5)];
        // Weights and scales alternate in this fixture, so neither run is
        // contiguous and the expert deliberately takes the six-region path.
        let entry = ExpertEntry { w_run: merge_run(&plans, true), s_run: merge_run(&plans, false), plans };
        assert!(entry.w_run.is_none() && entry.s_run.is_none(), "interleaved fixture must not coalesce");

        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let slab = read_expert(Some(&mut reactor), &entry, false)?;
        // slab regions are `Bytes`; compare their exposed byte slices to the source
        assert_eq!(&slab[0].0[..], &regions[0][..]);
        assert_eq!(&slab[0].1[..], &regions[1][..]);
        assert_eq!(&slab[1].0[..], &regions[2][..]);
        assert_eq!(&slab[1].1[..], &regions[3][..]);
        assert_eq!(&slab[2].0[..], &regions[4][..]);
        assert_eq!(&slab[2].1[..], &regions[5][..]);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    /// A coalesced read must be split by each tensor's own offset, not by its
    /// position. This fixture reproduces the real container's layout — the three
    /// `.qs` scales pooled first, then the three weights, both groups in
    /// **alphabetical** order (down, gate, up) rather than the gate/up/down order
    /// the slab wants — so a positional split silently loads gate's bytes into
    /// down's matrix. Every region carries distinct content so that swap fails
    /// here instead of turning into plausible-looking activations.
    #[test]
    fn coalesced_read_splits_by_offset_not_position() -> Result<(), Error> {
        let path = std::env::temp_dir().join(format!("peregrine_merge_{}", std::process::id()));
        // disk order: ds, gs, us, dw, gw, uw — scales pooled at the front.
        let regions: [Vec<u8>; 6] = [
            vec![0xD5; 24], // down scale
            vec![0x65; 8],  // gate scale
            vec![0x55; 8],  // up scale
            vec![0xDD; 96], // down weight
            vec![0x66; 96], // gate weight
            vec![0x5A; 96], // up weight
        ];
        let mut f = std::fs::File::create(&path)?;
        let mut offs = [0u64; 6];
        let mut cur = 0u64;
        for (i, r) in regions.iter().enumerate() {
            offs[i] = cur;
            f.write_all(r)?;
            cur += r.len() as u64;
        }
        f.sync_all()?;
        let rf = std::fs::File::open(&path)?;
        let fd = rf.as_raw_fd();
        let tp = |wi: usize, si: usize| TPlan {
            w_fd: fd,
            w_off: offs[wi],
            w_len: regions[wi].len(),
            s_fd: fd,
            s_off: offs[si],
            s_len: regions[si].len(),
            w_fd_direct: None,
            s_fd_direct: None,
            fmt: QuantFmt::Int4,
            o: 1,
            i: 1,
            gs: 0,
        };
        // gate, up, down — pointing at their scattered-on-disk homes.
        let plans = [tp(4, 1), tp(5, 2), tp(3, 0)];
        let entry = ExpertEntry { w_run: merge_run(&plans, true), s_run: merge_run(&plans, false), plans };
        let w = entry.w_run.ok_or_else(|| Error::Format("weights should coalesce".into()))?;
        let s = entry.s_run.ok_or_else(|| Error::Format("scales should coalesce".into()))?;
        assert_eq!((w.off, w.len), (offs[3], 96 * 3));
        assert_eq!((s.off, s.len), (offs[0], 24 + 8 + 8));
        assert_eq!(expert_regions(&entry, false).len(), 2, "coalesced expert must issue two reads");

        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let slab = read_expert(Some(&mut reactor), &entry, false)?;
        // gate/up/down, each paired with its own scale — the split has to undo
        // the alphabetical on-disk ordering.
        assert_eq!(&slab[0].0[..], &regions[4][..], "gate weight");
        assert_eq!(&slab[0].1[..], &regions[1][..], "gate scale");
        assert_eq!(&slab[1].0[..], &regions[5][..], "up weight");
        assert_eq!(&slab[1].1[..], &regions[2][..], "up scale");
        assert_eq!(&slab[2].0[..], &regions[3][..], "down weight");
        assert_eq!(&slab[2].1[..], &regions[0][..], "down scale");

        // The six views tile their two extents exactly, so the cache budgets a
        // coalesced expert at its real size rather than treble-counting it.
        let total: usize = slab.iter().map(|(w, s)| w.footprint() + s.footprint()).sum();
        assert_eq!(total, w.len + s.len);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    /// Build the completion-lane fixture: one file holding an interleaved
    /// (six-region) expert and a coalesced (two-region) expert, so a single
    /// claim mixes both per-expert region counts. Returns the open file (keeps
    /// the fd alive), its path, and the two entries.
    fn completion_fixture(
        tag: &str,
    ) -> Result<(std::fs::File, std::path::PathBuf, ExpertEntry, ExpertEntry), Error> {
        let path = std::env::temp_dir().join(format!("peregrine_completion_{}_{}", std::process::id(), tag));
        // Expert A: weight/scale alternate on disk → merge_run finds no
        // contiguous run and the expert reads six regions. Expert B mirrors the
        // real container: scales pooled, then weights, both alphabetical
        // (down, gate, up) → two coalesced regions. Distinct fills per region
        // so a positional mix-up fails loudly.
        let a_regions: Vec<Vec<u8>> = (0..6usize)
            .map(|k| (0..(24 + k * 11)).map(|b| (b as u8).wrapping_mul(3).wrapping_add(k as u8 * 37)).collect())
            .collect();
        let b_regions: [Vec<u8>; 6] = [
            vec![0xD5; 24], // down scale
            vec![0x65; 8],  // gate scale
            vec![0x55; 8],  // up scale
            vec![0xDD; 96], // down weight
            vec![0x66; 96], // gate weight
            vec![0x5A; 96], // up weight
        ];
        let mut f = std::fs::File::create(&path)?;
        let mut offs: Vec<u64> = Vec::new();
        let mut cur = 0u64;
        for r in a_regions.iter().chain(b_regions.iter()) {
            offs.push(cur);
            f.write_all(r)?;
            cur += r.len() as u64;
        }
        f.sync_all()?;
        let rf = std::fs::File::open(&path)?;
        let fd = rf.as_raw_fd();
        let lens: Vec<usize> = a_regions.iter().chain(b_regions.iter()).map(Vec::len).collect();
        let tp = |wi: usize, si: usize| TPlan {
            w_fd: fd,
            w_off: offs[wi],
            w_len: lens[wi],
            s_fd: fd,
            s_off: offs[si],
            s_len: lens[si],
            w_fd_direct: None,
            s_fd_direct: None,
            fmt: QuantFmt::Int4,
            o: 1,
            i: 1,
            gs: 0,
        };
        let a_plans = [tp(0, 1), tp(2, 3), tp(4, 5)];
        let entry_a = ExpertEntry { w_run: merge_run(&a_plans, true), s_run: merge_run(&a_plans, false), plans: a_plans };
        assert!(entry_a.w_run.is_none() && entry_a.s_run.is_none(), "interleaved expert must not coalesce");
        // gate, up, down — pointing at their scattered-on-disk homes (offsets
        // 6.. are expert B's block).
        let b_plans = [tp(6 + 4, 6 + 1), tp(6 + 5, 6 + 2), tp(6 + 3, 6)];
        let entry_b = ExpertEntry { w_run: merge_run(&b_plans, true), s_run: merge_run(&b_plans, false), plans: b_plans };
        assert!(entry_b.w_run.is_some() && entry_b.s_run.is_some(), "pooled expert must coalesce");
        assert_eq!(expert_regions(&entry_b, false).len(), 2);
        Ok((rf, path, entry_a, entry_b))
    }

    fn eplan(expert: usize, entry: ExpertEntry) -> EPlan {
        EPlan { pos: expert, expert, rows: Vec::new(), rw: Vec::new(), entry }
    }

    /// The owned-completion lane must deliver exactly the bytes the blocking
    /// wave path delivers — same regions, same carve — for both the six-region
    /// and the coalesced two-region expert shapes, with each expert forwarded
    /// exactly once. Delivery *order* is explicitly unspecified (completions
    /// may arrive in any order); byte content is not.
    #[test]
    fn completion_lane_bytes_identical_to_wave() -> Result<(), Error> {
        let (_rf, path, entry_a, entry_b) = completion_fixture("ident")?;
        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        // Oracle first: the wave path (untouched code) on the same entries.
        let want_a = read_expert(Some(&mut reactor), &entry_a, false)?;
        let want_b = read_expert(Some(&mut reactor), &entry_b, false)?;

        let eplans = [eplan(0, entry_a), eplan(1, entry_b)];
        let plan_refs: Vec<&EPlan> = eplans.iter().collect();
        let mut got: Vec<(usize, peregrine_io::ExpertSlab)> = Vec::new();
        let all = stream_experts_completion(&mut reactor, &plan_refs, false, |k, slab| {
            got.push((k, slab));
            true
        })?;
        assert!(all, "every expert must be forwarded when the consumer stays");
        got.sort_by_key(|(k, _)| *k);
        let ks: Vec<usize> = got.iter().map(|(k, _)| *k).collect();
        assert_eq!(ks, vec![0, 1], "each expert forwarded exactly once");
        for ((_, got), want) in got.iter().zip([&want_a, &want_b]) {
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(&g.0[..], &w.0[..], "weight bytes identical to the wave path");
                assert_eq!(&g.1[..], &w.1[..], "scale bytes identical to the wave path");
            }
        }
        // The lane must leave the ring clean: the legacy wave path (which
        // refuses to run while owned reads are in flight) works right after.
        let again = read_expert(Some(&mut reactor), &entry_a, false)?;
        for (g, w) in again.iter().zip(want_a.iter()) {
            assert_eq!(&g.0[..], &w.0[..]);
            assert_eq!(&g.1[..], &w.1[..]);
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }

    /// A consumer that disappears mid-stream (dropped channel → `forward`
    /// returns false) stops the stream with `Ok(false)` — and the quiesce on
    /// the way out leaves the ring reusable, not poisoned by in-flight owned
    /// reads.
    #[test]
    fn completion_lane_stops_cleanly_when_consumer_gone() -> Result<(), Error> {
        let (_rf, path, entry_a, entry_b) = completion_fixture("gone")?;
        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let eplans = [eplan(0, entry_a), eplan(1, entry_b)];
        let plan_refs: Vec<&EPlan> = eplans.iter().collect();
        let mut forwards = 0usize;
        let all = stream_experts_completion(&mut reactor, &plan_refs, false, |_, _| {
            forwards += 1;
            false // consumer gone after the first delivery
        })?;
        assert!(!all, "a gone consumer must report Ok(false), not success");
        assert_eq!(forwards, 1, "the stream must stop at the refusing forward");
        // Ring must be clean for the next claim on this lane.
        let want_a = read_expert(Some(&mut reactor), &entry_a, false)?;
        assert_eq!(want_a[0].0.len(), entry_a.plans[0].w_len);
        std::fs::remove_file(&path)?;
        Ok(())
    }
}

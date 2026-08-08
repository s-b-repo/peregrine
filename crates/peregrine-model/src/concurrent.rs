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

use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use peregrine_core::{Cfg, Context, Error, QtInfo, SafeTensors};
use peregrine_io::{Bytes, Reactor, ReadReq, WarmCache};

use crate::gpu::{GpuTier, HeatTable};
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
    /// Per-**sequence** routing history for batched decode: `route_log_multi[s]` is
    /// sequence `s`'s own history (position `s` ↔ sequence `s`), so each concurrent
    /// stream predicts and prefetches from its *own* routing rather than the weak
    /// cross-sequence union. `None` on the single-stream path (which uses `route_log`).
    pub route_log_multi: Option<&'a [&'a Mutex<RouteHistory>]>,
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

/// One expert's streaming+compute plan: which rows route to it (+ gate weights),
/// where its gate/up/down tensors live on disk, and its batch-union position
/// (`pos`) for the deterministic ordered reduce (GPU-resident experts take the
/// intervening positions).
struct EPlan {
    pos: usize,
    expert: usize,
    rows: Vec<usize>,
    rw: Vec<f32>,
    gate: TPlan,
    up: TPlan,
    down: TPlan,
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
    let (w_fd, w_off, w_len) = st.region(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))?;
    let sname = format!("{name}.qs");
    let (s_fd, s_off, s_len) = st.region(&sname).ok_or_else(|| Error::Format(format!("missing tensor {sname}")))?;
    let w_fd_direct = st.region_direct(name).map(|(fd, _, _)| fd);
    let s_fd_direct = st.region_direct(&sname).map(|(fd, _, _)| fd);
    Ok(TPlan { w_fd, w_off, w_len, s_fd, s_off, s_len, w_fd_direct, s_fd_direct, fmt, o, i, gs: info.gs as usize })
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
fn read_regions(r: &mut Reactor, regions: &[(RawFd, u64, usize)], direct: bool) -> Result<Vec<Bytes>, Error> {
    if direct && io_engine() != IoEngine::Pread {
        return r
            .read_direct_aligned(regions)
            .ctx(|| "io_uring O_DIRECT zero-copy expert read".to_string());
    }
    let mut bufs: Vec<Vec<u8>> = regions.iter().map(|&(_, _, len)| vec![0u8; len]).collect();
    {
        let mut reqs: Vec<ReadReq> = bufs
            .iter_mut()
            .zip(regions)
            .map(|(b, &(fd, off, _))| ReadReq { fd, offset: off, buf: b.as_mut_slice(), tag: 0 })
            .collect();
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
            IoEngine::Uring => r.read_many(&mut reqs).ctx(|| "io_uring batched expert read".to_string())?,
        };
        for (i, &n) in res.iter().enumerate() {
            if n < 0 {
                return Err(Error::Io(std::io::Error::from_raw_os_error((-n) as i32)));
            }
            let done = n as usize;
            if done < bufs[i].len() {
                let (fd, off, _) = regions[i];
                r.read_exact(fd, off + done as u64, &mut bufs[i][done..])
                    .ctx(|| "io_uring short-read completion".to_string())?;
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
fn io_engine() -> IoEngine {
    static V: std::sync::OnceLock<IoEngine> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("COLI_IO_ENGINE").as_deref() {
        Ok("pread") => IoEngine::Pread,
        Ok("regbuf") => IoEngine::RegBuf,
        // Historical spelling: `COLI_REGBUF=1` was documented and benchmarked
        // for a year while being read by no code at all. Honour it here so the
        // knob finally means something, rather than deleting it and silently
        // changing what a published benchmark arm did.
        _ if matches!(std::env::var("COLI_REGBUF").as_deref(), Ok("1") | Ok("true")) => IoEngine::RegBuf,
        _ => IoEngine::Uring,
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
fn read_expert(r: &mut Reactor, gate: &TPlan, up: &TPlan, down: &TPlan, direct: bool) -> Result<peregrine_io::ExpertSlab, Error> {
    // pick the weight/scale fd for a tensor: O_DIRECT twin when `direct`, else buffered.
    let wfd = |t: &TPlan| if direct { t.w_fd_direct.unwrap_or(t.w_fd) } else { t.w_fd };
    let sfd = |t: &TPlan| if direct { t.s_fd_direct.unwrap_or(t.s_fd) } else { t.s_fd };
    let regions = [
        (wfd(gate), gate.w_off, gate.w_len),
        (sfd(gate), gate.s_off, gate.s_len),
        (wfd(up), up.w_off, up.w_len),
        (sfd(up), up.s_off, up.s_len),
        (wfd(down), down.w_off, down.w_len),
        (sfd(down), down.s_off, down.s_len),
    ];
    pack_slab(read_regions(r, &regions, direct)?)
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
fn read_experts_batched(r: &mut Reactor, plans: &[&EPlan], direct: bool) -> Result<Vec<peregrine_io::ExpertSlab>, Error> {
    let n = plans.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // one (fd, offset, len) per region, in gate/up/down × (weight, scale) order. In
    // direct mode use the O_DIRECT twin fd (falling back per-region if a twin is
    // somehow missing); the reader applies the block alignment.
    let mut regions: Vec<(RawFd, u64, usize)> = Vec::with_capacity(6 * n);
    for p in plans {
        for t in [&p.gate, &p.up, &p.down] {
            let (wfd, sfd) = if direct {
                (t.w_fd_direct.unwrap_or(t.w_fd), t.s_fd_direct.unwrap_or(t.s_fd))
            } else {
                (t.w_fd, t.s_fd)
            };
            regions.push((wfd, t.w_off, t.w_len));
            regions.push((sfd, t.s_off, t.s_len));
        }
    }
    // Buffered path: hint the kernel to start readahead on every region before the
    // batched read. The advice fires from the same ring in one submit, so it costs
    // one extra syscall and can overlap NVMe queue depth with our submit ceremony.
    // Purely advisory — bytes returned are identical either way.
    if !direct && fadvise_main_enabled() {
        // Soft-failure only: readahead failures never affect correctness.
        if let Err(e) = r.fadvise_willneed_many(&regions) {
            peregrine_io::note_advisory_err("fadvise willneed (batched readahead)", &e);
        }
    }
    // one deep submit for all 6·n regions (buffered) or per-region aligned DMA
    // (direct, zero-copy); bytes come back in region order, six per expert.
    let bytes = match read_regions(r, &regions, direct) {
        Ok(b) => b,
        Err(e) if io_recovery_enabled() && !direct => {
            // Retry ladder: on a batched-read failure, re-issue each region as
            // an individual `read_exact_retry` (which handles EIO/EAGAIN/EINTR
            // with backoff). Trades throughput for resilience — used only after
            // the fast path has already failed. O_DIRECT is skipped because the
            // recovery path uses buffered reads.
            eprintln!("[io-recovery] batched read failed ({e}); retrying regions individually");
            read_regions_with_retry(r, &regions)?
        }
        Err(e) => return Err(e),
    };
    let mut bytes = bytes.into_iter();
    let mut slabs: Vec<peregrine_io::ExpertSlab> = Vec::with_capacity(n);
    for _ in 0..n {
        let six: Vec<Bytes> = bytes.by_ref().take(6).collect();
        slabs.push(pack_slab(six)?);
    }
    // Optional page-cache release for long-running RSS-bounded workloads. Only
    // useful when the warm cache is off / cold — a hit would otherwise re-read the
    // pages we just dropped. Purely advisory, so a soft failure is harmless.
    if !direct && fadvise_drop_enabled() {
        for &(fd, off, len) in &regions {
            if let Err(e) = r.fadvise_dontneed(fd, off, len) {
                peregrine_io::note_advisory_err("fadvise dontneed (page-cache release)", &e);
            }
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
fn read_regions_with_retry(r: &mut Reactor, regions: &[(RawFd, u64, usize)]) -> Result<Vec<Bytes>, Error> {
    let mut out: Vec<Bytes> = Vec::with_capacity(regions.len());
    for &(fd, off, len) in regions {
        let mut buf = vec![0u8; len];
        r.read_exact_retry(fd, off, &mut buf, 3)
            .ctx(|| format!("io_uring per-region retry @ off={off} len={len}"))?;
        out.push(Bytes::from(buf));
    }
    Ok(out)
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
    let use_direct = ctx.direct; // O_DIRECT streaming (page-cache-bypassing); Copy bool
    let reactors = ctx.reactors;
    if reactors.is_empty() {
        return Err(Error::Format("streaming mode without io_uring reactors".into()));
    }
    let hidden = cfg.hidden as usize;
    let (e_n, mi, k) = (cfg.n_experts as usize, cfg.moe_inter as usize, cfg.topk as usize);
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
                // Exhaustive on purpose: `GpuSpill` would need a mid-forward
                // upload path that does not exist, so it resolves to the CPU
                // lane. Spelling it out means adding a real spill is a compile
                // error here rather than a verdict that silently does nothing.
                match bal.choose(gpu_resident, heat) {
                    crate::lane::Placement::Gpu => true,
                    crate::lane::Placement::Cpu | crate::lane::Placement::GpuSpill => false,
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
            let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
            plans.push(EPlan {
                pos: this_pos,
                expert: e,
                rows,
                rw,
                gate: tplan(st, &p("gate_proj.weight"), mi, hidden)?,
                up: tplan(st, &p("up_proj.weight"), mi, hidden)?,
                down: tplan(st, &p("down_proj.weight"), hidden, mi)?,
            });
        }
    }
    let n = pos;

    // Apply the layout schedule: reorder the streamed `EPlan`s by the schedule's
    // rank of each expert. This changes only the io_uring submit order — the
    // deterministic reduce uses `pos` (batch-union index) as its scatter key, so
    // outputs stay bit-identical. Experts not in the schedule keep their
    // original ordering, appended at the end. No-op when there is no schedule.
    if let Some(sched) = ctx.layout_schedule {
        if let Some(row) = sched.get(layer) {
            let rank: std::collections::HashMap<u32, usize> =
                row.iter().enumerate().map(|(i, &e)| (e, i)).collect();
            plans.sort_by_key(|p| rank.get(&(p.expert as u32)).copied().unwrap_or(usize::MAX));
        }
    }
    // Co-activation affinity: hyperedge grouping + fused-pair adjacency on top
    // of (or instead of) the layout order. Reduce keys on `pos`, so any order
    // here is bit-identical.
    if let Some(aff) = ctx.affinity {
        apply_affinity_order(&mut plans, layer, aff);
    }

    // job: (disk-plan index, streamed gate/up/down bytes) from I/O lane → CPU pool.
    // `Bytes` regions so an O_DIRECT read can hand its aligned DMA buffer over the
    // channel with no copy (== peregrine_io::ExpertSlab).
    type Bytes3 = [(Bytes, Bytes); 3];
    let (job_tx, job_rx) = crossbeam_channel::bounded::<(usize, Bytes3)>(workers.max(1) * 2);
    // result: one computed expert keyed by `pos`, or — under
    // `COLI_CUDA_FUSED_REDUCE` — the GPU lane's single pre-accumulated
    // `[s_n, hidden]` partial standing in for all of its experts at once.
    let (res_tx, res_rx) = crossbeam_channel::bounded::<Result<LaneMsg, Error>>(workers.max(1) * 2);
    // Whether the GPU lane reduces on the device. Resolved once here rather than
    // per lane so the collector's expected message count and the lane's
    // behaviour cannot disagree.
    let fused_reduce = gpu.is_some() && !gplans.is_empty() && fused_reduce_enabled();

    let completed = AtomicUsize::new(0);
    // Shared cursor the I/O rings atomically claim expert-batches from (lock-free
    // work-stealing): each ring `fetch_add`s a batch, so no expert is read twice and
    // the rings never idle while work remains.
    let io_work = AtomicUsize::new(0);
    let plans_ref = &plans;
    let gplans_ref = &gplans;
    let x_ref = x;
    let completed_ref = &completed;
    let io_work_ref = &io_work;
    // Per-lane wall-time accumulator (or `None` for the no-tracking path). Copied
    // into each scoped thread so the atomic bumps in the accumulator's four counters
    // are the only synchronization the timing incurs.
    let timings_ref = ctx.timings;
    // Cache-admission gate: shared HeatTable ref + threshold, consulted per
    // streamed expert before inserting into the warm cache. Threshold 0 (the
    // default) admits everything.
    let heat_ref = ctx.heat;
    let admit_min_heat = cache_admit_min_heat();

    // `(per-expert slots, the GPU lane's device-reduced partial)`. The partial is
    // `None` unless `COLI_CUDA_FUSED_REDUCE` is on and the layer had GPU experts.
    type LaneResults = (Vec<Option<EOut>>, Option<Vec<f32>>);
    let results: Result<LaneResults, Error> = std::thread::scope(|scope| {
        // ---- I/O lanes: N io_uring rings in PARALLEL, lock-free (atomic) work-stealing ----
        // One thread per ring. Each atomically claims a batch of experts off `io_work`,
        // serves warm-tier hits immediately, and streams the misses through *its own*
        // ring in one deep submit — so N reads run concurrently (which also parallelizes
        // dm-crypt decryption on encrypted volumes). The `pos`-ordered reduce is
        // order-independent, so which ring reads which expert never changes the output.
        let n_plans = plans_ref.len();
        for ring in reactors.iter() {
            let job_tx = job_tx.clone();
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                loop {
                    let batch = experts_per_batch();
                    let start = io_work_ref.fetch_add(batch, Ordering::Relaxed);
                    if start >= n_plans {
                        break; // no work left for this ring
                    }
                    let end = (start + batch).min(n_plans);
                    // split the claimed range into warm-tier hits (dispatch now) and
                    // misses (one deep async submit on this ring)
                    let mut miss: Vec<usize> = Vec::new();
                    for (idx, plan) in plans_ref.iter().enumerate().take(end).skip(start) {
                        let key = (layer as u32, plan.expert as u32);
                        let hit = ecache.and_then(|c| c.lock().get(key));
                        match hit {
                            Some(bytes) => {
                                if job_tx.send((idx, bytes)).is_err() {
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
                    let slabs = {
                        let mut r = ring.lock(); // this ring, uncontended (owned by this thread)
                        read_experts_batched(&mut r, &chunk_plans, use_direct)
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
                        if let Some(c) = ecache {
                            let expert = plans_ref[idx].expert;
                            // Admission gate: only cache experts with demonstrated
                            // reuse (routing heat ≥ threshold). Heat is bumped after
                            // the reduce, so a first-ever routing reads 0 here —
                            // threshold 1 = "cache from the second routing on".
                            let admit = admit_min_heat == 0
                                || heat_ref.is_some_and(|h| h.get(layer, expert) >= admit_min_heat);
                            let mut c = c.lock();
                            c.note_disk_read(layer as u32);
                            if admit {
                                c.insert((layer as u32, expert as u32), bytes.clone());
                            }
                        }
                        if job_tx.send((idx, bytes)).is_err() {
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
                    let (idx, bytes) = match job_rx.recv() {
                        Ok(job) => job,
                        Err(crossbeam_channel::RecvError) => break,
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
                        gate: rebuild(&plan.gate, gw, gs),
                        up: rebuild(&plan.up, uw, us),
                        down: rebuild(&plan.down, dw, ds),
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
    Ok(out)
}

/// One expert queued for speculative prefetch: its `(layer, expert)` cache key and
/// the three tensor plans to stream. Built by [`prefetch_item`], consumed by
/// [`prefetch_read`] on the prefetch lane's own ring.
pub struct PrefetchItem {
    key: (u32, u32),
    plans: [TPlan; 3],
}

impl PrefetchItem {
    /// The `(layer, expert)` warm-cache key this item populates.
    pub fn key(&self) -> (u32, u32) {
        self.key
    }
}

/// Build the streaming plan for one routed expert (gate/up/down), for the prefetch
/// lane. Mirrors the disk-plan construction in [`moe_forward_concurrent`].
pub fn prefetch_item(st: &SafeTensors, cfg: &Cfg, layer: usize, expert: usize) -> Result<PrefetchItem, Error> {
    let hidden = cfg.hidden as usize;
    let mi = cfg.moe_inter as usize;
    let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{expert}.{t}");
    let gate = tplan(st, &p("gate_proj.weight"), mi, hidden)?;
    let up = tplan(st, &p("up_proj.weight"), mi, hidden)?;
    let down = tplan(st, &p("down_proj.weight"), hidden, mi)?;
    Ok(PrefetchItem { key: (layer as u32, expert as u32), plans: [gate, up, down] })
}

/// Stream one prefetch item's six regions through `reactor` into an owned slab
/// (one batched submit) — the exact bytes the I/O lane would read, so a later hit
/// is bit-identical.
pub fn prefetch_read(reactor: &mut Reactor, item: &PrefetchItem, direct: bool) -> Result<peregrine_io::ExpertSlab, Error> {
    read_expert(reactor, &item.plans[0], &item.plans[1], &item.plans[2], direct)
}

/// One expert queued for a page-cache *hint* (`fadvise(WILLNEED)`): its six
/// `(fd, offset, len)` regions to warm without streaming into the cache. Used for
/// low-confidence multi-path predictions, where a full prefetch isn't worth the
/// bandwidth but a cheap page-cache hint may still help a later miss.
pub struct HintItem {
    regions: [(RawFd, u64, usize); 6],
}

impl HintItem {
    /// The six `(fd, offset, len)` regions (gate/up/down × weight+scale) to hint.
    pub fn regions(&self) -> &[(RawFd, u64, usize); 6] {
        &self.regions
    }
}

/// Build a [`HintItem`] for one expert's six regions. Uses the **buffered** fds:
/// `fadvise` only populates the page cache, so it's a no-op for O_DIRECT reads and
/// the caller gates hints off under direct I/O.
pub fn prefetch_hint_item(st: &SafeTensors, cfg: &Cfg, layer: usize, expert: usize) -> Result<HintItem, Error> {
    let hidden = cfg.hidden as usize;
    let mi = cfg.moe_inter as usize;
    let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{expert}.{t}");
    let gate = tplan(st, &p("gate_proj.weight"), mi, hidden)?;
    let up = tplan(st, &p("up_proj.weight"), mi, hidden)?;
    let down = tplan(st, &p("down_proj.weight"), hidden, mi)?;
    let regions = [
        (gate.w_fd, gate.w_off, gate.w_len),
        (gate.s_fd, gate.s_off, gate.s_len),
        (up.w_fd, up.w_off, up.w_len),
        (up.s_fd, up.s_off, up.s_len),
        (down.w_fd, down.w_off, down.w_len),
        (down.s_fd, down.s_off, down.s_len),
    ];
    Ok(HintItem { regions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::QuantFmt;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

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
        let (gate, up, down) = (tp(0, 1), tp(2, 3), tp(4, 5));

        let mut reactor = match Reactor::new(16) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping: io_uring unavailable: {e}");
                std::fs::remove_file(&path)?;
                return Ok(());
            }
        };
        let slab = read_expert(&mut reactor, &gate, &up, &down, false)?;
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
}

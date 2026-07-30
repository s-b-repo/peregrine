//! peregrine-par — a tiny **persistent** scoped worker pool for data-parallel,
//! **bit-identical** compute.
//!
//! `std::thread::scope` spawns fresh threads per call; a batched forward runs
//! ~200+ small parallel ops (2 norms + 1 attention per layer × ~78 layers), so
//! per-op spawn overhead would dominate the small batches decode uses. This pool
//! parks a fixed set of threads once and dispatches disjoint index sub-ranges to
//! them with a **fork-join barrier**.
//!
//! Soundness of the one `unsafe` (handing workers a pointer to a borrowed
//! closure): the dispatcher does not return until it has received one done-signal
//! per dispatched chunk, so every worker's deref of the closure *happens-before*
//! the dispatcher's stack frame — which owns the closure — can unwind. The closure
//! is `Sync`, bounding the concurrent shared reads; workers only read through it,
//! and all writes go to **disjoint** output rows/slots, so there is no data race.
//!
//! Determinism: each `par_*` runs the caller's `Fn` over disjoint indices with no
//! cross-index reduction, so the result is **bit-identical to a serial loop**
//! regardless of how work is scheduled — the property the model crate's
//! `parallel_matches_serial` tests assert exactly (`f32::to_bits`).
//!
//! `std`-only: per-worker `mpsc` job channels + one shared `mpsc` done-channel
//! (no `crossbeam`/`rayon` dependency). In release (`panic = "abort"`) a panicking
//! task aborts loudly; in debug the `catch_unwind` net turns it into a re-raised
//! panic on the dispatcher, never a deadlock.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

/// Default parallel-eligibility gates (below these `n`, run serial — the barrier
/// isn't worth it). Centralized so the whole engine tunes in one place.
pub const PAR_ROWS_MIN: usize = 4;
pub const PAR_ATTN_MIN: usize = 2;
pub const PAR_MOE_MIN: usize = 2;
pub const PAR_MATMUL_MIN: usize = 8;

thread_local! {
    /// True on the pool's own worker threads, so a *nested* `par_*` (e.g. an
    /// expert's matmul inside a parallel MoE) runs serial instead of dispatching —
    /// which would deadlock a fixed pool (all workers blocked waiting on each other).
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A `Send`, lifetime-erased raw pointer to a borrowed `dyn Fn(usize) + Sync`.
/// Soundness discharged by the barrier in [`Pool::run_ranges`] (every deref
/// happens-before the pointee's frame unwinds) and the `Sync` bound on the pointee.
#[derive(Clone, Copy)]
struct TaskPtr(*const (dyn Fn(usize) + Sync + 'static));
// SAFETY: see the type doc — barrier + `Sync` pointee make cross-thread `&`-access sound.
unsafe impl Send for TaskPtr {}

struct Job {
    task: TaskPtr,
    start: usize,
    end: usize,
    done: Sender<()>,
    panicked: Arc<AtomicBool>,
}

/// A `Sync` raw base pointer used only to form **disjoint** row sub-slices; the
/// dispatch guarantees each row index runs on exactly one worker, so no two
/// threads ever form overlapping `&mut`.
struct SendPtr<T>(*mut T);
// SAFETY: only disjoint, non-overlapping sub-slices/slots are formed from this
// base (each index dispatched once), so shared `&`-access to the base is sound.
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// The base pointer, read via `&self` so a closure captures the whole `SendPtr`
    /// (a `Sync` wrapper) rather than the bare `*mut T` field — Rust 2021 disjoint
    /// closure capture would otherwise grab the field and lose the `Sync` impl.
    #[inline]
    fn get(&self) -> *mut T {
        self.0
    }
}

/// A persistent pool of parked worker threads reused across `par_*` calls.
pub struct Pool {
    job_txs: Vec<Sender<Job>>,
    handles: Vec<JoinHandle<()>>,
}

/// Optional worker-startup hook, run once by each worker thread with its index
/// before it starts serving jobs. Set it with [`set_worker_start_hook`] *before*
/// the first pool is built (the global pool is constructed lazily on first
/// `par_*` call). The intended consumer is NUMA thread pinning — this crate
/// stays std-only, so the affinity syscall lives in the caller's hook.
static WORKER_START_HOOK: std::sync::OnceLock<fn(usize)> = std::sync::OnceLock::new();

/// Install a hook run by every pool worker at startup with its worker index.
/// First caller wins; later calls are ignored (returns whether this call set it).
/// A hook installed after a pool has spawned only affects later-built pools.
pub fn set_worker_start_hook(f: fn(usize)) -> bool {
    WORKER_START_HOOK.set(f).is_ok()
}

/// Optional worker→group map for **hierarchical (two-level) dispatch**: group =
/// NUMA node (or socket). When set — `groups[i]` is worker `i`'s group — the
/// dispatcher splits the index range into contiguous per-group blocks sized
/// proportionally to each group's worker count, then per-worker chunks within a
/// block. Combined with node-grouped pinning, a group's block stays node-local.
/// Bit-identical to flat dispatch (each index still runs exactly once; writes
/// stay disjoint) — only which worker computes which range changes.
static WORKER_GROUPS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();

/// Install the worker→group map (first caller wins; returns whether set).
/// Install *before* the first `par_*` call for the map to shape dispatch from
/// the start (it is consulted per call, so late installation also works).
pub fn set_worker_groups(groups: Vec<usize>) -> bool {
    WORKER_GROUPS.set(groups).is_ok()
}

/// Plan `(worker, start, end)` range assignments over `0..n` for `w` workers.
///
/// Flat mode (`groups` absent or mismatched): the historical ceil-chunk split —
/// worker `i` gets `[i*chunk, (i+1)*chunk)`.
///
/// Hierarchical mode (`groups[i]` = worker `i`'s group): the range splits into
/// one contiguous block per group, sized proportionally to the group's worker
/// count (remainders spread over the leading groups), then ceil-chunked across
/// that group's workers. Every index appears in exactly one range either way —
/// the property the bit-identity of `par_*` rests on.
fn plan_assignments(n: usize, w: usize, groups: Option<&[usize]>) -> Vec<(usize, usize, usize)> {
    let mut out: Vec<(usize, usize, usize)> = Vec::with_capacity(w);
    let flat = |out: &mut Vec<(usize, usize, usize)>| {
        let chunk = n.div_ceil(w);
        let mut start = 0usize;
        let mut worker = 0usize;
        while start < n && worker < w {
            let end = (start + chunk).min(n);
            out.push((worker, start, end));
            worker += 1;
            start = end;
        }
    };
    match groups {
        Some(g) if g.len() == w && w > 0 => {
            // group id → its workers, in first-appearance order (deterministic).
            let mut order: Vec<usize> = Vec::new();
            let mut members: Vec<Vec<usize>> = Vec::new();
            for (i, &gid) in g.iter().enumerate() {
                match order.iter().position(|&x| x == gid) {
                    Some(k) => members[k].push(i),
                    None => {
                        order.push(gid);
                        members.push(vec![i]);
                    }
                }
            }
            // proportional block sizes; leading groups absorb the remainder.
            let n_groups = members.len();
            let mut sizes: Vec<usize> = members.iter().map(|m| n * m.len() / w).collect();
            let assigned: usize = sizes.iter().sum();
            for (k, s) in sizes.iter_mut().enumerate() {
                if k < n - assigned {
                    *s += 1;
                }
            }
            let _ = n_groups;
            let mut start = 0usize;
            for (m, &size) in members.iter().zip(&sizes) {
                if size == 0 {
                    continue;
                }
                let block_end = start + size;
                let chunk = size.div_ceil(m.len());
                let mut s = start;
                for &worker in m {
                    if s >= block_end {
                        break;
                    }
                    let e = (s + chunk).min(block_end);
                    out.push((worker, s, e));
                    s = e;
                }
                start = block_end;
            }
        }
        _ => flat(&mut out),
    }
    out
}

fn worker_loop(index: usize, rx: Receiver<Job>) {
    if let Some(hook) = WORKER_START_HOOK.get() {
        hook(index);
    }
    IN_POOL.with(|c| c.set(true)); // mark this thread so nested par_* run serial
    while let Ok(job) = rx.recv() {
        // SAFETY: `job.task.0` points to a `dyn Fn + Sync` owned by the dispatcher's
        // stack frame, which blocks on `job.done` (and its siblings) before that
        // frame unwinds — so this deref cannot dangle; `Sync` makes the read defined.
        let f: &(dyn Fn(usize) + Sync) = unsafe { &*job.task.0 };
        let r = catch_unwind(AssertUnwindSafe(|| {
            for i in job.start..job.end {
                f(i);
            }
        }));
        if r.is_err() {
            job.panicked.store(true, Ordering::Relaxed);
        }
        // Always signal, even on a caught panic, so the barrier can't deadlock.
        let _ = job.done.send(());
    }
}

impl Pool {
    /// A pool with `workers` threads (`>= 1`). `workers == 1` parks no threads and
    /// runs every `par_*` inline on the caller — the serial oracle for tests.
    pub fn new(workers: usize) -> Pool {
        let workers = workers.max(1);
        if workers == 1 {
            return Pool { job_txs: Vec::new(), handles: Vec::new() };
        }
        let mut job_txs = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let (tx, rx) = channel::<Job>();
            job_txs.push(tx);
            handles.push(std::thread::spawn(move || worker_loop(i, rx)));
        }
        Pool { job_txs, handles }
    }

    /// Run `f(i)` for every `i in 0..n` across the pool, blocking until all done.
    /// Serial when the pool has one worker or `n < min_parallel`. The load-bearing
    /// invariant: this fn does not return until it has received one done-signal per
    /// dispatched chunk, so the borrowed `f` outlives every worker deref.
    fn run_ranges(&self, n: usize, min_parallel: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        let w = self.job_txs.len();
        if w < 2 || n < min_parallel.max(1) || IN_POOL.with(|c| c.get()) {
            // serial: one worker, below the gate, or already inside a pool task
            // (nested dispatch would deadlock).
            for i in 0..n {
                f(i);
            }
            return;
        }
        let panicked = Arc::new(AtomicBool::new(false));
        // SAFETY: erase `f`'s borrow lifetime for the channel hop to the persistent
        // workers. The barrier below blocks until every worker has finished
        // dereferencing this pointer, and `f` is owned by this stack frame for that
        // whole span, so no worker can observe a dangling reference; `f: Sync` makes
        // the concurrent shared reads defined.
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), _>(f) };
        let task = TaskPtr(erased);
        let (done_tx, done_rx) = channel::<()>();
        // (worker, start, end) assignments: hierarchical two-level split when a
        // worker→group map is installed (contiguous per-group blocks proportional
        // to group size, then per-worker chunks within the block — node-local
        // ranges under node-grouped pinning), else the flat ceil-chunk split.
        let assignments = plan_assignments(n, w, WORKER_GROUPS.get().map(|g| g.as_slice()));
        let mut sent = 0usize;
        let mut abort_from: Option<usize> = None;
        for &(worker, start, end) in &assignments {
            let job = Job { task, start, end, done: done_tx.clone(), panicked: Arc::clone(&panicked) };
            if self.job_txs[worker].send(job).is_err() {
                // a worker is gone (shutdown race): finish this range inline and
                // remember to run every not-yet-sent range inline too.
                for i in start..end {
                    f(i);
                }
                abort_from = Some(sent + 1);
                break;
            }
            sent += 1;
        }
        if let Some(from) = abort_from {
            for &(_, start, end) in assignments.iter().skip(from) {
                for i in start..end {
                    f(i);
                }
            }
        }
        // Barrier — the soundness invariant above.
        for _ in 0..sent {
            let _ = done_rx.recv();
        }
        if panicked.load(Ordering::Relaxed) {
            // debug/unwind builds only (release aborts on panic); never a deadlock.
            resume_unwind(Box::new("peregrine-par: a parallel task panicked"));
        }
    }

    /// Apply `f(row_index, &mut row)` to each of `n` disjoint `stride`-element rows
    /// of `out`, in parallel (serial below `min_parallel`). Bit-identical to the
    /// serial loop: rows are disjoint, no cross-row reduction.
    pub fn par_rows_mut(&self, out: &mut [f32], stride: usize, n: usize, min_parallel: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
        let base = SendPtr(out.as_mut_ptr());
        let g = move |i: usize| {
            // SAFETY: rows are `stride` apart and each `i` is dispatched exactly
            // once, so this `&mut` never overlaps another worker's.
            let row = unsafe { std::slice::from_raw_parts_mut(base.get().add(i * stride), stride) };
            f(i, row);
        };
        self.run_ranges(n, min_parallel, &g);
    }

    /// Like [`Self::par_rows_mut`] but hands each row three disjoint strided slices
    /// (buffers `a`/`b`/`c` with `strides` `[sa,sb,sc]`) — for callers writing
    /// several row-strided outputs together. Buffers/strides are grouped to keep
    /// the argument list small.
    pub fn par_rows_mut3(
        &self,
        bufs: (&mut [f32], &mut [f32], &mut [f32]),
        strides: [usize; 3],
        n: usize,
        min_parallel: usize,
        f: impl Fn(usize, &mut [f32], &mut [f32], &mut [f32]) + Sync,
    ) {
        let (a, b, c) = bufs;
        let [sa, sb, sc] = strides;
        let (pa, pb, pc) = (SendPtr(a.as_mut_ptr()), SendPtr(b.as_mut_ptr()), SendPtr(c.as_mut_ptr()));
        let g = move |i: usize| {
            // SAFETY: three disjoint row-strided buffers; each `i` dispatched once.
            let (ra, rb, rc) = unsafe {
                (
                    std::slice::from_raw_parts_mut(pa.get().add(i * sa), sa),
                    std::slice::from_raw_parts_mut(pb.get().add(i * sb), sb),
                    std::slice::from_raw_parts_mut(pc.get().add(i * sc), sc),
                )
            };
            f(i, ra, rb, rc);
        };
        self.run_ranges(n, min_parallel, &g);
    }

    /// `f(i)` for `i in 0..n`, collected into `Vec<R>` in **index order** (regardless
    /// of completion order). Parallel above `min_parallel`. No shared output during
    /// compute — each task writes only its own slot — so it is deterministic.
    pub fn par_map<R: Send>(&self, n: usize, min_parallel: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
        let mut out: Vec<R> = Vec::with_capacity(n);
        let base: SendPtr<R> = SendPtr(out.as_mut_ptr());
        let g = move |i: usize| {
            let val = f(i);
            // SAFETY: slot `i` is written exactly once (disjoint dispatch) into
            // uninitialized capacity, so a raw write (no drop of a prior value) is
            // correct. `run_ranges` writes all `n` slots before it returns.
            unsafe { base.get().add(i).write(val) };
        };
        self.run_ranges(n, min_parallel, &g);
        // SAFETY: the barrier in `run_ranges` initialized all `n` slots (or unwound
        // before reaching here on a task panic, so this line never runs partial).
        unsafe { out.set_len(n) };
        out
    }

    /// Split `out` into contiguous row-chunks and apply `f(start, end, chunk)` to
    /// each disjoint chunk in parallel; serial (one whole chunk `f(0, n, out)`) below
    /// `min_parallel`, on a one-worker pool, or when nested. Unlike
    /// [`Self::par_rows_mut`] the callee gets a *range* so it can run one batched
    /// operation per chunk (e.g. a batched matmul with per-chunk scratch). Whatever
    /// `f` does per row must be independent → bit-identical to processing all rows
    /// in one serial call.
    pub fn par_chunks_mut(&self, out: &mut [f32], stride: usize, n: usize, min_parallel: usize, f: impl Fn(usize, usize, &mut [f32]) + Sync) {
        if n == 0 {
            return;
        }
        let w = self.job_txs.len();
        let serial = w < 2 || n < min_parallel.max(1) || IN_POOL.with(|c| c.get());
        let target = if serial { 1 } else { w.min(n) };
        let chunk = n.div_ceil(target);
        let nchunks = n.div_ceil(chunk);
        let base = SendPtr(out.as_mut_ptr());
        let g = move |ci: usize| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            // SAFETY: chunks are disjoint contiguous row ranges; each `ci` runs once.
            let seg = unsafe { std::slice::from_raw_parts_mut(base.get().add(start * stride), (end - start) * stride) };
            f(start, end, seg);
        };
        // dispatch over chunk indices; run_ranges is serial when nchunks == 1.
        self.run_ranges(nchunks, 2, &g);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.job_txs.clear(); // drop senders → workers' recv() errors → they exit
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// The process-wide pool, sized to `min(cores, 16)` — or `COLI_PAR_THREADS` when
/// set (`1` = fully serial, for A/B benchmarking or single-core deployments).
/// Built on first use; lives for the process.
pub fn global() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let w = std::env::var("COLI_PAR_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get().min(16)).unwrap_or(4));
        Pool::new(w)
    })
}

/// [`Pool::par_rows_mut`] on the global pool.
pub fn par_rows_mut(out: &mut [f32], stride: usize, n: usize, min_parallel: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    global().par_rows_mut(out, stride, n, min_parallel, f);
}

/// [`Pool::par_rows_mut3`] on the global pool.
pub fn par_rows_mut3(
    bufs: (&mut [f32], &mut [f32], &mut [f32]),
    strides: [usize; 3],
    n: usize,
    min_parallel: usize,
    f: impl Fn(usize, &mut [f32], &mut [f32], &mut [f32]) + Sync,
) {
    global().par_rows_mut3(bufs, strides, n, min_parallel, f);
}

/// [`Pool::par_map`] on the global pool.
pub fn par_map<R: Send>(n: usize, min_parallel: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    global().par_map(n, min_parallel, f)
}

/// [`Pool::par_chunks_mut`] on the global pool.
pub fn par_chunks_mut(out: &mut [f32], stride: usize, n: usize, min_parallel: usize, f: impl Fn(usize, usize, &mut [f32]) + Sync) {
    global().par_chunks_mut(out, stride, n, min_parallel, f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_assignments_covers_every_index_exactly_once() {
        // Flat and hierarchical plans must both partition 0..n: no index missed,
        // none duplicated — the invariant bit-identity rests on.
        for (n, w, groups) in [
            (1000usize, 4usize, None),
            (1000, 4, Some(vec![0usize, 0, 1, 1])),
            (7, 4, Some(vec![0, 0, 1, 1])),
            (3, 6, Some(vec![0, 0, 0, 1, 1, 1])),
            (1000, 6, Some(vec![0, 0, 0, 0, 1, 1])), // asymmetric groups
        ] {
            let plan = plan_assignments(n, w, groups.as_deref());
            let mut seen = vec![0u8; n];
            for &(worker, s, e) in &plan {
                assert!(worker < w);
                for slot in seen.iter_mut().take(e).skip(s) {
                    *slot += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "n={n} w={w} groups={groups:?}: each index exactly once");
        }
    }

    #[test]
    fn hierarchical_blocks_are_group_contiguous() {
        // Two groups of two workers: group 0's ranges must all precede group 1's
        // (contiguous per-group blocks — the node-locality property).
        let plan = plan_assignments(100, 4, Some(&[0, 0, 1, 1]));
        let g0_max = plan.iter().filter(|&&(w, _, _)| w < 2).map(|&(_, _, e)| e).max().unwrap_or(0);
        let g1_min = plan.iter().filter(|&&(w, _, _)| w >= 2).map(|&(_, s, _)| s).min().unwrap_or(usize::MAX);
        assert!(g0_max <= g1_min, "group 0's block ends before group 1's begins: {plan:?}");
        // Proportional: equal worker counts → ~equal halves.
        assert_eq!(g0_max, 50);
    }

    #[test]
    fn par_map_matches_serial_and_order() {
        // parallel result must equal the serial map, in index order, for n far
        // larger than the worker count (multiple chunks per worker).
        let pool = Pool::new(4);
        let n = 1000usize;
        let par = pool.par_map(n, 2, |i| (i as f32 * 1.5 - 3.0).sin());
        let serial: Vec<f32> = (0..n).map(|i| (i as f32 * 1.5 - 3.0).sin()).collect();
        assert_eq!(par.len(), n);
        assert!(par.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()), "par_map must be bit-identical + ordered");
    }

    #[test]
    fn par_rows_mut_matches_serial() {
        // fill via par_rows_mut vs a plain serial loop with an f64-accumulated
        // reduction (like rmsnorm) — assert exact bit equality.
        let (n, stride) = (257usize, 40usize);
        let src: Vec<f32> = (0..n * stride).map(|k| (k as f32).cos()).collect();
        let write = |i: usize, row: &mut [f32]| {
            let s: f64 = src[i * stride..i * stride + stride].iter().map(|&v| v as f64 * v as f64).sum();
            let inv = 1.0 / (s / stride as f64 + 1e-6).sqrt();
            for (d, out) in row.iter_mut().enumerate() {
                *out = (src[i * stride + d] as f64 * inv) as f32;
            }
        };
        let pool = Pool::new(4);
        let mut par = vec![0f32; n * stride];
        pool.par_rows_mut(&mut par, stride, n, 2, write);
        let mut serial = vec![0f32; n * stride];
        for i in 0..n {
            write(i, &mut serial[i * stride..i * stride + stride]);
        }
        assert!(par.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn min_parallel_and_pool_of_one_are_serial() {
        // both the gate and a 1-thread pool must run inline and match.
        let one = Pool::new(1);
        let many = Pool::new(8);
        let a = one.par_map(50, 2, |i| i as f32);
        let b = many.par_map(50, usize::MAX, |i| i as f32); // gate forces serial
        let c = many.par_map(50, 2, |i| i as f32); // parallel
        let ref_: Vec<f32> = (0..50).map(|i| i as f32).collect();
        assert!(a.iter().zip(&ref_).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(b.iter().zip(&ref_).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(c.iter().zip(&ref_).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn par_rows_mut3_matches_serial() {
        let n = 33usize;
        let (sa, sb, sc) = (4usize, 2usize, 3usize);
        let write = |i: usize, a: &mut [f32], b: &mut [f32], c: &mut [f32]| {
            for (d, x) in a.iter_mut().enumerate() {
                *x = (i * 10 + d) as f32;
            }
            for (d, x) in b.iter_mut().enumerate() {
                *x = (i * 100 + d) as f32;
            }
            for (d, x) in c.iter_mut().enumerate() {
                *x = -((i + d) as f32);
            }
        };
        let pool = Pool::new(4);
        let (mut pa, mut pb, mut pc) = (vec![0f32; n * sa], vec![0f32; n * sb], vec![0f32; n * sc]);
        pool.par_rows_mut3((&mut pa, &mut pb, &mut pc), [sa, sb, sc], n, 2, write);
        let (mut ra, mut rb, mut rc) = (vec![0f32; n * sa], vec![0f32; n * sb], vec![0f32; n * sc]);
        for i in 0..n {
            write(i, &mut ra[i * sa..i * sa + sa], &mut rb[i * sb..i * sb + sb], &mut rc[i * sc..i * sc + sc]);
        }
        assert_eq!((pa, pb, pc), (ra, rb, rc));
    }

    #[test]
    fn par_chunks_mut_matches_serial() {
        // parallel-chunked fill must equal one serial whole-buffer call; each row's
        // value derives from its global index, so chunk boundaries don't matter.
        let (n, stride) = (100usize, 7usize);
        let fill = |start: usize, end: usize, seg: &mut [f32]| {
            for (li, s) in (start..end).enumerate() {
                for d in 0..stride {
                    seg[li * stride + d] = (s * 31 + d) as f32;
                }
            }
        };
        let pool = Pool::new(4);
        let mut par = vec![0f32; n * stride];
        pool.par_chunks_mut(&mut par, stride, n, 8, fill);
        let mut ser = vec![0f32; n * stride];
        fill(0, n, &mut ser); // one whole call = the serial oracle
        assert!(par.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn nested_par_runs_serial_not_deadlock() {
        // a par_* called from inside another par_* task must run serial (the guard),
        // never deadlock the fixed pool. Result must still be correct.
        let pool = Pool::new(4);
        let outer = pool.par_map(16, 2, |i| {
            // nested par_map on the GLOBAL pool from a worker → must serialize
            let inner = par_map(8, 2, |j| i * 8 + j);
            inner.iter().sum::<usize>()
        });
        let expect: Vec<usize> = (0..16).map(|i| (0..8).map(|j| i * 8 + j).sum()).collect();
        assert_eq!(outer, expect);
    }

    #[test]
    fn panicking_task_surfaces_not_deadlocks() {
        // a task that panics must re-surface on the dispatcher (caught here), never
        // hang. `assert_eq!` triggers the panic without a P-audit `panic!(`/`unwrap`.
        let pool = Pool::new(4);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.par_map(64, 2, |i| {
                if i == 40 {
                    assert_eq!(1, 2, "intentional test panic");
                }
                i
            })
        }));
        assert!(r.is_err(), "a panicking parallel task must surface, not deadlock");
    }
}

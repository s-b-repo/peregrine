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
use peregrine_io::{Reactor, ReadReq, WarmCache};

use crate::gpu::{GpuTier, HeatTable};
use crate::mlp::Mlp;
use crate::router::{batch_union, route};
use crate::weight::{QtWeight, QuantFmt};

/// Shared per-forward state threaded through the layer/MoE compute: the
/// safetensors index, the streaming io_uring ring, the GPU tier, the CPU-lane
/// width, the config, and whether experts stream from disk. Passed by reference
/// so the layer/MoE entry points stay small (no long argument lists).
pub struct ForwardCtx<'a> {
    pub st: &'a SafeTensors,
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
    /// Per-layer routing history written after each layer's reduce: `route_log[layer]`
    /// = this forward's batch-union of routed experts. The prefetch lane reads it
    /// to predict the next token's experts. `None` on speculative-draft forwards
    /// (so drafts don't pollute the main-stream prediction) and when prefetch is off.
    pub route_log: Option<&'a Mutex<Vec<Vec<i32>>>>,
    /// Stream expert reads via O_DIRECT (bypass the page cache) when the shards
    /// opened O_DIRECT fds. Bytes are identical to the buffered path; only the
    /// cache behavior differs. `false` disables (buffered reads).
    pub direct: bool,
    /// Routing-frequency accumulator for heat-ranked VRAM residency: bumped once
    /// per routed expert per layer so [`crate::gpu::GpuTier::reheat`] can migrate
    /// hot experts into VRAM. `None` disables accumulation (no GPU tier / drafts).
    pub heat: Option<&'a HeatTable>,
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

/// Stream one expert's gate/up/down (six weight+scale regions) through the ring
/// in a **single batched submit** — one `submit_and_wait` for all six instead of
/// six sequential `read_exact`s (each its own enter syscall). Any short read (a
/// positioned read may legally return fewer bytes) is completed individually, so
/// the returned bytes are identical to six `read_exact`s — the streamed output
/// stays bit-identical to the resident path.
fn read_expert(r: &mut Reactor, gate: &TPlan, up: &TPlan, down: &TPlan, direct: bool) -> Result<peregrine_io::ExpertSlab, Error> {
    let mut gw = vec![0u8; gate.w_len];
    let mut gs = vec![0u8; gate.s_len];
    let mut uw = vec![0u8; up.w_len];
    let mut us = vec![0u8; up.s_len];
    let mut dw = vec![0u8; down.w_len];
    let mut ds = vec![0u8; down.s_len];

    // pick the weight/scale fd for a tensor: O_DIRECT twin when `direct`, else buffered.
    let wfd = |t: &TPlan| if direct { t.w_fd_direct.unwrap_or(t.w_fd) } else { t.w_fd };
    let sfd = |t: &TPlan| if direct { t.s_fd_direct.unwrap_or(t.s_fd) } else { t.s_fd };

    let mut done = [0usize; 6];
    {
        let mut reqs = [
            ReadReq { fd: wfd(gate), offset: gate.w_off, buf: &mut gw, tag: 0 },
            ReadReq { fd: sfd(gate), offset: gate.s_off, buf: &mut gs, tag: 1 },
            ReadReq { fd: wfd(up), offset: up.w_off, buf: &mut uw, tag: 2 },
            ReadReq { fd: sfd(up), offset: up.s_off, buf: &mut us, tag: 3 },
            ReadReq { fd: wfd(down), offset: down.w_off, buf: &mut dw, tag: 4 },
            ReadReq { fd: sfd(down), offset: down.s_off, buf: &mut ds, tag: 5 },
        ];
        let res = if direct {
            r.read_direct_many(&mut reqs).ctx(|| "io_uring O_DIRECT expert read".to_string())?
        } else {
            r.read_many(&mut reqs).ctx(|| "io_uring batched expert read".to_string())?
        };
        for (i, &n) in res.iter().enumerate() {
            if n < 0 {
                return Err(Error::Io(std::io::Error::from_raw_os_error((-n) as i32)));
            }
            done[i] = n as usize;
        }
    }
    // Complete any short reads (buffered path only; the direct reader returns full
    // length, so `done == len` and this is a no-op) — byte-identical to read_exact.
    for (i, (fd, off, buf)) in [
        (wfd(gate), gate.w_off, &mut gw),
        (sfd(gate), gate.s_off, &mut gs),
        (wfd(up), up.w_off, &mut uw),
        (sfd(up), up.s_off, &mut us),
        (wfd(down), down.w_off, &mut dw),
        (sfd(down), down.s_off, &mut ds),
    ]
    .into_iter()
    .enumerate()
    {
        if done[i] < buf.len() {
            r.read_exact(fd, off + done[i] as u64, &mut buf[done[i]..])
                .ctx(|| "io_uring short-read completion".to_string())?;
        }
    }
    Ok([(gw, gs), (uw, us), (dw, ds)])
}

/// How many experts' reads to submit to the ring at once. 6 regions/expert, so
/// `16 × 6 = 96` in-flight reads keep the io_uring queue deep (vs. 6 when reading
/// one expert at a time — the colibrì deep-queue model) while bounding the transient
/// landing-buffer memory to ~`16 × 18.9 MB ≈ 300 MB` (this box is RAM-contended, so
/// a bounded batch matters; a reusable slab arena would remove the ceiling entirely).
pub const EXPERTS_PER_BATCH: usize = 16;

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
    // one (fd, offset) + landing buffer per region, in gate/up/down × (weight,scale)
    // order. In direct mode use the O_DIRECT twin fd (falling back per-region if a
    // twin is somehow missing); the reader applies the block alignment.
    let mut regions: Vec<(RawFd, u64)> = Vec::with_capacity(6 * n);
    let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(6 * n);
    for p in plans {
        for t in [&p.gate, &p.up, &p.down] {
            let (wfd, sfd) = if direct {
                (t.w_fd_direct.unwrap_or(t.w_fd), t.s_fd_direct.unwrap_or(t.s_fd))
            } else {
                (t.w_fd, t.s_fd)
            };
            regions.push((wfd, t.w_off));
            bufs.push(vec![0u8; t.w_len]);
            regions.push((sfd, t.s_off));
            bufs.push(vec![0u8; t.s_len]);
        }
    }
    // one deep submit for all 6·n regions. Direct: O_DIRECT (page-cache-bypassing)
    // aligned reads that complete each region internally (results = full length).
    let results = {
        let mut reqs: Vec<ReadReq> = bufs
            .iter_mut()
            .zip(&regions)
            .map(|(b, &(fd, off))| ReadReq { fd, offset: off, buf: b.as_mut_slice(), tag: 0 })
            .collect();
        if direct {
            r.read_direct_many(&mut reqs).ctx(|| "io_uring O_DIRECT layer read".to_string())?
        } else {
            r.read_many(&mut reqs).ctx(|| "io_uring batched layer read".to_string())?
        }
    };
    // surface errors + complete any short reads individually (byte-identical result)
    for (i, &(fd, off)) in regions.iter().enumerate() {
        let got = results[i];
        if got < 0 {
            return Err(Error::Io(std::io::Error::from_raw_os_error((-got) as i32)));
        }
        let done = got as usize;
        if done < bufs[i].len() {
            r.read_exact(fd, off + done as u64, &mut bufs[i][done..])
                .ctx(|| "io_uring batched short-read completion".to_string())?;
        }
    }
    // assemble the six buffers of each expert into its slab
    let mut slabs: Vec<peregrine_io::ExpertSlab> = Vec::with_capacity(n);
    for e in 0..n {
        let b = e * 6;
        let gw = std::mem::take(&mut bufs[b]);
        let gs = std::mem::take(&mut bufs[b + 1]);
        let uw = std::mem::take(&mut bufs[b + 2]);
        let us = std::mem::take(&mut bufs[b + 3]);
        let dw = std::mem::take(&mut bufs[b + 4]);
        let ds = std::mem::take(&mut bufs[b + 5]);
        slabs.push([(gw, gs), (uw, us), (dw, ds)]);
    }
    Ok(slabs)
}

fn rebuild(t: &TPlan, wb: Vec<u8>, sb: Vec<u8>) -> QtWeight {
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
    let r = route(x, router_w, router_bias, s_n, hidden, e_n, k, cfg.norm_topk, cfg.routed_scale);

    // Partition the batch-union into GPU-resident (compute on device) and disk
    // (stream + CPU) experts, assigning each a global `pos` in batch-union order
    // so the final reduce stays deterministic regardless of which lane finishes.
    let mut plans: Vec<EPlan> = Vec::new();
    let mut gplans: Vec<GPlan> = Vec::new();
    let mut pos = 0usize;
    let uniq = batch_union(&r, s_n);
    for &e in uniq.iter() {
        let e = e as usize;
        let mut rows: Vec<usize> = Vec::new();
        let mut rw: Vec<f32> = Vec::new();
        for s in 0..s_n {
            for kk in 0..r.keff[s] as usize {
                if r.idx[s * r.k + kk] as usize == e {
                    rows.push(s);
                    rw.push(r.w[s * r.k + kk]);
                    break;
                }
            }
        }
        if rows.is_empty() {
            continue;
        }
        let this_pos = pos;
        pos += 1;
        if gpu.is_some_and(|g| g.has(layer, e)) {
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

    // job: (disk-plan index, streamed gate/up/down bytes) from I/O lane → CPU pool
    type Bytes3 = [(Vec<u8>, Vec<u8>); 3];
    let (job_tx, job_rx) = crossbeam_channel::bounded::<(usize, Bytes3)>(workers.max(1) * 2);
    // result: (pos, computed expert) from any lane → main reducer
    let (res_tx, res_rx) = crossbeam_channel::bounded::<Result<(usize, EOut), Error>>(workers.max(1) * 2);

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

    let results: Result<Vec<Option<EOut>>, Error> = std::thread::scope(|scope| {
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
                    let start = io_work_ref.fetch_add(EXPERTS_PER_BATCH, Ordering::Relaxed);
                    if start >= n_plans {
                        break; // no work left for this ring
                    }
                    let end = (start + EXPERTS_PER_BATCH).min(n_plans);
                    // split the claimed range into warm-tier hits (dispatch now) and
                    // misses (one deep async submit on this ring)
                    let mut miss: Vec<usize> = Vec::new();
                    for idx in start..end {
                        let key = (layer as u32, plans_ref[idx].expert as u32);
                        let hit = ecache.and_then(|c| c.lock().get(key).cloned());
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
                    let slabs = {
                        let mut r = ring.lock(); // this ring, uncontended (owned by this thread)
                        read_experts_batched(&mut r, &chunk_plans, use_direct)
                    };
                    let slabs: Vec<Bytes3> = match slabs {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = res_tx.send(Err(e));
                            return;
                        }
                    };
                    for (&idx, bytes) in miss.iter().zip(slabs) {
                        if let Some(c) = ecache {
                            let mut c = c.lock();
                            c.note_disk_read(layer as u32);
                            c.insert((layer as u32, plans_ref[idx].expert as u32), bytes.clone());
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
                    match g.compute(layer, &jobs, hidden) {
                        Ok(hs) => {
                            for (gp, h) in gplans_ref.iter().zip(hs) {
                                completed_ref.fetch_add(1, Ordering::Relaxed);
                                let out = EOut { rows: gp.rows.clone(), rw: gp.rw.clone(), h };
                                if res_tx.send(Ok((gp.pos, out))).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = res_tx.send(Err(e));
                        }
                    }
                });
            }
        }

        // ---- CPU lane: pool of workers computing SwiGLU per disk expert ----
        for _ in 0..workers.max(1) {
            let job_rx = job_rx.clone();
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                while let Ok((idx, bytes)) = job_rx.recv() {
                    let plan = &plans_ref[idx];
                    let [(gw, gs), (uw, us), (dw, ds)] = bytes;
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
                    completed_ref.fetch_add(1, Ordering::Relaxed);
                    let out = EOut { rows: plan.rows.clone(), rw: plan.rw.clone(), h };
                    if res_tx.send(Ok((plan.pos, out))).is_err() {
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
        loop {
            match res_rx.recv() {
                Ok(Ok((pos, eo))) => {
                    slots[pos] = Some(eo);
                    got += 1;
                    if got == n {
                        break;
                    }
                }
                Ok(Err(e)) => return Err(e),
                // channel closed: fine only if every expert already arrived
                Err(_) => {
                    if got == n {
                        break;
                    }
                    return Err(Error::Format(format!(
                        "concurrent MoE: io/cpu lane ended early ({got}/{n} experts)"
                    )));
                }
            }
        }
        Ok(slots)
    });
    let slots = results?;

    // ---- deterministic reduce: scatter in fixed batch-union order ----
    let mut out = vec![0f32; s_n * hidden];
    for eo in slots.into_iter().flatten() {
        for (ri, (&s, &wgt)) in eo.rows.iter().zip(&eo.rw).enumerate() {
            let dst = &mut out[s * hidden..s * hidden + hidden];
            let src = &eo.h[ri * hidden..ri * hidden + hidden];
            for d in 0..hidden {
                dst[d] += wgt * src[d];
            }
        }
    }
    if let Some(sh) = shared {
        let hs = sh.swiglu(x, s_n);
        for z in 0..s_n * hidden {
            out[z] += hs[z];
        }
    }

    // Accumulate routing frequency so the GPU tier can migrate hot experts into
    // VRAM (heat-ranked residency). Union hotness is the batched-relevant signal;
    // single-threaded here (after the reduce), so the lock-free bumps never race.
    if let Some(heat) = ctx.heat {
        for &e in &uniq {
            heat.bump(layer, e as usize);
        }
    }

    // Record this layer's routed set so the prefetch lane can predict the next
    // token's experts. Single-threaded here (after the reduce) — no race.
    if let Some(rl) = ctx.route_log {
        let mut h = rl.lock();
        if layer < h.len() {
            h[layer] = uniq;
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
            Err(_) => {
                std::fs::remove_file(&path)?;
                return Ok(()); // no io_uring on this host → skip
            }
        };
        let slab = read_expert(&mut reactor, &gate, &up, &down, false)?;
        assert_eq!(slab[0].0, regions[0]);
        assert_eq!(slab[0].1, regions[1]);
        assert_eq!(slab[1].0, regions[2]);
        assert_eq!(slab[1].1, regions[3]);
        assert_eq!(slab[2].0, regions[4]);
        assert_eq!(slab[2].1, regions[5]);
        std::fs::remove_file(&path)?;
        Ok(())
    }
}

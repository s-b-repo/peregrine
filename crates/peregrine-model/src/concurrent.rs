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
use peregrine_io::Reactor;

use crate::mlp::Mlp;
use crate::router::{batch_union, route};
use crate::weight::{QtWeight, QuantFmt};

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
    fmt: QuantFmt,
    o: usize,
    i: usize,
    gs: usize,
}

/// One expert's streaming+compute plan: which rows route to it (+ gate weights)
/// and where its gate/up/down tensors live on disk.
struct EPlan {
    rows: Vec<usize>,
    rw: Vec<f32>,
    gate: TPlan,
    up: TPlan,
    down: TPlan,
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
    Ok(TPlan { w_fd, w_off, w_len, s_fd, s_off, s_len, fmt, o, i, gs: info.gs as usize })
}

/// Stream one tensor's weight+scale bytes through the ring (both to completion).
fn read_tensor(r: &mut Reactor, t: &TPlan) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut w = vec![0u8; t.w_len];
    let mut s = vec![0u8; t.s_len];
    r.read_exact(t.w_fd, t.w_off, &mut w).ctx(|| format!("io_uring weight read @ {}", t.w_off))?;
    r.read_exact(t.s_fd, t.s_off, &mut s).ctx(|| format!("io_uring scale read @ {}", t.s_off))?;
    Ok((w, s))
}

fn rebuild(t: &TPlan, wb: Vec<u8>, sb: Vec<u8>) -> QtWeight {
    let scale: Vec<f32> = sb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    match t.fmt {
        QuantFmt::Int4Grouped => QtWeight::new_grouped(t.o, t.i, wb, scale, t.gs),
        f => QtWeight::new(f, t.o, t.i, wb, scale),
    }
}

/// Concurrent streamed MoE forward. Bit-identical to the sequential streamed
/// path; only faster (io_uring reads overlap CPU-pool compute).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward_concurrent(
    st: &SafeTensors,
    reactor: &Mutex<Reactor>,
    workers: usize,
    layer: usize,
    cfg: &Cfg,
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    shared: Option<&Mlp>,
    s_n: usize,
) -> Result<Vec<f32>, Error> {
    let hidden = cfg.hidden as usize;
    let (e_n, mi, k) = (cfg.n_experts as usize, cfg.moe_inter as usize, cfg.topk as usize);
    let r = route(x, router_w, router_bias, s_n, hidden, e_n, k, cfg.norm_topk, cfg.routed_scale);

    // Build one plan per unique routed expert (main thread; no disk I/O yet).
    let mut plans: Vec<EPlan> = Vec::new();
    for &e in batch_union(&r, s_n).iter() {
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
        let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
        plans.push(EPlan {
            rows,
            rw,
            gate: tplan(st, &p("gate_proj.weight"), mi, hidden)?,
            up: tplan(st, &p("up_proj.weight"), mi, hidden)?,
            down: tplan(st, &p("down_proj.weight"), hidden, mi)?,
        });
    }
    let n = plans.len();

    // job: (pos, streamed gate/up/down bytes) from I/O lane → CPU pool
    type Bytes3 = [(Vec<u8>, Vec<u8>); 3];
    let (job_tx, job_rx) = crossbeam_channel::bounded::<(usize, Bytes3)>(workers.max(1) * 2);
    // result: (pos, computed expert) from CPU pool → main reducer
    let (res_tx, res_rx) = crossbeam_channel::bounded::<Result<(usize, EOut), Error>>(workers.max(1) * 2);

    let completed = AtomicUsize::new(0);
    let plans_ref = &plans;
    let x_ref = x;
    let completed_ref = &completed;

    let results: Result<Vec<Option<EOut>>, Error> = std::thread::scope(|scope| {
        // ---- I/O lane: stream each expert's 6 regions through the ring ----
        {
            let job_tx = job_tx.clone();
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                for (pos, plan) in plans_ref.iter().enumerate() {
                    let read = {
                        let mut ring = reactor.lock();
                        read_tensor(&mut ring, &plan.gate).and_then(|g| {
                            let u = read_tensor(&mut ring, &plan.up)?;
                            let d = read_tensor(&mut ring, &plan.down)?;
                            Ok([g, u, d])
                        })
                    };
                    match read {
                        Ok(bytes) => {
                            if job_tx.send((pos, bytes)).is_err() {
                                break; // pool gone (error elsewhere)
                            }
                        }
                        Err(e) => {
                            let _ = res_tx.send(Err(e));
                            break;
                        }
                    }
                }
                // senders drop here → CPU pool drains and exits
            });
        }

        // ---- CPU lane: pool of workers computing SwiGLU per expert ----
        for _ in 0..workers.max(1) {
            let job_rx = job_rx.clone();
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                while let Ok((pos, bytes)) = job_rx.recv() {
                    let plan = &plans_ref[pos];
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
                    if res_tx.send(Ok((pos, out))).is_err() {
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
    Ok(out)
}

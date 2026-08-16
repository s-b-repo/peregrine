//! colibrì concurrent MoE scheduler (M4 core).
//!
//! The throughput lever from the plan: instead of the C engine's phased
//! "stream-then-compute", this overlaps the **I/O lane** (io_uring streaming of
//! disk-resident experts) with the **CPU lane** (computing RAM-resident experts)
//! on the same MoE layer, then merges. Output is identical (within float
//! reassociation) to the sequential path.
//!
//! This is the CPU∥SSD half of the three-lane design; the GPU lane composes the
//! same way (feature-gated FFI, validated on an NVIDIA box).

// Quality gates: no unsafe, no panicking error handling (denied in tests too).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod reconstruct;

use std::os::unix::io::RawFd;

use peregrine_core::{Context, Error};
use peregrine_model::{batch_union, route, Mlp, MoeCfg, Routed, RouterCfg};
use reconstruct::{mlp_from_segments, QtMeta};

/// Where an expert's weights live.
pub enum ExpertLoc<'a> {
    /// already in RAM — compute immediately on the CPU lane
    Resident(&'a Mlp),
    /// on disk — stream via the io_uring I/O lane, then compute. Boxed because a
    /// `DiskExpert` (3 tensors × regions) dwarfs the resident pointer, and it's
    /// always about to incur a disk read anyway.
    Disk(Box<DiskExpert>),
}

/// One on-disk quantized tensor: the packed-weight region and the f32-scale
/// region (each `(fd, offset, len)`), plus the format/shape to rebuild it. The
/// two regions are the actual safetensors tensor byte ranges — streamed in
/// place, no sidecar file.
#[derive(Clone, Copy, Debug)]
pub struct DiskQt {
    pub w_fd: RawFd,
    pub w_off: u64,
    pub w_len: usize,
    pub s_fd: RawFd,
    pub s_off: u64,
    pub s_len: usize,
    pub meta: QtMeta,
}

/// A streamable expert: its gate/up/down tensors, each streamed from the
/// checkpoint and reconstructed after the reads complete.
#[derive(Clone, Copy, Debug)]
pub struct DiskExpert {
    pub gate: DiskQt,
    pub up: DiskQt,
    pub down: DiskQt,
}

/// Owns the io_uring ring so it's set up **once** and reused across every
/// `moe_streamed` call (the ring is a syscall + a couple of mmaps to create —
/// per-layer-per-token setup would dominate). This is the persistent I/O lane;
/// there is no pread fallback — a missing ring is a hard error.
pub struct Streamer {
    reactor: peregrine_io::Reactor,
}

impl Streamer {
    /// Create a reusable streamer with a ring of `depth` submission slots.
    /// Errors if io_uring is unavailable (Linux without io_uring, or non-Linux).
    pub fn new(depth: u32) -> Result<Streamer, Error> {
        let reactor = peregrine_io::Reactor::new(depth.max(1)).ctx(|| "io_uring reactor init".to_string())?;
        Ok(Streamer { reactor })
    }

    /// Stream every disk expert's gate/up/down tensors (6 regions each) and
    /// reconstruct them into `Mlp`s. Each region is read to completion through
    /// the io_uring ring (short completions are retried by `read_exact`); any
    /// I/O error propagates — no fallback.
    fn read_experts(&mut self, experts: &[(usize, &DiskExpert)]) -> Result<Vec<(usize, Mlp)>, Error> {
        // One deep submit for every region of every expert. Reading them one at
        // a time (six blocking `read_exact`s per expert) put the ring at queue
        // depth 1 — 6·E serialized NVMe round-trips on the lane whose whole
        // purpose is to overlap them.
        let mut bufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(experts.len() * 3);
        for (_, de) in experts {
            for q in [&de.gate, &de.up, &de.down] {
                bufs.push((vec![0u8; q.w_len], vec![0u8; q.s_len]));
            }
        }
        {
            // Borrow every landing buffer at once so a single `read_many` covers
            // the whole batch.
            let mut reqs: Vec<peregrine_io::ReadReq> = Vec::with_capacity(bufs.len() * 2);
            let mut qts: Vec<&DiskQt> = Vec::with_capacity(bufs.len());
            for (_, de) in experts {
                qts.extend([&de.gate, &de.up, &de.down]);
            }
            for (q, (w, sc)) in qts.iter().zip(bufs.iter_mut()) {
                reqs.push(peregrine_io::ReadReq { fd: q.w_fd, offset: q.w_off, buf: w.as_mut_slice(), tag: 0 });
                reqs.push(peregrine_io::ReadReq { fd: q.s_fd, offset: q.s_off, buf: sc.as_mut_slice(), tag: 0 });
            }
            let results = self.reactor.read_many(&mut reqs).ctx(|| "io_uring batched expert read".to_string())?;
            // Complete any short region individually (the kernel may return less
            // than requested); a hard error propagates.
            for (j, n) in results.iter().enumerate() {
                let want = reqs[j].buf.len() as i64;
                if *n == want {
                    continue;
                }
                if *n < 0 {
                    return Err(Error::Format(format!(
                        "io_uring batched expert read failed at region {j} (errno {})",
                        -*n
                    )));
                }
                let done = (*n).max(0) as usize;
                let (fd, off) = (reqs[j].fd, reqs[j].offset + done as u64);
                self.reactor
                    .read_exact(fd, off, &mut reqs[j].buf[done..])
                    .ctx(|| format!("io_uring expert read completion @ {off}"))?;
            }
        }

        let mut out = Vec::with_capacity(experts.len());
        for (i, (eid, de)) in experts.iter().enumerate() {
            let metas = [de.gate.meta, de.up.meta, de.down.meta];
            let base = i * 3;
            let bufs6: [(Vec<u8>, Vec<u8>); 3] = [
                std::mem::take(&mut bufs[base]),
                std::mem::take(&mut bufs[base + 1]),
                std::mem::take(&mut bufs[base + 2]),
            ];
            out.push((*eid, mlp_from_segments(&metas, &bufs6)?));
        }
        Ok(out)
    }
}

/// Accumulate one expert's contribution into `out` (gather routed rows → SwiGLU
/// → weighted scatter). Identical math to `peregrine_model::moe_forward`'s inner loop.
fn contribute(out: &mut [f32], x: &[f32], mlp: &Mlp, r: &Routed, eid: usize, hidden: usize, s_n: usize) {
    let mut rows = Vec::new();
    let mut rw = Vec::new();
    for s in 0..s_n {
        // `keff` can never exceed `k`, but it arrives from the caller — clamp so
        // a malformed `Routed` cannot index past the row.
        for kk in 0..(r.keff[s].max(0) as usize).min(r.k) {
            if r.idx[s * r.k + kk] as usize == eid {
                rows.push(s);
                rw.push(r.w[s * r.k + kk]);
                break;
            }
        }
    }
    if rows.is_empty() {
        return;
    }
    let nr = rows.len();
    let mut xg = vec![0f32; nr * hidden];
    for (ri, &s) in rows.iter().enumerate() {
        xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
    }
    let h = mlp.swiglu(&xg, nr);
    for (ri, (&s, &wgt)) in rows.iter().zip(&rw).enumerate() {
        let dst = &mut out[s * hidden..s * hidden + hidden];
        let src = &h[ri * hidden..ri * hidden + hidden];
        for d in 0..hidden {
            dst[d] += wgt * src[d];
        }
    }
}

/// Concurrent MoE forward: I/O lane streams disk experts while the CPU lane
/// computes resident experts; results merge into one output `[s_n, hidden]`.
pub fn moe_streamed(
    streamer: &mut Streamer,
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    experts: &[ExpertLoc],
    shared: Option<&Mlp>,
    cfg: MoeCfg,
) -> Result<Vec<f32>, Error> {
    let MoeCfg { s_n, hidden, k: topk, norm_topk, routed_scale } = cfg;
    let e_n = experts.len();
    let r = route(x, router_w, router_bias, RouterCfg { s_n, d_n: hidden, e_n, k: topk, norm_topk, routed_scale, min_share: 0.0 });
    let uniq = batch_union(&r, s_n);

    // partition the batch-union by residency
    let mut resident: Vec<(usize, &Mlp)> = Vec::new();
    let mut disk: Vec<(usize, &DiskExpert)> = Vec::new();
    for &e in &uniq {
        match &experts[e as usize] {
            ExpertLoc::Resident(m) => resident.push((e as usize, m)),
            ExpertLoc::Disk(d) => disk.push((e as usize, d.as_ref())),
        }
    }

    // I/O lane (streaming, reusing the persistent ring) ∥ CPU lane (resident compute)
    let sr = &mut *streamer;
    let disk_ref = &disk;
    let (mut out, streamed) = std::thread::scope(|sc| {
        let io = sc.spawn(move || sr.read_experts(disk_ref));
        let mut out = vec![0f32; s_n * hidden];
        for &(eid, mlp) in &resident {
            contribute(&mut out, x, mlp, &r, eid, hidden, s_n);
        }
        // a panic in the io lane becomes an error here, never a re-panic
        let streamed = match io.join() {
            Ok(res) => res,
            Err(payload) => {
                let why = payload
                    .downcast_ref::<&str>()
                    .map(|m| (*m).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                Err(Error::Format(format!("io lane thread panicked: {why}")))
            }
        };
        (out, streamed)
    });

    // compute the streamed experts (now resident) and merge
    for (eid, mlp) in streamed? {
        contribute(&mut out, x, &mlp, &r, eid, hidden, s_n);
    }

    if let Some(sh) = shared {
        let hs = sh.swiglu(x, s_n);
        for z in 0..s_n * hidden {
            out[z] += hs[z];
        }
    }
    Ok(out)
}

/// [`moe_streamed`] as an installable [`peregrine_model::concurrent::MoeEngine`].
///
/// **This is what gives `moe_streamed` a production caller.** The direct wiring
/// is impossible: this crate depends on `peregrine-model`, so `peregrine-model`
/// calling into it would be a dependency cycle. The dependency is inverted
/// instead — `peregrine-model` declares the trait, this implements it, and a
/// binary installs it when `COLI_MOE_ENGINE=sched`.
///
/// **Opt-in, and it should stay opt-in.** `moe_streamed` is the two-lane
/// ancestor: one io_uring ring, no GPU lane, no warm cache, no prefetch, no lane
/// balancer. On any host with a GPU tier or a warm cache it will be slower than
/// the default `moe_forward_concurrent`. Its value is as a second, independently
/// written implementation that `streamed_matches_the_production_concurrent_path`
/// checks the first against — and, now, as a runtime A/B of that same pair.
///
/// The `Streamer` owns an io_uring ring and `moe_streamed` needs it mutably,
/// while the trait hands out `&self`; the `Mutex` is that adaptation and also
/// serialises layers, which is correct — the ring is not shareable.
pub struct SchedEngine {
    streamer: std::sync::Mutex<Streamer>,
}

impl SchedEngine {
    /// Build the engine with an io_uring ring `depth` entries deep.
    pub fn new(depth: u32) -> Result<SchedEngine, Error> {
        Ok(SchedEngine { streamer: std::sync::Mutex::new(Streamer::new(depth)?) })
    }
}

impl peregrine_model::concurrent::MoeEngine for SchedEngine {
    fn name(&self) -> &'static str {
        "sched"
    }

    fn moe_forward(
        &self,
        ctx: &peregrine_model::concurrent::ForwardCtx,
        call: peregrine_model::concurrent::MoeCall,
    ) -> Result<Vec<f32>, Error> {
        use peregrine_core::QtInfo;
        let peregrine_model::concurrent::MoeCall { layer, x, router_w, router_bias, shared, s_n } = call;
        let cfg = ctx.cfg;
        let hidden = cfg.hidden as usize;
        let mi = cfg.moe_inter as usize;
        let e_n = cfg.n_experts as usize;

        // Locate this layer's experts in the container the model already opened.
        // `SafeTensors::region` yields the same `(fd, offset, len)` triples
        // `concurrent.rs`'s private `tplan` builds its plans from, so both
        // engines read one file rather than two.
        let qt_of = |name: &str, o: usize, i: usize| -> Result<DiskQt, Error> {
            let info = QtInfo::detect(ctx.st, name, o as i64, i as i64);
            let sname = format!("{name}.qs");
            let (w_fd, w_off, w_len) =
                ctx.st.region(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))?;
            let (s_fd, s_off, s_len) =
                ctx.st.region(&sname).ok_or_else(|| Error::Format(format!("missing tensor {sname}")))?;
            Ok(DiskQt {
                w_fd,
                w_off,
                w_len,
                s_fd,
                s_off,
                s_len,
                meta: QtMeta { fmt: info.fmt, o, i, gs: info.gs as usize },
            })
        };
        let mut disk: Vec<DiskExpert> = Vec::with_capacity(e_n);
        for e in 0..e_n {
            let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
            disk.push(DiskExpert {
                gate: qt_of(&p("gate_proj.weight"), mi, hidden)?,
                up: qt_of(&p("up_proj.weight"), mi, hidden)?,
                down: qt_of(&p("down_proj.weight"), hidden, mi)?,
            });
        }
        let locs: Vec<ExpertLoc> = disk.iter().map(|d| ExpertLoc::Disk(Box::new(*d))).collect();

        let mut streamer = self
            .streamer
            .lock()
            .map_err(|e| Error::Format(format!("sched streamer lock poisoned: {e}")))?;
        moe_streamed(
            &mut streamer,
            x,
            router_w,
            router_bias,
            &locs,
            shared,
            MoeCfg {
                s_n,
                hidden,
                k: cfg.topk as usize,
                norm_topk: cfg.norm_topk,
                routed_scale: cfg.routed_scale,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::reconstruct::QtMeta;
    use super::*;
    use peregrine_core::pack::{f32_bytes, quant_i4};
    use peregrine_core::QtFmt;
    use peregrine_model::{moe_forward, Mlp, MoeCfg, QtWeight, QuantFmt};
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn qi4(w: &[f32], o: usize, i: usize) -> QtWeight {
        let (q, s) = quant_i4(w, o, i);
        QtWeight::new(QuantFmt::Int4, o, i, q, s)
    }

    fn make_mlp(r: &mut Lcg, hidden: usize, inter: usize) -> Mlp {
        let g: Vec<f32> = (0..inter * hidden).map(|_| r.f()).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|_| r.f()).collect();
        let d: Vec<f32> = (0..hidden * inter).map(|_| r.f()).collect();
        Mlp { gate: qi4(&g, inter, hidden), up: qi4(&u, inter, hidden), down: qi4(&d, hidden, inter) }
    }

    /// Append one QtWeight's weight + scale regions to `f`, returning its DiskQt
    /// (with offsets relative to the start of the file = absolute here).
    fn write_qt(
        f: &mut std::fs::File,
        cursor: &mut u64,
        fd: RawFd,
        w: &QtWeight,
        o: usize,
        i: usize,
    ) -> Result<DiskQt, std::io::Error> {
        let (q, s) = w.raw();
        let sb = f32_bytes(s);
        let w_off = *cursor;
        f.write_all(q)?;
        *cursor += q.len() as u64;
        let s_off = *cursor;
        f.write_all(&sb)?;
        *cursor += sb.len() as u64;
        Ok(DiskQt {
            w_fd: fd,
            w_off,
            w_len: q.len(),
            s_fd: fd,
            s_off,
            s_len: sb.len(),
            meta: QtMeta { fmt: QtFmt::Int4, o, i, gs: 0 },
        })
    }

    /// **The oracle this crate exists to be.**
    ///
    /// `peregrine-sched` has no dependents: production MoE is
    /// `peregrine-model/concurrent.rs::moe_forward_concurrent`, and until now
    /// nothing compared the two, so the crate was neither used nor useful — the
    /// `[R]` finding in `docs/BAD_PATTERNS.md`. A second, independently written
    /// implementation of the same computation is worth keeping *only* if
    /// something checks it against the first; otherwise it is a second thing to
    /// keep correct with no benefit.
    ///
    /// The comparison is possible because both entry points take the router
    /// weights as arguments, so both route identically by construction, and
    /// because `SafeTensors::region` exposes the same `(fd, offset, len)` triples
    /// `concurrent.rs`'s private `tplan` builds its own plans from. Pointing
    /// `DiskQt` at the container's own bytes is what makes this an equivalence
    /// test rather than two engines reading two different files.
    ///
    /// Tolerance, not bits: the lanes accumulate expert contributions in
    /// different orders, and `f32 +=` is not associative. Bit-identity is
    /// asserted *within* each engine (`concurrent`'s `pos`-keyed reduce), not
    /// across them.
    #[test]
    fn streamed_matches_the_production_concurrent_path() -> Result<(), peregrine_core::Error> {
        use peregrine_core::{Cfg, QtInfo, SafeTensors};
        use peregrine_model::concurrent::{moe_forward_concurrent, ForwardCtx};
        use peregrine_model::testkit::build_tiny_model_seeded;
        use parking_lot::Mutex;

        let dir = std::env::temp_dir().join(format!("peregrine_oracle_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        build_tiny_model_seeded(&dir, 0x5EED)?;
        let cfg = Cfg::load(&dir)?;
        let st = SafeTensors::open(&dir)?;

        let (hidden, e_n) = (cfg.hidden as usize, cfg.n_experts as usize);
        let mi = cfg.moe_inter as usize;
        let s_n = 3usize;
        // The first sparse layer: `first_dense` is the count of leading dense
        // layers, so it is also the index of the first layer with experts.
        let layer = cfg.first_dense as usize;

        let mut r = Lcg(0xC0FFEE);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        // Our own router, not the checkpoint's: both engines take it as an
        // argument, so supplying it directly removes the router from the
        // comparison and leaves only the expert path — which is what differs.
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| r.f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| r.f() * 0.1).collect();

        let reactors = vec![Mutex::new(peregrine_io::Reactor::new(64)?)];
        let ctx = ForwardCtx {
            st: &st,
            absorb: false,
            dsa: false,
            reactors: &reactors,
            gpu: None,
            workers: 1,
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
            // The oracle re-derives plans per request, like every path did before
            // `ExpertIndex` existed. `None` is the documented fallback and reads
            // the same bytes, which is the only property this test cares about.
            expert_index: None,
            fd_devices: None,
        };
        let production = moe_forward_concurrent(&ctx, layer, &x, &router_w, &router_bias, None, s_n)?;

        // Point this crate's `DiskQt`s at the *same* container bytes. `region`
        // is what `tplan` uses; a scale tensor is always `<name>.qs`.
        let qt_of = |name: &str, o: usize, i: usize| -> Result<DiskQt, peregrine_core::Error> {
            let info = QtInfo::detect(&st, name, o as i64, i as i64);
            let sname = format!("{name}.qs");
            let (w_fd, w_off, w_len) = st
                .region(name)
                .ok_or_else(|| peregrine_core::Error::Format(format!("missing tensor {name}")))?;
            let (s_fd, s_off, s_len) = st
                .region(&sname)
                .ok_or_else(|| peregrine_core::Error::Format(format!("missing tensor {sname}")))?;
            Ok(DiskQt {
                w_fd,
                w_off,
                w_len,
                s_fd,
                s_off,
                s_len,
                meta: QtMeta { fmt: info.fmt, o, i, gs: info.gs as usize },
            })
        };
        let mut disk: Vec<DiskExpert> = Vec::with_capacity(e_n);
        for e in 0..e_n {
            let p = |t: &str| format!("model.layers.{layer}.mlp.experts.{e}.{t}");
            disk.push(DiskExpert {
                gate: qt_of(&p("gate_proj.weight"), mi, hidden)?,
                up: qt_of(&p("up_proj.weight"), mi, hidden)?,
                down: qt_of(&p("down_proj.weight"), hidden, mi)?,
            });
        }
        let locs: Vec<ExpertLoc> = disk.iter().map(|d| ExpertLoc::Disk(Box::new(*d))).collect();

        let mut streamer = Streamer::new(64)?;
        let mcfg = MoeCfg {
            s_n,
            hidden,
            k: cfg.topk as usize,
            norm_topk: cfg.norm_topk,
            routed_scale: cfg.routed_scale,
        };
        let streamed = moe_streamed(&mut streamer, &x, &router_w, &router_bias, &locs, None, mcfg)?;

        assert_eq!(streamed.len(), production.len(), "both engines produce [s_n, hidden]");
        // A model whose output is all zeros would satisfy any tolerance, so
        // require the comparison to have had something to compare.
        assert!(
            production.iter().any(|v| v.abs() > 1e-6),
            "the reference output is entirely zero — the fixture routed nothing and this test proves nothing"
        );
        for z in 0..s_n * hidden {
            let tol = 1e-3 * production[z].abs().max(1.0);
            assert!(
                (production[z] - streamed[z]).abs() < tol,
                "z={z} concurrent={} streamed={}",
                production[z],
                streamed[z]
            );
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Historical name, kept: this compares `moe_streamed` against the fully
    /// **resident** `moe_forward`, not against the production concurrent path —
    /// see `streamed_matches_the_production_concurrent_path` for that. The old
    /// name read as if this were the cross-engine check
    /// (`docs/testing-and-quality.md` flags it), which is why the real one says
    /// what it compares in its name.
    #[test]
    fn streamed_matches_the_resident_reference() -> Result<(), peregrine_core::Error> {
        let (hidden, inter, e_n, k, s_n) = (16usize, 8usize, 6usize, 2usize, 4usize);
        let mut r = Lcg(0xF00D);

        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| r.f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| r.f() * 0.1).collect();
        let experts: Vec<Mlp> = (0..e_n).map(|_| make_mlp(&mut r, hidden, inter)).collect();
        let shared = make_mlp(&mut r, hidden, inter);

        // sequential reference: all experts resident
        let seq = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), MoeCfg { s_n, hidden, k, norm_topk: true, routed_scale: 2.5 });

        // write the odd-indexed experts' gate/up/down (6 regions each) to a file;
        // even ones stay resident — exercises the mixed CPU∥IO path.
        let path = std::env::temp_dir().join(format!("peregrine_sched_{}", std::process::id()));
        let mut f = std::fs::File::create(&path)?;
        let rf = std::fs::File::open(&path)?;
        let fd = rf.as_raw_fd();
        let mut cursor = 0u64;
        let mut disk: Vec<Option<DiskExpert>> = (0..e_n).map(|_| None).collect();
        for (e, m) in experts.iter().enumerate() {
            if e % 2 == 1 {
                let gate = write_qt(&mut f, &mut cursor, fd, &m.gate, inter, hidden)?;
                let up = write_qt(&mut f, &mut cursor, fd, &m.up, inter, hidden)?;
                let down = write_qt(&mut f, &mut cursor, fd, &m.down, hidden, inter)?;
                disk[e] = Some(DiskExpert { gate, up, down });
            }
        }
        f.sync_all()?;

        let locs: Vec<ExpertLoc> = experts
            .iter()
            .enumerate()
            .map(|(e, m)| match disk[e] {
                Some(de) => ExpertLoc::Disk(Box::new(de)),
                None => ExpertLoc::Resident(m),
            })
            .collect();

        // one persistent streamer, reused across calls (the ring is set up once)
        let mut streamer = Streamer::new(64)?;
        let conc = moe_streamed(&mut streamer, &x, &router_w, &router_bias, &locs, Some(&shared), MoeCfg { s_n, hidden, k, norm_topk: true, routed_scale: 2.5 })?;
        let conc2 = moe_streamed(&mut streamer, &x, &router_w, &router_bias, &locs, Some(&shared), MoeCfg { s_n, hidden, k, norm_topk: true, routed_scale: 2.5 })?;
        assert_eq!(conc, conc2, "reused streamer must give identical output");

        for z in 0..s_n * hidden {
            let tol = 1e-3 * seq[z].abs().max(1.0);
            assert!((seq[z] - conc[z]).abs() < tol, "z={z} seq={} conc={}", seq[z], conc[z]);
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }
}

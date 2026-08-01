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

    #[test]
    fn concurrent_matches_sequential() -> Result<(), peregrine_core::Error> {
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

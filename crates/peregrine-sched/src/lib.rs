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
use peregrine_model::{batch_union, route, Mlp, Routed};
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
        let read_qt = |reactor: &mut peregrine_io::Reactor, q: &DiskQt| -> Result<(Vec<u8>, Vec<u8>), Error> {
            let mut w = vec![0u8; q.w_len];
            let mut s = vec![0u8; q.s_len];
            reactor.read_exact(q.w_fd, q.w_off, &mut w).ctx(|| format!("io_uring expert weight read @ {}", q.w_off))?;
            reactor.read_exact(q.s_fd, q.s_off, &mut s).ctx(|| format!("io_uring expert scale read @ {}", q.s_off))?;
            Ok((w, s))
        };

        let mut out = Vec::with_capacity(experts.len());
        for (eid, de) in experts {
            let metas = [de.gate.meta, de.up.meta, de.down.meta];
            let bufs6 = [
                read_qt(&mut self.reactor, &de.gate)?,
                read_qt(&mut self.reactor, &de.up)?,
                read_qt(&mut self.reactor, &de.down)?,
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
        for kk in 0..r.keff[s] as usize {
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
#[allow(clippy::too_many_arguments)]
pub fn moe_streamed(
    streamer: &mut Streamer,
    x: &[f32],
    hidden: usize,
    s_n: usize,
    router_w: &[f32],
    router_bias: &[f32],
    topk: usize,
    norm_topk: bool,
    routed_scale: f32,
    experts: &[ExpertLoc],
    shared: Option<&Mlp>,
) -> Result<Vec<f32>, Error> {
    let e_n = experts.len();
    let r = route(x, router_w, router_bias, s_n, hidden, e_n, topk, norm_topk, routed_scale);
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
            Err(_) => Err(Error::Format("io lane thread panicked".into())),
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
    use peregrine_model::{moe_forward, Mlp, QtWeight, QuantFmt};
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
        let seq = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), s_n, hidden, k, true, 2.5);

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
        let conc = moe_streamed(&mut streamer, &x, hidden, s_n, &router_w, &router_bias, k, true, 2.5, &locs, Some(&shared))?;
        let conc2 = moe_streamed(&mut streamer, &x, hidden, s_n, &router_w, &router_bias, k, true, 2.5, &locs, Some(&shared))?;
        assert_eq!(conc, conc2, "reused streamer must give identical output");

        for z in 0..s_n * hidden {
            let tol = 1e-3 * seq[z].abs().max(1.0);
            assert!((seq[z] - conc[z]).abs() < tol, "z={z} seq={} conc={}", seq[z], conc[z]);
        }
        std::fs::remove_file(&path)?;
        Ok(())
    }
}

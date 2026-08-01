//! MLP compute: the SwiGLU block shared by the dense layers, the shared expert,
//! and every routed expert, plus the full MoE forward (route → gather → expert
//! SwiGLU → weighted scatter → shared expert). Ports `dense_mlp` (`c/glm.c:3201`)
//! and the MoE compute of `moe()` (Phase A–E), minus the streaming/tiering
//! (M2), CACHE_ROUTE, and EXPERT_BUDGET opt-ins.

use crate::math::silu_mul;
use crate::router::{batch_union, route, RouterCfg};
use crate::weight::QtWeight;

/// One expert (or the shared expert / a dense layer's MLP): gate, up, down.
pub struct Mlp {
    pub gate: QtWeight,
    pub up: QtWeight,
    pub down: QtWeight,
}

impl Mlp {
    /// SwiGLU: `down( silu(gate·x) ⊙ (up·x) )`. Input `x[s_n, gate.i]`, output
    /// `[s_n, down.o]`.
    pub fn swiglu(&self, x: &[f32], s_n: usize) -> Vec<f32> {
        let mut g = self.gate.apply_vec(x, s_n);
        let u = self.up.apply_vec(x, s_n);
        silu_mul(&mut g, &u);
        self.down.apply_vec(&g, s_n)
    }
}

/// The per-layer MoE configuration: batch shape plus the routing knobs. One
/// value instead of five positional parameters, so a call site cannot silently
/// transpose `hidden` and `k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeCfg {
    pub s_n: usize,
    pub hidden: usize,
    pub k: usize,
    pub norm_topk: bool,
    pub routed_scale: f32,
}

/// Full MoE layer forward. `x[s_n, hidden]` → `out[s_n, hidden]`.
///
/// Each unique routed expert is computed once over the positions that route to
/// it (the batch-union invariant), its output scattered back weighted by the
/// gate weight; the shared expert (if any) is added to every position.
pub fn moe_forward(
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    experts: &[Mlp],
    shared: Option<&Mlp>,
    cfg: MoeCfg,
) -> Vec<f32> {
    let MoeCfg { s_n, hidden, k, norm_topk, routed_scale } = cfg;
    let e_n = experts.len();
    let r = route(x, router_w, router_bias, RouterCfg { s_n, d_n: hidden, e_n, k, norm_topk, routed_scale, min_share: crate::router::route_min_share() });
    let mut out = vec![0f32; s_n * hidden];

    // Gather each routed expert's positions (+ gate weights) in batch-union order.
    struct Plan {
        e: usize,
        rows: Vec<usize>,
        rw: Vec<f32>,
    }
    // One pass over the routing table buckets every expert's rows; the previous
    // shape rescanned the whole table once per unique expert
    // (O(|union| × S × K) — millions of comparisons per sparse layer on a long
    // prefill) to recover information a single sweep already has.
    //
    // Emission stays in `batch_union` order and rows stay in ascending `s` with
    // the first matching `kk` per position, so the scatter order — and therefore
    // the f32 accumulation order — is bit-identical to the rescan version.
    let union = batch_union(&r, s_n);
    let mut slot_of: Vec<Option<usize>> = vec![None; e_n];
    let mut plans: Vec<Plan> = Vec::with_capacity(union.len());
    for &e in union.iter() {
        if let Some(slot) = usize::try_from(e).ok().filter(|&e| e < e_n) {
            if slot_of[slot].is_none() {
                slot_of[slot] = Some(plans.len());
                plans.push(Plan { e: slot, rows: Vec::new(), rw: Vec::new() });
            }
        }
    }
    for s in 0..s_n {
        for kk in 0..(r.keff[s].max(0) as usize).min(r.k) {
            let Ok(e) = usize::try_from(r.idx[s * r.k + kk]) else { continue };
            let Some(&Some(pi)) = slot_of.get(e) else { continue };
            // one row per position per expert — mirrors the original `break`
            if plans[pi].rows.last() == Some(&s) {
                continue;
            }
            plans[pi].rows.push(s);
            plans[pi].rw.push(r.w[s * r.k + kk]);
        }
    }
    plans.retain(|p| !p.rows.is_empty());

    // Compute each expert's SwiGLU on the pool (disjoint scratch), then scatter
    // SERIALLY in batch-union order — a row hit by two experts accumulates in the
    // same order as the serial loop, so the result is bit-identical (f32 `+=` is not
    // associative). This mirrors the concurrent streaming lane's post-scope reduce.
    let hs: Vec<Vec<f32>> = peregrine_par::par_map(plans.len(), peregrine_par::PAR_MOE_MIN, |i| {
        let p = &plans[i];
        let nr = p.rows.len();
        let mut xg = vec![0f32; nr * hidden];
        for (ri, &s) in p.rows.iter().enumerate() {
            xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
        }
        experts[p.e].swiglu(&xg, nr)
    });
    for (p, h) in plans.iter().zip(&hs) {
        for (ri, (&s, &wgt)) in p.rows.iter().zip(&p.rw).enumerate() {
            let dst = &mut out[s * hidden..s * hidden + hidden];
            let src = &h[ri * hidden..ri * hidden + hidden];
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::siluf;
    use crate::weight::test_support::quant_i4;
    use peregrine_kernels::matmul_f32;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    // Full-f32 reference SwiGLU using dequantized weights.
    fn swiglu_ref(m: &Mlp, x: &[f32], s_n: usize, hidden: usize, inter: usize) -> Vec<f32> {
        let mut g = vec![0f32; s_n * inter];
        let mut u = vec![0f32; s_n * inter];
        matmul_f32(&mut g, x, &m.gate.dequant(), s_n, hidden, inter);
        matmul_f32(&mut u, x, &m.up.dequant(), s_n, hidden, inter);
        for z in 0..s_n * inter {
            g[z] = siluf(g[z]) * u[z];
        }
        let mut h = vec![0f32; s_n * hidden];
        matmul_f32(&mut h, &g, &m.down.dequant(), s_n, inter, hidden);
        h
    }

    fn make_mlp(rng: &mut Lcg, hidden: usize, inter: usize) -> Mlp {
        let gate: Vec<f32> = (0..inter * hidden).map(|_| rng.f()).collect();
        let up: Vec<f32> = (0..inter * hidden).map(|_| rng.f()).collect();
        let down: Vec<f32> = (0..hidden * inter).map(|_| rng.f()).collect();
        Mlp {
            gate: quant_i4(&gate, inter, hidden),
            up: quant_i4(&up, inter, hidden),
            down: quant_i4(&down, hidden, inter),
        }
    }

    #[test]
    fn moe_forward_tracks_f32_reference() {
        let (hidden, inter, e_n, k, s_n) = (16usize, 8usize, 4usize, 2usize, 3usize);
        let mut rng = Lcg(0xdead_beef);

        let x: Vec<f32> = (0..s_n * hidden).map(|_| rng.f()).collect();
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| rng.f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| rng.f() * 0.1).collect();
        let experts: Vec<Mlp> = (0..e_n).map(|_| make_mlp(&mut rng, hidden, inter)).collect();
        let shared = make_mlp(&mut rng, hidden, inter);

        let out = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), MoeCfg { s_n, hidden, k, norm_topk: true, routed_scale: 2.5 });

        // Reference: identical routing (f32 router), f32 expert compute.
        let r = route(&x, &router_w, &router_bias, RouterCfg { s_n, d_n: hidden, e_n, k, norm_topk: true, routed_scale: 2.5, min_share: 0.0 });
        let mut refout = vec![0f32; s_n * hidden];
        for s in 0..s_n {
            for kk in 0..k {
                let e = r.idx[s * k + kk] as usize;
                let wgt = r.w[s * k + kk];
                let h = swiglu_ref(&experts[e], &x[s * hidden..s * hidden + hidden], 1, hidden, inter);
                for d in 0..hidden {
                    refout[s * hidden + d] += wgt * h[d];
                }
            }
        }
        let hs = swiglu_ref(&shared, &x, s_n, hidden, inter);
        for z in 0..s_n * hidden {
            refout[z] += hs[z];
        }

        for z in 0..s_n * hidden {
            let tol = 0.05 * (inter + hidden) as f32;
            assert!((out[z] - refout[z]).abs() < tol, "z={z} out={} ref={}", out[z], refout[z]);
        }
    }

    #[test]
    fn moe_forward_parallel_matches_serial() {
        // moe_forward now computes experts on the pool and scatters serially; it must
        // be bit-identical to a hand-serial gather→swiglu→scatter in batch-union order.
        let (hidden, inter, e_n, k, s_n) = (16usize, 8usize, 6usize, 2usize, 9usize);
        let mut rng = Lcg(0x9a1e);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| rng.f()).collect();
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| rng.f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| rng.f() * 0.1).collect();
        let experts: Vec<Mlp> = (0..e_n).map(|_| make_mlp(&mut rng, hidden, inter)).collect();
        let shared = make_mlp(&mut rng, hidden, inter);

        let par = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), MoeCfg { s_n, hidden, k, norm_topk: true, routed_scale: 2.5 });

        // serial oracle: gather → swiglu → scatter, in batch-union order
        let r = route(&x, &router_w, &router_bias, RouterCfg { s_n, d_n: hidden, e_n, k, norm_topk: true, routed_scale: 2.5, min_share: 0.0 });
        let mut ser = vec![0f32; s_n * hidden];
        for &e in batch_union(&r, s_n).iter() {
            let e = e as usize;
            let (mut rows, mut rw): (Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new());
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
            let nr = rows.len();
            let mut xg = vec![0f32; nr * hidden];
            for (ri, &s) in rows.iter().enumerate() {
                xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
            }
            let h = experts[e].swiglu(&xg, nr);
            for (ri, (&s, &wgt)) in rows.iter().zip(&rw).enumerate() {
                for d in 0..hidden {
                    ser[s * hidden + d] += wgt * h[ri * hidden + d];
                }
            }
        }
        let sh = shared.swiglu(&x, s_n);
        for z in 0..s_n * hidden {
            ser[z] += sh[z];
        }

        assert!(par.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()), "moe_forward must be bit-identical parallel vs serial");
    }

    #[test]
    fn dense_mlp_is_swiglu() {
        // a "dense" layer is just an Mlp::swiglu over all positions
        let (hidden, inter, s_n) = (12usize, 6usize, 2usize);
        let mut rng = Lcg(0x1234);
        let m = make_mlp(&mut rng, hidden, inter);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| rng.f()).collect();
        let y = m.swiglu(&x, s_n);
        let yref = swiglu_ref(&m, &x, s_n, hidden, inter);
        for z in 0..s_n * hidden {
            assert!((y[z] - yref[z]).abs() < 0.05 * (inter + hidden) as f32);
        }
    }
}

//! MLP compute: the SwiGLU block shared by the dense layers, the shared expert,
//! and every routed expert, plus the full MoE forward (route → gather → expert
//! SwiGLU → weighted scatter → shared expert). Ports `dense_mlp` (`c/glm.c:3201`)
//! and the MoE compute of `moe()` (Phase A–E), minus the streaming/tiering
//! (M2), CACHE_ROUTE, and EXPERT_BUDGET opt-ins.

use crate::math::silu_mul;
use crate::router::{batch_union, route};
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

/// Full MoE layer forward. `x[s_n, hidden]` → `out[s_n, hidden]`.
///
/// Each unique routed expert is computed once over the positions that route to
/// it (the batch-union invariant), its output scattered back weighted by the
/// gate weight; the shared expert (if any) is added to every position.
#[allow(clippy::too_many_arguments)]
pub fn moe_forward(
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    experts: &[Mlp],
    shared: Option<&Mlp>,
    s_n: usize,
    hidden: usize,
    k: usize,
    norm_topk: bool,
    routed_scale: f32,
) -> Vec<f32> {
    let e_n = experts.len();
    let r = route(x, router_w, router_bias, s_n, hidden, e_n, k, norm_topk, routed_scale);
    let mut out = vec![0f32; s_n * hidden];

    for &e in batch_union(&r, s_n).iter() {
        let e = e as usize;
        // gather the positions routing to expert e (+ their gate weights)
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
        let nr = rows.len();
        let mut xg = vec![0f32; nr * hidden];
        for (ri, &s) in rows.iter().enumerate() {
            xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
        }
        let h = experts[e].swiglu(&xg, nr);
        for (ri, (&s, &wgt)) in rows.iter().zip(&rw).enumerate() {
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

        let out = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), s_n, hidden, k, true, 2.5);

        // Reference: identical routing (f32 router), f32 expert compute.
        let r = route(&x, &router_w, &router_bias, s_n, hidden, e_n, k, true, 2.5);
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

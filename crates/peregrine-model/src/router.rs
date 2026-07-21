//! MoE router — Phase A of `moe()` (`c/glm.c:2705-2830`) plus the batch-union
//! (Phase B). Ported base path (CACHE_ROUTE / TOPP / TOPK overrides are later
//! opt-in features).
//!
//! Key subtlety carried over from the C: the correction **bias is used only to
//! select** the top-K experts; the stored gate weight is the plain sigmoid
//! `logit`, not the bias-augmented `choice`.

use peregrine_kernels::matmul_f32;

use crate::math::sigmoidf;

/// Routing decision for a batch of `S` positions, top-`K` each.
pub struct Routed {
    /// selected expert ids, `[S*K]`
    pub idx: Vec<i32>,
    /// gate weights (post norm/scale), `[S*K]`
    pub w: Vec<f32>,
    /// effective experts kept per position, `[S]`
    pub keff: Vec<i32>,
    pub k: usize,
}

/// Route `x[S,D]` through `router_w[E,D]` (+ `router_bias[E]`), selecting top-`k`
/// experts per position. `norm_topk` renormalizes the kept gate weights;
/// `routed_scale` multiplies them (DeepSeek `routed_scaling_factor`).
#[allow(clippy::too_many_arguments)] // router config is inherently wide
pub fn route(
    x: &[f32],
    router_w: &[f32],
    router_bias: &[f32],
    s_n: usize,
    d_n: usize,
    e_n: usize,
    k: usize,
    norm_topk: bool,
    routed_scale: f32,
) -> Routed {
    let mut logits = vec![0f32; s_n * e_n];
    matmul_f32(&mut logits, x, router_w, s_n, d_n, e_n);

    let mut idx = vec![0i32; s_n * k];
    let mut w = vec![0f32; s_n * k];
    let mut choice = vec![0f32; e_n];

    for s in 0..s_n {
        let logit = &mut logits[s * e_n..s * e_n + e_n];
        for e in 0..e_n {
            logit[e] = sigmoidf(logit[e]);
            choice[e] = logit[e] + router_bias[e];
        }
        // greedy top-k by `choice` (bias-augmented), no replacement; ties go to
        // the lowest index (strict `>`), matching the C selection loop.
        let ib = &mut idx[s * k..s * k + k];
        let wb = &mut w[s * k..s * k + k];
        for kk in 0..k {
            let mut best = -1i32;
            let mut bv = -1e30f32;
            for e in 0..e_n {
                if ib[..kk].contains(&(e as i32)) {
                    continue;
                }
                if choice[e] > bv {
                    bv = choice[e];
                    best = e as i32;
                }
            }
            ib[kk] = best;
            wb[kk] = logit[best as usize]; // weight is the sigmoid, not choice
        }
        if norm_topk {
            let mut sm = 0f32;
            for &wi in wb.iter() {
                sm += wi;
            }
            sm += 1e-20;
            for wi in wb.iter_mut() {
                *wi /= sm;
            }
        }
        for wi in wb.iter_mut() {
            *wi *= routed_scale;
        }
    }

    Routed { idx, w, keff: vec![k as i32; s_n], k }
}

/// Batch-union (Phase B): the set of distinct experts routed by any position, in
/// first-seen order. Each unique expert is computed once and applied to all its
/// rows — the invariant the concurrent scheduler (M4) enforces structurally.
pub fn batch_union(r: &Routed, s_n: usize) -> Vec<i32> {
    let mut seen = std::collections::HashSet::new();
    let mut uniq = Vec::new();
    for s in 0..s_n {
        for kk in 0..r.keff[s] as usize {
            let e = r.idx[s * r.k + kk];
            if seen.insert(e) {
                uniq.push(e);
            }
        }
    }
    uniq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_by_bias_weights_by_sigmoid() {
        // E=4, D=1, x=[1]. router rows = raw logits [2,-1,0.5,3]; bias pushes
        // expert 3 below expert 2 for *selection*, but weights stay sigmoid.
        let x = [1.0f32];
        let router_w = [2.0f32, -1.0, 0.5, 3.0]; // [E=4, D=1]
        let bias = [0.0f32, 0.0, 0.0, -0.5];
        let r = route(&x, &router_w, &bias, 1, 1, 4, 2, true, 1.0);

        // choice = [σ2, σ-1, σ0.5, σ3-0.5] = [.8808,.2689,.6225,.4526]
        // top-2 → experts 0 and 2
        assert_eq!(&r.idx[..], &[0, 2]);
        // weights = sigmoid(2), sigmoid(0.5), normalized
        let (s0, s2) = (sigmoidf(2.0), sigmoidf(0.5));
        let sum = s0 + s2;
        assert!((r.w[0] - s0 / sum).abs() < 1e-5);
        assert!((r.w[1] - s2 / sum).abs() < 1e-5);
        assert!((r.w[0] + r.w[1] - 1.0).abs() < 1e-5); // normalized
    }

    #[test]
    fn routed_scale_applies_after_norm() {
        let x = [1.0f32];
        let router_w = [2.0f32, 0.5];
        let bias = [0.0f32, 0.0];
        let r = route(&x, &router_w, &bias, 1, 1, 2, 2, true, 2.5);
        // normalized weights sum to 1, then ×2.5
        assert!((r.w[0] + r.w[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn batch_union_dedups() {
        // 2 positions, K=2: experts {0,2} and {2,3} → union {0,2,3}
        let r = Routed { idx: vec![0, 2, 2, 3], w: vec![1.0; 4], keff: vec![2, 2], k: 2 };
        let u = batch_union(&r, 2);
        assert_eq!(u, vec![0, 2, 3]); // first-seen order
    }

    #[test]
    fn ties_go_to_lowest_index() {
        // two experts with identical choice → lowest index selected first
        let x = [1.0f32];
        let router_w = [1.0f32, 1.0, 1.0];
        let bias = [0.0f32, 0.0, 0.0];
        let r = route(&x, &router_w, &bias, 1, 1, 3, 2, false, 1.0);
        assert_eq!(&r.idx[..], &[0, 1]);
    }
}

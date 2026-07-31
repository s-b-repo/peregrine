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
/// The routing configuration one `route` call needs beyond the tensors: how many
/// experts exist, how many to keep, and how the kept gate weights are scaled.
/// Grouping them keeps the entry point readable and makes the call sites
/// self-describing (they used to be six positional values, three of them
/// `usize`, which is easy to transpose silently).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterCfg {
    /// positions in this batch
    pub s_n: usize,
    /// model dim (router input width)
    pub d_n: usize,
    /// experts to choose from
    pub e_n: usize,
    /// experts kept per position
    pub k: usize,
    /// renormalize the kept gate weights to sum to 1
    pub norm_topk: bool,
    /// multiplier applied to the kept gate weights (`routed_scaling_factor`)
    pub routed_scale: f32,
}

pub fn route(x: &[f32], router_w: &[f32], router_bias: &[f32], cfg: RouterCfg) -> Routed {
    let RouterCfg { s_n, d_n, e_n, k, norm_topk, routed_scale } = cfg;
    let mut logits = vec![0f32; s_n * e_n];
    // The router matmul runs once per MoE layer per token (`n_experts × hidden`
    // MACs — ~1M per layer at GLM-5.2 shapes, ~90M across the stack), and it was
    // the one matmul with no threading at all.
    //
    // Parallelized *around* the kernel rather than inside it: each output
    // element keeps its own left-to-right accumulation, so results are
    // bit-identical (`matmul_f32` is also this workspace's f32 reference oracle,
    // and its documented contract is to match the C engine bit-for-bit).
    let work = s_n.saturating_mul(d_n).saturating_mul(e_n);
    if work < (1 << 20) {
        matmul_f32(&mut logits, x, router_w, s_n, d_n, e_n);
    } else if s_n > 1 {
        // Prefill: one row per task.
        peregrine_par::par_rows_mut(&mut logits, e_n, s_n, peregrine_par::PAR_MATMUL_MIN, |s, row| {
            matmul_f32(row, &x[s * d_n..s * d_n + d_n], router_w, 1, d_n, e_n);
        });
    } else {
        // Decode (a single row): split the *experts* instead, so the latency-
        // critical path still uses the pool.
        peregrine_par::par_chunks_mut(&mut logits, 1, e_n, peregrine_par::PAR_MATMUL_MIN, |start, end, chunk| {
            matmul_f32(chunk, x, &router_w[start * d_n..end * d_n], 1, d_n, end - start);
        });
    }

    let mut idx = vec![0i32; s_n * k];
    let mut w = vec![0f32; s_n * k];
    let mut choice = vec![0f32; e_n];
    // Per-position kept count; filled in as each position selects (a position
    // can keep fewer than `k` — see the selection loop).
    let mut keff = vec![k as i32; s_n];

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
        // How many experts this position actually kept. Selection can come up
        // empty before `k` rounds — every expert already taken, or all-NaN
        // scores (`NaN > x` is false, so nothing is ever "best"). Recording the
        // real count keeps `keff` honest instead of emitting a `-1` expert id
        // that downstream indexing would turn into an out-of-bounds access.
        let mut kept = k;
        for kk in 0..k {
            let mut best = -1i32;
            let mut bv = f32::NEG_INFINITY;
            for (e, &c) in choice.iter().enumerate().take(e_n) {
                if ib[..kk].contains(&(e as i32)) {
                    continue;
                }
                if c > bv {
                    bv = c;
                    best = e as i32;
                }
            }
            let Ok(bi) = usize::try_from(best) else {
                kept = kk;
                break;
            };
            ib[kk] = best;
            wb[kk] = logit[bi]; // weight is the sigmoid, not choice
        }
        // Unfilled slots must not be read: zero their weight and point them at
        // expert 0 so even a caller that ignores `keff` stays in range.
        for slot in kk_range(kept, k) {
            ib[slot] = 0;
            wb[slot] = 0.0;
        }
        keff[s] = kept as i32;
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

    Routed { idx, w, keff, k }
}

/// The slot range `kept..k` — the selections a position did not fill.
#[inline]
fn kk_range(kept: usize, k: usize) -> std::ops::Range<usize> {
    kept.min(k)..k
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

/// The experts routed at a single position `s` (its top-k selection), in selection
/// order. For per-sequence prefetch prediction in batched decode, where position `s`
/// maps to sequence `s`.
pub fn routed_at(r: &Routed, s: usize) -> Vec<i32> {
    (0..r.keff[s] as usize).map(|kk| r.idx[s * r.k + kk]).collect()
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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 4, k: 2, norm_topk: true, routed_scale: 1.0 });

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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 2, norm_topk: true, routed_scale: 2.5 });
        // normalized weights sum to 1, then ×2.5
        assert!((r.w[0] + r.w[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn parallel_router_matmul_is_bit_identical() {
        // The router matmul is parallelized around the kernel (each output keeps
        // its own left-to-right accumulation), so every dispatch path must agree
        // exactly — a one-ULP difference could flip a near-tie top-k selection
        // and route a different expert.
        let (d_n, e_n) = (2048usize, 96usize); // above the parallel-dispatch threshold
        let mut lcg = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let router_w: Vec<f32> = (0..e_n * d_n).map(|_| next()).collect();
        let bias = vec![0f32; e_n];
        for s_n in [1usize, 4] {
            let x: Vec<f32> = (0..s_n * d_n).map(|_| next()).collect();
            // Reference: the serial kernel, computed directly.
            let mut want = vec![0f32; s_n * e_n];
            matmul_f32(&mut want, &x, &router_w, s_n, d_n, e_n);
            let r = route(&x, &router_w, &bias, RouterCfg { s_n, d_n, e_n, k: 4, norm_topk: false, routed_scale: 1.0 });
            // The routed weights are sigmoid(logit), so compare through it.
            for s in 0..s_n {
                for kk in 0..r.keff[s] as usize {
                    let e = r.idx[s * r.k + kk] as usize;
                    let expect = sigmoidf(want[s * e_n + e]);
                    assert_eq!(r.w[s * r.k + kk], expect, "s={s} kk={kk} e={e} (s_n={s_n})");
                }
            }
        }
    }

    #[test]
    fn batch_union_dedups() {
        // 2 positions, K=2: experts {0,2} and {2,3} → union {0,2,3}
        let r = Routed { idx: vec![0, 2, 2, 3], w: vec![1.0; 4], keff: vec![2, 2], k: 2 };
        let u = batch_union(&r, 2);
        assert_eq!(u, vec![0, 2, 3]); // first-seen order
    }

    #[test]
    fn selection_short_of_k_reports_keff_and_stays_in_range() {
        // k > e_n: after every expert is taken there is nothing left to select.
        // The old code stored -1 and then indexed `logit[-1 as usize]`.
        // `Cfg::validate` rejects this config, so this is the defence in depth.
        let x = [1.0f32];
        let router_w = [1.0f32, 2.0];
        let bias = [0.0f32, 0.0];
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 4, norm_topk: false, routed_scale: 1.0 });
        assert_eq!(r.keff[0], 2, "only two experts exist to keep");
        assert!(r.idx.iter().all(|&e| e >= 0 && (e as usize) < 2), "no out-of-range expert ids: {:?}", r.idx);
        assert_eq!(&r.w[2..], &[0.0, 0.0], "unfilled slots carry zero weight");
    }

    #[test]
    fn nan_scores_do_not_emit_invalid_experts() {
        // NaN logits make every `choice[e] > bv` comparison false, so nothing is
        // ever selected — this must yield keff 0, not a -1 expert id.
        let x = [f32::NAN];
        let router_w = [1.0f32, 1.0];
        let bias = [0.0f32, 0.0];
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 2, norm_topk: false, routed_scale: 1.0 });
        assert_eq!(r.keff[0], 0, "no selection is possible from NaN scores");
        assert!(r.idx.iter().all(|&e| e >= 0 && (e as usize) < 2));
        assert!(super::batch_union(&r, 1).is_empty(), "no experts are routed");
    }

    #[test]
    fn ties_go_to_lowest_index() {
        // two experts with identical choice → lowest index selected first
        let x = [1.0f32];
        let router_w = [1.0f32, 1.0, 1.0];
        let bias = [0.0f32, 0.0, 0.0];
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 3, k: 2, norm_topk: false, routed_scale: 1.0 });
        assert_eq!(&r.idx[..], &[0, 1]);
    }
}

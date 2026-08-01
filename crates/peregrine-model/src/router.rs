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
    /// Drop trailing selections carrying less than this share of the position's
    /// gate mass (`0.0` = keep all `k`, the historical behaviour). See
    /// [`negligible_tail`].
    pub min_share: f32,
}

pub fn route(x: &[f32], router_w: &[f32], router_bias: &[f32], cfg: RouterCfg) -> Routed {
    let RouterCfg { s_n, d_n, e_n, k, norm_topk, routed_scale, min_share } = cfg;
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
        // Drop the negligible tail before normalization, so the `norm_topk`
        // block below renormalizes over the survivors and the MoE sum keeps its
        // original scale instead of quietly shrinking by the dropped mass.
        let drop = negligible_tail(wb, kept, min_share);
        for slot in (kept - drop)..kept {
            ib[slot] = 0;
            wb[slot] = 0.0;
        }
        kept -= drop;
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

    // Diagnostics only, and only when asked: how much of the routed set carries a
    // negligible share of the gate mass. Every routed expert costs a full weight
    // read whatever its weight, so this is the number that says whether truncating
    // the set is worth anything on this workload.
    if gate_stats_enabled() {
        use std::sync::atomic::Ordering;
        let mut below = [0u64; 4];
        for s in 0..s_n {
            let kept = (keff[s].max(0) as usize).min(k);
            gate_share_below(&w[s * k..s * k + k], kept, k, &GATE_THRESHOLDS, &mut below);
            GATE_TOTAL.fetch_add(kept as u64, Ordering::Relaxed);
        }
        for (c, n) in GATE_BELOW.iter().zip(below) {
            c.fetch_add(n, Ordering::Relaxed);
        }
    }

    Routed { idx, w, keff, k }
}

/// How many *trailing* selections to drop because each carries less than
/// `min_share` of the position's gate mass. Returns `0` when the feature is off
/// (`min_share <= 0`), the row is degenerate, or nothing qualifies.
///
/// Every routed expert costs a full weight read — ~18.9 MB at int4, and 600 of
/// them per token — regardless of how much it contributes. An expert carrying 1%
/// of the gate mass costs exactly what the top expert costs and contributes 1%.
///
/// Only a trailing run is dropped, and only from the end of the *selection*
/// order. Selection ranks by the bias-augmented `choice` while the stored weight
/// is the plain sigmoid, so the weights are not guaranteed monotonic — scanning
/// from the end is the conservative reading of "the tail", and it keeps the
/// surviving slots in their original order, which the batch-union and the
/// position-keyed reduce both depend on.
///
/// At least one expert always survives: a position with no expert would leave the
/// token with only the shared expert's contribution.
///
/// Pure — unit-testable without a model.
pub fn negligible_tail(w: &[f32], kept: usize, min_share: f32) -> usize {
    if min_share <= 0.0 || kept <= 1 {
        return 0;
    }
    let kept = kept.min(w.len());
    let sum: f32 = w[..kept].iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return 0; // degenerate row — leave the selection alone
    }
    let mut drop = 0usize;
    while drop < kept - 1 {
        let i = kept - 1 - drop;
        if w[i] / sum >= min_share {
            break;
        }
        drop += 1;
    }
    drop
}

/// Minimum gate share a routed expert must carry to be worth its weight read
/// (`COLI_ROUTE_MIN_SHARE`). `0`/unset disables truncation.
///
/// **This is the one knob in the engine that changes token values.** Every other
/// adaptive knob may only move latency or residency; this one drops a real (if
/// small) term from the MoE sum. It is off by default for that reason, and
/// `COLI_GATE_STATS` is how to size it before turning it on.
pub fn route_min_share() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COLI_ROUTE_MIN_SHARE")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|f| f.is_finite() && (0.0..1.0).contains(f))
            .unwrap_or(0.0)
    })
}

/// Gate-share thresholds the accumulator reports on.
pub const GATE_THRESHOLDS: [f32; 4] = [0.005, 0.01, 0.02, 0.05];

/// Process-global gate-share tallies: `[below 0.5%, below 1%, below 2%, below 5%]`
/// plus the total routed count. Monotonic, lock-free, diagnostic only — nothing
/// reads them on the forward path. Mirrors the CUDA backend's `g_group_*`
/// counters rather than threading a reference through `ForwardCtx`, because
/// `route` is the one chokepoint both MoE paths share.
static GATE_BELOW: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static GATE_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether to tally gate shares (`COLI_GATE_STATS=1`). Off by default: the tally
/// is a handful of adds per position per layer, but it is pure diagnostics.
fn gate_stats_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("COLI_GATE_STATS").as_deref(), Ok("1") | Ok("true")))
}

/// `(below-threshold counts, total routed)` accumulated since process start.
/// `None` when `COLI_GATE_STATS` is off, so a caller cannot mistake "not measured"
/// for "no tail".
pub fn gate_stats_snapshot() -> Option<([u64; 4], u64)> {
    if !gate_stats_enabled() {
        return None;
    }
    use std::sync::atomic::Ordering;
    let total = GATE_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let mut below = [0u64; 4];
    for (i, c) in GATE_BELOW.iter().enumerate() {
        below[i] = c.load(Ordering::Relaxed);
    }
    Some((below, total))
}

/// Count, per threshold, how many of one position's kept experts carry a gate
/// share below it. `out[t]` is incremented for threshold `thresholds[t]`.
///
/// This is the measurement that decides whether truncating the routed set is
/// worth anything. Each routed expert costs a full weight read (~18.9 MB at int4,
/// 600 reads ≈ 11.3 GB per token) regardless of how much it contributes, and
/// nothing in the engine has ever looked at the magnitude of a gate weight — the
/// reduce multiplies it in and moves on. If the 7th and 8th experts routinely
/// carry ~1% of the mass, the same fraction of the disk budget is buying ~1% of
/// the output.
///
/// Shares are computed relative to the position's own kept sum, so the result is
/// independent of `norm_topk` and `routed_scale` (which scale every weight
/// alike). Positions with no kept experts, or a non-positive sum, contribute
/// nothing rather than dividing by zero.
///
/// Pure and allocation-free — safe on the forward path, unit-testable without a
/// model.
pub fn gate_share_below(w: &[f32], keff: usize, k: usize, thresholds: &[f32], out: &mut [u64]) {
    let kept = keff.min(k).min(w.len());
    if kept == 0 {
        return;
    }
    let sum: f32 = w[..kept].iter().sum();
    // NaN must be excluded explicitly: `sum <= 0.0` is false for NaN, so a naive
    // positivity check would let it through and divide every share into NaN.
    if !sum.is_finite() || sum <= 0.0 {
        return; // all-zero or NaN weights carry no information about the tail
    }
    for &wi in &w[..kept] {
        let share = wi / sum;
        for (t, &thr) in thresholds.iter().enumerate() {
            if share < thr {
                if let Some(c) = out.get_mut(t) {
                    *c = c.saturating_add(1);
                }
            }
        }
    }
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
    fn min_share_zero_keeps_every_selection() {
        // Off is the default and must be exactly the historical behaviour.
        let w = vec![0.9f32, 0.05, 0.03, 0.02];
        assert_eq!(negligible_tail(&w, 4, 0.0), 0);
        assert_eq!(negligible_tail(&w, 4, -1.0), 0);
    }

    #[test]
    fn negligible_tail_drops_only_the_trailing_run() {
        // Shares: .9/.05/.03/.02 — the last three are each under 6%.
        let w = vec![0.9f32, 0.05, 0.03, 0.02];
        assert_eq!(negligible_tail(&w, 4, 0.06), 3);
        assert_eq!(negligible_tail(&w, 4, 0.04), 2, "only .03 and .02 are under 4%");
        assert_eq!(negligible_tail(&w, 4, 0.01), 0, "nothing is under 1%");
        // A big weight *after* a small one stops the scan: only a trailing run
        // goes, because selection order is not weight order (bias-augmented
        // `choice` picks, the plain sigmoid is stored).
        let interleaved = vec![0.5f32, 0.01, 0.48, 0.01];
        assert_eq!(negligible_tail(&interleaved, 4, 0.05), 1, "the 0.48 halts the scan");
    }

    #[test]
    fn negligible_tail_always_leaves_one_expert() {
        // Even an all-tiny row keeps something: a position with no routed expert
        // would fall back to the shared expert's contribution alone.
        let w = vec![0.25f32, 0.25, 0.25, 0.25];
        assert_eq!(negligible_tail(&w, 4, 0.99), 3, "at most k-1 dropped");
        assert_eq!(negligible_tail(&[1.0], 1, 0.99), 0, "a single expert is never dropped");
        // Degenerate rows are left alone rather than divided by zero.
        assert_eq!(negligible_tail(&[0.0, 0.0], 2, 0.5), 0);
        assert_eq!(negligible_tail(&[f32::NAN, 1.0], 2, 0.5), 0);
    }

    #[test]
    fn route_truncates_and_renormalizes_to_the_original_scale() {
        // End-to-end through `route`: one dominant expert and one negligible one.
        // The negligible expert must be dropped and the survivor must still carry
        // the full `routed_scale`, so the MoE sum does not silently shrink by the
        // dropped mass.
        let x = vec![1.0f32];
        // Two experts with very different logits → very different sigmoids.
        let router_w = vec![6.0f32, -6.0];
        let bias = vec![0.0f32, 0.0];
        let cfg = |ms: f32| RouterCfg {
            s_n: 1,
            d_n: 1,
            e_n: 2,
            k: 2,
            norm_topk: true,
            routed_scale: 2.5,
            min_share: ms,
        };
        let off = route(&x, &router_w, &bias, cfg(0.0));
        assert_eq!(off.keff[0], 2, "disabled keeps both");
        let on = route(&x, &router_w, &bias, cfg(0.05));
        assert_eq!(on.keff[0], 1, "the negligible expert is dropped");
        assert_eq!(on.idx[0], off.idx[0], "the survivor is still the top expert");
        // Renormalized: the kept weights sum to `routed_scale` in both cases.
        let sum_on: f32 = on.w[..on.keff[0] as usize].iter().sum();
        let sum_off: f32 = off.w[..off.keff[0] as usize].iter().sum();
        assert!((sum_on - 2.5).abs() < 1e-5, "kept mass must still be routed_scale, got {sum_on}");
        assert!((sum_off - 2.5).abs() < 1e-5, "baseline sums to routed_scale, got {sum_off}");
    }

    #[test]
    fn gate_share_counts_the_negligible_tail() {
        // A peaked position: two experts carry 90% of the mass, six split 10%.
        let w = vec![0.45f32, 0.45, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01];
        let thr = [0.05f32, 0.03, 0.015];
        let mut out = [0u64; 3];
        gate_share_below(&w, 8, 8, &thr, &mut out);
        assert_eq!(out[0], 6, "six experts below a 5% share");
        assert_eq!(out[1], 6, "six below 3%");
        assert_eq!(out[2], 2, "two below 1.5%");
    }

    #[test]
    fn gate_share_is_invariant_to_uniform_scaling() {
        // `norm_topk` and `routed_scale` multiply every weight alike, so the
        // measurement must not depend on whether they ran.
        let base = vec![0.5f32, 0.3, 0.15, 0.05];
        let thr = [0.1f32];
        let mut a = [0u64; 1];
        gate_share_below(&base, 4, 4, &thr, &mut a);
        let scaled: Vec<f32> = base.iter().map(|v| v * 2.5).collect();
        let mut b = [0u64; 1];
        gate_share_below(&scaled, 4, 4, &thr, &mut b);
        assert_eq!(a, b, "scaling every weight must not change the shares");
        assert_eq!(a[0], 1, "only the 0.05 share is below 10%");
    }

    #[test]
    fn gate_share_ignores_unfilled_slots_and_degenerate_rows() {
        // Only `keff` entries are real; the rest are zeroed padding and must not
        // be counted as a negligible tail.
        let w = vec![0.6f32, 0.4, 0.0, 0.0];
        let thr = [0.1f32];
        let mut out = [0u64; 1];
        gate_share_below(&w, 2, 4, &thr, &mut out);
        assert_eq!(out[0], 0, "padding slots are not routed experts");
        // All-zero and empty rows contribute nothing rather than dividing by zero.
        let mut z = [0u64; 1];
        gate_share_below(&[0.0, 0.0], 2, 2, &thr, &mut z);
        assert_eq!(z[0], 0);
        let mut e = [0u64; 1];
        gate_share_below(&[], 0, 0, &thr, &mut e);
        assert_eq!(e[0], 0);
    }

    #[test]
    fn gate_share_of_a_flat_router_finds_no_tail() {
        // Maximum routing entropy: eight equal experts, each a 12.5% share, so no
        // threshold below that fires. This is the case where truncation buys
        // nothing — the measurement has to be able to say so.
        let w = vec![1.0f32; 8];
        let thr = [0.05f32, 0.12];
        let mut out = [0u64; 2];
        gate_share_below(&w, 8, 8, &thr, &mut out);
        assert_eq!(out, [0, 0], "a uniform router has no negligible tail");
    }

    #[test]
    fn selects_by_bias_weights_by_sigmoid() {
        // E=4, D=1, x=[1]. router rows = raw logits [2,-1,0.5,3]; bias pushes
        // expert 3 below expert 2 for *selection*, but weights stay sigmoid.
        let x = [1.0f32];
        let router_w = [2.0f32, -1.0, 0.5, 3.0]; // [E=4, D=1]
        let bias = [0.0f32, 0.0, 0.0, -0.5];
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 4, k: 2, norm_topk: true, routed_scale: 1.0, min_share: 0.0 });

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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 2, norm_topk: true, routed_scale: 2.5, min_share: 0.0 });
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
            let r = route(&x, &router_w, &bias, RouterCfg { s_n, d_n, e_n, k: 4, norm_topk: false, routed_scale: 1.0, min_share: 0.0 });
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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 4, norm_topk: false, routed_scale: 1.0, min_share: 0.0 });
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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 2, k: 2, norm_topk: false, routed_scale: 1.0, min_share: 0.0 });
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
        let r = route(&x, &router_w, &bias, RouterCfg { s_n: 1, d_n: 1, e_n: 3, k: 2, norm_topk: false, routed_scale: 1.0, min_share: 0.0 });
        assert_eq!(&r.idx[..], &[0, 1]);
    }
}

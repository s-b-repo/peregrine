//! The router group: what the MoE layer is allowed to assume about the decision
//! it is handed.
//!
//! Routing is the one place where a wrong answer cannot be detected downstream.
//! Everything after it — the expert reads, the MoE sum, the reduce — faithfully
//! computes whatever it was given. So the properties here are about the
//! *decision*: that the gates are a distribution, that selection follows the
//! documented ordering rule (bias augments the choice, never the stored weight),
//! and that the controls which shrink the expert set can only shrink it.

use peregrine_model::router::{negligible_tail, route, route_ranks, RouterCfg};

use crate::{cert, rng, with_cases, Certificate, EPS_F32};

/// One router configuration used across the group. `norm_topk` on, no scaling:
/// the shape the production MoE layer uses at GLM-class widths.
fn cfg(s_n: usize, d_n: usize, e_n: usize, k: usize) -> RouterCfg {
    RouterCfg { s_n, d_n, e_n, k, norm_topk: true, routed_scale: 1.0, min_share: 0.0 }
}

pub fn certificates() -> Vec<Certificate> {
    vec![
        gates_are_a_distribution(),
        selection_matches_route_ranks(),
        bias_selects_but_does_not_weight(),
        keff_is_bounded_by_k(),
        negligible_tail_never_drops_a_leader(),
        expert_budget_only_shrinks(),
    ]
}

/// Renormalized top-k gates must be a distribution, or the layer rescales its own
/// output as a side effect of the routing decision — every token would carry a
/// different effective temperature. The tolerance is `k` ulps of the partial sums
/// that built it.
fn gates_are_a_distribution() -> Certificate {
    let mut r = rng();
    let mut worst = 0.0f64;
    let mut cases = 0usize;
    for (s_n, d_n, e_n, k) in [(1usize, 32usize, 8usize, 2usize), (4, 64, 16, 4), (8, 128, 32, 8)] {
        let x = r.f32_vec(s_n * d_n);
        let w = r.f32_vec(e_n * d_n);
        let b = r.f32_vec(e_n);
        let out = route(&x, &w, &b, cfg(s_n, d_n, e_n, k));
        for s in 0..s_n {
            let keep = out.keff[s].max(1) as usize;
            let sum: f64 = out.w[s * k..s * k + keep].iter().map(|v| f64::from(*v)).sum();
            worst = worst.max((sum - 1.0).abs());
            cases += 1;
        }
    }
    with_cases(
        cert(
            "router",
            "renormalized_gates_are_a_distribution",
            "With `norm_topk`, the surviving gates of every position sum to 1 within `k` f32 ulps, so routing cannot rescale the layer's output.",
            worst,
            8.0 * EPS_F32,
        ),
        cases,
    )
}

/// Two production functions compute the selection order — `route` for the layer,
/// `route_ranks` for the look-ahead's width profile — and the second is only
/// meaningful if it agrees with the first. They take different code paths (batch
/// versus single row, and a different fan-out threshold), so this is a real
/// comparison rather than a tautology.
fn selection_matches_route_ranks() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for (s_n, d_n, e_n, k) in [(1usize, 64usize, 8usize, 3usize), (4, 96, 16, 4), (2, 128, 64, 2)] {
        let x = r.f32_vec(s_n * d_n);
        let w = r.f32_vec(e_n * d_n);
        let b = r.f32_vec(e_n);
        let out = route(&x, &w, &b, cfg(s_n, d_n, e_n, k));
        for s in 0..s_n {
            let expect = route_ranks(&x[s * d_n..s * d_n + d_n], &w, &b, d_n, e_n, k);
            let got = &out.idx[s * k..s * k + k];
            if expect.len() != k || got.iter().zip(expect.iter()).any(|(a, b)| a != b) {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "router",
            "route_selection_matches_route_ranks",
            "For every position, `route`'s selected expert ids equal `route_ranks`' for the same row: the look-ahead's width profile describes the decision the layer actually makes.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The subtlety carried over from the C: the bias augments the *choice* used to
/// rank experts, but the stored gate weight is the plain sigmoid of the logit.
/// If the bias leaked into the weight, every biased expert would receive its
/// bias twice — once in who gets selected, once in how much it contributes.
///
/// The fixture makes both halves deterministic: with the bias pushing expert 0
/// out of contention and then pulling it back in, the selection is guaranteed to
/// change. The weights are then read per *expert*, not per position, because
/// `norm_topk` renormalizes over whatever survived — so the comparison is
/// "expert 4 carries the same gate weight in both runs", which is the claim.
fn bias_selects_but_does_not_weight() -> Certificate {
    let mut r = rng();
    let (d_n, e_n, k) = (64usize, 8usize, 2usize);
    let x = r.f32_vec(d_n);
    let w = r.f32_vec(e_n * d_n);
    let mut b_out = vec![0.0f32; e_n];
    b_out[0] = -100.0;
    let mut b_in = vec![0.0f32; e_n];
    b_in[0] = 100.0;
    // `norm_topk` off: the stored weight is the plain sigmoid, unrenormalized, so
    // the same expert's weight is comparable across two different selections.
    let plain = RouterCfg { norm_topk: false, ..cfg(1, d_n, e_n, k) };
    let a = route(&x, &w, &b_out, plain);
    let b = route(&x, &w, &b_in, plain);
    let selection_moved = a.idx != b.idx;
    // Compare the weight each run assigns to each expert that both selected.
    let shared: Vec<(f32, f32)> = a
        .idx
        .iter()
        .zip(a.w.iter())
        .filter_map(|(e, wa)| {
            let wb = b.idx.iter().position(|other| other == e).map(|j| b.w[j])?;
            Some((*wa, wb))
        })
        .collect();
    let weights_agree = !shared.is_empty() && shared.iter().all(|(x, y)| x.to_bits() == y.to_bits());
    with_cases(
        cert(
            "router",
            "the_router_bias_selects_without_weighting",
            "Moving one expert's router bias from -100 to +100 changes which experts are selected, and every expert selected in both runs carries a bit-identical gate weight: a biased expert is not also given its bias in the MoE sum.",
            f64::from(u8::from(!selection_moved)) + f64::from(u8::from(!weights_agree)),
            0.0,
        ),
        shared.len() + 1,
    )
}

/// `keff` is read by the batch union and the reduce as the authoritative count of
/// survivors, so it can never exceed `k` (which would index past the row) and can
/// never be zero (which would leave the token with no expert at all).
fn keff_is_bounded_by_k() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for min_share in [0.0f32, 0.01, 0.1, 0.49] {
        let (s_n, d_n, e_n, k) = (4usize, 64usize, 16usize, 4usize);
        let x = r.f32_vec(s_n * d_n);
        let w = r.f32_vec(e_n * d_n);
        let b = r.f32_vec(e_n);
        let c = RouterCfg { min_share, ..cfg(s_n, d_n, e_n, k) };
        let out = route(&x, &w, &b, c);
        for s in 0..s_n {
            let keep = out.keff[s];
            if keep < 1 || keep > k as i32 {
                bad += 1.0;
            }
            cases += 1;
        }
    }
    with_cases(
        cert(
            "router",
            "keff_is_never_zero_and_never_exceeds_k",
            "At every `min_share` from off to aggressive, each position reports between 1 and `k` surviving experts: at least one expert (a token with none would have no expert contribution) and never more than the row holds.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The truncation rule's own contract, in three parts: it never leaves fewer than
/// one expert, it drops nothing at all when the threshold is off, and what it
/// drops is a *trailing* run — scanning from the end of the selection order,
/// never from the middle.
///
/// That last part is deliberate and is why the claim is about the position of the
/// drop rather than its size: the stored weights are **not** monotonic in the
/// selection order, because selection ranks by the bias-augmented choice while
/// the weight is the plain sigmoid. A rule that dropped any small weight it found
/// would remove an expert from the middle of the row, and both the position-keyed
/// reduce and the batch union depend on the survivors keeping their original
/// order. So the certificate checks that the survivors are a *prefix* of the row:
/// re-running the same rule over the survivors changes nothing, while the dropped
/// suffix is exactly what the threshold rejected.
fn negligible_tail_never_drops_a_leader() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for kept in [1usize, 2, 5, 8] {
        for min_share in [0.0f32, 0.01, 0.2, 0.9] {
            for _ in 0..8 {
                let w: Vec<f32> = (0..kept).map(|_| r.sym().abs() + 1e-3).collect();
                let drop = negligible_tail(&w, kept, min_share);
                // Never fewer than one survivor.
                if drop + 1 > kept {
                    bad += 1.0;
                }
                if min_share == 0.0 && drop != 0 {
                    bad += 1.0;
                }
                // The survivors are a prefix: re-running the rule over them drops
                // nothing more, so the drop really was a suffix.
                if negligible_tail(&w[..kept - drop], kept - drop, min_share) != 0 {
                    bad += 1.0;
                }
                // Every dropped element was below the row's share threshold —
                // the definition of the tail — and the rule stopped at the first
                // element that was not. The one exception the contract allows:
                // when a single expert is left it may itself be below the
                // threshold, because "at least one expert always survives" wins.
                if drop > 0 {
                    let total: f32 = w[..kept].iter().sum();
                    let survivors = &w[..kept - drop];
                    if w[kept - drop..kept].iter().any(|v| *v / total >= min_share) {
                        bad += 1.0;
                    }
                    if survivors.len() > 1 && survivors.iter().all(|v| *v / total < min_share) {
                        // Nothing above the threshold survived, so the rule was not
                        // maximal: it would have kept dropping.
                        bad += 1.0;
                    }
                }
                cases += 1;
            }
        }
    }
    with_cases(
        cert(
            "router",
            "min_share_drops_only_a_trailing_tail",
            "`negligible_tail` never leaves fewer than one expert, drops nothing when the threshold is off, drops only elements below the row's share threshold, and stops at the first survivor — so the survivors are a prefix of the selection, except that a lone survivor may itself be below the threshold.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The expert budget is the other control that shrinks the routed set, and its
/// contract is one-directional in three parts: a budget of 0 must be a no-op,
/// no position may keep *more* experts than routing kept, and the slots behind a
/// position's `keff` must be zeroed — a caller that ignored `keff` and read a
/// stale weight would otherwise contribute an expert the budget had just
/// removed.
///
/// It makes no promise that every position keeps an expert: a position whose
/// experts were all outside the budget's top set does end up with `keff == 0`,
/// which is why the claim is stated as "the union is bounded", not "nothing is
/// emptied".
fn expert_budget_only_shrinks() -> Certificate {
    let mut r = rng();
    let (s_n, d_n, e_n, k) = (4usize, 32usize, 8usize, 4usize);
    let x = r.f32_vec(s_n * d_n);
    let w = r.f32_vec(e_n * d_n);
    let b = r.f32_vec(e_n);

    let mut off = route(&x, &w, &b, cfg(s_n, d_n, e_n, k));
    let before: Vec<i32> = off.keff.clone();
    let weights_before = off.w.clone();
    let dropped_off = peregrine_model::router::apply_expert_budget(&mut off, s_n, 0, true, 1.0);
    let no_op = dropped_off == 0 && off.keff == before && off.w == weights_before;

    let mut bad = f64::from(u8::from(!no_op));
    let mut cases = 1usize;
    for budget in [1usize, 2, 4] {
        let mut tight = route(&x, &w, &b, cfg(s_n, d_n, e_n, k));
        let start: Vec<i32> = tight.keff.clone();
        peregrine_model::router::apply_expert_budget(&mut tight, s_n, budget, true, 1.0);
        let union = peregrine_model::router::batch_union(&tight, s_n).len();
        cases += 1;
        if union > budget {
            bad += 1.0;
        }
        for (s, before) in start.iter().enumerate() {
            cases += 1;
            // One-directional: a budget may only remove experts.
            if tight.keff[s] > *before {
                bad += 1.0;
            }
            // Nothing stale behind keff: every dropped slot reads as weight zero.
            let keep = tight.keff[s].max(0) as usize;
            if tight.w[s * k + keep..s * k + k].iter().any(|v| *v != 0.0) {
                bad += 1.0;
            }
        }
    }
    with_cases(
        cert(
            "router",
            "the_expert_budget_only_shrinks_the_route",
            "A budget of 0 is a no-op; a budget of N leaves the batch union at or below N distinct experts, never grows a position's expert count, and zeroes every slot behind `keff` so no stale weight survives.",
            bad,
            0.0,
        ),
        cases,
    )
}

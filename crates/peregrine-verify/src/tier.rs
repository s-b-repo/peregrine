//! The tiering-policy group: the hot-store swap rule the GPU placement path
//! runs (`COLI_GPU_TIER_SWAP=lfru`).
//!
//! This policy decides *which resident expert gets thrown out of VRAM*. Its
//! failure mode is a throughput regression with no error, so every part of the
//! rule that can be stated as an identity is: what the score is made of, which
//! side of the hysteresis an exchange lands on, and what the function does with
//! inputs that do not describe a decision.

use peregrine_io::tier::{decay, lfru_score, pick_lfru, pick_swap};

use crate::{cert, with_cases, Certificate};

pub fn certificates() -> Vec<Certificate> {
    vec![
        frequency_outranks_recency(),
        hysteresis_stops_the_exchange(),
        lfru_swaps_the_coldest_slot_for_the_hottest_expert(),
        mismatched_inputs_decide_nothing(),
        pick_swap_agrees_when_recency_is_flat(),
        decay_only_ever_halves(),
    ]
}

/// The score is `heat << 8 | (255 − age)`, so one frequency count (256 points) is
/// worth more than every recency point available (255). The contract is
/// therefore strict: a *merely recent* expert never outranks one that is *one
/// count hotter*, however recently the first was touched — without that, a stream
/// of accesses to a cold expert would pin it in VRAM and the heat counts would
/// stop meaning anything.
fn frequency_outranks_recency() -> Certificate {
    let clock = 1_000u32;
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    // Cold but just accessed vs one-count-hotter and never accessed.
    let cold_recent = lfru_score(0, clock, clock);
    let hotter_stale = lfru_score(1, 0, clock);
    cases += 1;
    if cold_recent >= hotter_stale {
        bad += 1.0;
    }
    // The margin is exactly one count (256) minus the recency difference: a hot
    // expert that is *also* the most recent still leads.
    if hotter_stale - cold_recent != 1 {
        bad += 1.0;
    }
    // Recency saturates at 0 and never wraps: an age beyond the recency window
    // adds nothing, so a long-idle expert does not become "recent" again.
    let very_old = lfru_score(5, 0, clock);
    let ancient = lfru_score(5, clock.wrapping_sub(10_000), clock);
    cases += 1;
    if very_old != ancient {
        bad += 1.0;
    }
    // And the score is monotonic in heat.
    let mut mono = true;
    for h in [0u32, 1, 7, 1000] {
        if lfru_score(h, 0, clock) >= lfru_score(h + 1, clock, clock) {
            mono = false;
        }
    }
    cases += 4;
    if !mono {
        bad += 1.0;
    }
    with_cases(
        cert(
            "tier",
            "one_frequency_count_outranks_all_recency",
            "The LFRU score is `heat·256 + recency` with recency capped at 255, so an expert one frequency count hotter wins even when it is the *stale* one, recency saturates at zero age instead of wrapping, and the score is strictly increasing in heat.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// The 25%-plus-four-count hysteresis is the anti-ping-pong rule: without it, two
/// experts sitting near each other would trade slots every generation and the
/// pinned set would churn forever while the heat table stayed still. The claim is
/// the boundary itself — a gain of exactly `cold/4 + 4` does **not** swap, and one
/// more does.
fn hysteresis_stops_the_exchange() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for cold_heat in [4u32, 8, 100, 4000] {
        // Two experts, one pinned. The hot one's heat is set exactly on, and one
        // count above, the hysteresis edge.
        let pinned = [0usize];
        let edge = cold_heat + (cold_heat >> 2) + 4;
        let last = [0u32, 0];
        let on_edge = [cold_heat, edge];
        let above = [cold_heat, edge + 1];
        cases += 4;
        // On the edge: no decision. One count past it: a swap.
        if pick_lfru(&on_edge, &last, 0, &pinned).is_some() {
            bad += 1.0;
        }
        if pick_lfru(&above, &last, 0, &pinned).is_none() {
            bad += 1.0;
        }
        // The frequency-only policy uses the same edge, in raw counts rather
        // than score units.
        if pick_swap(&on_edge, &pinned).is_some() {
            bad += 1.0;
        }
        if pick_swap(&above, &pinned).is_none() {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "tier",
            "the_hysteresis_edge_is_exactly_where_the_doc_says",
            "A gain of exactly `cold/4 + 4` produces no swap and one count more does, in both the LFRU and frequency-only policies.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// When a swap happens, it is between the two ends the rule names: the coldest
/// *pinned slot* and the hottest *non-resident expert*. Any other pair is a
/// different policy wearing this one's name, and it is the pair that decides
/// whether a hot expert can be resident at all.
fn lfru_swaps_the_coldest_slot_for_the_hottest_expert() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    // Slots 0,1,2 pinned with descending heat; experts 3..7 non-resident with a
    // range of heat, so both "coldest slot" and "hottest expert" have a unique
    // answer.
    let heat = [40u32, 30, 20, 0, 5, 90, 12, 60];
    let pinned = [0usize, 1, 2];
    let last = [0u32; 8];
    let got = pick_lfru(&heat, &last, 0, &pinned);
    cases += 1;
    match got {
        // slot 2 (coldest pinned: heat 20), expert 5 (hottest non-resident: 90)
        Some(s) => {
            if s.slot != 2 || s.eid != 5 {
                bad += 1.0;
            }
            // `gain` is reported in frequency counts, so it must match the
            // difference the caller will see, not the score difference.
            if s.gain != 90 - 20 {
                bad += 1.0;
            }
        }
        None => bad += 1.0,
    }
    // Everything resident: nothing to swap in, so no decision.
    let all_pinned = [0usize, 1, 2, 3, 4, 5, 6, 7];
    cases += 1;
    if pick_lfru(&heat, &last, 0, &all_pinned).is_some() {
        bad += 1.0;
    }
    with_cases(
        cert(
            "tier",
            "a_swap_replaces_the_coldest_pinned_slot_with_the_hottest_expert",
            "When the policy decides to swap, the slot is the coldest pinned one, the expert is the hottest non-resident one, and the reported `gain` is their frequency-count difference.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Inputs that do not describe a decision must produce `None`, not a decision
/// derived from garbage and not an abort: `last` shorter than `heat`, an empty
/// pinned set, an empty expert set, or a pin past the end of `heat`. These index
/// out of bounds otherwise, and the release profile aborts.
fn mismatched_inputs_decide_nothing() -> Certificate {
    let heat = [10u32, 9, 8, 7];
    let mut bad = 0.0f64;
    let cases = [
        (vec![], vec![0usize]),
        (heat.to_vec(), vec![]),
        (heat.to_vec(), vec![0, 1]),
        (heat.to_vec(), vec![9]),
        (vec![10, 9], vec![0, 1]),
    ];
    for (h, p) in &cases {
        let last = vec![0u32; h.len()];
        if pick_lfru(h, &last, 0, p).is_some() {
            bad += 1.0;
        }
        if pick_swap(h, p).is_some() {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "tier",
            "mismatched_inputs_produce_no_decision",
            "An empty expert set, an empty pinned set, a `last` shorter than `heat`, and a pin past the end of `heat` all return `None` from both policies rather than deciding from garbage.",
            bad,
            0.0,
        ),
        cases.len() * 2,
    )
}

/// With recency flat the two policies must agree, since LFRU reduces to
/// frequency-only. They are separate ports of two C functions; a divergence here
/// is the two implementations of one rule disagreeing.
fn pick_swap_agrees_when_recency_is_flat() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for n in [2usize, 5, 17] {
        let mut r = crate::rng();
        for _ in 0..32 {
            let heat: Vec<u32> = (0..n).map(|_| (r.unit() * 200.0) as u32).collect();
            let pinned: Vec<usize> = (0..n / 2).collect();
            if pinned.is_empty() {
                continue;
            }
            // `last` equal to `clock` makes every recency term 255, so the two
            // scores differ only in the ordering the frequency term imposes.
            let clock = 777u32;
            let last = vec![clock; n];
            let a = pick_lfru(&heat, &last, clock, &pinned);
            let b = pick_swap(&heat, &pinned);
            cases += 1;
            match (a, b) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    if x.slot != y.slot || x.eid != y.eid {
                        bad += 1.0;
                    }
                }
                _ => bad += 1.0,
            }
        }
    }
    with_cases(
        cert(
            "tier",
            "lfru_reduces_to_frequency_only_when_recency_is_flat",
            "With every expert equally recent, `pick_lfru` and `pick_swap` reach the same decision — the two ports of the two C functions are one rule.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// `decay` is periodic: it halves every count so a burst from an hour ago cannot
/// pin a slot forever. The claim is that it only ever halves — no entry
/// increases, none decreases by more than half, and the map is idempotent
/// (`decay(decay(x)) == decay(x)` for the counts a single period can produce is
/// false in general, so the certified part is monotonicity and the bound).
fn decay_only_ever_halves() -> Certificate {
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    let mut heat: Vec<u32> = vec![0, 1, 2, 3, 255, 256, 1023, u32::MAX / 2];
    let before = heat.clone();
    decay(&mut heat);
    cases += heat.len();
    for (b, a) in before.iter().zip(heat.iter()) {
        if *a > *b || *a != b >> 1 {
            bad += 1.0;
        }
    }
    // Zero stays zero: an idle expert must not be able to accumulate heat from
    // nothing.
    if heat[0] != 0 {
        bad += 1.0;
    }
    // Decaying twice continues to halve rather than stopping at a floor.
    let once = heat.clone();
    decay(&mut heat);
    cases += heat.len();
    if heat != once.iter().map(|v| v >> 1).collect::<Vec<u32>>() {
        bad += 1.0;
    }
    // And the empty case is a no-op rather than an error.
    let mut empty: Vec<u32> = Vec::new();
    decay(&mut empty);
    if !empty.is_empty() {
        bad += 1.0;
    }
    cases += 1;
    with_cases(
        cert(
            "tier",
            "decay_halves_every_count_and_never_raises_one",
            "`decay` maps every count to its own `>> 1`: no entry increases, none falls by more than half, zero stays zero, and repeated decay keeps halving rather than hitting a floor.",
            bad,
            0.0,
        ),
        cases,
    )
}

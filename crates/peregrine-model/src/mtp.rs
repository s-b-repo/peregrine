//! MTP speculative decoding — the acceptance rule (`c/glm.c:4060-4062`,
//! Leviathan rejection sampling). The draft head proposes a token; the main
//! model verifies it in the same batched forward. Accepting with probability
//! `min(1, p/q)` and, on rejection, resampling from the residual `(p-q)+`
//! makes the emitted distribution **exactly** the target `p` — speculation is
//! invisible to the output.
//!
//! Here we implement and validate that acceptance rule (the correctness core).
//! Wiring the int8 MTP head + batched verify into `Model` is the remaining M6
//! integration.

/// Speculative-sampling acceptance. `p` = target distribution, `q` = draft
/// distribution (both normalized over the vocab), `drafted` = the token drawn
/// from `q`. `u_accept`/`u_resample` are two uniforms in `[0,1)`. Returns the
/// emitted token — distributed exactly as `p`.
pub fn speculative_sample(p: &[f32], q: &[f32], drafted: usize, u_accept: f64, u_resample: f64) -> usize {
    let qd = q[drafted].max(1e-20) as f64;
    let accept_prob = (p[drafted] as f64 / qd).min(1.0);
    if u_accept < accept_prob {
        return drafted;
    }
    // rejected → sample from the residual (p - q)+, renormalized
    let mut resid: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| (pi - qi).max(0.0) as f64).collect();
    let mut tot: f64 = resid.iter().sum();
    if tot <= 1e-12 {
        // degenerate (q dominates p everywhere) — fall back to sampling p
        resid = p.iter().map(|&x| x as f64).collect();
        tot = resid.iter().sum();
    }
    let target = u_resample * tot;
    let mut cum = 0.0;
    for (i, &r) in resid.iter().enumerate() {
        cum += r;
        if cum >= target {
            return i;
        }
    }
    resid.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn u01(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 * (1.0 / 9007199254740992.0)
        }
    }

    fn sample_from(dist: &[f32], u: f64) -> usize {
        let mut cum = 0.0;
        for (i, &d) in dist.iter().enumerate() {
            cum += d as f64;
            if cum >= u {
                return i;
            }
        }
        dist.len() - 1
    }

    #[test]
    fn accepts_when_target_exceeds_draft() {
        // if p[d] >= q[d], the draft is always accepted (min(1, p/q) = 1)
        let p = [0.1f32, 0.6, 0.3];
        let q = [0.2f32, 0.3, 0.5];
        // token 1: p=0.6 > q=0.3 → accept for any u_accept
        assert_eq!(speculative_sample(&p, &q, 1, 0.999, 0.5), 1);
    }

    #[test]
    fn output_distribution_equals_target() {
        // The key losslessness property: regardless of the (wrong) draft
        // distribution q, the emitted tokens are distributed as p.
        let p = [0.4f32, 0.1, 0.2, 0.25, 0.05];
        let q = [0.1f32, 0.5, 0.1, 0.1, 0.2]; // deliberately mismatched
        let v = p.len();
        let mut rng = Lcg(0xBADC0DE);
        let n = 60_000;
        let mut hist = vec![0u32; v];
        for _ in 0..n {
            let drafted = sample_from(&q, rng.u01());
            let tok = speculative_sample(&p, &q, drafted, rng.u01(), rng.u01());
            hist[tok] += 1;
        }
        for i in 0..v {
            let freq = hist[i] as f64 / n as f64;
            assert!((freq - p[i] as f64).abs() < 0.02, "token {i}: freq {freq:.3} vs p {}", p[i]);
        }
    }
}

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// How many draft rounds pass between pin-set refreshes. Read by
/// [`MtpPins::note_round`].
///
/// The pin set is re-derived from a counter that only ever grows, so a refresh
/// is worth doing often enough to track a workload shift and rarely enough that
/// the cache-lock hold and the warm enqueue it performs are lost in the noise of
/// the draft steps around it. 64 rounds is ~64 accepted-or-rejected speculation
/// windows — hundreds of draft steps — which on the streaming container is tens
/// of gigabytes of expert reads. The cost of one refresh is a 256-slot snapshot,
/// a sort, and one short cache-lock hold.
pub const PIN_REFRESH_ROUNDS: u64 = 64;

/// Per-expert draft-routing frequency for the MTP head's **own** expert pool
/// (layer index `n_layers`), plus the round clock its pin set is refreshed on.
///
/// **Deliberately not the [`crate::gpu::HeatTable`]**, and not because a second
/// counter is tidier. Two reasons, either of which alone rules the heat table
/// out:
///
/// 1. `Model::heat` is `gpu.as_ref().map(|_| HeatTable::new(..))` — the table
///    exists only when a GPU tier does. The regime this mechanism is for is the
///    CPU-only streaming deployment, where one draft step is ~300 MB of SSD at
///    `s_n = 1` with no batch-union amortization. There, there is no heat table
///    to write to at all, which is why `COLI_MTP_HEAT` is documented as inert
///    without a GPU tier and why flipping it could never have closed this.
/// 2. Heat is a **shared** budget: it ranks main-stream experts for VRAM
///    residency and gates warm-cache admission. An MTP row competing in it means
///    main-stream experts losing residency out of the same bytes — the trade
///    `COLI_MTP_HEAT` deliberately exposes as a knob. A pin is the opposite
///    shape: a fixed set, separately budgeted, that the main stream's eviction
///    order cannot reach.
///
/// It is also one live row where the heat table is `n_layers + 1` rows wide.
///
/// Counts are monotonic, like the heat table's, and for the same reason: a set
/// that re-ranks on differences a counter cannot distinguish moves experts in
/// and out of residency for no information, and every move is a re-read or a
/// PCIe upload.
///
/// **Fed only when experts stream.** `forward_layer` reaches
/// `moe_forward_dispatch` — and therefore the bump site — only under
/// `stream_experts`; a resident model computes its MoE through `mlp::moe_forward`,
/// which never sees a `ForwardCtx` at all. That is the right shape rather than a
/// gap: a resident model already holds every expert in RAM, so a pin has nothing
/// left to buy it, and the counter staying at zero is what makes every plan off
/// it empty there.
pub struct MtpPins {
    /// The one layer this table describes — `cfg.n_layers`. Held so [`Self::bump`]
    /// can be wired into the shared MoE bump site and reject anything else,
    /// rather than trusting that the draft `ForwardCtx` will only ever run this
    /// layer (it only does today; nothing in the type system says so).
    layer: usize,
    counts: Vec<AtomicU32>,
    rounds: AtomicU64,
    /// Size of the last applied pin set, `usize::MAX` before the first one, so
    /// [`Self::note_applied`] reports the first application as a change.
    applied: AtomicUsize,
}

impl MtpPins {
    /// A zeroed table for `layer`'s `n_experts` experts.
    pub fn new(layer: usize, n_experts: usize) -> MtpPins {
        MtpPins {
            layer,
            counts: (0..n_experts).map(|_| AtomicU32::new(0)).collect(),
            rounds: AtomicU64::new(0),
            applied: AtomicUsize::new(usize::MAX),
        }
    }

    /// The layer this table describes.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Record one routing of `expert` in `layer` (lock-free). Routings of any
    /// other layer, and out-of-range expert ids, are ignored.
    pub fn bump(&self, layer: usize, expert: usize) {
        if layer != self.layer {
            return;
        }
        if let Some(c) = self.counts.get(expert) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A plain snapshot of the counts, indexed by expert id.
    pub fn snapshot(&self) -> Vec<u32> {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Advance the draft-round clock and report whether this round is a refresh
    /// generation (every [`PIN_REFRESH_ROUNDS`]th).
    ///
    /// Driven from the draft path rather than from `Model::reheat` because
    /// `reheat` is called from exactly one place — `peregrine-serve`'s batched
    /// engine — so a pin refresh riding it would leave the CLI speculative path
    /// permanently unpinned. That is the same "the mechanism exists and nothing
    /// calls it" shape this pin was added to fix.
    pub fn note_round(&self) -> bool {
        (self.rounds.fetch_add(1, Ordering::Relaxed) + 1).is_multiple_of(PIN_REFRESH_ROUNDS)
    }

    /// Record that a pin set of `n` experts was applied, and report whether that
    /// is a **change** from the last one.
    ///
    /// The caller uses this to decide whether to say anything. A pin set is
    /// re-derived every refresh generation and is usually identical to the last;
    /// a line per generation would bury the log, and no line at all is how a
    /// mechanism ends up impossible to tell apart from an inert one. Reporting
    /// only the transitions keeps both.
    pub fn note_applied(&self, n: usize) -> bool {
        self.applied.swap(n, Ordering::Relaxed) != n
    }

    /// Size of the last applied pin set — `0` before the first application, so a
    /// scrape cannot tell "not yet applied" from "applied, empty". Both mean the
    /// same thing to a reader of the metric: nothing is pinned right now.
    pub fn applied(&self) -> usize {
        match self.applied.load(Ordering::Relaxed) {
            usize::MAX => 0,
            n => n,
        }
    }
}

/// Choose which of one layer's experts to hold resident, hottest first, until
/// `budget` bytes are spent. `bytes_of(expert)` is that expert's footprint in
/// the tier being planned. Pure and deterministic, so the policy is testable
/// without a cache, a GPU, or a checkpoint.
///
/// **An expert with no observed routing is never chosen.** A cold table says
/// nothing about which eighth of a 256-expert pool matters, so ranking it would
/// spend the entire budget on expert ids `0..K` and then hold them against the
/// experts drafting actually asks for. [`crate::gpu::rank_by_heat`] makes the
/// opposite choice — a cold rank there must reproduce `plan_residency`'s static
/// round-robin, because at load *something* has to be resident — but a pin
/// carries no such obligation: an empty pin set is a correct pin set, and the
/// mechanism starts working once drafting has revealed a shape.
///
/// Ties keep ascending expert id, so equal counts never reshuffle the set
/// between generations — the same stability `rank_by_heat` documents, and for
/// the same reason: every moved pin is a re-read or a PCIe upload.
///
/// Returned in **heat order**, not id order. A caller that must truncate further
/// (a shrunken cache budget, a PCIe quota) has to drop the coldest, and only
/// this order lets it; callers issuing reads sort by id first, since within one
/// layer ascending expert id is ascending disk offset.
pub fn plan_pins(counts: &[u32], budget: usize, bytes_of: impl Fn(usize) -> usize) -> Vec<usize> {
    if budget == 0 {
        return Vec::new();
    }
    let mut cand: Vec<usize> = (0..counts.len()).filter(|&e| counts[e] > 0).collect();
    // Stable sort on the descending count: equal counts keep ascending id.
    cand.sort_by_key(|&e| std::cmp::Reverse(counts[e]));
    let mut out = Vec::new();
    let mut used = 0usize;
    for e in cand {
        let b = bytes_of(e);
        if b == 0 || used.saturating_add(b) > budget {
            // `break`, not `continue`. The list is heat-ordered, so everything
            // past the first expert that does not fit is colder and worth less;
            // letting a colder-but-smaller one fill the gap makes a byte-budgeted
            // set depend on the size distribution rather than on the routing. The
            // two rules can only differ on a container whose experts vary in size
            // *within* one layer, which none does — `peregrine-requantize` picks
            // precision per layer — so this is about keeping the rule legible,
            // not about a case in the tree.
            break;
        }
        used += b;
        out.push(e);
    }
    out
}

/// Clamp a pin budget to half the tier it lives in, and say so **once** if it
/// was clamped. Returns the granted budget.
///
/// **Half, because a pin stops being a reservation past that point.** Neither
/// tier's evictor refuses to evict a pinned entry — `WarmCache::evict_to_budget`
/// takes the lowest `(prio, recency)` resident whatever its priority, and the
/// GPU tier's byte budget is a hard ceiling — so a tier that is mostly pins does
/// not hold them all: it starves everything else first and then evicts among the
/// pins anyway, on recency, which is the ordering the pin existed to remove.
///
/// Reported once rather than per generation because the clamp is recomputed
/// every refresh and a line each time would bury the log; reported at all
/// because a silently truncated budget reads as "the pin set is small" when what
/// happened is "the knob asked for more than the tier can give". The caller owns
/// the `Once`, so each tier reports its own clamp independently.
pub fn granted_pin_budget(
    knob: &str,
    tier: &str,
    asked: usize,
    tier_budget: usize,
    once: &std::sync::Once,
) -> usize {
    let granted = asked.min(tier_budget / 2);
    if granted < asked {
        once.call_once(|| {
            let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
            eprintln!(
                "peregrine: [mtp-pin] {knob} asked for {:.0} MB, granted {:.0} MB — a pin set past \
                 half the {tier} stops reserving and starts evicting what it shares the tier with",
                mb(asked),
                mb(granted),
            );
        });
    }
    granted
}

#[cfg(test)]
mod pin_tests {
    use super::{plan_pins, MtpPins, PIN_REFRESH_ROUNDS};

    #[test]
    fn a_cold_table_pins_nothing() {
        // The whole point: no routing observed means no information about which
        // experts matter, so the budget stays unspent rather than being handed
        // to expert ids 0..K.
        let counts = vec![0u32; 8];
        assert!(plan_pins(&counts, usize::MAX, |_| 16).is_empty());
    }

    #[test]
    fn hottest_first_and_budget_bounded() {
        let mut counts = vec![0u32; 8];
        counts[5] = 9;
        counts[2] = 40;
        counts[7] = 3;
        // Room for exactly two experts of 16 bytes.
        assert_eq!(plan_pins(&counts, 32, |_| 16), vec![2, 5]);
        // Room for all three.
        assert_eq!(plan_pins(&counts, 48, |_| 16), vec![2, 5, 7]);
        // Room for none.
        assert!(plan_pins(&counts, 8, |_| 16).is_empty());
    }

    #[test]
    fn equal_counts_keep_ascending_expert_id() {
        // Stability is what keeps a partially-warm table from reshuffling the
        // resident set — and every reshuffled pin is a re-read.
        let counts = vec![7u32, 7, 7, 7];
        assert_eq!(plan_pins(&counts, 32, |_| 16), vec![0, 1]);
    }

    #[test]
    fn bump_ignores_other_layers_and_out_of_range_experts() {
        let p = MtpPins::new(78, 4);
        p.bump(78, 1);
        p.bump(77, 1); // a main-stack layer must not reach this table
        p.bump(78, 9); // out of range
        assert_eq!(p.snapshot(), vec![0, 1, 0, 0]);
    }

    #[test]
    fn refresh_fires_once_per_generation() {
        let p = MtpPins::new(0, 1);
        let fired: u64 = (0..PIN_REFRESH_ROUNDS * 3).filter(|_| p.note_round()).count() as u64;
        assert_eq!(fired, 3, "one refresh per {PIN_REFRESH_ROUNDS} draft rounds");
    }
}

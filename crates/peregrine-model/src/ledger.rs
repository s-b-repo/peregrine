//! The byte ledger — decomposing "11.3 GB per token" into the numbers it is
//! actually made of.
//!
//! That figure is quoted as one number throughout this repo and it is not one.
//! It is the *arithmetic* denominator — `topk × sparse_layers × bytes_per_expert`
//! — and every real byte count sits below it for a different reason:
//!
//! | column | what it is | why it differs from the one above |
//! |---|---|---|
//! | **requested** | `Σ keff` over routed positions | the arithmetic figure |
//! | **unique** | distinct experts after the batch union | at B>1 one read serves every row that selected it |
//! | **cache-served** | warm-cache hits | never reached the drive |
//! | **from disk** | what the drive actually moved | the only column an operator feels |
//! | **prefetch waste** | speculative reads never hit | moved and bought nothing |
//!
//! # Why this exists
//!
//! Optimizing a sum without its decomposition is how a saving in one column
//! gets reported as a saving overall while another column silently absorbs it.
//! This repo has the scar: `COLI_ROUTE_MIN_SHARE` cut ~12.5 % of reads and cost
//! **27.9 %** of top-1 predictions — a real byte saving, priced out only
//! because someone insisted on computing the quality column beside it.
//!
//! So the rule this module enforces structurally: **a byte saving with no
//! quality figure beside it is not a result.** [`Ledger::verdict`] refuses to
//! present a saving without naming the flip-rate gate that has to qualify it.
//!
//! # What it does not have
//!
//! One column is **not** derivable from any counter this engine currently
//! keeps: bytes re-read because an eviction was wrong. Distinguishing a re-read
//! from a first read needs per-slab eviction history the warm cache does not
//! retain. That absence is reported rather than folded silently into
//! `from_disk`, because a missing column that looks present is worse than one
//! that is named — see the 0.6 % figure this project has already had to publish
//! a correction about.

/// One accounting of where a run's expert bytes went.
///
/// All fields are **bytes**, not slab counts. The counters they come from are
/// counts, and the conversion is `count × bytes_per_expert`, which is exact
/// only while every routed expert is the same size. A heat-tiered container
/// (`--tier-hot-frac`) breaks that, and [`Ledger::uniform_expert_size`] records
/// whether it held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    /// Positions × selections: what routing asked for, before any sharing.
    pub requested: u64,
    /// Distinct experts after the batch union — what a step actually needs.
    pub unique: u64,
    /// Served from the warm cache without touching the drive.
    pub cache_served: u64,
    /// Read from the drive.
    pub from_disk: u64,
    /// Speculatively read and never hit before eviction.
    pub prefetch_waste: u64,
    /// Bytes per routed expert, the conversion factor for every column above.
    pub bytes_per_expert: u64,
    /// Whether every routed expert really is that size. `false` on a tiered
    /// container, where these figures are estimates and say so.
    pub uniform_expert_size: bool,
}

impl Ledger {
    /// Batch amortization: what fraction of requested bytes the union removes.
    ///
    /// This is the number the continuous-batching claim rests on, and until now
    /// it was published only as a throughput ratio (4.4× aggregate at B=16),
    /// never as the byte identity underneath it.
    pub fn union_saving(&self) -> f64 {
        if self.requested == 0 {
            return 0.0;
        }
        1.0 - (self.unique.min(self.requested) as f64 / self.requested as f64)
    }

    /// Share of unique bytes the cache kept off the drive.
    pub fn cache_saving(&self) -> f64 {
        if self.unique == 0 {
            return 0.0;
        }
        self.cache_served.min(self.unique) as f64 / self.unique as f64
    }

    /// Prefetch waste as a share of the bytes that reached the drive. Not of
    /// `requested`: the waste is disk traffic, and expressing it against a
    /// denominator that never touched disk would understate it.
    pub fn waste_share(&self) -> f64 {
        if self.from_disk == 0 {
            return 0.0;
        }
        self.prefetch_waste as f64 / self.from_disk as f64
    }

    /// Bytes that actually left the drive per token, given the tokens the run
    /// produced. The figure an operator feels, as opposed to the arithmetic one.
    pub fn disk_per_token(&self, tokens: u64) -> f64 {
        if tokens == 0 {
            0.0
        } else {
            self.from_disk as f64 / tokens as f64
        }
    }

    /// The arithmetic figure per token — what "11.3 GB/token" means.
    pub fn requested_per_token(&self, tokens: u64) -> f64 {
        if tokens == 0 {
            0.0
        } else {
            self.requested as f64 / tokens as f64
        }
    }

    /// The ledger, rendered. `tokens` is the run's decoded token count.
    ///
    /// Every saving is printed with the gate that has to qualify it. A column
    /// this engine cannot source is printed as absent rather than omitted: an
    /// omitted column reads as an oversight and a fabricated one reads as a
    /// measurement, and neither is true.
    pub fn report(&self, tokens: u64) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let mut s = format!(
            "[ledger] expert bytes over {tokens} token(s), {} B/expert{}\n\
             [ledger] {:<18} {:>12.3} GB {:>10.3} GB/token\n\
             [ledger] {:<18} {:>12.3} GB {:>10.3} GB/token   ({:.1}% saved by batch union)\n\
             [ledger] {:<18} {:>12.3} GB {:>21}({:.1}% of unique)\n\
             [ledger] {:<18} {:>12.3} GB {:>10.3} GB/token\n\
             [ledger] {:<18} {:>12.3} GB {:>21}({:.1}% of disk traffic)\n",
            self.bytes_per_expert,
            if self.uniform_expert_size { "" } else { " (ESTIMATE — tiered container, experts differ in size)" },
            "requested",
            gb(self.requested),
            self.requested_per_token(tokens) / 1e9,
            "unique (union)",
            gb(self.unique),
            if tokens > 0 { self.unique as f64 / tokens as f64 / 1e9 } else { 0.0 },
            100.0 * self.union_saving(),
            "cache-served",
            gb(self.cache_served),
            "",
            100.0 * self.cache_saving(),
            "from disk",
            gb(self.from_disk),
            self.disk_per_token(tokens) / 1e9,
            "prefetch waste",
            gb(self.prefetch_waste),
            "",
            100.0 * self.waste_share(),
        );
        s.push_str(
            "[ledger] re-read after eviction: NOT MEASURED — no counter distinguishes a re-read \n\
             [ledger]   from a first read. Those bytes are inside `from disk` above, unlabelled.\n",
        );
        // The standing rule, enforced here rather than left to the reader.
        s.push_str(
            "[ledger] NOTE: every saving above is a BYTE figure with no quality figure beside it. \n\
             [ledger]   Gate any change that produced one with `peregrine flip-rate` before \n\
             [ledger]   quoting it. COLI_ROUTE_MIN_SHARE cut 12.5% of reads for 27.9% of top-1.\n",
        );
        s
    }

    /// Whether the columns are internally consistent.
    ///
    /// `unique` cannot exceed `requested` and `cache_served + from_disk` should
    /// account for `unique`. A violation means a counter was read from a
    /// different window than its neighbours — which produces a ledger that
    /// looks precise and is not.
    pub fn coherent(&self) -> bool {
        self.unique <= self.requested
            && self.cache_served <= self.unique
            && self.prefetch_waste <= self.from_disk
    }
}

/// Assemble a ledger from raw counter values.
///
/// Kept separate from the counters themselves so it is testable without a
/// model: every argument is a number some subsystem already tracks.
pub struct LedgerInput {
    /// `Σ keff` — routed selections, from `router::union_stats_snapshot`.
    pub selections: u64,
    /// Distinct experts after the union, same source.
    pub distinct: u64,
    /// Warm-cache hits (slab count).
    pub cache_hits: u64,
    /// Warm-cache misses across both paths (slab count).
    pub cache_misses: u64,
    /// Prefetched slabs evicted before ever being hit.
    pub prefetch_wasted: u64,
    pub bytes_per_expert: u64,
    pub uniform_expert_size: bool,
}

impl LedgerInput {
    pub fn build(&self) -> Ledger {
        let b = self.bytes_per_expert;
        // Misses are what reached the drive; hits are what did not. Both are
        // slab counts against the post-union stream, which is why `unique` is
        // the denominator for the cache share and `requested` is not.
        Ledger {
            requested: self.selections.saturating_mul(b),
            unique: distinct_guard(self.distinct, self.selections).saturating_mul(b),
            cache_served: self.cache_hits.saturating_mul(b),
            from_disk: self.cache_misses.saturating_mul(b),
            prefetch_waste: self.prefetch_wasted.saturating_mul(b),
            bytes_per_expert: b,
            uniform_expert_size: self.uniform_expert_size,
        }
    }
}

/// `distinct` can exceed `selections` only if the two were sampled from
/// different windows; clamping keeps the ledger coherent rather than emitting
/// a negative saving that would read as a bug in batching.
fn distinct_guard(distinct: u64, selections: u64) -> u64 {
    distinct.min(selections)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> LedgerInput {
        LedgerInput {
            selections: 1000,
            distinct: 250,
            cache_hits: 50,
            cache_misses: 200,
            prefetch_wasted: 120,
            bytes_per_expert: 18_900_000,
            uniform_expert_size: true,
        }
    }

    #[test]
    fn the_union_saving_is_the_batching_claim_as_a_byte_identity() {
        // 1000 selections collapsing to 250 distinct is a 75% byte saving, and
        // that identity is what the 4.4x aggregate throughput figure rests on.
        // It had never been published as bytes, only as a throughput ratio.
        let l = input().build();
        assert!((l.union_saving() - 0.75).abs() < 1e-9, "{}", l.union_saving());
        assert!(l.coherent());
    }

    #[test]
    fn prefetch_waste_is_measured_against_disk_traffic_not_against_requested() {
        // Wasted speculative reads are disk bytes. Expressing them against the
        // arithmetic denominator — which is 4x larger here — would report 2.3%
        // where the truth is 60%, and the flattering figure is the wrong one.
        let l = input().build();
        assert!((l.waste_share() - 0.6).abs() < 1e-9, "{}", l.waste_share());
        let against_requested = l.prefetch_waste as f64 / l.requested as f64;
        assert!(against_requested < l.waste_share(), "the wrong denominator flatters");
    }

    #[test]
    fn a_saving_is_never_printed_without_the_gate_that_qualifies_it() {
        // The standing rule from the review thread, enforced structurally: a
        // byte saving with no quality figure beside it is not a result.
        let r = input().build().report(64);
        assert!(r.contains("flip-rate"), "{r}");
        assert!(r.contains("27.9%"), "the priced-out precedent must travel with it: {r}");
    }

    #[test]
    fn the_column_this_engine_cannot_source_is_named_not_omitted() {
        // An omitted column reads as an oversight; a fabricated one reads as a
        // measurement. Neither is true, so it is printed as absent.
        let r = input().build().report(64);
        assert!(r.contains("NOT MEASURED"), "{r}");
        assert!(r.contains("re-read after eviction"), "{r}");
    }

    #[test]
    fn a_tiered_container_marks_its_own_figures_as_estimates() {
        // `count x bytes_per_expert` is exact only while experts are the same
        // size, and `--tier-hot-frac` breaks that by ~40%.
        let mut i = input();
        i.uniform_expert_size = false;
        let r = i.build().report(10);
        assert!(r.contains("ESTIMATE"), "{r}");
        assert!(r.contains("tiered container"), "{r}");
    }

    #[test]
    fn counters_from_mismatched_windows_cannot_produce_a_negative_saving() {
        // More distinct than selections is impossible within one window; if it
        // happens the two were sampled at different times, and a negative
        // union saving would read as "batching made it worse".
        let mut i = input();
        i.distinct = 5000;
        let l = i.build();
        assert!(l.coherent(), "the guard must keep the ledger coherent");
        assert_eq!(l.union_saving(), 0.0, "no saving, not a negative one");
    }

    #[test]
    fn an_empty_run_reports_zeros_rather_than_dividing() {
        let l = Ledger::default();
        assert_eq!(l.union_saving(), 0.0);
        assert_eq!(l.cache_saving(), 0.0);
        assert_eq!(l.waste_share(), 0.0);
        assert_eq!(l.disk_per_token(0), 0.0);
        assert!(l.coherent());
        assert!(!l.report(0).is_empty());
    }

    #[test]
    fn disk_per_token_is_the_column_an_operator_feels() {
        // The arithmetic figure and the real one differ by the union and the
        // cache, and quoting the first as if it were the second is the whole
        // reason this module exists.
        let l = input().build();
        let (arith, real) = (l.requested_per_token(10), l.disk_per_token(10));
        assert!(real < arith, "disk traffic must be below the arithmetic figure");
        assert!((arith / real - 5.0).abs() < 1e-9, "1000 selections vs 200 disk reads");
    }
}

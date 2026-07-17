//! Hot-store tiering policy — direct port of `c/tier.h`.
//!
//! LFRU: routing frequency (`heat`) is the primary signal; recency (`last` vs a
//! monotonic `clock`) only breaks close calls. A recent access is worth at most
//! 255 points while one frequency count is worth 256, so a merely-recent expert
//! cannot displace a genuinely hotter one. A 25%+4-count hysteresis prevents
//! ping-pong when promoting a hot streamed expert over a cold pinned one.

/// A proposed hot-store swap: replace `slot` (index into the pinned set) with
/// expert `eid`, gaining `gain` frequency counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swap {
    pub slot: usize,
    pub eid: usize,
    pub gain: i64,
}

/// Combined score: `heat << 8 | recent`. Port of `tier_lfru_score`.
pub fn lfru_score(heat: u32, last: u32, clock: u32) -> u64 {
    let age = clock.wrapping_sub(last);
    let recent = 255u32.saturating_sub(age); // 255-age, floored at 0
    ((heat as u64) << 8) | recent as u64
}

/// Pick a pinned slot to replace using LFRU. Port of `tier_pick_lfru`.
pub fn pick_lfru(heat: &[u32], last: &[u32], clock: u32, pinned: &[usize]) -> Option<Swap> {
    if heat.is_empty() || last.is_empty() || pinned.is_empty() {
        return None;
    }
    // coldest pinned slot
    let mut cold = 0usize;
    for z in 1..pinned.len() {
        if lfru_score(heat[pinned[z]], last[pinned[z]], clock) < lfru_score(heat[pinned[cold]], last[pinned[cold]], clock) {
            cold = z;
        }
    }
    // hottest non-resident expert
    let mut hot: Option<usize> = None;
    let mut hs = 0u64;
    for e in 0..heat.len() {
        if pinned.contains(&e) {
            continue;
        }
        let score = lfru_score(heat[e], last[e], clock);
        if hot.is_none() || score > hs {
            hot = Some(e);
            hs = score;
        }
    }
    let hot = hot?;
    let cs = lfru_score(heat[pinned[cold]], last[pinned[cold]], clock);
    // 25% + 4-frequency-count hysteresis, in score units
    if hs <= cs + (cs >> 2) + (4u64 << 8) {
        return None;
    }
    Some(Swap { slot: cold, eid: hot, gain: ((hs - cs) >> 8) as i64 })
}

/// Frequency-only swap pick. Port of `tier_pick_swap`.
pub fn pick_swap(heat: &[u32], pinned: &[usize]) -> Option<Swap> {
    if heat.is_empty() || pinned.is_empty() {
        return None;
    }
    let mut cold = 0usize;
    for z in 1..pinned.len() {
        if heat[pinned[z]] < heat[pinned[cold]] {
            cold = z;
        }
    }
    let mut hot: Option<usize> = None;
    let mut fh = 0u32;
    for e in 0..heat.len() {
        if pinned.contains(&e) {
            continue;
        }
        if heat[e] > fh {
            fh = heat[e];
            hot = Some(e);
        }
    }
    let hot = hot?;
    let fc = heat[pinned[cold]];
    if fh <= fc + (fc >> 2) + 4 {
        return None;
    }
    Some(Swap { slot: cold, eid: hot, gain: fh as i64 - fc as i64 })
}

/// Halve all heat counters (periodic decay). Port of `tier_decay`.
pub fn decay(heat: &mut [u32]) {
    for h in heat.iter_mut() {
        *h >>= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_dominates_recency() {
        // expert 0: very hot but old; expert 1: cold but just accessed
        let heat = [1000u32, 1];
        let last = [0u32, 100];
        let clock = 100;
        // hotter-but-old still outscores recent-but-cold
        assert!(lfru_score(heat[0], last[0], clock) > lfru_score(heat[1], last[1], clock));
    }

    #[test]
    fn hysteresis_blocks_marginal_swap() {
        // pinned expert 0 (heat 100); candidate expert 1 only slightly hotter
        let heat = [100u32, 110];
        // 110 <= 100 + 25 + 4 → no swap
        assert_eq!(pick_swap(&heat, &[0]), None);
    }

    #[test]
    fn clear_winner_swaps() {
        let heat = [10u32, 500];
        let last = [0u32, 0];
        let s = pick_lfru(&heat, &last, 0, &[0]).unwrap();
        assert_eq!(s.slot, 0);
        assert_eq!(s.eid, 1);
    }

    #[test]
    fn decay_halves() {
        let mut h = [8u32, 3, 0];
        decay(&mut h);
        assert_eq!(h, [4, 1, 0]);
    }
}

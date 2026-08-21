//! Tolerance-keyed state fingerprints — the identity half of token-equivalence
//! memoization (Q4).
//!
//! The idea being served: if two hidden states are close enough, the expert
//! computation for one might be reusable for the other, turning an ~18.9 MB
//! read plus a matmul into a cache lookup. That is *computation* memoization,
//! not token caching.
//!
//! # The defect this module exists to avoid
//!
//! The obvious implementation hashes the state, then accepts any cached entry
//! **within `eps`** of the query. That is not a key. It memoizes on a *range*,
//! and ranges overlap: two lookups can both "hit" entries computed under
//! different tolerances, and no later reader can say which computation
//! answered. The failure is silent and it compounds — a reused result becomes
//! the input to the next reuse.
//!
//! So the tolerance is part of the **identity**, not a post-hoc acceptance
//! test: [`fingerprint`] hashes the quantized vector *and* the band together.
//! Two keys built under different bands can never collide, by construction.
//!
//! # The guarantee, and its exact shape
//!
//! Quantizing coordinate-wise onto a grid of width `band` gives a **provable
//! one-sided bound**:
//!
//! > If two states produce the same [`StateKey`], they differ by at most
//! > `band` in every coordinate (`L∞ ≤ band`).
//!
//! because `round(a/band) == round(b/band)` implies `|a − b| ≤ band`. That is
//! the direction that matters: a hit can never mean "arbitrarily far apart".
//!
//! The converse does **not** hold, and saying so is the point. Two states
//! closer than `band` can still straddle a grid boundary and get different
//! keys — a *miss* where a reuse was legitimate. This scheme is therefore
//! **conservative**: it loses opportunities, it does not invent equivalences.
//! For a correctness-sensitive cache that is the only acceptable direction to
//! be wrong in.
//!
//! # What this does not do
//!
//! It does not skip any expert computation. Nothing consumes these keys yet;
//! the memoization they would key is unbuilt, and building it needs a decision
//! this module cannot make — whether an `L∞` bound on the *hidden state*
//! implies a usable bound on the *expert output*, through a gated SwiGLU and a
//! router. It probably does not in general. So the honest sequencing is: the
//! key first, with a provable property; the reuse behind a flip-rate gate; and
//! no claim of correctness-neutrality until that gate has run.

/// A state fingerprint that carries its own tolerance.
///
/// `band_bits` is the raw bit pattern of the band, so two keys built under
/// different tolerances differ even when their quantized vectors coincide —
/// which is the whole reason the band is in here rather than in the lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateKey {
    /// Hash of the quantized coordinates.
    pub digest: u64,
    /// Bit pattern of the band this key was built under.
    pub band_bits: u32,
    /// Vector length, so states of different widths cannot collide.
    pub len: u32,
}

impl StateKey {
    /// The band this key was built under.
    pub fn band(&self) -> f32 {
        f32::from_bits(self.band_bits)
    }

    /// Whether two keys were built under the same tolerance. A `false` here
    /// means the two are not comparable at all — not that the states differ.
    pub fn same_band(&self, other: &StateKey) -> bool {
        self.band_bits == other.band_bits
    }
}

/// Sentinel used for any coordinate that is not finite.
///
/// A NaN cannot be quantized onto a grid, and mapping it to 0 would make every
/// NaN-bearing state collide with a legitimate zero. It gets its own reserved
/// cell so such states collide only with each other — which in practice means a
/// broken forward memoizes against other broken forwards and never against a
/// good one.
const NONFINITE_CELL: i64 = i64::MIN;

/// Fingerprint `h` at tolerance `band`.
///
/// Returns `None` for a non-positive or non-finite band: a zero band would
/// divide by zero and a negative one would invert the grid, and both would
/// produce a key that looks valid. Refusing is the only safe answer, because
/// the caller's next move is to trust the key.
pub fn fingerprint(h: &[f32], band: f32) -> Option<StateKey> {
    if !band.is_finite() || band <= 0.0 {
        return None;
    }
    // FNV-1a over the quantized cells. An identity hash, not cryptography:
    // the adversary here is accidental collision, not a forger.
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: i64| {
        for b in v.to_le_bytes() {
            digest ^= u64::from(b);
            digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    for &x in h {
        let cell = if x.is_finite() {
            // `round` (half away from zero) rather than `floor`: floor makes
            // the bound one-sided (`0 <= a-b < band` only when both are in the
            // same cell), while rounding keeps `|a-b| <= band` symmetric.
            let q = (f64::from(x) / f64::from(band)).round();
            // Clamp rather than wrap: a coordinate large enough to overflow i64
            // at this band is pathological, and wrapping would alias it onto a
            // small cell — a false equivalence, the one outcome forbidden here.
            if q >= i64::MAX as f64 {
                i64::MAX
            } else if q <= (i64::MIN + 1) as f64 {
                i64::MIN + 1
            } else {
                q as i64
            }
        } else {
            NONFINITE_CELL
        };
        mix(cell);
    }
    // Fold the band and the length into the digest as well as carrying them in
    // the struct, so even a caller that compares only `digest` cannot cross
    // bands.
    mix(i64::from(band.to_bits()));
    mix(h.len() as i64);
    Some(StateKey { digest, band_bits: band.to_bits(), len: h.len() as u32 })
}

/// Largest coordinate-wise difference between two equal-length vectors.
///
/// The quantity the key's guarantee is stated in. Returns `None` on a length
/// mismatch rather than comparing a prefix, which would report a bound that
/// was never checked.
pub fn linf(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random floats — no clock, no rand dependency.
    fn vecs(n: usize, len: usize, seed: u64, scale: f32) -> Vec<Vec<f32>> {
        let mut z = seed | 1;
        let mut next = || {
            z ^= z >> 12;
            z ^= z << 25;
            z ^= z >> 27;
            z = z.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        (0..n).map(|_| (0..len).map(|_| (next() - 0.5) * 2.0 * scale).collect()).collect()
    }

    #[test]
    fn equal_keys_imply_the_bound_they_advertise() {
        // The module's one provable claim, checked exhaustively over colliding
        // pairs: same key => L-infinity <= band. If this ever fails, every
        // reuse the scheme would license is unbounded.
        // Low dimension and a coarse band relative to the spread, so
        // collisions are common. The first fixture used 8 dimensions at a fine
        // band and produced none — the assertion at the bottom caught that the
        // test was passing without testing anything, which is the failure mode
        // this whole repo keeps finding.
        let band = 0.4f32;
        let v = vecs(400, 3, 0xA11CE, 0.5);
        let mut compared = 0;
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                let (Some(ki), Some(kj)) = (fingerprint(&v[i], band), fingerprint(&v[j], band)) else {
                    continue;
                };
                if ki == kj {
                    compared += 1;
                    let d = linf(&v[i], &v[j]).unwrap_or(f32::INFINITY);
                    assert!(
                        d <= band * (1.0 + 1e-6),
                        "keys collided at L-inf {d} > band {band} — the bound is not held"
                    );
                }
            }
        }
        assert!(compared > 0, "the fixture produced no collisions, so nothing was actually tested");
    }

    #[test]
    fn the_scheme_is_conservative_not_permissive() {
        // The converse deliberately does NOT hold: states closer than the band
        // can straddle a grid boundary and miss. Documented as the safe
        // direction to be wrong in, and pinned so nobody "fixes" it into a
        // permissive matcher.
        let band = 1.0f32;
        let a = [0.49f32];
        let b = [0.51f32];
        assert!(linf(&a, &b).unwrap_or(1.0) < band, "these are within the band");
        assert_ne!(
            fingerprint(&a, band),
            fingerprint(&b, band),
            "a boundary straddle must MISS — losing a reuse is safe, inventing one is not"
        );
    }

    #[test]
    fn different_bands_can_never_share_a_key() {
        // The defect the module exists to prevent: without the band in the
        // identity, a lookup can hit an entry computed under a different
        // tolerance and no reader can tell which computation answered.
        let h = [0.1f32, 0.2, 0.3];
        let a = fingerprint(&h, 0.01);
        let b = fingerprint(&h, 0.02);
        assert!(a.is_some() && b.is_some());
        assert_ne!(a, b, "same vector, different tolerance, must not be the same key");
        if let (Some(a), Some(b)) = (a, b) {
            assert_ne!(a.digest, b.digest, "the band must be folded into the digest, not only the struct");
            assert!(!a.same_band(&b));
        }
    }

    #[test]
    fn a_post_hoc_epsilon_match_admits_pairs_the_key_does_not() {
        // Why the band belongs in the key rather than in the lookup. The
        // "hash then accept anything within eps" design is transitive by
        // accident: a~b and b~c does not give a~c, but a range-based cache will
        // happily serve `a` from `c`. Demonstrated rather than asserted.
        let band = 1.0f32;
        let (a, b, c) = ([0.0f32], [0.9f32], [1.8f32]);
        let within = |x: &[f32], y: &[f32]| linf(x, y).unwrap_or(f32::INFINITY) <= band;
        assert!(within(&a, &b) && within(&b, &c), "each neighbouring pair is inside the band");
        assert!(!within(&a, &c), "but the ends are 1.8 apart — twice the tolerance");
        // The key-based scheme cannot chain them: a and c land in different
        // cells and simply do not match.
        assert_ne!(fingerprint(&a, band), fingerprint(&c, band));
    }

    #[test]
    fn an_unusable_band_is_refused_rather_than_keyed() {
        // A zero band divides by zero; a negative one inverts the grid. Both
        // would return a key that looks valid, and the caller's next move is to
        // trust it.
        let h = [1.0f32, 2.0];
        assert!(fingerprint(&h, 0.0).is_none());
        assert!(fingerprint(&h, -0.5).is_none());
        assert!(fingerprint(&h, f32::NAN).is_none());
        assert!(fingerprint(&h, f32::INFINITY).is_none());
        assert!(fingerprint(&h, 1e-6).is_some(), "a small but positive band is legitimate");
    }

    #[test]
    fn a_broken_forward_memoizes_only_against_other_broken_forwards() {
        // NaN cannot be placed on a grid. Mapping it to cell 0 would collide
        // every NaN-bearing state with a legitimate zero vector, which is the
        // one collision that would silently poison good results.
        let nan = [f32::NAN, 1.0f32];
        let zero = [0.0f32, 1.0f32];
        assert_ne!(fingerprint(&nan, 0.1), fingerprint(&zero, 0.1));
        let nan2 = [f32::NAN, 1.0f32];
        assert_eq!(fingerprint(&nan, 0.1), fingerprint(&nan2, 0.1), "NaN states share their own cell");
    }

    #[test]
    fn vectors_of_different_widths_cannot_collide() {
        // A shorter state is a different computation, not a nearby one.
        let a = [1.0f32, 0.0];
        let b = [1.0f32];
        assert_ne!(fingerprint(&a, 0.5), fingerprint(&b, 0.5));
        assert_eq!(fingerprint(&a, 0.5).map(|k| k.len), Some(2));
    }

    #[test]
    fn fingerprints_are_deterministic_across_calls() {
        // A key that varied per call would make the cache a no-op that looked
        // like a working one.
        let v = vecs(1, 32, 7, 1.0);
        assert_eq!(v.len(), 1, "the fixture must produce the vector under test");
        let h = &v[0];
        assert_eq!(fingerprint(h, 0.01), fingerprint(h, 0.01));
    }

    #[test]
    fn a_wider_band_never_produces_fewer_collisions() {
        // Monotonicity in the tolerance: coarsening the grid can only merge
        // cells. If this failed, "increase eps to get more hits" would be
        // unsound advice.
        let v = vecs(120, 6, 0xBEEF, 0.5);
        let count = |band: f32| {
            let keys: Vec<Option<StateKey>> = v.iter().map(|x| fingerprint(x, band)).collect();
            let mut n = 0;
            for i in 0..keys.len() {
                for j in (i + 1)..keys.len() {
                    if keys[i].is_some() && keys[i] == keys[j] {
                        n += 1;
                    }
                }
            }
            n
        };
        let (narrow, wide) = (count(0.05), count(0.5));
        assert!(wide >= narrow, "a wider band gave fewer collisions ({wide} < {narrow})");
    }

    #[test]
    fn linf_refuses_a_length_mismatch_rather_than_comparing_a_prefix() {
        // Comparing a prefix would report a bound that was never checked.
        assert!(linf(&[1.0, 2.0], &[1.0]).is_none());
        assert_eq!(linf(&[1.0, 2.0], &[1.5, 2.0]), Some(0.5));
    }
}

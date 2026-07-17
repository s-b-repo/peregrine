//! Token sampling — port of `argmax_v` / `dist_build` / `dist_sample` /
//! `pick_tok` and the xorshift64 RNG (`c/glm.c:4063-4101`).
//!
//! `temp <= 0` is greedy argmax (deterministic — the token-exact validation
//! mode). Otherwise: softmax at `temp`, optional nucleus (top-p) truncation,
//! then inverse-CDF sampling with an optional banned token (used by speculative
//! rejection sampling so the draft stays invisible to the output distribution).

/// Greedy argmax; lowest index wins ties (strict `>`), matching `argmax_v`.
pub fn argmax(lo: &[f32]) -> usize {
    let mut b = 0usize;
    for i in 1..lo.len() {
        if lo[i] > lo[b] {
            b = i;
        }
    }
    b
}

/// Stateful sampler: holds the RNG stream and reused distribution buffers so a
/// decode loop is single-threaded and reproducible from `seed`.
pub struct Sampler {
    rng: u64,
    pub temp: f32,
    pub nucleus: f32,
    p: Vec<f32>,
    idx: Vec<usize>,
}

impl Sampler {
    /// `temp <= 0` → greedy. `nucleus` in (0,1) enables top-p truncation.
    /// The default RNG seed matches the C engine (`0x9E3779B97F4A7C15`).
    pub fn new(temp: f32, nucleus: f32, seed: u64) -> Sampler {
        Sampler { rng: seed, temp, nucleus, p: Vec::new(), idx: Vec::new() }
    }

    /// xorshift64 → uniform double in [0,1). Port of `rndu`.
    fn rndu(&mut self) -> f64 {
        let mut g = self.rng;
        g ^= g << 13;
        g ^= g >> 7;
        g ^= g << 17;
        self.rng = g;
        (g >> 11) as f64 * (1.0 / 9007199254740992.0) // 2^53
    }

    /// Build the target distribution into `self.p`: softmax(lo/temp), optionally
    /// truncated to the top-p `nucleus` mass and renormalized. Port of `dist_build`.
    fn dist_build(&mut self, lo: &[f32]) {
        let v = lo.len();
        self.p.resize(v, 0.0);
        let mx = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let invt = 1.0 / self.temp.max(1e-4);
        let mut s = 0f64;
        for i in 0..v {
            self.p[i] = ((lo[i] - mx) * invt).exp();
            s += self.p[i] as f64;
        }
        for i in 0..v {
            self.p[i] /= s as f32;
        }
        if self.nucleus > 0.0 && self.nucleus < 1.0 {
            self.idx.clear();
            self.idx.extend(0..v);
            let p = &self.p;
            // total_cmp is a total order over f32 (NaN-safe) — `partial_cmp` would
            // return None on a NaN probability and panic on unwrap, and a NaN in a
            // sort comparator also violates sort's ordering contract.
            self.idx.sort_by(|&a, &b| p[b].total_cmp(&p[a])); // desc
            let mut cum = 0f64;
            let mut keep = v;
            for i in 0..v {
                cum += self.p[self.idx[i]] as f64;
                if cum >= self.nucleus as f64 {
                    keep = i + 1;
                    break;
                }
            }
            for i in keep..v {
                self.p[self.idx[i]] = 0.0;
            }
            let mut s2 = 0f64;
            for i in 0..keep {
                s2 += self.p[self.idx[i]] as f64;
            }
            for i in 0..keep {
                self.p[self.idx[i]] /= s2 as f32;
            }
        }
    }

    /// Inverse-CDF sample from `self.p`; `ban >= 0` excludes that token,
    /// renormalizing on the fly. Port of `dist_sample`.
    fn dist_sample(&mut self, v: usize, ban: i32) -> usize {
        let banned = if ban >= 0 { self.p[ban as usize] as f64 } else { 0.0 };
        let mut z = 1.0 - banned;
        if z <= 1e-12 {
            z = 1e-12;
        }
        let u = self.rndu() * z;
        let mut cum = 0f64;
        for i in 0..v {
            if i as i32 == ban {
                continue;
            }
            cum += self.p[i] as f64;
            if cum >= u {
                return i;
            }
        }
        for i in (0..v).rev() {
            if i as i32 != ban && self.p[i] > 0.0 {
                return i;
            }
        }
        0
    }

    /// Next token from logits. Greedy if `temp <= 0`, else sampled. `ban < 0`
    /// means no banned token.
    pub fn pick(&mut self, lo: &[f32], ban: i32) -> usize {
        if self.temp <= 0.0 {
            return argmax(lo);
        }
        self.dist_build(lo);
        self.dist_sample(lo.len(), ban)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_lowest_index_wins_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 5.0, 1.0]), 0); // tie → lowest index
    }

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(0.0, 0.9, 1);
        assert_eq!(s.pick(&[0.1, 0.9, 0.3, 0.2], -1), 1);
    }

    #[test]
    fn sampling_is_reproducible() {
        let lo = [1.0f32, 2.0, 0.5, 3.0, 1.5];
        let mut a = Sampler::new(1.0, 0.9, 42);
        let mut b = Sampler::new(1.0, 0.9, 42);
        for _ in 0..20 {
            assert_eq!(a.pick(&lo, -1), b.pick(&lo, -1));
        }
    }

    #[test]
    fn peaked_logits_mostly_pick_peak() {
        // one dominant logit → sampled overwhelmingly (softmax still stochastic)
        let lo = [0.0f32, 0.0, 12.0, 0.0];
        let mut s = Sampler::new(1.0, 1.0, 7);
        let hits = (0..200).filter(|_| s.pick(&lo, -1) == 2).count();
        assert!(hits > 190, "peak picked {hits}/200");
    }

    #[test]
    fn nucleus_excludes_tail() {
        // tiny nucleus keeps only the top token → tail never sampled
        let lo = [0.0f32, 1.0, 2.0, 5.0];
        let mut s = Sampler::new(1.0, 0.3, 3);
        for _ in 0..100 {
            assert_eq!(s.pick(&lo, -1), 3);
        }
    }

    #[test]
    fn nan_logits_do_not_panic() {
        // a NaN logit (e.g. from a corrupted forward) must not panic the sort or
        // the sampler — total_cmp gives a total order and the result stays valid.
        let lo = [1.0f32, f32::NAN, 2.0, 0.5];
        let mut s = Sampler::new(1.0, 0.9, 5);
        let t = s.pick(&lo, -1);
        assert!(t < lo.len());
    }

    #[test]
    fn ban_is_never_sampled() {
        let lo = [2.0f32, 2.0, 2.0, 2.0];
        let mut s = Sampler::new(1.0, 1.0, 99);
        for _ in 0..200 {
            assert_ne!(s.pick(&lo, 1), 1);
        }
    }
}

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

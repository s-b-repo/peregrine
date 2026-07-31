//! Elementary numerics ported exactly from `c/glm.c` — the building blocks of
//! the forward pass. RMSNorm accumulates in f64 then rounds to f32 (matching the
//! C), RoPE uses GLM's interleaved-in / split-half-out layout.

use peregrine_core::Cfg;

/// RMSNorm: `out[i] = x[i] * rsqrt(mean(x²) + eps) * w[i]`. Port of `rmsnorm`
/// (`c/glm.c:1208`) — the sum of squares is accumulated in f64.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    // An empty row has no mean square to normalize by (0/0 = NaN); a norm
    // weight shorter than the row would index out of bounds. Both mean the
    // caller passed mismatched tensors, so leave `out` untouched.
    let d = x.len();
    if d == 0 || w.len() < d || out.len() < d {
        return;
    }
    let mut ms = 0f64;
    for &v in x {
        ms += v as f64 * v as f64;
    }
    let r = 1.0 / ((ms / d as f64) as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * w[i];
    }
}

/// In-place RMSNorm — identical arithmetic to [`rmsnorm`] with `out == x`, but
/// without the caller-side scratch copy that aliasing would otherwise force.
/// The sum of squares is accumulated over the whole row before any element is
/// written, so reading and writing the same buffer is safe.
pub fn rmsnorm_inplace(v: &mut [f32], w: &[f32], eps: f32) {
    let d = v.len();
    if d == 0 || w.len() < d {
        return;
    }
    let mut ms = 0f64;
    for &x in v.iter() {
        ms += x as f64 * x as f64;
    }
    let r = 1.0 / ((ms / d as f64) as f32 + eps).sqrt();
    for i in 0..d {
        v[i] = v[i] * r * w[i];
    }
}

/// Classic LayerNorm (mean + variance, weight + bias) — used by the DSA indexer
/// k_norm. Port of `layernorm` (`c/glm.c:1213`).
pub fn layernorm(v: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = v.len();
    if n == 0 || w.len() < n || b.len() < n {
        return; // mismatched norm tensors — see `rmsnorm`
    }
    let mut mu = 0f64;
    for &x in v.iter() {
        mu += x as f64;
    }
    mu /= n as f64;
    let mut var = 0f64;
    for &x in v.iter() {
        let d = x as f64 - mu;
        var += d * d;
    }
    var /= n as f64;
    let r = 1.0 / (var as f32 + eps).sqrt();
    for i in 0..n {
        v[i] = (v[i] - mu as f32) * r * w[i] + b[i];
    }
}

/// In-place softmax with max subtraction. Port of `softmax` (`c/glm.c:1219`).
///
/// Degenerate rows fall back to a uniform distribution rather than producing
/// NaNs: an all-`-inf` row (every key masked) would otherwise compute
/// `-inf - -inf = NaN` for every entry, and a non-finite sum would divide the
/// whole row to NaN. A NaN row poisons the logits and makes greedy decode latch
/// onto one token forever, so it must never leave this function.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        let u = 1.0 / x.len() as f32;
        x.fill(u);
        return;
    }
    let mut s = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if !s.is_finite() || s <= 0.0 {
        let u = 1.0 / x.len() as f32;
        x.fill(u);
        return;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

#[inline]
pub fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn siluf(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU elementwise: `g[i] = silu(g[i]) * u[i]`, in place — the fused
/// gate/up activation used by every MLP (`c/glm.c:3097`, etc.).
pub fn silu_mul(g: &mut [f32], u: &[f32]) {
    for (gi, &ui) in g.iter_mut().zip(u) {
        *gi = siluf(*gi) * ui;
    }
}

/// The inverse frequencies `theta^(-2j/qk)` for one RoPE lane count. They depend
/// only on the config, so they are computed once and shared by every rotation
/// instead of being re-derived with `powf` per head, per token, per layer.
#[derive(Clone, Debug)]
pub struct RopeTable {
    inv: Vec<f32>,
}

impl RopeTable {
    /// Build the table for `qk` lanes (`qk/2` pairs) at `theta`.
    pub fn new(qk: usize, theta: f32) -> RopeTable {
        let half = qk / 2;
        let inv = (0..half).map(|j| theta.powf(-2.0 * j as f32 / qk as f32)).collect();
        RopeTable { inv }
    }

    /// The table for a config's attention RoPE lanes.
    pub fn from_cfg(c: &Cfg) -> RopeTable {
        RopeTable::new(c.qk_rope.max(0) as usize, c.theta)
    }

    /// Number of rotated pairs (`qk/2`).
    pub fn pairs(&self) -> usize {
        self.inv.len()
    }
}

/// Interleaved RoPE on the first `qk_rope` lanes of `v` at position `pos`.
/// Port of `rope_interleave` (`c/glm.c:1225`): input pairs are interleaved
/// `(2j, 2j+1)`, output is split-half `(j, half+j)`.
pub fn rope_interleave(v: &mut [f32], pos: usize, c: &Cfg) {
    rope_interleave_with(v, pos, &RopeTable::from_cfg(c));
}

/// [`rope_interleave`] against a precomputed [`RopeTable`] — the hot-path form.
/// Rotating `2*pairs` lanes, this is bit-identical to the `powf`-per-call
/// version (same frequencies, same order of operations).
pub fn rope_interleave_with(v: &mut [f32], pos: usize, table: &RopeTable) {
    let half = table.inv.len();
    // A row shorter than the rotated span would leave the rotation half-applied
    // (some lanes rotated and permuted, the rest stale) — refuse instead.
    if half == 0 || v.len() < 2 * half {
        return;
    }
    let inp: Vec<f32> = v[..2 * half].to_vec();
    for (j, &inv) in table.inv.iter().enumerate() {
        let ang = pos as f32 * inv;
        let (cs, sn) = (ang.cos(), ang.sin());
        let (a, b) = (inp[2 * j], inp[2 * j + 1]);
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_rope(qk_rope: i64, theta: f32) -> Result<Cfg, peregrine_core::Error> {
        // minimal Cfg for RoPE tests — only qk_rope/theta are read
        let j = serde_json::json!({
            "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
            "n_routed_experts": 1, "num_experts_per_tok": 1, "moe_intermediate_size": 4,
            "intermediate_size": 4, "first_k_dense_replace": 0, "q_lora_rank": 0,
            "kv_lora_rank": 1, "qk_nope_head_dim": 1, "qk_rope_head_dim": qk_rope,
            "v_head_dim": 1, "n_shared_experts": 0, "vocab_size": 8, "n_group": 1,
            "topk_group": 1, "rope_parameters": {"rope_theta": theta}
        });
        Cfg::from_json(&j)
    }

    #[test]
    fn rmsnorm_unit_weight() {
        let x = [3.0f32, 4.0]; // mean(x²) = 12.5, rms = sqrt(12.5)
        let w = [1.0f32, 1.0];
        let mut out = [0f32; 2];
        rmsnorm(&mut out, &x, &w, 0.0);
        let r = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * r).abs() < 1e-6);
        assert!((out[1] - 4.0 * r).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn rope_pos0_is_deinterleave() -> Result<(), peregrine_core::Error> {
        // pos=0 → all angles 0 → cs=1,sn=0 → output is evens-then-odds
        let c = cfg_with_rope(4, 10000.0)?;
        let mut v = [10.0f32, 11.0, 12.0, 13.0];
        rope_interleave(&mut v, 0, &c);
        assert_eq!(v, [10.0, 12.0, 11.0, 13.0]); // [a0,a1 | b0,b1]
        Ok(())
    }

    #[test]
    fn softmax_degenerate_rows_stay_finite() {
        // Every key masked: the max is -inf, so `x - m` would be NaN. A NaN row
        // reaches argmax and locks greedy decode onto one token forever.
        let mut all_masked = [f32::NEG_INFINITY; 4];
        softmax(&mut all_masked);
        assert!(all_masked.iter().all(|v| v.is_finite()), "no NaNs: {all_masked:?}");
        assert!((all_masked.iter().sum::<f32>() - 1.0).abs() < 1e-6, "uniform fallback sums to 1");
        // Empty rows are a no-op rather than a panic.
        softmax(&mut []);
    }

    #[test]
    fn rmsnorm_inplace_matches_out_of_place() {
        let x = [3.0f32, -4.0, 0.5, 2.0];
        let w = [1.5f32, 0.25, -2.0, 1.0];
        let mut out = [0f32; 4];
        rmsnorm(&mut out, &x, &w, 1e-5);
        let mut inplace = x;
        rmsnorm_inplace(&mut inplace, &w, 1e-5);
        assert_eq!(out, inplace, "in-place RMSNorm must be bit-identical");
    }

    #[test]
    fn norms_ignore_mismatched_tensors() {
        // A short norm weight used to index out of bounds (an abort in release).
        let mut v = [1.0f32, 2.0, 3.0];
        rmsnorm_inplace(&mut v, &[1.0], 1e-5);
        assert_eq!(v, [1.0, 2.0, 3.0], "left untouched");
        let mut lv = [1.0f32, 2.0];
        layernorm(&mut lv, &[1.0], &[0.0], 1e-5);
        assert_eq!(lv, [1.0, 2.0]);
    }

    #[test]
    fn rope_table_matches_powf_path() -> Result<(), peregrine_core::Error> {
        let c = cfg_with_rope(8, 10000.0)?;
        let orig = [1.0f32, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
        let mut a = orig;
        rope_interleave(&mut a, 13, &c);
        let mut b = orig;
        rope_interleave_with(&mut b, 13, &RopeTable::from_cfg(&c));
        assert_eq!(a, b, "cached frequencies must be bit-identical");
        Ok(())
    }

    #[test]
    fn rope_preserves_pair_energy() -> Result<(), peregrine_core::Error> {
        // each (a,b) pair is a rotation → magnitude preserved
        let c = cfg_with_rope(4, 10000.0)?;
        let orig = [1.0f32, 2.0, -3.0, 0.5];
        let mut v = orig;
        rope_interleave(&mut v, 7, &c);
        let half = 2;
        for j in 0..half {
            let e_in = orig[2 * j].powi(2) + orig[2 * j + 1].powi(2);
            let e_out = v[j].powi(2) + v[half + j].powi(2);
            assert!((e_in - e_out).abs() < 1e-4, "pair {j}");
        }
        Ok(())
    }
}

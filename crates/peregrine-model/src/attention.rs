//! MLA (Multi-head Latent Attention) — `attention_rows` (`c/glm.c:2329-2638`).
//!
//! Two cores share one projection + KV-append front end:
//!   - [`mla_attention`] — dense reconstruction: rebuild `[k_nope|v]` for every
//!     cached position via `kv_b`, then standard scored attention.
//!   - [`mla_attention_absorb`] — the decode optimization: absorb `kv_b`'s
//!     `k_nope` rows into the query (`qabs`) and score against the latent `Lc`
//!     directly, and project the latent context average through `kv_b`'s value
//!     rows once — no per-token reconstruction. Algebraically identical to dense.
//!
//! DSA sparse selection is deferred (M5 remainder); attending over all cached
//! keys is exactly the "DSA selects everything" case the C engine validates as
//! reproducing dense attention.

use crate::math::{rmsnorm, rope_interleave, softmax};
use crate::weight::QtWeight;
use peregrine_core::Cfg;

/// Per-layer KV cache holding the compressed latents and roped keys. Positions
/// are appended in order (`pos == len`), matching sequential prefill/decode.
pub struct LayerKv {
    pub lc: Vec<f32>, // [len, kv_lora]
    pub rc: Vec<f32>, // [len, qk_rope]
    pub len: usize,
    kv_lora: usize,
    qk_rope: usize,
}

impl LayerKv {
    pub fn new(kv_lora: usize, qk_rope: usize) -> LayerKv {
        LayerKv { lc: Vec::new(), rc: Vec::new(), len: 0, kv_lora, qk_rope }
    }
    fn append(&mut self, pos: usize, lc_row: &[f32], rc_row: &[f32]) {
        assert_eq!(pos, self.len, "KV cache must be appended in position order");
        debug_assert_eq!(lc_row.len(), self.kv_lora);
        debug_assert_eq!(rc_row.len(), self.qk_rope);
        self.lc.extend_from_slice(lc_row);
        self.rc.extend_from_slice(rc_row);
        self.len += 1;
    }
}

/// The five projection weights of one attention block.
pub struct AttnWeights<'a> {
    pub q_a: &'a QtWeight,
    pub q_a_ln: &'a [f32],
    pub q_b: &'a QtWeight,
    pub kv_a: &'a QtWeight,
    pub kv_a_ln: &'a [f32],
    pub kv_b: &'a QtWeight,
    pub o: &'a QtWeight,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Shared front end: q/kv projections, per-head query RoPE, and appending the
/// compressed KV (`Lc` normalized latent, `Rc` roped key) for the new tokens.
/// Returns the roped queries `Q[s_n, H*qk_head]`.
fn project(w: &AttnWeights, x: &[f32], s_n: usize, pos_base: usize, cache: &mut LayerKv, c: &Cfg) -> Vec<f32> {
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let kvl = c.kv_lora as usize;
    let cw = kvl + qk_rope;
    let q_lora = c.q_lora as usize;

    let mut qr = w.q_a.apply_vec(x, s_n);
    for s in 0..s_n {
        let row = &mut qr[s * q_lora..s * q_lora + q_lora];
        let tmp: Vec<f32> = row.to_vec();
        rmsnorm(row, &tmp, w.q_a_ln, c.eps);
    }
    let mut q = w.q_b.apply_vec(&qr, s_n);
    let comp = w.kv_a.apply_vec(x, s_n);

    for s in 0..s_n {
        let pos = pos_base + s;
        for h in 0..h_n {
            let off = s * h_n * qh + h * qh + qk_nope;
            rope_interleave(&mut q[off..off + qk_rope], pos, c);
        }
        let cs = &comp[s * cw..s * cw + cw];
        let mut lc_row = cs[..kvl].to_vec();
        let tmp = lc_row.clone();
        rmsnorm(&mut lc_row, &tmp, w.kv_a_ln, c.eps);
        let mut rc_row = cs[kvl..cw].to_vec();
        rope_interleave(&mut rc_row, pos, c);
        cache.append(pos, &lc_row, &rc_row);
    }
    q
}

/// Dense core: reconstruct `[k_nope|v]` for all cached positions via `kv_b`,
/// then causal scored attention. Returns `ctx[s_n, H*v_head]`.
fn attend_dense(w: &AttnWeights, q: &[f32], s_n: usize, pos_base: usize, cache: &LayerKv, c: &Cfg) -> Vec<f32> {
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let vh = c.v_head as usize;
    let kvl = c.kv_lora as usize;
    let kvb_head = qk_nope + vh;

    let tk = cache.len;
    let kvb_all = w.kv_b.apply_vec(&cache.lc[..tk * kvl], tk);

    let mut ctx = vec![0f32; s_n * h_n * vh];
    for s in 0..s_n {
        let pos = pos_base + s;
        for h in 0..h_n {
            let qp = &q[s * h_n * qh + h * qh..s * h_n * qh + h * qh + qh];
            let (q_nope, q_rope) = qp.split_at(qk_nope);
            let nt = pos + 1;
            let mut sc = vec![0f32; nt];
            for t in 0..nt {
                let base = t * h_n * kvb_head + h * kvb_head;
                let kn = &kvb_all[base..base + qk_nope];
                let kr = &cache.rc[t * qk_rope..t * qk_rope + qk_rope];
                sc[t] = (dot(q_nope, kn) + dot(q_rope, kr)) * c.attn_scale;
            }
            softmax(&mut sc);
            let cx = &mut ctx[(s * h_n + h) * vh..(s * h_n + h) * vh + vh];
            for t in 0..nt {
                let base = t * h_n * kvb_head + h * kvb_head + qk_nope;
                let vv = &kvb_all[base..base + vh];
                let a = sc[t];
                for d in 0..vh {
                    cx[d] += a * vv[d];
                }
            }
        }
    }
    ctx
}

/// Absorb core: fold `kv_b`'s k_nope rows into the query, score against `Lc`
/// directly, and project the latent average through `kv_b`'s value rows.
fn attend_absorb(w: &AttnWeights, q: &[f32], s_n: usize, pos_base: usize, cache: &LayerKv, c: &Cfg) -> Vec<f32> {
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let vh = c.v_head as usize;
    let kvl = c.kv_lora as usize;

    let mut ctx = vec![0f32; s_n * h_n * vh];
    for s in 0..s_n {
        let pos = pos_base + s;
        for h in 0..h_n {
            let qp = &q[s * h_n * qh + h * qh..s * h_n * qh + h * qh + qh];
            let (q_nope, q_rope) = qp.split_at(qk_nope);
            let rbase = h * (qk_nope + vh);

            // qabs = Σ_d q_nope[d] · kv_b_row(rbase+d)   (absorb k_nope into q)
            let mut qabs = vec![0f32; kvl];
            for d in 0..qk_nope {
                let row = w.kv_b.dequant_row(rbase + d);
                for i in 0..kvl {
                    qabs[i] += q_nope[d] * row[i];
                }
            }

            let nt = pos + 1;
            let mut sc = vec![0f32; nt];
            for t in 0..nt {
                let lt = &cache.lc[t * kvl..t * kvl + kvl];
                let kr = &cache.rc[t * qk_rope..t * qk_rope + qk_rope];
                sc[t] = (dot(&qabs, lt) + dot(q_rope, kr)) * c.attn_scale;
            }
            softmax(&mut sc);

            // clat = Σ_t sc[t] · Lc[t]; ctx = kv_b value rows · clat
            let mut clat = vec![0f32; kvl];
            for t in 0..nt {
                let lt = &cache.lc[t * kvl..t * kvl + kvl];
                let a = sc[t];
                for i in 0..kvl {
                    clat[i] += a * lt[i];
                }
            }
            let cx = &mut ctx[(s * h_n + h) * vh..(s * h_n + h) * vh + vh];
            for j in 0..vh {
                let row = w.kv_b.dequant_row(rbase + qk_nope + j);
                cx[j] = dot(&row, &clat);
            }
        }
    }
    ctx
}

/// MLA attention (dense reconstruction). `s_n` new tokens from `pos_base`,
/// appended to `cache`; returns `out[s_n, hidden]`.
pub fn mla_attention(w: &AttnWeights, x: &[f32], s_n: usize, pos_base: usize, cache: &mut LayerKv, c: &Cfg) -> Vec<f32> {
    let q = project(w, x, s_n, pos_base, cache, c);
    let ctx = attend_dense(w, &q, s_n, pos_base, cache, c);
    w.o.apply_vec(&ctx, s_n)
}

/// MLA attention (weight absorption). Same result as [`mla_attention`], via the
/// decode-optimized latent-space core.
pub fn mla_attention_absorb(w: &AttnWeights, x: &[f32], s_n: usize, pos_base: usize, cache: &mut LayerKv, c: &Cfg) -> Vec<f32> {
    let q = project(w, x, s_n, pos_base, cache, c);
    let ctx = attend_absorb(w, &q, s_n, pos_base, cache, c);
    w.o.apply_vec(&ctx, s_n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::test_support::quant_i4;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn cfg() -> Result<Cfg, peregrine_core::Error> {
        let j = serde_json::json!({
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "intermediate_size": 8, "first_k_dense_replace": 1, "q_lora_rank": 12,
            "kv_lora_rank": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
            "v_head_dim": 6, "n_shared_experts": 1, "vocab_size": 32, "n_group": 1,
            "topk_group": 1, "rope_parameters": {"rope_theta": 10000.0}, "rms_norm_eps": 1e-5
        });
        Cfg::from_json(&j)
    }

    struct Weights {
        q_a: QtWeight,
        q_a_ln: Vec<f32>,
        q_b: QtWeight,
        kv_a: QtWeight,
        kv_a_ln: Vec<f32>,
        kv_b: QtWeight,
        o: QtWeight,
    }
    impl Weights {
        fn view(&self) -> AttnWeights<'_> {
            AttnWeights {
                q_a: &self.q_a,
                q_a_ln: &self.q_a_ln,
                q_b: &self.q_b,
                kv_a: &self.kv_a,
                kv_a_ln: &self.kv_a_ln,
                kv_b: &self.kv_b,
                o: &self.o,
            }
        }
    }

    fn make_weights(c: &Cfg, seed: u64) -> Weights {
        let mut r = Lcg(seed);
        let (hidden, h, qh, vh) = (c.hidden as usize, c.n_heads as usize, c.qk_head as usize, c.v_head as usize);
        let (ql, kvl, qkr, qkn) = (c.q_lora as usize, c.kv_lora as usize, c.qk_rope as usize, c.qk_nope as usize);
        let w = |r: &mut Lcg, n: usize| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
        Weights {
            q_a: quant_i4(&w(&mut r, ql * hidden), ql, hidden),
            q_a_ln: (0..ql).map(|_| 0.5 + r.f() * 0.1).collect(),
            q_b: quant_i4(&w(&mut r, h * qh * ql), h * qh, ql),
            kv_a: quant_i4(&w(&mut r, (kvl + qkr) * hidden), kvl + qkr, hidden),
            kv_a_ln: (0..kvl).map(|_| 0.5 + r.f() * 0.1).collect(),
            kv_b: quant_i4(&w(&mut r, h * (qkn + vh) * kvl), h * (qkn + vh), kvl),
            o: quant_i4(&w(&mut r, hidden * h * vh), hidden, h * vh),
        }
    }

    fn new_cache(c: &Cfg) -> LayerKv {
        LayerKv::new(c.kv_lora as usize, c.qk_rope as usize)
    }

    #[test]
    fn single_token_is_value_projection() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 1);
        let (h, qkn, vh, kvl) = (c.n_heads as usize, c.qk_nope as usize, c.v_head as usize, c.kv_lora as usize);
        let mut r = Lcg(555);
        let x: Vec<f32> = (0..c.hidden as usize).map(|_| r.f()).collect();
        let mut cache = new_cache(&c);
        let out = mla_attention(&w.view(), &x, 1, 0, &mut cache, &c);
        let kvb = w.kv_b.apply_vec(&cache.lc[..kvl], 1);
        let kvb_head = qkn + vh;
        let mut v = vec![0f32; h * vh];
        for hh in 0..h {
            let base = hh * kvb_head + qkn;
            v[hh * vh..hh * vh + vh].copy_from_slice(&kvb[base..base + vh]);
        }
        let expect = w.o.apply_vec(&v, 1);
        for d in 0..c.hidden as usize {
            assert!((out[d] - expect[d]).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn attention_is_causal() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 2);
        let hidden = c.hidden as usize;
        let s_n = 4;
        let mut r = Lcg(999);
        let mut x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut ca = new_cache(&c);
        let out_a = mla_attention(&w.view(), &x, s_n, 0, &mut ca, &c);
        for d in 0..hidden {
            x[(s_n - 1) * hidden + d] += 1.0;
        }
        let mut cb = new_cache(&c);
        let out_b = mla_attention(&w.view(), &x, s_n, 0, &mut cb, &c);
        for p in 0..s_n - 1 {
            for d in 0..hidden {
                assert!((out_a[p * hidden + d] - out_b[p * hidden + d]).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn decode_step_matches_prefill() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 3);
        let hidden = c.hidden as usize;
        let mut r = Lcg(24680);
        let x: Vec<f32> = (0..4 * hidden).map(|_| r.f()).collect();
        let mut full = new_cache(&c);
        let out_full = mla_attention(&w.view(), &x, 4, 0, &mut full, &c);
        let mut inc = new_cache(&c);
        let _ = mla_attention(&w.view(), &x[..3 * hidden], 3, 0, &mut inc, &c);
        let out_dec = mla_attention(&w.view(), &x[3 * hidden..4 * hidden], 1, 3, &mut inc, &c);
        for d in 0..hidden {
            assert!((out_full[3 * hidden + d] - out_dec[d]).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn absorb_approximates_dense() -> Result<(), peregrine_core::Error> {
        // absorb is an algebraic rearrangement of dense; they differ only by
        // kv_b activation quantization in the dense reconstruction.
        let c = cfg()?;
        let w = make_weights(&c, 7);
        let hidden = c.hidden as usize;
        let s_n = 5;
        let mut r = Lcg(31415);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut cd = new_cache(&c);
        let dense = mla_attention(&w.view(), &x, s_n, 0, &mut cd, &c);
        let mut cabs = new_cache(&c);
        let absorb = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut cabs, &c);
        for z in 0..s_n * hidden {
            let scale = dense[z].abs().max(1.0);
            assert!((dense[z] - absorb[z]).abs() < 0.1 * scale, "z={z} dense={} absorb={}", dense[z], absorb[z]);
        }
        Ok(())
    }

    #[test]
    fn absorb_is_causal() -> Result<(), peregrine_core::Error> {
        // the absorb core must be causal too (exact, no tolerance)
        let c = cfg()?;
        let w = make_weights(&c, 8);
        let hidden = c.hidden as usize;
        let s_n = 4;
        let mut r = Lcg(271);
        let mut x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut ca = new_cache(&c);
        let a = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut ca, &c);
        for d in 0..hidden {
            x[(s_n - 1) * hidden + d] += 1.0;
        }
        let mut cb = new_cache(&c);
        let b = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut cb, &c);
        for p in 0..s_n - 1 {
            for d in 0..hidden {
                assert!((a[p * hidden + d] - b[p * hidden + d]).abs() < 1e-6);
            }
        }
        Ok(())
    }
}

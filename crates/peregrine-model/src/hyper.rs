//! mHC hyper-connections (Manifold-Constrained Hyper-Connections, Xie et al.
//! 2026) — the Glm5Next residual stream. Instead of one residual vector per
//! position, the stack carries `hc_mult` parallel streams `[H, d]`; every
//! attention/FFN site owns a learned mapping that (a) collapses the streams
//! into the single sublayer input, and (b) places the sublayer output back
//! while mixing the streams through a Sinkhorn-normalized (approximately
//! doubly-stochastic) H×H matrix.
//!
//! Math per site, per position (HF `DeepseekV4HyperConnection.forward`,
//! extracted 2026-08-27; GLM-5.3-Flash inherits it unchanged):
//!
//! ```text
//! flat   = UnweightedRMSNorm(streams.flatten())            # [H*d], fp32
//! logits = fn · flat                                       # [(2+H)*H]
//! pre    = σ(logits[..H]      · scale₀ + base[..H])   + ε
//! post   = 2σ(logits[H..2H]   · scale₁ + base[H..2H])
//! comb   = softmax₋₁(logits[2H..]·scale₂ + base[2H..]) + ε # [H, H]
//! comb   = colnorm(comb); repeat (iters-1): rownorm, colnorm
//! x      = Σ_h pre[h] · streams[h]                         # sublayer input
//! ...sublayer runs on x...
//! streams'[h] = post[h] · sublayer_out + Σ_g comb[g][h] · streams[g]
//! ```
//!
//! Everything here runs in f32 — the engine's native precision — where the HF
//! reference downcasts `post`/`comb` to bf16 before applying them. That is a
//! (strictly finer) numerical difference the flip-rate gate measures, not a
//! structural one.

use crate::math::sigmoidf;
use crate::weight::QtWeight;
use peregrine_core::{Cfg, Error, SafeTensors};

/// One site's mHC parameters (a layer has two sites: attention and FFN).
pub struct HyperConn {
    /// `fn`: `[(2+H)*H, H*hidden]`, stored int8 + `.qs` in the container
    /// (small and gate-critical — its logits pass through σ/softmax, so it gets
    /// the embed-grade precision, not the int4 the big projections take).
    fnw: QtWeight,
    /// `base`: `[(2+H)*H]`.
    base: Vec<f32>,
    /// `scale`: `[3]` — one learned scale per output (pre, post, comb).
    scale: Vec<f32>,
    hc: usize,
}

/// One position's mixing weights, ready to apply.
pub struct HcMix {
    /// Stream-collapse weights `[H]` (σ + ε, strictly positive).
    pub pre: Vec<f32>,
    /// Block-output placement weights `[H]`, range (0, 2).
    pub post: Vec<f32>,
    /// Stream mixer `[H, H]` row-major `[src][dst]` after Sinkhorn.
    pub comb: Vec<f32>,
}

impl HyperConn {
    /// Load one site's tensors (`{prefix}{site}_base` / `_fn` / `_scale`),
    /// `Ok(None)` when the site is absent (a conventional-residual layer, e.g.
    /// the MTP head layer, which the checkpoint ships without hc tensors).
    pub fn load(st: &SafeTensors, prefix: &str, site: &str, cfg: &Cfg) -> Result<Option<HyperConn>, Error> {
        let fn_name = format!("{prefix}hc_{site}_fn");
        if !st.has(&fn_name) {
            return Ok(None);
        }
        let hc = cfg.hc_mult.max(1) as usize;
        let mix = (2 + hc) * hc;
        let d = cfg.hidden as usize;
        let mut base = vec![0f32; mix];
        st.read_f32(&format!("{prefix}hc_{site}_base"), &mut base)?;
        let mut scale = vec![0f32; 3];
        st.read_f32(&format!("{prefix}hc_{site}_scale"), &mut scale)?;
        Ok(Some(HyperConn { fnw: QtWeight::load(st, &fn_name, mix, hc * d)?, base, scale, hc }))
    }

    /// Compute one position's mixing weights from its streams (`[H*d]`).
    pub fn mix(&self, streams: &[f32], cfg: &Cfg) -> HcMix {
        let hc = self.hc;
        debug_assert_eq!(streams.len() % hc, 0);
        // Unweighted RMSNorm over the whole flattened width.
        let n = streams.len() as f32;
        let ms = streams.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ms + cfg.eps).sqrt();
        let flat: Vec<f32> = streams.iter().map(|v| v * inv).collect();
        let logits = self.fnw.apply_vec(&flat, 1); // [(2+H)*H]

        let eps = cfg.hc_eps;
        let pre: Vec<f32> =
            (0..hc).map(|h| sigmoidf(logits[h] * self.scale[0] + self.base[h]) + eps).collect();
        let post: Vec<f32> =
            (0..hc).map(|h| 2.0 * sigmoidf(logits[hc + h] * self.scale[1] + self.base[hc + h])).collect();

        // comb: softmax over each row (destination axis), + ε, then Sinkhorn.
        let mut comb = vec![0f32; hc * hc];
        for i in 0..hc {
            let row = &mut comb[i * hc..(i + 1) * hc];
            let mut mx = f32::NEG_INFINITY;
            for (j, r) in row.iter_mut().enumerate() {
                let z = 2 * hc + i * hc + j;
                *r = logits[z] * self.scale[2] + self.base[z];
                mx = mx.max(*r);
            }
            let mut sum = 0.0f32;
            for r in row.iter_mut() {
                *r = (*r - mx).exp();
                sum += *r;
            }
            for r in row.iter_mut() {
                *r = *r / sum + eps;
            }
        }
        // Column normalize first, then (iters-1) × (row, column) — the exact
        // reference order.
        let colnorm = |m: &mut [f32]| {
            for j in 0..hc {
                let mut s = 0.0f32;
                for i in 0..hc {
                    s += m[i * hc + j];
                }
                let inv = 1.0 / (s + eps);
                for i in 0..hc {
                    m[i * hc + j] *= inv;
                }
            }
        };
        let rownorm = |m: &mut [f32]| {
            for i in 0..hc {
                let row = &mut m[i * hc..(i + 1) * hc];
                let s: f32 = row.iter().sum();
                let inv = 1.0 / (s + eps);
                for r in row.iter_mut() {
                    *r *= inv;
                }
            }
        };
        colnorm(&mut comb);
        for _ in 1..cfg.hc_sinkhorn.max(1) {
            rownorm(&mut comb);
            colnorm(&mut comb);
        }
        HcMix { pre, post, comb }
    }
}

impl HcMix {
    /// Collapse the streams into the sublayer input: `Σ_h pre[h]·streams[h]`.
    pub fn collapse(&self, streams: &[f32], d: usize, out: &mut [f32]) {
        out[..d].fill(0.0);
        for (h, &p) in self.pre.iter().enumerate() {
            let s = &streams[h * d..(h + 1) * d];
            for (o, &v) in out[..d].iter_mut().zip(s) {
                *o += p * v;
            }
        }
    }

    /// Place the sublayer output and mix the residual streams:
    /// `streams'[h] = post[h]·y + Σ_g comb[g][h]·streams[g]`, in place.
    pub fn place(&self, streams: &mut [f32], y: &[f32], d: usize, scratch: &mut Vec<f32>) {
        let hc = self.pre.len();
        scratch.clear();
        scratch.extend_from_slice(&streams[..hc * d]);
        for h in 0..hc {
            let dst = &mut streams[h * d..(h + 1) * d];
            let ph = self.post[h];
            for (o, &yv) in dst.iter_mut().zip(y) {
                *o = ph * yv;
            }
            for g in 0..hc {
                let c = self.comb[g * hc + h];
                let src = &scratch[g * d..(g + 1) * d];
                for (o, &v) in dst.iter_mut().zip(src) {
                    *o += c * v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::test_support::quant_i8;

    fn tiny_cfg() -> Result<Cfg, Error> {
        Cfg::from_json(&serde_json::json!({
            "model_type": "glm5_next_text",
            "vocab_size": 32, "hidden_size": 4,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 8, "intermediate_size": 8,
            "mlp_layer_types": ["sparse"],
            "layer_types": ["deepseek_sparse_attention"],
            "q_lora_rank": 12, "kv_lora_rank": 8,
            "qk_nope_head_dim": 4, "qk_rope_head_dim": 0, "v_head_dim": 4,
            "n_shared_experts": 1, "norm_topk_prob": true,
            "rms_norm_eps": 1e-5, "swiglu_limit": 10.0,
            "hc_mult": 3, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "eos_token_id": 0
        }))
    }

    fn make_hc(cfg: &Cfg, seed: u64) -> HyperConn {
        let hc = cfg.hc_mult as usize;
        let d = cfg.hidden as usize;
        let mix = (2 + hc) * hc;
        let mut s = seed;
        let mut f = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let fnw: Vec<f32> = (0..mix * hc * d).map(|_| f() * 0.5).collect();
        HyperConn {
            fnw: quant_i8(&fnw, mix, hc * d),
            base: (0..mix).map(|_| f() * 0.5).collect(),
            scale: vec![1.0 + f() * 0.1, 1.0 + f() * 0.1, 1.0 + f() * 0.1],
            hc,
        }
    }

    #[test]
    fn comb_is_approximately_doubly_stochastic_after_sinkhorn() -> Result<(), Error> {
        let cfg = tiny_cfg()?;
        let hcw = make_hc(&cfg, 3);
        let hc = cfg.hc_mult as usize;
        let d = cfg.hidden as usize;
        let streams: Vec<f32> = (0..hc * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let m = hcw.mix(&streams, &cfg);
        for i in 0..hc {
            let row: f32 = m.comb[i * hc..(i + 1) * hc].iter().sum();
            assert!((row - 1.0).abs() < 1e-3, "row {i} sums to {row}");
        }
        for j in 0..hc {
            let col: f32 = (0..hc).map(|i| m.comb[i * hc + j]).sum();
            assert!((col - 1.0).abs() < 1e-3, "col {j} sums to {col}");
        }
        assert!(m.comb.iter().all(|&v| v > 0.0), "strictly positive mixing");
        assert!(m.pre.iter().all(|&v| v > 0.0));
        assert!(m.post.iter().all(|&v| v > 0.0 && v < 2.0));
        Ok(())
    }

    #[test]
    fn place_with_identity_comb_and_zero_output_preserves_streams() -> Result<(), Error> {
        let cfg = tiny_cfg()?;
        let hc = cfg.hc_mult as usize;
        let d = cfg.hidden as usize;
        let mut ident = vec![0f32; hc * hc];
        for i in 0..hc {
            ident[i * hc + i] = 1.0;
        }
        let m = HcMix { pre: vec![1.0; hc], post: vec![0.5; hc], comb: ident };
        let orig: Vec<f32> = (0..hc * d).map(|i| i as f32).collect();
        let mut streams = orig.clone();
        let y = vec![0f32; d];
        let mut scratch = Vec::new();
        m.place(&mut streams, &y, d, &mut scratch);
        assert!(streams.iter().zip(&orig).all(|(a, b)| (a - b).abs() < 1e-7));
        // And collapse with pre = 1 is the plain stream sum.
        let mut x = vec![0f32; d];
        m.collapse(&streams, d, &mut x);
        for j in 0..d {
            let want: f32 = (0..hc).map(|h| streams[h * d + j]).sum();
            assert!((x[j] - want).abs() < 1e-5);
        }
        Ok(())
    }
}

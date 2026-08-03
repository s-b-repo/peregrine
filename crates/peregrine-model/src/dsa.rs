//! DSA lightning indexer (M5): per-query key selection for sparse long-context
//! attention. Faithful to colibrì's indexer (`glm.c` `moe`/indexer path). The
//! indexer scores every cached key against the query and keeps the top
//! `index_topk`; [`crate::attention::mla_attention_dsa`] then attends only those.
//!
//! Activation: the C engine runs the indexer only once the context exceeds
//! `index_topk` (below that, attention is already dense over ≤ index_topk keys,
//! so selection is a no-op). The indexer weights ship in checkpoints converted
//! with `--indexer`; this module implements + unit-tests the scoring and top-k
//! selection, which compose with the (separately tested) sparse attend path.

use peregrine_core::{Cfg, Error, SafeTensors};

use crate::math::rope_interleave;
use crate::weight::QtWeight;

/// Score cached keys for one query: `s[t] = (1/√nh) · Σ_h w[h]·ReLU((q_h·k_t)/√hd)`.
/// `qi` is the roped per-head query `[nh*hd]`, `w` the per-head weights `[nh]`,
/// `keys` the cached indexer keys `[nkeys*hd]`. This is the exact colibrì scoring.
pub fn score_keys(qi: &[f32], w: &[f32], keys: &[f32], nh: usize, hd: usize) -> Vec<f32> {
    let nkeys = keys.len() / hd;
    let rs = 1.0 / (hd as f32).sqrt();
    let wsc = 1.0 / (nh as f32).sqrt();
    let mut out = vec![0f32; nkeys];
    for (t, s) in out.iter_mut().enumerate() {
        let kt = &keys[t * hd..t * hd + hd];
        let mut a = 0.0f32;
        for h in 0..nh {
            let qh = &qi[h * hd..h * hd + hd];
            let d0: f32 = qh.iter().zip(kt).map(|(&x, &y)| x * y).sum::<f32>() * rs;
            if d0 > 0.0 {
                a += w[h] * d0; // ReLU on the per-head score before weighting
            }
        }
        *s = a * wsc;
    }
    out
}

/// Indices of the top-`k` scores (highest first, ties broken by lower index),
/// returned in **ascending** order for causal-order attention. `k >= len` keeps
/// all indices (the dense case).
pub fn select_topk(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    if k >= scores.len() {
        return idx;
    }
    idx.sort_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let mut sel: Vec<usize> = idx.into_iter().take(k).collect();
    sel.sort_unstable();
    sel
}

/// Per-layer lightning indexer weights: its own q/k/weight projections and key
/// LayerNorm. Present only when the checkpoint carries
/// `model.layers.{i}.self_attn.indexer*` tensors.
///
/// **Weights only — no cache.** These were one struct, which made the indexer
/// structurally single-sequence: `keys`/`len` are per-*sequence* state, so one
/// indexer per layer on a `Model` serving concurrent requests would have let
/// two sequences interleave keys into one buffer. That is silent cross-sequence
/// corruption, not a crash, and nothing downstream would have caught it. The
/// cache now lives beside the KV latents in `LayerKv`, which already has
/// exactly the lifecycle it needs: append in order, truncate on a speculative
/// rewind, share a common prefix by refcount.
pub struct IndexerWeights {
    wq: QtWeight,       // [nh*hd, q_lora]  query projection (from the q-LoRA)
    wk: QtWeight,       // [hd, hidden]     key projection (from the hidden)
    wp: QtWeight,       // [nh, hidden]     per-head weights projection
    k_norm_w: Vec<f32>, // [hd] key LayerNorm weight
    k_norm_b: Vec<f32>, // [hd] key LayerNorm bias
    nh: usize,
    hd: usize,
    topk: usize,
}

impl IndexerWeights {
    /// Load the indexer for `layer` if present; `Ok(None)` otherwise.
    pub fn load(st: &SafeTensors, layer: usize, cfg: &Cfg) -> Result<Option<IndexerWeights>, Error> {
        let base = format!("model.layers.{layer}.self_attn");
        let wqn = format!("{base}.indexer_projections.wq_b");
        if !st.has(&wqn) {
            return Ok(None);
        }
        let (nh, hd) = (cfg.index_nh as usize, cfg.index_hd as usize);
        let (hidden, ql) = (cfg.hidden as usize, cfg.q_lora as usize);
        let mut k_norm_w = vec![0f32; hd];
        let mut k_norm_b = vec![0f32; hd];
        st.read_f32(&format!("{base}.indexer.k_norm.weight"), &mut k_norm_w)?;
        st.read_f32(&format!("{base}.indexer.k_norm.bias"), &mut k_norm_b)?;
        Ok(Some(IndexerWeights {
            wq: QtWeight::load(st, &wqn, nh * hd, ql)?,
            wk: QtWeight::load(st, &format!("{base}.indexer_projections.wk"), hd, hidden)?,
            wp: QtWeight::load(st, &format!("{base}.indexer_projections.weights_proj"), nh, hidden)?,
            k_norm_w,
            k_norm_b,
            nh,
            hd,
            topk: cfg.index_topk.max(0) as usize,
        }))
    }

    /// Width of one cached indexer key.
    pub fn hd(&self) -> usize {
        self.hd
    }

    /// How many keys a query keeps. Selection is a no-op at or below this
    /// context length — attention is already dense over that many keys — which
    /// is the activation rule the C engine uses.
    pub fn topk(&self) -> usize {
        self.topk
    }

    /// Project, LayerNorm (eps 1e-6) and RoPE one position's key. The caller
    /// caches it; this borrows nothing mutable, so a layer's weights stay
    /// shareable across concurrent sequences.
    pub fn key_row(&self, x_row: &[f32], pos: usize, cfg: &Cfg) -> Vec<f32> {
        let mut k = self.wk.apply_vec(x_row, 1); // [hd]
        layernorm_affine(&mut k, &self.k_norm_w, &self.k_norm_b, 1e-6);
        rope_interleave(&mut k, pos, cfg);
        k
    }

    /// Select the top-`index_topk` key indices for a query at `pos`, over the
    /// caller's cached `keys` (`[nkeys*hd]`, causal order). `qr` is the query's
    /// **post-`rmsnorm`** q-LoRA row `[q_lora]` — the same tensor `q_b` consumes
    /// — and `x_row` its hidden `[hidden]`.
    pub fn select(&self, qr: &[f32], x_row: &[f32], pos: usize, cfg: &Cfg, keys: &[f32]) -> Vec<usize> {
        let (nh, hd) = (self.nh, self.hd);
        let mut qi = self.wq.apply_vec(qr, 1); // [nh*hd]
        for h in 0..nh {
            rope_interleave(&mut qi[h * hd..h * hd + hd], pos, cfg);
        }
        let w = self.wp.apply_vec(x_row, 1); // [nh]
        let nt = (pos + 1).min(keys.len().checked_div(hd).unwrap_or(0));
        let scores = score_keys(&qi, &w, &keys[..nt * hd], nh, hd);
        select_topk(&scores, self.topk)
    }
}

/// In-place affine LayerNorm over a single vector (mean/var across all elements).
fn layernorm_affine(v: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..v.len() {
        v[i] = (v[i] - mean) * inv * w[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_picks_highest_and_is_causal_ordered() {
        let scores = [0.1, 0.9, 0.3, 0.7, 0.2];
        // top-2 by value are indices 1 (0.9) and 3 (0.7), returned ascending
        assert_eq!(select_topk(&scores, 2), vec![1, 3]);
        // k >= len keeps all, in order
        assert_eq!(select_topk(&scores, 9), vec![0, 1, 2, 3, 4]);
        // ties break toward the lower index
        assert_eq!(select_topk(&[0.5, 0.5, 0.1], 1), vec![0]);
    }

    #[test]
    fn score_keys_relu_weighted_dot() {
        // one head, hd=2: score = w0 * ReLU(q·k / √2)
        let qi = [1.0f32, 0.0];
        let w = [2.0f32];
        let keys = [1.0f32, 0.0, -1.0, 0.0, 0.0, 1.0]; // k0=+q dir, k1=-q dir, k2 orthogonal
        let s = score_keys(&qi, &w, &keys, 1, 2);
        let rs = 1.0 / 2f32.sqrt();
        assert!((s[0] - 2.0 * rs).abs() < 1e-6); // positive dot → weighted
        assert_eq!(s[1], 0.0); // negative dot → ReLU zero
        assert_eq!(s[2], 0.0); // orthogonal → zero
    }
}

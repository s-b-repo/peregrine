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

fn cmp_desc(a: f32, ia: usize, b: f32, ib: usize) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
    }
    .then(ia.cmp(&ib))
}

/// Indices of the top-`k` scores (highest first, ties broken by lower index),
/// returned in **ascending** order for causal-order attention. `k >= len` keeps
/// all indices (the dense case).
pub fn select_topk(scores: &[f32], k: usize) -> Vec<usize> {
    let n = scores.len();
    if k == 0 {
        return Vec::new();
    }
    if k >= n {
        return (0..n).collect();
    }
    let block = k.max(1024);
    let limit = k.saturating_add(block).min(n);
    let compare = |&a: &usize, &b: &usize| cmp_desc(scores[a], a, scores[b], b);
    let mut cand = Vec::with_capacity(limit);
    cand.extend(0..k);
    let mut threshold = *cand.select_nth_unstable_by(k - 1, compare).1;
    for i in k..n {
        if compare(&i, &threshold).is_lt() {
            cand.push(i);
            if cand.len() == limit {
                threshold = *cand.select_nth_unstable_by(k - 1, compare).1;
                cand.truncate(k);
            }
        }
    }
    if cand.len() > k {
        cand.select_nth_unstable_by(k - 1, compare);
        cand.truncate(k);
    }
    cand.sort_unstable();
    cand
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
    /// Glm5Next k-pool compression: `Some` scores pools of `kpool` consecutive
    /// tokens (learned per-channel within-pool weighting) instead of tokens.
    kpool: Option<KpoolWeights>,
}

/// The k-pool half of a Glm5Next indexer: pool size, the learned gate that
/// produces per-token pooling logits, and the pool-slot embedding.
struct KpoolWeights {
    /// Pool width in tokens (`index_kpool`).
    k: usize,
    /// Whether the incomplete tail pool's tokens are always selected.
    tail: bool,
    /// `index_kpool_compress_gate`: `[hd, hidden]` — per-token pooling logits.
    gate: QtWeight,
    /// `index_kpool_compress_ape`: `[k, hd]` — additive per-slot embedding.
    ape: Vec<f32>,
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
        // k-pool compression: required when the config declares it — an indexer
        // silently falling back to per-token scoring would select different
        // keys than the checkpoint was trained to.
        let kpool = if cfg.index_kpool > 0 && cfg.arch == peregrine_core::Arch::Glm5Next {
            let gate_name = format!("{base}.indexer.index_kpool_compress_gate");
            if !st.has(&gate_name) {
                return Err(Error::Format(format!(
                    "config declares index_kpool={} but {gate_name} is missing",
                    cfg.index_kpool
                )));
            }
            let k = cfg.index_kpool as usize;
            let mut ape = vec![0f32; k * hd];
            st.read_f32(&format!("{base}.indexer.index_kpool_compress_ape"), &mut ape)?;
            Some(KpoolWeights {
                k,
                tail: cfg.index_kpool_tail,
                gate: QtWeight::load(st, &gate_name, hd, hidden)?,
                ape,
            })
        } else {
            None
        };
        Ok(Some(IndexerWeights {
            wq: QtWeight::load(st, &wqn, nh * hd, ql)?,
            wk: QtWeight::load(st, &format!("{base}.indexer_projections.wk"), hd, hidden)?,
            wp: QtWeight::load(st, &format!("{base}.indexer_projections.weights_proj"), nh, hidden)?,
            k_norm_w,
            k_norm_b,
            nh,
            hd,
            topk: cfg.index_topk.max(0) as usize,
            kpool,
        }))
    }

    /// Width of one cached indexer key.
    pub fn hd(&self) -> usize {
        self.hd
    }

    /// Width of one cached indexer **row** — `hd` for the per-token indexer,
    /// `2*hd` for the k-pool form (key ‖ pooling gate logits). This, not
    /// [`Self::hd`], is what sizes reads of the `LayerKv` index stream.
    pub fn row_width(&self) -> usize {
        if self.kpool.is_some() {
            2 * self.hd
        } else {
            self.hd
        }
    }

    /// How many keys a query keeps. Selection is a no-op at or below this
    /// context length — attention is already dense over that many keys — which
    /// is the activation rule the C engine uses. (Exact for the k-pool form
    /// too: at `nt <= topk` every complete pool fits the pool budget and the
    /// tail is appended, so the selection is all of `0..nt`.)
    pub fn topk(&self) -> usize {
        self.topk
    }

    /// Project, LayerNorm (eps 1e-6) and RoPE one position's key. The caller
    /// caches it; this borrows nothing mutable, so a layer's weights stay
    /// shareable across concurrent sequences. In k-pool form the row is the
    /// key with the pooling gate logits appended (`[2*hd]`) — both are
    /// per-token quantities that later selections need for every past position.
    pub fn key_row(&self, x_row: &[f32], pos: usize, cfg: &Cfg) -> Vec<f32> {
        let mut k = self.wk.apply_vec(x_row, 1); // [hd]
        layernorm_affine(&mut k, &self.k_norm_w, &self.k_norm_b, 1e-6);
        rope_interleave(&mut k, pos, cfg); // no-op under NoPE (qk_rope = 0)
        if let Some(kp) = &self.kpool {
            k.extend(kp.gate.apply_vec(x_row, 1)); // [hd] gate logits
        }
        k
    }

    /// Select the cached key indices a query at `pos` attends, over the
    /// caller's cached `keys` (`[nkeys*row_width]`, causal order). `qr` is the
    /// query's **post-`rmsnorm`** q-LoRA row `[q_lora]` — the same tensor `q_b`
    /// consumes — and `x_row` its hidden `[hidden]`.
    pub fn select(&self, qr: &[f32], x_row: &[f32], pos: usize, cfg: &Cfg, keys: &[f32]) -> Vec<usize> {
        let (nh, hd) = (self.nh, self.hd);
        let mut qi = self.wq.apply_vec(qr, 1); // [nh*hd]
        for h in 0..nh {
            rope_interleave(&mut qi[h * hd..h * hd + hd], pos, cfg);
        }
        let w = self.wp.apply_vec(x_row, 1); // [nh]
        let rw = self.row_width();
        let nt = (pos + 1).min(keys.len().checked_div(rw).unwrap_or(0));
        let Some(kp) = &self.kpool else {
            let scores = score_keys(&qi, &w, &keys[..nt * hd], nh, hd);
            return select_topk(&scores, self.topk);
        };

        // --- k-pool form -------------------------------------------------
        // 1. Compress each complete pool of `k` consecutive tokens into one
        //    key: per *channel*, softmax over the pool's (gate logit + slot
        //    embedding), then the weighted sum of the member keys.
        let k = kp.k;
        let n_full = nt / k;
        let mut pool_keys = vec![0f32; n_full * hd];
        let mut probs = vec![0f32; k];
        for p in 0..n_full {
            for c in 0..hd {
                let mut mx = f32::NEG_INFINITY;
                for (slot, pr) in probs.iter_mut().enumerate() {
                    *pr = keys[(p * k + slot) * rw + hd + c] + kp.ape[slot * hd + c];
                    mx = mx.max(*pr);
                }
                let mut sum = 0f32;
                for pr in probs.iter_mut() {
                    *pr = (*pr - mx).exp();
                    sum += *pr;
                }
                let inv = 1.0 / sum;
                let mut acc = 0f32;
                for (slot, pr) in probs.iter().enumerate() {
                    acc += (pr * inv) * keys[(p * k + slot) * rw + c];
                }
                pool_keys[p * hd + c] = acc;
            }
        }
        // 2. Score pools exactly as the per-token form scores keys, keep the
        //    top `topk/k` pools, expand each back to its member tokens.
        let budget = (self.topk / k).max(1);
        let sel_pools = if n_full > 0 {
            let scores = score_keys(&qi, &w, &pool_keys, nh, hd);
            select_topk(&scores, budget)
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(sel_pools.len() * k + k);
        for p in sel_pools {
            out.extend(p * k..(p + 1) * k);
        }
        // 3. The incomplete tail pool's tokens ride along (the query's own
        //    position lives there), so recent context is never invisible.
        if kp.tail {
            out.extend(n_full * k..nt);
        }
        out.sort_unstable();
        out
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

    fn oracle(scores: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| {
            if scores[a].is_nan() && !scores[b].is_nan() {
                std::cmp::Ordering::Greater
            } else if scores[b].is_nan() && !scores[a].is_nan() {
                std::cmp::Ordering::Less
            } else {
                scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            }
        });
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }

    fn assert_matches_oracle(scores: &[f32], k: usize) {
        let selected = select_topk(scores, k);
        assert_eq!(selected, oracle(scores, k), "n={} k={k}", scores.len());
        if k == 0 {
            assert_eq!(selected.capacity(), 0);
        } else if k < scores.len() {
            assert!(selected.capacity() <= k.saturating_add(k.max(1024)).min(scores.len()));
        }
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 40) as f32 / (1u32 << 24) as f32
    }

    #[test]
    fn topk_edge_cases() {
        assert_eq!(select_topk(&[0.1, 0.9, 0.3, 0.7, 0.2], 2), vec![1, 3]);
        assert_eq!(select_topk(&[0.1, 0.9, 0.3, 0.7, 0.2], 9), vec![0, 1, 2, 3, 4]);
        assert_eq!(select_topk(&[0.5, 0.5, 0.1], 1), vec![0]);
        assert_eq!(select_topk(&[], 0), Vec::<usize>::new());
        assert_eq!(select_topk(&[], 5), Vec::<usize>::new());
        assert_eq!(select_topk(&[1.0], 0), Vec::<usize>::new());
        assert_eq!(select_topk(&[1.0], 1), vec![0]);
        assert_eq!(select_topk(&[3.0, 1.0, 2.0], 1), vec![0]);
        let n = 4096usize;
        let scores: Vec<f32> = (0..n).map(|x| x as f32 / 7.0).collect();
        assert_matches_oracle(&scores, 0);
        assert_matches_oracle(&scores, 1);
        assert_matches_oracle(&scores, n - 1);
        assert_matches_oracle(&scores, n);
        assert_matches_oracle(&scores, 17);
    }

    #[test]
    fn topk_all_ties() {
        let n = 513;
        let scores = vec![0.0f32; n];
        for k in [0, 1, 2, 100, n - 1, n] {
            assert_matches_oracle(&scores, k);
        }
        let ties = vec![-1.5f32; 200];
        assert_eq!(select_topk(&ties, 3), vec![0, 1, 2]);
    }

    #[test]
    fn topk_sorted_and_reverse() {
        let asc: Vec<f32> = (0..300).map(|i| i as f32).collect();
        assert_matches_oracle(&asc, 50);
        let mut rev = asc.clone();
        rev.reverse();
        assert_matches_oracle(&rev, 50);
    }

    #[test]
    fn topk_special_floats() {
        let scores = [f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 1.0, -1.0];
        assert_matches_oracle(&scores, 3);
        assert_matches_oracle(&scores, 6);
        assert_matches_oracle(&scores, 0);
        let zs = [0.0f32, -0.0];
        assert_eq!(select_topk(&zs, 1), vec![0]);
        let zs2 = [-0.0f32, 0.0];
        assert_eq!(select_topk(&zs2, 1), vec![0]);
        let nn = [f32::NAN, f32::NAN, 5.0];
        assert_eq!(select_topk(&nn, 1), vec![2]);
        assert_eq!(select_topk(&nn, 2), vec![0, 2]);
        assert_eq!(select_topk(&nn, 3), vec![0, 1, 2]);
        let all_nan = vec![f32::NAN; 64];
        assert_matches_oracle(&all_nan, 10);
        let mixed = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::NAN];
        for k in 0..=6 {
            assert_matches_oracle(&mixed, k);
        }
    }

    #[test]
    fn topk_randomized_matches_full_sort() {
        let mut state = 0x9E3779B97F4A7C15u64;
        for trial in 0..200 {
            let n = 1usize + (trial * 37) % 1500;
            let k = (trial * 13) % (n + 1);
            let mode = trial % 6;
            let scores: Vec<f32> = (0..n)
                .map(|_| match mode {
                    0 => lcg(&mut state),
                    1 => {
                        let v = lcg(&mut state);
                        if v < 0.2 { f32::NAN } else { v }
                    }
                    2 => {
                        let v = lcg(&mut state) - 0.5;
                        if v == 0.0 { 0.0 } else { v }
                    }
                    3 => match (lcg(&mut state) * 4.0) as u32 {
                        0 => f32::INFINITY,
                        1 => f32::NEG_INFINITY,
                        2 => f32::NAN,
                        _ => lcg(&mut state),
                    },
                    4 => ((trial / 6) % 5) as f32,
                    _ => {
                        let r = lcg(&mut state);
                        if r < 0.5 { 0.0 } else if r < 0.9 { 1.0 } else { r }
                    }
                })
                .collect();
            for k2 in [k, k.min(3), n.saturating_sub(1)] {
                assert_matches_oracle(&scores, k2.min(n));
            }
        }
    }

    #[test]
    fn topk_short_inputs_all_k() {
        let values = [f32::NAN, f32::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY];
        for n in 0..=4 {
            for mut pattern in 0..values.len().pow(n) {
                let scores: Vec<f32> = (0..n)
                    .map(|_| {
                        let value = values[pattern % values.len()];
                        pattern /= values.len();
                        value
                    })
                    .collect();
                for k in 0..=scores.len() + 1 {
                    assert_matches_oracle(&scores, k);
                }
                assert_matches_oracle(&scores, usize::MAX);
            }
        }
    }

    #[test]
    fn topk_comparator_total_order() {
        let scores = [
            f32::from_bits(0xffc00001),
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0x7f800001),
            f32::from_bits(0x7fc00002),
        ];
        let order = [6, 5, 3, 4, 2, 1, 0, 7, 8];
        for (rank_a, &a) in order.iter().enumerate() {
            for (rank_b, &b) in order.iter().enumerate() {
                assert_eq!(cmp_desc(scores[a], a, scores[b], b), rank_a.cmp(&rank_b));
            }
        }
        for k in 0..=scores.len() {
            let mut expected = order[..k].to_vec();
            expected.sort_unstable();
            assert_eq!(select_topk(&scores, k), expected);
        }
    }

    #[test]
    fn topk_large_inputs_and_random_bits() {
        let mut state = 0x123456789abcdef0u64;
        let n = 65537;
        for mode in 0..7 {
            let scores: Vec<f32> = (0..n)
                .map(|i| match mode {
                    0 => i as f32,
                    1 => (n - i) as f32,
                    2 => 1.0,
                    3 => f32::NAN,
                    4 => if i % 2 == 0 { -0.0 } else { 0.0 },
                    _ => {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let bits = (state >> 32) as u32;
                        if mode == 5 {
                            f32::from_bits(bits)
                        } else {
                            f32::from_bits(bits & 0xff7fffff)
                        }
                    }
                })
                .collect();
            for k in [0, 1, 2, 63, 1023, 1024, 1025, 4096, n / 2, n - 1, n, usize::MAX] {
                assert_matches_oracle(&scores, k);
            }
        }
    }

    #[test]
    fn topk_block_boundary_stability() {
        for n in [63usize, 64, 65, 1023, 1024, 1025, 1026, 2047, 2048, 2049, 4097] {
            let scores: Vec<f32> =
                (0..n).map(|i| ((i as u64 * 2654435761 % 97) as f32) / 97.0).collect();
            for k in [0, 1, 64, n / 2, n.saturating_sub(1), n] {
                assert_matches_oracle(&scores, k);
            }
        }
    }
}

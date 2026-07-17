//! Top-level GLM-5.2 model: weight loading by the container naming scheme, the
//! per-layer forward loop, and the generate loop. Ports the structure of
//! `model_load` (`c/glm.c:1425-1469`) and `layer_forward_rows` (`c/glm.c:3629`).
//!
//! Experts are held resident (fine for the tiny/oracle model); disk streaming
//! for the 744B model is M2. Absorption/DSA are M5 — attention runs the dense
//! reconstruction path.

use peregrine_core::{Cfg, Error, SafeTensors};

use crate::attention::{mla_attention, AttnWeights, LayerKv};
use crate::math::rmsnorm;
use crate::mlp::{moe_forward, Mlp};
use crate::sample::Sampler;
use crate::weight::QtWeight;

/// Per-layer weights.
struct LayerW {
    in_ln: Vec<f32>,
    post_ln: Vec<f32>,
    q_a: QtWeight,
    q_a_ln: Vec<f32>,
    q_b: QtWeight,
    kv_a: QtWeight,
    kv_a_ln: Vec<f32>,
    kv_b: QtWeight,
    o: QtWeight,
    sparse: bool,
    dense: Option<Mlp>,          // dense layers (i < first_dense)
    router: Vec<f32>,            // [E, hidden] (sparse only)
    router_bias: Vec<f32>,       // [E]
    shared: Option<Mlp>,
    experts: Vec<Mlp>,
}

impl LayerW {
    fn attn(&self) -> AttnWeights<'_> {
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

/// A loaded model plus its per-layer KV cache.
pub struct Model {
    pub cfg: Cfg,
    embed: Vec<f32>, // [vocab, hidden], dequantized
    layers: Vec<LayerW>,
    final_norm: Vec<f32>,
    lm_head: QtWeight,
    kv: Vec<LayerKv>,
}

fn load_f32(st: &SafeTensors, name: &str, n: usize) -> Result<Vec<f32>, Error> {
    let mut v = vec![0f32; n];
    st.read_f32(name, &mut v)?;
    Ok(v)
}

impl Model {
    /// Load a model directory (config.json + `*.safetensors` in the int4/int8
    /// container format).
    pub fn load(dir: &std::path::Path) -> Result<Model, Error> {
        let cfg = Cfg::load(dir)?;
        let st = SafeTensors::open(dir)?;
        let (d, h) = (cfg.hidden as usize, cfg.n_heads as usize);
        let (qkh, vh) = (cfg.qk_head as usize, cfg.v_head as usize);
        let (ql, kvl, qkr, qkn) = (cfg.q_lora as usize, cfg.kv_lora as usize, cfg.qk_rope as usize, cfg.qk_nope as usize);
        let vocab = cfg.vocab as usize;

        let embed = QtWeight::load(&st, "model.embed_tokens.weight", vocab, d)?.dequant();
        let lm_head = QtWeight::load(&st, "lm_head.weight", vocab, d)?;
        let final_norm = load_f32(&st, "model.norm.weight", d)?;

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            let sparse = i >= cfg.first_dense as usize;

            let (mut dense, mut router, mut router_bias, mut shared, mut experts) =
                (None, Vec::new(), Vec::new(), None, Vec::new());
            if !sparse {
                let di = cfg.dense_inter as usize;
                dense = Some(Mlp {
                    gate: QtWeight::load(&st, &p("mlp.gate_proj.weight"), di, d)?,
                    up: QtWeight::load(&st, &p("mlp.up_proj.weight"), di, d)?,
                    down: QtWeight::load(&st, &p("mlp.down_proj.weight"), d, di)?,
                });
            } else {
                let (e_n, mi, si) = (cfg.n_experts as usize, cfg.moe_inter as usize, (cfg.moe_inter * cfg.n_shared) as usize);
                router = load_f32(&st, &p("mlp.gate.weight"), e_n * d)?;
                router_bias = load_f32(&st, &p("mlp.gate.e_score_correction_bias"), e_n)?;
                shared = Some(Mlp {
                    gate: QtWeight::load(&st, &p("mlp.shared_experts.gate_proj.weight"), si, d)?,
                    up: QtWeight::load(&st, &p("mlp.shared_experts.up_proj.weight"), si, d)?,
                    down: QtWeight::load(&st, &p("mlp.shared_experts.down_proj.weight"), d, si)?,
                });
                for e in 0..e_n {
                    let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                    experts.push(Mlp {
                        gate: QtWeight::load(&st, &pe("gate_proj.weight"), mi, d)?,
                        up: QtWeight::load(&st, &pe("up_proj.weight"), mi, d)?,
                        down: QtWeight::load(&st, &pe("down_proj.weight"), d, mi)?,
                    });
                }
            }

            layers.push(LayerW {
                in_ln: load_f32(&st, &p("input_layernorm.weight"), d)?,
                post_ln: load_f32(&st, &p("post_attention_layernorm.weight"), d)?,
                q_a: QtWeight::load(&st, &p("self_attn.q_a_proj.weight"), ql, d)?,
                q_a_ln: load_f32(&st, &p("self_attn.q_a_layernorm.weight"), ql)?,
                q_b: QtWeight::load(&st, &p("self_attn.q_b_proj.weight"), h * qkh, ql)?,
                kv_a: QtWeight::load(&st, &p("self_attn.kv_a_proj_with_mqa.weight"), kvl + qkr, d)?,
                kv_a_ln: load_f32(&st, &p("self_attn.kv_a_layernorm.weight"), kvl)?,
                kv_b: QtWeight::load(&st, &p("self_attn.kv_b_proj.weight"), h * (qkn + vh), kvl)?,
                o: QtWeight::load(&st, &p("self_attn.o_proj.weight"), d, h * vh)?,
                sparse,
                dense,
                router,
                router_bias,
                shared,
                experts,
            });
        }

        let kv = (0..cfg.n_layers).map(|_| LayerKv::new(kvl, qkr)).collect();
        Ok(Model { cfg, embed, layers, final_norm, lm_head, kv })
    }

    /// Clear the KV cache to start a fresh sequence.
    pub fn reset(&mut self) {
        let (kvl, qkr) = (self.cfg.kv_lora as usize, self.cfg.qk_rope as usize);
        for k in &mut self.kv {
            *k = LayerKv::new(kvl, qkr);
        }
    }

    fn rmsnorm_rows(x: &[f32], w: &[f32], s_n: usize, d: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0f32; s_n * d];
        for s in 0..s_n {
            let src = x[s * d..s * d + d].to_vec();
            rmsnorm(&mut out[s * d..s * d + d], &src, w, eps);
        }
        out
    }

    /// Run `tokens` (new positions starting at `pos_base`) through all layers,
    /// appending to the KV cache. Returns logits `[S, vocab]`.
    pub fn forward_step(&mut self, tokens: &[i32], pos_base: usize) -> Vec<f32> {
        let s_n = tokens.len();
        let d = self.cfg.hidden as usize;
        let eps = self.cfg.eps;

        // embedding lookup — clamp out-of-range ids (a malformed prompt token, or
        // a negative id) into `0..vocab` so a bad input can't index out of bounds.
        let vocab = self.cfg.vocab as usize;
        let mut x = vec![0f32; s_n * d];
        for (s, &t) in tokens.iter().enumerate() {
            let tid = (t.max(0) as usize).min(vocab.saturating_sub(1));
            x[s * d..s * d + d].copy_from_slice(&self.embed[tid * d..tid * d + d]);
        }

        // split disjoint fields so attention can borrow layers (imm) + kv (mut)
        let Model { cfg, layers, kv, .. } = self;
        for (li, l) in layers.iter().enumerate() {
            let nrm = Self::rmsnorm_rows(&x, &l.in_ln, s_n, d, eps);
            let attn = mla_attention(&l.attn(), &nrm, s_n, pos_base, &mut kv[li], cfg);
            for z in 0..s_n * d {
                x[z] += attn[z];
            }
            let nrm2 = Self::rmsnorm_rows(&x, &l.post_ln, s_n, d, eps);
            let ffn = if l.sparse {
                moe_forward(
                    &nrm2,
                    &l.router,
                    &l.router_bias,
                    &l.experts,
                    l.shared.as_ref(),
                    s_n,
                    d,
                    cfg.topk as usize,
                    cfg.norm_topk,
                    cfg.routed_scale,
                )
            } else {
                l.dense.as_ref().unwrap().swiglu(&nrm2, s_n)
            };
            for z in 0..s_n * d {
                x[z] += ffn[z];
            }
        }

        let xf = Self::rmsnorm_rows(&x, &self.final_norm, s_n, d, eps);
        self.lm_head.apply_vec(&xf, s_n)
    }

    /// Greedy/sampled generation: prefill `prompt`, then decode `n_new` tokens.
    /// Resets the KV cache first. Returns the newly generated token ids.
    pub fn generate(&mut self, prompt: &[i32], n_new: usize, sampler: &mut Sampler) -> Vec<i32> {
        // an empty prompt has no last-position logits to sample from, and no
        // requested tokens is a no-op — both would otherwise underflow below.
        if prompt.is_empty() || n_new == 0 {
            return Vec::new();
        }
        self.reset();
        let vocab = self.cfg.vocab as usize;
        let logits = self.forward_step(prompt, 0);
        let mut next = sampler.pick(&logits[(prompt.len() - 1) * vocab..prompt.len() * vocab], -1) as i32;
        let mut out = vec![next];
        for step in 1..n_new {
            let pos = prompt.len() + step - 1; // first decode attends at prompt.len()
            let lg = self.forward_step(&[next], pos);
            next = sampler.pick(&lg[..vocab], -1) as i32;
            out.push(next);
        }
        out
    }

    /// Teacher-forcing predictions: greedy argmax at each position of a single
    /// full forward over `tokens`. The shape the oracle gate compares against.
    pub fn teacher_forcing(&mut self, tokens: &[i32]) -> Vec<i32> {
        self.reset();
        let vocab = self.cfg.vocab as usize;
        let logits = self.forward_step(tokens, 0);
        (0..tokens.len())
            .map(|s| crate::sample::argmax(&logits[s * vocab..s * vocab + vocab]) as i32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::build_tiny_model;
    use std::path::PathBuf;

    fn tmp_model_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("peregrine_model_{}_{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&d);
        build_tiny_model(&d);
        d
    }

    #[test]
    fn loads_and_runs_forward() {
        let dir = tmp_model_dir("fwd");
        let mut m = Model::load(&dir).unwrap();
        let logits = m.forward_step(&[1, 5, 9, 2], 0);
        assert_eq!(logits.len(), 4 * m.cfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_is_deterministic_greedy() {
        let dir = tmp_model_dir("gen");
        let mut m = Model::load(&dir).unwrap();
        let prompt = [3, 7, 1, 4];
        let mut s1 = Sampler::new(0.0, 0.9, 1); // greedy
        let a = m.generate(&prompt, 8, &mut s1);
        let mut s2 = Sampler::new(0.0, 0.9, 1);
        let b = m.generate(&prompt, 8, &mut s2);
        assert_eq!(a, b, "greedy generation must be deterministic");
        assert!(a.iter().all(|&t| (t as usize) < m.cfg.vocab as usize));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_matches_teacher_forcing_prefix() {
        // greedy decode's first token == teacher-forcing argmax at the last
        // prompt position (both are argmax of the same prefill logits).
        let dir = tmp_model_dir("tf");
        let mut m = Model::load(&dir).unwrap();
        let prompt = [2, 6, 3, 8, 1];
        let tf = m.teacher_forcing(&prompt);
        let mut s = Sampler::new(0.0, 0.9, 1);
        let gen = m.generate(&prompt, 1, &mut s);
        assert_eq!(gen[0], tf[prompt.len() - 1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn handles_empty_prompt_and_out_of_range_tokens() {
        // regression: empty prompt / zero n_new must not underflow, and out-of-
        // range or negative token ids must be clamped, not index out of bounds.
        let dir = tmp_model_dir("edge");
        let mut m = Model::load(&dir).unwrap();
        let mut s = Sampler::new(0.0, 0.9, 1);
        assert!(m.generate(&[], 4, &mut s).is_empty());
        assert!(m.generate(&[1, 2], 0, &mut s).is_empty());
        let logits = m.forward_step(&[9999, -3, 0], 0);
        assert_eq!(logits.len(), 3 * m.cfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

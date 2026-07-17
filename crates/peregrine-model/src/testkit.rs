//! Synthetic-model generation — build a tiny GLM-5.2-shaped model directory
//! (config.json + int4/int8 `model.safetensors`) with random weights, using
//! only `peregrine_core::pack` (no torch/numpy). Used by tests and the `peregrine-engine`
//! demo mode to exercise loading + the full forward end-to-end.

use peregrine_core::pack::{f32_bytes, quant_i4, quant_i8, write_safetensors, Blob};
use peregrine_core::Cfg;
use std::path::Path;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// The tiny GLM-5.2-shaped config: 3 layers (1 dense + 2 sparse MoE).
pub fn tiny_cfg_json() -> serde_json::Value {
    serde_json::json!({
        "hidden_size": 16, "num_hidden_layers": 3, "num_attention_heads": 2,
        "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
        "intermediate_size": 8, "first_k_dense_replace": 1, "q_lora_rank": 12,
        "kv_lora_rank": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 6, "n_shared_experts": 1, "vocab_size": 32, "n_group": 1,
        "topk_group": 1, "rope_parameters": {"rope_theta": 10000.0}, "rms_norm_eps": 1e-5,
        "routed_scaling_factor": 2.5, "norm_topk_prob": true, "index_topk": 4096
    })
}

/// Write a tiny random model into `dir`, seeded by `seed` for reproducibility.
pub fn build_tiny_model_seeded(dir: &Path, seed: u64) {
    let cfg: Cfg = Cfg::from_json(&tiny_cfg_json()).unwrap();
    let mut r = Lcg(seed);
    let rnd = |n: usize, r: &mut Lcg| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
    let (d, h) = (cfg.hidden as usize, cfg.n_heads as usize);
    let (qkh, vh) = (cfg.qk_head as usize, cfg.v_head as usize);
    let (ql, kvl, qkr, qkn) = (cfg.q_lora as usize, cfg.kv_lora as usize, cfg.qk_rope as usize, cfg.qk_nope as usize);
    let vocab = cfg.vocab as usize;

    let mut blobs = Vec::new();
    let w4 = |blobs: &mut Vec<Blob>, name: &str, o: usize, i: usize, r: &mut Lcg| {
        let w = rnd(o * i, r);
        let (q, s) = quant_i4(&w, o, i);
        blobs.push(Blob::new(name.to_string(), "U8", vec![o as i64, (i.div_ceil(2)) as i64], q));
        blobs.push(Blob::new(format!("{name}.qs"), "F32", vec![o as i64], f32_bytes(&s)));
    };
    let wf = |blobs: &mut Vec<Blob>, name: &str, n: usize, r: &mut Lcg| {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + r.f() * 0.1).collect();
        blobs.push(Blob::new(name.to_string(), "F32", vec![n as i64], f32_bytes(&v)));
    };

    // embed + lm_head (int8) + final norm
    let w = rnd(vocab * d, &mut r);
    let (q, s) = quant_i8(&w, vocab, d);
    blobs.push(Blob::new("model.embed_tokens.weight", "U8", vec![vocab as i64, d as i64], q));
    blobs.push(Blob::new("model.embed_tokens.weight.qs", "F32", vec![vocab as i64], f32_bytes(&s)));
    let w = rnd(vocab * d, &mut r);
    let (q, s) = quant_i8(&w, vocab, d);
    blobs.push(Blob::new("lm_head.weight", "U8", vec![vocab as i64, d as i64], q));
    blobs.push(Blob::new("lm_head.weight.qs", "F32", vec![vocab as i64], f32_bytes(&s)));
    wf(&mut blobs, "model.norm.weight", d, &mut r);

    for i in 0..cfg.n_layers as usize {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        wf(&mut blobs, &p("input_layernorm.weight"), d, &mut r);
        wf(&mut blobs, &p("post_attention_layernorm.weight"), d, &mut r);
        w4(&mut blobs, &p("self_attn.q_a_proj.weight"), ql, d, &mut r);
        wf(&mut blobs, &p("self_attn.q_a_layernorm.weight"), ql, &mut r);
        w4(&mut blobs, &p("self_attn.q_b_proj.weight"), h * qkh, ql, &mut r);
        w4(&mut blobs, &p("self_attn.kv_a_proj_with_mqa.weight"), kvl + qkr, d, &mut r);
        wf(&mut blobs, &p("self_attn.kv_a_layernorm.weight"), kvl, &mut r);
        w4(&mut blobs, &p("self_attn.kv_b_proj.weight"), h * (qkn + vh), kvl, &mut r);
        w4(&mut blobs, &p("self_attn.o_proj.weight"), d, h * vh, &mut r);
        if i < cfg.first_dense as usize {
            let di = cfg.dense_inter as usize;
            w4(&mut blobs, &p("mlp.gate_proj.weight"), di, d, &mut r);
            w4(&mut blobs, &p("mlp.up_proj.weight"), di, d, &mut r);
            w4(&mut blobs, &p("mlp.down_proj.weight"), d, di, &mut r);
        } else {
            let (e_n, mi, si) = (cfg.n_experts as usize, cfg.moe_inter as usize, (cfg.moe_inter * cfg.n_shared) as usize);
            let rw = rnd(e_n * d, &mut r);
            blobs.push(Blob::new(p("mlp.gate.weight"), "F32", vec![e_n as i64, d as i64], f32_bytes(&rw)));
            let rb: Vec<f32> = (0..e_n).map(|_| r.f() * 0.1).collect();
            blobs.push(Blob::new(p("mlp.gate.e_score_correction_bias"), "F32", vec![e_n as i64], f32_bytes(&rb)));
            w4(&mut blobs, &p("mlp.shared_experts.gate_proj.weight"), si, d, &mut r);
            w4(&mut blobs, &p("mlp.shared_experts.up_proj.weight"), si, d, &mut r);
            w4(&mut blobs, &p("mlp.shared_experts.down_proj.weight"), d, si, &mut r);
            for e in 0..e_n {
                let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                w4(&mut blobs, &pe("gate_proj.weight"), mi, d, &mut r);
                w4(&mut blobs, &pe("up_proj.weight"), mi, d, &mut r);
                w4(&mut blobs, &pe("down_proj.weight"), d, mi, &mut r);
            }
        }
    }

    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&tiny_cfg_json()).unwrap()).unwrap();
    write_safetensors(dir, &blobs).unwrap();
}

/// Convenience: build the tiny model with the default seed.
pub fn build_tiny_model(dir: &Path) {
    build_tiny_model_seeded(dir, 0xC0FFEE);
}

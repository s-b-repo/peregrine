//! Synthetic-model generation — build a tiny GLM-5.2-shaped model directory
//! (config.json + int4/int8 `model.safetensors`) with random weights, using
//! only `peregrine_core::pack` (no torch/numpy). Used by tests and the `peregrine-engine`
//! demo mode to exercise loading + the full forward end-to-end.

use peregrine_core::pack::{f32_bytes, quant_i4, quant_i8, write_safetensors, Blob};
use peregrine_core::{Cfg, Error};
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

/// [`tiny_cfg_json`] with the DSA lightning indexer configured: two heads of
/// width 4, keeping the top `topk` keys.
///
/// A *separate* config rather than a flag on the shared one, because
/// `index_topk` here is deliberately tiny — small enough that a handful of
/// prompt tokens crosses it and the sparse path actually engages. The shared
/// tiny model keeps 4096, which no test prompt will ever exceed.
#[cfg(test)]
pub(crate) fn tiny_indexer_cfg_json(topk: i64) -> serde_json::Value {
    let mut v = tiny_cfg_json();
    if let Some(o) = v.as_object_mut() {
        o.insert("index_n_heads".into(), serde_json::json!(2));
        o.insert("index_head_dim".into(), serde_json::json!(4));
        o.insert("index_topk".into(), serde_json::json!(topk));
    }
    v
}

/// Write a tiny random model into `dir`, seeded by `seed` for reproducibility.
pub fn build_tiny_model_seeded(dir: &Path, seed: u64) -> Result<(), Error> {
    build_tiny_model_cfg(dir, seed, tiny_cfg_json())
}

/// A tiny model that also carries DSA indexer tensors, keeping the top `topk`
/// keys. Separate from [`build_tiny_model_seeded`] so the several hundred tests
/// built on the shared fixture keep loading a checkpoint *without* an indexer —
/// the state a real laptop-converted container is in.
#[cfg(test)]
pub(crate) fn build_tiny_model_with_indexer(dir: &Path, seed: u64, topk: i64) -> Result<(), Error> {
    build_tiny_model_cfg(dir, seed, tiny_indexer_cfg_json(topk))
}

/// Write a tiny random model for an arbitrary tiny config.
fn build_tiny_model_cfg(dir: &Path, seed: u64, cfg_json: serde_json::Value) -> Result<(), Error> {
    let cfg: Cfg = Cfg::from_json(&cfg_json)?;
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

    // layers 0..n_layers are the main stack; layer n_layers is the MTP head layer.
    for i in 0..=cfg.n_layers as usize {
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
        // DSA lightning indexer, when the config configures one. Same tensor
        // names the real converter emits with `--indexer`, so `IndexerWeights`
        // is loaded by the production path rather than a test-only shortcut.
        if cfg.index_nh > 0 && cfg.index_hd > 0 {
            let (inh, ihd) = (cfg.index_nh as usize, cfg.index_hd as usize);
            w4(&mut blobs, &p("self_attn.indexer_projections.wq_b"), inh * ihd, ql, &mut r);
            w4(&mut blobs, &p("self_attn.indexer_projections.wk"), ihd, d, &mut r);
            w4(&mut blobs, &p("self_attn.indexer_projections.weights_proj"), inh, d, &mut r);
            wf(&mut blobs, &p("self_attn.indexer.k_norm.weight"), ihd, &mut r);
            let kb: Vec<f32> = (0..ihd).map(|_| r.f() * 0.1).collect();
            blobs.push(Blob::new(p("self_attn.indexer.k_norm.bias"), "F32", vec![ihd as i64], f32_bytes(&kb)));
        }
        // the MTP head layer also carries the embed/hidden projection + norms.
        if i == cfg.n_layers as usize {
            w4(&mut blobs, &p("eh_proj.weight"), d, 2 * d, &mut r);
            wf(&mut blobs, &p("enorm.weight"), d, &mut r);
            wf(&mut blobs, &p("hnorm.weight"), d, &mut r);
            wf(&mut blobs, &p("shared_head.norm.weight"), d, &mut r);
        }
    }

    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg_json)?)?;
    write_safetensors(dir, &blobs)?;
    Ok(())
}

/// Convenience: build the tiny model with the default seed.
pub fn build_tiny_model(dir: &Path) -> Result<(), Error> {
    build_tiny_model_seeded(dir, 0xC0FFEE)
}

/// The tiny classic-Qwen3 (dense GQA) config — Track C's GQA core fixture.
/// Dims mirror `peregrine_core::config`'s tests and C2's importer fixture.
pub fn tiny_qwen_cfg_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 8, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "eos_token_id": 0
    })
}

/// The tiny Qwen3.5-hybrid config: 3 layers (linear, linear, full), the
/// output-gated attention, partial rotary 0.25 — every hybrid mechanism at toy
/// dims. Kept in sync with the config tests and C2's importer fixture.
pub fn tiny_hybrid_cfg_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 8,
            "num_hidden_layers": 3, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4, "attn_output_gate": true,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
            "rms_norm_eps": 1e-6, "eos_token_id": 0
        }
    })
}

/// A dense-GQA config at caller-chosen width — the fixture for tests whose
/// *numerics* must be representative rather than merely structural. The 16-wide
/// [`tiny_qwen_cfg_json`] exercises every code path but sits inside a single
/// WMMA tile, where edge handling and short accumulations make GPU/CPU
/// agreement look worse than it is at real widths.
pub fn sized_qwen_cfg_json(hidden: i64, inter: i64, n_heads: i64, n_kv_heads: i64, head_dim: i64) -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3", "vocab_size": 32, "hidden_size": hidden,
        "intermediate_size": inter, "num_hidden_layers": 2,
        "num_attention_heads": n_heads, "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "eos_token_id": 0
    })
}

/// [`sized_qwen_cfg_json`] written to disk as a loadable model.
pub fn build_sized_qwen_model(dir: &Path, seed: u64, cfg_json: serde_json::Value) -> Result<(), Error> {
    build_tiny_gqa_family(dir, seed, cfg_json)
}

/// A hybrid config at caller-chosen width. The 16-wide [`tiny_hybrid_cfg_json`]
/// exercises every code path structurally, but it is below the width at which
/// the device GEMV entry engages — so a GPU test built on it passes without the
/// device ever computing. Tests that must actually reach the kernel use this.
pub fn sized_hybrid_cfg_json(hidden: i64, inter: i64, head_dim: i64, lin_dim: i64) -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32, "hidden_size": hidden, "intermediate_size": inter,
            "num_hidden_layers": 3, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": head_dim,
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": lin_dim, "linear_value_head_dim": lin_dim,
            "linear_conv_kernel_dim": 4, "attn_output_gate": true,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
            "rms_norm_eps": 1e-6, "eos_token_id": 0
        }
    })
}

/// [`sized_hybrid_cfg_json`] written to disk as a loadable model.
pub fn build_sized_hybrid_model(dir: &Path, seed: u64, cfg_json: serde_json::Value) -> Result<(), Error> {
    build_tiny_gqa_family(dir, seed, cfg_json)
}

/// Write a tiny random dense-GQA (classic Qwen3) model into `dir`.
pub fn build_tiny_qwen_model(dir: &Path, seed: u64) -> Result<(), Error> {
    build_tiny_gqa_family(dir, seed, tiny_qwen_cfg_json())
}

/// Write a tiny random Qwen3.5-hybrid model into `dir`.
pub fn build_tiny_hybrid_model(dir: &Path, seed: u64) -> Result<(), Error> {
    build_tiny_gqa_family(dir, seed, tiny_hybrid_cfg_json())
}

/// Shared writer for the two Qwen-family fixtures: emits exactly the Track C
/// tensor contract (HF names verbatim, int4 + `.qs` matrices, float norms and
/// GDN scalars, int8 embed/lm_head) so the production loader is what the tests
/// exercise — no test-only naming.
fn build_tiny_gqa_family(dir: &Path, seed: u64, cfg_json: serde_json::Value) -> Result<(), Error> {
    let cfg: Cfg = Cfg::from_json(&cfg_json)?;
    let hybrid = cfg.arch == peregrine_core::Arch::HybridGdn;
    let mut r = Lcg(seed);
    let rnd = |n: usize, r: &mut Lcg| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
    let d = cfg.hidden as usize;
    let (nh, nkv, hd) = (cfg.n_heads as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
    let vocab = cfg.vocab as usize;
    let di = cfg.dense_inter as usize;

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

    let stem = if hybrid { "model.language_model" } else { "model" };
    let w = rnd(vocab * d, &mut r);
    let (q, s) = quant_i8(&w, vocab, d);
    blobs.push(Blob::new(format!("{stem}.embed_tokens.weight"), "U8", vec![vocab as i64, d as i64], q));
    blobs.push(Blob::new(format!("{stem}.embed_tokens.weight.qs"), "F32", vec![vocab as i64], f32_bytes(&s)));
    let w = rnd(vocab * d, &mut r);
    let (q, s) = quant_i8(&w, vocab, d);
    blobs.push(Blob::new("lm_head.weight".to_string(), "U8", vec![vocab as i64, d as i64], q));
    blobs.push(Blob::new("lm_head.weight.qs".to_string(), "F32", vec![vocab as i64], f32_bytes(&s)));
    wf(&mut blobs, &format!("{stem}.norm.weight"), d, &mut r);

    for i in 0..cfg.n_layers as usize {
        let p = |s: &str| format!("{stem}.layers.{i}.{s}");
        wf(&mut blobs, &p("input_layernorm.weight"), d, &mut r);
        wf(&mut blobs, &p("post_attention_layernorm.weight"), d, &mut r);
        let full = !hybrid || cfg.full_attn.get(i).copied().unwrap_or(false);
        if full {
            let q_rows = if cfg.attn_gate { 2 * nh * hd } else { nh * hd };
            w4(&mut blobs, &p("self_attn.q_proj.weight"), q_rows, d, &mut r);
            w4(&mut blobs, &p("self_attn.k_proj.weight"), nkv * hd, d, &mut r);
            w4(&mut blobs, &p("self_attn.v_proj.weight"), nkv * hd, d, &mut r);
            w4(&mut blobs, &p("self_attn.o_proj.weight"), d, nh * hd, &mut r);
            wf(&mut blobs, &p("self_attn.q_norm.weight"), hd, &mut r);
            wf(&mut blobs, &p("self_attn.k_norm.weight"), hd, &mut r);
        } else {
            let (kh, vh, kd, vd, taps) = (
                cfg.lin_k_heads as usize,
                cfg.lin_v_heads as usize,
                cfg.lin_k_dim as usize,
                cfg.lin_v_dim as usize,
                cfg.lin_conv_k as usize,
            );
            let conv_dim = 2 * kh * kd + vh * vd;
            w4(&mut blobs, &p("linear_attn.in_proj_qkv.weight"), conv_dim, d, &mut r);
            w4(&mut blobs, &p("linear_attn.in_proj_z.weight"), vh * vd, d, &mut r);
            w4(&mut blobs, &p("linear_attn.in_proj_a.weight"), vh, d, &mut r);
            w4(&mut blobs, &p("linear_attn.in_proj_b.weight"), vh, d, &mut r);
            let conv = rnd(conv_dim * taps, &mut r);
            blobs.push(Blob::new(
                p("linear_attn.conv1d.weight"),
                "F32",
                vec![conv_dim as i64, 1, taps as i64],
                f32_bytes(&conv),
            ));
            let a_log: Vec<f32> = (0..vh).map(|_| r.f() * 0.5).collect();
            blobs.push(Blob::new(p("linear_attn.A_log"), "F32", vec![vh as i64], f32_bytes(&a_log)));
            let dtb: Vec<f32> = (0..vh).map(|_| r.f() * 0.5).collect();
            blobs.push(Blob::new(p("linear_attn.dt_bias"), "F32", vec![vh as i64], f32_bytes(&dtb)));
            wf(&mut blobs, &p("linear_attn.norm.weight"), vd, &mut r);
            w4(&mut blobs, &p("linear_attn.out_proj.weight"), d, vh * vd, &mut r);
        }
        w4(&mut blobs, &p("mlp.gate_proj.weight"), di, d, &mut r);
        w4(&mut blobs, &p("mlp.up_proj.weight"), di, d, &mut r);
        w4(&mut blobs, &p("mlp.down_proj.weight"), d, di, &mut r);
    }

    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg_json)?)?;
    write_safetensors(dir, &blobs)?;
    Ok(())
}

//! Import a bf16 HF checkpoint (Qwen3 / Qwen3.5-hybrid family) into the
//! peregrine int4/int8 container format — Track C2.
//!
//! A sibling of `peregrine-requantize`, deliberately not an extension of it:
//! requantize is container→container (packed `QtView` decode, `.qs` scale
//! sources, expert dims resolved from a MoE config), while an HF checkpoint is
//! dense bf16 tensors whose logical shapes live in their own headers. What the
//! two share is the *output* layer — [`crate::requant::ShardWriter`], the
//! fsync-then-rename shard commit, and `peregrine_core::pack`'s quantizers —
//! and that is exactly what this reuses.
//!
//! The tensor policy is the Track C REV 2 contract
//! (`bench-data/coordination-2026-08-15.md`), names VERBATIM from the shipped
//! index:
//! - **int4 + `.qs`**: every projection matrix — `self_attn.{q,k,v,o}_proj`,
//!   `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`,
//!   `mlp.{gate,up,down}_proj`, `lm_head.weight`, `mtp.fc.weight` (and the
//!   same families under `mtp.layers.*`).
//! - **int8 + `.qs`**: `embed_tokens.weight` (the GLM embed convention).
//! - **float (widened to F32)**: everything tiny or shape-odd —
//!   `conv1d.weight` `[conv_dim,1,4]`, `A_log`, `dt_bias`, and every norm.
//! - **skip**: the vision tower `model.visual.*`.
//! - **refuse**: anything else, by name, loudly. On a brand-new `model_type`,
//!   silently quantizing (or silently dropping) an unrecognized family is how
//!   a container loads cleanly and computes garbage; an error naming the
//!   tensor is how the contract gets extended on purpose instead.

use peregrine_core::config::Cfg;
use peregrine_core::pack::{quant_i4, quant_i8};
use peregrine_core::Dtype;
use std::collections::{BTreeMap, BTreeSet};
use peregrine_core::safetensors::SafeTensors;
use peregrine_core::{Context, Error};
use std::path::Path;

use crate::requant::ShardWriter;

/// What the policy table says about one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Vision tower — not imported.
    SkipVisual,
    /// Projection matrix → int4 + per-row `.qs`.
    Int4,
    /// Token embedding — and, on `glm5_next`, every gate-critical matrix
    /// (mHC `fn`, KDA forget/output-gate low-ranks, `b_proj`, the k-pool
    /// compress gate) → int8 + per-row `.qs`.
    Int8Embed,
    /// Small/odd tensors → F32 passthrough at their original shape.
    Float,
    /// A `*_scale_inv` block-scale sibling of an FP8 tensor — consumed while
    /// dequantizing its base tensor, never imported on its own.
    SkipScale,
}

/// The REV 2 name policy. Suffix-matched so the same table covers the plain
/// `model.` stem (classic dense Qwen3), the hybrid's `model.language_model.`
/// stem, and the `mtp.layers.*` head layer without three copies of itself.
pub fn classify(name: &str) -> Result<Family, Error> {
    if name.starts_with("model.visual.") {
        return Ok(Family::SkipVisual);
    }
    if name.ends_with(".qs") {
        return Err(Error::Format(format!(
            "'{name}': the source carries peregrine scale tensors — this is already a peregrine \
             container, not an HF checkpoint (use peregrine-requantize for container→container)"
        )));
    }
    const INT4_SUFFIX: &[&str] = &[
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.out_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
        // DFlash2 / speculative decoding projection matrices
        ".hidden_projection.weight",
        ".output_projection.weight",
        ".predecessor_codebook",
        ".successor_codebook",
        ".attention_conv.kernel_projection.weight",
        ".mlp_conv.kernel_projection.weight",
    ];
    const FLOAT_SUFFIX: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".linear_attn.conv1d.weight",
        ".linear_attn.A_log",
        ".linear_attn.dt_bias",
        ".linear_attn.norm.weight",
        // DFlash2 norm layers and conv kernels
        ".hidden_norm.weight",
        ".attention_conv.base_kernel",
        ".mlp_conv.base_kernel",
    ];
    if INT4_SUFFIX.iter().any(|s| name.ends_with(s)) {
        return Ok(Family::Int4);
    }
    if FLOAT_SUFFIX.iter().any(|s| name.ends_with(s)) {
        return Ok(Family::Float);
    }
    match name {
        // Both stems: hybrid (language_model) and classic dense.
        "model.language_model.embed_tokens.weight" | "model.embed_tokens.weight" => Ok(Family::Int8Embed),
        "model.language_model.norm.weight" | "model.norm.weight" => Ok(Family::Float),
        // Untied on this family (tie_word_embeddings false).
        "lm_head.weight" => Ok(Family::Int4),
        // The MTP head's own pieces outside its full-attn-shaped layer.
        "mtp.fc.weight" => Ok(Family::Int4),
        "mtp.norm.weight" | "mtp.pre_fc_norm_embedding.weight" | "mtp.pre_fc_norm_hidden.weight" => {
            Ok(Family::Float)
        }
        // DFlash2 / speculative decoding tensors (projection matrices → int4)
        "fc.weight" |
        "hidden_norm.weight" |
        "norm.weight" => Ok(Family::Float),
        // mask token embedding (if present)
        "mask_token_embed.weight" => Ok(Family::Int8Embed),
        _ => Err(Error::Format(format!(
            "'{name}': not in the Track C REV 2 tensor contract — refusing to guess whether it \
             quantizes, passes through, or is skipped. Extend `classify` in import.rs (and the \
             contract in bench-data/coordination-2026-08-15.md) deliberately."
        ))),
    }
}

/// Rename one GLM-5.3-Flash HF tensor to its peregrine container name, or
/// `None` when it does not travel (the vision tower; `*_scale_inv` siblings
/// are classified separately so the loop can account for them).
///
/// Two renames, both load-bearing:
/// - the `model.language_model.` stem becomes `model.` — the expert-streaming
///   lane, reshard and every layout tool are written against
///   `model.layers.{l}.mlp.experts.{e}.`, and `Arch::Glm5Next`'s
///   `layer_prefix` matches;
/// - the DSA indexer's own projections move under `indexer_projections.` with
///   their `.weight` suffix dropped, the GLM-5.2 container convention
///   `IndexerWeights::load` reads.
pub fn glm5next_rename(name: &str) -> Option<String> {
    if name.starts_with("model.visual.") {
        return None;
    }
    let stemmed = match name.strip_prefix("model.language_model.") {
        Some(rest) => format!("model.{rest}"),
        None => name.to_string(),
    };
    for proj in ["wq_b", "wk", "weights_proj"] {
        let hf = format!(".self_attn.indexer.{proj}.weight");
        if let Some(pre) = stemmed.strip_suffix(&hf) {
            return Some(format!("{pre}.self_attn.indexer_projections.{proj}"));
        }
    }
    Some(stemmed)
}

/// The GLM-5.3-Flash tensor policy, applied to **renamed** names. Same
/// refuse-the-unknown discipline as [`classify`]; the split is: int4 for the
/// big projections and experts, int8 for everything whose output passes
/// through a gate nonlinearity (σ/softmax/decay — where int4 rounding moves
/// routing and state rather than blurring an activation), F32 for norms,
/// convs, per-channel vectors and the router.
pub fn classify_glm5next(name: &str) -> Result<Family, Error> {
    if name.starts_with("model.visual.") {
        return Ok(Family::SkipVisual);
    }
    if name.ends_with("_scale_inv") {
        return Ok(Family::SkipScale);
    }
    if name.ends_with(".qs") {
        return Err(Error::Format(format!(
            "'{name}': the source carries peregrine scale tensors — this is already a peregrine \
             container, not an HF checkpoint (use peregrine-requantize for container→container)"
        )));
    }
    const INT4_SUFFIX: &[&str] = &[
        // NoPE MLA
        ".self_attn.q_a_proj.weight",
        ".self_attn.q_b_proj.weight",
        ".self_attn.kv_a_proj_with_mqa.weight",
        ".self_attn.kv_b_proj.weight",
        ".self_attn.o_proj.weight",
        // KDA projections
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        // dense MLP / shared expert / routed experts (suffix covers all three)
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
        ".gate_proj.weight",
        ".up_proj.weight",
        ".down_proj.weight",
        // DSA indexer projections (post-rename names)
        ".self_attn.indexer_projections.wq_b",
        ".self_attn.indexer_projections.wk",
        ".self_attn.indexer_projections.weights_proj",
        // MTP head
        ".eh_proj.weight",
    ];
    const INT8_SUFFIX: &[&str] = &[
        ".self_attn.f_a_proj.weight",
        ".self_attn.f_b_proj.weight",
        ".self_attn.g_a_proj.weight",
        ".self_attn.g_b_proj.weight",
        ".self_attn.b_proj.weight",
        ".self_attn.indexer.index_kpool_compress_gate",
        ".hc_attn_fn",
        ".hc_ffn_fn",
    ];
    const FLOAT_SUFFIX: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".self_attn.q_a_layernorm.weight",
        ".self_attn.kv_a_layernorm.weight",
        ".self_attn.o_norm.weight",
        ".self_attn.q_conv1d.weight",
        ".self_attn.k_conv1d.weight",
        ".self_attn.v_conv1d.weight",
        ".self_attn.A_log",
        ".self_attn.dt_bias",
        ".self_attn.indexer.k_norm.weight",
        ".self_attn.indexer.k_norm.bias",
        ".self_attn.indexer.index_kpool_compress_ape",
        ".hc_attn_base",
        ".hc_attn_scale",
        ".hc_ffn_base",
        ".hc_ffn_scale",
        // Router: selection-critical, tiny, stays exact.
        ".mlp.gate.weight",
        ".mlp.gate.e_score_correction_bias",
        // MTP head norms
        ".enorm.weight",
        ".hnorm.weight",
        ".shared_head.norm.weight",
    ];
    if INT8_SUFFIX.iter().any(|s| name.ends_with(s)) {
        return Ok(Family::Int8Embed);
    }
    if FLOAT_SUFFIX.iter().any(|s| name.ends_with(s)) {
        return Ok(Family::Float);
    }
    if INT4_SUFFIX.iter().any(|s| name.ends_with(s)) {
        return Ok(Family::Int4);
    }
    match name {
        "model.embed_tokens.weight" => Ok(Family::Int8Embed),
        "model.norm.weight" => Ok(Family::Float),
        "lm_head.weight" => Ok(Family::Int4),
        _ => Err(Error::Format(format!(
            "'{name}': not in the glm5_next tensor contract — refusing to guess whether it \
             quantizes, passes through, or is skipped. Extend `classify_glm5next` in import.rs \
             deliberately."
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen4Tensor {
    Dense(Family),
    GateUp,
    Down,
}

fn qwen4_rename(name: &str) -> String {
    let stem = match name.strip_prefix("model.language_model.") {
        Some(rest) => format!("model.{rest}"),
        None => name.to_string(),
    };
    stem.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
}

fn qwen4_contract(cfg: &Cfg) -> Result<BTreeMap<String, (Qwen4Tensor, Vec<i64>)>, Error> {
    let q = cfg.qwen4.as_ref().ok_or_else(|| Error::Format("qwen4_exp: missing validated Qwen4Cfg".into()))?;
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, family, shape: Vec<i64>| {
        tensors.insert(name, (family, shape));
    };
    let f32 = Qwen4Tensor::Dense(Family::Float);
    let int4 = Qwen4Tensor::Dense(Family::Int4);
    let d = cfg.hidden;
    let hc = q.hc_count * d;
    add("model.embed_tokens.weight".into(), Qwen4Tensor::Dense(Family::Int8Embed), vec![cfg.vocab, d]);
    add("lm_head.weight".into(), int4, vec![cfg.vocab, d]);
    for layer in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{layer}.{s}");
        add(p("mlp.gate.weight"), f32, vec![cfg.n_experts, d]);
        add(p("mlp.shared_expert_gate.weight"), f32, vec![1, d]);
        add(p("mlp.experts.gate_up_proj"), Qwen4Tensor::GateUp, vec![cfg.n_experts, 2 * cfg.moe_inter, d]);
        add(p("mlp.experts.down_proj"), Qwen4Tensor::Down, vec![cfg.n_experts, d, cfg.moe_inter]);
        for proj in ["gate", "up", "down"] {
            let shape = if proj == "down" { vec![d, cfg.dense_inter] } else { vec![cfg.dense_inter, d] };
            add(p(&format!("mlp.shared_experts.{proj}_proj.weight")), int4, shape);
        }
        if cfg.full_attn[layer as usize] {
            let qsa = q.qsa.as_ref().ok_or_else(|| Error::Format("qwen4_exp: missing validated QSA config".into()))?;
            for (proj, rows, cols) in [
                ("q", 2 * cfg.n_heads * cfg.head_dim, d),
                ("k", cfg.n_kv_heads * cfg.head_dim, d),
                ("v", cfg.n_kv_heads * cfg.head_dim, d),
                ("o", d, cfg.n_heads * cfg.head_dim),
            ] {
                add(p(&format!("self_attn.{proj}_proj.weight")), int4, vec![rows, cols]);
            }
            for norm in ["q_norm", "k_norm"] {
                add(p(&format!("self_attn.{norm}.weight")), f32, vec![cfg.head_dim]);
            }
            add(p("self_attn.indexer.index_qk_proj.weight"), f32, vec![(qsa.n_heads + 1) * qsa.head_dim, d]);
            for norm in ["q_layernorm", "k_layernorm"] {
                add(p(&format!("self_attn.indexer.{norm}.weight")), f32, vec![qsa.head_dim]);
            }
        } else {
            let k = cfg.lin_k_heads * cfg.lin_k_dim;
            let v = cfg.lin_v_heads * cfg.lin_v_dim;
            add(p("linear_attn.in_proj_qkv.weight"), int4, vec![2 * k + v, d]);
            add(p("linear_attn.out_proj.weight"), int4, vec![d, v]);
            add(p("linear_attn.in_proj_z.weight"), f32, vec![v, d]);
            for proj in ["a", "b"] {
                add(p(&format!("linear_attn.in_proj_{proj}.weight")), f32, vec![cfg.lin_v_heads, d]);
            }
            for vector in ["A_log", "dt_bias"] {
                add(p(&format!("linear_attn.{vector}")), f32, vec![cfg.lin_v_heads]);
            }
            add(p("linear_attn.norm.weight"), f32, vec![cfg.lin_v_dim]);
            add(p("linear_attn.conv1d.weight"), f32, vec![2 * k + v, 1, cfg.lin_conv_k]);
        }
        if q.ple_layer_ids.contains(&(layer + 1)) {
            add(p("ple.key_proj.weight"), f32, vec![hc, q.ple_embed_dim]);
            add(p("ple.value_proj.weight"), f32, vec![d, q.ple_embed_dim]);
            for norm in ["norm_key", "norm_query", "norm_conv"] {
                add(p(&format!("ple.{norm}.weight")), f32, vec![hc]);
            }
            add(p("ple.conv1d.weight"), f32, vec![hc, 1, q.ple_conv_kernel_size]);
        }
    }
    let mut residuals = vec!["model.hyper_connection_mixer".to_string()];
    for layer in 0..cfg.n_layers {
        for site in ["attn", "mlp"] {
            let prefix = format!("model.layers.{layer}.{site}_hyper_connection");
            add(format!("{prefix}.block_inject_weight.weight"), f32, vec![q.hc_count, hc]);
            residuals.push(prefix);
        }
    }
    for prefix in residuals {
        add(format!("{prefix}.hc_norm.weight"), f32, vec![hc]);
        add(format!("{prefix}.input_mix_weight_down.weight"), f32, vec![q.hc_lowrank, hc]);
        add(format!("{prefix}.input_mix_weight_up.weight"), f32, vec![hc, q.hc_lowrank]);
    }
    Ok(tensors)
}

struct Qwen4Ple {
    prefix: String,
    shards: Vec<(String, i64)>,
    rows: i64,
    cols: i64,
    buffers: Vec<(String, Vec<i64>)>,
}

fn qwen4_ple_geometry(cfg: &Cfg) -> Result<Vec<Qwen4Ple>, Error> {
    let q = cfg.qwen4.as_ref().ok_or_else(|| Error::Format("qwen4_exp: missing PLE config".into()))?;
    let heads = (q.ngram_size - 1) * q.heads_per_ngram;
    let mut prime = q.ngram_vocab_size_base - 1;
    let mut plans = Vec::new();
    for (index, layer) in q.ple_layer_ids.iter().enumerate() {
        let prefix = format!("model.layers.{}.ple.ple_embedding", layer - 1);
        let mut sizes = Vec::new();
        let mut offsets = Vec::new();
        let mut total = 0i64;
        for _ in 0..heads {
            loop {
                prime = prime.checked_add(1).ok_or_else(|| Error::Format("qwen4_exp: prime overflow".into()))?;
                let mut divisor = 2;
                while divisor <= prime / divisor && prime % divisor != 0 {
                    divisor += if divisor == 2 { 1 } else { 2 };
                }
                if divisor > prime / divisor { break; }
            }
            offsets.push(total);
            sizes.push(prime);
            total = total.checked_add(prime).ok_or_else(|| Error::Format("qwen4_exp: PLE rows overflow".into()))?;
        }
        let rows = total.checked_add(q.ngram_vocab_divisor - 1)
            .map(|v| v / q.ngram_vocab_divisor * q.ngram_vocab_divisor)
            .ok_or_else(|| Error::Format("qwen4_exp: padded PLE rows overflow".into()))?;
        let cols = q.ple_embed_dim / heads;
        rows.checked_mul(cols).and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Format("qwen4_exp: PLE byte size overflow".into()))?;
        let shard_rows = (rows + q.split_ngram_parts - 1) / q.split_ngram_parts;
        if (rows + shard_rows - 1) / shard_rows != q.split_ngram_parts {
            return Err(Error::Format("qwen4_exp: split_ngram_parts cannot produce the configured number of nonempty row shards".into()));
        }
        let bound = (i64::MAX / cfg.vocab / 2).max(1) as u64;
        let seed = (q.seed as u64).wrapping_add(10007u64.wrapping_mul(index as u64));
        let multipliers = (0..q.ngram_size).map(|i| {
            let mut value = seed.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(i as u64 + 1));
            value = value.wrapping_add(0x9e3779b97f4a7c15);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            (2 * ((value ^ (value >> 31)) % bound) + 1) as i64
        }).collect();
        plans.push(Qwen4Ple {
            shards: (0..q.split_ngram_parts).map(|i| (format!("{prefix}.ngram_embedding.shard_{i}.weight"), shard_rows.min(rows - i * shard_rows))).collect(),
            rows, cols,
            buffers: vec![(format!("{prefix}.layer_multipliers"), multipliers),
                (format!("{prefix}.ngram_heads_vocab_sizes"), sizes),
                (format!("{prefix}.ngram_heads_offsets"), offsets)],
            prefix,
        });
    }
    Ok(plans)
}

fn qwen4_skip_mtp(name: &str) -> bool {
    if matches!(name, "mtp.fc_embedding.weight" | "mtp.fc_hidden.weight"
        | "mtp.pre_fc_norm_embedding.weight" | "mtp.pre_fc_norm_hidden.weight"
        | "mtp.hyper_connection_mixer.hc_norm.weight"
        | "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
        | "mtp.hyper_connection_mixer.input_mix_weight_up.weight") {
        return true;
    }
    let Some(suffix) = name.strip_prefix("mtp.layers.0.") else { return false; };
    matches!(suffix,
        "attn_hyper_connection.block_inject_weight.weight" | "attn_hyper_connection.hc_norm.weight"
        | "attn_hyper_connection.input_mix_weight_down.weight" | "attn_hyper_connection.input_mix_weight_up.weight"
        | "mlp_hyper_connection.block_inject_weight.weight" | "mlp_hyper_connection.hc_norm.weight"
        | "mlp_hyper_connection.input_mix_weight_down.weight" | "mlp_hyper_connection.input_mix_weight_up.weight"
        | "mlp.experts.down_proj" | "mlp.experts.gate_up_proj" | "mlp.gate.weight"
        | "mlp.shared_expert.down_proj.weight" | "mlp.shared_expert.gate_proj.weight"
        | "mlp.shared_expert.up_proj.weight" | "mlp.shared_expert_gate.weight"
        | "self_attn.indexer.index_qk_proj.weight" | "self_attn.indexer.k_layernorm.weight"
        | "self_attn.indexer.q_layernorm.weight" | "self_attn.k_norm.weight" | "self_attn.q_norm.weight"
        | "self_attn.k_proj.weight" | "self_attn.q_proj.weight" | "self_attn.v_proj.weight" | "self_attn.o_proj.weight")
}

fn qwen4_preflight(st: &SafeTensors, cfg: &Cfg) -> Result<Vec<Qwen4Ple>, Error> {
    let mut expected = qwen4_contract(cfg)?;
    let mut plans = qwen4_ple_geometry(cfg)?;
    let mut buffers = BTreeMap::new();
    for plan in &plans {
        for (name, rows) in &plan.shards {
            expected.insert(name.clone(), (Qwen4Tensor::Dense(Family::Int8Embed), vec![*rows, plan.cols]));
        }
        for (name, values) in &plan.buffers {
            buffers.insert(name.clone(), values);
        }
    }
    let mut seen = BTreeSet::new();
    for t in st.tensors() {
        if t.name.starts_with("model.visual.") {
            continue;
        }
        if qwen4_skip_mtp(&t.name) { continue; }
        let name = qwen4_rename(&t.name);
        if !seen.insert(name.clone()) {
            return Err(Error::Format(format!("qwen4_exp: rename collision for '{name}'")));
        }
        if let Some(values) = buffers.get(&name) {
            if t.dtype != Dtype::I64 || t.shape != [values.len() as i64]
                || t.compression != peregrine_core::compress::Compression::None || t.layout.is_some() {
                return Err(Error::Format(format!("qwen4_exp: '{name}' requires native uncompressed I64 shape [{}]", values.len())));
            }
            let mut actual = vec![0; values.len()];
            st.read_i64(&t.name, &mut actual)?;
            if actual != **values {
                return Err(Error::Format(format!("qwen4_exp: '{name}' buffer mismatch: expected {values:?}, got {actual:?}")));
            }
            continue;
        }
        let (_, shape) = expected.get(&name).ok_or_else(|| Error::Format(format!(
            "'{}': not in the qwen4_exp import foundation contract; unknown tensors, MTP and ngram layouts are not skipped", t.name
        )))?;
        if &t.shape != shape {
            return Err(Error::Format(format!("'{}': qwen4_exp expected shape {shape:?}, got {:?}", t.name, t.shape)));
        }
        if !matches!(t.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32)
            || t.compression != peregrine_core::compress::Compression::None || t.layout.is_some()
        {
            return Err(Error::Format(format!("'{}': qwen4_exp requires uncompressed, native-layout BF16/F16/F32; packed and FP8 sources are unsupported", t.name)));
        }
    }
    for name in expected.keys() {
        if !seen.contains(name) {
            return Err(Error::Format(format!("qwen4_exp: missing required tensor '{name}'; refusing incomplete conversion")));
        }
    }
    let sources: BTreeMap<_, _> = st.tensors().iter().map(|t| (qwen4_rename(&t.name), t.name.clone())).collect();
    for plan in &mut plans {
        for (name, _) in &mut plan.shards {
            *name = sources.get(name).ok_or_else(|| Error::Format(format!("missing PLE shard {name}")))?.clone();
        }
    }
    Ok(plans)
}

fn qwen4_write_ple(st: &SafeTensors, plan: &Qwen4Ple, outdir: &Path, rep: &mut ImportReport) -> Result<(), Error> {
    use crate::stwrite::{PieceMeta, write_streaming};
    use std::io::{Seek, SeekFrom, Write};
    let name = format!("{}.ngram_embedding.weight", plan.prefix);
    let path = outdir.join(format!("model-ple-{}.safetensors", plan.prefix));
    let spool_path = path.with_extension("scales.part");
    let mut spool = std::fs::OpenOptions::new().create_new(true).read(true).write(true).open(&spool_path)?;
    std::fs::remove_file(&spool_path)?;
    let pieces = [
        PieceMeta { name: name.clone(), dtype: "U8".into(), shape: vec![plan.rows, plan.cols], nbytes: (plan.rows * plan.cols) as u64, extra: vec![] },
        PieceMeta { name: format!("{name}.qs"), dtype: "F32".into(), shape: vec![plan.rows], nbytes: plan.rows as u64 * 4, extra: vec![] },
    ];
    let result = write_streaming(&path, &[("peregrine.import.contract".into(), "qwen4-exp-ple-v1".into())], &pieces, |piece, output| {
        if piece == 1 {
            spool.seek(SeekFrom::Start(0))?;
            std::io::copy(&mut spool, output)?;
            return Ok(());
        }
        let chunk_rows = (262144 / plan.cols).max(1);
        let mut dense = vec![0f32; (chunk_rows * plan.cols) as usize];
        for (source, rows) in &plan.shards {
            let mut row = 0;
            while row < *rows {
                let count = chunk_rows.min(rows - row);
                let values = &mut dense[..(count * plan.cols) as usize];
                st.read_slice_f32(source, row * plan.cols, count * plan.cols, values)?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Format(format!("qwen4_exp: '{source}' non-finite PLE tensor")));
                }
                let (packed, scales) = quant_i8(values, count as usize, plan.cols as usize);
                output.write_all(&packed)?;
                let bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
                spool.write_all(&bytes)?;
                row += count;
            }
        }
        Ok(())
    });
    if result.is_err() { let _ = std::fs::remove_file(path.with_extension("safetensors.part")); }
    result?;
    rep.imported_int8 += 1;
    rep.bytes_out += pieces.iter().map(|p| p.nbytes).sum::<u64>();
    Ok(())
}

fn qwen4_write_experts(
    st: &SafeTensors, name: &str, out_name: &str, cfg: &Cfg,
    writer: &mut ShardWriter, rep: &mut ImportReport,
) -> Result<(), Error> {
    let gate_up = out_name.ends_with(".gate_up_proj");
    let suffix = if gate_up { ".gate_up_proj" } else { ".down_proj" };
    let prefix = out_name.strip_suffix(suffix).ok_or_else(|| Error::Format(format!("qwen4_exp: invalid fused expert '{name}'")))?;
    let (rows, cols) = if gate_up { (cfg.moe_inter, cfg.hidden) } else { (cfg.hidden, cfg.moe_inter) };
    let count = rows.checked_mul(cols).ok_or_else(|| Error::Format(format!("'{name}': expert size overflow")))?;
    let mut dense = vec![0f32; usize::try_from(count).map_err(|_| Error::Format(format!("'{name}': expert too large")))?];
    let projections: &[&str] = if gate_up { &["gate", "up"] } else { &["down"] };
    for expert in 0..cfg.n_experts {
        for (part, projection) in projections.iter().enumerate() {
            let offset = (expert * projections.len() as i64 + part as i64) * count;
            st.read_slice_f32(name, offset, count, &mut dense)?;
            if dense.iter().any(|v| !v.is_finite()) {
                return Err(Error::Format(format!("'{name}': non-finite expert {expert} {projection}")));
            }
            let (packed, scales) = quant_i4(&dense, rows as usize, cols as usize);
            let output = format!("{prefix}.{expert}.{projection}_proj.weight");
            rep.imported_int4 += 1;
            rep.bytes_out += (packed.len() + scales.len() * 4) as u64;
            writer.push(&output, "U8", vec![rows, (cols + 1) / 2], packed)?;
            writer.push(&format!("{output}.qs"), "F32", vec![rows], scales.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        }
    }
    Ok(())
}

/// Read a dense tensor as f32, folding in its FP8 block scales when a
/// `{name}_scale_inv` sibling exists (128×128-class blocks; the per-axis block
/// size is derived from the scale tensor's own shape, so any block geometry
/// the checkpoint declares round-trips).
fn read_dense_dequant(st: &SafeTensors, name: &str, shape: &[i64]) -> Result<Vec<f32>, Error> {
    let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
    let mut dense = vec![0f32; numel];
    st.read_f32(name, &mut dense)?;
    let scale_name = format!("{name}_scale_inv");
    if !st.has(&scale_name) {
        return Ok(dense);
    }
    let (&[o, i], true) = (shape, shape.len() == 2) else {
        return Err(Error::Format(format!(
            "'{name}': carries FP8 block scales but its shape {shape:?} is not a 2-D matrix"
        )));
    };
    let (o, i) = (o.max(0) as usize, i.max(0) as usize);
    let st_shape = match st.tensors().iter().find(|t| t.name == scale_name) {
        Some(t) => t.shape.clone(),
        None => {
            return Err(Error::Format(format!(
                "'{scale_name}': present by `has` but absent from the index — corrupt header"
            )))
        }
    };
    let (&[so, si], true) = (st_shape.as_slice(), st_shape.len() == 2) else {
        return Err(Error::Format(format!("'{scale_name}': expected a 2-D block-scale tensor, got {st_shape:?}")));
    };
    let (so, si) = (so.max(1) as usize, si.max(1) as usize);
    let (br, bc) = (o.div_ceil(so), i.div_ceil(si));
    let mut scales = vec![0f32; so * si];
    st.read_f32(&scale_name, &mut scales)?;
    for r in 0..o {
        let srow = &scales[(r / br) * si..(r / br) * si + si];
        for c in 0..i {
            dense[r * i + c] *= srow[c / bc];
        }
    }
    Ok(dense)
}

/// What an import did, for the operator line and the tests.
#[derive(Debug, Default, Clone)]
pub struct ImportReport {
    pub tensors_total: usize,
    pub imported_int4: usize,
    pub imported_int8: usize,
    pub imported_float: usize,
    pub skipped_visual: usize,
    pub skipped_mtp: usize,
    pub imported_i64: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub sidecars: Vec<String>,
}

/// Import every text-stack tensor of the HF checkpoint at `indir` into a
/// peregrine container at `outdir`. Names travel verbatim; shapes come from
/// the bf16 headers (unlike a packed container, a dense tensor's header shape
/// IS its logical shape). `config.json` and the tokenizer sidecars are copied
/// unmodified — the loader's `Arch` detection reads the HF config directly.
pub fn import_hf(indir: &Path, outdir: &Path, shard_bytes: u64) -> Result<ImportReport, Error> {
    // The contract is keyed on the checkpoint's own declaration, same rule as
    // `Cfg::from_json`: a glm5_next checkpoint takes the GLM-5.3 table (stem
    // rename + FP8 block dequant), everything else the Track C REV 2 table.
    let cfg_bytes = std::fs::read(indir.join("config.json"))
        .ctx(|| format!("{}: config.json (the import contract is keyed on model_type)", indir.display()))?;
    let cfg_root: serde_json::Value = serde_json::from_slice(&cfg_bytes)?;
    let glm5next = cfg_root
        .get("model_type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t.starts_with("glm5_next"));
    let qwen4 = if matches!(cfg_root.get("model_type").and_then(|v| v.as_str()), Some("qwen4_exp" | "qwen4_exp_text")) {
        Some(Cfg::from_json(&cfg_root)?)
    } else {
        None
    };
    let st = SafeTensors::open(indir).ctx(|| if qwen4.is_some() {
        "qwen4_exp source: requires native BF16/F16/F32 weights and I64 PLE buffers".into()
    } else {
        format!("open {}", indir.display())
    })?;
    let ple = qwen4.as_ref().map(|cfg| qwen4_preflight(&st, cfg)).transpose()?.unwrap_or_default();
    if qwen4.is_some() && outdir.exists() && std::fs::read_dir(outdir)?.next().is_some() {
        return Err(Error::Format("qwen4_exp: output directory must be empty".into()));
    }
    std::fs::create_dir_all(outdir).ctx(|| format!("create {}", outdir.display()))?;
    let contract = if qwen4.is_some() { "qwen4-exp-foundation" } else if glm5next { "glm5-next" } else { "track-c-rev2" };
    let skipped_mtp: Vec<_> = st.tensors().iter().filter(|t| qwen4.is_some() && qwen4_skip_mtp(&t.name)).map(|t| t.name.clone()).collect();
    let mut w = ShardWriter::new(outdir, "model", shard_bytes).with_metadata(vec![
        ("peregrine.import.tool".into(), "peregrine-import-hf".into()),
        ("peregrine.import.contract".into(), contract.into()),
        ("peregrine.import.source".into(), indir.display().to_string()),
        ("peregrine.import.skipped_mtp".into(), serde_json::to_string(&skipped_mtp)?),
    ]);
    let mut rep = ImportReport::default();
    let mut ple_sources = BTreeSet::new();
    for plan in &ple {
        qwen4_write_ple(&st, plan, outdir, &mut rep)?;
        for (name, _) in &plan.shards { ple_sources.insert(name.clone()); }
        for (name, values) in &plan.buffers {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            rep.bytes_out += bytes.len() as u64;
            rep.imported_i64 += 1;
            w.push(name, "I64", vec![values.len() as i64], bytes)?;
        }
    }
    let ple_buffers: BTreeSet<_> = ple.iter().flat_map(|p| p.buffers.iter().map(|(name, _)| name.clone())).collect();
    let qwen4_tensors = qwen4.as_ref().map(qwen4_contract).transpose()?;

    let meta: Vec<(String, Vec<i64>)> =
        st.tensors().iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    for (name, shape) in &meta {
        rep.tensors_total += 1;
        rep.bytes_in += st.uncompressed_nbytes(name).unwrap_or(0).max(0) as u64;
        // Resolve the output name and family per contract.
        let (family, out_name) = if let (Some(cfg), Some(contract)) = (&qwen4, &qwen4_tensors) {
            if name.starts_with("model.visual.") {
                rep.skipped_visual += 1;
                continue;
            }
            if qwen4_skip_mtp(name) {
                rep.skipped_mtp += 1;
                continue;
            }
            let out_name = qwen4_rename(name);
            if ple_sources.contains(name) || ple_buffers.contains(&out_name) { continue; }
            let (kind, _) = contract.get(&out_name).ok_or_else(|| Error::Format(format!("qwen4_exp: unplanned tensor '{name}'")))?;
            match kind {
                Qwen4Tensor::Dense(family) => (*family, out_name),
                Qwen4Tensor::GateUp | Qwen4Tensor::Down => {
                    qwen4_write_experts(&st, name, &out_name, cfg, &mut w, &mut rep)?;
                    continue;
                }
            }
        } else if glm5next {
            let out_name = match glm5next_rename(name) {
                Some(n) => n,
                None => {
                    rep.skipped_visual += 1;
                    continue;
                }
            };
            (classify_glm5next(&out_name)?, out_name)
        } else {
            (classify(name)?, name.clone())
        };
        match family {
            Family::SkipVisual => {
                rep.skipped_visual += 1;
                continue;
            }
            // Consumed while dequantizing its base tensor below.
            Family::SkipScale => continue,
            _ => {}
        }
        // One dense read covers every family: `read_f32` widens bf16/f16/fp8
        // exactly, and `read_dense_dequant` folds FP8 block scales back in.
        let dense = read_dense_dequant(&st, name, shape)?;
        if qwen4.is_some() && dense.iter().any(|v| !v.is_finite()) {
            return Err(Error::Format(format!("'{name}': qwen4_exp non-finite tensor")));
        }
        match family {
            Family::Int4 | Family::Int8Embed => {
                let (&[o, i], true) = (shape.as_slice(), shape.len() == 2) else {
                    return Err(Error::Format(format!(
                        "'{name}': the contract quantizes this family, but its shape {shape:?} \
                         is not a 2-D matrix — the checkpoint does not match the contract"
                    )));
                };
                let (o, i) = (o.max(0) as usize, i.max(0) as usize);
                let (q, s, out_cols, count) = if family == Family::Int4 {
                    let (q, s) = quant_i4(&dense, o, i);
                    (q, s, i.div_ceil(2) as i64, &mut rep.imported_int4)
                } else {
                    let (q, s) = quant_i8(&dense, o, i);
                    (q, s, i as i64, &mut rep.imported_int8)
                };
                *count += 1;
                rep.bytes_out += (q.len() + s.len() * 4) as u64;
                w.push(&out_name, "U8", vec![o as i64, out_cols], q)?;
                let s_bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.push(&format!("{out_name}.qs"), "F32", vec![s.len() as i64], s_bytes)?;
            }
            Family::Float => {
                // Original shape preserved — conv1d stays 3-D.
                rep.imported_float += 1;
                rep.bytes_out += (dense.len() * 4) as u64;
                let bytes: Vec<u8> = dense.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.push(&out_name, "F32", shape.clone(), bytes)?;
            }
            Family::SkipVisual | Family::SkipScale => {}
        }
    }
    w.flush()?;

    // Same rule as requantize: shards alone are not a model. The HF config is
    // copied VERBATIM — `Cfg`'s Arch detection is written against it.
    for side in ["config.json", "tokenizer.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"] {
        let src = indir.join(side);
        if src.exists() {
            std::fs::copy(&src, outdir.join(side)).ctx(|| format!("copy {side}"))?;
            rep.sidecars.push(side.to_string());
        }
    }
    if !rep.sidecars.iter().any(|s| s == "config.json") {
        return Err(Error::Format(format!(
            "{}: no config.json next to the shards — the output would not be loadable",
            indir.display()
        )));
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_core::pack::{write_safetensors, Blob};

    fn tiny_qwen4_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen4_exp_text", "hidden_size": 4, "vocab_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 4, "num_experts": 2, "num_experts_per_tok": 1,
            "moe_intermediate_size": 3, "shared_expert_intermediate_size": 2,
            "hc_count": 2, "hc_lowrank": 3, "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 2
        })
    }

    fn tiny_qwen4_blobs() -> Vec<Blob> {
        let mut blobs = vec![
            bf16_blob("model.language_model.embed_tokens.weight", vec![8, 4], 1),
            bf16_blob("lm_head.weight", vec![8, 4], 2),
        ];
        let p = "model.language_model.layers.0";
        for (name, shape) in [
            ("mlp.gate.weight", vec![2, 4]),
            ("mlp.shared_expert_gate.weight", vec![1, 4]),
            ("mlp.shared_expert.gate_proj.weight", vec![2, 4]),
            ("mlp.shared_expert.up_proj.weight", vec![2, 4]),
            ("mlp.shared_expert.down_proj.weight", vec![4, 2]),
            ("mlp.experts.gate_up_proj", vec![2, 6, 4]),
            ("mlp.experts.down_proj", vec![2, 4, 3]),
            ("linear_attn.in_proj_qkv.weight", vec![6, 4]),
            ("linear_attn.in_proj_z.weight", vec![2, 4]),
            ("linear_attn.in_proj_a.weight", vec![1, 4]),
            ("linear_attn.in_proj_b.weight", vec![1, 4]),
            ("linear_attn.out_proj.weight", vec![4, 2]),
            ("linear_attn.conv1d.weight", vec![6, 1, 2]),
            ("linear_attn.A_log", vec![1]),
            ("linear_attn.dt_bias", vec![1]),
            ("linear_attn.norm.weight", vec![2]),
        ] {
            blobs.push(bf16_blob(&format!("{p}.{name}"), shape, 17));
        }
        for prefix in [format!("{p}.attn_hyper_connection"), format!("{p}.mlp_hyper_connection"), "model.language_model.hyper_connection_mixer".into()] {
            blobs.push(bf16_blob(&format!("{prefix}.hc_norm.weight"), vec![8], 19));
            blobs.push(bf16_blob(&format!("{prefix}.input_mix_weight_down.weight"), vec![3, 8], 20));
            blobs.push(bf16_blob(&format!("{prefix}.input_mix_weight_up.weight"), vec![8, 3], 21));
            if prefix.contains("layers") {
                blobs.push(bf16_blob(&format!("{prefix}.block_inject_weight.weight"), vec![2, 8], 22));
            }
        }
        blobs
    }

    fn write_qwen4_fixture(dir: &Path, cfg: &serde_json::Value, blobs: &[Blob]) -> Result<(), Error> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), serde_json::to_vec(cfg)?)?;
        write_safetensors(dir, blobs)?;
        Ok(())
    }

    fn tiny_ple_fixture(dtype: &str, buffers: bool) -> (serde_json::Value, Vec<Blob>, Vec<f32>) {
        let mut cfg = tiny_qwen4_config();
        for (key, value) in [("ple_embed_dim", 12), ("ngram_size", 3), ("heads_per_ngram", 2),
            ("ngram_vocab_size_base", 11), ("make_ngram_vocab_size_divisible_by", 8),
            ("split_ngram_parts", 16), ("eos_token_id", 0), ("seed", 1234)] {
            cfg[key] = serde_json::json!(value);
        }
        cfg["ple_layer_ids"] = serde_json::json!([1]);
        let mut blobs = tiny_qwen4_blobs();
        let prefix = "model.language_model.layers.0.ple";
        for (suffix, shape) in [("key_proj.weight", vec![8, 12]), ("value_proj.weight", vec![4, 12]),
            ("norm_key.weight", vec![8]), ("norm_query.weight", vec![8]),
            ("norm_conv.weight", vec![8]), ("conv1d.weight", vec![8, 1, 4])] {
            blobs.push(bf16_blob(&format!("{prefix}.{suffix}"), shape, 25));
        }
        let dense: Vec<f32> = (0..64).flat_map(|r| {
            let a = (r + 1) as f32 / 16.0;
            [a, -a * 0.5, a * 0.25]
        }).collect();
        for shard in (0..16).rev() {
            let values = &dense[shard * 12..(shard + 1) * 12];
            let bytes = match dtype {
                "BF16" => values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<_>>(),
                "F16" => values.iter().flat_map(|&v| peregrine_core::f32_to_f16(v).to_le_bytes()).collect(),
                _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            blobs.push(Blob::new(format!("{prefix}.ple_embedding.ngram_embedding.shard_{shard}.weight"), dtype, vec![4, 3], bytes));
        }
        if buffers {
            let bound = (i64::MAX / 8 / 2) as u64;
            let multipliers: Vec<i64> = (1..=3).map(|i| {
                let mut z = 1234u64.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(i + 1));
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                (2 * ((z ^ (z >> 31)) % bound) + 1) as i64
            }).collect();
            for (suffix, values) in [("layer_multipliers", multipliers),
                ("ngram_heads_vocab_sizes", vec![11i64, 13, 17, 19]),
                ("ngram_heads_offsets", vec![0, 11, 24, 41])] {
                blobs.push(Blob::new(format!("{prefix}.ple_embedding.{suffix}"), "I64", vec![values.len() as i64],
                    values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()));
            }
        }
        (cfg, blobs, dense)
    }

    #[test]
    fn qwen4_ple_preflight_checks_numeric_shards_and_buffers() -> Result<(), Error> {
        let (dir, _) = fixture_dirs("ple_preflight")?;
        let (json, blobs, _) = tiny_ple_fixture("BF16", true);
        write_qwen4_fixture(&dir, &json, &blobs)?;
        let plans = qwen4_preflight(&SafeTensors::open(&dir)?, &Cfg::from_json(&json)?)?;
        assert_eq!(plans.len(), 1);
        assert_eq!((plans[0].rows, plans[0].cols), (64, 3));
        assert_eq!(plans[0].buffers[1].1, [11, 13, 17, 19]);
        assert_eq!(plans[0].buffers[2].1, [0, 11, 24, 41]);
        for (i, (name, rows)) in plans[0].shards.iter().enumerate() {
            assert!(name.ends_with(&format!("shard_{i}.weight")));
            assert_eq!(*rows, 4);
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn qwen4_ple_import_exact_int8_scales_i64_and_optional_buffers() -> Result<(), Error> {
        for dtype in ["BF16", "F16", "F32"] {
            for buffers in [false, true] {
                let (dir, out) = fixture_dirs(&format!("ple_roundtrip_{dtype}_{buffers}"))?;
                let (json, blobs, dense) = tiny_ple_fixture(dtype, buffers);
                write_qwen4_fixture(&dir, &json, &blobs)?;
                let report = import_hf(&dir, &out, 64)?;
                assert_eq!(report.imported_int8, 2);
                assert_eq!(report.imported_i64, 3);
                let st = SafeTensors::open(&out)?;
                let name = "model.layers.0.ple.ple_embedding.ngram_embedding.weight";
                let tensor = st.find(name).ok_or_else(|| Error::Format(name.into()))?;
                assert_eq!(tensor.shape, [64, 3]);
                assert_eq!(tensor.dtype, Dtype::U8);
                let qs = st.find(&format!("{name}.qs")).ok_or_else(|| Error::Format(format!("{name}.qs")))?;
                assert_eq!(qs.shape, [64]);
                let mut packed = vec![0u8; dense.len()];
                let mut scales = vec![0.0; 64];
                st.read_raw(name, &mut packed)?;
                st.read_f32(&format!("{name}.qs"), &mut scales)?;
                let expected = quant_i8(&dense, 64, 3);
                assert_eq!(packed, expected.0);
                assert_eq!(scales, expected.1);
                for row in 0..64 {
                    assert_eq!(packed[row * 3], 127);
                    assert_eq!(packed[row * 3 + 1], ((dense[row * 3 + 1] / scales[row]).round_ties_even() as i8) as u8);
                    assert_eq!(packed[row * 3 + 2], 32);
                    assert_eq!(scales[row], dense[row * 3] / 127.0);
                    for col in 0..3 {
                        let i = row * 3 + col;
                        assert!(((packed[i] as i8 as f32) * scales[row] - dense[i]).abs() <= scales[row] * 0.501);
                    }
                }
                for (buffer, values) in qwen4_ple_geometry(&Cfg::from_json(&json)?)?.remove(0).buffers {
                    let mut got = vec![0; values.len()];
                    st.read_i64(&buffer, &mut got)?;
                    assert_eq!(got, values);
                }
                assert!(!st.tensors().iter().any(|t| t.name.contains("shard_")));
                assert_eq!(report.bytes_out, st.tensors().iter().map(|t| t.nbytes as u64).sum::<u64>());
                std::fs::remove_dir_all(dir)?;
                std::fs::remove_dir_all(out)?;
            }
        }
        Ok(())
    }

    #[test]
    fn qwen4_ple_refuses_mismatch_missing_extra_shape_dtype_and_collision() -> Result<(), Error> {
        for case in ["multipliers", "sizes", "offsets", "missing", "extra", "shape", "width", "fp8", "collision", "i64_dtype", "total"] {
            let (dir, out) = fixture_dirs(&format!("ple_reject_{case}"))?;
            let (mut json, mut blobs, _) = tiny_ple_fixture("BF16", true);
            let message = match case {
                "multipliers" | "sizes" | "offsets" => {
                    let suffix = match case { "multipliers" => "layer_multipliers", "sizes" => "ngram_heads_vocab_sizes", _ => "ngram_heads_offsets" };
                    let b = blobs.iter_mut().find(|b| b.name.ends_with(suffix)).ok_or_else(|| Error::Format(suffix.into()))?;
                    b.bytes[0] ^= 1;
                    "buffer mismatch"
                }
                "missing" => { blobs.retain(|b| !b.name.ends_with("shard_2.weight")); "missing required tensor" }
                "extra" => {
                    blobs.push(bf16_blob("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_16.weight", vec![4, 3], 0));
                    "shard_16"
                }
                "total" => { json["split_ngram_parts"] = serde_json::json!(49); "nonempty row shards" }
                _ => {
                    let b = blobs.iter_mut().find(|b| b.name.ends_with("shard_0.weight")).ok_or_else(|| Error::Format("shard_0".into()))?;
                    match case {
                        "shape" => { b.shape = vec![3, 4]; "expected shape" }
                        "width" => { b.shape = vec![1, 12]; "expected shape" }
                        "fp8" => { b.dtype = "F8_E4M3".into(); b.bytes = vec![0; 12]; "FP8 sources are unsupported" }
                        "collision" => {
                            let copy = Blob::new(qwen4_rename(&b.name), "BF16", b.shape.clone(), b.bytes.clone());
                            blobs.push(copy);
                            "rename collision"
                        }
                        _ => {
                            let b = blobs.iter_mut().find(|b| b.name.ends_with("layer_multipliers")).ok_or_else(|| Error::Format("layer_multipliers".into()))?;
                            b.dtype = "F32".into(); b.bytes = vec![0; 12];
                            "requires native uncompressed I64"
                        }
                    }
                }
            };
            write_qwen4_fixture(&dir, &json, &blobs)?;
            let error = import_hf(&dir, &out, 64).err().ok_or_else(|| Error::Format(format!("{case}: unexpectedly accepted")))?.to_string();
            assert!(error.contains(message), "{case}: {error}");
            assert!(!out.exists());
            std::fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    #[test]
    fn qwen4_tiny_import_splits_exact_experts_and_loads() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("qwen4_exact")?;
        let blobs = tiny_qwen4_blobs();
        write_qwen4_fixture(&dir, &tiny_qwen4_config(), &blobs)?;
        let report = import_hf(&dir, &out, 64)?;
        assert_eq!(report.imported_int4, 12);
        assert_eq!(report.imported_int8, 1);
        assert_eq!(report.imported_float, 20);
        let st = SafeTensors::open(&out)?;
        let mut expected = BTreeSet::new();
        for blob in &blobs {
            let name = qwen4_rename(&blob.name);
            if name.ends_with("mlp.experts.gate_up_proj") || name.ends_with("mlp.experts.down_proj") {
                continue;
            }
            expected.insert(name.clone());
            let source = bf16_exact(blob.bytes.len() / 2, match blob.name.as_str() {
                "model.language_model.embed_tokens.weight" => 1,
                "lm_head.weight" => 2,
                n if n.ends_with("hc_norm.weight") => 19,
                n if n.ends_with("input_mix_weight_down.weight") => 20,
                n if n.ends_with("input_mix_weight_up.weight") => 21,
                n if n.ends_with("block_inject_weight.weight") => 22,
                _ => 17,
            });
            let quantized = name == "lm_head.weight" || name == "model.embed_tokens.weight"
                || name.contains(".shared_experts.") || name.ends_with("in_proj_qkv.weight") || name.ends_with("out_proj.weight");
            if quantized {
                expected.insert(format!("{name}.qs"));
            } else {
                let tensor = st.find(&name).ok_or_else(|| Error::Format(name.clone()))?;
                assert_eq!(tensor.dtype, Dtype::F32, "{name}");
                assert_eq!(tensor.shape, blob.shape, "{name}");
                let mut got = vec![0f32; source.len()];
                st.read_f32(&name, &mut got)?;
                assert_eq!(got, source, "{name}");
            }
        }
        for expert in 0..2 {
            for proj in ["gate", "up", "down"] {
                let (rows, cols, total, offset) = match proj {
                    "gate" => (3, 4, 48, expert * 24),
                    "up" => (3, 4, 48, expert * 24 + 12),
                    _ => (4, 3, 24, expert * 12),
                };
                let dense = bf16_exact(total, 17);
                let (packed, scales) = quant_i4(&dense[offset..offset + 12], rows, cols);
                let name = format!("model.layers.0.mlp.experts.{expert}.{proj}_proj.weight");
                expected.insert(name.clone());
                expected.insert(format!("{name}.qs"));
                let tensor = st.find(&name).ok_or_else(|| Error::Format(name.clone()))?;
                assert_eq!(tensor.shape, vec![rows as i64, cols.div_ceil(2) as i64]);
                let mut got = vec![0u8; packed.len()];
                st.read_raw(&name, &mut got)?;
                assert_eq!(got, packed, "{name}");
                let mut got_scales = vec![0f32; rows];
                st.read_f32(&format!("{name}.qs"), &mut got_scales)?;
                assert_eq!(got_scales, scales, "{name}");
            }
        }
        assert_eq!(st.tensors().iter().map(|t| t.name.clone()).collect::<BTreeSet<_>>(), expected);
        assert_eq!(std::fs::read(out.join("config.json"))?, std::fs::read(dir.join("config.json"))?);
        // The single-stream resident forward is implemented — an imported
        // container must load (it previously stopped at the production guard).
        let _m = peregrine_model::Model::load(&out)?;
        std::fs::remove_dir_all(dir)?;
        std::fs::remove_dir_all(out)?;
        Ok(())
    }

    #[test]
    fn qwen4_native_f16_f32_sources_and_foundation_metadata() -> Result<(), Error> {
        for dtype in ["F16", "F32"] {
            let (dir, out) = fixture_dirs(&format!("qwen4_{dtype}"))?;
            let blobs = tiny_qwen4_blobs().into_iter().map(|blob| {
                let dense: Vec<f32> = blob.bytes.as_chunks::<2>().0.iter()
                    .map(|b| peregrine_core::bf16_to_f32(u16::from_le_bytes(*b))).collect();
                let bytes = if dtype == "F16" {
                    dense.iter().flat_map(|&v| peregrine_core::f32_to_f16(v).to_le_bytes()).collect::<Vec<_>>()
                } else {
                    dense.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
                };
                Blob::new(qwen4_rename(&blob.name), dtype, blob.shape, bytes)
            }).collect::<Vec<_>>();
            write_qwen4_fixture(&dir, &tiny_qwen4_config(), &blobs)?;
            import_hf(&dir, &out, 1 << 20)?;
            let st = SafeTensors::open(&out)?;
            let mut got = vec![0f32; 8];
            st.read_f32("model.hyper_connection_mixer.hc_norm.weight", &mut got)?;
            assert_eq!(got, bf16_exact(8, 19));
            let bytes = std::fs::read(out.join("model-00000.safetensors"))?;
            let header_len = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| Error::Format("fixture header".into()))?) as usize;
            let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
            assert_eq!(header["__metadata__"]["peregrine.import.contract"], "qwen4-exp-foundation");
            std::fs::remove_dir_all(dir)?;
            std::fs::remove_dir_all(out)?;
        }
        Ok(())
    }

    #[test]
    fn qwen4_rejections_leave_no_output() -> Result<(), Error> {
        for case in ["config", "ple", "unknown", "missing", "shape", "collision", "fp8", "mtp", "ngram"] {
            let (dir, out) = fixture_dirs(&format!("qwen4_reject_{case}"))?;
            let mut cfg = tiny_qwen4_config();
            let mut blobs = tiny_qwen4_blobs();
            let message = match case {
                "config" => {
                    cfg["hc_count"] = serde_json::json!(1);
                    "hc_count"
                }
                "ple" => {
                    cfg["ple_layer_ids"] = serde_json::json!([1]);
                    cfg["eos_token_id"] = serde_json::json!(1);
                    cfg["heads_per_ngram"] = serde_json::json!(1);
                    blobs.push(Blob::new("model.language_model.layers.0.ple.ple_embedding.layer_multipliers", "I64", vec![3],
                        [i64::MAX, (1i64 << 54) + 3, 17].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()));
                    "buffer mismatch"
                }
                "missing" => {
                    blobs.remove(0);
                    "missing required tensor"
                }
                "shape" => {
                    let b = blobs.iter_mut().find(|b| b.name.ends_with("experts.gate_up_proj")).ok_or_else(|| Error::Format("fixture".into()))?;
                    b.shape = vec![2, 4, 6];
                    "expected shape"
                }
                "collision" => {
                    blobs.push(bf16_blob("model.embed_tokens.weight", vec![8, 4], 1));
                    "rename collision"
                }
                "fp8" => {
                    blobs[0] = Blob::new("model.language_model.embed_tokens.weight", "F8_E4M3", vec![8, 4], vec![0u8; 32]);
                    "FP8 sources are unsupported"
                }
                "mtp" => {
                    blobs.push(bf16_blob("mtp.fc.weight", vec![4, 4], 1));
                    "mtp.fc.weight"
                }
                "ngram" => {
                    blobs.push(bf16_blob("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.0.weight", vec![4, 4], 1));
                    "ngram_embedding.0.weight"
                }
                _ => {
                    blobs.push(bf16_blob("model.language_model.layers.0.linear_attn.mystery.weight", vec![4, 4], 1));
                    "mystery"
                }
            };
            write_qwen4_fixture(&dir, &cfg, &blobs)?;
            if case == "config" {
                std::fs::remove_file(dir.join("model.safetensors"))?;
            }
            let error = import_hf(&dir, &out, 1).err().ok_or_else(|| Error::Format(format!("{case}: unexpectedly accepted")))?.to_string();
            assert!(error.contains(message), "{case}: {error}");
            assert!(!out.exists(), "{case} created output");
            std::fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    #[test]
    fn qwen4_validated_partial_ple_policy_and_qsa_import() -> Result<(), Error> {
        let mut json = tiny_qwen4_config();
        json["ple_layer_ids"] = serde_json::json!([1]);
        json["heads_per_ngram"] = serde_json::json!(1);
        json["eos_token_id"] = serde_json::json!(1);
        let cfg = Cfg::from_json(&json)?;
        let contract = qwen4_contract(&cfg)?;
        for (suffix, shape) in [
            ("key_proj.weight", vec![8, 4]), ("value_proj.weight", vec![4, 4]),
            ("norm_key.weight", vec![8]), ("norm_query.weight", vec![8]),
            ("norm_conv.weight", vec![8]), ("conv1d.weight", vec![8, 1, 4]),
        ] {
            assert_eq!(contract.get(&format!("model.layers.0.ple.{suffix}")), Some(&(Qwen4Tensor::Dense(Family::Float), shape)));
        }
        json["ple_layer_ids"] = serde_json::json!([]);
        json["layer_types"] = serde_json::json!(["full_attention"]);
        for (key, value) in [("indexer_n_heads", 2), ("indexer_kv_heads", 1), ("indexer_head_dim", 4), ("indexer_budget", 4), ("indexer_compress_ratio", 2)] {
            json[key] = serde_json::json!(value);
        }
        let wrapped = serde_json::json!({"model_type": "qwen4_exp", "text_config": json});
        let mut blobs = tiny_qwen4_blobs();
        blobs.retain(|b| !b.name.contains("linear_attn"));
        for (suffix, shape) in [
            ("q_proj.weight", vec![8, 4]), ("k_proj.weight", vec![4, 4]),
            ("v_proj.weight", vec![4, 4]), ("o_proj.weight", vec![4, 4]),
            ("q_norm.weight", vec![4]), ("k_norm.weight", vec![4]),
            ("indexer.index_qk_proj.weight", vec![12, 4]),
            ("indexer.q_layernorm.weight", vec![4]), ("indexer.k_layernorm.weight", vec![4]),
        ] {
            blobs.push(bf16_blob(&format!("model.language_model.layers.0.self_attn.{suffix}"), shape, 23));
        }
        blobs.push(bf16_blob("model.visual.patch_embed.proj.weight", vec![2, 2], 1));
        let (dir, out) = fixture_dirs("qwen4_qsa")?;
        write_qwen4_fixture(&dir, &wrapped, &blobs)?;
        let report = import_hf(&dir, &out, 128)?;
        assert_eq!(report.skipped_visual, 1);
        let st = SafeTensors::open(&out)?;
        for suffix in ["index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight"] {
            let name = format!("model.layers.0.self_attn.indexer.{suffix}");
            let t = st.find(&name).ok_or_else(|| Error::Format(name.clone()))?;
            assert_eq!(t.dtype, Dtype::F32);
            assert!(!st.has(&format!("{name}.qs")));
            let mut got = vec![0f32; t.numel as usize];
            st.read_f32(&name, &mut got)?;
            assert_eq!(got, bf16_exact(t.numel as usize, 23));
        }
        let error = import_hf(&dir, &out, 128).err().ok_or_else(|| Error::Format("nonempty output unexpectedly accepted".into()))?.to_string();
        assert!(error.contains("must be empty"), "{error}");
        std::fs::remove_dir_all(dir)?;
        std::fs::remove_dir_all(out)?;
        Ok(())
    }

    /// f32 → bf16 by truncation. Test fixtures generate values that are exact
    /// in bf16, so the widening read is bit-exact and comparisons need no
    /// tolerance for the float family.
    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    }

    /// Deterministic values that survive f32→bf16→f32 exactly.
    fn bf16_exact(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|k| {
                let x = ((k as u32).wrapping_mul(2654435761).wrapping_add(seed) % 255) as f32;
                (x - 127.0) / 64.0 // multiples of 1/64 in a small range: exact in bf16
            })
            .collect()
    }

    fn bf16_blob(name: &str, shape: Vec<i64>, seed: u32) -> Blob {
        let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
        Blob::new(name.to_string(), "BF16", shape, bf16_bytes(&bf16_exact(numel, seed)))
    }

    /// A synthetic bf16 HF checkpoint at the canonical tiny-hybrid dims
    /// (`peregrine_model::testkit::tiny_hybrid_cfg_json`): 3 layers
    /// [linear, linear, full], gate on, plus a vision tensor to skip.
    /// `mystery` adds one tensor from outside the contract, for the refusal test.
    fn build_hf_hybrid_fixture(dir: &std::path::Path, mystery: bool) -> Result<(), Error> {
        let cfg = peregrine_model::testkit::tiny_hybrid_cfg_json();
        let (d, di, vocab) = (16usize, 8usize, 32usize);
        let (nh, hd) = (4usize, 8usize);
        let nkv = 2usize;
        let (kh, vh, kd, vd, taps) = (2usize, 4usize, 4usize, 4usize, 4usize);
        let conv_dim = 2 * kh * kd + vh * vd;
        let stem = "model.language_model";

        let mut blobs = Vec::new();
        let mut seed = 1u32;
        let mut push = |blobs: &mut Vec<Blob>, name: String, shape: Vec<i64>| {
            seed += 1;
            blobs.push(bf16_blob(&name, shape, seed));
        };
        push(&mut blobs, format!("{stem}.embed_tokens.weight"), vec![vocab as i64, d as i64]);
        push(&mut blobs, "lm_head.weight".into(), vec![vocab as i64, d as i64]);
        push(&mut blobs, format!("{stem}.norm.weight"), vec![d as i64]);
        push(&mut blobs, "model.visual.patch_embed.proj.weight".into(), vec![4, 4]);
        for l in 0..3usize {
            let p = |s: &str| format!("{stem}.layers.{l}.{s}");
            push(&mut blobs, p("input_layernorm.weight"), vec![d as i64]);
            push(&mut blobs, p("post_attention_layernorm.weight"), vec![d as i64]);
            if l == 2 {
                // full attention, output gate on: q emits query|gate.
                push(&mut blobs, p("self_attn.q_proj.weight"), vec![(2 * nh * hd) as i64, d as i64]);
                push(&mut blobs, p("self_attn.k_proj.weight"), vec![(nkv * hd) as i64, d as i64]);
                push(&mut blobs, p("self_attn.v_proj.weight"), vec![(nkv * hd) as i64, d as i64]);
                push(&mut blobs, p("self_attn.o_proj.weight"), vec![d as i64, (nh * hd) as i64]);
                push(&mut blobs, p("self_attn.q_norm.weight"), vec![hd as i64]);
                push(&mut blobs, p("self_attn.k_norm.weight"), vec![hd as i64]);
            } else {
                push(&mut blobs, p("linear_attn.in_proj_qkv.weight"), vec![conv_dim as i64, d as i64]);
                push(&mut blobs, p("linear_attn.in_proj_z.weight"), vec![(vh * vd) as i64, d as i64]);
                push(&mut blobs, p("linear_attn.in_proj_a.weight"), vec![vh as i64, d as i64]);
                push(&mut blobs, p("linear_attn.in_proj_b.weight"), vec![vh as i64, d as i64]);
                push(&mut blobs, p("linear_attn.conv1d.weight"), vec![conv_dim as i64, 1, taps as i64]);
                push(&mut blobs, p("linear_attn.A_log"), vec![vh as i64]);
                push(&mut blobs, p("linear_attn.dt_bias"), vec![vh as i64]);
                push(&mut blobs, p("linear_attn.norm.weight"), vec![vd as i64]);
                push(&mut blobs, p("linear_attn.out_proj.weight"), vec![d as i64, (vh * vd) as i64]);
            }
            push(&mut blobs, p("mlp.gate_proj.weight"), vec![di as i64, d as i64]);
            push(&mut blobs, p("mlp.up_proj.weight"), vec![di as i64, d as i64]);
            push(&mut blobs, p("mlp.down_proj.weight"), vec![d as i64, di as i64]);
        }
        if mystery {
            push(&mut blobs, format!("{stem}.layers.0.linear_attn.mystery.weight"), vec![4, 4]);
        }
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).map_err(|e| Error::Format(e.to_string()))?)?;
        write_safetensors(dir, &blobs)?;
        Ok(())
    }

    fn fixture_dirs(tag: &str) -> Result<(std::path::PathBuf, std::path::PathBuf), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_import_{}_{}", std::process::id(), tag));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        Ok((dir, out))
    }

    #[test]
    fn classify_covers_the_rev2_contract_and_refuses_the_rest() {
        let p = "model.language_model.layers.7.";
        for n in [
            format!("{p}self_attn.q_proj.weight"),
            format!("{p}linear_attn.in_proj_qkv.weight"),
            format!("{p}linear_attn.out_proj.weight"),
            format!("{p}mlp.down_proj.weight"),
            "lm_head.weight".into(),
            "mtp.fc.weight".into(),
            "mtp.layers.0.self_attn.o_proj.weight".into(),
        ] {
            assert_eq!(classify(&n).ok(), Some(Family::Int4), "{n}");
        }
        for n in [
            format!("{p}linear_attn.conv1d.weight"),
            format!("{p}linear_attn.A_log"),
            format!("{p}linear_attn.dt_bias"),
            format!("{p}linear_attn.norm.weight"),
            format!("{p}input_layernorm.weight"),
            format!("{p}self_attn.k_norm.weight"),
            "model.language_model.norm.weight".into(),
            "mtp.pre_fc_norm_hidden.weight".into(),
            "mtp.layers.0.post_attention_layernorm.weight".into(),
        ] {
            assert_eq!(classify(&n).ok(), Some(Family::Float), "{n}");
        }
        assert_eq!(classify("model.language_model.embed_tokens.weight").ok(), Some(Family::Int8Embed));
        assert_eq!(classify("model.embed_tokens.weight").ok(), Some(Family::Int8Embed), "dense-qwen stem too");
        assert_eq!(classify("model.visual.blocks.3.attn.qkv.weight").ok(), Some(Family::SkipVisual));

        // The property the whole tool leans on: unknown names are hard errors.
        let msg = match classify("model.language_model.layers.7.linear_attn.in_proj_ba.weight") {
            Err(e) => e.to_string(),
            Ok(f) => format!("wrongly accepted as {f:?}"),
        };
        assert!(msg.contains("REV 2"), "an unknown family must refuse, got: {msg}");
        let msg = match classify("model.language_model.layers.7.mlp.down_proj.weight.qs") {
            Err(e) => e.to_string(),
            Ok(f) => format!("wrongly accepted as {f:?}"),
        };
        assert!(msg.contains("already a peregrine container"), "got: {msg}");
    }

    #[test]
    fn the_hybrid_fixture_imports_loads_and_generates() -> Result<(), Error> {
        // The full C2 acceptance in miniature: bf16 HF fixture → import →
        // the *production loader* consumes the result and decodes. This is the
        // free cross-check against C1: their loader tests pin the container
        // conventions, this pins that the importer emits them.
        let (dir, out) = fixture_dirs("hybrid")?;
        build_hf_hybrid_fixture(&dir, false)?;
        let rep = import_hf(&dir, &out, 1 << 30)?;
        assert_eq!(rep.skipped_visual, 1, "the vision tensor is skipped");
        // int4: full layer q/k/v/o (4) + 2 linear layers × 5 in/out projections
        // (10) + 3 layers × 3 mlp (9) + lm_head (1) = 24.
        assert_eq!(rep.imported_int4, 24);
        assert_eq!(rep.imported_int8, 1, "embed only");
        assert!(rep.imported_float > 0);
        assert!(rep.sidecars.iter().any(|s| s == "config.json"));

        let st = SafeTensors::open(&out)?;
        assert!(st.has("model.language_model.layers.0.linear_attn.in_proj_qkv.weight.qs"), "quantized matrices carry scales");
        assert!(!st.has("model.visual.patch_embed.proj.weight"), "vision stays out");
        // conv1d keeps its 3-D shape and its exact (bf16-representable) values.
        let conv = match st
            .tensors()
            .iter()
            .find(|t| t.name == "model.language_model.layers.0.linear_attn.conv1d.weight")
        {
            Some(t) => t.shape.clone(),
            None => Vec::new(), // the assert below then reports the absence
        };
        assert_eq!(conv, vec![32, 1, 4], "conv_dim 32 = 2*2*4 + 4*4, depthwise, 4 taps");

        // The decisive check: C1's production loader loads the imported
        // container and decodes tokens.
        let mut m = peregrine_model::Model::load(&out)?;
        let mut greedy = peregrine_model::Sampler::new(0.0, 0.9, 1);
        let toks = m.generate(&[1, 5, 9], 4, &mut greedy)?;
        assert!(!toks.is_empty(), "the imported container must decode");
        assert!(toks.iter().all(|&t| (0..32).contains(&t)), "tokens in vocab: {toks:?}");
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn an_unknown_tensor_fails_the_whole_import() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("refuse")?;
        build_hf_hybrid_fixture(&dir, true)?;
        let verdict = import_hf(&dir, &out, 1 << 30);
        let msg = match verdict {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(msg.contains("mystery"), "the refusal must name the tensor: {msg}");
        std::fs::remove_dir_all(&dir)?;
        if out.exists() {
            std::fs::remove_dir_all(&out)?;
        }
        Ok(())
    }

    /// Encode one f32 that is exactly representable in e4m3 (test fixtures
    /// only generate such values, so the brute-force search is total; a
    /// non-representable input is a fixture-generator bug, surfaced as Err).
    fn f8_exact(v: f32) -> Result<u8, Error> {
        (0u8..=255)
            .find(|&b| {
                let d = peregrine_core::dtype::f8e4m3_to_f32(b);
                d == v && d.is_sign_positive() == v.is_sign_positive()
            })
            .ok_or_else(|| Error::Format(format!("{v} is not e4m3-representable — fix the fixture generator")))
    }

    /// e4m3-exact values: multiples of 0.25 in [-1.75, 1.75].
    fn f8_exact_vals(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|k| {
                let x = ((k as u32).wrapping_mul(2654435761).wrapping_add(seed) % 15) as f32;
                (x - 7.0) * 0.25
            })
            .collect()
    }

    /// A synthetic GLM-5.3-Flash-style HF checkpoint at the canonical tiny
    /// dims (`peregrine_model::testkit::tiny_glm53_cfg_json`): the
    /// `model.language_model.` stem, bf16 attention/hc tensors, **FP8 experts
    /// with `weight_scale_inv` block scales**, the HF indexer names, a vision
    /// tensor to skip, and the GLM-dialect MTP head layer.
    fn build_hf_glm53_fixture(dir: &std::path::Path) -> Result<(), Error> {
        let cfg = peregrine_model::testkit::tiny_glm53_cfg_json();
        let (d, vocab) = (16usize, 32usize);
        let (ql, kvl) = (12usize, 8usize);
        let (h, qkh, vh) = (2usize, 4usize, 4usize);
        let (lh, ld, taps) = (2usize, 4usize, 4usize);
        let lqkv = lh * ld;
        let (inh, ihd, kp) = (2usize, 4usize, 4usize);
        let (e_n, mi, di) = (4usize, 8usize, 8usize);
        let hc = 4usize;
        let mix = (2 + hc) * hc;
        let stem = "model.language_model";

        let mut blobs = Vec::new();
        let seed = std::cell::Cell::new(1u32);
        let bf = |blobs: &mut Vec<Blob>, name: String, shape: Vec<i64>| {
            seed.set(seed.get() + 1);
            blobs.push(bf16_blob(&name, shape, seed.get()));
        };
        let f32b = |blobs: &mut Vec<Blob>, name: String, vals: &[f32], shape: Vec<i64>| {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            blobs.push(Blob::new(name, "F32", shape, bytes));
        };
        // FP8 matrix + block scales: 2×2 scale grid over [o, i], scale values
        // exact powers of two so dequant is exact in f32.
        let fp8 = |blobs: &mut Vec<Blob>, name: String, o: usize, i: usize| -> Result<(), Error> {
            seed.set(seed.get() + 1);
            let vals = f8_exact_vals(o * i, seed.get());
            let payload: Vec<u8> = vals.iter().map(|&v| f8_exact(v)).collect::<Result<_, _>>()?;
            blobs.push(Blob::new(name.clone(), "F8_E4M3", vec![o as i64, i as i64], payload));
            let scales = [0.5f32, 2.0, 1.0, 4.0];
            let bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            blobs.push(Blob::new(format!("{name}_scale_inv"), "F32", vec![2, 2], bytes));
            Ok(())
        };

        bf(&mut blobs, format!("{stem}.embed_tokens.weight"), vec![vocab as i64, d as i64]);
        bf(&mut blobs, "lm_head.weight".into(), vec![vocab as i64, d as i64]);
        bf(&mut blobs, format!("{stem}.norm.weight"), vec![d as i64]);
        bf(&mut blobs, "model.visual.patch_embed.proj.weight".into(), vec![4, 4]);
        // 4 main layers [kda, kda, dsa, kda] + the MTP head layer (4, DSA).
        for l in 0..5usize {
            let p = |s: &str| format!("{stem}.layers.{l}.{s}");
            let full = l == 2 || l == 4;
            bf(&mut blobs, p("input_layernorm.weight"), vec![d as i64]);
            bf(&mut blobs, p("post_attention_layernorm.weight"), vec![d as i64]);
            if full {
                bf(&mut blobs, p("self_attn.q_a_proj.weight"), vec![ql as i64, d as i64]);
                bf(&mut blobs, p("self_attn.q_a_layernorm.weight"), vec![ql as i64]);
                bf(&mut blobs, p("self_attn.q_b_proj.weight"), vec![(h * qkh) as i64, ql as i64]);
                bf(&mut blobs, p("self_attn.kv_a_proj_with_mqa.weight"), vec![kvl as i64, d as i64]);
                bf(&mut blobs, p("self_attn.kv_a_layernorm.weight"), vec![kvl as i64]);
                bf(&mut blobs, p("self_attn.kv_b_proj.weight"), vec![(h * (qkh + vh)) as i64, kvl as i64]);
                bf(&mut blobs, p("self_attn.o_proj.weight"), vec![d as i64, (h * vh) as i64]);
                bf(&mut blobs, p("self_attn.indexer.wq_b.weight"), vec![(inh * ihd) as i64, ql as i64]);
                bf(&mut blobs, p("self_attn.indexer.wk.weight"), vec![ihd as i64, d as i64]);
                bf(&mut blobs, p("self_attn.indexer.weights_proj.weight"), vec![inh as i64, d as i64]);
                bf(&mut blobs, p("self_attn.indexer.k_norm.weight"), vec![ihd as i64]);
                bf(&mut blobs, p("self_attn.indexer.k_norm.bias"), vec![ihd as i64]);
                bf(&mut blobs, p("self_attn.indexer.index_kpool_compress_gate"), vec![ihd as i64, d as i64]);
                bf(&mut blobs, p("self_attn.indexer.index_kpool_compress_ape"), vec![kp as i64, ihd as i64]);
            } else {
                bf(&mut blobs, p("self_attn.q_proj.weight"), vec![lqkv as i64, d as i64]);
                bf(&mut blobs, p("self_attn.k_proj.weight"), vec![lqkv as i64, d as i64]);
                bf(&mut blobs, p("self_attn.v_proj.weight"), vec![lqkv as i64, d as i64]);
                for t in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    bf(&mut blobs, p(&format!("self_attn.{t}.weight")), vec![lqkv as i64, 1, taps as i64]);
                }
                bf(&mut blobs, p("self_attn.f_a_proj.weight"), vec![ld as i64, d as i64]);
                bf(&mut blobs, p("self_attn.f_b_proj.weight"), vec![lqkv as i64, ld as i64]);
                f32b(&mut blobs, p("self_attn.dt_bias"), &f8_exact_vals(lqkv, 77), vec![lqkv as i64]);
                f32b(&mut blobs, p("self_attn.A_log"), &f8_exact_vals(lh, 78), vec![lh as i64]);
                bf(&mut blobs, p("self_attn.b_proj.weight"), vec![lh as i64, d as i64]);
                bf(&mut blobs, p("self_attn.g_a_proj.weight"), vec![ld as i64, d as i64]);
                bf(&mut blobs, p("self_attn.g_b_proj.weight"), vec![lqkv as i64, ld as i64]);
                bf(&mut blobs, p("self_attn.o_norm.weight"), vec![ld as i64]);
                bf(&mut blobs, p("self_attn.o_proj.weight"), vec![d as i64, lqkv as i64]);
            }
            if l < 4 {
                for site in ["attn", "ffn"] {
                    f32b(&mut blobs, p(&format!("hc_{site}_base")), &f8_exact_vals(mix, 80), vec![mix as i64]);
                    bf(&mut blobs, p(&format!("hc_{site}_fn")), vec![mix as i64, (hc * d) as i64]);
                    f32b(&mut blobs, p(&format!("hc_{site}_scale")), &[1.0, 1.0, 1.0], vec![3]);
                }
            }
            if l == 0 {
                bf(&mut blobs, p("mlp.gate_proj.weight"), vec![di as i64, d as i64]);
                bf(&mut blobs, p("mlp.up_proj.weight"), vec![di as i64, d as i64]);
                bf(&mut blobs, p("mlp.down_proj.weight"), vec![d as i64, di as i64]);
            } else {
                bf(&mut blobs, p("mlp.gate.weight"), vec![e_n as i64, d as i64]);
                f32b(&mut blobs, p("mlp.gate.e_score_correction_bias"), &f8_exact_vals(e_n, 81), vec![e_n as i64]);
                bf(&mut blobs, p("mlp.shared_experts.gate_proj.weight"), vec![mi as i64, d as i64]);
                bf(&mut blobs, p("mlp.shared_experts.up_proj.weight"), vec![mi as i64, d as i64]);
                bf(&mut blobs, p("mlp.shared_experts.down_proj.weight"), vec![d as i64, mi as i64]);
                for e in 0..e_n {
                    let pe = |s: &str| format!("{stem}.layers.{l}.mlp.experts.{e}.{s}");
                    fp8(&mut blobs, pe("gate_proj.weight"), mi, d)?;
                    fp8(&mut blobs, pe("up_proj.weight"), mi, d)?;
                    fp8(&mut blobs, pe("down_proj.weight"), d, mi)?;
                }
            }
            if l == 4 {
                bf(&mut blobs, p("eh_proj.weight"), vec![d as i64, (2 * d) as i64]);
                bf(&mut blobs, p("enorm.weight"), vec![d as i64]);
                bf(&mut blobs, p("hnorm.weight"), vec![d as i64]);
                bf(&mut blobs, p("shared_head.norm.weight"), vec![d as i64]);
            }
        }
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).map_err(|e| Error::Format(e.to_string()))?)?;
        write_safetensors(dir, &blobs)?;
        Ok(())
    }

    #[test]
    fn glm5next_rename_moves_the_stem_and_the_indexer_projections() {
        assert_eq!(
            glm5next_rename("model.language_model.layers.7.mlp.experts.3.up_proj.weight").as_deref(),
            Some("model.layers.7.mlp.experts.3.up_proj.weight")
        );
        assert_eq!(
            glm5next_rename("model.language_model.layers.3.self_attn.indexer.wk.weight").as_deref(),
            Some("model.layers.3.self_attn.indexer_projections.wk")
        );
        assert_eq!(
            glm5next_rename("model.language_model.layers.3.self_attn.indexer.k_norm.weight").as_deref(),
            Some("model.layers.3.self_attn.indexer.k_norm.weight"),
            "k_norm stays under indexer. — only the three projections move"
        );
        assert_eq!(glm5next_rename("model.visual.blocks.0.attn.qkv.weight"), None);
        assert_eq!(glm5next_rename("lm_head.weight").as_deref(), Some("lm_head.weight"));
    }

    #[test]
    fn fp8_block_scales_dequantize_exactly() -> Result<(), Error> {
        // A 4×4 FP8 tensor with a 2×2 scale grid: every element must come back
        // as value × its block's scale, exactly (both sides are powers of two).
        let (dir, _out) = fixture_dirs("fp8deq")?;
        let vals = f8_exact_vals(16, 5);
        let payload: Vec<u8> = vals.iter().map(|&v| f8_exact(v)).collect::<Result<_, _>>()?;
        let scales = [0.5f32, 2.0, 4.0, 0.25];
        std::fs::create_dir_all(&dir)?;
        write_safetensors(
            &dir,
            &[
                Blob::new("w", "F8_E4M3", vec![4, 4], payload),
                Blob::new(
                    "w_scale_inv",
                    "F32",
                    vec![2, 2],
                    scales.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                ),
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        let got = read_dense_dequant(&st, "w", &[4, 4])?;
        for r in 0..4 {
            for c in 0..4 {
                let want = vals[r * 4 + c] * scales[(r / 2) * 2 + (c / 2)];
                assert_eq!(got[r * 4 + c], want, "({r},{c})");
            }
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn the_glm53_fixture_imports_renames_dequantizes_and_generates() -> Result<(), Error> {
        // The GLM-5.3 acceptance in miniature: FP8 HF fixture → import (stem
        // rename + block dequant + per-family precision) → the production
        // loader consumes the result and decodes through the whole hybrid
        // KDA/DSA/mHC stack.
        let (dir, out) = fixture_dirs("glm53")?;
        build_hf_glm53_fixture(&dir)?;
        let rep = import_hf(&dir, &out, 1 << 30)?;
        assert_eq!(rep.skipped_visual, 1, "the vision tensor is skipped");
        assert!(rep.imported_int4 > 0 && rep.imported_int8 > 0 && rep.imported_float > 0);

        let st = SafeTensors::open(&out)?;
        assert!(st.has("model.layers.1.mlp.experts.0.gate_proj.weight"), "stem renamed to model.");
        assert!(st.has("model.layers.1.mlp.experts.0.gate_proj.weight.qs"), "experts carry int4 scales");
        assert!(st.has("model.layers.2.self_attn.indexer_projections.wk"), "indexer projections renamed");
        assert!(st.has("model.layers.0.hc_attn_fn.qs"), "hc fn is int8+qs");
        assert!(st.has("model.layers.4.eh_proj.weight"), "the MTP head layer travels");
        assert!(!st.has("model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv"), "scales are folded, not copied");
        assert!(!st.has("model.language_model.embed_tokens.weight"), "no language_model stem survives");

        let mut m = peregrine_model::Model::load(&out)?;
        let mut greedy = peregrine_model::Sampler::new(0.0, 0.9, 1);
        let toks = m.generate(&[1, 5, 9], 4, &mut greedy)?;
        assert_eq!(toks.len(), 4, "the imported container must decode");
        assert!(toks.iter().all(|&t| (0..32).contains(&t)), "tokens in vocab: {toks:?}");
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }
}

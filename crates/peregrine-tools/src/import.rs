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

use peregrine_core::pack::{quant_i4, quant_i8};
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
    let st = SafeTensors::open(indir)?;
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
    std::fs::create_dir_all(outdir).ctx(|| format!("create {}", outdir.display()))?;
    let contract = if glm5next { "glm5-next" } else { "track-c-rev2" };
    let mut w = ShardWriter::new(outdir, "model", shard_bytes).with_metadata(vec![
        ("peregrine.import.tool".into(), "peregrine-import-hf".into()),
        ("peregrine.import.contract".into(), contract.into()),
        ("peregrine.import.source".into(), indir.display().to_string()),
    ]);
    let mut rep = ImportReport::default();

    let meta: Vec<(String, Vec<i64>)> =
        st.tensors().iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    for (name, shape) in &meta {
        rep.tensors_total += 1;
        rep.bytes_in += st.uncompressed_nbytes(name).unwrap_or(0).max(0) as u64;
        // Resolve the output name and family per contract.
        let (family, out_name) = if glm5next {
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

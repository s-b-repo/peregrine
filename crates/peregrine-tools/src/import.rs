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
    /// Token embedding → int8 + per-row `.qs`.
    Int8Embed,
    /// Small/odd tensors → F32 passthrough at their original shape.
    Float,
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
    std::fs::create_dir_all(outdir).ctx(|| format!("create {}", outdir.display()))?;
    let mut w = ShardWriter::new(outdir, "model", shard_bytes).with_metadata(vec![
        ("peregrine.import.tool".into(), "peregrine-import-hf".into()),
        ("peregrine.import.contract".into(), "track-c-rev2".into()),
        ("peregrine.import.source".into(), indir.display().to_string()),
    ]);
    let mut rep = ImportReport::default();

    let meta: Vec<(String, Vec<i64>)> =
        st.tensors().iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    for (name, shape) in &meta {
        rep.tensors_total += 1;
        let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
        rep.bytes_in += st.uncompressed_nbytes(name).unwrap_or(0).max(0) as u64;
        let family = classify(name)?;
        if family == Family::SkipVisual {
            rep.skipped_visual += 1;
            continue;
        }
        // One dense read covers every family: `read_f32` widens bf16 exactly.
        let mut dense = vec![0f32; numel];
        st.read_f32(name, &mut dense)?;
        match family {
            Family::Int4 | Family::Int8Embed => {
                let (&[o, i], true) = (shape.as_slice(), shape.len() == 2) else {
                    return Err(Error::Format(format!(
                        "'{name}': the contract quantizes this family, but its shape {shape:?} \
                         is not a 2-D matrix — the checkpoint does not match REV 2"
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
                w.push(name, "U8", vec![o as i64, out_cols], q)?;
                let s_bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.push(&format!("{name}.qs"), "F32", vec![s.len() as i64], s_bytes)?;
            }
            Family::Float => {
                // Original shape preserved — conv1d stays 3-D.
                rep.imported_float += 1;
                rep.bytes_out += (dense.len() * 4) as u64;
                let bytes: Vec<u8> = dense.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.push(name, "F32", shape.clone(), bytes)?;
            }
            Family::SkipVisual => {}
        }
    }
    w.flush()?;

    // Same rule as requantize: shards alone are not a model. The HF config is
    // copied VERBATIM — `Cfg`'s Arch detection is written against it.
    for side in ["config.json", "tokenizer.json", "generation_config.json", "tokenizer_config.json"] {
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
        assert!(toks.iter().all(|&t| t >= 0 && t < 32), "tokens in vocab: {toks:?}");
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
}

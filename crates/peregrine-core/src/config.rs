//! Model configuration — the Rust equivalent of the C `Cfg` struct and
//! `load_cfg` (`c/glm.c:1258-1308`).
//!
//! Field names and defaults are ported exactly, including the derived
//! `qk_head`/`attn_scale`, the DSA `idx_type` per-layer schedule, and the
//! `CKR` bounds validation from PR #25 (hostile config.json must not pass).

use crate::{Context, Error};
use serde_json::Value;
use std::path::Path;

/// Which transformer architecture a checkpoint declares. The engine was built
/// for GLM-5.2's MLA + routed experts; `DenseGqa` (added 2026-08-15, Track C,
/// for Qwen3.8) is a plain dense stack with grouped-query attention — no
/// latents, no routers, every layer's MLP on the dense path the engine already
/// computes for GLM's `first_k_dense_replace` layers. Detection is by
/// `model_type`, never by guessing from field presence alone, so an unknown
/// checkpoint fails loudly instead of loading as the wrong math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.2 shape: MLA attention (kv_lora latents), MoE with routed experts.
    GlmMla,
    /// Qwen3-family dense shape: GQA attention with per-head q/k RMS norm and
    /// full-head-dim rotate-half RoPE, SwiGLU MLP every layer.
    DenseGqa,
    /// Qwen3.5/3.8 hybrid: `layer_types` interleaves output-gated GQA
    /// full-attention layers (partial rotate-half RoPE) with gated-DeltaNet
    /// linear-attention layers carrying a per-stream recurrent state instead
    /// of KV. Dense SwiGLU MLP every layer, like [`Arch::DenseGqa`].
    HybridGdn,
    /// GLM-5.3-Flash (`model_type: glm5_next[_text]`): a hybrid of KDA
    /// linear-attention layers (Kimi Delta Attention — per-*channel* forget
    /// gates, where [`Arch::HybridGdn`]'s GDN decays per head) and NoPE MLA
    /// layers (`qk_rope_head_dim = 0`) carrying a k-pool-compressed DSA
    /// indexer. Routed-expert MoE like [`Arch::GlmMla`] (sigmoid router +
    /// correction bias), but with the SwiGLU pre-activations clamped at
    /// `swiglu_limit`, and the whole stack running on `hc_mult` mHC
    /// hyper-connection residual streams (Sinkhorn-normalized mixing at every
    /// attention/FFN site) instead of a single residual stream.
    Glm5Next,
    Qwen4Exp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4OutputGate {
    Sigmoid,
    Silu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4QsaCfg {
    pub n_heads: i64,
    pub head_dim: i64,
    pub budget: i64,
    pub compress_ratio: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4Cfg {
    pub hc_count: i64,
    pub hc_lowrank: i64,
    pub output_gate: Qwen4OutputGate,
    pub qsa: Option<Qwen4QsaCfg>,
    pub ple_layer_ids: Vec<i64>,
    pub ple_embed_dim: i64,
    pub ple_conv_kernel_size: i64,
    pub ngram_size: i64,
    pub heads_per_ngram: i64,
    pub ngram_vocab_size_base: i64,
    pub ngram_vocab_divisor: i64,
    pub seed: i64,
    pub split_ngram_parts: i64,
}

/// Parsed `config.json`. Mirrors `Cfg` in `c/glm.c`.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub arch: Arch,
    pub qwen4: Option<Qwen4Cfg>,
    /// GQA: number of key/value heads (`num_key_value_heads`); equals `n_heads`
    /// under MLA (where every head shares the latent anyway — unused there).
    pub n_kv_heads: i64,
    /// GQA: per-head dimension (`head_dim`, defaulting to hidden/n_heads).
    /// 0 under MLA, whose head geometry is qk_nope/qk_rope/v_head.
    pub head_dim: i64,
    /// HybridGdn: which layers run full attention (`layer_types`); `true` =
    /// full attention, `false` = gated-DeltaNet linear attention. Empty for
    /// the other architectures (every layer is whatever the arch says).
    pub full_attn: Vec<bool>,
    /// HybridGdn: whether q_proj emits `[n_heads*head_dim*2]` — query in the
    /// first flat half, a sigmoid output gate in the second
    /// (`attn_output_gate`). The gate multiplies the attention output before
    /// o_proj.
    pub attn_gate: bool,
    /// Gated-DeltaNet geometry (`linear_*` in config.json); zero elsewhere.
    /// Glm5Next reuses these for its KDA layers (`linear_attn_config`), where
    /// k-heads == v-heads and k-dim == v-dim by construction.
    pub lin_k_heads: i64,
    pub lin_v_heads: i64,
    pub lin_k_dim: i64,
    pub lin_v_dim: i64,
    pub lin_conv_k: i64,
    /// Glm5Next KDA forget-gate lower bound (`linear_attn_config.gate_lower_bound`).
    /// `Some(b)`: the log-decay is `b · sigmoid(exp(A_log) · g)` (bounded in
    /// `(b, 0)`); `None`: the unbounded `-exp(A_log) · softplus(g)` form.
    pub lin_gate_lb: Option<f32>,
    pub hidden: i64,
    pub n_layers: i64,
    pub n_heads: i64,
    pub n_experts: i64,
    pub topk: i64,
    pub moe_inter: i64,
    pub dense_inter: i64,
    pub first_dense: i64,
    pub q_lora: i64,
    pub kv_lora: i64,
    pub qk_nope: i64,
    pub qk_rope: i64,
    pub v_head: i64,
    pub n_shared: i64,
    pub vocab: i64,
    pub n_group: i64,
    pub topk_group: i64,
    pub norm_topk: bool,
    pub eps: f32,
    pub routed_scale: f32,
    pub theta: f32,
    /// eos_token_id(s) — GLM-5.2 has three (endoftext, user, observation).
    pub stop_ids: Vec<i32>,
    // DSA lightning indexer
    pub index_topk: i64,
    pub index_nh: i64,
    pub index_hd: i64,
    /// per-layer indexer type: `true` = full indexer layer, `false` = shared.
    pub idx_type: Vec<bool>,
    /// Glm5Next: the indexer scores *pools* of this many consecutive tokens
    /// instead of individual tokens (`index_kpool`); 0 = per-token indexer.
    pub index_kpool: i64,
    /// Glm5Next: always append the incomplete tail pool's tokens to the
    /// selection (`index_kpool_always_select_tail`).
    pub index_kpool_tail: bool,
    /// Glm5Next: SwiGLU pre-activation clamp (`swiglu_limit`) — gate clamped to
    /// `<= limit`, up to `[-limit, limit]`, in every dense/shared/routed MLP.
    /// 0.0 = no clamp (every other architecture).
    pub swiglu_limit: f32,
    /// Glm5Next mHC hyper-connections: number of parallel residual streams
    /// (`hc_mult`). 0 or 1 = a single conventional residual stream.
    pub hc_mult: i64,
    /// mHC Sinkhorn numerical floor (`hc_eps`).
    pub hc_eps: f32,
    /// mHC Sinkhorn iteration count (`hc_sinkhorn_iters`).
    pub hc_sinkhorn: i64,
    // derived
    pub qk_head: i64,
    pub attn_scale: f32,
}

/// `gi()` in the C engine: read an integer field, default 0 if absent.
/// JSON numbers may be float-encoded, so fall back to `as_f64`.
fn gi(root: &Value, key: &str) -> i64 {
    match root.get(key) {
        Some(v) => v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or(0),
        None => 0,
    }
}

fn gf(root: &Value, key: &str, default: f32) -> f32 {
    root.get(key).and_then(|v| v.as_f64()).map(|f| f as f32).unwrap_or(default)
}

impl Cfg {
    /// Load and validate `<dir>/config.json`, folding in
    /// `<dir>/generation_config.json`'s stop tokens when the checkpoint ships
    /// one. HF splits EOS across the two files — Qwen3.8 declares 248044 in
    /// config.json but the ChatML turn terminator <|im_end|> (248046) only in
    /// generation_config.json, so an engine reading one file serves answers
    /// with a trailing <|im_end|> it never stops on. Union, never replace:
    /// every id either file declares is kept, same rule as the array parse.
    pub fn load(dir: &Path) -> Result<Cfg, Error> {
        let path = dir.join("config.json");
        // read through the io_uring lane (no std::fs read path)
        let bytes = peregrine_io::read_file(&path).ctx(|| path.display().to_string())?;
        let root: Value = serde_json::from_slice(&bytes)?;
        let mut cfg = Cfg::from_json(&root)?;
        // Absent is the normal case (GLM containers ship none), so only an
        // existing-but-unreadable file is worth saying anything about — and it
        // is an advisory, not a fatal: the model still runs, it just stops on
        // config.json's ids alone.
        let gen_path = dir.join("generation_config.json");
        let gen_bytes = if gen_path.exists() {
            match peregrine_io::read_file(&gen_path) {
                Ok(b) => Some(b),
                Err(e) => {
                    peregrine_io::note_advisory_err("generation_config.json read", &e);
                    None
                }
            }
        } else {
            None
        };
        if let Some(bytes) = gen_bytes {
            match serde_json::from_slice::<Value>(&bytes) {
                Ok(g) => {
                    let extra: Vec<i64> = match g.get("eos_token_id") {
                        Some(Value::Number(n)) => n.as_i64().into_iter().collect(),
                        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_i64()).collect(),
                        _ => Vec::new(),
                    };
                    for id in extra {
                        let id = id as i32;
                        if !cfg.stop_ids.contains(&id) {
                            cfg.stop_ids.push(id);
                        }
                    }
                }
                Err(e) => peregrine_io::note_advisory_err("generation_config.json parse", &e),
            }
        }
        Ok(cfg)
    }

    /// Parse a config from an already-decoded JSON value (used by tests).
    pub fn from_json(root: &Value) -> Result<Cfg, Error> {
        // Architecture dispatch, on the checkpoint's own declaration. Absent
        // `model_type` keeps the historical GLM path (laptop-converted GLM
        // containers predate this field being read), but a *declared* type this
        // engine does not implement is a loud error, not a GLM-shaped guess.
        match root.get("model_type").and_then(|v| v.as_str()) {
            Some("qwen4_exp" | "qwen4_exp_text") => {
                return Cfg::from_json_qwen4(root.get("text_config").unwrap_or(root));
            }
            // Hybrid families first: "qwen3_5"/"qwen3_next" (and the VL wrapper,
            // whose text stack lives under `text_config`) — checked before the
            // "qwen3" prefix would swallow them into the pure-dense path.
            Some(t) if t.starts_with("qwen3_5") || t.starts_with("qwen3_next") => {
                return Cfg::from_json_hybrid(root.get("text_config").unwrap_or(root))
            }
            Some(t) if t.starts_with("qwen3") => return Cfg::from_json_gqa(root),
            // GLM-5.3-Flash — checked before the "glm" prefix would swallow it
            // into the MLA-only path. The multimodal checkpoint nests the text
            // stack under `text_config` (model_type "glm5_next"); a text-only
            // export is flat (model_type "glm5_next_text").
            Some(t) if t.starts_with("glm5_next") => {
                return Cfg::from_json_glm5next(root.get("text_config").unwrap_or(root))
            }
            Some(t) if t.starts_with("glm") || t.starts_with("deepseek") => {}
            None => {}
            Some(other) => {
                return Err(Error::Format(format!(
                    "config: model_type \"{other}\" is not supported (glm/deepseek MLA-MoE, glm5_next KDA+MLA-MoE, qwen3 dense-GQA, or qwen3_5 hybrid)"
                )))
            }
        }
        let n_layers = gi(root, "num_hidden_layers");

        // stop tokens: eos_token_id is a scalar or an array. Every listed id is
        // kept — truncating the list would let generation run past a stop token
        // the checkpoint declares.
        let mut stop_ids = Vec::new();
        match root.get("eos_token_id") {
            Some(Value::Number(n)) => match n.as_i64() {
                Some(id) => stop_ids.push(id as i32),
                None => return Err(Error::Format(format!("config: eos_token_id={n} is not an integer"))),
            },
            Some(Value::Array(a)) => {
                for v in a.iter() {
                    match v.as_i64() {
                        Some(id) => stop_ids.push(id as i32),
                        None => return Err(Error::Format(format!("config: eos_token_id entry {v} is not an integer"))),
                    }
                }
            }
            _ => {}
        }

        // DSA indexer per-layer schedule: explicit list or freq/offset formula
        let index_topk = gi(root, "index_topk");
        let mut idx_type = vec![false; n_layers.max(0) as usize];
        {
            let types = root.get("indexer_types").and_then(|v| v.as_array());
            let mut freq = gi(root, "index_topk_freq");
            if freq < 1 {
                freq = 1;
            }
            let off = root
                .get("index_skip_topk_offset")
                .and_then(|v| v.as_i64())
                .unwrap_or(2);
            for (i, slot) in idx_type.iter_mut().enumerate() {
                *slot = match types.and_then(|t| t.get(i)).and_then(|v| v.as_str()) {
                    Some(s) => s == "full",
                    None => {
                        let v = (i as i64) - off + 1;
                        let v = if v < 0 { 0 } else { v };
                        v % freq == 0
                    }
                };
            }
        }

        let qk_nope = gi(root, "qk_nope_head_dim");
        let qk_rope = gi(root, "qk_rope_head_dim");
        // `rope_theta` lives under `rope_parameters` in newer transformers
        // exports and at the top level in the long-standing HF layout. Read both
        // (nested wins): a checkpoint using the top-level spelling would
        // otherwise silently fall back to 10000.0 and scramble every position.
        let theta_json = root
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .or_else(|| root.get("rope_theta"));
        let theta = match theta_json {
            Some(v) => match v.as_f64() {
                Some(f) if f.is_finite() && f > 0.0 => f as f32,
                _ => return Err(Error::Format(format!("config: rope_theta={v} is not a positive number"))),
            },
            // Absent theta with RoPE lanes in play means the parse missed the
            // field rather than the model being RoPE-free — refuse to guess.
            None if qk_rope > 0 => {
                return Err(Error::Format(
                    "config: rope_theta not found (looked in rope_parameters.rope_theta and top-level rope_theta)".into(),
                ))
            }
            None => 10000.0,
        };

        let mut c = Cfg {
            qwen4: None,
            arch: Arch::GlmMla,
            n_kv_heads: gi(root, "num_attention_heads"),
            head_dim: 0,
            full_attn: Vec::new(),
            attn_gate: false,
            lin_k_heads: 0,
            lin_v_heads: 0,
            lin_k_dim: 0,
            lin_v_dim: 0,
            lin_conv_k: 0,
            lin_gate_lb: None,
            hidden: gi(root, "hidden_size"),
            n_layers,
            n_heads: gi(root, "num_attention_heads"),
            n_experts: gi(root, "n_routed_experts"),
            topk: gi(root, "num_experts_per_tok"),
            moe_inter: gi(root, "moe_intermediate_size"),
            dense_inter: gi(root, "intermediate_size"),
            first_dense: gi(root, "first_k_dense_replace"),
            q_lora: gi(root, "q_lora_rank"),
            kv_lora: gi(root, "kv_lora_rank"),
            qk_nope,
            qk_rope,
            v_head: gi(root, "v_head_dim"),
            n_shared: gi(root, "n_shared_experts"),
            vocab: gi(root, "vocab_size"),
            n_group: gi(root, "n_group"),
            topk_group: gi(root, "topk_group"),
            norm_topk: root.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(false),
            eps: gf(root, "rms_norm_eps", 1e-5),
            routed_scale: gf(root, "routed_scaling_factor", 1.0),
            theta,
            stop_ids,
            index_topk,
            index_nh: gi(root, "index_n_heads"),
            index_hd: gi(root, "index_head_dim"),
            idx_type,
            index_kpool: 0,
            index_kpool_tail: false,
            swiglu_limit: 0.0,
            hc_mult: 0,
            hc_eps: 0.0,
            hc_sinkhorn: 0,
            qk_head: qk_nope + qk_rope,
            attn_scale: 0.0,
        };
        c.attn_scale = 1.0 / (c.qk_head as f32).sqrt();

        if c.n_group != 1 {
            return Err(Error::Format("this engine requires n_group=1 (GLM-5.2)".into()));
        }
        c.validate()?;
        Ok(c)
    }

    fn from_json_qwen4(root: &Value) -> Result<Cfg, Error> {
        if !root.is_object() {
            return Err(Error::Format("config: qwen4_exp text_config must be an object".into()));
        }
        let integer = |key: &str, default: i64, lo: i64, hi: i64| -> Result<i64, Error> {
            let value = match root.get(key) {
                None => default,
                Some(v) => v.as_i64().ok_or_else(|| Error::Format(format!("config: {key} must be an integer")))?,
            };
            if !(lo..=hi).contains(&value) {
                return Err(Error::Format(format!("config: {key}={value} is outside [{lo},{hi}]")));
            }
            Ok(value)
        };
        let boolean = |key: &str, default: bool| -> Result<bool, Error> {
            match root.get(key) {
                None => Ok(default),
                Some(v) => v.as_bool().ok_or_else(|| Error::Format(format!("config: {key} must be boolean"))),
            }
        };
        let hidden = integer("hidden_size", 2048, 1, 1 << 20)?;
        let n_layers = integer("num_hidden_layers", 40, 1, 128)?;
        let n_heads = integer("num_attention_heads", 16, 1, 1024)?;
        let n_kv_heads = integer("num_key_value_heads", 2, 1, n_heads)?;
        let head_dim = integer("head_dim", 256, 2, 1 << 16)?;
        let n_experts = integer("num_experts", 512, 1, 4096)?;
        let topk = integer("num_experts_per_tok", 10, 1, n_experts.min(64))?;
        let moe_inter = integer("moe_intermediate_size", 512, 1, 1 << 20)?;
        let shared_inter = integer("shared_expert_intermediate_size", 512, 1, 1 << 20)?;
        let vocab = integer("vocab_size", 248320, 1, 1 << 24)?;
        let hc_count = integer("hc_count", 4, 2, 16)?;
        let hc_lowrank = integer("hc_lowrank", 320, 1, 1 << 20)?;
        let interval = integer("full_attention_interval", 4, 1, 128)?;
        let full_attn = match root.get("layer_types") {
            None | Some(Value::Null) => (0..n_layers).map(|i| (i + 1) % interval == 0).collect(),
            Some(Value::Array(types)) if types.len() == n_layers as usize => types.iter().enumerate().map(|(i, t)| {
                match t.as_str() {
                    Some("linear_attention") => Ok(false),
                    Some("full_attention" | "qwen_sparse_attention") => Ok(true),
                    _ => Err(Error::Format(format!("config: unsupported qwen4_exp layer_types[{i}]={t}"))),
                }
            }).collect::<Result<Vec<_>, _>>()?,
            _ => return Err(Error::Format("config: layer_types must match num_hidden_layers".into())),
        };
        let hidden_act = root.get("hidden_act").and_then(Value::as_str).unwrap_or("silu");
        if hidden_act != "silu" || root.get("hidden_act").is_some_and(|v| !v.is_string()) {
            return Err(Error::Format("config: qwen4_exp requires hidden_act=silu".into()));
        }
        let output_gate = match root.get("output_gate_type") {
            None | Some(Value::Null) => Qwen4OutputGate::Silu,
            Some(Value::String(s)) if s == "sigmoid" => Qwen4OutputGate::Sigmoid,
            Some(Value::String(s)) if s == "silu" => Qwen4OutputGate::Silu,
            _ => return Err(Error::Format("config: output_gate_type must be sigmoid or silu".into())),
        };
        if boolean("attention_bias", false)? || boolean("tie_word_embeddings", false)? {
            return Err(Error::Format("config: qwen4_exp attention bias and tied embeddings are not implemented".into()));
        }
        let rp = root.get("rope_parameters");
        if rp.is_some_and(|v| !v.is_object() && !v.is_null()) {
            return Err(Error::Format("config: rope_parameters must be an object".into()));
        }
        if rp.and_then(|v| v.get("rope_type")).is_some_and(|v| v.as_str() != Some("default")) {
            return Err(Error::Format("config: qwen4_exp only default text RoPE is implemented".into()));
        }
        let positive = |value: Option<&Value>, default: f64, key: &str| -> Result<f32, Error> {
            let f = match value {
                None => default,
                Some(v) => v.as_f64().ok_or_else(|| Error::Format(format!("config: {key} must be numeric")))?,
            } as f32;
            if !f.is_finite() || f <= 0.0 {
                return Err(Error::Format(format!("config: {key} must be positive and finite")));
            }
            Ok(f)
        };
        let theta = positive(rp.and_then(|v| v.get("rope_theta")).or_else(|| root.get("rope_theta")), 10000.0, "rope_theta")?;
        let partial = positive(rp.and_then(|v| v.get("partial_rotary_factor")).or_else(|| root.get("partial_rotary_factor")), 1.0, "partial_rotary_factor")?;
        let rotary_dim = (head_dim as f64 * partial as f64) as i64;
        if partial > 1.0 || rotary_dim < 2 || rotary_dim % 2 != 0 {
            return Err(Error::Format("config: qwen4_exp rotary span must be even and in [2,head_dim]".into()));
        }
        let qsa_keys = ["indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"];
        let qsa = if qsa_keys.iter().any(|k| root.get(k).is_some_and(|v| !v.is_null())) {
            if qsa_keys.iter().any(|k| root.get(k).is_none_or(Value::is_null)) {
                return Err(Error::Format("config: QSA requires all five indexer fields".into()));
            }
            let qsa = Qwen4QsaCfg {
                n_heads: integer("indexer_n_heads", 0, 1, 1024)?,
                head_dim: integer("indexer_head_dim", 0, 2, 1 << 16)?,
                budget: integer("indexer_budget", 0, 1, 1 << 20)?,
                compress_ratio: integer("indexer_compress_ratio", 0, 1, 1024)?,
            };
            integer("indexer_kv_heads", 0, 1, 1)?;
            if qsa.budget % qsa.compress_ratio != 0 || rotary_dim > qsa.head_dim {
                return Err(Error::Format("config: QSA budget must divide into complete blocks and RoPE must fit its head".into()));
            }
            Some(qsa)
        } else {
            None
        };
        if full_attn.iter().any(|&v| v) && qsa.is_none() {
            return Err(Error::Format("config: qwen4_exp sparse attention requires QSA indexer fields".into()));
        }
        let mut stop_ids = Vec::new();
        let eos = match root.get("eos_token_id") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a.iter().collect(),
            Some(v) => vec![v],
        };
        for v in eos {
            let id = v.as_i64().filter(|&id| id >= 0 && id < vocab)
                .ok_or_else(|| Error::Format("config: eos_token_id must be an integer in the vocabulary".into()))?;
            stop_ids.push(id as i32);
        }
        let mut ple_layer_ids = match root.get("ple_layer_ids") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a.iter().map(|v| v.as_i64().filter(|&id| id >= 1 && id <= n_layers)
                .ok_or_else(|| Error::Format("config: ple_layer_ids must contain one-indexed decoder layer ids".into())))
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(Error::Format("config: ple_layer_ids must be an array".into())),
        };
        ple_layer_ids.sort_unstable();
        ple_layer_ids.dedup();
        let ple_embed_dim = if root.get("ple_embed_dim").is_some_and(Value::is_null) { hidden } else { integer("ple_embed_dim", hidden, 1, 1 << 20)? };
        let ngram_size = integer("ngram_size", 3, 2, 32)?;
        let heads_per_ngram = integer("heads_per_ngram", 8, 1, 256)?;
        if !ple_layer_ids.is_empty() && (stop_ids.is_empty()
            || ple_embed_dim % ((ngram_size - 1) * heads_per_ngram) != 0
            || ple_layer_ids.iter().any(|&id| full_attn[(id - 1) as usize])) {
            return Err(Error::Format("config: PLE requires EOS, divisible embedding heads, and linear-only layer ids".into()));
        }
        let qwen4 = Qwen4Cfg {
            hc_count, hc_lowrank, output_gate, qsa, ple_layer_ids, ple_embed_dim,
            ple_conv_kernel_size: integer("ple_conv_kernel_size", 4, 1, 64)?,
            ngram_size, heads_per_ngram,
            ngram_vocab_size_base: integer("ngram_vocab_size_base", 20_000_000, 2, 1 << 31)?,
            ngram_vocab_divisor: integer("make_ngram_vocab_size_divisible_by", 128, 1, 1 << 20)?,
            seed: integer("seed", 1234, i64::MIN, i64::MAX)?,
            split_ngram_parts: integer("split_ngram_parts", 512, 1, 1 << 20)?,
        };
        let c = Cfg {
            arch: Arch::Qwen4Exp,
            qwen4: Some(qwen4),
            n_kv_heads, head_dim, full_attn, attn_gate: true,
            lin_k_heads: integer("linear_num_key_heads", 16, 1, 1024)?,
            lin_v_heads: integer("linear_num_value_heads", 32, 1, 4096)?,
            lin_k_dim: integer("linear_key_head_dim", 128, 1, 1 << 16)?,
            lin_v_dim: integer("linear_value_head_dim", 128, 1, 1 << 16)?,
            lin_conv_k: integer("linear_conv_kernel_dim", 4, 1, 64)?,
            lin_gate_lb: None,
            hidden, n_layers, n_heads, n_experts, topk, moe_inter,
            dense_inter: shared_inter, first_dense: 0,
            q_lora: 0, kv_lora: 0, qk_nope: head_dim - rotary_dim, qk_rope: rotary_dim,
            v_head: head_dim, n_shared: 1, vocab, n_group: 1, topk_group: 1,
            norm_topk: boolean("norm_topk_prob", true)?,
            eps: positive(root.get("rms_norm_eps"), 1e-6, "rms_norm_eps")?,
            routed_scale: 1.0, theta, stop_ids,
            index_topk: 0, index_nh: 0, index_hd: 0, idx_type: Vec::new(),
            index_kpool: 0, index_kpool_tail: false, swiglu_limit: 0.0,
            hc_mult: 0, hc_eps: 0.0, hc_sinkhorn: 0,
            qk_head: head_dim, attn_scale: 1.0 / (head_dim as f32).sqrt(),
        };
        c.validate_gqa()?;
        c.validate_hybrid()?;
        Ok(c)
    }

    /// Parse a Qwen3-family dense-GQA config. The MoE/MLA fields are filled
    /// with the degenerate values that keep every existing invariant true —
    /// `first_dense = n_layers` means no layer ever takes the sparse path, so
    /// `n_experts = topk = 1` are never consulted by routing — rather than
    /// with zeros that would trip bounds checks or divide-by-zero downstream.
    fn from_json_gqa(root: &Value) -> Result<Cfg, Error> {
        let n_layers = gi(root, "num_hidden_layers");
        let hidden = gi(root, "hidden_size");
        let n_heads = gi(root, "num_attention_heads");
        let n_kv_heads = match gi(root, "num_key_value_heads") {
            0 => n_heads, // MHA spelling: absent field means every head has its own KV
            n => n,
        };
        let head_dim = match gi(root, "head_dim") {
            0 if n_heads > 0 => hidden / n_heads,
            hd => hd,
        };
        let mut stop_ids = Vec::new();
        match root.get("eos_token_id") {
            Some(Value::Number(n)) => match n.as_i64() {
                Some(id) => stop_ids.push(id as i32),
                None => return Err(Error::Format(format!("config: eos_token_id={n} is not an integer"))),
            },
            Some(Value::Array(a)) => {
                for v in a.iter() {
                    match v.as_i64() {
                        Some(id) => stop_ids.push(id as i32),
                        None => return Err(Error::Format(format!("config: eos_token_id entry {v} is not an integer"))),
                    }
                }
            }
            _ => {}
        }
        let theta_json = root
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .or_else(|| root.get("rope_theta"));
        let theta = match theta_json.and_then(|v| v.as_f64()) {
            Some(f) if f.is_finite() && f > 0.0 => f as f32,
            // GQA rotates every head lane; an unparseable theta would scramble
            // every position, so there is no safe default here at all.
            _ => return Err(Error::Format("config: qwen3 checkpoint without a positive rope_theta".into())),
        };
        let dense_inter = gi(root, "intermediate_size");
        let c = Cfg {
            qwen4: None,
            arch: Arch::DenseGqa,
            n_kv_heads,
            head_dim,
            full_attn: Vec::new(),
            attn_gate: false,
            lin_k_heads: 0,
            lin_v_heads: 0,
            lin_k_dim: 0,
            lin_v_dim: 0,
            lin_conv_k: 0,
            lin_gate_lb: None,
            hidden,
            n_layers,
            n_heads,
            // Degenerate MoE: no layer is sparse (first_dense = n_layers), so
            // these exist only to satisfy shared bounds checks.
            n_experts: 1,
            topk: 1,
            moe_inter: dense_inter,
            dense_inter,
            first_dense: n_layers,
            q_lora: 0,
            kv_lora: 0,
            qk_nope: 0,
            // The full head is rotated, so head_dim doubles as the RoPE span —
            // which keeps `RopeTable::from_cfg` and `attn_scale` correct without
            // a parallel set of derived fields.
            qk_rope: head_dim,
            v_head: head_dim,
            n_shared: 0,
            vocab: gi(root, "vocab_size"),
            n_group: 1,
            topk_group: 1,
            norm_topk: false,
            eps: gf(root, "rms_norm_eps", 1e-6),
            routed_scale: 1.0,
            theta,
            stop_ids,
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            idx_type: vec![false; n_layers.max(0) as usize],
            index_kpool: 0,
            index_kpool_tail: false,
            swiglu_limit: 0.0,
            hc_mult: 0,
            hc_eps: 0.0,
            hc_sinkhorn: 0,
            qk_head: head_dim,
            attn_scale: 1.0 / (head_dim.max(1) as f32).sqrt(),
        };
        c.validate_gqa()?;
        Ok(c)
    }

    /// Parse a Qwen3.5/3.8-family hybrid config (`root` is already the text
    /// sub-config when the checkpoint is the VL wrapper). Everything the
    /// dense-GQA parse establishes holds; on top of it: the per-layer
    /// full/linear schedule, the attention output gate, partial rotary, and
    /// the gated-DeltaNet geometry.
    fn from_json_hybrid(root: &Value) -> Result<Cfg, Error> {
        let mut c = Cfg::from_json_gqa(root)?;
        c.arch = Arch::HybridGdn;
        let n_layers = c.n_layers.max(0) as usize;
        // The explicit list wins; `full_attention_interval = k` (every k-th
        // layer, 1-indexed: layers k-1, 2k-1, …) is the fallback spelling.
        c.full_attn = match root.get("layer_types").and_then(|v| v.as_array()) {
            Some(types) => {
                if types.len() != n_layers {
                    return Err(Error::Format(format!(
                        "config: layer_types lists {} layers but num_hidden_layers={n_layers}",
                        types.len()
                    )));
                }
                let mut out = Vec::with_capacity(n_layers);
                for (i, t) in types.iter().enumerate() {
                    match t.as_str() {
                        Some("full_attention") => out.push(true),
                        Some("linear_attention") => out.push(false),
                        other => {
                            return Err(Error::Format(format!(
                                "config: layer_types[{i}] = {other:?} (expected full_attention | linear_attention)"
                            )))
                        }
                    }
                }
                out
            }
            None => {
                let k = gi(root, "full_attention_interval").max(1) as usize;
                (0..n_layers).map(|i| (i + 1) % k == 0).collect()
            }
        };
        c.attn_gate = root.get("attn_output_gate").and_then(|v| v.as_bool()).unwrap_or(false);
        c.lin_k_heads = gi(root, "linear_num_key_heads");
        c.lin_v_heads = gi(root, "linear_num_value_heads");
        c.lin_k_dim = gi(root, "linear_key_head_dim");
        c.lin_v_dim = gi(root, "linear_value_head_dim");
        c.lin_conv_k = gi(root, "linear_conv_kernel_dim");
        // Partial rotary narrows the RoPE span (qk_rope doubles as that span —
        // see from_json_gqa); the score scale stays 1/sqrt(head_dim).
        let pr = root
            .get("partial_rotary_factor")
            .or_else(|| root.get("rope_parameters").and_then(|rp| rp.get("partial_rotary_factor")))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        if !(pr > 0.0 && pr <= 1.0) {
            return Err(Error::Format(format!("config: partial_rotary_factor={pr} is outside (0,1]")));
        }
        c.qk_rope = ((c.head_dim as f64 * pr) as i64) & !1; // even, per the RoPE pair rule
        if c.qk_rope < 2 {
            return Err(Error::Format(format!(
                "config: partial rotary span {} is too small to rotate a pair",
                c.qk_rope
            )));
        }
        c.validate_hybrid()?;
        Ok(c)
    }

    /// Parse a GLM-5.3-Flash config (`root` is already the text sub-config when
    /// the checkpoint is the multimodal wrapper). The MoE half is the GLM-5.2
    /// shape (sigmoid router, correction bias, shared expert); attention is a
    /// per-layer schedule of KDA linear attention and NoPE MLA with a
    /// k-pool-compressed DSA indexer; the residual stream is `hc_mult` mHC
    /// hyper-connection streams; and every SwiGLU clamps at `swiglu_limit`.
    fn from_json_glm5next(root: &Value) -> Result<Cfg, Error> {
        let n_layers = gi(root, "num_hidden_layers");
        let n = n_layers.max(0) as usize;

        let mut stop_ids = Vec::new();
        match root.get("eos_token_id") {
            Some(Value::Number(num)) => match num.as_i64() {
                Some(id) => stop_ids.push(id as i32),
                None => return Err(Error::Format(format!("config: eos_token_id={num} is not an integer"))),
            },
            Some(Value::Array(a)) => {
                for v in a.iter() {
                    match v.as_i64() {
                        Some(id) => stop_ids.push(id as i32),
                        None => return Err(Error::Format(format!("config: eos_token_id entry {v} is not an integer"))),
                    }
                }
            }
            _ => {}
        }

        // Per-layer attention schedule. The explicit list wins; the fallback is
        // the HF class's own default (every 4th layer, 0-indexed 3, 7, 11, … is
        // the MLA/DSA layer, the rest KDA).
        let full_attn: Vec<bool> = match root.get("layer_types").and_then(|v| v.as_array()) {
            Some(types) => {
                if types.len() != n {
                    return Err(Error::Format(format!(
                        "config: layer_types lists {} layers but num_hidden_layers={n}",
                        types.len()
                    )));
                }
                let mut out = Vec::with_capacity(n);
                for (i, t) in types.iter().enumerate() {
                    match t.as_str() {
                        // "full_attention" is the HF normalization alias for the
                        // MLA/DSA lane.
                        Some("deepseek_sparse_attention") | Some("full_attention") => out.push(true),
                        Some("linear_attention") => out.push(false),
                        other => {
                            return Err(Error::Format(format!(
                                "config: layer_types[{i}] = {other:?} (expected deepseek_sparse_attention | linear_attention)"
                            )))
                        }
                    }
                }
                out
            }
            None => (0..n).map(|i| i % 4 == 3).collect(),
        };

        // KDA geometry: the `linear_attn_config` dict, with the flat spellings
        // and the HF class defaults (64 heads × 128, 4 conv taps) as fallback.
        let lac = root.get("linear_attn_config");
        let lac_i = |key: &str, flat: &str, default: i64| -> i64 {
            match lac.and_then(|d| d.get(key)).and_then(|v| v.as_i64()) {
                Some(v) => v,
                None => match gi(root, flat) {
                    0 => default,
                    v => v,
                },
            }
        };
        let lin_heads = lac_i("num_heads", "linear_num_heads", 64);
        let lin_dim = lac_i("head_dim", "linear_head_dim", 128);
        let lin_conv_k = lac_i("short_conv_kernel_size", "linear_conv_kernel_dim", 4);
        // gate_lower_bound: a number bounds the log-decay at that value; absent
        // (or null) falls back to -5.0 unless `safe_gate: false` opts into the
        // unbounded softplus form.
        let lin_gate_lb = match lac.and_then(|d| d.get("gate_lower_bound")).and_then(|v| v.as_f64()) {
            Some(b) => Some(b as f32),
            None => {
                let safe = lac.and_then(|d| d.get("safe_gate")).and_then(|v| v.as_bool()).unwrap_or(true);
                safe.then_some(-5.0)
            }
        };

        // MoE layer schedule: `mlp_layer_types` (dense-prefix + sparse rest) or
        // `first_k_dense_replace`. A non-prefix dense pattern would need a
        // per-layer sparse vector this Cfg does not carry — refuse rather than
        // load a layer with the wrong MLP kind.
        let first_dense = match root.get("mlp_layer_types").and_then(|v| v.as_array()) {
            Some(types) => {
                if types.len() != n {
                    return Err(Error::Format(format!(
                        "config: mlp_layer_types lists {} layers but num_hidden_layers={n}",
                        types.len()
                    )));
                }
                let dense_prefix = types.iter().take_while(|t| t.as_str() == Some("dense")).count();
                for (i, t) in types.iter().enumerate().skip(dense_prefix) {
                    match t.as_str() {
                        Some("sparse") => {}
                        Some("dense") => {
                            return Err(Error::Format(format!(
                                "config: mlp_layer_types has a dense layer at index {i} after sparse layers — only a dense prefix is supported"
                            )))
                        }
                        other => {
                            return Err(Error::Format(format!(
                                "config: mlp_layer_types[{i}] = {other:?} (expected dense | sparse)"
                            )))
                        }
                    }
                }
                dense_prefix as i64
            }
            None => gi(root, "first_k_dense_replace"),
        };

        // DSA indexer per-layer schedule — same explicit-list-or-formula rule
        // as the GLM-5.2 parse (GLM-5.3-Flash ships all-"full").
        let mut idx_type = vec![false; n];
        {
            let types = root.get("indexer_types").and_then(|v| v.as_array());
            let mut freq = gi(root, "index_topk_freq");
            if freq < 1 {
                freq = 1;
            }
            let off = root.get("index_skip_topk_offset").and_then(|v| v.as_i64()).unwrap_or(2);
            for (i, slot) in idx_type.iter_mut().enumerate() {
                *slot = match types.and_then(|t| t.get(i)).and_then(|v| v.as_str()) {
                    Some(s) => s == "full",
                    None => {
                        let v = ((i as i64) - off + 1).max(0);
                        v % freq == 0
                    }
                };
            }
        }

        let qk_nope = gi(root, "qk_nope_head_dim");
        let qk_rope = gi(root, "qk_rope_head_dim");
        let n_heads = gi(root, "num_attention_heads");
        let mut c = Cfg {
            qwen4: None,
            arch: Arch::Glm5Next,
            n_kv_heads: match gi(root, "num_key_value_heads") {
                0 => n_heads,
                v => v,
            },
            head_dim: 0,
            full_attn,
            attn_gate: false,
            lin_k_heads: lin_heads,
            lin_v_heads: lin_heads,
            lin_k_dim: lin_dim,
            lin_v_dim: lin_dim,
            lin_conv_k,
            lin_gate_lb,
            hidden: gi(root, "hidden_size"),
            n_layers,
            n_heads,
            n_experts: gi(root, "n_routed_experts"),
            topk: gi(root, "num_experts_per_tok"),
            moe_inter: gi(root, "moe_intermediate_size"),
            dense_inter: gi(root, "intermediate_size"),
            first_dense,
            q_lora: gi(root, "q_lora_rank"),
            kv_lora: gi(root, "kv_lora_rank"),
            qk_nope,
            qk_rope,
            v_head: gi(root, "v_head_dim"),
            n_shared: gi(root, "n_shared_experts"),
            vocab: gi(root, "vocab_size"),
            n_group: match gi(root, "n_group") {
                0 => 1,
                v => v,
            },
            topk_group: match gi(root, "topk_group") {
                0 => 1,
                v => v,
            },
            norm_topk: root.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(false),
            eps: gf(root, "rms_norm_eps", 1e-5),
            routed_scale: gf(root, "routed_scaling_factor", 1.0),
            // NoPE: no attention lane is ever rotated (qk_rope = 0 is enforced
            // below), so theta exists only to satisfy RopeTable's signature.
            theta: 10000.0,
            stop_ids,
            index_topk: gi(root, "index_topk"),
            index_nh: gi(root, "index_n_heads"),
            index_hd: gi(root, "index_head_dim"),
            idx_type,
            index_kpool: match gi(root, "index_kpool") {
                0 => 16, // the HF class default when the field is absent
                v => v,
            },
            index_kpool_tail: root
                .get("index_kpool_always_select_tail")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            swiglu_limit: gf(root, "swiglu_limit", 10.0),
            hc_mult: match gi(root, "hc_mult") {
                0 => 4, // the HF class default when the field is absent
                v => v,
            },
            hc_eps: gf(root, "hc_eps", 1e-6),
            hc_sinkhorn: match gi(root, "hc_sinkhorn_iters") {
                0 => 20,
                v => v,
            },
            qk_head: qk_nope + qk_rope,
            attn_scale: 0.0,
        };
        c.attn_scale = 1.0 / (c.qk_head.max(1) as f32).sqrt();
        c.validate_glm5next()?;
        Ok(c)
    }

    /// The GLM-5.3-Flash choke point: the GLM-5.2 bounds where the shapes are
    /// shared, plus the KDA/mHC/k-pool geometry, minus the RoPE checks (the
    /// architecture is NoPE — `qk_rope_head_dim = 0` is *required*, matching
    /// the HF implementation's own validation).
    fn validate_glm5next(&self) -> Result<(), Error> {
        let ck = |name: &str, v: i64, lo: i64, hi: i64| -> Result<(), Error> {
            if v < lo || v > hi {
                Err(Error::Format(format!("config: {name}={v} is outside [{lo},{hi}]")))
            } else {
                Ok(())
            }
        };
        ck("hidden_size", self.hidden, 1, 1 << 20)?;
        ck("num_hidden_layers", self.n_layers, 1, 128)?;
        ck("num_attention_heads", self.n_heads, 1, 1024)?;
        ck("n_routed_experts", self.n_experts, 1, 4096)?;
        ck("num_experts_per_tok", self.topk, 1, 64)?;
        ck("moe_intermediate_size", self.moe_inter, 1, 1 << 20)?;
        ck("intermediate_size", self.dense_inter, 1, 1 << 24)?;
        ck("first_k_dense_replace", self.first_dense, 0, self.n_layers)?;
        ck("q_lora_rank", self.q_lora, 1, 1 << 20)?;
        ck("kv_lora_rank", self.kv_lora, 1, 1 << 20)?;
        ck("qk_nope_head_dim", self.qk_nope, 1, 1 << 16)?;
        ck("v_head_dim", self.v_head, 1, 1 << 16)?;
        ck("n_shared_experts", self.n_shared, 0, 64)?;
        ck("vocab_size", self.vocab, 1, 1 << 24)?;
        if self.qk_rope != 0 {
            return Err(Error::Format(format!(
                "config: qk_rope_head_dim={} — glm5_next attention is NoPE, expected 0",
                self.qk_rope
            )));
        }
        if self.topk > self.n_experts {
            return Err(Error::Format(format!(
                "config: num_experts_per_tok={} exceeds n_routed_experts={}",
                self.topk, self.n_experts
            )));
        }
        if self.n_group != 1 || self.topk_group != 1 {
            return Err(Error::Format(format!(
                "config: n_group={}/topk_group={} — this engine requires ungrouped routing (both 1)",
                self.n_group, self.topk_group
            )));
        }
        if self.full_attn.len() != self.n_layers.max(0) as usize {
            return Err(Error::Format("config: layer_types length mismatch".into()));
        }
        // KDA geometry, only when a linear layer exists in the schedule.
        if self.full_attn.iter().any(|f| !f) {
            ck("linear_attn num_heads", self.lin_k_heads, 1, 1024)?;
            ck("linear_attn head_dim", self.lin_k_dim, 1, 1 << 16)?;
            ck("linear_attn short_conv_kernel_size", self.lin_conv_k, 1, 64)?;
        }
        // DSA indexer + k-pool geometry, only when configured.
        if self.index_nh > 0 && self.index_hd > 0 {
            ck("index_topk", self.index_topk, 1, 1 << 20)?;
            ck("index_n_heads", self.index_nh, 1, 1024)?;
            ck("index_head_dim", self.index_hd, 1, 1 << 16)?;
            ck("index_kpool", self.index_kpool, 1, 1 << 10)?;
            if self.index_topk % self.index_kpool != 0 {
                return Err(Error::Format(format!(
                    "config: index_topk={} is not divisible by index_kpool={}",
                    self.index_topk, self.index_kpool
                )));
            }
        }
        ck("hc_mult", self.hc_mult, 1, 16)?;
        ck("hc_sinkhorn_iters", self.hc_sinkhorn, 1, 256)?;
        if !(self.swiglu_limit > 0.0 && self.swiglu_limit.is_finite()) {
            return Err(Error::Format(format!(
                "config: swiglu_limit={} must be a positive finite clamp",
                self.swiglu_limit
            )));
        }
        if !(self.hc_eps > 0.0 && self.hc_eps.is_finite()) {
            return Err(Error::Format(format!("config: hc_eps={} must be a positive finite floor", self.hc_eps)));
        }
        Ok(())
    }

    /// The linear-attention geometry checks on top of [`Self::validate_gqa`]
    /// (which already ran inside [`Self::from_json_gqa`]).
    fn validate_hybrid(&self) -> Result<(), Error> {
        let ck = |name: &str, v: i64, lo: i64, hi: i64| -> Result<(), Error> {
            if v < lo || v > hi {
                Err(Error::Format(format!("config: {name}={v} is outside [{lo},{hi}]")))
            } else {
                Ok(())
            }
        };
        // A hybrid with no linear layers is just dense GQA misdeclared; with no
        // full layers it still works, so only the geometry is bounds-checked.
        if self.full_attn.iter().any(|f| !f) {
            ck("linear_num_key_heads", self.lin_k_heads, 1, 1024)?;
            ck("linear_num_value_heads", self.lin_v_heads, 1, 4096)?;
            ck("linear_key_head_dim", self.lin_k_dim, 1, 1 << 16)?;
            ck("linear_value_head_dim", self.lin_v_dim, 1, 1 << 16)?;
            ck("linear_conv_kernel_dim", self.lin_conv_k, 1, 64)?;
            if self.lin_v_heads % self.lin_k_heads != 0 {
                return Err(Error::Format(format!(
                    "config: linear_num_value_heads={} is not a multiple of linear_num_key_heads={}",
                    self.lin_v_heads, self.lin_k_heads
                )));
            }
        }
        Ok(())
    }

    /// Bounds checks for the dense-GQA shape — the same hostile-config choke
    /// point [`Self::validate`] is for MLA, over the fields GQA actually reads.
    fn validate_gqa(&self) -> Result<(), Error> {
        let ck = |name: &str, v: i64, lo: i64, hi: i64| -> Result<(), Error> {
            if v < lo || v > hi {
                Err(Error::Format(format!("config: {name}={v} is outside [{lo},{hi}]")))
            } else {
                Ok(())
            }
        };
        ck("hidden_size", self.hidden, 1, 1 << 20)?;
        ck("num_hidden_layers", self.n_layers, 1, 128)?;
        ck("num_attention_heads", self.n_heads, 1, 1024)?;
        ck("num_key_value_heads", self.n_kv_heads, 1, self.n_heads)?;
        ck("head_dim", self.head_dim, 2, 1 << 16)?;
        ck("intermediate_size", self.dense_inter, 1, 1 << 24)?;
        ck("vocab_size", self.vocab, 1, 1 << 24)?;
        // Grouped queries share KV heads in whole groups; a ragged ratio would
        // leave some queries with no defined KV head.
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(Error::Format(format!(
                "config: num_attention_heads={} is not a multiple of num_key_value_heads={}",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.head_dim % 2 != 0 {
            return Err(Error::Format(format!(
                "config: head_dim={} must be even (RoPE rotates lane pairs)",
                self.head_dim
            )));
        }
        Ok(())
    }

    /// Width of one cached row in the KV cache's first slot: the MLA compressed
    /// latent, or all GQA key heads for one position. [`LayerKv`] is
    /// width-parameterized, which is what lets both architectures share every
    /// cache mechanism (prefix cache, disk sessions, truncate/clone) unchanged.
    pub fn kv_row_a(&self) -> i64 {
        match self.arch {
            Arch::GlmMla | Arch::Glm5Next => self.kv_lora,
            Arch::DenseGqa | Arch::HybridGdn | Arch::Qwen4Exp => self.n_kv_heads * self.head_dim,
        }
    }

    /// Width of one cached row in the second slot: MLA's rope keys, or all GQA
    /// value heads for one position.
    pub fn kv_row_b(&self) -> i64 {
        match self.arch {
            // Glm5Next is NoPE (qk_rope = 0): its second slot is legitimately
            // zero-width — LayerKv is width-parameterized and appends empty rows.
            Arch::GlmMla | Arch::Glm5Next => self.qk_rope,
            Arch::DenseGqa | Arch::HybridGdn | Arch::Qwen4Exp => self.n_kv_heads * self.head_dim,
        }
    }

    /// The `CKR` bounds checks from `load_cfg` — a single choke point that
    /// rejects hostile dimensions before any downstream allocation.
    fn validate(&self) -> Result<(), Error> {
        let ck = |name: &str, v: i64, lo: i64, hi: i64| -> Result<(), Error> {
            if v < lo || v > hi {
                Err(Error::Format(format!("config: {name}={v} is outside [{lo},{hi}]")))
            } else {
                Ok(())
            }
        };
        ck("hidden_size", self.hidden, 1, 1 << 20)?;
        ck("num_hidden_layers", self.n_layers, 1, 128)?;
        ck("num_attention_heads", self.n_heads, 1, 1024)?;
        ck("n_routed_experts", self.n_experts, 1, 4096)?;
        ck("num_experts_per_tok", self.topk, 1, 64)?;
        ck("moe_intermediate_size", self.moe_inter, 1, 1 << 20)?;
        ck("intermediate_size", self.dense_inter, 1, 1 << 24)?;
        ck("first_k_dense_replace", self.first_dense, 0, self.n_layers)?;
        ck("q_lora_rank", self.q_lora, 0, 1 << 20)?;
        ck("kv_lora_rank", self.kv_lora, 1, 1 << 20)?;
        ck("qk_nope_head_dim", self.qk_nope, 1, 1 << 16)?;
        ck("qk_rope_head_dim", self.qk_rope, 1, 1 << 16)?;
        ck("v_head_dim", self.v_head, 1, 1 << 16)?;
        ck("n_shared_experts", self.n_shared, 0, 64)?;
        ck("vocab_size", self.vocab, 1, 1 << 24)?;
        ck("index_topk", self.index_topk, 0, 1 << 20)?;
        ck("index_n_heads", self.index_nh, 0, 1024)?;
        ck("index_head_dim", self.index_hd, 0, 1 << 16)?;
        // The router selects `topk` distinct experts without replacement, so a
        // topk above the expert count has no valid selection to make.
        if self.topk > self.n_experts {
            return Err(Error::Format(format!(
                "config: num_experts_per_tok={} exceeds n_routed_experts={}",
                self.topk, self.n_experts
            )));
        }
        // RoPE rotates (2j, 2j+1) pairs, so an odd lane count would leave the
        // final lane un-rotated *and* in the wrong output slot.
        for (name, v) in [("qk_rope_head_dim", self.qk_rope), ("index_head_dim", self.index_hd)] {
            if v % 2 != 0 {
                return Err(Error::Format(format!("config: {name}={v} must be even (RoPE rotates lane pairs)")));
            }
        }
        // The indexer keeps the top-`index_topk` keys; zero would select no keys
        // at all and yield an identically-zero attention context.
        if self.index_nh > 0 && self.index_hd > 0 && self.index_topk < 1 {
            return Err(Error::Format(
                "config: index_topk must be >= 1 when the DSA indexer is configured".into(),
            ));
        }
        // `topk_group` is only meaningful with grouped routing, which this
        // engine does not implement (n_group=1 is enforced above).
        if self.topk_group != 1 {
            return Err(Error::Format(format!(
                "config: topk_group={} is unsupported (this engine requires topk_group=1)",
                self.topk_group
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny-oracle config from `c/tools/make_glm_oracle.py`.
    fn tiny_json() -> Value {
        serde_json::json!({
            "vocab_size": 256,
            "hidden_size": 128,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 5,
            "first_k_dense_replace": 3,
            "num_attention_heads": 4,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 24,
            "qk_rope_head_dim": 8,
            "v_head_dim": 32,
            "index_topk": 4096,
            "index_head_dim": 16,
            "index_n_heads": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "rms_norm_eps": 1e-5,
            "eos_token_id": [1, 2, 3]
        })
    }

    fn qwen4_json() -> Result<Value, Error> {
        serde_json::from_str(concat!(
            r#"{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","#,
            r#""hidden_size":2560,"vocab_size":248320,"num_hidden_layers":48,"#,
            r#""num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"#,
            r#""num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"#,
            r#""shared_expert_intermediate_size":640,"linear_num_key_heads":16,"#,
            r#""linear_num_value_heads":48,"linear_key_head_dim":128,"#,
            r#""linear_value_head_dim":128,"linear_conv_kernel_dim":4,"#,
            r#""output_gate_type":"sigmoid","hc_count":4,"hc_lowrank":320,"#,
            r#""ple_layer_ids":[2],"ple_embed_dim":2560,"heads_per_ngram":8,"#,
            r#""ngram_size":3,"ngram_vocab_size_base":20000000,"#,
            r#""make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":128,"#,
            r#""ple_conv_kernel_size":4,"eos_token_id":248044,"indexer_n_heads":4,"#,
            r#""indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"#,
            r#""indexer_compress_ratio":4,"#,
            r#""rope_parameters":{"rope_type":"default","rope_theta":10000000,"#,
            r#""partial_rotary_factor":0.25,"mrope_interleaved":true,"#,
            r#""mrope_section":[11,11,10]}}}"#
        )).map_err(|e| Error::Format(format!("fixture: {e}")))
    }

    #[test]
    fn qwen4_official_geometry_is_distinct_and_preserved() -> Result<(), Error> {
        let c = Cfg::from_json(&qwen4_json()?)?;
        let q = c.qwen4.as_ref().ok_or_else(|| Error::Format("missing Qwen4 config".into()))?;
        assert_eq!(c.arch, Arch::Qwen4Exp);
        assert_eq!((c.hidden, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim), (2560, 48, 24, 2, 256));
        assert_eq!((c.n_experts, c.topk, c.moe_inter, c.dense_inter), (512, 10, 640, 640));
        assert_eq!((c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim), (16, 48, 128, 128));
        assert_eq!((c.hc_mult, c.index_topk, c.first_dense), (0, 0, 0));
        assert_eq!((c.qk_rope, c.kv_row_a(), c.kv_row_b()), (64, 512, 512));
        assert_eq!(c.full_attn, (0..48).map(|i| i % 4 == 3).collect::<Vec<_>>());
        assert_eq!((q.hc_count, q.hc_lowrank), (4, 320));
        assert_eq!(q.output_gate, Qwen4OutputGate::Sigmoid);
        assert_eq!(q.ple_layer_ids, [2]);
        assert_eq!(q.seed, 1234);
        assert_eq!(q.split_ngram_parts, 128);
        assert_eq!(q.qsa, Some(Qwen4QsaCfg { n_heads: 4, head_dim: 128, budget: 2048, compress_ratio: 4 }));
        assert!(c.attn_gate && c.norm_topk);
        assert_eq!(c.stop_ids, [248044]);
        let flat = Cfg::from_json(&qwen4_json()?["text_config"])?;
        assert_eq!(flat.qwen4, c.qwen4);
        assert_eq!(flat.arch, c.arch);
        Ok(())
    }

    #[test]
    fn qwen4_alias_schedule_defaults_and_shared_width() -> Result<(), Error> {
        let mut j = qwen4_json()?["text_config"].clone();
        j["num_hidden_layers"] = serde_json::json!(4);
        j["layer_types"] = serde_json::json!(["linear_attention", "linear_attention", "linear_attention", "full_attention"]);
        j["output_gate_type"] = Value::Null;
        j["ple_layer_ids"] = serde_json::json!([2, 1, 2]);
        j["ple_embed_dim"] = Value::Null;
        j["shared_expert_intermediate_size"] = serde_json::json!(1024);
        let c = Cfg::from_json(&j)?;
        let q = c.qwen4.as_ref().ok_or_else(|| Error::Format("missing Qwen4 config".into()))?;
        assert_eq!(q.output_gate, Qwen4OutputGate::Silu);
        assert_eq!(q.ple_layer_ids, [1, 2]);
        assert_eq!(q.ple_embed_dim, c.hidden);
        assert_eq!((c.moe_inter, c.dense_inter), (640, 1024));
        j["layer_types"][3] = serde_json::json!("qwen_sparse_attention");
        assert_eq!(Cfg::from_json(&j)?.full_attn, c.full_attn);
        Ok(())
    }

    #[test]
    fn qwen4_rejects_malformed_or_unsupported_geometry() -> Result<(), Error> {
        for (key, value) in [
            ("num_hidden_layers", serde_json::json!(i64::MAX)),
            ("num_hidden_layers", serde_json::json!(2.5)),
            ("num_attention_heads", serde_json::json!(0)),
            ("num_key_value_heads", serde_json::json!(5)),
            ("num_experts_per_tok", serde_json::json!(513)),
            ("num_experts", serde_json::json!(0)),
            ("linear_num_value_heads", serde_json::json!(47)),
            ("linear_conv_kernel_dim", serde_json::json!(0)),
            ("hc_count", serde_json::json!(1)),
            ("hc_lowrank", serde_json::json!(0)),
            ("indexer_budget", serde_json::json!(7)),
            ("indexer_kv_heads", serde_json::json!(2)),
            ("indexer_head_dim", serde_json::json!(32)),
            ("indexer_n_heads", Value::Null),
            ("ple_layer_ids", serde_json::json!([0])),
            ("ple_layer_ids", serde_json::json!([49])),
            ("ple_layer_ids", serde_json::json!([4])),
            ("ple_embed_dim", serde_json::json!(15)),
            ("ngram_size", serde_json::json!(1)),
            ("eos_token_id", serde_json::json!([])),
            ("eos_token_id", serde_json::json!(248320)),
            ("output_gate_type", serde_json::json!("relu")),
            ("hidden_act", serde_json::json!("gelu")),
            ("attention_bias", serde_json::json!(true)),
            ("norm_topk_prob", serde_json::json!(1)),
            ("tie_word_embeddings", serde_json::json!(true)),
            ("rms_norm_eps", serde_json::json!(1e100)),
            ("full_attention_interval", serde_json::json!(0)),
            ("layer_types", serde_json::json!(["linear_attention"])),
            ("rope_parameters", serde_json::json!({"rope_type": "yarn"})),
            ("rope_parameters", serde_json::json!({"partial_rotary_factor": 0.014})),
            ("rope_parameters", serde_json::json!({"rope_theta": 1e100})),
        ] {
            let mut j = qwen4_json()?;
            j["text_config"][key] = value;
            assert!(Cfg::from_json(&j).is_err(), "accepted invalid {key}");
        }
        let mut j = qwen4_json()?;
        j["text_config"] = Value::Null;
        assert!(Cfg::from_json(&j).is_err());
        j["model_type"] = serde_json::json!("qwen4_exp_unknown");
        assert!(Cfg::from_json(&j).is_err());
        Ok(())
    }

    #[test]
    fn parses_tiny_oracle() -> Result<(), Error> {
        let c = Cfg::from_json(&tiny_json())?;
        assert_eq!(c.hidden, 128);
        assert_eq!(c.n_layers, 5);
        assert_eq!(c.first_dense, 3);
        assert_eq!(c.n_experts, 8);
        assert_eq!(c.topk, 2);
        assert_eq!(c.qk_nope, 24);
        assert_eq!(c.qk_rope, 8);
        assert_eq!(c.qk_head, 32); // derived: 24 + 8
        assert!((c.attn_scale - 1.0 / 32f32.sqrt()).abs() < 1e-9);
        assert!(c.norm_topk);
        assert_eq!(c.routed_scale, 2.5);
        assert_eq!(c.theta, 10000.0);
        assert_eq!(c.stop_ids, vec![1, 2, 3]);
        assert_eq!(c.idx_type.len(), 5);
        Ok(())
    }

    #[test]
    fn rejects_n_group_ne_1() {
        let mut j = tiny_json();
        j["n_group"] = serde_json::json!(2);
        assert!(Cfg::from_json(&j).is_err());
    }

    #[test]
    fn rejects_out_of_bounds() {
        let mut j = tiny_json();
        j["num_experts_per_tok"] = serde_json::json!(9999); // > 64
        assert!(Cfg::from_json(&j).is_err());
    }

    #[test]
    fn scalar_eos_token() -> Result<(), Error> {
        let mut j = tiny_json();
        j["eos_token_id"] = serde_json::json!(7);
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.stop_ids, vec![7]);
        Ok(())
    }

    #[test]
    fn rejects_topk_above_expert_count() {
        // Both fields are individually in range, but the router cannot select 8
        // distinct experts out of 4 — previously this loaded and then indexed
        // out of bounds on the first sparse layer.
        let mut j = tiny_json();
        j["n_routed_experts"] = serde_json::json!(4);
        j["num_experts_per_tok"] = serde_json::json!(8);
        assert!(Cfg::from_json(&j).is_err(), "topk > n_experts must be rejected at load");
    }

    #[test]
    fn rejects_odd_rope_dims() {
        for field in ["qk_rope_head_dim", "index_head_dim"] {
            let mut j = tiny_json();
            j[field] = serde_json::json!(7);
            assert!(Cfg::from_json(&j).is_err(), "{field} must be even");
        }
    }

    #[test]
    fn rejects_zero_index_topk_with_indexer() {
        // index_topk=0 (also the missing-field default) selects no keys, which
        // makes DSA attention output identically zero with no error.
        let mut j = tiny_json();
        j["index_topk"] = serde_json::json!(0);
        assert!(Cfg::from_json(&j).is_err());
        // ...but a model with no indexer at all is still fine.
        let mut j2 = tiny_json();
        j2["index_topk"] = serde_json::json!(0);
        j2["index_n_heads"] = serde_json::json!(0);
        j2["index_head_dim"] = serde_json::json!(0);
        assert!(Cfg::from_json(&j2).is_ok(), "no-indexer model needs no index_topk");
    }

    #[test]
    fn reads_top_level_rope_theta() -> Result<(), Error> {
        // The long-standing HF layout puts rope_theta at the top level; reading
        // only the nested spelling silently defaulted it to 10000.0.
        let mut j = tiny_json();
        j["rope_parameters"] = serde_json::json!({ "rope_type": "default" });
        j["rope_theta"] = serde_json::json!(1_000_000.0);
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.theta, 1_000_000.0);
        // The nested spelling still wins when both are present.
        let mut j2 = tiny_json();
        j2["rope_theta"] = serde_json::json!(1_000_000.0);
        assert_eq!(Cfg::from_json(&j2)?.theta, 10000.0);
        Ok(())
    }

    #[test]
    fn rejects_missing_rope_theta_when_roped() {
        let mut j = tiny_json();
        j["rope_parameters"] = serde_json::json!({ "rope_type": "default" });
        assert!(Cfg::from_json(&j).is_err(), "a roped model must not silently default theta");
    }

    /// The Track-C tiny Qwen-shaped config (kept in sync with
    /// `peregrine_model::testkit::tiny_qwen_cfg_json` — C2's importer fixture
    /// matches these dims).
    fn tiny_qwen_json() -> Value {
        serde_json::json!({
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 0
        })
    }

    #[test]
    fn qwen3_config_parses_as_dense_gqa() -> Result<(), Error> {
        let c = Cfg::from_json(&tiny_qwen_json())?;
        assert_eq!(c.arch, Arch::DenseGqa);
        assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (4, 2, 4));
        // Degenerate MoE invariants: no sparse layer can ever engage.
        assert_eq!(c.first_dense, c.n_layers, "every layer must take the dense path");
        assert_eq!((c.n_experts, c.topk), (1, 1));
        // head_dim doubles as the RoPE span and the score scale basis.
        assert_eq!(c.qk_rope, 4);
        assert_eq!(c.qk_head, 4);
        assert!((c.attn_scale - 0.5).abs() < 1e-9); // 1/sqrt(4)
        // KV rows: all K heads in slot a, all V heads in slot b.
        assert_eq!((c.kv_row_a(), c.kv_row_b()), (8, 8));
        Ok(())
    }

    #[test]
    fn qwen3_defaults_head_dim_and_kv_heads_when_absent() -> Result<(), Error> {
        let mut j = tiny_qwen_json();
        if let Some(o) = j.as_object_mut() {
            o.remove("head_dim");
            o.remove("num_key_value_heads");
        }
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.head_dim, 4, "head_dim defaults to hidden/n_heads");
        assert_eq!(c.n_kv_heads, 4, "absent num_key_value_heads means MHA");
        Ok(())
    }

    #[test]
    fn declared_unknown_model_type_is_a_loud_error() {
        let mut j = tiny_json();
        j["model_type"] = serde_json::json!("llama");
        assert!(Cfg::from_json(&j).is_err(), "an unimplemented declared arch must not load as GLM");
        // ...while GLM-family declarations and the historical absent field both load.
        let mut j2 = tiny_json();
        j2["model_type"] = serde_json::json!("glm_moe");
        assert!(Cfg::from_json(&j2).is_ok());
        assert_eq!(Cfg::from_json(&tiny_json()).map(|c| c.arch).ok(), Some(Arch::GlmMla));
    }

    #[test]
    fn gqa_rejects_ragged_head_grouping_and_theta_less_configs() {
        let mut j = tiny_qwen_json();
        j["num_key_value_heads"] = serde_json::json!(3); // 4 % 3 != 0
        assert!(Cfg::from_json(&j).is_err());
        let mut j2 = tiny_qwen_json();
        if let Some(o) = j2.as_object_mut() {
            o.remove("rope_theta");
        }
        assert!(Cfg::from_json(&j2).is_err(), "GQA rotates every lane; theta cannot default");
    }

    /// The Track-C tiny hybrid config: 3 layers (linear, linear, full), the
    /// qwen3_5 shape at toy dims. Kept in sync with
    /// `peregrine_model::testkit::tiny_hybrid_cfg_json` and C2's importer fixture.
    fn tiny_hybrid_json() -> Value {
        serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 8,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "full_attention_interval": 3,
                "layer_types": ["linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "attn_output_gate": true,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
                "rms_norm_eps": 1e-6,
                "eos_token_id": 0
            }
        })
    }

    #[test]
    fn qwen3_5_config_parses_as_hybrid() -> Result<(), Error> {
        let c = Cfg::from_json(&tiny_hybrid_json())?;
        assert_eq!(c.arch, Arch::HybridGdn);
        assert_eq!(c.full_attn, vec![false, false, true]);
        assert!(c.attn_gate);
        assert_eq!((c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, c.lin_conv_k), (2, 4, 4, 4, 4));
        // Partial rotary: span = head_dim * 0.25 = 2 lanes; scale on full head_dim.
        assert_eq!(c.qk_rope, 2);
        assert!((c.attn_scale - 1.0 / (8f32).sqrt()).abs() < 1e-9);
        assert_eq!(c.theta, 10_000_000.0);
        // KV rows cover only full-attention layers' geometry.
        assert_eq!((c.kv_row_a(), c.kv_row_b()), (16, 16));
        assert_eq!(c.first_dense, c.n_layers, "hybrid MLPs all take the dense path");
        Ok(())
    }

    #[test]
    fn hybrid_layer_types_must_match_layer_count() {
        let mut j = tiny_hybrid_json();
        j["text_config"]["layer_types"] = serde_json::json!(["linear_attention", "full_attention"]);
        assert!(Cfg::from_json(&j).is_err(), "a 2-entry schedule for 3 layers must not load");
    }

    #[test]
    fn hybrid_interval_fallback_matches_the_shipped_schedule() -> Result<(), Error> {
        // Drop the explicit list; full_attention_interval=3 must reproduce it
        // (1-indexed every-3rd: layers 2, 5, … → here just layer index 2).
        let mut j = tiny_hybrid_json();
        if let Some(o) = j["text_config"].as_object_mut() {
            o.remove("layer_types");
        }
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.full_attn, vec![false, false, true]);
        Ok(())
    }

    #[test]
    fn generation_config_stop_tokens_are_unioned_in() -> Result<(), Error> {
        // config.json says 248044; generation_config.json says [248046, 248044]
        // — the loaded set must carry both, each exactly once.
        let d = std::env::temp_dir().join(format!("peregrine_genconf_{}", std::process::id()));
        if d.exists() {
            std::fs::remove_dir_all(&d)?;
        }
        std::fs::create_dir_all(&d)?;
        let mut j = tiny_qwen_json();
        j["eos_token_id"] = serde_json::json!(248044);
        std::fs::write(d.join("config.json"), serde_json::to_vec(&j)?)?;
        std::fs::write(
            d.join("generation_config.json"),
            serde_json::to_vec(&serde_json::json!({"eos_token_id": [248046, 248044]}))?,
        )?;
        let c = Cfg::load(&d)?;
        assert_eq!(c.stop_ids, vec![248044, 248046], "union, deduped, config.json order first");
        // Absent generation_config keeps the historical single-file behaviour.
        std::fs::remove_file(d.join("generation_config.json"))?;
        assert_eq!(Cfg::load(&d)?.stop_ids, vec![248044]);
        std::fs::remove_dir_all(&d)?;
        Ok(())
    }

    #[test]
    fn keeps_every_eos_token() -> Result<(), Error> {
        let mut j = tiny_json();
        j["eos_token_id"] = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.stop_ids.len(), 10, "no stop id may be dropped");
        Ok(())
    }

    /// A miniature of the real GLM-5.3-Flash config: the multimodal wrapper's
    /// nesting, the hybrid KDA/DSA schedule, the k-pool indexer, mHC and the
    /// SwiGLU clamp — every field the Glm5Next parse is responsible for.
    fn tiny_glm5next_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "vocab_size": 32, "hidden_size": 16,
                "num_hidden_layers": 4, "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 8, "intermediate_size": 8,
                "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
                "indexer_types": ["full", "full", "full", "full"],
                "q_lora_rank": 12, "kv_lora_rank": 8,
                "qk_nope_head_dim": 4, "qk_rope_head_dim": 0, "v_head_dim": 4,
                "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
                "norm_topk_prob": true, "routed_scaling_factor": 2.5,
                "rms_norm_eps": 1e-5, "swiglu_limit": 10.0,
                "linear_attn_config": {
                    "num_heads": 2, "head_dim": 4,
                    "short_conv_kernel_size": 3, "gate_lower_bound": -5.0
                },
                "index_topk": 8, "index_n_heads": 2, "index_head_dim": 4,
                "index_kpool": 4, "index_kpool_always_select_tail": true,
                "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
                "mhc": true,
                "eos_token_id": [0, 3]
            }
        })
    }

    #[test]
    fn glm5next_config_parses_the_wrapped_and_flat_spellings() -> Result<(), Error> {
        let c = Cfg::from_json(&tiny_glm5next_json())?;
        assert_eq!(c.arch, Arch::Glm5Next);
        assert_eq!(c.full_attn, vec![false, false, false, true], "3 KDA + 1 DSA");
        assert_eq!(c.first_dense, 1, "one dense-prefix layer from mlp_layer_types");
        assert_eq!((c.lin_k_heads, c.lin_k_dim, c.lin_conv_k), (2, 4, 3));
        assert_eq!(c.lin_gate_lb, Some(-5.0));
        assert_eq!((c.index_kpool, c.index_kpool_tail), (4, true));
        assert_eq!((c.hc_mult, c.hc_sinkhorn), (4, 20));
        assert!((c.swiglu_limit - 10.0).abs() < 1e-9);
        assert_eq!((c.qk_rope, c.qk_head), (0, 4), "NoPE: qk_head is nope alone");
        assert_eq!(c.kv_row_b(), 0, "no rope slot in the KV cache");
        assert_eq!(c.stop_ids, vec![0, 3]);
        // A flat text-only export parses identically.
        let flat = tiny_glm5next_json()["text_config"].clone();
        let cf = Cfg::from_json(&flat)?;
        assert_eq!(cf.arch, Arch::Glm5Next);
        assert_eq!(cf.full_attn, c.full_attn);
        Ok(())
    }

    #[test]
    fn glm5next_refuses_rope_lanes_and_interleaved_dense() {
        // Same shape as the import-contract refusal tests: extract the message
        // through a match so a wrong acceptance reports *what* was accepted.
        let refusal = |j: &Value| -> String {
            match Cfg::from_json(j) {
                Err(e) => e.to_string(),
                Ok(c) => format!("wrongly accepted as {:?}", c.arch),
            }
        };
        let mut j = tiny_glm5next_json();
        j["text_config"]["qk_rope_head_dim"] = serde_json::json!(4);
        let msg = refusal(&j);
        assert!(msg.contains("NoPE"), "rope lanes must refuse: {msg}");

        let mut j = tiny_glm5next_json();
        j["text_config"]["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "dense", "sparse"]);
        let msg = refusal(&j);
        assert!(msg.contains("dense prefix"), "interleaved dense must refuse: {msg}");

        let mut j = tiny_glm5next_json();
        j["text_config"]["index_topk"] = serde_json::json!(6); // not divisible by kpool 4
        let msg = refusal(&j);
        assert!(msg.contains("divisible"), "ragged k-pool must refuse: {msg}");
    }

    #[test]
    fn glm5next_defaults_match_the_hf_class() -> Result<(), Error> {
        // Drop every field the HF class defaults, keep only the shapes: the
        // parse must land on the class defaults, not zeros.
        let mut j = tiny_glm5next_json()["text_config"].clone();
        if let Some(o) = j.as_object_mut() {
            for k in ["linear_attn_config", "index_kpool", "index_kpool_always_select_tail",
                      "hc_mult", "hc_eps", "hc_sinkhorn_iters", "swiglu_limit", "layer_types"] {
                o.remove(k);
            }
        }
        // The defaulted kpool is 16, so index_topk must be a multiple of it.
        j["index_topk"] = serde_json::json!(32);
        let c = Cfg::from_json(&j)?;
        assert_eq!((c.lin_k_heads, c.lin_k_dim, c.lin_conv_k), (64, 128, 4));
        assert_eq!(c.lin_gate_lb, Some(-5.0));
        assert_eq!(c.index_kpool, 16);
        assert!(c.index_kpool_tail);
        assert_eq!((c.hc_mult, c.hc_sinkhorn), (4, 20));
        assert!((c.swiglu_limit - 10.0).abs() < 1e-9);
        // layer_types fallback: every 4th layer (0-indexed 3) is the DSA layer.
        assert_eq!(c.full_attn, vec![false, false, false, true]);
        // index_topk=8 is not divisible by the defaulted kpool 16 — but the
        // divisibility gate only applies with an indexer configured, which this
        // shape still has, so fix topk for the assertion above to have run.
        Ok(())
    }
}

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
}

/// Parsed `config.json`. Mirrors `Cfg` in `c/glm.c`.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub arch: Arch,
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
    pub lin_k_heads: i64,
    pub lin_v_heads: i64,
    pub lin_k_dim: i64,
    pub lin_v_dim: i64,
    pub lin_conv_k: i64,
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
            // Hybrid families first: "qwen3_5"/"qwen3_next" (and the VL wrapper,
            // whose text stack lives under `text_config`) — checked before the
            // "qwen3" prefix would swallow them into the pure-dense path.
            Some(t) if t.starts_with("qwen3_5") || t.starts_with("qwen3_next") => {
                return Cfg::from_json_hybrid(root.get("text_config").unwrap_or(root))
            }
            Some(t) if t.starts_with("qwen3") => return Cfg::from_json_gqa(root),
            Some(t) if t.starts_with("glm") || t.starts_with("deepseek") => {}
            None => {}
            Some(other) => {
                return Err(Error::Format(format!(
                    "config: model_type \"{other}\" is not supported (glm/deepseek MLA-MoE, qwen3 dense-GQA, or qwen3_5 hybrid)"
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
            Arch::GlmMla => self.kv_lora,
            Arch::DenseGqa | Arch::HybridGdn => self.n_kv_heads * self.head_dim,
        }
    }

    /// Width of one cached row in the second slot: MLA's rope keys, or all GQA
    /// value heads for one position.
    pub fn kv_row_b(&self) -> i64 {
        match self.arch {
            Arch::GlmMla => self.qk_rope,
            Arch::DenseGqa | Arch::HybridGdn => self.n_kv_heads * self.head_dim,
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
}

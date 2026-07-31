//! Model configuration — the Rust equivalent of the C `Cfg` struct and
//! `load_cfg` (`c/glm.c:1258-1308`).
//!
//! Field names and defaults are ported exactly, including the derived
//! `qk_head`/`attn_scale`, the DSA `idx_type` per-layer schedule, and the
//! `CKR` bounds validation from PR #25 (hostile config.json must not pass).

use crate::{Context, Error};
use serde_json::Value;
use std::path::Path;

/// Parsed `config.json`. Mirrors `Cfg` in `c/glm.c`.
#[derive(Clone, Debug)]
pub struct Cfg {
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
    /// Load and validate `<dir>/config.json`.
    pub fn load(dir: &Path) -> Result<Cfg, Error> {
        let path = dir.join("config.json");
        // read through the io_uring lane (no std::fs read path)
        let bytes = peregrine_io::read_file(&path).ctx(|| path.display().to_string())?;
        let root: Value = serde_json::from_slice(&bytes)?;
        Cfg::from_json(&root)
    }

    /// Parse a config from an already-decoded JSON value (used by tests).
    pub fn from_json(root: &Value) -> Result<Cfg, Error> {
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

    #[test]
    fn keeps_every_eos_token() -> Result<(), Error> {
        let mut j = tiny_json();
        j["eos_token_id"] = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let c = Cfg::from_json(&j)?;
        assert_eq!(c.stop_ids.len(), 10, "no stop id may be dropped");
        Ok(())
    }
}

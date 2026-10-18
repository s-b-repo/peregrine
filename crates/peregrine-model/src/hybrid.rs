use std::path::Path;

use peregrine_core::{Compression, Dtype, Error, QtInfo, SafeTensors};

use crate::mlp::Mlp;
use crate::weight::{QtWeight, QuantFmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPolicy {
    QuantPreferred,
    FullPreferred,
    FullOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightPolicyConfig {
    pub default: WeightPolicy,
    pub critical: WeightPolicy,
    pub experts: WeightPolicy,
}

impl Default for WeightPolicyConfig {
    fn default() -> Self {
        Self {
            default: WeightPolicy::QuantPreferred,
            critical: WeightPolicy::FullPreferred,
            experts: WeightPolicy::QuantPreferred,
        }
    }
}

impl WeightPolicyConfig {
    pub fn from_env() -> Self {
        let mut out = Self::default();
        if let Some(v) = std::env::var("COLI_HYBRID_DEFAULT")
            .ok()
            .and_then(|v| parse_policy(&v))
        {
            out.default = v;
        }
        if let Some(v) = std::env::var("COLI_HYBRID_CRITICAL")
            .ok()
            .and_then(|v| parse_policy(&v))
        {
            out.critical = v;
        }
        if let Some(v) = std::env::var("COLI_HYBRID_EXPERTS")
            .ok()
            .and_then(|v| parse_policy(&v))
        {
            out.experts = v;
        }
        out
    }
}

fn parse_policy(value: &str) -> Option<WeightPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "quant" | "quant-preferred" | "q" => Some(WeightPolicy::QuantPreferred),
        "full" | "bf16" | "full-preferred" | "f" => Some(WeightPolicy::FullPreferred),
        "full-only" | "full_only" | "bf16-only" => Some(WeightPolicy::FullOnly),
        _ => None,
    }
}

pub struct HybridWeights {
    quant: SafeTensors,
    full: Option<SafeTensors>,
    policy: WeightPolicyConfig,
}

pub struct WeightResolver<'a> {
    quant: &'a SafeTensors,
    full: Option<&'a SafeTensors>,
    policy: WeightPolicyConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertSource {
    Quant,
    Full,
}

impl HybridWeights {
    pub fn open(quant_dir: &Path, reference_dir: Option<&Path>) -> Result<Self, Error> {
        let quant = SafeTensors::open(quant_dir)?;
        let full = reference_dir.map(SafeTensors::open).transpose()?;
        Ok(Self {
            quant,
            full,
            policy: WeightPolicyConfig::from_env(),
        })
    }

    pub fn resolver(&self) -> WeightResolver<'_> {
        WeightResolver {
            quant: &self.quant,
            full: self.full.as_ref(),
            policy: self.policy,
        }
    }

    pub fn quant(&self) -> &SafeTensors {
        &self.quant
    }

    pub fn full(&self) -> Option<&SafeTensors> {
        self.full.as_ref()
    }

    pub fn policy(&self) -> WeightPolicyConfig {
        self.policy
    }

    pub fn into_parts(self) -> (SafeTensors, Option<SafeTensors>, WeightPolicyConfig) {
        (self.quant, self.full, self.policy)
    }
}

impl<'a> WeightResolver<'a> {
    pub fn new(
        quant: &'a SafeTensors,
        full: Option<&'a SafeTensors>,
        policy: WeightPolicyConfig,
    ) -> Self {
        Self {
            quant,
            full,
            policy,
        }
    }

    pub fn quant(&self) -> &'a SafeTensors {
        self.quant
    }

    pub fn full(&self) -> Option<&'a SafeTensors> {
        self.full
    }

    pub fn policy_for(&self, name: &str) -> WeightPolicy {
        if is_expert(name) {
            self.policy.experts
        } else if is_critical(name) {
            self.policy.critical
        } else {
            self.policy.default
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.quant.has(name) || self.full_name(name).is_some()
    }

    pub fn has_quant_matrix(&self, name: &str, o: usize, i: usize) -> bool {
        quant_matrix(self.quant, name, o, i)
    }

    pub fn has_full_matrix(&self, name: &str, o: usize, i: usize) -> bool {
        self.full_name(name)
            .is_some_and(|n| self.full.is_some_and(|st| full_matrix(st, &n, o, i)))
    }

    pub fn load_matrix(&self, name: &str, o: usize, i: usize) -> Result<QtWeight, Error> {
        self.load_matrix_with_policy(name, o, i, self.policy_for(name))
    }

    pub fn load_matrix_with_policy(
        &self,
        name: &str,
        o: usize,
        i: usize,
        policy: WeightPolicy,
    ) -> Result<QtWeight, Error> {
        let full_name = self.full_name(name);
        let has_quant =
            quant_matrix(self.quant, name, o, i) || native_matrix(self.quant, name, o, i);
        let has_full = full_name
            .as_ref()
            .is_some_and(|n| self.full.is_some_and(|st| full_matrix(st, n, o, i)));
        let use_full = match policy {
            WeightPolicy::QuantPreferred => !has_quant && has_full,
            WeightPolicy::FullPreferred => has_full,
            WeightPolicy::FullOnly => {
                if !has_full {
                    return Err(Error::Format(format!(
                        "full-only tensor '{name}' is missing"
                    )));
                }
                true
            }
        };
        if use_full {
            let n = full_name
                .ok_or_else(|| Error::Format(format!("full tensor '{name}' is missing")))?;
            let st = self
                .full
                .ok_or_else(|| Error::Format(format!("full tensor '{name}' is unavailable")))?;
            return QtWeight::load(st, &n, o, i);
        }
        if quant_matrix(self.quant, name, o, i) {
            return QtWeight::load(self.quant, name, o, i);
        }
        if native_matrix(self.quant, name, o, i) {
            return QtWeight::load(self.quant, name, o, i);
        }
        Err(Error::Format(format!(
            "tensor '{name}' is missing from both weight stores"
        )))
    }

    pub fn read_f32(&self, name: &str, out: &mut [f32]) -> Result<(), Error> {
        let full = self
            .full
            .and_then(|st| self.full_name(name).map(|full_name| (st, full_name)));
        let has_quant = native_values(self.quant, name, out.len());
        let has_full = full
            .as_ref()
            .is_some_and(|(st, full_name)| native_values(st, full_name, out.len()));
        let use_full = match self.policy_for(name) {
            WeightPolicy::QuantPreferred => !has_quant && has_full,
            WeightPolicy::FullPreferred => has_full,
            WeightPolicy::FullOnly => {
                if !has_full {
                    return Err(Error::Format(format!(
                        "full-only tensor '{name}' is missing"
                    )));
                }
                true
            }
        };
        if use_full {
            let (st, full_name) =
                full.ok_or_else(|| Error::Format(format!("full tensor '{name}' is unavailable")))?;
            st.read_f32(&full_name, out).map(|_| ())
        } else {
            self.quant.read_f32(name, out).map(|_| ())
        }
    }

    pub fn expert_source(
        &self,
        layer: usize,
        expert: usize,
        hidden: usize,
        inter: usize,
    ) -> Result<ExpertSource, Error> {
        let p = |suffix: &str| format!("model.layers.{layer}.mlp.experts.{expert}.{suffix}");
        let q = [
            (p("gate_proj.weight"), inter, hidden),
            (p("up_proj.weight"), inter, hidden),
            (p("down_proj.weight"), hidden, inter),
        ];
        let full = q.iter().all(|(n, o, i)| self.has_full_matrix(n, *o, *i));
        let quant = q.iter().all(|(n, o, i)| {
            self.has_quant_matrix(n, *o, *i) || native_matrix(self.quant, n, *o, *i)
        });
        let policy = self.policy.experts;
        match policy {
            WeightPolicy::FullOnly if !full => Err(Error::Format(format!(
                "full-only expert {layer}/{expert} is incomplete"
            ))),
            WeightPolicy::FullOnly => Ok(ExpertSource::Full),
            WeightPolicy::FullPreferred if full => Ok(ExpertSource::Full),
            _ if quant => Ok(ExpertSource::Quant),
            _ if full => Ok(ExpertSource::Full),
            _ => Err(Error::Format(format!(
                "expert {layer}/{expert} is missing from both weight stores"
            ))),
        }
    }

    pub fn load_expert(
        &self,
        layer: usize,
        expert: usize,
        hidden: usize,
        inter: usize,
        limit: f32,
    ) -> Result<Mlp, Error> {
        let p = |suffix: &str| format!("model.layers.{layer}.mlp.experts.{expert}.{suffix}");
        Ok(Mlp {
            gate: self.load_matrix(&p("gate_proj.weight"), inter, hidden)?,
            up: self.load_matrix(&p("up_proj.weight"), inter, hidden)?,
            down: self.load_matrix(&p("down_proj.weight"), hidden, inter)?,
            limit,
        })
    }

    fn full_name(&self, name: &str) -> Option<String> {
        let full = self.full?;
        if full.has(name) {
            return Some(name.to_string());
        }
        let alias = if let Some(rest) = name.strip_prefix("model.layers.") {
            format!("model.language_model.layers.{rest}")
        } else if name == "model.embed_tokens.weight" {
            "model.language_model.embed_tokens.weight".to_string()
        } else if name == "model.norm.weight" {
            "model.language_model.norm.weight".to_string()
        } else {
            return None;
        };
        full.has(&alias).then_some(alias)
    }
}

fn is_expert(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

fn is_critical(name: &str) -> bool {
    if matches!(
        name,
        "model.embed_tokens.weight" | "model.language_model.embed_tokens.weight" | "lm_head.weight"
    ) || name.contains(".mlp.gate.")
        || name.contains(".indexer.")
        || name.contains(".hc_attn")
        || name.contains(".hc_ffn")
    {
        return true;
    }
    name.ends_with(".f_a_proj.weight")
        || name.ends_with(".f_b_proj.weight")
        || name.ends_with(".g_a_proj.weight")
        || name.ends_with(".g_b_proj.weight")
        || name.ends_with(".b_proj.weight")
        || name.contains("layernorm")
        || name.contains(".norm")
        || name.ends_with(".A_log")
        || name.ends_with(".dt_bias")
}

fn quant_matrix(st: &SafeTensors, name: &str, o: usize, i: usize) -> bool {
    st.find(name).is_some_and(|_| {
        QuantFmt::from_qt(QtInfo::detect(st, name, o as i64, i as i64).fmt).is_some()
    })
}

fn native_matrix(st: &SafeTensors, name: &str, o: usize, i: usize) -> bool {
    let Some(t) = st.find(name) else { return false };
    t.shape == [o as i64, i as i64] && native_values(st, name, o.saturating_mul(i))
}

fn native_values(st: &SafeTensors, name: &str, len: usize) -> bool {
    st.find(name).is_some_and(|t| {
        t.numel == len as i64
            && matches!(t.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32)
            && t.compression == Compression::None
            && t.layout.is_none()
    })
}

fn full_matrix(st: &SafeTensors, name: &str, o: usize, i: usize) -> bool {
    native_matrix(st, name, o, i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_core::pack::{f32_bytes, quant_i4, write_safetensors, Blob};

    fn dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("peregrine_hybrid_{}_{}", std::process::id(), tag))
    }

    #[test]
    fn policy_selects_quant_then_full_and_missing_falls_back() -> Result<(), Error> {
        let qdir = dir("q");
        let fdir = dir("f");
        let _ = std::fs::remove_dir_all(&qdir);
        let _ = std::fs::remove_dir_all(&fdir);
        let values: Vec<f32> = (0..6).map(|n| n as f32 * 0.25).collect();
        let (q, scale) = quant_i4(&values, 2, 3);
        write_safetensors(
            &qdir,
            &[
                Blob::new("w", "U8", vec![2, 2], q),
                Blob::new("w.qs", "F32", vec![2], f32_bytes(&scale)),
            ],
        )?;
        let bf16 = |v: &[f32]| {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<u8>>()
        };
        write_safetensors(
            &fdir,
            &[
                Blob::new("w", "BF16", vec![2, 3], bf16(&values)),
                Blob::new("fallback", "BF16", vec![1, 3], bf16(&[1.0, 2.0, 3.0])),
            ],
        )?;
        let stores = HybridWeights::open(&qdir, Some(&fdir))?;
        let (quant, full, _) = stores.into_parts();
        let resolver = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::QuantPreferred,
                critical: WeightPolicy::QuantPreferred,
                experts: WeightPolicy::QuantPreferred,
            },
        );
        assert!(!resolver.load_matrix("w", 2, 3)?.is_full());
        assert!(resolver.load_matrix("fallback", 1, 3)?.is_full());
        let full_resolver = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::FullPreferred,
                critical: WeightPolicy::FullPreferred,
                experts: WeightPolicy::FullPreferred,
            },
        );
        assert!(full_resolver.load_matrix("w", 2, 3)?.is_full());
        std::fs::remove_dir_all(qdir)?;
        std::fs::remove_dir_all(fdir)?;
        Ok(())
    }

    #[test]
    fn scalar_reads_obey_policy_and_full_only_does_not_fall_back() -> Result<(), Error> {
        let qdir = dir("scalar_q");
        let fdir = dir("scalar_f");
        let _ = std::fs::remove_dir_all(&qdir);
        let _ = std::fs::remove_dir_all(&fdir);
        write_safetensors(
            &qdir,
            &[Blob::new("scalar", "F32", vec![2], f32_bytes(&[1.0, 2.0]))],
        )?;
        write_safetensors(
            &fdir,
            &[Blob::new(
                "scalar",
                "BF16",
                vec![2],
                vec![0x40, 0x40, 0x80, 0x40],
            )],
        )?;
        let (quant, full, _) = HybridWeights::open(&qdir, Some(&fdir))?.into_parts();
        let quant_resolver = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::QuantPreferred,
                critical: WeightPolicy::FullPreferred,
                experts: WeightPolicy::QuantPreferred,
            },
        );
        let mut values = [0.0; 2];
        quant_resolver.read_f32("scalar", &mut values)?;
        assert_eq!(values, [1.0, 2.0]);
        let full_resolver = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::FullPreferred,
                critical: WeightPolicy::FullPreferred,
                experts: WeightPolicy::QuantPreferred,
            },
        );
        full_resolver.read_f32("scalar", &mut values)?;
        assert_eq!(values, [3.0, 4.0]);
        let full_only = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::FullOnly,
                critical: WeightPolicy::FullOnly,
                experts: WeightPolicy::QuantPreferred,
            },
        );
        assert!(full_only.read_f32("missing", &mut values).is_err());
        assert!(is_critical("model.language_model.embed_tokens.weight"));
        std::fs::remove_dir_all(qdir)?;
        std::fs::remove_dir_all(fdir)?;
        Ok(())
    }

    #[test]
    fn full_only_experts_never_select_quantized_weights() -> Result<(), Error> {
        let qdir = dir("expert_q");
        let fdir = dir("expert_f");
        let _ = std::fs::remove_dir_all(&qdir);
        let _ = std::fs::remove_dir_all(&fdir);
        let gate = [1.0, 2.0, 3.0];
        let up = [4.0, 5.0, 6.0];
        let down = [7.0, 8.0, 9.0];
        let (gate_q, gate_s) = quant_i4(&gate, 1, 3);
        let (up_q, up_s) = quant_i4(&up, 1, 3);
        let (down_q, down_s) = quant_i4(&down, 3, 1);
        let gate_name = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let up_name = "model.layers.0.mlp.experts.0.up_proj.weight";
        let down_name = "model.layers.0.mlp.experts.0.down_proj.weight";
        write_safetensors(
            &qdir,
            &[
                Blob::new(gate_name, "U8", vec![1, 2], gate_q),
                Blob::new(
                    &format!("{gate_name}.qs"),
                    "F32",
                    vec![1],
                    f32_bytes(&gate_s),
                ),
                Blob::new(up_name, "U8", vec![1, 2], up_q),
                Blob::new(&format!("{up_name}.qs"), "F32", vec![1], f32_bytes(&up_s)),
                Blob::new(down_name, "U8", vec![3, 1], down_q),
                Blob::new(
                    &format!("{down_name}.qs"),
                    "F32",
                    vec![3],
                    f32_bytes(&down_s),
                ),
            ],
        )?;
        let bf16 = |v: &[f32]| {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<u8>>()
        };
        write_safetensors(
            &fdir,
            &[
                Blob::new(gate_name, "BF16", vec![1, 3], bf16(&gate)),
                Blob::new(up_name, "BF16", vec![1, 3], bf16(&up)),
                Blob::new(down_name, "BF16", vec![3, 1], bf16(&down)),
            ],
        )?;
        let (quant, full, _) = HybridWeights::open(&qdir, Some(&fdir))?.into_parts();
        let resolver = WeightResolver::new(
            &quant,
            full.as_ref(),
            WeightPolicyConfig {
                default: WeightPolicy::QuantPreferred,
                critical: WeightPolicy::QuantPreferred,
                experts: WeightPolicy::FullOnly,
            },
        );
        assert_eq!(resolver.expert_source(0, 0, 3, 1)?, ExpertSource::Full);
        assert!(resolver.load_expert(0, 0, 3, 1, 10.0)?.gate.is_full());
        std::fs::remove_dir_all(qdir)?;
        std::fs::remove_dir_all(fdir)?;
        Ok(())
    }
}

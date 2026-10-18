use std::collections::{HashMap, HashSet};

use peregrine_core::Error;

use crate::predict::{PredictSource, RouteHistory};

const MAX_CONTEXT: usize = 2;
const MAX_LAYERS: usize = 128;
const MAX_ENTRIES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FuturePrefetchMode {
    #[default]
    Off,
    History,
    Learned,
    Both,
}

impl FuturePrefetchMode {
    pub fn from_env_value(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim).unwrap_or("off") {
            "" | "0" | "off" | "false" => Ok(Self::Off),
            "history" => Ok(Self::History),
            "learned" => Ok(Self::Learned),
            "both" => Ok(Self::Both),
            other => Err(format!(
                "unknown future-prefetch mode {other:?}; use off|history|learned|both"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::History => "history",
            Self::Learned => "learned",
            Self::Both => "both",
        }
    }

    pub fn enabled(self) -> bool {
        self != Self::Off
    }

    pub fn uses_history(self) -> bool {
        matches!(self, Self::History | Self::Both)
    }

    pub fn uses_learned(self) -> bool {
        matches!(self, Self::Learned | Self::Both)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuturePrefetchSettings {
    pub mode: FuturePrefetchMode,
    pub width: usize,
    pub max_pending: usize,
}

impl Default for FuturePrefetchSettings {
    fn default() -> Self {
        Self {
            mode: FuturePrefetchMode::Off,
            width: 8,
            max_pending: 1024,
        }
    }
}

impl FuturePrefetchSettings {
    pub fn from_env() -> Result<Self, String> {
        Self::from_values(
            std::env::var("COLI_FUTURE_PREFETCH").ok().as_deref(),
            std::env::var("COLI_FUTURE_PREFETCH_WIDTH").ok().as_deref(),
            std::env::var("COLI_FUTURE_PREFETCH_MAX_PENDING")
                .ok()
                .as_deref(),
        )
    }

    pub fn from_values(
        mode: Option<&str>,
        width: Option<&str>,
        max_pending: Option<&str>,
    ) -> Result<Self, String> {
        let width = width
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|_| "COLI_FUTURE_PREFETCH_WIDTH must be a positive integer".to_string())?
            .unwrap_or(8);
        if width == 0 {
            return Err("COLI_FUTURE_PREFETCH_WIDTH must be greater than zero".into());
        }
        let max_pending = max_pending
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|_| "COLI_FUTURE_PREFETCH_MAX_PENDING must be an integer".to_string())?
            .unwrap_or(1024)
            .clamp(1, 65_536);
        Ok(Self {
            mode: FuturePrefetchMode::from_env_value(mode)?,
            width,
            max_pending,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenRouteTable {
    tag: String,
    layers: Vec<HashMap<Vec<i32>, HashMap<u32, u32>>>,
    observations: u64,
}

impl TokenRouteTable {
    pub fn new(n_layers: usize, tag: String) -> Self {
        Self {
            tag,
            layers: (0..n_layers).map(|_| HashMap::new()).collect(),
            observations: 0,
        }
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn entries(&self) -> usize {
        self.layers.iter().map(HashMap::len).sum()
    }

    pub fn observe(&mut self, layer: usize, context: &[i32], experts: &[i32]) {
        self.observe_with_limit(layer, context, experts, MAX_ENTRIES);
    }

    fn observe_with_limit(
        &mut self,
        layer: usize,
        context: &[i32],
        experts: &[i32],
        max_entries: usize,
    ) {
        if context.is_empty() || experts.is_empty() {
            return;
        }
        let total_entries = self.entries();
        if total_entries >= max_entries {
            return;
        }
        let Some(rows) = self.layers.get_mut(layer) else {
            return;
        };
        let start = context.len().saturating_sub(MAX_CONTEXT);
        let available = context.len() - start;
        let mut remaining = max_entries.saturating_sub(total_entries);
        for len in 1..=available.min(MAX_CONTEXT) {
            let key = context[context.len() - len..].to_vec();
            let is_new = !rows.contains_key(&key);
            if is_new && remaining == 0 {
                continue;
            }
            if is_new {
                remaining -= 1;
            }
            let counts = rows.entry(key).or_default();
            for &expert in experts {
                if expert < 0 {
                    continue;
                }
                let count = counts.entry(expert as u32).or_insert(0);
                *count = count.saturating_add(1);
            }
        }
        self.observations = self.observations.saturating_add(1);
    }

    pub fn predict_layer(&self, layer: usize, context: &[i32], width: usize) -> Vec<(u32, u32)> {
        if width == 0 || context.is_empty() {
            return Vec::new();
        }
        let Some(rows) = self.layers.get(layer) else {
            return Vec::new();
        };
        let available = context.len().min(MAX_CONTEXT);
        for len in (1..=available).rev() {
            let start = context.len() - len;
            let Some(counts) = rows.get(&context[start..]) else {
                continue;
            };
            let mut ranked: Vec<(u32, u32)> = counts
                .iter()
                .map(|(&expert, &count)| (expert, count))
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            ranked.truncate(width);
            return ranked;
        }
        Vec::new()
    }

    pub fn to_json(&self) -> serde_json::Value {
        let layers: Vec<serde_json::Value> = self
            .layers
            .iter()
            .map(|rows| {
                let mut entries: Vec<(&Vec<i32>, &HashMap<u32, u32>)> = rows.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                let rows: Vec<serde_json::Value> = entries
                    .into_iter()
                    .map(|(context, counts)| {
                        let mut experts: Vec<(&u32, &u32)> = counts.iter().collect();
                        experts.sort_by_key(|&(expert, _)| *expert);
                        let experts: Vec<serde_json::Value> = experts
                            .into_iter()
                            .map(|(e, c)| serde_json::json!([e, c]))
                            .collect();
                        serde_json::json!([context, experts])
                    })
                    .collect();
                serde_json::Value::Array(rows)
            })
            .collect();
        serde_json::json!({
            "version": 1,
            "tag": self.tag,
            "context_len": MAX_CONTEXT,
            "n_layers": self.layers.len(),
            "observations": self.observations,
            "layers": layers,
        })
    }

    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        Self::from_json_inner(value, None)
    }

    pub fn from_json_with_shape(
        value: &serde_json::Value,
        expected_layers: usize,
        expected_experts: usize,
    ) -> Option<Self> {
        if expected_layers == 0 {
            return None;
        }
        Self::from_json_inner(value, Some((expected_layers, expected_experts)))
    }

    fn from_json_inner(value: &serde_json::Value, shape: Option<(usize, usize)>) -> Option<Self> {
        if value.get("version")?.as_u64()? != 1 {
            return None;
        }
        let tag = value.get("tag")?.as_str()?.to_string();
        let n_layers = usize::try_from(value.get("n_layers")?.as_u64()?).ok()?;
        if let Some((expected_layers, _)) = shape {
            if expected_layers == 0 || n_layers != expected_layers {
                return None;
            }
        }
        let context_len = usize::try_from(value.get("context_len")?.as_u64()?).ok()?;
        if n_layers == 0 || n_layers > MAX_LAYERS || context_len == 0 || context_len > MAX_CONTEXT {
            return None;
        }
        let layer_values = value.get("layers")?.as_array()?;
        if layer_values.len() != n_layers {
            return None;
        }
        let observations = value
            .get("observations")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let mut table = Self::new(n_layers, tag);
        let mut total = 0usize;
        for (layer, rows) in layer_values.iter().enumerate() {
            let rows = rows.as_array()?;
            total = total.saturating_add(rows.len());
            if total > MAX_ENTRIES {
                return None;
            }
            for row in rows {
                let row = row.as_array()?;
                let context = row.first()?.as_array()?;
                if context.is_empty() || context.len() > context_len {
                    return None;
                }
                let context: Vec<i32> = context
                    .iter()
                    .map(|value| i32::try_from(value.as_i64()?).ok())
                    .collect::<Option<Vec<_>>>()?;
                let experts = row.get(1)?.as_array()?;
                let counts = table.layers[layer].entry(context).or_default();
                for expert in experts {
                    let pair = expert.as_array()?;
                    let expert = u32::try_from(pair.first()?.as_u64()?).ok()?;
                    if let Some((_, expected_experts)) = shape {
                        if u64::try_from(expected_experts)
                            .map_or(true, |limit| u64::from(expert) >= limit)
                        {
                            return None;
                        }
                    }
                    let count = u32::try_from(pair.get(1)?.as_u64()?).ok()?;
                    let slot = counts.entry(expert).or_insert(0);
                    *slot = slot.saturating_add(count);
                }
            }
        }
        table.observations = observations;
        Some(table)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Error> {
        let bytes = serde_json::to_vec_pretty(&self.to_json())
            .map_err(|error| Error::Format(format!("serialize token route table: {error}")))?;
        peregrine_core::write_atomic(path, &bytes)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FutureRoutePlan {
    pub layers: Vec<Vec<u32>>,
    pub history_hits: u64,
    pub learned_hits: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn plan_future_routes(
    mode: FuturePrefetchMode,
    predictor: &PredictSource,
    learned: Option<&TokenRouteTable>,
    history: &RouteHistory,
    committed: &[i32],
    pending: i32,
    draft_paths: &[Vec<i32>],
    first_layer: usize,
    n_layers: usize,
    topk: usize,
    max_per_layer: usize,
) -> FutureRoutePlan {
    if !mode.enabled() || draft_paths.iter().all(Vec::is_empty) || n_layers <= first_layer {
        return FutureRoutePlan::default();
    }
    let topk = topk.max(1);
    let width = max_per_layer.max(1);
    let mut out = FutureRoutePlan {
        layers: vec![Vec::new(); n_layers],
        ..FutureRoutePlan::default()
    };
    let mut scores: Vec<HashMap<u32, u32>> =
        (first_layer..n_layers).map(|_| HashMap::new()).collect();
    let mut locals: Vec<RouteHistory> = draft_paths
        .iter()
        .map(|_| clone_history(history, n_layers))
        .collect();
    let mut seen_prefixes = HashSet::new();
    for (path_index, path) in draft_paths.iter().enumerate() {
        let mut context = tail(committed, MAX_CONTEXT);
        let mut prefix = Vec::with_capacity(path.len() + 1);
        prefix.push(pending);
        let mut targets = Vec::with_capacity(path.len() + 1);
        targets.push(pending);
        targets.extend_from_slice(path);
        for (target_index, target) in targets.into_iter().enumerate() {
            if target_index > 0 {
                prefix.push(target);
            }
            let first_prefix = seen_prefixes.insert(prefix.clone());
            for layer in first_layer..n_layers {
                let mut selected = Vec::new();
                if mode.uses_history() {
                    let ranked = predictor.predict_layer(layer, &locals[path_index]);
                    if first_prefix && !ranked.is_empty() {
                        out.history_hits = out.history_hits.saturating_add(1);
                    }
                    for &(expert, _) in ranked.iter().take(topk) {
                        if let Ok(expert) = i32::try_from(expert) {
                            selected.push(expert);
                            if first_prefix {
                                let score = scores[layer - first_layer]
                                    .entry(expert as u32)
                                    .or_insert(0);
                                *score = score.saturating_add(1);
                            }
                        }
                    }
                    locals[path_index].push_layer(layer, selected.clone());
                }
                if mode.uses_learned() {
                    if let Some(table) = learned {
                        let ranked = table.predict_layer(layer, &context, topk);
                        if first_prefix && !ranked.is_empty() {
                            out.learned_hits = out.learned_hits.saturating_add(1);
                        }
                        for &(expert, _) in &ranked {
                            let Ok(expert_i32) = i32::try_from(expert) else {
                                continue;
                            };
                            selected.push(expert_i32);
                            if first_prefix {
                                let score = scores[layer - first_layer].entry(expert).or_insert(0);
                                *score = score.saturating_add(1);
                            }
                        }
                    }
                }
            }
            context.push(target);
            if context.len() > MAX_CONTEXT {
                context.drain(..context.len() - MAX_CONTEXT);
            }
        }
    }
    for (offset, layer_scores) in scores.into_iter().enumerate() {
        let mut ranked: Vec<(u32, u32)> = layer_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(width);
        out.layers[first_layer + offset] = ranked.into_iter().map(|(expert, _)| expert).collect();
    }
    out
}

fn clone_history(history: &RouteHistory, n_layers: usize) -> RouteHistory {
    let mut out = RouteHistory::new(n_layers, history.depth());
    for layer in 0..n_layers {
        let frames: Vec<Vec<i32>> = history.frames(layer).cloned().collect();
        for frame in frames.into_iter().rev() {
            out.push_layer(layer, frame);
        }
    }
    out
}

fn tail(tokens: &[i32], width: usize) -> Vec<i32> {
    tokens
        .iter()
        .copied()
        .rev()
        .take(width)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::{Momentum, TransitionTable};

    #[test]
    fn modes_parse_without_latching() {
        assert_eq!(
            FuturePrefetchMode::from_env_value(None),
            Ok(FuturePrefetchMode::Off)
        );
        assert_eq!(
            FuturePrefetchMode::from_env_value(Some("history")),
            Ok(FuturePrefetchMode::History)
        );
        assert_eq!(
            FuturePrefetchMode::from_env_value(Some("learned")),
            Ok(FuturePrefetchMode::Learned)
        );
        assert_eq!(
            FuturePrefetchMode::from_env_value(Some("both")),
            Ok(FuturePrefetchMode::Both)
        );
        assert!(FuturePrefetchMode::from_env_value(Some("wat")).is_err());
    }

    #[test]
    fn settings_validate_and_clamp_shared_limits() {
        assert_eq!(
            FuturePrefetchSettings::from_values(Some("both"), Some(" 17 "), Some("65537")),
            Ok(FuturePrefetchSettings {
                mode: FuturePrefetchMode::Both,
                width: 17,
                max_pending: 65_536
            })
        );
        assert_eq!(
            FuturePrefetchSettings::from_values(Some("history"), Some("0"), Some("8")),
            Err("COLI_FUTURE_PREFETCH_WIDTH must be greater than zero".into())
        );
        assert!(FuturePrefetchSettings::from_values(Some("bad"), None, None).is_err());
        assert_eq!(
            FuturePrefetchSettings::from_values(None, Some("wat"), Some("0")),
            Err("COLI_FUTURE_PREFETCH_WIDTH must be a positive integer".into())
        );
        assert_eq!(
            FuturePrefetchSettings::from_values(None, None, Some("0")),
            Ok(FuturePrefetchSettings {
                mode: FuturePrefetchMode::Off,
                width: 8,
                max_pending: 1
            })
        );
    }

    #[test]
    fn learned_table_prefers_the_longest_context_and_falls_back() {
        let mut table = TokenRouteTable::new(1, "tag".into());
        table.observe(0, &[3], &[1, 2]);
        table.observe(0, &[3, 4], &[5, 6]);
        table.observe(0, &[3, 4], &[5, 6]);
        assert_eq!(table.predict_layer(0, &[3, 4], 2), vec![(5, 2), (6, 2)]);
        assert_eq!(table.predict_layer(0, &[3], 2), vec![(1, 1), (2, 1)]);
        assert!(table.predict_layer(0, &[9], 2).is_empty());
    }

    #[test]
    fn learned_table_round_trips() {
        let mut table = TokenRouteTable::new(2, "cfg".into());
        table.observe(0, &[1, 2], &[3, 4]);
        table.observe(1, &[2], &[5]);
        let back = TokenRouteTable::from_json(&table.to_json());
        assert_eq!(back, Some(table));
    }

    #[test]
    fn malformed_artifacts_are_rejected() {
        let mut table = TokenRouteTable::new(1, "cfg".into());
        table.observe(0, &[1], &[2]);
        let mut value = table.to_json();
        value["context_len"] = serde_json::json!(0);
        assert!(TokenRouteTable::from_json(&value).is_none());
        let mut value = table.to_json();
        value["layers"][0][0][0] = serde_json::json!([1, 2, 3]);
        assert!(TokenRouteTable::from_json(&value).is_none());
    }

    #[test]
    fn observe_caps_rows_across_layers() {
        let mut table = TokenRouteTable::new(2, "cfg".into());
        table.observe_with_limit(0, &[1], &[2], 2);
        table.observe_with_limit(0, &[3, 4], &[5], 2);
        assert_eq!(table.entries(), 2);
        table.observe_with_limit(1, &[6], &[7], 2);
        assert_eq!(table.entries(), 2);
        assert_eq!(table.observations(), 2);
    }

    #[test]
    fn shaped_parse_rejects_invalid_model_shapes() {
        let mut table = TokenRouteTable::new(1, "cfg".into());
        table.observe(0, &[1], &[1]);
        let value = table.to_json();
        assert_eq!(
            TokenRouteTable::from_json_with_shape(&value, 1, 2),
            Some(table.clone())
        );
        assert!(TokenRouteTable::from_json_with_shape(&value, 0, 2).is_none());
        assert!(TokenRouteTable::from_json_with_shape(&value, 2, 2).is_none());
        let mut invalid_expert = value.clone();
        invalid_expert["layers"][0][0][1][0][0] = serde_json::json!(2);
        assert!(TokenRouteTable::from_json_with_shape(&invalid_expert, 1, 2).is_none());
        assert!(TokenRouteTable::from_json(&invalid_expert).is_some());
        let empty = serde_json::json!({
            "version": 1,
            "tag": "cfg",
            "n_layers": 0,
            "context_len": 2,
            "layers": [],
        });
        assert!(TokenRouteTable::from_json_with_shape(&empty, 0, 2).is_none());
        assert!(TokenRouteTable::from_json_with_shape(&empty, 1, 2).is_none());
    }

    #[test]
    fn history_planner_unions_successive_route_states() {
        let mut transitions = TransitionTable::new(1, "t".into());
        for _ in 0..3 {
            transitions.observe(0, &[1], &[2, 3]);
            transitions.observe(0, &[2, 3], &[3, 4]);
        }
        let predictor = PredictSource::Automaton {
            table: std::sync::Arc::new(transitions),
            fallback: Momentum::default(),
        };
        let mut history = RouteHistory::new(1, 4);
        history.push_layer(0, vec![1]);
        let plan = plan_future_routes(
            FuturePrefetchMode::History,
            &predictor,
            None,
            &history,
            &[8],
            9,
            &[vec![10, 11]],
            0,
            1,
            2,
            8,
        );
        let mut experts = plan.layers[0].clone();
        experts.sort_unstable();
        assert_eq!(experts, vec![2, 3, 4]);
        assert!(plan.history_hits >= 3);
    }

    #[test]
    fn shared_draft_prefixes_are_scored_once() {
        let mut learned = TokenRouteTable::new(1, "t".into());
        learned.layers[0].insert(vec![0], std::collections::HashMap::from([(3, 1)]));
        learned.layers[0].insert(vec![1, 2], std::collections::HashMap::from([(4, 1)]));
        learned.layers[0].insert(vec![1, 3], std::collections::HashMap::from([(0, 1)]));
        assert_eq!(learned.predict_layer(0, &[0], 1), vec![(3, 1)]);
        assert_eq!(learned.predict_layer(0, &[1, 2], 1), vec![(4, 1)]);
        assert_eq!(learned.predict_layer(0, &[1, 3], 1), vec![(0, 1)]);
        let history = RouteHistory::new(1, 1);
        let plan = plan_future_routes(
            FuturePrefetchMode::Learned,
            &PredictSource::default(),
            Some(&learned),
            &history,
            &[],
            0,
            &[vec![1, 2, 9], vec![1, 3, 9]],
            0,
            1,
            1,
            1,
        );
        assert_eq!(plan.layers[0], vec![0]);
        assert_eq!(plan.learned_hits, 3);
    }

    #[test]
    fn learned_and_history_modes_merge_deduplicated_candidates() {
        let mut transitions = TransitionTable::new(1, "t".into());
        transitions.observe(0, &[1], &[2, 3]);
        let predictor = PredictSource::Automaton {
            table: std::sync::Arc::new(transitions),
            fallback: Momentum::default(),
        };
        let mut learned = TokenRouteTable::new(1, "t".into());
        learned.observe(0, &[8], &[3, 4]);
        let mut history = RouteHistory::new(1, 1);
        history.push_layer(0, vec![1]);
        let plan = plan_future_routes(
            FuturePrefetchMode::Both,
            &predictor,
            Some(&learned),
            &history,
            &[8],
            9,
            &[vec![10]],
            0,
            1,
            2,
            8,
        );
        let mut experts = plan.layers[0].clone();
        experts.sort_unstable();
        assert_eq!(experts, vec![1, 2, 3, 4]);
        assert!(plan.history_hits > 0);
        assert!(plan.learned_hits > 0);
    }
}

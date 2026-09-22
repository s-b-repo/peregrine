//! L3 — the execution layer and `C_t`.
//!
//! Derivations are registered closures. Results are memoised on a key built
//! from the derivation name, its literal inputs, *and* the fingerprints of the
//! nodes it depends on — so when an upstream fact is corrected the key changes,
//! the memo misses, and the value is recomputed rather than silently served
//! stale. That is the whole point of keying on provenance instead of on the
//! question text.

use std::collections::HashMap;

use crate::graph::{DcrError, MemoryGraph};
use crate::ids::digest;
use crate::nodes::{Derivation, Kind, NewNode, Node, NodeMeta, Origin};

pub type Inputs = Vec<(String, f64)>;
pub type DerivationFn = Box<dyn Fn(&Inputs) -> f64>;

#[derive(Debug, Clone)]
pub struct MemoEntry {
    pub key: String,
    pub value: f64,
    pub inputs: Inputs,
    pub deps: Vec<String>,
    pub hits: u32,
}

#[derive(Default)]
pub struct ExecutionLayer {
    derivations: HashMap<String, DerivationFn>,
    memo: HashMap<String, MemoEntry>,
    pub executions: u32,
    pub memo_hits: u32,
}

impl std::fmt::Debug for ExecutionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionLayer")
            .field("derivations", &self.derivation_names())
            .field("memo_entries", &self.memo.len())
            .field("executions", &self.executions)
            .field("memo_hits", &self.memo_hits)
            .finish()
    }
}

impl ExecutionLayer {
    pub fn register(&mut self, name: &str, f: impl Fn(&Inputs) -> f64 + 'static) {
        self.derivations.insert(name.to_string(), Box::new(f));
    }

    pub fn derivation_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.derivations.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Memo key: the derivation, its literal inputs, and the *content* of every
    /// node it depends on.
    pub fn key_for(
        &self,
        graph: &MemoryGraph,
        name: &str,
        inputs: &Inputs,
        deps: &[String],
    ) -> String {
        let mut sorted_inputs = inputs.clone();
        sorted_inputs.sort_by(|a, b| a.0.cmp(&b.0));
        let inputs_repr = sorted_inputs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        let mut sorted_deps: Vec<&String> = deps.iter().collect();
        sorted_deps.sort();
        let fingerprints = sorted_deps
            .iter()
            .map(|d| match graph.get(d) {
                Some(node) => node.fingerprint(),
                None => (*d).clone(),
            })
            .collect::<Vec<_>>()
            .join(",");
        digest(&[name, &inputs_repr, &fingerprints], 16)
    }

    pub fn call(
        &mut self,
        graph: &MemoryGraph,
        name: &str,
        inputs: &Inputs,
        deps: &[String],
    ) -> Result<f64, DcrError> {
        if !self.derivations.contains_key(name) {
            return Err(DcrError::UnknownDerivation(name.to_string()));
        }
        let key = self.key_for(graph, name, inputs, deps);
        if let Some(entry) = self.memo.get_mut(&key) {
            entry.hits += 1;
            self.memo_hits += 1;
            return Ok(entry.value);
        }
        let value = (self.derivations[name])(inputs);
        self.memo.insert(
            key.clone(),
            MemoEntry {
                key,
                value,
                inputs: inputs.clone(),
                deps: deps.to_vec(),
                hits: 0,
            },
        );
        self.executions += 1;
        Ok(value)
    }

    /// Execute the derivation attached to a node and cache the result at L3.
    pub fn run(&mut self, graph: &MemoryGraph, node: &Node) -> Result<Option<f64>, DcrError> {
        let Some(derivation) = node.meta.derivation.clone() else {
            return Ok(None);
        };
        let value = self.call(
            graph,
            &derivation.name,
            &derivation.inputs,
            &node.dependencies,
        )?;
        node.level_cache.borrow_mut().l3 = Some(value);
        Ok(Some(value))
    }

    /// Run a derivation and materialise the result as a calculation node.
    pub fn compute_node(
        &mut self,
        graph: &MemoryGraph,
        name: &str,
        inputs: Inputs,
        deps: Vec<String>,
        key: Option<String>,
    ) -> Result<Node, DcrError> {
        let value = self.call(graph, name, &inputs, &deps)?;
        let node = NewNode::new(Kind::Calculation, crate::ladder::format_number(value))
            // Not read from anywhere — produced by running a derivation. The
            // window labels it as such so a computed value is never mistaken
            // for one the source material actually stated.
            .origin(Origin::Computed)
            .deps(deps)
            .key(key.unwrap_or_else(|| name.to_string()))
            .meta(NodeMeta {
                derivation: Some(Derivation {
                    name: name.to_string(),
                    inputs,
                }),
                ..Default::default()
            })
            .build();
        node.level_cache.borrow_mut().l3 = Some(value);
        Ok(node)
    }

    pub fn memo_len(&self) -> usize {
        self.memo.len()
    }

    pub fn stats(&self) -> String {
        format!(
            "derivations: {:?}, memo_entries: {}, executions: {}, memo_hits: {}",
            self.derivation_names(),
            self.memo.len(),
            self.executions,
            self.memo_hits
        )
    }
}

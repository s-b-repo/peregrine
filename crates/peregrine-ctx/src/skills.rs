//! Skill loading — admission of a capability as the trigger to load it.
//!
//! A [`Kind::Skill`] node names a *capability* rather than a fact: its value is
//! a description the planner can match a query against, and `meta.source`
//! carries the content-addressed artifact (a LoRA adapter hash, a URI) that a
//! serving stack would load. The planner treats the node like any other
//! candidate; what makes a skill different happens after the knapsack — an
//! admitted skill fires [`SkillLoader::activate`] once per turn.
//!
//! No serving stack is assumed. [`RecordingLoader`] is inert and records, so
//! the admission → activation path is testable offline; a real backend
//! (peregrine, vLLM multi-LoRA) implements the same two methods.

use crate::graph::DcrError;
use crate::nodes::{Kind, Node};

/// The loadable artifact a skill node points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRef {
    /// Skill name — the node's key (`code-review`).
    pub name: String,
    /// Content-addressed artifact reference (`sha256:…`, a URI).
    pub adapter: String,
    /// What the skill does — the text similarity matched against.
    pub description: String,
}

impl SkillRef {
    /// The skill a node names, if it is a well-formed skill node. A missing
    /// artifact reference is `None`: the graph rejects such nodes at insert,
    /// so `None` here means the node is not a skill at all.
    pub fn from_node(node: &Node) -> Option<SkillRef> {
        if node.kind != Kind::Skill {
            return None;
        }
        Some(SkillRef {
            name: node.key.clone().unwrap_or_default(),
            adapter: node.meta.source.clone()?,
            description: node.value.clone(),
        })
    }
}

/// The runtime's hook into a serving stack. Called once per turn per admitted
/// skill, after the final plan of the turn is known.
pub trait SkillLoader: Send {
    /// Bring the artifact online (load an adapter, warm an endpoint). Must be
    /// idempotent: a skill admitted on consecutive turns is activated each turn.
    fn activate(&mut self, skill: &SkillRef) -> Result<(), DcrError>;
    /// References of everything currently loaded, for telemetry and display.
    fn active(&self) -> Vec<String>;
}

/// Inert default: loads nothing, records everything. This is what makes the
/// admission → activation path observable in tests and benches without a
/// model server, and what a CLI run reports through `Answer::skills`.
#[derive(Debug, Default)]
pub struct RecordingLoader {
    /// Adapter references in first-activation order, deduplicated.
    loaded: Vec<String>,
    /// Every activation, including repeats — the per-turn firing record.
    pub activations: Vec<String>,
}

impl RecordingLoader {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SkillLoader for RecordingLoader {
    fn activate(&mut self, skill: &SkillRef) -> Result<(), DcrError> {
        self.activations.push(skill.adapter.clone());
        if !self.loaded.contains(&skill.adapter) {
            self.loaded.push(skill.adapter.clone());
        }
        Ok(())
    }

    fn active(&self) -> Vec<String> {
        self.loaded.clone()
    }
}

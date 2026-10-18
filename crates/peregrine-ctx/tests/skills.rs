//! Skill nodes: the planner admits capabilities, the runtime activates them.
//!
//! A skill is the first node kind whose admission has a side effect beyond the
//! window — it fires the loader. These pin the admission → activation path,
//! the grounding rule for capabilities, and the two escalation edge cases a
//! level-less node introduces.

use std::path::PathBuf;

use peregrine_ctx::graph::DcrError;
use peregrine_ctx::nodes::Kind;
use peregrine_ctx::runtime::Dcr;

const ADAPTER: &str = "sha256:0123456789abcdef";

fn skill_runtime() -> Dcr {
    let mut rt = Dcr::new(800);
    rt.add_skill(
        "code-review",
        "reviews code diffs for correctness and style",
        ADAPTER,
    )
    .expect("add_skill");
    rt
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dcr-skills-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn add_skill_registers_a_retrievable_node() {
    let rt = skill_runtime();
    let skills: Vec<_> = rt
        .graph
        .nodes()
        .iter()
        .filter(|n| n.kind == Kind::Skill)
        .collect();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].key.as_deref(), Some("code-review"));
    assert_eq!(skills[0].meta.source.as_deref(), Some(ADAPTER));
}

#[test]
fn a_skill_without_an_artifact_reference_is_rejected() {
    let mut rt = Dcr::new(800);
    let result = rt.add_skill("groundless", "names nothing loadable", "  ");
    assert!(
        matches!(result, Err(DcrError::Provenance(_))),
        "a capability with no content-addressed adapter must not enter the graph: {result:?}"
    );
}

#[test]
fn a_matching_query_admits_the_skill_under_its_own_group() {
    let mut rt = skill_runtime();
    let context = rt.plan("code review", None);
    assert!(
        context.entries.iter().any(|e| e.kind == Kind::Skill),
        "expected the skill in the plan: {:?}",
        context.node_ids()
    );
    for rendered in [context.render(), context.render_prefix_stable()] {
        let group = rendered.find("## SKILLS (activated)");
        let entry = rendered.find("code-review");
        assert!(
            group.is_some() && entry.is_some() && group < entry,
            "skill must render under its group: {rendered}"
        );
    }
}

#[test]
fn admission_fires_the_loader_once_per_turn() {
    let mut rt = skill_runtime();
    let answer = rt.ask("code review", None);
    assert_eq!(answer.skills, vec!["code-review".to_string()]);
    assert_eq!(
        rt.skills.active(),
        vec![ADAPTER.to_string()],
        "the loader saw exactly the admitted artifact"
    );
}

#[test]
fn an_irrelevant_query_neither_admits_nor_activates() {
    let mut rt = skill_runtime();
    // Competing content the query actually matches, so the seed floor drops
    // the skill rather than the skill winning an empty field.
    rt.ingest("The server ip is 10.0.4.12 and the port is 8080.", Some("ops"))
        .unwrap();
    let answer = rt.ask("what is the server ip?", None);
    assert!(answer.skills.is_empty(), "activated: {:?}", answer.skills);
    assert!(rt.skills.active().is_empty());
    assert!(
        !answer
            .context
            .entries
            .iter()
            .any(|e| e.kind == Kind::Skill),
        "skill admitted by an ops query: {:?}",
        answer.context.node_ids()
    );
}

/// A skill has no richer level than L2. The model can still `#ESCALATE` one —
/// the re-plan must hold it at L2 rather than evict it for lacking L0, and
/// the runtime's own escalation proxy must not target it at all.
#[test]
fn escalation_cannot_evict_a_skill() {
    let mut rt = skill_runtime();
    rt.auto_escalate = true;
    // Partial overlap: thin enough that the escalation machinery would fire on
    // an ordinary node.
    let answer = rt.ask("review checklist", None);
    assert!(
        answer.context.entries.iter().any(|e| e.kind == Kind::Skill),
        "escalation dropped the skill from the final plan: {:?}",
        answer.context.node_ids()
    );
    assert_eq!(answer.skills, vec!["code-review".to_string()]);
    // Once, for the final plan — not once per re-plan.
    assert_eq!(rt.skills.active().len(), 1);
}

#[test]
fn skill_nodes_survive_a_store_round_trip() {
    let dir = scratch("persist");
    let path = dir.join("store.dcr.json");
    let rt = skill_runtime();
    rt.save(&path).expect("save");
    let loaded = Dcr::load(&path, 800).expect("load");
    let skills: Vec<_> = loaded
        .graph
        .nodes()
        .iter()
        .filter(|n| n.kind == Kind::Skill)
        .collect();
    assert_eq!(skills.len(), 1, "kind did not round-trip");
    assert_eq!(
        skills[0].meta.source.as_deref(),
        Some(ADAPTER),
        "the adapter reference is the skill's grounding; losing it orphans the node"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

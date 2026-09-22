//! DCR — the runtime facade.
//!
//! Wires the components into the per-turn data flow from the architecture doc:
//!
//! ```text
//! event → index → graph update → plan(k + r) → model → state update → prefetch
//! ```
//!
//! The struct is deliberately small; all the interesting decisions live in the
//! components it composes. What it owns is the *loop*: escalation, consistency
//! between the Reasoner and the Memory Runtime, and the telemetry that says
//! whether any of this is working.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::context_store::{Checkpoint, ContextError, ContextStore, ObjectKind, ObjectRecord};
use crate::execute::{ExecutionLayer, Inputs};
use crate::graph::{DcrError, MemoryGraph};
use crate::ids::Clock;
use crate::index::HybridIndex;
use crate::indexer::{IndexResult, IngestCtx, StateIndexer};
use crate::json::{Json, parse};
use crate::ladder::{Ladder, Level};
use crate::llm::{LocalReasoner, REASONER_SYSTEM, Reasoner, parse_escalation};
use crate::nodes::{
    Derivation, EdgeType, Kind, LevelCache, NewNode, Node, NodeIdx, NodeMeta, Origin, Status,
};
use crate::planner::{ActiveContext, PlanCtx, RelevancePlanner, Weights};
use crate::policy::DecisionPolicy;
use crate::spans::{Document, RawStore, Span};
use crate::speculation::Speculator;
use crate::telemetry::{Telemetry, Turn};
use crate::tokens::Estimator;
use crate::trust::Signer;

use std::cell::{Cell, RefCell};

#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    pub context: ActiveContext,
    pub escalations: u32,
    pub cited: Vec<String>,
    pub tokens: usize,  // audit-allow: LM token count, not a credential
    /// True when a mid-turn invalidation forced the workspace to be rebuilt.
    pub replanned: bool,
    /// The reasoner asked for a richer level and the runtime could not serve
    /// it — the escalation budget was spent, or the node id did not resolve.
    ///
    /// Reported rather than swallowed: the turn produced no answer, and a
    /// caller that cannot tell that apart from "the history does not contain
    /// it" is being handed the degraded path without being told.
    pub escalation_refused: bool,
}

/// Returned when the reasoner requested a level the runtime would not serve.
///
/// Deliberately the same sentence the reasoner produces when nothing in the
/// window is usable, because that is what the situation is from the caller's
/// side: the model declined to answer from what it was given.
pub const UNSERVED_ESCALATION: &str = "I don't have that in the active context.";

pub struct Dcr {
    pub clock: Clock,
    pub raw: RawStore,
    pub graph: MemoryGraph,
    pub index: HybridIndex,
    pub ladder: Ladder,
    pub indexer: StateIndexer,
    pub policy: DecisionPolicy,
    pub planner: RelevancePlanner,
    pub speculator: Speculator,
    pub execution: ExecutionLayer,
    pub telemetry: Telemetry,
    pub budget: usize,
    /// How many times a turn may be re-planned at a richer level after the
    /// model reports insufficiency. Zero disables the escalation protocol.
    pub max_escalations: u32,
    /// Compute the escalation decision in the runtime when the reasoner did not
    /// send one.
    ///
    /// The poor fit this addresses is "reasoners that cannot signal": with
    /// escalation disabled the ablation drops to 6/7, losing the buried-detail
    /// probe. The model-side trigger is a function of the query and the
    /// rendered window only, so [`DecisionPolicy::needs_escalation`] can
    /// evaluate the same thing here. Off by default: it costs an extra plan on
    /// turns that would not have escalated, and a harness that *can* signal
    /// should be believed rather than second-guessed.
    pub auto_escalate: bool,
    /// Every source span that has appeared in an assembled context, ever.
    ///
    /// The dual of `audit_path_completeness`: that metric is a property of
    /// answers, and so is conditioned on the probe set knowing what to ask.
    /// This is a property of the assembly history alone — conditioned on
    /// nothing — so the spans it never contains are exactly the region a
    /// query-layer degradation check cannot see. Credit: this metric was
    /// proposed by the commenter "vespermind" against the DCR report.
    /// Planner stage clocks and rejection counts, summed over every plan this
    /// runtime has produced — including the extra plans an escalation causes,
    /// so it is the planning cost of a *turn* rather than of a call.
    pub planning: crate::planner::StageProfile,
    /// Turns planned, so `planning` can be reported per turn.
    pub plans: u64,
    assembled_spans: std::collections::HashSet<String>,
    /// Ordered pairs of spans ever rendered at L0 in the *same* window.
    assembled_pairs: std::collections::HashSet<(String, String)>,
    reasoner: Box<dyn Reasoner>,
}

impl std::fmt::Debug for Dcr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dcr")
            .field("budget", &self.budget)
            .field("graph", &self.graph.stats())
            .finish_non_exhaustive()
    }
}

impl Dcr {
    pub fn new(budget: usize) -> Self {
        Self {
            clock: Clock::new(),
            raw: RawStore::default(),
            graph: MemoryGraph::new(),
            index: HybridIndex::default(),
            ladder: Ladder::default(),
            indexer: StateIndexer::default(),
            policy: DecisionPolicy::default(),
            planner: RelevancePlanner {
                budget,
                ..Default::default()
            },
            speculator: Speculator::default(),
            execution: ExecutionLayer::default(),
            telemetry: Telemetry::default(),
            budget,
            max_escalations: 2,
            auto_escalate: false,
            planning: crate::planner::StageProfile::default(),
            plans: 0,
            assembled_spans: std::collections::HashSet::new(),
            assembled_pairs: std::collections::HashSet::new(),
            reasoner: Box::new(LocalReasoner::new()),
        }
    }

    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.planner.weights = weights;
        self
    }

    pub fn with_estimator(mut self, estimator: Estimator) -> Self {
        self.ladder.estimator = estimator;
        self
    }

    pub fn with_reasoner(mut self, reasoner: Box<dyn Reasoner>) -> Self {
        self.reasoner = reasoner;
        self
    }

    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.planner.budget = budget;
    }

    // -- ingestion ---------------------------------------------------------
    pub fn ingest(&mut self, text: &str, doc_id: Option<&str>) -> Result<IndexResult, DcrError> {
        self.ingest_with_source(text, doc_id, None)
    }

    pub fn ingest_with_source(
        &mut self,
        text: &str,
        doc_id: Option<&str>,
        source: Option<String>,
    ) -> Result<IndexResult, DcrError> {
        let mut ctx = IngestCtx {
            raw: &mut self.raw,
            graph: &mut self.graph,
            index: &mut self.index,
            ladder: &self.ladder,
            clock: &self.clock,
        };
        let result = self.indexer.ingest(&mut ctx, text, doc_id, source)?;
        self.telemetry.history_tokens += self.ladder.estimator.count(text);
        Ok(result)
    }

    pub fn ingest_file(&mut self, path: &Path) -> Result<IndexResult, DcrError> {
        let text = std::fs::read_to_string(path).map_err(|e| DcrError::Io(e.to_string()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        self.ingest_with_source(&text, Some(&name), Some(path.display().to_string()))
    }

    // -- planning and answering -------------------------------------------
    pub fn plan(&mut self, query: &str, budget: Option<usize>) -> ActiveContext {
        self.plan_with(query, budget, &[], &HashMap::new())
    }

    pub fn plan_with(
        &mut self,
        query: &str,
        budget: Option<usize>,
        pin: &[NodeIdx],
        force_level: &HashMap<NodeIdx, Level>,
    ) -> ActiveContext {
        self.graph.drain_pending(32);
        let ctx = PlanCtx {
            raw: &self.raw,
            graph: &self.graph,
            index: &self.index,
            ladder: &self.ladder,
            policy: &self.policy,
            now: self.clock.now(),
            evidence_by_span: &self.indexer.evidence_by_span,
        };
        let context = self.planner.plan(
            &ctx,
            Some(&mut self.speculator),
            query,
            budget.or(Some(self.budget)),
            pin,
            force_level,
        );
        self.planning += context.profile;
        self.plans += 1;
        context
    }

    pub fn prompt_for(&self, context: &ActiveContext) -> String {
        format!("{}\n\nQUESTION: {}\n", context.render(), context.query)
    }

    pub fn ask(&mut self, query: &str, budget: Option<usize>) -> Answer {
        let mut reasoner = std::mem::replace(
            &mut self.reasoner,
            Box::new(LocalReasoner::new()) as Box<dyn Reasoner>,
        );
        let answer = self.ask_with(query, budget, reasoner.as_mut());
        self.reasoner = reasoner;
        answer
    }

    pub fn ask_with(
        &mut self,
        query: &str,
        budget: Option<usize>,
        reasoner: &mut dyn Reasoner,
    ) -> Answer {
        self.ask_with_consolidation(query, budget, reasoner, &mut |_| {})
    }

    /// `ask`, with a hook that runs after the model returns and before the
    /// consistency check.
    ///
    /// This is where Solution B's background consolidation lands in a real
    /// deployment: it runs while Solution A is thinking, and it may invalidate
    /// a fact A is mid-way through reasoning over. Passing it explicitly keeps
    /// that race testable instead of hypothetical.
    pub fn ask_with_consolidation(
        &mut self,
        query: &str,
        budget: Option<usize>,
        reasoner: &mut dyn Reasoner,
        consolidate: &mut dyn FnMut(&mut MemoryGraph),
    ) -> Answer {
        let budget = budget.unwrap_or(self.budget);
        let mut force: HashMap<NodeIdx, Level> = HashMap::new();
        let mut pin: Vec<NodeIdx> = Vec::new();
        let mut escalations = 0u32;
        // Nodes already escalated this turn. Without this an inferred
        // escalation can re-request the node it just promoted, since the
        // promoted line still scores highest.
        let mut auto_escalated: HashSet<NodeIdx> = HashSet::new();
        let mut escalation_refused = false;
        let mut replanned = false;
        let mut consistency_retries = 0u32;

        let (text, context) = loop {
            let context = self.plan_with(query, Some(budget), &pin, &force);
            let text = reasoner.complete(&self.prompt_for(&context), REASONER_SYSTEM);
            consolidate(&mut self.graph);

            // The Reasoner has been thinking; the Memory Runtime may have
            // consolidated underneath it. Snapshot isolation with an interrupt:
            // if anything in the working set was invalidated mid-turn, the
            // answer is discarded and the workspace rebuilt rather than
            // returned from state that no longer holds.
            if self.consistency_violation(&context) && consistency_retries < 1 {
                consistency_retries += 1;
                replanned = true;
                continue;
            }
            // A signalled escalation names a node id; an inferred one names an
            // index directly. Both land in the same force/pin/re-plan step, so
            // the mechanism being exercised is identical either way and the
            // ablation rows are comparable.
            let requested: Option<NodeIdx> = match parse_escalation(&text) {
                Some(node_id) => self.graph.idx_of(&node_id),
                None if self.auto_escalate => {
                    self.policy.needs_escalation(query, &context, &auto_escalated)
                }
                None => None,
            };
            if let Some(idx) = requested {
                if escalations < self.max_escalations {
                    escalations += 1;
                    auto_escalated.insert(idx);
                    force.insert(idx, Level::L0);
                    if !pin.contains(&idx) {
                        pin.push(idx);
                    }
                    continue;
                }
            }
            // An escalation the runtime cannot serve must not be returned as
            // the answer.
            //
            // It used to be. With `max_escalations = 0` the reasoner still
            // emitted `#ESCALATE <node>`, nothing consumed it, and the control
            // token fell through to the caller as the turn's output. That made
            // the `no escalation` ablation row fail for a reason unrelated to
            // the mechanism it was ablating: the probe was scored against the
            // literal string `#ESCALATE clai_...`, so the row could not have
            // passed however reachable the answer was. A control that cannot
            // pass is as uninformative as one that cannot fail.
            if parse_escalation(&text).is_some() {
                escalation_refused = true;
                break (UNSERVED_ESCALATION.to_string(), context);
            }
            break (text, context);
        };

        let cited: Vec<String> = context
            .entries
            .iter()
            .filter(|e| text.contains(&e.node_id))
            .map(|e| e.node_id.clone())
            .collect();
        let cited_idx: HashSet<NodeIdx> = context
            .entries
            .iter()
            .filter(|e| cited.contains(&e.node_id))
            .map(|e| e.node)
            .collect();
        for &idx in &cited_idx {
            let node = self.graph.node(idx);
            node.reads.set(node.reads.get() + 1);
        }
        let stale_reads = context
            .entries
            .iter()
            .filter(|e| self.graph.node(e.node).status == Status::Stale)
            .count() as u32;

        self.telemetry.record_turn(Turn {
            query: query.to_string(),
            qtype: context.qtype.unwrap_or(crate::policy::QueryType::Open),
            tokens: context.tokens,
            budget: context.budget,
            admitted: context.entries.len(),
            considered: context.considered,
            escalations,
            stale_reads,
            demotions: context.demoted.len(),
            overflow: context.overflow,
        });
        self.speculator.feedback(&cited_idx);
        self.mark_assembled(&context);

        Answer {
            tokens: context.tokens,
            text,
            context,
            escalations,
            cited,
            replanned,
            escalation_refused,
        }
    }

    /// Record the source spans of every node the model was shown this turn.
    ///
    /// A span counts as read if any node grounded in it was admitted, at any
    /// ladder level — an L2 fact conveys the span's content just as an L0 quote
    /// does. Spans that never appear here were ingested, indexed, and never
    /// looked at; that is where a confident wrong answer comes from.
    fn mark_assembled(&mut self, context: &ActiveContext) {
        let mut shown_this_turn: Vec<String> = Vec::new();
        for entry in &context.entries {
            // Only L0 renders a span's actual bytes. L1 summarises, and L2/L3
            // convey a value derived from the spans — and because corroboration
            // collapses many agreeing spans behind one node, any level above L0
            // would mark spans whose specific content was never shown (a single
            // L1 fact can sit on hundreds of spans). A silent wrong answer hides
            // in exactly such an unshown span, so coverage counts L0 alone. A
            // node cheap enough to admit at L0 has few spans, so this does not
            // undercount by symmetry.
            if entry.level != Level::L0 {
                continue;
            }
            for span in &self.graph.node(entry.node).source_spans {
                self.assembled_spans.insert(span.clone());
                shown_this_turn.push(span.clone());
            }
        }
        // Co-occurrence, not just occurrence. A reviewer pointed out that span
        // coverage is unconditioned along one axis only: A and B can each be
        // covered while never once appearing together, and a multi-hop answer
        // needing both in scope fails while both look healthy in the table.
        shown_this_turn.sort();
        shown_this_turn.dedup();
        for (i, a) in shown_this_turn.iter().enumerate() {
            for b in &shown_this_turn[i + 1..] {
                self.assembled_pairs.insert((a.clone(), b.clone()));
            }
        }
    }

    /// Read coverage: the fraction of L0 spans that have ever been assembled.
    ///
    /// Runs offline, replays no queries, and is conditioned on nothing. The
    /// question worth watching is whether the covered count grows with history
    /// or stays flat while `N` grows — see `bench --coverage`.
    pub fn coverage(&self) -> Coverage {
        let total = self.raw.len();
        let assembled = self
            .assembled_spans
            .iter()
            .filter(|id| self.raw.has_span(id))
            .count();
        Coverage {
            total_spans: total,
            assembled_spans: assembled,
            fraction: if total == 0 {
                0.0
            } else {
                assembled as f64 / total as f64
            },
        }
    }

    /// Co-occurrence coverage: of the span pairs the graph says are
    /// load-bearing together, how many have ever been in the same window?
    ///
    /// Span coverage answers "was this ever rendered". It does not answer "was
    /// it ever rendered *alongside* the thing that makes it matter" — a
    /// reviewer's point, and it bounds what [`Self::coverage`] can claim. An
    /// answer needing A and B in scope simultaneously can fail while both spans
    /// sit in the covered column.
    ///
    /// Enumerating all pairs is `O(N^2)` and mostly meaningless, since most
    /// pairs are unrelated. The denominator here is the set the graph already
    /// knows about: for every dependency edge, the cross product of the two
    /// nodes' source spans. That is where co-occurrence is load-bearing, it is
    /// `O(edges)` rather than `O(N^2)`, and the map is already in the graph
    /// rather than needing to be inferred.
    pub fn pair_coverage(&self) -> PairCoverage {
        let mut linked: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for node in self.graph.nodes() {
            for dep_id in &node.dependencies {
                let Some(dep_idx) = self.graph.idx_of(dep_id) else {
                    continue;
                };
                for a in &node.source_spans {
                    for b in &self.graph.node(dep_idx).source_spans {
                        if a == b {
                            continue;
                        }
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        linked.insert((lo.clone(), hi.clone()));
                    }
                }
            }
        }
        let covered = linked
            .iter()
            .filter(|pair| self.assembled_pairs.contains(*pair))
            .count();
        let total = linked.len();
        PairCoverage {
            linked_pairs: total,
            covered_pairs: covered,
            fraction: if total == 0 {
                0.0
            } else {
                covered as f64 / total as f64
            },
        }
    }

    /// Did background consolidation invalidate anything in this plan?
    fn consistency_violation(&self, context: &ActiveContext) -> bool {
        let touched = self.graph.invalidated_since(context.snapshot_version);
        context.entries.iter().any(|e| touched.contains(&e.node))
    }

    // -- state updates -----------------------------------------------------
    /// Cache a settled conclusion. Rejected unless it is groundable.
    pub fn commit_fact(
        &mut self,
        value: &str,
        key: Option<&str>,
        source_spans: &[String],
        dependencies: &[String],
        confidence: f32,
        kind: Kind,
    ) -> Result<NodeIdx, DcrError> {
        let node = NewNode::new(kind, value)
            .spans(source_spans.to_vec())
            .deps(dependencies.to_vec())
            .confidence(confidence)
            .maybe_key(key.map(str::to_string))
            .build();
        let conflicts: Vec<NodeIdx> = match key {
            Some(key) if kind == Kind::Claim => self
                .graph
                .by_key(key, true)
                .into_iter()
                .filter(|&i| {
                    let other = self.graph.node(i);
                    other.id != node.id && other.kind == Kind::Claim && other.value != node.value
                })
                .collect(),
            _ => Vec::new(),
        };
        let idx = self.graph.upsert(node, &self.raw, &self.clock)?;
        self.reindex(idx);
        for other in conflicts {
            self.graph.contradict(idx, other);
            if self.indexer.supersede_on_conflict {
                self.graph.supersede(other, idx);
                self.index.remove_node(other);
            }
        }
        Ok(idx)
    }

    /// Run a registered derivation (L3) and cache it as a calculation node.
    pub fn compute(
        &mut self,
        name: &str,
        inputs: Inputs,
        deps: Vec<String>,
        key: Option<&str>,
    ) -> Result<NodeIdx, DcrError> {
        let node = self.execution.compute_node(
            &self.graph,
            name,
            inputs,
            deps,
            key.map(str::to_string),
        )?;
        let idx = self.graph.upsert(node, &self.raw, &self.clock)?;
        self.reindex(idx);
        Ok(idx)
    }

    pub fn register(&mut self, name: &str, f: impl Fn(&Inputs) -> f64 + 'static) {
        self.execution.register(name, f);
    }

    /// Re-ground a stale node: recompute if derived, drop cheap levels if not.
    pub fn revalidate(&mut self, idx: NodeIdx) -> Result<(), DcrError> {
        let derivation: Option<Derivation> = self.graph.node(idx).meta.derivation.clone();
        if derivation.is_some() {
            let node = self.graph.node(idx);
            if let Some(value) = self.execution.run(&self.graph, node)? {
                self.graph.node_mut(idx).value = crate::ladder::format_number(value);
            }
        }
        {
            let node = self.graph.node(idx);
            let mut cache = node.level_cache.borrow_mut();
            cache.l1 = None;
            cache.l2 = None;
        }
        self.graph.revalidate(idx, &self.clock);
        self.reindex(idx);
        Ok(())
    }

    fn reindex(&mut self, idx: NodeIdx) {
        let mut ctx = IngestCtx {
            raw: &mut self.raw,
            graph: &mut self.graph,
            index: &mut self.index,
            ladder: &self.ladder,
            clock: &self.clock,
        };
        self.indexer.reindex(&mut ctx, idx);
    }

    // -- audit -------------------------------------------------------------
    pub fn explain(&mut self, node_id: &str) -> Result<String, DcrError> {
        let idx = self
            .graph
            .idx_of(node_id)
            .ok_or_else(|| DcrError::UnknownNode(node_id.to_string()))?;
        let path = self.graph.explain(idx, None);
        self.telemetry.record_explain(path.complete);
        let mut lines = vec![format!("explain({node_id})  complete={}", path.complete)];
        for &node_idx in &path.nodes {
            let node = self.graph.node(node_idx);
            let indent = "  ".repeat(path.depths.get(&node_idx).copied().unwrap_or(0));
            let head = match &node.key {
                Some(key) => format!("{key} = {}", node.value),
                None => node.value.clone(),
            };
            let head: String = head.chars().take(96).collect();
            lines.push(format!(
                "{indent}- [{}] {}: {head}  (conf={:.2}, {})",
                node.id,
                node.kind.as_str(),
                node.confidence,
                node.status.as_str()
            ));
        }
        for span_id in &path.spans {
            let text: String = self.raw.text_of(span_id).chars().take(120).collect();
            lines.push(format!("  source {span_id}: \"{text}\""));
        }
        if !path.complete {
            let ids: Vec<&str> = path
                .truncated_at
                .iter()
                .map(|&i| self.graph.node(i).id.as_str())
                .collect();
            lines.push(format!("  ! truncated at: {}", ids.join(", ")));
        }
        Ok(lines.join("\n"))
    }

    pub fn explain_plan(&self, context: &ActiveContext) -> String {
        self.planner.explain_plan(&self.graph, context)
    }

    // -- background work ---------------------------------------------------
    /// Solution B's background pass: drain deferred invalidation and collapse
    /// nodes that keep being read raw without yielding anything new.
    pub fn consolidate(&mut self, limit: usize) -> (usize, Vec<String>) {
        let drained = self.graph.drain_pending(limit).len();
        let mut collapsed = Vec::new();
        for idx in 0..self.graph.len() {
            let node = self.graph.node(idx);
            if node.status != Status::Fresh || !self.policy.should_deescalate(node) {
                continue;
            }
            collapsed.push(node.id.clone());
            self.graph.node_mut(idx).meta.deescalated = true;
        }
        (drained, collapsed)
    }

    /// Destroy the working set and rebuild it from Solution B alone.
    ///
    /// The wiki claims this invariant; the number that makes it meaningful is
    /// the rebuild cost, so it is measured here.
    pub fn rebuild_workspace(&mut self, query: &str, budget: Option<usize>) -> RebuildReport {
        let mut cleared = 0usize;
        for node in self.graph.nodes() {
            let mut cache = node.level_cache.borrow_mut();
            cleared += usize::from(cache.l1.is_some())
                + usize::from(cache.l2.is_some())
                + usize::from(cache.l3.is_some());
            *cache = LevelCache::default();
        }
        let before = self.ladder.builds();
        let context = self.plan(query, budget);
        let after = self.ladder.builds();
        RebuildReport {
            cleared_level_cache_entries: cleared,
            rebuilt_l1: after.l1 - before.l1,
            rebuilt_l2: after.l2 - before.l2,
            rebuilt_l3: after.l3 - before.l3,
            tokens: context.tokens,
            nodes: context.entries.len(),
        }
    }

    // -- reporting ---------------------------------------------------------
    pub fn report(&self) -> String {
        // Publish the speculator's counters through the same report as the
        // rest of the evaluation metrics, rather than only in its own line.
        let mut telemetry = self.telemetry.clone();
        telemetry.record_prefetch(
            self.speculator.issued,
            self.speculator.hits,
            self.speculator.wasted_builds,
        );
        let mut out = telemetry.report().to_string();
        out.push_str(&format!("\n[graph]\n  {}\n", self.graph.stats()));
        out.push_str(&format!("[index]\n  {}\n", self.index.stats()));
        out.push_str(&format!("[execution]\n  {}\n", self.execution.stats()));
        out.push_str(&format!("[speculation]\n  {}\n", self.speculator.stats()));
        out.push_str(&format!(
            "\nraw: {} spans, {} chars\nladder builds: {}\n",
            self.raw.len(),
            self.raw.total_chars(),
            self.ladder.builds()
        ));
        out
    }

    // -- persistence -------------------------------------------------------
    pub fn save(&self, path: &Path) -> Result<(), DcrError> {
        let documents = Json::Arr(
            self.raw
                .documents()
                .iter()
                .map(|d| {
                    Json::obj(vec![
                        ("doc_id", Json::str(&d.id)),
                        ("text", Json::str(&d.text)),
                        ("ts", Json::num(d.ts as f64)),
                        ("source", d.source.clone().map_or(Json::Null, Json::Str)),
                    ])
                })
                .collect(),
        );
        let spans = Json::Arr(
            self.raw
                .spans()
                .iter()
                .map(|s| {
                    Json::obj(vec![
                        ("span_id", Json::str(&s.id)),
                        ("doc_id", Json::str(&s.doc_id)),
                        ("start", Json::num(s.start as f64)),
                        ("end", Json::num(s.end as f64)),
                        ("seq", Json::num(s.seq as f64)),
                        ("ts", Json::num(s.ts as f64)),
                    ])
                })
                .collect(),
        );
        let nodes = Json::Arr(self.graph.nodes().iter().map(node_to_json).collect());
        let mut edges = Vec::new();
        for idx in 0..self.graph.len() {
            for edge in self.graph.out_edges(idx) {
                edges.push(Json::obj(vec![
                    ("src", Json::str(&self.graph.node(edge.src).id)),
                    ("dst", Json::str(&self.graph.node(edge.dst).id)),
                    ("type", Json::str(edge.kind.as_str())),
                ]));
            }
        }
        let mut evidence: Vec<(String, Json)> = self
            .indexer
            .evidence_by_span
            .iter()
            .map(|(k, v)| (k.clone(), Json::str(v)))
            .collect();
        evidence.sort_by(|a, b| a.0.cmp(&b.0));

        let payload = Json::obj(vec![
            ("clock", Json::num(self.clock.now() as f64)),
            (
                "raw",
                Json::obj(vec![("documents", documents), ("spans", spans)]),
            ),
            (
                "graph",
                Json::obj(vec![
                    ("nodes", nodes),
                    ("edges", Json::Arr(edges)),
                    ("version", Json::num(self.graph.version as f64)),
                ]),
            ),
            (
                "predictor",
                Json::obj(vec![
                    (
                        "weights",
                        Json::Arr(
                            self.speculator
                                .predictor
                                .weights
                                .iter()
                                .map(|w| Json::num(*w as f64))
                                .collect(),
                        ),
                    ),
                    (
                        "updates",
                        Json::num(self.speculator.predictor.updates as f64),
                    ),
                ]),
            ),
            (
                "history_tokens",
                Json::num(self.telemetry.history_tokens as f64),
            ),
            ("evidence_by_span", Json::Obj(evidence)),
        ]);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| DcrError::Io(e.to_string()))?;
            }
        }
        std::fs::write(path, payload.to_json_string()).map_err(|e| DcrError::Io(e.to_string()))
    }

    /// Persist into a `.context` container and seal it as a new generation.
    ///
    /// Same state as [`Dcr::save`], written as immutable content-addressed
    /// objects under a Merkle root instead of one rewritable file. The
    /// difference that matters is not the layout: a `.dcr.json` that has been
    /// edited reloads without complaint, and an edited container does not.
    ///
    /// Nodes are written as derived objects citing the spans they came from,
    /// so the provenance rule the graph enforces in memory is enforced again
    /// by the store on the way to disk.
    pub fn save_context(
        &self,
        dir: &Path,
        signer: Option<&dyn Signer>,
    ) -> Result<Checkpoint, DcrError> {
        let mut store = match ContextStore::open(dir) {
            Ok(store) => store,
            Err(ContextError::Io(_)) => {
                ContextStore::create(dir, "dcr").map_err(context_err)?
            }
            Err(e) => return Err(context_err(e)),
        };

        // L0 first: spans cite documents, nodes cite spans, so every object
        // exists before anything points at it.
        let mut doc_objects: HashMap<String, String> = HashMap::new();
        for document in self.raw.documents() {
            let record = ObjectRecord::new(
                ObjectKind::Document,
                &document.id,
                Json::obj(vec![
                    ("text", Json::str(&document.text)),
                    ("ts", Json::num(document.ts as f64)),
                    (
                        "source",
                        document.source.clone().map_or(Json::Null, Json::Str),
                    ),
                ]),
            );
            let id = store.put(record).map_err(context_err)?;
            doc_objects.insert(document.id.clone(), id);
        }

        let mut span_objects: HashMap<String, String> = HashMap::new();
        for span in self.raw.spans() {
            let parent = doc_objects.get(&span.doc_id).cloned();
            let mut record = ObjectRecord::new(
                ObjectKind::Span,
                &span.id,
                Json::obj(vec![
                    ("doc_id", Json::str(&span.doc_id)),
                    ("start", Json::num(span.start as f64)),
                    ("end", Json::num(span.end as f64)),
                    ("seq", Json::num(span.seq as f64)),
                    ("ts", Json::num(span.ts as f64)),
                ]),
            );
            if let Some(parent) = parent {
                record = record.derived_from(vec![parent]);
            }
            let id = store.put(record).map_err(context_err)?;
            span_objects.insert(span.id.clone(), id);
        }

        let mut usage: Vec<(String, Json)> = Vec::new();
        for node in self.graph.nodes() {
            let sources: Vec<String> = node
                .source_spans
                .iter()
                .filter_map(|span_id| span_objects.get(span_id).cloned())
                .collect();
            // Read counters move to a single usage object. Left in the node,
            // they would give every admitted node a new address on every
            // query, and the store would grow with reads rather than with what
            // it actually knows.
            usage.push((
                node.id.clone(),
                Json::obj(vec![
                    ("reads", Json::num(node.reads.get() as f64)),
                    ("admits", Json::num(node.admits.get() as f64)),
                    ("l0_reads", Json::num(node.meta.l0_reads.get() as f64)),
                    ("l0_yield", Json::num(node.meta.l0_yield.get() as f64)),
                ]),
            ));
            let mut record =
                ObjectRecord::new(ObjectKind::Node, &node.id, node_object(node));
            // A node grounded only in other nodes still has provenance; it is
            // the ungrounded one the store must refuse.
            if !sources.is_empty() || !node.dependencies.is_empty() {
                record = record.derived_from(if sources.is_empty() {
                    node.dependencies.clone()
                } else {
                    sources
                });
            }
            store.put(record).map_err(context_err)?;
        }

        for idx in 0..self.graph.len() {
            for edge in self.graph.out_edges(idx) {
                let (src, dst) = (
                    self.graph.node(edge.src).id.clone(),
                    self.graph.node(edge.dst).id.clone(),
                );
                let record = ObjectRecord::new(
                    ObjectKind::Edge,
                    format!("{src}->{dst}:{}", edge.kind.as_str()),
                    Json::obj(vec![
                        ("src", Json::str(&src)),
                        ("dst", Json::str(&dst)),
                        ("type", Json::str(edge.kind.as_str())),
                    ]),
                );
                store.put(record).map_err(context_err)?;
            }
        }

        usage.sort_by(|a, b| a.0.cmp(&b.0));
        if !usage.is_empty() {
            store
                .put(ObjectRecord::new(
                    ObjectKind::Usage,
                    "usage",
                    Json::Obj(usage),
                ))
                .map_err(context_err)?;
        }

        store
            .commit(self.clock.now(), signer)
            .map_err(context_err)
    }

    /// Open a `.context` container.
    ///
    /// Every object goes through [`ContextStore::admit`] on the way in, so
    /// nothing reaches the graph that failed integrity, provenance, generation,
    /// schema or policy. A refused object is an error, not a warning: a
    /// partially admitted memory is worse than one that refuses to open.
    pub fn open_context(dir: &Path, budget: usize) -> Result<Dcr, DcrError> {
        let store = ContextStore::open(dir).map_err(context_err)?;
        let mut runtime = Dcr::new(budget);

        let mut documents: Vec<Document> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        let mut nodes: Vec<Json> = Vec::new();
        let mut edges: Vec<(String, String, EdgeType)> = Vec::new();
        let mut usage: Vec<(String, Json)> = Vec::new();

        let ids: Vec<String> = store.object_ids().cloned().collect();
        for id in ids {
            let (record, _admitted) = store.admit(&id, None).map_err(context_err)?;
            let body = &record.body;
            match record.kind {
                ObjectKind::Document => documents.push(Document {
                    id: record.logical_id.clone(),
                    text: body
                        .get("text")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    ts: body.get("ts").and_then(Json::as_u64).unwrap_or(0),
                    source: body.get("source").and_then(Json::as_str).map(str::to_string),
                }),
                ObjectKind::Span => spans.push(Span {
                    id: record.logical_id.clone(),
                    doc_id: body
                        .get("doc_id")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    start: body.get("start").and_then(Json::as_usize).unwrap_or(0),
                    end: body.get("end").and_then(Json::as_usize).unwrap_or(0),
                    seq: body.get("seq").and_then(Json::as_usize).unwrap_or(0),
                    ts: body.get("ts").and_then(Json::as_u64).unwrap_or(0),
                }),
                ObjectKind::Node => nodes.push(body.clone()),
                ObjectKind::Usage => {
                    // Later generations win: usage is a running total, and the
                    // newest object is the current one.
                    if let Json::Obj(pairs) = body {
                        if record.generation
                            >= usage.first().map(|_| record.generation).unwrap_or(0)
                        {
                            usage = pairs.clone();
                        }
                    }
                }
                ObjectKind::Edge => {
                    if let (Some(src), Some(dst), Some(kind)) = (
                        body.get("src").and_then(Json::as_str),
                        body.get("dst").and_then(Json::as_str),
                        body.get("type")
                            .and_then(Json::as_str)
                            .and_then(EdgeType::parse),
                    ) {
                        edges.push((src.to_string(), dst.to_string(), kind));
                    }
                }
            }
        }

        // Insertion order is load-bearing for spans (seq) and for the graph's
        // own indices, so restore in the order the runtime wrote them.
        documents.sort_by(|a, b| a.ts.cmp(&b.ts).then(a.id.cmp(&b.id)));
        spans.sort_by(|a, b| a.seq.cmp(&b.seq));
        let clock = spans.iter().map(|s| s.ts).max().unwrap_or(0);
        runtime.raw.restore(documents, spans);

        nodes.sort_by_key(|n| n.get("timestamp").and_then(Json::as_u64).unwrap_or(0));
        for node in &nodes {
            runtime.graph.insert_restored(node_from_json(node));
        }
        for (src, dst, kind) in edges {
            runtime.graph.restore_edge(&src, &dst, kind);
        }
        for (node_id, counters) in &usage {
            let Some(idx) = runtime.graph.idx_of(node_id) else {
                continue;
            };
            let node = runtime.graph.node(idx);
            let count = |name: &str| counters.get(name).and_then(Json::as_u64).unwrap_or(0) as u32;
            node.reads.set(count("reads"));
            node.admits.set(count("admits"));
            node.meta.l0_reads.set(count("l0_reads"));
            node.meta.l0_yield.set(count("l0_yield"));
        }
        runtime.clock.restore(
            clock.max(
                nodes
                    .iter()
                    .filter_map(|n| n.get("timestamp").and_then(Json::as_u64))
                    .max()
                    .unwrap_or(0),
            ),
        );

        for idx in 0..runtime.raw.len() {
            let text = runtime.raw.text_of(&runtime.raw.span(idx).id).to_string();
            runtime.index.add_span(idx, &text);
        }
        for idx in 0..runtime.graph.len() {
            runtime.reindex(idx);
        }
        Ok(runtime)
    }

    pub fn load(path: &Path, budget: usize) -> Result<Dcr, DcrError> {
        let text = std::fs::read_to_string(path).map_err(|e| DcrError::Io(e.to_string()))?;
        let data = parse(&text).map_err(DcrError::Parse)?;
        let mut runtime = Dcr::new(budget);
        runtime
            .clock
            .restore(data.get("clock").and_then(Json::as_u64).unwrap_or(0));

        let raw = data
            .get("raw")
            .ok_or_else(|| DcrError::Parse("no raw".into()))?;
        let documents: Vec<Document> = raw
            .get("documents")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|d| Document {
                id: d
                    .get("doc_id")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string(),
                text: d
                    .get("text")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string(),
                ts: d.get("ts").and_then(Json::as_u64).unwrap_or(0),
                source: d.get("source").and_then(Json::as_str).map(str::to_string),
            })
            .collect();
        let spans: Vec<Span> = raw
            .get("spans")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|s| Span {
                id: s
                    .get("span_id")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string(),
                doc_id: s
                    .get("doc_id")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string(),
                start: s.get("start").and_then(Json::as_usize).unwrap_or(0),
                end: s.get("end").and_then(Json::as_usize).unwrap_or(0),
                seq: s.get("seq").and_then(Json::as_usize).unwrap_or(0),
                ts: s.get("ts").and_then(Json::as_u64).unwrap_or(0),
            })
            .collect();
        runtime.raw.restore(documents, spans);

        let graph = data
            .get("graph")
            .ok_or_else(|| DcrError::Parse("no graph".into()))?;
        for node in graph.get("nodes").and_then(Json::as_array).unwrap_or(&[]) {
            runtime.graph.insert_restored(node_from_json(node));
        }
        for edge in graph.get("edges").and_then(Json::as_array).unwrap_or(&[]) {
            let (Some(src), Some(dst), Some(kind)) = (
                edge.get("src").and_then(Json::as_str),
                edge.get("dst").and_then(Json::as_str),
                edge.get("type")
                    .and_then(Json::as_str)
                    .and_then(EdgeType::parse),
            ) else {
                continue;
            };
            runtime.graph.restore_edge(src, dst, kind);
        }
        if let Some(predictor) = data.get("predictor") {
            if let Some(weights) = predictor.get("weights").and_then(Json::as_array) {
                for (slot, value) in runtime
                    .speculator
                    .predictor
                    .weights
                    .iter_mut()
                    .zip(weights.iter())
                {
                    if let Some(v) = value.as_f64() {
                        *slot = v as f32;
                    }
                }
            }
            runtime.speculator.predictor.updates =
                predictor.get("updates").and_then(Json::as_u64).unwrap_or(0) as u32;
        }
        runtime.telemetry.history_tokens = data
            .get("history_tokens")
            .and_then(Json::as_usize)
            .unwrap_or(0);
        if let Some(Json::Obj(pairs)) = data.get("evidence_by_span") {
            for (span_id, node_id) in pairs {
                if let Some(id) = node_id.as_str() {
                    runtime
                        .indexer
                        .evidence_by_span
                        .insert(span_id.clone(), id.to_string());
                }
            }
        }

        // Rebuild the retrieval index from state — it is derived, not source.
        for idx in 0..runtime.raw.len() {
            let text = runtime.raw.text_of(&runtime.raw.span(idx).id).to_string();
            runtime.index.add_span(idx, &text);
        }
        for idx in 0..runtime.graph.len() {
            runtime.reindex(idx);
        }
        Ok(runtime)
    }
}

/// Read coverage — the offline, unconditioned dual of audit-path completeness.
#[derive(Debug, Clone, Copy)]
pub struct Coverage {
    pub total_spans: usize,
    pub assembled_spans: usize,
    pub fraction: f64,
}

/// The two-dimensional dual of [`Coverage`]: see [`Dcr::pair_coverage`].
#[derive(Debug, Clone, Copy)]
pub struct PairCoverage {
    /// Span pairs joined by a dependency edge — where co-occurrence matters.
    pub linked_pairs: usize,
    /// Of those, how many were ever rendered at L0 in the same window.
    pub covered_pairs: usize,
    pub fraction: f64,
}

impl std::fmt::Display for Coverage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} spans ever assembled ({:.1}% read coverage); {} spans never seen",
            self.assembled_spans,
            self.total_spans,
            self.fraction * 100.0,
            self.total_spans - self.assembled_spans
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RebuildReport {
    pub cleared_level_cache_entries: usize,
    pub rebuilt_l1: usize,
    pub rebuilt_l2: usize,
    pub rebuilt_l3: usize,
    pub tokens: usize,  // audit-allow: LM token count, not a credential
    pub nodes: usize,
}

impl std::fmt::Display for RebuildReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cleared_level_cache_entries: {}, rebuilt: {{L1: {}, L2: {}, L3: {}}}, tokens: {}, nodes: {}",
            self.cleared_level_cache_entries,
            self.rebuilt_l1,
            self.rebuilt_l2,
            self.rebuilt_l3,
            self.tokens,
            self.nodes
        )
    }
}

/// Container failures become `DcrError`s at the runtime boundary, keeping the
/// distinction the message carries: refused, rolled back and corrupt are not
/// the same event, and flattening them to "load failed" would lose the only
/// information an operator can act on.
fn context_err(error: ContextError) -> DcrError {
    match error {
        ContextError::Io(m) => DcrError::Io(m),
        other => DcrError::Parse(other.to_string()),
    }
}

fn strings(values: &[String]) -> Json {
    Json::Arr(values.iter().map(Json::str).collect())
}

/// A node as an immutable container object: everything [`node_to_json`] holds
/// except the read counters, which are usage rather than content.
fn node_object(node: &Node) -> Json {
    const VOLATILE: [&str; 2] = ["reads", "admits"];
    const VOLATILE_META: [&str; 2] = ["l0_reads", "l0_yield"];
    let Json::Obj(fields) = node_to_json(node) else {
        return node_to_json(node);
    };
    Json::Obj(
        fields
            .into_iter()
            .filter(|(key, _)| !VOLATILE.contains(&key.as_str()))
            .map(|(key, value)| match (key.as_str(), &value) {
                ("meta", Json::Obj(meta)) => (
                    key.clone(),
                    Json::Obj(
                        meta.iter()
                            .filter(|(k, _)| !VOLATILE_META.contains(&k.as_str()))
                            .cloned()
                            .collect(),
                    ),
                ),
                _ => (key, value),
            })
            .collect(),
    )
}

fn node_to_json(node: &Node) -> Json {
    let derivation = match &node.meta.derivation {
        Some(d) => Json::obj(vec![
            ("name", Json::str(&d.name)),
            (
                "inputs",
                Json::Obj(
                    d.inputs
                        .iter()
                        .map(|(k, v)| (k.clone(), Json::num(*v)))
                        .collect(),
                ),
            ),
        ]),
        None => Json::Null,
    };
    Json::obj(vec![
        ("node_id", Json::str(&node.id)),
        ("kind", Json::str(node.kind.as_str())),
        ("origin", Json::str(node.origin.as_str())),
        ("value", Json::str(&node.value)),
        ("source_spans", strings(&node.source_spans)),
        ("dependencies", strings(&node.dependencies)),
        ("confidence", Json::num(node.confidence as f64)),
        ("timestamp", Json::num(node.timestamp as f64)),
        ("status", Json::str(node.status.as_str())),
        ("key", node.key.clone().map_or(Json::Null, Json::Str)),
        ("reads", Json::num(node.reads.get() as f64)),
        ("admits", Json::num(node.admits.get() as f64)),
        (
            "l1",
            node.level_cache
                .borrow()
                .l1
                .clone()
                .map_or(Json::Null, Json::Str),
        ),
        (
            "l3",
            node.level_cache.borrow().l3.map_or(Json::Null, Json::num),
        ),
        (
            "meta",
            Json::obj(vec![
                ("corrective", Json::Bool(node.meta.corrective)),
                ("grounded", Json::Bool(node.meta.grounded)),
                (
                    "superseded_by",
                    node.meta
                        .superseded_by
                        .clone()
                        .map_or(Json::Null, Json::Str),
                ),
                ("supersedes", strings(&node.meta.supersedes)),
                ("contradicts", strings(&node.meta.contradicts)),
                ("superseded_source", Json::Bool(node.meta.superseded_source)),
                ("corrected_by", strings(&node.meta.corrected_by)),
                ("corroborations", Json::num(node.meta.corroborations as f64)),
                ("l0_reads", Json::num(node.meta.l0_reads.get() as f64)),
                ("l0_yield", Json::num(node.meta.l0_yield.get() as f64)),
                ("deescalated", Json::Bool(node.meta.deescalated)),
                ("derivation", derivation),
            ]),
        ),
    ])
}

fn node_from_json(data: &Json) -> Node {
    let meta_json = data.get("meta");
    let get_meta = |key: &str| meta_json.and_then(|m| m.get(key));
    let derivation = get_meta("derivation").and_then(|d| {
        let name = d.get("name")?.as_str()?.to_string();
        let inputs = match d.get("inputs") {
            Some(Json::Obj(pairs)) => pairs
                .iter()
                .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
                .collect(),
            _ => Vec::new(),
        };
        Some(Derivation { name, inputs })
    });
    let meta = NodeMeta {
        corrective: get_meta("corrective")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        grounded: get_meta("grounded").and_then(Json::as_bool).unwrap_or(true),
        superseded_by: get_meta("superseded_by")
            .and_then(Json::as_str)
            .map(str::to_string),
        supersedes: get_meta("supersedes")
            .map(Json::string_list)
            .unwrap_or_default(),
        contradicts: get_meta("contradicts")
            .map(Json::string_list)
            .unwrap_or_default(),
        superseded_source: get_meta("superseded_source")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        corrected_by: get_meta("corrected_by")
            .map(Json::string_list)
            .unwrap_or_default(),
        corroborations: get_meta("corroborations")
            .and_then(Json::as_u64)
            .unwrap_or(0) as u32,
        l0_reads: Cell::new(get_meta("l0_reads").and_then(Json::as_u64).unwrap_or(0) as u32),
        l0_yield: Cell::new(get_meta("l0_yield").and_then(Json::as_u64).unwrap_or(0) as u32),
        deescalated: get_meta("deescalated")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        derivation,
        source: None,
    };
    Node {
        id: data
            .get("node_id")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
        kind: data
            .get("kind")
            .and_then(Json::as_str)
            .and_then(Kind::parse)
            .unwrap_or(Kind::Claim),
        // A store written before origins existed holds material this runtime
        // read directly, so `observed` is the honest default rather than a
        // placeholder — same forward-compatibility rule as the fields above.
        origin: data
            .get("origin")
            .and_then(Json::as_str)
            .and_then(Origin::parse)
            .unwrap_or_default(),
        value: data
            .get("value")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
        source_spans: data
            .get("source_spans")
            .map(Json::string_list)
            .unwrap_or_default(),
        dependencies: data
            .get("dependencies")
            .map(Json::string_list)
            .unwrap_or_default(),
        confidence: data.get("confidence").and_then(Json::as_f64).unwrap_or(1.0) as f32,
        timestamp: data.get("timestamp").and_then(Json::as_u64).unwrap_or(0),
        status: data
            .get("status")
            .and_then(Json::as_str)
            .and_then(Status::parse)
            .unwrap_or(Status::Fresh),
        key: data.get("key").and_then(Json::as_str).map(str::to_string),
        level_cache: RefCell::new(LevelCache {
            l1: data.get("l1").and_then(Json::as_str).map(str::to_string),
            l2: None,
            l3: data.get("l3").and_then(Json::as_f64),
            // Not persisted: it is a memo of a derivable quantity, and a stored
            // one would have to be trusted against a span list that reloads
            // separately.
            l0_sizes: None,
        }),
        meta,
        reads: Cell::new(data.get("reads").and_then(Json::as_u64).unwrap_or(0) as u32),
        admits: Cell::new(data.get("admits").and_then(Json::as_u64).unwrap_or(0) as u32),
    }
}

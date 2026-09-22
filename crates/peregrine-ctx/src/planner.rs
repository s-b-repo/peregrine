//! Relevance Planner — decides what the model actually sees.
//!
//! This is the component that keeps `k` small:
//!
//! 1. **Seed** from `q_t` by index search at the cheapest plausible level.
//! 2. **Expand** along dependency edges (capped depth *and* fan-out — the
//!    over-expansion failure mode pulls in the transitive world).
//! 3. **Level-assign** each node via the decision policy.
//! 4. **Bound** to `B_attention` with the knapsack, which demotes before it
//!    drops.
//! 5. **Speculate** on what is left over.
//!
//! `U(x)` is the dominant failure mode of the whole design: get it wrong and
//! the optimiser confidently fills the window with cheap-and-useless material
//! *while looking efficient*. It is therefore a small, inspectable weighted sum
//! with every term named, and [`RelevancePlanner::explain_plan`] prints the
//! per-node arithmetic so a bad answer can be traced to the term that caused it.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::budget::{Candidate, Choice, solve};
use crate::graph::MemoryGraph;
use crate::index::{HybridIndex, Namespace};
use crate::ladder::{Ladder, Level};
use crate::nodes::{EdgeType, Kind, Node, NodeIdx, Status};
use crate::policy::{DecisionPolicy, QueryType};
use crate::spans::RawStore;
use crate::speculation::{Prediction, Speculator};

#[derive(Debug, Clone)]
pub struct Weights {
    pub similarity: f32,
    pub proximity: f32,
    pub explain_path: f32,
    pub confidence: f32,
    pub read_through: f32,
    pub recency: f32,
    pub stale_penalty: f32,
    /// Penalty for evidence whose only dependents were superseded — otherwise a
    /// corrected fact's original wording keeps competing with the correction.
    pub superseded_source: f32,
    /// A node with a live `contradicts` edge is *more* useful, not less: the
    /// model should adjudicate rather than inherit whichever side won.
    pub contradiction_bonus: f32,
    pub decay: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            similarity: 1.0,
            proximity: 0.8,
            explain_path: 1.5,
            confidence: 0.4,
            read_through: 0.3,
            recency: 0.2,
            stale_penalty: 2.0,
            superseded_source: 1.2,
            contradiction_bonus: 0.6,
            decay: 0.6,
        }
    }
}

impl Weights {
    pub fn kind_prior(&self, kind: Kind) -> f32 {
        match kind {
            Kind::Goal | Kind::Constraint => 0.5,
            Kind::Decision => 0.25,
            Kind::Claim | Kind::Calculation => 0.2,
            Kind::OpenQuestion => 0.1,
            Kind::Evidence => 0.0,
        }
    }
}

/// Fidelity of a level *for a given question shape*.
fn level_fit(qtype: QueryType, level: Level) -> f32 {
    use Level::{L0, L1, L2, L3};
    match (qtype, level) {
        (QueryType::QuoteExact, L0) => 1.0,
        (QueryType::QuoteExact, L1) => 0.55,
        (QueryType::QuoteExact, L2) => 0.35,
        (QueryType::QuoteExact, L3) => 0.4,
        (QueryType::ValueLookup, L0) => 0.85,
        (QueryType::ValueLookup, L1) => 0.8,
        (QueryType::ValueLookup, L2) => 1.0,
        (QueryType::ValueLookup, L3) => 0.9,
        (QueryType::Justify, L0) => 0.9,
        (QueryType::Justify, L1) => 0.85,
        (QueryType::Justify, L2) => 0.95,
        (QueryType::Justify, L3) => 0.9,
        (QueryType::Recompute, L0) => 0.7,
        (QueryType::Recompute, L1) => 0.5,
        (QueryType::Recompute, L2) => 0.8,
        (QueryType::Recompute, L3) => 1.0,
        (QueryType::Summarize, L0) => 0.7,
        (QueryType::Summarize, L1) => 1.0,
        (QueryType::Summarize, L2) => 0.85,
        (QueryType::Summarize, L3) => 0.6,
        (QueryType::Open, L0) => 0.7,
        (QueryType::Open, L1) => 0.85,
        (QueryType::Open, L2) => 1.0,
        (QueryType::Open, L3) => 0.8,
    }
}

/// The named terms of `U(x)`, kept so a plan can be audited term by term.
#[derive(Debug, Clone, Copy, Default)]
pub struct UtilityTerms {
    pub similarity: f32,
    pub proximity: f32,
    pub explain_path: f32,
    pub confidence: f32,
    pub read_through: f32,
    pub recency: f32,
    pub kind_prior: f32,
    pub contradiction: f32,
    pub stale: f32,
    pub superseded_source: f32,
}

impl UtilityTerms {
    /// All weight in one term, for the constant-cost scorer control. Not a
    /// scoring mode — it exists so the stage can be removed without disturbing
    /// the candidate set the other stages see.
    pub fn constant(value: f32) -> Self {
        Self {
            similarity: value,
            ..Default::default()
        }
    }

    pub fn base(&self) -> f32 {
        self.similarity
            + self.proximity
            + self.explain_path
            + self.confidence
            + self.read_through
            + self.recency
            + self.kind_prior
            + self.contradiction
            + self.stale
            + self.superseded_source
    }

    pub fn named(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("similarity", self.similarity),
            ("proximity", self.proximity),
            ("explain_path", self.explain_path),
            ("confidence", self.confidence),
            ("read_through", self.read_through),
            ("recency", self.recency),
            ("kind_prior", self.kind_prior),
            ("contradiction", self.contradiction),
            ("stale", self.stale),
            ("superseded_source", self.superseded_source),
        ]
    }
}

/// Per-stage clocks and rejection counts for one plan.
///
/// Proposed by [@cwahq](https://www.moltbook.com/post/78237a57-17ef-4c78-b05f-8c1e5a944196):
/// *"split candidate generation, scoring, and graph expansion into separate
/// clocks, then publish the rejected-candidate count at each stage… 'planning'
/// is still too broad to own the cost."*
///
/// Every latency figure this repo published before this struct existed was
/// taken at the outer edge of a turn, which makes planning a *residual* rather
/// than a measurement. A stage with no clock can absorb any amount of the
/// budget and never appear in a table, and the failure mode is not that it is
/// slow — it is that the table then reads as complete.
///
/// The rejection counts are the same argument at the correctness layer. A stage
/// that discards candidates without publishing how many is unfalsifiable in the
/// way `stale_fact_read_rate: 0.0` was: nothing about it can fail.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageProfile {
    // -- clocks ------------------------------------------------------------
    /// One clock around the whole `plan()` call, measured independently of the
    /// per-stage clocks below.
    ///
    /// Without this, `total()` is the *sum* of the stages, so "the stages
    /// account for the planner" is a tautology: work in an unclocked stage does
    /// not appear as a residual, it does not appear at all. That is a control
    /// that cannot fail, and it was sitting inside instrumentation built
    /// precisely because an unclocked stage had hidden a cost. A reader summed
    /// the published columns and asked what the leftover meant; the honest
    /// answer was that the leftover could not mean anything until this existed.
    pub measured_time: Duration,
    pub seed_time: Duration,
    pub expand_time: Duration,
    pub pin_time: Duration,
    pub score_time: Duration,
    pub knapsack_time: Duration,
    pub speculate_time: Duration,

    // -- seed --------------------------------------------------------------
    /// Hits the index returned before any of this stage's filters ran.
    pub seed_hits: usize,
    pub seed_dropped_floor: usize,
    pub seed_dropped_recency: usize,
    pub seed_dropped_status: usize,
    pub seed_from_spans: usize,
    pub seeds_kept: usize,

    // -- expand ------------------------------------------------------------
    pub expand_reached: usize,
    /// `max_candidates` stopped the BFS early. If this is true at large N then
    /// "planning cost grows with N" and "the cap is silently truncating" are
    /// the same observation, and only one of them is a latency problem.
    pub expand_capped: bool,

    // -- pin ---------------------------------------------------------------
    pub pinned_added: usize,
    /// Nodes `by_kind` visited to find them. Uncapped and linear in the graph.
    pub pin_scanned: usize,

    // -- score -------------------------------------------------------------
    pub score_considered: usize,
    pub score_dropped_cap: usize,
    pub score_dropped_superseded: usize,
    pub score_dropped_stale: usize,
    pub score_dropped_no_level: usize,
    /// Source spans concatenated while pricing the candidates of one plan.
    ///
    /// `Ladder::available` renders `raw_text` — every source span of the node,
    /// joined — just to ask whether the text exceeds 40 tokens. Corroboration
    /// collapses agreeing spans behind one node, so a node's span list grows
    /// with history, and this counter is what shows the scoring stage doing
    /// work linear in N over a candidate set that is capped at 120.
    pub score_spans_priced: usize,

    // -- knapsack ----------------------------------------------------------
    pub knapsack_candidates: usize,
    pub knapsack_dropped: usize,
    pub knapsack_demoted: usize,

    /// Rendering the admitted set after the knapsack has chosen it.
    ///
    /// This stage had no clock in the first version of this struct, which is
    /// the exact failure the struct exists to prevent.
    pub admit_time: Duration,
    pub admitted: usize,
}

impl std::ops::AddAssign for StageProfile {
    /// Sum stage costs across the re-plans of one turn, or across a whole run.
    /// `expand_capped` is a disjunction: the cap bound at least once.
    fn add_assign(&mut self, rhs: Self) {
        self.measured_time += rhs.measured_time;
        self.seed_time += rhs.seed_time;
        self.expand_time += rhs.expand_time;
        self.pin_time += rhs.pin_time;
        self.score_time += rhs.score_time;
        self.knapsack_time += rhs.knapsack_time;
        self.speculate_time += rhs.speculate_time;
        self.seed_hits += rhs.seed_hits;
        self.seed_dropped_floor += rhs.seed_dropped_floor;
        self.seed_dropped_recency += rhs.seed_dropped_recency;
        self.seed_dropped_status += rhs.seed_dropped_status;
        self.seed_from_spans += rhs.seed_from_spans;
        self.seeds_kept += rhs.seeds_kept;
        self.expand_reached += rhs.expand_reached;
        self.expand_capped |= rhs.expand_capped;
        self.pinned_added += rhs.pinned_added;
        self.pin_scanned += rhs.pin_scanned;
        self.score_considered += rhs.score_considered;
        self.score_dropped_cap += rhs.score_dropped_cap;
        self.score_dropped_superseded += rhs.score_dropped_superseded;
        self.score_dropped_stale += rhs.score_dropped_stale;
        self.score_dropped_no_level += rhs.score_dropped_no_level;
        self.knapsack_candidates += rhs.knapsack_candidates;
        self.knapsack_dropped += rhs.knapsack_dropped;
        self.knapsack_demoted += rhs.knapsack_demoted;
        self.score_spans_priced += rhs.score_spans_priced;
        self.admit_time += rhs.admit_time;
        self.admitted += rhs.admitted;
    }
}

impl StageProfile {
    /// Total time inside the planner. Named `planning` everywhere it is
    /// reported, so the residual can finally be checked against a sum.
    pub fn total(&self) -> Duration {
        self.seed_time
            + self.expand_time
            + self.pin_time
            + self.score_time
            + self.knapsack_time
            + self.admit_time
            + self.speculate_time
    }

    /// What the per-stage clocks failed to account for.
    ///
    /// Saturating, because the stages are sampled inside a call the outer clock
    /// brackets, so tiny negative values are measurement noise rather than
    /// findings. A residual that grows with N is an unclocked stage.
    pub fn residual(&self) -> Duration {
        self.measured_time.saturating_sub(self.total())
    }

    pub fn clocks(&self) -> [(&'static str, Duration); 7] {
        [
            ("seed", self.seed_time),
            ("expand", self.expand_time),
            ("pin", self.pin_time),
            ("score", self.score_time),
            ("knapsack", self.knapsack_time),
            ("admit", self.admit_time),
            ("speculate", self.speculate_time),
        ]
    }

    /// Candidates discarded per stage, in pipeline order.
    pub fn rejections(&self) -> [(&'static str, usize); 8] {
        [
            ("seed:floor", self.seed_dropped_floor),
            ("seed:recency", self.seed_dropped_recency),
            ("seed:status", self.seed_dropped_status),
            ("expand:capped", usize::from(self.expand_capped)),
            ("score:cap", self.score_dropped_cap),
            ("score:superseded", self.score_dropped_superseded),
            ("score:stale", self.score_dropped_stale),
            ("knapsack:dropped", self.knapsack_dropped),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub node: NodeIdx,
    pub node_id: String,
    pub kind: Kind,
    pub label: String,
    pub level: Level,
    pub cost: usize,
    pub utility: f32,
    pub reason: &'static str,
    pub pinned: bool,
    pub rendered: String,
    pub terms: UtilityTerms,
}

/// The tiny working set — `k` — plus everything needed to audit it.
#[derive(Debug, Clone, Default)]
pub struct ActiveContext {
    pub query: String,
    pub qtype: Option<QueryType>,
    pub budget: usize,
    pub entries: Vec<Admission>,
    pub tokens: usize,  // audit-allow: LM token count, not a credential
    pub seeds: Vec<NodeIdx>,
    pub dropped: Vec<NodeIdx>,
    pub demoted: Vec<(NodeIdx, Level, Level)>,
    pub prefetch: Vec<Prediction>,
    pub considered: usize,
    pub snapshot_version: u64,
    pub explain_paths: Vec<(NodeIdx, Vec<NodeIdx>)>,
    pub stale_seen: Vec<NodeIdx>,
    pub overflow: bool,
    /// Per-stage clocks and rejection counts. See [`StageProfile`].
    pub profile: StageProfile,
}

impl ActiveContext {
    pub fn node_ids(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.node_id.as_str()).collect()
    }

    pub fn nodes(&self) -> Vec<NodeIdx> {
        self.entries.iter().map(|e| e.node).collect()
    }

    pub fn level_of(&self, node_id: &str) -> Option<Level> {
        self.entries
            .iter()
            .find(|e| e.node_id == node_id)
            .map(|e| e.level)
    }

    /// The prompt text. Grouped by role, not by score, because a model reads a
    /// labelled state block far more reliably than a ranked list.
    pub fn render(&self) -> String {
        self.render_with_header(true)
    }

    pub fn render_with_header(&self, with_header: bool) -> String {
        let mut sorted: Vec<&Admission> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.utility
                .total_cmp(&a.utility)
                .then(a.node_id.cmp(&b.node_id))
        });

        let order = [
            (Kind::Goal, "GOALS"),
            (Kind::Constraint, "CONSTRAINTS"),
            (Kind::Claim, "FACTS (cached state)"),
            (Kind::Calculation, "COMPUTED"),
            (Kind::Decision, "DECISIONS"),
            (Kind::OpenQuestion, "OPEN QUESTIONS"),
            (Kind::Evidence, "EVIDENCE (raw spans)"),
        ];
        let mut blocks: Vec<String> = Vec::new();
        if with_header {
            blocks.push(format!(
                "# ACTIVE CONTEXT  (k={}/{} tokens, {} of {} candidates, query type: {})",
                self.tokens,
                self.budget,
                self.entries.len(),
                self.considered,
                self.qtype.map(QueryType::as_str).unwrap_or("open")
            ));
        }
        for (kind, title) in order {
            let group: Vec<&&Admission> = sorted.iter().filter(|e| e.kind == kind).collect();
            if group.is_empty() {
                continue;
            }
            blocks.push(format!("\n## {title}"));
            for entry in group {
                blocks.push(entry.rendered.clone());
            }
        }
        blocks.join("\n")
    }

    pub fn manifest(&self) -> Vec<(String, Level, usize, f32)> {
        self.entries
            .iter()
            .map(|e| (e.node_id.clone(), e.level, e.cost, e.utility))
            .collect()
    }
}

/// Everything the planner reads. Borrowed, never owned — the runtime owns the
/// components and hands out one view of them per turn.
pub struct PlanCtx<'a> {
    pub raw: &'a RawStore,
    pub graph: &'a MemoryGraph,
    pub index: &'a HybridIndex,
    pub ladder: &'a Ladder,
    pub policy: &'a DecisionPolicy,
    pub now: u64,
    pub evidence_by_span: &'a HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RelevancePlanner {
    pub weights: Weights,
    pub budget: usize,
    pub seed_k: usize,
    pub max_depth: usize,
    pub max_fanout: usize,
    pub max_candidates: usize,
    /// Seeds scoring below this fraction of the top hit are dropped. Without
    /// it, `seed_k` always returns `k` nodes however weak, and weak seeds are
    /// exactly the cheap-but-useless material a utility-per-token optimiser is
    /// happy to pack the window with.
    pub seed_min_ratio: f32,
    /// Drop candidates older than this fraction of the store's age span before
    /// scoring, rather than scoring everything.
    ///
    /// Suggested by a reader who measured a halving of latency at no accuracy
    /// cost on a 1.5k-node graph over a vector store. `0.0` disables it, which
    /// is the default — the trade is real but it is a correctness knob wearing a
    /// latency costume, and the probes it should break first are exactly the
    /// ones that reach furthest back: `old fact, never repeated` and
    /// `corrected fact (mid-history)`.
    pub recency_cutoff: f32,
    /// Causal control only. When true the utility function is replaced by a
    /// constant, leaving the candidate set, the corpus, the node count and every
    /// other stage untouched.
    ///
    /// Proposed by a reader as the intervention that "scoring is 85% of
    /// planning" needs: an 85% share is a correlation until you remove the
    /// stage and watch the curve. If latency collapses, scoring is the cause;
    /// if it persists, the share was real and the explanation incomplete.
    /// Answers are meaningless under this flag — it exists to move a clock, not
    /// to serve a query. Never enable in real use.
    pub stub_scorer: bool,
    /// Negative control only. When true, stale nodes are planned as if fresh
    /// instead of being skipped — used to prove `stale_fact_read_rate` can
    /// return nonzero, so that a zero in production is a fired guard rather
    /// than a guard that was never exercised. Never enable in real use.
    pub admit_stale: bool,
}

impl Default for RelevancePlanner {
    fn default() -> Self {
        Self {
            weights: Weights::default(),
            budget: 1200,
            seed_k: 8,
            max_depth: 3,
            max_fanout: 6,
            max_candidates: 120,
            seed_min_ratio: 0.5,
            recency_cutoff: 0.0,
            stub_scorer: false,
            admit_stale: false,
        }
    }
}

impl RelevancePlanner {
    /// A stable digest of every knob that decides which candidate is rejected
    /// and why.
    ///
    /// A reader's point: a receipt reading `guard fired 4/4` records that *a*
    /// rule rejected the adversarial candidate, not *which version* of it. Change
    /// a weight, a cap or the seed floor and every historical trace silently
    /// re-interprets — the output is identical and its meaning is not. Binding
    /// this to the result makes a pass attributable to a named, versioned rule
    /// rather than to "the policy as it stands today".
    ///
    /// Covers the utility weights, the seed floor, the depth and fan-out caps,
    /// the candidate cap and the two control flags. It does **not** cover the
    /// routing table or the extractor, both of which also change what gets
    /// rejected — so this narrows the gap rather than closing it, and the
    /// credits say so.
    pub fn policy_digest(&self) -> String {
        let w = &self.weights;
        let text = format!(
            "w:{:?}|floor:{}|depth:{}|fanout:{}|cands:{}|decay:{}|stale:{}",
            [
                w.similarity, w.proximity, w.explain_path, w.confidence,
                w.read_through, w.recency, w.stale_penalty, w.superseded_source,
                w.contradiction_bonus, w.decay,
            ],
            self.seed_min_ratio,
            self.max_depth,
            self.max_fanout,
            self.max_candidates,
            self.recency_cutoff,
            self.admit_stale,
        );
        crate::hash::sha256(text.as_bytes()).short(12)
    }

    pub fn plan(
        &self,
        ctx: &PlanCtx<'_>,
        speculator: Option<&mut Speculator>,
        query: &str,
        budget: Option<usize>,
        pin: &[NodeIdx],
        force_level: &HashMap<NodeIdx, Level>,
    ) -> ActiveContext {
        // Independent of every per-stage clock below, so that
        // `profile.residual()` can detect a stage nobody timed.
        let whole_call = Instant::now();
        let budget = budget.unwrap_or(self.budget);
        let qtype = ctx.policy.classify(query);
        let mut active = ActiveContext {
            query: query.to_string(),
            qtype: Some(qtype),
            budget,
            snapshot_version: ctx.graph.version,
            ..Default::default()
        };

        let query_vec = ctx.ladder.query_vector(query);

        let clock = Instant::now();
        let seeds = self.seed(ctx, query, &query_vec, &mut active);
        active.profile.seed_time = clock.elapsed();
        active.profile.seeds_kept = seeds.len();

        let seed_scores: HashMap<NodeIdx, f32> = seeds.iter().copied().collect();

        let clock = Instant::now();
        let mut distances = self.expand(ctx, &seeds, qtype, pin, &mut active);
        active.profile.expand_time = clock.elapsed();
        active.profile.expand_reached = distances.len();

        let clock = Instant::now();
        let pinned = self.pinned(ctx, qtype, &mut distances, pin, &mut active.profile);
        active.profile.pin_time = clock.elapsed();

        let clock = Instant::now();
        active.profile.score_dropped_cap = distances.len().saturating_sub(self.max_candidates);
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut preferred: HashMap<NodeIdx, Level> = HashMap::new();
        let mut scratch: HashMap<NodeIdx, (UtilityTerms, Level)> = HashMap::new();

        for &(idx, distance) in distances.iter().take(self.max_candidates) {
            active.profile.score_considered += 1;
            let node = ctx.graph.node(idx);
            active.profile.score_spans_priced += node.source_spans.len();
            if node.status == Status::Superseded {
                active.profile.score_dropped_superseded += 1;
                continue;
            }
            if node.status == Status::Stale && !pinned.contains(&idx) && !self.admit_stale {
                active.profile.score_dropped_stale += 1;
                active.stale_seen.push(idx);
                continue;
            }
            let available = ctx.ladder.available(ctx.raw, node);
            if available.is_empty() {
                active.profile.score_dropped_no_level += 1;
                continue;
            }
            let want = force_level
                .get(&idx)
                .copied()
                .unwrap_or_else(|| ctx.policy.preferred_level(node, qtype, &available));
            preferred.insert(idx, want);
            // Under the causal control the utility function is skipped entirely
            // and every other stage sees an identical candidate.
            let terms = if self.stub_scorer {
                UtilityTerms::constant(1.0)
            } else {
                self.utility_terms(
                    ctx,
                    (idx, node, distance),
                    &query_vec,
                    &seed_scores,
                    &active,
                )
            };
            let base = terms.base().max(0.01);
            let mut options: Vec<Choice> = Vec::new();
            for level in &available {
                if let Some(forced) = force_level.get(&idx) {
                    if level != forced {
                        continue;
                    }
                }
                options.push(Choice::new(
                    *level,
                    ctx.ladder.cost(ctx.raw, node, *level),
                    base * level_fit(qtype, *level),
                ));
            }
            if options.is_empty() {
                active.profile.score_dropped_no_level += 1;
                continue;
            }
            candidates.push(Candidate::new(idx, options, pinned.contains(&idx)));
            scratch.insert(idx, (terms, want));
        }

        active.profile.score_time = clock.elapsed();

        active.considered = candidates.len();
        active.profile.knapsack_candidates = candidates.len();
        let clock = Instant::now();
        let allocation = solve(&candidates, budget, None, &preferred);
        active.profile.knapsack_time = clock.elapsed();

        let clock = Instant::now();
        let mut chosen: Vec<(NodeIdx, Choice)> = allocation.chosen.into_iter().collect();
        chosen.sort_by_key(|(idx, _)| *idx);
        for (idx, option) in chosen {
            let node = ctx.graph.node(idx);
            let (terms, _) = scratch[&idx];
            let rendered = ctx.ladder.render(ctx.raw, node, option.level);
            let decision =
                ctx.policy
                    .decide(node, qtype, &ctx.ladder.available(ctx.raw, node), true);
            node.admits.set(node.admits.get() + 1);
            if option.level == Level::L0 {
                node.meta.l0_reads.set(node.meta.l0_reads.get() + 1);
            }
            active.entries.push(Admission {
                node: idx,
                node_id: node.id.clone(),
                kind: node.kind,
                label: node.label(),
                level: option.level,
                cost: option.cost,
                utility: option.utility,
                reason: decision.reason,
                pinned: pinned.contains(&idx),
                rendered,
                terms,
            });
        }
        active.profile.admit_time = clock.elapsed();
        active.profile.admitted = active.entries.len();

        active.tokens = allocation.tokens;
        active.dropped = allocation.dropped;
        active.demoted = allocation.demoted;
        active.overflow = allocation.overflow;
        active.profile.knapsack_dropped = active.dropped.len();
        active.profile.knapsack_demoted = active.demoted.len();

        if let Some(speculator) = speculator {
            let clock = Instant::now();
            self.speculate(ctx, speculator, &distances, &query_vec, &active);
            active.profile.speculate_time = clock.elapsed();
        }
        active.profile.measured_time = whole_call.elapsed();
        active
    }

    // -- steps -------------------------------------------------------------
    fn seed(
        &self,
        ctx: &PlanCtx<'_>,
        query: &str,
        query_vec: &[f32],
        active: &mut ActiveContext,
    ) -> Vec<(NodeIdx, f32)> {
        let mut hits = ctx
            .index
            .search(Namespace::Node, query, query_vec, self.seed_k);
        active.profile.seed_hits = hits.len();
        if let Some((_, top)) = hits.first().copied() {
            let floor = top * self.seed_min_ratio;
            let before = hits.len();
            hits.retain(|(_, score)| *score >= floor);
            active.profile.seed_dropped_floor = before - hits.len();
        }
        let mut seeds: Vec<(NodeIdx, f32)> = Vec::new();
        // Recency prefilter, off by default. Applied before scoring so it is a
        // cost saving rather than a re-ranking: anything older than the cutoff
        // never reaches the utility function at all.
        let age_floor = if self.recency_cutoff > 0.0 {
            let stamps: Vec<u64> = ctx.graph.nodes().iter().map(|n| n.timestamp).collect();
            match (stamps.iter().min(), stamps.iter().max()) {
                (Some(&lo), Some(&hi)) if hi > lo => {
                    Some(lo + ((hi - lo) as f32 * self.recency_cutoff) as u64)
                }
                _ => None,
            }
        } else {
            None
        };
        for (idx, score) in hits {
            let node = ctx.graph.node(idx);
            if let Some(floor) = age_floor {
                if node.timestamp < floor {
                    active.profile.seed_dropped_recency += 1;
                    continue;
                }
            }
            // Evidence whose every live dependent has been superseded supports
            // nothing current, and it still carries the old value verbatim. It
            // used to be admitted with a NOTE telling the model not to treat it
            // as current, which is only a guard if the model reads notes: on the
            // adversarial mutation set the superseded sentence out-scores the
            // correction on lexical overlap, so a matcher that ignores the note
            // answers from it every time. Exclude it from seeding for the same
            // reason a superseded claim is excluded. `corrected_by` evidence is
            // deliberately left in — it still supports live facts, so its note
            // has to do the work.
            if node.meta.superseded_source && !self.admit_stale {
                active.profile.seed_dropped_status += 1;
                active.stale_seen.push(idx);
                continue;
            }
            match node.status {
                Status::Superseded => {
                    active.profile.seed_dropped_status += 1;
                    continue;
                }
                // Never seed from a stale node without re-grounding it first —
                // unless the negative control has bypassed the guard.
                Status::Stale if !self.admit_stale => {
                    active.profile.seed_dropped_status += 1;
                    active.stale_seen.push(idx);
                }
                // Contradicted and unresolved nodes seed like fresh ones: they
                // are current, and they carry their marker into the window.
                status if status.is_live() || self.admit_stale => seeds.push((idx, score)),
                _ => {
                    active.profile.seed_dropped_status += 1;
                    continue;
                }
            }
        }
        // Fall back to the raw lexical index when state search finds nothing —
        // the material may be ingested but not yet extracted into state.
        if seeds.len() < 2 {
            for (span_idx, score) in
                ctx.index
                    .search(Namespace::Span, query, query_vec, self.seed_k)
            {
                let span_id = &ctx.raw.span(span_idx).id;
                if let Some(evidence) = ctx
                    .evidence_by_span
                    .get(span_id)
                    .and_then(|id| ctx.graph.idx_of(id))
                {
                    if !seeds.iter().any(|(i, _)| *i == evidence) {
                        active.profile.seed_from_spans += 1;
                        seeds.push((evidence, score * 0.8));
                    }
                }
            }
        }
        active.seeds = seeds.iter().map(|(i, _)| *i).collect();
        seeds
    }

    /// BFS over dependency edges, capped in depth and fan-out.
    fn expand(
        &self,
        ctx: &PlanCtx<'_>,
        seeds: &[(NodeIdx, f32)],
        qtype: QueryType,
        pin: &[NodeIdx],
        active: &mut ActiveContext,
    ) -> Vec<(NodeIdx, usize)> {
        let mut distances: Vec<(NodeIdx, usize)> = Vec::new();
        let mut seen: HashMap<NodeIdx, usize> = HashMap::new();
        let mut queue: VecDeque<(NodeIdx, usize)> = VecDeque::new();
        // The same rule `seed` applies, applied here too.
        //
        // Seeding excludes evidence whose every live dependent has been
        // superseded, because it still carries the old value verbatim and a
        // matcher that ignores notes will answer from it. Expansion used to
        // re-admit exactly those nodes through a dependency or dependent edge,
        // so the guard held only for material that arrived by lexical match and
        // was silently bypassed by material that arrived through the graph.
        //
        // That is a guard on one entrance of a room with two doors. It was
        // found by tightening an unrelated threshold, which changed which nodes
        // seeded, which changed which expansions ran, and let the excluded node
        // in through the second door on an adversarial corpus where the
        // superseded sentence is the better lexical match for the query.
        let admit_stale = self.admit_stale;
        let excluded = |idx: NodeIdx| -> bool {
            !admit_stale && ctx.graph.node(idx).meta.superseded_source
        };
        let push = |distances: &mut Vec<(NodeIdx, usize)>,
                    seen: &mut HashMap<NodeIdx, usize>,
                    idx: NodeIdx,
                    depth: usize| {
            if excluded(idx) {
                return false;
            }
            match seen.entry(idx) {
                Entry::Vacant(slot) => {
                    slot.insert(depth);
                    distances.push((idx, depth));
                    true
                }
                Entry::Occupied(_) => false,
            }
        };
        for &(idx, _) in seeds {
            if push(&mut distances, &mut seen, idx, 0) {
                queue.push_back((idx, 0));
            }
        }
        for &idx in pin {
            if idx < ctx.graph.len() && push(&mut distances, &mut seen, idx, 0) {
                queue.push_back((idx, 0));
            }
        }
        while let Some((idx, depth)) = queue.pop_front() {
            if distances.len() >= self.max_candidates {
                active.profile.expand_capped = true;
                continue;
            }
            if depth >= self.max_depth {
                continue;
            }
            let node = ctx.graph.node(idx);
            for dep in node.dependencies.iter().take(self.max_fanout) {
                if let Some(dep_idx) = ctx.graph.idx_of(dep) {
                    if push(&mut distances, &mut seen, dep_idx, depth + 1) {
                        queue.push_back((dep_idx, depth + 1));
                    }
                }
            }
            // One hop *up* to whatever was derived from this node, so a fact
            // brings its decision with it rather than arriving orphaned.
            if depth == 0 {
                for dependent in ctx
                    .graph
                    .neighbors(idx, None, false)
                    .into_iter()
                    .take(self.max_fanout)
                {
                    if push(&mut distances, &mut seen, dependent, 1) {
                        queue.push_back((dependent, self.max_depth));
                    }
                }
            }
            for other in ctx.graph.neighbors(idx, Some(EdgeType::Contradicts), true) {
                if ctx.graph.node(other).status != Status::Superseded {
                    push(&mut distances, &mut seen, other, depth + 1);
                }
            }
        }

        if qtype == QueryType::Justify {
            // A justification demands the whole audit path, whatever it costs.
            let seeds: Vec<NodeIdx> = active.seeds.iter().copied().take(2).collect();
            for seed in seeds {
                let path = ctx.graph.explain(seed, None);
                active.explain_paths.push((seed, path.nodes.clone()));
                for (i, node) in path.nodes.iter().enumerate() {
                    push(&mut distances, &mut seen, *node, i.min(self.max_depth));
                }
            }
        }
        distances
    }

    /// Goals and constraints are admitted every turn whether or not the query
    /// mentioned them — they are cheap, they bound what any answer may say, and
    /// a plan that silently drops a constraint is worse than one that costs
    /// twenty more tokens. They join the candidate set even when nothing in the
    /// query reached them.
    fn pinned(
        &self,
        ctx: &PlanCtx<'_>,
        qtype: QueryType,
        distances: &mut Vec<(NodeIdx, usize)>,
        pin: &[NodeIdx],
        profile: &mut StageProfile,
    ) -> HashSet<NodeIdx> {
        let mut pinned: HashSet<NodeIdx> = pin.iter().copied().collect();
        let mut known: HashSet<NodeIdx> = distances.iter().map(|(i, _)| *i).collect();
        for kind in ctx.policy.pinned_kinds(qtype) {
            // `by_kind` filters the whole node vector, so this stage visits the
            // entire graph once per pinned kind. It is the only uncapped step
            // in the planner, and `pin_scanned` is here so that fact appears in
            // a table rather than being inferred from a latency curve.
            profile.pin_scanned += ctx.graph.len();
            for idx in ctx.graph.by_kind(kind, true) {
                pinned.insert(idx);
                if known.insert(idx) {
                    profile.pinned_added += 1;
                    distances.push((idx, 1));
                }
            }
        }
        pinned
    }

    fn utility_terms(
        &self,
        ctx: &PlanCtx<'_>,
        (idx, node, distance): (NodeIdx, &Node, usize),
        query_vec: &[f32],
        seed_scores: &HashMap<NodeIdx, f32>,
        active: &ActiveContext,
    ) -> UtilityTerms {
        let w = &self.weights;
        let similarity = ctx.ladder.similarity(ctx.raw, node, query_vec).max(0.0);
        let seed_score = seed_scores.get(&idx).copied().unwrap_or(0.0);
        let proximity = w.decay.powi(distance as i32);
        let age = ctx.now.saturating_sub(node.timestamp).max(1) as f32;
        let recency = 1.0 / (1.0 + age.sqrt() / 8.0);
        let on_path = active
            .explain_paths
            .iter()
            .any(|(_, nodes)| nodes.contains(&idx));
        let contradicted = !ctx
            .graph
            .neighbors(idx, Some(EdgeType::Contradicts), true)
            .is_empty();
        UtilityTerms {
            similarity: w.similarity * similarity.max(seed_score),
            proximity: w.proximity * proximity,
            explain_path: if on_path { w.explain_path } else { 0.0 },
            confidence: w.confidence * node.confidence,
            read_through: w.read_through * node.read_rate(),
            recency: w.recency * recency,
            kind_prior: w.kind_prior(node.kind),
            contradiction: if contradicted {
                w.contradiction_bonus
            } else {
                0.0
            },
            stale: if node.status == Status::Stale {
                -w.stale_penalty
            } else {
                0.0
            },
            superseded_source: if node.meta.superseded_source {
                -w.superseded_source
            } else if !node.meta.corrected_by.is_empty() {
                -w.superseded_source / 3.0
            } else {
                0.0
            },
        }
    }

    fn speculate(
        &self,
        ctx: &PlanCtx<'_>,
        speculator: &mut Speculator,
        distances: &[(NodeIdx, usize)],
        query_vec: &[f32],
        active: &ActiveContext,
    ) {
        let admitted: HashSet<NodeIdx> = active.entries.iter().map(|e| e.node).collect();
        let scored: Vec<(NodeIdx, &Node, f32, f32)> = distances
            .iter()
            .filter(|(idx, _)| !admitted.contains(idx))
            .filter_map(|&(idx, distance)| {
                let node = ctx.graph.node(idx);
                if node.status == Status::Superseded {
                    return None;
                }
                Some((
                    idx,
                    node,
                    ctx.ladder.similarity(ctx.raw, node, query_vec),
                    self.weights.decay.powi(distance as i32),
                ))
            })
            .collect();
        let predictions = speculator.predict(&scored, ctx.now);
        speculator.prefetch(&predictions, ctx.graph.nodes(), ctx.ladder, ctx.raw);
    }

    // -- introspection -----------------------------------------------------
    pub fn explain_plan(&self, graph: &MemoryGraph, active: &ActiveContext) -> String {
        let mut lines = vec![
            format!(
                "query: {:?}  type={}  budget={}  used={}  candidates={}",
                active.query,
                active.qtype.map(QueryType::as_str).unwrap_or("open"),
                active.budget,
                active.tokens,
                active.considered
            ),
            format!(
                "seeds: {}",
                if active.seeds.is_empty() {
                    "(none)".to_string()
                } else {
                    active
                        .seeds
                        .iter()
                        .map(|&i| graph.node(i).id.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ),
        ];
        let mut entries: Vec<&Admission> = active.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.utility
                .total_cmp(&a.utility)
                .then(a.node_id.cmp(&b.node_id))
        });
        for entry in entries {
            let terms = entry
                .terms
                .named()
                .into_iter()
                .filter(|(_, v)| *v != 0.0)
                .map(|(name, v)| format!("{name}={v:+.2}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "  ADMIT {} {:<22} {} cost={:<4} U={:.2} {}({})",
                entry.node_id,
                entry.label,
                entry.level.as_str(),
                entry.cost,
                entry.utility,
                if entry.pinned { "[pinned] " } else { "" },
                terms
            ));
        }
        for (idx, wanted, got) in &active.demoted {
            lines.push(format!(
                "  DEMOTE {} {} -> {} (budget pressure)",
                graph.node(*idx).id,
                wanted.as_str(),
                got.as_str()
            ));
        }
        for idx in &active.dropped {
            lines.push(format!("  DROP   {}", graph.node(*idx).id));
        }
        for idx in &active.stale_seen {
            lines.push(format!(
                "  SKIP   {} (stale — needs revalidation)",
                graph.node(*idx).id
            ));
        }
        for prediction in &active.prefetch {
            lines.push(format!(
                "  PREFETCH {} p={:.2}",
                graph.node(prediction.node).id,
                prediction.p
            ));
        }
        lines.join("\n")
    }
}

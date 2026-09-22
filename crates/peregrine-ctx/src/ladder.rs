//! The representation ladder.
//!
//! ```text
//! L0 = raw bytes            exact, expensive
//! L1 = chunk summary        lossy prose, cheap
//! L2 = state object         the structured fact + provenance pointers
//! L3 = executable derivation / computed result
//! ```
//!
//! L2 deserves a note, because the wiki describes it as "semantic/state
//! vectors" and a vector is not something a model reads. Here the vector is
//! L2's *index key* and the structured state object is L2's *payload*:
//! admitting a node at L2 means the model sees
//! `server.ip = 10.0.9.7 [conf 0.90, spans …]` rather than the paragraph it
//! came from. That is the level the fact cache is built on, and it is why 27k
//! tokens of transcript collapse into a few hundred.
//!
//! Higher levels are built on first demand and cached on the node, so ingesting
//! a large document stays cheap until something actually asks about it.

use std::cell::Cell;

use crate::embed::{Embedder, HashingEmbedder, cosine};
use crate::nodes::{Kind, L0Sizes, Node, Origin, Status};
use crate::spans::RawStore;
use crate::summarize::ExtractiveSummarizer;
use crate::text::clip;
use crate::tokens::Estimator;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    L0,
    L1,
    L2,
    L3,
}

impl Level {
    pub const ALL: [Level; 4] = [Level::L0, Level::L1, Level::L2, Level::L3];

    pub fn as_str(self) -> &'static str {
        match self {
            Level::L0 => "L0",
            Level::L1 => "L1",
            Level::L2 => "L2",
            Level::L3 => "L3",
        }
    }

    pub fn parse(text: &str) -> Option<Level> {
        Level::ALL.into_iter().find(|l| l.as_str() == text)
    }

    /// Cost ordering used for demotion and escalation. L3 sits above L1 because
    /// a computed result is usually compact but had to be *derived*; L0 is
    /// always the most expensive admission.
    pub fn order(self) -> u8 {
        match self {
            Level::L2 => 0,
            Level::L1 => 1,
            Level::L3 => 2,
            Level::L0 => 3,
        }
    }

    pub fn cheaper_than(self, other: Level) -> bool {
        self.order() < other.order()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Builds {
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

impl std::fmt::Display for Builds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{L1: {}, L2: {}, L3: {}}}", self.l1, self.l2, self.l3)
    }
}

#[derive(Debug)]
pub struct Ladder {
    pub summarizer: ExtractiveSummarizer,
    pub estimator: Estimator,
    /// How text becomes a vector. Swap this to change the vector channel; the
    /// default is the bundled 256-dimensional hashing embedder.
    pub embedder: Box<dyn Embedder>,
    /// Collapse the ladder to a single level - the ablation that asks what the
    /// other three levels are actually buying.
    pub flatten_to_l2: bool,
    /// Times the L0 span concatenation has actually been performed.
    ///
    /// Deterministic, unlike a clock, which matters here: the cost this
    /// memoises is what made the planner super-linear, and on a shared machine
    /// the timings carry enough spread to hide the difference. This counter
    /// does not.
    l0_builds: Cell<usize>,
    /// Source spans concatenated by those builds.
    l0_build_spans: Cell<usize>,
    /// Whether [`Self::l0_sizes`] may reuse its memo.
    ///
    /// Exists so the memo can be turned off and the cost it removes measured
    /// rather than asserted. A speed-up nobody can make disappear on purpose
    /// is not a measured speed-up.
    pub memoise_l0: bool,
    builds: Cell<Builds>,
}

impl Default for Ladder {
    fn default() -> Self {
        Self {
            summarizer: ExtractiveSummarizer::default(),
            estimator: Estimator::default(),
            embedder: Box::new(HashingEmbedder::default()),
            flatten_to_l2: false,
            l0_builds: Cell::new(0),
            l0_build_spans: Cell::new(0),
            memoise_l0: true,
            builds: Cell::new(Builds::default()),
        }
    }
}

impl Ladder {
    pub fn builds(&self) -> Builds {
        self.builds.get()
    }

    fn note_build(&self, level: Level) {
        let mut builds = self.builds.get();
        match level {
            Level::L1 => builds.l1 += 1,
            Level::L2 => builds.l2 += 1,
            Level::L3 => builds.l3 += 1,
            Level::L0 => {}
        }
        self.builds.set(builds);
    }

    // -- materialisation ---------------------------------------------------
    pub fn raw_text(&self, raw: &RawStore, node: &Node) -> String {
        if node.source_spans.is_empty() {
            node.value.clone()
        } else {
            raw.text_of_many(&node.source_spans)
        }
    }

    /// Both L0 token counts, built once per node and reused. See [`L0Sizes`].
    ///
    /// This is the planner's dominant cost made cheap. `available()` and
    /// `cost()` are each called for every candidate of every plan, and both
    /// concatenated the node's whole span list — one to compare a length
    /// against 40, the other to price an admission that would usually be
    /// dropped. They are computed together because they share that
    /// concatenation.
    /// How many times the L0 span concatenation has run. See [`Self::l0_builds`].
    pub fn l0_builds(&self) -> usize {
        self.l0_builds.get()
    }

    /// Source spans concatenated across those builds — the work itself, rather
    /// than the number of times it was entered.
    pub fn l0_build_spans(&self) -> usize {
        self.l0_build_spans.get()
    }

    pub fn l0_sizes(&self, raw: &RawStore, node: &Node) -> L0Sizes {
        let key = (node.source_spans.len(), node.value.len());
        if self.memoise_l0 {
            if let Some(sizes) = node.level_cache.borrow().l0_sizes {
                if sizes.key == key {
                    return sizes;
                }
            }
        }
        self.l0_builds.set(self.l0_builds.get() + 1);
        self.l0_build_spans
            .set(self.l0_build_spans.get() + node.source_spans.len());
        let text = self.raw_text(raw, node);
        let sizes = L0Sizes {
            key,
            raw_tokens: self.estimator.count(&text),
            rendered_cost: self.estimator.count(&Self::l0_render(node, &text)),
        };
        node.level_cache.borrow_mut().l0_sizes = Some(sizes);
        sizes
    }

    fn l0_render(node: &Node, text: &str) -> String {
        format!("[{} L0 {}] \"{}\"", node.id, node.label(), text)
    }

    /// The payload the model sees for this node at this level.
    pub fn render(&self, raw: &RawStore, node: &Node, level: Level) -> String {
        match level {
            Level::L0 => Self::l0_render(node, &self.raw_text(raw, node)),
            Level::L1 => format!(
                "[{} L1 {}] {}",
                node.id,
                node.label(),
                self.summary(raw, node)
            ),
            Level::L2 => format!("[{} L2] {}", node.id, self.state_object(node)),
            Level::L3 => match self.result(node) {
                Some(value) => {
                    format!("[{} L3 {}] {}", node.id, node.label(), format_number(value))
                }
                None => String::new(),
            },
        }
    }

    pub fn summary(&self, raw: &RawStore, node: &Node) -> String {
        if let Some(cached) = node.level_cache.borrow().l1.clone() {
            return cached;
        }
        let built = self.summarizer.summarize(&self.raw_text(raw, node));
        node.level_cache.borrow_mut().l1 = Some(built.clone());
        self.note_build(Level::L1);
        built
    }

    pub fn vector(&self, raw: &RawStore, node: &Node) -> Vec<f32> {
        if let Some(cached) = node.level_cache.borrow().l2.clone() {
            return cached;
        }
        let raw_text = self.raw_text(raw, node);
        let head: String = raw_text.chars().take(400).collect();
        let basis = [
            node.key.clone().unwrap_or_default(),
            node.value.clone(),
            head,
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
        let built = self.embedder.embed(&basis);
        node.level_cache.borrow_mut().l2 = Some(built.clone());
        self.note_build(Level::L2);
        built
    }

    /// The compact structured form — the fact cache entry.
    ///
    /// The value is clipped: L2 is a *handle* on the material, and an evidence
    /// node whose value is a whole paragraph must still cost less at L2 than at
    /// L1, or demotion stops being a saving.
    pub fn state_object(&self, node: &Node) -> String {
        let value = clip(&node.value, 120);
        let mut bits = vec![match &node.key {
            Some(key) => format!("{key} = {value}"),
            None => value,
        }];
        if node.confidence < 0.999 {
            bits.push(format!("conf={:.2}", node.confidence));
        }
        if node.status != Status::Fresh {
            bits.push(format!("status={}", node.status.as_str()));
        }
        // Derived material says so. An inference or a hypothesis that reads
        // like an observation is the failure this label exists to prevent.
        if node.origin != Origin::Observed {
            bits.push(format!("origin={}", node.origin.as_str()));
        }
        if !node.source_spans.is_empty() {
            bits.push(format!("spans={}", join_head(&node.source_spans, 3)));
        } else if !node.dependencies.is_empty() {
            bits.push(format!("via={}", join_head(&node.dependencies, 3)));
        }
        if node.status.is_live() && !node.meta.contradicts.is_empty() {
            bits.push(format!(
                "CONTRADICTS={} \u{2014} adjudicate, do not pick blindly",
                join_head(&node.meta.contradicts, 2)
            ));
        }
        if node.meta.superseded_source {
            bits.push("NOTE=source of a superseded fact; do not treat as current".to_string());
        } else if !node.meta.corrected_by.is_empty() {
            bits.push(format!(
                "NOTE=contains a value corrected later; current value in {}",
                join_head(&node.meta.corrected_by, 2)
            ));
        }
        bits.join(" \u{b7} ")
    }

    /// The computed result, if one has been materialised.
    ///
    /// Running a derivation needs `&mut ExecutionLayer`, which rendering must
    /// not require — so materialisation happens in
    /// [`crate::runtime::Dcr::compute`] and revalidation, and this only reads
    /// what is already there.
    pub fn result(&self, node: &Node) -> Option<f64> {
        if let Some(cached) = node.level_cache.borrow().l3 {
            return Some(cached);
        }
        if node.kind == Kind::Calculation {
            return node.value.parse::<f64>().ok();
        }
        None
    }

    // -- costing -----------------------------------------------------------
    pub fn cost(&self, raw: &RawStore, node: &Node, level: Level) -> usize {
        // L0 is priced from the memo rather than by rendering: the render is a
        // full concatenation of the node's spans, and the planner prices every
        // candidate at every available level before the knapsack drops most of
        // them.
        if level == Level::L0 {
            return self.l0_sizes(raw, node).rendered_cost;
        }
        let text = self.render(raw, node, level);
        if text.is_empty() {
            0
        } else {
            self.estimator.count(&text)
        }
    }

    /// Which levels this node can actually be admitted at, cheapest first.
    pub fn available(&self, raw: &RawStore, node: &Node) -> Vec<Level> {
        if self.flatten_to_l2 {
            return vec![Level::L2];
        }
        let mut levels = vec![Level::L2];
        if !node.source_spans.is_empty() {
            levels.push(Level::L0);
            if self.l0_sizes(raw, node).raw_tokens > 40 {
                levels.push(Level::L1);
            }
        }
        if node.kind == Kind::Calculation || node.meta.derivation.is_some() {
            levels.push(Level::L3);
        }
        if node.kind == Kind::Evidence && !levels.contains(&Level::L0) {
            levels.push(Level::L0);
        }
        levels.sort_by_key(|l| l.order());
        levels.dedup();
        levels
    }

    pub fn similarity(&self, raw: &RawStore, node: &Node, query_vec: &[f32]) -> f32 {
        cosine(&self.vector(raw, node), query_vec)
    }

    pub fn query_vector(&self, query: &str) -> Vec<f32> {
        self.embedder.embed(query)
    }

    /// Speculative materialisation. Costs storage/compute budget, never
    /// attention budget.
    pub fn prewarm(&self, raw: &RawStore, node: &Node) -> usize {
        let mut built = 0;
        if node.level_cache.borrow().l1.is_none() && !node.source_spans.is_empty() {
            self.summary(raw, node);
            built += 1;
        }
        if node.level_cache.borrow().l2.is_none() {
            self.vector(raw, node);
            built += 1;
        }
        built
    }
}

fn join_head(items: &[String], take: usize) -> String {
    items
        .iter()
        .take(take)
        .cloned()
        .collect::<Vec<_>>()
        .join(",")
}

/// Integers print as integers; the demo's `2160` should not read `2160.0`.
pub fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

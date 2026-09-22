//! Typed state nodes — `S_t`, `C_t` and the leaves of `E_t`.
//!
//! The one rule worth restating: `source_spans` and `dependencies` are not
//! metadata, they are what make a node admissible at all. A cached fact with no
//! spans is a hallucination with a database row.

use std::cell::{Cell, RefCell};

use crate::ids::make_id;

pub type NodeIdx = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Evidence,
    Claim,
    Calculation,
    Decision,
    Goal,
    Constraint,
    OpenQuestion,
}

impl Kind {
    pub const ALL: [Kind; 7] = [
        Kind::Evidence,
        Kind::Claim,
        Kind::Calculation,
        Kind::Decision,
        Kind::Goal,
        Kind::Constraint,
        Kind::OpenQuestion,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Evidence => "evidence",
            Kind::Claim => "claim",
            Kind::Calculation => "calculation",
            Kind::Decision => "decision",
            Kind::Goal => "goal",
            Kind::Constraint => "constraint",
            Kind::OpenQuestion => "open_question",
        }
    }

    pub fn parse(text: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.as_str() == text)
    }

    /// Short prefix used in node ids.
    fn prefix(self) -> &'static str {
        &self.as_str()[..4]
    }
}

/// Where a node's content came from — its *epistemic origin*, which is
/// orthogonal to [`Kind`] (what role it plays) and [`Status`] (whether it is
/// still current).
///
/// The distinction exists so a reasoner is never handed a hypothesis dressed
/// as an observation. `server.ip = 10.0.9.7` extracted from a log line and
/// `server.ip = 10.0.9.7` guessed from two adjacent facts are the same string,
/// the same kind and the same status; they are not the same claim, and only
/// one of them should settle an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Origin {
    /// Read directly out of material this runtime ingested.
    #[default]
    Observed,
    /// Came from outside — another agent, a fetched document, a tool result.
    ExternallySourced,
    /// Produced by running a derivation over known inputs.
    Computed,
    /// Concluded by a model rather than read or computed.
    Inferred,
    /// Proposed, not established. Admissible for reasoning *about*, never as
    /// grounds.
    Hypothetical,
}

impl Origin {
    pub const ALL: [Origin; 5] = [
        Origin::Observed,
        Origin::ExternallySourced,
        Origin::Computed,
        Origin::Inferred,
        Origin::Hypothetical,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Observed => "observed",
            Origin::ExternallySourced => "externally-sourced",
            Origin::Computed => "computed",
            Origin::Inferred => "inferred",
            Origin::Hypothetical => "hypothetical",
        }
    }

    pub fn parse(text: &str) -> Option<Origin> {
        Origin::ALL.into_iter().find(|o| o.as_str() == text)
    }

    /// Whether material of this origin may be cited as grounds.
    ///
    /// Observed, external and computed material can support a conclusion;
    /// an inference or a hypothesis is a conclusion, and citing one as its own
    /// evidence is how a guess becomes a fact.
    pub fn is_grounds(self) -> bool {
        matches!(
            self,
            Origin::Observed | Origin::ExternallySourced | Origin::Computed
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Fresh,
    Stale,
    Superseded,
    /// Live, and in an unresolved disagreement with another live node. Both
    /// sides are kept: the contradiction is the finding, not an error to
    /// resolve by picking one.
    Contradicted,
    /// Live and open — an [`Kind::OpenQuestion`] nothing has answered yet.
    Unresolved,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Fresh => "fresh",
            Status::Stale => "stale",
            Status::Superseded => "superseded",
            Status::Contradicted => "contradicted",
            Status::Unresolved => "unresolved",
        }
    }

    pub fn parse(text: &str) -> Option<Status> {
        match text {
            "fresh" => Some(Status::Fresh),
            "stale" => Some(Status::Stale),
            "superseded" => Some(Status::Superseded),
            "contradicted" => Some(Status::Contradicted),
            "unresolved" => Some(Status::Unresolved),
            _ => None,
        }
    }

    /// Current, and usable without revalidation.
    ///
    /// Contradicted and unresolved nodes are live: they enter the window
    /// *carrying their marker*, because hiding a contradiction is worse than
    /// showing one. Stale needs re-grounding first; superseded is history.
    pub fn is_live(self) -> bool {
        matches!(
            self,
            Status::Fresh | Status::Contradicted | Status::Unresolved
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// evidence → claim
    Supports,
    /// calculation → inputs
    DerivedFrom,
    /// conflicting nodes, both retained
    Contradicts,
    /// replacement, preserving history
    Supersedes,
}

impl EdgeType {
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeType::Supports => "supports",
            EdgeType::DerivedFrom => "derived-from",
            EdgeType::Contradicts => "contradicts",
            EdgeType::Supersedes => "supersedes",
        }
    }

    pub fn parse(text: &str) -> Option<EdgeType> {
        match text {
            "supports" => Some(EdgeType::Supports),
            "derived-from" => Some(EdgeType::DerivedFrom),
            "contradicts" => Some(EdgeType::Contradicts),
            "supersedes" => Some(EdgeType::Supersedes),
            _ => None,
        }
    }

    /// Edges the planner and `explain()` walk. `contradicts`/`supersedes` are
    /// revision bookkeeping, not dependency.
    pub fn is_dependency(self) -> bool {
        matches!(self, EdgeType::Supports | EdgeType::DerivedFrom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub src: NodeIdx,
    pub dst: NodeIdx,
    pub kind: EdgeType,
}

/// A registered derivation plus the literal inputs it was called with.
#[derive(Debug, Clone, PartialEq)]
pub struct Derivation {
    pub name: String,
    pub inputs: Vec<(String, f64)>,
}

/// Typed replacement for the Python original's untyped `meta` dict.
#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    /// Extracted from an explicitly corrective statement ("Correction: …").
    pub corrective: bool,
    /// Stamped once grounding has been verified; makes the check inductive.
    pub grounded: bool,
    pub superseded_by: Option<String>,
    pub supersedes: Vec<String>,
    pub contradicts: Vec<String>,
    /// Evidence whose every live dependent has been superseded.
    pub superseded_source: bool,
    /// Evidence that still supports live facts but also carries a value that
    /// was later corrected; points at where the current value lives.
    pub corrected_by: Vec<String>,
    pub corroborations: u32,
    /// How often this node was admitted at L0, and whether that ever produced
    /// anything new — the de-escalation signal.
    pub l0_reads: Cell<u32>,
    pub l0_yield: Cell<u32>,
    pub deescalated: bool,
    pub derivation: Option<Derivation>,
    pub source: Option<String>,
}

/// Cached higher ladder levels. Built on first demand, never eagerly.
#[derive(Debug, Clone, Default)]
pub struct LevelCache {
    pub l1: Option<String>,
    pub l2: Option<Vec<f32>>,
    pub l3: Option<f64>,
    /// Sizes of this node's L0 material. See [`L0Sizes`].
    pub l0_sizes: Option<L0Sizes>,
}

/// The two token counts the planner needs from a node's raw spans, memoised.
///
/// Both were recomputed for every candidate of every plan, and computing
/// either means concatenating the node's entire span list. Corroboration grows
/// that list with history, so the planner's scoring stage was doing work linear
/// in N over a candidate set capped at 120 — which is how "planning cost grows
/// with N" survived a candidate cap that should have bounded it.
///
/// Neither count is persisted: both are memos of a derivable quantity, and a
/// stored one would have to be trusted against a span list that reloads
/// separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L0Sizes {
    /// `(source span count, value length)` — everything the counts below
    /// depend on. Stored so the memo invalidates itself when new corroboration
    /// lands or the value is revised, rather than going quietly stale.
    pub key: (usize, usize),
    /// Tokens in the concatenated span text. Decides whether L1 is offered.
    pub raw_tokens: usize,  // audit-allow: LM token count, not a credential
    /// Tokens in the *rendered* L0 admission. Prices it for the knapsack.
    pub rendered_cost: usize,
}

#[derive(Debug)]
pub struct Node {
    pub id: String,
    pub kind: Kind,
    /// Epistemic origin. Rendered whenever it is not `Observed`, so derived
    /// material is never silently indistinguishable from read material.
    pub origin: Origin,
    pub value: String,
    pub source_spans: Vec<String>,
    pub dependencies: Vec<String>,
    pub confidence: f32,
    pub timestamp: u64,
    pub status: Status,
    /// Subject key (e.g. `server.ip`). Two fresh claims sharing a key and
    /// disagreeing on value are a contradiction the indexer must flag.
    pub key: Option<String>,
    /// Interior mutability on purpose: materialising a cheaper representation
    /// is logically a read, and forcing `&mut` here would make the planner
    /// unable to hold the graph while pricing what is in it.
    pub level_cache: RefCell<LevelCache>,
    pub meta: NodeMeta,
    /// How often this node was admitted *and* cited. Feeds `U(x)`.
    pub reads: Cell<u32>,
    pub admits: Cell<u32>,
}

impl Node {
    /// Identity of the node's *content* — changes when the value or its inputs
    /// change, which is what invalidates memoised derivations.
    pub fn fingerprint(&self) -> String {
        let mut deps = self.dependencies.clone();
        deps.sort();
        let mut spans = self.source_spans.clone();
        spans.sort();
        make_id(
            "fp",
            &[
                self.kind.as_str(),
                &self.value,
                &deps.join(","),
                &spans.join(","),
            ],
        )
    }

    pub fn is_grounded(&self) -> bool {
        !self.source_spans.is_empty() || !self.dependencies.is_empty()
    }

    pub fn label(&self) -> String {
        match &self.key {
            Some(key) => format!("{}:{key}", self.kind.as_str()),
            None => self.kind.as_str().to_string(),
        }
    }

    pub fn read_rate(&self) -> f32 {
        let admits = self.admits.get();
        if admits == 0 {
            0.0
        } else {
            self.reads.get() as f32 / admits as f32
        }
    }
}

/// Builder for a node. `node_id` is derived from kind, key, value and spans —
/// the value is part of the identity, because two claims sharing a key but
/// disagreeing on the value are *different nodes* that contradict each other,
/// not one node overwriting the other.
pub struct NewNode {
    pub kind: Kind,
    pub origin: Origin,
    pub value: String,
    pub source_spans: Vec<String>,
    pub dependencies: Vec<String>,
    pub confidence: f32,
    pub key: Option<String>,
    pub meta: NodeMeta,
}

impl NewNode {
    pub fn new(kind: Kind, value: impl Into<String>) -> Self {
        Self {
            kind,
            origin: Origin::default(),
            value: value.into(),
            source_spans: Vec::new(),
            dependencies: Vec::new(),
            confidence: 1.0,
            key: None,
            meta: NodeMeta::default(),
        }
    }

    pub fn spans(mut self, spans: impl IntoIterator<Item = String>) -> Self {
        self.source_spans = spans.into_iter().collect();
        self
    }

    pub fn deps(mut self, deps: impl IntoIterator<Item = String>) -> Self {
        self.dependencies = deps.into_iter().collect();
        self
    }

    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    pub fn maybe_key(mut self, key: Option<String>) -> Self {
        self.key = key;
        self
    }

    pub fn meta(mut self, meta: NodeMeta) -> Self {
        self.meta = meta;
        self
    }

    pub fn origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    pub fn build(self) -> Node {
        let id = make_id(
            self.kind.prefix(),
            &[
                self.kind.as_str(),
                self.key.as_deref().unwrap_or(""),
                &self.value,
                &self.source_spans.join(","),
            ],
        );
        Node {
            id,
            kind: self.kind,
            origin: self.origin,
            value: self.value,
            source_spans: self.source_spans,
            dependencies: self.dependencies,
            confidence: self.confidence,
            timestamp: 0,
            status: if self.kind == Kind::OpenQuestion {
                Status::Unresolved
            } else {
                Status::Fresh
            },
            key: self.key,
            level_cache: RefCell::new(LevelCache::default()),
            meta: self.meta,
            reads: Cell::new(0),
            admits: Cell::new(0),
        }
    }
}

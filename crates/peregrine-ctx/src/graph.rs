//! The dynamic memory graph — `M_t = (R_t, S_t, C_t, E_t)` as a typed graph.
//!
//! Three invariants from the spec are enforced here rather than documented and
//! hoped for:
//!
//! 1. every non-evidence node reaches at least one evidence node via
//!    `dependencies` (unsourced facts are rejected at `upsert` time);
//! 2. nothing is deleted — revision is `supersedes`, conflict is `contradicts`;
//! 3. a `stale` node never enters the active context without revalidation.
//!
//! Invalidation cascades are *bounded*. An unbounded cascade from one small
//! correction can mark a whole subgraph stale and force a full re-derivation;
//! the implementation caps the eager cascade and defers the rest to a lazy
//! queue drained on read.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ids::Clock;
use crate::nodes::{Edge, EdgeType, Kind, Node, NodeIdx, Status};
use crate::spans::{RawStore, StoreError};

#[derive(Debug)]
pub enum DcrError {
    /// A node would enter the graph ungrounded.
    Provenance(String),
    UnknownNode(String),
    UnknownDerivation(String),
    Store(StoreError),
    Io(String),
    Parse(String),
}

impl std::fmt::Display for DcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DcrError::Provenance(m) => write!(f, "provenance error: {m}"),
            DcrError::UnknownNode(id) => write!(f, "unknown node: {id}"),
            DcrError::UnknownDerivation(name) => write!(f, "unknown derivation: {name}"),
            DcrError::Store(e) => write!(f, "{e}"),
            DcrError::Io(m) => write!(f, "io error: {m}"),
            DcrError::Parse(m) => write!(f, "parse error: {m}"),
        }
    }
}

impl std::error::Error for DcrError {}

impl From<StoreError> for DcrError {
    fn from(value: StoreError) -> Self {
        DcrError::Store(value)
    }
}

/// The audit path returned by [`MemoryGraph::explain`].
#[derive(Debug, Default, Clone)]
pub struct ExplainPath {
    pub root: NodeIdx,
    pub nodes: Vec<NodeIdx>,
    pub edges: Vec<Edge>,
    pub spans: Vec<String>,
    /// False when the walk hit the depth cap before reaching evidence — the
    /// audit-path-completeness metric in the evaluation design.
    pub complete: bool,
    pub truncated_at: Vec<NodeIdx>,
    pub depths: HashMap<NodeIdx, usize>,
}

#[derive(Debug)]
pub struct MemoryGraph {
    nodes: Vec<Node>,
    by_id: HashMap<String, NodeIdx>,
    out_edges: Vec<Vec<Edge>>,
    in_edges: Vec<Vec<Edge>>,
    by_key: HashMap<String, Vec<NodeIdx>>,
    /// Non-superseded nodes per key — the same pool `by_key(_, true)` filters
    /// to, maintained incrementally so the hot ingest checks (`duplicate`,
    /// `conflicts`) read O(live) instead of copying + sorting the whole
    /// append-only `by_key` bucket every call. Membership boundary is
    /// `status != Superseded` (so `Stale`/`Contradicted` stay in), matching
    /// exactly what `by_key(key, true)` returns.
    live_by_key: HashMap<String, Vec<NodeIdx>>,
    pub max_cascade: usize,
    pub max_depth: usize,
    /// Bumped on every mutation. The planner records it on an assembled
    /// context so a mid-turn consolidation can be detected.
    pub version: u64,
    invalidated_since: Vec<(u64, Vec<NodeIdx>)>,
    /// Deferred tail of a bounded cascade; drained lazily.
    pending_invalidation: VecDeque<NodeIdx>,
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            by_id: HashMap::new(),
            out_edges: Vec::new(),
            in_edges: Vec::new(),
            by_key: HashMap::new(),
            live_by_key: HashMap::new(),
            max_cascade: 64,
            max_depth: 8,
            version: 0,
            invalidated_since: Vec::new(),
            pending_invalidation: VecDeque::new(),
        }
    }

    // -- mutation ----------------------------------------------------------
    pub fn upsert(
        &mut self,
        node: Node,
        raw: &RawStore,
        clock: &Clock,
    ) -> Result<NodeIdx, DcrError> {
        self.upsert_inner(node, Some(raw), clock)
    }

    /// Insert a node exactly as persisted: no clock tick, no re-validation, no
    /// edge derivation. Only for restoring a store that was already validated
    /// on the way in — timestamps are load-bearing for staleness, so a restore
    /// must not renumber them.
    pub fn insert_restored(&mut self, node: Node) -> NodeIdx {
        let idx = self.nodes.len();
        if let Some(key) = node.key.clone() {
            self.by_key.entry(key.clone()).or_default().push(idx);
            // Restored nodes carry their persisted status; only live ones join
            // live_by_key, so a restored superseded node stays out — the
            // in-memory maintenance and the rebuild-from-disk agree.
            if node.status != Status::Superseded {
                self.live_by_key.entry(key).or_default().push(idx);
            }
        }
        self.by_id.insert(node.id.clone(), idx);
        self.nodes.push(node);
        self.out_edges.push(Vec::new());
        self.in_edges.push(Vec::new());
        idx
    }

    fn upsert_inner(
        &mut self,
        mut node: Node,
        raw: Option<&RawStore>,
        clock: &Clock,
    ) -> Result<NodeIdx, DcrError> {
        if let Some(raw) = raw {
            self.require_grounding(&node, raw)?;
        }
        node.meta.grounded = true;
        node.timestamp = clock.tick();

        let existing = self.by_id.get(&node.id).copied();
        let idx = match existing {
            Some(idx) => {
                let old = &self.nodes[idx];
                node.reads.set(old.reads.get());
                node.admits.set(old.admits.get());
                let changed = old.fingerprint() != node.fingerprint();
                let old_key = old.key.clone();
                let new_key = node.key.clone();
                self.nodes[idx] = node;
                self.reindex_key(idx, old_key.as_deref(), new_key.as_deref());
                // The replacement is always live; keep live_by_key in step.
                // reindex_key moved by_key on a key change but early-returns on
                // an equal key, so the equal-key resurrection case (re-upserting
                // over a superseded id) is handled here too.
                if old_key != new_key {
                    if let Some(k) = old_key.as_deref() {
                        if let Some(list) = self.live_by_key.get_mut(k) {
                            list.retain(|&i| i != idx);
                        }
                    }
                }
                if let Some(k) = new_key {
                    let list = self.live_by_key.entry(k).or_default();
                    if !list.contains(&idx) {
                        list.push(idx);
                    }
                }
                if changed {
                    self.invalidate(idx, false);
                }
                idx
            }
            None => {
                let idx = self.nodes.len();
                if let Some(key) = node.key.clone() {
                    self.by_key.entry(key.clone()).or_default().push(idx);
                    // A newly inserted node is always live (Fresh/Unresolved).
                    self.live_by_key.entry(key).or_default().push(idx);
                }
                self.by_id.insert(node.id.clone(), idx);
                self.nodes.push(node);
                self.out_edges.push(Vec::new());
                self.in_edges.push(Vec::new());
                idx
            }
        };

        let deps = self.nodes[idx].dependencies.clone();
        let kind = self.nodes[idx].kind;
        let status = self.nodes[idx].status;
        for dep in deps {
            if let Some(dep_idx) = self.by_id.get(&dep).copied() {
                let edge_type = if kind == Kind::Claim {
                    EdgeType::Supports
                } else {
                    EdgeType::DerivedFrom
                };
                self.add_edge(idx, dep_idx, edge_type);
                if status != Status::Superseded {
                    self.nodes[dep_idx].meta.superseded_source = false;
                }
            }
        }
        self.bump();
        Ok(idx)
    }

    fn require_grounding(&self, node: &Node, raw: &RawStore) -> Result<(), DcrError> {
        if node.kind == Kind::Evidence {
            if node.source_spans.is_empty() {
                return Err(DcrError::Provenance(
                    "evidence node must carry at least one source span".into(),
                ));
            }
        } else if !node.is_grounded() {
            return Err(DcrError::Provenance(format!(
                "{} {} has neither source spans nor dependencies; a cached fact with no \
                 provenance is a hallucination with a database row",
                node.kind.as_str(),
                node.id
            )));
        }
        let missing: Vec<&String> = node
            .source_spans
            .iter()
            .filter(|s| !raw.has_span(s))
            .collect();
        if !missing.is_empty() {
            return Err(DcrError::Provenance(format!(
                "unknown source spans: {missing:?}"
            )));
        }
        if node.kind != Kind::Evidence
            && node.source_spans.is_empty()
            && !node.dependencies.iter().any(|d| self.reaches_evidence(d))
        {
            return Err(DcrError::Provenance(format!(
                "{} {} does not reach any evidence node",
                node.kind.as_str(),
                node.id
            )));
        }
        Ok(())
    }

    /// O(1) via the grounding flag stamped at insert time.
    ///
    /// Grounding is inductive — a node is grounded if any dependency is — so it
    /// does not need a depth-limited walk, and must not have one: a
    /// legitimately deep derivation chain would be rejected at the cap.
    fn reaches_evidence(&self, node_id: &str) -> bool {
        match self.by_id.get(node_id) {
            Some(&idx) => {
                let node = &self.nodes[idx];
                node.kind == Kind::Evidence || !node.source_spans.is_empty() || node.meta.grounded
            }
            None => false,
        }
    }

    pub fn add_edge(&mut self, src: NodeIdx, dst: NodeIdx, kind: EdgeType) -> Edge {
        let edge = Edge { src, dst, kind };
        if self.out_edges[src].contains(&edge) {
            return edge;
        }
        self.out_edges[src].push(edge);
        self.in_edges[dst].push(edge);
        self.bump();
        edge
    }

    /// Revision without deletion: the old node stays, marked superseded.
    pub fn supersede(&mut self, old: NodeIdx, new: NodeIdx) {
        self.add_edge(new, old, EdgeType::Supersedes);
        let old_id = self.nodes[old].id.clone();
        let new_id = self.nodes[new].id.clone();
        self.set_status(old, Status::Superseded);
        self.nodes[old].meta.superseded_by = Some(new_id);
        self.nodes[new].meta.supersedes.push(old_id.clone());
        // Supersession *resolves* the contradiction, so the survivor stops
        // carrying the warning; the edge and the loser both stay for the audit.
        self.nodes[new].meta.contradicts.retain(|c| *c != old_id);
        let new_id = self.nodes[new].id.clone();
        self.nodes[old].meta.contradicts.retain(|c| *c != new_id);
        // …and stops *reading* as contested, once nothing live disagrees with
        // it. Leaving the survivor marked would make every correction look
        // like an open dispute forever.
        self.settle(new);
        self.invalidate(old, false);
        self.mark_orphaned_evidence(old);
        self.bump();
    }

    /// Flag evidence whose every live dependent has been superseded.
    ///
    /// The span itself stays true as a record — the log really did say
    /// 10.0.4.12 at the time — but admitting it next to the corrected fact is
    /// how a "solved" contradiction walks back into the window.
    fn mark_orphaned_evidence(&mut self, node: NodeIdx) {
        let deps = self.nodes[node].dependencies.clone();
        let corrected = self.nodes[node].meta.superseded_by.clone();
        for dep in deps {
            let Some(&dep_idx) = self.by_id.get(&dep) else {
                continue;
            };
            if self.nodes[dep_idx].kind != Kind::Evidence {
                continue;
            }
            let live = self.in_edges[dep_idx]
                .iter()
                .filter(|e| e.kind.is_dependency())
                .any(|e| self.nodes[e.src].status != Status::Superseded);
            self.nodes[dep_idx].meta.superseded_source = !live;
            if let Some(pointer) = corrected.clone() {
                let marks = &mut self.nodes[dep_idx].meta.corrected_by;
                if !marks.contains(&pointer) {
                    marks.push(pointer);
                }
            }
        }
    }

    /// Both sides retained with their evidence so the model can adjudicate.
    pub fn contradict(&mut self, a: NodeIdx, b: NodeIdx) {
        self.add_edge(a, b, EdgeType::Contradicts);
        self.add_edge(b, a, EdgeType::Contradicts);
        // Stamped on the node so the rendered state object carries the warning.
        // Two conflicting facts in the window without a marker is worse than
        // either one alone — the model has no way to know they disagree.
        let (a_id, b_id) = (self.nodes[a].id.clone(), self.nodes[b].id.clone());
        for (src, other) in [(a, b_id), (b, a_id)] {
            let marks = &mut self.nodes[src].meta.contradicts;
            if !marks.contains(&other) {
                marks.push(other);
            }
            // A node in a live disagreement is not `fresh`, and saying so in
            // the status is what lets the state itself report the dispute
            // rather than leaving it to be inferred from the edges.
            if self.nodes[src].status == Status::Fresh {
                self.set_status(src, Status::Contradicted);
            }
        }
    }

    /// Return a node to `fresh` once nothing live contradicts it any more.
    fn settle(&mut self, idx: NodeIdx) {
        if self.nodes[idx].status != Status::Contradicted {
            return;
        }
        let live = self.nodes[idx]
            .meta
            .contradicts
            .iter()
            .filter_map(|id| self.by_id.get(id))
            .any(|&other| self.nodes[other].status.is_live());
        if !live {
            self.set_status(idx, Status::Fresh);
        }
    }

    /// Mark a node and its transitive dependents stale, cascade-bounded.
    pub fn invalidate(&mut self, node: NodeIdx, include_self: bool) -> Vec<NodeIdx> {
        let mut marked: Vec<NodeIdx> = Vec::new();
        let mut seen: HashSet<NodeIdx> = HashSet::new();
        let mut queue: VecDeque<(NodeIdx, usize)> = VecDeque::new();
        if include_self {
            queue.push_back((node, 0));
        } else {
            for edge in self.in_edges[node]
                .iter()
                .filter(|e| e.kind.is_dependency())
            {
                queue.push_back((edge.src, 1));
            }
        }
        while let Some((current, depth)) = queue.pop_front() {
            if !seen.insert(current) {
                continue;
            }
            if marked.len() >= self.max_cascade || depth > self.max_depth {
                // Bounded cascade: the tail is revalidated lazily on read
                // instead of eagerly invalidating an arbitrarily large subgraph.
                self.pending_invalidation.push_back(current);
                continue;
            }
            if self.nodes[current].status == Status::Superseded {
                continue;
            }
            if self.nodes[current].kind != Kind::Evidence {
                self.set_status(current, Status::Stale);
                self.nodes[current].level_cache.borrow_mut().l3 = None;
            }
            marked.push(current);
            let dependents: Vec<NodeIdx> = self.in_edges[current]
                .iter()
                .filter(|e| e.kind.is_dependency())
                .map(|e| e.src)
                .collect();
            for src in dependents {
                queue.push_back((src, depth + 1));
            }
        }
        if !marked.is_empty() {
            let version = self.version;
            self.invalidated_since.push((version, marked.clone()));
            self.bump();
        }
        marked
    }

    /// Resume a deferred cascade tail. Called before planning.
    pub fn drain_pending(&mut self, limit: usize) -> Vec<NodeIdx> {
        let mut marked = Vec::new();
        for _ in 0..limit.min(self.pending_invalidation.len()) {
            if let Some(idx) = self.pending_invalidation.pop_front() {
                marked.extend(self.invalidate(idx, true));
            }
        }
        marked
    }

    /// Clear staleness once the node has been re-grounded or recomputed.
    pub fn revalidate(&mut self, idx: NodeIdx, clock: &Clock) {
        self.set_status(idx, Status::Fresh);
        self.nodes[idx].timestamp = clock.tick();
        self.bump();
    }

    fn reindex_key(&mut self, idx: NodeIdx, old_key: Option<&str>, new_key: Option<&str>) {
        if old_key == new_key {
            return;
        }
        if let Some(key) = old_key {
            if let Some(list) = self.by_key.get_mut(key) {
                list.retain(|i| *i != idx);
            }
        }
        if let Some(key) = new_key {
            self.by_key.entry(key.to_string()).or_default().push(idx);
        }
    }

    /// Set a node's status while keeping `live_by_key` consistent. Membership is
    /// `status != Superseded`; only crossing that boundary moves the index.
    fn set_status(&mut self, idx: NodeIdx, new: Status) {
        let old = self.nodes[idx].status;
        self.nodes[idx].status = new;
        if (old != Status::Superseded) == (new != Status::Superseded) {
            return; // no crossing of the live/superseded boundary
        }
        let Some(key) = self.nodes[idx].key.clone() else {
            return;
        };
        let list = self.live_by_key.entry(key).or_default();
        if new != Status::Superseded {
            if !list.contains(&idx) {
                list.push(idx);
            }
        } else {
            list.retain(|&i| i != idx);
        }
    }

    fn bump(&mut self) {
        self.version += 1;
    }

    // -- reads -------------------------------------------------------------
    pub fn node(&self, idx: NodeIdx) -> &Node {
        &self.nodes[idx]
    }

    pub fn node_mut(&mut self, idx: NodeIdx) -> &mut Node {
        &mut self.nodes[idx]
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn idx_of(&self, node_id: &str) -> Option<NodeIdx> {
        self.by_id.get(node_id).copied()
    }

    pub fn get(&self, node_id: &str) -> Option<&Node> {
        self.idx_of(node_id).map(|i| &self.nodes[i])
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn edge_count(&self) -> usize {
        self.out_edges.iter().map(Vec::len).sum()
    }

    pub fn out_edges(&self, idx: NodeIdx) -> &[Edge] {
        &self.out_edges[idx]
    }

    pub fn in_edges(&self, idx: NodeIdx) -> &[Edge] {
        &self.in_edges[idx]
    }

    pub fn neighbors(&self, idx: NodeIdx, kind: Option<EdgeType>, outgoing: bool) -> Vec<NodeIdx> {
        let edges = if outgoing {
            &self.out_edges[idx]
        } else {
            &self.in_edges[idx]
        };
        edges
            .iter()
            .filter(|e| kind.is_none_or(|k| e.kind == k))
            .map(|e| if outgoing { e.dst } else { e.src })
            .collect()
    }

    /// Nodes carrying `key`, oldest first. Superseded ones are excluded unless
    /// asked for — history is kept, not served.
    pub fn by_key(&self, key: &str, fresh_only: bool) -> Vec<NodeIdx> {
        let mut found: Vec<NodeIdx> = self
            .by_key
            .get(key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|&i| !fresh_only || self.nodes[i].status != Status::Superseded)
            .collect();
        found.sort_by_key(|&i| self.nodes[i].timestamp);
        found
    }

    /// The non-superseded nodes for `key`, unsorted — the pool `by_key(key,
    /// true)` returns, without the per-call bucket copy + filter + sort.
    pub fn live_by_key(&self, key: &str) -> &[NodeIdx] {
        self.live_by_key.get(key).map(Vec::as_slice).unwrap_or(&[])
    }

    /// `live_by_key`, copied and timestamp-sorted — byte-identical to
    /// `by_key(key, true)` but over the O(live) set. For callers that observe
    /// order (e.g. `conflicts`).
    pub fn live_by_key_sorted(&self, key: &str) -> Vec<NodeIdx> {
        let mut found = self.live_by_key(key).to_vec();
        found.sort_by_key(|&i| self.nodes[i].timestamp);
        // Debug-only, and legitimately so: an internal-invariant check that runs
        // in debug test builds, NOT a control guarding a release-path claim. The
        // maintenance it verifies (`set_status` + the insert/restore hooks) is
        // identical in release, so a debug run that passes is sufficient; there
        // is nothing here that must fire under `--release`. Positional `==` on
        // two timestamp-sorted Vecs — it checks order, not just set membership.
        debug_assert_eq!(
            found,
            self.by_key(key, true),
            "live_by_key out of sync with by_key(key, true) for key {key:?}"
        );
        found
    }

    /// Every key currently indexed (live or superseded). Introspection/tests.
    pub fn keys(&self) -> Vec<&str> {
        self.by_key.keys().map(String::as_str).collect()
    }

    pub fn by_kind(&self, kind: Kind, fresh_only: bool) -> Vec<NodeIdx> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == kind && (!fresh_only || n.status == Status::Fresh))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn active(&self) -> Vec<NodeIdx> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.status != Status::Superseded)
            .map(|(i, _)| i)
            .collect()
    }

    /// Transitive closure down to evidence — the audit path.
    pub fn explain(&self, root: NodeIdx, max_depth: Option<usize>) -> ExplainPath {
        let max_depth = max_depth.unwrap_or(self.max_depth);
        let mut path = ExplainPath {
            root,
            complete: true,
            ..Default::default()
        };
        let mut seen: HashSet<NodeIdx> = HashSet::new();
        let mut queue: VecDeque<(NodeIdx, usize)> = VecDeque::from([(root, 0usize)]);
        let mut reached_evidence = false;
        while let Some((current, depth)) = queue.pop_front() {
            if !seen.insert(current) {
                continue;
            }
            let node = &self.nodes[current];
            path.nodes.push(current);
            path.depths.insert(current, depth);
            for span in &node.source_spans {
                if !path.spans.contains(span) {
                    path.spans.push(span.clone());
                }
            }
            if node.kind == Kind::Evidence || !node.source_spans.is_empty() {
                reached_evidence = true;
            }
            if depth >= max_depth {
                if !node.dependencies.is_empty() {
                    path.complete = false;
                    path.truncated_at.push(current);
                }
                continue;
            }
            for edge in self.out_edges[current]
                .iter()
                .filter(|e| e.kind.is_dependency())
            {
                path.edges.push(*edge);
                queue.push_back((edge.dst, depth + 1));
            }
        }
        path.complete = path.complete && reached_evidence;
        path
    }

    /// Nodes invalidated at or after `version` — the consistency check.
    pub fn invalidated_since(&self, version: u64) -> HashSet<NodeIdx> {
        self.invalidated_since
            .iter()
            .filter(|(v, _)| *v >= version)
            .flat_map(|(_, nodes)| nodes.iter().copied())
            .collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_invalidation.len()
    }

    pub fn stats(&self) -> GraphStats {
        let mut by_kind: Vec<(Kind, usize)> = Vec::new();
        for kind in Kind::ALL {
            let count = self.nodes.iter().filter(|n| n.kind == kind).count();
            if count > 0 {
                by_kind.push((kind, count));
            }
        }
        GraphStats {
            nodes: self.nodes.len(),
            edges: self.edge_count(),
            by_kind,
            stale: self
                .nodes
                .iter()
                .filter(|n| n.status == Status::Stale)
                .count(),
            superseded: self
                .nodes
                .iter()
                .filter(|n| n.status == Status::Superseded)
                .count(),
            version: self.version,
            pending_invalidation: self.pending_invalidation.len(),
        }
    }

    /// Restore edges after loading a persisted store.
    pub fn restore_edge(&mut self, src: &str, dst: &str, kind: EdgeType) {
        if let (Some(s), Some(d)) = (self.idx_of(src), self.idx_of(dst)) {
            self.add_edge(s, d, kind);
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub by_kind: Vec<(Kind, usize)>,
    pub stale: usize,
    pub superseded: usize,
    pub version: u64,
    pub pending_invalidation: usize,
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kinds: Vec<String> = self
            .by_kind
            .iter()
            .map(|(k, c)| format!("{}: {c}", k.as_str()))
            .collect();
        write!(
            f,
            "nodes: {}, edges: {}, by_kind: {{{}}}, stale: {}, superseded: {}, version: {}, \
             pending_invalidation: {}",
            self.nodes,
            self.edges,
            kinds.join(", "),
            self.stale,
            self.superseded,
            self.version,
            self.pending_invalidation
        )
    }
}

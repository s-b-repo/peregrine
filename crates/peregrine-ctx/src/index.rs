//! Retrieval index over both text chunks and state nodes.
//!
//! Two signals, because neither alone is enough: BM25 finds exact identifiers
//! (`10.0.4.12`, `ERR_CONN_REFUSED`) that an embedding smears away, and the
//! vector side finds paraphrases that share no tokens with the query. Indexing
//! *state nodes* — not only text — is what makes graph-seeded retrieval work at
//! all, and it is the point where DCR stops looking like RAG.
//!
//! Span vectors are deliberately not built eagerly: spans get the cheap lexical
//! index at ingest, and only nodes (few, small) get vectors.

use std::collections::{HashMap, HashSet};

use crate::embed::{DIM, cosine, hashing_embed};
use crate::text::content_tokens;

/// Ties break on the item's own index, never on hash order — the plan a query
/// produces must not depend on which way a `HashMap` happened to iterate.
fn rank(mut scored: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

/// BM25 over whatever text it is given.
#[derive(Debug)]
pub struct LexicalIndex {
    k1: f32,
    b: f32,
    postings: HashMap<String, HashMap<usize, u32>>,
    lengths: HashMap<usize, usize>,
    total_len: usize,
}

impl Default for LexicalIndex {
    fn default() -> Self {
        Self {
            k1: 1.4,
            b: 0.72,
            postings: HashMap::new(),
            lengths: HashMap::new(),
            total_len: 0,
        }
    }
}

impl LexicalIndex {
    pub fn add(&mut self, id: usize, text: &str) {
        if self.lengths.contains_key(&id) {
            self.remove(id);
        }
        let tokens = content_tokens(text);
        if tokens.is_empty() {
            self.lengths.insert(id, 0);
            return;
        }
        let mut counts: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        for (token, count) in counts {
            self.postings.entry(token).or_default().insert(id, count);
        }
        self.lengths.insert(id, tokens.len());
        self.total_len += tokens.len();
    }

    pub fn remove(&mut self, id: usize) {
        let length = self.lengths.remove(&id).unwrap_or(0);
        self.total_len = self.total_len.saturating_sub(length);
        self.postings.retain(|_, posting| {
            posting.remove(&id);
            !posting.is_empty()
        });
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let n = self.lengths.len().max(1) as f32;
        let avg = if self.total_len == 0 {
            1.0
        } else {
            (self.total_len as f32 / n).max(1.0)
        };
        let mut scores: HashMap<usize, f32> = HashMap::new();
        let mut seen: Vec<String> = content_tokens(query);
        seen.sort();
        seen.dedup();
        for token in seen {
            let Some(posting) = self.postings.get(&token) else {
                continue;
            };
            let df = posting.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (&id, &tf) in posting {
                let length = self.lengths.get(&id).copied().unwrap_or(0).max(1) as f32;
                let tf = tf as f32;
                let denom = tf + self.k1 * (1.0 - self.b + self.b * length / avg);
                *scores.entry(id).or_insert(0.0) += idf * (tf * (self.k1 + 1.0)) / denom;
            }
        }
        rank(scores.into_iter().collect(), k)
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }
}

/// Random-hyperplane LSH over the hashing embedding.
///
/// The exact path scores every vector, so retrieval cost grows with the store
/// even though assembled context does not — "flat attention, linear retrieval",
/// the gap the cost model says has to close before `O(k + r)` means anything at
/// scale.
///
/// For cosine similarity the standard trick is a sign-random-projection
/// signature: the probability two vectors land on the same side of a random
/// hyperplane is `1 - theta/pi`, so agreeing signature bits estimate the angle.
/// Vectors are bucketed by signature and a query scores only its own bucket,
/// which is sub-linear when the store is larger than the bucket.
///
/// Planes are generated from a fixed seed by splitmix64, because everything in
/// this runtime has to be reproducible — an ablation cannot be attributed if two
/// runs of the same configuration differ.
///
/// Recall is protected two ways: each vector is indexed in several independent
/// tables, and a query whose buckets yield too few candidates falls back to the
/// exact scan rather than returning a short list. That makes the structure a
/// pruning step, never a source of silent misses.
#[derive(Debug)]
struct Lsh {
    planes: Vec<Vec<f32>>,
    tables: Vec<HashMap<u32, Vec<usize>>>,
}

const LSH_TABLES: usize = 8;
const LSH_BITS: usize = 12;
const LSH_SEED: u64 = 0x5DEE_CE66_D1CE_F00D;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Lsh {
    fn new(dim: usize) -> Self {
        let mut state = LSH_SEED;
        let planes = (0..LSH_TABLES * LSH_BITS)
            .map(|_| {
                (0..dim)
                    // uniform in [-1, 1); adequate for sign projection, and
                    // cheaper than a Gaussian we would only take the sign of.
                    .map(|_| (splitmix64(&mut state) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        Self {
            planes,
            tables: (0..LSH_TABLES).map(|_| HashMap::new()).collect(),
        }
    }

    fn signature(&self, table: usize, v: &[f32]) -> u32 {
        let mut sig = 0u32;
        for bit in 0..LSH_BITS {
            let plane = &self.planes[table * LSH_BITS + bit];
            let dot: f32 = v
                .iter()
                .zip(plane.iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot >= 0.0 {
                sig |= 1 << bit;
            }
        }
        sig
    }

    fn insert(&mut self, id: usize, v: &[f32]) {
        for t in 0..LSH_TABLES {
            let sig = self.signature(t, v);
            self.tables[t].entry(sig).or_default().push(id);
        }
    }

    fn remove(&mut self, id: usize, v: &[f32]) {
        for t in 0..LSH_TABLES {
            let sig = self.signature(t, v);
            if let Some(bucket) = self.tables[t].get_mut(&sig) {
                bucket.retain(|&other| other != id);
            }
        }
    }

    /// Candidate ids: the query's own bucket in every table, plus each bucket
    /// one bit away (multi-probe), which recovers most near-boundary misses
    /// without another table.
    fn candidates(&self, v: &[f32]) -> HashSet<usize> {
        let mut out = HashSet::new();
        for t in 0..LSH_TABLES {
            let sig = self.signature(t, v);
            if let Some(ids) = self.tables[t].get(&sig) {
                out.extend(ids.iter().copied());
            }
            for bit in 0..LSH_BITS {
                if let Some(ids) = self.tables[t].get(&(sig ^ (1 << bit))) {
                    out.extend(ids.iter().copied());
                }
            }
        }
        out
    }
}

#[derive(Debug)]
pub struct VectorIndex {
    vectors: HashMap<usize, Vec<f32>>,
    lsh: Lsh,
    /// Score every vector instead of pruning. Kept switchable so the two paths
    /// can be compared on the same corpus rather than argued about.
    pub exact: bool,
    /// Below this many vectors a full scan is already cheap and bucketing only
    /// costs recall, so the structure is bypassed.
    min_for_ann: usize,
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self {
            vectors: HashMap::new(),
            lsh: Lsh::new(DIM),
            exact: true,
            min_for_ann: 256,
        }
    }
}

impl VectorIndex {
    pub fn add(&mut self, id: usize, vector: Vec<f32>) {
        self.lsh.insert(id, &vector);
        self.vectors.insert(id, vector);
    }

    pub fn add_text(&mut self, id: usize, text: &str) {
        self.add(id, hashing_embed(text, DIM));
    }

    pub fn remove(&mut self, id: usize) {
        if let Some(v) = self.vectors.remove(&id) {
            self.lsh.remove(id, &v);
        }
    }

    pub fn search(&self, query_vec: &[f32], k: usize) -> Vec<(usize, f32)> {
        let score = |ids: Box<dyn Iterator<Item = usize> + '_>| -> Vec<(usize, f32)> {
            ids.filter_map(|id| {
                let v = self.vectors.get(&id)?;
                let s = cosine(query_vec, v);
                (s > 0.0).then_some((id, s))
            })
            .collect()
        };
        if self.exact || self.vectors.len() < self.min_for_ann {
            return rank(score(Box::new(self.vectors.keys().copied())), k);
        }
        let candidates = self.lsh.candidates(query_vec);
        // A bucket that cannot supply k results is not evidence that no match
        // exists, so fall back rather than return a short list. Pruning must
        // never be a source of silent misses.
        if candidates.len() < k.max(8) {
            return rank(score(Box::new(self.vectors.keys().copied())), k);
        }
        rank(score(Box::new(candidates.into_iter())), k)
    }

    /// How many vectors a query would actually score, for the scaling table.
    pub fn probe_width(&self, query_vec: &[f32], k: usize) -> usize {
        if self.exact || self.vectors.len() < self.min_for_ann {
            return self.vectors.len();
        }
        let c = self.lsh.candidates(query_vec).len();
        if c < k.max(8) { self.vectors.len() } else { c }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Namespace {
    /// State nodes — a hit here is a seed the planner can expand from.
    Node,
    /// Raw chunks — a hit here is material that must first be grounded.
    Span,
}

/// How the two channels' rankings are combined into one.
///
/// External writing on production retrieval favours reciprocal rank fusion over
/// a weighted score blend, on the grounds that scores from two channels are not
/// on a common scale and normalising by the top hit makes a channel's influence
/// depend on the *shape* of its score distribution: a channel with one dominant
/// hit has everything below it crushed toward zero, while a flat channel keeps
/// its whole list near its weight. Ranks have no such problem.
///
/// Kept as an option rather than a replacement. `Linear` is the default because
/// every published figure was measured on it, and `bench --fusion` measures what
/// the swap costs instead of assuming the newer method is better here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fusion {
    /// Max-normalised weighted blend of the two channels' scores.
    Linear,
    /// Reciprocal rank fusion: a hit contributes `w / (k0 + rank)`, so only its
    /// position in each channel matters. `k0` damps the top of the list — the
    /// usual value is 60.
    ///
    /// Note what this does to anything downstream that reads the score as a
    /// magnitude: `1/(60+1)` and `1/(60+12)` differ by 15%, so a rank-fused list
    /// is nearly flat. The planner's seed floor is `top * seed_min_ratio`, which
    /// is scale-free and therefore safe, but it stops *binding* under RRF. That
    /// is a real consequence of the method rather than a bug, and `bench
    /// --fusion` reports it.
    Rrf { k0: f32 },
}

/// Lexical + vector, score-normalised and blended.
#[derive(Debug)]
pub struct HybridIndex {
    node_lexical: LexicalIndex,
    node_vector: VectorIndex,
    span_lexical: LexicalIndex,
    span_vector: VectorIndex,
    pub lexical_weight: f32,
    /// How the two channels are combined. `Linear` by default.
    pub fusion: Fusion,
}

impl Default for HybridIndex {
    fn default() -> Self {
        Self {
            node_lexical: LexicalIndex::default(),
            node_vector: VectorIndex::default(),
            span_lexical: LexicalIndex::default(),
            span_vector: VectorIndex::default(),
            lexical_weight: 0.55,
            fusion: Fusion::Linear,
        }
    }
}

impl HybridIndex {
    pub fn add_node(&mut self, idx: usize, text: &str, vector: Vec<f32>) {
        self.node_lexical.add(idx, text);
        self.node_vector.add(idx, vector);
    }

    pub fn remove_node(&mut self, idx: usize) {
        self.node_lexical.remove(idx);
        self.node_vector.remove(idx);
    }

    pub fn add_span(&mut self, idx: usize, text: &str) {
        self.span_lexical.add(idx, text);
    }

    /// Opt-in eager L2 over raw chunks; off by default (laziness rule).
    pub fn add_span_vector(&mut self, idx: usize, text: &str) {
        self.span_vector.add_text(idx, text);
    }

    pub fn search(
        &self,
        namespace: Namespace,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let (lexical, vector) = match namespace {
            Namespace::Node => (&self.node_lexical, &self.node_vector),
            Namespace::Span => (&self.span_lexical, &self.span_vector),
        };
        let lex = lexical.search(query, k * 3);
        let vec = vector.search(query_vec, k * 3);
        let mut combined: HashMap<usize, f32> = HashMap::new();
        match self.fusion {
            Fusion::Linear => {
                if let Some(top) = lex.first().map(|(_, s)| *s).filter(|s| *s > 0.0) {
                    for (id, score) in lex {
                        *combined.entry(id).or_insert(0.0) += self.lexical_weight * (score / top);
                    }
                }
                if let Some(top) = vec.first().map(|(_, s)| *s).filter(|s| *s > 0.0) {
                    for (id, score) in vec {
                        *combined.entry(id).or_insert(0.0) +=
                            (1.0 - self.lexical_weight) * (score / top);
                    }
                }
            }
            Fusion::Rrf { k0 } => {
                // A zero-scoring hit is not a ranked hit. The lexical channel
                // pads with zeros where nothing matched, and rewarding those by
                // position would fuse noise.
                for (pos, (id, score)) in lex.iter().enumerate() {
                    if *score <= 0.0 {
                        continue;
                    }
                    *combined.entry(*id).or_insert(0.0) +=
                        self.lexical_weight / (k0 + pos as f32 + 1.0);
                }
                for (pos, (id, score)) in vec.iter().enumerate() {
                    if *score <= 0.0 {
                        continue;
                    }
                    *combined.entry(*id).or_insert(0.0) +=
                        (1.0 - self.lexical_weight) / (k0 + pos as f32 + 1.0);
                }
                // Rescale to top = 1.0. Without this the fused scores land two
                // orders of magnitude below the linear path's, and every
                // downstream consumer that reads a score as a magnitude would
                // change behaviour for a reason that has nothing to do with
                // ranking. Rescaling keeps the comparison about order.
                if let Some(top) = combined.values().copied().fold(None::<f32>, |a, b| {
                    Some(a.map_or(b, |a: f32| a.max(b)))
                }) {
                    if top > 0.0 {
                        for v in combined.values_mut() {
                            *v /= top;
                        }
                    }
                }
            }
        }
        rank(combined.into_iter().collect(), k)
    }

    /// Both channels' rankings, unfused.
    ///
    /// `search` blends these and returns one list, which makes it impossible to
    /// ask how much the two channels *disagree* — and the docs claim they are
    /// "correlated rather than independent evidence" without ever having
    /// measured it. This returns them separately so `bench --channels` can.
    pub fn channels(
        &self,
        namespace: Namespace,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> (Vec<(usize, f32)>, Vec<(usize, f32)>) {
        let (lexical, vector) = match namespace {
            Namespace::Node => (&self.node_lexical, &self.node_vector),
            Namespace::Span => (&self.span_lexical, &self.span_vector),
        };
        (lexical.search(query, k), vector.search(query_vec, k))
    }

    /// Score every vector instead of pruning, on both node and span stores.
    /// Present so the exact and approximate paths can be measured against each
    /// other on the same corpus.
    pub fn set_exact(&mut self, exact: bool) {
        self.node_vector.exact = exact;
        self.span_vector.exact = exact;
    }

    /// How many node vectors a query would score, for the scaling table.
    pub fn node_probe_width(&self, query: &str, k: usize) -> (usize, usize) {
        let v = crate::embed::hashing_embed(query, crate::embed::DIM);
        (self.node_vector.probe_width(&v, k), self.node_vector.len())
    }

    pub fn span_probe_width(&self, query: &str, k: usize) -> (usize, usize) {
        let v = crate::embed::hashing_embed(query, crate::embed::DIM);
        (self.span_vector.probe_width(&v, k), self.span_vector.len())
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            nodes_indexed: self.node_lexical.len(),
            spans_indexed: self.span_lexical.len(),
            node_vectors: self.node_vector.len(),
            span_vectors: self.span_vector.len(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IndexStats {
    pub nodes_indexed: usize,
    pub spans_indexed: usize,
    pub node_vectors: usize,
    pub span_vectors: usize,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nodes_indexed: {}, spans_indexed: {}, node_vectors: {}, span_vectors: {}",
            self.nodes_indexed, self.spans_indexed, self.node_vectors, self.span_vectors
        )
    }
}

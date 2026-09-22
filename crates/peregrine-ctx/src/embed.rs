//! L2 vectors without a model dependency.
//!
//! A hashing embedder is enough for the runtime's actual requirement: making
//! state nodes findable by meaning-ish similarity so the planner has seeds to
//! expand from. Deterministic, dependency-free and instant, which keeps the
//! whole runtime runnable offline.

use std::collections::HashMap;

use crate::text::content_tokens;

pub const DIM: usize = 256;

/// FNV-1a over the token, so persisted vectors stay comparable across runs.
fn hash_index(token: &str, dim: usize) -> usize {
    let mut hash: u32 = 2_166_136_261;
    for byte in token.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    (hash as usize) % dim
}

/// Word + character-trigram hashing embedding, L2-normalised.
///
/// The trigrams give partial credit for morphological variants and typos,
/// which plain word hashing cannot do.
pub fn hashing_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    let tokens = content_tokens(text);
    if tokens.is_empty() {
        return vec;
    }
    let mut counts: HashMap<&str, u32> = HashMap::new();
    for token in &tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    for (token, count) in counts {
        let weight = 1.0 + (count as f32).ln();
        vec[hash_index(token, dim)] += weight;
        let chars: Vec<char> = token.chars().collect();
        if chars.len() >= 3 {
            for window in chars.windows(3) {
                let trigram: String = window.iter().collect();
                vec[hash_index(&trigram, dim)] += 0.35 * weight;
            }
        }
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

pub fn embed(text: &str) -> Vec<f32> {
    hashing_embed(text, DIM)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Anything that turns text into a vector.
///
/// The documentation has claimed since the first release that "the interface
/// accepts any `str -> Vec<f32>`". It did not: `Ladder` called
/// [`hashing_embed`] directly, so swapping the embedder meant editing the
/// runtime. This is that claim made true.
///
/// The default stays [`HashingEmbedder`], because the whole runtime is offline,
/// deterministic and dependency-free, and a learned embedder is none of those.
/// What changes is that replacing it is now a constructor argument rather than
/// a patch.
pub trait Embedder: std::fmt::Debug {
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Dimensionality. Vectors from one embedder are never comparable with
    /// another's, so this is checked rather than assumed where it matters.
    fn dim(&self) -> usize;
}

/// The bundled embedder: word + character-trigram hashing, 256 dimensions.
///
/// Fast, offline and reproducible across runs. It finds material that shares
/// vocabulary, which is why paraphrase retrieval is a documented poor fit —
/// `bench --channels` measures how much this channel actually agrees with the
/// lexical one rather than leaving it asserted.
#[derive(Debug, Clone, Copy)]
pub struct HashingEmbedder {
    pub dim: usize,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self { dim: DIM }
    }
}

impl Embedder for HashingEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        hashing_embed(text, self.dim)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

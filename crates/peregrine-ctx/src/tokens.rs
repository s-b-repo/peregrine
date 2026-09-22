//! Token accounting.
//!
//! `B_attention` is only meaningful if costs are measured in the same unit the
//! model bills in. The default estimator is a cheap character heuristic so the
//! runtime works offline; swap in a real tokenizer by setting
//! [`Estimator::Custom`].

pub const CHARS_PER_TOKEN: usize = 4;

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.len().div_ceil(CHARS_PER_TOKEN).max(1)
}

/// How the runtime prices text.
#[derive(Default)]
pub enum Estimator {
    /// `len / 4`, no network, good enough to keep a budget honest.
    #[default]
    Heuristic,
    /// Anything exact — a real tokenizer, or the Anthropic token-counting
    /// endpoint behind a cache.
    Custom(Box<dyn Fn(&str) -> usize>),
}

impl Estimator {
    pub fn count(&self, text: &str) -> usize {
        match self {
            Estimator::Heuristic => estimate_tokens(text),
            Estimator::Custom(f) => f(text),
        }
    }
}

impl std::fmt::Debug for Estimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Estimator::Heuristic => f.write_str("Estimator::Heuristic"),
            Estimator::Custom(_) => f.write_str("Estimator::Custom(..)"),
        }
    }
}

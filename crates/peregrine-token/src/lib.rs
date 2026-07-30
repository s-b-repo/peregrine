//! peregrine-token — vendored stable-toolchain subset of
//! [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) v0.10.0 (MIT):
//! the BPE engine (`bpe/`), the SIMD pretokenizers (`pretokenize/`, `std::arch`,
//! runtime-dispatched), and the HuggingFace `tokenizer.json` loader (`hf`).
//!
//! What the subset drops relative to upstream (and why):
//! - the SentencePiece engine (`portable_simd` → nightly-only); peregrine's
//!   serve layer falls back to the HF `tokenizers` crate for those models,
//! - the PyO3/numpy bindings (would link libpython into the serve binary),
//! - the batch/file/parquet/hub layers (peregrine feeds in-memory prompts).
//!
//! The public surface for peregrine is the small [`GigaTokenizer`] facade;
//! the vendored modules keep their upstream structure so future re-vendors
//! diff cleanly. Attribution: LICENSE-MIT-gigatoken at the crate root.

// Vendored code keeps upstream style; don't hold it to the engine crates'
// panic-free / clippy-pedantic gates (the facade below IS held to them).
// dead_code / unused_* fall out of trimming the SentencePiece + batch layers
// while keeping the remaining files upstream-verbatim.
#![allow(clippy::all)]
#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unused_mut)]

pub(crate) mod input;
pub mod bpe;
pub mod hf;
pub mod pretokenize;
#[cfg(test)]
pub(crate) mod test_hub;
pub mod tiktoken_load;
pub(crate) mod token;

pub use bpe::Tokenizer;
pub use token::TokenId;

/// The facade peregrine-serve consumes: load a BPE `tokenizer.json`, encode
/// with the memoizing cache (added/special tokens handled), decode back to
/// bytes. `encode` is `&mut` because the pretoken memo cache learns as it
/// goes — hold one instance for the process and repeated chat-template
/// prefixes encode from cache ("cross-request memo cache").
pub struct GigaTokenizer {
    inner: bpe::Tokenizer,
}

impl GigaTokenizer {
    /// Build from in-memory `tokenizer.json` bytes. Errors (with a
    /// descriptive message) on SentencePiece-style (`byte_fallback`) and
    /// non-BPE models — the caller falls back to the HF `tokenizers` crate.
    pub fn from_hf_json_bytes(data: &[u8]) -> Result<GigaTokenizer, String> {
        match hf::load_hf_slice(data) {
            Ok(hf::HfTokenizer::Bpe(t)) => Ok(GigaTokenizer { inner: t }),
            Err(e) => Err(format!("{e:#}")),
        }
    }

    /// Encode `text` to token ids (added/special tokens matched atomically,
    /// pretokens served from the memo cache when repeated).
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.inner.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
        out
    }

    /// Decode token ids back to bytes. Unknown ids are skipped (mirrors the
    /// lenient behavior serving needs — a stale id must not kill a stream).
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let toks: Vec<TokenId> = ids.iter().map(|&i| TokenId(i)).collect();
        self.inner.decode(&toks).collect()
    }

    /// A new tokenizer sharing the immutable model data with a fresh memo
    /// cache — for per-worker encoding without cache contention.
    pub fn fork(&self) -> GigaTokenizer {
        GigaTokenizer { inner: self.inner.fork() }
    }

    /// Vocabulary size (including added tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[cfg(test)]
mod facade_tests {
    use super::*;

    fn gpt2() -> GigaTokenizer {
        let path = crate::test_hub::gpt2_tokenizer_json();
        let bytes = std::fs::read(path).expect("committed GPT-2 fixture must exist");
        GigaTokenizer::from_hf_json_bytes(&bytes).expect("fixture is a BPE tokenizer.json")
    }

    #[test]
    fn encode_decode_round_trips_gpt2() {
        let mut t = gpt2();
        for text in [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "  multiple   spaces\tand\nnewlines ",
            "naïve façade — ünïcodé ✓ 🦀🦅",
            "fn main() { println!(\"{}\", 42); }",
        ] {
            let ids = t.encode(text);
            assert!(!ids.is_empty());
            let bytes = t.decode(&ids);
            assert_eq!(String::from_utf8_lossy(&bytes), text, "round trip for {text:?}");
        }
    }

    #[test]
    fn known_gpt2_ids() {
        // Anchor against the canonical GPT-2 encoding of "Hello world"
        // (well-known reference: [15496, 995]).
        let mut t = gpt2();
        assert_eq!(t.encode("Hello world"), vec![15496, 995]);
    }

    #[test]
    fn fork_shares_model_fresh_cache() {
        let mut a = gpt2();
        let ids1 = a.encode("repeated prefix repeated prefix");
        let mut b = a.fork();
        let ids2 = b.encode("repeated prefix repeated prefix");
        assert_eq!(ids1, ids2, "fork encodes identically");
        assert_eq!(a.vocab_size(), b.vocab_size());
    }

    #[test]
    fn memo_cache_is_deterministic_across_repeats() {
        let mut t = gpt2();
        let first = t.encode("The same sentence, encoded twice.");
        let second = t.encode("The same sentence, encoded twice.");
        assert_eq!(first, second, "cached encode == cold encode");
    }

    #[test]
    fn sentencepiece_json_is_refused_with_reason() {
        // byte_fallback=true marks SentencePiece-style models — the facade must
        // return a descriptive error (serve falls back to HF tokenizers).
        let sp_json = br#"{"model":{"type":"BPE","byte_fallback":true,"vocab":{},"merges":[]}}"#;
        let err = match GigaTokenizer::from_hf_json_bytes(sp_json) {
            Ok(_) => panic!("byte_fallback JSON must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("SentencePiece"), "got: {err}");
    }
}

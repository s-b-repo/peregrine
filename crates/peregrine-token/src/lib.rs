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
    /// Lazily-built forked workers for [`Self::encode_batch`], kept across
    /// calls so their construction (a zeroed, vocab-seeded cache table each)
    /// amortizes the way upstream's per-run batch workers do — and their
    /// memo caches stay warm call-to-call. Empty until the first batch call;
    /// costs nothing on the serve path.
    pool: Vec<bpe::Tokenizer>,
}

impl GigaTokenizer {
    /// Build from in-memory `tokenizer.json` bytes. Errors (with a
    /// descriptive message) on SentencePiece-style (`byte_fallback`) and
    /// non-BPE models — the caller falls back to the HF `tokenizers` crate.
    pub fn from_hf_json_bytes(data: &[u8]) -> Result<GigaTokenizer, String> {
        match hf::load_hf_slice(data) {
            Ok(hf::HfTokenizer::Bpe(t)) => Ok(GigaTokenizer { inner: t, pool: Vec::new() }),
            Err(e) => Err(format!("{e:#}")),
        }
    }

    /// Encode `text` to token ids (added/special tokens matched atomically,
    /// pretokens served from the memo cache when repeated).
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// [`Self::encode`] appending into a caller-owned buffer — the
    /// allocation-free entry point for bulk work. One call over a whole
    /// document runs ~3–4× faster than line-at-a-time calls: the SIMD
    /// pretokenizer state, span-batch machinery, and added-token scan are
    /// set up once per call, so short inputs are dominated by that fixed
    /// cost (ids are identical either way).
    pub fn encode_into(&mut self, text: &str, out: &mut Vec<u32>) {
        self.inner.encode_with_added_tokens_flat(text.as_bytes(), out);
    }

    /// Encode many texts in parallel with `workers` forked engines
    /// (upstream gigatoken's batch-layer shape, which the vendored subset
    /// otherwise drops). Outputs are in input order and id-for-id identical
    /// to serial [`Self::encode`] — the memo cache is pure memoization, so
    /// per-worker caches cannot change ids.
    ///
    /// Worker construction (a vocab-seeded cache table per fork) only pays
    /// for itself on bulk input, so small batches run serially on `self`:
    /// parallelism engages at ≥ 2 MB of input per worker. `workers` is a
    /// ceiling; pass `std::thread::available_parallelism()`. Workers persist
    /// on this instance after the first call (construction amortizes across
    /// calls, caches stay warm) — a `GigaTokenizer` that has done batch work
    /// holds that pool's memory until dropped.
    pub fn encode_batch(&mut self, texts: &[&str], workers: usize) -> Vec<Vec<u32>> {
        self.encode_batch_with(texts, workers, 2 << 20)
    }

    fn encode_batch_with(
        &mut self,
        texts: &[&str],
        workers: usize,
        min_bytes_per_worker: usize,
    ) -> Vec<Vec<u32>> {
        let total: usize = texts.iter().map(|t| t.len()).sum();
        let amortize_cap = if min_bytes_per_worker == 0 {
            texts.len().max(1)
        } else {
            (total / min_bytes_per_worker).max(1)
        };
        let n = workers.clamp(1, texts.len().max(1)).min(amortize_cap);
        if n <= 1 {
            return texts.iter().map(|t| self.encode(t)).collect();
        }

        // Contiguous chunks balanced by bytes (order-preserving).
        let target = total / n + 1;
        let mut bounds = vec![0usize];
        let mut acc = 0usize;
        for (i, t) in texts.iter().enumerate() {
            if acc >= target && bounds.len() < n {
                bounds.push(i);
                acc = 0;
            }
            acc += t.len();
        }
        bounds.push(texts.len());

        // Grow the persistent pool to n workers, each pre-sized for an even
        // share of this input (a capacity hint — tables still grow past it).
        // Construction happens once; later calls reuse the warm workers.
        while self.pool.len() < n {
            self.pool.push(self.inner.fork_sized(total / n));
        }

        let mut results: Vec<Option<Vec<Vec<u32>>>> = Vec::new();
        let chunks: Vec<&[&str]> = bounds.windows(2).map(|w| &texts[w[0]..w[1]]).collect();
        std::thread::scope(|s| {
            let handles: Vec<_> = self
                .pool
                .iter_mut()
                .zip(&chunks)
                .map(|(worker, chunk)| {
                    s.spawn(move || {
                        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(chunk.len());
                        let mut flat: Vec<u32> = Vec::new();
                        for t in *chunk {
                            flat.clear();
                            worker.encode_with_added_tokens_flat(t.as_bytes(), &mut flat);
                            ids.push(flat.clone());
                        }
                        ids
                    })
                })
                .collect();
            results = handles.into_iter().map(|h| h.join().ok()).collect();
        });

        let mut out: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        let mut poisoned = false;
        for (chunk, r) in chunks.iter().zip(results) {
            match r {
                Some(ids) => out.extend(ids),
                // A worker thread died (it cannot return an error, only
                // panic); redo its chunk serially so the call still returns
                // the full, correct id stream, and rebuild the pool next
                // call rather than reuse a worker of unknown state.
                None => {
                    poisoned = true;
                    out.extend(chunk.iter().map(|t| self.encode(t)));
                }
            }
        }
        if poisoned {
            self.pool.clear();
        }
        out
    }

    /// Decode token ids back to bytes. Ids outside the vocabulary are skipped:
    /// a model whose `lm_head` is padded past the tokenizer vocab can sample one,
    /// and the inner decoder indexes its vocab directly, so filtering here is
    /// what makes the documented lenient behavior true (a stale id must not kill
    /// a stream — under `panic = "abort"` it would kill the whole server).
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let vocab = self.inner.vocab_size() as u32;
        let toks: Vec<TokenId> = ids.iter().filter(|&&i| i < vocab).map(|&i| TokenId(i)).collect();
        self.inner.decode(&toks).collect()
    }

    /// A new tokenizer sharing the immutable model data with a fresh memo
    /// cache — for per-worker encoding without cache contention.
    pub fn fork(&self) -> GigaTokenizer {
        GigaTokenizer { inner: self.inner.fork(), pool: Vec::new() }
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
    fn decode_skips_out_of_vocab_ids() {
        // A model whose lm_head is padded past the tokenizer vocab can sample an
        // id the vocab has no entry for. The inner decoder indexes its vocab
        // directly, so without the facade's filter this aborts the process.
        let mut t = gpt2();
        let ids = t.encode("Hello world");
        let vocab = t.vocab_size() as u32;
        let mut padded = ids.clone();
        padded.push(vocab);
        padded.push(u32::MAX);
        assert_eq!(t.decode(&padded), t.decode(&ids), "out-of-range ids are skipped, not fatal");
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
    fn encode_into_matches_encode() {
        let mut t = gpt2();
        let text = "Hello world <|endoftext|> and ünïcode ✓ tails";
        let via_encode = t.encode(text);
        let mut via_into = vec![9999]; // pre-existing content must be preserved
        t.encode_into(text, &mut via_into);
        assert_eq!(via_into[0], 9999);
        assert_eq!(&via_into[1..], &via_encode[..]);
    }

    #[test]
    fn encode_batch_matches_serial_ids() {
        let mut t = gpt2();
        let texts: Vec<String> = (0..97)
            .map(|i| {
                format!(
                    "line {i}: the quick brown fox <|endoftext|> jumps {} times — naïve ✓",
                    i * 7 % 13
                )
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let serial: Vec<Vec<u32>> = refs.iter().map(|s| t.encode(s)).collect();
        // min_bytes_per_worker = 0 forces the parallel path on a tiny input.
        let parallel = t.encode_batch_with(&refs, 4, 0);
        assert_eq!(parallel, serial, "parallel batch must be id-for-id serial");
        // Second call reuses the now-warm persistent pool — still identical.
        let parallel_warm = t.encode_batch_with(&refs, 4, 0);
        assert_eq!(parallel_warm, serial, "warm-pool batch must be id-for-id serial");
        // The public gate keeps tiny inputs serial — ids identical either way.
        let gated = t.encode_batch(&refs, 4);
        assert_eq!(gated, serial);
        // Degenerate shapes.
        assert!(t.encode_batch(&[], 4).is_empty());
        assert_eq!(t.encode_batch_with(&["", "a"], 8, 0), vec![t.encode(""), t.encode("a")]);
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

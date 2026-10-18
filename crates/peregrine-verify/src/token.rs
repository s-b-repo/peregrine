//! The tokenizer group: losslessness, and the parallel path's agreement with the
//! serial one.
//!
//! Tokenization is the one place where a silent error rewrites the model's
//! input. A dropped byte, a merged space, or a batch encoder that disagrees with
//! the serial one produces different ids for the same text — and a model fed
//! different ids answers differently with no error anywhere.

use std::path::PathBuf;

use peregrine_token::GigaTokenizer;

use crate::{cert, exact, failed, with_cases, Certificate};

/// The corpus of round-trip fixtures. Deliberately includes the characters that
/// are the usual suspects: whitespace runs, newlines, tabs, combining marks,
/// emoji (multi-byte, outside the BMP), CJK, quotes and backslashes.
const TEXTS: [&str; 12] = [
    "",
    "hello world",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\r\nmixed",
    "punctuation: \"quoted\", 'single', `back`, \\backslash\\",
    "combining e\u{0301} vs precomposed \u{e9}",
    "emoji \u{1f680}\u{1f9ea} with ZWJ \u{1f469}\u{200d}\u{1f4bb}",
    "CJK \u{6f22}\u{5b57}\u{4e0c} mixed with latin",
    "repeated repeated repeated repeated repeated repeated",
    "the quick brown fox jumps over the lazy dog, again and again",
    "\u{0}\u{1f}\u{7f} control characters",
    "numbers 0123456789 -1.5e-9 +42 0xDEAD_BEEF",
];

/// The vendored GPT-2 BPE fixture that ships with `peregrine-token`. A
/// certificate that needs a tokenizer.json is only useful with a real one, and
/// this is the file the crate's own parity suite runs against.
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json")
}

fn load() -> Result<GigaTokenizer, String> {
    let path = fixture();
    let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    GigaTokenizer::from_hf_json_bytes(&bytes)
}

pub fn certificates() -> Vec<Certificate> {
    vec![encode_decode_is_lossless(), batch_encode_matches_serial(), ids_are_a_function_of_the_text()]
}

/// `encode` then `decode` must return the exact input bytes. Not "equivalent"
/// text — the same bytes — because the lossless property is what lets the serve
/// layer echo a prompt back verbatim and lets a request be re-tokenized after a
/// cache eviction without changing what the model sees.
fn encode_decode_is_lossless() -> Certificate {
    let mut tok = match load() {
        Ok(t) => t,
        Err(e) => {
            return cert(
                "token",
                "encode_then_decode_returns_the_input_bytes",
                "Every fixture text round-trips through `encode` → `decode` to the identical bytes, including whitespace runs, control characters, combining marks, emoji and CJK.",
                failed("load gpt2 fixture", &e),
                0.0,
            )
        }
    };
    let mut bad = 0.0f64;
    for text in TEXTS {
        let ids = tok.encode(text);
        let back = tok.decode(&ids);
        if back != text.as_bytes() {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "token",
            "encode_then_decode_returns_the_input_bytes",
            "Every fixture text round-trips through `encode` → `decode` to the identical bytes, including whitespace runs, control characters, combining marks, emoji and CJK.",
            bad,
            0.0,
        ),
        TEXTS.len(),
    )
}

/// The batch encoder forks one engine per worker, each with its own memo cache.
/// It claims to be id-for-id identical to the serial path because memoization is
/// pure — and that claim is exactly what a parallel path has to earn. A worker
/// that cached a pretoken differently would produce different ids for the same
/// text, and the symptom would be a request-dependent answer.
fn batch_encode_matches_serial() -> Certificate {
    let mut serial = match load() {
        Ok(t) => t,
        Err(e) => {
            return cert(
                "token",
                "batch_encode_is_id_for_id_serial",
                "`encode_batch` over four workers with the parallelism forced on produces exactly the ids `encode` produces, for every fixture text.",
                failed("load gpt2 fixture", &e),
                0.0,
            )
        }
    };
    // Repeat each fixture enough times to cross the byte-per-worker floor, so the
    // parallel branch is genuinely taken rather than short-circuited to serial.
    let mut corpus: Vec<&str> = Vec::new();
    for _ in 0..24 {
        for t in TEXTS {
            corpus.push(t);
        }
    }
    let batched = serial.encode_batch(&corpus, 4);
    let mut bad = 0.0f64;
    if batched.len() != corpus.len() {
        bad += 1.0;
    }
    for (text, ids) in corpus.iter().zip(batched.iter()) {
        if serial.encode(text) != *ids {
            bad += 1.0;
        }
    }
    with_cases(
        cert(
            "token",
            "batch_encode_is_id_for_id_serial",
            "`encode_batch` over four workers with the parallelism forced on produces exactly the ids `encode` produces, for every fixture text.",
            bad,
            0.0,
        ),
        corpus.len(),
    )
}

/// Encoding must be a function of the text alone: no state carried between calls
/// beyond memoization may change an id. The fixture deliberately encodes the same
/// string first alone and then between two others — the cross-request memo cache
/// is a real production path, and a cache that leaked state would make the answer
/// depend on what was tokenized before it.
fn ids_are_a_function_of_the_text() -> Certificate {
    let mut tok = match load() {
        Ok(t) => t,
        Err(e) => {
            return cert(
                "token",
                "encoding_does_not_depend_on_history",
                "Encoding the same text first, then between other texts, then again yields identical ids every time: the cross-request memo cache is pure memoization.",
                failed("load gpt2 fixture", &e),
                0.0,
            )
        }
    };
    let probe = "the memo cache must not change an id";
    let first = tok.encode(probe);
    for filler in TEXTS {
        tok.encode(filler);
    }
    let amid = tok.encode(probe);
    let again = tok.encode(probe);
    let mut fresh = match load() {
        Ok(t) => t,
        Err(e) => {
            return cert(
                "token",
                "encoding_does_not_depend_on_history",
                "Encoding the same text first, then between other texts, then again yields identical ids every time: the cross-request memo cache is pure memoization.",
                failed("load fresh gpt2 fixture", &e),
                0.0,
            )
        }
    };
    let virgin = fresh.encode(probe);
    let identical = first == amid && amid == again && first == virgin;
    with_cases(
        cert(
            "token",
            "encoding_does_not_depend_on_history",
            "Encoding the same text first, then between other texts, then again yields identical ids every time — and matches a freshly constructed tokenizer: the cross-request memo cache is pure memoization.",
            exact(identical),
            0.0,
        ),
        3,
    )
}
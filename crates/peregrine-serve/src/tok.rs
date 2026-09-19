//! Tokenizer: the vendored gigatoken BPE engine (`peregrine-token`) is the
//! **only** runtime tokenizer. Models whose `tokenizer.json` gigatoken cannot
//! load (SentencePiece / non-BPE flavors) fail at boot with a descriptive
//! error rather than silently degrading. Correctness is gated by the id-for-id
//! parity suite against the HF `tokenizers` oracle (dev-dependency only, see
//! `tests/tokenizer_parity.rs`).
//!
//! Forked gigatoken instances are process-persistent in [`TokenBackend`]'s
//! shard pool, so each shard's pretoken memo cache warms **across requests**
//! — a repeated chat-template prefix encodes from cache. One instance would
//! serialize every concurrent request on a single lock; the pool spreads
//! encodes round-robin instead (each shard's memo learns independently).
//! Streaming decode does NOT take a shard per token: each stream takes a
//! [`DecodeHandle`] (an `Arc` view of the immutable vocab) once and decodes
//! at token rate with no shared state — N concurrent streams would otherwise
//! contend at token frequency for what is a read of an immutable table.

use peregrine_core::{Context, Error};
use peregrine_token::{DecodeHandle, GigaTokenizer};

/// Shard count for [`TokenBackend`]: `COLI_TOKENIZER_WORKERS`, default the
/// machine's parallelism capped at 8. Each shard is a `fork` sharing the
/// immutable model data (`Arc`) with its own memo cache (~MBs), so the
/// default costs tens of MB against GBs of model weights. More shards absorb
/// bigger arrival bursts in parallel; fewer warm their memo caches faster
/// (each shard learns prefixes independently — same steady state, N× slower
/// warmup). Floor 1: a single shard is the historical behavior.
fn tokenizer_workers() -> usize {
    std::env::var("COLI_TOKENIZER_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get().min(8)).unwrap_or(4)
        })
        .max(1)
}

/// The process-wide tokenizer: a pool of forked [`GigaTokenizer`]s behind one
/// mutex each. `encode` is `&mut` (the pretoken memo cache learns), so one
/// instance serialized every concurrent request on a single lock — a burst of
/// B arrivals parked B blocking-pool threads for whole encodes. Forks share
/// the immutable model data and carry independent memos (pure memoization, so
/// ids are unaffected — see `encode_batch`'s identical guarantee), and a
/// round-robin cursor spreads concurrent encodes across shards. More
/// concurrent encodes than shards share one exactly like before: never wrong,
/// just briefly serial. Decode paths route through the same cursor; `decode`
/// itself is `&self`-immutable, so even the lock acquisition is the only
/// shared cost there.
pub struct TokenBackend {
    shards: Box<[parking_lot::Mutex<GigaTokenizer>]>,
    cursor: std::sync::atomic::AtomicUsize,
}

impl TokenBackend {
    /// Construct from the model dir's `tokenizer.json`. Logs the vocab size to
    /// stderr (a server boots once; the operator should see the tokenizer came
    /// up). Non-BPE models are a hard boot error by design.
    pub fn load(dir: &std::path::Path) -> Result<TokenBackend, Error> {
        let path = dir.join("tokenizer.json");
        // Through the ring like every other file the engine opens; the
        // helper falls back to `pread` on a host without io_uring.
        let bytes = peregrine_io::read_file(&path).ctx(|| path.display().to_string())?;
        match GigaTokenizer::from_hf_json_bytes(&bytes) {
            Ok(t) => {
                let n = tokenizer_workers();
                eprintln!("[tokenizer] gigatoken BPE active, vocab={}, encode_shards={n}", t.vocab_size());
                Ok(Self::sharded(t, n))
            }
            Err(e) => Err(Error::Format(format!(
                "gigatoken can't load this model's tokenizer.json \
                 (SentencePiece/non-BPE models are unsupported): {e}"
            ))),
        }
    }

    /// Pool `base` into `n` shards (shard 0 is `base` itself, the rest forks).
    /// `n` floors at 1. Forking is boot-time work; steady state pays nothing.
    fn sharded(base: GigaTokenizer, n: usize) -> TokenBackend {
        let mut shards = vec![parking_lot::Mutex::new(base)];
        while shards.len() < n.max(1) {
            // One-time at construction; contention here is impossible yet.
            let forked = shards[0].lock().fork();
            shards.push(parking_lot::Mutex::new(forked));
        }
        TokenBackend { shards: shards.into_boxed_slice(), cursor: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// Next shard, round-robin. Cursor overflow wraps (`fetch_add` wraps by
    /// definition); the modulo keeps every value a valid index.
    fn shard(&self) -> parking_lot::MutexGuard<'_, GigaTokenizer> {
        let i = self.cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.shards.len();
        self.shards[i].lock()
    }

    /// Encode `text` to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.shard().encode(text))
    }

    /// Decode token ids to text (whole-stream form, used by the non-streaming
    /// path). Streaming callers want [`IncrementalDecoder`] instead.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let bytes = self.shard().decode(ids);
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// A lock-free per-token decode view for one stream: one shard acquisition
    /// here, then [`DecodeHandle::token_bytes`] at token rate with no lock
    /// and no allocation. Token boundaries do not respect character
    /// boundaries, so streaming callers must join bytes through
    /// [`IncrementalDecoder`]. Out-of-vocab ids read as `None` (same lenient
    /// contract as [`Self::decode`]'s skip — a padded `lm_head` can sample one,
    /// and the inner decoder indexes its vocab directly, so filtering here is
    /// what makes the documented lenient behavior true (a stale id must not kill
    /// the stream).
    pub fn decode_handle(&self) -> DecodeHandle {
        self.shard().decode_handle()
    }

    /// The active tokenizer name (for logs / health output).
    pub fn name(&self) -> &'static str {
        "gigatoken"
    }

    /// Test-only constructor from an already-built tokenizer — the parity
    /// fixture path, so sptc tests exercise the real tokenizer without a
    /// model directory on disk.
    #[cfg(test)]
    pub(crate) fn from_giga_for_test(t: GigaTokenizer) -> TokenBackend {
        Self::sharded(t, 1)
    }
}

/// Turns a token-id stream into text deltas, one token at a time.
///
/// A multi-byte character can span two tokens, so decoding each token in
/// isolation would emit replacement characters, and re-decoding the whole
/// prefix per token (the previous approach) is both O(n²) and unsound: pass
/// *n* ends in a 3-byte `U+FFFD` that pass *n+1* replaces with the real 2- or
/// 4-byte character, so the previous text is no longer a byte-prefix of the new
/// one and slicing at its length lands mid-character — a panic, which under
/// `panic = "abort"` takes the whole server down.
///
/// Instead this buffers the undecodable tail: each token's bytes are appended,
/// the longest valid UTF-8 prefix is emitted, and an incomplete trailing
/// sequence is held back until the next token completes it. Bytes that can
/// never form valid UTF-8 are emitted as `U+FFFD` (matching lossy decoding)
/// rather than stalling the stream. Concatenating every delta reproduces the
/// lossy decode of the whole id sequence exactly.
#[derive(Default)]
pub struct IncrementalDecoder {
    /// Bytes received but not yet emitted — always a proper prefix of some
    /// multi-byte character (at most 3 bytes).
    pending: Vec<u8>,
}

impl IncrementalDecoder {
    pub fn new() -> IncrementalDecoder {
        IncrementalDecoder { pending: Vec::new() }
    }

    /// Feed one token's raw bytes; returns the text that is now complete
    /// (empty when the token only extended an unfinished character).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    // The valid prefix is always emittable.
                    if good > 0 {
                        out.push_str(&String::from_utf8_lossy(&self.pending[..good]));
                    }
                    match e.error_len() {
                        // Genuinely invalid bytes: emit one replacement char and
                        // keep scanning the remainder.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                        // Truncated but still-valid sequence: hold it for the
                        // next token to complete.
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// Flush any trailing bytes that never completed a character (end of
    /// stream), as replacement characters.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::{IncrementalDecoder, TokenBackend};
    use peregrine_token::GigaTokenizer;

    /// The committed GPT-2 fixture both parity suites read — a real BPE
    /// tokenizer small enough to fork repeatedly in a unit test.
    fn fixture_tokenizer() -> Result<GigaTokenizer, peregrine_core::Error> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| peregrine_core::Error::Format(format!("committed GPT-2 fixture: {e}")))?;
        GigaTokenizer::from_hf_json_bytes(&bytes)
            .map_err(|e| peregrine_core::Error::Format(format!("fixture is BPE: {e}")))
    }

    fn sharding_texts() -> Vec<String> {
        let prefix = "[gMASK]<sop><|user|>Summarize the following: ".to_string();
        vec![
            String::new(),
            "hello".to_string(),
            "Hello, world!".to_string(),
            "  leading whitespace\tand\nnewlines  ".to_string(),
            "naïve café — 你好 world! 🦅🦀".to_string(),
            prefix.clone() + &"The quick brown fox jumps over the lazy dog. ".repeat(20),
            // Same prefix twice: the second pass must hit the memo caches.
            prefix + "Second document, same template prefix.",
        ]
    }

    /// Sharded encodes must agree id-for-id with the single instance — the
    /// memo is pure memoization, so per-shard caches cannot change ids.
    /// Two rounds: the second exercises the warmed memo on every shard.
    #[test]
    fn sharded_encode_matches_single() -> Result<(), peregrine_core::Error> {
        let texts = sharding_texts();
        let single = TokenBackend::sharded(fixture_tokenizer()?, 1);
        let sharded = TokenBackend::sharded(fixture_tokenizer()?, 4);
        for _ in 0..2 {
            for t in &texts {
                let (a, b) = (single.encode(t)?, sharded.encode(t)?);
                assert_eq!(a, b, "shard divergence on {t:?}");
            }
        }
        // Decodes agree too (round-tripped through one shard's ids).
        let ids = single.encode(&texts[5])?;
        assert_eq!(sharded.decode(&ids)?, single.decode(&ids)?);
        Ok(())
    }

    /// Concurrent encodes on one shared backend agree with the sequential
    /// reference: round-robin handout shares no mutable state between shards.
    #[test]
    fn sharded_encode_agrees_concurrently() -> Result<(), peregrine_core::Error> {
        use std::sync::Arc;
        let texts = sharding_texts();
        let backend = Arc::new(TokenBackend::sharded(fixture_tokenizer()?, 4));
        let single = TokenBackend::sharded(fixture_tokenizer()?, 1);
        let reference: Vec<Vec<u32>> =
            texts.iter().map(|t| single.encode(t)).collect::<Result<_, _>>()?;
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let (b, texts) = (Arc::clone(&backend), &texts);
                handles.push(s.spawn(move || {
                    texts.iter().map(|t| b.encode(t)).collect::<Result<Vec<_>, _>>()
                }));
            }
            for h in handles {
                let got = h
                    .join()
                    .map_err(|_| peregrine_core::Error::Format("worker failed".into()))??;
                assert_eq!(got, reference, "concurrent encode diverged");
            }
            Ok::<_, peregrine_core::Error>(())
        })
    }

    /// Feeding bytes one at a time must reproduce the lossy whole-buffer decode.
    fn assert_streams_like_lossy(chunks: &[&[u8]]) {
        let mut d = IncrementalDecoder::new();
        let mut got = String::new();
        for c in chunks {
            got.push_str(&d.push(c));
        }
        got.push_str(&d.finish());
        let all: Vec<u8> = chunks.concat();
        assert_eq!(got, String::from_utf8_lossy(&all), "chunks {chunks:?}");
    }

    #[test]
    fn multibyte_char_split_across_tokens() {
        // The exact SSE crash: a 4-byte emoji arriving as two token payloads.
        let emoji = "🦅".as_bytes();
        assert_streams_like_lossy(&[&emoji[..2], &emoji[2..]]);
        // ...and byte-at-a-time, the worst case.
        assert_streams_like_lossy(&[&emoji[..1], &emoji[1..2], &emoji[2..3], &emoji[3..]]);
        // Nothing is emitted until the character is complete.
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(&emoji[..3]), "", "incomplete char must not be emitted early");
        assert_eq!(d.push(&emoji[3..]), "🦅");
    }

    #[test]
    fn cjk_and_mixed_text_stream_correctly() {
        let text = "你好, world! naïve — 🦀🦅 done";
        let bytes = text.as_bytes();
        // Split at every byte boundary; each split must still reconstruct.
        for cut in 0..bytes.len() {
            let mut d = IncrementalDecoder::new();
            let mut got = String::new();
            got.push_str(&d.push(&bytes[..cut]));
            got.push_str(&d.push(&bytes[cut..]));
            got.push_str(&d.finish());
            assert_eq!(got, text, "split at {cut}");
        }
    }

    #[test]
    fn invalid_bytes_become_replacement_not_a_stall() {
        // A byte that can never start a character must not wedge the stream.
        assert_streams_like_lossy(&[b"ok", &[0xFF], b"more"]);
        // A truncated sequence at end-of-stream flushes as replacement.
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(&[0xE4, 0xBD]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }

    #[test]
    fn ascii_passes_through_unbuffered() {
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(b"hello"), "hello");
        assert_eq!(d.push(b" world"), " world");
        assert_eq!(d.finish(), "");
    }
}

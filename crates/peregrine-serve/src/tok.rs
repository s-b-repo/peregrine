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
//! encodes across shards instead (each shard's memo learns independently,
//! try-lock cascade dodges busy shards rather than queueing behind them).
//! Decode takes no shard at all: [`TokenBackend`] holds one `Arc` view of
//! the immutable vocab, so `decode`/`decode_handle` never lock — N
//! concurrent streams and a long non-streaming decode proceed with no shared
//! state, and neither can delay (or be delayed by) an encode.

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
/// just briefly serial. Decode takes no shard at all: `decode_view` is a
/// clone-cheap `Arc` of the immutable vocabulary, so `decode`/`decode_handle`
/// never lock, never touch the atomic, and can never delay (or be delayed
/// by) an encode.
pub struct TokenBackend {
    shards: Box<[parking_lot::Mutex<GigaTokenizer>]>,
    cursor: std::sync::atomic::AtomicUsize,
    decode_view: DecodeHandle,
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
    /// The decode view is cloned from `base` up front so no decode path ever
    /// takes a shard lock (see the struct docs).
    fn sharded(base: GigaTokenizer, n: usize) -> TokenBackend {
        let decode_view = base.decode_handle();
        let mut shards = vec![parking_lot::Mutex::new(base)];
        while shards.len() < n.max(1) {
            // One-time at construction; contention here is impossible yet.
            let forked = shards[0].lock().fork();
            shards.push(parking_lot::Mutex::new(forked));
        }
        TokenBackend {
            shards: shards.into_boxed_slice(),
            cursor: std::sync::atomic::AtomicUsize::new(0),
            decode_view,
        }
    }

    /// Next free shard: probe from the round-robin cursor with non-blocking
    /// `try_lock` and take the first free one. Pure round-robin parks a
    /// request behind a slow encode on its assigned shard while other shards
    /// sit free (head-of-line blocking); the cascade only blocks when *every*
    /// shard is busy, and then on the cursor shard — the one this request
    /// would have taken historically. One probe pass, never a spin.
    fn shard(&self) -> parking_lot::MutexGuard<'_, GigaTokenizer> {
        let n = self.shards.len();
        let start = self.cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % n;
        for offset in 0..n {
            if let Some(guard) = self.shards[(start + offset) % n].try_lock() {
                return guard;
            }
        }
        self.shards[start].lock()
    }

    /// Encode `text` to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.shard().encode(text))
    }

    /// Decode token ids to text (whole-stream form, used by the non-streaming
    /// path). Streaming callers want [`IncrementalDecoder`] instead.
    ///
    /// Lock-free: concatenating the immutable vocab rows is byte-identical to
    /// `GigaTokenizer::decode` (same table, same skip-out-of-vocab filter),
    /// so a long non-streaming decode never occupies an encode shard.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Some(b) = self.decode_view.token_bytes(id) {
                bytes.extend_from_slice(b);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// A lock-free per-token decode view for one stream: one `Arc` clone here,
    /// then [`DecodeHandle::token_bytes`] at token rate with no lock, no
    /// atomic, and no allocation. Token boundaries do not respect character
    /// boundaries, so streaming callers must join bytes through
    /// [`IncrementalDecoder`]. Out-of-vocab ids read as `None` (same lenient
    /// contract as [`Self::decode`]'s skip — a padded `lm_head` can sample one,
    /// and the inner decoder indexes its vocab directly, so filtering here is
    /// what makes the documented lenient behavior true (a stale id must not kill
    /// the stream).
    pub fn decode_handle(&self) -> DecodeHandle {
        self.decode_view.clone()
    }

    /// Decode many id sequences at once, in input order, byte-for-byte
    /// identical to calling [`Self::decode`] per sequence. The batch is split
    /// into one contiguous chunk per shard and each chunk decodes on its own
    /// shard in a scoped thread — so shards (with their warm memo caches)
    /// decode in parallel while no chunk ever waits on another shard's lock.
    /// Small batches run serially on the cursor shard.
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        let n = batch.len();
        if n == 0 {
            return Vec::new();
        }
        let nshards = self.shards.len();
        if nshards <= 1 || n < 2 {
            let refs: Vec<&[u32]> = batch.iter().map(Vec::as_slice).collect();
            return self.shards[0].lock().decode_batch(&refs, 1).into_iter().map(|b| String::from_utf8_lossy(&b).into_owned()).collect();
        }
        let nchunks = nshards.min(n);
        let chunk = n.div_ceil(nchunks);
        let mut out: Vec<String> = Vec::with_capacity(n);
        out.resize_with(n, String::new);
        std::thread::scope(|s| {
            for (shard_i, (out_c, batch_c)) in out.chunks_mut(chunk).zip(batch.chunks(chunk)).enumerate() {
                let shard = &self.shards[shard_i % nshards];
                s.spawn(move || {
                    let guard = shard.lock();
                    let refs: Vec<&[u32]> = batch_c.iter().map(Vec::as_slice).collect();
                    let bytes = guard.decode_batch(&refs, 1);
                    for (dst, b) in out_c.iter_mut().zip(bytes) {
                        *dst = String::from_utf8_lossy(&b).into_owned();
                    }
                });
            }
        });
        out
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

    /// `decode` through the lock-free view is byte-identical to the
    /// tokenizer's own `decode` — same table, same out-of-vocab skip —
    /// including ids past the vocabulary (a padded `lm_head` sampling).
    #[test]
    fn decode_matches_tokenizer_decode() -> Result<(), peregrine_core::Error> {
        let mut tok = fixture_tokenizer()?;
        let backend = TokenBackend::sharded(fixture_tokenizer()?, 4);
        let mut ids = tok.encode("Hello, world! naïve 🦅");
        let oov = u32::try_from(tok.vocab_size())
            .map_err(|e| peregrine_core::Error::Format(format!("vocab size: {e}")))?
            .saturating_add(5);
        ids.push(oov);
        ids.push(u32::MAX);
        assert_eq!(backend.decode(&ids)?, String::from_utf8_lossy(&tok.decode(&ids)));
        // The per-stream handle reads the same rows the whole decode does.
        let handle = backend.decode_handle();
        let mut bytes = Vec::new();
        for &id in &ids {
            if (id as usize) < tok.vocab_size() {
                bytes.extend_from_slice(handle.token_bytes(id).ok_or_else(|| {
                    peregrine_core::Error::Format("handle dropped an in-vocab id".into())
                })?);
            } else {
                assert!(handle.token_bytes(id).is_none(), "OOV id {id} must read as None");
            }
        }
        assert_eq!(bytes, tok.decode(&ids));
        Ok(())
    }

    /// `decode_batch` fans out across shards but reads the same rows as
    /// `decode`: empty, OOV-bearing, and unicode sequences all match their
    /// single-decode reference in input order.
    #[test]
    fn decode_batch_matches_single_decodes() -> Result<(), peregrine_core::Error> {
        let mut tok = fixture_tokenizer()?;
        let backend = TokenBackend::sharded(fixture_tokenizer()?, 4);
        let oov = u32::try_from(tok.vocab_size())
            .map_err(|e| peregrine_core::Error::Format(format!("vocab size: {e}")))?
            .saturating_add(5);
        let mut batch: Vec<Vec<u32>> = sharding_texts().iter().map(|t| tok.encode(t)).collect();
        batch.push(Vec::new());
        batch.push(vec![oov, u32::MAX]);
        let reference: Vec<String> =
            batch.iter().map(|ids| backend.decode(ids)).collect::<Result<_, _>>()?;
        assert_eq!(backend.decode_batch(&batch), reference);
        // Single-sequence batch collapses to the same answer.
        assert_eq!(backend.decode_batch(&batch[..1]), reference[..1]);
        assert!(backend.decode_batch(&[]).is_empty());
        Ok(())
    }

    /// Concurrent encodes on one shared backend agree with the sequential
    /// reference: round-robin handout shares no mutable state between shards.
    #[test]
    fn sharded_encode_agrees_concurrently() -> Result<(), peregrine_core::Error> {        use std::sync::Arc;
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

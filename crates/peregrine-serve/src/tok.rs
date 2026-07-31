//! Tokenizer: the vendored gigatoken BPE engine (`peregrine-token`) is the
//! **only** runtime tokenizer. Models whose `tokenizer.json` gigatoken cannot
//! load (SentencePiece / non-BPE flavors) fail at boot with a descriptive
//! error rather than silently degrading. Correctness is gated by the id-for-id
//! parity suite against the HF `tokenizers` oracle (dev-dependency only, see
//! `tests/tokenizer_parity.rs`).
//!
//! The gigatoken instance is process-persistent behind a mutex, so its
//! pretoken memo cache warms **across requests** — a repeated chat-template
//! prefix encodes from cache. Encode is one short critical section per
//! request; decode takes the same lock briefly (streaming decodes are one
//! call per emitted token over a short id slice).

use peregrine_core::{Context, Error};
use peregrine_token::GigaTokenizer;

/// The process-wide tokenizer. `Mutex` because encode is `&mut` (the memo
/// cache learns); kept for the process so the cache persists across requests.
pub struct TokenBackend {
    giga: Box<parking_lot::Mutex<GigaTokenizer>>,
}

impl TokenBackend {
    /// Construct from the model dir's `tokenizer.json`. Logs the vocab size to
    /// stderr (a server boots once; the operator should see the tokenizer came
    /// up). Non-BPE models are a hard boot error by design.
    pub fn load(dir: &std::path::Path) -> Result<TokenBackend, Error> {
        let path = dir.join("tokenizer.json");
        let bytes = std::fs::read(&path).ctx(|| path.display().to_string())?;
        match GigaTokenizer::from_hf_json_bytes(&bytes) {
            Ok(t) => {
                eprintln!("[tokenizer] gigatoken BPE active, vocab={}", t.vocab_size());
                Ok(TokenBackend { giga: Box::new(parking_lot::Mutex::new(t)) })
            }
            Err(e) => Err(Error::Format(format!(
                "gigatoken can't load this model's tokenizer.json \
                 (SentencePiece/non-BPE models are unsupported): {e}"
            ))),
        }
    }

    /// Encode `text` to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.giga.lock().encode(text))
    }

    /// Decode token ids to text (whole-stream form, used by the non-streaming
    /// path). Streaming callers want [`IncrementalDecoder`] instead.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let bytes = self.giga.lock().decode(ids);
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Decode token ids to their raw bytes, without the lossy UTF-8 conversion.
    /// Token boundaries do not respect character boundaries, so a streaming
    /// caller must join the bytes itself before deciding what is a complete
    /// character.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        self.giga.lock().decode(ids)
    }

    /// The active tokenizer name (for logs / health output).
    pub fn name(&self) -> &'static str {
        "gigatoken"
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
    use super::IncrementalDecoder;

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

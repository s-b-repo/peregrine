//! Serving tokenizers share one interface. Gigatoken handles supported
//! ByteLevel BPE tokenizers and shares an immutable vocabulary for decoding.
//! The Hugging Face backend handles the other supported tokenizer.json formats.

use peregrine_core::{Context, Error};
use peregrine_token::{DecodeHandle, GigaTokenizer};

pub enum TokenBackend {
    Giga {
        encoder: Box<parking_lot::Mutex<GigaTokenizer>>,
        vocab: DecodeHandle,
    },
    HuggingFace(Box<tokenizers::Tokenizer>),
}

impl TokenBackend {
    pub fn load(dir: &std::path::Path) -> Result<TokenBackend, Error> {
        let path = dir.join("tokenizer.json");
        let bytes = peregrine_io::read_file(&path).ctx(|| path.display().to_string())?;
        Self::from_bytes(&bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<TokenBackend, Error> {
        match GigaTokenizer::from_hf_json_bytes(bytes) {
            Ok(encoder) => {
                let vocab = encoder.decode_handle();
                eprintln!("[tokenizer] gigatoken BPE active, vocab={}", encoder.vocab_size());
                Ok(Self::Giga { encoder: Box::new(parking_lot::Mutex::new(encoder)), vocab })
            }
            Err(fast_error) => {
                let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|error| {
                    Error::Format(format!("tokenizer.json: {error} (gigatoken: {fast_error})"))
                })?;
                eprintln!("[tokenizer] Hugging Face active, vocab={}", tokenizer.get_vocab_size(true));
                Ok(Self::HuggingFace(Box::new(tokenizer)))
            }
        }
    }

    /// Encode `text` to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        match self {
            Self::Giga { encoder, .. } => Ok(encoder.lock().encode(text)),
            // The server already renders the model's conversation template.
            Self::HuggingFace(tokenizer) => tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| Error::Format(format!("encode prompt: {error}"))),
        }
    }

    /// Decode token ids to text (whole-stream form, used by the non-streaming
    /// path). Streaming callers want [`IncrementalDecoder`] instead.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        match self {
            Self::Giga { vocab, .. } => {
                let mut bytes = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(piece) = vocab.token_bytes(*id) {
                        bytes.extend_from_slice(piece);
                    }
                }
                Ok(match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
                })
            }
            Self::HuggingFace(tokenizer) => tokenizer
                .decode(ids, false)
                .map_err(|error| Error::Format(format!("decode completion: {error}"))),
        }
    }

    pub fn piece(&self, id: u32) -> Option<String> {
        match self {
            Self::Giga { vocab, .. } => vocab.token_bytes(id)
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
            Self::HuggingFace(tokenizer) => tokenizer.id_to_token(id),
        }
    }

    pub fn stream(&self) -> StreamDecoder {
        match self {
            Self::Giga { vocab, .. } => StreamDecoder::Incremental(vocab.clone(), IncrementalDecoder::new()),
            // Some decoders revise the prefix when more tokens arrive, for
            // example when applying WordPiece punctuation cleanup.
            Self::HuggingFace(_) => StreamDecoder::Buffered,
        }
    }

    /// The active tokenizer name (for logs / health output).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Giga { .. } => "gigatoken",
            Self::HuggingFace(_) => "huggingface",
        }
    }
}

pub enum StreamDecoder {
    Incremental(DecodeHandle, IncrementalDecoder),
    Buffered,
}

impl StreamDecoder {
    pub fn push(&mut self, token: u32) -> String {
        match self {
            Self::Incremental(vocab, decoder) => decoder.push(vocab.token_bytes(token).unwrap_or(&[])),
            Self::Buffered => String::new(),
        }
    }

    pub fn finish(&mut self, tokenizer: &TokenBackend, ids: &[u32]) -> Result<String, Error> {
        match self {
            Self::Incremental(_, decoder) => Ok(decoder.finish()),
            Self::Buffered => tokenizer.decode(ids),
        }
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

    #[test]
    fn wordpiece_uses_its_sequence_decoder_for_both_formats() -> Result<(), peregrine_core::Error> {
        let tokenizer = TokenBackend::from_bytes(br###"{
            "version": "1.0",
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "!": 3}
            }
        }"###)?;
        assert_eq!(tokenizer.name(), "huggingface");
        let ids = tokenizer.encode("hello world!")?;
        assert_eq!(ids, [1, 2, 3]);
        let mut stream = tokenizer.stream();
        assert!(ids.iter().all(|&id| stream.push(id).is_empty()));
        let text = tokenizer.decode(&ids)?;
        assert_eq!(stream.finish(&tokenizer, &ids)?, text);
        assert_eq!(text, "hello world!");
        Ok(())
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

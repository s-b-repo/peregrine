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

    /// Decode token ids to text. The SSE path diffs consecutive decodes, so
    /// partial-UTF-8 at a token boundary is handled by the lossy conversion.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let bytes = self.giga.lock().decode(ids);
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The active tokenizer name (for logs / health output).
    pub fn name(&self) -> &'static str {
        "gigatoken"
    }
}

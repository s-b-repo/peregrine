//! Tokenizer backend selection: the vendored gigatoken BPE fast path
//! (`peregrine-token`) with the HuggingFace `tokenizers` crate as fallback.
//!
//! - `COLI_TOKENIZER=giga` — require gigatoken (hard error if the model's
//!   tokenizer.json isn't a supported BPE flavor);
//! - `COLI_TOKENIZER=hf` — force the HF crate;
//! - unset — try gigatoken, fall back to HF with the reason logged.
//!
//! The gigatoken instance is process-persistent behind a mutex, so its
//! pretoken memo cache warms **across requests** — a repeated chat-template
//! prefix encodes from cache. Encode is one short critical section per
//! request; decode takes the same lock briefly (streaming decodes are one
//! call per emitted token over a short id slice).

use peregrine_token::GigaTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

/// The active tokenizer backend.
pub enum TokenBackend {
    /// Vendored gigatoken BPE (fast path). `Mutex` because encode is `&mut`
    /// (the memo cache learns); kept for the process so the cache persists.
    Giga(Box<parking_lot::Mutex<GigaTokenizer>>),
    /// HuggingFace `tokenizers` fallback (SentencePiece / non-BPE models, or
    /// forced via env).
    Hf(Box<HfTokenizer>),
}

impl TokenBackend {
    /// Choose and construct the backend from the model dir's `tokenizer.json`.
    /// Logs the decision to stderr (a server boots once; the operator should
    /// see which path is active).
    pub fn load(dir: &std::path::Path) -> Result<TokenBackend, String> {
        let path = dir.join("tokenizer.json");
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let forced = std::env::var("COLI_TOKENIZER").ok();
        match forced.as_deref() {
            Some("hf") => {
                eprintln!("[tokenizer] HF tokenizers (forced by COLI_TOKENIZER=hf)");
                Self::load_hf(&bytes)
            }
            Some("giga") => match GigaTokenizer::from_hf_json_bytes(&bytes) {
                Ok(t) => {
                    eprintln!("[tokenizer] gigatoken BPE active (forced), vocab={}", t.vocab_size());
                    Ok(TokenBackend::Giga(Box::new(parking_lot::Mutex::new(t))))
                }
                Err(e) => Err(format!("COLI_TOKENIZER=giga but gigatoken can't load this model: {e}")),
            },
            _ => match GigaTokenizer::from_hf_json_bytes(&bytes) {
                Ok(t) => {
                    eprintln!("[tokenizer] gigatoken BPE active, vocab={}", t.vocab_size());
                    Ok(TokenBackend::Giga(Box::new(parking_lot::Mutex::new(t))))
                }
                Err(e) => {
                    eprintln!("[tokenizer] falling back to HF tokenizers: {e}");
                    Self::load_hf(&bytes)
                }
            },
        }
    }

    fn load_hf(bytes: &[u8]) -> Result<TokenBackend, String> {
        let t = HfTokenizer::from_bytes(bytes).map_err(|e| format!("HF tokenizer: {e}"))?;
        Ok(TokenBackend::Hf(Box::new(t)))
    }

    /// Encode `text` to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        match self {
            TokenBackend::Giga(t) => Ok(t.lock().encode(text)),
            TokenBackend::Hf(t) => {
                let enc = t.encode(text, false).map_err(|e| format!("encode: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
        }
    }

    /// Decode token ids to text. Both backends produce plain text with
    /// special tokens materialized as their content is; the SSE path diffs
    /// consecutive decodes, so partial-UTF-8 at a token boundary is handled
    /// by the lossy conversion the same way for both.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        match self {
            TokenBackend::Giga(t) => {
                let bytes = t.lock().decode(ids);
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            TokenBackend::Hf(t) => t.decode(ids, true).map_err(|e| format!("decode: {e}")),
        }
    }

    /// Which backend is active (for logs / health output).
    pub fn name(&self) -> &'static str {
        match self {
            TokenBackend::Giga(_) => "gigatoken",
            TokenBackend::Hf(_) => "hf-tokenizers",
        }
    }
}

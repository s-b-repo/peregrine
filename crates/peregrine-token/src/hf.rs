// Vendored from marcelroed/gigatoken v0.10.0 (MIT) — src/load_tokenizer/hf.rs.
// Local modifications: SentencePiece branch removed (peregrine's serve layer
// falls back to the HF `tokenizers` crate for byte_fallback models); the
// hub-downloading test module dropped (fixture tests live in the facade).
//! Load HuggingFace tokenizer.json files.
//!
//! Supports two styles:
//! - SentencePiece BPE (`byte_fallback=true`, e.g. Llama) is NOT supported in this
//!   vendored subset — `load_hf_slice` returns a descriptive error instead
//! - ByteLevel BPE without byte_fallback (e.g. GPT-2) → [`load_hf_bpe`]

// The tokenizer variants differ greatly in size
#![allow(clippy::large_enum_variant)]

use crate::bpe;
use crate::token::TokenId;
use eyre::{Context, Result, ensure};
use rustc_hash::FxBuildHasher;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// JSON schema (only the fields we need)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(default)]
    normalizer: Option<NormalizerJson>,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    normalizers: Vec<NormalizerJson>,
    /// `Prepend` normalizer: the prefix (Llama 2's "▁").
    #[serde(default)]
    prepend: Option<String>,
    /// `Replace` normalizer: pattern and replacement content.
    #[serde(default)]
    pattern: Option<PatternJson>,
    #[serde(default)]
    content: Option<String>,
    /// `Strip` normalizer sides.
    #[serde(default)]
    strip_left: Option<bool>,
    #[serde(default)]
    strip_right: Option<bool>,
    /// `Precompiled` normalizer: base64-encoded sentencepiece charsmap.
    #[serde(default)]
    precompiled_charsmap: Option<String>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreTokenizerJson>,
    #[serde(default)]
    pattern: Option<PatternJson>,
    /// `Metaspace` fields. `add_prefix_space` is the pre-0.15 spelling of
    /// `prepend_scheme`.
    #[serde(default)]
    replacement: Option<String>,
    #[serde(default)]
    prepend_scheme: Option<String>,
    #[serde(default)]
    add_prefix_space: Option<bool>,
    #[serde(default)]
    split: Option<bool>,
    /// `Split` field (e.g. "MergedWithPrevious" for the gemma-3/4 no-op
    /// space Split).
    #[serde(default)]
    behavior: Option<String>,
}

#[derive(Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
    #[serde(rename = "String", default)]
    literal: Option<String>,
}

#[derive(Deserialize)]
struct Model {
    /// tokenizer.json files written before tokenizers 0.9 (e.g. the original
    /// GPT-2 upload) omit `model.type`; those are always BPE.
    #[serde(rename = "type", default = "legacy_bpe_type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(deserialize_with = "deserialize_merges")]
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
    /// HF BPE `ignore_merges`: a pretoken whose whole byte string is a vocab
    /// entry encodes as that single ID, skipping the merge loop (GLM-5.2,
    /// DeepSeek V3, Llama 3).
    #[serde(default)]
    ignore_merges: bool,
}

fn legacy_bpe_type() -> String {
    "BPE".to_string()
}

/// Merges appear as `["a", "b"]` arrays in current tokenizer.json files and
/// as `"a b"` strings in older ones; accept both.
fn deserialize_merges<'de, D>(deserializer: D) -> Result<Vec<[String; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Merge {
        Pair([String; 2]),
        Legacy(String),
    }
    let raw = Vec::<Merge>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|m| match m {
            Merge::Pair(pair) => Ok(pair),
            Merge::Legacy(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| {
                    serde::de::Error::custom(format!("invalid merge entry: {s:?}"))
                })?;
                Ok([a.to_string(), b.to_string()])
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

// ---------------------------------------------------------------------------
// Token string → raw bytes conversion
// ---------------------------------------------------------------------------

/// Parse a byte-fallback token string `<0xHH>` into its byte.
fn parse_byte_fallback(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Convert a HuggingFace vocab string to raw bytes.
///
/// - Byte-fallback tokens `<0xHH>` → the single byte.
/// - Everything else → its UTF-8 bytes (▁ is kept as-is).
fn token_str_to_bytes(s: &str) -> Vec<u8> {
    match parse_byte_fallback(s) {
        Some(byte) => vec![byte],
        None => s.as_bytes().to_vec(),
    }
}

/// Added tokens may live outside model.vocab (e.g. Qwen2's <|endoftext|>,
/// Phi-3's placeholders); extend the vocab so their IDs decode to the
/// literal content.
fn extend_vocab_with_added_tokens(vocab: &mut Vec<Arc<[u8]>>, added_tokens: &[AddedToken]) {
    for t in added_tokens {
        let id = t.id as usize;
        if id >= vocab.len() {
            vocab.resize(id + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[id].is_empty() {
            vocab[id] = t.content.as_bytes().into();
        }
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// A tokenizer loaded from HuggingFace `tokenizer.json` data: the model's
/// `byte_fallback` flag decides which of the two supported styles applies.
pub enum HfTokenizer {
    Bpe(bpe::tiktoken::Tokenizer),
}

/// Probes `model.type` alone, so an unsupported model family (WordPiece,
/// Unigram, ...) is refused by name BEFORE the full BPE-shaped schema is
/// applied —
/// those files are valid JSON with a different `model.vocab`/`merges`
/// shape, and the full parse would report a misleading deserializer error
/// ("missing field `merges`", "invalid type: sequence, expected a map").
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model: Option<ModelTypeOnly>,
}

#[derive(Deserialize)]
struct ModelTypeOnly {
    #[serde(rename = "type")]
    model_type: Option<String>,
    /// Family markers for untyped legacy files (pre-0.9 `tokenizers`
    /// omitted `model.type`): `unk_id` only exists on Unigram models
    /// (e.g. t5-small, xlm-roberta) and `max_input_chars_per_word` only on
    /// WordPiece (e.g. bert-base-uncased). `continuing_subword_prefix`
    /// would NOT work for WordPiece detection: BPE serializes it too (the
    /// original gpt2 upload has `"continuing_subword_prefix": ""`).
    unk_id: Option<u64>,
    max_input_chars_per_word: Option<u64>,
}

fn parse_tokenizer_json(data: &[u8]) -> Result<TokenizerJson> {
    if let Ok(ModelTypeProbe { model: Some(m) }) = sonic_rs::from_slice::<ModelTypeProbe>(data) {
        let family = match m.model_type.as_deref() {
            Some("BPE") => None,
            Some(other) => Some(other.to_string()),
            None if m.unk_id.is_some() => Some("Unigram (untyped legacy file)".to_string()),
            None if m.max_input_chars_per_word.is_some() => {
                Some("WordPiece (untyped legacy file)".to_string())
            }
            None => None, // untyped BPE (pre-0.9 GPT-2-style files)
        };
        if let Some(family) = family {
            return Err(eyre::eyre!(
                "Unsupported model type \"{family}\": gigatoken supports BPE tokenizers \
                 (byte-level, or SentencePiece-style with byte_fallback)"
            ));
        }
    }
    // Inline the deserializer's own message (offending field, position,
    // snippet): the first line is often all that surfaces in test summaries
    // and short tracebacks.
    sonic_rs::from_slice(data).map_err(|e| eyre::eyre!("Failed to parse tokenizer JSON: {e}"))
}

fn read_tokenizer_json(path: impl AsRef<Path>) -> Result<TokenizerJson> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    parse_tokenizer_json(&data).with_context(|| format!("Failed to parse {}", path.display()))
}

/// Load a tokenizer from in-memory `tokenizer.json` contents, choosing the
/// SentencePiece or ByteLevel BPE style from the model's `byte_fallback` flag.
pub fn load_hf_slice(data: &[u8]) -> Result<HfTokenizer> {
    let tj = parse_tokenizer_json(data)?;
    if tj.model.byte_fallback {
        Err(eyre::eyre!(
            "SentencePiece-style tokenizer (byte_fallback=true) — not supported by the \
             vendored BPE subset; use the HF tokenizers fallback"
        ))
    } else {
        Ok(HfTokenizer::Bpe(build_bpe(&tj)?))
    }
}


/// Determine whether the tokenizer's normalizer is NFC (the only kind we
/// support for ByteLevel BPE). Returns `true` for NFC, `false` for no
/// normalizer, and an error for anything else — silently skipping an unknown
/// normalizer would produce token IDs that diverge from HF.
fn detect_nfc_normalizer(normalizer: &Option<NormalizerJson>) -> Result<bool> {
    fn is_nfc(n: &NormalizerJson) -> Result<bool> {
        match n.kind.as_str() {
            "NFC" => Ok(true),
            "Sequence" => n
                .normalizers
                .iter()
                .try_fold(false, |acc, c| Ok(acc | is_nfc(c)?)),
            other => Err(eyre::eyre!("Unsupported normalizer type: {other}")),
        }
    }
    normalizer.as_ref().map_or(Ok(false), is_nfc)
}

// ---------------------------------------------------------------------------
// Pre-tokenizer detection
// ---------------------------------------------------------------------------

/// Determine the pretokenization scheme from a tokenizer.json `pre_tokenizer`.
///
/// Handles a bare `ByteLevel` (GPT-2 style, `use_regex: true`) and
/// `Sequence`s whose `Split` regexes (in order) form a known scheme —
/// either a single known regex or DeepSeek's digits/CJK/main triple.
fn detect_pretokenizer_type(
    pre_tokenizer: &Option<PreTokenizerJson>,
) -> Result<crate::pretokenize::PretokenizerType> {
    use crate::pretokenize::PretokenizerType;

    fn collect_split_regexes<'a>(pt: &'a PreTokenizerJson, out: &mut Vec<&'a str>) {
        if pt.kind == "Split"
            && let Some(PatternJson { regex: Some(re), .. }) = &pt.pattern
        {
            out.push(re);
        }
        for child in &pt.pretokenizers {
            collect_split_regexes(child, out);
        }
    }

    let Some(pt) = pre_tokenizer else {
        // No pre_tokenizer at all; keep the historical default.
        return Ok(PretokenizerType::GPT2);
    };
    let mut regexes = Vec::new();
    collect_split_regexes(pt, &mut regexes);
    if regexes.is_empty() {
        // ByteLevel with use_regex (the default) splits with the GPT-2 regex.
        if pt.kind == "ByteLevel" {
            return Ok(PretokenizerType::GPT2);
        }
        return Err(eyre::eyre!(
            "Unsupported pre_tokenizer type: {} (no Split regex found)",
            pt.kind
        ));
    }
    PretokenizerType::from_split_regexes(&regexes).ok_or_else(|| {
        eyre::eyre!("Unknown pre_tokenizer Split regexes, no fast pretokenizer for: {regexes:?}")
    })
}

/// Whether a `ByteLevel` pre-tokenizer anywhere in the chain sets
/// `add_prefix_space` (see the `Tokenizer::add_prefix_space` field for the
/// semantics).
fn detect_add_prefix_space(pre_tokenizer: &Option<PreTokenizerJson>) -> bool {
    fn walk(pt: &PreTokenizerJson) -> bool {
        (pt.kind == "ByteLevel" && pt.add_prefix_space == Some(true))
            || pt.pretokenizers.iter().any(walk)
    }
    pre_tokenizer.as_ref().is_some_and(walk)
}

// ---------------------------------------------------------------------------
// GPT-2 / ByteLevel BPE loader
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping table.
/// Returns (byte_to_unicode, unicode_to_byte).
fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut b2u = ['\0'; 256];
    for &b in &allowed {
        b2u[b as usize] = b as char;
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if b2u[b as usize] == '\0' {
            b2u[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
    (b2u, u2b)
}

/// Decode a GPT-2 ByteLevel unicode string back to raw bytes.
///
/// Byte-level vocab strings consist solely of table chars; a string with any
/// other char is stored raw (e.g. DeepSeek V4 keeps its special tokens
/// unencoded in `model.vocab`) and taken as literal UTF-8 content.
fn unicode_to_bytes(s: &str, u2b: &HashMap<char, u8>) -> Vec<u8> {
    if s.chars().all(|c| u2b.contains_key(&c)) {
        s.chars().map(|c| u2b[&c]).collect()
    } else {
        s.as_bytes().to_vec()
    }
}

/// Load a HuggingFace `tokenizer.json` that uses ByteLevel BPE without
/// byte_fallback (e.g. GPT-2, RoBERTa).
///
/// Returns a [`bpe::tiktoken::Tokenizer`] with byte remapping.
pub fn load_hf_bpe(path: impl AsRef<Path>) -> Result<bpe::tiktoken::Tokenizer> {
    build_bpe(&read_tokenizer_json(path)?)
}

fn build_bpe(tj: &TokenizerJson) -> Result<bpe::tiktoken::Tokenizer> {
    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        !tj.model.byte_fallback,
        "byte_fallback tokenizers should use load_hf_sentencepiece instead"
    );

    let (_b2u, u2b) = build_byte_unicode_tables();

    // Build vocab sorted by ID — each entry is the raw bytes for that token
    let max_id = tj.model.vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tj.model.vocab.len(), FxBuildHasher);
    for (tok_str, &id) in &tj.model.vocab {
        let bytes: Arc<[u8]> = unicode_to_bytes(tok_str, &u2b).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    extend_vocab_with_added_tokens(&mut vocab, &tj.added_tokens);

    // Build merges from the merge list. Each merge "a b" means:
    // look up token IDs for "a" and "b", the merged token is vocab[concat(a,b)].
    let mut entries: Vec<(TokenId, TokenId, TokenId)> = Vec::with_capacity(tj.model.merges.len());
    for [str_a, str_b] in &tj.model.merges {
        let bytes_a = unicode_to_bytes(str_a, &u2b);
        let bytes_b = unicode_to_bytes(str_b, &u2b);
        let id_a = match vocab_inv.get(bytes_a.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let id_b = match vocab_inv.get(bytes_b.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let mut merged_bytes = bytes_a;
        merged_bytes.extend_from_slice(&bytes_b);
        let id_merged = match vocab_inv.get(merged_bytes.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        entries.push((id_a, id_b, id_merged));
    }

    let byte_remapping = bpe::ByteRemapping::from_byte_vocab(&vocab)?;
    let vocab: Vec<Vec<u8>> = vocab.into_iter().map(|a| a.to_vec()).collect();

    // The fast merge loops take the merged token's ID as the merge priority,
    // which is only correct when the merge list produces IDs in rank order
    // (true for every tiktoken-style vocab: GPT-2, cl100k, o200k, Qwen,
    // Llama-3, ...). Fairseq-heritage vocabs (RoBERTa/OPT/DeBERTa) order IDs
    // by corpus frequency instead; those carry their explicit list position
    // as the rank.
    let id_order_ok = entries.is_sorted_by_key(|&(_, _, merged)| merged);
    let mut tokenizer = if id_order_ok {
        let mut merges: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (id_a, id_b, id_merged) in entries {
            merges.entry((id_a, id_b)).or_insert(id_merged);
        }
        bpe::tiktoken::Tokenizer::new(merges, vocab, byte_remapping)
    } else {
        let mut merges: bpe::tiktoken::RankedMerges =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (rank, (id_a, id_b, id_merged)) in entries.into_iter().enumerate() {
            merges
                .entry(bpe::ranked_merge_key(id_a, id_b))
                .or_insert((id_merged, rank as u32));
        }
        bpe::tiktoken::Tokenizer::new_ranked(merges, vocab, byte_remapping)
    };
    tokenizer.set_pretokenizer_type(detect_pretokenizer_type(&tj.pre_tokenizer)?);
    tokenizer.set_normalize_nfc(detect_nfc_normalizer(&tj.normalizer)?);
    tokenizer.set_add_prefix_space(detect_add_prefix_space(&tj.pre_tokenizer));
    tokenizer.set_ignore_merges(tj.model.ignore_merges);
    // All added tokens (special and non-special) are matched atomically in the
    // raw input by HF's AddedVocabulary; mirror that, including the
    // whitespace-stripping flags.
    tokenizer.set_added_tokens(
        tj.added_tokens
            .iter()
            .map(|t| bpe::tiktoken::AddedTokenDef {
                content: t.content.as_bytes().into(),
                id: TokenId::from(t.id),
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            })
            .collect(),
    );
    Ok(tokenizer)
}

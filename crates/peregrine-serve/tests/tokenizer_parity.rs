//! Parity gate: the vendored gigatoken BPE backend must encode **id-for-id**
//! identically to the HuggingFace `tokenizers` crate (the reference
//! implementation) on the committed GPT-2 fixture, and both decodes must
//! reproduce the input text. This is the correctness bar that lets serve pick
//! the fast path by default.

use peregrine_token::GigaTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

/// The committed fixture, or the reason it could not be read.
///
/// Returns `Result` rather than unwrapping: `clippy.toml` sets
/// `allow-expect-in-tests = false`, and an integration test is its own crate so
/// it inherits no crate-root `deny` to enforce that — the policy has to be kept
/// by hand here.
fn fixture_bytes() -> Result<Vec<u8>, String> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json");
    std::fs::read(&path).map_err(|e| format!("committed GPT-2 fixture at {}: {e}", path.display()))
}

/// Both tokenizers over the same fixture bytes.
fn both(bytes: &[u8]) -> Result<(GigaTokenizer, HfTokenizer), String> {
    let giga = GigaTokenizer::from_hf_json_bytes(bytes).map_err(|e| format!("fixture is BPE: {e}"))?;
    let hf = HfTokenizer::from_bytes(bytes).map_err(|e| format!("HF loads the same fixture: {e}"))?;
    Ok((giga, hf))
}

const CORPUS: &[&str] = &[
    "Hello world",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and   runs of spaces  ",
    "tabs\tand\nnewlines\r\nand\r carriage returns",
    "numbers 1234567890 and mixed a1b2c3",
    "punctuation!!! ??? ... ---- ###",
    "naïve façade — ünïcodé, résumé, coöperate",
    "emoji 🦀🦅✓ and CJK 你好世界 and RTL שלום عالم",
    "fn main() { println!(\"{}\", 42); } // rust code",
    "{\"json\": true, \"nested\": {\"k\": [1, 2, 3]}}",
    "x = (a + b) * c / d - e % f; # math-ish",
    "[gMASK]<sop><|user|>\nchat template markup<|assistant|>\n",
    "the the the the the repeated repeated repeated words words",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "'s 't 're 've 'm 'll 'd contraction suffixes",
    "",
    " ",
    "\n",
    "a",
];

/// The real GLM-5.2 `tokenizer.json` (154 820 vocab / 321 649 merges / 36 added
/// tokens — the scheme serve actually runs, on the `Olmo3` fast arm), or `None`
/// to skip: the checkpoint is 20 MB and lives outside the repo, so this gate
/// runs where the model does (`COLI_MODEL`, else the box-conventional
/// `~/models/GLM-5.2`) and skips — never fails — elsewhere. The GPT-2 fixture
/// tests above keep the committed-corpus bar; these keep the production-vocab
/// bar.
fn glm_tokenizer_bytes() -> Option<Vec<u8>> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("COLI_MODEL") {
        candidates.push(std::path::PathBuf::from(dir).join("tokenizer.json"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(std::path::PathBuf::from(home).join("models/GLM-5.2/tokenizer.json"));
    }
    candidates.into_iter().find_map(|p| std::fs::read(&p).ok())
}

/// GLM-specific additions to [`CORPUS`]: the real chat markers (genuine added
/// tokens in this vocab, unlike under GPT-2 where the same strings tokenize as
/// plain text), digit runs against the `\p{N}{1,3}` pretokenizer arm, CJK-heavy
/// prose, and the newline-run arm (`\s*[\r\n]+`) that distinguishes Olmo3 from
/// the GPT-4 family.
const GLM_EXTRA: &[&str] = &[
    "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\nhi<|assistant|>\n",
    "<|user|>\n<|assistant|>\n<|observation|>\n<|endoftext|>",
    "12345678901234567890 and 1 22 333 4444 55555",
    "你好世界。这是一个中文句子，包含标点符号！还有数字１２３和English混排。",
    "line\n\n\nruns\r\n\r\nof\n \n newlines",
    "    indented code block\n\tdef f(x):\n\t\treturn x ** 2\n",
    "混合 mixed 语言 language 文本 text ，。！？【】《》",
];

#[test]
fn glm_matches_hf_id_for_id() -> Result<(), String> {
    let Some(bytes) = glm_tokenizer_bytes() else {
        eprintln!("skipping: GLM tokenizer.json absent (set COLI_MODEL or place ~/models/GLM-5.2)");
        return Ok(());
    };
    let (mut giga, hf) = both(&bytes)?;
    for text in CORPUS.iter().chain(GLM_EXTRA) {
        let g: Vec<u32> = giga.encode(text);
        let encoded = hf.encode(*text, false).map_err(|e| format!("hf encode {text:?}: {e}"))?;
        assert_eq!(g, encoded.get_ids().to_vec(), "encode mismatch on {text:?}");
        let decoded = giga.decode(&g);
        assert_eq!(String::from_utf8_lossy(&decoded), *text, "decode round trip on {text:?}");
    }
    Ok(())
}

#[test]
fn glm_encode_into_and_batch_match_encode() -> Result<(), String> {
    let Some(bytes) = glm_tokenizer_bytes() else {
        eprintln!("skipping: GLM tokenizer.json absent (set COLI_MODEL or place ~/models/GLM-5.2)");
        return Ok(());
    };
    let (mut giga, _hf) = both(&bytes)?;

    // `encode_into` is the same engine entry as `encode` minus the fresh Vec —
    // assert it anyway, so a future fast path through one and not the other
    // cannot drift silently.
    let mut buf: Vec<u32> = Vec::new();
    for text in CORPUS.iter().chain(GLM_EXTRA) {
        buf.clear();
        giga.encode_into(text, &mut buf);
        assert_eq!(buf, giga.encode(text), "encode_into mismatch on {text:?}");
    }

    // Bulk documents big enough to clear the per-worker byte gate, so the
    // parallel path (chunking + worker handout) actually engages rather than
    // silently falling back to the serial loop it is being compared against.
    let base: String = CORPUS.iter().chain(GLM_EXTRA).flat_map(|s| s.chars()).collect();
    let doc: String = std::iter::repeat_with(|| base.as_str()).take(64).collect();
    let docs: Vec<&str> = std::iter::repeat_with(|| doc.as_str()).take(48).collect();
    let par = giga.encode_batch(&docs, 2);
    assert_eq!(par.len(), docs.len(), "one id vector per document");
    let serial = giga.encode(&doc);
    for (i, ids) in par.iter().enumerate() {
        assert_eq!(ids, &serial, "encode_batch doc {i} mismatch vs serial encode");
    }
    Ok(())
}

#[test]
fn giga_matches_hf_id_for_id() -> Result<(), String> {
    let bytes = fixture_bytes()?;
    let (mut giga, hf) = both(&bytes)?;
    for text in CORPUS {
        let g: Vec<u32> = giga.encode(text);
        let encoded = hf.encode(*text, false).map_err(|e| format!("hf encode {text:?}: {e}"))?;
        assert_eq!(g, encoded.get_ids().to_vec(), "encode mismatch on {text:?}");
    }
    Ok(())
}

#[test]
fn giga_decode_round_trips() -> Result<(), String> {
    let bytes = fixture_bytes()?;
    let (mut giga, _hf) = both(&bytes)?;
    for text in CORPUS {
        let ids = giga.encode(text);
        let decoded = giga.decode(&ids);
        assert_eq!(
            String::from_utf8_lossy(&decoded),
            *text,
            "decode round trip on {text:?}"
        );
    }
    Ok(())
}

#[test]
fn repeated_prefix_hits_memo_cache_consistently() -> Result<(), String> {
    // The cross-request memo cache must never change ids: encode the same
    // chat-template prefix many times (cache-hot) and compare against a
    // fresh fork (cache-cold) every round.
    let bytes = fixture_bytes()?;
    let (mut warm, _hf) = both(&bytes)?;
    let prefix = "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n";
    for i in 0..50 {
        let text = format!("{prefix}request number {i}<|assistant|>\n");
        let hot = warm.encode(&text);
        let cold = warm.fork().encode(&text);
        assert_eq!(hot, cold, "cache-hot ids must equal cache-cold ids (round {i})");
    }
    Ok(())
}

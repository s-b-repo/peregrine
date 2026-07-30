//! Parity gate: the vendored gigatoken BPE backend must encode **id-for-id**
//! identically to the HuggingFace `tokenizers` crate (the reference
//! implementation) on the committed GPT-2 fixture, and both decodes must
//! reproduce the input text. This is the correctness bar that lets serve pick
//! the fast path by default.

use peregrine_token::GigaTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn fixture_bytes() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json");
    std::fs::read(path).expect("committed GPT-2 fixture")
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

#[test]
fn giga_matches_hf_id_for_id() {
    let bytes = fixture_bytes();
    let mut giga = GigaTokenizer::from_hf_json_bytes(&bytes).expect("fixture is BPE");
    let hf = HfTokenizer::from_bytes(&bytes).expect("HF loads the same fixture");
    for text in CORPUS {
        let g: Vec<u32> = giga.encode(text);
        let h: Vec<u32> = hf.encode(*text, false).expect("hf encode").get_ids().to_vec();
        assert_eq!(g, h, "encode mismatch on {text:?}");
    }
}

#[test]
fn giga_decode_round_trips() {
    let bytes = fixture_bytes();
    let mut giga = GigaTokenizer::from_hf_json_bytes(&bytes).expect("fixture is BPE");
    for text in CORPUS {
        let ids = giga.encode(text);
        let decoded = giga.decode(&ids);
        assert_eq!(
            String::from_utf8_lossy(&decoded),
            *text,
            "decode round trip on {text:?}"
        );
    }
}

#[test]
fn repeated_prefix_hits_memo_cache_consistently() {
    // The cross-request memo cache must never change ids: encode the same
    // chat-template prefix many times (cache-hot) and compare against a
    // fresh fork (cache-cold) every round.
    let bytes = fixture_bytes();
    let mut warm = GigaTokenizer::from_hf_json_bytes(&bytes).expect("fixture is BPE");
    let prefix = "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n";
    for i in 0..50 {
        let text = format!("{prefix}request number {i}<|assistant|>\n");
        let hot = warm.encode(&text);
        let cold = warm.fork().encode(&text);
        assert_eq!(hot, cold, "cache-hot ids must equal cache-cold ids (round {i})");
    }
}

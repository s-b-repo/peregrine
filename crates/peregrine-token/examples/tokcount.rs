//! Token-usage benchmark: how many tokens inputs use.
//!
//! Usage: `cargo run -p peregrine-token --example tokcount -- <tokenizer.json> [file ...]`
//! With no files, built-in samples are encoded. Prints per-input bytes,
//! tokens, and tokens/byte, plus totals and throughput (single + batch).

use peregrine_token::GigaTokenizer;
use std::time::Instant;

fn samples() -> Vec<String> {
    vec![
        "Hello world".to_string(),
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "  multiple   spaces\tand\nnewlines ".to_string(),
        "naïve façade — ünïcodé ✓ 🦀🦅".to_string(),
        "fn main() { println!(\"{}\", 42); }".to_string(),
        "The quick brown fox jumps over the lazy dog. ".repeat(200),
    ]
}

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    // First arg names a tokenizer only when it looks like one; otherwise the
    // fixture is used and every arg is an input.
    let tok_path = if args.first().is_some_and(|a| a.ends_with(".json")) {
        args.remove(0)
    } else {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/gpt2_tokenizer.json");
        p.to_string_lossy().into_owned()
    };
    let rest = args;
    let data = std::fs::read(&tok_path).expect("read tokenizer.json");
    let mut tok =
        GigaTokenizer::from_hf_json_bytes(&data).expect("tokenizer.json is a BPE model");
    println!("vocab={} tokenizer={tok_path}", tok.vocab_size());

    let mut texts: Vec<String> = Vec::new();
    if rest.is_empty() {
        texts = samples();
    } else {
        for f in &rest {
            match std::fs::read(f) {
                Ok(b) => texts.push(String::from_utf8_lossy(&b).into_owned()),
                Err(e) => {
                    // Not a file? Treat the argument itself as the text.
                    eprintln!("note: {f} is not a file ({e}); encoding it literally");
                    texts.push(f.clone());
                }
            }
        }
    }

    let mut total_bytes = 0usize;
    let mut total_ids = 0usize;
    let t0 = Instant::now();
    for (i, t) in texts.iter().enumerate() {
        let ids = tok.encode(t);
        total_bytes += t.len();
        total_ids += ids.len();
        let per_byte = ids.len() as f64 / t.len().max(1) as f64;
        let preview: String = t.chars().take(48).collect();
        println!("[{i}] bytes={} tokens={} tok/byte={per_byte:.3} {preview:?}", t.len(), ids.len());
    }
    let dt = t0.elapsed().as_secs_f64().max(1e-9);

    // Batch path over the same inputs (workers = online cores).
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let b0 = Instant::now();
    let out = tok.batch_encode(&refs, workers);
    let bdt = b0.elapsed().as_secs_f64().max(1e-9);
    let batch_ids: usize = out.iter().map(Vec::len).sum();
    assert_eq!(batch_ids, total_ids, "batch must agree with serial");

    println!("---");
    println!("inputs={} bytes={total_bytes} tokens={total_ids}", texts.len());
    println!(
        "serial: {:.1} MB/s  {:.0} tok/s",
        total_bytes as f64 / dt / 1e6,
        total_ids as f64 / dt
    );
    println!(
        "batch (workers={workers}): {:.1} MB/s  {:.0} tok/s",
        total_bytes as f64 / bdt / 1e6,
        batch_ids as f64 / bdt
    );
}

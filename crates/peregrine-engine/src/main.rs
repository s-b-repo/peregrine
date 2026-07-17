//! colibrì Rust engine binary. Two modes:
//!
//! - `peregrine <model-dir>` (or `COLI_MODEL=<dir> peregrine`): serve mode.
//!   Emits the `READY` sentinel, then answers line requests, each terminated by
//!   the `END` sentinel — the same handshake `c/openai_server.py` uses to drive
//!   `c/glm` as a resident subprocess (so the Rust binary is a drop-in). The
//!   request grammar here is a minimal token-id protocol; full OpenAI-header
//!   framing is the remaining M7 integration.
//! - `peregrine demo`: builds a tiny synthetic model, loads it, and generates
//!   — a self-contained end-to-end smoke test that needs no model files.

use std::io::{BufRead, Write};
use std::path::Path;

use peregrine_model::{Model, Sampler};

// Match c/openai_server.py's framing sentinels.
const READY: &[u8] = b"\x01\x01READY\x01\x01\n";
const END: &[u8] = b"\x01\x01END\x01\x01\n";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("demo") {
        run_demo();
        return;
    }
    // `build <dir>`: write a tiny synthetic model to <dir> (for serve testing).
    if args.get(1).map(String::as_str) == Some("build") {
        let Some(dir) = args.get(2) else {
            eprintln!("usage: peregrine build <dir>");
            std::process::exit(2);
        };
        peregrine_model::testkit::build_tiny_model(Path::new(dir));
        eprintln!("wrote demo model to {dir}");
        return;
    }
    let dir = std::env::var("COLI_MODEL").ok().or_else(|| args.get(1).cloned());
    let Some(dir) = dir else {
        eprintln!("usage: peregrine <model-dir>   (or COLI_MODEL=<dir>)   |   peregrine demo");
        std::process::exit(2);
    };
    let mut model = match Model::load(Path::new(&dir)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = serve(&mut model) {
        // the peer closing the pipe mid-response is a normal end, not a failure
        if e.kind() != std::io::ErrorKind::BrokenPipe {
            eprintln!("serve I/O error: {e}");
            std::process::exit(1);
        }
    }
}

/// stdio serve loop. Requests (one per line):
///   `GEN <ngen> <tok0> <tok1> ...`  → greedy-generate `ngen` tokens
///   `QUIT`                          → exit
/// Each response is the space-separated generated token ids, then `END`.
///
/// Returns `Ok` on a clean shutdown (EOF/QUIT). A write error (e.g. the client
/// closed the pipe) is propagated so the caller can exit quietly rather than
/// panicking mid-response.
fn serve(model: &mut Model) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();
    out.write_all(READY)?;
    out.flush()?;

    let mut line = String::new();
    loop {
        line.clear();
        // a read error is treated as EOF — the peer is gone either way
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t == "QUIT" {
            break;
        }
        let mut it = t.split_whitespace();
        if it.next() == Some("GEN") {
            let ngen: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let prompt: Vec<i32> = it.filter_map(|s| s.parse().ok()).collect();
            if !prompt.is_empty() && ngen > 0 {
                let mut sampler = Sampler::new(0.0, 0.9, 1); // greedy = deterministic
                let toks = model.generate(&prompt, ngen, &mut sampler);
                let rendered: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                out.write_all(rendered.join(" ").as_bytes())?;
                out.write_all(b"\n")?;
            }
        }
        out.write_all(END)?;
        out.flush()?;
    }
    Ok(())
}

fn run_demo() {
    let dir = std::env::temp_dir().join(format!("coli_engine_demo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    peregrine_model::testkit::build_tiny_model(&dir);
    let mut model = Model::load(&dir).expect("load demo model");
    let prompt = [1, 5, 9, 2];
    let mut sampler = Sampler::new(0.0, 0.9, 1);
    let toks = model.generate(&prompt, 8, &mut sampler);
    println!("peregrine — demo");
    println!("  model: {} layers, vocab {}, hidden {}", model.cfg.n_layers, model.cfg.vocab, model.cfg.hidden);
    println!("  prompt {prompt:?} -> generated {toks:?}");
    assert!(toks.iter().all(|&t| (t as i64) < model.cfg.vocab));
    let _ = std::fs::remove_dir_all(&dir);
}

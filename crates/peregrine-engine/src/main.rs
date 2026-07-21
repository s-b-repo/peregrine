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

// Quality gates: no unsafe, no panicking error handling.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{BufRead, Write};
use std::path::Path;

use peregrine_core::Error;
use peregrine_model::{Model, Sampler};

// Match c/openai_server.py's framing sentinels.
const READY: &[u8] = b"\x01\x01READY\x01\x01\n";
const END: &[u8] = b"\x01\x01END\x01\x01\n";

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("demo") => run_demo(),
        // `build <dir>`: write a tiny synthetic model to <dir> (for serve testing).
        Some("build") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine build <dir>".into()))?;
            peregrine_model::testkit::build_tiny_model(Path::new(dir))?;
            eprintln!("wrote demo model to {dir}");
            Ok(())
        }
        _ => {
            let dir = std::env::var("COLI_MODEL").ok().or_else(|| args.get(1).cloned()).ok_or_else(|| {
                Error::Format("usage: peregrine <model-dir>  (or COLI_MODEL=<dir>)  |  peregrine demo".into())
            })?;
            let mut model = Model::load(Path::new(&dir))?;
            // the peer closing the pipe mid-response is a normal end, not a failure
            match serve(&mut model) {
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => other,
            }
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
fn serve(model: &mut Model) -> Result<(), Error> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();
    out.write_all(READY)?;
    out.flush()?;

    let mut line = String::new();
    loop {
        line.clear();
        // EOF (Ok(0)) ends the loop; a genuine read error propagates.
        if reader.read_line(&mut line)? == 0 {
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
            // Parse the count and ids strictly; a malformed request is reported
            // to stderr and answered with an empty frame (never silently coerced).
            match parse_gen(&mut it) {
                Ok((ngen, prompt)) if ngen > 0 && !prompt.is_empty() => {
                    let mut sampler = Sampler::new(0.0, 0.9, 1); // greedy = deterministic
                    let toks = model.generate(&prompt, ngen, &mut sampler)?;
                    let rendered: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                    out.write_all(rendered.join(" ").as_bytes())?;
                    out.write_all(b"\n")?;
                }
                Ok(_) => {}
                Err(msg) => eprintln!("peregrine: bad GEN request: {msg}"),
            }
        }
        out.write_all(END)?;
        out.flush()?;
    }
    Ok(())
}

/// Parse a `GEN <ngen> <id...>` request body (the iterator is positioned after
/// the `GEN` token). Returns a descriptive [`Error`] on any malformed field.
fn parse_gen<'a>(it: &mut impl Iterator<Item = &'a str>) -> Result<(usize, Vec<i32>), Error> {
    let cnt = it.next().ok_or_else(|| Error::Format("missing token count".into()))?;
    let ngen: usize = match cnt.parse() {
        Ok(n) => n,
        Err(e) => return Err(Error::Format(format!("token count '{cnt}': {e}"))),
    };
    let mut prompt = Vec::new();
    for s in it {
        match s.parse::<i32>() {
            Ok(v) => prompt.push(v),
            Err(e) => return Err(Error::Format(format!("token id '{s}': {e}"))),
        }
    }
    Ok((ngen, prompt))
}

fn run_demo() -> Result<(), Error> {
    let dir = std::env::temp_dir().join(format!("coli_engine_demo_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    peregrine_model::testkit::build_tiny_model(&dir)?;
    let mut model = Model::load(&dir)?;
    let prompt = [1, 5, 9, 2];
    let mut sampler = Sampler::new(0.0, 0.9, 1);
    let toks = model.generate(&prompt, 8, &mut sampler)?;
    println!("peregrine — demo");
    println!("  model: {} layers, vocab {}, hidden {}", model.cfg.n_layers, model.cfg.vocab, model.cfg.hidden);
    println!("  prompt {prompt:?} -> generated {toks:?}");
    if !toks.iter().all(|&t| (t as i64) < model.cfg.vocab) {
        return Err(Error::Format("demo generated an out-of-range token id".into()));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

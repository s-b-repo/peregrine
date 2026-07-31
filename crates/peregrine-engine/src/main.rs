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

use peregrine_core::{Context, Error};
use peregrine_model::{pick_batch_greedy, Model, Sampler, SeqKv};

// Match c/openai_server.py's framing sentinels.
const READY: &[u8] = b"\x01\x01READY\x01\x01\n";
const END: &[u8] = b"\x01\x01END\x01\x01\n";

fn main() {
    // Cap glibc arenas before spawning the streaming/compute worker pools, so the
    // engine no longer needs `MALLOC_ARENA_MAX=2` in the environment to stay flat.
    peregrine_model::cap_malloc_arenas();
    if let Err(e) = run() {
        eprintln!("peregrine: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("demo") => run_demo(),
        // `bench [B ...]`: aggregate decode-throughput sweep over batch sizes,
        // driving forward_step_batched on COLI_MODEL. Shows the batching
        // amortization (streaming) or compute scaling (resident).
        Some("bench") => run_bench(&args[2..]),
        // `build <dir>`: write a tiny synthetic model to <dir> (for serve testing).
        Some("build") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine build <dir>".into()))?;
            peregrine_model::testkit::build_tiny_model(Path::new(dir))?;
            eprintln!("wrote demo model to {dir}");
            Ok(())
        }
        // `build-automaton <model-dir> [corpus-len]`: offline pass that runs a corpus
        // through the model, accumulates the expert-transition automaton, and writes
        // `<model-dir>/automaton.json` (auto-loaded on the next model load).
        Some("build-automaton") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine build-automaton <model-dir> [corpus-len]".into()))?;
            let len = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            let mut model = Model::load_streaming(Path::new(dir), true)?;
            let corpus = synth_corpus(model.cfg.vocab as usize, len);
            let table = model.build_automaton(&corpus)?;
            let out = Path::new(dir).join("automaton.json");
            peregrine_model::save_automaton(&table, &out)?;
            eprintln!("wrote automaton ({len}-token corpus) to {}", out.display());
            Ok(())
        }
        // `galactic <model-dir> [corpus-len]`: the one-shot offline preprocessing
        // pass — one corpus run emits EVERY artifact the loader consumes:
        // automaton.json, macrostates.json, routes.json, schedule.json
        // (co-occurrence → Louvain communities), and a route_stats.json seed.
        Some("galactic") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine galactic <model-dir> [corpus-len]".into()))?;
            let len = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            let dirp = Path::new(dir);
            let mut model = Model::load_streaming(dirp, true)?;
            let corpus = synth_corpus(model.cfg.vocab as usize, len);
            let (table, macros, trace) = model.build_artifacts(&corpus)?;
            peregrine_model::save_automaton(&table, &dirp.join("automaton.json"))?;
            peregrine_model::save_macrostates(&macros, &dirp.join("macrostates.json"))?;
            let routes_json = serde_json::to_vec(&trace).map_err(|e| Error::Format(format!("serialize trace: {e}")))?;
            peregrine_core::write_atomic(&dirp.join("routes.json"), &routes_json)?;
            // layout schedule from the same trace (Louvain + 2-opt refinement)
            let mut ordered = peregrine_tools::order_experts(&trace, "louvain")?;
            for (l, row) in ordered.iter_mut().enumerate() {
                let w = peregrine_tools::build_cooccurrence(&trace, l);
                peregrine_tools::two_opt(row, &w);
            }
            peregrine_tools::write_schedule(dirp, &ordered)?;
            // storage-tier placement (hypergraph: whole communities per tier),
            // emitted when the operator declares tier byte budgets.
            let vram_mb: u64 = std::env::var("COLI_TIER_VRAM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let ram_mb: u64 = std::env::var("COLI_TIER_RAM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            if vram_mb > 0 || ram_mb > 0 {
                // Size an expert from what the checkpoint actually stores rather
                // than assuming int4: an int8 model is twice as large per
                // expert, so the hardcoded formula under-counted by 2× and the
                // emitted tier plan overcommitted VRAM.
                let hidden = model.cfg.hidden as u64;
                let inter = model.cfg.moe_inter as u64;
                let bytes_per_expert = model
                    .expert_bytes_on_disk(model.cfg.first_dense as usize, 0)
                    .unwrap_or_else(|| (3 * inter * hidden) / 2);
                let n_layers = model.cfg.n_layers as usize;
                let sparse0 = model.cfg.first_dense as usize;
                let per_layer_v = (vram_mb << 20) / (n_layers.saturating_sub(sparse0).max(1) as u64);
                let per_layer_r = (ram_mb << 20) / (n_layers.saturating_sub(sparse0).max(1) as u64);
                let mut vram_all: Vec<(usize, i32)> = Vec::new();
                let mut ram_all: Vec<(usize, i32)> = Vec::new();
                for l in sparse0..n_layers {
                    let w = peregrine_tools::build_cooccurrence(&trace, l);
                    let heat = peregrine_tools::trace_heat(&trace, l);
                    let (v, r) = peregrine_tools::assign_tiers(&w, &heat, bytes_per_expert, per_layer_v, per_layer_r);
                    vram_all.extend(v.into_iter().map(|e| (l, e)));
                    ram_all.extend(r.into_iter().map(|e| (l, e)));
                }
                peregrine_tools::write_tiers(dirp, &vram_all, &ram_all)?;
            }
            // heat + history + co-activation seed for the next session
            model.save_route_stats(dirp)?;
            eprintln!(
                "galactic pass complete ({len}-token corpus): automaton.json, macrostates.json, \
                 routes.json, schedule.json, route_stats.json written to {dir}"
            );
            Ok(())
        }
        // `compile-plan <model-dir>`: bundle every offline artifact present in the
        // model dir (automaton, macrostates, schedule, tiers, learned policy from
        // route_stats) into one `plan.json` — the "compiled execution plan" the
        // loader consumes in one shot. Profile-guided: every input is a recorded
        // profile. Missing artifacts are simply omitted.
        Some("compile-plan") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine compile-plan <model-dir>".into()))?;
            let dirp = Path::new(dir);
            // A missing artifact is normal (that part of the plan is simply
            // absent); a corrupt one is not, and silently omitting it produced a
            // plan that looked complete while quietly dropping a section.
            let read_json = |name: &str| -> Result<Option<serde_json::Value>, Error> {
                let path = dirp.join(name);
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => return Err(Error::Io(e)).ctx(|| format!("read {}", path.display())),
                };
                let v: serde_json::Value = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::Format(format!("{}: not valid JSON ({e})", path.display())))?;
                Ok(Some(v))
            };
            let mut plan = serde_json::Map::new();
            plan.insert("version".into(), serde_json::json!(1));
            let mut parts: Vec<&str> = Vec::new();
            if let Some(v) = read_json("automaton.json")? {
                plan.insert("automaton".into(), v);
                parts.push("automaton");
            }
            if let Some(v) = read_json("macrostates.json")? {
                plan.insert("macrostates".into(), v);
                parts.push("macrostates");
            }
            if let Some(v) = read_json("schedule.json")? {
                plan.insert("schedule".into(), v);
                parts.push("schedule");
            }
            if let Some(v) = read_json("tiers.json")? {
                plan.insert("tiers".into(), v);
                parts.push("tiers");
            }
            if let Some(v) = read_json("route_stats.json")? {
                if let Some(learn) = v.get("learn") {
                    if !learn.is_null() {
                        plan.insert("learn".into(), learn.clone());
                        parts.push("learn");
                    }
                }
            }
            if parts.is_empty() {
                return Err(Error::Format(
                    "no artifacts found — run `peregrine galactic <model-dir>` first".into(),
                ));
            }
            let bytes = serde_json::to_vec_pretty(&serde_json::Value::Object(plan))
                .map_err(|e| Error::Format(format!("serialize plan: {e}")))?;
            peregrine_core::write_atomic(&dirp.join("plan.json"), &bytes)?;
            eprintln!("compiled execution plan ({}) → {dir}/plan.json", parts.join(" + "));
            Ok(())
        }
        // `dump-routes <model-dir> <out.json> [corpus-len]`: write the raw per-forward
        // routing trace (for offline inspection / custom automaton building).
        Some("dump-routes") => {
            let dir = args.get(2).ok_or_else(|| Error::Format("usage: peregrine dump-routes <model-dir> <out.json> [corpus-len]".into()))?;
            let out = args.get(3).ok_or_else(|| Error::Format("usage: peregrine dump-routes <model-dir> <out.json> [corpus-len]".into()))?;
            let len = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
            let mut model = Model::load_streaming(Path::new(dir), true)?;
            let corpus = synth_corpus(model.cfg.vocab as usize, len);
            let n = model.dump_routes_to(&corpus, Path::new(out))?;
            eprintln!("wrote {n} forwards of routing trace to {out}");
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

/// A deterministic pseudo-random token stream for offline automaton building — there
/// is no real prompt corpus in this environment, so an LCG stands in. Same seed →
/// same corpus → reproducible automaton.
fn synth_corpus(vocab: usize, n: usize) -> Vec<i32> {
    let vocab = vocab.max(1) as u64;
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % vocab) as i32
        })
        .collect()
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
                    // warm-tier diagnostics (stderr): how much of the expert I/O the
                    // cache absorbed this request.
                    if let Some((h, m, d)) = model.ecache_stats() {
                        let hr = 100.0 * h as f64 / (h + m).max(1) as f64;
                        let pf = model.ecache_prefetch_reads().unwrap_or(0);
                        eprintln!("[ecache] hits={h} misses={m} disk_reads={d} prefetch_reads={pf} hit_rate={hr:.1}%");
                        // prefetch effectiveness: how many speculative reads paid off,
                        // plus fadvise hints and (opt-in) verify mismatches.
                        let (used, wasted) = model.ecache_prefetch_effectiveness().unwrap_or((0, 0));
                        let acc = 100.0 * model.prefetch_accuracy().unwrap_or(0.0);
                        let fadv = model.ecache_fadvise_hints().unwrap_or(0);
                        let vm = model.ecache_verify_mismatch().unwrap_or(0);
                        eprintln!("[prefetch] used={used} wasted={wasted} accuracy={acc:.1}% fadvise={fadv} verify_mismatch={vm}");
                    }
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

/// Aggregate decode-throughput sweep. For each batch size B, run `COLI_BENCH_STEPS`
/// batched decode steps over B independent sequences (via `forward_step_batched`)
/// and report aggregate tokens/sec. On a streaming model this shows the disk
/// amortization (experts read once per step, shared across B); on a resident model
/// it shows compute scaling. `COLI_MODEL` selects the model; args override the B set.
fn run_bench(batch_args: &[String]) -> Result<(), Error> {
    let dir = std::env::var("COLI_MODEL")
        .ok()
        .ok_or_else(|| Error::Format("bench needs COLI_MODEL=<dir>  (e.g. a tiny model from `peregrine build`)".into()))?;
    let steps: usize =
        std::env::var("COLI_BENCH_STEPS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(3);
    let batches: Vec<usize> = if batch_args.is_empty() {
        vec![1, 4, 16]
    } else {
        batch_args.iter().filter_map(|s| s.parse().ok()).filter(|&b| b > 0).collect()
    };

    let t0 = std::time::Instant::now();
    let model = Model::load(Path::new(&dir))?;
    let vocab = model.cfg.vocab as usize;
    eprintln!("peregrine bench: loaded {} layers, vocab {}, in {:.1}s", model.cfg.n_layers, vocab, t0.elapsed().as_secs_f64());
    println!("  batch   steps   tokens    seconds     agg tok/s   per-seq tok/s");
    for &b in &batches {
        let mut seqs: Vec<SeqKv> = (0..b).map(|_| SeqKv::new(&model.cfg)).collect();
        // distinct starting token per sequence so their routing diverges
        let mut toks: Vec<i32> = (0..b).map(|i| (i as i32 * 7 + 1) % vocab.max(1) as i32).collect();
        let (d0, _) = model.ecache_stats().map(|(_, _, d)| (d, ())).unwrap_or((0, ()));

        let t = std::time::Instant::now();
        for step in 0..steps {
            let pos_of: Vec<usize> = vec![step; b];
            let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
            let logits = model.forward_step_batched(&toks, &mut refs, &pos_of, None)?;
            pick_batch_greedy(&logits, vocab, &mut toks); // next tokens feed the following step
        }
        let dt = t.elapsed().as_secs_f64().max(1e-9);
        let tokens = b * steps;
        let agg = tokens as f64 / dt;
        println!("  {b:5}   {steps:5}   {tokens:6}   {dt:8.2}   {agg:11.3}   {:13.4}", agg / b as f64);
        if let Some((_, _, d)) = model.ecache_stats() {
            eprintln!("    [ecache] disk_reads this batch: {}", d.saturating_sub(d0));
        }
    }
    Ok(())
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
    // Both ends: a negative id is as out of range as an oversized one, and
    // `< vocab` alone accepts every negative value.
    if !toks.iter().all(|&t| (0..model.cfg.vocab).contains(&(t as i64))) {
        return Err(Error::Format("demo generated an out-of-range token id".into()));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

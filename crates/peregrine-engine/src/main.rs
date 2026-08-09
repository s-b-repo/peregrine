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


/// Install an alternative MoE expert-dispatch engine when `COLI_MOE_ENGINE`
/// asks for one. Unset or `concurrent` leaves the default three-lane path.
///
/// **This is where `peregrine-sched` becomes reachable from production.** That
/// crate depends on `peregrine-model`, so the model cannot call it — the cycle
/// is why `moe_streamed` sat unreachable while `README.md` and three docs
/// described it as shipped. A binary may depend on both, so the install happens
/// here and the model only knows the `MoeEngine` trait.
///
/// **`sched` is slower and that is not a bug.** `moe_streamed` is the two-lane
/// ancestor: one io_uring ring, no GPU lane, no warm cache, no prefetch, no lane
/// balancer. It exists as an independently written second implementation of the
/// same computation — valuable as an A/B against the default, not as a default.
fn install_moe_engine() {
    // Matched through `as_deref()` — the idiom every other env gate in this
    // repo uses. `unwrap_or_default()` is a [P] hit and `Err(_)` a [B] one,
    // and both audits are right: neither says which case it is handling.
    let var = std::env::var("COLI_MOE_ENGINE");
    match var.as_deref() {
        Ok("sched") => {}
        // Unset, empty, or the default engine: nothing to install.
        Ok("") | Ok("concurrent") => return,
        Ok(other) => {
            eprintln!("peregrine: COLI_MOE_ENGINE={other} is not a known engine (concurrent|sched); using concurrent");
            return;
        }
        _ => return,
    }
    let depth: u32 = std::env::var("COLI_IO_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    match peregrine_sched::SchedEngine::new(depth) {
        Ok(engine) => {
            if peregrine_model::concurrent::install_moe_engine(Box::new(engine)) {
                eprintln!("peregrine: MoE engine = sched (peregrine-sched::moe_streamed, ring depth {depth}) — \
                           two-lane: no GPU lane, no warm cache, no prefetch");
            }
        }
        // A failed ring is not a reason to abort the process: the default engine
        // is fully functional, so report and carry on with it.
        Err(e) => eprintln!("peregrine: COLI_MOE_ENGINE=sched requested but the io_uring ring failed ({e}); using concurrent"),
    }
    // Confirm against the dispatch path rather than against the branch above.
    // Every message in this function reports an *intent*; this one reports what
    // `moe_forward_dispatch` will actually do, and the two differ whenever the
    // ring failed or something installed first.
    if !peregrine_model::concurrent::moe_engine_installed() {
        eprintln!("peregrine: MoE dispatch = concurrent (no alternative engine installed)");
    }
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    // Machine banner on stderr, before anything else. Every run's log now says
    // which SIMD family computed it, whether the GPU lane was even built, and
    // what the PCIe link negotiated — the three facts that most often explain a
    // number nobody can reproduce. Suppressed for `build`, which writes a tiny
    // fixture and has no machine-dependent behavior worth annotating.
    if args.get(1).map(String::as_str) != Some("build") {
        eprintln!("{}", peregrine_model::startup_banner());
    }
    install_moe_engine();
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
                //
                // Sized **per expert**, not from one probe. Probing
                // `(first_dense, 0)` and reusing the answer everywhere assumes a
                // uniform container, which `peregrine-requantize --tier-hot-frac`
                // stopped producing: a heat-tiered checkpoint mixes int4 and int2
                // experts within a layer. The int4 fallback formula stays for
                // tensors the index cannot size, per expert rather than globally.
                let hidden = model.cfg.hidden as u64;
                let inter = model.cfg.moe_inter as u64;
                let int4_guess = (3 * inter * hidden) / 2;
                let n_layers = model.cfg.n_layers as usize;
                let sparse0 = model.cfg.first_dense as usize;
                let per_layer_v = (vram_mb << 20) / (n_layers.saturating_sub(sparse0).max(1) as u64);
                let per_layer_r = (ram_mb << 20) / (n_layers.saturating_sub(sparse0).max(1) as u64);
                let mut vram_all: Vec<(usize, i32)> = Vec::new();
                let mut ram_all: Vec<(usize, i32)> = Vec::new();
                for l in sparse0..n_layers {
                    let w = peregrine_tools::build_cooccurrence(&trace, l);
                    let heat = peregrine_tools::trace_heat(&trace, l);
                    let bytes_of = |e: i32| -> u64 {
                        usize::try_from(e)
                            .ok()
                            .and_then(|eu| model.expert_bytes_on_disk(l, eu))
                            .unwrap_or(int4_guess)
                    };
                    let (v, r) = peregrine_tools::assign_tiers(&w, &heat, bytes_of, per_layer_v, per_layer_r);
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
        // `route-stats <routes.json> [n_experts]`: read a trace written by
        // `dump-routes`/`galactic` and report what the router actually does —
        // consecutive-token expert overlap against the independence null, and
        // union growth over speculative windows and batch proxies.
        //
        // This exists because the repo's headline "0.6% cross-token locality"
        // is a **warm-cache hit rate** (58/9600 at a 10 GB cache,
        // `docs/peregrine-vs-colibri.md` §5.2) that four documents then gloss as
        // a statement about the router. Those are different quantities; only the
        // first was ever measured. The routing figure is what decides whether
        // batching amortizes expert reads and whether speculative verification
        // is byte-neutral, so it needs measuring on its own terms.
        //
        // Takes the trace, not the model: it is pure analysis over recorded
        // routing, so it runs on a box that cannot load the checkpoint.
        // `flip-rate <source-dir> <candidate-dir> [--text FILE] [--tokens N]`:
        // the quality gate for a lossy container, and the one thing standing
        // between "int2 converted" and "int2 usable".
        //
        // Every other gate in this repo is bit-identity, which a requantized
        // container fails by construction — so `peregrine-requantize` has always
        // ended by telling the operator to measure `prediction_flip_rate`, with
        // nothing in the tree that could. `requant.rs` says why it is here and
        // not there: the tool links `peregrine-core` alone on purpose, and
        // flip-rate needs the whole inference stack.
        //
        // **The models load one at a time.** Two streaming loads would hold two
        // warm caches and two resident sets against one page cache, and the
        // slower container would be measured while the faster one evicted it.
        // Peak RSS here is one model's, not two.
        Some("flip-rate") => {
            let usage = || {
                Error::Format(
                    "usage: peregrine flip-rate <source-dir> <candidate-dir> [--text FILE] [--tokens N]".into(),
                )
            };
            let src = args.get(2).filter(|s| !s.starts_with("--")).ok_or_else(usage)?;
            let cand = args.get(3).filter(|s| !s.starts_with("--")).ok_or_else(usage)?;
            let text = flag_value(&args, "--text");
            let n_tokens: usize = flag_value(&args, "--tokens")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);

            let toks = match &text {
                Some(path) => {
                    let raw = std::fs::read_to_string(Path::new(path))
                        .map_err(|e| Error::Format(format!("flip-rate: read {path}: {e}")))?;
                    // The tokenizer travels with the source container, so the
                    // ids are the ones that container was converted from.
                    let tj = Path::new(src).join("tokenizer.json");
                    let json = std::fs::read(&tj).map_err(|e| {
                        Error::Format(format!("flip-rate: --text needs {}: {e}", tj.display()))
                    })?;
                    let mut tk = peregrine_token::GigaTokenizer::from_hf_json_bytes(&json)
                        .map_err(|e| Error::Format(format!("flip-rate: tokenizer: {e}")))?;
                    let ids: Vec<i32> = tk.encode(&raw).iter().map(|&i| i as i32).collect();
                    if ids.is_empty() {
                        return Err(Error::Format("flip-rate: --text encoded to no tokens".into()));
                    }
                    ids.into_iter().take(n_tokens).collect::<Vec<i32>>()
                }
                None => {
                    // Deliberately loud. A flip rate over uniform-random ids
                    // measures how two containers disagree on inputs the model
                    // was never trained to see, which is a different and much
                    // less useful number than the one the operator wants.
                    eprintln!(
                        "peregrine: flip-rate has no --text, so it is running on a SYNTHETIC \
                         corpus of uniform-random token ids. Treat the result as a smoke test \
                         of the harness, not as a quality figure for the container."
                    );
                    Vec::new()
                }
            };

            let run = |dir: &str, toks: &[i32]| -> Result<(Vec<i32>, Vec<i32>), Error> {
                let mut m = Model::load_streaming(Path::new(dir), true)?;
                let t = if toks.is_empty() {
                    synth_corpus(m.cfg.vocab as usize, n_tokens)
                } else {
                    toks.to_vec()
                };
                let t0 = std::time::Instant::now();
                let out = m.teacher_forcing(&t)?;
                eprintln!(
                    "peregrine: flip-rate: {dir} — {} positions in one forward, {:.1}s",
                    out.len(),
                    t0.elapsed().as_secs_f64()
                );
                Ok((out, t))
            };
            let (a, used) = run(src, &toks)?;
            let (b, _) = run(cand, &used)?;

            let rate = Model::prediction_flip_rate(&a, &b)
                .ok_or_else(|| Error::Format("flip-rate: the two runs disagree on length".into()))?;
            println!("positions   {}", a.len());
            println!("flips       {}", a.iter().zip(&b).filter(|(x, y)| x != y).count());
            println!("flip_rate   {rate:.6}");
            println!("source      {src}");
            println!("candidate   {cand}");
            // Top-1 agreement only, and on one text. It is a floor on quality,
            // not a summary of it: a container can hold top-1 and still shift
            // the distribution underneath, which this cannot see.
            eprintln!(
                "peregrine: flip-rate is top-1 agreement under teacher forcing on ONE text. \
                 A low rate does not license the container on its own."
            );
            Ok(())
        }
        Some("route-stats") => {
            let path = args
                .get(2)
                .ok_or_else(|| Error::Format("usage: peregrine route-stats <routes.json> [n_experts]".into()))?;
            let trace = peregrine_tools::read_routes(Path::new(path))?;
            // The pool size is not recorded in the trace, so it is an argument.
            // Defaulting it to 0 rather than guessing keeps the null honest: with
            // no pool size the report prints no null instead of a wrong one.
            let n_experts = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            if n_experts == 0 {
                eprintln!(
                    "peregrine: no expert-pool size given — reporting overlap without its \
                     independence null. Pass it as the 3rd argument (GLM-5.2: 256)."
                );
            }
            print!("{}", peregrine_tools::format_route_stats(&trace, n_experts));
            Ok(())
        }
        _ => {
            // The positional model dir is the first argument that is not a flag
            // or a flag's value. Taking `args[1]` blindly made `peregrine
            // --draft 4` (with COLI_MODEL unset) try to load a directory called
            // "--draft" and report a file-not-found instead of the usage line.
            let dir = std::env::var("COLI_MODEL").ok().or_else(|| positional_dir(&args)).ok_or_else(|| {
                Error::Format(
                    "usage: peregrine <model-dir> [--draft N]  (or COLI_MODEL=<dir>)  |  peregrine demo".into(),
                )
            })?;
            let mut model = Model::load(Path::new(&dir))?;
            // Speculative decode needs an MTP head, which only checkpoints
            // converted with `--mtp` carry. Refuse loudly rather than silently
            // decoding non-speculatively: an operator who asked for `--draft`
            // and got the historical path with no signal would benchmark the
            // wrong thing and conclude speculation does not help — which is
            // precisely the reading this feature exists to re-test.
            let draft = draft_depth(&args);
            if draft > 0 && !model.has_mtp() {
                return Err(Error::Format(format!(
                    "--draft {draft} needs an MTP head, and this checkpoint has none \
                     (convert with --mtp, or drop the flag)"
                )));
            }
            if draft > 0 {
                eprintln!("peregrine: MTP speculative decode on, {draft} draft tokens per round");
            }
            // the peer closing the pipe mid-response is a normal end, not a failure
            match serve(&mut model, draft) {
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => other,
            }
        }
    }
}

/// A deterministic pseudo-random token stream for offline automaton building — there
/// is no real prompt corpus in this environment, so an LCG stands in. Same seed →
/// same corpus → reproducible automaton.
/// `--name value` lookup over the raw argv. Deliberately not a flag parser:
/// this binary's subcommands are positional and the two flags `flip-rate` takes
/// do not justify pulling one in.
fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

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

/// Draft depth for MTP speculative decode (`--draft N` or `COLI_DRAFT=N`);
/// 0 = off, the historical non-speculative path.
///
/// **Why a default of 4–6 is the guidance and 2 is not.** `generate_speculative`
/// has shipped complete, tested and reachable by nothing since M5, and the
/// repo's stance ("MTP is a net loss on MoE decode") traces to two figures taken
/// at draft depth 2 — where the theoretical ceiling is 3 accepted tokens, so the
/// measured 2.46 was already 82% of what that configuration could ever reach.
/// Production stacks run 5–6 draft tokens on this model class, and the GLM-5
/// report measures an accept length of 2.76 at 4 steps.
///
/// The amortization only exists because the verify pass is *one* forward over
/// `1+γ` rows through `batch_union` — a single shared expert read for every
/// drafted token. Whether that is cheap depends on how much the routed union
/// grows with γ, which is exactly what `peregrine route-stats` measures.
fn draft_depth(args: &[String]) -> usize {
    args.iter()
        .position(|a| a == "--draft")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| std::env::var("COLI_DRAFT").ok().and_then(|s| s.trim().parse::<usize>().ok()))
        .unwrap_or(0)
}

/// The first positional argument, skipping `--flag` tokens and the value that
/// follows each one. `args[0]` is the binary name.
///
/// Kept separate and pure so the "is this a flag or the model dir" rule is one
/// definition rather than a condition repeated at each call site — the shape
/// that let `--draft` be mistaken for a directory in the first place.
fn positional_dir(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += 2; // skip the flag and its value
        } else {
            return Some(args[i].clone());
        }
    }
    None
}

/// stdio serve loop. Requests (one per line):
///   `GEN <ngen> <tok0> <tok1> ...`  → greedy-generate `ngen` tokens
///   `QUIT`                          → exit
/// Each response is the space-separated generated token ids, then `END`.
///
/// With `draft > 0` the MTP head drafts `draft` tokens per round and the main
/// model verifies them in one batched forward. Greedy acceptance means the
/// emitted sequence is **identical** to the non-speculative path — speculation
/// buys wall-clock, never different tokens — which is what
/// `speculative_matches_greedy` asserts.
///
/// Returns `Ok` on a clean shutdown (EOF/QUIT). A write error (e.g. the client
/// closed the pipe) is propagated so the caller can exit quietly rather than
/// panicking mid-response.
fn serve(model: &mut Model, draft: usize) -> Result<(), Error> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();
    // `COLI_PERF_COUNTERS=1`. Opened **here**, on the thread that will run the
    // decode loop, because `perf_event_open` follows the calling thread
    // (pid = 0): opened anywhere else it would count a thread that does no
    // inference. `None` when the kernel refuses — paranoid level, seccomp, or
    // no PMU, which is most VMs — and the report below simply says nothing.
    //
    // Handed to the model rather than kept local, so the per-forward delta is
    // available to `COLI_PERF_PREFETCH_FEEDBACK`. The shutdown report below now
    // reads it back through `Model::llc_misses`, which is the same counter and
    // the same thread — moving ownership does not change what is measured.
    if let Some(c) = peregrine_model::open_l3_miss_counter() {
        model.attach_perf_counter(c);
    }
    // `COLI_PREDICT_SOURCE`: force a specific prefetch predictor. Load picks the
    // strongest the artifacts allow, so this exists to select a *weaker* arm —
    // which is what `COLI_PREDICT_EVAL`'s scoreboard needs to be actionable.
    if let Some(name) = model.apply_predictor_override() {
        eprintln!("peregrine: prefetch predictor = {name} (COLI_PREDICT_SOURCE)");
    }
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
                    let toks = if draft > 0 {
                        // Greedy acceptance, so this returns exactly what the
                        // line below would have.
                        model.generate_speculative(&prompt, ngen, draft)?
                    } else {
                        let mut sampler = Sampler::new(0.0, 0.9, 1); // greedy = deterministic
                        model.generate(&prompt, ngen, &mut sampler)?
                    };
                    let rendered: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                    out.write_all(rendered.join(" ").as_bytes())?;
                    out.write_all(b"\n")?;
                    // warm-tier diagnostics (stderr): how much of the expert I/O the
                    // cache absorbed this request.
                    if let Some((h, m, d)) = model.ecache_stats() {
                        let hr = 100.0 * h as f64 / (h + m).max(1) as f64;
                        let pf = model.ecache_prefetch_reads().unwrap_or(0);
                        eprintln!("[ecache] hits={h} misses={m} disk_reads={d} prefetch_reads={pf} hit_rate={hr:.1}%");
                        // Per-layer breakdown for the busiest sparse layers.
                        // The aggregate hides the thing an operator acts on:
                        // reads are not spread evenly across layers, and the
                        // layout schedule and tier plan are both per-layer
                        // decisions. `ecache_disk_reads_for_layer` existed for
                        // this and had no caller, test or otherwise.
                        let first = model.cfg.first_dense as usize;
                        let last = model.cfg.n_layers as usize;
                        let mut per: Vec<(usize, u64)> = (first..last)
                            .filter_map(|l| model.ecache_disk_reads_for_layer(l).map(|n| (l, n)))
                            .filter(|&(_, n)| n > 0)
                            .collect();
                        per.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        if !per.is_empty() {
                            let top: Vec<String> =
                                per.iter().take(5).map(|(l, n)| format!("L{l}={n}")).collect();
                            eprintln!("[ecache] busiest layers by disk reads: {}", top.join(" "));
                        }
                        // prefetch effectiveness: how many speculative reads paid off,
                        // plus fadvise hints and (opt-in) verify mismatches.
                        let (used, wasted) = model.ecache_prefetch_effectiveness().unwrap_or((0, 0));
                        let acc = 100.0 * model.prefetch_accuracy().unwrap_or(0.0);
                        let fadv = model.ecache_fadvise_hints().unwrap_or(0);
                        let vm = model.ecache_verify_mismatch().unwrap_or(0);
                        // `accuracy` is `used/(used+wasted)`, and `wasted` is only
                        // incremented on **eviction** (`warmcache.rs`) — a prefetched
                        // slab still resident at shutdown is in neither term. So the
                        // denominator is the classified subset, not what was issued,
                        // and quoting `accuracy` alone is survivorship-biased: the
                        // first serving-path run classified 64 of 433 reads. Both
                        // rates are printed so neither can be read as the other.
                        let unclassified = pf.saturating_sub(used + wasted);
                        let yield_pct = if pf > 0 { 100.0 * used as f64 / pf as f64 } else { 0.0 };
                        eprintln!(
                            "[prefetch] used={used} wasted={wasted} unclassified={unclassified} \
                             accuracy={acc:.1}% (of {} classified) yield={yield_pct:.1}% (of {pf} issued) \
                             fadvise={fadv} verify_mismatch={vm}",
                            used + wasted
                        );
                        // Compression only reports when COLI_CACHE_COMPRESS is on.
                        // Printing the achieved ratio keeps the feature honest: the
                        // payload is packed int4 nibbles, so ~1.2x is the ceiling and
                        // a value near 1.0 means every hit pays a decode for nothing.
                        if let Some(r) = model.ecache_compression_ratio() {
                            eprintln!("[ecache-compress] ratio={r:.2}x");
                        }
                    }
                    // GPU expert-lane transfer counters. Silent when the lane never
                    // ran (no CUDA build, or COLI_GPU unset), so a CPU-only run is
                    // unchanged. `transfer_frac` needs COLI_CUDA_PROFILE; when it is
                    // high the lane is PCIe-bound and COLI_PCIE_BUDGET_MB is the knob.
                    // Routing gate mass (COLI_GATE_STATS=1). Each routed expert costs a
                    // full weight read regardless of its gate weight, so the share
                    // below each threshold is the share of the disk budget that
                    // bought almost nothing. Silent unless asked.
                    report_gate_stats();
                    // Router look-ahead (COLI_ROUTER_LOOKAHEAD, on by default).
                    // `issued` is speculative reads started from the *next* layer's
                    // own router during the boundary the disk would otherwise spend
                    // idle. Deliberately reported apart from the [prefetch] line
                    // above: a speculative read is not a demand access, and folding
                    // it in would make a look-ahead that guessed wrong look like a
                    // cache that performed badly. Silent when it never fired.
                    let la = peregrine_model::lookahead_issued();
                    if la > 0 {
                        eprintln!("[lookahead] issued={la}");
                    }
                    // Predictor scoreboard (COLI_PREDICT_EVAL=1). Recall is the share
                    // of the next layer's real routing each predictor named; `p@r` is
                    // precision by rank, and a steep profile is what justifies
                    // prefetching only the head of a ranking. Compare every arm
                    // against `prev-token`: that baseline costs nothing and the warm
                    // cache already exploits it, so a predictor that fails to beat it
                    // has bought nothing whatever its recall looks like alone.
                    if let Some((arms, layers)) = model.predict_eval_report() {
                        eprintln!("[predict-eval] scored {layers} layer transitions, width={}",
                            arms.first().map(|a| a.precision_at.len()).unwrap_or(0));
                        for a in &arms {
                            let by_rank: Vec<String> =
                                a.precision_at.iter().map(|p| format!("{:.0}%", 100.0 * p)).collect();
                            eprintln!(
                                "[predict-eval] {:<16} recall={:.1}% precision={:.1}% silent={}/{} p@r=[{}]",
                                a.name,
                                100.0 * a.recall,
                                100.0 * a.precision,
                                a.silent,
                                a.asked,
                                by_rank.join(" ")
                            );
                        }
                    }
                    // LLC misses on this thread (COLI_PERF_COUNTERS=1). Scoped
                    // deliberately: the counter follows one thread, so this is
                    // attention and the deterministic reduce, *not* the io_uring
                    // workers or the `peregrine-par` pool. A whole-process figure
                    // would need a counter per thread, and reporting this one as
                    // if it were that is how a number stops meaning anything.
                    if let Some(misses) = model.llc_misses() {
                        eprintln!("[perf] llc-misses={misses} (decode thread only)");
                    }
                    // Batch-union sharing (COLI_UNION_STATS=1). `share` is how many
                    // routed selections each distinct expert read actually served —
                    // the amortization batching is supposed to buy. benchmarks.md
                    // credits the 4.4x aggregate gain at B=16 "entirely" to this,
                    // while a union model over GLM-5.2's 256-expert top-8 layers
                    // predicts only ~1.26x. This line reads it off the live engine
                    // rather than deriving it. Silent unless asked.
                    report_union_stats();
                    let g = model.telemetry().gpu;
                    if g.calls > 0 {
                        match g.transfer_fraction() {
                            Some(f) => eprintln!(
                                "[gpu] calls={} experts={} rows={} h2d={:.1}ms kernel={:.1}ms d2h={:.1}ms transfer_frac={:.0}%",
                                g.calls, g.experts, g.rows, g.h2d_ms, g.kernel_ms, g.d2h_ms, 100.0 * f
                            ),
                            None => eprintln!(
                                "[gpu] calls={} experts={} rows={} (set COLI_CUDA_PROFILE=1 for transfer timings)",
                                g.calls, g.experts, g.rows
                            ),
                        }
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
/// batched decode steps over B independent sequences and report aggregate tokens/sec.
/// On a streaming model this shows the disk amortization (experts read once per
/// step, shared across B); on a resident model it shows compute scaling.
/// `COLI_MODEL` selects the model; args override the B set.
///
/// Speculation: `--draft N` (or `COLI_DRAFT=N`) engages **batched MTP speculative
/// verify** — each step drafts `γ` tokens per sequence with the MTP head, then the
/// B·(1+γ) rows share **one** read-union per step via `verify_drafts_batched`. The
/// disk-rate ceiling is the same as for one decode forward, so every accepted draft
/// is a decoded token the disk did not pay for — the central lever on a
/// bandwidth-bound engine. Bit-identical to greedy per-token decode (asserted by
/// `speculative_rows_emit_exactly_what_greedy_would` in the model tests).
///
/// Reports both the forward-issued tokens/sec (`agg`) and the verified tokens/sec
/// (`tok/s`), so a draft that inflates `agg` with low acceptance can't hide.
/// Falls back to the plain `forward_step_batched` path when `--draft 0` (the
/// default) or the checkpoint has no MTP head — bit-identical to before this flag
/// existed.
/// Routing gate mass (`COLI_GATE_STATS=1`). Each routed expert costs a full
/// weight read regardless of its gate weight, so the share below each threshold
/// is the share of the disk budget that bought almost nothing. Silent unless
/// asked.
fn report_gate_stats() {
    if let Some((below, total)) = peregrine_model::gate_stats_snapshot() {
        let pct = |n: u64| 100.0 * n as f64 / total as f64;
        eprintln!(
            "[gate] routed={total} below_0.5%={:.1}% below_1%={:.1}% below_2%={:.1}% below_5%={:.1}%",
            pct(below[0]),
            pct(below[1]),
            pct(below[2]),
            pct(below[3])
        );
    }
}

/// Batch-union sharing and the all-low-gate ceiling (`COLI_UNION_STATS=1`).
///
/// **Shared with `run_bench` on purpose, and that is the point of extracting
/// it.** Both figures lived only in `serve`, which is strictly single-sequence —
/// so `s_n` was a prefill chunk length and the numbers described sharing across
/// *positions of one prompt*, not across concurrent sequences. The question
/// they exist to answer (§13: is per-token precision selection worth building?)
/// is specifically about what happens as the batch grows, which is exactly the
/// regime `serve` cannot enter. `run_bench` is the only batched entry point and
/// printed neither.
///
/// Read them together. `share` is how many routed selections each distinct read
/// served — the amortization batching buys. `all-low-gate` is the fraction of
/// reads where *every* row wanting the expert wanted it weakly, which is the
/// ceiling on any per-token precision scheme: a read is issued per union entry,
/// not per row, so one row leaning on an expert forces the wider precision for
/// all of them. The two move in opposite directions as B grows, and that tension
/// is the finding — not either number alone.
fn report_union_stats() {
    if let Some((sel, distinct, calls)) = peregrine_model::union_stats_snapshot() {
        let share = if distinct > 0 { sel as f64 / distinct as f64 } else { 0.0 };
        eprintln!("[union] selections={sel} distinct={distinct} calls={calls} share={share:.3}x");
    }
    if let Some((all_low, distinct)) = peregrine_model::union_low_gate_snapshot() {
        let f = if distinct > 0 { all_low as f64 / distinct as f64 } else { 0.0 };
        eprintln!(
            "[union] all-low-gate reads={all_low}/{distinct} ({:.1}%) — the ceiling on \
             per-token precision selection under batch union",
            f * 100.0
        );
    }
}

fn run_bench(batch_args: &[String]) -> Result<(), Error> {
    // `bench` does not pass a flag-style arg to `serve`, but it DOES read `--draft`
    // / `COLI_DRAFT` here. Using the same `draft_depth` helper that serve uses
    // keeps the meaning of `--draft 4` identical across the two entry points.
    let draft = draft_depth(batch_args);
    let dir = std::env::var("COLI_MODEL")
        .ok()
        .ok_or_else(|| Error::Format("bench needs COLI_MODEL=<dir>  (e.g. a tiny model from `peregrine build`)".into()))?;
    let steps: usize =
        std::env::var("COLI_BENCH_STEPS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(3);
    let batches: Vec<usize> = if batch_args.is_empty() {
        vec![1, 4, 16]
    } else {
        // Skip the `--draft` flag and its integer value, so `--draft 4 16` parses
        // as the batch list `[16]` rather than `[4, 16]`. `--draft`-with-no-value
        // is illegal (the parser already returned 0 above) so this `step_by(2)`
        // can't run off the end of `args`.
        let mut filtered: Vec<String> = Vec::with_capacity(batch_args.len());
        let mut skip_next = false;
        for a in batch_args {
            if skip_next {
                skip_next = false;
                continue;
            }
            if a == "--draft" {
                skip_next = true;
                continue;
            }
            filtered.push(a.clone());
        }
        filtered.into_iter().filter_map(|s| s.parse().ok()).filter(|&b| b > 0).collect()
    };

    let t0 = std::time::Instant::now();
    let model = Model::load(Path::new(&dir))?;
    let vocab = model.cfg.vocab as usize;
    let has_mtp = model.has_mtp();
    if draft > 0 && !has_mtp {
        return Err(Error::Format(format!(
            "bench --draft {draft} needs an MTP head, and this checkpoint has none \
             (convert with --mtp, or drop the flag)"
        )));
    }
    eprintln!(
        "peregrine bench: loaded {} layers, vocab {}, mtp={}, draft=γ={draft}, in {:.1}s",
        model.cfg.n_layers,
        vocab,
        if has_mtp { "yes" } else { "no" },
        t0.elapsed().as_secs_f64()
    );
    let spec = draft > 0 && has_mtp;
    if spec {
        eprintln!("peregrine bench: speculative batched-verify on (γ={draft} draft tokens per sequence per step)");
    }
    println!("  batch   steps   tokens    seconds     agg tok/s   tok/s   per-seq tok/s");
    let d = model.cfg.hidden as usize;
    for &b in &batches {
        let mut seqs: Vec<SeqKv> = (0..b).map(|_| SeqKv::new(&model.cfg)).collect();
        // distinct starting token per sequence so their routing diverges
        let mut toks: Vec<i32> = (0..b).map(|i| (i as i32 * 7 + 1) % vocab.max(1) as i32).collect();
        // per-sequence last accepted position — speculation rewinds KV to it each step
        let mut pos_of: Vec<usize> = (0..b).map(|_| 0).collect();
        // Hidden states of the last accepted row per sequence — seeded inside the
        // speculative branch's first forward (the one that consumes the starting
        // token) and consumed by the next draft round. Left empty on the
        // non-speculative path; the bench harness reads it only when `spec`.
        let mut hlast: Vec<f32>;
        let (d0, _) = model.ecache_stats().map(|(_, _, d)| (d, ())).unwrap_or((0, ()));

        let t = std::time::Instant::now();
        let mut verified_tokens = 0usize;
        if !spec {
            // Historical non-speculative path: keep the existing bench loop. The
            // bench harness documented in `docs/benchmarks.md` runs this arm, so
            // removing it would silently change the headline numbers.
            for step in 0..steps {
                let pos: Vec<usize> = vec![step; b];
                let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
                let logits = model.forward_step_batched(&toks, &mut refs, &pos, None)?;
                pick_batch_greedy(&logits, vocab, &mut toks);
                verified_tokens += b;
            }
        } else {
            // Speculative batched verify. Step 0 must seed `hlast` with the
            // hidden state of each sequence's starting token, since a draft
            // starts from its committed hidden state — the verify pass would
            // compute it anyway, but doing it once up front lets the round-1
            // draft overlap with nothing on the disk and still be correct.
            //
            // `forward_rows_batched_hidden` returns the hidden of every row it
            // was given, so the same one-shot forward serves both as the
            // step-0 broadcast and as the seed for the first speculative round.
            {
                let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
                let owner: Vec<usize> = (0..b).collect();
                let pos0: Vec<usize> = vec![0; b];
                let (_logits, hidden) = model.forward_rows_batched_hidden(&toks, &owner, &mut refs, &pos0, None)?;
                // pick next tokens from this forward, and stash each row's
                // pre-final-norm hidden for the next round's `mtp_draft`.
                let next: Vec<i32> = (0..b)
                    .map(|s| {
                        let row = &_logits[s * vocab..(s + 1) * vocab];
                        peregrine_model::argmax(row) as i32
                    })
                    .collect();
                // Each sequence's first committed row is the starting token
                // at position 0, so its hidden is row `s`. The next round
                // decodes `next[s]` at position 1.
                hlast = (0..b).flat_map(|s| hidden[s * d..s * d + d].iter().copied()).collect();
                toks = next;
                pos_of.iter_mut().for_each(|p| *p = 1);
                verified_tokens += b; // each sequence committed its starting token
            }
            for _step in 0..steps.saturating_sub(1) {
                // draft γ tokens per sequence off its own last committed hidden.
                // Each sequence drafts independently — drafts do not touch the main
                // KV, so this is all `&Model` and could run in parallel across
                // sequences; a depth-4 draft on GLM-5.2 is ~1 MTP layer vs 78
                // sparse layers, so the serial cost is well under 1 % of step time.
                let mut drafts: Vec<Vec<i32>> = Vec::with_capacity(b);
                for s in 0..b {
                    let hs = &hlast[s * d..(s + 1) * d];
                    let draft = model.mtp_draft(toks[s], draft, hs)?;
                    drafts.push(draft);
                }
                // verify `[next, draft...]` for every sequence in ONE forward —
                // the union of routed experts is shared across all B sequences
                // and their drafts, so the per-step disk budget is one read per
                // distinct expert in that union, not per row.
                let mut refs: Vec<&mut SeqKv> = seqs.iter_mut().collect();
                let (verified, hidden_after) =
                    model.verify_drafts_batched(&toks, &drafts, &mut refs, &pos_of)?;
                // count accepted drafts + the always-confirmed `next` token
                let mut next_hlast = vec![0f32; b * d];
                let mut next_toks = vec![0i32; b];
                for (s, v) in verified.iter().enumerate() {
                    verified_tokens += v.tokens.len();
                    next_toks[s] = v.next;
                    pos_of[s] += v.tokens.len();
                    next_hlast[s * d..(s + 1) * d].copy_from_slice(&hidden_after[s * d..(s + 1) * d]);
                }
                toks = next_toks;
                hlast = next_hlast;
            }
        }
        let dt = t.elapsed().as_secs_f64().max(1e-9);
        let agg = (b * steps) as f64 / dt; // forward-issued tokens
        let vps = verified_tokens as f64 / dt; // actually decoded tokens
        let token_count = b * steps;
        println!(
            "  {b:5}   {steps:5}   {token_count:6}   {dt:8.2}   {agg:11.3}   {vps:8.3}   {:13.4}",
            vps / b as f64
        );
        if let Some((_, _, d)) = model.ecache_stats() {
            eprintln!("    [ecache] disk_reads this batch: {}", d.saturating_sub(d0));
        }
    }
    // The batched figures §13 needs. Printed once at the end rather than per
    // batch point because the counters are process-wide cumulative — and note
    // `benchmarks.md`'s finding that running several batch sizes in one process
    // inflates the later ones, so a sweep's union numbers are a blend. **Use one
    // batch size per fresh process** when the number is meant to be attributed
    // to a specific B.
    report_gate_stats();
    report_union_stats();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(rest: &[&str]) -> Vec<String> {
        std::iter::once("peregrine".to_string()).chain(rest.iter().map(|s| s.to_string())).collect()
    }

    #[test]
    fn positional_dir_skips_flags_and_their_values() {
        assert_eq!(positional_dir(&argv(&["/models/glm"])).as_deref(), Some("/models/glm"));
        // The case that motivated this: the flag's *value* must not be mistaken
        // for the directory either, so "4" is skipped along with "--draft".
        assert_eq!(positional_dir(&argv(&["--draft", "4", "/models/glm"])).as_deref(), Some("/models/glm"));
        assert_eq!(positional_dir(&argv(&["/models/glm", "--draft", "4"])).as_deref(), Some("/models/glm"));
        // Flags only, no directory → None, so the caller prints usage instead of
        // trying to load a path named "--draft".
        assert_eq!(positional_dir(&argv(&["--draft", "4"])), None);
        assert_eq!(positional_dir(&argv(&[])), None);
    }

    #[test]
    fn draft_depth_defaults_off_and_reads_the_flag() {
        // Off unless asked: the historical non-speculative path is the default.
        // (COLI_DRAFT is process-wide, so this asserts the flag path and the
        // no-flag path only when the env is unset — the same care
        // `union_stats_are_silent_unless_asked` takes.)
        assert_eq!(draft_depth(&argv(&["--draft", "6", "/m"])), 6);
        assert_eq!(draft_depth(&argv(&["/m", "--draft", "1"])), 1);
        if std::env::var("COLI_DRAFT").is_err() {
            assert_eq!(draft_depth(&argv(&["/m"])), 0, "off by default");
            // A malformed value is not silently taken as 1 or as "on".
            assert_eq!(draft_depth(&argv(&["--draft", "banana", "/m"])), 0);
            assert_eq!(draft_depth(&argv(&["--draft"])), 0, "flag with no value");
        }
    }
}

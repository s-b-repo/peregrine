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
                        // prefetch effectiveness: how many speculative reads paid off,
                        // plus fadvise hints and (opt-in) verify mismatches.
                        let (used, wasted) = model.ecache_prefetch_effectiveness().unwrap_or((0, 0));
                        let acc = 100.0 * model.prefetch_accuracy().unwrap_or(0.0);
                        let fadv = model.ecache_fadvise_hints().unwrap_or(0);
                        let vm = model.ecache_verify_mismatch().unwrap_or(0);
                        eprintln!("[prefetch] used={used} wasted={wasted} accuracy={acc:.1}% fadvise={fadv} verify_mismatch={vm}");
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
                    // Batch-union sharing (COLI_UNION_STATS=1). `share` is how many
                    // routed selections each distinct expert read actually served —
                    // the amortization batching is supposed to buy. benchmarks.md
                    // credits the 4.4x aggregate gain at B=16 "entirely" to this,
                    // while a union model over GLM-5.2's 256-expert top-8 layers
                    // predicts only ~1.26x. This line reads it off the live engine
                    // rather than deriving it. Silent unless asked.
                    if let Some((sel, distinct, calls)) = peregrine_model::union_stats_snapshot() {
                        let share = if distinct > 0 { sel as f64 / distinct as f64 } else { 0.0 };
                        eprintln!(
                            "[union] selections={sel} distinct={distinct} calls={calls} share={share:.3}x"
                        );
                    }
                    // The number that decides gate-mass mixed-precision loading.
                    // A read is issued per *union entry*, not per row, so an
                    // expert one row leans on and another barely wants must be
                    // read at the higher precision. Only experts low-gate for
                    // *every* row that wants them could be read narrower — and
                    // that share shrinks as the batch grows, which is the
                    // tension with the amortization the line above measures.
                    if let Some((all_low, distinct)) = peregrine_model::union_low_gate_snapshot() {
                        let f = if distinct > 0 { all_low as f64 / distinct as f64 } else { 0.0 };
                        eprintln!(
                            "[union] all-low-gate reads={all_low}/{distinct} ({:.1}%) — the ceiling on \
                             per-token precision selection under batch union",
                            f * 100.0
                        );
                    }
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

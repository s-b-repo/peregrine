//! Command line interface.
//!
//! ```text
//! peregrine-ctx demo
//! peregrine-ctx ingest notes/            index a file or a directory
//! peregrine-ctx ask "what is the server ip?" --budget 600
//! peregrine-ctx plan "why did we roll back?" --explain
//! peregrine-ctx explain <node_id>
//! peregrine-ctx stats
//! peregrine-ctx bench [--turns N] [--window N] [--scaling]
//! ```
//!
//! Globs are left to the shell, which already does them better than a
//! hand-rolled matcher would.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use peregrine_ctx::bench::{
    run_ablation, run_baselines, run_benchmark, run_consolidation, run_coverage, run_decay, run_multi_hop, run_channels, run_mutation_probe, run_stages,
    run_cache_layout, run_fusion, run_recall, run_scaling_diverse, run_subject_control,
    run_poison,
    run_rebuild, run_scaling, run_sweep, run_tamper,
};
use peregrine_ctx::demo::run_demo;
use peregrine_ctx::graph::DcrError;
use peregrine_ctx::context_store::ContextStore;
use peregrine_ctx::runtime::Dcr;
use peregrine_ctx::scrub::{ScrubOptions, scrub};

const DEFAULT_STORE: &str = ".dcr.json";
const USAGE: &str = "\
peregrine-ctx — Dynamic Context Runtime

usage: peregrine-ctx [--store PATH] [--budget N] <command> [options]

commands:
  ingest <path>...            index files or directories into the memory runtime
  ask <query> [--show-context]  plan a working set and answer a question
  plan <query> [--explain]    show the active context without calling a model
  skill add <name> <adapter> [description...]
                              register a capability (adapter = content hash/URI)
  skill list                  show registered skills
  explain <node_id>           audit path from a node down to raw spans
  stats                       telemetry report
  verify                      check a .context container: objects, chain, root
  scrub [--repair]            detect bit rot; repair from a verified replica
  checkpoint                  seal the current state as a new generation
  quarantine                  list objects that failed verification
  demo                        worked example through all four ladder levels
  serve [--port N] [--sessions N]
                              share sessions over TCP: ASK/INGEST, one runtime
                              per conversation (default port 7878, 256 sessions)
  serve-context [--dir PATH] [--max-hits N] [--once] [--idle-ms N]
                              answer unifur's context requests over the file
                              channel in --dir (default /srv/m-sda/unifur/interop):
                              every <id>.request.json gets a planned working set
                              written back as <id>.response.json, and every
                              <id>.put.json is ingested and acknowledged as
                              <id>.put-ack.json
  bench [--turns N] [--window N]
                              DCR vs full context vs sliding window
        --scaling             does k stay flat while history grows?
        --ablate              which mechanism carries which probe?
        --mutate              is a correction served once the original has dependents?
        --multihop            does graph expansion buy anything when a join is needed?
        --decay               does a recency prefilter buy latency without costing recall?
        --consolidate         correctness when the store is written mid-turn
        --diverse             scaling on a lexically varied corpus, to millions of tokens
        --subject             does it identify the subject, or the document that mentions it?
        --cache               how much of each turn is a cacheable prefix of the last?
        --recall              approximate top-k overlap against the exact scan
        --fusion              reciprocal rank fusion against the linear blend
        --sweep               correctness and cost against B_attention
        --coverage            read coverage as history grows (offline dual)
        --poison              positive control: can stale_fact_read_rate fire?
        --baselines           DCR vs RAG, summarize-all and recursive context
        --rebuild             what does destroying and rebuilding the workspace cost?
        --tamper              can the container actually detect tampering?

options:
  --store PATH                persisted memory (default: .dcr.json).
                              No extension (.context, memory/) is a tamper-evident
                              container; a file extension is the plain JSON store.
  --replica PATH              extra full copy of the object store; repeatable.
                              Repair reads only from a copy that verifies.
  --budget N                  B_attention, in tokens (default: 1200)
  --prefix-stable             ask/plan: render with a turn-invariant header and
                              content-addressed ordering, so consecutive turns
                              share a cacheable prefix (see bench --cache)
  --stickiness W              ask/plan: utility bonus W for nodes retained from
                              the previous window (default: 0, off)
  --endpoint HOST:PORT        ask: answer with an OpenAI-compatible local server
                              (peregrine-serve, vLLM) instead of the offline
                              line-matcher
  --model NAME                model field sent to --endpoint (default: peregrine)
   --save-every N              ingest: checkpoint the store every N files (default 0, only at
                               the end). A long ingest that crashes keeps the last batch.
  --quiet                     ingest: drop the per-file report (a 208k-file corpus makes it
                               the bulk of the output)
  --fit-gate                  ask: when the whole store fits in the window, send
                               raw history instead of planning (needs a reasoner
                               that reads raw text — see --endpoint)

serve-context options:
  --dir PATH                   the channel directory unifur submits into
                               (default /srv/m-sda/unifur/interop)
  --max-hits N                 cap on hits per answer, whatever the query asks
                               for (default 8)
  --once                       answer one pass and exit, instead of looping
  --idle-ms N                  sleep between empty passes (default 200)
";


struct Args {
    store: PathBuf,
    budget: usize,
    command: String,
    positional: Vec<String>,
    flags: Vec<String>,
    turns: usize,
    window: usize,
    replicas: Vec<PathBuf>,
    stickiness: Option<f32>,
    endpoint: Option<String>,
    model: Option<String>,
    port: u16,
    sessions: usize,
    channel_dir: PathBuf,
    max_hits: usize,
    once: bool,
    idle_ms: u64,
    save_every: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut store = PathBuf::from(DEFAULT_STORE);
    let mut budget = 1200usize;
    let mut turns = 300usize;
    let mut window = 8000usize;
    let mut command = String::new();
    let mut positional = Vec::new();
    let mut flags = Vec::new();
    let mut replicas: Vec<PathBuf> = Vec::new();
    let mut stickiness: Option<f32> = None;
    let mut endpoint: Option<String> = None;
    let mut model: Option<String> = None;
    let mut port: u16 = 7878;
    let mut sessions: usize = 256;
    let mut channel_dir = PathBuf::from(peregrine_ctx::interop::DEFAULT_CHANNEL_DIR);
    let mut max_hits: usize = 8;
    let mut once = false;
    let mut idle_ms: u64 = 200;
    let mut save_every: usize = 0;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--store" => store = PathBuf::from(argv.next().ok_or("--store needs a path")?),
            "--replica" => {
                replicas.push(PathBuf::from(argv.next().ok_or("--replica needs a path")?))
            }
            "--budget" => {
                budget = argv
                    .next()
                    .ok_or("--budget needs a number")?
                    .parse()
                    .map_err(|_| "--budget must be a number")?
            }
            "--turns" => {
                turns = argv
                    .next()
                    .ok_or("--turns needs a number")?
                    .parse()
                    .map_err(|_| "--turns must be a number")?
            }
            "--window" => {
                window = argv
                    .next()
                    .ok_or("--window needs a number")?
                    .parse()
                    .map_err(|_| "--window must be a number")?
            }
            "--stickiness" => {
                stickiness = Some(
                    argv.next()
                        .ok_or("--stickiness needs a weight")?
                        .parse()
                        .map_err(|_| "--stickiness must be a number")?,
                )
            }
            "--endpoint" => {
                endpoint = Some(argv.next().ok_or("--endpoint needs host:port")?)
            }
            "--model" => model = Some(argv.next().ok_or("--model needs a name")?),
            "--port" => {
                port = argv
                    .next()
                    .ok_or("--port needs a number")?
                    .parse()
                    .map_err(|_| "--port must be a number")?
            }
            "--sessions" => {
                sessions = argv
                    .next()
                    .ok_or("--sessions needs a number")?
                    .parse()
                    .map_err(|_| "--sessions must be a number")?
            }
            "--dir" => {
                channel_dir = PathBuf::from(argv.next().ok_or("--dir needs a path")?)
            }
            "--max-hits" => {
                max_hits = argv
                    .next()
                    .ok_or("--max-hits needs a number")?
                    .parse()
                    .map_err(|_| "--max-hits must be a number")?
            }
            "--save-every" => {
                save_every = argv
                    .next()
                    .ok_or("--save-every needs a number")?
                    .parse()
                    .map_err(|_| "--save-every must be a number")?
            }
            "--once" => once = true,
            "--idle-ms" => {
                idle_ms = argv
                    .next()
                    .ok_or("--idle-ms needs a number")?
                    .parse()
                    .map_err(|_| "--idle-ms must be a number")?
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            flag if flag.starts_with("--") => flags.push(flag.to_string()),
            value if command.is_empty() => command = value.to_string(),
            value => positional.push(value.to_string()),
        }
    }
    if command.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(Args {
        store,
        budget,
        command,
        positional,
        flags,
        turns,
        window,
        replicas,
        stickiness,
        endpoint,
        model,
        port,
        sessions,
        channel_dir,
        max_hits,
        once,
        idle_ms,
        save_every,
    })
}

/// Which format `--store` names.
///
/// An existing directory is a container. A path that does not exist yet is a
/// container when it has no file extension (`.context`, `memory/`) and a plain
/// JSON store when it has one (`.dcr.json`) — so the same flag creates either,
/// the default `.dcr.json` keeps working, and the choice can be read off the
/// path instead of remembered. A leading dot is not an extension, which is
/// what makes `.context` a directory rather than a file.
fn is_container(store: &Path) -> bool {
    if store.is_dir() {
        return true;
    }
    if store.is_file() {
        return false;
    }
    store.extension().is_none()
}

/// Does a container already exist here, as opposed to being where one goes?
fn container_exists(store: &Path) -> bool {
    store.join("manifest").is_file()
}

fn open(store: &Path, budget: usize) -> Result<Dcr, DcrError> {
    if is_container(store) {
        if !container_exists(store) {
            // The directory is where the container will be written, not one to
            // read. First ingest creates it.
            return Ok(Dcr::new(budget));
        }
        let mut runtime = Dcr::open_context(store, budget)?;
        runtime.set_budget(budget);
        return Ok(runtime);
    }
    if store.exists() {
        let mut runtime = Dcr::load(store, budget)?;
        runtime.set_budget(budget);
        Ok(runtime)
    } else {
        Ok(Dcr::new(budget))
    }
}

/// Persist back to whichever format the store is in. Sealing a container is a
/// new generation, so every write is an appended, chained state rather than an
/// overwrite.
fn persist(runtime: &Dcr, store: &Path) -> Result<(), DcrError> {
    if is_container(store) {
        let checkpoint = runtime.save_context(store, None)?;
        println!(
            "generation {} · root {} · {} objects",
            checkpoint.generation,
            checkpoint.merkle_root.short(12),
            checkpoint.object_count
        );
        Ok(())
    } else {
        runtime.save(store)
    }
}

/// Open the container behind `--store`, or explain that there is not one.
fn container(store: &Path) -> Result<ContextStore, String> {
    if !container_exists(store) {
        return Err(format!(
            "{} is not a .context container.\n\
             Create one with: peregrine-ctx ingest <path> --store {}",
            store.display(),
            store.display()
        ));
    }
    ContextStore::open(store).map_err(|e| e.to_string())
}

/// A file, or every file under a directory — modification-time ordered.
///
/// Ingest order is revision order, so a directory listed by name would let
/// `a.md` supersede `z.md`.
fn expand(pattern: &str) -> Vec<PathBuf> {
    let path = Path::new(pattern);
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    if path.is_dir() {
        let mut found: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(meta) = entry.metadata() {
                    let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                    found.push((modified, path));
                }
            }
        }
        found.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        return found.into_iter().map(|(_, p)| p).collect();
    }
    Vec::new()
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let to_err = |e: DcrError| e.to_string();

    match args.command.as_str() {
        "ingest" => {
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            let mut total = 0usize;
            let mut done = 0usize;
            // Checkpoint as it goes. A full-corpus ingest is a long job —
            // the cybercorpus tree is ~208k files, measured at roughly 19
            // hours on this box — and saving only at the end means a crash, a
            // reboot or an OOM throws away the whole run. `--save-every`
            // bounds the loss to the last batch; `--quiet` drops the per-file
            // chatter, which at 208k files is both the bulk of the output and
            // a measurable part of the wall time.
            let save_every = args.save_every;
            let quiet = args.flags.iter().any(|f| f == "--quiet");
            for pattern in &args.positional {
                let matches = expand(pattern);
                if matches.is_empty() {
                    eprintln!("no match: {pattern}");
                }
                for path in matches {
                    let result = runtime.ingest_file(&path).map_err(to_err)?;
                    total += result.nodes.len();
                    done += 1;
                    if !quiet {
                        println!("{}: {}", path.display(), result.summary());
                        for (old, new) in &result.contradictions {
                            println!("  contradiction: {old} superseded by {new}");
                        }
                    }
                    if save_every > 0 && done % save_every == 0 {
                        persist(&runtime, &args.store).map_err(to_err)?;
                        eprintln!("checkpoint at {done} file(s), {total} state nodes");
                    }
                }
            }
            persist(&runtime, &args.store).map_err(to_err)?;
            if quiet {
                eprintln!("{done} file(s); {total} state nodes; store: {}", args.store.display());
            } else {
                println!("\n{total} state nodes; store: {}", args.store.display());
            }
        }
        "ask" => {
            let query = args.positional.first().ok_or("ask needs a query")?;
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            runtime.prefix_stable = args.flags.iter().any(|f| f == "--prefix-stable");
            runtime.fit_gate = args.flags.iter().any(|f| f == "--fit-gate");
            if let Some(w) = args.stickiness {
                runtime.planner.weights.stickiness = w;
            }
            if let Some(endpoint) = &args.endpoint {
                runtime.set_reasoner(Box::new(peregrine_ctx::llm::HttpReasoner::new(
                    endpoint.clone(),
                    args.model.clone().unwrap_or_else(|| "peregrine".to_string()),
                )));
            }
            if runtime.fit_gate && args.endpoint.is_none() {
                eprintln!(
                    "warning: --fit-gate without --endpoint sends raw history to the offline \
                     line-matcher, which cannot read it — answers will be empty. The gate is \
                     meant for a real reasoner (--endpoint HOST:PORT)."
                );
            }
            let answer = runtime.ask(query, None);
            println!("{}", answer.text);
            println!(
                "\n-- {}/{} tokens, {} nodes, {} escalation(s), cited {}{}{}",
                answer.tokens,
                args.budget,
                answer.context.entries.len(),
                answer.escalations,
                if answer.cited.is_empty() {
                    "nothing".to_string()
                } else {
                    answer.cited.join(", ")
                },
                if answer.skills.is_empty() {
                    String::new()
                } else {
                    format!(", skills active: {}", answer.skills.join(", "))
                },
                if answer.bypassed { ", fit bypass" } else { "" }
            );
            if args.flags.iter().any(|f| f == "--show-context") {
                println!("\n{}", answer.context.render());
            }
            persist(&runtime, &args.store).map_err(to_err)?;
        }
        "plan" => {
            let query = args.positional.first().ok_or("plan needs a query")?;
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            if let Some(w) = args.stickiness {
                runtime.planner.weights.stickiness = w;
            }
            let context = runtime.plan(query, None);
            if args.flags.iter().any(|f| f == "--prefix-stable") {
                println!("{}", context.render_prefix_stable());
            } else {
                println!("{}", context.render());
            }
            if args.flags.iter().any(|f| f == "--explain") {
                println!("\n{}", runtime.explain_plan(&context));
            }
        }
        "skill" => {
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            match args.positional.first().map(String::as_str) {
                Some("add") => {
                    let name = args.positional.get(1).ok_or("skill add needs a name")?;
                    let adapter = args
                        .positional
                        .get(2)
                        .ok_or("skill add needs an adapter reference (hash or URI)")?;
                    let description = match args.positional.get(3..) {
                        Some(words) if !words.is_empty() => words.join(" "),
                        _ => name.clone(),
                    };
                    runtime.add_skill(name, &description, adapter).map_err(to_err)?;
                    persist(&runtime, &args.store).map_err(to_err)?;
                    println!("skill {name} registered ({adapter})");
                }
                Some("list") => {
                    let mut found = false;
                    for node in runtime.graph.nodes() {
                        if node.kind != peregrine_ctx::Kind::Skill {
                            continue;
                        }
                        found = true;
                        println!(
                            "{:<20} {:<40} {}",
                            node.key.as_deref().unwrap_or("-"),
                            node.meta.source.as_deref().unwrap_or("-"),
                            node.value
                        );
                    }
                    if !found {
                        println!("no skills registered");
                    }
                }
                _ => {
                    return Err(
                        "usage: peregrine-ctx skill add <name> <adapter> [description...] | peregrine-ctx skill list"
                            .to_string(),
                    )
                }
            }
        }
        "explain" => {
            let node_id = args.positional.first().ok_or("explain needs a node id")?;
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            println!("{}", runtime.explain(node_id).map_err(to_err)?);
        }
        "stats" => {
            let runtime = open(&args.store, args.budget).map_err(to_err)?;
            print!("{}", runtime.report());
        }
        "verify" => {
            let store = container(&args.store)?;
            let report = store.verify(None);
            println!("{report}");
            println!(
                "\ngeneration {} · root {} · {}",
                store.generation(),
                store.root_hash().short(12),
                if store.manifest().signing_key_id.is_some() {
                    match store.manifest().signing_is_cryptographic {
                        true => "signed",
                        // Never let a development signature read as protection.
                        false => "signed with a NON-CRYPTOGRAPHIC key",
                    }
                } else {
                    "unsigned"
                }
            );
            if !report.ok() {
                return Err("verification failed".to_string());
            }
        }
        "scrub" => {
            let mut store = container(&args.store)?;
            for replica in args.replicas.iter() {
                store.add_replica(replica);
            }
            let options = if args.flags.iter().any(|f| f == "--repair") {
                ScrubOptions::repairing(store.generation() + 1)
            } else {
                ScrubOptions::default()
            };
            let report = scrub(&mut store, &options, None).map_err(|e| e.to_string())?;
            print!("{report}");
            if !report.clean() {
                return Err("scrub found unrepaired corruption".to_string());
            }
        }
        "checkpoint" => {
            let mut store = container(&args.store)?;
            let checkpoint = store
                .commit(store.generation() + 1, None)
                .map_err(|e| e.to_string())?;
            println!(
                "generation {} · root {} · parent {} · {} objects",
                checkpoint.generation,
                checkpoint.merkle_root.short(12),
                checkpoint.parent_root.short(12),
                checkpoint.object_count
            );
        }
        "quarantine" => {
            let store = container(&args.store)?;
            let entries = store.quarantined();
            if entries.is_empty() {
                println!("quarantine is empty");
            }
            for (id, reason) in entries {
                println!("{} — {reason}", &id[..id.len().min(16)]);
            }
        }
        "demo" => {
            run_demo(args.budget).map_err(to_err)?;
        }
        "serve" => {
            let listener = std::net::TcpListener::bind(("127.0.0.1", args.port))
                .map_err(|e| format!("bind 127.0.0.1:{}: {e}", args.port))?;
            println!(
                "peregrine-ctx serve listening on 127.0.0.1:{} — {} session slots, budget {}",
                args.port, args.sessions, args.budget
            );
            let router = std::sync::Arc::new(std::sync::Mutex::new(peregrine_ctx::SessionRouter::new(
                args.budget,
                args.sessions,
            )));
            peregrine_ctx::sessions::serve(listener, router);
        }
        "serve-context" => {
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            // Ingest whatever the operator points at before answering, so the
            // first request already has a corpus behind it. Paths come from
            // `positional` (`peregrine-ctx serve-context cybercorpus/text`), because a
            // responder that can only serve what it was started with is not a
            // context runtime, it is a fixture.
            for pattern in &args.positional {
                let matches = expand(pattern);
                if matches.is_empty() {
                    eprintln!("no match: {pattern}");
                }
                for path in matches {
                    let result = runtime.ingest_file(&path).map_err(to_err)?;
                    println!("{}: {}", path.display(), result.summary());
                }
            }
            let mut responder = peregrine_ctx::interop::Responder::new(
                &args.channel_dir,
                runtime,
                args.max_hits,
            )
            .map_err(|e| e.to_string())?;
            println!(
                "peregrine-ctx serve-context watching {} — budget {}, max {} hit(s) per answer{}",
                responder.dir().display(),
                args.budget,
                args.max_hits,
                if args.once { ", one pass" } else { "" }
            );
            if args.once {
                let report = responder.poll_once().map_err(|e| e.to_string())?;
                print!("{}", report.render());
                persist(responder.runtime_mut(), &args.store).map_err(to_err)?;
                if !report.failures.is_empty() {
                    return Err(format!(
                        "{} request(s) or write(s) could not be served; they are recorded, \
                         not fabricated",
                        report.failures.len()
                    ));
                }
            } else {
                // The loop persists its own ingested writes: a responder that
                // acknowledged a document and then restarted without it would
                // leave the peer holding an acknowledgement for a store that no
                // longer has that document. `--once` above persists on the way
                // out instead, so the flag is the only thing that differs.
                responder
                    .with_store(&args.store)
                    .serve(std::time::Duration::from_millis(args.idle_ms))
                    .map_err(|e| e.to_string())?;
            }
        }
        "bench" => {
            let has = |name: &str| args.flags.iter().any(|f| f == name);
            if has("--scaling") {
                run_scaling(&[100, 300, 1000, 3000], args.budget).map_err(to_err)?;
            } else if has("--recall") {
                run_recall(&[300, 1000, 3000], args.budget).map_err(to_err)?;
            } else if has("--fusion") {
                run_fusion(args.turns, args.budget).map_err(to_err)?;
            } else if has("--cache") {
                run_cache_layout(args.turns, args.budget).map_err(to_err)?;
            } else if has("--subject") {
                run_subject_control(args.turns, args.budget).map_err(to_err)?;
            } else if has("--diverse") {
                run_scaling_diverse(&[3_000, 10_000, 30_000, 80_000], args.budget).map_err(to_err)?;
            } else if has("--consolidate") {
                run_consolidation(args.turns, args.budget).map_err(to_err)?;
            } else if has("--decay") {
                run_decay(args.turns, args.budget).map_err(to_err)?;
            } else if has("--multihop") {
                run_multi_hop(args.turns, args.budget).map_err(to_err)?;
            } else if has("--mutate") {
                run_mutation_probe(args.turns, args.budget).map_err(to_err)?;
            } else if has("--channels") {
                run_channels(args.turns, args.budget).map_err(to_err)?;
            } else if has("--stages") {
                run_stages(&[100, 300, 1000, 3000], args.budget).map_err(to_err)?;
            } else if has("--ablate") {
                run_ablation(args.turns, args.budget).map_err(to_err)?;
            } else if has("--sweep") {
                run_sweep(args.turns, &[120, 200, 300, 500, 800, 1200, 2000, 4000])
                    .map_err(to_err)?;
            } else if has("--coverage") {
                run_coverage(&[100, 300, 1000, 3000], args.budget).map_err(to_err)?;
            } else if has("--poison") {
                run_poison(args.budget).map_err(to_err)?;
            } else if has("--baselines") {
                run_baselines(args.turns, args.budget).map_err(to_err)?;
            } else if has("--rebuild") {
                run_rebuild(args.turns, args.budget).map_err(to_err)?;
            } else if has("--tamper") {
                run_tamper(args.budget).map_err(to_err)?;
            } else {
                run_benchmark(args.turns, args.budget, args.window).map_err(to_err)?;
            }
        }
        other => return Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

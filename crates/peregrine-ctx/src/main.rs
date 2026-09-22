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
  explain <node_id>           audit path from a node down to raw spans
  stats                       telemetry report
  verify                      check a .context container: objects, chain, root
  scrub [--repair]            detect bit rot; repair from a verified replica
  checkpoint                  seal the current state as a new generation
  quarantine                  list objects that failed verification
  demo                        worked example through all four ladder levels
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
            for pattern in &args.positional {
                let matches = expand(pattern);
                if matches.is_empty() {
                    eprintln!("no match: {pattern}");
                }
                for path in matches {
                    let result = runtime.ingest_file(&path).map_err(to_err)?;
                    total += result.nodes.len();
                    println!("{}: {}", path.display(), result.summary());
                    for (old, new) in &result.contradictions {
                        println!("  contradiction: {old} superseded by {new}");
                    }
                }
            }
            persist(&runtime, &args.store).map_err(to_err)?;
            println!("\n{total} state nodes; store: {}", args.store.display());
        }
        "ask" => {
            let query = args.positional.first().ok_or("ask needs a query")?;
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            let answer = runtime.ask(query, None);
            println!("{}", answer.text);
            println!(
                "\n-- {}/{} tokens, {} nodes, {} escalation(s), cited {}",
                answer.tokens,
                args.budget,
                answer.context.entries.len(),
                answer.escalations,
                if answer.cited.is_empty() {
                    "nothing".to_string()
                } else {
                    answer.cited.join(", ")
                }
            );
            if args.flags.iter().any(|f| f == "--show-context") {
                println!("\n{}", answer.context.render());
            }
            persist(&runtime, &args.store).map_err(to_err)?;
        }
        "plan" => {
            let query = args.positional.first().ok_or("plan needs a query")?;
            let mut runtime = open(&args.store, args.budget).map_err(to_err)?;
            let context = runtime.plan(query, None);
            println!("{}", context.render());
            if args.flags.iter().any(|f| f == "--explain") {
                println!("\n{}", runtime.explain_plan(&context));
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

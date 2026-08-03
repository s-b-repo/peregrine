//! `peregrine-prune` — router-weighted expert pruning (REAP).
//!
//! Reads a routing trace, ranks each layer's experts by the gate mass they
//! carried, and writes a container with the least-salient fraction removed and
//! the survivors renumbered.
//!
//! Depends on `peregrine-core` alone, like `peregrine-requantize`: a batch job
//! that may run for hours has no business linking io_uring, the scheduler, or a
//! GPU backend.

use std::path::{Path, PathBuf};

use peregrine_core::Error;
use peregrine_tools::prune::{default_outdir, load_trace, plan_keep, prune, PruneReport, SAFE_FRAC};

const USAGE: &str = "\
peregrine-prune <model-dir> [out-dir] --trace <routes.json> [options]

Router-weighted expert pruning. Ranks each layer's experts by the gate mass
they carried over the trace and drops the least salient.

  --trace <path>     routing trace to rank from (required)
  --frac <f>         fraction of each layer's experts to drop (default 0.25)
  --keep-min <n>     floor on surviving experts per layer (default: top-k)
  --shard-gb <f>     output shard size in GiB (default 4)
  --dry-run          report the plan and the sizes; write nothing
  --force            proceed past the safety warning above 25%

WHAT THIS DOES AND DOES NOT BUY

  Pruning does NOT reduce bytes per token. Top-k is unchanged, so the same k
  experts are read per position whatever the pool size. What shrinks is the
  working set: fewer distinct experts to hold, cache, prefetch and lay out.

  Use 25%, not 50%. GLM-4.5-Air lost 11.2% on coding and 25.8% on
  multiple-choice at 50%, and retention does not improve with model size.

  The calibration trace dominates the result. Generic web text collapsed code
  performance in the published runs. Trace the workload you actually serve.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-prune: {e}");
        std::process::exit(1);
    }
}

/// Parsed arguments. Split out so the parse is unit-testable without a model.
struct Args {
    indir: PathBuf,
    outdir: Option<PathBuf>,
    trace: PathBuf,
    frac: f64,
    keep_min: Option<usize>,
    shard_bytes: u64,
    dry_run: bool,
    force: bool,
}

fn parse(argv: &[String]) -> Result<Args, Error> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut trace: Option<PathBuf> = None;
    let (mut frac, mut keep_min, mut dry_run, mut force) = (0.25f64, None, false, false);
    let mut shard_gb = 4.0f64;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--trace" => trace = it.next().map(PathBuf::from),
            "--frac" => frac = next_f64(&mut it, "--frac")?,
            "--keep-min" => keep_min = Some(next_f64(&mut it, "--keep-min")? as usize),
            "--shard-gb" => shard_gb = next_f64(&mut it, "--shard-gb")?,
            "--dry-run" => dry_run = true,
            "--force" => force = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(Error::Format(format!("unknown flag {s}\n\n{USAGE}"))),
            s => positional.push(PathBuf::from(s)),
        }
    }
    let mut pos = positional.into_iter();
    let Some(indir) = pos.next() else {
        return Err(Error::Format(format!("a model directory is required\n\n{USAGE}")));
    };
    let Some(trace) = trace else {
        return Err(Error::Format(format!("--trace is required: pruning on no evidence is refused\n\n{USAGE}")));
    };
    if !(0.0..1.0).contains(&frac) {
        return Err(Error::Format(format!("--frac {frac} is outside [0, 1) — 1.0 would remove every expert")));
    }
    Ok(Args {
        indir,
        outdir: pos.next(),
        trace,
        frac,
        keep_min,
        shard_bytes: (shard_gb.max(0.001) * (1u64 << 30) as f64) as u64,
        dry_run,
        force,
    })
}

fn next_f64(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<f64, Error> {
    let Some(v) = it.next() else {
        return Err(Error::Format(format!("{flag} needs a value")));
    };
    v.trim().parse::<f64>().map_err(|e| Error::Format(format!("{flag}: {e}")))
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        print!("{USAGE}");
        return Ok(());
    }
    let args = parse(&argv)?;
    let cfg = peregrine_core::Cfg::load(&args.indir)?;

    // Above the published safe fraction this stops being a residency trade and
    // starts being a quality decision, so it needs an explicit act.
    if args.frac > SAFE_FRAC && !args.force {
        return Err(Error::Format(format!(
            "--frac {:.2} is above the {:.0}% the evidence supports.\n\
             GLM-4.5-Air lost 11.2% on coding and 25.8% on multiple-choice at 50%, and retention\n\
             does not improve with model size. Re-run with --force if that is the trade you want.",
            args.frac,
            SAFE_FRAC * 100.0
        )));
    }

    let sal = load_trace(&args.trace)?;
    let n_layers = cfg.n_layers as usize;
    let n_experts = cfg.n_experts as usize;
    let keep_min = args.keep_min.unwrap_or(cfg.topk.max(1) as usize);
    // `n_layers + 1`: the MTP head is a sparse layer with its own router, and
    // `config.json` carries one `n_routed_experts` for all of them, so the plan
    // has to cover it or the container comes out inconsistent with its config.
    let plan = plan_keep(&sal, n_layers + 1, n_experts, args.frac, keep_min);

    eprintln!(
        "peregrine-prune: {} positions traced; {} of {} layers ranked on aggregate saliency; \
         keeping {} of {} experts per layer",
        sal.positions,
        plan.layers_by_aggregate,
        plan.keep.len(),
        plan.n_experts_out,
        n_experts,
    );

    if args.dry_run {
        let rep = PruneReport {
            layers: n_layers,
            experts_in: n_experts,
            experts_kept: plan.n_experts_out,
            layers_without_evidence: plan.layers_by_aggregate,
            frequency_only: false,
            ..PruneReport::default()
        };
        println!("{}", rep.summary());
        println!("\n(dry run — nothing written)");
        return Ok(());
    }

    let outdir = args.outdir.unwrap_or_else(|| default_outdir(&args.indir, args.frac));
    refuse_overwrite(&args.indir, &outdir)?;
    eprintln!("peregrine-prune: writing {}", outdir.display());
    let rep = prune(&args.indir, &outdir, &plan, args.shard_bytes)?;
    println!("{}", rep.summary());
    println!(
        "\nNext: measure it. `Model::prediction_flip_rate` against the source container on the\n\
         workload you traced — a pruned checkpoint that was never compared is a guess."
    );
    Ok(())
}

/// A conversion that wrote into its own source would destroy the thing it was
/// reading, hours in and irrecoverably.
fn refuse_overwrite(indir: &Path, outdir: &Path) -> Result<(), Error> {
    let same = std::fs::canonicalize(indir).ok().zip(std::fs::canonicalize(outdir).ok()).is_some_and(|(a, b)| a == b);
    if same {
        return Err(Error::Format("the output directory is the source directory — refusing to overwrite".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_trace_is_required_because_pruning_on_no_evidence_is_refused() {
        let err = match parse(&argv(&["/m"])) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("--trace is required"), "got: {err}");
    }

    #[test]
    fn frac_is_bounded_below_one() {
        // 1.0 removes every expert, which is not a smaller model but a broken
        // one — the router would have nothing to select.
        for bad in ["1.0", "1.5", "-0.1"] {
            let err = match parse(&argv(&["/m", "--trace", "/t.json", "--frac", bad])) {
                Err(e) => e.to_string(),
                Ok(_) => String::new(),
            };
            assert!(err.contains("outside [0, 1)"), "--frac {bad} should be rejected, got: {err}");
        }
        let ok = parse(&argv(&["/m", "--trace", "/t.json", "--frac", "0.25"]));
        assert!(ok.is_ok());
    }

    #[test]
    fn defaults_are_the_conservative_ones() -> Result<(), Error> {
        let a = parse(&argv(&["/m", "--trace", "/t.json"]))?;
        assert_eq!(a.frac, SAFE_FRAC, "the default fraction is the one the evidence supports");
        assert!(!a.force, "and it does not silently carry --force");
        assert!(!a.dry_run);
        assert_eq!(a.keep_min, None, "the floor defaults to the model's top-k, not a guess");
        Ok(())
    }

    #[test]
    fn an_unknown_flag_is_reported_rather_than_ignored() {
        // A typo'd flag that parses as a positional would silently become the
        // output directory, which is how a conversion writes somewhere nobody
        // meant it to.
        let err = match parse(&argv(&["/m", "--trace", "/t.json", "--fraq", "0.5"])) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("unknown flag --fraq"), "got: {err}");
    }

    #[test]
    fn the_output_directory_cannot_be_the_source() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_prune_same_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let err = match refuse_overwrite(&dir, &dir) {
            Err(e) => e.to_string(),
            Ok(()) => String::new(),
        };
        assert!(err.contains("refusing to overwrite"), "got: {err}");
        let other = dir.join("out");
        std::fs::create_dir_all(&other)?;
        assert!(refuse_overwrite(&dir, &other).is_ok());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn the_default_outdir_names_the_fraction_it_used() {
        // Two prunes of one model at different fractions must not collide, and
        // the artifact should say which it is without opening it.
        let a = default_outdir(Path::new("/models/glm52_i4"), 0.25);
        let b = default_outdir(Path::new("/models/glm52_i4"), 0.5);
        assert_ne!(a, b);
        assert!(a.ends_with("glm52_i4_pruned25"), "got {}", a.display());
        assert!(b.ends_with("glm52_i4_pruned50"), "got {}", b.display());
    }
}

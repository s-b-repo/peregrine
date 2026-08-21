//! `peregrine-basisfit` — cross-expert factorization, priced as rate–distortion
//! on activations.
//!
//! Fits `W_e = B + Δ_e` across a layer's experts, holds `B` resident, and
//! measures what the engine would actually pay: bytes streamed per token, and
//! the activation-weighted distortion the residual's quantization introduces.
//! It writes nothing and converts nothing — the deliverable is the verdict,
//! because whether a basis is worth building is exactly what is unknown.

// Each binary target is its own crate root, so the panic-lint denials the
// library carries have to be repeated here.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use peregrine_core::Error;
use peregrine_tools::basis::{measure, measure_control, FitConfig, Grouping, Scope};
use peregrine_tools::requant::{load_calib, Target};

const USAGE: &str = "\
peregrine-basisfit <model-dir> [--calib <sidecar>] [options]

Fits a shared basis across each layer's routed experts and prices the result the
way the engine would feel it: bytes moved per token against the distortion that
buys. Measures only — no container is written.

  --calib <path>     calibration sidecar from `peregrine calib-capture`. Without
                     it every projection is scored UNWEIGHTED (plain Frobenius),
                     which is the measurement this tool exists to argue against;
                     the report says so rather than letting it read as calibrated
  --rank <n>         basis vectors beyond the group mean (default 0, the
                     W = B + delta form)
  --groups <n>       groups per layer sharing one basis (default 1)
  --residual <fmt>   precision the residual is stored at (default int2-g64)
  --baseline <fmt>   precision the no-basis comparison uses (defaults to
                     whatever --residual is, so both arms stream the SAME bytes
                     and the only variable is the basis). Setting it to the
                     container's own precision makes it lossless by
                     construction; the report refuses that rather than scoring it
  --layers <n>       sparse layers to measure, from the first (default 2, 0=all)
  --experts <n>      experts per layer to include (default 0 = all)
  --control [seed]   also run the shuffled-grouping control at equal rank.
                     Runs several random partitions, not one: the learned arm is
                     fit and scored on the same experts, so a positive margin is
                     the NULL expectation and needs a spread to be read against
  --shuffled <seed>  run ONLY the shuffled arm

  formats: int8 int4 int4-g<N> int3-g64 int2 int2-g64

WHY THIS MEASURES DISTORTION AND NOT RECONSTRUCTION ERROR

  ||W - (B + delta)||_F is minimized by construction when B is the group mean
  and delta is stored exactly. It cannot fail, and its success says nothing
  about the container: a basis can lower Frobenius error while making the
  residual high-entropy and HOSTILE to the int4/block quantization it then has
  to survive. Weight-space error is structurally blind to that, because it
  never quantizes anything.

  So the residual is actually quantized, the error is weighted by calibrated
  per-channel activation magnitude, and every basis arm is reported beside a
  no-basis baseline at the same precision. If the residual quantizes no better
  than the weight it replaced, the basis moved the entropy around and the
  verdict says so.

WHY --control MATTERS

  If a learned grouping does not beat a RANDOM one at equal rank, the basis is
  capturing layer-wide structure that one per-layer mean would also capture --
  not cross-expert redundancy. The byte saving would be real and the
  explanation wrong. Run it before believing a win.

WHAT THIS DOES NOT MEASURE

  Distortion is not flip rate. Gate the winning arm with `peregrine flip-rate`
  before trusting any of it.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-basisfit: {e}");
        std::process::exit(1);
    }
}

struct Args {
    indir: PathBuf,
    calib: Option<PathBuf>,
    fit: FitConfig,
    scope: Scope,
    control: Option<u64>,
}

fn fmt(s: &str) -> Result<Target, Error> {
    Target::parse(s).ok_or_else(|| Error::Format(format!("unknown format {s}\n\n{USAGE}")))
}

fn num(v: Option<&String>, what: &str) -> Result<usize, Error> {
    v.and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| Error::Format(format!("{what} needs a number\n\n{USAGE}")))
}

fn parse(argv: &[String]) -> Result<Args, Error> {
    let mut indir: Option<PathBuf> = None;
    let mut calib = None;
    let mut fit = FitConfig::default();
    let mut scope = Scope::default();
    let mut control = None;
    // Whether `--baseline` was given. Unset, it mirrors `--residual`, so both
    // arms stream identical bytes and the comparison isolates the basis
    // instead of confounding it with a precision change.
    let mut baseline_set = false;
    let mut it = argv.iter().peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--calib" => calib = it.next().map(PathBuf::from),
            "--rank" => fit.rank = num(it.next(), "--rank")?,
            "--groups" => fit.groups = num(it.next(), "--groups")?.max(1),
            "--residual" => {
                let s = it.next().ok_or_else(|| Error::Format("--residual needs a format".into()))?;
                fit.residual = fmt(s)?;
            }
            "--baseline" => {
                let s = it.next().ok_or_else(|| Error::Format("--baseline needs a format".into()))?;
                fit.baseline = fmt(s)?;
                baseline_set = true;
            }
            "--layers" => scope.layers = num(it.next(), "--layers")?,
            "--experts" => scope.experts = num(it.next(), "--experts")?,
            "--control" => {
                // Optional seed: only consume the next token if it is one.
                let seed = match it.peek().and_then(|s| s.parse::<u64>().ok()) {
                    Some(v) => {
                        it.next();
                        v
                    }
                    None => 0xC0FFEE,
                };
                control = Some(seed);
            }
            "--shuffled" => {
                let seed = num(it.next(), "--shuffled")? as u64;
                fit.grouping = Grouping::Shuffled(seed);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(Error::Format(format!("unknown flag {s}\n\n{USAGE}"))),
            s => indir = Some(PathBuf::from(s)),
        }
    }
    let indir = indir.ok_or_else(|| Error::Format(format!("no model dir\n\n{USAGE}")))?;
    if !baseline_set {
        fit.baseline = fit.residual;
    }
    Ok(Args { indir, calib, fit, scope, control })
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        print!("{USAGE}");
        return Ok(());
    }
    let args = parse(&argv)?;
    let calib = match &args.calib {
        Some(p) => Some(load_calib(p)?),
        None => None,
    };
    // An uncalibrated run is the exact experiment this tool argues against, so
    // it is announced up front rather than only in the trailing caveats.
    if calib.is_none() {
        eprintln!(
            "peregrine-basisfit: no --calib sidecar, so every projection is scored UNWEIGHTED \
             (plain Frobenius). That is weight-space reconstruction error wearing an activations \
             label — produce a sidecar with `peregrine calib-capture` before quoting these numbers."
        );
    }
    println!(
        "rank={} groups={} residual={} baseline={} grouping={} layers={} experts={}",
        args.fit.rank,
        args.fit.groups,
        args.fit.residual.label(),
        args.fit.baseline.label(),
        args.fit.grouping.label(),
        if args.scope.layers == 0 { "all".into() } else { args.scope.layers.to_string() },
        if args.scope.experts == 0 { "all".into() } else { args.scope.experts.to_string() },
    );

    match args.control {
        Some(seed) => {
            let ctl = measure_control(&args.indir, calib.as_ref(), &args.fit, &args.scope, seed)?;
            println!("\n--- learned grouping ---\n{}", ctl.learned.verdict());
            // Only the first draw is printed in full: the remaining draws exist
            // to give the control a spread, and printing four near-identical
            // verdicts would bury the one comparison that matters.
            if let Some(first) = ctl.shuffled.first() {
                println!("\n--- shuffled grouping (draw 1 of {}) ---\n{}", ctl.shuffled.len(), first.verdict());
            }
            println!("\n--- control (seed {seed:#x}) ---\n{}", ctl.verdict());
            for c in ctl.learned.caveats() {
                println!("\nCAVEAT: {c}");
            }
        }
        None => {
            let rep = measure(&args.indir, calib.as_ref(), &args.fit, &args.scope)?;
            println!("\n{}", rep.verdict());
            for c in rep.caveats() {
                println!("\nCAVEAT: {c}");
            }
        }
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
    fn control_takes_an_optional_seed_without_eating_the_model_dir() -> Result<(), Error> {
        // `--control` with no seed followed by a positional is the shape a user
        // types first; consuming the path as a seed would fail far downstream
        // with a confusing "no model dir".
        let a = parse(&argv(&["--control", "/tmp/m"]))?;
        assert_eq!(a.indir, PathBuf::from("/tmp/m"));
        assert_eq!(a.control, Some(0xC0FFEE), "an absent seed must take the default");
        let b = parse(&argv(&["/tmp/m", "--control", "42"]))?;
        assert_eq!(b.control, Some(42), "an explicit seed must be read");
        assert_eq!(b.indir, PathBuf::from("/tmp/m"));
        Ok(())
    }

    #[test]
    fn an_unknown_format_is_refused_rather_than_defaulted() {
        // Silently falling back to int4 would make a sweep report the wrong
        // precision's numbers under the label the user asked for.
        assert!(parse(&argv(&["/tmp/m", "--residual", "int5"])).is_err());
        assert!(parse(&argv(&["/tmp/m", "--residual", "int2-g64"])).is_ok());
    }

    #[test]
    fn groups_never_falls_to_zero() -> Result<(), Error> {
        // A zero group count would partition into nothing and report an empty
        // sweep as a clean run.
        let a = parse(&argv(&["/tmp/m", "--groups", "0"]))?;
        assert_eq!(a.fit.groups, 1);
        Ok(())
    }

    #[test]
    fn shuffled_only_mode_is_reachable_without_the_control() -> Result<(), Error> {
        let a = parse(&argv(&["/tmp/m", "--shuffled", "7"]))?;
        assert_eq!(a.fit.grouping, Grouping::Shuffled(7));
        assert!(a.control.is_none(), "--shuffled alone must not imply the paired control");
        Ok(())
    }
}

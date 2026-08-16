//! `peregrine-skipbound` — the offline prototype for pre-read expert skipping.
//!
//! Computes each expert's contribution bound from the container, then measures
//! against a routing trace how often that bound is tight enough to license a
//! skip. It writes a sidecar and a verdict; it does **not** change the read
//! path, because whether that is worth doing is exactly what this measures.

// The last first-party crate to adopt the panic-lint denials the other nine
// already carry. It qualified all along — zero unwrap/expect/panic in this
// crate's sources — so this is a ratchet, not a cleanup. Each binary target
// is its own crate root, so the attribute has to be repeated per target.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use peregrine_core::Error;
use peregrine_tools::skipbound::{compute_bounds, load_bounds, load_frames, measure};

const USAGE: &str = "\
peregrine-skipbound <model-dir> [--trace <routes.json>] [--out <bounds.json>]
peregrine-skipbound --bounds <bounds.json> --trace <routes.json>

Offline prototype for pre-read expert skipping. Computes a per-expert bound on
how much that expert can contribute, and — given a trace — measures how often
the bound is tight enough that the expert's ~18.9 MB read is provably skippable.

  --trace <path>   routing trace to measure tightness against
  --out <path>     write the bound sidecar here (default <model-dir>/expert_bounds.json)
  --no-write       measure only; write nothing
  --bounds <path>  measure against an existing sidecar instead of computing.
                   Skips the container pass (hundreds of GB of reads), so a
                   trace analysis costs one JSON read; no model dir is needed
                   and nothing is written

THE BOUND

  ||contribution|| <= gate * C_e * ||x||^2,  C_e = ||W_down||_F ||W_gate||_F ||W_up||_F

  ||x||^2 is common to every expert at a position, so the ranking a runtime
  skip would use needs no hidden state — only the gate weights in the trace.

WHY THIS IS A PROTOTYPE

  An upper bound is one-sided: a small bound proves an expert cannot matter, a
  large one proves nothing. If few reads clear the threshold, a runtime check
  costs per-token work and eliminates almost nothing. Read the verdict before
  writing any read-path code.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-skipbound: {e}");
        std::process::exit(1);
    }
}

struct Args {
    /// `None` only in `--bounds` mode, where no container is touched.
    indir: Option<PathBuf>,
    trace: Option<PathBuf>,
    out: Option<PathBuf>,
    no_write: bool,
    bounds: Option<PathBuf>,
}

fn parse(argv: &[String]) -> Result<Args, Error> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let (mut trace, mut out, mut no_write, mut bounds) = (None, None, false, None);
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--trace" => trace = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--no-write" => no_write = true,
            "--bounds" => bounds = it.next().map(PathBuf::from),
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(Error::Format(format!("unknown flag {s}\n\n{USAGE}"))),
            s => positional.push(PathBuf::from(s)),
        }
    }
    let indir = positional.into_iter().next();
    if bounds.is_none() && indir.is_none() {
        return Err(Error::Format(format!("a model directory is required\n\n{USAGE}")));
    }
    if bounds.is_some() && out.is_some() {
        // Rewriting a sidecar from itself is a lossy copy pretending to be a
        // computation; refuse rather than let a stale file masquerade as fresh.
        return Err(Error::Format(format!("--bounds and --out are mutually exclusive\n\n{USAGE}")));
    }
    Ok(Args { indir, trace, out, no_write, bounds })
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        print!("{USAGE}");
        return Ok(());
    }
    let args = parse(&argv)?;

    let bounds = match (&args.bounds, &args.indir) {
        (Some(sidecar), _) => {
            let b = load_bounds(sidecar)?;
            eprintln!("peregrine-skipbound: loaded {} expert bounds from {}", b.c.len(), sidecar.display());
            b
        }
        (None, Some(indir)) => {
            eprintln!("peregrine-skipbound: computing bounds (one pass over every routed expert)");
            let b = compute_bounds(indir)?;
            eprintln!("peregrine-skipbound: {} experts bounded", b.c.len());
            if !args.no_write {
                let out = args.out.clone().unwrap_or_else(|| indir.join("expert_bounds.json"));
                let bytes = serde_json::to_vec_pretty(&b.to_json())
                    .map_err(|e| Error::Format(format!("serialize bounds: {e}")))?;
                peregrine_core::durable::write_atomic(&out, &bytes)?;
                eprintln!("peregrine-skipbound: wrote {}", out.display());
            }
            b
        }
        // parse() guarantees one of the two is present.
        (None, None) => return Err(Error::Format(format!("a model directory is required\n\n{USAGE}"))),
    };

    match args.trace {
        Some(t) => {
            let frames = load_frames(&t)?;
            println!("{}", measure(&bounds, &frames).verdict());
        }
        None => println!(
            "No --trace given, so tightness was not measured — and tightness is the whole question.\n\
             The sidecar alone licenses nothing: re-run with a trace of the workload you serve."
        ),
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
    fn a_model_directory_is_required() {
        let err = match parse(&argv(&["--trace", "/t.json"])) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("a model directory is required"), "got: {err}");
    }

    #[test]
    fn a_typod_flag_is_reported_rather_than_taken_as_the_model_path() {
        let err = match parse(&argv(&["/m", "--trase", "/t.json"])) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("unknown flag --trase"), "got: {err}");
    }

    #[test]
    fn bounds_mode_needs_no_model_dir_and_refuses_out() -> Result<(), Error> {
        // The whole point of --bounds is not touching the container; requiring
        // a model dir anyway would defeat it.
        let a = parse(&argv(&["--bounds", "/b.json", "--trace", "/t.json"]))?;
        assert!(a.indir.is_none());
        assert_eq!(a.bounds.as_deref(), Some(std::path::Path::new("/b.json")));
        // Rewriting a sidecar from itself must be refused, not silently done.
        let err = match parse(&argv(&["--bounds", "/b.json", "--out", "/b2.json"])) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("mutually exclusive"), "got: {err}");
        Ok(())
    }

    #[test]
    fn the_trace_is_optional_but_its_absence_is_the_default() -> Result<(), Error> {
        // Computing bounds without measuring them is legitimate (you may want
        // the sidecar), but it answers nothing — which the run says out loud.
        let a = parse(&argv(&["/m"]))?;
        assert!(a.trace.is_none());
        assert!(!a.no_write, "the sidecar is written by default");
        Ok(())
    }
}

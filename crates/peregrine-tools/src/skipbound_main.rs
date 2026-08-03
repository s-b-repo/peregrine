//! `peregrine-skipbound` — the offline prototype for pre-read expert skipping.
//!
//! Computes each expert's contribution bound from the container, then measures
//! against a routing trace how often that bound is tight enough to license a
//! skip. It writes a sidecar and a verdict; it does **not** change the read
//! path, because whether that is worth doing is exactly what this measures.

use std::path::PathBuf;

use peregrine_core::Error;
use peregrine_tools::skipbound::{compute_bounds, load_frames, measure};

const USAGE: &str = "\
peregrine-skipbound <model-dir> [--trace <routes.json>] [--out <bounds.json>]

Offline prototype for pre-read expert skipping. Computes a per-expert bound on
how much that expert can contribute, and — given a trace — measures how often
the bound is tight enough that the expert's ~18.9 MB read is provably skippable.

  --trace <path>   routing trace to measure tightness against
  --out <path>     write the bound sidecar here (default <model-dir>/expert_bounds.json)
  --no-write       measure only; write nothing

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
    indir: PathBuf,
    trace: Option<PathBuf>,
    out: Option<PathBuf>,
    no_write: bool,
}

fn parse(argv: &[String]) -> Result<Args, Error> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let (mut trace, mut out, mut no_write) = (None, None, false);
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--trace" => trace = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--no-write" => no_write = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(Error::Format(format!("unknown flag {s}\n\n{USAGE}"))),
            s => positional.push(PathBuf::from(s)),
        }
    }
    let Some(indir) = positional.into_iter().next() else {
        return Err(Error::Format(format!("a model directory is required\n\n{USAGE}")));
    };
    Ok(Args { indir, trace, out, no_write })
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        print!("{USAGE}");
        return Ok(());
    }
    let args = parse(&argv)?;

    eprintln!("peregrine-skipbound: computing bounds (one pass over every routed expert)");
    let bounds = compute_bounds(&args.indir)?;
    eprintln!("peregrine-skipbound: {} experts bounded", bounds.c.len());

    if !args.no_write {
        let out = args.out.unwrap_or_else(|| args.indir.join("expert_bounds.json"));
        let bytes = serde_json::to_vec_pretty(&bounds.to_json())
            .map_err(|e| Error::Format(format!("serialize bounds: {e}")))?;
        peregrine_core::durable::write_atomic(&out, &bytes)?;
        eprintln!("peregrine-skipbound: wrote {}", out.display());
    }

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
    fn the_trace_is_optional_but_its_absence_is_the_default() -> Result<(), Error> {
        // Computing bounds without measuring them is legitimate (you may want
        // the sidecar), but it answers nothing — which the run says out loud.
        let a = parse(&argv(&["/m"]))?;
        assert!(a.trace.is_none());
        assert!(!a.no_write, "the sidecar is written by default");
        Ok(())
    }
}

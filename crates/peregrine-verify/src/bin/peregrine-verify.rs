//! `peregrine-verify` — the numerical certificate gate.
//!
//! ```text
//! peregrine-verify verify              # every certificate; exit 1 on any failure
//! peregrine-verify verify --group kernels
//! peregrine-verify list                # the theorems, without running them
//! peregrine-verify preflight           # the token-exactness subset
//! ```
//!
//! Exit codes: `0` every certificate passed, `1` at least one failed, `2` the
//! arguments or the command were wrong. An unknown group name is `2`, never `0`
//! — a gate that passes because it matched nothing is worse than no gate.

use std::process::ExitCode;

use peregrine_verify::{run_all, run_group, GROUPS};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("usage: peregrine-verify <verify|list|preflight> [--group <name>]");
        return ExitCode::from(2);
    };
    let mut group: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--group needs a group name; known groups: {}", GROUPS.join(", "));
                    return ExitCode::from(2);
                };
                group = Some(value.clone());
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    match command.as_str() {
        "verify" => {
            let report = match &group {
                Some(name) => run_group(name),
                None => run_all(),
            };
            if report.is_empty() {
                // Either an unknown group, or a group whose certificates were
                // all filtered away. Both are a caller error, not a pass.
                eprintln!("no certificates in group '{}'; known groups: {}", group.unwrap_or_default(), GROUPS.join(", "));
                return ExitCode::from(2);
            }
            print!("{}", report.render());
            ExitCode::from(if report.passed() { 0 } else { 1 })
        }
        "preflight" => {
            let report = peregrine_verify::preflight();
            print!("{}", report.render());
            ExitCode::from(if report.passed() { 0 } else { 1 })
        }
        "list" => {
            let report = match &group {
                Some(name) => run_group(name),
                None => run_all(),
            };
            if report.is_empty() {
                eprintln!("no certificates in group '{}'; known groups: {}", group.unwrap_or_default(), GROUPS.join(", "));
                return ExitCode::from(2);
            }
            for c in &report.certificates {
                println!("{}/{}: {}", c.group, c.name, c.theorem);
            }
            ExitCode::from(0)
        }
        other => {
            eprintln!("unknown command: {other}; expected verify, list or preflight");
            ExitCode::from(2)
        }
    }
}
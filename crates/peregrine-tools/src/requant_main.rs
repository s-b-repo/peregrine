//! `peregrine-requantize` — rewrite a container at a different on-disk precision.
//!
//! Separate from `peregrine build` on purpose: that name already means "write a
//! tiny synthetic model", and a multi-hour batch job has no business living in
//! the inference binary's dependency surface. This links `peregrine-core` only.
//!
//! ```text
//! peregrine-requantize <indir> <outdir> [--target int2] [--include .mlp.experts.]
//!                                       [--shard-bytes N] [--dry-run]
//! ```

// The last first-party crate to adopt the panic-lint denials the other nine
// already carry. It qualified all along — zero unwrap/expect/panic in this
// crate's sources — so this is a ratchet, not a cleanup. Each binary target
// is its own crate root, so the attribute has to be repeated per target.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use peregrine_tools::requant::{load_calib, plan_sizes, requantize, DownPolicy, HeatTier, Plan, Target};
use std::path::Path;

fn usage() -> i32 {
    eprintln!(
        "usage: peregrine-requantize <indir> <outdir> [options]\n\
         \n\
         options:\n\
         \x20 --target <scheme>    int8 | int4 | int4-g<N> | int3-g64 | int2-g64 | int2\n\
         \x20                      (default int2 — per-row, effectively ternary; see\n\
         \x20                      todo.md §13 first: every *uniform* sub-int4 rung is a\n\
         \x20                      measured negative on GLM-5.2 — int2-g64 flip_rate\n\
         \x20                      1.000, int3-g64 flip_rate 0.514. The asymmetric\n\
         \x20                      recipe below is the open question)\n\
         \x20 --include <substr>   only tensors containing this      (default .mlp.experts.)\n\
         \x20 --shard-bytes <N>    roll output shards at N bytes     (default 5000000000)\n\
         \x20 --dry-run            report the size plan from headers, write nothing\n\
         \x20 --down <policy>      .down_proj precision: same (follow --target), keep\n\
         \x20                      (byte-identical pass-through), or its own scheme.\n\
         \x20                      The evidenced low-bit recipes quantize gate/up harder\n\
         \x20                      than down — down feeds the residual stream directly\n\
         \x20 --mtp-target <scheme> precision for the MTP head's own expert pool\n\
         \x20                      (layer n_layers). That layer is int8 in a GLM-5.2\n\
         \x20                      container -- 2x a normal expert's bytes -- and is\n\
         \x20                      read once per DRAFT STEP with no batch-union\n\
         \x20                      amortization. Needs no flip gate: a draft is kept\n\
         \x20                      only where it equals the verify pass's own argmax,\n\
         \x20                      so a coarser head moves the acceptance rate and\n\
         \x20                      cannot move a served token. Beats --keep-last-layers\n\
         \x20                      and --down for that layer.\n\
         \x20 --keep-last-layers N pass the last N expert layer indices through\n\
         \x20                      untouched (the MTP head row counts as one) — the\n\
         \x20                      hybrid hedge for the layers closest to the logits\n\
         \x20 --calib <sidecar>    importance-weighted rounding from a calibration\n\
         \x20                      sidecar (mean |x| per hidden channel per layer,\n\
         \x20                      from a teacher-forcing capture pass). int3-g64\n\
         \x20                      only; applies to gate/up (input width = hidden),\n\
         \x20                      down and dense layers stay data-free. Same bytes\n\
         \x20                      on disk — only the rounding objective changes\n\
         \x20 --tier-hot-frac <f>  heat-tiered precision: keep this fraction of each\n\
         \x20                      layer's experts (by routing heat from the source's\n\
         \x20                      route_stats.json) at --tier-hot, rest at --target\n\
         \x20 --tier-hot <scheme>  precision for the hot fraction (default int4)\n\
         \n\
         Requantizing is lossy: it changes token values. Measure the cost with\n\
         Model::prediction_flip_rate against the source container before relying\n\
         on the output."
    );
    2
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<&str> = Vec::new();
    let mut plan = Plan::default();
    let mut dry_run = false;
    let mut tier_frac: Option<f64> = None;
    let mut tier_hot = Target::Int4;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--dry-run" => dry_run = true,
            "--target" => {
                i += 1;
                match args.get(i).and_then(|s| Target::parse(s)) {
                    Some(t) => plan.target = t,
                    None => {
                        eprintln!("peregrine-requantize: unknown --target (int8|int4|int4-g<N>|int2)");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--include" => {
                i += 1;
                match args.get(i) {
                    Some(s) => plan.include = s.clone(),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--shard-bytes" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    Some(n) if n > 0 => plan.shard_bytes = n,
                    _ => {
                        eprintln!("peregrine-requantize: --shard-bytes needs a positive integer");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--down" => {
                i += 1;
                match args.get(i).and_then(|s| DownPolicy::parse(s)) {
                    Some(d) => plan.down = d,
                    None => {
                        eprintln!("peregrine-requantize: --down needs keep | same | a scheme (e.g. int4)");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--mtp-target" => {
                i += 1;
                match args.get(i).and_then(|s| Target::parse(s)) {
                    Some(t) => plan.mtp_target = Some(t),
                    None => {
                        eprintln!("peregrine-requantize: unknown --mtp-target (int8|int4|int4-g<N>|int3-g64|int2-g64|int2)");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--keep-last-layers" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) => plan.keep_last_layers = n,
                    None => {
                        eprintln!("peregrine-requantize: --keep-last-layers needs an integer");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--calib" => {
                i += 1;
                match args.get(i) {
                    Some(p) => match load_calib(Path::new(p)) {
                        Ok(c) => plan.calib = Some(c),
                        Err(e) => {
                            eprintln!("peregrine-requantize: {e}");
                            return std::process::ExitCode::from(2);
                        }
                    },
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--tier-hot-frac" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<f64>().ok()) {
                    Some(f) if (0.0..=1.0).contains(&f) => tier_frac = Some(f),
                    _ => {
                        eprintln!("peregrine-requantize: --tier-hot-frac needs a fraction in [0,1]");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "--tier-hot" => {
                i += 1;
                match args.get(i).and_then(|s| Target::parse(s)) {
                    Some(t) => tier_hot = t,
                    None => {
                        eprintln!("peregrine-requantize: unknown --tier-hot scheme");
                        return std::process::ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => return std::process::ExitCode::from(usage() as u8),
            other => positional.push(other),
        }
        i += 1;
    }
    let (indir, outdir) = match positional.as_slice() {
        [a, b] => (Path::new(*a), Path::new(*b)),
        _ => return std::process::ExitCode::from(usage() as u8),
    };
    if let Some(frac) = tier_frac {
        let cfg = match peregrine_core::config::Cfg::load(indir) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("peregrine-requantize: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let (nl, ne) = (cfg.n_layers as usize, cfg.n_experts as usize);
        let heat = match HeatTier::from_route_stats(indir, nl, ne) {
            Ok(h) => h,
            Err(e) => {
                eprintln!(
                    "peregrine-requantize: heat tiering needs routing data — {e}\n\
                     Produce it by running the model with COLI_ROUTE_STATS_PERSIST=1, or with \
                     `peregrine dump-routes`. Refusing rather than tiering everything cold, \
                     which would look like it worked."
                );
                return std::process::ExitCode::FAILURE;
            }
        };
        let tier = HeatTier { hot_frac: frac, hot: tier_hot, cold: plan.target, heat, n_experts: ne };
        // The number that decides whether this is worth doing at all.
        let ratio = tier.weighted_ratio(cfg.moe_inter as usize, cfg.hidden as usize);
        eprintln!(
            "peregrine-requantize: heat-tiered — hottest {:.0}% at {} / rest at {}; \
             per-token bytes {:.0}% of an all-{} checkpoint",
            frac * 100.0,
            tier_hot.label(),
            plan.target.label(),
            ratio * 100.0,
            plan.target.label(),
        );
        plan.tier = Some(tier);
    }

    if dry_run {
        return match plan_sizes(indir, &plan) {
            Ok(rep) => {
                eprintln!(
                    "peregrine-requantize --dry-run: {} tensors ({} would requantize to {}) | \
                     {:.2} GB -> {:.2} GB ({:.1}% of source), about {} shard(s) at {:.1} GB each\n\
                     both containers must fit at once: plan for {:.2} GB of free space.",
                    rep.tensors_total,
                    rep.tensors_requantized,
                    plan.target.label(),
                    rep.bytes_in as f64 / 1e9,
                    rep.bytes_out as f64 / 1e9,
                    rep.ratio() * 100.0,
                    rep.shards,
                    plan.shard_bytes as f64 / 1e9,
                    rep.bytes_out as f64 / 1e9,
                );
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("peregrine-requantize: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    let mut extras = String::new();
    if plan.down != DownPolicy::Same {
        extras.push_str(&format!(" | down {}", plan.down.label()));
    }
    if plan.keep_last_layers > 0 {
        extras.push_str(&format!(" | keep-last-layers {}", plan.keep_last_layers));
    }
    if let Some(c) = &plan.calib {
        extras.push_str(&format!(" | calib {:016x}", c.fp));
    }
    eprintln!(
        "peregrine-requantize: {} -> {} | target {} | include '{}'{extras}",
        indir.display(),
        outdir.display(),
        plan.target.label(),
        plan.include
    );
    match requantize(indir, outdir, &plan) {
        Ok(rep) => {
            // The calibrated count is part of the report so an inert sidecar
            // (nothing matched its widths) is visible, not silent.
            let calibrated = if plan.calib.is_some() {
                format!(", {} calibrated", rep.tensors_calibrated)
            } else {
                String::new()
            };
            eprintln!(
                "peregrine-requantize: {} tensors ({} requantized{calibrated}) -> {} shards | {:.2} GB -> {:.2} GB ({:.1}% of source)",
                rep.tensors_total,
                rep.tensors_requantized,
                rep.shards,
                rep.bytes_in as f64 / 1e9,
                rep.bytes_out as f64 / 1e9,
                rep.ratio() * 100.0,
            );
            for n in rep.skipped_unsupported.iter().take(8) {
                eprintln!("peregrine-requantize: copied through (no packed form): {n}");
            }
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("peregrine-requantize: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

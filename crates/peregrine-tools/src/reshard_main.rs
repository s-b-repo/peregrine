//! `peregrine-reshard` — re-pack a sharded MoE checkpoint so each sparse
//! layer's routed experts split across bandwidth-proportional device groups,
//! or place complete transformer layers directly on several drives.
//!
//! ```text
//! peregrine-reshard --model <dir> --out <dir> \
//!     --groups <name>:<weight>,<name>:<weight>,... \
//!     [--route-stats <path>] [--dry-run] [--verify]
//! ```

use peregrine_tools::reshard::{parse_groups, plan, plan_layers, verify, write, write_layers, Options, Plan, WriteReport};
use std::path::{Path, PathBuf};

fn usage() -> i32 {
    eprintln!(
        "usage: peregrine-reshard --model <dir> --out <dir> --groups <name>:<w>,... [options]\n\
         \x20      peregrine-reshard --model <dir> --out <dir> --layer-drive <dir> [--layer-drive <dir> ...] [options]\n\
         \n\
         options:\n\
         \x20 --groups <spec>       comma-separated <name>:<weight> storage groups; weights are\n\
         \x20                       relative device bandwidth. Trunk (non-expert) tensors ride\n\
         \x20                       the FIRST group listed.\n\
         \x20 --route-stats <path>  route_stats.json-shaped heat for the greedy packer\n\
         \x20                       (uniform heat when omitted)\n\
         \x20 --layer-drive <dir>   alternate whole layers across these new directories\n\
         \x20                       and write model_paths.json under a new --out directory\n\
         \x20 --dry-run             print per-group file counts / bytes / expected per-token\n\
         \x20                       share, write nothing\n\
         \x20 --verify              after writing, byte-compare EVERY tensor against the source;\n\
         \x20                       with --dry-run, verify an existing --out instead\n\
         \n\
         Output: experts-l<layer>-<group>.safetensors per (layer, group),\n\
         rolling trunk-<firstgroup>-NNNNN.safetensors, and a manifest.json\n\
         recording files, bytes, and the layer -> expert -> group assignment."
    );
    2
}

fn print_plan(p: &Plan) {
    for (g, spec) in p.groups.iter().enumerate() {
        eprintln!(
            "peregrine-reshard: group '{}' (weight {}): {} file(s), {:.3} GB total \
             ({:.3} GB routed experts), expected per-token routed share {:.1}%",
            spec.name,
            spec.weight,
            p.files_in_group(g),
            p.group_bytes[g] as f64 / 1e9,
            p.routed_bytes[g] as f64 / 1e9,
            p.token_share[g] * 100.0,
        );
    }
}

fn print_layer_plan(out: &Path, drives: &[PathBuf], files: &[peregrine_tools::reshard::FilePlan]) {
    for (group, path) in std::iter::once(out).chain(drives.iter().map(PathBuf::as_path)).enumerate() {
        let (count, bytes) = files.iter().filter(|f| f.group == group)
            .fold((0, 0u64), |(count, bytes), f| (count + 1, bytes.saturating_add(f.bytes)));
        eprintln!("peregrine-reshard: {} holds {count} shard(s), {:.3} GB", path.display(), bytes as f64 / 1e9);
    }
}

fn print_written(rep: WriteReport) {
    eprintln!(
        "peregrine-reshard: wrote {} file(s), {:.3} GB, manifest {}",
        rep.files.len(), rep.bytes_written as f64 / 1e9, rep.manifest.display()
    );
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut groups_spec: Option<String> = None;
    let mut layer_drives = Vec::new();
    let mut route_stats: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut do_verify = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--dry-run" => dry_run = true,
            "--verify" => do_verify = true,
            "--model" => {
                i += 1;
                match args.get(i) {
                    Some(s) => model = Some(PathBuf::from(s)),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--out" => {
                i += 1;
                match args.get(i) {
                    Some(s) => out = Some(PathBuf::from(s)),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--groups" => {
                i += 1;
                match args.get(i) {
                    Some(s) => groups_spec = Some(s.clone()),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--route-stats" => {
                i += 1;
                match args.get(i) {
                    Some(s) => route_stats = Some(PathBuf::from(s)),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "--layer-drive" => {
                i += 1;
                match args.get(i) {
                    Some(s) => layer_drives.push(PathBuf::from(s)),
                    None => return std::process::ExitCode::from(usage() as u8),
                }
            }
            "-h" | "--help" => return std::process::ExitCode::from(usage() as u8),
            other => {
                eprintln!("peregrine-reshard: unknown argument '{other}'");
                return std::process::ExitCode::from(usage() as u8);
            }
        }
        i += 1;
    }
    let (Some(model), Some(out)) = (model, out) else {
        return std::process::ExitCode::from(usage() as u8);
    };
    let result = if let Some(groups_spec) = groups_spec.as_deref().filter(|_| layer_drives.is_empty()) {
        let groups = match parse_groups(groups_spec) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("peregrine-reshard: {e}");
                return std::process::ExitCode::from(2);
            }
        };
        let mut opts = Options::new(groups);
        opts.route_stats = route_stats;
        plan(&model, &opts).and_then(|p| {
            print_plan(&p);
            if !dry_run {
                print_written(write(&model, &out, &p)?);
            }
            Ok(())
        })
    } else if !layer_drives.is_empty() && groups_spec.is_none() && route_stats.is_none() {
        plan_layers(&model, layer_drives.len()).and_then(|p| {
            print_layer_plan(&out, &layer_drives, &p);
            if !dry_run {
                print_written(write_layers(&model, &out, &layer_drives, &p)?);
            }
            Ok(())
        })
    } else {
        return std::process::ExitCode::from(usage() as u8);
    };
    if let Err(e) = result {
        eprintln!("peregrine-reshard: {e}");
        return std::process::ExitCode::FAILURE;
    }
    if dry_run {
        eprintln!("peregrine-reshard: --dry-run, nothing written");
    }

    if do_verify {
        match verify(&model, &out) {
            Ok(rep) => {
                for f in &rep.files {
                    if f.mismatches.is_empty() {
                        eprintln!("peregrine-reshard: verify {} OK ({} tensors)", f.file, f.tensors);
                    } else {
                        for m in &f.mismatches {
                            eprintln!("peregrine-reshard: verify {} MISMATCH: {m}", f.file);
                        }
                    }
                }
                for m in &rep.missing {
                    eprintln!("peregrine-reshard: verify MISSING from output: {m}");
                }
                for m in &rep.extra {
                    eprintln!("peregrine-reshard: verify EXTRA in output: {m}");
                }
                if !rep.ok() {
                    eprintln!("peregrine-reshard: verification FAILED");
                    return std::process::ExitCode::FAILURE;
                }
                eprintln!("peregrine-reshard: verification passed (byte-identical)");
            }
            Err(e) => {
                eprintln!("peregrine-reshard: verify: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    std::process::ExitCode::SUCCESS
}

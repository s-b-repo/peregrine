//! `peregrine-layout-reorg`: rearrange experts in a routing-aware order so
//! contiguous batched reads coalesce more of the disk queue. Thin CLI over
//! [`peregrine_tools`] — see the library docs for the methods.

use peregrine_tools::{order_experts, read_routes, write_schedule};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-layout-reorg: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut routes_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut method: String = "cluster".to_string();
    let mut optimize = false;
    let mut apply = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--routes" => {
                i += 1;
                routes_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--method" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    method = v.clone();
                }
            }
            "--optimize" => optimize = true,
            "--apply" => apply = true,
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    let routes = routes_path.ok_or_else(|| "missing --routes <routes.json>".to_string())?;
    let out = out_path.ok_or_else(|| "missing --out <out_dir>".to_string())?;
    let trace = read_routes(&routes)?;
    let mut ordered = order_experts(&trace, &method)?;
    if optimize {
        for (l, row) in ordered.iter_mut().enumerate() {
            let w = peregrine_tools::build_cooccurrence(&trace, l);
            peregrine_tools::two_opt(row, &w);
        }
    }
    write_schedule(&out, &ordered)?;
    if apply {
        // Self-reorganizing rewrite: turn the schedule into physical disk
        // adjacency inside model.safetensors (bit-identical reads after).
        peregrine_tools::apply_layout(&out, &ordered)?;
        eprintln!("applied physical re-layout to {}/model.safetensors", out.display());
    }
    eprintln!(
        "wrote per-layer schedule for {} layers to {}/schedule.json",
        ordered.len(),
        out.display()
    );
    Ok(())
}

fn usage() {
    eprintln!(
        "usage: peregrine-layout-reorg --routes <routes.json> --out <model_dir> \
    [--method cluster|greedy|louvain|spectral|hilbert] [--optimize] [--apply]"
    );
}

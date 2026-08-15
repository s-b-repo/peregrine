//! `peregrine-import-hf` — import a bf16 HF checkpoint (Qwen3 / Qwen3.5-hybrid
//! family) into the peregrine int4/int8 container format. Track C2's container
//! lane; the tensor policy is the REV 2 contract in
//! `bench-data/coordination-2026-08-15.md` and lives in [`peregrine_tools::import`].

// The same panic-lint ratchet every binary target in this crate carries.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use peregrine_tools::import::import_hf;
use std::path::Path;

fn usage() -> i32 {
    eprintln!(
        "usage: peregrine-import-hf <hf-checkpoint-dir> <outdir> [--shard-bytes N]\n\
         \n\
         Imports the text stack of a bf16 HF checkpoint (Qwen3 dense GQA or\n\
         Qwen3.5 hybrid) into the peregrine container format: projection\n\
         matrices to int4 + .qs, embed_tokens to int8 + .qs, norms and the\n\
         gated-DeltaNet scalars (conv1d, A_log, dt_bias) kept float, the\n\
         vision tower skipped, names verbatim. A tensor outside the Track C\n\
         REV 2 contract fails the import by name — extend the contract\n\
         deliberately, never by silent guess.\n\
         \n\
         The output is lossless-except-quantization: gate it with the dense\n\
         teacher-forcing parity check against the HF reference before serving."
    );
    2
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<&str> = Vec::new();
    let mut shard_bytes: u64 = 5_000_000_000;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--shard-bytes" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    Some(n) if n > 0 => shard_bytes = n,
                    _ => {
                        eprintln!("peregrine-import-hf: --shard-bytes needs a positive integer");
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
    eprintln!("peregrine-import-hf: {} -> {} (track-c-rev2 contract)", indir.display(), outdir.display());
    match import_hf(indir, outdir, shard_bytes) {
        Ok(rep) => {
            eprintln!(
                "peregrine-import-hf: {} tensors -> {} int4 + {} int8 + {} float, {} vision skipped | \
                 {:.2} GB -> {:.2} GB | sidecars: {}",
                rep.tensors_total,
                rep.imported_int4,
                rep.imported_int8,
                rep.imported_float,
                rep.skipped_visual,
                rep.bytes_in as f64 / 1e9,
                rep.bytes_out as f64 / 1e9,
                rep.sidecars.join(", "),
            );
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("peregrine-import-hf: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

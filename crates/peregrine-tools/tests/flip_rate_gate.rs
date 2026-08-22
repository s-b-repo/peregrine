//! The quality gate for lossy containers, gated itself.
//!
//! `prediction_flip_rate` is the repo's only non-equality quality check: every
//! other anchor is bit-identity, which a requantized container fails by
//! construction. That makes it the one gate whose *sensitivity* nobody can
//! infer from a passing suite — a flip rate that always reads 0.000 would look
//! exactly like a lossless conversion, and would license an int2 checkpoint on
//! the strength of a broken measurement.
//!
//! So both directions are asserted here: a container against itself must flip
//! nothing, and a genuinely lossy requantization must flip something.

use peregrine_model::{testkit, Model};

fn predictions(dir: &std::path::Path, toks: &[i32]) -> Result<Vec<i32>, peregrine_core::Error> {
    let mut m = Model::load(dir)?;
    m.teacher_forcing(toks)
}

#[test]
fn the_flip_rate_gate_detects_loss_and_only_loss() -> Result<(), peregrine_core::Error> {
    let base = std::env::temp_dir().join(format!("peregrine_flip_{}", std::process::id()));
    let lossy = base.with_extension("int2");
    for d in [&base, &lossy] {
        if d.exists() {
            std::fs::remove_dir_all(d)?;
        }
    }
    testkit::build_tiny_model(&base)?;
    // Long enough that a single lucky agreement cannot carry the assertion.
    let toks: Vec<i32> = (0..64).map(|i| (i * 7 + 3) % 11).collect();

    let want = predictions(&base, &toks)?;

    // Zero direction: identical inputs, identical container. A gate that cannot
    // return 0.0 reports loss that is not there, and every requantization would
    // look equally bad.
    let same = predictions(&base, &toks)?;
    assert_eq!(
        Model::prediction_flip_rate(&want, &same),
        Some(0.0),
        "a container against itself must flip nothing — otherwise the gate is measuring its own noise"
    );

    // Non-zero direction. Per-row int2 is the strongest loss this fixture can
    // express: `amax / 1` makes the `-2` level unreachable, so the format is
    // effectively ternary (see docs/todo.md §13), and the tiny model's expert
    // projections are too narrow for the grouped variants to be legal.
    peregrine_tools::requant::requantize(
        &base,
        &lossy,
        &peregrine_tools::requant::Plan {
            target: peregrine_tools::requant::Target::Int2,
            ..Default::default()
        },
    )?;
    let got = predictions(&lossy, &toks)?;
    let rate = Model::prediction_flip_rate(&want, &got).ok_or_else(|| {
        peregrine_core::Error::Format("same token count in, same prediction count out".into())
    })?;
    assert!(
        rate > 0.0,
        "int4 → per-row int2 dropped ~2 bits per weight and the gate saw no prediction change — \
         a gate that cannot detect this cannot license a container against it"
    );

    for d in [&base, &lossy] {
        std::fs::remove_dir_all(d)?;
    }
    Ok(())
}

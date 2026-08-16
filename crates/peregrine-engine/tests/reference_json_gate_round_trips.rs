//! `flip-rate --reference-json` is the container-vs-dump half of the Track C
//! parity gate: the reference arm (HF bf16, disk-offloaded — too heavy to run
//! in-process here) dumps `{"tokens", "argmax"}`, and the gate compares a
//! peregrine container's teacher-forced argmax against it. These tests pin the
//! plumbing on the hybrid architecture with the container's own predictions as
//! the dump: agreement with yourself must read 0.000 (the harness has no
//! self-flips), a corrupted dump must read the exact planted flip count, and
//! the mode must refuse the argument combinations that would silently measure
//! the wrong thing.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_peregrine"))
}

fn tiny_hybrid(tag: &str) -> Result<PathBuf, peregrine_core::Error> {
    let dir = std::env::temp_dir().join(format!("peregrine_refjson_{tag}_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    peregrine_model::testkit::build_tiny_hybrid_model(&dir, 46)?;
    Ok(dir)
}

/// The container's own teacher-forced argmax, dumped the way the HF runner
/// dumps its reference — the fixture for a self-agreement gate.
fn self_dump(dir: &PathBuf, tokens: &[i32]) -> Result<Vec<i32>, peregrine_core::Error> {
    let mut m = peregrine_model::Model::load(dir)?;
    m.teacher_forcing(tokens)
}

#[test]
fn self_agreement_reads_zero_and_planted_flips_read_exactly() -> Result<(), peregrine_core::Error> {
    let dir = tiny_hybrid("selfzero")?;
    let tokens: Vec<i32> = vec![1, 5, 9, 2, 7, 4, 8, 3];
    let argmax = self_dump(&dir, &tokens)?;
    let dump = dir.join("ref.json");
    std::fs::write(&dump, serde_json::json!({ "tokens": tokens, "argmax": argmax }).to_string())?;
    let out = bin().args(["flip-rate", dir.to_str().unwrap_or("."), "--reference-json"]).arg(&dump).output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("flip_rate   0.000000"),
        "self-agreement must be exact; got:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Corrupt two reference predictions: the gate must count exactly those.
    let mut wrong = self_dump(&dir, &tokens)?;
    wrong[1] = (wrong[1] + 1) % 32;
    wrong[5] = (wrong[5] + 1) % 32;
    // Give the dump top-k rows arranged so exactly one of the two planted
    // flips is a "near-tie" (the container's answer sits in the reference
    // top-k) and the other is a real departure — the containment line must
    // split them 1 of 2.
    let honest = self_dump(&dir, &tokens)?;
    let topk: Vec<Vec<i32>> = (0..tokens.len())
        .map(|i| {
            if i == 1 {
                vec![wrong[1], honest[1]] // flip position, container's answer IS in top-k
            } else {
                vec![wrong.get(i).copied().unwrap_or(0), -1] // position 5's honest answer is NOT
            }
        })
        .collect();
    std::fs::write(
        &dump,
        serde_json::json!({ "tokens": tokens, "argmax": wrong, "topk": topk }).to_string(),
    )?;
    let out = bin().args(["flip-rate", dir.to_str().unwrap_or("."), "--reference-json"]).arg(&dump).output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("flips       2"), "exactly the planted flips must be counted; got:\n{stdout}");
    assert!(
        stdout.contains("flips_in_reference_top2   1 of 2 flips"),
        "the near-tie split must read 1 of 2; got:\n{stdout}"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The paired-analysis dump: the positions it names must be exactly the
/// positions that disagree — that is the whole basis of a McNemar comparison
/// between two arms, and an off-by-one or a truncated list would silently
/// weaken every conclusion drawn from it.
#[test]
fn dump_flips_names_exactly_the_disagreeing_positions() -> Result<(), peregrine_core::Error> {
    let dir = tiny_hybrid("dumpflips")?;
    let tokens: Vec<i32> = vec![1, 5, 9, 2, 7, 4, 8, 3];
    let mut wrong = self_dump(&dir, &tokens)?;
    // Plant disagreements at known positions.
    let planted = [2usize, 6];
    for &i in &planted {
        wrong[i] = (wrong[i] + 1) % 32;
    }
    let dump = dir.join("ref.json");
    std::fs::write(&dump, serde_json::json!({ "tokens": tokens, "argmax": wrong }).to_string())?;
    let flips_path = dir.join("flips.json");
    let out = bin()
        .args(["flip-rate", dir.to_str().unwrap_or("."), "--reference-json"])
        .arg(&dump)
        .arg("--dump-flips")
        .arg(&flips_path)
        .output()?;
    assert!(out.status.success(), "gate must succeed: {}", String::from_utf8_lossy(&out.stderr));
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&flips_path)?)
        .map_err(|e| peregrine_core::Error::Format(format!("flips json: {e}")))?;
    let got: Vec<usize> = doc["flips"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();
    assert_eq!(got, planted.to_vec(), "dumped positions must be exactly the planted disagreements");
    assert_eq!(doc["positions"].as_u64(), Some(tokens.len() as u64));
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn the_mode_refuses_ambiguous_argument_combinations() -> Result<(), peregrine_core::Error> {
    let dir = tiny_hybrid("refuse")?;
    let dump = dir.join("ref.json");
    std::fs::write(&dump, serde_json::json!({ "tokens": [1, 2], "argmax": [3, 4] }).to_string())?;
    // A candidate dir AND a reference dump: two different measurements, refuse.
    let out = bin()
        .args(["flip-rate", dir.to_str().unwrap_or("."), dir.to_str().unwrap_or("."), "--reference-json"])
        .arg(&dump)
        .output()?;
    assert!(!out.status.success(), "candidate dir + reference json must refuse");
    // --text alongside the dump: the dump's ids win silently otherwise — refuse.
    let out = bin()
        .args(["flip-rate", dir.to_str().unwrap_or("."), "--reference-json"])
        .arg(&dump)
        .args(["--text", "/nonexistent"])
        .output()?;
    assert!(!out.status.success(), "--text alongside --reference-json must refuse");
    // Mismatched lengths in the dump itself.
    std::fs::write(&dump, serde_json::json!({ "tokens": [1, 2, 3], "argmax": [3, 4] }).to_string())?;
    let out = bin().args(["flip-rate", dir.to_str().unwrap_or("."), "--reference-json"]).arg(&dump).output()?;
    assert!(!out.status.success(), "length-mismatched dump must refuse");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

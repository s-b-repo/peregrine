//! `flip-rate --candidate-env` exists because the knobs it A/Bs latch in
//! process-global `OnceLock`s (`route_min_share` first among them): exporting
//! the var sets it for both arms, and the gate compares the knob to itself —
//! 0.000, indistinguishable from a lossless candidate. That is the vacuous
//! reading `flip_rate_gate.rs` guards the *measure* against; these tests guard
//! the *harness*: the env demonstrably reaches the candidate arm and only it,
//! a key the parent also holds is refused rather than measured, and the child
//! arm computes exactly what the in-process arm computes when its env is inert.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_peregrine"));
    // The test runner's own environment must not become an accidental arm:
    // a COLI_ROUTE_MIN_SHARE inherited from the shell would either trip the
    // refusal (test 2's territory) or truncate the source arm too.
    c.env_remove("COLI_ROUTE_MIN_SHARE");
    c
}

fn tiny_model(tag: &str) -> Result<PathBuf, peregrine_core::Error> {
    let dir = std::env::temp_dir().join(format!("peregrine_cand_env_{tag}_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    peregrine_model::testkit::build_tiny_model(&dir)?;
    Ok(dir)
}

/// Same container on both sides, a truncation knob on the candidate side only:
/// a nonzero flip rate is only reachable if the env landed in exactly one arm.
/// Both-arms leakage (the OnceLock failure this flag exists to dodge) and a
/// dropped env both read 0.000 — which is why the assertion is `> 0` and not a
/// literal, and why the arm's own echo of its environment is pinned too: the
/// echo is read back from the child's environment, not from the parent's argv.
#[test]
fn the_candidate_env_reaches_the_candidate_arm_and_only_it() -> Result<(), peregrine_core::Error> {
    let dir = tiny_model("reach")?;
    let out = bin()
        .args([
            "flip-rate",
            &dir.display().to_string(),
            &dir.display().to_string(),
            "--tokens",
            "48",
            "--candidate-env",
            "COLI_ROUTE_MIN_SHARE=0.999",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "flip-rate failed:\n{stderr}");
    assert!(
        stderr.contains("flip-arm: env: COLI_ROUTE_MIN_SHARE=0.999"),
        "the candidate arm did not echo the knob from its own environment:\n{stderr}"
    );
    let rate: f64 = stdout
        .lines()
        .find_map(|l| l.strip_prefix("flip_rate"))
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| peregrine_core::Error::Format(format!("no flip_rate line in:\n{stdout}")))?;
    assert!(
        rate > 0.0,
        "dropping every sub-99.9%-share expert on one arm flipped nothing — the knob \
         either reached both arms or neither (rate {rate}, stdout:\n{stdout})"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A key the parent also holds means the source arm would run with it too.
/// Refused, not warned: a warning scrolls past a flawless-looking 0.000.
#[test]
fn a_key_the_parent_also_holds_is_refused() -> Result<(), peregrine_core::Error> {
    let dir = tiny_model("refuse")?;
    let out = bin()
        .env("COLI_ROUTE_MIN_SHARE", "0.5")
        .args([
            "flip-rate",
            &dir.display().to_string(),
            &dir.display().to_string(),
            "--tokens",
            "8",
            "--candidate-env",
            "COLI_ROUTE_MIN_SHARE=0.5",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an ambiguous env split must not produce a number");
    assert!(
        stderr.contains("set in this environment"),
        "the refusal should name the hazard:\n{stderr}"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// With an env var nothing reads, the child arm must reproduce the in-process
/// arm bit for bit — 0.000, same container both sides. This is the harness
/// half of `flip_rate_gate.rs`'s zero direction: a child that tokenized,
/// forwarded, or framed its output differently would report loss that is not
/// there, and every candidate measured through it would look equally bad.
#[test]
fn an_inert_candidate_env_leaves_the_arms_identical() -> Result<(), peregrine_core::Error> {
    let dir = tiny_model("inert")?;
    let out = bin()
        .args([
            "flip-rate",
            &dir.display().to_string(),
            &dir.display().to_string(),
            "--tokens",
            "48",
            "--candidate-env",
            "PEREGRINE_TEST_INERT=1",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "flip-rate failed:\n{stderr}");
    assert!(
        stdout.contains("flip_rate   0.000000"),
        "the child arm disagreed with the in-process arm on identical input:\n{stdout}"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

//! Self-reorganizing rewrite equality gate: physically re-laying out the
//! checkpoint must be invisible to the model — identical teacher-forcing
//! predictions before and after `apply_layout`.

use peregrine_model::{testkit, Model};

#[test]
fn apply_layout_is_bit_identical() -> Result<(), peregrine_core::Error> {
    let dir = std::env::temp_dir().join(format!("peregrine_apply_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    testkit::build_tiny_model(&dir)?;
    let toks = [3i32, 7, 1, 4, 9, 2];

    // Reference predictions on the original layout.
    let want = {
        let mut m = Model::load(&dir)?;
        m.teacher_forcing(&toks)?
    };

    // Physically rewrite with a deliberately reversed expert order per layer.
    let n_experts = 4i32;
    let n_layers = 2usize;
    let ordered: Vec<Vec<i32>> = (0..n_layers).map(|_| (0..n_experts).rev().collect()).collect();
    peregrine_tools::apply_layout(&dir, &ordered).map_err(peregrine_core::Error::Format)?;

    // Same predictions from the re-laid checkpoint.
    let got = {
        let mut m = Model::load(&dir)?;
        m.teacher_forcing(&toks)?
    };
    assert_eq!(got, want, "physical re-layout must not change any prediction");

    // The rewrite really moved the expert tensors: layer-1 expert 3 now
    // precedes expert 0 in the file (offset order matches the schedule).
    let st = peregrine_core::SafeTensors::open(&dir)?;
    let off = |name: &str| st.find(name).map(|t| t.off).unwrap_or(u64::MAX);
    let e3 = off("model.layers.1.mlp.experts.3.gate_proj.weight");
    let e0 = off("model.layers.1.mlp.experts.0.gate_proj.weight");
    assert!(e3 < e0, "reversed schedule → expert 3 stored before expert 0 ({e3} vs {e0})");

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

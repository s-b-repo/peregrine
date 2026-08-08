//! The MoE dispatch report must read the dispatch path, not the environment.
//!
//! An **integration** test, not a unit test, and deliberately so: the engine
//! lives in a process-global `OnceLock` that is first-call-wins, so installing
//! one inside `peregrine-model`'s unit-test binary would change what every other
//! test in that binary dispatches through, permanently and in whatever order the
//! harness happened to run them. Each file under `tests/` is its own process, so
//! this one can install an engine and still assert the *uninstalled* state first.

use peregrine_model::concurrent::{
    install_moe_engine, moe_engine_installed, moe_engine_name, ForwardCtx, MoeCall, MoeEngine,
};
use peregrine_core::Error;

/// Never dispatched — the test asserts reporting, and a real forward would need
/// a container. `moe_forward` returning an error rather than a plausible vector
/// is the point: if some future change made this reachable, it fails loudly.
struct NamedEngine;

impl MoeEngine for NamedEngine {
    fn name(&self) -> &'static str {
        "test-engine"
    }
    fn moe_forward(&self, _ctx: &ForwardCtx, _call: MoeCall) -> Result<Vec<f32>, Error> {
        Err(Error::Format("NamedEngine is a reporting fixture and must never dispatch".into()))
    }
}

#[test]
fn the_report_follows_the_dispatch_path_and_not_the_env_var() {
    // `COLI_MOE_ENGINE` set with nothing installed is exactly the state a failed
    // ring leaves behind, and the state the old report got wrong.
    std::env::set_var("COLI_MOE_ENGINE", "sched");
    assert!(!moe_engine_installed(), "nothing has installed an engine yet");
    assert_eq!(
        moe_engine_name(),
        "concurrent",
        "with nothing installed the report must say concurrent, whatever COLI_MOE_ENGINE claims"
    );

    assert!(install_moe_engine(Box::new(NamedEngine)), "first install must win");
    assert!(moe_engine_installed());
    assert_eq!(moe_engine_name(), "test-engine", "the report must come from MoeEngine::name");

    // First-call-wins: a second installer loses, and the report must keep naming
    // the engine that is actually dispatching rather than the last one offered.
    assert!(!install_moe_engine(Box::new(NamedEngine)), "second install must be rejected");
    assert_eq!(moe_engine_name(), "test-engine");
}

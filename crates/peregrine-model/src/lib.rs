//! colibrì GLM-5.2 forward pass (M1, in progress).
//!
//! Ported piece by piece from `c/glm.c`, each validated in isolation before the
//! full forward is wired up. Present: the elementary numerics ([`math`]) and the
//! MoE router ([`router`]). Next: MLA attention, the MoE/dense-MLP expert compute
//! on [`peregrine_kernels`], the full layer/forward loop, and — on a machine with
//! `transformers` — the token-exact gate against `c/ref_glm.json`.

// Explicit index loops mirror the C forward pass (for verification) and mostly
// index several tensors at once — `needless_range_loop` is noise in this crate.

// Quality gates: no unsafe here, and no panicking error handling anywhere
// (denied even in tests).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod attention;
pub mod concurrent;
pub mod dsa;
pub mod gpu;
pub mod iotune;
pub mod lane;
pub mod learn;
pub mod math;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod predeval;
pub mod predict;
pub mod ram;
pub mod rlm;
pub mod router;
pub mod sample;
pub mod telemetry;
pub mod testkit;
pub mod topic;
pub mod weight;
pub mod wmma_tune;
pub mod workload;

pub use attention::{
    mla_attention, mla_attention_absorb, mla_attention_batched, mla_attention_rows, AttnWeights, KvDtype, KvSpan,
    LayerKv, RowAttn, RowLayout,
};
pub use math::{
    layernorm, rmsnorm, rmsnorm_inplace, rope_interleave, rope_interleave_with, sigmoidf, siluf, silu_mul, softmax,
    RopeTable,
};
pub use mlp::{moe_forward, Mlp, MoeCfg};
pub use model::{
    accept_run, accept_run_sampled, kv_dtype, lookahead_issued, save_automaton, save_macrostates, KvExport,
    KvLayerExport, Model, SeqKv, Verified,
};
pub use iotune::{IoTuner, IowqCap};
pub use lane::{Bias, BubbleTuner, LaneBalancer, LaneTimings, LaneTimingsAccum, Placement};
pub use learn::{learn_mode, BanditScheduler, KnobArm, LearnMode, QAction, QScheduler, QState};
pub use telemetry::{open_l3_miss_counter, PlanOptimizer, RuntimeTelemetry};
pub use wmma_tune::{KernelShape, TileConfig, WmmaTuner};
pub use workload::{classify_str, TokenClass};
pub use predeval::{ArmReport, PredictEval};
pub use predict::{phase_boost, Momentum, PredictSource, PrefetchTuner, RouteHistory, TransitionTable};
pub use rlm::{rlm_enabled, rlm_layers, rlm_margin, rlm_max_depth, RLMController};
pub use mtp::speculative_sample;
pub use router::{
    batch_union, gate_share_below, gate_stats_snapshot, route, union_low_gate_snapshot, union_stats_snapshot, Routed,
    RouterCfg,
};
pub use sample::{argmax, pick_batch_greedy, Sampler};
pub use weight::{QtWeight, QuantFmt};

/// Re-exported so the binaries can cap glibc malloc arenas at startup (drops the
/// `MALLOC_ARENA_MAX=2` requirement). See [`peregrine_io::cap_malloc_arenas`].
pub use peregrine_io::cap_malloc_arenas;

/// One-line-per-fact description of the machine this process will run on:
/// SIMD tier, CPU/NUMA topology, GPU backend status, and PCIe link per display
/// device.
///
/// **This function exists because four separate probes were written for exactly
/// this purpose and none of them had a caller.** `peregrine_kernels::cpu_kernel_tier`
/// ("for startup logging"), `peregrine_cuda::is_available` / `status`
/// ("for startup logging"), and `peregrine_io::topo::pcie_link_by_bdf` were all
/// reachable only from tests, while `docs/` described runtime topology discovery
/// as a shipped feature. They are assembled here rather than in the binary
/// because `peregrine-engine` depends on neither `peregrine-kernels`,
/// `peregrine-io` nor `peregrine-cuda` directly — this crate is the only one
/// that can see all four.
///
/// Returned as a `String` rather than printed so the caller decides the stream
/// and the tests can assert on the content.
pub fn startup_banner() -> String {
    let topo = peregrine_io::topo::snapshot();
    let mut s = format!(
        "peregrine: cpu={} | {} logical CPUs, {} NUMA node(s)",
        peregrine_kernels::cpu_kernel_tier(),
        topo.logical_cpus,
        topo.numa.len().max(1)
    );
    // The GPU half is cfg-split rather than absent off-CUDA: "not built" is a
    // fact worth printing, and it is the one people misread as "no GPU here".
    #[cfg(feature = "cuda")]
    {
        // `probe_device_count`, not `device_count`: the banner runs before any
        // `init`, and `device_count` reports contexts this process has built —
        // which is 0 at this point on every host, GPU or not.
        s.push_str(&format!(
            "\nperegrine: gpu={} ({}, {} device(s) visible)",
            if peregrine_cuda::is_available() { "available" } else { "unavailable" },
            peregrine_cuda::status(),
            peregrine_cuda::probe_device_count()
        ));
    }
    #[cfg(not(feature = "cuda"))]
    {
        s.push_str("\nperegrine: gpu=unavailable (CUDA backend not built — rebuild with `--features cuda`)");
    }
    for (bdf, link) in peregrine_io::topo::gpu_pcie_links() {
        // An x16 card negotiated to x4 makes `COLI_PCIE_BUDGET_MB` the wrong
        // knob to reach for, and nothing else in the engine would tell you.
        s.push_str(&format!("\nperegrine: pcie {bdf} {} x{}", link.speed, link.width));
    }
    s
}

#[cfg(test)]
mod banner_tests {
    /// The banner must actually consult each probe, not print a placeholder.
    ///
    /// Written as content assertions rather than "it returned a string" because
    /// the whole point of the function is that four probes acquired a caller —
    /// a test that only checks it is non-empty would still pass if someone
    /// replaced the body with a constant, which is precisely how these symbols
    /// became unreachable in the first place.
    #[test]
    fn banner_reports_every_probe_it_was_built_to_wire() {
        let b = super::startup_banner();
        assert!(b.contains("cpu="), "SIMD tier line missing: {b}");
        assert!(b.contains("gpu="), "GPU status line missing: {b}");
        assert!(b.contains("logical CPUs"), "topology line missing: {b}");
        assert!(b.contains("NUMA node"), "NUMA count missing: {b}");
        // `cpu_kernel_tier` returns one of three known strings; the banner must
        // carry whichever this host reports rather than a hardcoded guess.
        let tier = peregrine_kernels::cpu_kernel_tier();
        assert!(b.contains(tier), "banner must carry cpu_kernel_tier()'s answer ({tier}): {b}");
    }

    /// `is_available()` must not be `device_count() > 0`. That form is false
    /// until `init` has run, which made it useless for gating `init` and made
    /// it report "unavailable" on a working GPU. On a non-CUDA build both are
    /// 0 and this is vacuous; on a CUDA host it is the regression guard.
    #[test]
    fn gpu_availability_does_not_depend_on_having_initialized_first() {
        #[cfg(feature = "cuda")]
        {
            // Deliberately does NOT call init() first.
            let probed = peregrine_cuda::probe_device_count();
            assert_eq!(
                peregrine_cuda::is_available(),
                probed > 0,
                "is_available must track the driver's device count, not initialized contexts"
            );
        }
    }
}

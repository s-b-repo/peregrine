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
pub mod router;
pub mod sample;
pub mod telemetry;
pub mod testkit;
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
pub use model::{lookahead_issued, save_automaton, save_macrostates, Model, SeqKv};
pub use iotune::{IoTuner, IowqCap};
pub use lane::{Bias, BubbleTuner, LaneBalancer, LaneTimings, LaneTimingsAccum, Placement};
pub use learn::{learn_mode, BanditScheduler, KnobArm, LearnMode, QAction, QScheduler, QState};
pub use telemetry::{open_l3_miss_counter, PlanOptimizer, RuntimeTelemetry};
pub use wmma_tune::{KernelShape, TileConfig, WmmaTuner};
pub use workload::{classify_str, PhaseTracker, TokenClass};
pub use predeval::{ArmReport, PredictEval};
pub use predict::{Momentum, PredictSource, PrefetchTuner, RouteHistory, TransitionTable};
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

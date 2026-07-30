//! colibrì GLM-5.2 forward pass (M1, in progress).
//!
//! Ported piece by piece from `c/glm.c`, each validated in isolation before the
//! full forward is wired up. Present: the elementary numerics ([`math`]) and the
//! MoE router ([`router`]). Next: MLA attention, the MoE/dense-MLP expert compute
//! on [`peregrine_kernels`], the full layer/forward loop, and — on a machine with
//! `transformers` — the token-exact gate against `c/ref_glm.json`.

// Explicit index loops mirror the C forward pass (for verification) and mostly
// index several tensors at once — `needless_range_loop` is noise in this crate.
#![allow(clippy::needless_range_loop)]

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
pub mod predict;
pub mod router;
pub mod sample;
pub mod telemetry;
pub mod testkit;
pub mod weight;
pub mod wmma_tune;
pub mod workload;

pub use attention::{mla_attention, mla_attention_absorb, mla_attention_batched, AttnWeights, LayerKv, RowAttn};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoidf, siluf, silu_mul, softmax};
pub use mlp::{moe_forward, Mlp};
pub use model::{save_automaton, save_macrostates, Model, SeqKv};
pub use iotune::{IoTuner, IowqCap};
pub use lane::{Bias, BubbleTuner, LaneBalancer, LaneTimings, LaneTimingsAccum, Placement};
pub use learn::{learn_mode, BanditScheduler, KnobArm, LearnMode, QAction, QScheduler, QState};
pub use telemetry::{open_l3_miss_counter, PlanOptimizer, RuntimeTelemetry};
pub use wmma_tune::{KernelShape, TileConfig, WmmaTuner};
pub use workload::{classify_str, PhaseTracker, TokenClass};
pub use predict::{Momentum, PredictSource, PrefetchTuner, RouteHistory, TransitionTable};
pub use mtp::speculative_sample;
pub use router::{batch_union, route, Routed};
pub use sample::{argmax, pick_batch_greedy, Sampler};
pub use weight::{QtWeight, QuantFmt};

/// Re-exported so the binaries can cap glibc malloc arenas at startup (drops the
/// `MALLOC_ARENA_MAX=2` requirement). See [`peregrine_io::cap_malloc_arenas`].
pub use peregrine_io::cap_malloc_arenas;

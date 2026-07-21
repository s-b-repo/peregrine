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
pub mod math;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod router;
pub mod sample;
pub mod testkit;
pub mod weight;

pub use attention::{mla_attention, mla_attention_absorb, AttnWeights, LayerKv};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoidf, siluf, silu_mul, softmax};
pub use mlp::{moe_forward, Mlp};
pub use model::Model;
pub use mtp::speculative_sample;
pub use router::{batch_union, route, Routed};
pub use sample::{argmax, Sampler};
pub use weight::{QtWeight, QuantFmt};

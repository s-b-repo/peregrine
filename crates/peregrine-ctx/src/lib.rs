//! DCR — Dynamic Context Runtime.
//!
//! Ported from `s-b-repo/subnext` (crate `dcr`) for use as peregrine's
//! context plane (RLM working-set assembly). Upstream files are kept
//! verbatim apart from the crate prefix (`dcr::` → `peregrine_ctx::`) and
//! the CLI binary name, so future re-ports diff cleanly.
//!
//! Vendored code keeps upstream style; it is exempt from the engine crates'
//! panic-free gates (the integration seam in `peregrine-model` is held to
//! them). Like `peregrine-token`, the whole crate allows rather than denies.
//!
//! An implementation of the specification in `docs/`: unlimited external state
//! plus a small, planned working set.
//!
//! ```
//! use peregrine_ctx::Dcr;
//!
//! let mut rt = Dcr::new(800);
//! rt.ingest("The server ip is 10.0.4.12.", None).unwrap();
//! let answer = rt.ask("what is the server ip?", None);
//! println!("{} ({} tokens)", answer.text, answer.tokens);
//! ```
//!
//! The modules map one-to-one onto the wiki:
//!
//! | module | spec page |
//! |---|---|
//! | [`spans`] | L0 immutable raw store (`R_t`) |
//! | [`ladder`] | representation ladder L0–L3 |
//! | [`nodes`] | typed state schema |
//! | [`graph`] | dynamic memory graph (`S_t`, `E_t`) |
//! | [`execute`] | execution layer + memoisation (`C_t`) |
//! | [`indexer`] | state indexer / node extraction |
//! | [`index`] | hybrid retrieval over text *and* state |
//! | [`policy`] | runtime decision policy |
//! | [`budget`] | attention-budget knapsack (`B_attention`) |
//! | [`planner`] | relevance planner (`k + r`) |
//! | [`speculation`] | speculative context (`tau`, prefetch) |
//! | [`hash`] | SHA-256, the one primitive this crate owns |
//! | [`merkle`] | Merkle root and inclusion proofs over the container |
//! | [`telemetry`] | escalation / staleness / prefetch metrics |
//! | [`trust`] | the context gateway: verify → inspect → admit |
//! | [`context_store`] | the `.context` container: objects, checkpoints, chain |
//! | [`scrub`] | bit-rot detection and repair from verified replicas |
//! | [`baselines`] | RAG / summarise-all / recursive context assemblies |
//! | [`runtime`] | the facade tying it together |

pub mod baselines;
pub mod bench;
pub mod budget;
pub mod context_store;
pub mod demo;
pub mod embed;
pub mod execute;
pub mod graph;
pub mod hash;
pub mod ids;
pub mod index;
pub mod indexer;
pub mod json;
pub mod ladder;
pub mod llm;
pub mod merkle;
pub mod nodes;
pub mod planner;
pub mod policy;
pub mod runtime;
pub mod scrub;
pub mod spans;
pub mod speculation;
pub mod summarize;
pub mod telemetry;
pub mod text;
pub mod tokens;
pub mod trust;

pub use budget::{Allocation, Candidate, Choice, solve};
pub use context_store::{ContextError, ContextStore, ObjectKind, ObjectRecord};
pub use execute::ExecutionLayer;
pub use graph::{DcrError, ExplainPath, MemoryGraph};
pub use hash::{Digest, Sha256, sha256};
pub use index::HybridIndex;
pub use indexer::{HeuristicExtractor, IndexResult, StateIndexer};
pub use ladder::{Ladder, Level};
pub use llm::{CommandReasoner, LocalReasoner, Reasoner};
pub use merkle::{MerkleTree, verify_proof};
pub use nodes::{Edge, EdgeType, Kind, Node, NodeIdx, Status};
pub use planner::{ActiveContext, RelevancePlanner, Weights};
pub use policy::{DecisionPolicy, QueryType};
pub use runtime::{Answer, Dcr};
pub use scrub::{ScrubOptions, ScrubReport, scrub};
pub use spans::{RawStore, Span};
pub use speculation::{Predictor, Speculator};
pub use telemetry::Telemetry;
pub use trust::{
    AdmissionPolicy, Admitted, Candidate as TrustCandidate, ContextGateway, KeyId, KeyRole,
    Rejection, Signer, TrustLabel, Verification, Verifier,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

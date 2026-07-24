//! colibrì engine core (M0): model-directory formats shared by every later
//! crate — config parsing, the safetensors index, quantized-tensor format
//! detection, and dtype conversion. All ported to match the C engine
//! (`c/glm.c`, `c/st.h`) byte-for-byte so the Rust loader reproduces its
//! dequantization exactly.

// Quality gates (requested): no panicking error handling anywhere, no unsafe in
// this crate. `unwrap`/`expect`/`panic!` are denied even in tests.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod config;
pub mod dtype;
pub mod pack;
pub mod qt;
pub mod safetensors;

pub use config::Cfg;
pub use dtype::{bf16_to_f32, f16_to_f32, Dtype};
pub use qt::{detect_group_size, QtFmt, QtInfo};
pub use safetensors::{SafeTensors, TensorInfo};

use thiserror::Error as ThisError;

/// Errors from loading and running a model. Structured (thiserror): the `#[from]`
/// conversions let call sites use a plain `?` for I/O and JSON failures, and the
/// [`Context`] extension adds a human message (path/offset/operation) without a
/// `.map_err(|e| ...)` at the call site.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// malformed config/safetensors, out-of-bounds dimensions, missing tensors
    #[error("{0}")]
    Format(String),
    /// an inner error annotated with what was being attempted
    #[error("{msg}: {source}")]
    Context { msg: String, source: Box<Error> },
}

/// Attach a context message to any error that converts into [`Error`], turning
/// `foo().ctx(|| format!("reading {path}"))?` into a `Context` error. The single
/// `map_err` is encapsulated here, so no error-mapping closures appear at call
/// sites and the underlying error is preserved as `source`.
pub trait Context<T> {
    fn ctx(self, f: impl FnOnce() -> String) -> Result<T, Error>;
}

impl<T, E: Into<Error>> Context<T> for Result<T, E> {
    fn ctx(self, f: impl FnOnce() -> String) -> Result<T, Error> {
        self.map_err(|e| Error::Context { msg: f(), source: Box::new(e.into()) })
    }
}

//! colibrì engine core (M0): model-directory formats shared by every later
//! crate — config parsing, the safetensors index, quantized-tensor format
//! detection, and dtype conversion. All ported to match the C engine
//! (`c/glm.c`, `c/st.h`) byte-for-byte so the Rust loader reproduces its
//! dequantization exactly.

pub mod config;
pub mod dtype;
pub mod pack;
pub mod qt;
pub mod safetensors;

pub use config::Cfg;
pub use dtype::{bf16_to_f32, f16_to_f32, Dtype};
pub use qt::{detect_group_size, QtFmt, QtInfo};
pub use safetensors::{SafeTensors, TensorInfo};

use std::fmt;

/// Errors from loading a model directory.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// malformed config/safetensors, out-of-bounds dimensions, missing tensors
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Json(e) => write!(f, "json: {e}"),
            Error::Format(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

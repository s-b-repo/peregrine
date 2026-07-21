//! Reconstruct a streamed expert from the raw bytes of its safetensors tensors.
//!
//! Unlike the C engine's coalesced ~19 MB sidecar blob, peregrine streams each
//! expert's gate/up/down tensors **in place** from the checkpoint (6 regions:
//! packed weights + f32 scales, per projection). This module turns those raw
//! byte buffers back into `QtWeight`s / an `Mlp`, honoring the container format
//! (per-row int8/int4/int2 or grouped int4).

use peregrine_core::{Error, QtFmt};
use peregrine_model::{Mlp, QtWeight, QuantFmt};

/// Format + shape of one on-disk quantized tensor `[o, i]`.
#[derive(Clone, Copy, Debug)]
pub struct QtMeta {
    pub fmt: QtFmt,
    pub o: usize,
    pub i: usize,
    /// group size for [`QtFmt::Int4Grouped`], else 0
    pub gs: usize,
}

/// Decode f32 scales from their little-endian byte region.
fn scales_from_bytes(s: &[u8]) -> Vec<f32> {
    s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Rebuild one quantized weight from its streamed weight-bytes + scale-bytes.
/// Errors if the streamed metadata claims an unquantized (F32) tensor, which has
/// no compute path here.
pub fn qt_from_raw(meta: QtMeta, w: &[u8], s: &[u8]) -> Result<QtWeight, Error> {
    let scale = scales_from_bytes(s);
    match meta.fmt {
        QtFmt::Int4Grouped => Ok(QtWeight::new_grouped(meta.o, meta.i, w.to_vec(), scale, meta.gs)),
        fmt => {
            let qf = QuantFmt::from_qt(fmt)
                .ok_or_else(|| Error::Format("streamed expert tensor is unquantized (F32)".into()))?;
            Ok(QtWeight::new(qf, meta.o, meta.i, w.to_vec(), scale))
        }
    }
}

/// Rebuild an expert from its three streamed (weight, scale) tensor buffers.
/// `metas`/`bufs` are ordered gate, up, down.
pub fn mlp_from_segments(metas: &[QtMeta; 3], bufs: &[(Vec<u8>, Vec<u8>); 3]) -> Result<Mlp, Error> {
    Ok(Mlp {
        gate: qt_from_raw(metas[0], &bufs[0].0, &bufs[0].1)?,
        up: qt_from_raw(metas[1], &bufs[1].0, &bufs[1].1)?,
        down: qt_from_raw(metas[2], &bufs[2].0, &bufs[2].1)?,
    })
}

//! Serialize a resident expert to a single on-disk blob and reconstruct it
//! after streaming — the analog of the C engine's coalesced ~19 MB gate/up/down
//! read. Blob layout: `[gate_q | gate_s | up_q | up_s | down_q | down_s]`, int4.

use peregrine_core::pack::f32_bytes;
use peregrine_core::QtFmt;
use peregrine_model::{Mlp, QtWeight};

/// Expert dimensions needed to slice a blob back into gate/up/down.
#[derive(Clone, Copy, Debug)]
pub struct MlpDims {
    pub hidden: usize,
    pub inter: usize,
}

fn i4_qlen(o: usize, i: usize) -> usize {
    o * (i.div_ceil(2))
}

/// Serialize an int4 expert (gate/up/down) into one contiguous blob.
pub fn mlp_to_blob(m: &Mlp) -> Vec<u8> {
    let mut out = Vec::new();
    for w in [&m.gate, &m.up, &m.down] {
        let (q, s) = w.raw();
        out.extend_from_slice(q);
        out.extend_from_slice(&f32_bytes(s));
    }
    out
}

fn take_i4(blob: &[u8], cursor: &mut usize, o: usize, i: usize) -> QtWeight {
    let qlen = i4_qlen(o, i);
    let q = blob[*cursor..*cursor + qlen].to_vec();
    *cursor += qlen;
    let slen = o * 4;
    let s: Vec<f32> = blob[*cursor..*cursor + slen]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *cursor += slen;
    QtWeight::new(QtFmt::Int4, o, i, q, s)
}

/// Reconstruct an int4 expert from its blob given its dimensions.
pub fn mlp_from_blob(blob: &[u8], dims: MlpDims) -> Mlp {
    let (h, inter) = (dims.hidden, dims.inter);
    let mut cur = 0usize;
    Mlp {
        gate: take_i4(blob, &mut cur, inter, h),
        up: take_i4(blob, &mut cur, inter, h),
        down: take_i4(blob, &mut cur, h, inter),
    }
}

/// Byte length of an int4 expert blob for the given dims.
pub fn blob_len(dims: MlpDims) -> usize {
    let (h, inter) = (dims.hidden, dims.inter);
    // gate + up ([inter,h]) and down ([h,inter]); each = q bytes + o*4 scale bytes
    2 * (i4_qlen(inter, h) + inter * 4) + i4_qlen(h, inter) + h * 4
}

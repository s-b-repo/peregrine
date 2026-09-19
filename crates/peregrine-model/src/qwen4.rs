//! Qwen3.8-Flash-Next (`qwen4_exp`) mechanisms: gated residuals, GDN output
//! gating, QSA sparse selection, ngram PLE hashing, softmax top-k routing, and
//! the single-stream forward they compose into
//! (`Model::forward_hidden_qwen4`).
//!
//! Supported today (all exercised by `qwen4_tests` on the tiny fixture):
//! single-stream resident-CPU inference — `generate`, `forward_step`,
//! `forward_hidden`, `teacher_forcing` — with bit-identical chunked-prefill
//! vs stepwise-decode state handling (GDN recurrence, PLE hash/conv history,
//! KV and indexer-key appends, all sequential).
//!
//! Not supported — each refuses loudly with a `qwen4_exp`-prefixed error
//! rather than running a degraded path:
//! * expert streaming (`COLI_STREAM=1`): the shared-gate MoE only has a
//!   resident implementation;
//! * batched / concurrent serving (`forward_rows_batched*`, the serve engine):
//!   per-sequence GDN/PLE recurrence and stream-form residuals have no
//!   multi-sequence counterpart yet;
//! * MTP drafts and speculative decoding (no MTP head is ever loaded;
//!   `generate_speculative` without a head falls back to plain greedy, which
//!   is correct, while `mtp_draft*` without a head errors);
//! * token trees (MLA-only everywhere);
//! * RLM recursive replay (replays a `[hidden]`-wide hidden the stream-form
//!   layers cannot consume);
//! * GPU lanes (the Qwen4 forward never consults them — CPU-only, no silent
//!   fallback involved);
//! * prefetch / route-history / tuner side channels (the Qwen4 forward logs
//!   into none of them; correctness-neutral omissions, not errors).
//!
//! Parallelism: the heavy projections were already parallel —
//! `QtWeight::apply_vec` splits output rows above a `1<<20`-MAC gate — so the
//! serial leftovers parallelized here are the Qwen4-local ops
//! (`Matrix::apply`, the mixer token loops, norm groups, MoE plans), each
//! behind the same style of work gate. Every split is over independent
//! outputs with reductions kept in serial order, hence bit-identical;
//! `qwen4_parallel_matches_serial` proves it, comparing both directly
//! (`to_bits`) and through a compact FNV-1a hash. Within one decode token the
//! only available axis is the output-row split, which is why decode stays
//! matmul-bound and prefill gains the token axis too.
use crate::math::{sigmoidf, siluf};
use peregrine_core::{Cfg, Error};
use peregrine_core::config::Qwen4OutputGate;

#[cfg(test)]
#[path = "qwen4_tests.rs"]
mod tests;

pub(crate) fn require_runtime(_cfg: &Cfg) -> Result<(), Error> {
    // Single-stream resident forward (gated residuals, GDN, QSA-selected GQA,
    // PLE, softmax MoE) is implemented in `Model::forward_hidden_qwen4` and
    // exercised by `qwen4_tiny_model_loads_and_decodes_deterministically`.
    // Expert streaming and batched serving remain unsupported and fail with
    // explicit errors on those paths instead of at load.
    Ok(())
}

pub fn mtp_unsupported() -> Error {
    Error::Format("qwen4_exp: MTP and speculative decoding are not supported on this architecture".into())
}

/// Loud refusal for one of the paths listed as unsupported in the module docs.
/// One message per path so a failure names the capability that is missing
/// instead of pointing at MTP (which was never the reason batched serving or
/// RLM replay cannot run here).
pub(crate) fn unsupported(what: &str) -> Error {
    Error::Format(format!("qwen4_exp: {what} is not supported on this architecture"))
}

fn invalid(message: &str) -> Error {
    Error::Format(format!("qwen4_exp: {message}"))
}

fn finite(values: &[f32]) -> Result<(), Error> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite mechanism input or output"));
    }
    Ok(())
}



pub fn rms_norm(x: &[f32], weight: &[f32], group_size: usize, eps: f32) -> Result<Vec<f32>, Error> {
    if group_size == 0 || weight.is_empty() || !weight.len().is_multiple_of(group_size)
        || !x.len().is_multiple_of(weight.len()) || !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("invalid grouped RMS norm geometry or epsilon"));
    }
    finite(x)?;
    finite(weight)?;
    let mut out = x.to_vec();
    // Groups normalize disjoint segments, so groups split across workers
    // bit-identically. Gated like `rmsnorm_rows`: narrow groups (QSA query
    // probes, single pooled keys) stay serial; wide stream rows split.
    let gate = if group_size >= 256 { peregrine_par::PAR_ROWS_MIN } else { usize::MAX };
    peregrine_par::par_rows_mut(&mut out, group_size, x.len() / group_size, gate, |g, group| {
        let inv = (group.iter().map(|v| v * v).sum::<f32>() / group_size as f32 + eps).sqrt().recip();
        let start = (g * group_size) % weight.len();
        for (j, v) in group.iter_mut().enumerate() {
            *v = *v * inv * (1.0 + weight[start + j]);
        }
    });
    finite(&out)?;
    Ok(out)
}

pub fn gdn_output_norm(x: &[f32], z: &[f32], weight: &[f32], eps: f32, gate: Qwen4OutputGate) -> Result<Vec<f32>, Error> {
    if weight.is_empty() || !x.len().is_multiple_of(weight.len()) || x.len() != z.len()
        || !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("invalid GDN output norm geometry or epsilon"));
    }
    finite(x)?;
    finite(z)?;
    finite(weight)?;
    let mut out = x.to_vec();
    for (h, row) in out.chunks_exact_mut(weight.len()).enumerate() {
        let inv = (row.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + eps).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            let z = z[h * weight.len() + j];
            let activation = match gate { Qwen4OutputGate::Sigmoid => sigmoidf(z), Qwen4OutputGate::Silu => siluf(z) };
            *v = *v * inv * weight[j] * activation;
        }
    }
    finite(&out)?;
    Ok(out)
}

pub fn split_query_gate(projected: &[f32], heads: usize, head_dim: usize) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let width = heads.checked_mul(head_dim).and_then(|v| v.checked_mul(2)).filter(|&v| v > 0)
        .ok_or_else(|| invalid("invalid query/gate geometry"))?;
    if !projected.len().is_multiple_of(width) {
        return Err(invalid("ragged query/gate projection"));
    }
    finite(projected)?;
    let mut query = Vec::with_capacity(projected.len() / 2);
    let mut gate = Vec::with_capacity(projected.len() / 2);
    for head in projected.chunks_exact(2 * head_dim) {
        query.extend_from_slice(&head[..head_dim]);
        gate.extend_from_slice(&head[head_dim..]);
    }
    Ok((query, gate))
}

pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, Error> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(values.len()) {
            return Err(invalid("invalid matrix geometry"));
        }
        finite(&values)?;
        Ok(Self { rows, cols, values })
    }

    pub fn load_critical(st: &peregrine_core::SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Self, Error> {
        use peregrine_core::{Dtype, QtFmt, QtInfo};
        let count = rows.checked_mul(cols).filter(|&n| n > 0).ok_or_else(|| invalid("invalid critical matrix geometry"))?;
        let tensor = st.tensors().iter().find(|t| t.name == name).ok_or_else(|| invalid(&format!("missing {name}")))?;
        let fmt = QtInfo::detect(st, name, rows as i64, cols as i64).fmt;
        let values = match fmt {
            QtFmt::F32 if tensor.dtype == Dtype::F32 && tensor.shape == [rows as i64, cols as i64] => {
                let mut values = vec![0.0; count];
                st.read_f32(name, &mut values)?;
                values
            }
            QtFmt::Int8 => crate::weight::QtWeight::load(st, name, rows, cols)?.dequant(),
            _ => return Err(invalid(&format!("critical tensor {name} requires F32 or int8, not {fmt:?}"))),
        };
        Self::new(rows, cols, values)
    }

    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        if !x.len().is_multiple_of(self.cols) {
            return Err(invalid("matrix input width mismatch"));
        }
        finite(x)?;
        let in_rows = x.len() / self.cols;
        let mut out = vec![0.0; in_rows * self.rows];
        // Parallel over output rows above a work gate: each output is one
        // independent dot product summed left-to-right exactly as below, so
        // any split across workers is bit-identical to the serial loop. Same
        // `1<<20`-MAC gate as `QtWeight::apply_vec` — the mixer projections
        // that dominate decode (`[lowrank, hidden*hc]`) clear it, tiny test
        // shapes stay serial.
        let macs = in_rows.saturating_mul(self.rows).saturating_mul(self.cols);
        let gate = if macs >= 1 << 20 { peregrine_par::PAR_MATMUL_MIN } else { usize::MAX };
        let (rows, cols, vals) = (self.rows, self.cols, &self.values);
        peregrine_par::par_rows_mut(&mut out, 1, in_rows * rows, gate, |r, slot| {
            let xr = &x[(r / rows) * cols..(r / rows + 1) * cols];
            let wr = &vals[(r % rows) * cols..(r % rows + 1) * cols];
            slot[0] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
        });
        finite(&out)?;
        Ok(out)
    }
}

pub struct GatedResidual {
    hidden: usize,
    streams: usize,
    eps: f32,
    norm: Vec<f32>,
    down: Matrix,
    up: Matrix,
    inject: Option<Matrix>,
}

pub struct ResidualInput {
    pub mixed: Vec<f32>,
    pub injection: Option<Vec<f32>>,
}

impl GatedResidual {
    pub fn new(hidden: usize, streams: usize, eps: f32, norm: Vec<f32>, down: Matrix, up: Matrix, inject: Option<Matrix>) -> Result<Self, Error> {
        let width = hidden.checked_mul(streams).ok_or_else(|| invalid("residual width overflow"))?;
        if hidden == 0 || streams < 2 || norm.len() != width || down.cols != width || up.rows != width
            || up.cols != down.rows || inject.as_ref().is_some_and(|w| w.cols != width || w.rows != streams)
            || !eps.is_finite() || eps <= 0.0 {
            return Err(invalid("invalid gated residual geometry"));
        }
        finite(&norm)?;
        Ok(Self { hidden, streams, eps, norm, down, up, inject })
    }

    pub fn mix(&self, x: &[f32]) -> Result<ResidualInput, Error> {
        let n = rms_norm(x, &self.norm, self.hidden, self.eps)?;
        let low: Vec<f32> = self.down.apply(&n)?.into_iter().map(|v| siluf(v / self.streams as f32)).collect();
        let gates = self.up.apply(&low)?;
        let width = self.hidden * self.streams;
        let mut mixed = vec![0.0; x.len() / width * self.hidden];
        // Per-token rows are independent — each accumulates its own streams in
        // stream order — so tokens split across workers bit-identically.
        // Serial at decode (`s_n == 1`); the matmuls above carry that case.
        let (hidden, streams) = (self.hidden, self.streams);
        peregrine_par::par_rows_mut(&mut mixed, hidden, x.len() / width, peregrine_par::PAR_ROWS_MIN, |t, row| {
            for h in 0..streams {
                for (j, value) in row.iter_mut().enumerate() {
                    let index = t * width + h * hidden + j;
                    *value += sigmoidf(gates[index]) * n[index];
                }
            }
            for v in row.iter_mut() { *v /= streams as f32; }
        });
        let injection = self.inject.as_ref().map(|w| w.apply(&n).map(|v| v.into_iter()
            .map(|v| 2.0 * sigmoidf(v / self.streams as f32)).collect())).transpose()?;
        finite(&mixed)?;
        Ok(ResidualInput { mixed, injection })
    }

    pub fn combine(&self, residual: &[f32], block: &[f32], injection: &[f32]) -> Result<Vec<f32>, Error> {
        let width = self.hidden * self.streams;
        let rows = residual.len() / width;
        if !residual.len().is_multiple_of(width) || block.len() != rows * self.hidden || injection.len() != rows * self.streams {
            return Err(invalid("residual injection shape mismatch"));
        }
        finite(residual)?;
        finite(block)?;
        finite(injection)?;
        let mut out = residual.to_vec();
        // Purely elementwise — no reductions at all — so any row split is
        // bit-identical. Same token-row gate as `mix`.
        let (hidden, streams) = (self.hidden, self.streams);
        peregrine_par::par_rows_mut(&mut out, width, rows, peregrine_par::PAR_ROWS_MIN, |t, row| {
            for h in 0..streams {
                for j in 0..hidden {
                    row[h * hidden + j] += block[t * hidden + j] * injection[t * streams + h];
                }
            }
        });
        finite(&out)?;
        Ok(out)
    }
}

pub fn softmax_topk(logits: &[f32], k: usize, normalize: bool) -> Result<Vec<(usize, f32)>, Error> {
    if k == 0 || k > logits.len() { return Err(invalid("invalid softmax top-k")); }
    finite(logits)?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    order.truncate(k);
    let sum = if normalize { order.iter().map(|&i| probs[i]).sum() } else { probs.iter().sum::<f32>() };
    Ok(order.into_iter().map(|i| (i, probs[i] / sum)).collect())
}

#[derive(Clone)]
pub struct NgramHash {
    vocab: i64,
    eos: i64,
    multipliers: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    history: Vec<i64>,
    heads_per_order: usize,
    pub padded_vocab: i64,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn prime(value: i64) -> bool {
    if value < 2 { return false; }
    if value % 2 == 0 { return value == 2; }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value % divisor == 0 { return false; }
        divisor += 2;
    }
    true
}

impl NgramHash {
    pub fn new(cfg: &Cfg, layer_id: i64) -> Result<Self, Error> {
        let q = cfg.qwen4.as_ref().ok_or_else(|| invalid("ngram config missing"))?;
        let layer_index = q.ple_layer_ids.iter().position(|&id| id == layer_id).ok_or_else(|| invalid("layer has no PLE"))?;
        let eos = cfg.stop_ids.first().copied().ok_or_else(|| invalid("ngram EOS missing"))? as i64;
        if cfg.vocab <= 0 || eos < 0 || eos >= cfg.vocab || !(2..=32).contains(&q.ngram_size)
            || !(1..=256).contains(&q.heads_per_ngram) || !(2..=1 << 31).contains(&q.ngram_vocab_size_base)
            || q.ngram_vocab_divisor <= 0 {
            return Err(invalid("invalid ngram geometry"));
        }
        let heads = ((q.ngram_size - 1) * q.heads_per_ngram) as usize;
        let bound = (i64::MAX / cfg.vocab / 2).max(1) as u64;
        let seed = (q.seed as u64).wrapping_add(10007 * layer_index as u64);
        let multipliers = (0..q.ngram_size).map(|i| {
            let value = seed.wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(i as u64 + 1));
            (2 * (splitmix64(value) % bound) + 1) as i64
        }).collect();
        let mut size = q.ngram_vocab_size_base - 1;
        let mut sizes = Vec::with_capacity(heads);
        let mut offsets = Vec::with_capacity(heads);
        let mut total = 0i64;
        for index in 0..(layer_index + 1) * heads {
            size += 1;
            while !prime(size) { size += 1; }
            if index >= layer_index * heads {
                sizes.push(size);
                offsets.push(total);
                total = total.checked_add(size).ok_or_else(|| invalid("ngram vocabulary overflow"))?;
            }
        }
        let divisor = q.ngram_vocab_divisor;
        let padded_vocab = total.checked_add(divisor - 1).and_then(|v| (v / divisor).checked_mul(divisor))
            .ok_or_else(|| invalid("ngram padded vocabulary overflow"))?;
        Ok(Self { vocab: cfg.vocab, eos, multipliers, sizes, offsets,
            history: vec![eos; q.ngram_size as usize - 1], heads_per_order: q.heads_per_ngram as usize, padded_vocab })
    }

    pub fn reset(&mut self) { self.history.fill(self.eos); }

    pub fn ids(&mut self, tokens: &[i32]) -> Result<Vec<i64>, Error> {
        if tokens.iter().any(|&t| t < 0 || t as i64 >= self.vocab) {
            return Err(invalid("ngram token outside vocabulary"));
        }
        let count = tokens.len().checked_mul(self.sizes.len()).ok_or_else(|| invalid("ngram output overflow"))?;
        let mut out = Vec::with_capacity(count);
        for &token in tokens {
            let mut mixed = (token as i64).wrapping_mul(self.multipliers[0]);
            let mut crossed_eos = false;
            for order in 1..self.multipliers.len() {
                let previous = self.history[self.history.len() - order];
                crossed_eos |= previous == self.eos;
                let previous = if crossed_eos { self.eos } else { previous };
                mixed ^= previous.wrapping_mul(self.multipliers[order]);
                for h in 0..self.heads_per_order {
                    let index = (order - 1) * self.heads_per_order + h;
                    out.push(mixed.rem_euclid(self.sizes[index]) + self.offsets[index]);
                }
            }
            self.history.rotate_left(1);
            if let Some(last) = self.history.last_mut() { *last = token as i64; }
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub(crate) struct PleState {
    hash: NgramHash,
    history: Vec<f32>,
}

impl PleState {
    pub(crate) fn bytes(&self) -> usize { self.history.len() * 4 + self.hash.history.len() * 8 }
}

pub(crate) fn load_vector(st: &peregrine_core::SafeTensors, name: &str, shape: &[i64]) -> Result<Vec<f32>, Error> {
    let t = st.tensors().iter().find(|t| t.name == name).ok_or_else(|| invalid(&format!("missing {name}")))?;
    if t.dtype != peregrine_core::Dtype::F32 || t.shape != shape { return Err(invalid(&format!("{name}: expected F32 {shape:?}"))); }
    let mut values = vec![0.0; shape.iter().product::<i64>() as usize];
    st.read_f32(name, &mut values)?;
    finite(&values)?;
    Ok(values)
}

impl GatedResidual {
    pub(crate) fn load(st: &peregrine_core::SafeTensors, prefix: &str, cfg: &Cfg, combine: bool) -> Result<Self, Error> {
        let q = cfg.qwen4.as_ref().ok_or_else(|| invalid("missing config"))?;
        let (d, h, r) = (cfg.hidden as usize, q.hc_count as usize, q.hc_lowrank as usize);
        let p = |s: &str| format!("{prefix}.{s}.weight");
        Self::new(d, h, cfg.eps, load_vector(st, &p("hc_norm"), &[(d * h) as i64])?,
            Matrix::load_critical(st, &p("input_mix_weight_down"), r, d * h)?,
            Matrix::load_critical(st, &p("input_mix_weight_up"), d * h, r)?,
            if combine { Some(Matrix::load_critical(st, &p("block_inject_weight"), h, d * h)?) } else { None })
    }
}

pub(crate) struct PleTable {
    name: String,
    rows: usize,
    width: usize,
    int8: bool,
}

impl PleTable {
    fn load(st: &peregrine_core::SafeTensors, name: String, rows: usize, width: usize) -> Result<Self, Error> {
        let t = st.tensors().iter().find(|t| t.name == name).ok_or_else(|| invalid(&format!("missing {name}")))?;
        let int8 = st.has(&format!("{name}.qs"));
        if t.shape != [rows as i64, width as i64] || !matches!(t.compression, peregrine_core::Compression::None)
            || (!int8 && t.dtype != peregrine_core::Dtype::F32)
            || (int8 && t.dtype != peregrine_core::Dtype::U8) {
            return Err(invalid(&format!("{name}: PLE requires uncompressed F32 or per-row int8 [{rows},{width}]")));
        }
        if int8 {
            let sc = st.tensors().iter().find(|t| t.name == format!("{name}.qs")).ok_or_else(|| invalid("missing PLE scales"))?;
            if sc.dtype != peregrine_core::Dtype::F32 || sc.shape != [rows as i64]
                || !matches!(sc.compression, peregrine_core::Compression::None) { return Err(invalid("invalid PLE row scales")); }
        }
        Ok(Self { name, rows, width, int8 })
    }

    fn lookup(&self, st: &peregrine_core::SafeTensors, ids: &[i64]) -> Result<Vec<f32>, Error> {
        let mut out = Vec::with_capacity(ids.len() * self.width);
        for &id in ids {
            let row = usize::try_from(id).ok().filter(|&r| r < self.rows).ok_or_else(|| invalid("PLE row outside table"))?;
            let mut values = vec![0.0; self.width];
            if self.int8 {
                let (fd, offset, _) = st.region(&self.name).ok_or_else(|| invalid("missing PLE region"))?;
                let mut raw = vec![0u8; self.width];
                let mut done = 0;
                while done < raw.len() {
                    let result = peregrine_io::pread_many(&mut [peregrine_io::ReadReq { fd, offset: offset + (row * self.width + done) as u64, buf: &mut raw[done..], tag: 0 }])[0];
                    if result <= 0 { return Err(invalid("PLE row read failed or truncated")); }
                    done += result as usize;
                }
                let mut scale = [0.0];
                st.read_slice_f32(&format!("{}.qs", self.name), row as i64, 1, &mut scale)?;
                for (v, b) in values.iter_mut().zip(raw) { *v = b as i8 as f32 * scale[0]; }
            } else {
                st.read_slice_f32(&self.name, (row * self.width) as i64, self.width as i64, &mut values)?;
            }
            finite(&values)?;
            out.extend(values);
        }
        Ok(out)
    }
}

pub(crate) struct Ple {
    hash: NgramHash,
    table: PleTable,
    key: Matrix,
    value: Matrix,
    key_norm: Vec<f32>,
    query_norm: Vec<f32>,
    conv_norm: Vec<f32>,
    conv: Vec<f32>,
    kernel: usize,
    dilation: usize,
    hidden: usize,
    streams: usize,
    eps: f32,
}

impl Ple {
    fn load(st: &peregrine_core::SafeTensors, layer: usize, cfg: &Cfg) -> Result<Self, Error> {
        let q = cfg.qwen4.as_ref().ok_or_else(|| invalid("missing config"))?;
        let hash = NgramHash::new(cfg, layer as i64 + 1)?;
        let (d, h, e) = (cfg.hidden as usize, q.hc_count as usize, q.ple_embed_dim as usize);
        let prefix = format!("model.layers.{layer}.ple");
        let ep = format!("{prefix}.ple_embedding");
        for (suffix, expected) in [("layer_multipliers", &hash.multipliers), ("ngram_heads_vocab_sizes", &hash.sizes), ("ngram_heads_offsets", &hash.offsets)] {
            let name = format!("{ep}.{suffix}");
            if let Some(t) = st.tensors().iter().find(|t| t.name == name) {
                if t.dtype != peregrine_core::Dtype::I64 || t.shape != [expected.len() as i64] { return Err(invalid("invalid persistent ngram buffer")); }
                let mut values = vec![0i64; expected.len()];
                st.read_i64(&name, &mut values)?;
                if &values != expected { return Err(invalid("persistent ngram buffer disagrees with config")); }
            }
        }
        let table = PleTable::load(st, format!("model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight"), hash.padded_vocab as usize, e / hash.sizes.len())?;
        Ok(Self { hash, table,
            key: Matrix::load_critical(st, &format!("{prefix}.key_proj.weight"), d * h, e)?,
            value: Matrix::load_critical(st, &format!("{prefix}.value_proj.weight"), d, e)?,
            key_norm: load_vector(st, &format!("{prefix}.norm_key.weight"), &[(d * h) as i64])?,
            query_norm: load_vector(st, &format!("{prefix}.norm_query.weight"), &[(d * h) as i64])?,
            conv_norm: load_vector(st, &format!("{prefix}.norm_conv.weight"), &[(d * h) as i64])?,
            conv: load_vector(st, &format!("{prefix}.conv1d.weight"), &[(d * h) as i64, 1, q.ple_conv_kernel_size])?,
            kernel: q.ple_conv_kernel_size as usize, dilation: q.ngram_size as usize, hidden: d, streams: h, eps: cfg.eps })
    }

    pub(crate) fn inject(&self, st: &peregrine_core::SafeTensors, token: i32, x: &mut [f32], state: &mut Option<PleState>) -> Result<(), Error> {
        let width = self.hidden * self.streams;
        let history_len = (self.kernel - 1) * self.dilation;
        if x.len() != width { return Err(invalid("PLE residual width mismatch")); }
        let state = state.get_or_insert_with(|| PleState { hash: self.hash.clone(), history: vec![0.0; history_len * width] });
        let embedding = self.table.lookup(st, &state.hash.ids(&[token])?)?;
        let key = rms_norm(&self.key.apply(&embedding)?, &self.key_norm, self.hidden, self.eps)?;
        let value = self.value.apply(&embedding)?;
        let query = rms_norm(x, &self.query_norm, self.hidden, self.eps)?;
        let mut gated = vec![0.0; width];
        for h in 0..self.streams {
            let range = h * self.hidden..(h + 1) * self.hidden;
            let score = key[range.clone()].iter().zip(&query[range.clone()]).map(|(a, b)| a * b).sum::<f32>() / (self.hidden as f32).sqrt();
            let score = if score == 0.0 { 0.0 } else { score.signum() * score.abs().max(1e-6).sqrt() };
            for (dst, &v) in gated[range].iter_mut().zip(&value) { *dst = sigmoidf(score) * v; }
        }
        let normalized = rms_norm(&gated, &self.conv_norm, self.hidden, self.eps)?;
        for (ch, dst) in x.iter_mut().enumerate() {
            let mut acc = 0.0;
            for tap in 0..self.kernel {
                let v = if tap + 1 == self.kernel { normalized[ch] } else { state.history[tap * self.dilation * width + ch] };
                acc += self.conv[ch * self.kernel + tap] * v;
            }
            *dst += gated[ch] + siluf(acc);
        }
        if history_len > 0 {
            state.history.copy_within(width.., 0);
            state.history[(history_len - 1) * width..].copy_from_slice(&normalized);
        }
        finite(x)
    }
}

pub(crate) struct Layer {
    pub attn: GatedResidual,
    pub mlp: GatedResidual,
    pub ple: Option<Ple>,
    pub shared_gate: Matrix,
    pub index: Option<(Matrix, Qsa)>,
    pub gdn_abz: Option<(Matrix, Matrix, Matrix)>,
}

impl Layer {
    pub(crate) fn load(st: &peregrine_core::SafeTensors, layer: usize, cfg: &Cfg) -> Result<Self, Error> {
        let q = cfg.qwen4.as_ref().ok_or_else(|| invalid("missing config"))?;
        let pre = format!("model.layers.{layer}");
        let d = cfg.hidden as usize;
        let index = if cfg.full_attn[layer] {
            let ix = q.qsa.as_ref().ok_or_else(|| invalid("missing QSA config"))?;
            Some((Matrix::load_critical(st, &format!("{pre}.self_attn.indexer.index_qk_proj.weight"), ((ix.n_heads + 1) * ix.head_dim) as usize, d)?,
                Qsa::new(cfg, load_vector(st, &format!("{pre}.self_attn.indexer.q_layernorm.weight"), &[ix.head_dim])?,
                    load_vector(st, &format!("{pre}.self_attn.indexer.k_layernorm.weight"), &[ix.head_dim])?)?))
        } else { None };
        let gdn_abz = if !cfg.full_attn[layer] {
            Some((Matrix::load_critical(st, &format!("{pre}.linear_attn.in_proj_a.weight"), cfg.lin_v_heads as usize, d)?,
                Matrix::load_critical(st, &format!("{pre}.linear_attn.in_proj_b.weight"), cfg.lin_v_heads as usize, d)?,
                Matrix::load_critical(st, &format!("{pre}.linear_attn.in_proj_z.weight"), (cfg.lin_v_heads * cfg.lin_v_dim) as usize, d)?))
        } else { None };
        Ok(Self { attn: GatedResidual::load(st, &format!("{pre}.attn_hyper_connection"), cfg, true)?,
            mlp: GatedResidual::load(st, &format!("{pre}.mlp_hyper_connection"), cfg, true)?,
            ple: q.ple_layer_ids.contains(&(layer as i64 + 1)).then(|| Ple::load(st, layer, cfg)).transpose()?,
            shared_gate: Matrix::load_critical(st, &format!("{pre}.mlp.shared_expert_gate.weight"), 1, d)?, index, gdn_abz })
    }
}

pub(crate) fn route(x: &[f32], weights: &[f32], cfg: &Cfg) -> Result<crate::router::Routed, Error> {
    let d = cfg.hidden as usize;
    let k = cfg.topk as usize;
    let mut routed = crate::router::Routed { idx: Vec::new(), w: Vec::new(), keff: vec![k as i32; x.len() / d], k };
    for row in x.chunks_exact(d) {
        let logits: Vec<f32> = weights.chunks_exact(d).map(|w| w.iter().zip(row).map(|(a, b)| a * b).sum()).collect();
        for (e, w) in softmax_topk(&logits, k, cfg.norm_topk)? { routed.idx.push(e as i32); routed.w.push(w); }
    }
    Ok(routed)
}

/// Softmax-MoE forward for Qwen4Exp: `route` (plain softmax, no bias) selects
/// the experts, each selected expert's SwiGLU is scattered back weighted, and
/// the shared expert is added gated by `sigmoid(shared_gate·x)`.
///
/// Batch-union ordered like [`crate::mlp::moe_forward`] (one pass to bucket,
/// serial scatter in union order) so prefill-chunked and stepwise decodes are
/// bit-identical — the property `generate` depends on.
pub(crate) fn moe_forward_qwen4(
    x: &[f32],
    router_w: &[f32],
    shared_gate: &Matrix,
    experts: &[crate::mlp::Mlp],
    shared: Option<&crate::mlp::Mlp>,
    cfg: &Cfg,
    s_n: usize,
) -> Result<Vec<f32>, Error> {
    let d = cfg.hidden as usize;
    if s_n == 0 || x.len() != s_n * d || experts.is_empty() {
        return Err(invalid("invalid Qwen4 MoE batch geometry"));
    }
    finite(x)?;
    finite(router_w)?;
    let r = route(x, router_w, cfg)?;
    let mut out = vec![0.0f32; s_n * d];
    struct Plan { e: usize, rows: Vec<usize>, rw: Vec<f32> }
    let union = crate::router::batch_union(&r, s_n);
    let mut slot_of: Vec<Option<usize>> = vec![None; experts.len()];
    let mut plans: Vec<Plan> = Vec::with_capacity(union.len());
    for &e in union.iter() {
        if let Some(slot) = usize::try_from(e).ok().filter(|&e| e < experts.len()) {
            if slot_of[slot].is_none() {
                slot_of[slot] = Some(plans.len());
                plans.push(Plan { e: slot, rows: Vec::new(), rw: Vec::new() });
            }
        }
    }
    for s in 0..s_n {
        for kk in 0..(r.keff[s].max(0) as usize).min(r.k) {
            let Ok(e) = usize::try_from(r.idx[s * r.k + kk]) else { continue };
            let Some(&Some(pi)) = slot_of.get(e) else { continue };
            if plans[pi].rows.last() == Some(&s) { continue; }
            plans[pi].rows.push(s);
            plans[pi].rw.push(r.w[s * r.k + kk]);
        }
    }
    plans.retain(|p| !p.rows.is_empty());
    // Experts compute on the pool (disjoint scratch per plan); the scatter
    // stays serial in batch-union order, so `f32 +=` accumulation order — and
    // every bit — matches the serial loop. Same shape as `mlp::moe_forward`.
    let hs: Vec<Vec<f32>> = peregrine_par::par_map(plans.len(), peregrine_par::PAR_MOE_MIN, |i| {
        let p = &plans[i];
        let nr = p.rows.len();
        let mut xg = vec![0.0f32; nr * d];
        for (ri, &s) in p.rows.iter().enumerate() {
            xg[ri * d..ri * d + d].copy_from_slice(&x[s * d..s * d + d]);
        }
        experts[p.e].swiglu(&xg, nr)
    });
    for (p, h) in plans.iter().zip(&hs) {
        for (ri, (&s, &wgt)) in p.rows.iter().zip(&p.rw).enumerate() {
            let dst = &mut out[s * d..s * d + d];
            let src = &h[ri * d..ri * d + d];
            for j in 0..d {
                dst[j] += wgt * src[j];
            }
        }
    }
    if let Some(sh) = shared {
        let hs = sh.swiglu(x, s_n);
        let gates = shared_gate.apply(x)?;
        if gates.len() != s_n {
            return Err(invalid("shared-gate width mismatch"));
        }
        for s in 0..s_n {
            let g = sigmoidf(gates[s]);
            let dst = &mut out[s * d..s * d + d];
            let src = &hs[s * d..s * d + d];
            for j in 0..d {
                dst[j] += g * src[j];
            }
        }
    }
    finite(&out)?;
    Ok(out)
}

pub struct Qsa {
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    ratio: usize,
    budget: usize,
    theta: f32,
    eps: f32,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
}

impl Qsa {
    pub fn new(cfg: &Cfg, query_norm: Vec<f32>, key_norm: Vec<f32>) -> Result<Self, Error> {
        let qsa = cfg.qwen4.as_ref().and_then(|q| q.qsa.as_ref()).ok_or_else(|| invalid("QSA config missing"))?;
        if qsa.n_heads <= 0 || qsa.head_dim <= 0 || qsa.compress_ratio <= 0 || qsa.budget <= 0
            || qsa.budget % qsa.compress_ratio != 0 || cfg.qk_rope < 2 || cfg.qk_rope % 2 != 0
            || cfg.qk_rope > qsa.head_dim || query_norm.len() != qsa.head_dim as usize || key_norm.len() != query_norm.len()
            || !cfg.theta.is_finite() || cfg.theta <= 0.0 || !cfg.eps.is_finite() || cfg.eps <= 0.0 {
            return Err(invalid("invalid QSA geometry"));
        }
        finite(&query_norm)?;
        finite(&key_norm)?;
        Ok(Self { heads: qsa.n_heads as usize, head_dim: qsa.head_dim as usize,
            rotary_dim: cfg.qk_rope as usize, ratio: qsa.compress_ratio as usize, budget: qsa.budget as usize,
            theta: cfg.theta, eps: cfg.eps, query_norm, key_norm })
    }

    fn rotate(&self, head: &mut [f32], position: usize) {
        let half = self.rotary_dim / 2;
        for j in 0..half {
            let angle = position as f32 / self.theta.powf((2 * j) as f32 / self.rotary_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (head[j], head[j + half]);
            head[j] = a * cos - b * sin;
            head[j + half] = b * cos + a * sin;
        }
    }

    pub fn select(&self, raw_query: &[f32], raw_keys: &[f32], positions: &[usize], query_position: usize, visible: &[usize]) -> Result<Vec<usize>, Error> {
        if raw_query.len() != self.heads * self.head_dim || raw_keys.len() / self.head_dim != positions.len()
            || !raw_keys.len().is_multiple_of(self.head_dim) || visible.iter().any(|&i| i >= positions.len())
            || visible.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("QSA query, key, position or visibility shape mismatch"));
        }
        finite(raw_keys)?;
        let mut query = rms_norm(raw_query, &self.query_norm, self.head_dim, self.eps)?;
        for q in query.chunks_exact_mut(self.head_dim) { self.rotate(q, query_position); }
        let complete = visible.len() / self.ratio;
        let mut scores = Vec::with_capacity(complete);
        for block in visible[..complete * self.ratio].chunks_exact(self.ratio) {
            let mut pooled = vec![0.0; self.head_dim];
            for &token in block {
                for j in 0..self.head_dim { pooled[j] += raw_keys[token * self.head_dim + j]; }
            }
            for v in &mut pooled { *v /= self.ratio as f32; }
            let mut key = rms_norm(&pooled, &self.key_norm, self.head_dim, self.eps)?;
            self.rotate(&mut key, positions[block[0]]);
            let score = query.chunks_exact(self.head_dim).map(|q| q.iter().zip(&key)
                .map(|(a, b)| a * b).sum::<f32>().max(0.0)).sum::<f32>() / (self.head_dim as f32).sqrt();
            scores.push(score);
        }
        finite(&scores)?;
        let blocks = crate::dsa::select_topk(&scores, self.budget / self.ratio);
        let mut selected = Vec::with_capacity(self.budget.min(visible.len()) + self.ratio - 1);
        for block in blocks { selected.extend_from_slice(&visible[block * self.ratio..(block + 1) * self.ratio]); }
        selected.extend_from_slice(&visible[complete * self.ratio..]);
        Ok(selected)
    }
}

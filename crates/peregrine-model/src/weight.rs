//! `QtWeight` — the bridge from `peregrine-core` container formats to `peregrine-kernels`
//! matmuls. Owns one quantized weight `[O, I]` (packed bytes + per-row scales)
//! and applies it to f32 activations via the IDOT path (quantize activations
//! with `qrow_i8`, then integer-dot).
//!
//! Supports every container format GLM-5.2 experts ship in — per-row int8
//! (fmt 1), per-row packed int4 (fmt 2), grouped packed int4 (fmt 4, the
//! coherence-critical format for GLM-5.2), and packed int2 (fmt 3).

use peregrine_core::{Error, QtFmt, QtInfo, SafeTensors};
use peregrine_io::Bytes;
use peregrine_kernels::{
    dot_i2i8, dot_i2i8_g64, dot_i3i8_g64, matmul_i4_from_f32, matmul_i4g_from_f32, matmul_i8_from_f32, qrow_i8, ActScratch, MatShape,
};

/// The quantized formats the compute path supports. F32 is rejected at load, so
/// it is deliberately *unrepresentable* here — this makes `apply`/`dequant`
/// exhaustive with no `unreachable!`/panic branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFmt {
    Int8,
    Int4,
    Int4Grouped,
    Int2,
    /// int3 with per-group scales, group 64 (colibrì fmt 5) — 3.5 bits/weight.
    Int3G64,
    /// **affine** int2 with a scale *and* zero-point per 64-value group, the two
    /// interleaved in the scale array as `[s, z]`. 3.0 bits/weight effective
    /// (the two f32 per group cost a full bit/weight on top of the payload).
    /// Unlike
    /// [`QuantFmt::Int2`] all four levels carry weight, because the bias is the
    /// group's own zero-point rather than a fixed `-2`.
    Int2G64,
}

impl QuantFmt {
    /// Narrow a detected container format to a computable one. `F32` (a tensor
    /// with no `.qs` scale sibling) and `Unknown` (a payload matching no
    /// container for the requested shape) have no quantized compute path.
    pub fn from_qt(f: QtFmt) -> Option<QuantFmt> {
        match f {
            QtFmt::Int8 => Some(QuantFmt::Int8),
            QtFmt::Int4 => Some(QuantFmt::Int4),
            QtFmt::Int4Grouped => Some(QuantFmt::Int4Grouped),
            QtFmt::Int2 => Some(QuantFmt::Int2),
            QtFmt::Int3G64 => Some(QuantFmt::Int3G64),
            QtFmt::Int2G64 => Some(QuantFmt::Int2G64),
            QtFmt::F32 | QtFmt::Unknown => None,
        }
    }
}

/// A VRAM-resident copy of a weight, when one exists. Uninhabited without the
/// `cuda` feature, so `Option<DevMatrix>` costs nothing and every non-cuda
/// build proves at compile time that no device path can be taken.
#[cfg(feature = "cuda")]
pub type DevMatrix = peregrine_cuda::GpuMatrix;
#[cfg(not(feature = "cuda"))]
pub type DevMatrix = std::convert::Infallible;

/// One quantized weight matrix `[O, I]`.
pub struct QtWeight {
    pub fmt: QuantFmt,
    pub o: usize,
    pub i: usize,
    /// packed weight bytes: int8 as u8 (reinterpreted), int4 as nibbles, int2 as
    /// 2-bit fields (4/byte). A [`Bytes`] region so a streamed expert can own its
    /// O_DIRECT aligned DMA buffer directly (zero-copy); resident loads use a plain
    /// `Vec`. Every kernel reads it as `&[u8]` via `Deref`, so the shape is invisible
    /// to the compute path.
    q: Bytes,
    /// scales: `O` per-row (int8/int4/int2) or `O*ceil(I/gs)` grouped (fmt 4),
    /// laid out `scale[o*ng + g]`
    scale: Vec<f32>,
    /// group size for [`QtFmt::Int4Grouped`] (weights per shared scale), else 0
    gs: usize,
    /// A VRAM-resident copy, when this weight was uploaded ([`Self::upload_to_device`]).
    /// Consulted only at `s_n == 1` — decode — where the device GEMV both wins
    /// on time and keeps f32 throughout; batched shapes stay on the CPU path,
    /// which is already parallel over output rows.
    dev: Option<DevMatrix>,
}

/// `COLI_ACT_F32=1`: compute quantized matmuls against **f32** activations
/// instead of the int8 ones `qrow_i8` produces ([`QtWeight::apply_vec_f32_act`]).
///
/// A `OnceLock` here rather than a build-time read, against this repo's usual
/// preference, and the reason is specific: this is a measurement knob with no
/// config plumbed to the matmul hot path, and the one harness that A/Bs it —
/// `peregrine flip-rate` — already runs each arm in its own **process**
/// (`--candidate-env`, built precisely because latched knobs cannot be
/// toggled in-process). So the latch cannot produce the vacuous same-arm
/// comparison it produces for serving knobs.
fn act_f32() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("COLI_ACT_F32").as_deref(), Ok("1") | Ok("true")))
}

impl QtWeight {
    /// Build from already-quantized per-row data (also the test constructor).
    /// For grouped-int4 use [`Self::new_grouped`].
    pub fn new(fmt: QuantFmt, o: usize, i: usize, q: impl Into<Bytes>, scale: Vec<f32>) -> QtWeight {
        debug_assert!(
            matches!(fmt, QuantFmt::Int8 | QuantFmt::Int4 | QuantFmt::Int2 | QuantFmt::Int3G64 | QuantFmt::Int2G64),
            "QtWeight::new is for formats with an implicit group layout (got {fmt:?}); \
             use new_grouped for grouped int4, whose group size is a runtime value"
        );
        QtWeight { fmt, o, i, q: q.into(), scale, gs: 0, dev: None }
    }

    /// Build a grouped-int4 weight: `scale` holds `o*ceil(i/gs)` entries laid out
    /// `scale[o*ng + g]`.
    pub fn new_grouped(o: usize, i: usize, q: impl Into<Bytes>, scale: Vec<f32>, gs: usize) -> QtWeight {
        debug_assert!(gs > 0 && gs.is_multiple_of(16), "grouped-int4 gs must be a positive multiple of 16");
        QtWeight { fmt: QuantFmt::Int4Grouped, o, i, q: q.into(), scale, gs, dev: None }
    }

    /// Load a container weight `[O, I]` (`name` + `name.qs`) from a model dir.
    pub fn load(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<QtWeight, Error> {
        let info = QtInfo::detect(st, name, o as i64, i as i64);
        let fmt = match QuantFmt::from_qt(info.fmt) {
            Some(f) => f,
            None if info.fmt == QtFmt::Unknown => {
                return Err(Error::Format(format!(
                    "weight '{name}': {} bytes match no quantized container for [{o},{i}] \
                     (truncated tensor, or the shape disagrees with the checkpoint)",
                    st.uncompressed_nbytes(name).unwrap_or(0)
                )))
            }
            None => {
                return Err(Error::Format(format!(
                    "weight '{name}': no `.qs` scale sibling (runtime-f32 weights are not \
                     supported on the quantized expert path)"
                )))
            }
        };
        let nb = match fmt {
            QuantFmt::Int8 => o * i,
            QuantFmt::Int4 | QuantFmt::Int4Grouped => o * (i.div_ceil(2)),
            QuantFmt::Int2 => o * (i.div_ceil(4)),
            QuantFmt::Int3G64 => o * i.div_ceil(peregrine_core::pack::I3_GROUP) * peregrine_core::pack::I3_GROUP_BYTES,
            QuantFmt::Int2G64 => o * i.div_ceil(peregrine_core::pack::I2G_GROUP) * peregrine_core::pack::I2G_GROUP_BYTES,
        };
        let mut q = vec![0u8; nb];
        st.read_raw(name, &mut q)?;
        let mut scale = vec![0f32; info.scale_count as usize];
        st.read_f32(&format!("{name}.qs"), &mut scale)?;
        Ok(QtWeight { fmt, o, i, q: q.into(), scale, gs: info.gs as usize, dev: None })
    }

    /// Raw quantized payload: `(packed_bytes, per_row_scales)`. Lets the
    /// scheduler serialize a resident expert to a disk blob and reconstruct it
    /// after streaming (`QtWeight::new`).
    pub fn raw(&self) -> (&[u8], &[f32]) {
        (&self.q[..], &self.scale)
    }

    /// `q` viewed as int8 (only meaningful for [`QuantFmt::Int8`]). `u8` and `i8`
    /// are both `Pod` with identical layout, so `bytemuck` reinterprets the slice
    /// with no `unsafe` and no copy.
    fn as_i8(&self) -> &[i8] {
        bytemuck::cast_slice::<u8, i8>(&self.q[..])
    }

    /// `(bytes per weight row, scales per weight row)` — the row pitch an output-row
    /// window slices on. `None` for the formats whose `apply` arm is a bespoke
    /// scalar loop rather than one of the batched `matmul_*` kernels; those keep
    /// the whole-matrix path (they are the rare compression rungs, never the
    /// formats a served model actually runs).
    fn row_pitch(&self) -> Option<(usize, usize)> {
        match self.fmt {
            QuantFmt::Int8 => Some((self.i, 1)),
            QuantFmt::Int4 => Some((self.i.div_ceil(2), 1)),
            QuantFmt::Int4Grouped => Some((self.i.div_ceil(2), self.i.div_ceil(self.gs))),
            QuantFmt::Int2 | QuantFmt::Int3G64 | QuantFmt::Int2G64 => None,
        }
    }

    /// [`Self::apply`] restricted to output rows `rows`, for a **single** activation
    /// row. `y` receives exactly `rows.len()` floats.
    ///
    /// Weights are `[O, I]` row-major, so an output-row range is a contiguous slice
    /// of both the packed bytes and the scales — the window is a borrow, never a
    /// copy, and each row's dot is the same one the full-matrix path computes.
    fn apply_window(&self, rows: std::ops::Range<usize>, x: &[f32], act: ActScratch<'_>, y: &mut [f32]) {
        let Some((rb, sp)) = self.row_pitch() else { return };
        let shape = MatShape::new(1, self.i, rows.end - rows.start);
        let q = &self.q[rows.start * rb..rows.end * rb];
        let sc = &self.scale[rows.start * sp..rows.end * sp];
        match self.fmt {
            QuantFmt::Int8 => matmul_i8_from_f32(y, x, bytemuck::cast_slice::<u8, i8>(q), sc, shape, act),
            QuantFmt::Int4 => matmul_i4_from_f32(y, x, q, sc, shape, act),
            QuantFmt::Int4Grouped => matmul_i4g_from_f32(y, x, q, sc, shape, self.gs, act),
            // `row_pitch` returned None for these, so they never reach here.
            QuantFmt::Int2 | QuantFmt::Int3G64 | QuantFmt::Int2G64 => {}
        }
    }

    /// `y[s_n, O] = apply(self, x[s_n, I])`. Caller provides int8 activation
    /// scratch `xq[s_n*I]`, per-row scale scratch `sx[s_n]`, and output `y`.
    pub fn apply(&self, x: &[f32], s_n: usize, xq: &mut [i8], sx: &mut [f32], y: &mut [f32]) {
        match self.fmt {
            QuantFmt::Int8 => matmul_i8_from_f32(y, x, self.as_i8(), &self.scale, MatShape::new(s_n, self.i, self.o), ActScratch { xq, sx }),
            QuantFmt::Int4 => matmul_i4_from_f32(y, x, &self.q[..], &self.scale, MatShape::new(s_n, self.i, self.o), ActScratch { xq, sx }),
            QuantFmt::Int4Grouped => {
                matmul_i4g_from_f32(y, x, &self.q[..], &self.scale, MatShape::new(s_n, self.i, self.o), self.gs, ActScratch { xq, sx })
            }
            QuantFmt::Int2 => {
                // Quantize activations, then int2·int8 dot each row (AVX2 when
                // available, else scalar — bit-identical). int2 is rare (extra
                // compression); the scalar path remains the token-exact reference.
                for s in 0..s_n {
                    sx[s] = qrow_i8(&x[s * self.i..s * self.i + self.i], &mut xq[s * self.i..s * self.i + self.i]);
                }
                let rb = self.i.div_ceil(4);
                for o in 0..self.o {
                    let w = &self.q[o * rb..o * rb + rb];
                    let sc = self.scale[o];
                    for s in 0..s_n {
                        let d = dot_i2i8(w, &xq[s * self.i..s * self.i + self.i], self.i) as f32;
                        y[s * self.o + o] = d * sc * sx[s];
                    }
                }
            }
            QuantFmt::Int3G64 => {
                // Per-group scales, so the integer dot is accumulated and scaled
                // per 64-value group — the same shape as grouped int4, which is
                // why this composes with the int8-activation path at all.
                for s in 0..s_n {
                    sx[s] = qrow_i8(&x[s * self.i..s * self.i + self.i], &mut xq[s * self.i..s * self.i + self.i]);
                }
                let ng = self.i.div_ceil(peregrine_core::pack::I3_GROUP);
                let rb = ng * peregrine_core::pack::I3_GROUP_BYTES;
                for o in 0..self.o {
                    let w = &self.q[o * rb..o * rb + rb];
                    let sc = &self.scale[o * ng..o * ng + ng];
                    for s in 0..s_n {
                        let d = dot_i3i8_g64(w, &xq[s * self.i..s * self.i + self.i], sc, self.i);
                        y[s * self.o + o] = d * sx[s];
                    }
                }
            }
            QuantFmt::Int2G64 => {
                // Same grouped shape as int3-g64, with two scales per group
                // instead of one. The affine zero-point is folded inside the
                // kernel (`s·Σqx + z·Σx`), so nothing extra is needed here — and
                // like int3, the per-row activation scale `sx[s]` multiplies the
                // whole group-scaled sum, since `qrow_i8` is per row.
                use peregrine_core::pack::{I2G_GROUP, I2G_GROUP_BYTES, I2G_SCALES_PER_GROUP};
                for s in 0..s_n {
                    sx[s] = qrow_i8(&x[s * self.i..s * self.i + self.i], &mut xq[s * self.i..s * self.i + self.i]);
                }
                let ng = self.i.div_ceil(I2G_GROUP);
                let rb = ng * I2G_GROUP_BYTES;
                let sw = ng * I2G_SCALES_PER_GROUP;
                for o in 0..self.o {
                    let w = &self.q[o * rb..o * rb + rb];
                    let sc = &self.scale[o * sw..o * sw + sw];
                    for s in 0..s_n {
                        let d = dot_i2i8_g64(w, &xq[s * self.i..s * self.i + self.i], sc, self.i);
                        y[s * self.o + o] = d * sx[s];
                    }
                }
            }
        }
    }

    /// Allocating convenience over [`Self::apply`]. Output rows are independent, so
    /// the matmul runs as batched row-chunks on the persistent compute pool above
    /// `PAR_MATMUL_MIN` — each chunk does one batched `apply` with its own quantized
    /// scratch, so the result is bit-identical to the serial whole-matrix call
    /// (guarded by `apply_vec_parallel_matches_serial`). Serial below the gate or
    /// when nested (e.g. an expert matmul inside a parallel MoE), so no
    /// oversubscription. This parallelizes every projection, lm_head, and expert.
    /// Upload this weight to `device` if it is per-row int4 and fits the
    /// device's free VRAM with `headroom` to spare. Returns whether it landed.
    ///
    /// Only per-row int4 (`fmt = 2`) uploads raw; every other format would be
    /// dequantized to f32 on the way, costing 8x the VRAM for the same weight,
    /// so those are declined rather than silently made expensive. Idempotent:
    /// an already-resident weight reports `true` without re-uploading.
    #[cfg(feature = "cuda")]
    pub fn upload_to_device(&mut self, device: i32, headroom: usize) -> Result<bool, Error> {
        if self.dev.is_some() {
            return Ok(true);
        }
        if self.fmt != QuantFmt::Int4 {
            return Ok(false);
        }
        let need = self.packed_bytes();
        let (free, _) = peregrine_cuda::mem_info(device)?;
        if free < need.saturating_add(headroom) {
            return Ok(false);
        }
        self.dev = Some(peregrine_cuda::GpuMatrix::upload_int4(device, &self.q[..], &self.scale, self.o, self.i)?);
        Ok(true)
    }

    /// Without the backend there is no device to upload to, so this is always
    /// `false`. Split per-config rather than cfg'd inside one body: it keeps
    /// the parameters honestly unused here instead of needing a suppression.
    #[cfg(not(feature = "cuda"))]
    pub fn upload_to_device(&mut self, _device: i32, _headroom: usize) -> Result<bool, Error> {
        Ok(false)
    }

    /// Packed bytes this weight occupies in RAM — the denominator for "how
    /// much of the model is on the device".
    pub fn packed_bytes(&self) -> usize {
        self.q.len() + 4 * self.scale.len()
    }

    /// Bytes this weight occupies in VRAM (0 when it is not resident).
    pub fn device_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.dev.as_ref().map_or(0, |d| d.bytes())
        }
        // Without the backend `DevMatrix` is uninhabited, so this arm is
        // unreachable by construction — and reading the field here is what
        // proves it, rather than leaving a never-read field behind.
        #[cfg(not(feature = "cuda"))]
        {
            self.dev.as_ref().map_or(0, |d| match *d {})
        }
    }

    pub fn apply_vec(&self, x: &[f32], s_n: usize) -> Vec<f32> {
        // Decode against a VRAM-resident weight: the device GEMV is both faster
        // (measured 6.7x on a real layer shape) and more accurate (f32
        // throughout, rms 1.1e-7 vs the CPU path's 3.0e-3 — this path never
        // quantizes activations). Only `s_n == 1`: batched shapes take the WMMA
        // kernels or the CPU's output-row split, both already efficient there.
        // A device failure falls back rather than failing the request.
        #[cfg(feature = "cuda")]
        if s_n == 1 {
            if let Some(d) = &self.dev {
                match d.matvec(x) {
                    Ok(y) => return y,
                    Err(e) => peregrine_io::note_advisory_err("device matvec (CPU fallback)", &e),
                }
            }
        }
        // Measurement-only escape from activation quantization (`COLI_ACT_F32=1`).
        // Never a serving path — it dequantizes a weight row per output and dots
        // in f32, which is correct and slow. See [`Self::apply_vec_f32_act`].
        if act_f32() {
            return self.apply_vec_f32_act(x, s_n);
        }
        let mut y = vec![0f32; s_n * self.o];
        // DECODE (`s_n == 1`) splits the OUTPUT rows, not the batch rows.
        //
        // The batch split below is `par_chunks_mut(.., n = s_n, ..)`, which goes
        // serial whenever `s_n < PAR_MATMUL_MIN` — and decode's `s_n` is 1, so
        // every matmul of every layer ran on ONE worker while the other 11 sat
        // idle (measured: 190% CPU of a possible 1200% on the resident Qwen
        // serve). A resident dense model spends essentially all of its time
        // here, so that is most of its decode cost. Output rows are independent
        // and `[O, I]` row-major, so splitting them is both bit-identical and a
        // contiguous slice; `y` is `[1, O]`, so each chunk is contiguous too.
        //
        // Above the same `i·o` work gate as the batch path: a matmul too small
        // to pay for pool dispatch stays serial either way.
        if s_n == 1 && self.i * self.o >= 1 << 20 && self.row_pitch().is_some() {
            peregrine_par::par_chunks_mut(&mut y, 1, self.o, peregrine_par::PAR_MATMUL_MIN, |o0, o1, y_chunk| {
                // Per-worker activation scratch. The quantization of `x` is
                // repeated per chunk rather than hoisted: it is O(I) against the
                // chunk's O(I·(o1-o0)) of matmul (≈0.1% at Qwen's shapes), and
                // keeping it inside means the window path shares one code path
                // with `apply` instead of needing a pre-quantized variant.
                let mut xq = vec![0i8; self.i];
                let mut sx = vec![0f32; 1];
                self.apply_window(o0..o1, x, ActScratch { xq: &mut xq, sx: &mut sx }, y_chunk);
            });
            return y;
        }
        // Only parallelize when the per-row matmul work (`i·o` MACs) is large enough
        // that the pool dispatch pays off; tiny matrices (e.g. the test model) stay
        // serial regardless of batch, avoiding overhead-dominated slowdowns. Real
        // GLM-5.2 projections (6144×6144 ≈ 38M) clear this by ~36×.
        let mut gate = if self.i * self.o >= 1 << 20 { peregrine_par::PAR_MATMUL_MIN } else { usize::MAX };
        // Per-shape dispatch specialization (`COLI_SHAPE_SPECIALIZE=1`): the
        // first calls per (fmt, o, i) shape probe serial vs parallel and memoize
        // whichever measured faster — the heuristic gate above becomes a
        // measured, shape-specific decision. Dispatch-level specialization, not
        // codegen; bit-identical either way (both paths already are).
        let probing = shape_dispatch_pre(self.fmt, self.o, self.i, s_n, &mut gate);
        let t0 = probing.map(|_| std::time::Instant::now());
        peregrine_par::par_chunks_mut(&mut y, self.o, s_n, gate, |start, end, y_chunk| {
            let n = end - start;
            let mut xq = vec![0i8; n * self.i];
            let mut sx = vec![0f32; n];
            self.apply(&x[start * self.i..end * self.i], n, &mut xq, &mut sx, y_chunk);
        });
        if let (Some(used_par), Some(t0)) = (probing, t0) {
            shape_dispatch_post(self.fmt, self.o, self.i, used_par, t0.elapsed().as_nanos() as u64);
        }
        y
    }

    /// The same matmul with **no activation quantization**: weights dequantized
    /// exactly, activations left in f32.
    ///
    /// Exists to answer a question the engine could not otherwise ask about
    /// itself. Every quantized matmul here computes int4/int8 weights against
    /// **int8 activations** (`qrow_i8`), so a container's served error has two
    /// sources — the weights it stores and the activations the kernel quantizes
    /// on the fly — and every quantization gate this repo has ever run measured
    /// their sum while attributing it to the weights alone. Flipping this knob
    /// holds the weights fixed and removes the activation term, so the two
    /// become separable: the difference between a gate run with it and without
    /// it *is* the activation contribution.
    ///
    /// Measured motivation: on one MLP shape the device's fp16-activation path
    /// sat 29x closer to an f32 truth than this crate's int8-activation path
    /// (1.09e-4 vs 3.18e-3 RMS), which is large enough to move a flip rate on
    /// its own and therefore large enough to have been silently inflating them.
    ///
    /// Deliberately unoptimized — a reference, not a kernel. It allocates a row
    /// buffer per output row and does f32 dots with no SIMD path; at GLM/Qwen
    /// widths expect it to be far slower than [`Self::apply_vec`]. That is the
    /// correct trade for a measurement instrument: a fast one would have to
    /// reintroduce the very approximations it exists to remove.
    fn apply_vec_f32_act(&self, x: &[f32], s_n: usize) -> Vec<f32> {
        let mut y = vec![0f32; s_n * self.o];
        let mut row = vec![0f32; self.i];
        for o in 0..self.o {
            self.dequant_row_into(o, &mut row);
            for s in 0..s_n {
                let xr = &x[s * self.i..(s + 1) * self.i];
                // Same accumulation order as the reference dots elsewhere in
                // this file, so the only difference from `apply_vec` is the
                // activation precision — which is the whole point.
                y[s * self.o + o] = row.iter().zip(xr).map(|(w, v)| w * v).sum();
            }
        }
        y
    }

    /// Dequantize a single output row `o` to f32 `[I]` — used by the MLA
    /// absorption path (`qt_addrow` / `qt_matvec_rows`), which reads individual
    /// `kv_b` rows rather than a batched matmul.
    pub fn dequant_row(&self, o: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.i];
        self.dequant_row_into(o, &mut out);
        out
    }

    /// [`Self::dequant_row`] into a caller-owned buffer — the allocation-free
    /// form for hot loops. The MLA absorb core calls this once per (head,
    /// output element) per token per layer; allocating a fresh `Vec` each time
    /// cost on the order of a million allocations per decoded token, all of them
    /// re-deriving the same immutable weight.
    ///
    /// `out` must hold at least `self.i` floats; a shorter buffer is left
    /// untouched (the caller sized it wrong).
    pub fn dequant_row_into(&self, o: usize, out: &mut [f32]) {
        if out.len() < self.i {
            return;
        }
        let out = &mut out[..self.i];
        match self.fmt {
            QuantFmt::Int8 => {
                let s = self.scale[o];
                let q = self.as_i8();
                for (i, dst) in out.iter_mut().enumerate() {
                    *dst = q[o * self.i + i] as f32 * s;
                }
            }
            QuantFmt::Int4 => {
                let s = self.scale[o];
                let rb = self.i.div_ceil(2);
                for (i, dst) in out.iter_mut().enumerate() {
                    let byte = self.q[o * rb + (i >> 1)];
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    *dst = (nib - 8) as f32 * s;
                }
            }
            QuantFmt::Int4Grouped => {
                let rb = self.i.div_ceil(2);
                let ng = self.i.div_ceil(self.gs);
                for (i, dst) in out.iter_mut().enumerate() {
                    let s = self.scale[o * ng + i / self.gs];
                    let byte = self.q[o * rb + (i >> 1)];
                    let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    *dst = (nib - 8) as f32 * s;
                }
            }
            QuantFmt::Int2 => {
                let s = self.scale[o];
                let rb = self.i.div_ceil(4);
                for (i, dst) in out.iter_mut().enumerate() {
                    let byte = self.q[o * rb + (i >> 2)];
                    let field = ((byte >> (2 * (i & 3))) & 0x03) as i32;
                    *dst = (field - 2) as f32 * s;
                }
            }
            QuantFmt::Int3G64 => {
                use peregrine_core::pack::{i3_value, I3_GROUP, I3_GROUP_BYTES, I3_LOW_BYTES};
                let ng = self.i.div_ceil(I3_GROUP);
                let rb = ng * I3_GROUP_BYTES;
                for (i, dst) in out.iter_mut().enumerate() {
                    let (g, k) = (i / I3_GROUP, i % I3_GROUP);
                    let gb = o * rb + g * I3_GROUP_BYTES;
                    let v = i3_value(self.q[gb + (k >> 2)], self.q[gb + I3_LOW_BYTES + (k >> 3)], k);
                    *dst = v as f32 * self.scale[o * ng + g];
                }
            }
            QuantFmt::Int2G64 => {
                use peregrine_core::pack::{i2g_field, I2G_GROUP, I2G_GROUP_BYTES, I2G_SCALES_PER_GROUP};
                let ng = self.i.div_ceil(I2G_GROUP);
                let rb = ng * I2G_GROUP_BYTES;
                for (i, dst) in out.iter_mut().enumerate() {
                    let (g, k) = (i / I2G_GROUP, i % I2G_GROUP);
                    let si = (o * ng + g) * I2G_SCALES_PER_GROUP;
                    // Affine: `q·s + z`, not `(q - bias)·s`.
                    *dst = i2g_field(self.q[o * rb + g * I2G_GROUP_BYTES + (k >> 2)], k) as f32
                        * self.scale[si]
                        + self.scale[si + 1];
                }
            }
        }
    }

    /// Dequantize to a full f32 `[O, I]` matrix — for reference/validation paths.
    pub fn dequant(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.o * self.i];
        for o in 0..self.o {
            out[o * self.i..o * self.i + self.i].copy_from_slice(&self.dequant_row(o));
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Quantize an f32 weight `[O, I]` into a [`super::QtWeight`] for tests.
    use super::QtWeight;
    use super::QuantFmt;

    pub fn quant_i8(w: &[f32], o: usize, i: usize) -> QtWeight {
        let mut q = vec![0u8; o * i];
        let mut sc = vec![0f32; o];
        for oo in 0..o {
            let row = &w[oo * i..oo * i + i];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = (amax / 127.0).max(1e-12);
            sc[oo] = s;
            for ii in 0..i {
                q[oo * i + ii] = ((row[ii] / s).round_ties_even() as i32 as i8) as u8;
            }
        }
        QtWeight::new(QuantFmt::Int8, o, i, q, sc)
    }

    /// Grouped-int4 quantizer (matches colibrì `quant_int4_grouped`): one scale
    /// per `gs`-element group along the input dim, `sc[o*ng + g]`.
    pub fn quant_i4_grouped(w: &[f32], o: usize, i: usize, gs: usize) -> QtWeight {
        let rb = i.div_ceil(2);
        let ng = i.div_ceil(gs);
        let mut q = vec![0u8; o * rb];
        let mut sc = vec![0f32; o * ng];
        for oo in 0..o {
            let row = &w[oo * i..oo * i + i];
            for g in 0..ng {
                let (s, e) = (g * gs, ((g + 1) * gs).min(i));
                let amax = row[s..e].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let scale = (amax / 7.0).max(1e-12);
                sc[oo * ng + g] = scale;
                for ii in s..e {
                    let v = (row[ii] / scale).round_ties_even().clamp(-8.0, 7.0) as i32;
                    let bias = (v + 8) as u8 & 0x0F;
                    if ii & 1 == 0 {
                        q[oo * rb + (ii >> 1)] |= bias;
                    } else {
                        q[oo * rb + (ii >> 1)] |= bias << 4;
                    }
                }
            }
        }
        QtWeight::new_grouped(o, i, q, sc, gs)
    }

    pub fn quant_i4(w: &[f32], o: usize, i: usize) -> QtWeight {
        let rb = i.div_ceil(2);
        let mut q = vec![0u8; o * rb];
        let mut sc = vec![0f32; o];
        for oo in 0..o {
            let row = &w[oo * i..oo * i + i];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = (amax / 7.0).max(1e-12);
            sc[oo] = s;
            for ii in 0..i {
                let v = (row[ii] / s).round_ties_even().clamp(-8.0, 7.0) as i32;
                let bias = (v + 8) as u8 & 0x0F;
                if ii & 1 == 0 {
                    q[oo * rb + (ii >> 1)] |= bias;
                } else {
                    q[oo * rb + (ii >> 1)] |= bias << 4;
                }
            }
        }
        QtWeight::new(QuantFmt::Int4, o, i, q, sc)
    }
}

#[cfg(test)]
mod tests {

    /// A VRAM-resident weight must compute the SAME matmul as the CPU one at
    /// decode — the property every served token depends on once the tier is on.
    /// Tolerance rather than equality, because the device path is the more
    /// accurate of the two (f32 throughout vs int8 activations), and the
    /// direction is asserted as well as the magnitude: uploading a weight must
    /// not make it worse.
    /// The device matvec across the SHAPES A REAL MODEL ACTUALLY HAS, not one
    /// convenient square-ish one.
    ///
    /// Written after a squat [384,256] test passed while a serving run went
    /// wrong: Qwen's weights span from [48,5120] (the DeltaNet gates — 48
    /// output rows against a 5120-long reduction, so few blocks each reducing a
    /// long row) to [17408,5120] and [5120,17408]. Those are different kernel
    /// regimes, and a reduction bug shows in one and not the others. Any shape
    /// that reduces wrong is caught here rather than in a token stream.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_matvec_is_correct_across_real_model_shapes() -> Result<(), peregrine_core::Error> {
        // Shares the one device with every other GPU test in the crate.
        let _g = crate::gpu_test_lock::gpu_guard();
        if peregrine_cuda::init(&[0]) < 1 {
            return Ok(());
        }
        // (o, i): GDN gates, GDN qkv-ish, attention q/o, MLP gate/up, MLP down,
        // plus degenerate single-row and odd widths.
        for (o, i) in [(48usize, 5120usize), (2048, 5120), (5120, 5120), (1024, 5120), (5120, 1024), (1, 4096), (7, 33)] {
            let mut seed = 0xB0A7u64 ^ ((o * 131 + i) as u64);
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            };
            let w: Vec<f32> = (0..o * i).map(|_| rnd() * 0.1).collect();
            let mut qt = super::test_support::quant_i4(&w, o, i);
            let x: Vec<f32> = (0..i).map(|_| rnd()).collect();
            let cpu = qt.apply_vec(&x, 1);
            if !qt.upload_to_device(0, 16 << 20)? {
                continue; // no headroom for this one right now
            }
            let gpu = qt.apply_vec(&x, 1);
            let mut row = vec![0f32; i];
            let mut truth = vec![0f32; o];
            for oo in 0..o {
                qt.dequant_row_into(oo, &mut row);
                truth[oo] = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            }
            let rms = |v: &[f32]| {
                (v.iter().zip(&truth).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / v.len() as f32).sqrt()
            };
            let (e_cpu, e_gpu) = (rms(&cpu), rms(&truth.iter().zip(&gpu).map(|(_, g)| *g).collect::<Vec<_>>()));
            let scale = truth.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            // A shape the device entry declines (it validates before computing)
            // falls back to the CPU, and the two results are then bit-identical.
            // That is correct behaviour, not a wrong answer, so it is
            // distinguished rather than asserted against — what must never
            // happen is the device RUNNING and returning something wrong.
            let fell_back = cpu.iter().zip(&gpu).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "[{o}x{i}] rms vs truth: cpu {e_cpu:.3e}  gpu {e_gpu:.3e}  (scale {scale:.3e}){}",
                if fell_back { "  [declined -> CPU]" } else { "" }
            );
            if fell_back {
                continue;
            }
            assert!(
                e_gpu / scale < 1e-3,
                "device matvec is wrong at [{o}x{i}]: rms {e_gpu:.3e} on scale {scale:.3e}"
            );
            assert!(e_gpu <= e_cpu, "device must not be less accurate at [{o}x{i}]");
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_resident_weight_matvecs_like_the_cpu_path_only_better() -> Result<(), peregrine_core::Error> {
        // Shares the one device with every other GPU test in the crate.
        let _g = crate::gpu_test_lock::gpu_guard();
        if peregrine_cuda::init(&[0]) < 1 {
            return Ok(());
        }
        let (o, i) = (512usize, 256usize);
        let mut seed = 0xD00Du64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..o * i).map(|_| rnd() * 0.1).collect();
        let mut qt = super::test_support::quant_i4(&w, o, i);
        let x: Vec<f32> = (0..i).map(|_| rnd()).collect();

        let cpu = qt.apply_vec(&x, 1);
        if !qt.upload_to_device(0, 16 << 20)? {
            return Ok(()); // no headroom on the card right now
        }
        assert!(qt.device_bytes() > 0, "a resident weight must report its VRAM footprint");
        let gpu = qt.apply_vec(&x, 1); // same call — dispatch is transparent

        // Truth over the weights the container actually holds.
        let mut row = vec![0f32; i];
        let mut truth = vec![0f32; o];
        for oo in 0..o {
            qt.dequant_row_into(oo, &mut row);
            truth[oo] = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
        let rms = |v: &[f32]| {
            (v.iter().zip(&truth).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let (e_cpu, e_gpu) = (rms(&cpu), rms(&gpu));
        println!("matvec rms vs truth: cpu {e_cpu:.4e}  gpu {e_gpu:.4e}");
        assert!(e_gpu <= e_cpu, "uploading a weight must not make it less accurate ({e_gpu:.4e} vs {e_cpu:.4e})");

        // And batched shapes must still take the CPU path, untouched.
        let x2: Vec<f32> = (0..2 * i).map(|_| rnd()).collect();
        let batched = qt.apply_vec(&x2, 2);
        assert_eq!(batched.len(), 2 * o, "s_n > 1 keeps the existing path and shape");
        Ok(())
    }

    /// The f32-activation instrument must differ from the production path ONLY
    /// in activation precision — same weights, same shapes, same outputs to
    /// within the quantization error it removes. A bug that made it compute a
    /// different matmul entirely would make every attribution drawn from it
    /// wrong, so this pins it against a hand-rolled reference over the same
    /// dequantized weights (which is what it should equal exactly).
    #[test]
    fn the_f32_activation_path_computes_the_same_matmul_without_quantizing_x() {
        let (o, i, s_n) = (12usize, 32usize, 2usize);
        let mut seed = 0xACC0u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..o * i).map(|_| rnd()).collect();
        let qt = super::test_support::quant_i4(&w, o, i);
        let x: Vec<f32> = (0..s_n * i).map(|_| rnd()).collect();

        // Reference: dequantized weights, f32 activations, plain dots.
        let mut row = vec![0f32; i];
        let mut want = vec![0f32; s_n * o];
        for oo in 0..o {
            qt.dequant_row_into(oo, &mut row);
            for s in 0..s_n {
                want[s * o + oo] =
                    row.iter().zip(&x[s * i..(s + 1) * i]).map(|(a, b)| a * b).sum();
            }
        }
        let got = qt.apply_vec_f32_act(&x, s_n);
        assert!(
            got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "the instrument must be exactly the dequantized-weight f32 matmul"
        );

        // And it must be measurably CLOSER to that reference than the
        // production int8-activation path — the property the attribution rests
        // on. (Not merely different: closer.)
        let prod = qt.apply_vec(&x, s_n);
        let err = |v: &[f32]| v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(
            err(&got) <= err(&prod),
            "instrument err {:.3e} must not exceed production err {:.3e}",
            err(&got),
            err(&prod)
        );
    }
    use super::test_support::*;
    use super::QtWeight;
    use peregrine_kernels::{matmul_f32, ActScratch};

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn grouped_int4_loads_from_disk_and_matches() -> Result<(), peregrine_core::Error> {
        // Emit a grouped-int4 weight in the container format the converter
        // writes, then load it through QtWeight::load (format detection + read)
        // and confirm the loaded weight forwards identically to the in-memory one.
        use super::QuantFmt;
        use peregrine_core::pack::{f32_bytes, quant_i4_grouped, write_safetensors, Blob};
        use peregrine_core::SafeTensors;

        let (o, i, gs) = (4usize, 64usize, 16usize); // ng=4 → 16 scales > o → grouped
        let mut rng = Lcg(0x515e);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let (q, s) = quant_i4_grouped(&wf, o, i, gs);

        let dir = std::env::temp_dir().join(format!("coli_wg_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        write_safetensors(
            &dir,
            &[
                Blob::new("w", "U8", vec![o as i64, i.div_ceil(2) as i64], q.clone()),
                Blob::new("w.qs", "F32", vec![(o * (i / gs)) as i64], f32_bytes(&s)),
            ],
        )?;

        let st = SafeTensors::open(&dir)?;
        let loaded = QtWeight::load(&st, "w", o, i)?;
        assert_eq!(loaded.fmt, QuantFmt::Int4Grouped);
        assert_eq!(loaded.gs, gs);

        let mem = QtWeight::new_grouped(o, i, q, s, gs);
        let x: Vec<f32> = (0..2 * i).map(|_| rng.f()).collect();
        assert_eq!(loaded.apply_vec(&x, 2), mem.apply_vec(&x, 2), "disk-loaded grouped == in-memory");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn int2_g64_apply_matches_its_dequantized_matmul() {
        // Same contract as int3-g64 below, and it matters more here: the affine
        // form splits into `s·Σqx + z·Σx`, so a kernel that dropped the second
        // term would still produce plausible-looking numbers on centred weights.
        // A deliberately *off-centre* weight distribution makes the zero-point
        // carry real mass, so omitting it fails loudly.
        use peregrine_core::pack::quant_i2_g64;
        let (o, i, s_n) = (5usize, 128usize, 3usize);
        let mut rng = Lcg(0xC0FFEE);
        let wf: Vec<f32> = (0..o * i).map(|_| 2.0 + rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();
        let (q, sc) = quant_i2_g64(&wf, o, i);
        let w = QtWeight::new(super::QuantFmt::Int2G64, o, i, q, sc);

        let y = w.apply_vec(&xf, s_n);
        let wdq = w.dequant();
        let mut yref = vec![0f32; s_n * o];
        matmul_f32(&mut yref, &xf, &wdq, s_n, i, o);
        for k in 0..s_n * o {
            let tol = 0.05 * i as f32;
            assert!((y[k] - yref[k]).abs() < tol, "k={k} kernel={} dequant-matmul={}", y[k], yref[k]);
        }
    }

    #[test]
    fn int3_g64_apply_matches_its_dequantized_matmul() {
        // New format, new kernel: the contract is that `dot_i3i8_g64` computes
        // the same thing as dequantizing and doing an f32 matmul, within the
        // activation-quantization error the IDOT path already carries for every
        // other format. Without this the kernel could decode the two planes in
        // the wrong order and still produce plausible numbers.
        use peregrine_core::pack::quant_i3_g64;
        let (o, i, s_n) = (5usize, 128usize, 3usize);
        let mut rng = Lcg(0xC0FFEE);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();
        let (q, sc) = quant_i3_g64(&wf, o, i);
        let w = QtWeight::new(super::QuantFmt::Int3G64, o, i, q, sc);

        let y = w.apply_vec(&xf, s_n);
        let wdq = w.dequant();
        let mut yref = vec![0f32; s_n * o];
        matmul_f32(&mut yref, &xf, &wdq, s_n, i, o);
        for k in 0..s_n * o {
            let tol = 0.05 * i as f32;
            assert!((y[k] - yref[k]).abs() < tol, "k={k} kernel={} dequant-matmul={}", y[k], yref[k]);
        }
    }

    #[test]
    fn core_dequant_matches_qtweight() {
        // `peregrine_core::pack::QtView` decodes container bytes for the offline
        // tools, which depend on core alone and cannot reach `QtWeight`. Two
        // implementations of one format is a drift risk, so this pins them: same
        // bytes in, bit-identical floats out, for every format the tools touch.
        use peregrine_core::pack::QtView;
        use peregrine_core::qt::QtFmt;
        let (o, i) = (6usize, 40usize);
        let mut rng = Lcg(0x5EED);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let i2 = {
            // int2 has no test_support constructor; build it from core's producer.
            let (q, sc) = peregrine_core::pack::quant_i2(&wf, o, i);
            QtWeight::new(super::QuantFmt::Int2, o, i, q, sc)
        };
        let i3g = {
            let (q, sc) = peregrine_core::pack::quant_i3_g64(&wf, o, i);
            QtWeight::new(super::QuantFmt::Int3G64, o, i, q, sc)
        };
        let i2g = {
            let (q, sc) = peregrine_core::pack::quant_i2_g64(&wf, o, i);
            QtWeight::new(super::QuantFmt::Int2G64, o, i, q, sc)
        };
        for (w, fmt, gs) in [
            (quant_i8(&wf, o, i), QtFmt::Int8, 0usize),
            (quant_i4(&wf, o, i), QtFmt::Int4, 0),
            (i2, QtFmt::Int2, 0),
            (quant_i4_grouped(&wf, o, i, 16), QtFmt::Int4Grouped, 16),
            // The grouped formats were missing from this pin, so core and engine
            // could have drifted on exactly the two decoders with the most
            // arithmetic in them (two planes; a zero-point).
            (i3g, QtFmt::Int3G64, 64),
            (i2g, QtFmt::Int2G64, 64),
        ] {
            let (q, scale) = w.raw();
            let view = QtView { fmt, o, i, gs, q, scale };
            let mut a = vec![0f32; i];
            let mut b = vec![0f32; i];
            for r in 0..o {
                w.dequant_row_into(r, &mut a);
                view.dequant_row_into(r, &mut b);
                for k in 0..i {
                    assert_eq!(a[k].to_bits(), b[k].to_bits(), "{fmt:?} row {r} col {k}: core != engine");
                }
            }
        }
    }

    #[test]
    fn dequant_row_into_is_bit_identical_to_the_full_matrix() {
        // `Model::embed` keeps the table packed and pulls one row per token
        // instead of materializing the whole `[vocab, hidden]` f32 matrix
        // (2.85 GB of resident set at GLM-5.2 shapes). That substitution is only
        // sound while row `r` of `dequant()` is bit-for-bit `dequant_row_into(r)`
        // — true by construction today, asserted here so it stays true.
        let (o, i) = (7usize, 48usize);
        let mut rng = Lcg(0xB19);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            let whole = w.dequant();
            let mut row = vec![0f32; i];
            for r in 0..o {
                row.iter_mut().for_each(|v| *v = f32::NAN); // no stale-buffer pass
                w.dequant_row_into(r, &mut row);
                for k in 0..i {
                    assert_eq!(
                        row[k].to_bits(),
                        whole[r * i + k].to_bits(),
                        "fmt {:?} row {r} col {k}: row-wise dequant differs from the full matrix",
                        w.fmt
                    );
                }
            }
        }
    }

    #[test]
    fn dequant_apply_tracks_f32() {
        let (o, i, s_n) = (5usize, 40usize, 3usize);
        let mut rng = Lcg(0x77);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();

        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            // apply() (quantized activations) vs a full-f32 matmul with the
            // dequantized weights — must agree within quant error.
            let y = w.apply_vec(&xf, s_n);
            let wdq = w.dequant();
            let mut yref = vec![0f32; s_n * o];
            matmul_f32(&mut yref, &xf, &wdq, s_n, i, o);
            for k in 0..s_n * o {
                let tol = 0.03 * i as f32;
                assert!((y[k] - yref[k]).abs() < tol, "fmt {:?} k={k} y={} ref={}", w.fmt, y[k], yref[k]);
            }
        }
    }

    #[test]
    fn apply_vec_parallel_matches_serial() {
        // apply_vec runs batched row-chunks on the pool when the matrix is large
        // enough (i·o ≥ 1<<20) and s_n ≥ PAR_MATMUL_MIN. Use 1024×1024 so the parallel
        // path actually engages, and assert it is bit-identical to one whole serial
        // `apply` for every format.
        let (o, i, s_n) = (1024usize, 1024usize, 16usize);
        let mut rng = Lcg(0x1234);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();
        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            let par = w.apply_vec(&xf, s_n);
            let mut xq = vec![0i8; s_n * i];
            let mut sx = vec![0f32; s_n];
            let mut ser = vec![0f32; s_n * o];
            w.apply(&xf, s_n, &mut xq, &mut sx, &mut ser);
            assert!(
                par.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()),
                "apply_vec parallel must match serial for {:?}",
                w.fmt
            );
        }
    }

    #[test]
    fn decode_splits_output_rows_and_stays_bit_identical() {
        // The decode case: ONE activation row. The batch split cannot engage
        // here (s_n = 1 < PAR_MATMUL_MIN), which is exactly why the output-row
        // window exists — before it, every decode matmul ran single-threaded.
        // Same 1024×1024 shape so `i·o ≥ 1<<20` puts us on the new path, and the
        // result must still be bit-identical to one whole serial `apply`.
        let (o, i, s_n) = (1024usize, 1024usize, 1usize);
        let mut rng = Lcg(0xBEEF);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..s_n * i).map(|_| rng.f()).collect();
        for w in [quant_i8(&wf, o, i), quant_i4(&wf, o, i), quant_i4_grouped(&wf, o, i, 16)] {
            assert!(w.row_pitch().is_some(), "{:?} should take the window path", w.fmt);
            let win = w.apply_vec(&xf, s_n);
            let mut xq = vec![0i8; s_n * i];
            let mut sx = vec![0f32; s_n];
            let mut ser = vec![0f32; s_n * o];
            w.apply(&xf, s_n, &mut xq, &mut sx, &mut ser);
            assert!(
                win.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()),
                "output-row window must match serial for {:?}",
                w.fmt
            );
        }
    }

    #[test]
    fn a_window_is_the_matching_slice_of_the_whole_matmul() {
        // Directly pin the window's contract: rows [o0,o1) of the full result,
        // computed from a borrowed slice of the weights, equal the same rows of
        // the whole-matrix answer. This is what makes splitting them sound.
        let (o, i) = (1024usize, 1024usize);
        let mut rng = Lcg(0x5151);
        let wf: Vec<f32> = (0..o * i).map(|_| rng.f()).collect();
        let xf: Vec<f32> = (0..i).map(|_| rng.f()).collect();
        let w = quant_i4(&wf, o, i);
        let mut xq = vec![0i8; i];
        let mut sx = vec![0f32; 1];
        let mut full = vec![0f32; o];
        w.apply(&xf, 1, &mut xq, &mut sx, &mut full);
        for (o0, o1) in [(0usize, 1usize), (0, 333), (333, 700), (700, o), (o - 1, o)] {
            let mut part = vec![0f32; o1 - o0];
            let mut xq2 = vec![0i8; i];
            let mut sx2 = vec![0f32; 1];
            w.apply_window(o0..o1, &xf, ActScratch { xq: &mut xq2, sx: &mut sx2 }, &mut part);
            assert!(
                part.iter().zip(&full[o0..o1]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "window [{o0},{o1}) must equal that slice of the full matmul"
            );
        }
    }
}

/// Per-shape dispatch specialization (`COLI_SHAPE_SPECIALIZE=1`): the runtime
/// pendant to the global SIMD selection. For each `(fmt, o, i)` matmul shape,
/// the first `2 × PROBES` batched calls alternate serial vs parallel dispatch
/// while timing them; afterwards the measured-faster mode is memoized and used
/// unconditionally. Dispatch-level "runtime specialization of hot paths" — no
/// codegen, and bit-identical (both dispatch modes already are).
mod shape_dispatch {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const PROBES: u32 = 4;

    #[derive(Default, Clone, Copy)]
    struct Stat {
        serial_ns: u64,
        serial_n: u32,
        par_ns: u64,
        par_n: u32,
    }

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| matches!(std::env::var("COLI_SHAPE_SPECIALIZE").as_deref(), Ok("1") | Ok("true")))
    }

    /// Shape key → probe statistics.
    type ShapeTable = Mutex<HashMap<(u8, usize, usize), Stat>>;

    fn table() -> &'static ShapeTable {
        static T: OnceLock<ShapeTable> = OnceLock::new();
        T.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Pre-dispatch: overrides `gate` per the memoized (or probing) decision.
    /// Returns `Some(used_par)` while probing (the caller then reports timing
    /// via [`post`]), `None` when disabled or already locked in.
    pub fn pre(fmt: u8, o: usize, i: usize, s_n: usize, gate: &mut usize) -> Option<bool> {
        if !enabled() || s_n == 0 {
            return None;
        }
        let Ok(mut t) = table().lock() else { return None };
        let stat = t.entry((fmt, o, i)).or_default();
        if stat.serial_n >= PROBES && stat.par_n >= PROBES {
            // locked in: pick the faster measured mean (per call)
            let s_mean = stat.serial_ns / stat.serial_n.max(1) as u64;
            let p_mean = stat.par_ns / stat.par_n.max(1) as u64;
            *gate = if p_mean < s_mean { 1 } else { usize::MAX };
            return None;
        }
        // probing: alternate, filling whichever side has fewer samples
        let use_par = stat.par_n <= stat.serial_n;
        *gate = if use_par { 1 } else { usize::MAX };
        Some(use_par)
    }

    /// Post-dispatch: record the probe timing.
    pub fn post(fmt: u8, o: usize, i: usize, used_par: bool, ns: u64) {
        let Ok(mut t) = table().lock() else { return };
        let stat = t.entry((fmt, o, i)).or_default();
        if used_par {
            stat.par_ns = stat.par_ns.saturating_add(ns);
            stat.par_n = stat.par_n.saturating_add(1);
        } else {
            stat.serial_ns = stat.serial_ns.saturating_add(ns);
            stat.serial_n = stat.serial_n.saturating_add(1);
        }
    }
}

/// [`shape_dispatch::pre`] adapter taking the typed format.
fn shape_dispatch_pre(fmt: QuantFmt, o: usize, i: usize, s_n: usize, gate: &mut usize) -> Option<bool> {
    shape_dispatch::pre(fmt as u8, o, i, s_n, gate)
}

/// [`shape_dispatch::post`] adapter taking the typed format.
fn shape_dispatch_post(fmt: QuantFmt, o: usize, i: usize, used_par: bool, ns: u64) {
    shape_dispatch::post(fmt as u8, o, i, used_par, ns);
}

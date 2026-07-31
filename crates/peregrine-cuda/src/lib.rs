//! CUDA GPU lane (M3) — FFI to the validated kernels in `c/backend_cuda.cu`.
//!
//! Behind the `cuda` feature, `build.rs` compiles the `.cu` with nvcc and links
//! cudart, exposing the flat C ABI over an opaque `ColiCudaTensor` handle
//! (`c/backend_cuda.h`). Reusing the proven kernels avoids re-validating GPU
//! math. Without the feature (the default on hosts with no GPU/nvcc), this is a
//! stub reporting the backend unavailable, so the workspace always builds.
//!
//! The GPU lane composes with the M4 scheduler exactly like the CPU/IO lanes:
//! VRAM-resident experts are dispatched via [`expert_group`] while the io_uring
//! lane streams disk experts and the CPU lane computes RAM experts — all at
//! once. That integration is exercised on an NVIDIA box.
//!
//! Quality gates: CUDA FFI is the only (irreducible) `unsafe` here — every block
//! is encapsulated behind a safe API with a `# Safety` note; no panicking error
//! handling.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use peregrine_core::Error;

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_int, c_void};

    /// Opaque device tensor handle — the host never dereferences it.
    #[repr(C)]
    pub struct ColiCudaTensor {
        _private: [u8; 0],
    }

    /// Opaque captured+instantiated CUDA graph handle — never dereferenced on host.
    #[repr(C)]
    pub struct ColiCudaGraph {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn coli_cuda_init(devices: *const c_int, count: c_int) -> c_int;
        pub fn coli_cuda_shutdown();
        pub fn coli_cuda_device_count() -> c_int;
        pub fn coli_cuda_mem_info(device: c_int, free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
        pub fn coli_cuda_tensor_upload(
            tensor: *mut *mut ColiCudaTensor,
            weights: *const c_void,
            scales: *const f32,
            fmt: c_int,
            i: c_int,
            o: c_int,
            device: c_int,
        ) -> c_int;
        pub fn coli_cuda_tensor_free(tensor: *mut ColiCudaTensor);
        pub fn coli_cuda_expert_group(
            gates: *const *mut ColiCudaTensor,
            ups: *const *mut ColiCudaTensor,
            downs: *const *mut ColiCudaTensor,
            rows: *const c_int,
            count: c_int,
            y: *mut f32,
            x: *const f32,
        ) -> c_int;
        pub fn coli_cuda_group_stats(
            calls: *mut u64,
            experts: *mut u64,
            rows: *mut u64,
            h2d_ms: *mut f64,
            kernel_ms: *mut f64,
            d2h_ms: *mut f64,
        );
        pub fn coli_cuda_graph_begin(device: c_int) -> c_int;
        pub fn coli_cuda_graph_end(device: c_int, out: *mut *mut ColiCudaGraph) -> c_int;
        pub fn coli_cuda_graph_launch(g: *mut ColiCudaGraph) -> c_int;
        pub fn coli_cuda_graph_free(g: *mut ColiCudaGraph);
    }

    // Device-pointer pipe primitives — currently exercised only by the graph
    // capture test; un-gate when the resident decode path (full A8) issues them.
    #[cfg(test)]
    extern "C" {
        pub fn coli_cuda_pipe_alloc(device: c_int, bytes: usize) -> *mut c_void;
        pub fn coli_cuda_pipe_free(device: c_int, p: *mut c_void);
        pub fn coli_cuda_pipe_upload(device: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn coli_cuda_pipe_download(device: c_int, src: *const c_void, dst: *mut c_void, bytes: usize) -> c_int;
        pub fn coli_cuda_pipe_silu_mul(device: c_int, gate_dev: *mut f32, up_dev: *const f32, n: usize) -> c_int;
        pub fn coli_cuda_pipe_add(device: c_int, x_dev: *mut f32, t_dev: *const f32, n: usize) -> c_int;
    }
}

/// Cumulative counters for the GPU expert-group lane (process-global, monotonic).
/// `h2d_ms`/`kernel_ms`/`d2h_ms` accumulate only when `COLI_CUDA_PROFILE` is set;
/// they let a profiler show the H2D / kernel / D2H overlap the async path gives.
#[derive(Debug, Clone, Copy, Default)]
pub struct GroupStats {
    pub calls: u64,
    pub experts: u64,
    pub rows: u64,
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
}

/// Number of usable CUDA devices (0 when the backend is not built).
pub fn device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        unsafe { ffi::coli_cuda_device_count() as i32 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Whether the GPU lane can run on this host.
pub fn is_available() -> bool {
    device_count() > 0
}

/// Human-readable backend status for startup logging.
pub fn status() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA backend linked (c/backend_cuda.cu)"
    }
    #[cfg(not(feature = "cuda"))]
    {
        "CUDA backend not built — rebuild with `--features cuda` on an NVIDIA host"
    }
}

/// Initialize the given CUDA devices. Returns the number initialized, or 0 when
/// the backend is not built.
#[cfg(feature = "cuda")]
pub fn init(devices: &[i32]) -> i32 {
    unsafe { ffi::coli_cuda_init(devices.as_ptr(), devices.len() as i32) as i32 }
}
#[cfg(not(feature = "cuda"))]
pub fn init(_devices: &[i32]) -> i32 {
    0
}

/// Release all CUDA device contexts and resources. Safe to call once at teardown;
/// a no-op when the backend is not built.
pub fn shutdown() {
    #[cfg(feature = "cuda")]
    {
        // SAFETY: idempotent teardown of the process-global CUDA state; always safe.
        unsafe { ffi::coli_cuda_shutdown() };
    }
}

#[cfg(feature = "cuda")]
use std::os::raw::{c_int, c_void};

/// Free and total VRAM on `device`, in bytes. Errors if the backend is not built
/// or the query fails.
#[cfg(feature = "cuda")]
pub fn mem_info(device: i32) -> Result<(usize, usize), Error> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both out-pointers are valid; the C call only writes through them and
    // returns 1 on success, 0 on a bad device / null pointer.
    let ok = unsafe { ffi::coli_cuda_mem_info(device as c_int, &mut free, &mut total) };
    if ok == 1 {
        Ok((free, total))
    } else {
        Err(Error::Format(format!("cuda mem_info(device={device}) failed")))
    }
}
#[cfg(not(feature = "cuda"))]
pub fn mem_info(_device: i32) -> Result<(usize, usize), Error> {
    Err(Error::Format("cuda backend not built".into()))
}

/// One MoE expert's gate/up/down weights resident in VRAM as f32 (`fmt=0`).
/// RAII — `Drop` frees the three device tensors. Uploaded once at load, then
/// dispatched many times via [`expert_group`].
#[cfg(feature = "cuda")]
pub struct GpuExpert {
    gate: *mut ffi::ColiCudaTensor,
    up: *mut ffi::ColiCudaTensor,
    down: *mut ffi::ColiCudaTensor,
}

// SAFETY: the fields are opaque device-tensor handles, never dereferenced on the
// host. The CUDA runtime API is process-global and thread-safe, and the compute
// entry (`expert_group`) only reads the tensors, so moving/​sharing a `GpuExpert`
// across the scheduler's worker threads is sound.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuExpert {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuExpert {}

#[cfg(feature = "cuda")]
impl GpuExpert {
    /// Upload one expert's dequantized f32 weights to `device`. `hidden` is the
    /// model dim, `inter` the MoE intermediate dim; `gate`/`up` are `[inter,
    /// hidden]` row-major, `down` is `[hidden, inter]`.
    pub fn upload(
        device: i32,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden: usize,
        inter: usize,
    ) -> Result<GpuExpert, Error> {
        if gate.len() != inter * hidden || up.len() != inter * hidden || down.len() != hidden * inter {
            return Err(Error::Format("gpu expert upload: weight length mismatch".into()));
        }
        // Upload in order; on any failure free the ones already uploaded (no leak).
        let g = upload_tensor(device, gate, hidden, inter)?;
        let u = match upload_tensor(device, up, hidden, inter) {
            Ok(t) => t,
            Err(e) => {
                free_tensor(g);
                return Err(e);
            }
        };
        let d = match upload_tensor(device, down, inter, hidden) {
            Ok(t) => t,
            Err(e) => {
                free_tensor(g);
                free_tensor(u);
                return Err(e);
            }
        };
        Ok(GpuExpert { gate: g, up: u, down: d })
    }

    /// Upload one expert's **per-row int4** (`fmt=2`) weights directly — no dequant,
    /// so it is ~8× denser in VRAM than [`Self::upload`] (18.9 MB vs 151 MB per
    /// GLM-5.2 expert). Each tensor is `(packed_nibbles, per_row_scales)`:
    /// `gate`/`up` are `[inter, hidden]`, `down` is `[hidden, inter]`. Computed by
    /// the GPU int4 grouped-expert kernel (the `all_s4` path); numerics differ from
    /// the f32 tier (int4 on device), guarded by tolerance tests. Only valid for a
    /// per-row-int4 source (grouped int4 / `fmt=4` needs a requantize first).
    pub fn upload_int4(
        device: i32,
        gate: (&[u8], &[f32]),
        up: (&[u8], &[f32]),
        down: (&[u8], &[f32]),
        hidden: usize,
        inter: usize,
    ) -> Result<GpuExpert, Error> {
        let ok = |b: &[u8], s: &[f32], o: usize, i: usize| b.len() == o * i.div_ceil(2) && s.len() == o;
        if !ok(gate.0, gate.1, inter, hidden) || !ok(up.0, up.1, inter, hidden) || !ok(down.0, down.1, hidden, inter) {
            return Err(Error::Format("gpu int4 expert upload: byte/scale length mismatch".into()));
        }
        let g = upload_tensor_i4(device, gate.0, gate.1, hidden, inter)?;
        let u = match upload_tensor_i4(device, up.0, up.1, hidden, inter) {
            Ok(t) => t,
            Err(e) => {
                free_tensor(g);
                return Err(e);
            }
        };
        let d = match upload_tensor_i4(device, down.0, down.1, inter, hidden) {
            Ok(t) => t,
            Err(e) => {
                free_tensor(g);
                free_tensor(u);
                return Err(e);
            }
        };
        Ok(GpuExpert { gate: g, up: u, down: d })
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuExpert {
    fn drop(&mut self) {
        free_tensor(self.gate);
        free_tensor(self.up);
        free_tensor(self.down);
    }
}

/// Upload one f32 weight tensor `[o, i]` as `fmt=0` (no scales).
#[cfg(feature = "cuda")]
fn upload_tensor(device: i32, w: &[f32], i: usize, o: usize) -> Result<*mut ffi::ColiCudaTensor, Error> {
    let mut t: *mut ffi::ColiCudaTensor = std::ptr::null_mut();
    // SAFETY: `t` is a valid out-slot; `w` points to `o*i` valid f32; `fmt=0`
    // permits a null `scales`. The call copies host→device and returns 1 on success.
    let ok = unsafe {
        ffi::coli_cuda_tensor_upload(
            &mut t,
            w.as_ptr() as *const c_void,
            std::ptr::null(),
            0,
            i as c_int,
            o as c_int,
            device as c_int,
        )
    };
    if ok == 1 && !t.is_null() {
        Ok(t)
    } else {
        free_tensor(t);
        Err(Error::Format(format!("cuda tensor_upload failed (i={i}, o={o})")))
    }
}

/// Upload one **per-row int4** (`fmt=2`) tensor `[o, i]`: `o*ceil(i/2)` packed
/// nibble bytes plus `o` row scales.
#[cfg(feature = "cuda")]
fn upload_tensor_i4(device: i32, w: &[u8], scales: &[f32], i: usize, o: usize) -> Result<*mut ffi::ColiCudaTensor, Error> {
    let mut t: *mut ffi::ColiCudaTensor = std::ptr::null_mut();
    // SAFETY: `w` holds o*ceil(i/2) packed bytes and `scales` holds `o` f32; fmt=2
    // (int4) requires non-null scales. The call copies host→device, returns 1 on ok.
    let ok = unsafe {
        ffi::coli_cuda_tensor_upload(
            &mut t,
            w.as_ptr() as *const c_void,
            scales.as_ptr(),
            2,
            i as c_int,
            o as c_int,
            device as c_int,
        )
    };
    if ok == 1 && !t.is_null() {
        Ok(t)
    } else {
        free_tensor(t);
        Err(Error::Format(format!("cuda int4 tensor_upload failed (i={i}, o={o})")))
    }
}

#[cfg(feature = "cuda")]
fn free_tensor(t: *mut ffi::ColiCudaTensor) {
    // SAFETY: `t` was returned by `tensor_upload` or is null; `tensor_free` is NULL-safe.
    unsafe { ffi::coli_cuda_tensor_free(t) };
}

/// Compute a batch of VRAM-resident experts in one call: `experts[k]` processes
/// `rows[k]` consecutive token-rows of `x` (`x` is `[Σrows, hidden]` f32); the
/// full expert SwiGLU runs on the GPU. Returns `y` `[Σrows, hidden]` f32.
#[cfg(feature = "cuda")]
pub fn expert_group(experts: &[&GpuExpert], rows: &[i32], x: &[f32], hidden: usize) -> Result<Vec<f32>, Error> {
    if experts.len() != rows.len() {
        return Err(Error::Format("expert_group: experts/rows length mismatch".into()));
    }
    if rows.iter().any(|&r| r < 0) {
        return Err(Error::Format("expert_group: negative row count".into()));
    }
    let total: usize = rows.iter().map(|&r| r as usize).sum();
    if x.len() != total * hidden {
        return Err(Error::Format("expert_group: x length != sum(rows)*hidden".into()));
    }
    let gates: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.gate).collect();
    let ups: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.up).collect();
    let downs: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.down).collect();
    let mut y = vec![0f32; total * hidden];
    // SAFETY: gates/ups/downs each hold `count` valid handles; `rows` has `count`
    // entries; `x` has Σrows*hidden f32 and `y` the same length. The call blocks
    // until the kernels finish (internal stream sync) and returns 1 on success.
    let ok = unsafe {
        ffi::coli_cuda_expert_group(
            gates.as_ptr(),
            ups.as_ptr(),
            downs.as_ptr(),
            rows.as_ptr(),
            experts.len() as c_int,
            y.as_mut_ptr(),
            x.as_ptr(),
        )
    };
    if ok == 1 {
        Ok(y)
    } else {
        Err(Error::Format("cuda expert_group failed".into()))
    }
}

/// Snapshot the cumulative GPU expert-group counters (see [`GroupStats`]) — how
/// many batched calls/experts/rows ran and, under `COLI_CUDA_PROFILE`, the
/// H2D/kernel/D2H milliseconds. All zero when the backend is not built or no
/// group has run yet. Used by the profiler to show the async lane's overlap.
#[cfg(feature = "cuda")]
pub fn group_stats() -> GroupStats {
    let (mut calls, mut experts, mut rows) = (0u64, 0u64, 0u64);
    let (mut h2d, mut kernel, mut d2h) = (0f64, 0f64, 0f64);
    // SAFETY: all six out-pointers are valid and distinct; the C call only writes
    // scalar counters through them and returns nothing.
    unsafe {
        ffi::coli_cuda_group_stats(&mut calls, &mut experts, &mut rows, &mut h2d, &mut kernel, &mut d2h);
    }
    GroupStats { calls, experts, rows, h2d_ms: h2d, kernel_ms: kernel, d2h_ms: d2h }
}
#[cfg(not(feature = "cuda"))]
pub fn group_stats() -> GroupStats {
    GroupStats::default()
}

/// A captured + instantiated CUDA graph of a device's managed stream. Capture a
/// stable-shape op sequence once with [`capture`], then [`Graph::launch`] replays
/// it, skipping the per-op launch cost — the point of CUDA Graphs for the
/// steady-state decode step. RAII: `Drop` frees the graph.
#[cfg(feature = "cuda")]
pub struct Graph {
    ptr: *mut ffi::ColiCudaGraph,
}

// SAFETY: the handle is an opaque device-side graph, never dereferenced on the
// host; a launch serializes on the owning device's stream.
#[cfg(feature = "cuda")]
unsafe impl Send for Graph {}

#[cfg(feature = "cuda")]
impl Graph {
    /// Replay the captured graph on its device's stream; blocks until it completes.
    pub fn launch(&self) -> Result<(), Error> {
        // SAFETY: `ptr` is a valid instantiated graph returned by `capture`.
        let ok = unsafe { ffi::coli_cuda_graph_launch(self.ptr) };
        if ok == 1 {
            Ok(())
        } else {
            Err(Error::Format("cuda graph launch failed".into()))
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `capture`; `graph_free` is null-safe.
        unsafe { ffi::coli_cuda_graph_free(self.ptr) };
    }
}

/// Capture the ops `body` issues on `device`'s managed stream into a replayable
/// [`Graph`]. `body` must issue only async, same-stream ops (the device-pointer
/// `pipe_*` primitives) — a synchronous copy inside fails the capture. The capture
/// is always ended (even if `body` errors) so the stream stays usable afterward.
#[cfg(feature = "cuda")]
pub fn capture(device: i32, body: impl FnOnce() -> Result<(), Error>) -> Result<Graph, Error> {
    // SAFETY: begin/end bracket a capture on the device's managed stream.
    if unsafe { ffi::coli_cuda_graph_begin(device as c_int) } != 1 {
        return Err(Error::Format("cuda graph begin failed".into()));
    }
    let body_res = body();
    let mut ptr: *mut ffi::ColiCudaGraph = std::ptr::null_mut();
    // SAFETY: `ptr` is a valid out-slot; end_capture writes the instantiated graph.
    let ended = unsafe { ffi::coli_cuda_graph_end(device as c_int, &mut ptr) };
    // Take ownership of whatever the capture produced BEFORE propagating a body
    // error: `graph_end` can succeed even when the body failed, and returning
    // early with a bare pointer would leak the instantiated graph.
    let graph = if ended == 1 && !ptr.is_null() { Some(Graph { ptr }) } else { None };
    body_res?; // surface a body error only after the capture is properly ended
    match graph {
        Some(g) => Ok(g),
        None => Err(Error::Format("cuda graph end/instantiate failed".into())),
    }
}

/// Reference f32 SwiGLU on the CPU, matching what the GPU `expert_group`
/// computes for one expert: `down( silu(x·gateᵀ) ⊙ (x·upᵀ) )`.
#[cfg(all(test, feature = "cuda"))]
fn cpu_swiglu(x: &[f32], gate: &[f32], up: &[f32], down: &[f32], n: usize, hidden: usize, inter: usize) -> Vec<f32> {
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut y = vec![0f32; n * hidden];
    for t in 0..n {
        let xt = &x[t * hidden..t * hidden + hidden];
        let mut h = vec![0f32; inter];
        for o in 0..inter {
            let (gw, uw) = (&gate[o * hidden..o * hidden + hidden], &up[o * hidden..o * hidden + hidden]);
            let g: f32 = xt.iter().zip(gw).map(|(&a, &b)| a * b).sum();
            let u: f32 = xt.iter().zip(uw).map(|(&a, &b)| a * b).sum();
            h[o] = silu(g) * u;
        }
        for d in 0..hidden {
            let dw = &down[d * inter..d * inter + inter];
            y[t * hidden + d] = h.iter().zip(dw).map(|(&a, &b)| a * b).sum();
        }
    }
    y
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    // GPU tests share the device's global scratch (`ctx->x/y/group_desc`), so they
    // must run serially — mirroring the production invariant that one GPU-lane
    // thread issues `expert_group`. `unwrap_or_else` recovers a poisoned lock
    // (a panicking test) without violating the no-`unwrap` gate.
    static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
        GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Per-row int4 quantize `w[o,i]` → (packed nibbles `o*ceil(i/2)`, `o` scales),
    /// matching the model's scheme (scale = amax/7, nibble = clamp(round(v/s),-8,7)+8).
    fn quant_i4(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
        let rb = i.div_ceil(2);
        let mut q = vec![0u8; o * rb];
        let mut sc = vec![0f32; o];
        for oo in 0..o {
            let row = &w[oo * i..oo * i + i];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = (amax / 7.0).max(1e-12);
            sc[oo] = s;
            for ii in 0..i {
                let v = (row[ii] / s).round().clamp(-8.0, 7.0) as i32;
                let bias = (v + 8) as u8 & 0x0F;
                if ii & 1 == 0 {
                    q[oo * rb + (ii >> 1)] |= bias;
                } else {
                    q[oo * rb + (ii >> 1)] |= bias << 4;
                }
            }
        }
        (q, sc)
    }

    /// Inverse of [`quant_i4`]: (nibble − 8) · row scale.
    fn dequant_i4(q: &[u8], sc: &[f32], o: usize, i: usize) -> Vec<f32> {
        let rb = i.div_ceil(2);
        let mut out = vec![0f32; o * i];
        for oo in 0..o {
            let s = sc[oo];
            for ii in 0..i {
                let byte = q[oo * rb + (ii >> 1)];
                let nib = if ii & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                out[oo * i + ii] = (nib - 8) as f32 * s;
            }
        }
        out
    }

    #[test]
    fn int4_expert_group_matches_dequant() -> Result<(), Error> {
        // The int4-resident tier (upload_int4, no dequant) must compute the same
        // SwiGLU as running the dequantized-int4 weights on the CPU — proving the 8×
        // denser VRAM path is correct (up to int4 + kernel tolerance).
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 32usize, 3usize);
        let mut r = Lcg(0x1234);
        let gatef: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let upf: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let downf: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let (gq, gs) = quant_i4(&gatef, inter, hidden);
        let (uq, us) = quant_i4(&upf, inter, hidden);
        let (dq, ds) = quant_i4(&downf, hidden, inter);
        let e = GpuExpert::upload_int4(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter)?;
        let y_gpu = expert_group(&[&e], &[n as i32], &x, hidden)?;

        // reference: dequantize the SAME int4 weights and run f32 SwiGLU on the CPU
        let gd = dequant_i4(&gq, &gs, inter, hidden);
        let ud = dequant_i4(&uq, &us, inter, hidden);
        let dd = dequant_i4(&dq, &ds, hidden, inter);
        let y_cpu = cpu_swiglu(&x, &gd, &ud, &dd, n, hidden, inter);

        assert_eq!(y_gpu.len(), y_cpu.len());
        for k in 0..y_cpu.len() {
            let tol = 3e-2 * y_cpu[k].abs().max(1.0);
            assert!((y_gpu[k] - y_cpu[k]).abs() < tol, "k={k} gpu={} cpu={}", y_gpu[k], y_cpu[k]);
        }
        Ok(())
    }

    #[test]
    fn graph_capture_replays_silu_mul() -> Result<(), Error> {
        // Capture a real device kernel (pipe_silu_mul on the managed stream) into a
        // CUDA graph, then replay it: the output must equal eager silu(gate)*up, and
        // a second replay must reproduce it — proving the instantiated graph re-runs
        // (the A8 capture/replay mechanism; wiring the full decode step is follow-up).
        use std::os::raw::c_void;
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let n = 256usize;
        let mut r = Lcg(0xA8A8);
        let gate: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let up: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let want: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        let nb = n * 4;

        // SAFETY: allocate two n-float device buffers on device 0.
        let (gate_dev, up_dev) =
            unsafe { (ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32, ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32) };
        assert!(!gate_dev.is_null() && !up_dev.is_null(), "device alloc");

        let upload = |dst: *mut f32, src: &[f32]| {
            // SAFETY: `dst` has `nb` bytes; `src` has `n` f32.
            unsafe { ffi::coli_cuda_pipe_upload(0, dst as *mut c_void, src.as_ptr() as *const c_void, nb) };
        };
        let download = |src: *mut f32| -> Vec<f32> {
            let mut out = vec![0f32; n];
            // SAFETY: `src` has `nb` bytes; `out` has `n` f32.
            unsafe { ffi::coli_cuda_pipe_download(0, src as *const c_void, out.as_mut_ptr() as *mut c_void, nb) };
            out
        };

        upload(up_dev, &up);
        upload(gate_dev, &gate);
        let graph = capture(0, || {
            // SAFETY: valid device buffers; runs silu_mul on the managed (captured) stream.
            let ok = unsafe { ffi::coli_cuda_pipe_silu_mul(0, gate_dev, up_dev, n) };
            if ok == 1 {
                Ok(())
            } else {
                Err(Error::Format("silu_mul capture".into()))
            }
        })?;

        graph.launch()?;
        let out1 = download(gate_dev);
        for k in 0..n {
            let tol = 1e-4 * want[k].abs().max(1.0);
            assert!((out1[k] - want[k]).abs() < tol, "launch1 k={k} got={} want={}", out1[k], want[k]);
        }

        upload(gate_dev, &gate); // silu_mul overwrote gate_dev — restore before replay
        graph.launch()?;
        let out2 = download(gate_dev);
        assert_eq!(out1, out2, "graph replay must reproduce the first launch");

        // SAFETY: both pointers came from pipe_alloc; free is null-safe.
        unsafe {
            ffi::coli_cuda_pipe_free(0, gate_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, up_dev as *mut c_void);
        }
        Ok(())
    }

    #[test]
    fn graph_capture_multi_kernel() -> Result<(), Error> {
        // A real decode step is many kernels; capture TWO dependent ops (silu_mul
        // then add) into one graph and confirm replay == eager for the composite —
        // proving the capture mechanism scales past a single kernel (the step toward
        // capturing a full resident decode step; that needs an on-device forward).
        use std::os::raw::c_void;
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let n = 256usize;
        let mut r = Lcg(0xA8FF);
        let gate: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let up: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r.f() * 0.5).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let want: Vec<f32> = (0..n).map(|k| silu(gate[k]) * up[k] + bias[k]).collect();
        let nb = n * 4;

        // SAFETY: three n-float device buffers on device 0.
        let (g_dev, u_dev, b_dev) = unsafe {
            (
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
            )
        };
        assert!(!g_dev.is_null() && !u_dev.is_null() && !b_dev.is_null(), "device alloc");
        let upload = |dst: *mut f32, src: &[f32]| {
            // SAFETY: `dst` has `nb` bytes; `src` has `n` f32.
            unsafe { ffi::coli_cuda_pipe_upload(0, dst as *mut c_void, src.as_ptr() as *const c_void, nb) };
        };
        upload(u_dev, &up);
        upload(b_dev, &bias);
        upload(g_dev, &gate);

        let graph = capture(0, || {
            // SAFETY: g_dev = silu(g_dev)*u_dev, then g_dev += b_dev — two kernels on the stream.
            let a = unsafe { ffi::coli_cuda_pipe_silu_mul(0, g_dev, u_dev, n) };
            let b = unsafe { ffi::coli_cuda_pipe_add(0, g_dev, b_dev, n) };
            if a == 1 && b == 1 {
                Ok(())
            } else {
                Err(Error::Format("multi-kernel capture".into()))
            }
        })?;

        graph.launch()?;
        let mut out = vec![0f32; n];
        // SAFETY: `g_dev` has `nb` bytes; `out` has `n` f32.
        unsafe { ffi::coli_cuda_pipe_download(0, g_dev as *const c_void, out.as_mut_ptr() as *mut c_void, nb) };
        for k in 0..n {
            let tol = 1e-4 * want[k].abs().max(1.0);
            assert!((out[k] - want[k]).abs() < tol, "k={k} got={} want={}", out[k], want[k]);
        }
        // SAFETY: all three came from pipe_alloc; free is null-safe.
        unsafe {
            ffi::coli_cuda_pipe_free(0, g_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, u_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, b_dev as *mut c_void);
        }
        Ok(())
    }

    #[test]
    fn expert_group_matches_cpu_f32() -> Result<(), Error> {
        // Skip gracefully on a box with the feature built but no usable GPU.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 32usize, 3usize);
        let mut r = Lcg(0xC0DA);
        let gate: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let up: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let down: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let e = GpuExpert::upload(0, &gate, &up, &down, hidden, inter)?;
        let y_gpu = expert_group(&[&e], &[n as i32], &x, hidden)?;
        let y_cpu = cpu_swiglu(&x, &gate, &up, &down, n, hidden, inter);

        assert_eq!(y_gpu.len(), y_cpu.len());
        for k in 0..y_cpu.len() {
            let tol = 1e-3 * y_cpu[k].abs().max(1.0);
            assert!((y_gpu[k] - y_cpu[k]).abs() < tol, "k={k} gpu={} cpu={}", y_gpu[k], y_cpu[k]);
        }
        Ok(())
    }

    #[test]
    fn expert_group_handles_over_64_and_counts_stats() -> Result<(), Error> {
        // Regression for the 64→256 cap: a group of >64 experts must dispatch in
        // one call (it returned 0/err before), and group_stats must count it.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, count) = (32usize, 16usize, 70usize); // 70 > the old 64 cap
        let mut r = Lcg(0x6464);
        // one resident expert plus its host weights, kept for the CPU reference check
        struct OwnedExpert {
            gpu: GpuExpert,
            gate: Vec<f32>,
            up: Vec<f32>,
            down: Vec<f32>,
        }
        let mut owned: Vec<OwnedExpert> = Vec::with_capacity(count);
        for _ in 0..count {
            let gate: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
            let up: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
            let down: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
            let gpu = GpuExpert::upload(0, &gate, &up, &down, hidden, inter)?;
            owned.push(OwnedExpert { gpu, gate, up, down });
        }
        let refs: Vec<&GpuExpert> = owned.iter().map(|o| &o.gpu).collect();
        let rows = vec![1i32; count]; // one row per expert
        let x: Vec<f32> = (0..count * hidden).map(|_| r.f()).collect();

        let before = group_stats();
        let y = expert_group(&refs, &rows, &x, hidden)?;
        let after = group_stats();

        assert_eq!(y.len(), count * hidden);
        assert!(after.calls > before.calls, "group_stats must count the call");
        assert!(after.experts >= before.experts + count as u64, "group_stats must count experts");
        // spot-check each expert's row against the CPU reference
        for (c, o) in owned.iter().enumerate() {
            let yc = cpu_swiglu(&x[c * hidden..c * hidden + hidden], &o.gate, &o.up, &o.down, 1, hidden, inter);
            for k in 0..hidden {
                let tol = 1e-3 * yc[k].abs().max(1.0);
                assert!((y[c * hidden + k] - yc[k]).abs() < tol, "c={c} k={k}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_unavailable_without_feature() {
        // On the dev host (no GPU, feature off) the backend is a stub.
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(device_count(), 0);
            assert!(!is_available());
            assert!(status().contains("not built"));
        }
        // With the feature the API exists; availability depends on the box.
        #[cfg(feature = "cuda")]
        {
            assert!(device_count() >= 0);
            assert!(!status().is_empty());
        }
    }
}

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
    //! Declarations only — every signature here mirrors one in `backend_cuda.h`.

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
        pub fn coli_cuda_probe_device_count() -> c_int;
        pub fn coli_cuda_mem_info(device: c_int, free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
        pub fn coli_cuda_largest_free_block(device: c_int, out: *mut usize) -> c_int;
        pub fn coli_cuda_tensor_upload(
            tensor: *mut *mut ColiCudaTensor,
            weights: *const c_void,
            scales: *const f32,
            fmt: c_int,
            i: c_int,
            o: c_int,
            device: c_int,
        ) -> c_int;
        pub fn coli_cuda_tensor_upload_async(
            tensor: *mut *mut ColiCudaTensor,
            weights: *const c_void,
            scales: *const f32,
            fmt: c_int,
            i: c_int,
            o: c_int,
            device: c_int,
        ) -> c_int;
        pub fn coli_cuda_host_register(ptr: *mut c_void, bytes: usize) -> c_int;
        pub fn coli_cuda_host_unregister(ptr: *mut c_void) -> c_int;
        pub fn coli_cuda_stream_sync(device: c_int) -> c_int;
        pub fn coli_cuda_tensor_free(tensor: *mut ColiCudaTensor);
        pub fn coli_cuda_shared_mlp_w4a16(
            gate: *mut ColiCudaTensor,
            up: *mut ColiCudaTensor,
            down: *mut ColiCudaTensor,
            y: *mut f32,
            x: *const f32,
            s: c_int,
        ) -> c_int;
        pub fn coli_cuda_dense_mlp_gemv(
            gate: *mut ColiCudaTensor,
            up: *mut ColiCudaTensor,
            down: *mut ColiCudaTensor,
            y: *mut f32,
            x: *const f32,
        ) -> c_int;
        pub fn coli_cuda_w4_matvec(w: *mut ColiCudaTensor, y: *mut f32, x: *const f32) -> c_int;
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
        pub fn coli_cuda_expert_group_tiled(
            gates: *const *mut ColiCudaTensor,
            ups: *const *mut ColiCudaTensor,
            downs: *const *mut ColiCudaTensor,
            rows: *const c_int,
            count: c_int,
            y: *mut f32,
            x: *const f32,
            tile_m: c_int,
            tile_n: c_int,
            tile_k: c_int,
            arm_out: *mut c_int,
        ) -> c_int;
        pub fn coli_cuda_expert_group_reduce(
            gates: *const *mut ColiCudaTensor,
            ups: *const *mut ColiCudaTensor,
            downs: *const *mut ColiCudaTensor,
            rows: *const c_int,
            count: c_int,
            row_ptr: *const c_int,
            row_idx: *const c_int,
            rw: *const f32,
            s_n: c_int,
            out: *mut f32,
            x: *const f32,
        ) -> c_int;
        pub fn coli_cuda_graph_cache_stats(
            captures: *mut u64,
            replays: *mut u64,
            invalidations: *mut u64,
            uncacheable: *mut u64,
        );
    }

    // Device-pointer pipe primitives — exercised by the graph-capture tests.
    //
    // Every one of these runs on `ctx->stream` (see the ordering note above
    // `coli_cuda_pipe_rmsnorm` in the `.cu`) — which is what makes them
    // capturable, and, more urgently, what makes a chain of them ordered at all:
    // that stream is `cudaStreamNonBlocking`, so an op on the default stream is
    // not synchronized against the rest of the chain.
    //
    // **Still `#[cfg(test)]`, deliberately.** Un-gating so a plain
    // `cargo check --features cuda` type-checks them against the header is
    // worth doing — but only *with* the production caller. `mod ffi` is private,
    // so a `pub fn` in it with no caller is dead code, and un-gating on its own
    // trades one gate for nine `never used` warnings against a repo whose stated
    // bar is zero. `#[allow(dead_code)]` is not the escape: the bad-patterns
    // audit's `[C]` section treats lint suppression as a strict failure, which is
    // the correct call. Un-gate when the device-resident forward issues them.
    #[cfg(test)]
    extern "C" {
        pub fn coli_cuda_pipe_alloc(device: c_int, bytes: usize) -> *mut c_void;
        pub fn coli_cuda_pipe_free(device: c_int, p: *mut c_void);
        pub fn coli_cuda_pipe_upload(device: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn coli_cuda_pipe_download(device: c_int, src: *const c_void, dst: *mut c_void, bytes: usize) -> c_int;
        pub fn coli_cuda_pipe_silu_mul(device: c_int, gate_dev: *mut f32, up_dev: *const f32, n: usize) -> c_int;
        pub fn coli_cuda_pipe_add(device: c_int, x_dev: *mut f32, t_dev: *const f32, n: usize) -> c_int;
        pub fn coli_cuda_pipe_rmsnorm(
            device: c_int,
            y_dev: *mut f32,
            x_dev: *const f32,
            w_dev: *const f32,
            s: c_int,
            d: c_int,
            eps: f32,
        ) -> c_int;
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

/// Devices the CUDA **driver** reports, without initializing any of them.
///
/// Distinct from [`device_count`], which counts the contexts *this process*
/// built. Use this one to decide whether to call [`init`]; use that one to know
/// which device indices are addressable afterwards.
pub fn probe_device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        unsafe { ffi::coli_cuda_probe_device_count() as i32 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Whether the GPU lane can run on this host.
///
/// **This was `device_count() > 0` until 2026-08-07, which made it false on a
/// working GPU.** `device_count` reports initialized contexts, so the answer was
/// 0 until `init` had already run — circular for anything trying to decide
/// *whether* to init, and it reported "unavailable" on an RTX 3060 that the
/// test suite was simultaneously driving. The defect was invisible because the
/// function had no caller outside tests; wiring it into the startup banner is
/// what surfaced it.
pub fn is_available() -> bool {
    probe_device_count() > 0
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

/// Largest single VRAM allocation that currently succeeds on `device`, found by
/// binary-searching real `cudaMalloc` probes (~13 of them on a 12 GB card, 2 MB
/// grain). `free − largest` from [`mem_info`] is the fragmentation the defrag-pool
/// question (`docs/todo.md` §2) turns on. Diagnostic only: every probe is a live
/// allocation, so never call this on a forward path.
#[cfg(feature = "cuda")]
pub fn largest_free_block(device: i32) -> Result<usize, Error> {
    let mut out = 0usize;
    // SAFETY: the out-pointer is valid; the C call only writes through it and
    // returns 1 on success, 0 on a bad device / null pointer / failed query.
    let ok = unsafe { ffi::coli_cuda_largest_free_block(device as c_int, &mut out) };
    if ok == 1 {
        Ok(out)
    } else {
        Err(Error::Format(format!("cuda largest_free_block(device={device}) failed")))
    }
}
#[cfg(not(feature = "cuda"))]
pub fn largest_free_block(_device: i32) -> Result<usize, Error> {
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

    /// Async twin of [`Self::upload_int4`] — the GPU end of the disk→GPU lane.
    ///
    /// Identical bytes and identical device state, but the three H2D copies are
    /// queued on the device's managed stream and **not waited on**, so a batch of
    /// uploads overlaps the host work that produces the next one instead of
    /// stalling on each in turn.
    ///
    /// Two obligations come with that, and both are the caller's:
    ///
    /// 1. Call [`stream_sync`] on `device` before any kernel reads these weights.
    /// 2. Keep `gate`/`up`/`down` alive and unmodified until that sync returns —
    ///    the DMA reads them after this function has already come back.
    ///
    /// The payoff only lands if the byte slices are **pinned** (see
    /// [`pin_host`]). An async copy from pageable memory is legal but the driver
    /// must bounce it through its own staging buffer, which serializes the very
    /// thing this exists to overlap.
    pub fn upload_int4_async(
        device: i32,
        gate: (&[u8], &[f32]),
        up: (&[u8], &[f32]),
        down: (&[u8], &[f32]),
        hidden: usize,
        inter: usize,
    ) -> Result<GpuExpert, Error> {
        let ok = |b: &[u8], s: &[f32], o: usize, i: usize| b.len() == o * i.div_ceil(2) && s.len() == o;
        if !ok(gate.0, gate.1, inter, hidden) || !ok(up.0, up.1, inter, hidden) || !ok(down.0, down.1, hidden, inter) {
            return Err(Error::Format("gpu int4 async expert upload: byte/scale length mismatch".into()));
        }
        let g = upload_tensor_i4_async(device, gate.0, gate.1, hidden, inter)?;
        let u = match upload_tensor_i4_async(device, up.0, up.1, hidden, inter) {
            Ok(t) => t,
            Err(e) => {
                // The failed call left nothing queued, but `g`'s copy may still be
                // in flight; drain before freeing its device memory.
                let _ = stream_sync(device);
                free_tensor(g);
                return Err(e);
            }
        };
        let d = match upload_tensor_i4_async(device, down.0, down.1, inter, hidden) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_sync(device);
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
///
/// # The host and device disagree about what a nibble means
///
/// The bytes go over verbatim, but the two sides decode them differently:
///
/// - **Host** (`QtWeight::dequant_row_into`, `dot_i4i8_avx2`, every CPU kernel):
///   **bias-8**. The nibble is unsigned `0..15` and the value is `nibble - 8`.
/// - **Device** (`w4a16_matmul_t`, `w4_gemv_rows`, every `.cu` kernel):
///   **two's-complement**. The nibble is a signed 4-bit field and the value is
///   `a & 8 ? a - 16 : a`.
///
/// Both map `0..15` onto `-8..7`, but they are DIFFERENT permutations of it, so
/// reading device bytes with the host rule (or vice versa) yields plausible
/// garbage rather than an error. That has now cost two debugging sessions — a
/// gate-layout bug and a GEMV kernel that benchmarked 3x faster while computing
/// nonsense (rms 4.2e-1 against an f32 truth the CPU path hits at 3.0e-3).
/// Neither was caught by a timing harness; both were caught by an
/// accuracy-versus-truth assertion. **A new device kernel must follow the
/// device rule, and must ship with a test that compares against f32 truth.**
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

/// Live pinning state — what the disk→GPU lane actually got, not what it asked
/// for. See [`pin_host`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinStats {
    pub buffers: usize,
    pub bytes: u64,
    pub declined: usize,
}

/// Buffers below this are declined by policy. `cudaHostRegister` walks and pins
/// page tables, far too expensive to pay on a small allocation that will never
/// be an H2D source; the expert slabs this lane exists for are ~18.9 MB.
const MIN_PIN_BYTES: usize = 1 << 20;

static PINNED_BUFS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PINNED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PIN_DECLINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Current pinning counters.
///
/// `declined` is the number that matters when the lane underperforms: a high
/// count with `buffers` near zero means the lane is nominally on and doing
/// nothing, and every upload is quietly bouncing through the driver's own
/// staging buffer.
///
/// Note it is **not** `RLIMIT_MEMLOCK` that refuses these. `cudaHostRegister`
/// pins through the NVIDIA driver, which does its own accounting: measured on
/// this box, it registers 256 MB happily with `ulimit -l` at 8192 KB. That
/// limit binds `IORING_REGISTER_BUFFERS` (see `concurrent.rs`), which is a
/// different mechanism on the same lane — do not chase the wrong knob. Real
/// refusals here mean host memory pressure or a driver that will not pin the
/// range.
pub fn pin_stats() -> PinStats {
    use std::sync::atomic::Ordering::Relaxed;
    PinStats {
        buffers: PINNED_BUFS.load(Relaxed),
        bytes: PINNED_BYTES.load(Relaxed),
        declined: PIN_DECLINED.load(Relaxed),
    }
}

/// Pin an aligned host allocation so io_uring can DMA disk bytes into it and
/// CUDA can then DMA those same bytes to the device, with no copy in between.
/// Returns whether the pin took.
///
/// **This is a `peregrine_io::set_pin_hook` callback and nothing else.** It is a
/// safe `fn` because that hook's type is a safe fn pointer; its actual contract
/// is the hook's contract — `peregrine_io::AlignedBuf` calls it with the base
/// and length of an allocation it has just made, and calls [`unpin_host`] with
/// the same pair from its own `Drop`, while the pages are still mapped. Calling
/// it with anything else is unsound. It lives here rather than in
/// `peregrine-model` because that crate denies `unsafe`, and this is FFI.
///
/// Failure is ordinary rather than exceptional, which is why it returns `bool`
/// instead of an error: a declined buffer costs the pinned lane and nothing
/// else — the pageable upload path still works. See [`pin_stats`] for what does
/// and does not cause a refusal.
pub fn pin_host(ptr: *mut u8, len: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if len < MIN_PIN_BYTES {
        return false; // declined by policy, not by the driver — not counted
    }
    // SAFETY: the hook contract above — a live, still-mapped allocation of
    // exactly `len` bytes at `ptr`, unregistered before it is freed.
    if unsafe { host_register(ptr, len) } {
        PINNED_BUFS.fetch_add(1, Relaxed);
        PINNED_BYTES.fetch_add(len as u64, Relaxed);
        true
    } else {
        PIN_DECLINED.fetch_add(1, Relaxed);
        false
    }
}

/// Undo a [`pin_host`]. Same contract: a `peregrine_io::set_pin_hook` callback,
/// called from `AlignedBuf::drop` while the pages are still mapped.
pub fn unpin_host(ptr: *mut u8, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    // SAFETY: the hook contract — a pointer `pin_host` returned true for, still
    // mapped.
    if unsafe { host_unregister(ptr) } {
        PINNED_BUFS.fetch_sub(1, Relaxed);
        PINNED_BYTES.fetch_sub(len as u64, Relaxed);
    } else {
        eprintln!("[peregrine advisory] cudaHostUnregister failed; pinned pages leaked");
    }
}

/// Pin an existing host allocation with `cudaHostRegister`, so io_uring can DMA
/// disk bytes into it and CUDA can then DMA those same bytes to the device with
/// no copy in between. Returns whether the pin took.
///
/// **Failure is ordinary, not exceptional**, which is why this returns `bool`
/// rather than an `Error`: the caller keeps a pageable buffer and the blocking
/// upload path still works. Host memory pressure and a driver that will not pin
/// the range are the real causes; `RLIMIT_MEMLOCK` is not one of them, despite
/// the folklore — see [`pin_stats`].
///
/// # Safety
/// `ptr` must point to `bytes` of live host memory that stays mapped, and at the
/// same address, until [`host_unregister`] is called on it.
#[cfg(feature = "cuda")]
pub unsafe fn host_register(ptr: *mut u8, bytes: usize) -> bool {
    // SAFETY: forwarded caller contract — a live mapping of `bytes` at `ptr`.
    unsafe { ffi::coli_cuda_host_register(ptr as *mut c_void, bytes) == 1 }
}

/// Undo a [`host_register`]. Must run while the pages are still mapped:
/// unregistering a freed pointer is undefined.
///
/// # Safety
/// `ptr` must be a pointer a previous [`host_register`] returned `true` for, and
/// its memory must still be mapped.
#[cfg(feature = "cuda")]
pub unsafe fn host_unregister(ptr: *mut u8) -> bool {
    // SAFETY: forwarded caller contract — a still-mapped, previously registered
    // pointer.
    unsafe { ffi::coli_cuda_host_unregister(ptr as *mut c_void) == 1 }
}

#[cfg(not(feature = "cuda"))]
/// # Safety
/// Never dereferences `ptr`; the no-CUDA build has nothing to pin.
pub unsafe fn host_register(_ptr: *mut u8, _bytes: usize) -> bool {
    false
}

#[cfg(not(feature = "cuda"))]
/// # Safety
/// Never dereferences `ptr`; the no-CUDA build has nothing to unpin.
pub unsafe fn host_unregister(_ptr: *mut u8) -> bool {
    false
}

/// Drain `device`'s managed stream — the completion boundary for a batch of
/// [`GpuExpert::upload_int4_async`] calls. Until this returns, none of those
/// weights may be read by a kernel and none of their host source buffers may be
/// freed or reused.
#[cfg(feature = "cuda")]
pub fn stream_sync(device: i32) -> Result<(), Error> {
    // SAFETY: plain FFI call taking a device ordinal; validated C-side.
    if unsafe { ffi::coli_cuda_stream_sync(device as c_int) } == 1 {
        Ok(())
    } else {
        Err(Error::Format(format!("cuda stream_sync failed on device {device}")))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn stream_sync(_device: i32) -> Result<(), Error> {
    Ok(())
}

/// Async twin of [`upload_tensor_i4`]: same bytes, same conversion, but issued
/// on the device's managed stream and *not* waited on. The caller owes a
/// [`stream_sync`] before any kernel reads the result and before `w`/`scales`
/// are dropped.
#[cfg(feature = "cuda")]
fn upload_tensor_i4_async(
    device: i32,
    w: &[u8],
    scales: &[f32],
    i: usize,
    o: usize,
) -> Result<*mut ffi::ColiCudaTensor, Error> {
    let mut t: *mut ffi::ColiCudaTensor = std::ptr::null_mut();
    // SAFETY: `w` holds o*ceil(i/2) packed bytes and `scales` holds `o` f32;
    // fmt=2 requires non-null scales. The copies are queued on the device's
    // stream and complete no later than the caller's `stream_sync`, which the
    // caller must issue while `w` and `scales` are still alive.
    let ok = unsafe {
        ffi::coli_cuda_tensor_upload_async(
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
        Err(Error::Format(format!("cuda int4 tensor_upload_async failed (i={i}, o={o})")))
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
    // Through the untiled C entry, not `expert_group_tiled(.., None)`. The two
    // are the same code path on the C side, and routing everything through the
    // tiled one would be tidier — but it would also leave the default,
    // overwhelmingly common call reaching the kernels via an argument list that
    // only the autotuner exercises. Keeping the historical entry point in use
    // means the untuned path is the one that has always been running.
    expert_group_dispatch(experts, rows, x, hidden, None, None)
}

/// Run one **dense** SwiGLU MLP entirely on the device: `down(silu(gate·x) ⊙
/// (up·x))` with int4 weights and f32 activations (`w4a16`), for `s_n` rows of
/// `x[s_n, hidden]`, returning `[s_n, hidden]`.
///
/// The C kernel this binds (`coli_cuda_shared_mlp_w4a16`) was written for GLM's
/// *shared* expert and then never called from Rust — it had no binding at all
/// until Track D. A dense transformer layer's MLP is the same shape, which is
/// what makes a resident dense model (Qwen3.8: 64 MLPs, 8.57 GB at int4) a GPU
/// workload with no new kernel.
///
/// Requires the tensors to have been uploaded as **per-row int4**
/// ([`GpuExpert::upload_int4`], `fmt=2`); the C entry validates the format and
/// the shape triple and returns failure rather than computing something else.
/// A failure here is reported, never fatal: the caller falls back to the CPU
/// MLP for that layer, which is the same result by a slower road.
/// One **per-row int4** weight resident on the device: the single-matrix analogue
/// of [`GpuExpert`], for the projections a dense layer applies one at a time
/// (attention `q/k/v/o`, GDN `in_proj_*`/`out_proj`, `lm_head`). Together those
/// are the per-token weight bytes the MLP triple does not cover.
#[cfg(feature = "cuda")]
pub struct GpuMatrix {
    t: *mut ffi::ColiCudaTensor,
    o: usize,
    i: usize,
}

// SAFETY: as for `GpuExpert` — an opaque device handle, never dereferenced on the
// host, read-only in the compute entry.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuMatrix {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuMatrix {}

#[cfg(feature = "cuda")]
impl GpuMatrix {
    /// Upload a per-row int4 weight `[o, i]`: `o*ceil(i/2)` packed bytes plus `o`
    /// row scales — the same encoding [`GpuExpert::upload_int4`] takes, and the
    /// same host/device nibble asymmetry applies (see `upload_tensor_i4`).
    pub fn upload_int4(device: i32, packed: &[u8], scales: &[f32], o: usize, i: usize) -> Result<GpuMatrix, Error> {
        if packed.len() != o * i.div_ceil(2) || scales.len() != o {
            return Err(Error::Format(format!(
                "gpu matrix upload: {} bytes / {} scales for [{o},{i}]",
                packed.len(),
                scales.len()
            )));
        }
        Ok(GpuMatrix { t: upload_tensor_i4(device, packed, scales, i, o)?, o, i })
    }

    /// Device bytes this matrix holds — what a VRAM budget is spent in.
    pub fn bytes(&self) -> usize {
        self.o * self.i.div_ceil(2) + self.o * std::mem::size_of::<f32>()
    }

    /// `y[o] = W · x[i]` for a single activation row, in GEMV form (see
    /// [`dense_mlp_w4a16`] for why decode does not want WMMA here).
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        if x.len() != self.i {
            return Err(Error::Format(format!("w4 matvec: x is {} floats, expected {}", x.len(), self.i)));
        }
        let mut y = vec![0f32; self.o];
        // SAFETY: `self.t` is a live device tensor owned by this `GpuMatrix`
        // (freed only in `Drop`); `x`/`y` are host buffers of exactly the lengths
        // the C entry reads and writes, checked above.
        let ok = unsafe { ffi::coli_cuda_w4_matvec(self.t, y.as_mut_ptr(), x.as_ptr()) };
        if ok == 0 {
            return Err(Error::Format("w4 matvec: device call failed (format/shape/launch)".into()));
        }
        Ok(y)
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuMatrix {
    fn drop(&mut self) {
        free_tensor(self.t);
    }
}

/// Decode (`s_n == 1`) takes the GEMV entry instead: a WMMA fragment is 16 rows
/// wide, so at one activation row the `w4a16` kernel computes 15 idle rows for
/// every real one — measured at 3.85 ms on Qwen's 17408×5120 SwiGLU against a
/// 0.37 ms bandwidth floor. No fragment shape fixes that (every WMMA shape
/// wastes M when M=1), so the fix is shape, not tuning. Set `COLI_CUDA_GEMV=0`
/// to force the WMMA path for an A/B; batched shapes never take GEMV, because
/// there the fragments are full and WMMA is the right kernel.
#[cfg(feature = "cuda")]
fn gemv_enabled() -> bool {
    !matches!(std::env::var("COLI_CUDA_GEMV").as_deref(), Ok("0") | Ok("false"))
}

#[cfg(feature = "cuda")]
pub fn dense_mlp_w4a16(e: &GpuExpert, x: &[f32], s_n: usize, hidden: usize) -> Result<Vec<f32>, Error> {
    if s_n == 1 && gemv_enabled() {
        let mut y = vec![0f32; hidden];
        if x.len() != hidden {
            return Err(Error::Format(format!("dense_mlp_gemv: x is {} floats, expected {hidden}", x.len())));
        }
        // SAFETY: same contract as the w4a16 call below — `e`'s handles are live
        // device tensors owned by `e`, and `x`/`y` are host buffers sized exactly
        // as the C entry reads and writes them (one row of `hidden`), checked here.
        let ok = unsafe { ffi::coli_cuda_dense_mlp_gemv(e.gate, e.up, e.down, y.as_mut_ptr(), x.as_ptr()) };
        if ok == 0 {
            return Err(Error::Format("dense_mlp_gemv: device MLP failed (format/shape/launch)".into()));
        }
        return Ok(y);
    }
    if s_n == 0 || x.len() != s_n * hidden {
        return Err(Error::Format(format!(
            "dense_mlp_w4a16: x is {} floats, expected {s_n} x {hidden}",
            x.len()
        )));
    }
    let mut y = vec![0f32; s_n * hidden];
    // SAFETY: `e`'s handles are live device tensors owned by `e` (freed only in
    // its `Drop`); `x`/`y` are host buffers sized exactly as the C entry reads
    // and writes them, checked above; `s_n` fits `c_int` for any batch this
    // engine assembles.
    let ok = unsafe {
        ffi::coli_cuda_shared_mlp_w4a16(
            e.gate,
            e.up,
            e.down,
            y.as_mut_ptr(),
            x.as_ptr(),
            s_n as std::os::raw::c_int,
        )
    };
    if ok == 0 {
        return Err(Error::Format("dense_mlp_w4a16: device MLP failed (format/shape/launch)".into()));
    }
    Ok(y)
}

/// Which kernel arm an `expert_group` call took.
///
/// Reported rather than re-derived: only [`GroupArm::W4A16`] consults the tile,
/// so a tuner that guessed the arm from the same environment variables the
/// backend reads would be a second copy of that decision — and a tuner recording
/// a "winning tile" from a run that never took the tiled arm is recording noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupArm {
    /// int4 Tensor Core (`COLI_CUDA_TC_INT4`). One legal WMMA shape, 8×8×32.
    Int4Tc,
    /// fp16 Tensor Core (`COLI_CUDA_TC_W4A16`) — **the only tile-sensitive arm**.
    W4A16,
    /// Packed-W4 scalar kernels.
    PackedW4,
    /// Generic per-format scalar kernels.
    Generic,
}

impl GroupArm {
    /// Gated with its only caller (`expert_group_tiled`): the enum is public and
    /// useful to match on either way, but nothing decodes a C arm code in a
    /// build with no C side, and an ungated private helper warns there.
    #[cfg(feature = "cuda")]
    fn from_c(v: i32) -> GroupArm {
        match v {
            0 => GroupArm::Int4Tc,
            1 => GroupArm::W4A16,
            2 => GroupArm::PackedW4,
            _ => GroupArm::Generic,
        }
    }
}

/// [`expert_group`] with an explicit WMMA fragment shape for the W4A16 Tensor
/// Core arm, returning the output **and the arm that produced it**. `None` is
/// the 16×16×16 default, and is what `expert_group` passes.
#[cfg(feature = "cuda")]
pub fn expert_group_tiled(
    experts: &[&GpuExpert],
    rows: &[i32],
    x: &[f32],
    hidden: usize,
    tile: Option<(u16, u16, u16)>,
) -> Result<(Vec<f32>, GroupArm), Error> {
    let mut arm: c_int = -1;
    let y = expert_group_dispatch(experts, rows, x, hidden, tile, Some(&mut arm))?;
    Ok((y, GroupArm::from_c(arm as i32)))
}

#[cfg(feature = "cuda")]
fn expert_group_dispatch(
    experts: &[&GpuExpert],
    rows: &[i32],
    x: &[f32],
    hidden: usize,
    tile: Option<(u16, u16, u16)>,
    arm_out: Option<&mut c_int>,
) -> Result<Vec<f32>, Error> {
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
    // SAFETY (both arms): gates/ups/downs each hold `count` valid handles; `rows`
    // has `count` entries; `x` has Σrows*hidden f32 and `y` the same length. The
    // call blocks until the kernels finish (internal stream sync) and returns 1.
    // Which C entry, keyed on whether the caller wants the arm reported — NOT on
    // whether it named a tile.
    //
    // The untiled entry has no `arm_out` parameter, so it cannot answer that
    // question: `arm` stays at its `-1` sentinel, which `GroupArm::from_c` maps
    // to `Generic`. Keying this match on `tile` alone (as it did until
    // 2026-08-08) therefore made `expert_group_tiled(.., None)` report `Generic`
    // no matter which arm ran — and `None` is exactly what `gpu.rs` passes when
    // the tuner selects an int4 tile, so the int4-arm observation its `match arm`
    // deliberately keeps was dropped every time, and dropped permanently once
    // `Int4Tc` became the recorded best for a shape.
    //
    // `{0,0,0}` is the C side's "default tile", so asking for the arm costs
    // nothing beyond an argument list. `expert_group` passes both `None`s and so
    // still reaches the kernels the historical way; see its own note.
    let ok = match (tile, arm_out) {
        (None, None) => unsafe {
            ffi::coli_cuda_expert_group(
                gates.as_ptr(),
                ups.as_ptr(),
                downs.as_ptr(),
                rows.as_ptr(),
                experts.len() as c_int,
                y.as_mut_ptr(),
                x.as_ptr(),
            )
        },
        (tile, arm_out) => unsafe {
            let (tm, tn, tk) = tile.unwrap_or((0, 0, 0));
            ffi::coli_cuda_expert_group_tiled(
                gates.as_ptr(),
                ups.as_ptr(),
                downs.as_ptr(),
                rows.as_ptr(),
                experts.len() as c_int,
                y.as_mut_ptr(),
                x.as_ptr(),
                tm as c_int,
                tn as c_int,
                tk as c_int,
                arm_out.map_or(std::ptr::null_mut(), |a| a as *mut c_int),
            )
        },
    };
    if ok == 1 {
        Ok(y)
    } else {
        Err(Error::Format("cuda expert_group failed".into()))
    }
}

/// The CSR layout [`expert_group_reduce`] accumulates through, built once from
/// a per-y-row destination and weight.
///
/// Split out and made public so the *ordering* — the thing that fixes the
/// result bit-for-bit — is testable without a GPU. [`Self::build`] is the only
/// place that decides it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReduceLayout {
    /// `[s_n + 1]`, ascending; `row_ptr[s_n] == total`.
    pub row_ptr: Vec<i32>,
    /// `[total]`, the y-row indices contributing to each output row, **ascending
    /// within each row** — which is what makes the device sum deterministic.
    pub row_idx: Vec<i32>,
}

impl ReduceLayout {
    /// Build the CSR from `dst[k]` = which output row y-row `k` contributes to.
    ///
    /// A counting sort, not a sort-by-key: it visits `k` in ascending order and
    /// appends, so each output row's contribution list comes out ascending in
    /// `k` — and `k` is batch-union (`pos`) order, so the device sums experts in
    /// the same order the host reduce does. Getting this wrong would not fail
    /// anything; it would just quietly change the low bits.
    ///
    /// `None` when any `dst` is out of range, rather than dropping the row: a
    /// silently discarded contribution is a token computed from fewer experts
    /// than the router selected.
    pub fn build(dst: &[usize], s_n: usize) -> Option<ReduceLayout> {
        if dst.iter().any(|&s| s >= s_n) {
            return None;
        }
        let mut counts = vec![0i32; s_n + 1];
        for &s in dst {
            counts[s + 1] += 1;
        }
        for s in 0..s_n {
            counts[s + 1] += counts[s];
        }
        let row_ptr = counts.clone();
        let mut fill = counts;
        let mut row_idx = vec![0i32; dst.len()];
        for (k, &s) in dst.iter().enumerate() {
            let at = fill[s] as usize;
            row_idx[at] = i32::try_from(k).ok()?;
            fill[s] += 1;
        }
        Some(ReduceLayout { row_ptr, row_idx })
    }
}

/// [`expert_group`] with the layer-level gate-weighted reduce fused on the
/// device: returns `[s_n, hidden]` instead of `[Σrows, hidden]`.
///
/// `dst[k]` is the batch row y-row `k` contributes to and `rw[k]` its router
/// weight. The D2H shrinks from `Σrows` rows to `s_n` — at a saturated batch
/// that is the expert-per-row factor, ~5× on the measured GLM-5.2 unions at
/// B=16, and exactly 1× at B=1, which is why this is a knob and not a default.
///
/// **The summation order changes** relative to running the host reduce over the
/// same experts (GPU contributions now accumulate among themselves before
/// meeting the CPU lane's), so this is not bit-identical to
/// [`expert_group`] plus a host reduce. It *is* stable run to run — see
/// `grouped_reduce` in the `.cu` for why there are no atomics.
#[cfg(feature = "cuda")]
pub fn expert_group_reduce(
    experts: &[&GpuExpert],
    rows: &[i32],
    x: &[f32],
    hidden: usize,
    layout: &ReduceLayout,
    rw: &[f32],
    s_n: usize,
) -> Result<Vec<f32>, Error> {
    if experts.len() != rows.len() {
        return Err(Error::Format("expert_group_reduce: experts/rows length mismatch".into()));
    }
    if rows.iter().any(|&r| r < 0) {
        return Err(Error::Format("expert_group_reduce: negative row count".into()));
    }
    let total: usize = rows.iter().map(|&r| r as usize).sum();
    if x.len() != total * hidden {
        return Err(Error::Format("expert_group_reduce: x length != sum(rows)*hidden".into()));
    }
    // Every one of these would index out of bounds inside the kernel, where the
    // failure is a wrong number or a fault rather than an error return.
    if rw.len() != total || layout.row_idx.len() != total {
        return Err(Error::Format("expert_group_reduce: weights/row_idx length != sum(rows)".into()));
    }
    if s_n == 0 || layout.row_ptr.len() != s_n + 1 {
        return Err(Error::Format("expert_group_reduce: row_ptr length != s_n + 1".into()));
    }
    if layout.row_ptr.last() != Some(&(total as i32)) || layout.row_ptr.first() != Some(&0) {
        return Err(Error::Format("expert_group_reduce: row_ptr does not span [0, sum(rows)]".into()));
    }
    if layout.row_idx.iter().any(|&k| k < 0 || k as usize >= total) {
        return Err(Error::Format("expert_group_reduce: row_idx out of range".into()));
    }
    let gates: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.gate).collect();
    let ups: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.up).collect();
    let downs: Vec<*mut ffi::ColiCudaTensor> = experts.iter().map(|e| e.down).collect();
    let mut out = vec![0f32; s_n * hidden];
    // SAFETY: lengths are all checked above against `total`/`s_n`; the call
    // blocks until the kernels finish and returns 1 on success.
    let ok = unsafe {
        ffi::coli_cuda_expert_group_reduce(
            gates.as_ptr(),
            ups.as_ptr(),
            downs.as_ptr(),
            rows.as_ptr(),
            experts.len() as c_int,
            layout.row_ptr.as_ptr(),
            layout.row_idx.as_ptr(),
            rw.as_ptr(),
            s_n as c_int,
            out.as_mut_ptr(),
            x.as_ptr(),
        )
    };
    if ok == 1 {
        Ok(out)
    } else {
        Err(Error::Format("cuda expert_group_reduce failed".into()))
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

/// Cumulative counters for the expert-group graph cache (`COLI_CUDA_GRAPH=1`).
///
/// These exist because the cache's failure mode is *silent underperformance*,
/// not an error. A launch-shape key that churns — a batch whose per-expert row
/// counts differ every tick, say — captures a new graph on every call and is
/// strictly slower than the eager path, while every output stays correct and
/// every test passes. `replays` staying near zero while `captures` tracks
/// `GroupStats::calls` is what that looks like from outside.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphCacheStats {
    /// Launch shapes recorded (one per new shape, or per shape after an
    /// invalidation).
    pub captures: u64,
    /// Calls served by replaying an already-captured graph — the win.
    pub replays: u64,
    /// Cached graphs discarded because a scratch buffer was reallocated under
    /// them. Persistently nonzero means residency or batch size is still moving.
    pub invalidations: u64,
    /// Calls that fell through to the eager path with the knob on: the W4A16
    /// arm, `COLI_CUDA_PROFILE`, or `COLI_CUDA_ASYNC=0`.
    pub uncacheable: u64,
}

/// Snapshot the [`GraphCacheStats`]. All zero when the backend is not built.
#[cfg(feature = "cuda")]
pub fn graph_cache_stats() -> GraphCacheStats {
    let (mut captures, mut replays, mut invalidations, mut uncacheable) = (0u64, 0u64, 0u64, 0u64);
    // SAFETY: four valid, distinct out-pointers; the C call only writes scalars.
    unsafe {
        ffi::coli_cuda_graph_cache_stats(&mut captures, &mut replays, &mut invalidations, &mut uncacheable);
    }
    GraphCacheStats { captures, replays, invalidations, uncacheable }
}
#[cfg(not(feature = "cuda"))]
pub fn graph_cache_stats() -> GraphCacheStats {
    GraphCacheStats::default()
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

    /// Track D's first equivalence check: the device dense-MLP path must agree
    /// with the CPU SwiGLU it replaces, on the same int4 weights.
    ///
    /// Tolerance, not bit-identity, and deliberately so: the GPU reduces in a
    /// different order (WMMA fragments) than the CPU's row loop, so the two
    /// cannot be bit-equal and a test demanding that would be testing the wrong
    /// property. What must hold is that the kernel computes *this* MLP —
    /// dequantizing the same nibbles against the same per-row scales — which a
    /// relative-error bound catches while wrong-layout or wrong-scale bugs
    /// (the failure mode that cost a night on the CPU side) blow straight
    /// through it.
    #[test]
    fn the_device_dense_mlp_agrees_with_the_cpu_swiglu() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, s_n) = (256usize, 512usize, 3usize);
        let mut r = Lcg(0x5EED);
        let gatef: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let upf: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let downf: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let (gq, gs) = quant_i4(&gatef, inter, hidden);
        let (uq, us) = quant_i4(&upf, inter, hidden);
        let (dq, ds) = quant_i4(&downf, hidden, inter);
        let e = GpuExpert::upload_int4(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter)?;
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();

        let got = dense_mlp_w4a16(&e, &x, s_n, hidden)?;
        assert_eq!(got.len(), s_n * hidden);

        // CPU reference over the DEQUANTIZED weights — the same values the
        // device holds, so this compares kernels rather than quantizers.
        let deq = |q: &[u8], sc: &[f32], o: usize, i: usize| -> Vec<f32> {
            let mut w = vec![0f32; o * i];
            for oo in 0..o {
                for ii in 0..i {
                    let byte = q[oo * i.div_ceil(2) + ii / 2];
                    let nib = if ii % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    w[oo * i + ii] = (nib as f32 - 8.0) * sc[oo];
                }
            }
            w
        };
        let (gw, uw, dw) =
            (deq(&gq, &gs, inter, hidden), deq(&uq, &us, inter, hidden), deq(&dq, &ds, hidden, inter));
        let mut want = vec![0f32; s_n * hidden];
        for srow in 0..s_n {
            let xr = &x[srow * hidden..(srow + 1) * hidden];
            let mut h = vec![0f32; inter];
            for j in 0..inter {
                let g: f32 = (0..hidden).map(|k| gw[j * hidden + k] * xr[k]).sum();
                let u: f32 = (0..hidden).map(|k| uw[j * hidden + k] * xr[k]).sum();
                h[j] = (g / (1.0 + (-g).exp())) * u; // silu(g) * u
            }
            for o in 0..hidden {
                want[srow * hidden + o] = (0..inter).map(|j| dw[o * inter + j] * h[j]).sum();
            }
        }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("dense-mlp gpu-vs-cpu: worst abs {worst:.3e} on scale {scale:.3e}");
        assert!(
            worst / scale < 5e-3,
            "device dense MLP must match the CPU SwiGLU (worst {worst:.3e}, scale {scale:.3e})"
        );
        Ok(())
    }

    /// The single-weight GEMV — the form the attention/GDN projections and
    /// `lm_head` need — must compute the same matvec the CPU does.
    ///
    /// This test exists because its kernel is the one that shipped WRONG once:
    /// the first `w4_gemv_rows` decoded nibbles with the HOST's bias-8 rule while
    /// the device holds two's-complement (`upload` re-encodes, see
    /// `upload_tensor_i4`). It benchmarked 3x faster than the kernel it replaced
    /// and computed nonsense, and no timing harness could tell. A GEMV against
    /// a dequantized CPU reference catches exactly that.
    #[test]
    fn the_device_matvec_agrees_with_the_cpu_matvec() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        // A non-square shape, so a transposed index would not accidentally pass.
        let (o, i) = (384usize, 256usize);
        let mut r = Lcg(0xA11CE);
        let wf: Vec<f32> = (0..o * i).map(|_| r.f() * 0.1).collect();
        let (wq, ws) = quant_i4(&wf, o, i);
        let m = GpuMatrix::upload_int4(0, &wq, &ws, o, i)?;
        let x: Vec<f32> = (0..i).map(|_| r.f()).collect();
        let got = m.matvec(&x)?;
        assert_eq!(got.len(), o);

        // Reference over the DEQUANTIZED host weights (bias-8, the host rule),
        // so this compares kernels rather than quantizers.
        let mut want = vec![0f32; o];
        for oo in 0..o {
            let mut acc = 0f32;
            for ii in 0..i {
                let byte = wq[oo * i.div_ceil(2) + ii / 2];
                let nib = if ii % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                acc += (nib as f32 - 8.0) * ws[oo] * x[ii];
            }
            want[oo] = acc;
        }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("w4 matvec gpu-vs-cpu: worst abs {worst:.3e} on scale {scale:.3e}");
        assert!(worst / scale < 5e-3, "device matvec must match the CPU matvec (worst {worst:.3e}, scale {scale:.3e})");
        // `bytes()` is what a VRAM budget is spent in, so it must not drift from
        // the upload's own length checks.
        assert_eq!(m.bytes(), o * i.div_ceil(2) + o * 4);
        Ok(())
    }

    /// The measurement `docs/todo.md` §2's defrag-pool item asked for instead of the
    /// pool. The engine's VRAM workload is exactly two block sizes (`int4_bytes`,
    /// `f32_bytes`); this reproduces the worst churn `reheat`'s precision ladder
    /// can produce — interleaved frees with every gap refilled at the *other*
    /// format — and then asks the allocator whether free VRAM is still reachable
    /// as one block. The bar is half: a fragmenting allocator collapses
    /// `largest` to the ~24 MB block size, three orders below it, while a
    /// coalescing one keeps `largest ≈ free` minus runtime headroom. If this
    /// ever fails, the `cudaMallocAsync` pool earns its build; until then the
    /// item stays closed on this number.
    #[test]
    fn vram_churn_of_the_two_expert_block_sizes_leaves_free_memory_in_one_block() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let base = largest_free_block(0)?;
        let (free0, _) = mem_info(0)?;
        assert!(base > 0 && base <= free0, "probe sanity: 0 < largest ({base}) <= free ({free0})");

        // ~8 MB per f32 tensor triple at this shape; int4 is ~16× smaller.
        let (hidden, inter) = (1024usize, 2048usize);
        let mut r = Lcg(0xC0FFEE);
        let gatef: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let upf: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let downf: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let (gq, gs) = quant_i4(&gatef, inter, hidden);
        let (uq, us) = quant_i4(&upf, inter, hidden);
        let (dq, ds) = quant_i4(&downf, hidden, inter);

        let up_f32 = |_: usize| GpuExpert::upload(0, &gatef, &upf, &downf, hidden, inter);
        let up_int4 =
            |_: usize| GpuExpert::upload_int4(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter);

        for round in 0..3usize {
            // Alternate formats across 16 slots, drop every other slot, refill
            // each gap at the other format: an f32-sized hole receives an
            // int4-sized tenant and vice versa, which is the only way this
            // workload can fragment at all.
            let mut slots: Vec<Option<GpuExpert>> = Vec::new();
            for i in 0..16usize {
                let flip = (i + round) % 2 == 0;
                slots.push(Some(if flip { up_f32(i)? } else { up_int4(i)? }));
            }
            for (i, s) in slots.iter_mut().enumerate() {
                if i % 2 == 0 {
                    *s = None;
                }
            }
            for (i, s) in slots.iter_mut().enumerate() {
                if s.is_none() {
                    let flip = (i + round) % 2 == 0;
                    *s = Some(if flip { up_int4(i)? } else { up_f32(i)? });
                }
            }
        }

        let (free1, _) = mem_info(0)?;
        let largest = largest_free_block(0)?;
        println!("vram-frag probe: free {free1} B, largest block {largest} B ({:.1}%)",
                 largest as f64 / free1 as f64 * 100.0);
        assert!(
            largest.saturating_mul(2) >= free1,
            "churning the two expert block sizes fragmented VRAM: largest allocatable block \
             {largest} B is under half of free {free1} B — the defrag-pool item just reopened"
        );
        Ok(())
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

    /// What `key` is set to right now, if anything.
    ///
    /// A function rather than `std::env::var(key).ok()` at each call site: as a
    /// tail expression the `.ok()` is a conversion, whereas `let p = ….ok();` is
    /// the shape the bad-patterns audit flags as a discarded `Result` — and the
    /// audit is right to be blunt about it, so the fix is one place that reads
    /// unambiguously rather than three that each need arguing about.
    fn env_snapshot(key: &str) -> Option<String> {
        std::env::var(key).ok()
    }

    /// Set `key` for the duration of a closure and restore it after, so one
    /// test's knob never leaks into the next. Safe against the other GPU tests
    /// because every one of them holds `gpu_guard()`.
    fn with_env<T>(key: &str, val: &str, f: impl FnOnce() -> T) -> T {
        let prev = env_snapshot(key);
        std::env::set_var(key, val);
        let out = f();
        match prev {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        out
    }

    /// [`with_env`] specialized to the graph knob, which most of these tests
    /// toggle.
    fn with_graph_knob<T>(on: bool, f: impl FnOnce() -> T) -> T {
        with_env("COLI_CUDA_GRAPH", if on { "1" } else { "0" }, f)
    }

    /// One int4 expert with reproducible weights, for the graph-cache tests.
    fn int4_expert(seed: u64, hidden: usize, inter: usize) -> Result<GpuExpert, Error> {
        let mut r = Lcg(seed);
        let gatef: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let upf: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let downf: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let (gq, gs) = quant_i4(&gatef, inter, hidden);
        let (uq, us) = quant_i4(&upf, inter, hidden);
        let (dq, ds) = quant_i4(&downf, hidden, inter);
        GpuExpert::upload_int4(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter)
    }

    #[test]
    fn graph_cached_expert_group_matches_eager() -> Result<(), Error> {
        // The cache replays the *same* kernels with the *same* arguments in the
        // *same* order, so it must be bit-identical to eager — not merely close.
        // A tolerance here would hide precisely the bug this is guarding: a
        // replay reading a scratch buffer that has moved.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 32usize, 2usize);
        let e = int4_expert(0x9001, hidden, inter)?;
        let mut r = Lcg(0x51DE);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let eager = with_graph_knob(false, || expert_group(&[&e], &[n as i32], &x, hidden))?;
        // First graphed call captures; the second must replay the same graph.
        let before = graph_cache_stats();
        let captured = with_graph_knob(true, || expert_group(&[&e], &[n as i32], &x, hidden))?;
        let replayed = with_graph_knob(true, || expert_group(&[&e], &[n as i32], &x, hidden))?;
        let after = graph_cache_stats();

        assert_eq!(eager, captured, "the captured graph must reproduce the eager result exactly");
        assert_eq!(eager, replayed, "a replay must reproduce the eager result exactly");
        assert!(after.captures > before.captures, "the first graphed call must capture");
        assert!(
            after.replays > before.replays,
            "the second call at the same shape must REPLAY, not re-capture — a shape key that \
             churns makes this feature slower than not having it"
        );
        Ok(())
    }

    #[test]
    fn a_grown_scratch_buffer_invalidates_cached_graphs() -> Result<(), Error> {
        // **The one silent failure in this design.** `reserve` frees before it
        // reallocates, so a larger call moves `ctx->x`/`ctx->y` and every graph
        // captured against the old addresses is left pointing at freed VRAM.
        // Nothing errors: the allocator hands those pages out again and the
        // replay reads whatever landed there.
        //
        // Small shape (captures) → large shape (forces the realloc) → small
        // shape again (must NOT replay the stale graph).
        //
        // **The order below is load-bearing, and this test had it wrong from the
        // day it was written until 2026-08-08.** `reserve` is grow-only, so
        // taking the large shape's eager reference *first* sized the scratch for
        // it; the graphed large call then found `*cap >= bytes`, returned without
        // freeing anything, `note_realloc` never ran, the generation never moved
        // and the invalidation this asserts never happened. It went unnoticed
        // because the GPU was unavailable on the dev box, so the whole test
        // self-skipped and reported `ok`. The large reference is therefore taken
        // *after* the graphed sequence: eager recomputes from the weights and
        // the input either way, so when it runs cannot change what it returns.
        //
        // `BIG_ROWS` also dominates every other shape in this module (the next
        // largest is 96×64), so no earlier test in the process can have already
        // satisfied the reservation and quietly re-break the premise.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        const BIG_ROWS: i32 = 1024;
        let (hidden, inter) = (64usize, 32usize);
        let e = int4_expert(0x9002, hidden, inter)?;
        let mut r = Lcg(0x6120);
        let small: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
        let big: Vec<f32> = (0..BIG_ROWS as usize * hidden).map(|_| r.f()).collect();

        // Sizes the scratch for exactly one row, which is what the capture below
        // wants: a reservation already satisfied, so it records against a
        // generation that is not about to move under it.
        let want_small = with_graph_knob(false, || expert_group(&[&e], &[1], &small, hidden))?;

        let before = graph_cache_stats();
        with_graph_knob(true, || expert_group(&[&e], &[1], &small, hidden))?; // capture at gen G
        let got_big = with_graph_knob(true, || expert_group(&[&e], &[BIG_ROWS], &big, hidden))?; // reallocs → gen G+1
        let got_small = with_graph_knob(true, || expert_group(&[&e], &[1], &small, hidden))?;
        let after = graph_cache_stats();

        // Only now — see the ordering note above.
        let want_big = with_graph_knob(false, || expert_group(&[&e], &[BIG_ROWS], &big, hidden))?;

        assert_eq!(want_big, got_big, "the growing call must be correct");
        assert_eq!(
            want_small, got_small,
            "the small shape's cached graph was captured before the scratch moved — replaying it \
             reads freed VRAM, and the generation guard is what stops that"
        );
        assert!(
            after.invalidations > before.invalidations,
            "growing the scratch must invalidate the graph captured under the old generation — if \
             this fails, check the premise before the guard: something must have sized the scratch \
             for {BIG_ROWS} rows before the graphed sequence ran, in which case no reallocation \
             happened and there was nothing to invalidate"
        );
        Ok(())
    }

    #[test]
    fn a_replayed_graph_picks_up_new_expert_weights() -> Result<(), Error> {
        // The point of caching by *shape*: one graph serves every residency
        // generation at that shape. That only works because the descriptor
        // upload is inside the graph and sources from a pinned buffer the host
        // rewrites between replays. If the descriptors were baked as kernel
        // arguments instead, this test would return expert A's answer for
        // expert B — correct-looking numbers for the wrong weights.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 32usize, 2usize);
        let a = int4_expert(0xA1, hidden, inter)?;
        let b = int4_expert(0xB2, hidden, inter)?;
        let mut r = Lcg(0xD1FF);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let want_a = with_graph_knob(false, || expert_group(&[&a], &[n as i32], &x, hidden))?;
        let want_b = with_graph_knob(false, || expert_group(&[&b], &[n as i32], &x, hidden))?;
        assert_ne!(want_a, want_b, "the two experts must actually differ or this proves nothing");

        let got_a = with_graph_knob(true, || expert_group(&[&a], &[n as i32], &x, hidden))?;
        let got_b = with_graph_knob(true, || expert_group(&[&b], &[n as i32], &x, hidden))?;
        assert_eq!(want_a, got_a);
        assert_eq!(want_b, got_b, "a replay at the same shape must use the CURRENT descriptors");
        Ok(())
    }

    #[test]
    fn the_w4a16_arm_is_never_graph_cached() -> Result<(), Error> {
        // That arm passes device weight pointers as kernel arguments, so a
        // captured graph is bound to the expert set it was recorded with — the
        // exact bug the test above rules out for the other arms. It must fall
        // through to the eager path and say so in the counters, rather than
        // capture a graph that is wrong the moment residency changes.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 32usize, 24usize); // ≥ the 16-row W4A16 floor
        let e = int4_expert(0x9004, hidden, inter)?;
        let mut r = Lcg(0x16A1);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let (before, got, after) = with_env("COLI_CUDA_TC_W4A16", "1", || {
            let before = graph_cache_stats();
            let got = with_graph_knob(true, || expert_group(&[&e], &[n as i32], &x, hidden));
            (before, got, graph_cache_stats())
        });
        got?;

        assert_eq!(after.captures, before.captures, "the W4A16 arm must not be captured");
        assert!(
            after.uncacheable > before.uncacheable,
            "a call skipped with the knob on must be counted, or 'the cache is doing nothing' \
             is indistinguishable from 'the cache is working'"
        );
        Ok(())
    }

    #[test]
    fn every_w4a16_tile_instantiation_agrees_with_the_default() -> Result<(), Error> {
        // `COLI_W4A16_TILES` emits three template instantiations and the
        // dispatch picks one by runtime triple. Until 2026-08-08 only 16×16×16
        // had ever executed — the other two were known to *compile*, which for a
        // templated kernel proves the fragment shapes are legal and nothing
        // else. A shape that is legal can still index its tile wrong, and the
        // dispatch's silent fallback to 16×16×16 means a triple that never
        // matches produces plausible output forever.
        //
        // So: assert each triple reaches the W4A16 arm, and that all three agree
        // with the default. They accumulate in a different order, so this is a
        // tolerance check rather than the bit-identity the graph tests use.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, n) = (64usize, 64usize, 32usize); // ≥ the 16-row W4A16 floor
        let e = int4_expert(0x9005, hidden, inter)?;
        let mut r = Lcg(0x7113);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();

        let out = with_env("COLI_CUDA_TC_W4A16", "1", || {
            let run = |tile: Option<(u16, u16, u16)>| {
                with_graph_knob(false, || expert_group_tiled(&[&e], &[n as i32], &x, hidden, tile))
            };
            [None, Some((16, 16, 16)), Some((32, 8, 16)), Some((8, 32, 16))].map(run)
        });

        let mut got = Vec::new();
        for o in out {
            got.push(o?);
        }
        let (want, want_arm) = &got[0];
        assert_eq!(*want_arm, GroupArm::W4A16, "the tile only means anything on the W4A16 arm");
        for (i, (y, arm)) in got.iter().enumerate().skip(1) {
            assert_eq!(*arm, GroupArm::W4A16, "tile {i} did not reach the W4A16 arm");
            assert_eq!(y.len(), want.len());
            let worst = y
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 2e-2,
                "tile {i} disagrees with 16×16×16 by {worst} — a legal fragment shape that \
                 computes the wrong thing, which compiling it could never have caught"
            );
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

    /// Pins the stream fix, and does it *deterministically* rather than by
    /// hoping to observe a race.
    ///
    /// `cudaStreamBeginCapture` records `ctx->stream`. An op launched on any
    /// other stream is not an error and not a warning — it simply is not in the
    /// graph. Until 2026-08-07 `pipe_rmsnorm` launched on the **default** stream,
    /// so capturing it and replaying produced a graph that quietly skipped the
    /// normalization: the eager pass during capture still ran it (which is why a
    /// naive "capture then check the buffer" test would have passed), but the
    /// *replay* would not.
    ///
    /// So this captures rmsnorm→silu_mul, then replays onto **freshly uploaded
    /// input** and requires the composite result. With the op on the wrong
    /// stream, the replay leaves the normalization undone and the values are
    /// wrong by the norm factor.
    ///
    /// The same fix is what makes the chain *ordered*: `ctx->stream` is
    /// `cudaStreamNonBlocking`, so a default-stream kernel is not synchronized
    /// against it at all, and `silu_mul` could read what `rmsnorm` had not
    /// finished writing.
    #[test]
    fn graph_capture_records_ops_only_from_the_context_stream() -> Result<(), Error> {
        use std::os::raw::c_void;
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let d = 256usize; // one row, so rmsnorm is over the whole buffer
        let mut r = Lcg(0x511E);
        let x: Vec<f32> = (0..d).map(|_| r.f()).collect();
        let w: Vec<f32> = (0..d).map(|_| r.f() * 0.5 + 1.0).collect();
        let up: Vec<f32> = (0..d).map(|_| r.f()).collect();
        let eps = 1e-6f32;

        // Reference: rmsnorm(x) * w, then silu(.) * up — the composite the graph
        // must reproduce on replay.
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let want: Vec<f32> = (0..d).map(|k| silu(x[k] * inv * w[k]) * up[k]).collect();

        let nb = d * 4;
        // SAFETY: four d-float device buffers on device 0.
        let (x_dev, w_dev, y_dev, u_dev) = unsafe {
            (
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
                ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32,
            )
        };
        assert!(
            !x_dev.is_null() && !w_dev.is_null() && !y_dev.is_null() && !u_dev.is_null(),
            "device alloc"
        );
        let upload = |dst: *mut f32, src: &[f32]| {
            // SAFETY: `dst` has `nb` bytes; `src` has `d` f32.
            unsafe { ffi::coli_cuda_pipe_upload(0, dst as *mut c_void, src.as_ptr() as *const c_void, nb) };
        };
        upload(x_dev, &x);
        upload(w_dev, &w);
        upload(u_dev, &up);

        let graph = capture(0, || {
            // SAFETY: y = rmsnorm(x) * w over one row of `d`, then y = silu(y)*up.
            let a = unsafe { ffi::coli_cuda_pipe_rmsnorm(0, y_dev, x_dev, w_dev, 1, d as c_int, eps) };
            let b = unsafe { ffi::coli_cuda_pipe_silu_mul(0, y_dev, u_dev, d) };
            if a == 1 && b == 1 {
                Ok(())
            } else {
                Err(Error::Format("rmsnorm capture".into()))
            }
        })?;

        // Scribble over `y_dev` so a replay that skips the rmsnorm cannot pass on
        // leftovers from the eager pass that ran during capture.
        let poison = vec![-7.0f32; d];
        upload(y_dev, &poison);

        graph.launch()?;
        let mut out = vec![0f32; d];
        // SAFETY: `y_dev` has `nb` bytes; `out` has `d` f32.
        unsafe { ffi::coli_cuda_pipe_download(0, y_dev as *const c_void, out.as_mut_ptr() as *mut c_void, nb) };
        for k in 0..d {
            let tol = 1e-4 * want[k].abs().max(1.0);
            assert!(
                (out[k] - want[k]).abs() < tol,
                "k={k} got={} want={} — a replayed graph missing its rmsnorm means the op was on the wrong stream",
                out[k],
                want[k]
            );
        }
        // SAFETY: all four came from pipe_alloc; free is null-safe.
        unsafe {
            ffi::coli_cuda_pipe_free(0, x_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, w_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, y_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, u_dev as *mut c_void);
        }
        Ok(())
    }

    #[test]
    fn a_download_observes_work_already_queued_on_the_context_stream() -> Result<(), Error> {
        use std::os::raw::c_void;
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        // `pipe_upload`/`pipe_download` were blocking `cudaMemcpy` on the legacy
        // default stream until 2026-08-08. `ctx->stream` is `cudaStreamNonBlocking`,
        // so the two were not ordered against each other and a download could return
        // bytes a queued kernel had not written yet. The 2026-08-07 pass fixed this
        // for the compute primitives and missed the staging ones; the graph-capture
        // tests could not catch it because they always sync via `graph.launch()`
        // before downloading.
        //
        // The defect is a race, so this makes the race one-sided instead of relying
        // on luck: ROUNDS launches over N floats is tens of milliseconds of device
        // work standing against a ~32 MiB copy. A correctly ordered download waits
        // for the whole queue and can only observe ROUNDS; an unordered one would
        // have to beat every launch to report ROUNDS by accident.
        const N: usize = 8 << 20; // 8 Mi floats = 32 MiB
        const ROUNDS: usize = 200;
        let nb = N * 4;
        // SAFETY: two N-float device buffers on device 0.
        let (x_dev, t_dev) = unsafe {
            (ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32, ffi::coli_cuda_pipe_alloc(0, nb) as *mut f32)
        };
        assert!(!x_dev.is_null() && !t_dev.is_null(), "device alloc");

        let zero = vec![0f32; N];
        let one = vec![1f32; N];
        // SAFETY: both device buffers hold `nb` bytes; both slices hold N f32.
        let up = unsafe {
            ffi::coli_cuda_pipe_upload(0, x_dev as *mut c_void, zero.as_ptr() as *const c_void, nb)
                & ffi::coli_cuda_pipe_upload(0, t_dev as *mut c_void, one.as_ptr() as *const c_void, nb)
        };
        assert_eq!(up, 1, "pipe_upload");
        for _ in 0..ROUNDS {
            // SAFETY: x += t over N elements, both device-resident, both `nb` bytes.
            let ok = unsafe { ffi::coli_cuda_pipe_add(0, x_dev, t_dev, N) };
            assert_eq!(ok, 1, "pipe_add launch");
        }
        // Deliberately no `pipe_sync` and no graph launch: the download itself is
        // what has to carry the ordering.
        let mut out = vec![f32::NAN; N];
        // SAFETY: `x_dev` holds `nb` bytes; `out` holds N f32.
        let ok =
            unsafe { ffi::coli_cuda_pipe_download(0, x_dev as *const c_void, out.as_mut_ptr() as *mut c_void, nb) };
        // Scan the whole buffer, not index 0: a copy that lands while the adds are
        // still running gives a correct prefix and a stale tail.
        let bad = out.iter().position(|&v| v != ROUNDS as f32).map(|i| (i, out[i]));
        // SAFETY: both came from pipe_alloc; free is null-safe.
        unsafe {
            ffi::coli_cuda_pipe_free(0, x_dev as *mut c_void);
            ffi::coli_cuda_pipe_free(0, t_dev as *mut c_void);
        }
        assert_eq!(ok, 1, "pipe_download");
        if let Some((i, got)) = bad {
            // `bad` is Some only where the value already differs, so this always
            // trips — it is an assert rather than a `panic!` because the repo bars
            // panicking error handling in tests too (`clippy.toml`).
            assert_eq!(
                got, ROUNDS as f32,
                "out[{i}] — pipe_download returned the buffer before the queued adds finished"
            );
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
    fn fused_reduce_matches_the_host_reduce() -> Result<(), Error> {
        // Two experts, both contributing to both batch rows — the shape the
        // fusion exists for (`total` = 4 y-rows collapsing to `s_n` = 2).
        // Compared against the host reduce over `expert_group`'s own output, so
        // this checks the *reduce*, not the SwiGLU (which its own tests cover).
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, s_n) = (64usize, 32usize, 2usize);
        let a = int4_expert(0xF1, hidden, inter)?;
        let b = int4_expert(0xF2, hidden, inter)?;
        let mut r = Lcg(0xFEED);
        let batch: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();

        // y-row k: (expert 0 -> row 0), (expert 0 -> row 1), (expert 1 -> row 0),
        // (expert 1 -> row 1). Gathered inputs repeat the batch rows accordingly.
        let dst = [0usize, 1, 0, 1];
        let rw = [0.25f32, 0.5, 0.75, 1.5];
        let mut x = Vec::new();
        for &s in &dst {
            x.extend_from_slice(&batch[s * hidden..s * hidden + hidden]);
        }

        let y = expert_group(&[&a, &b], &[2, 2], &x, hidden)?;
        let mut want = vec![0f32; s_n * hidden];
        for (k, (&s, &w)) in dst.iter().zip(&rw).enumerate() {
            for d in 0..hidden {
                want[s * hidden + d] += w * y[k * hidden + d];
            }
        }

        let layout = ReduceLayout::build(&dst, s_n).ok_or_else(|| Error::Format("layout".into()))?;
        let got = expert_group_reduce(&[&a, &b], &[2, 2], &x, hidden, &layout, &rw, s_n)?;
        assert_eq!(got.len(), s_n * hidden);
        for k in 0..got.len() {
            let tol = 1e-4 * want[k].abs().max(1.0);
            assert!((got[k] - want[k]).abs() < tol, "k={k} fused={} host={}", got[k], want[k]);
        }
        Ok(())
    }

    #[test]
    fn fused_reduce_is_bit_stable_across_repeats() -> Result<(), Error> {
        // **The test an atomic scatter fails.** `f32 +=` is not associative, so
        // a reduce that let threads race would return a slightly different
        // vector each run — and every tolerance-based test above would still
        // pass. Identical bits, three times, is the only assertion that catches
        // it, and it is the property the engine's reproducibility rests on.
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, s_n) = (64usize, 32usize, 3usize);
        let experts: Vec<GpuExpert> =
            (0..4).map(|i| int4_expert(0x700 + i, hidden, inter)).collect::<Result<_, Error>>()?;
        let refs: Vec<&GpuExpert> = experts.iter().collect();
        let mut r = Lcg(0x5EED);
        // Every expert contributes to every batch row: maximal contention, which
        // is exactly where an atomic implementation would diverge.
        let dst: Vec<usize> = (0..4 * s_n).map(|k| k % s_n).collect();
        let rw: Vec<f32> = (0..dst.len()).map(|_| r.f().abs() + 0.1).collect();
        let x: Vec<f32> = (0..dst.len() * hidden).map(|_| r.f()).collect();
        let rows = vec![s_n as i32; 4];
        let layout = ReduceLayout::build(&dst, s_n).ok_or_else(|| Error::Format("layout".into()))?;

        let first = expert_group_reduce(&refs, &rows, &x, hidden, &layout, &rw, s_n)?;
        for attempt in 1..3 {
            let again = expert_group_reduce(&refs, &rows, &x, hidden, &layout, &rw, s_n)?;
            assert_eq!(first, again, "run {attempt} differs bit-for-bit — the reduce is not ordered");
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

    /// The async lane must land the **same device state** as the blocking one.
    ///
    /// This is the gate on the one genuinely risky part of the disk→GPU lane:
    /// `coli_cuda_tensor_upload_async` moves the `offset_to_signed_s4` int4
    /// conversion kernel from the default stream onto `ctx->stream`. Get that
    /// wrong and the kernel is unordered against the copy feeding it, so it
    /// converts nibbles that have not arrived — which this repo has already
    /// shipped once, in the other direction, and which no timing harness
    /// notices. Byte-exact equality of the two paths' outputs is what catches
    /// it: both read the same weights through the same kernel, so any
    /// difference is the ordering.
    #[test]
    fn the_async_int4_upload_lands_what_the_blocking_one_does() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        let (hidden, inter, s_n) = (256usize, 512usize, 3usize);
        let mut r = Lcg(0xA5EED);
        let gatef: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let upf: Vec<f32> = (0..inter * hidden).map(|_| r.f() * 0.1).collect();
        let downf: Vec<f32> = (0..hidden * inter).map(|_| r.f() * 0.1).collect();
        let (gq, gs) = quant_i4(&gatef, inter, hidden);
        let (uq, us) = quant_i4(&upf, inter, hidden);
        let (dq, ds) = quant_i4(&downf, hidden, inter);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();

        let blocking = GpuExpert::upload_int4(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter)?;
        let want = dense_mlp_w4a16(&blocking, &x, s_n, hidden)?;
        drop(blocking);

        let async_e = GpuExpert::upload_int4_async(0, (&gq, &gs), (&uq, &us), (&dq, &ds), hidden, inter)?;
        // The caller's obligation: nothing may read these weights, and the host
        // buffers may not be dropped, until the stream drains.
        stream_sync(0)?;
        let got = dense_mlp_w4a16(&async_e, &x, s_n, hidden)?;

        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "element {k}: async upload {g} != blocking upload {w} — \
                 the same weights through the same kernel must be bit-identical, \
                 so a difference here is the int4 conversion running unordered \
                 against its copy"
            );
        }
        Ok(())
    }

    /// `stream_sync` on an idle stream must succeed rather than error — the
    /// drain path calls it unconditionally at the end of a generation, including
    /// generations that queued nothing.
    #[test]
    fn stream_sync_on_an_idle_stream_succeeds() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        stream_sync(0)?;
        stream_sync(0)?;
        Ok(())
    }

    /// `pin_host` must pin a page-aligned buffer and `unpin_host` must release
    /// it, with the counters moving together. A refusal (small `RLIMIT_MEMLOCK`)
    /// is a legitimate outcome on some hosts and must be reported as declined
    /// rather than counted as pinned.
    #[test]
    fn pin_host_registers_and_releases_an_aligned_buffer() -> Result<(), Error> {
        let _g = gpu_guard();
        if init(&[0]) < 1 {
            return Ok(());
        }
        const LEN: usize = 4 << 20; // over MIN_PIN_BYTES, and page-aligned below
        let Ok(layout) = std::alloc::Layout::from_size_align(LEN, 4096) else {
            return Ok(());
        };
        // SAFETY: non-zero size, power-of-two alignment; freed exactly once below.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Ok(());
        }
        let before = pin_stats();
        let pinned = pin_host(ptr, LEN);
        let after = pin_stats();
        if pinned {
            assert_eq!(after.buffers, before.buffers + 1, "a pinned buffer must be counted");
            assert_eq!(after.bytes, before.bytes + LEN as u64, "its bytes must be counted");
            unpin_host(ptr, LEN);
            let end = pin_stats();
            assert_eq!(end.buffers, before.buffers, "unpin must give the count back");
            assert_eq!(end.bytes, before.bytes, "unpin must give the bytes back");
        } else {
            // The honest failure: RLIMIT_MEMLOCK too small for a 4 MB pin.
            assert_eq!(after.declined, before.declined + 1, "a refusal must be counted as declined");
            assert_eq!(after.buffers, before.buffers, "a refused buffer must not be counted as pinned");
            eprintln!(
                "note: cudaHostRegister refused 4 MB, so the pinned lane is not exercised here. \
                 This is NOT `ulimit -l`: the NVIDIA driver pins through its own accounting \
                 (measured registering 256 MB with `ulimit -l` at 8192 KB)."
            );
        }
        // SAFETY: same pointer and layout as the allocation, freed once, and
        // already unregistered above if it was ever registered.
        unsafe { std::alloc::dealloc(ptr, layout) };
        Ok(())
    }

    /// A buffer under the policy floor must be declined without ever reaching
    /// the driver: `cudaHostRegister` is a page-table walk, far too expensive to
    /// pay on an allocation that will never be an H2D source.
    #[test]
    fn pin_host_declines_small_buffers_without_calling_the_driver() {
        let before = pin_stats();
        let mut small = [0u8; 64];
        assert!(!pin_host(small.as_mut_ptr(), small.len()), "a 64-byte buffer must be declined");
        let after = pin_stats();
        assert_eq!(after, before, "a policy refusal must not touch the driver or the counters");
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

    /// The fused reduce's *ordering* is what fixes its result, and ordering is
    /// decided entirely on the host. These run on any box, including one with
    /// no GPU — which matters, because every test that exercises the kernel
    /// skips itself without a device.
    #[test]
    fn reduce_layout_lists_contributions_in_ascending_y_row_order() -> Result<(), &'static str> {
        // Interleaved destinations: three y-rows for batch row 0, two for row 1.
        // Ascending order within each list is not incidental — `k` is
        // batch-union (`pos`) order, so it is the order the host reduce sums in.
        let l = ReduceLayout::build(&[0, 1, 0, 1, 0], 2).ok_or("build rejected a valid layout")?;
        assert_eq!(l.row_ptr, vec![0, 3, 5]);
        assert_eq!(l.row_idx, vec![0, 2, 4, 1, 3]);
        for s in 0..2 {
            let (lo, hi) = (l.row_ptr[s] as usize, l.row_ptr[s + 1] as usize);
            assert!(l.row_idx[lo..hi].windows(2).all(|w| w[0] < w[1]), "row {s} is not ascending");
        }
        Ok(())
    }

    #[test]
    fn reduce_layout_covers_every_contribution_exactly_once() -> Result<(), &'static str> {
        // A CSR that dropped or duplicated a y-row would compute a token from
        // the wrong set of experts — and would still produce plausible numbers.
        let dst = [2usize, 0, 2, 1, 1, 2, 0];
        let l = ReduceLayout::build(&dst, 3).ok_or("build rejected a valid layout")?;
        let mut seen: Vec<i32> = l.row_idx.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..dst.len() as i32).collect::<Vec<_>>());
        assert_eq!(l.row_ptr.last().copied(), Some(dst.len() as i32));
        for (s, expected) in [(0usize, 2), (1, 2), (2, 3)] {
            let (lo, hi) = (l.row_ptr[s] as usize, l.row_ptr[s + 1] as usize);
            assert_eq!(hi - lo, expected, "batch row {s} contribution count");
            assert!(l.row_idx[lo..hi].iter().all(|&k| dst[k as usize] == s));
        }
        Ok(())
    }

    #[test]
    fn reduce_layout_refuses_an_out_of_range_destination() -> Result<(), &'static str> {
        // Refuse, never drop: a discarded contribution is a token computed from
        // fewer experts than the router chose, and nothing downstream can see it.
        assert_eq!(ReduceLayout::build(&[0, 5], 2), None);
        // An empty batch of contributions is legal — every row is simply zero.
        // `.ok_or(..)?`, not `.unwrap_or_default()`: the default IS an empty
        // layout, so the old form asserted the same thing whether `build`
        // returned the empty layout or refused outright.
        let empty = ReduceLayout::build(&[], 2).ok_or("an empty batch is legal")?;
        assert_eq!(empty.row_ptr, vec![0, 0, 0]);
        assert!(empty.row_idx.is_empty());
        Ok(())
    }
}

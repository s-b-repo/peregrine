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
    }
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

    #[test]
    fn expert_group_matches_cpu_f32() -> Result<(), Error> {
        // Skip gracefully on a box with the feature built but no usable GPU.
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
            let _ = device_count();
            let _ = status();
        }
    }
}

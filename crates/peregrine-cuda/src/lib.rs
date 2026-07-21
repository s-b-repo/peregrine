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

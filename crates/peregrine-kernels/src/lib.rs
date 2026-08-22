//! colibrì CPU integer-dot kernels (M1): the token-exactness-critical matmul
//! path. Scalar reference implementations define the exact arithmetic; AVX2 and
//! AVX-VNNI variants are validated bit-identical to them. Ported from the IDOT
//! kernels in `c/glm.c` (`qrow_i8`, `dot_i8i8`, `dot_i4i8`, `matmul_*_idot`).

// Explicit index loops are the deliberate house style in these numeric kernels:
// they mirror the C ports (for line-by-line verification) and most index several
// slices at once, so `needless_range_loop` is noise here.
// Quality gates: SIMD intrinsics are the only (irreducible) `unsafe` here; no
// panicking error handling in library code.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod idot;
pub mod matmul;
pub mod quant;

pub use idot::{
    dot_i2i8, dot_i2i8_g64, dot_i2i8_g64_scalar, dot_i2i8_scalar, dot_i3i8_g64, dot_i3i8_g64_scalar, dot_i4i8, dot_i4i8_grouped, dot_i4i8_grouped_scalar,
    dot_i4i8_scalar,
    dot_i8i8, dot_i8i8_scalar,
};
pub use matmul::{
    matmul_f32, matmul_i4_from_f32, matmul_i4_idot, matmul_i4g_from_f32, matmul_i4g_idot,
    matmul_i8_from_f32, matmul_q_idot, ActScratch, MatShape,
};
pub use quant::qrow_i8;

/// Human-readable summary of which SIMD kernel family this CPU will use, for
/// startup logging and to make the token-exactness caveat visible (batched/GPU
/// paths may round differently; the S=1 CPU path here is the reference).
pub fn cpu_kernel_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnni") {
            return "x86_64 AVX-VNNI (vpdpbusd)";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "x86_64 AVX2 (maddubs)";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return "aarch64 NEON+dotprod (sdot)";
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return "aarch64 NEON (smull)";
        }
    }
    "portable scalar"
}

#[cfg(test)]
mod tests {
    /// A SIMD kernel that is never reached is worse than none — it looks like
    /// coverage. On aarch64 the dispatcher must land on a vector path, so that
    /// an equivalence test passing cannot mean "both sides ran the scalar
    /// reference".
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_does_not_silently_fall_back_to_scalar() {
        let b = super::cpu_kernel_tier();
        assert!(b.starts_with("aarch64"), "expected a NEON backend on aarch64, got {b:?}");
    }
}

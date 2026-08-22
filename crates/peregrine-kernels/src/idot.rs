//! Integer dot products — port of `dot_i8i8` and `dot_i4i8` (`c/glm.c:602,690`).
//!
//! These are the token-exactness anchor. The scalar versions are the reference;
//! the SIMD versions must produce the **identical** i32 accumulator (integer
//! addition is associative, so any lane grouping gives the same sum — there is
//! no rounding until the final f32 scale multiply in [`crate::matmul`]).
//!
//! Weights: int8 `w[0..I]`, or packed int4 `w4[0..(I+1)/2]` where element `i`
//! is the low nibble of byte `i/2` (even `i`) or the high nibble (odd `i`),
//! biased by −8 into `[-8, 7]`. Activations: int8 `x[0..I]` (from `qrow_i8`).

/// int8·int8 dot → i32. Reference implementation.
#[inline]
pub fn dot_i8i8_scalar(w: &[i8], x: &[i8], n: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..n {
        sum += w[i] as i32 * x[i] as i32;
    }
    sum
}

/// packed-int4·int8 dot → i32. Reference implementation.
#[inline]
pub fn dot_i4i8_scalar(w4: &[u8], x: &[i8], n: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..n {
        let byte = w4[i >> 1];
        let nib = if i & 1 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
        sum += (nib - 8) * x[i] as i32;
    }
    sum
}

/// Best available int8·int8 dot for this CPU (runtime feature dispatch).
#[inline]
pub fn dot_i8i8(w: &[i8], x: &[i8], n: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // The VNNI kernel is `#[target_feature(enable = "avx2,avxvnni")]`, so BOTH
        // must be present — checking only avxvnni would be unsound on a
        // (hypothetical) CPU reporting avxvnni without avx2.
        // SAFETY: each branch calls a target_feature fn only after detecting the
        // exact features it requires at runtime.
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni") {
            return unsafe { x86::dot_i8i8_vnni(w, x, n) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::dot_i8i8_avx2(w, x, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: each branch calls a target_feature fn only after detecting the
        // exact features it requires at runtime.
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { aarch64::dot_i8i8_dotprod(w, x, n) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { aarch64::dot_i8i8_neon(w, x, n) };
        }
    }
    dot_i8i8_scalar(w, x, n)
}

/// Best available packed-int4·int8 dot for this CPU.
#[inline]
pub fn dot_i4i8(w4: &[u8], x: &[i8], n: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the avx2 kernel is only called after detecting avx2 at runtime.
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::dot_i4i8_avx2(w4, x, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: as above — detection precedes each target_feature call.
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { aarch64::dot_i4i8_dotprod(w4, x, n) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { aarch64::dot_i4i8_neon(w4, x, n) };
        }
    }
    dot_i4i8_scalar(w4, x, n)
}

/// grouped packed-int4·int8 dot → f32. The weight row is split into groups of
/// `gs` input elements, each with its own f32 scale `scales[g]`; the result is
/// `Σ_g scales[g] · (int4·int8 dot of group g)`. This is the coherence-critical
/// path for GLM-5.2: 128-block scales preserve the fine-grained weight structure
/// per-row scales crush (see colibrì `issue_grouped_quant.md`).
///
/// Every valid group size is a multiple of 16 (see [`detect_group_size`]), so a
/// group starts at an even element index and its packed nibbles begin on a byte
/// boundary (`start >> 1`) — the same alignment invariant [`dot_i4i8`] relies on.
/// Reference version: uses the scalar inner dot.
#[inline]
pub fn dot_i4i8_grouped_scalar(w4: &[u8], x: &[i8], scales: &[f32], n: usize, gs: usize) -> f32 {
    let mut acc = 0f32;
    let mut start = 0usize;
    let mut g = 0usize;
    while start < n {
        let len = gs.min(n - start);
        let d = dot_i4i8_scalar(&w4[start >> 1..], &x[start..], len);
        acc += scales[g] * d as f32;
        start += gs;
        g += 1;
    }
    acc
}

/// grouped packed-int4·int8 dot → f32, using the best available inner dot.
///
/// The per-group integer dot is bit-identical between scalar and SIMD (integer
/// addition is associative), and the f32 group-scale accumulation runs in fixed
/// group order, so this equals [`dot_i4i8_grouped_scalar`] exactly.
#[inline]
pub fn dot_i4i8_grouped(w4: &[u8], x: &[i8], scales: &[f32], n: usize, gs: usize) -> f32 {
    let mut acc = 0f32;
    let mut start = 0usize;
    let mut g = 0usize;
    while start < n {
        let len = gs.min(n - start);
        let d = dot_i4i8(&w4[start >> 1..], &x[start..], len);
        acc += scales[g] * d as f32;
        start += gs;
        g += 1;
    }
    acc
}

/// packed-int2·int8 dot → i32. Reference implementation. Element `i` is the
/// `2·(i&3)`-shifted 2-bit field of byte `i>>2`, biased by −2 into `[-2, 1]`.
#[inline]
pub fn dot_i2i8_scalar(w2: &[u8], x: &[i8], n: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..n {
        let byte = w2[i >> 2];
        let field = ((byte >> (2 * (i & 3))) & 0x03) as i32;
        sum += (field - 2) * x[i] as i32;
    }
    sum
}

/// Best available packed-int2·int8 dot for this CPU. Bit-identical to
/// [`dot_i2i8_scalar`] (integer addition is associative).
#[inline]
pub fn dot_i2i8(w2: &[u8], x: &[i8], n: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the avx2 kernel is only called after detecting avx2 at runtime.
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::dot_i2i8_avx2(w2, x, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // No plain-NEON int2 twin: int2 is a closed negative on this engine
        // (uniform int2-g64 measured `flip_rate` 1.000), so the format is a
        // completeness case rather than a hot path. `sdot` costs nothing extra
        // once the unpack is written; a second hand-rolled variant would be risk
        // without a caller.
        // SAFETY: detection precedes the target_feature call.
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { aarch64::dot_i2i8_dotprod(w2, x, n) };
        }
    }
    dot_i2i8_scalar(w2, x, n)
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    //! NEON (`smull`/`sadalp`) and ARMv8.2 dot-product (`sdot`) kernels — the
    //! ARM twins of the AVX2 and AVX-VNNI paths above, and written to the same
    //! contract: process vector-width chunks, fall through to the scalar tail,
    //! produce the **identical** i32 accumulator.
    //!
    //! `sdot` is the direct analogue of VNNI's `vpdpbusd`, and easier to use:
    //! it is signed×signed→i32, so none of the `maddubs` sign-trick ceremony is
    //! needed. Each instruction consumes 4 int8 pairs per 32-bit lane.
    //!
    //! NEON is architecturally mandatory on ARMv8-A, so the `neon` kernels are
    //! the real floor here; the scalar reference is reached only if feature
    //! detection somehow reports no NEON at all.
    use super::{dot_i2i8_scalar, dot_i4i8_scalar, dot_i8i8_scalar};
    use std::arch::aarch64::*;

    /// `sdot acc.4s, a.16b, b.16b` — four int8 products summed into each of the
    /// four i32 lanes.
    ///
    /// Written as inline assembly rather than `vdotq_s32` because that intrinsic
    /// is still nightly-only (`stdarch_neon_dotprod`, rust-lang/rust#117224) and
    /// this workspace is stable-only. `asm!` *is* stable, so the instruction is
    /// reachable without a nightly toolchain; the equivalence tests below pin the
    /// result against the scalar reference either way.
    ///
    /// # Safety
    /// The CPU must support NEON and the dot-product extension.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut out = acc;
        core::arch::asm!(
            "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
        out
    }

    /// Unpack 16 packed-int4 bytes into 32 int8 values in element order,
    /// biased into `[-8, 7]`. Returns `(elems 0..15, elems 16..31)`.
    ///
    /// `vzip1q_u8`/`vzip2q_u8` are the NEON spelling of `_mm_unpacklo_epi8` /
    /// `_mm_unpackhi_epi8`: element `i` is the low nibble of byte `i/2` for even
    /// `i` and the high nibble for odd `i`, so zipping the low-nibble and
    /// high-nibble vectors restores element order exactly.
    ///
    /// # Safety
    /// Caller must have NEON, and `by` must be a valid 16-byte load.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn unpack_i4(by: uint8x16_t) -> (int8x16_t, int8x16_t) {
        let m4 = vdupq_n_u8(0x0F);
        let b8 = vdupq_n_s8(8);
        let lo = vandq_u8(by, m4);
        let hi = vshrq_n_u8::<4>(by);
        let n0 = vzip1q_u8(lo, hi);
        let n1 = vzip2q_u8(lo, hi);
        (
            vsubq_s8(vreinterpretq_s8_u8(n0), b8),
            vsubq_s8(vreinterpretq_s8_u8(n1), b8),
        )
    }

    /// int8·int8 via ARMv8.2 `sdot`.
    ///
    /// Two accumulators for the same reason the AVX2 kernel uses two: `sdot` has
    /// multi-cycle latency and one accumulator serializes the loop on it.
    /// Bit-identical regardless — integer addition is associative.
    ///
    /// # Safety
    /// The CPU must support NEON and the dot-product extension. [`super::dot_i8i8`]
    /// checks both with `is_aarch64_feature_detected!` before calling.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_i8i8_dotprod(w: &[i8], x: &[i8], n: usize) -> i32 {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 32 <= n {
            acc0 = sdot(acc0, vld1q_s8(w.as_ptr().add(i)), vld1q_s8(x.as_ptr().add(i)));
            acc1 = sdot(acc1, vld1q_s8(w.as_ptr().add(i + 16)), vld1q_s8(x.as_ptr().add(i + 16)));
            i += 32;
        }
        while i + 16 <= n {
            acc0 = sdot(acc0, vld1q_s8(w.as_ptr().add(i)), vld1q_s8(x.as_ptr().add(i)));
            i += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        if i < n {
            sum += dot_i8i8_scalar(&w[i..], &x[i..], n - i);
        }
        sum
    }

    /// int8·int8 on plain NEON, for ARMv8.0/8.1 without the dot-product
    /// extension: `smull` widens i8×i8 to i16, `sadalp` pairwise-accumulates
    /// into i32. Products are bounded by 128·128, so no intermediate saturates.
    ///
    /// # Safety
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_i8i8_neon(w: &[i8], x: &[i8], n: usize) -> i32 {
        let mut acc = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 16 <= n {
            let wv = vld1q_s8(w.as_ptr().add(i));
            let xv = vld1q_s8(x.as_ptr().add(i));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(wv), vget_low_s8(xv)));
            acc = vpadalq_s16(acc, vmull_high_s8(wv, xv));
            i += 16;
        }
        let mut sum = vaddvq_s32(acc);
        if i < n {
            sum += dot_i8i8_scalar(&w[i..], &x[i..], n - i);
        }
        sum
    }

    /// packed-int4·int8 via `sdot`. This is GLM-5.2's hot path: every routed
    /// expert weight is int4 with group scales, and [`super::dot_i4i8_grouped`]
    /// reaches it once per group.
    ///
    /// # Safety
    /// The CPU must support NEON and the dot-product extension.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_i4i8_dotprod(w4: &[u8], x: &[i8], n: usize) -> i32 {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 32 <= n {
            let (w0, w1) = unpack_i4(vld1q_u8(w4.as_ptr().add(i >> 1)));
            acc0 = sdot(acc0, w0, vld1q_s8(x.as_ptr().add(i)));
            acc1 = sdot(acc1, w1, vld1q_s8(x.as_ptr().add(i + 16)));
            i += 32;
        }
        let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        if i < n {
            // `i` is a multiple of 32, so `i >> 1` lands on a byte boundary and
            // the tail's nibble parity is preserved.
            sum += dot_i4i8_scalar(&w4[i >> 1..], &x[i..], n - i);
        }
        sum
    }

    /// packed-int4·int8 on plain NEON, without the dot-product extension.
    ///
    /// # Safety
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_i4i8_neon(w4: &[u8], x: &[i8], n: usize) -> i32 {
        let mut acc = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 32 <= n {
            let (w0, w1) = unpack_i4(vld1q_u8(w4.as_ptr().add(i >> 1)));
            let x0 = vld1q_s8(x.as_ptr().add(i));
            let x1 = vld1q_s8(x.as_ptr().add(i + 16));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
            acc = vpadalq_s16(acc, vmull_high_s8(w0, x0));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
            acc = vpadalq_s16(acc, vmull_high_s8(w1, x1));
            i += 32;
        }
        let mut sum = vaddvq_s32(acc);
        if i < n {
            sum += dot_i4i8_scalar(&w4[i >> 1..], &x[i..], n - i);
        }
        sum
    }

    /// packed-int2·int8 via `sdot`. 16 bytes carry 64 elements, so one load
    /// feeds four `sdot`s.
    ///
    /// The 4-way interleave restores element order: byte `k` holds elements
    /// `4k..4k+3` in its four 2-bit fields, so zipping `(f0,f1)` and `(f2,f3)`
    /// as bytes and then zipping *those* as 16-bit lanes lands
    /// `f0[k],f1[k],f2[k],f3[k]` consecutively.
    ///
    /// # Safety
    /// The CPU must support NEON and the dot-product extension.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_i2i8_dotprod(w2: &[u8], x: &[i8], n: usize) -> i32 {
        let m2 = vdupq_n_u8(0x03);
        let b2 = vdupq_n_s8(2);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0usize;
        while i + 64 <= n {
            let by = vld1q_u8(w2.as_ptr().add(i >> 2));
            let f0 = vandq_u8(by, m2);
            let f1 = vandq_u8(vshrq_n_u8::<2>(by), m2);
            let f2 = vandq_u8(vshrq_n_u8::<4>(by), m2);
            let f3 = vshrq_n_u8::<6>(by);
            let a0 = vreinterpretq_u16_u8(vzip1q_u8(f0, f1));
            let a1 = vreinterpretq_u16_u8(vzip2q_u8(f0, f1));
            let b0 = vreinterpretq_u16_u8(vzip1q_u8(f2, f3));
            let b1 = vreinterpretq_u16_u8(vzip2q_u8(f2, f3));
            let z0 = vsubq_s8(vreinterpretq_s8_u16(vzip1q_u16(a0, b0)), b2);
            let z1 = vsubq_s8(vreinterpretq_s8_u16(vzip2q_u16(a0, b0)), b2);
            let z2 = vsubq_s8(vreinterpretq_s8_u16(vzip1q_u16(a1, b1)), b2);
            let z3 = vsubq_s8(vreinterpretq_s8_u16(vzip2q_u16(a1, b1)), b2);
            acc0 = sdot(acc0, z0, vld1q_s8(x.as_ptr().add(i)));
            acc1 = sdot(acc1, z1, vld1q_s8(x.as_ptr().add(i + 16)));
            acc0 = sdot(acc0, z2, vld1q_s8(x.as_ptr().add(i + 32)));
            acc1 = sdot(acc1, z3, vld1q_s8(x.as_ptr().add(i + 48)));
            i += 64;
        }
        let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        if i < n {
            sum += dot_i2i8_scalar(&w2[i >> 2..], &x[i..], n - i);
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 (maddubs sign-trick) and AVX-VNNI (`vpdpbusd`) dot kernels. Each
    //! processes vector-width chunks then falls through to a scalar tail, so any
    //! `n` is handled and the result equals the scalar reference exactly.
    use super::{dot_i2i8_scalar, dot_i4i8_scalar, dot_i8i8_scalar};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hsum256(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let mut s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }

    /// int8·int8 via AVX2 `maddubs`: |w| (unsigned) × x·sign(w). Adjacent-pair
    /// products stay < 32767 (bound 2·128·127) so the 16-bit intermediate never
    /// saturates — the sum is exact.
    ///
    /// # Safety
    /// The CPU must support AVX2. The [`super::dot_i8i8`] dispatcher verifies this
    /// with `is_x86_feature_detected!` before calling.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8i8_avx2(w: &[i8], x: &[i8], n: usize) -> i32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 32 <= n {
            let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
            i += 32;
        }
        let mut sum = hsum256(acc);
        if i < n {
            sum += dot_i8i8_scalar(&w[i..], &x[i..], n - i);
        }
        sum
    }

    /// int8·int8 via AVX-VNNI `vpdpbusd` (256-bit): u8·s8 → s32 directly.
    ///
    /// # Safety
    /// The CPU must support AVX2 and AVX-VNNI. [`super::dot_i8i8`] checks both
    /// with `is_x86_feature_detected!` before calling.
    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn dot_i8i8_vnni(w: &[i8], x: &[i8], n: usize) -> i32 {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 32 <= n {
            let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let xs = _mm256_sign_epi8(xv, wv); // x · sign(w)
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_abs_epi8(wv), xs);
            i += 32;
        }
        let mut sum = hsum256(acc);
        if i < n {
            sum += dot_i8i8_scalar(&w[i..], &x[i..], n - i);
        }
        sum
    }

    /// packed-int4·int8 via AVX2. Unpacks 16 bytes → 32 nibbles in order,
    /// biases by −8, then the same maddubs sign-trick as int8.
    ///
    /// # Safety
    /// The CPU must support AVX2. [`super::dot_i4i8`] verifies this before calling.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i4i8_avx2(w4: &[u8], x: &[i8], n: usize) -> i32 {
        let m4 = _mm_set1_epi8(0x0F);
        let b8 = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        // Two independent accumulators over two 32-element blocks per iteration.
        // The `maddubs → madd → add` chain is several cycles deep and a single
        // accumulator serializes every iteration on it; two chains interleave so
        // one block's unpack overlaps the other's arithmetic. Bit-identical by
        // the module's own argument — integer addition is associative, so the
        // split and the final fold reassociate a sum with no rounding in it
        // (`i4_simd_matches_scalar` pins it against the scalar reference).
        //
        // Written out rather than factored into a closure on purpose: a closure
        // declared inside a `#[target_feature]` fn does NOT inherit the feature,
        // so its intrinsics stop inlining and the "tidier" version measured 3.5x
        // SLOWER (4.8 vs 16.9 GB/s at n=5120, `examples/idotbench.rs`).
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 64 <= n {
            let by = _mm_loadu_si128(w4.as_ptr().add(i >> 1) as *const __m128i); // 16B = 32 nibbles
            let lo = _mm_and_si128(by, m4);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
            let n0 = _mm_unpacklo_epi8(lo, hi); // elems 0..15 (in order)
            let n1 = _mm_unpackhi_epi8(lo, hi); // elems 16..31
            let wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8); // → [-8,7]
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p, ones));

            let j = i + 32;
            let by_b = _mm_loadu_si128(w4.as_ptr().add(j >> 1) as *const __m128i);
            let lo_b = _mm_and_si128(by_b, m4);
            let hi_b = _mm_and_si128(_mm_srli_epi16::<4>(by_b), m4);
            let n0_b = _mm_unpacklo_epi8(lo_b, hi_b);
            let n1_b = _mm_unpackhi_epi8(lo_b, hi_b);
            let wv_b = _mm256_sub_epi8(_mm256_set_m128i(n1_b, n0_b), b8);
            let xv_b = _mm256_loadu_si256(x.as_ptr().add(j) as *const __m256i);
            let p_b = _mm256_maddubs_epi16(_mm256_sign_epi8(wv_b, wv_b), _mm256_sign_epi8(xv_b, wv_b));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p_b, ones));
            i += 64;
        }
        // Odd trailing 32-element block (n not a multiple of 64).
        while i + 32 <= n {
            let by = _mm_loadu_si128(w4.as_ptr().add(i >> 1) as *const __m128i);
            let lo = _mm_and_si128(by, m4);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
            let n0 = _mm_unpacklo_epi8(lo, hi);
            let n1 = _mm_unpackhi_epi8(lo, hi);
            let wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8);
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p, ones));
            i += 32;
        }
        let mut sum = hsum256(_mm256_add_epi32(acc0, acc1));
        if i < n {
            // `i` is a multiple of 32 (even), so element `i` is the low nibble of
            // byte `i/2` — slicing both w4 (by byte) and x (by element) stays aligned.
            sum += dot_i4i8_scalar(&w4[i >> 1..], &x[i..], n - i);
        }
        sum
    }

    /// packed-int2·int8 via AVX2. Unpacks 8 bytes → 32 fields in order: extract the
    /// four 2-bit fields (`&3`, `>>2&3`, `>>4&3`, `>>6&3`) into byte lanes, then a
    /// two-level `unpack` interleaves them so lane `4k+j` holds byte `k`'s field
    /// `j`; bias by −2 into `[-2,1]` and run the same `maddubs` sign-trick as int4.
    ///
    /// # Safety
    /// The CPU must support AVX2. [`super::dot_i2i8`] verifies this before calling.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i2i8_avx2(w2: &[u8], x: &[i8], n: usize) -> i32 {
        let m2 = _mm_set1_epi8(0x03);
        let b2 = _mm256_set1_epi8(2);
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 32 <= n {
            let by = _mm_loadl_epi64(w2.as_ptr().add(i >> 2) as *const __m128i); // 8B = 32 fields
            // f_j[k] = field j of byte k (>>2 clears higher bits before the mask)
            let f0 = _mm_and_si128(by, m2);
            let f1 = _mm_and_si128(_mm_srli_epi16::<2>(by), m2);
            let f2 = _mm_and_si128(_mm_srli_epi16::<4>(by), m2);
            let f3 = _mm_and_si128(_mm_srli_epi16::<6>(by), m2);
            // interleave to [f0[0],f1[0],f2[0],f3[0], f0[1],...] = elements in order
            let lo01 = _mm_unpacklo_epi8(f0, f1);
            let lo23 = _mm_unpacklo_epi8(f2, f3);
            let r_lo = _mm_unpacklo_epi16(lo01, lo23); // elements 0..15
            let r_hi = _mm_unpackhi_epi16(lo01, lo23); // elements 16..31
            let wv = _mm256_sub_epi8(_mm256_set_m128i(r_hi, r_lo), b2); // → [-2,1]
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
            i += 32;
        }
        let mut sum = hsum256(acc);
        if i < n {
            // `i` is a multiple of 32, so element `i` is field 0 of byte `i/4`.
            sum += dot_i2i8_scalar(&w2[i >> 2..], &x[i..], n - i);
        }
        sum
    }
}

/// **affine** int2-g64·int8 dot → f32. Reference implementation.
///
/// Each 64-value group is 16 bytes (four 2-bit fields per byte, the int2
/// layout) and carries **two** f32 in `scales`, interleaved `[s, z]`. Fields are
/// unsigned `[0, 3]`; the bias is the group's own zero-point, not a constant.
///
/// The affine form decomposes so the integer dot survives intact:
///
/// ```text
///   Σ (q·s + z)·x  =  s·Σ(q·x)  +  z·Σ(x)
/// ```
///
/// so a group needs the usual integer dot *plus* a plain sum of that group's
/// activations. That second accumulator is the whole cost of the zero-point —
/// no float enters until the per-group scale multiply, exactly as in
/// [`dot_i3i8_g64_scalar`].
pub fn dot_i2i8_g64_scalar(w2: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
    const GROUP: usize = 64;
    const GROUP_BYTES: usize = 16;
    let mut acc = 0f32;
    let mut start = 0usize;
    let mut g = 0usize;
    while start < n {
        let len = GROUP.min(n - start);
        let base = g * GROUP_BYTES;
        let mut d = 0i32; // Σ q·x
        let mut sx = 0i32; // Σ x
        for k in 0..len {
            let field = ((w2[base + (k >> 2)] >> (2 * (k & 3))) & 0x03) as i32;
            let xv = x[start + k] as i32;
            d += field * xv;
            sx += xv;
        }
        acc += scales[2 * g] * d as f32 + scales[2 * g + 1] * sx as f32;
        start += GROUP;
        g += 1;
    }
    acc
}

/// affine int2-g64·int8 dot → f32, using the best available inner kernel.
///
/// Bit-identical to [`dot_i2i8_g64_scalar`]: both accumulate the group's two
/// integer sums exactly and apply the same two float multiplies in the same
/// order. Gated by `int2_g64_avx2_matches_scalar`.
pub fn dot_i2i8_g64(w2: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the avx2 kernel is only called after detecting avx2 at runtime.
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86_i2g::dot_i2i8_g64_avx2(w2, x, scales, n) };
        }
    }
    dot_i2i8_g64_scalar(w2, x, scales, n)
}

#[cfg(target_arch = "x86_64")]
mod x86_i2g {
    use super::x86::hsum256;
    use std::arch::x86_64::*;

    const GROUP: usize = 64;
    const GROUP_BYTES: usize = 16;

    /// `(Σ q·x, Σ x)` over 32 consecutive elements starting at packed byte
    /// `wp` and activation pointer `xp`.
    ///
    /// The unsigned field layout is what makes this simpler than the biased
    /// per-row int2 kernel: `q ∈ [0,3]` is already a valid unsigned operand for
    /// `maddubs`, so the sign-shuffle dance that kernel needs is unnecessary
    /// here. Products stay in `[-384, 381]` and pairwise sums in `[-768, 762]`,
    /// well inside i16.
    ///
    /// # Safety
    /// AVX2 required; `wp` must have 8 readable bytes and `xp` 32.
    #[target_feature(enable = "avx2")]
    unsafe fn block32(wp: *const u8, xp: *const i8) -> (__m256i, __m256i) {
        let m2 = _mm_set1_epi8(0x03);
        let ones16 = _mm256_set1_epi16(1);
        let by = _mm_loadl_epi64(wp as *const __m128i); // 8 B = 32 fields
        let f0 = _mm_and_si128(by, m2);
        let f1 = _mm_and_si128(_mm_srli_epi16::<2>(by), m2);
        let f2 = _mm_and_si128(_mm_srli_epi16::<4>(by), m2);
        let f3 = _mm_and_si128(_mm_srli_epi16::<6>(by), m2);
        let lo01 = _mm_unpacklo_epi8(f0, f1);
        let lo23 = _mm_unpacklo_epi8(f2, f3);
        let r_lo = _mm_unpacklo_epi16(lo01, lo23); // elements 0..15
        let r_hi = _mm_unpackhi_epi16(lo01, lo23); // elements 16..31
        let wv = _mm256_set_m128i(r_hi, r_lo); // unsigned [0,3], in order
        let xv = _mm256_loadu_si256(xp as *const __m256i);
        // Σ q·x — q is the unsigned operand, x the signed one.
        let d = _mm256_madd_epi16(_mm256_maddubs_epi16(wv, xv), ones16);
        // Σ x — same shape with every weight forced to 1.
        let sx = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), xv), ones16);
        (d, sx)
    }

    /// # Safety
    /// AVX2 required. [`super::dot_i2i8_g64`] verifies this before calling.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i2i8_g64_avx2(w2: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
        let mut acc = 0f32;
        let mut start = 0usize;
        let mut g = 0usize;
        while start < n {
            let len = GROUP.min(n - start);
            let base = g * GROUP_BYTES;
            let (d, sx) = if len == GROUP {
                // A group is exactly two 32-element blocks, so the vector loop
                // never straddles a group boundary and the per-group scale stays
                // exact.
                let (d0, s0) = block32(w2.as_ptr().add(base), x.as_ptr().add(start));
                let (d1, s1) = block32(w2.as_ptr().add(base + 8), x.as_ptr().add(start + 32));
                (hsum256(_mm256_add_epi32(d0, d1)), hsum256(_mm256_add_epi32(s0, s1)))
            } else {
                // Ragged final group: fall back rather than masking. Integer
                // sums are order-independent, so this is still bit-identical.
                let mut d = 0i32;
                let mut sxs = 0i32;
                for k in 0..len {
                    let field = ((w2[base + (k >> 2)] >> (2 * (k & 3))) & 0x03) as i32;
                    let xv = x[start + k] as i32;
                    d += field * xv;
                    sxs += xv;
                }
                (d, sxs)
            };
            acc += scales[2 * g] * d as f32 + scales[2 * g + 1] * sx as f32;
            start += GROUP;
            g += 1;
        }
        acc
    }
}

/// int3-g64·int8 dot → f32 (colibrì's fmt 5).
///
/// Each 64-value group is 24 bytes — a 16-byte low plane holding bits 0-1 in the
/// int2 layout, then an 8-byte high plane holding bit 2 one bit per value — plus
/// one f32 scale. Values are stored biased `+4`, so they decode to `[-4, 3]`.
///
/// The integer dot is accumulated per group and scaled once, the same shape as
/// [`dot_i4i8_grouped`]: per-group scales cannot be factored out of a single
/// integer accumulator. colibrì has no int8-activation path for this format
/// ("int3 has no IDOT in v1: int8 activations don't compose with per-group
/// accumulation without a kernel restructure") — that restructure is what
/// `dot_i4i8_grouped` already is here, so int3 composes with it for free.
pub fn dot_i3i8_g64_scalar(w3: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
    const GROUP: usize = 64;
    const LOW_BYTES: usize = 16;
    const GROUP_BYTES: usize = 24;
    let mut acc = 0f32;
    let mut start = 0usize;
    let mut g = 0usize;
    while start < n {
        let len = GROUP.min(n - start);
        let base = g * GROUP_BYTES;
        let mut d = 0i32;
        for k in 0..len {
            let lo = w3[base + (k >> 2)];
            let hi = w3[base + LOW_BYTES + (k >> 3)];
            let low = ((lo >> (2 * (k & 3))) & 0x03) as i32;
            let high = ((hi >> (k & 7)) & 0x01) as i32;
            d += ((low | (high << 2)) - 4) * x[start + k] as i32;
        }
        acc += scales[g] * d as f32;
        start += GROUP;
        g += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    #[test]
    fn int2_g64_avx2_matches_scalar() {
        // Same contract as int3-g64 below: the scalar kernel is the reference
        // and the SIMD variant must equal it bit-for-bit, not within a
        // tolerance. Both integer accumulators are exact and the two float
        // multiplies happen in the same order, so equality is achievable.
        //
        // Packing is done here rather than through `quant_i2_g64`, so this pits
        // the two kernels against each other rather than against a shared
        // producer bug.
        fn pack(codes: &[u8]) -> Vec<u8> {
            let ng = codes.len().div_ceil(64);
            let mut q = vec![0u8; ng * 16];
            for (k, &u) in codes.iter().enumerate() {
                let (g, j) = (k / 64, k % 64);
                q[g * 16 + (j >> 2)] |= (u & 0x03) << (2 * (j & 3));
            }
            q
        }
        // Widths that exercise a whole group, several groups, and every ragged
        // remainder class around the 32-element vector block.
        for n in [64usize, 128, 192, 200, 63, 65, 32, 31, 1] {
            let codes: Vec<u8> = (0..n).map(|k| (k * 7 % 4) as u8).collect();
            let q = pack(&codes);
            let ng = n.div_ceil(64);
            // Interleaved [scale, zero]; zeros deliberately non-zero and signed,
            // since a zero-point of 0 would hide the whole Σx term.
            let sc: Vec<f32> = (0..ng)
                .flat_map(|g| [0.125 + g as f32 * 0.5, -0.75 + g as f32 * 0.25])
                .collect();
            // Activations spanning the i8 extremes: the Σx accumulator is where
            // a saturating or unsigned mistake would show up.
            let x: Vec<i8> = (0..n).map(|k| ((((k * 37) % 255) as i32) - 128) as i8).collect();
            let want = dot_i2i8_g64_scalar(&q, &x, &sc, n);
            let got = dot_i2i8_g64(&q, &x, &sc, n);
            assert_eq!(want.to_bits(), got.to_bits(), "n={n}: avx2 must be bit-identical to scalar");
        }
    }

    #[test]
    fn int2_g64_dot_matches_dequantized_reference() {
        // The kernel and the container decoder must agree: quantize, then check
        // the kernel against a plain f32 dot over what `QtView` says the bytes
        // mean. This is the pairing that would catch a bias or ordering mismatch
        // between producer and consumer, which no kernel-vs-kernel test can see.
        let n = 128usize;
        let codes: Vec<u8> = (0..n).map(|k| (k * 3 % 4) as u8).collect();
        let ng = n.div_ceil(64);
        let mut q = vec![0u8; ng * 16];
        for (k, &u) in codes.iter().enumerate() {
            q[(k / 64) * 16 + ((k % 64) >> 2)] |= (u & 0x03) << (2 * ((k % 64) & 3));
        }
        let sc: Vec<f32> = (0..ng).flat_map(|g| [0.5 + g as f32, -1.25 * (g + 1) as f32]).collect();
        let x: Vec<i8> = (0..n).map(|k| ((k % 17) as i32 - 8) as i8).collect();

        let mut want = 0f64;
        for k in 0..n {
            let g = k / 64;
            let w = codes[k] as f64 * sc[2 * g] as f64 + sc[2 * g + 1] as f64;
            want += w * x[k] as f64;
        }
        let got = dot_i2i8_g64(&q, &x, &sc, n) as f64;
        assert!((got - want).abs() < 1e-3, "kernel {got} vs dequantized reference {want}");
    }

    #[test]
    fn int3_g64_avx2_matches_scalar() {
        // Every format here ships a scalar reference and a SIMD variant checked
        // bit-for-bit against it (docs/testing-and-quality.md). int3 shipped with
        // only the scalar half, so an int3 checkpoint computed slower than the
        // int4 it replaced. The per-group integer dot is exact in both paths, so
        // equality here is bit-equality, not a tolerance.
        //
        // The packing is done here rather than via the encoder, so this tests the
        // two kernels against each other and not against a shared producer bug.
        fn pack(codes: &[u8]) -> Vec<u8> {
            let ng = codes.len().div_ceil(64);
            let mut q = vec![0u8; ng * 24];
            for (k, &u) in codes.iter().enumerate() {
                let (g, j) = (k / 64, k % 64);
                let base = g * 24;
                q[base + (j >> 2)] |= (u & 0x03) << (2 * (j & 3));
                q[base + 16 + (j >> 3)] |= ((u >> 2) & 0x01) << (j & 7);
            }
            q
        }
        for n in [64usize, 128, 200, 63, 65, 32] {
            // every code in [0,7] appears, so both planes vary
            let codes: Vec<u8> = (0..n).map(|k| (k * 5 % 8) as u8).collect();
            let q = pack(&codes);
            let ng = n.div_ceil(64);
            let sc: Vec<f32> = (0..ng).map(|g| 0.125 + g as f32 * 0.5).collect();
            let x: Vec<i8> = (0..n).map(|k| ((k * 53 % 255) as i32 - 127) as i8).collect();
            let want = dot_i3i8_g64_scalar(&q, &x, &sc, n);
            let got = dot_i3i8_g64(&q, &x, &sc, n);
            assert_eq!(want.to_bits(), got.to_bits(), "n={n}: SIMD path diverged from the reference");
        }
    }

    use super::*;

    // Deterministic LCG so tests need no rng crate and no Date/random.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn i8_full(&mut self) -> i8 {
            (self.next() >> 33) as i8
        }
        fn i8_127(&mut self) -> i8 {
            ((self.next() >> 33) as i32 % 255 - 127) as i8
        }
        fn nib(&mut self) -> i32 {
            (self.next() >> 40) as i32 % 16 - 8 // [-8,7]
        }
    }

    fn pack_i4(vals: &[i32]) -> Vec<u8> {
        let rb = vals.len().div_ceil(2);
        let mut out = vec![0u8; rb];
        for (i, &v) in vals.iter().enumerate() {
            let bias = (v + 8) as u8 & 0x0F;
            if i & 1 == 0 {
                out[i >> 1] |= bias;
            } else {
                out[i >> 1] |= bias << 4;
            }
        }
        out
    }

    // Lengths chosen to exercise the SIMD body plus every tail remainder.
    const LENS: [usize; 12] = [1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 257];

    #[test]
    fn i8_simd_matches_scalar() {
        let mut rng = Lcg(0x1234_5678);
        for &n in &LENS {
            let w: Vec<i8> = (0..n).map(|_| rng.i8_full()).collect();
            let x: Vec<i8> = (0..n).map(|_| rng.i8_127()).collect();
            let reference = dot_i8i8_scalar(&w, &x, n);
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { x86::dot_i8i8_avx2(&w, &x, n) }, reference, "avx2 n={n}");
                }
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni") {
                    assert_eq!(unsafe { x86::dot_i8i8_vnni(&w, &x, n) }, reference, "vnni n={n}");
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                // Both ARM kernels, not just the one dispatch picks: a box
                // without `dotprod` runs the `smull` path and nothing else would
                // ever check it.
                if std::arch::is_aarch64_feature_detected!("neon") {
                    assert_eq!(unsafe { aarch64::dot_i8i8_neon(&w, &x, n) }, reference, "neon n={n}");
                }
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    assert_eq!(unsafe { aarch64::dot_i8i8_dotprod(&w, &x, n) }, reference, "sdot n={n}");
                }
            }
            assert_eq!(dot_i8i8(&w, &x, n), reference, "dispatch n={n}");
        }
    }

    #[test]
    fn i4_grouped_scalar_matches_dispatch_and_hand() {
        let mut rng = Lcg(0xdead_beef);
        // (n, gs) pairs: exact multiples and ragged tails, gs>n, single group.
        for &(n, gs) in &[(32usize, 16usize), (96, 32), (128, 128), (70, 16), (10, 16), (256, 64)] {
            let vals: Vec<i32> = (0..n).map(|_| rng.nib()).collect();
            let w4 = pack_i4(&vals);
            let x: Vec<i8> = (0..n).map(|_| rng.i8_127()).collect();
            let ng = n.div_ceil(gs);
            let scales: Vec<f32> = (0..ng).map(|g| 0.01 + 0.05 * g as f32).collect();

            // hand reference: Σ_g scale[g] · Σ_{i in g} val[i]·x[i]
            let mut hand = 0f32;
            for (g, &scale) in scales.iter().enumerate().take(ng) {
                let (s, e) = (g * gs, ((g + 1) * gs).min(n));
                let d: i32 = (s..e).map(|i| vals[i] * x[i] as i32).sum();
                hand += scale * d as f32;
            }
            let scal = dot_i4i8_grouped_scalar(&w4, &x, &scales, n, gs);
            let disp = dot_i4i8_grouped(&w4, &x, &scales, n, gs);
            assert_eq!(scal, hand, "grouped scalar n={n} gs={gs}");
            assert_eq!(disp, scal, "grouped dispatch==scalar n={n} gs={gs}");
        }
    }

    #[test]
    fn i2_scalar_matches_hand() {
        let mut rng = Lcg(0x2222_2222);
        for &n in &LENS {
            let vals: Vec<i32> = (0..n).map(|_| (rng.next() >> 40) as i32 % 4 - 2).collect(); // [-2,1]
            // pack 4 fields/byte, field i in bits [2*(i&3) .. +2)
            let mut w2 = vec![0u8; n.div_ceil(4)];
            for (i, &v) in vals.iter().enumerate() {
                w2[i >> 2] |= (((v + 2) as u8) & 0x03) << (2 * (i & 3));
            }
            let x: Vec<i8> = (0..n).map(|_| rng.i8_127()).collect();
            let hand: i32 = vals.iter().zip(&x).map(|(&v, &xi)| v * xi as i32).sum();
            assert_eq!(dot_i2i8_scalar(&w2, &x, n), hand, "int2 n={n}");
        }
    }

    #[test]
    fn i2_simd_matches_scalar() {
        let mut rng = Lcg(0x1357_2468);
        for &n in &LENS {
            let vals: Vec<i32> = (0..n).map(|_| (rng.next() >> 40) as i32 % 4 - 2).collect(); // [-2,1]
            let mut w2 = vec![0u8; n.div_ceil(4)];
            for (i, &v) in vals.iter().enumerate() {
                w2[i >> 2] |= (((v + 2) as u8) & 0x03) << (2 * (i & 3));
            }
            let x: Vec<i8> = (0..n).map(|_| rng.i8_127()).collect();
            let reference = dot_i2i8_scalar(&w2, &x, n);
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { x86::dot_i2i8_avx2(&w2, &x, n) }, reference, "avx2 i2 n={n}");
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    assert_eq!(unsafe { aarch64::dot_i2i8_dotprod(&w2, &x, n) }, reference, "sdot i2 n={n}");
                }
            }
            assert_eq!(dot_i2i8(&w2, &x, n), reference, "dispatch i2 n={n}");
        }
    }

    #[test]
    fn i4_simd_matches_scalar() {
        let mut rng = Lcg(0x9e37_79b9);
        for &n in &LENS {
            let vals: Vec<i32> = (0..n).map(|_| rng.nib()).collect();
            let w4 = pack_i4(&vals);
            let x: Vec<i8> = (0..n).map(|_| rng.i8_127()).collect();
            // scalar sanity: unpack must reproduce the packed values
            let reference = dot_i4i8_scalar(&w4, &x, n);
            let hand: i32 = vals.iter().zip(&x).map(|(&v, &xi)| v * xi as i32).sum();
            assert_eq!(reference, hand, "pack/unpack n={n}");
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { x86::dot_i4i8_avx2(&w4, &x, n) }, reference, "avx2 i4 n={n}");
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    assert_eq!(unsafe { aarch64::dot_i4i8_neon(&w4, &x, n) }, reference, "neon i4 n={n}");
                }
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    assert_eq!(unsafe { aarch64::dot_i4i8_dotprod(&w4, &x, n) }, reference, "sdot i4 n={n}");
                }
            }
            assert_eq!(dot_i4i8(&w4, &x, n), reference, "dispatch i4 n={n}");
        }
    }
}

/// int3-g64·int8 dot → f32, using the best available inner kernel.
///
/// Bit-identical to [`dot_i3i8_g64_scalar`]: the per-group integer dot is exact
/// in both paths (no float rounding until the per-group scale multiply, which is
/// applied in the same order), so the SIMD path is a speed change only. Gated by
/// `int3_g64_avx2_matches_scalar`.
pub fn dot_i3i8_g64(w3: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the avx2 kernel is only called after detecting avx2 at runtime.
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86_i3::dot_i3i8_g64_avx2(w3, x, scales, n) };
        }
    }
    dot_i3i8_g64_scalar(w3, x, scales, n)
}

#[cfg(target_arch = "x86_64")]
mod x86_i3 {
    use super::dot_i3i8_g64_scalar;
    use super::x86::hsum256;
    use std::arch::x86_64::*;

    const GROUP: usize = 64;
    const LOW_BYTES: usize = 16;
    const GROUP_BYTES: usize = 24;

    /// One full 64-value group as two 32-value halves.
    ///
    /// The low plane is byte-for-byte the int2 layout, so its unpack is the same
    /// shuffle `dot_i2i8_avx2` uses. The high plane is one bit per value: four
    /// bytes cover 32 values, so each source byte is broadcast across eight lanes
    /// and tested against a per-lane bit mask, yielding 4 where the bit is set.
    /// `low | high` reproduces the biased `[0,7]` code, and subtracting 4 gives
    /// the stored `[-4,3]`.
    ///
    /// # Safety
    /// Caller has verified AVX2. `w3` must hold a whole 24-byte group at `base`
    /// and `x` 64 readable elements at `xoff`.
    #[target_feature(enable = "avx2")]
    unsafe fn group_dot(w3: &[u8], base: usize, x: &[i8], xoff: usize) -> i32 {
        let m2 = _mm_set1_epi8(0x03);
        let b4 = _mm256_set1_epi8(4);
        let ones = _mm256_set1_epi16(1);
        let bitmask = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128i8, 1, 2, 4, 8, 16, 32, 64, -128i8, 1, 2, 4, 8, 16, 32, 64,
            -128i8, 1, 2, 4, 8, 16, 32, 64, -128i8,
        );
        // byte k of the shuffle result takes source byte k/8, per 128-bit lane
        let spread = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        );
        let mut acc = _mm256_setzero_si256();
        for half in 0..2 {
            let k = half * 32; // first value of this half within the group
            // low plane: 8 bytes = 32 two-bit fields, identical to int2
            let by = _mm_loadl_epi64(w3.as_ptr().add(base + (k >> 2)) as *const __m128i);
            let f0 = _mm_and_si128(by, m2);
            let f1 = _mm_and_si128(_mm_srli_epi16::<2>(by), m2);
            let f2 = _mm_and_si128(_mm_srli_epi16::<4>(by), m2);
            let f3 = _mm_and_si128(_mm_srli_epi16::<6>(by), m2);
            let lo01 = _mm_unpacklo_epi8(f0, f1);
            let lo23 = _mm_unpacklo_epi8(f2, f3);
            let r_lo = _mm_unpacklo_epi16(lo01, lo23);
            let r_hi = _mm_unpackhi_epi16(lo01, lo23);
            let low = _mm256_set_m128i(r_hi, r_lo);

            // high plane: 4 bytes = 32 single bits
            let hp = w3.as_ptr().add(base + LOW_BYTES + (k >> 3)) as *const i32;
            let bcast = _mm256_shuffle_epi8(_mm256_set1_epi32(hp.read_unaligned()), spread);
            let isset = _mm256_cmpeq_epi8(_mm256_and_si256(bcast, bitmask), bitmask);
            let high = _mm256_and_si256(isset, b4);

            let wv = _mm256_sub_epi8(_mm256_or_si256(low, high), b4); // → [-4,3]
            let xv = _mm256_loadu_si256(x.as_ptr().add(xoff + k) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
        }
        hsum256(acc)
    }

    /// # Safety
    /// Caller has verified AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i3i8_g64_avx2(w3: &[u8], x: &[i8], scales: &[f32], n: usize) -> f32 {
        let mut acc = 0f32;
        let mut start = 0usize;
        let mut g = 0usize;
        while start < n {
            let len = GROUP.min(n - start);
            let base = g * GROUP_BYTES;
            // Only whole groups vectorize; a ragged final group falls back to the
            // reference, which is the same arithmetic.
            let d = if len == GROUP {
                group_dot(w3, base, x, start)
            } else {
                let tail = dot_i3i8_g64_scalar(&w3[base..], &x[start..], &scales[g..g + 1], len);
                // `dot_i3i8_g64_scalar` already applied the scale; undo the double-apply
                // by accumulating it directly and skipping the multiply below.
                acc += tail;
                start += GROUP;
                g += 1;
                continue;
            };
            acc += scales[g] * d as f32;
            start += GROUP;
            g += 1;
        }
        acc
    }
}

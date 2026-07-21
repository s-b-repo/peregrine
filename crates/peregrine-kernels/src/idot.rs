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

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 (maddubs sign-trick) and AVX-VNNI (`vpdpbusd`) dot kernels. Each
    //! processes vector-width chunks then falls through to a scalar tail, so any
    //! `n` is handled and the result equals the scalar reference exactly.
    use super::{dot_i4i8_scalar, dot_i8i8_scalar};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256i) -> i32 {
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
        let mut acc = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 32 <= n {
            let by = _mm_loadu_si128(w4.as_ptr().add(i >> 1) as *const __m128i); // 16B = 32 nibbles
            let lo = _mm_and_si128(by, m4);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(by), m4);
            let n0 = _mm_unpacklo_epi8(lo, hi); // elems 0..15 (in order)
            let n1 = _mm_unpackhi_epi8(lo, hi); // elems 16..31
            let wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8); // → [-8,7]
            let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
            i += 32;
        }
        let mut sum = hsum256(acc);
        if i < n {
            // `i` is a multiple of 32 (even), so element `i` is the low nibble of
            // byte `i/2` — slicing both w4 (by byte) and x (by element) stays aligned.
            sum += dot_i4i8_scalar(&w4[i >> 1..], &x[i..], n - i);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
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
            for g in 0..ng {
                let (s, e) = (g * gs, ((g + 1) * gs).min(n));
                let d: i32 = (s..e).map(|i| vals[i] * x[i] as i32).sum();
                hand += scales[g] * d as f32;
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
            assert_eq!(dot_i4i8(&w4, &x, n), reference, "dispatch i4 n={n}");
        }
    }
}

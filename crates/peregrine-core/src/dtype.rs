//! Storage dtypes and exact BF16/F16 → F32 conversion.
//!
//! Ported byte-for-byte from `c/st.h` (`st_dtype_code`, `bf16_to_f32`,
//! `f16_to_f32`) so the Rust loader reproduces the C engine's dequantization
//! bit-for-bit. The container uses U8/I8 for already-quantized int4/int8/int2
//! weights (see [`crate::qt`]).

/// Storage dtype of a safetensors tensor.
///
/// The numeric discriminants match `st_dtype_code` in `c/st.h`
/// (BF16=0, F16=1, F32=2, U8/I8=3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// bfloat16
    Bf16 = 0,
    /// IEEE float16
    F16 = 1,
    /// IEEE float32
    F32 = 2,
    /// raw bytes — quantized int4/int8/int2 container payloads
    U8 = 3,
    /// OCP FP8 E4M3 (bias 7, no infinities, S.1111.111 = NaN) — the dtype
    /// GLM-5.3-Flash ships its block-scaled expert weights in. Read-side only:
    /// `peregrine-import-hf` widens and dequantizes it; the container never
    /// stores it.
    F8E4M3 = 4,
    I64 = 5,
}

impl Dtype {
    /// Parse a safetensors `dtype` string. `I8` maps to `U8` (both are raw
    /// quantized bytes here), matching `st_dtype_code`. `None` for anything this
    /// engine does not support.
    ///
    /// The same parse is available as [`std::str::FromStr`] (which is what makes
    /// this inherent name legitimate rather than a shadow of the trait).
    pub fn parse(s: &str) -> Option<Dtype> {
        match s {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            "I64" => Some(Dtype::I64),
            "U8" | "I8" => Some(Dtype::U8),
            "F8_E4M3" => Some(Dtype::F8E4M3),
            _ => None,
        }
    }

    /// Bytes per element for the float dtypes; `U8` is 1.
    pub fn elem_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I64 => 8,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::U8 | Dtype::F8E4M3 => 1,
        }
    }
}

impl std::str::FromStr for Dtype {
    type Err = crate::Error;

    /// The safetensors spelling of a dtype (see [`Dtype::parse`]).
    fn from_str(s: &str) -> Result<Dtype, crate::Error> {
        Dtype::parse(s).ok_or_else(|| crate::Error::Format(format!("unsupported dtype: {s}")))
    }
}

/// bfloat16 → f32: place the 16 bits in the high half of the f32. Exact.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// OCP FP8 E4M3 → f32. Exact: every e4m3 value (bias 7, subnormals at
/// `man/8 · 2⁻⁶`, no infinities, `S.1111.111` = NaN — `S.1111.110` is the
/// ordinary finite ±448) is representable in f32.
#[inline]
pub fn f8e4m3_to_f32(b: u8) -> f32 {
    let sign = ((b & 0x80) as u32) << 24;
    let exp = ((b >> 3) & 0xF) as u32;
    let man = (b & 0x7) as u32;
    let u = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: renormalize into the f32 exponent range
            let mut e = 127 - 7 + 1;
            let mut m = man;
            while m & 0x8 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x7) << 20)
        }
    } else if exp == 0xF && man == 0x7 {
        sign | 0x7FC0_0000 // NaN
    } else {
        sign | ((exp + 120) << 23) | (man << 20)
    };
    f32::from_bits(u)
}

/// IEEE float16 → f32. Direct port of `f16_to_f32` in `c/st.h`, including the
/// subnormal-renormalization loop and inf/nan handling.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign: u32 = ((h & 0x8000) as u32) << 16;
    let mut exp: u32 = ((h >> 10) & 0x1F) as u32;
    let mut man: u32 = (h & 0x3FF) as u32;
    let u: u32 = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: renormalize into the f32 exponent range
            exp = 127 - 15 + 1;
            while man & 0x400 == 0 {
                man <<= 1;
                exp -= 1;
            }
            man &= 0x3FF;
            sign | (exp << 23) | (man << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        // `exp + 112`, not the C source's `exp - 15 + 127`. Both name the same
        // rebias, but `exp` is unsigned and every f16 below 1.0 has `exp < 15`,
        // so the C spelling underflows — defined wraparound in C and in a Rust
        // release build, an overflow panic in a debug one. That covered the
        // whole of [2^-14, 1), which no test reached until `f32_to_f16`'s
        // exhaustive round trip walked all 65 536 encodings.
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(u)
}

/// f32 → IEEE float16, round-to-nearest-even — the inverse of [`f16_to_f32`].
///
/// The container never needed this (it only ever *reads* half-precision), but
/// the KV cache stores what the engine computes, so narrowing it needs an
/// encoder. Written to the IEEE rule rather than the convenient one: truncation
/// would bias every stored latent toward zero, and a KV cache is read thousands
/// of times per sequence, so a systematic bias compounds where a rounding error
/// cancels.
///
/// Values past f16's range saturate to ±inf rather than wrapping, subnormals
/// are encoded exactly (not flushed), and NaN stays NaN.
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let man = x & 0x007F_FFFF;
    // f16 exponent, rebiased from f32's 127 to f16's 15.
    let exp = (((x >> 23) & 0xFF) as i32) - 112;

    if exp == 0xFF - 112 {
        // inf, or NaN — which must not round down into an infinity.
        return sign | 0x7C00 | if man != 0 { ((man >> 13) as u16).max(1) } else { 0 };
    }
    if exp >= 0x1F {
        return sign | 0x7C00; // beyond f16's largest finite value
    }
    if exp <= 0 {
        // Subnormal: no implicit leading 1 in the result, so shift the f32's
        // explicit significand down to the fixed 2^-24 grid.
        if exp < -10 {
            return sign; // below half of the smallest subnormal
        }
        let m = man | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let h = (m >> shift) as u16;
        let round = 1u32 << (shift - 1);
        let up = (m & round) != 0 && ((m & (round - 1)) != 0 || (h & 1) != 0);
        return sign | (h + u16::from(up));
    }
    // Normal. A mantissa carry propagates into the exponent, and out of the top
    // into an infinity, which is the correct result in both cases.
    let h = ((exp as u16) << 10) | ((man >> 13) as u16);
    let up = (man & 0x1000) != 0 && ((man & 0x0FFF) != 0 || (h & 1) != 0);
    sign | (h + u16::from(up))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_encode_is_the_exact_inverse_where_it_can_be() {
        // Every value f16 can represent must survive a round trip untouched.
        for bits in 0u32..=0xFFFF {
            let h = bits as u16;
            if (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0 {
                continue; // NaN payloads are not required to be preserved
            }
            let back = f32_to_f16(f16_to_f32(h));
            // -0.0 and +0.0 both round-trip to their own sign.
            assert_eq!(back, h, "f16 {h:#06x} did not survive the round trip");
        }
    }

    #[test]
    fn f16_encode_rounds_to_nearest_even_and_saturates() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        // Smallest positive subnormal, and half of it (ties-to-even → 0).
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.5), 0x0001);
        // Past the largest finite f16 (65504) saturates rather than wrapping.
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(1e6), 0x7C00);
        assert_eq!(f32_to_f16(-1e6), 0xFC00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan(), "NaN must not round into an infinity");
        // Midway between two representable values rounds to the even one, both ways.
        let (a, b) = (f16_to_f32(0x3C00), f16_to_f32(0x3C01)); // 1.0 and its successor
        assert_eq!(f32_to_f16((a + b) * 0.5), 0x3C00, "tie below rounds to even");
        let c = f16_to_f32(0x3C02);
        assert_eq!(f32_to_f16((b + c) * 0.5), 0x3C02, "tie above rounds to even");
        // f32 subnormals are far below f16's range and must not produce garbage.
        assert_eq!(f32_to_f16(f32::from_bits(1)), 0x0000);
    }

    #[test]
    fn bf16_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0); // 1.0
        assert_eq!(bf16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(bf16_to_f32(0xBF80), -1.0); // -1.0
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f16_exact() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(f16_to_f32(0xC000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // smallest positive subnormal: 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn dtype_parse() {
        assert_eq!(Dtype::parse("BF16"), Some(Dtype::Bf16));
        assert_eq!(Dtype::parse("I8"), Some(Dtype::U8));
        assert_eq!(Dtype::parse("F8_E4M3"), Some(Dtype::F8E4M3));
        assert_eq!(Dtype::parse("F64"), None);
        assert_eq!(Dtype::Bf16 as i32, 0);
        assert_eq!(Dtype::U8 as i32, 3);
    }

    #[test]
    fn f8e4m3_decodes_the_reference_values() {
        // Spot values from the OCP FP8 spec.
        assert_eq!(f8e4m3_to_f32(0x00), 0.0);
        assert_eq!(f8e4m3_to_f32(0x80), -0.0);
        assert_eq!(f8e4m3_to_f32(0x38), 1.0); // exp 7, man 0
        assert_eq!(f8e4m3_to_f32(0xC0), -2.0);
        assert_eq!(f8e4m3_to_f32(0x7E), 448.0); // S.1111.110: largest finite
        assert_eq!(f8e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal: 1/8 · 2⁻⁶
        assert_eq!(f8e4m3_to_f32(0x07), 7.0 / 8.0 * 2f32.powi(-6)); // largest subnormal
        assert_eq!(f8e4m3_to_f32(0x08), 2f32.powi(-6)); // smallest normal
        assert!(f8e4m3_to_f32(0x7F).is_nan() && f8e4m3_to_f32(0xFF).is_nan());
        // Every non-NaN encoding is finite (e4m3 has no infinities), and the
        // decode is monotone over the positive range.
        let mut prev = -1.0f32;
        for b in 0u8..0x7F {
            let v = f8e4m3_to_f32(b);
            assert!(v.is_finite(), "{b:#04x} decoded to non-finite {v}");
            assert!(v > prev, "{b:#04x}: {v} not monotone after {prev}");
            prev = v;
        }
    }
}

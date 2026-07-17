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
}

impl Dtype {
    /// Parse a safetensors `dtype` string. `I8` maps to `U8` (both are raw
    /// quantized bytes here), matching `st_dtype_code`. Returns `Option` (not
    /// `Result`), so this is deliberately not the `FromStr` trait method.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Dtype> {
        match s {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            "U8" | "I8" => Some(Dtype::U8),
            _ => None,
        }
    }

    /// Bytes per element for the float dtypes; `U8` is 1.
    pub fn elem_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::U8 => 1,
        }
    }
}

/// bfloat16 → f32: place the 16 bits in the high half of the f32. Exact.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
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
        sign | ((exp - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(u)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(Dtype::from_str("BF16"), Some(Dtype::Bf16));
        assert_eq!(Dtype::from_str("I8"), Some(Dtype::U8));
        assert_eq!(Dtype::from_str("F64"), None);
        assert_eq!(Dtype::Bf16 as i32, 0);
        assert_eq!(Dtype::U8 as i32, 3);
    }
}

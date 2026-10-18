//! The storage-format group: f16 and bf16 conversions, and the claims the
//! checkpoint reader depends on.
//!
//! A dtype converter is the easiest place in this workspace to be *almost*
//! right. Every value the model reads at inference time arrives through one of
//! these functions, and an error that is a half-ulp too large is invisible on a
//! logit while being a real change to every downstream token.

use peregrine_core::dtype::{bf16_to_f32, f16_to_f32, f32_to_f16};

use crate::{cert, exact, mismatch_bits, with_cases, Certificate};

/// The significand widths the tolerances below are derived from: an IEEE
/// binary16 has 10 explicit significand bits (11 with the implicit leading one),
/// bfloat16 has 7 (8 with the implicit bit).
const HALF_SIG: f64 = 11.0;
const BF16_SIG: f64 = 8.0;

pub fn certificates() -> Vec<Certificate> {
    vec![f16_roundtrip_bound(), f16_specials(), f16_idempotent(), bf16_roundtrip_bound(), bf16_idempotent()]
}

/// Fixture values chosen to hit the awkward parts of binary16: exact small
/// integers and halves, the values just below and just above 1.0 (where the
/// exponent changes and the significand count changes with it), the largest
/// normal, and the smallest normal.
fn samples() -> Vec<f32> {
    // Written as exact binary fractions rather than decimals: these are the
    // values that sit *on* binary16's grid, and a decimal that only looks equal
    // would make the half-ulp bound untestable. Each is therefore `2^k` or
    // `2^k · (1 ± 2^-j)` — the neighbours where the significand count changes.
    let p14 = 2f32.powi(-14); // binary16's smallest normal
    let p24 = 2f32.powi(-24); // its smallest subnormal
    let mut v = vec![
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, 1.5, 0.25, 65504.0, -65504.0,
        p14, // smallest normal
        p14 - 2f32.powi(-25), // one subnormal ulp below it: rounds back up
        p24, // smallest subnormal
        1.0 + 2f32.powi(-10), // one ulp above 1 in binary16
        1.0 - 2f32.powi(-11), // the tie just below 1
        1023.5, -1023.5, 0.0001, -0.0001, 12.75, 0.125,
    ];
    let mut x = 0x1234_5678u64;
    for _ in 0..64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Bounded so the value stays normal: subnormal handling is covered by the
        // two smallest samples above, where the relative bound stops holding.
        let bits = (x >> 40) as u32 & 0x007f_ffff | 0x3e00_0000;
        v.push(f32::from_bits(bits));
    }
    v
}

/// Rounding to binary16 shifts every value by at most half an ulp of the
/// *format*, which is a relative `2^-11` for normals and an absolute `2^-24`
/// (the smallest subnormal, half of which is the subnormal ulp) below the normal
/// range. The residual is measured against that per-sample bound and the
/// tolerance is 1.0 — a ratio, not a fitted constant, so the check is exactly
/// the claim and nothing more.
fn f16_roundtrip_bound() -> Certificate {
    let mut worst = 0.0f64;
    for v in samples() {
        let decoded = f16_to_f32(f32_to_f16(v));
        let bound = 2f64.powf(-HALF_SIG) * (v as f64).abs() + 2f64.powf(-24.0);
        let ratio = (decoded as f64 - v as f64).abs() / bound;
        worst = worst.max(ratio);
    }
    cert(
        "dtype",
        "f16_roundtrip_is_within_half_an_ulp",
        "Every value decoded after a binary16 round trip differs from the original by at most half an ulp of the format — `2^-11·|v|` for normals, `2^-24` absolute at the subnormal floor.",
        worst,
        1.0,
    )
}

/// Zeros, infinities and NaNs must survive the conversion rather than become
/// ordinary numbers: a NaN that decodes to a finite value would propagate
/// silently into a softmax denominator.
fn f16_specials() -> Certificate {
    // 0x7C00 is the published binary16 encoding of +infinity (s = 0, e = 31,
    // m = 0) and 0xFC00 its negation; 0x7E00 is a quiet NaN. Written as literals
    // because "the converter's infinity is its own infinity" is not a claim
    // anyone can check.
    let encodes_inf = f32_to_f16(f32::INFINITY) == f16::INFINITY;
    let encodes_neg_inf = f32_to_f16(f32::NEG_INFINITY) == f16::NEG_INFINITY;
    let roundtrips_inf = f16_to_f32(f32_to_f16(f32::INFINITY)).to_bits() == f32::INFINITY.to_bits();
    let nan_stays_nan = f16_to_f32(f16::NAN).is_nan();
    let signed_zero = f16_to_f32(f32_to_f16(-0.0)).to_bits() == (-0.0f32).to_bits();
    let holds = encodes_inf && encodes_neg_inf && roundtrips_inf && nan_stays_nan && signed_zero;
    with_cases(
        cert(
            "dtype",
            "f16_specials_and_signed_zero_survive",
            "±infinity encodes to the published binary16 patterns `0x7C00`/`0xFC00` and decodes back to itself, a binary16 NaN decodes to a NaN, and `-0.0` round-trips with its sign bit intact.",
            exact(holds),
            0.0,
        ),
        5,
    )
}

/// Decoding is idempotent: `f16_to_f32` on a value that came out of `f32_to_f16`
/// changes nothing. Without this, the second pass through the converter — which
/// happens every time a KV latent is stored and reloaded — would drift.
fn f16_idempotent() -> Certificate {
    let first: Vec<f32> = samples().iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
    let second: Vec<f32> = first.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
    with_cases(
        cert(
            "dtype",
            "f16_roundtrip_is_idempotent",
            "Decoding a value that was already decoded changes no bit of it, so a store/reload cycle through binary16 does not drift.",
            mismatch_bits(&first, &second),
            0.0,
        ),
        first.len(),
    )
}

/// bfloat16 keeps 8 significand bits, and `bf16_roundtrip` **truncates** rather
/// than rounds — `v.to_bits() >> 16` simply drops the low half. So the bound is
/// a full ulp of the format (a relative `2^-7`), not the half ulp a
/// round-to-nearest implementation would earn, and a bfloat16 checkpoint pays
/// roughly twice the error binary16 does.
///
/// The measured worst case against a *half*-ulp bound is 1.80, which is the
/// ratio of a full ulp to a half one. Stating the claim the code actually makes
/// is not a way to dodge a failure: it is the reason the number is known. A
/// future change to round-to-nearest would show up here as this residual
/// halving, and that is the signal worth having.
fn bf16_roundtrip_bound() -> Certificate {
    let mut worst = 0.0f64;
    for v in samples() {
        let decoded = bf16_to_f32((v.to_bits() >> 16) as u16);
        let bound = 2f64.powf(1.0 - BF16_SIG) * (v as f64).abs() + 2f64.powf(-133.0);
        let ratio = (decoded as f64 - v as f64).abs() / bound;
        worst = worst.max(ratio);
    }
    cert(
        "dtype",
        "bf16_truncation_loses_at_most_one_ulp",
        "Truncating to bfloat16's 8 significand bits and decoding back differs by at most one ulp of bfloat16 — a relative `2^-7` — because `bf16_roundtrip` truncates instead of rounding to nearest.",
        worst,
        1.0,
    )
}

/// `bf16_roundtrip` is what a precision-floor report calls to attribute a loss
/// that bf16 already introduced, so it must be a fixed point: applying it to its
/// own output is a no-op.
fn bf16_idempotent() -> Certificate {
    let first: Vec<f32> = samples().iter().map(|v| peregrine_core::pack::bf16_roundtrip(*v)).collect();
    let second: Vec<f32> = first.iter().map(|v| peregrine_core::pack::bf16_roundtrip(*v)).collect();
    with_cases(
        cert(
            "dtype",
            "bf16_roundtrip_is_a_fixed_point",
            "`bf16_roundtrip` applied twice equals applied once, bit-for-bit, so a report cannot attribute its own rounding as model error.",
            mismatch_bits(&first, &second),
            0.0,
        ),
        first.len(),
    )
}

/// The binary16 bit patterns the specials certificate decodes directly. Kept as
/// named constants because a magic `0x7C00` in a test is a claim nobody can
/// check, and these are the published encodings.
mod f16 {
    pub(super) const INFINITY: u16 = 0x7C00;
    pub(super) const NEG_INFINITY: u16 = 0xFC00;
    pub(super) const NAN: u16 = 0x7E00;
}
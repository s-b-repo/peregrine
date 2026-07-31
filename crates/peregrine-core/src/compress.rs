//! Optional zstd compression for on-disk and in-memory expert payloads.
//!
//! Correctness-neutral: the compressed representation round-trips byte-for-byte
//! to the raw quantized bytes. Whether or not compression is used only affects
//! disk footprint and RAM cache density.
//!
//! Two directions:
//!
//! - [`encode`] — compress a raw payload (quantized weight bytes) with a chosen
//!   `level`. Higher levels shrink more but take longer; the packer picks a level
//!   once (default 3, matched to inference-time decode cost) and stamps it into
//!   the safetensors header alongside the tensor entry.
//! - [`decode`] — decompress into a preallocated buffer of the known original
//!   size. Fails on trailing garbage or truncation.

/// The on-disk / in-memory compression scheme used for a payload. Kept as a
/// small tagged type so future codecs (LZ4, Snappy) can slot in as variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    /// No compression — raw bytes.
    None,
    /// Zstandard, standard framed encoding.
    Zstd,
}

impl Compression {
    /// Parse the string tag used in safetensors headers ("zstd" → Zstd; absent
    /// or unknown → None).
    pub fn from_tag(s: Option<&str>) -> Compression {
        match s {
            Some("zstd") => Compression::Zstd,
            _ => Compression::None,
        }
    }

    /// The header tag for this scheme. `None` when uncompressed (so the tag can
    /// be omitted from the JSON entry rather than encoded as `"none"`).
    pub fn tag(&self) -> Option<&'static str> {
        match self {
            Compression::None => None,
            Compression::Zstd => Some("zstd"),
        }
    }
}

/// Compress `raw` with the given scheme. `level` is the zstd compression level
/// (1..=22 typical; 3 is a sensible default for streaming inference). Returns
/// the raw bytes unchanged for [`Compression::None`].
pub fn encode(raw: &[u8], scheme: Compression, level: i32) -> Result<Vec<u8>, crate::Error> {
    use crate::{Context, Error};
    match scheme {
        Compression::None => Ok(raw.to_vec()),
        Compression::Zstd => {
            // Outside 1..=22 zstd returns an opaque codec error; name the real
            // problem instead.
            if !(1..=22).contains(&level) {
                return Err(Error::Format(format!("zstd level {level} is outside the valid range 1..=22")));
            }
            zstd::stream::encode_all(raw, level).ctx(|| "zstd encode".to_string())
        }
    }
}

/// Decompress `payload` into a fresh `Vec<u8>` of at most `orig_len` bytes.
/// `orig_len` is the pre-compression size stored in the header — used to
/// preallocate and to detect truncation.
pub fn decode(payload: &[u8], scheme: Compression, orig_len: usize) -> Result<Vec<u8>, crate::Error> {
    use crate::{Context, Error};
    match scheme {
        Compression::None => {
            if payload.len() != orig_len {
                return Err(Error::Format(format!(
                    "uncompressed length mismatch: header says {orig_len}, got {}",
                    payload.len()
                )));
            }
            Ok(payload.to_vec())
        }
        Compression::Zstd => {
            // Bounded decode: the frame header is attacker/corruption-controlled,
            // so a few KB can expand to hundreds of GB. Reading at most
            // `orig_len + 1` caps memory while still detecting an overrun (the
            // extra byte makes "too long" observable), and the length check
            // below catches truncation.
            use std::io::Read;
            let mut out = Vec::with_capacity(orig_len.min(1 << 30));
            let dec = zstd::stream::Decoder::new(payload).ctx(|| "zstd decoder".to_string())?;
            dec.take(orig_len as u64 + 1).read_to_end(&mut out).ctx(|| "zstd decode".to_string())?;
            if out.len() != orig_len {
                return Err(Error::Format(format!(
                    "zstd decompressed length mismatch: header says {orig_len}, got {}",
                    out.len()
                )));
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_is_bounded_by_the_declared_size() -> Result<(), crate::Error> {
        // A "zstd bomb": a small frame that expands enormously. The declared
        // size is known up front, so decoding must stop there instead of
        // materializing the whole expansion and only then noticing the mismatch.
        let huge = vec![0u8; 8 << 20]; // 8 MiB of zeros compresses to a few hundred bytes
        let frame = encode(&huge, Compression::Zstd, 3)?;
        assert!(frame.len() < 4096, "the bomb frame is small: {} bytes", frame.len());
        // Claiming a small original size must fail without expanding 8 MiB.
        let err = decode(&frame, Compression::Zstd, 1024);
        assert!(err.is_err(), "an over-expanding frame must be rejected");
        // The honest size still round-trips.
        assert_eq!(decode(&frame, Compression::Zstd, huge.len())?.len(), huge.len());
        Ok(())
    }

    #[test]
    fn encode_rejects_out_of_range_levels() {
        assert!(encode(b"data", Compression::Zstd, 0).is_err());
        assert!(encode(b"data", Compression::Zstd, 99).is_err());
        assert!(encode(b"data", Compression::Zstd, 3).is_ok());
    }

    #[test]
    fn none_is_verbatim() -> Result<(), crate::Error> {
        let raw = b"hello world".to_vec();
        let enc = encode(&raw, Compression::None, 0)?;
        assert_eq!(enc, raw);
        let back = decode(&enc, Compression::None, raw.len())?;
        assert_eq!(back, raw);
        Ok(())
    }

    #[test]
    fn zstd_round_trip() -> Result<(), crate::Error> {
        // Real weight-shaped bytes: repetitive quantized nibbles compress well.
        let mut raw = Vec::with_capacity(1 << 16);
        for i in 0..(1 << 16) {
            raw.push((i % 251) as u8);
        }
        let enc = encode(&raw, Compression::Zstd, 3)?;
        assert!(!enc.is_empty());
        // Compression ratio is a byproduct — assert on round-trip only.
        let back = decode(&enc, Compression::Zstd, raw.len())?;
        assert_eq!(back, raw);
        Ok(())
    }

    #[test]
    fn zstd_truncation_is_error() -> Result<(), crate::Error> {
        let raw = vec![7u8; 4096];
        let enc = encode(&raw, Compression::Zstd, 3)?;
        // Wrong `orig_len` must fail rather than silently produce a truncated buffer.
        assert!(decode(&enc, Compression::Zstd, 1000).is_err());
        Ok(())
    }

    #[test]
    fn from_tag_and_back() {
        assert_eq!(Compression::from_tag(Some("zstd")), Compression::Zstd);
        assert_eq!(Compression::from_tag(None), Compression::None);
        assert_eq!(Compression::from_tag(Some("garbage")), Compression::None);
        assert_eq!(Compression::Zstd.tag(), Some("zstd"));
        assert_eq!(Compression::None.tag(), None);
    }
}

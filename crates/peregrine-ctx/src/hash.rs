//! SHA-256 — the one cryptographic primitive this crate owns.
//!
//! [`crate::ids`] addresses content *inside one process*: FNV-1a, twelve hex
//! characters, fast and non-cryptographic by design. That is the right tool for
//! naming a span, and the wrong tool for answering "has this object been
//! modified since it was written". This module answers the second question.
//!
//! Why implement it rather than take a dependency: a hash is the one primitive
//! that can be verified against published output. The NIST vectors in the tests
//! below check this code against FIPS 180-4 rather than against itself, so a
//! wrong digest is a failing test, not a silent weakening. That argument does
//! **not** extend to signatures or AEAD — those live behind the traits in
//! [`crate::trust`] with no bundled implementation, because hand-rolled
//! asymmetric crypto and unverified constant-time behaviour are a liability the
//! zero-dependency rule does not justify.

use std::fmt;

/// A 32-byte SHA-256 digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Digest([u8; 32]);

impl Digest {
    pub const LEN: usize = 32;

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Digest(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex, 64 characters.
    pub fn to_hex(self) -> String {
        const HEX: [u8; 16] = *b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    /// The first `n` hex characters — for human-facing output only. A truncated
    /// digest is a label, never an integrity check.
    pub fn short(self, n: usize) -> String {
        let mut hex = self.to_hex();
        hex.truncate(n);
        hex
    }

    /// Parse 64 lowercase or uppercase hex characters.
    pub fn parse_hex(text: &str) -> Option<Digest> {
        if text.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        let bytes = text.as_bytes();
        for (i, slot) in out.iter_mut().enumerate() {
            let hi = (bytes[i * 2] as char).to_digit(16)?;
            let lo = (bytes[i * 2 + 1] as char).to_digit(16)?;
            *slot = ((hi << 4) | lo) as u8;
        }
        Some(Digest(out))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Streaming SHA-256. `update` any number of times, then `finalize`.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    /// Message length in bits, which is what the padding encodes.
    bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            state: INIT,
            block: [0u8; 64],
            filled: 0,
            bits: 0,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.bits = self.bits.wrapping_add((bytes.len() as u64) * 8);
        let mut rest = bytes;

        // Finish a partially filled block first, then run whole blocks
        // straight out of the input without copying them.
        if self.filled > 0 {
            let want = 64 - self.filled;
            let take = want.min(rest.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&rest[..take]);
            self.filled += take;
            rest = &rest[take..];
            if self.filled == 64 {
                let block = self.block;
                self.compress(&block);
                self.filled = 0;
            }
        }
        while rest.len() >= 64 {
            let (block, tail) = rest.split_at(64);
            let mut chunk = [0u8; 64];
            chunk.copy_from_slice(block);
            self.compress(&chunk);
            rest = tail;
        }
        if !rest.is_empty() {
            self.block[..rest.len()].copy_from_slice(rest);
            self.filled = rest.len();
        }
    }

    pub fn finalize(mut self) -> Digest {
        // 0x80, then zeroes, then the 64-bit big-endian bit length.
        let bits = self.bits;
        self.update_raw(&[0x80]);
        while self.filled != 56 {
            self.update_raw(&[0x00]);
        }
        self.update_raw(&bits.to_be_bytes());

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        Digest(out)
    }

    /// `update` without counting the bytes — padding must not extend the length
    /// it is encoding.
    fn update_raw(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.block[self.filled] = *byte;
            self.filled += 1;
            if self.filled == 64 {
                let block = self.block;
                self.compress(&block);
                self.filled = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, slot) in w.iter_mut().take(16).enumerate() {
            *slot = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in self
            .state
            .iter_mut()
            .zip([a, b, c, d, e, f, g, h].into_iter())
        {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// SHA-256 of one byte string.
pub fn sha256(bytes: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// SHA-256 of several parts under a domain tag.
///
/// The tag and every part are length-prefixed, so `("ab", "c")` and
/// `("a", "bc")` cannot collide and a digest computed for one purpose cannot be
/// replayed as a digest for another. Every structural hash in the container —
/// object identity, Merkle nodes, the checkpoint chain — goes through here.
pub fn tagged(tag: &str, parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(&(tag.len() as u64).to_be_bytes());
    hasher.update(tag.as_bytes());
    for part in parts {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4 / NIST CAVP vectors. These check the implementation against
    /// published output rather than against itself.
    #[test]
    fn nist_vectors() {
        assert_eq!(
            sha256(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").to_hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Two blocks — the case that exercises the padding overflow path.
        assert_eq!(
            sha256(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            )
            .to_hex(),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn million_a() {
        let mut hasher = Sha256::new();
        for _ in 0..1000 {
            hasher.update(&[b'a'; 1000]);
        }
        assert_eq!(
            hasher.finalize().to_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// Streaming in arbitrary chunk sizes must equal the one-shot digest —
    /// the buffering in `update` is where a hash implementation usually breaks.
    #[test]
    fn chunking_does_not_change_the_digest() {
        let message: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let once = sha256(&message);
        for chunk in [1usize, 7, 63, 64, 65, 127, 128, 333] {
            let mut hasher = Sha256::new();
            for part in message.chunks(chunk) {
                hasher.update(part);
            }
            assert_eq!(hasher.finalize(), once, "chunk size {chunk}");
        }
    }

    #[test]
    fn hex_round_trips() {
        let digest = sha256(b"round trip");
        assert_eq!(Digest::parse_hex(&digest.to_hex()), Some(digest));
        assert_eq!(Digest::parse_hex("short"), None);
        assert_eq!(Digest::parse_hex(&"z".repeat(64)), None);
    }

    #[test]
    fn tagging_separates_domains() {
        assert_ne!(tagged("leaf", &[b"x"]), tagged("node", &[b"x"]));
        // Length prefixes: the parts cannot be re-split into the same stream.
        assert_ne!(tagged("t", &[b"ab", b"c"]), tagged("t", &[b"a", b"bc"]));
    }
}

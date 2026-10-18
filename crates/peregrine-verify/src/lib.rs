//! The numerical certificate suite — the workspace's other gate, next to
//! `scripts/audit-bad-patterns.sh`.
//!
//! # What "certified" means here
//!
//! Nothing in this crate is a proof in the formal sense. Rust is not a proof
//! assistant, and a floating-point implementation of a real-valued identity can
//! at best be *correct to a stated tolerance*. What this crate does instead is
//! make every load-bearing numerical claim in the workspace **falsifiable and
//! continuously checked**:
//!
//! - the claim is written down as a theorem string, in prose, next to the code
//!   that checks it, so a reader can judge whether it is worth checking
//!   independently of whether it currently passes;
//! - the check produces a **residual**: a single number that is zero exactly
//!   when the claim holds;
//! - the residual is compared against a **tolerance chosen from the arithmetic**,
//!   never from whatever the code currently happens to produce.
//!
//! A claim that cannot be reduced to a residual is not listed.
//!
//! # Which claims are load-bearing
//!
//! Load-bearing means: **if this stopped holding, the engine would produce wrong
//! output silently rather than fail.** That is the only bar, and it is what
//! separates a certificate from a unit test. Concretely, the classes here are:
//!
//! | Group | What is certified |
//! |---|---|
//! | `kernels` | every SIMD integer-dot variant is bit-identical to its scalar reference, for every length and alignment; the quantized matmuls index the same values the float path does; and the packed row a quantizer writes is the row the dot kernel reads |
//! | `dtype` | f16/bf16 round-trips stay inside half an ulp of the storage format, round-trips are idempotent, and the non-finite values survive |
//! | `pack` | int4/int8/int2/int2-g64/int3-g64 packing round-trips exactly on values the format represents, the k-block byte permutation is an involution, and the affine zero-point convention dequantizes a constant group back to its own minimum |
//! | `math` | RMSNorm produces unit RMS, softmax a distribution, SwiGLU composes its two production halves, RoPE at position 0 is the exact identity, and rotation preserves the 2-norm |
//! | `router` | renormalized gates are a distribution, selection is the top-k of the *plain* logit (the bias selects but does not weight), and the budget/min-share controls can only shrink `keff` |
//! | `ctx` | SHA-256 against published vectors, tagged hashing is domain-separated and length-prefixed, and a Merkle root is a function of the *set* while proofs verify and forgeries fail |
//! | `io` | a mirror-striped read returns the same bytes as a single-drive read, at both a splitting and a non-splitting size; the expert cache evicts strictly least-recently-used and its hit rate is the ratio it reports |
//! | `token` | BPE `encode` → `decode` is lossless, and the parallel batch encoder is id-for-id identical to the serial one |
//!
//! Not load-bearing, and therefore absent: anything that fails loudly. A bad
//! length, a malformed header, a missing tensor — those return an `Err`, and an
//! `Err` is already a test failure.
//!
//! # Running it
//!
//! ```text
//! cargo run -p peregrine-verify --release -- verify
//! cargo run -p peregrine-verify --release -- verify --group kernels
//! cargo run -p peregrine-verify --release -- list
//! ```
//!
//! `verify` exits non-zero the moment any identity the implementation rests on
//! stops holding, so a regression in, say, the int4 bias convention shows up as
//! a named failure rather than as slightly worse perplexity months later.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod ctx;
pub mod dtype;
pub mod io;
pub mod kernels;
pub mod math;
pub mod pack;
pub mod router;
pub mod sample;
pub mod tier;
pub mod token;

use std::fmt;

/// One checked claim.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub group: &'static str,
    pub name: &'static str,
    /// The claim, stated so a reader can judge whether checking it is
    /// worthwhile independently of whether it currently passes.
    pub theorem: &'static str,
    /// Measured deviation from the claim; zero exactly when it holds.
    pub residual: f64,
    /// Largest residual consistent with the claim, given the arithmetic.
    pub tolerance: f64,
    /// How many cases the residual aggregates. A `0.0` residual over 64 cases
    /// says far more than the same number over one, and the distinction is
    /// invisible in a table row without it.
    pub cases: usize,
}

impl Certificate {
    pub fn passed(&self) -> bool {
        self.residual.is_finite() && self.residual <= self.tolerance
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}: {} (residual {:e}, tolerance {:e})",
            self.group, self.name, self.theorem, self.residual, self.tolerance
        )
    }
}

/// Build one certificate. Every certificate in the workspace goes through here,
/// so `tolerance` is always written next to the claim that justifies it.
pub(crate) fn cert(
    group: &'static str,
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate { group, name, theorem, residual, tolerance, cases: 1 }
}

/// Record how many cases a residual aggregates. A free function rather than a
/// method because `with_cases` takes `self` by value, and a by-value `with_*`
/// reads like a builder on a borrowed value.
pub(crate) fn with_cases(c: Certificate, cases: usize) -> Certificate {
    Certificate { cases, ..c }
}

/// The residual of a check that could not run: infinite, so the certificate
/// fails, with the reason on stderr rather than a panic and rather than a skip.
/// A certificate that cannot be evaluated is not a passing certificate.
pub(crate) fn failed(what: &str, err: &str) -> f64 {
    eprintln!("peregrine-verify: {what}: {err}");
    f64::INFINITY
}

/// Turn an exact boolean claim into a residual: `0.0` when it holds, `1.0` when
/// it does not. Measured this way rather than as "a near-zero" so that a future
/// tolerance cannot quietly absorb a broken identity — an audit failure has to
/// be a number that is clearly out of bounds.
pub(crate) fn exact(holds: bool) -> f64 {
    f64::from(u8::from(!holds))
}

/// [`mismatch_count`] for `f32`, compared through `to_bits` so that a sign of
/// zero, a NaN payload, or a `-0.0` where `0.0` was required is a failure rather
/// than an equality.
pub(crate) fn mismatch_bits(expected: &[f32], observed: &[f32]) -> f64 {
    if expected.len() != observed.len() {
        return f64::INFINITY;
    }
    let bad = expected
        .iter()
        .zip(observed.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    bad as f64
}

/// The largest absolute deviation between two `f32` sequences, as `f64`. `inf`
/// when the lengths differ, which is a shape bug and not a numerical one.
pub(crate) fn max_abs_diff(expected: &[f32], observed: &[f32]) -> f64 {
    if expected.len() != observed.len() {
        return f64::INFINITY;
    }
    expected
        .iter()
        .zip(observed.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
}

/// `f32::EPSILON` as an `f64`, the unit roundoff bound for one f32 operation.
pub(crate) const EPS_F32: f64 = f32::EPSILON as f64;

/// A deterministic pseudo-random `f32` in `[-1, 1)`, seeded so every run of the
/// suite sees the same fixture. A certificate whose input changes between runs
/// is a certificate you cannot debug; the xorshift is 6 lines and has no
/// dependency.
pub(crate) fn rng() -> Rng {
    Rng { state: 0x9E37_79B9_7F4A_7C15 }
}

pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    pub(crate) fn i8_vec(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.sym() * 127.0).round() as i8).collect()
    }

    pub(crate) fn u8_vec(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.sym() * 127.0).round() as i8 as u8).collect()
    }

    pub(crate) fn f32_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sym()).collect()
    }
}

/// The result of a verification run.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub certificates: Vec<Certificate>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.certificates.iter().all(Certificate::passed)
    }

    pub fn failures(&self) -> Vec<&Certificate> {
        self.certificates.iter().filter(|c| !c.passed()).collect()
    }

    pub fn num_passed(&self) -> usize {
        self.certificates.iter().filter(|c| c.passed()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    /// One-line summary, for logs where the full table is too much.
    pub fn summary(&self) -> String {
        match self.failures().as_slice() {
            [] => format!("{} certificates passed", self.certificates.len()),
            failures => format!(
                "{}/{} certificates FAILED: {}",
                failures.len(),
                self.certificates.len(),
                failures.iter().map(|c| format!("{}::{}", c.group, c.name)).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    /// Human-readable table, grouped, with the residual/tolerance ratio so a
    /// certificate drifting toward its bound is visible long before it fails.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<10} {:<44} {:>11} {:>11} {:>7}  {}\n",
            "group", "certificate", "residual", "tolerance", "margin", "status"
        ));
        out.push_str(&"-".repeat(100));
        out.push('\n');

        let mut last_group = "";
        for c in &self.certificates {
            let group = if c.group == last_group { "" } else { c.group };
            last_group = c.group;
            let margin = if c.tolerance > 0.0 && c.residual.is_finite() {
                format!("{:.2}x", c.residual / c.tolerance)
            } else {
                "-".to_string()
            };
            let cases = if c.cases > 1 { format!(" (x{})", c.cases) } else { String::new() };
            out.push_str(&format!(
                "{:<10} {:<44} {:>11.3e} {:>11.3e} {:>7}  {}{}\n",
                group,
                format!("{}{}", c.name, cases),
                c.residual,
                c.tolerance,
                margin,
                if c.passed() { "ok" } else { "FAILED" },
                ""
            ));
        }

        out.push_str(&"-".repeat(100));
        out.push('\n');
        out.push_str(&format!("{} / {} certificates passed\n", self.num_passed(), self.certificates.len()));
        for f in self.failures() {
            out.push_str(&format!("\nFAILED {}::{}\n  claim:     {}\n  residual:  {:e}\n  tolerance: {:e}\n", f.group, f.name, f.theorem, f.residual, f.tolerance));
        }
        out
    }
}

/// Every certificate group, in the order [`run_all`] emits them.
pub const GROUPS: [&str; 10] =
    ["kernels", "dtype", "pack", "math", "router", "sample", "tier", "ctx", "io", "token"];

/// Run every certificate.
pub fn run_all() -> Report {
    let mut certificates = Vec::new();
    certificates.extend(kernels::certificates());
    certificates.extend(dtype::certificates());
    certificates.extend(pack::certificates());
    certificates.extend(math::certificates());
    certificates.extend(router::certificates());
    certificates.extend(sample::certificates());
    certificates.extend(tier::certificates());
    certificates.extend(ctx::certificates());
    certificates.extend(io::certificates());
    certificates.extend(token::certificates());
    Report { certificates }
}

/// Run only the certificates in `group`.
///
/// Returns an empty report for an unknown group; a caller that treats "no
/// certificates" as success must check [`Report::is_empty`] first — the CLI
/// does, which is why `verify --group typo` fails instead of passing vacuously.
pub fn run_group(group: &str) -> Report {
    let mut report = run_all();
    report.certificates.retain(|c| c.group == group);
    report
}

/// The subset worth running *before* a decode: the token-exactness kernels and
/// the dtype conversions. These are the properties a run depends on for its
/// whole lifetime and the only ones cheap enough to check on every start.
pub fn preflight() -> Report {
    let mut certificates = kernels::certificates();
    certificates.extend(dtype::certificates());
    Report { certificates }
}
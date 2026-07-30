[« Docs index](README.md)

# Testing & quality gates

**282 tests passing, 0 warnings, clippy clean** (debug + release), plus a
strict bad-patterns audit. This page is what to run, what each gate enforces,
and the correctness philosophy behind the test suite.

## The gates

```bash
cargo test --workspace                     # 282 tests, CPU-only, no GPU needed
cargo clippy --workspace --all-targets     # clean
scripts/audit-bad-patterns.sh --strict     # panic-vector / UB gate (CI)
cargo test -p peregrine-cuda --features cuda   # GPU-gated tests (NVIDIA host only)
```

Run all of them after any change to the streaming, scheduler, or serve paths.

## Correctness philosophy

The suite is built around **bit-identity anchors** rather than tolerances:

- **Scalar kernels are the reference.** The scalar integer-dot kernels are
  the token-exactness reference; AVX2/AVX-VNNI variants are checked
  bit-for-bit against them (accumulation order is controlled via `std::arch`,
  never `portable_simd`).
- **Parallel == serial, exactly.** The `peregrine-par` pool is bit-identical
  to serial execution (`f32::to_bits`-exact tests) for rmsnorm, resident MoE,
  per-row attention, and every matmul — fixed-order reduces make this
  possible.
- **Concurrent == sequential.** The 3-lane scheduler's output equals the
  sequential path regardless of lane interleaving (deterministic
  position-keyed reduce).
- **Chunked == whole.** Chunked prefill is bit-identical to whole-prompt
  prefill (`engine_chunked_prefill_matches_reference`).
- **Adaptive knobs are bit-identical when off**, and correctness-neutral when
  on (they may only change latency/residency, never token values).
- **Format round-trips.** Config / safetensors index / QT formats / dtype
  round-trips; zstd and kblock layouts decode byte-identically;
  `apply_layout_is_bit_identical` gates the physical checkpoint rewrite with
  teacher-forcing equality.
- **io_uring reads validated byte-for-byte vs `pread`** on real hardware.
- **Tokenizer parity.** Id-for-id equality with the HF `tokenizers` oracle
  over an edge-case corpus (`crates/peregrine-serve/tests/tokenizer_parity.rs`);
  the HF crate exists *only* as this test oracle.
- **Statistical gates** where exactness isn't the contract:
  `speculative_sample` rejection sampling is statistically lossless.

## The bad-patterns audit

`scripts/audit-bad-patterns.sh` (`--strict` = the CI gate) is a re-runnable
workspace-wide scan documented in [BAD_PATTERNS.md](BAD_PATTERNS.md):

- **[P] Panicking error handling — must be zero.** `unwrap`/`expect`/
  `panic!`/`unreachable!`/`todo!`/`unwrap_or_default` on the engine paths; a
  panic mid-inference takes the whole process down. The crates also
  `#![deny(clippy::unwrap_used, expect_used, panic)]`; the gate enforces it
  uniformly and catches what the clippy lints miss.
- **[B] Silent error swallowing — must be zero.** `let _ = fallible()`,
  `if let Ok(x)`-only handling, `Err(_) =>`, error-discarding
  `map_err`/`or_else`. Advisory operations (madvise/fadvise hints, NUMA
  pinning, route-stats persistence) stay correctness-neutral but must report
  through `note_advisory_err` (surfaced by `COLI_DEBUG=1`), never vanish.
- **[U] UB / concurrency footguns — must be zero.** `static mut`,
  `transmute`, `mem::forget`, `assume_init`.
- **[I] `unsafe` outside the expected crates — informational.** `unsafe` is
  expected only in `peregrine-io` (io_uring, madvise/mbind, perf),
  `peregrine-cuda` (FFI), `peregrine-kernels` (SIMD); `peregrine-core` is
  `#![forbid(unsafe_code)]`, and `peregrine-serve` too.

Current status: `--strict` green — **P=0, B=0, U=0, I=0** (51 files;
`peregrine-token` excluded as vendored — its gate is the parity suite).
Waivers use `// audit-allow:` comments; there is exactly one production
`assert!` (KV-cache append order) guarding an invariant whose silent
violation would corrupt attention output.

## Error handling contract

Every fallible public API returns `peregrine_core::Error` (thiserror + the
`Context`/`.ctx()` extension) — structured errors workspace-wide, no
`Result<_, String>` outside the vendored crate. Binaries print
`peregrine: <error>` and exit 1; advisory (non-fatal) failures are reported
via a shared reporter that `COLI_DEBUG=1` surfaces on stderr.

## Toolchain

Stable Rust, edition 2021 (the vendored `peregrine-token` is edition 2024
crate-locally for let-chains — still stable toolchain). Release profile:
`opt-level 3`, fat LTO, single codegen unit, `panic = "abort"`. `clippy.toml`
at the workspace root tightens the lint set.

## Writing new code

- No `unwrap`/`expect`/`panic` in engine crates — return
  `peregrine_core::Error` with `.ctx(|| …)` context.
- New adaptive features must be env-gated, default-off (or
  default-historical), and bit-identical when off; add the A/B test proving it.
- New kernels need a scalar reference and a bit-exact comparison test.
- Keep `unsafe` inside the three sanctioned crates; anything else will show
  up in the audit's [I] section.
- Re-run all four gates before calling work done.

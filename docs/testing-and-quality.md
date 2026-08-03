[« Docs index](README.md)

# Testing & quality gates

**482 tests passing, 0 warnings, clippy clean** (debug + release), plus a
strict bad-patterns audit. This page is what to run, what each gate enforces,
and the correctness philosophy behind the test suite.

## The gates

```bash
cargo test --workspace                     # 482 tests, CPU-only, no GPU needed
cargo clippy --workspace --all-targets     # clean
scripts/audit-bad-patterns.sh --strict     # panic / UB / suppression / Cargo gate (CI)
scripts/audit-reachability.py --list       # [R] shipped-but-unreachable pass
cargo check --features cuda --all-targets -p peregrine-model -p peregrine-cuda
cargo test -p peregrine-cuda --features cuda   # GPU-gated tests (NVIDIA host only)
```

Run all of them after any change to the streaming, scheduler, or serve paths.

`--all-targets` on the `cuda` check is not optional: without it `cargo check` skips
test targets, so a signature change can leave the `#[cfg(all(test, feature = "cuda"))]
modules uncompilable while the check still reports success. That is exactly how the
`GpuTier::build` call sites in `gpu.rs` went stale. It does not link (no `nvcc` needed),
so it runs anywhere.

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
- **Concurrent == sequential.** Deterministic position-keyed reduce, so lane
  interleaving cannot change the result. Read the anchor carefully though:
  `concurrent_matches_sequential` lives in `peregrine-sched` and exercises
  `moe_streamed` — the **two-lane ancestor no crate links**. The production
  3-lane path (`peregrine-model/concurrent.rs`) is covered at the attention
  level by `batched_matches_sequential`, not by the test whose name implies it.
- **Chunked == whole.** Chunked prefill is bit-identical to whole-prompt
  prefill (`engine_chunked_prefill_matches_reference`).
- **Adaptive knobs are bit-identical when off.** Almost all are also
  correctness-neutral when on — they may change latency or residency, never
  token values. **Five deliberate exceptions:**
  - `COLI_ROUTE_MIN_SHARE` drops routed experts carrying a negligible share of the
    gate mass, which removes a real (if small) term from the MoE sum. It is off by
    default, and it is gated by `Model::prediction_flip_rate` rather than by an
    equality assertion, because a lossy change fails every bit-identity anchor by
    construction. `COLI_GATE_STATS` is how to size it before enabling it.
  - `COLI_MLA_ABSORB` swaps the dense attention reconstruction for weight
    absorption. The two are algebraically equal but not numerically identical:
    dense pushes the cached latent back through the quantized `kv_b`, absorb folds
    that weight into the query instead. `absorb_approximates_dense` bounds *one
    call* at 10% relative; end-to-end that difference compounds through the stack,
    and it has not been measured on a real checkpoint. Off by default, and its
    test asserts only what is verifiable — inert when unset, genuinely wired when
    set — rather than an invented closeness bound. **It did not reach the batched
    decode path until 2026-08-03**: `forward_layer_batched` called an absorb-only
    core unconditionally, so a served request ran its prefill dense and every
    decode token absorbed — two numerically different cores inside one response,
    whatever the knob said. The dense core now takes per-row cache owners, and
    `batched_decode_honours_the_absorb_knob_and_defaults_to_dense` asserts the
    documented contract from both sides: at the default, batched decode is
    bit-identical to the single-sequence decode; with the knob set, it differs.
  - `COLI_KV_DTYPE=f16` stores the KV latents rounded, halving resident KV
    exactly. Unlike the two above, its cost is *bounded by construction* on the
    absorb path — the latent is dotted in f32, so the error is f16's own
    precision (1.8e-4 measured) — while the dense path measures ~100× worse,
    1.7e-2. That gap is **not** f16: `kv_b.apply_vec` quantizes activations to
    int8 at `amax / 127`, so a perturbation that moves a row's maximum rescales
    the whole grid. Recorded as a per-core tolerance in
    `f16_kv_halves_the_bytes_and_tracks_f32_closely`, which also asserts the two
    dtypes *do* differ — a lossy knob whose test would pass unchanged if the
    knob did nothing is not testing the knob.
  - `COLI_DSA` runs the lightning indexer, so each query attends the top
    `index_topk` cached keys instead of all of them. **Inert two ways, both
    asserted bit-exact**: with no indexer in the checkpoint there is nothing to
    run, and at or below `index_topk` cached positions the selection is the
    identity, so the scoring pass is skipped rather than run and discarded
    (`dsa_is_inert_without_an_indexer_and_below_index_topk`). Above the
    threshold it changes token values, and
    `dsa_selects_a_subset_once_context_exceeds_index_topk` asserts that it
    *does* — a sparse-attention flag whose test would pass unchanged if
    selection did nothing is not testing selection.
  - **A requantized checkpoint** (`peregrine-requantize`) changes token values by
    existing, not by being toggled, so it has no knob row. int4 → int2 is a double
    quantization; measure it with `prediction_flip_rate` against the source
    container rather than assuming the halved bytes are free.
- **Format round-trips.** Config / safetensors index / QT formats / dtype
  round-trips; zstd and kblock layouts decode byte-identically;
  `apply_layout_is_bit_identical` gates the physical checkpoint rewrite with
  teacher-forcing equality.
- **io_uring reads validated byte-for-byte vs `pread`** — *when the harness can
  run them.* Every such test returns `Ok(())` on a kernel without io_uring or a
  filesystem that rejects O_DIRECT (tmpfs, overlayfs, many CI containers), and
  `cargo test` swallows the `eprintln!` skip notice without `--nocapture`.
  Nothing asserts that at least one of them executed, so a green run is
  compatible with this whole story never having been exercised. Check it
  explicitly on any machine you intend to trust the claim on.
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
- **[C] Lint suppression — must be zero.** `#[allow(...)]`, item-level
  `#[deny(...)]`, `#[ignore]`. The workspace removed all 14 of its `#[allow]`s by
  fixing what they hid; this keeps that true by gate rather than by prose.
- **[Q] Cargo.toml hygiene — must be zero.** Wildcard versions and unpinned git
  dependencies. A dependency moving under you can shift the bit-identity anchors
  this suite is built on without a commit to blame.
- **[R] Shipped but unreachable — informational.** `pub fn`s no production path
  reaches, via `scripts/audit-reachability.py`. This is the one class greps
  cannot see and tests cannot catch — the code under test is correct, it simply
  runs nowhere, and `pub` keeps clippy's `dead_code` quiet. Five instances had
  shipped, each documented as live. **Run it before marking anything complete.**
- **[I] `unsafe` outside the expected crates — informational.** `unsafe` is
  expected only in `peregrine-io` (io_uring, madvise/mbind, perf),
  `peregrine-cuda` (FFI), `peregrine-kernels` (SIMD); `peregrine-core` is
  `#![forbid(unsafe_code)]`, and `peregrine-serve` too.

Current status: `--strict` green — **P=0, B=0, U=0, C=0, Q=0, I=0** (55 files;
`peregrine-token` excluded as vendored — its gate is the parity suite).
Waivers use `// audit-allow:` comments, and none remain in first-party code.
There are **zero production `assert!`s**: the last one (KV-cache append order)
became `LayerKv::append -> Result`, so a violation fails that one request instead
of aborting the process — the release profile sets `panic = "abort"`, which would
have taken every concurrent sequence down with it.

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
  If a feature changes token values when on, say so at its definition, in its
  knob-table row, and here — and gate it with a bounded flip rate, not equality.
- New kernels need a scalar reference and a bit-exact comparison test.
- Keep `unsafe` inside the three sanctioned crates; anything else will show
  up in the audit's [I] section.
- Re-run all four gates before calling work done.

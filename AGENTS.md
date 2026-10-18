# AGENTS.md — the law of this repository

`peregrine` is a Rust MoE inference engine (spin-off of *colibrì*) that drives
CPU, GPU, SSD and RAM concurrently. The crates, in dependency order:

| Crate | What lives there |
|---|---|
| `peregrine-kernels` | the token-exactness-critical CPU path: scalar integer-dot reference kernels, AVX2/AVX-VNNI variants, quant row packing |
| `peregrine-io` | io_uring reactor, mirror-striped reads, LRU/LFRU cache tiers, topology/sensor probes |
| `peregrine-par` | the worker pool behind every `par_*` call |
| `peregrine-cuda` | FFI to `cuda/backend_cuda.cu` (feature `cuda`) |
| `peregrine-core` | formats: `Cfg`, safetensors index, QT quant, dtype conversion, compression, durable writes |
| `peregrine-token` | **vendored third-party** (marcelroed/gigatoken, MIT) — upstream style preserved verbatim |
| `peregrine-model` | MLA, router, MoE, DSA, MTP, hybrid attention, and the production concurrent path |
| `peregrine-sched` | two-lane streaming scheduler (io_uring ∥ CPU) |
| `peregrine-serve` | HTTP + stdio serve protocol |
| `peregrine-tools` | import, reshard, offline artifact builders |
| `peregrine-ctx` | Dynamic Context Runtime: external object store, Merkle tree, trust, planning. **Vendored port** of subnext/DCR (`dcr`, pinned at 609451b) — see rule 7 |
| `peregrine-engine` | the `peregrine` binary: CLI + stdio serve protocol |
| `peregrine-verify` | the numerical certificate suite (`verify` subcommand), a gate like the audit |

Start at [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/testing-and-quality.md`](docs/testing-and-quality.md).

## The gates — run all of these before you stop

```bash
cargo test --workspace                            # 0 failures; CPU-only, no GPU needed
cargo clippy --workspace --all-targets            # no new warnings
scripts/audit-bad-patterns.sh --strict            # exit 0 required
scripts/verify-certificates.sh                    # every certificate, exit 0 required
```

The last two have the same shape on purpose: one command each, non-zero on
failure, reason on stderr. `scripts/audit-bad-patterns.sh` ratchets — it fails
when a section's count *exceeds* its budget in the `BUDGETS` table, not when the
count is non-zero — and its budgets only move down. A budget that needs raising
because new code added a hit is a review conversation, not a bookkeeping update.

`--all-targets` on clippy is not optional: without it, `cargo check` skips test
targets, so a signature change can leave a `#[cfg(all(test, feature = "cuda"))]`
module uncompilable while the check still reports success. That is exactly how
the `GpuTier::build` call sites in `gpu.rs` went stale.

A gate you cannot run in this environment (no GPU, no `nvcc`) is not a gate you
passed: say so explicitly in your final message rather than implying it is green.

## The law

**1. No panicking error handling. Anywhere. Including tests.** No `unwrap`,
`expect`, `panic!`, `unreachable!`, `todo!`, `unimplemented!`, no
`unwrap_or_default`. Every crate root carries
`#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` and
`clippy.toml` sets `allow-unwrap-in-tests = false` — extend them, do not relax
them. The release profile is `panic = "abort"`: there is no unwinding, so a panic
in the streaming/scheduler/serve path is not a failed request, it is a dead
server and a dropped in-flight batch.

*Tests may `assert!`/`assert_eq!`.* A test that cannot assert anything is not a
test. What a test may not do is `unwrap` a value the production path would have
had to handle.

**2. Errors are `peregrine_core::Error`, and they say what failed.** Every fallible
public API returns `peregrine_core::Error` (thiserror). Add context with the
`Context` extension — `foo().ctx(|| format!("reading {path}"))` — not with a
`.map_err(|e| …)` at the call site. A new crate does **not** reach for `anyhow`:
this workspace has one error type on purpose, and no `Result<_, String>` surfaces
remain outside the vendored crate.

**3. Nothing is swallowed.** A discarded `Result` hides a real failure. For
best-effort operations (madvise/fadvise hints, NUMA pinning, route-stats
persistence, shutdown signalling, optional-artifact loading) the answer is
`peregrine_io::note_advisory_err("what was attempted", &err)` — correctness-neutral
by design, **visible** by design (`COLI_DEBUG=1` surfaces it on stderr). For
control-flow errors, `match` the concrete variant (`VarError::NotPresent`,
`TryRecvError::Empty`) and name it. `let _ = fallible()` and `let _tmp = fallible()`
are both bugs.

The one exemption is the RAII keep-alive idiom `let _g = gpu_guard();`, where
the named binding is the point — `let _ =` would drop the guard immediately.

The audit scans **every** `.rs` file under `crates/` — `src/`, `tests/`,
`examples/` and the `src/bin/` targets — not just `src/`. Integration tests and
examples are separate crates that inherit no crate-root `#![deny(...)]`, so the
audit is the only thing holding the law there.

**4. No lint suppression.** No `#[allow(…)]`, no item-level `#[deny(…)]`, no
`#[ignore]`, and no bare `// audit-allow:` — a waiver needs a reason after the
colon. The workspace removed 14 `#[allow]`s that way (`too_many_arguments` →
`MatShape`/`RouterCfg`/`MoeCfg`, `should_implement_trait` → implement `FromStr`),
so a new one has to be argued in review rather than merged: fix the code, or make
the case. The working tree currently carries 5 (`C=5` in the audit — four
`too_many_arguments` and one `dead_code`); they are owed the same treatment.

**5. `unsafe` is confined to four crates**: `peregrine-io` (io_uring, aligned
alloc, madvise/mbind, perf counters), `peregrine-cuda` (FFI), `peregrine-kernels`
(SIMD intrinsics), `peregrine-par` (one documented lifetime-erasure behind a
`#SAFETY` note and a fork-join barrier). `peregrine-core`, `peregrine-serve` and
`peregrine-verify` are `#![forbid(unsafe_code)]`. `transmute` outside
`peregrine-par` is a hard error, as are `static mut`, `mem::forget`,
`assume_init()` and `unwrap_unchecked`.

**6. No secrets, no unpinned dependencies.** No API keys, tokens, `PRIVATE KEY`
blocks or `("admin","admin")` literals in the tree. No `dep = "*"` and no git
dependency without `rev`/`tag`/`branch`: the correctness anchors are
bit-identity tests, and a dependency moving under you shifts them without a commit
to blame.

**7. `peregrine-token` is vendored.** Do not "fix" upstream style in it, and do
not copy its idioms (upstream `unwrap`s, unsafe SIMD) into first-party code. Its
gate is the id-for-id parity suite against the HF `tokenizers` oracle.

**7a. `peregrine-ctx` is vendored the same way.** It is a verbatim port of
subnext/DCR: of its 34 `src` files, 32 are byte-identical to upstream, and the
two that differ (`lib.rs`, `main.rs`) differ only by the `dcr::` →
`peregrine_ctx::` prefix rename and the CLI binary name. Do not "fix" upstream
style in the ported files, and do not copy their idioms into first-party code.
Its gate is omni's cross-repo sync check (`scripts/verify-cross-repo.sh`), which
fails if any ported file drifts from upstream or a file goes missing — so
"cleaning up" a ported file is not a cleanup, it is a fork.

Exactly three files in `peregrine-ctx` are peregrine's own work and are held to
the engine crates' gates like any other first-party code:
`src/lib.rs`, `src/main.rs`, `examples/ctxbench.rs`. Those are the files to fix
when the audit points into `peregrine-ctx`.

**8. Temp files get unique names.** An atomic counter plus the pid. Parallel
tests share `/tmp`, and two agents on one box share the tree.

## Numerics

**Bit-identity is the contract; a tolerance is an admission.** Scalar kernels are
the reference, and the SIMD variants are checked bit-for-bit against them —
accumulation order is controlled explicitly, never left to a compiler. When a
comparison genuinely cannot be bit-exact (lanes accumulate in different orders),
the tolerance is derived **from the arithmetic** — `n · eps_f32 · scale`, times a
documented factor — and written down next to the assertion. A tolerance fitted to
whatever the code currently produces is not a tolerance, it is a description.

Inside `peregrine-kernels`, explicit index loops are the house style: they mirror
the C ports line-for-line so the two can be compared, and most of them index
several slices at once, which makes `needless_range_loop` noise. Do not
refactor them into iterators.

New numerical code goes in `peregrine-kernels` when it is on the token-exact
path, `peregrine-model` when it is model math, and nowhere else.

## Certificates

Every load-bearing numerical identity gets a certificate in `peregrine-verify`
(see `crates/peregrine-verify/src/lib.rs` for the harness,
`crates/peregrine-verify/src/*.rs` for the certificates themselves, and
[`docs/testing-and-quality.md`](docs/testing-and-quality.md) for the test suite
they complement). Load-bearing means: if this stopped holding, the engine would
produce *wrong output silently* rather than fail — SIMD against scalar, parallel
against serial, lanes against each other, quant against f32 truth, a striped read
against a single-drive read, dtype conversion against the file it came from. Not
load-bearing: anything that fails loudly instead.

- the claim is written down as a **theorem string** a reader can judge
  independently of whether it currently passes;
- the check produces a **residual** — one number, zero exactly when the claim
  holds;
- the residual is compared against a **tolerance** chosen from the arithmetic;
  exact identities get `0.0`;
- the check runs the **production function**, never a reimplementation of the
  formula. A certificate that reimplements the formula proves the formula, which
  was never in doubt — an inline copy of `weighted_switch_loss` kept passing while
  a mutant that dropped a factor shipped;
- exact-boolean identities are measured as `f64::from(u8::from(violation))`, so
  the residual is `1.0` and not a near-zero that a future tolerance could absorb;
- a check that could not run is a **failure with the reason on stderr**, not a
  skip and never a panic;
- the `margin` column (residual ÷ tolerance) is the early-warning signal: a
  certificate drifting toward its bound is visible long before it fails.

Add a certificate with the code it certifies, in the same change. The harness is
~200 lines in `crates/peregrine-verify/src/lib.rs`; a new group is a module
exporting `pub fn certificates() -> Vec<Certificate>`, registered in `GROUPS` and
in `run_all`.

## Conventions

- Config and format fields get `#[serde(default)]`, so an old checkpoint sidecar
  still parses.
- Runtime knobs are `COLI_`-prefixed environment variables (the prefix is kept
  from colibrì for drop-in compatibility) and are documented in
  [`docs/configuration.md`](docs/configuration.md).
- A new crate joins `members` in the root `Cargo.toml` and gets its own
  `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` at the
  crate root — note each binary target is its own crate root, so it carries the
  attribute again.
- Advisory failures must not become hard errors, and hard errors must not become
  advisory ones. Which one a thing is, is a design decision worth writing down.
- Docs are claims, and a claim that is not true is worse than no doc. When a
  change makes a doc wrong, fix the doc in the same change. Do not create new
  `.md` files unless asked — the contract lives in rustdoc, tests and
  certificates.

## Do not

- Edit `scripts/audit-bad-patterns.sh` to make a finding go away. It is the gate;
  if it is wrong, argue it in review. (It is *not* a claim of style: the audit
  counts are in `docs/BAD_PATTERNS.md` and must be regenerated from a run, never
  hand-edited.)
- Weaken a gate to make a commit land — `--strict` must exit 0, certificates must
  pass, tests must be green, or the reason must be in your final message.
- "Improve" a passing test's tolerance, or delete a test that is inconvenient.
- Benchmark on one run and quote it. `scripts/bench-*-envarms.sh` rotates arm
  order and reports medians; `iobench` defaults to `REPS=5` and refuses to resolve
  a gap smaller than the measured spread.
- Commit secrets, absolute machine-specific paths into tracked scripts, or
  generated weights/logs. The root `.gitignore` already covers the model shards
  and logs — keep it that way.
- Claim a benchmark, a certificate or a gate result you did not run.
# Bad-pattern catalogue

A re-runnable quality gate for the peregrine workspace, adapted from the
[rustsploit](https://github.com/s-b-repo/rustsploit) audit. It catches crash-vectors
and UB that `cargo clippy` does not flag by default, workspace-wide (per-crate
`#![deny(...)]` attributes can drift; this gate does not).

```bash
scripts/audit-bad-patterns.sh            # full report (shows offending lines)
scripts/audit-bad-patterns.sh --strict   # exit non-zero on any P/U hit — the CI gate
```

Scope: Rust under `crates/*/src`; comment lines are ignored.

## [P] Panicking error handling — **strict, must be zero**

A panic in the streaming/scheduler/serve path takes the whole process down
mid-inference. Return `Result::Err` (or `?` with `.ctx(|| …)`) instead.

| Pattern | Fix |
|---|---|
| `.unwrap()` / `.expect(` | `?` with context, or `match` returning `Err` |
| `.unwrap_or_default()` | it *lies* about errors (turns `Err` into `T::default()`) — handle the case |
| `.unwrap_err()` / `.expect_err(` | `match` the `Result` |
| `panic!` / `unreachable!` / `todo!` / `unimplemented!` | return a descriptive `Err` |

peregrine already `#![deny(clippy::unwrap_used, expect_used, panic)]` in its crates;
this gate enforces it **uniformly** and also catches `unwrap_or_default`,
`unreachable!`, and `todo!`, which the clippy lints miss.

## [U] UB / concurrency footguns — **strict, must be zero**

`static mut`, `transmute`, `mem::forget`, `MaybeUninit::…assume_init()`. None belong
in this codebase; the legitimate low-level work is confined to the crates below.

## [I] `unsafe` outside the expected crates — informational

`unsafe` is expected and reviewed in three domains only:
- **peregrine-io** — io_uring submission, aligned-buffer (`AlignedBuf`) allocation, and the
  OS-interface helpers: `madvise` hugepage/dontneed hints (`mem.rs`), `sched_setaffinity` /
  `mbind` NUMA pinning (`mem.rs`), and the `perf_event_open` counter (`perf.rs`);
- **peregrine-cuda** — the CUDA FFI;
- **peregrine-kernels** — hand-written AVX2 / AVX-VNNI SIMD intrinsics;
- **peregrine-token** — vendored third-party code (marcelroed/gigatoken, MIT):
  upstream style is kept verbatim (unsafe SIMD, unwrap/expect in engine
  internals and tests), so the whole crate is **excluded from this audit's
  file set** — its correctness gate is the id-for-id parity suite against the
  HF `tokenizers` oracle plus the vendored upstream tests.

Any `unsafe` elsewhere is reported for review (the pure-logic crates —
`peregrine-core`, `peregrine-model`, `peregrine-sched`, `peregrine-engine` — should
have none; `peregrine-core` is `#![forbid(unsafe_code)]`).

## What is intentionally NOT flagged

Unlike the security-tool original, the noisy sections don't apply to a numeric
kernel engine: numeric `as` casts (`as usize`/`as f32`/…) and array indexing are
pervasive and intentional; there is no async/HTTP/SQL/crypto surface. Those sections
are omitted rather than waived en masse.

## Current status

`--strict` is green: **P=0, U=0, I=0** (51 files; `peregrine-token` excluded as
vendored). Note: the root-level `audit-bad-patterns.sh` was previously a stale
copy that resolved its repo root incorrectly and scanned zero files — it is now
a shim delegating to `scripts/audit-bad-patterns.sh`, the canonical gate.
Re-run after any change to the streaming, scheduler, or serve paths.

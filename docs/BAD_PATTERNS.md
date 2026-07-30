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

## [B] Silent error swallowing — **strict, must be zero**

A discarded `Result` hides real failures. Every fallible call handles its `Err`
explicitly. Each discard position matches **both** the `_` wildcard and any
`_name` binding (`Err(_e)`, `map_err(|_err|`, `let _tmp =`), so renaming the
ignored binding cannot dodge the gate:

| Pattern | Fix |
|---|---|
| `let _ = fallible()` / `let _named = fallible()` | `if let Err(e) = … { note_advisory_err("…", &e) }` for best-effort ops; or handle the case |
| `if let Ok(x) = …` / `while let Ok(x) = …` | full `match` — the `Err` arm notes the failure or matches the concrete control-flow variant (`VarError::NotPresent`, `TryRecvError::Empty`) |
| `Err(_) =>` / `Err(_e) =>` / `Err(..) =>` | bind the error and include it in the message/log |
| `Err(e) => {}` (bound but empty arm) | do something with `e` — an empty arm is the same silent drop |
| `map_err(\|_\| …)` / `or_else(\|_\| …)` (incl. `unwrap_or_else`/`map_or_else`) | bind and propagate the source error |
| `.to_str().ok()` | explicit match documenting the lax handling |
| `.ok();` / `.err();` as a statement | the whole `Result` is thrown away — handle or `note_advisory_err` |

One exemption: the RAII keep-alive idiom `let _g = gpu_guard();` (any
`…guard()` callee) is *not* flagged — there the named binding is the point,
since `let _ =` would drop the guard immediately, which would be the actual
bug. The three `while let Ok(…) = rx.recv()` worker loops carry
`// audit-allow:` waivers: a channel `recv`'s only error is `Disconnected`,
which *is* the concrete shutdown variant.

**Advisory operations** (madvise/fadvise hints, NUMA pinning, route-stats
persistence, shutdown signalling) are correctness-neutral by design and must not
become hard errors — they report through `peregrine_io::note_advisory_err`
(re-exported from `peregrine-core`), which prints to stderr **only when
`COLI_DEBUG=1`**. Bool-returning hint APIs (kernel may decline) are called as
bare statements — the bool is informational by documented contract.

## [U] UB / concurrency footguns — **strict, must be zero**

`static mut`, `transmute`, `mem::forget`, `MaybeUninit::…assume_init()`,
`unwrap_unchecked`. None belong in this codebase; the legitimate low-level work
is confined to the crates below.

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

`--strict` is green: **P=0, B=0, U=0, I=0** (51 files; `peregrine-token`
excluded as vendored). Note: the root-level `audit-bad-patterns.sh` was previously a stale
copy that resolved its repo root incorrectly and scanned zero files — it is now
a shim delegating to `scripts/audit-bad-patterns.sh`, the canonical gate.
Beyond the gate, error plumbing is structured workspace-wide: every fallible
public API returns `peregrine_core::Error` (thiserror + the `Context`/`.ctx()`
extension) — no `Result<_, String>` surfaces remain outside the vendored crate.
The one production `assert!` (KV-cache append order, `attention.rs`) carries an
`// audit-allow:` waiver: it guards an engine-internal invariant whose silent
violation would corrupt attention output.
Re-run after any change to the streaming, scheduler, or serve paths.

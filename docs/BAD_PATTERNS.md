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
bug. The `while let Ok(…) = rx.recv()` worker loops were rewritten as explicit
`match`es on `RecvError` — a channel `recv`'s only error is `Disconnected`,
which *is* the concrete shutdown variant, so naming it is the fix.

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

## [L] `let Ok(x) = … else { … }` — informational

The let-else twin of the gated `if let Ok(`: the `else` arm drops the `Err` on
the floor. The codebase uses it as the sanctioned style for best-effort reads
(sysfs probes in `sensors.rs`/`topo.rs`, optional JSON caches in `model.rs`),
so it is reported for review rather than gated.

The `ErrorKind`-aware upgrade (`NotFound` = expected absence, anything else
reported through `note_advisory_err`) has landed at the sites where a
non-`NotFound` failure would otherwise be indistinguishable from "not present" —
`topo::pcie_link_by_bdf` via `read_sysfs_opt`, and the artifact readers in
`engine/main.rs` (`compile-plan` now separates a corrupt artifact from a missing
one). The remaining sites are genuine either-way probes.

## What is intentionally NOT flagged

Unlike the security-tool original, the noisy sections don't apply to a numeric
kernel engine: numeric `as` casts (`as usize`/`as f32`/…) and array indexing are
pervasive and intentional; there is no async/HTTP/SQL/crypto surface. Those sections
are omitted rather than waived en masse. Also not flagged: `Ok(_) => {}` arms
(ignoring a success *value* inside a full `match` is normal — the `Err` arm is
what the gate cares about, and it is covered) and `unwrap_or(`/`unwrap_or_else(`
with a used binding (the grep can't tell `Option` defaults from `Result`
swallows; the ignored-closure form `…or_else(|_e|` *is* gated).

## Current status

`--strict` is green: **P=0, B=0, U=0, I=0** (L=26 informational let-else
sites; 51 files; `peregrine-token` excluded as vendored). **No `// audit-allow:`
waivers and no `#[allow(...)]` attributes remain anywhere in first-party code** —
the waiver mechanism still exists, but every former use was replaced with real
handling (see below). Note: the root-level `audit-bad-patterns.sh` was previously a stale
copy that resolved its repo root incorrectly and scanned zero files — it is now
a shim delegating to `scripts/audit-bad-patterns.sh`, the canonical gate.
Beyond the gate, error plumbing is structured workspace-wide: every fallible
public API returns `peregrine_core::Error` (thiserror + the `Context`/`.ctx()`
extension) — no `Result<_, String>` surfaces remain outside the vendored crate.
The former waivers are all gone, each replaced by the handling it was standing in for:
- the production `assert!` on KV-cache append order became `LayerKv::append -> Result`,
  propagated through `mla_attention*` — the invariant is still checked on every
  append, but a violation fails that one request instead of aborting the process
  (the release profile sets `panic = "abort"`, so the assert would have taken
  every concurrent sequence down with it);
- the three `while let Ok(..) = rx.recv()` worker loops became explicit
  `match`es on `RecvError`, naming disconnection as the shutdown signal;
- the off-Linux `let _cpu` / `let _bdf` bindings became cfg-split function
  signatures whose parameters are declared unused (`_cpu: u32`), and the sensors
  test now asserts on the value it used to discard.

Likewise the 14 `#[allow(...)]` attributes were removed by fixing what they
suppressed: `clippy::too_many_arguments` by introducing `MatShape`/`ActScratch`
(kernels), `RouterCfg` (router) and `MoeCfg` (MoE layer); `should_implement_trait`
by implementing `FromStr` for `Dtype` (the inherent parser is now `Dtype::parse`);
`type_complexity` by naming `OfflineArtifacts`; and the three crate-wide
`needless_range_loop` opt-outs by rewriting each flagged loop in iterator form.
Re-run after any change to the streaming, scheduler, or serve paths.

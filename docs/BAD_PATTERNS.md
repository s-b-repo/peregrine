# Bad-pattern catalogue

A re-runnable quality gate for the peregrine workspace, adapted from the
[rustsploit](https://github.com/s-b-repo/rustsploit) audit. It catches crash-vectors
and UB that `cargo clippy` does not flag by default, workspace-wide (per-crate
`#![deny(...)]` attributes can drift; this gate does not).

```bash
scripts/audit-bad-patterns.sh            # full report (shows offending lines)
scripts/audit-bad-patterns.sh --strict   # exit non-zero on any strict hit — the CI gate
scripts/audit-reachability.py --list     # the [R] pass, also invoked by the above
```

Scope: Rust under `crates/*/src`; comment lines are ignored. Strict sections are
**P, B, U, C, Q**; **I, L, R** are informational.

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
persistence, shutdown signalling, optional-artifact loading) are
correctness-neutral by design and must not become hard errors — they report
through `peregrine_io::note_advisory_err` (re-exported from `peregrine-core`),
which prints to stderr **only when `COLI_DEBUG=1`**. Bool-returning hint APIs
(kernel may decline) are called as bare statements — the bool is informational by
documented contract.

**"Optional" is not a licence to be silent about a broken input.** The optional
sidecars (`plan.json`, `tiers.json`, `automaton.json`, `macrostates.json`,
`route_stats.json`) loaded with `let Ok(bytes) = fs::read(…) else { return }`,
which collapsed three distinct situations into one: the file is absent (normal),
the file is corrupt (the operator wants to know), and the file is for a different
model (the fingerprint guard's job). `read_optional_artifact` now separates them
— absent is silent, unreadable or malformed goes to `note_advisory_err`, stale
stays silent because that guard exists to make mixing checkpoints a no-op. This
closed twelve `[L]` findings (28 → 16), and `docs/model-format.md` no longer
claims malformed files are "silently ignored", which contradicted this section.

## [U] UB / concurrency footguns — **strict, must be zero**

`static mut`, `transmute`, `mem::forget`, `MaybeUninit::…assume_init()`,
`unwrap_unchecked`. None belong in this codebase; the legitimate low-level work
is confined to the crates below.

## [C] Lint suppression — **strict, must be zero**

`#[allow(...)]`, `#[deny(...)]` on an item, `#[ignore]` on a test.

The workspace removed all 14 of its `#[allow]`s by fixing what each one hid
(`too_many_arguments` by introducing `MatShape`/`RouterCfg`/`MoeCfg`,
`should_implement_trait` by implementing `FromStr`, and so on). That was a
claim in prose; this makes it a gate, so the next one has to be argued rather
than merged. `#[deny]` belongs at the crate root, not per-item, and `#[ignore]`
rots a test in place — delete it or fix it.

## [Q] Cargo.toml hygiene — **strict, must be zero**

A wildcard version (`dep = "*"`) or a git dependency without `rev`/`tag`/`branch`.

Both make the build non-reproducible. That matters more here than in most
projects: this engine's correctness rests on bit-identity anchors — SIMD kernels
checked bit-for-bit against scalar references, `to_bits()`-exact parallel-vs-serial
tests — and a transitive dependency moving under you can shift those without a
commit to blame.

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

## [R] Shipped but unreachable — informational, and the one grep cannot see

Code that compiles, passes its own unit tests, is documented as live, and **no
production path reaches it**. Every other section here is a regex over lines;
this one needs a definition-and-reference pass, so it lives in a companion
script:

```bash
scripts/audit-reachability.py --list     # also run by the main audit
```

It reports `pub fn`s whose only references are their own definition, a
re-export, or a `#[cfg(test)]` region. Informational by nature — a workspace
legitimately exposes API its binaries don't call — so the number is not a gate.
**The signal is a symbol appearing in this list that a doc or `todo.md` calls
shipped.** Run it before marking anything complete.

It exists because five instances shipped, and every one was found by hand:

| What | Claimed | Reality |
|---|---|---|
| `solve_residency_greedy` / `_sized` | ✅ in `todo.md`; `docs/gpu-cuda.md` described it as the initial-placement policy | zero production callers — placement was round-robin by index |
| `Placement::GpuSpill` | produced by `LaneBalancer::choose`, unit-tested | the sole consumer matched `Placement::Gpu`, so a spill verdict silently became CPU |
| `COLI_REGBUF` → `register_read_buffers` / `read_fixed` | ✅ in `todo.md`; an operator-settable knob in `docs/configuration.md` | no code reads the variable; the `IORING_OP_READ_FIXED` path has no caller. It was then set in a published benchmark arm |
| `COLI_PERF_COUNTERS` → `open_l3_miss_counter` | ✅ twice in `todo.md`, plus `DESIGN.md`, `README.md`, `docs/configuration.md`, `docs/io-and-storage.md`, `docs/adaptive-runtime.md` — "consumers feed `PerfCounter::read` deltas into the prefetch tuner" | that consumer does not exist; nothing calls the opener |
| `SlabPool::checkout_tagged` / `checkin_tagged` | ✅ in `todo.md` as recycling-by-generation; `docs/io-and-storage.md` describes the safety property as active | untagged `checkout`/`checkin` are used (7 and 5 sites); the generation-tagged variants that stop a straggler write into a recycled slab have zero callers, including tests |

*The "Reality" column records the state **at discovery** and is deliberately not
rewritten — this table is the audit's evidence, not a status board. Resolutions
are tracked in `todo.md`. As of 2026-08-06: `solve_residency_sized`,
`Placement::GpuSpill`, `COLI_REGBUF` and `COLI_PERF_COUNTERS` are wired (the
perf counter is opened on the decode thread in `peregrine-engine`'s `serve` and
read at shutdown). Note the counter's row overstates what was fixed: the opener
now has a caller, but the **documented consumer** — miss deltas steering the
prefetch tuner — is a separate piece of work, and a row can be half-resolved
without the list saying so.*

**The audit could not see its own headline example, until 2026-08-02.** It
recorded one definition site per name (`defs.setdefault`) and excluded exactly
that line from the reference scan. A symbol declared twice — the
`#[cfg(target_os = "linux")]` implementation and its `#[cfg(not(...))]` stub —
therefore had its second declaration counted as a *caller* of the first: the two
vouched for each other and the symbol scored as reached. `register_read_buffers`
is exactly that shape, so `COLI_REGBUF` — the row above, the case the script is
named for — was invisible to the script. Now every definition site is recorded
and excluded; the count went 49 → 55, newly surfacing `register_read_buffers`,
`read_direct_many`, `fixed_buffer_count`, `is_registered`, `with_compression`
and `pcie_link_by_bdf`. The lesson generalises: **a detector that cannot find
the defect it was written for is not evidence of absence.** When this list
shrinks, check that the symbol was wired or deleted — not that the scan stopped
seeing it.

### The 2026-08-02 triage

A second sweep found **four instances larger than any of the five above**, none
of which the script had surfaced — three because of the bare-name blind spot
below, one because nobody had checked. Recorded here so the next run's *diff* is
the signal rather than the count.

| Feature | Claimed | Reality |
|---|---|---|
| **LFRU eviction** (`tier.rs`) | six doc sites + two ✅ roadmap entries; `todo.md` specifically said "**not** plain LRU" | zero callers anywhere. The live policy is `warmcache.rs::evict_to_budget` — priority-weighted LRU. Heat never enters the victim score |
| **DSA sparse attention** (`dsa.rs`, `mla_attention_dsa`) | `DESIGN.md` M5, `model-format.md` "auto-detected", vs-colibrì feature table | `Indexer` is never constructed and nothing appends a key, so the path cannot run even if something called it |
| **`peregrine-sched`** | `README.md` status table: ✅, "`moe_streamed` overlaps io_uring streaming ∥ CPU compute" | **no crate depends on it.** Production MoE is `peregrine-model/concurrent.rs`. It is not used as an oracle either — no test compares the two *(resolved 2026-08-06: `streamed_matches_the_production_concurrent_path` now compares them over the same container bytes; still no dependents, which is now the point rather than the defect)* |
| **MTP speculative decode** | vs-colibrì: "MTP head wired" | `generate_speculative` has no caller in either binary; there is no `--draft` flag or `DRAFT` knob |

All four doc sets are now corrected. The code is left in place: deleting a
tested subsystem is a product decision about intent, not a hygiene fix, and
`todo.md` tracks each as 🟡 rather than ✅.

**The scan's own limits, stated so absence is not read as proof.** References
match by **bare name** with no type resolution, so any method sharing a name with
something called elsewhere scores as reached — `Indexer::{load, select, reset}`
are invisible for exactly this reason, which is why an entire unreachable module
went unreported. Treat common method names (`new`, `load`, `get`, `push`, `len`,
`clear`, `reset`, `select`) as **unchecked, not clean**. A separate bug fixed the
same day: lines beginning `*` were skipped as comment continuations, which also
ate dereference assignments, so `f16_to_f32` — reachable only via
`*o = f16_to_f32(...)` — was reported dead. Block comments are now tracked
properly.

The common shape is worth naming: **the feature was built, the documentation was
written from the intent, and the wiring was never done.** Tests do not catch it
because the code under test is genuinely correct — it simply runs nowhere. Nor
does clippy: `pub` items are reachable by definition from outside the crate, so
`dead_code` stays quiet. The audit's other sections are all "this line is
dangerous"; this one is "this line never runs."

## What is intentionally NOT flagged

The upstream rustsploit catalogue runs to 125+ patterns across A–Q. Most of its
categories describe a different program — an async, network-facing module
framework — and are omitted here rather than waived en masse:

- **Network / HTTP / SQL / crypto / mass-scan wrappers** (its G2, H, K0, L, M):
  no such surface exists in an inference engine.
- **Async and blocking** (its F): peregrine's engine thread is *deliberately*
  blocking and owns the model; `std::thread::sleep` and `std::fs` are correct
  there, not defects.
- **Numeric `as` casts** (its E1/E2) and **array indexing** (its D): pervasive and
  intentional in kernel code, where bounds are structurally guaranteed by shape.
  Gating them would produce thousands of hits and teach the team to ignore the
  audit — the failure mode a gate must not have.
- **`assert!` in production** (its A15): peregrine had exactly one, on KV-cache
  append order, and it was removed for a different reason — the release profile
  sets `panic = "abort"`, so an assert would have taken every concurrent sequence
  down with the one bad request. It is now a propagated `Result`.

What *was* worth adopting: **C** (lint suppression) and **Q** (Cargo hygiene),
both mechanical and both already at zero, so gating them costs nothing and keeps
prose claims honest. Also not flagged: `Ok(_) => {}` arms
(ignoring a success *value* inside a full `match` is normal — the `Err` arm is
what the gate cares about, and it is covered) and `unwrap_or(`/`unwrap_or_else(`
with a used binding (the grep can't tell `Option` defaults from `Result`
swallows; the ignored-closure form `…or_else(|_e|` *is* gated).

## Current status

`--strict` is green: **P=0, B=0, U=0, C=0, Q=0, I=0** (informational: L=23
let-else sites, R=21 unreferenced `pub fn`s; 62 files; `peregrine-token` excluded
as vendored).

*These counts had drifted — this block read `L=16, R=50, 55 files` while the gate
reported `L=23, R=21, 62 files`, which is the same failure the gate exists to
catch, one level up: a status line nobody re-ran. Regenerate it from
`scripts/audit-bad-patterns.sh --strict` rather than editing the numbers by hand.
The gate also reads the **working tree**, not the index, so "green" is a property
of what is checked out — a green HEAD says nothing about a dirty tree, and a
2026-08-08 session proved it by regressing all five strict sections to
`P=9, B=18, C=2` without touching a committed line.*

**All ten first-party crates now carry
`#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`** —
`peregrine-tools` was the last holdout and qualified all along, so adopting it
was a ratchet rather than a cleanup. Note each binary target is its own crate
root, so that crate carries the attribute five times. Integration tests under
`crates/*/tests/` are separate crates again and inherit **no** crate-root
attribute, so `clippy.toml`'s `allow-expect-in-tests = false` is unenforced
there and has to be kept by hand; the audit does not scan them either
(`*/src/*` only).

Of the R list, five entries were confirmed as genuine unreachable
features and are catalogued in [R] above; the rest are library API the binaries
do not happen to call. Two of the five — the `perf_event_open` counter and the
slab pool's generation tagging — were downgraded from ✅ to 🟡 in `todo.md` when
this pass found them. **No `// audit-allow:`
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

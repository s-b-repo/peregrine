# Bad-pattern catalogue

A re-runnable quality gate for the peregrine workspace. It started from the
[rustsploit](https://github.com/s-b-repo/rustsploit) audit and has since merged in
the `unifur` matrix (`/home/cortix/Downloads/inc/unifur/audit-bad-patterns.sh`),
which widens coverage from 8 sections to 17 and adds the per-section CLI. It
catches crash-vectors and UB that `cargo clippy` does not flag by default,
workspace-wide (per-crate `#![deny(...)]` attributes can drift; this gate does
not).

```bash
scripts/audit-bad-patterns.sh                # full report (shows offending lines)
scripts/audit-bad-patterns.sh --strict       # exit non-zero if a section is over budget — the CI gate
scripts/audit-bad-patterns.sh --section A    # one section (repeatable; old letters accepted)
scripts/audit-bad-patterns.sh --files f.txt  # scan only the .rs files listed, one per line
scripts/audit-bad-patterns.sh --max-report 0 # every hit, instead of the first 20 per pattern
scripts/audit-bad-patterns.sh --budgets      # print each section against its ratchet budget
scripts/audit-reachability.py --list         # the [Z] pass, also invoked by the above
```

Scope: **every** Rust file under `crates/` — `src/`, `tests/`, `examples/` and the
`src/bin/` targets, excluding the two vendored crates (`peregrine-token`,
`peregrine-ctx`) but **including** the three `peregrine-ctx` files peregrine
authored (`src/lib.rs`, `src/main.rs`, `examples/ctxbench.rs`). See `[E]`.
Integration tests and examples are *separate crates*: they inherit no crate-root
`#![deny(...)]`, so nothing else enforces the law there. They were excluded until
2026-10-06 for the honest reason that they had never been scanned; that added 240
hits on the first run, which is the number the ratchet below starts from.

## Ratchet, not zeroes

**`--strict` fails when a section's count *exceeds* its budget**, not when it is
non-zero. Seven sections are at zero and gated absolutely (**J** secrets, **M**
injection, **N** UB, **T** Cargo hygiene, **H**, **I**, **K**, **O**, **P**); the
rest carry a budget equal to the debt they already have.

That is a deliberate trade, and the reasoning is worth stating because the
alternative looks stricter. Two of these sections contradict a documented house
style outright: `[E]` reports ~2.3k numeric casts and `unsafe` sites, which the
kernels need (they mirror the C ports line-for-line), and `[D]` reports ~480
indexing sites in code where bounds are guaranteed by shape. Gating those at zero
means rewriting the numerical core into a different style to satisfy a linter —
and a gate that is red from the first second teaches the team to ignore it, which
is the one failure mode a gate must not have. A ratchet has the opposite
property: **new code cannot add a hit**, the existing debt is a visible number
rather than a suppressed one, and the budget can only move down.

Raising a budget is therefore a review conversation, not a bookkeeping update.

**Letter map.** unifur's letters won the collision (its matrix is the superset),
so three peregrine letters moved. The old letters still work as aliases to
`--section`, and nothing in the tree depends on the old names.

| Was | Now | Meaning |
|---|---|---|
| `P` | **A** | panicking error handling |
| `B` | **B** | silent error swallowing (its old info-only `[L]` twin is now `[B]`, informational) |
| `U` | **N** | UB / concurrency footguns |
| `C` | **C** | lint suppression and lint-bypass idioms |
| `Q` | **T** | Cargo.toml hygiene |
| `I` | **E** | `unsafe` / numeric casts (informational) |
| `R` | **Z** | shipped-but-unreachable (informational) |
| `K` | **X** | the hand-run audits grep cannot do (`X1`–`X3` below) |
| — | **D, F, G, H, I, J, K, M, O, P** | new, from the unifur matrix |

**Waivers.** A line carrying `// audit-allow: <reason>` is dropped from the counts,
and the reason is mandatory: a bare `// audit-allow:` with nothing after it is a
strict `[C]` hit, because a silent exemption is indistinguishable from a rule
nobody enforced. Every waiver in the tree is a token-count field name that the
secret regexes (`[J]`) would otherwise read as a credential.

## [A] Panicking error handling — **strict, must be zero**

A panic in the streaming/scheduler/serve path takes the whole process down
mid-inference. The release profile is `panic = "abort"`, so there is no unwinding
and no partial recovery: an `unwrap` in the serve loop is a dead server and a
dropped in-flight batch. Return `Result::Err` (or `?` with `.ctx(|| …)`) instead.

| Pattern | Fix |
|---|---|
| `.unwrap()` / `.expect(` | `?` with context, or `match` returning `Err` |
| `.unwrap_or_default()` | it *lies* about errors (turns `Err` into `T::default()`) — handle the case |
| `.unwrap_err()` / `.expect_err(` | `match` the `Result` |
| `panic!` / `unreachable!` / `todo!` / `unimplemented!` | return a descriptive `Err` |

Tests are **not** exempt: `clippy.toml` sets `allow-unwrap-in-tests = false`,
`allow-expect-in-tests = false` and `allow-panic-in-tests = false`, and this gate
reads the whole file, `#[cfg(test)]` regions included. The compound shapes from
the unifur matrix — `.first().unwrap()`, `.lock().unwrap()`, `.parse().unwrap()`,
`.chars().next().unwrap()`, `.position(..).unwrap()` — are all subsumed by
`\.unwrap\(\)`; only the base patterns are listed so the counts stay comparable
across runs.

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
| `if let Ok(x) = …` / `while let Ok(x) = …` / `if let Err(_) = …` | full `match` — the `Err` arm notes the failure or matches the concrete control-flow variant (`VarError::NotPresent`, `TryRecvError::Empty`) |
| `Err(_) =>` / `Err(_e) =>` / `Err(..) =>` | bind the error and include it in the message/log |
| `Err(e) => {}` (bound but empty arm) | do something with `e` — an empty arm is the same silent drop |
| `map_err(\|_\| …)` / `or_else(\|_\| …)` / `unwrap_or_else(\|_\| …)` | bind and propagate the source error |
| `.to_str().ok()` | explicit match documenting the lax handling |
| `.ok();` / `.err();` / `.await.ok()` as a statement | the whole `Result` is thrown away — handle or `note_advisory_err` |

`if let Err(_)` deliberately matches **only** the discarding form. `if let Err(e) = …`
that then does something with `e` is handling the error, which is the entire point
of the rule; gating the shape rather than the behaviour is what makes a gate
teachable.

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
closed twelve let-else findings (28 → 16), and `docs/model-format.md` no longer
claims malformed files are "silently ignored", which contradicted this section.

### [B/info] `let Ok(x) = … else { … }` — informational

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

## [C] Lint suppression — **strict, must be zero**

`#[allow(...)]`, `#[deny(...)]` on an item, `#[ignore]` on a test, and a bare
`// audit-allow:` with no reason after it.

The workspace removed all 14 of its `#[allow]`s by fixing what each one hid
(`too_many_arguments` by introducing `MatShape`/`RouterCfg`/`MoeCfg`,
`should_implement_trait` by implementing `FromStr`, and so on). That was a
claim in prose; this makes it a gate, so the next one has to be argued rather
than merged. `#[deny]` belongs at the crate root, not per-item, and `#[ignore]`
rots a test in place — delete it or fix it.

### [C/info] lint-bypass idioms — informational

Underscore-prefixed parameters (`fn f(_device: &Device)`) hide a value from the
unused-parameter lint. 52 sites, most of them real: phantom device handles,
cfg-split signatures where one platform has no use for the argument. Informational
because the fix — `_`-prefixing — is already the compiler-sanctioned answer; the
review question is only whether the argument should have been dropped outright.

## [T] Cargo.toml hygiene — **strict, must be zero**

A wildcard version (`dep = "*"`) or a git dependency without `rev`/`tag`/`branch`.

Both make the build non-reproducible. That matters more here than in most
projects: this engine's correctness rests on bit-identity anchors — SIMD kernels
checked bit-for-bit against scalar references, `to_bits()`-exact parallel-vs-serial
tests, and the numerical certificates in `peregrine-verify` — and a transitive
dependency moving under you can shift those without a commit to blame.

## [E] `unsafe` and numeric casts — informational

Numeric `as` casts are intentional here (the kernels mirror the C ports
line-for-line, where PyTorch layouts and `f32`/`u8` boundaries are explicit), and
`unsafe` is **expected** in four domains only:

- **peregrine-io** — io_uring submission, aligned-buffer (`AlignedBuf`) allocation, and the
  OS-interface helpers: `madvise` hugepage/dontneed hints (`mem.rs`), `sched_setaffinity` /
  `mbind` NUMA pinning (`mem.rs`), and the `perf_event_open` counter (`perf.rs`);
- **peregrine-cuda** — the CUDA FFI;
- **peregrine-kernels** — hand-written AVX2 / AVX-VNNI SIMD intrinsics;
- **peregrine-par** — one documented lifetime-erasure (a borrowed closure handed to a
  persistent worker pool), made sound by a fork-join barrier and hidden behind a
  safe API.

`unsafe`, `extern "C"`, `*const`/`*mut` casts and `mem::zeroed` outside those
crates are reported for review (the pure-logic crates — `peregrine-core`,
`peregrine-model`, `peregrine-sched`, `peregrine-engine`, `peregrine-verify` —
should have none; `peregrine-core` is `#![forbid(unsafe_code)]`). `mem::zeroed`
is informational rather than a `[N]` hit on purpose: all four of its uses are
zero-initialising a POD libc struct (`rlimit`, `cpu_set_t`, `stat`, `rusage`),
which is the sanctioned use and the wrong thing to flag next to a real one.

**peregrine-token** is vendored third-party code (marcelroed/gigatoken, MIT):
upstream style is kept verbatim (unsafe SIMD, unwrap/expect in engine internals and
tests), so the whole crate is **excluded from this audit's file set** — its
correctness gate is the id-for-id parity suite against the HF `tokenizers` oracle
plus the vendored upstream tests.

**peregrine-ctx** is vendored the same way, and used to be scanned. It is a
verbatim port of subnext/DCR (crate `dcr`, pinned at 609451b): of its 34 `src`
files, 32 are byte-identical to upstream and the two that differ (`lib.rs`,
`main.rs`) differ only by the `dcr::` → `peregrine_ctx::` prefix rename and the
CLI binary name; its 22 tests and 2 of its 3 examples are upstream byte-for-byte
after that same rename. Auditing ported files audits subnext, which peregrine has
no mandate over — and "fixing" one is not a cleanup but a fork, because omni's
`verify-cross-repo.sh` fails on any ported file that drifts. Its gate is that
cross-repo check.

Three files in `peregrine-ctx` are peregrine's own work and **stay in the file
set**: `src/lib.rs`, `src/main.rs`, `examples/ctxbench.rs`. Those are the files
to fix when a finding points into `peregrine-ctx`.

Neither casts nor `unsafe` are gated: ~2.2k cast sites and ~120 unsafe sites,
almost all intentional. Gating a construct this pervasive would produce thousands
of hits and teach the team to ignore the audit — the failure mode a gate must not
have. What *is* gated is the UB class they can imply ([N]).

## [D] Panic vectors (indexing and slicing) — informational

Literal indices, slice patterns, `split_at`, `chars().nth()`. The same reasoning
as `[E]`: a kernel engine indexes deliberately, bounds are structurally guaranteed
by `MatShape`/`RouterCfg`, and the house style is an explicit index loop that
mirrors the C port for line-by-line verification.

The signal is not the count, it is the **kind**: a slice built from a length that
came off the wire, an mmap, or a tensor header is where a short-buffer panic lives,
and that is worth a look when one appears here. `Option::get`-shaped code and
`get(..n)` on a length-checked buffer are noise.

## [F] Blocking calls — informational

`std::thread::sleep`, `std::fs::*`, `std::process::Command`, `std::net::*`,
`std::io::stdin`. The reactor exists to keep blocking syscalls off the hot path,
so a blocking call in a request path is a finding; the same call in a bench, a
setup path, or `peregrine-engine`'s deliberately-blocking engine thread is not.
Grep cannot tell the two apart — that is what makes this informational — so when
a `[F]` hit is on the hot path, the fix is to route it through
`peregrine_io::{pread_many, mmap_many, fadvise_many}` rather than to silence it.

## [G] Logging hygiene — informational

`dbg!`, `format!("{:?}", … err)`, `.context("")`, `println!` in library code.
`println!` is the interesting one: stdout is the serve protocol on stdout, so a
stray print corrupts a response frame. Everything here is also a signal about
*where errors are being dropped*, which `[B]` catches structurally.

## [H] HTTP layer — informational

A `reqwest::Client` built per request loses connection reuse; an unchecked status
is a silent 500. **Zero hits is the expected result and that is the point** — the
engine speaks its own JSON protocol over stdio and its own socket server, so this
section is a tripwire against an HTTP client creeping into a path where the
connection cost would be paid per request.

## [I] Iterator glitches — informational

`collect::<Result<Vec<_>, _>>().unwrap()`, `.zip(..).unwrap()`,
`.iter().find(..).expect(..)` — panicking shapes that survived because they sit
inside an iterator chain a reader skims past. Rare (zero at last run), kept because
the cost of the regex is nil.

## [J] Secrets — **strict** · style — informational

Two different rules sharing one section, because both used to be one catalogue
entry. The **strict** half is the one that matters: API keys, bearer tokens, AWS
keys, GitHub and Slack tokens, `-----BEGIN … PRIVATE KEY-----` blocks, and
`("admin","admin")`-shaped literals. Nothing in this tree may contain one, and
`--strict` fails on a single line.

The **informational** half is style: `== ""` instead of `is_empty()`,
`.len() == 0`, double `.to_string().to_string()`, `Box<dyn …>` in a hot path,
`use foo as _`. `Box<dyn>` is the interesting one here — trait objects in the
request path are a real cost — but there are 15 of them in API surface that is
either cold or behind a feature, so it reports rather than gates.

The `TODO|FIXME|HACK` regex from the upstream matrix was **dropped on purpose**:
comment lines are stripped before matching, so it could only ever fire on a
non-comment line, i.e. never on the thing it was written for.

## [K] Crypto — informational

MD5, SHA-1, DES/3DES, RC4, ECB-mode AES, `rand::thread_rng()`, `rand::random()`.
Zero hits at last run, and zero is the point: the Merkle tree in `peregrine-ctx`
makes SHA-256 load-bearing for content addressing, and reproducible *test* RNGs
(deterministic xorshift fixtures, `hash.rs`'s own counter) are how the
bit-identity tests stay reproducible. A non-CSPRNG in a production path is a
finding; in a fixture it is the mechanism.

## [M] SQL / command injection — **strict, must be zero**

`format!("SELECT … {…}")` (a formatted query string is a parameterisation that was
not written), and shelling out through `/bin/sh -c` with an interpolated argument.
There is no SQL in this workspace, so the query rules are a tripwire; the shell
rules are not hypothetical — `scripts/` drives `bench-*`, `setup-*` and the serve
wrappers, and a `Command::new("sh").arg("-c")` in a crate would be the first place
an operator-supplied string reached a shell.

## [N] UB / concurrency footguns — **strict, must be zero**

`static mut`, `mem::forget`, `assume_init()`, `MaybeUninit::uninit()`,
`unwrap_unchecked`, and `transmute` outside `peregrine-par`. None belong in this
codebase; the legitimate low-level work is confined to the crates in `[E]`, and
the one lifetime-erasure `peregrine-par` needs is encapsulated behind a safe API
precisely so it can be listed here.

## [O] Performance footguns — informational

`.iter().count()`, `.clone().iter().enumerate()`, `with_capacity(0)`, nested
`Box::new`, `Regex::new(..).unwrap`. All zero at last run. The rule that generalises:
in an engine, a `clone()` on a hot path is a throughput decision, and the decision
belongs in a comment or a certificate, not to be rediscovered in a profile.

## [P] API hygiene — informational

A `pub const … = "http…"` endpoint baked into a library, and credential-shaped
field names in a `pub struct`. The second is the interesting one: a public struct
with a `password`/`secret`/`api_key` field is a `Debug`-print hazard and an
accidental-log hazard, and this section is where a new one shows up.

## [Z] Shipped but unreachable — informational, and the one grep cannot see

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

> **Corrected 2026-10-06.** The LFRU row is stale and understates the code by a
> long way. `peregrine-model/src/gpu.rs` calls `peregrine_io::pick_lfru` to pick
> the hot-store swap (`COLI_GPU_TIER_SWAP=lfru`), with `gpu.rs`'s tests pinning
> the hysteresis edge; the policy is reachable from the production GPU path and is
> merely **off by default** (`COLI_GPU_TIER_SWAP` defaults to `replan`), and
> `peregrine_io::tier::decay` has the same problem. It is now certified rather
> than merely tested — see `tier` in `peregrine-verify`. Two *other* consequences
> of that row being wrong are worth naming, because they are the general shape:
> the reachability scanner cannot see a caller in another crate by bare name, and
> "off by default" is invisible to it just as much as "never called" is.
| **DSA sparse attention** (`dsa.rs`, `mla_attention_dsa`) | `DESIGN.md` M5, `model-format.md` "auto-detected", vs-colibrì feature table | `Indexer` is never constructed and nothing appends a key, so the path cannot run even if something called it |

> **Corrected 2026-10-06.** Also stale. `model.rs` dispatches to
> `mla_attention_dsa_indexed` whenever `IndexerWeights` load (the `dsa.rs` path
> the row was written about is the older *stateful* `Indexer`, which the engine
> does not use — it selects with `dsa::select_topk` per query instead), and
> `qwen4.rs` calls that same selection. What is true is narrower and worth
> saying precisely: the *sparse selection* is live, while the incremental
> key-append index that would amortize it across tokens is not.
| **`peregrine-sched`** | `README.md` status table: ✅, "`moe_streamed` overlaps io_uring streaming ∥ CPU compute" | **no crate depends on it.** Production MoE is `peregrine-model/concurrent.rs`. It is not used as an oracle either — no test compares the two *(resolved 2026-08-06: `streamed_matches_the_production_concurrent_path` now compares them over the same container bytes; still no dependents, which is now the point rather than the defect)* |

> **Corrected 2026-10-06.** "No crate depends on it" is wrong: `peregrine-serve`
> and `peregrine-engine` both list `peregrine-sched` as a dependency. The
> remaining true part is that `moe_streamed` itself is still test-only, so the
> crate is a dependency without carrying the feature the dependency is for —
> which is a different, smaller, and more actionable claim than the one recorded
> here.
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

## [X] The hand-run audits — informational, and grep cannot see these either

Everything above is a regex over lines. The three rules in this section are
judgements: no script can apply them, so they are written down instead, and
they are the ones that have actually cost time.

### [X1] A device kernel without an accuracy test

**A new GPU kernel ships with a test that compares it against a CPU or f32
reference. A timing harness is not evidence of correctness, because it
structurally cannot see a fast wrong answer.**

This is a rule with two scars behind it, both from 2026-08-16, both in the same
seam, from opposite directions:

- A gated-attention layout bug: the host packs `[query | gate]` per head
  interleaved, the port read it as flat halves.
- A decode GEMV kernel that **benchmarked 3× faster than the kernel it replaced
  while computing nonsense** (rms 4.2e-1 against an f32 truth the CPU path hits
  at 3.0e-3). Cause: host int4 is **bias-8** (`nibble - 8`), device int4 is
  **two's-complement** (`a & 8 ? a - 16 : a`) because `coli_cuda_tensor_upload`
  runs `offset_to_signed_s4` on the way over. Both map `0..15` onto `-8..7`, by
  different permutations — so reading one with the other's rule yields plausible
  garbage rather than an error or a crash.

The benchmark was perfectly happy in both cases. What caught the second one, on
its first run, was an assertion of RMS-versus-f32-truth added to the same commit
as the kernel. That is the whole rule: **the accuracy test is not optional
polish, it is the only defence against a kernel that is fast and wrong**, and
the encoding asymmetry that produces those is documented at the seam
(`upload_tensor_i4` in `peregrine-cuda`).

Not a gate — "is there a test for this kernel" is a judgement, not a regex. The
signal is a `.cu` entry point whose only Rust-side exercise is a benchmark or an
example.

### [X2] …and it must prove the device actually ran

The companion failure, found the same week and strictly worse than a failing
test: **a green test that certifies a path it never executed.** One device test
passed twice while proving nothing —

- on a 16-wide fixture, because the C entry *validates the shape and declines*,
  so the upload succeeded, the kernel never launched, and the CPU fallback
  produced the (correct) answer the assertion then checked;
- at 256-wide, because the test drove `forward_step` over a 5-token prompt —
  which is **prefill** (`s_n = 5`), and the device dispatch is `s_n == 1` only,
  so decode was never reached.

Both are the same shape: the fallback is *correct*, so every assertion about
correctness passes, and the test reports success for a kernel that did not run.
A benchmark cannot see this either — the fallback is merely slower.

The fix is a **witness that the device ran**, not more correctness assertions.
The cheap one is to assert the device result *differs* from the CPU path: these
kernels are deliberately more accurate (f32 throughout vs the CPU's int8
activations), so bit-identical output is proof of a fallback, and the test fails
on it. Size fixtures past the entry's own validation, and drive the shape the
dispatch actually takes.

Together with [X1]: a timing harness cannot see a kernel that is fast and wrong; a
correctness harness cannot see a kernel that never ran. A device test needs both
guards.

### [X3] Empty output is not evidence of broken decode

**A response filter that drops a region makes an empty completion a *legitimate*
render.** `OutputFilter` removes `<think>…</think>` by design (reasoning is
dropped, not shown), so a request whose token budget expires before the closing
tag renders every generated token invisible: `completion_tokens == max_tokens`,
zero SSE content deltas, `content: ""`, `finish_reason: stop`. Nothing is broken.
The model reasoned, the filter ate it, the budget ran out mid-thought.

This cost most of a night in 2026-08-17, and the expensive part was not finding
the filter — it was that the symptom is **content-dependent**, so it looks like
whatever else changed. Two prompts of *identical* token length landed on opposite
sides (one answered fast, one reasoned longer), and enabling the GPU flipped
prompts across the line because slightly different numerics change the reasoning
*trajectory*. That produced a clean, reproducible, and completely wrong
attribution to the GPU placement path — a three-arm partition where every arm's
result was real and the conclusion was not.

**The corollary is the practical part:** when a filter can eat the output, the
fastest discriminator is not more instrumentation, it is *a prompt whose answer
the filter cannot swallow*. Vary the input before you bisect the code.

And the general form, which is the rule the whole [X] family serves: **a
differential experiment is only as good as the axis you vary, and the axis worth
varying is the one you are most confident does not matter.** The confound here
sat in the axis held fixed — one prompt, reused all night for "verification",
which is also what hid the bug in the first place.

`POST /v1/debug/tokenize` exists for this: it returns the rendered prompt
verbatim (control markup included), the exact ids, and each id's rendered piece.
The pairing is what earns its keep — an id that renders to nothing is invisible
in a list of ids and obvious beside its piece.

## What is intentionally NOT flagged

The upstream rustsploit catalogue runs to 125+ patterns. After the merge, most of
its categories are present as **informational** sections rather than being waived
en masse — a rule that reports is still a rule, and a rule that reports teaches
nothing except by being read. What follows is the part that is genuinely absent:

- **`assert!` / `assert_eq!` in production code** (its A15): peregrine had exactly
  one `assert!`, on KV-cache append order, and it was removed for a different
  reason — the release profile sets `panic = "abort"`, so an assert would have
  taken every concurrent sequence down with the one bad request. It is now a
  propagated `Result`. `assert_eq!` and friends stay legal inside `#[cfg(test)]`
  and are **not** gated there, because a test that cannot assert anything is not a
  test.
- **`Ok(_) => {}` arms**: ignoring a success *value* inside a full `match` is
  normal. The `Err` arm is what the gate cares about, and it is covered.
- **`unwrap_or(` / `unwrap_or_else(` with a *used* binding**: the grep cannot tell
  an `Option` default from a `Result` swallow. The ignored-closure forms
  `…or_else(|_e|` / `…map_err(|_e|` **are** gated.
- **`todo!` / `unimplemented!` in a `#[cfg(feature = …)]` stub**: there are none.
  If one appears, it is gated by `[A]` like any other — the point of a feature stub
  is that the code behind it is unreachable *today*, not that its error handling
  is exempt.
- **The `TODO|FIXME|HACK` regex** — dropped outright, because comment lines are
  stripped before matching and it could therefore only ever fire on a non-comment
  line. A rule that cannot match its own motivating case is worse than no rule.
- **Vendored code**: `peregrine-token` and the ported files of `peregrine-ctx`
  are excluded from the file set (see `[E]`). The three `peregrine-ctx` files
  peregrine authored are not vendored and are still scanned.

What the merge added that is genuinely worth gating, and why these three:
`[C]` (lint suppression, mechanical and at zero), `[T]` (Cargo hygiene, mechanical
and at zero), and `[J]` secrets — the first two cost nothing to gate and keep a
prose claim honest; the third is the only rule here whose violation is not
recoverable by a later commit.

## Current status

Regenerate this block from `scripts/audit-bad-patterns.sh --budgets`; do not edit
the numbers by hand. The gate reads the **working tree**, not the index, so "green"
is a property of what is checked out — a green HEAD says nothing about a dirty
tree, and a 2026-08-08 session proved it by regressing every strict section to
`A=9, B=18, C=2` without touching a committed line.

Measured 2026-10-16 over the full file set (120 files, `peregrine-token` and the
ported files of `peregrine-ctx` excluded, the three `peregrine-ctx` files
peregrine authored included), on a working tree with uncommitted work in
`peregrine-ctx`, `peregrine-model`, `peregrine-serve`, `peregrine-tools`:

```
== peregrine bad-patterns audit (120 files) ==
gated:   A/P=37  B=67  C=7  J=0  M=0  N/U=0  T/Q=0
ratchet: D=425  E/I=2158  F=190  G=387  H=0  I=0
         J/style=11  K=0  O=0  P=0  R/Z=46
         B/let-else=33  C/bypass=48
→ --strict exits 0 (every section at or under budget)
```

Read that honestly. The budgets did **not** move — they are still A=285, B=109,
C=8, Z=65 — but the measured counts fell far below them, because exempting
`peregrine-ctx`'s ported files removed 413 `[A]` hits of upstream code from the
scanned set (the file set went 166 → 120 files, and `[Z]` went 67 → 46 with
`audit-reachability.py` moved in step). So the ratchet is currently *loose*
rather than tight: `[A]` has 248 hits of headroom. The honest follow-up is to
regenerate these budgets with `--budgets` and move them down to the measured
values in a separate change, because a scope edit and a budget edit should be
argued separately. It has not been done yet.

The previous measurement (2026-10-06, 166 files, `peregrine-ctx` still scanned)
read `A=285 B=109 C=8 … R/Z=65`, i.e. every section exactly at its budget, which
is what the ratchet was calibrated against. Of those 285 `[A]` hits, 240 were
`.expect`/`.unwrap` in integration tests and examples, and most of them lived in
the now-exempt `peregrine-ctx` port — which is why the count dropped by 248
without a single line of code changing.

Read that honestly. `[A]`/`[B]`/`[C]` are **not** at zero. The measured `[A]`=37
splits 22 in examples, 8 in `src`, 7 in tests — still concentrated in
integration tests and examples, which inherit no crate-root `#![deny(...)]`,
which is why they were the first thing the audit was extended to cover. Five
`[A]` hits were removed in the 2026-10-06 pass and the budget followed them down
(290 → 285): the `.unwrap`/`.unwrap_err` sites that clippy's `deny` was *already*
failing on, converted to named errors. A ratchet whose budget drifts upward is
not a ratchet. `[J]` secrets, `[M]` injection, `[N]` UB and `[T]` Cargo hygiene
**are** at zero — and `[M]` reached zero for a real reason that round: the
extended scan found `iobench` shelling out to `sh -c 'ulimit -l'` to report
`RLIMIT_MEMLOCK`, which is now a direct `getrlimit` through
`peregrine_io::memlock_limit()`.

`--strict` is green: **A=37, B=67, C=7, J=0, M=0, N=0, T=0**, every one at or
under its ratchet budget (informational: 33 let-else sites, 46 unreferenced
`pub fn`s; 120 files; `peregrine-token` and the ported `peregrine-ctx` files
excluded as vendored). This line used the pre-ratchet letter scheme
(`P=0, B=0, U=0, … 62 files`) and was left behind when the budget table landed.

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

Of the `[Z]` list, five entries were confirmed as genuine unreachable
features and are catalogued in [Z] above; the rest are library API the binaries
do not happen to call. Two of the five — the `perf_event_open` counter and the
slab pool's generation tagging — were downgraded from ✅ to 🟡 in `todo.md` when
this pass found them. **No bare `// audit-allow:`
waivers and no `#[allow(...)]` attributes remain anywhere in first-party code** —
the waiver mechanism still exists, but every former use was replaced with real
handling (see below), and the ten that remain are all *reasoned*
(`// audit-allow: LM token count, not a credential`) token-count field names that
`[J]` would otherwise read as secrets. Note: the root-level `audit-bad-patterns.sh` was previously a stale
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

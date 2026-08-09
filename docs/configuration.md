[« Docs index](README.md)

# Configuration reference

Every knob is an environment variable, and every one has a working default — you
should not need this page to run the engine. Reach for it when you want to know
what a specific knob does. If your question is instead *"decode is slow, what
should I turn?"*, start at [Performance tuning](performance-tuning.md), which
ranks the same knobs by measured effect.

The `COLI_` prefix is kept from colibrì for drop-in compatibility.

**Reading the tables.** Boolean gates accept `1`/`0` (and `true`/`false` where
noted). Knobs whose behaviour needs more than a line have a `###` note below their
table; the table cell links to it. Nothing on this page is a secret handshake —
every default is in the code at the path given.

## ⚠ Two classes of knob

**Most knobs are output-neutral**: they change how fast a token arrives, never
which token it is. Those can be A/B'd freely against a bit-identity assertion.

**Five knobs change token values.** They are real quality/performance trades, not
tuning, and each must be gated with `Model::prediction_flip_rate` against the
unmodified configuration before you rely on it:

| Knob | What moves |
|---|---|
| [`COLI_KV_DTYPE=f16`](#coli_kv_dtype) | stored KV latents lose precision |
| [`COLI_ROUTE_MIN_SHARE`](#coli_route_min_share) | drops low-gate experts from the MoE sum |
| [`COLI_DSA`](#coli_dsa) | attends only the indexer's top-k cached positions |
| [`COLI_MLA_ABSORB`](#coli_mla_absorb) | absorb and dense agree algebraically, not numerically |
| [`COLI_CUDA_FUSED_REDUCE`](#coli_cuda_fused_reduce) | low bits only — `f32 +=` is not associative |

> Earlier revisions of this page opened by claiming *"every one of them affects
> performance only: the token stream is unchanged"*. That was never true of the
> five above, four of which said so in their own entries. The invariant holds for
> everything else on this page.

---

## Core

| Var | Default | Effect |
|---|---|---|
| `COLI_MODEL` | unset | model directory (serve mode; wins over the positional arg; the only source for `bench`). The directory may carry a `model_paths.json` — see below |
| `COLI_STREAM` | auto | `1`/`0` force/disable expert streaming (auto-decided from available RAM vs model size) |
| `COLI_RSS_GUARD_GB` | projected peak | RSS ceiling the runtime guard enforces — [note](#coli_rss_guard_gb) |
| `COLI_RAM_OVERCOMMIT` | off | skip the pre-load RAM check — [note](#coli_ram_overcommit) |
| `COLI_DIRECT_LOAD` | off | read the resident trunk through O_DIRECT at load (`peregrine-core/src/safetensors.rs`) |
| `COLI_DEBUG` | off | surface advisory (non-fatal) failures on stderr |
| `COLI_NO_ARENA_CAP` | off | skip the automatic `M_ARENA_MAX=2` malloc-arena cap (also skipped if `MALLOC_ARENA_MAX` is set) |

### `model_paths.json` — a model split across several drives

A model directory may carry a `model_paths.json`:

```json
{"paths": ["/srv/modelstripe/GLM-5.2-r3", "/srv/model600p/GLM-5.2-r3"]}
```

The loader then indexes `*.safetensors` from the primary directory **and** every
listed directory (relative entries resolve against the primary). This is how a
checkpoint split across drives is served without a symlink farm: the primary dir
holds `config.json`, the tokenizer, and this file; each drive holds its own
folder of shards. Pair it with `peregrine-reshard`, which groups each sparse
layer's experts by device so every layer reads from all drives at once.

Two rules keep it deterministic: shards sort by **file name** regardless of
which directory they live in (so the tensor order is placement-independent),
and a file name present in two directories is a hard error — the loader will
not guess which copy to serve. A listed directory that is missing or not
mounted fails the load immediately with the path named, rather than serving a
silently incomplete model.

### `COLI_RSS_GUARD_GB`

Every 16 decoded tokens the engine reads its **measured** resident set and, if it
is meaningfully over (2 % + 300 MB tolerance), lowers the warm-cache budget by the
overshoot — *lowering* it, not just evicting, so the cache cannot refill to the old
ceiling. It exists because the pre-load projection is an estimate: colibrì recorded
one at 74.4 GB against a real 115.6 GB and three kernel kills. Correctness-neutral
(an evicted slab re-reads from disk); `0` disables it.

### `COLI_RAM_OVERCOMMIT`

Load projects its peak footprint from the safetensors headers before allocating and
**refuses to start** when it cannot fit, rather than being OOM-killed part-way
through. Set this when you know your swap or cgroup limits better than
`MemAvailable` does. The refusal looks like this, and is worth reading rather than
overriding:

```
[ram] avail 27.4 GB | resident 10.9 + KV 0.7 + stream 21.1 + slack 1.4 -> peak 36.3 GB
Error: ... 6.8 GB short, so the kernel would OOM-kill this run part-way through loading
```

---

## I/O & streaming

| Var | Default | Effect |
|---|---|---|
| `COLI_IO_RINGS` | 4 | io_uring rings, each on its own thread — [note](#coli_io_rings) |
| `COLI_IO_BATCH` | 16 | **upper bound** on experts claimed per ring — [note](#coli_io_batch) |
| `COLI_IO_ENGINE` | `uring` | `uring` \| `pread` \| `regbuf` — [note](#coli_io_engine) |
| `COLI_IO_SPLIT_MB` | 0 (off) | split streamed regions larger than this into sub-reads of the same buffer, raising a ring's submit depth (~4 → ~10 at decode) without touching claim sizing; on LUKS each in-flight read is an independent dm-crypt unit. Byte-identical; unmeasured — the M3 A/B decides its default |
| `COLI_IO_THREADS` | workers | worker threads for `COLI_IO_ENGINE=pread`; 8 is colibrì's harness figure |
| `COLI_REGBUF` | off | alias for `COLI_IO_ENGINE=regbuf` — [note](#coli_regbuf) |
| `COLI_REGBUF_SLOTS` | 16 | registered buffers = queue depth for `regbuf` — [note](#coli_regbuf) |
| `COLI_IO_DEPTH` | 256 | ring depth for `COLI_MOE_ENGINE=sched` (`peregrine-serve/src/main.rs`) |
| `COLI_DIRECT` | off | O_DIRECT lane: DMA into aligned buffers, bypassing the page cache. **Measured −23 %** — see [note](#coli_io_engine) |
| `COLI_FORCE_ASYNC` | on | force `IOSQE_ASYNC` on buffered reads — [note](#coli_force_async) |
| `COLI_EXPERT_MERGE` | on | coalesce an expert's adjacent regions: two reads instead of six. `0` forces the six-region path; bit-identical either way |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before every main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded runs) |
| `COLI_IO_TUNE` | on | adaptive `set_iowq_max_workers` from the `IoTuner` EWMA |
| `COLI_IO_RECOVERY` | on | per-region retry ladder on batched-read failure (transient EIO/EAGAIN/EINTR) |
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |
| `COLI_MOE_ENGINE` | `concurrent` | `concurrent` (3-lane) or `sched` — [note](#coli_moe_engine) |

### `COLI_IO_RINGS`

**Streaming buffers scale with this.** 4 rings project 11.5 GB, 8 rings 21.1 GB —
and on a 46 GB box with a VM resident the pre-load RAM guard refuses 8 outright.
Ring count is capped by *memory*, not by the device. Raise it only with the
headroom to match. Measured:
[`bench-data/2026-08-09-decode-levers/`](../bench-data/2026-08-09-decode-levers/README.md).

### `COLI_IO_BATCH`

An **upper bound** on the claim, not the claim size. The lane ceil-divides the
layer's work across rings, because a fixed 16 was larger than a decode token's
entire 8-expert layer: ring 0 claimed all of it and the other rings broke out
without issuing a read, leaving **one ring doing four rings' work on every decode
token** (measured 24 % io duty across 4 rings). Fixing that took decode
**21.83 → 16.08 s/tok** and duty **24 % → 84 %**. Prefill is unaffected — its
~69-expert union still yields the full 16. Detail:
[the concurrent scheduler](concurrent-scheduler.md#the-three-lanes).

### `COLI_IO_ENGINE`

`uring` (batched io_uring submit, the historical path), `pread` (N threads of
blocking `pread`, bypassing io_uring), or `regbuf` (io_uring through pre-registered
fixed buffers). **Output is byte-identical across all three** — same regions, same
offsets, only the syscall shape changes.

`pread` exists to test the dm-crypt hypothesis, and **that test has run and does
not support it.** At the shipped 4 rings over 5 reps: `uring` **1.12 GB/s** vs
`pread` **1.06 GB/s** — a 5.7 % gap against a 5–11 % measured spread, i.e. below
the noise floor rather than merely close.

What the same run *does* show is that **O_DIRECT is the slow arm**: 0.86 vs
1.12 GB/s, −23 %, outside both spreads. That is why `COLI_DIRECT` defaults off; the
buffered arm keeps kernel readahead that O_DIRECT by definition discards.

The often-quoted *"0.84 GB/s against colibrì's 2.02"* was **a two-variable
comparison** — `pread` implies no O_DIRECT, so it pitted uring-*with* against
pread-*without*. Working:
[`M5-io-engine.md`](../bench-data/2026-08-09-prefetch-causes/M5-io-engine.md).

### `COLI_REGBUF`

**Now wired** — it was inert for a year while being documented and set in a
published benchmark arm. `COLI_REGBUF=1` is an alias for `COLI_IO_ENGINE=regbuf`,
kept so the historical spelling finally does what it said. `read_fixed` was depth-1
(one `submit_and_wait` per region, the same defect that crippled the O_DIRECT lane),
so `read_fixed_many` had to land first.

`COLI_REGBUF_SLOTS` is the engine's queue depth — one in-flight op per buffer.
These are **pinned** pages charged against `RLIMIT_MEMLOCK` (8 MB by default), so
16 slots at ~6 MB expert regions needs 96 MB and registration fails with `ENOMEM`
— which reads as "out of memory" but means "out of *lockable* memory". The engine
falls back to the plain submit with an advisory line. Raise `ulimit -l` first.

### `COLI_FORCE_ASYNC`

Default on because an inline cold read serialises the submitter. **Measured
2026-08-09 and it makes no resolvable difference here**: 1.13 vs 1.15 GB/s at 32
reads, 0.87 vs 0.85 at the engine's own 96-deep submit, against 14–27 % spread.
Leave it on. The premise is that a *fast* device flips the answer, and the drive it
was measured against delivers 1.1 GB/s — so this is evidence about the knob **here**,
not about the knob.

### `COLI_MOE_ENGINE`

`concurrent` (default, 3-lane) or `sched` (peregrine-sched's 2-lane `moe_streamed`).
`sched` is **slower by construction** — no GPU lane, no warm cache, no prefetch. It
exists as a runtime A/B against an independently written second implementation that
the oracle test already compares. `COLI_IO_DEPTH` sets its ring depth.

---

## Compute & scheduling

| Var | Default | Effect |
|---|---|---|
| `COLI_PAR_THREADS` | ncpus (≤ 16) | `peregrine-par` pool size; `1` = fully serial (the A/B baseline) |
| `COLI_LANE_BALANCE` | off | `LaneBalancer` overrides static residency: downgrade cold GPU residents to CPU when GPU is bottlenecked |
| `COLI_REPLICATE_K` | 0 | top-K hottest GPU-residents also warmed into the CPU warm cache each `reheat` |
| `COLI_NUMA_PIN` | off | pin workers round-robin across NUMA nodes; hierarchical pool dispatch; NUMA-bind ≥ 2 MB buffers |
| `COLI_SHAPE_SPECIALIZE` | off | per-shape probe-then-memoize serial-vs-parallel matmul dispatch |
| `COLI_PREFILL_CHUNK_DIV` | 0 | prefill chunk becomes `max(64, pos/d)` — [note](#coli_prefill_chunk_div) |

### `COLI_PREFILL_CHUNK_DIV`

A fixed chunk makes prefill quadratic in prompt length, since attention re-derives
every cached position per call. Chunk size cannot change output — only how long one
prefill step blocks decode.

---

## Warm cache

| Var | Default | Effect |
|---|---|---|
| `COLI_ECACHE_GB` | 10 % avail (cap 2 GiB) | warm expert RAM cache budget — [note](#coli_ecache_gb) |
| `COLI_MLOCK` | off | `mlockall(MCL_CURRENT)` over the resident trunk — [note](#coli_mlock) |
| `COLI_CACHE_SWEEP` | off | evict highest-layer first instead of LRU — [note](#coli_cache_sweep) |
| `COLI_CACHE_LFRU` | off | rank victims by `heat<<8\|recency` — [note](#coli_cache_lfru) |
| `COLI_CACHE_ADMIT_MIN_HEAT` | 0 | admit only at ≥ N routings — [note](#coli_cache_admit_min_heat) |
| `COLI_CACHE_COMPRESS` | off | zstd-compress warm-cache slabs on admit, decode on hit |
| `COLI_CACHE_COMPRESS_IDLE` | off | background-recompress cold slots while the engine is idle |
| `COLI_CACHE_NEGATIVE_TTL` | 0 | evict unhit warm-cache slots older than N clock ticks |

### `COLI_ECACHE_GB`

"Avail" is the *smaller* of `/proc/meminfo` `MemAvailable` and the cgroup v2/v1
memory limit — `/proc/meminfo` is not namespaced, so inside a container it
describes the host.

**More cache is not monotonically better.** Past the point where the cache competes
with the resident trunk, a hit becomes a page fault, and throughput can collapse
while the cache's own hit rate keeps climbing. The engine warns when the granted
cache exceeds half of available memory. If decode is slow, measure a *smaller*
value before a larger one.

Measured ceiling on the payoff: tripling 4.29 → 12.88 GB bought **3.8 % fewer disk
reads** and moved decode 356 → 353 s. The expert pool is ~363 GB and one token
routes 11.3 GB of it, so no cache that fits in RAM changes the shape of the problem
— see [Performance tuning](performance-tuning.md#things-that-look-like-levers-and-are-not).

### `COLI_MLOCK`

`MCL_CURRENT` after the resident weights load and before the warm cache fills, so
the trunk cannot be paged out under later memory pressure. `MCL_CURRENT` and **not**
`MCL_FUTURE` is the whole design: the cache is *supposed* to stay reclaimable. This
does not raise the memory ceiling, it removes variance — a machine that was going to
swap now fails honestly instead. Needs `RLIMIT_MEMLOCK` headroom (`ulimit -l
unlimited` or `CAP_IPC_LOCK`); a refusal is reported and the run continues unwired.

### `COLI_CACHE_SWEEP`

Evict the **highest layer** first instead of the least-recently-used slot, so the
cache converges on a stable low-layer band. Expert reads are a deterministic
front-to-back layer sweep with no intra-pass reuse, and the cache's `used` counter
advances in layer order — so at the end of a pass the least-recently-used slot is
*by construction* the earliest-layer expert, i.e. exactly what the next token asks
for first. Measured at 227 slots against a 600-expert working set, pure LRU returned
**0 hits in 9,944 lookups** with the cache 100 % full throughout.

**Only helps below the threshold.** Once the budget clears one token's working set
(`sparse_layers × topk × bytes_per_expert`, ~11.3 GB on GLM-5.2) the previous pass
fits entirely, there is no complement to retain, and plain LRU captures ~67 % of the
available reuse on its own.

### `COLI_CACHE_LFRU`

Frequency is the slot's own hit count, so it does not need a GPU heat table.
**Its recency term is miscalibrated here**: the 255-step window was sized for
`HeatTable`'s per-*forward* clock but is fed the warm cache's per-*access* clock
(~624 ticks/token), so anything older than ~31 layers of sweep scores `recent = 0`
and the policy is effectively frequency-only.

### `COLI_CACHE_ADMIT_MIN_HEAT`

**Requires a GPU tier**: the heat table is only built alongside one. Until
2026-08-09 a missing table meant "admit nothing", so setting this to 1 on a CPU-only
box silently turned demand-path admission off entirely while the prefetch lane —
which has no such gate — kept admitting everything. It now treats a missing table as
"gate does not apply".

---

## Prediction & prefetch

| Var | Default | Effect |
|---|---|---|
| `COLI_PREFETCH_LOOKAHEAD` | on | per-layer look-ahead emission depth |
| `COLI_PREFETCH_LANES` | **1** | background prefetch lanes — [note](#coli_prefetch_lanes) |
| `COLI_PREFETCH_WARM_PATHS` | `usize::MAX` | warm-tier breadth (experts actually streamed) — [note](#prefetch-breadth) |
| `COLI_PREFETCH_HINT_PATHS` | **0** | fadvise-hint-tier breadth (`WILLNEED` only) — [note](#prefetch-breadth) |
| `COLI_PREFETCH_WARM_PATHS_<CLASS>` / `COLI_PREFETCH_HINT_PATHS_<CLASS>` | unset | per-class override (`CODE`/`JSON`/`MATH`/`PROSE`/`MIXED`) |
| `COLI_PREFETCH_TUNE` | **off** | EWMA distance tuner — [note](#prefetch-breadth) |
| `COLI_PREFETCH_DIST` / `COLI_PREFETCH_DIST_MAX` | 4 / 16 | tuner start distance and clamp |
| `COLI_PREFETCH_PROTECT` | **auto** | protect the predicted set from eviction — [note](#coli_prefetch_protect) |
| `COLI_PREFETCH_MIN_READS` / `COLI_PREFETCH_MIN_YIELD` | 512 / 2 | self-limit — [note](#coli_prefetch_min_reads--coli_prefetch_min_yield) |
| `COLI_PREFETCH_VERIFY` | off | re-read + byte-compare each speculative load; accuracy log at shutdown |
| `COLI_ROUTER_LOOKAHEAD` | on | ask layer L+1's router early — [note](#coli_router_lookahead) |
| `COLI_ROUTER_LOOKAHEAD_N` | 6 | how many ranked experts the look-ahead streams — a *window*, not a tuning constant |
| `COLI_ROUTER_LOOKAHEAD_BATCH` | on | extend the look-ahead to batched decode (B > 1); `0` restores B == 1 only (`model.rs`) |
| `COLI_PREDICT_SOURCE` | unset | force `momentum` or `phase-aware` — [note](#coli_predict_source) |
| `COLI_ROUTE_HIST_DEPTH` | 4 | K-deep routing history feeding the predictor |
| `COLI_PHASE_THRESHOLD` | 0.6 | Jaccard distance declaring a phase shift — [note](#coli_phase_threshold) |
| `COLI_ENTROPY_ADAPT` | off | routing-entropy-adaptive breadth (needs `COLI_PREFETCH_TUNE`) |

> **Prefetch is worth keeping on.** Turning it off measured *slower* (23.8 vs
> 21.9 s/tok), and `disk_reads` is identical either way — `prefetch_reads` is a
> **subset** of `disk_reads`, not additional I/O.

### `COLI_PREFETCH_LANES`

Listed as "on" until 2026-08-08, which read as "this is enabled" — it is an
**integer**, and at the default of **1 the pool is a single lane**, so the
"parallel-async" part of the feature is opt-in and off unless you set it. Only the
**batched serving** path spreads across lanes (one lane per sequence, keyed on a
stable per-sequence id); single-stream decode, the per-layer emitter and the router
look-ahead all use lane 0 deliberately, because one stream's staggered emissions
must not overtake each other. Raising it costs a thread and a ring per lane.

### Prefetch breadth

`WARM_PATHS` is how many experts are actually streamed into the cache;
`HINT_PATHS` is `WILLNEED` only, no read, and skipped under O_DIRECT. Both were
listed as "on" until 2026-08-08; they are **counts**, and **the hint tier is off by
default**. A shutdown line reading `fadvise=0` is therefore the default working as
configured, not a broken counter.

`COLI_PREFETCH_TUNE` is opt-in, and this matters more than a default usually would:
**the tuner is the only thing that bounds prefetch breadth.** With it off,
`WARM_PATHS` defaults to `usize::MAX` and every candidate the predictor names is
streamed. Enabling it **overrides `WARM_PATHS` entirely** (breadth becomes the
tuner's `distance()`, clamped to `[1, DIST_MAX]`), so the two cannot be combined —
setting warm paths to 0 with the tuner on silently re-widens to 1. Its evidence base
is partial: it EWMAs `used`/`wasted`, and `wasted` only counts on eviction, so slabs
still resident are invisible to it.

### `COLI_PREFETCH_PROTECT`

**The default follows the cache budget**, because this mechanism has opposite signs
either side of one token's working set. Measured at 8 decode tokens: below the
threshold it is worth **+193 hits** (193 vs 0 — a layer sweep drives pure recency to
*exactly* zero); above it it **costs 40 %** of them (564 vs 945) plus 381 disk reads.
On when the budget cannot hold a pass, off when it can; `0`/`1` still force it. The
`[workingset]` line at load says which side you are on.

### `COLI_PREFETCH_MIN_READS` / `COLI_PREFETCH_MIN_YIELD`

After this many speculative reads, keep issuing only if at least this percent were
used. Measured at 8 tokens, 4,034 speculative reads bought **3 hits (0.3 %) for
+41 % wall time**, so an unconditional lane is a wall-time tax on a saturated disk.
One-way once tripped; `_MIN_YIELD=0` disables the guard.

### `COLI_ROUTER_LOOKAHEAD`

At the end of layer `L`, run layer `L+1`'s own post-attention norm and router
against layer `L`'s output, and prefetch the experts that ranking names. A different
*kind* of predictor from everything else in this section: those are statistics over
the router's past answers, **this asks the router**. Correctness-neutral — the
authoritative router still runs at `L+1` and still decides, so this only changes
*when* bytes are read (`router_lookahead_cannot_move_a_token`).

`COLI_ROUTER_LOOKAHEAD_N` is a window, not a tuning constant — the right value is
however many reads fit in the layer boundary, a property of your disk and model.
Widening past the boundary buys steadily worse guesses that displace reads the
engine actually needs.

### `COLI_PREDICT_SOURCE`

Load picks the strongest source the artifacts support, so this exists to select a
*weaker* arm for A/B — `COLI_PREDICT_EVAL` scores the arms, and without this it
could only grade a choice nobody could change. Names needing artifacts (`automaton`,
`macro`) are ignored rather than silently degraded.

### `COLI_PHASE_THRESHOLD`

A fraction in `[0,1]` above which a routing **phase shift** is declared, making
`PredictSource::PhaseAware` fold a dominating vote onto the newest frame. **This
governed nothing until 2026-08-08**: its only reader was `PhaseTracker`, which has no
production caller, while the live predictor used a hardcoded `6000` bp. It is now
converted to basis points and read by the predictor, so the documented default is
finally the one in force. Only takes effect with `COLI_PREDICT_SOURCE=phase-aware`.

---

## Serving: batching, memo, KV

| Var | Default | Effect |
|---|---|---|
| `COLI_BATCH_SLA_MS` | unset | shrink the working batch cap on latency overrun; regrow on slack |
| `COLI_ADAPTIVE_WINDOW` | 1 | run prefill every Nth engine tick (decode-heavy window) |
| `COLI_FUSE_PREFILL` | off | prefill chunk rides the decode batch's forward — [note](#coli_fuse_prefill) |
| `COLI_MEMO_ENTRIES` / `COLI_MEMO_MB` | 32 / 64 | exact response memo — [note](#coli_memo_entries--coli_memo_mb) |
| `COLI_KV_BUDGET_MB` | 0 | resident-KV byte ceiling for admission — [note](#coli_kv_budget_mb) |
| `COLI_KV_POOL_MB` | 0 | recycle a retired sequence's KV allocations — [note](#coli_kv_pool_mb) |
| `COLI_PREFIX_CACHE_MB` | 0 | cross-request KV prefix cache; matched by comparing tokens, not hashing |
| `COLI_DRAFT` | 0 | MTP speculative-decode depth — [note](#coli_draft) |
| `COLI_DRAFT_SAMPLED` | off | extend speculation to temperature > 0 — [note](#coli_draft_sampled) |
| `X-Peregrine-Priority` | (HTTP header) | `high`/`1`/`true` → drained ahead of normal-priority requests |

### `COLI_FUSE_PREFILL`

Without it a mixed tick runs **two disjoint forwards**, each streaming its own
routed-expert union off disk — ~11.3 GB per token apiece at GLM-5.2 shapes — for
work the MoE lane would do in one pass, since it unions routed experts across *rows*
and does not care which sequence a row came from.

**Output-neutral, and asserted rather than argued**:
`a_fused_chunk_is_indistinguishable_from_two_separate_forwards` requires bit-identical
logits, and `fused_prefill_emits_the_same_tokens_as_the_two_forward_tick` requires an
identical token stream. Confirmed empirically 2026-08-09 — a fused stream's completion
was byte-identical to the unfused baseline.

**Measured: 322 s vs 356 s on two concurrent streams (1.11×).** That understates it:
the test used `max_tokens=4`, which is ~68 % prefill, and union sharing is a *decode*
effect. Fusion only engages when there is something to fuse with (a live decode batch
and a queued prefill); otherwise the tick takes the historical path.

### `COLI_MEMO_ENTRIES` / `COLI_MEMO_MB`

Serves a byte-identical repeat request from a prior certified completion **without
entering the model** — worth more here than on most servers, since one token costs a
pass over gigabytes of streamed experts. Bounded by both entry count and bytes;
either at `0` disables it.

Three rules keep it safe: the key is the *complete* request semantics (prompt token
ids, `max_tokens`, `top_p`, model id — compared field-by-field, **never hashed**, so
a collision cannot serve one caller another's answer); only `temperature == 0`
requests are eligible, because replaying a stored sample would silently turn a
sampling endpoint into a deterministic one; and a hit is answered before the engine
is touched, so it can never become a KV boundary. Entries hold token ids, not wire
bytes, so a streaming call can be served from a non-streaming entry. Counters on
`/health`.

### `COLI_KV_BUDGET_MB`

Applied alongside `--max-batch`. Admission is otherwise capped by a *count* and never
reads bytes, so default flags admit a ~53 GB worst case at GLM-5.2 shapes — and no KV
saving can raise concurrency while concurrency is a count. A high-water gate: it stops
admitting once resident KV crosses the budget, so the peak overshoots by at most one
sequence. Counts each sequence's private tail plus **one** charge per distinct shared
prefix however many sequences view it — charging a refcounted system prompt once per
request would refuse admissions over RAM that was never allocated.

### `COLI_KV_POOL_MB`

Recycle a retired sequence's KV latent allocations into the next admission instead of
returning them to the allocator. **Additive to `COLI_KV_BUDGET_MB`** — total KV RSS ≈
budget + pool — because charging admission for memory that is free and about to be
reused would refuse sequences over nothing. Output-neutral: recycled buffers are
cleared and readers only see `[0, len)`.

### `COLI_DRAFT`

Draft depth for **both** the stdio server (equivalent to `--draft N`) and the batched
HTTP engine. In the batched engine each sequence's `1 + γ` rows ride the *same*
forward as every other sequence's, so B sequences speculating share one routed-expert
union — speculation only pays on a disk-bound engine if the verify is shared.

**Greedy requests only by default**: accepting on argmax makes it sequence-identical
to greedy decoding, while at temperature > 0 it is merely distribution-preserving, so
a sampled request drafts nothing and takes the one-row path in the same batch — unless
`COLI_DRAFT_SAMPLED` is set. Speculated rows record into a *scratch* routing history,
never the sequence's own: a rejected draft must not warm experts for a token that
never existed. Needs a checkpoint with an MTP head; the engine refuses loudly without
one.

> **⚠ Measured as a regression on streaming GLM-5.2 (2026-08-09).** This entry used to
> advise "use 4–6, the net-loss figures were taken at depth 2". At depth 4:
> **p50 24.3 s/token against 15.5 s without it — 1.57× slower.** Speculation does
> accept (8 tokens from 6 forwards), but each forward verifies γ+1 rows and its routed
> union grows faster than acceptance repays. Output is sequence-identical, as
> documented; it is simply slower. The advice may still hold where experts are
> **resident** rather than streamed.
> [`bench-data/2026-08-09-decode-levers/`](../bench-data/2026-08-09-decode-levers/README.md)

### `COLI_DRAFT_SAMPLED`

Leviathan rejection sampling (`speculative_sample`): accept a draft with probability
`min(1, p/q)`, else resample the residual `(p−q)+`. The emitted **distribution** is
exactly the request's own — asserted over 40,000 draws — but the emitted **sequence**
is not what the same seed produces unspeculated, because rejection sampling draws two
uniforms per draft where plain decode draws one per token. A caller pinning outputs by
seed would see different text with no error anywhere, which is why this is opt-in
rather than the obvious win it looks like. Needs `COLI_DRAFT` too.

---

## Knobs that change token values

Every knob in this section is a **quality trade**. Gate each with
`Model::prediction_flip_rate` against the unmodified configuration.

### `COLI_KV_DTYPE`

`f32` (historical, output-neutral) or `f16`. f16 halves resident KV exactly —
175.5 KiB/token becomes 87.8 at GLM-5.2 shapes — which under `COLI_KV_BUDGET_MB`
converts straight into batch slots. Worth doing before any cleverer scheme: every
published KV-quantization result is measured against an fp16 baseline, so at f32 the
engine starts a full 2× behind the number it would be compared to.

**Pair it with `COLI_MLA_ABSORB`.** Absorb dots the stored latent in f32 and its
error stays at f16's own precision (measured **1.8e-4**), while the dense path pushes
the latent back through `kv_b.apply_vec`, whose per-row int8 activation scale
(`amax / 127`) can be moved by the perturbation and rescale the whole grid — measured
**1.7e-2**, two orders of magnitude worse, and from int8 activations rather than from
f16. An unrecognised value is reported and treated as `f32` rather than silently
guessed.

### `COLI_ROUTE_MIN_SHARE`

Drop trailing routed experts carrying less than this share of a position's gate mass,
renormalizing the survivors. It removes a real (if small) term from the MoE sum. Size
it with `COLI_GATE_STATS`. `0` disables it.

### `COLI_DSA`

Run the DSA lightning indexer on layers whose checkpoint carries one (`--indexer` at
conversion). Each query scores every cached key and attends only the top `index_topk`
— the largest workload reduction available on long context, because attention stops
growing with the cache.

**Inert two ways**: with no indexer tensors there is nothing to run, and at or below
`index_topk` cached positions the selection is the identity, so the scoring pass is
skipped and the output is bit-identical (the C engine's activation rule). Above that
it changes token values. Indexer keys are cached from position 0 whenever this is on,
because a later selection needs them. **Single-sequence path only**: selection is
implemented against the dense attention core, and the batched decode engine runs the
*absorb* core, which has no sparse form — so a batched server sees this only during
prefill.

### `COLI_MLA_ABSORB`

Run MLA attention through weight absorption instead of dense reconstruction. Works in
the 512-wide latent space rather than rebuilding `[k_nope|v]` for every cached
position on every step, so its cost stops growing with context. The two agree
algebraically but not numerically, because dense pushes the cached latent back through
the quantized `kv_b`. **Unvalidated on a real checkpoint.**

**This reaches the batched decode path too, as of 2026-08-03 — it did not before.**
`forward_layer_batched` called an absorb-only core unconditionally, so a served
request ran its prefill dense and every decode token absorbed: two numerically
different implementations inside one response, whatever this knob said. The dense core
now takes per-row cache owners, so batched decode at the default is bit-identical to
single-sequence decode.

**For serving, `1` is worth considering**: dense reconstructs `[k_nope|v]` once per
*cache*, and in a decode batch every sequence has its own, so nothing is shared and
the cost grows with context — which is the problem absorb exists to solve. That is now
your decision against a documented default rather than one the code took silently.

---

## GPU (feature = `cuda`)

Only knobs in this section require a CUDA build. Everything else on this page works
on a CPU-only binary.

| Var | Default | Effect |
|---|---|---|
| `COLI_GPU` | off | enable the GPU lane. **Also the only way to populate routing `heat`** — see [Tools](tools.md#heat-tiering-and-its-prerequisite) |
| `COLI_GPU_INT4` | off | int4 VRAM expert tier (needs per-row int4 experts) |
| `COLI_GPU_F32_FRAC` | unset | adaptive per-expert precision: hottest fraction of residents promoted to f32 |
| `COLI_GPU_TIER_SWAP` | `replan` | VRAM residency policy for `reheat` — [note](#coli_gpu_tier_swap) |
| `COLI_PCIE_BUDGET_MB` | unlimited | cap on bytes one `reheat` generation may upload; the coldest deferred to the next |
| `COLI_CUDA_ASYNC` | **on** | async H2D/kernel/D2H; `=0` forces synchronous (`cuda/backend_cuda.cu`) |
| `COLI_CUDA_TC_INT4` | off | int4 Tensor Core arm; one legal WMMA shape, 8×8×32 — [note](#kernel-arm-selection) |
| `COLI_CUDA_TC_MIN_ROWS` | 8 | every group must have ≥ this many rows or the int4 TC arm is skipped |
| `COLI_CUDA_TC_W4A16` | off | fp16 Tensor Core arm — **the only tile-sensitive one**. Needs all-int4 experts and compute capability ≥ 7.0 |
| `COLI_CUDA_TC_W4A16_MIN` | 16 | minimum rows before the W4A16 arm is taken |
| `COLI_CUDA_W4_PACKED` | **on** | packed-W4 arm when every expert is int4; `=0` falls through to the generic kernel |
| `COLI_CUDA_DUAL_PROJ` | **on** | in the packed-W4 arm, compute gate and up in one fused kernel; `=0` runs two passes |
| `COLI_CUDA_GRAPH` | off | capture/replay `expert_group` launches — [note](#coli_cuda_graph) |
| `COLI_CUDA_AUTOTUNE` | off | online WMMA tile selection — [note](#coli_cuda_autotune) |
| `COLI_CUDA_FUSED_REDUCE` | off | device-side gate-weighted reduce — [note](#coli_cuda_fused_reduce) |
| `COLI_CUDA_PROFILE` | off | per-call H2D/kernel/D2H timings. **Disables `COLI_CUDA_GRAPH`** for the calls it measures — the event records are not part of the work a replay repeats, so a graph captured with them would time itself |

### Kernel arm selection

`select_arm` in `cuda/backend_cuda.cu` picks one of four arms in this order, and the
first match wins:

1. **`ARM_TC_INT4`** — needs `COLI_CUDA_TC_INT4=1`, all-int4 experts, `D` and `I` both
   divisible by 32, and *every* group at ≥ `COLI_CUDA_TC_MIN_ROWS` rows.
2. **`ARM_W4A16`** — needs `COLI_CUDA_TC_W4A16=1`, all-int4 experts and compute
   capability ≥ 7.0.
3. **`ARM_W4_PACKED`** — the default for all-int4 experts; disable with
   `COLI_CUDA_W4_PACKED=0`.
4. **`ARM_GENERIC`** — the fallback.

The arm is also the first component of the CUDA-graph key: two calls of the same shape
on different arms are different launch sequences. This is why a knob that "does
nothing" may simply have failed a precondition — the backend reports which arm ran.

> **Three `COLI_CUDA_*` names are compile-time macros, not environment variables**:
> `COLI_W4A16_TILES` (emits the three template instantiations),
> `COLI_CUDA_GRAPH_CACHE` (16) and `COLI_CUDA_MAX_DEVICES` (16). Setting them in the
> environment does nothing.

### `COLI_GPU_TIER_SWAP`

`replan` (default, historical) re-ranks every candidate each generation, so any expert
whose heat rank moved is an upload — unbounded churn that `COLI_PCIE_BUDGET_MB` exists
to truncate. `lfru` and `freq` instead move **at most one expert per layer per
generation** using `peregrine-io`'s hot-store rules, which is the shape those rules
were written for (per layer, the resident set *is* a fixed slot array). Both carry a
25 %-plus-4-count hysteresis, so a generation where nothing meaningfully changed
uploads nothing at all. `lfru` adds a recency term worth at most 255 points against one
routing count's 256 — narrow by design, deciding only cases sitting exactly on `freq`'s
threshold. The re-plan stays the default because it is the only one that can *resize*
the resident set. Ignored (with an advisory) while `COLI_GPU_F32_FRAC` is set.

### `COLI_CUDA_GRAPH`

**Bit-identical by construction** — same kernels, same arguments, same order — so this
is a pure wall-clock knob. Off by default anyway, because its failure mode is silent:
the scratch buffers are grow-only and *free before they reallocate*, so a graph
captured before a larger call holds dangling device pointers. A generation counter
invalidates those, and `GET /metrics` reports
`graph_captures`/`graph_replays`/`graph_invalidations`/`graph_uncacheable` so "the knob
is on and replaying nothing" is visible rather than merely slow. The
`COLI_CUDA_TC_W4A16` arm is never cached — it passes device weight pointers as kernel
arguments, so a replay would compute against the previous residency generation.

### `COLI_CUDA_AUTOTUNE`

Online WMMA tile selection for the `COLI_CUDA_TC_W4A16` arm, persisted to
`<dir>/kernel_tuning.json`. A **second** opt-in on top of that arm, because the tile
reaches only it — the backend reports which arm actually ran, so a group that missed
the arm's row-count floor is not credited to the tile the tuner picked. Explores all
three legal fp16 fragment shapes (16×16×16, 32×8×16, 8×32×16) before exploiting one.
Whether the three are bit-identical is *expected* (same `K`, same k-loop) but
**unverified on hardware**.

### `COLI_CUDA_FUSED_REDUCE`

Fuse the layer-level gate-weighted accumulation of GPU-resident experts onto the
device: the D2H carries `s_n` rows instead of `Σrows`, ~5× fewer at B=16 on the
measured GLM-5.2 unions and exactly 1× at B=1. **Changes the GPU arm's low bits** —
GPU experts now sum among themselves before meeting the CPU lane's contributions
instead of interleaving in batch-union order, and `f32 +=` is not associative. It stays
*stable* run to run (the device reduce is CSR-ordered, no atomics); it simply is not the
host reduce's sum.

---

## Governors & learning

| Var | Default | Effect |
|---|---|---|
| `COLI_THERMAL_LIMIT_C` | unset | shrink CPU workers above this package temperature; regrow 8 °C below |
| `COLI_POWER_CAP_W` | unset | shrink workers when RAPL watts exceed the cap; regrow below 80 % |
| `COLI_BW_GOVERNOR` | off | shrink workers on a CPU-lane GB/s plateau; periodic regrow probe |
| `COLI_LEARN_SCHED` | off | ε-greedy bandit over knob configurations; policy persisted in `route_stats.json` |
| `COLI_RL_SCHED` | off | tabular Q-learning scheduler (the bandit wins if both are set) |

## Scheduling affinity & offline artifacts

| Var | Default | Effect |
|---|---|---|
| `COLI_FUSE_THRESHOLD` | 0.9 | co-firing rate above which expert pairs stay adjacent in dispatch |
| `COLI_HYPER_SCHED` | off | group co-activation components into one io-claim window |
| `COLI_TIER_VRAM_MB` / `COLI_TIER_RAM_MB` | unset | `galactic`: tier byte budgets → emit `tiers.json` |
| `COLI_TIER_SEED` | on | prefetch-warm the planned RAM tier at model load |
| `COLI_LAYOUT_SCHEDULE` | on | consume `schedule.json` to sort disk reads — [note](#coli_layout_schedule) |
| `COLI_ROUTE_STATS_PERSIST` | on | save `route_stats.json` at Drop; auto-load a matching one at `Model::load` |

### `COLI_LAYOUT_SCHEDULE`

**Now a fallback**: when the expert map resolved (any streaming load), submit order
comes from the real `(fd, offset)` instead. The schedule's rank is a routing-community
order that only matches disk order after a `--apply` rewrite, which is
single-shard-only and so cannot run on a sharded container. See
[Layout tools](layout-tools.md).

## Diagnostics

Nothing here is on the forward path; all of it prints at shutdown or on `/metrics`.

| Var | Default | Effect |
|---|---|---|
| `COLI_PERF_COUNTERS` | off | LLC-miss counter on the decode thread — [note](#coli_perf_counters) |
| `COLI_PERF_PREFETCH_FEEDBACK` | off | feed per-forward LLC-miss deltas to the prefetch-distance tuner (rising misses widen). Requires `COLI_PERF_COUNTERS=1`. A **second** opt-in on purpose: the counter is a measurement, this is a control loop driven by it, and the direction is a hypothesis rather than a measured result |
| `COLI_GATE_STATS` | off | per routed expert, whether its gate share is below 0.5/1/2/5 % of its position's mass; printed as `[gate]` |
| `COLI_UNION_STATS` | off | batch-union sharing — [note](#coli_union_stats) |
| `COLI_PREDICT_EVAL` | off | predictor scoreboard — [note](#coli_predict_eval) |
| `COLI_PREDICT_EVAL_N` | `topk` | candidates per arm the scoreboard scores, so recall is comparable with a real routing decision's width |

### `COLI_PERF_COUNTERS`

**Now wired** — it was documented as live for a year while `open_l3_miss_counter` had
no caller. Prints `[perf] llc-misses=N` at shutdown. **Scoped to one thread on
purpose**: `perf_event_open` follows the calling thread, so this counts attention and
the deterministic reduce, *not* the io_uring workers or the `peregrine-par` pool. A
whole-process figure needs a counter per thread; reporting this one as if it were that
is how a number stops meaning anything. Silent when the kernel refuses — paranoid
level, seccomp, or no PMU, which is most VMs.

### `COLI_UNION_STATS`

Prints `[union] selections=… distinct=… share=N.NNNx`, plus `[union] all-low-gate
reads=…/…` — the fraction of expert reads whose *every* routing row was under 1 % of
its position's gate mass, which is the ceiling on gate-mass mixed-precision loading (a
read is issued per union entry, not per row, so an expert one row leans on must be read
at full width whatever the others wanted). `share` is how many routed selections each
distinct expert read actually served — the amortization batching is supposed to buy.
`benchmarks.md` credits the 4.4× at B=16 "entirely" to this while a union model over
GLM-5.2's 256-expert top-8 layers predicts only ~1.26×; this reads it off the live
engine.

### `COLI_PREDICT_EVAL`

Scores the router look-ahead, the configured `PredictSource` and a previous-token
baseline against the routing that actually happened, and prints recall +
precision-by-rank as `[predict-eval]` at shutdown. Decode only. This is how you find
out whether the prediction machinery is earning its complexity on *your* container
rather than on someone else's.

## Bench & misc

| Var | Default | Effect |
|---|---|---|
| `COLI_BENCH_STEPS` | 3 | decode steps per batch size in `peregrine bench` |
| `PEREGRINE_API_KEY` | unset | bearer-auth key for `peregrine-serve` (same as `--api-key`) |

---

Deep dives on what the knobs actually steer:
[Performance tuning](performance-tuning.md) ·
[Measurement discipline](measurement.md) ·
[Adaptive runtime](adaptive-runtime.md) ·
[Prefetch & caching](prefetch-and-caching.md) ·
[I/O & storage](io-and-storage.md) · [GPU](gpu-cuda.md) ·
[Serving](serving.md) · [Tools](tools.md).

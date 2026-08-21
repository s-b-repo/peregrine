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

**Six knobs change token values.** They are real quality/performance trades, not
tuning, and each must be gated with `Model::prediction_flip_rate` against the
unmodified configuration before you rely on it:

| Knob | What moves |
|---|---|
| [`COLI_KV_DTYPE=f16`](#coli_kv_dtype) | stored KV latents lose precision |
| [`COLI_ROUTE_MIN_SHARE`](#coli_route_min_share) | drops low-gate experts from the MoE sum |
| [`COLI_DSA`](#coli_dsa) | attends only the indexer's top-k cached positions |
| [`COLI_MLA_ABSORB`](#coli_mla_absorb) | absorb and dense agree algebraically, not numerically |
| [`COLI_RLM`](#coli_rlm--recursive-refinement-at-contested-decode-positions) | recursive refinement of contested decode positions, may shift the argmax |
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
| `COLI_IO_ENGINE` | auto | `uring` \| `pread` \| `regbuf`; unset probes io_uring and falls back to `pread` — [note](#coli_io_engine) |
| `COLI_IO_COMPLETION` | on | forward each expert as its own reads complete (uring only); `0` restores the blocking whole-wave submit — [note](#coli_io_completion) |
| `COLI_SQPOLL` | off | kernel submission-polling thread on the streaming rings: no enter syscall per submit — [note](#coli_sqpoll) |
| `COLI_SQPOLL_IDLE_MS` | 2000 | how long the SQPOLL kthread busy-polls before sleeping (the next submit wakes it transparently) |
| `COLI_SQPOLL_CPU` | unset | pin the SQPOLL kthread to this CPU |
| `COLI_IO_SPLIT_MB` | 0 (off) | split streamed regions larger than this into sub-reads of the same buffer, raising a ring's submit depth (~4 → ~10 at decode) without touching claim sizing; on LUKS each in-flight read is an independent dm-crypt unit. Byte-identical; unmeasured — the M3 A/B decides its default |
| `COLI_IO_THREADS` | workers | worker threads for `COLI_IO_ENGINE=pread`; 8 is colibrì's harness figure |
| `COLI_REGBUF` | off | alias for `COLI_IO_ENGINE=regbuf` — [note](#coli_regbuf) |
| `COLI_REGBUF_SLOTS` | 16 | registered buffers = queue depth for `regbuf` — [note](#coli_regbuf) |
| `COLI_IO_DEPTH` | 256 | ring depth for `COLI_MOE_ENGINE=sched` (`peregrine-serve/src/main.rs`) |
| `COLI_DIRECT` | off | O_DIRECT lane: DMA into aligned buffers, bypassing the page cache. **Measured −23 %** — see [note](#coli_io_engine) |
| `COLI_FORCE_ASYNC` | on (off under SQPOLL) | force `IOSQE_ASYNC` on buffered reads — [note](#coli_force_async) |
| `COLI_EXPERT_MERGE` | on | coalesce an expert's adjacent regions: two reads instead of six. `0` forces the six-region path; bit-identical either way |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before every main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded runs) |
| `COLI_IO_TUNE` | on | adaptive `set_iowq_max_workers` from the `IoTuner` EWMA |
| `COLI_IO_RECOVERY` | on | per-region retry ladder on batched-read failure (transient EIO/EAGAIN/EINTR) |
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |
| `COLI_MOE_ENGINE` | `concurrent` | `concurrent` (3-lane) or `sched` — [note](#coli_moe_engine) |
| `COLI_IO_DEVICE_SCHED` | off | device-aware ring scheduling: claims are grouped per physical device instead of taken off one device-blind cursor, with cross-device work stealing. Built only when it can differ — streaming, >1 ring, shards genuinely on >1 device |
| `COLI_IO_DEVICE_MAP` | probed | override the shard→device-ordinal mapping the above schedules on, for a topology the prober reads wrongly. Read once per model open, not latched |

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

**Unset is not the same as `uring`.** Left unset, the engine probes io_uring once
and resolves to `pread` if it cannot create a ring — an older kernel,
`kernel.io_uring_disabled=2`, or a container whose seccomp profile blocks
`io_uring_setup` (Docker's default profile does). The probe prints one line when
it fails, and streaming then runs with **zero rings** rather than refusing to
load, which is what it used to do. Setting `COLI_IO_ENGINE=uring` *explicitly*
keeps the old strict behaviour: a benchmark arm that asked for io_uring should
fail loudly rather than quietly become a `pread` arm and publish its number under
the wrong name.

One consequence worth knowing: **`pread` cannot do O_DIRECT.** O_DIRECT needs
block-aligned buffers and only the ring path has them (`read_direct_aligned`),
so `COLI_DIRECT=1` on a ringless host is reported as off rather than failing
every read with `EINVAL`.

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
not about the knob. Under `COLI_SQPOLL=1` the default flips to **off**: the poll
kthread already issues ops nonblocking and hands cold reads to io-wq itself, so
`IOSQE_ASYNC` would only force a pointless io-wq bounce on page-cache-warm reads.
An explicit `COLI_FORCE_ASYNC=1`/`0` still wins either way.

### `COLI_IO_COMPLETION`

Default on. The I/O lane submits every claimed expert's regions with
Reactor-owned landing buffers and forwards each expert to the CPU pool the
moment its **last region** lands, instead of waiting for the whole claim's
wave — compute on expert 1 overlaps the reads of experts 2..N, and warm-cache
admission (including the zstd encode) moves off the I/O lane onto the worker.
Byte-identical to the wave by construction: same regions, same carve, and the
reduce keys on `pos`, never arrival order. `0` restores the blocking wave
path. `COLI_IO_ENGINE=pread|regbuf` also implies the wave — those measurement
arms are wave-shaped on purpose, and reshaping their requests would change
what they measure.

### `COLI_SQPOLL`

Default off. Opt-in `IORING_SETUP_SQPOLL` on the **streaming rings only**
(prefetch/loader rings keep the plain setup): a kernel thread polls the
submission queue, so a submit is a shared-memory write with **no syscall** —
on top of the completion lane this removes the last per-wave enter. The
kthread busy-polls a full core until `COLI_SQPOLL_IDLE_MS` (default 2000) of
quiet, which is usually the wrong trade on a CPU-contended box — hence
default off; the `sqpoll-on`/`sqpoll-off` bench arms decide. Unprivileged
since kernel 5.13; on any setup failure the reactor falls back to the plain
`COOP_TASKRUN` ring with an advisory note. `COLI_SQPOLL_CPU` pins the
kthread. Under SQPOLL, `IOSQE_ASYNC` defaults off (the kthread already issues
ops nonblocking and punts cold reads to io-wq itself) and the io-wq tuner's
`sq_full` trigger goes quiet (the kthread drains the SQ continuously) — its
read-µs EWMA remains live.

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
| `COLI_PREFILL_CHUNK_DIV` | **4** (2026-08-13) | prefill chunk becomes `max(64, pos/d)`; `0` = fixed 64 — [note](#coli_prefill_chunk_div) |

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
| `COLI_ECACHE_AUTO_FRAC` | 0.80 | fraction of post-load `MemAvailable` that `COLI_ECACHE_GB=auto` claims, still capped by the transient reserve + 1 GiB safety |

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
| `COLI_PREFETCH_STALE_DROP` | off | the prefetch worker drops a queued speculative warm **before its disk read** once the forward sweep has moved past its emit stamp. Motivated by measurement: at B=16, 98.6 % of speculative reads (40 352/41 159) arrived too late to be anything but waste. Advisory lane only, so output is untouched by construction |
| `COLI_PREFETCH_STALE_SLACK` | 1 | how many layer-steps past the emit stamp a queued warm survives before the drop above takes it. `[prefetch] stale_dropped=/used=` says whether it is cutting waste or fresh work |

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
| `COLI_FUSE_PREFILL` | **on** (2026-08-13) | prefill chunk rides the decode batch's forward; `=0` restores two forwards — [note](#coli_fuse_prefill) |
| `COLI_MEMO_ENTRIES` / `COLI_MEMO_MB` | 32 / 64 | exact response memo — [note](#coli_memo_entries--coli_memo_mb) |
| `COLI_KV_BUDGET_MB` | 0 | resident-KV byte ceiling for admission — [note](#coli_kv_budget_mb) |
| `COLI_KV_POOL_MB` | 0 | recycle a retired sequence's KV allocations — [note](#coli_kv_pool_mb) |
| `COLI_PREFIX_CACHE_MB` | **2048** (2026-08-13) | cross-request KV prefix cache (`0` disables); caches prompts *and* generated tokens, matched by comparing tokens, not hashing |
| `COLI_MAX_BATCH_ROWS` | 0 | ceiling on rows in one fused forward (chunk yields first; bounds draft depth); `0` = uncapped |
| `COLI_QUEUE_DEPTH` | 0 | admission backlog cap; a submit at the cap gets an OpenAI-shaped 503 instead of queueing forever; `0` = unbounded |
| `COLI_TOK_MERGE_SIMD` | auto | tokenizer short-merge tier A/B (`auto`\|`scalar`\|`avx2`\|`avx512`); auto = measured scalar default — see [tokenizer.md](tokenizer.md) |
| `COLI_DRAFT` | 0 | MTP speculative-decode depth — [note](#coli_draft) |
| `COLI_DRAFT_SAMPLED` | off | extend speculation to temperature > 0 — [note](#coli_draft_sampled) |
| `COLI_SPEC_GDN` | off | allow speculation on a **recurrent** (Qwen3.5-hybrid) arch — [note](#coli_spec_gdn) |
| `COLI_SPEC_GDN_MAX_B` | 0 | batch width above which `COLI_SPEC_GDN` stops drafting; `0` = uncapped |
| `COLI_DRAFT_NGRAM` | 0 | prompt-lookup drafting: match suffixes up to this length — [note](#coli_draft_ngram) |
| `COLI_SPEC_UNION_MAX` | 0 | ceiling on a tick's projected routed-expert union, in expert-read requests — [note](#coli_spec_union_max) |
| `COLI_DRAFT_TREE` | off | verify both draft sources as a token tree instead of choosing one — [note](#coli_draft_tree) |
| `COLI_MTP_HEAT` | off | let the MTP head's experts accumulate residency heat — [note](#coli_mtp_heat) |
| `X-Peregrine-Priority` | (HTTP header) | `high`/`1`/`true` → drained ahead of normal-priority requests |
| `COLI_KV_STORE_DIR` | unset | disk-persisted KV sessions: completed prefixes ≥256 tokens checkpoint here (fingerprint + checksum + full-token compare) and a restarted server restores them instead of re-prefilling. The in-memory prefix cache's disk extension |
| `COLI_KV_STORE_MB` | unset | byte cap on that store; the LRU trims to fit |
| `COLI_KV_STORE_TRIM` | unset | how much the store trims past the cap when it evicts, so eviction is not one entry per admission |
| `COLI_KV_STORE_SYNC` | off | serialize + fsync a checkpoint **on the engine thread** instead of the background writer. The control arm for the async-writer latency A/B, not a production setting: a synchronous checkpoint makes every other live stream's next token wait behind it |
| `COLI_TOPIC_ROUTING` | off | per-`TokenClass` residency steering: cache tiebreaks prefer experts this topic has routed before. **Accumulated nothing under `peregrine-serve` until 2026-08-20** — the batched forward never fed it, and the shutdown path wrote the empty profile out anyway |
| `COLI_TOPIC_HALFLIFE` | 512 | decay half-life for those topic profiles, scaled by the routing-entropy EWMA, so a profile tracks recent routing and re-forms on a topic shift instead of anchoring to all-time counts |
| `COLI_QWEN_THINK` | off | keep Qwen's `<think>` block in the response. Off pre-closes the block in the assistant turn (the shipped template's `enable_thinking=false` form) — with it open, any run whose token budget expired before the closing tag rendered as an **empty completion** |

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



### `COLI_GPU_DENSE`

Place a **dense** model's MLP weights in VRAM and compute them on the device.
Refused when the checkpoint carries routed experts — this is the dense-model
counterpart of the expert tier, not a replacement for it.

**It changes token values, and not by a rounding error.** The device path is far
more accurate than the CPU one: measured **rms 1.1e-7 against 3.0e-3** at decode,
because the CPU path quantizes activations to int8 and the device path stays in
f32. So *which* layers are resident changes the tokens produced — the GPU arm is
the better one, but it is a different one.

That is why `COLI_GPU_DENSE_LAYERS` exists. Left unset, the tier takes whatever
fits in free VRAM, which depends on whatever else holds the card — so two boots
of the same container can produce different output. Fine for serving, where the
better path wins and the boot log says what ran; **not** fine for a measurement
arm, where the comparison has to be repeatable. Pin the count for any gate.

Gated on the *value*, not on the variable being present: `COLI_GPU_DENSE=0`
means off. It read `is_ok()` once, so `=0` switched it **on** — found while
trying to isolate a regression by turning it off, which is exactly the moment
that bug costs the most.

### `COLI_ACT_F32`

Compute quantized matmuls against **f32** activations instead of the int8 ones
`qrow_i8` produces. A quality/throughput trade in the opposite direction from
the precision ladder: same weights, more accurate activations.

Latched in a `OnceLock`, against this repo's usual preference, and the exception
is specific: there is no config plumbed to the matmul hot path, and the one
harness that A/Bs it — `peregrine flip-rate --candidate-env` — already runs each
arm in its **own process**, precisely because latched knobs cannot be toggled
in-process. So the latch cannot produce the vacuous same-arm comparison it would
produce for a serving knob.

### `COLI_SPEC_GDN`

Speculation on a **recurrent** architecture (`Arch::HybridGdn` — Qwen3.5 and
Qwen3-Next). Default **off**; output is unaffected either way.

A KV-only arch rejects a draft by truncation: the rows are dropped and nothing
remembers them. A hybrid's linear-attention layers do not keep rows at all —
they keep a delta-rule state that the verify forward has already folded every
drafted token into, and `truncate` cannot reach it. Before this knob the engine
simply refused to draft there (`spec_reject_is_kv_only`), printing a line that
said so.

The rollback it enables is `GdnState`'s documented protocol: snapshot every
recurrent layer before the verify forward; on **full** acceptance drop the
snapshot, because the state is already exactly right; on partial acceptance
restore it, rewind the KV to the same boundary, and re-advance over precisely
the rows the client was sent. All the partially-accepting sequences of a tick
re-advance in **one** forward, not one each.

**Measured 2026-08-20 on the resident Qwen container and it is a net loss
there**: 0.786 → 0.506 tok/s at B=1, 1.55× slower, because that container's MTP
head agrees with the main model only ~9.5 % of the time (against 83 % on
streaming GLM). The knob works; the head it drives on that container does not
earn its rows. See
[`bench-data/2026-08-20-spec-gdn-qwen/`](../bench-data/2026-08-20-spec-gdn-qwen/README.md),
which also records that the cost is the **replays**, not the snapshot below —
the snapshot measured at under 1 % of the overhead.

It is a knob and not an unconditional enable because the snapshot is not free:
~3.1 MB per linear layer, so **≈151 MB per drafting sequence per tick** at
Qwen3.5-27B's 48 linear layers, charged per *sequence* while the resident
weight read that dominates a forward is shared across the batch. Past some
width the copies cost more than the accepted tokens repay, and
`COLI_SPEC_GDN_MAX_B` is where the operator puts that width. `/metrics` reports
the cost directly under `spec`: `gdn_snapshot_bytes`, `gdn_replays` (ticks that
missed the free full-acceptance path) and `gdn_replay_rows`.

`gdn_replays` is the number to watch. Full acceptance is free and partial
acceptance costs a second forward, so the useful lever is whatever makes whole
runs land — which is the `COLI_SPEC_CONF` floor, measured at +37 % on the
streaming arch for exactly that reason. Note that a per-token accept rate of
0.83 is only a 0.39 chance of a *complete* run at depth 5, so expect replays to
be common until the floor is tuned.

Output-neutral: `accept_run` still decides acceptance by argmax identity, so a
greedy request emits the same stream speculated or not. That is asserted end to
end by `recurrent_speculation_does_not_change_the_served_token_stream`, and the
rollback itself by `a_rejected_draft_leaves_no_recurrent_trace` and
`a_partially_accepted_draft_re_advances_exactly_the_committed_rows` — each with
a negative-control arm, because a rollback test that passes when the rollback
does nothing is testing nothing.


### `COLI_DRAFT_NGRAM`

A second speculative draft source, costing no weights, no forward pass and no
training. Set it to the longest suffix to match (typical `3`); `0` is off, and
anything below the 2-token floor reads as off. It needs `COLI_DRAFT` non-zero
for the depth, but **not** an MTP head — a container converted without `--mtp`
can speculate through this alone.

It proposes whatever followed the most recent earlier occurrence of the
sequence's current suffix. That is the right guess exactly when the output
repeats something already in context: quoted code, an edited file, a repeated
identifier, a list being walked, or a model that has fallen into a loop.

**Why it is worth a second source rather than a tuning knob on the first.** An
MTP draft step is a full sparse-MoE layer at `s_n = 1` — on the streaming
container ~300 MB of SSD per step, with none of the batch-union amortization
the verify forward gets. This is a backward `memcmp` over the token log. So
when it matches it is not marginally cheaper, it is free; and
[`ideas-from-colibri.md`](ideas-from-colibri.md) records why that matters
more here than elsewhere — on a disk-bound engine speculation only *loses* when
drafts are rejected, so a source whose acceptance approaches 1 sidesteps the
failure mode both engines measured.

Prompt-lookup takes priority whenever it matches; the head drafts when it does
not. The two are alternatives per tick rather than a chain, because `mtp_draft`
continues from a hidden state that assumes its own prefix and cannot be seeded
with someone else's tokens.

**Greedy requests only**, and not as a policy choice: an n-gram draft is not
drawn from any distribution, so `accept_run_sampled` would have no `q` to score
it against and the distribution-preserving guarantee would be void.
`COLI_DRAFT_SAMPLED` does not extend to it.

It never *invents* a continuation — it only replays what the history holds — so
a run of identical tokens drafts one per tick rather than filling the depth.
That is deliberate: on the streaming track an invented token that misses costs a
verify row's worth of expert reads.

`/metrics` reports it under its own `ngram` block (`proposed` / `accepted` /
`accept_rate`) rather than pooling it into `spec`, because averaging a free
source with an expensive one produces a number that decides nothing. Output is
unaffected — `accept_run` still decides by argmax identity, asserted by
`prompt_lookup_drafts_do_not_change_the_served_token_stream` and, on a headless
checkpoint, `prompt_lookup_speculates_without_an_mtp_head`.


### `COLI_SPEC_UNION_MAX`

The **cost-side** twin of [`COLI_SPEC_CONF`](#coli_spec_conf). Default **off**.

Speculation's economics on the streaming track is one fraction:

```
speedup = (1 + accepted) / union_growth
```

`COLI_SPEC_CONF` prunes drafts by expected **acceptance** — the numerator — and
that alone inverted the `COLI_DRAFT=4` regression into +37 %. Nothing pruned
them by expected **cost**, which is the term the measured 2.63× union growth at
γ=4 actually lives in. This is that knob: a ceiling on the routed-expert union
entries a single tick may cost.

The projection is deliberately **conservative**. Expected entries are
`rows × (entries per row)`, where the per-row figure is an EWMA of what recent
ticks actually cost (`ecache` hits + misses is exactly the union entries the
warm tier resolved). A real union is *sublinear* in rows — that sublinearity is
the whole batching win — so a linear projection overestimates and the ceiling
cuts sooner than strictly necessary. For a budget that is the safe direction,
and it is written down here rather than rediscovered from a disappointing sweep.

A ceiling may stop **speculation** and never stops **progress**: a budget too
small for even one row per sequence still yields depth 0, not a refusal to
decode. Depth-only, exactly like the confidence floor, so a greedy stream is
bit-identical whatever the setting — swept and asserted by
`the_union_ceiling_is_depth_only_and_changes_no_token`, with the arithmetic
itself pinned separately in `union_depth_cap_prices_rows_and_never_stops_progress`
because on a resident fixture the gate is inert and an engine test alone would
pass without exercising it.

Inert on a resident model, where `ecache_stats` is `None` because no expert is
ever read. `/metrics` reports `spec.union_stops` beside `spec.conf_stops`:
together they say which term of the fraction is actually limiting a run.

**Unset by default and deliberately untuned.** The number that should set it is
`decode.tokens_emitted` against `ecache`, measured on the real container.
Picking a ceiling before that measurement exists would be tuning against a
quantity nobody has measured, which is the failure
[`measurement.md`](measurement.md) opens with.


### `COLI_DRAFT_TREE`

Verify prompt-lookup **and** the MTP head in one forward, instead of choosing
one. Default **off**.

Today they are alternatives: when the n-gram matches it wins and the head's
chain is discarded unseen. They are frequently right about different
continuations, and one forward can check both — root = the pending token, one
branch per source. Whichever the model's own argmax follows is what commits, by
the same greedy-identity rule a chain uses, so the served stream is unchanged.
The hedge is skipped when the two sources agree on their first token: the
branches would not be alternatives, and paying two rows to verify the same
candidate twice is strictly worse than the longer chain.

Committing a tree is the one thing a chain never needed. An accepted path is a
**non-contiguous** subset of the block's cache slots, with the rejected siblings
interleaved, so `truncate` — which can only drop a suffix — cannot express it.
`SeqKv::retain_tail` gathers the kept rows down instead, and that is a pure move
rather than a recomputation because each row was roped at its **tree depth**,
which is exactly the position it lands on once its ancestors pack below it.

**MLA only.** A recurrent layer advances one delta-rule state row by row, so
siblings would chain rather than branch; the batched GQA path takes no key set,
so a mask there is silently ignored — the worst outcome of the three, because it
looks like it worked. Both are refused. Which means this runs on the *streaming*
track, where an extra verify row costs disk bytes, and not on the resident
track, where extra rows would be nearly free. That is backwards from where the
value is, and it is why tree width must be spent against
[`COLI_SPEC_UNION_MAX`](#coli_spec_union_max) rather than set to a constant.

Two costs to weigh before enabling it, both real and neither hidden:

- **Every branch row is a full row of the routed-expert union.** A two-branch
  hedge roughly doubles a sequence's draft rows, and on the streaming container
  that is bytes, not FLOPs.
- **A tree row's key set is explicit**, so it is O(context) to build and walks an
  index list where a dense row runs a tight loop. At a 4 k context and five nodes
  that is ~160 KB per forward; at 100 k it is megabytes. Trees are therefore
  cheapest at **short** context — the opposite of the usual intuition. The fix is
  a compact `prefix + extras` key-set representation so the prefix stays a range;
  until that exists, this knob is for short-context workloads. Rows that are
  *not* part of a tree are unaffected: their entry is `None`, meaning dense, so
  one sequence hedging never drags the rest of the batch off the fast path.

Greedy requests only — a sampled request needs `accept_run_sampled` and the `q`
each draft was drawn from, and there is no tree analogue of that rule.

`/metrics` reports `spec.trees` (hedges taken) and `spec.tree_branch_wins`
(hedges that paid — the accepted path left the prompt-lookup branch). `trees`
climbing while `branch_wins` stays flat means the extra rows are buying nothing
and the knob should go back off.


### `COLI_MTP_HEAT`

Let the MTP head's experts accumulate residency heat. Default **off**.

Every other field the draft `ForwardCtx` withholds is withheld because a
speculative draft must not feed a main-stream signal — prediction, calibration,
lane balance. Heat looks like one of those, and for a draft running the *main*
stack it would be. But the MTP head is layer index `n_layers`, and **nothing
except drafting ever executes that layer**: its heat row has no main-stream
competitor to skew.

The heat table has been sized `n_layers + 1` since 2026-08-09 precisely so that
row exists — before that resize, `bump` dropped out-of-range writes silently and
the LFRU eviction score and the VRAM reheat ranking were blind to a whole
layer's experts. The draft path's blanket `heat: None` has kept the row empty
ever since, so they still are. This closes that.

It matters most on the streaming container, where that layer is read in the
worst regime the engine has: once per **draft step**, at `s_n = 1`, with no
batch-union amortization, and stored int8 until
[`--mtp-target`](tools.md#--mtp-target-the-one-rung-on-this-ladder-with-no-quality-gate)
converts it. Per byte it is the strongest resident candidate in the container.

A knob rather than a default because it is a genuine trade: heat drives eviction
and VRAM promotion, so MTP experts earning residency means main-stream experts
losing it, out of the same 12 GB. Output-neutral on the CPU path; on a GPU build
it changes which arm computes an expert — a residency decision, not a value one,
but the reason this is opt-in rather than assumed. Inert without a GPU tier,
where no heat table is built at all.

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

### `COLI_RLM` — recursive refinement at contested decode positions

Enable the **Recursive Language Model** controller in the decode loop. After a token's
ordinary forward produces logits, the controller inspects them (top-2 margin for
greedy, distribution entropy for sampled) and, if the pass was *uncertain*, triggers
one or more recursive passes that re-run a configurable subset of transformer layers
on the refined hidden state, then recompute the head. Easy tokens terminate after
pass 0 (the ordinary forward); contested tokens spend extra compute for sharper
logits. Same horizontal-vs-vertical dynamicity split as MTP, on a different axis:
MTP runs layers ahead *across positions*, RLM runs layers *deeper at one position*.

| Sub-knob | Default | Effect |
|---|---|---|
| `COLI_RLM` | off | master switch |
| `COLI_RLM_DEPTH` | 2 | max recursive passes per token (cap 4) |
| `COLI_RLM_LAYERS` | 4 | how many of the last transformer layers each pass re-runs |
| `COLI_RLM_MARGIN` | 0.1 | greedy: top-2 logit gap below which to recurse |
| — | — | sampled: recursion threshold is entropy > 0.5 (hardcoded, see `rlm.rs:128`) |

**Why this is the right shape for peregrine specifically.** The recursive pass
re-runs the warm/expert-cache-hot path, **not** the cold-disk path. On the second pass
for a contested token the routed expert set is mostly the same as pass 1 (the hidden
has barely moved), so the warm cache serves most of it — measured at 100 % hit on a
repeated forward (README's "warm cache 3.58×" figure). The recursive pass therefore
runs from `ecache`, not SSD: extra compute for one contested token costs a fraction of
its first-pass time. This is also why the recursive pass goes through `forward_ctx()`
with `route_log: None` and `timings: None` — drafts must not skew the prefetch
predictor (`modeled after `mtp_draft_with`'s isolation contract).

**Composition with MTP** — `generate_speculative` recurses only at the
post-acceptance contested position (the row whose logits decide the next round's
`next`): drafted tokens are accepted/rejected by argmax of the verify forward, and
any accepted or rejected position is *already decided* — recursion there would either
contradict the spec-decode contract or burn compute on a token that won't be emitted.
Position 0 (`next`) is committed as argmax before the recursion loop and is left
alone. The contract `speculative_matches_greedy` enforces (each token is the model's
argmax) holds: the refined `next` is still just an argmax, of a sharper
distribution.

**Telemetry.** `Model::rlm_stats()` returns `(recursive_passes_emitted,
tokens_that_triggered_at_least_one_pass)`, surfaced on serve's `/metrics`
(`rlm` object), the engine `[rlm]` shutdown line, and `(0, 0)` when `COLI_RLM`
is unset — same role as `lookahead_issued()` for the router look-ahead.

**Status**: wired in both decode surfaces (2026-08-13) — the stdio engine via
`model.rs::generate` / `generate_speculative`, and `peregrine-serve`'s batched
accept loop via `Model::rlm_refine_external` over each request's own KV (same
policy, same depth cap, local depth counter; composed with MTP exactly as the
model-resident path: raw logits decide acceptance, only the post-acceptance
contested row refines, sampled speculative runs are never refined). Arity /
margin decisions in `crates/peregrine-model/src/rlm.rs`. Off-by-default,
structurally inert (the enabled check precedes any copy or clone, so the
bit-identity gates stay byte-identical). **Quality unmeasured on a real
checkpoint.** Size the trade against `Model::prediction_flip_rate` — that is
the offline metric for "did recursion move the argmax on contested tokens" —
which is exactly what an operator considering `COLI_RLM=1` wants to know first.

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
| `COLI_GPU_DENSE` | off | place a **dense** model's MLP weights in VRAM and run them on the device. Gated on the *value* (`=0` means off — it once read `is_ok()`, so `=0` enabled it). Dense containers only: refused when the checkpoint has routed experts. **Changes token values** — see [below](#coli_gpu_dense) |
| `COLI_GPU_DENSE_LAYERS` | fit to free VRAM | pin the resident layer count instead of taking whatever fits. **Set this for any measurement arm**: fitting to free VRAM makes placement depend on whatever else holds the card, so two boots of one container can differ |
| `COLI_GPU_DENSE_HEADROOM_MB` | 1024 | VRAM the dense tier refuses to spend, leaving room for activations and the context |
| `COLI_GPU_SPILL` | off | act on the lane balancer's `GpuSpill` verdicts by queueing the spilled `(layer, expert)` pairs for the next residency generation. Off = the verdict stays advisory, the historical behaviour |
| `COLI_CUDA_GEMV` | **on** | decode-shaped GEMV kernel for M=1 (a GEMM wastes M there); `=0` falls back to the general path |
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
| `COLI_IO_LATENCY` | off | per-read submit→complete **distribution** plus the thread's page-fault delta — [note](#coli_io_latency) |
| `COLI_CALIB_CAPTURE` | unset | path to write an activation-importance trace to, for `peregrine-requantize --calib`. Read per model load rather than latched, so it can co-run with `COLI_PREDICT_EVAL` in one instrumented pass; `peregrine calib-capture` is the standalone subcommand |

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

### `COLI_IO_LATENCY`

Records a **histogram** of per-read submit→complete times on the streaming
rings, plus this thread's minor/major page-fault delta over the same window.
Printed at shutdown as `[latency] expert-read: …`.

**Deliberately a histogram and not an EWMA.** Every other I/O figure this engine
publishes is a mean or a steady-state aggregate — GB/s in `iobench`, tok/s in
`bench`, per-lane wall-time in `telemetry.rs` — and *a mean is exactly the
statistic an SSD garbage-collection tail survives*. Periodic
multi-hundred-millisecond stalls would move a mean by a few percent while
dominating the tail. `IoTuner` smooths that signal by design, so a second EWMA
would have inherited the blindness it was added to remove. Nothing here decays.

Sampling is **per completion**, not per wave: a wave mean would average away the
single slow read being hunted.

**Reading the output.**

- The tail figure is **p99/p50**, not p99/mean. The first version used the mean
  and was wrong in the only case that matters: one large outlier drags the mean
  *above* p99, so a fat tail reported as a ratio below 1 and read as flat. The
  median is not moved by the outliers being hunted.
- `max/p50` is reported beside it, because **p99 structurally cannot resolve
  fewer than `count/100` stalls** — one stall in a hundred reads reports flat,
  correctly. A window where p99 is flat but `max` is not is reported as "rare
  stalls p99 cannot resolve at this sample count", not as "no tail".
- An undersampled quantile prints `p99=n/a(20<100)` rather than a number. A p99
  from twenty samples is the slowest of twenty wearing the name.
- **This is not a device measurement.** Submit→complete includes queueing behind
  the ring's own depth cap and io-wq scheduling; the report says so rather than
  letting a fat tail read as "the drive stalled".

Off by default: a tail hunt is a diagnostic run, not something the steady-state
path should pay an `Instant::now()` per completion for.

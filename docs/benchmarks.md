[« Docs index](README.md)

# Benchmarks

The full same-hardware study — peregrine (Rust) vs colibrì (C) on the real
**GLM-5.2 744B** int4 model, with methodology, all numbers, and limitations —
is [**peregrine-vs-colibri.md**](peregrine-vs-colibri.md). This page is the
summary.

## Headline numbers

Reference box: single RTX 3060 (12 GB) / Ryzen 5 5500 / 46 GB RAM,
CPU-streaming decode.

| | peregrine (Rust) | colibrì (C) |
|---|---|---|
| Decode, single sequence (steady state) | **0.062 tok/s** (16.08 s/tok) — was 0.054; see [the io-claim fix](#the-io-claim-fix-2026-08-09) | **0.077 tok/s** |
| **Batched decode, B=16 (aggregate)** | **0.280 tok/s** (4.4× over B=1) — see the note below on why the 2026-08-01 pass reads 0.224 | — |
| Warm cache on a repeated forward | **3.58×** (100 % hit, 0 disk) | learned pin |
| Raw device read rate (after I/O ports) | **~980 MB/s** | ~870 MB/s — but see [the laptop re-measurement](#second-box-glm-52-on-a-7-gb-laptop), which inverts this |
| Warm-cache hit rate, sustained decode (10 GB cache) | 0.6 % (measured) | — |
| Tokenizer throughput | **204 MB/s** (gigatoken) | — (HF: 6 MB/s) |

### The io-claim fix (2026-08-09)

Decode was **21.83 s/tok** and is now **16.08 s/tok** — **1.36×**, from four lines.
The I/O lane claimed work with

```rust
let start = io_work_ref.fetch_add(batch, Ordering::Relaxed);
if start >= n_plans { break; }     // no work left for this ring
```

where `batch` was `COLI_IO_BATCH` (default **16**). A decode token routes **8
experts per layer**, so ring 0 claimed all 8 and rings 1–3 got starts 16/32/48 —
every one `>= n_plans` — and **broke without issuing a single read**. One ring did
four rings' work, on every sparse layer of every decode token. The default is
correct for prefill, whose per-layer union is ~69 experts; it collapsed decode to
1× parallelism.

The claim now ceil-divides across rings, keeping the configured value as a ceiling:
decode gets `ceil(8/4) = 2` and all four rings run, prefill still gets its full 16.

| at defaults | before | after |
|---|---|---|
| decode | 21.83 s/tok | **16.08 s/tok** |
| p50 inter-token | 21.9 s | **15.5 s** |
| **io duty** | **24 % of 4 rings** | **84 %** |

**It was invisible until the lane counters gained a wall-clock denominator.**
`io_us`/`cpu_us`/`gpu_us` are summed over *threads*, so 17 s of `io_us` is equally
four rings busy for 4.3 s or one ring busy for 17 s. `LaneTimings::lane_wall_us`
distinguishes them, and `io duty 24 % of 4 rings` pointed straight at the claim
loop. Reproduced independently before any code change by setting `COLI_IO_BATCH=2`
(14.80 s/tok, 90 % duty). Full working:
[`bench-data/2026-08-09-decode-levers/`](../bench-data/2026-08-09-decode-levers/README.md).

## How to read them

- **Both engines are disk-bandwidth-bound** on one box: each token routes to
  600 experts ≈ 11.3 GB that must come off the SSD, and no cache that fits in
  46 GB can hold a meaningful slice of the ~370 GB expert working set.
  colibrì is ~1.4× faster at raw *single-sequence* streaming (more mature
  streaming path); peregrine's O_DIRECT + parallel-rings ports have since
  pushed its raw device read rate past colibrì's on the same box.
- **Continuous batching is where the concurrent design starts paying now**:
  decoding B sequences together reads each routed expert **once per step and
  shares it across the batch**, so step time grows only 3.6× for 16× the
  tokens — a measured **4.4× aggregate gain at B=16** (0.064 → 0.280 tok/s).
  The win is amortization of the byte budget, not a faster drive.
- **The warm cache works but never fits**: 3.58× on a repeated forward proves
  the mechanism; the 0.6 % hit rate on sustained decode is a *capacity*
  result — a 10 GB cache against a ~180 GB 16-token working set can hit ~5 %
  at best, however the router behaves. colibrì independently finds its PILOT
  prefetch neutral and MTP a net loss on disk-saturated MoE decode.
  **That entropy attribution was an inference, and it has now been measured
  and refuted** (2026-08-09, `bench-data/2026-08-09-prefetch-causes/`):
  `route-stats` over a real-text GLM-5.2 trace puts consecutive-token overlap at
  **33.55 % against a 3.12 % independence null** — 10.7×, i.e. routing is
  strongly predictable from the previous token. The low hit rate is therefore a
  capacity or policy result, not a router result; one token's routed set is
  ~11.3 GB and the cache was 4 GB, so a slab cannot survive to be reused. Run
  `COLI_UNION_STATS=1` for the live per-step sharing factor; see the
  correction in [the study](peregrine-vs-colibri.md#52-cache--locality-analysis-peregrine-measured).
- **The 3-lane scheduler's full advantage is latent** without expert
  residency: colibrì reaches **6.84 tok/s on 6× RTX 5090** (full residency),
  which is exactly the regime peregrine's concurrent design targets.
- On the resident (no-disk) path the `peregrine-par` compute pool lifts B=256
  aggregate to **79.6k vs 66.3k tok/s serial (1.2×)** with no small-batch
  regression; that lever scales with hidden size.

## Reproducing

```bash
# aggregate decode-throughput sweep over batch sizes
COLI_MODEL=/path/to/model cargo run --release --bin peregrine -- bench 1 4 16

# tokenizer throughput (no weights loaded)
cargo run --release -p peregrine-serve -- --model /path/to/model \
    --bench-tokenizer big_text_file.txt
```

`COLI_BENCH_STEPS` sets decode steps per batch size (default 3). The
[study](peregrine-vs-colibri.md) documents the full methodology, the exact
env configurations, and threats to validity.

**Caveat:** the study's measurements predate the adaptive-runtime and
completion sweeps — every knob those added is env-gated and bit-identical
when off, so the numbers stand as the baseline. That re-run has now happened:
The 2026-08-01 re-measure is [below](#benchmark-pass--2026-08-01-post-improvement-re-measure), no longer a separate page.

**Harness caveat:** a single `bench 1 4 16` invocation runs every batch point
in one process, and the later points inherit the earlier ones' allocator and
cache state — baseline B=16 measures 0.143 tok/s in-sweep vs 0.224 isolated.
Use it to observe batch scaling *within* one configuration; to compare
configurations against each other, run one batch size per fresh process (see
the 2026-08-01 pass for a runner that does).

## Second box: GLM-5.2 on a 7 GB laptop

A same-hardware A/B on a much smaller machine than the reference box, against a
**freshly converted GLM-5.2 container** (`zai-org/GLM-5.2-FP8` → colibrì's
`convert_fp8_to_int4.py`, per-row int4 experts / int8 embed+lm_head, DSA and MTP
skipped; 141 shards, **349 GB**).

**Box:** Intel i5-1235U (2P+8E, 12 threads), **7.4 GB RAM + 7.6 GB swap**,
LUKS-encrypted NVMe, no GPU.

| Metric | peregrine | colibrì |
|---|---|---|
| Tokenizer, whole-buffer (GLM-5.2 vocab) | **258.8 MB/s** | n/a — consumes ids |
| Tokenizer, pooled parallel ×12 (warm) | **707.0 MB/s** | n/a |
| *reference: HF `tokenizers`, same corpus* | *1.71 MB/s* | — |
| Disk read, O_DIRECT, 8-way | 0.84 GB/s | **2.02 GB/s** |
| Disk read, O_DIRECT, 1-way | 0.54 GB/s | **0.84 GB/s** |
| Disk read, buffered, 8-way | 1.40 GB/s | **1.64 GB/s** |
| Decode, any batch size | ⛔ OOM at load | ⛔ OOM at load |

**Decode does not run on this box, for either engine.** Measured from the
container headers, the always-resident (non-expert) weights are **10.59 GB**
against ~5.8 GB of free RAM+swap, so both engines are OOM-killed part-way
through load. That is a hardware-envelope result, not an engine defect, and it
hits both sides identically — the box needs ≥ 16 GB of RAM+swap to produce
decode rows at all.

**The I/O rows invert the headline table above**, and the first cause turned out
to be a defect rather than the hardware. `Reactor::read_direct_aligned` — the
call the streaming lane makes under `COLI_DIRECT=1` — looped **one region at a
time**, each its own `submit_and_wait`. The O_DIRECT lane therefore ran at queue
depth 1 no matter what `COLI_IO_BATCH`/`COLI_IO_RINGS` said, while the buffered
sibling submitted all 96 regions of a 16-expert batch in one call. That is why
direct reads measured *slower* than buffered ones, which is backwards.

Batching the submission (2026-08-02) is worth a consistent **1.2–1.3×** across
four configurations, controlled A/B on the same shard sizes, run-to-run spread
±0.01 GB/s:

| config | before | after |
|---|---|---|
| 6 MB × 24 regions, 8 rings | 0.60 GB/s | **0.73** |
| 6 MB × 24 regions, 4 rings | 0.53 | **0.66** |
| 64 MB × 4, 8 rings | 0.67 | **0.86** |
| 64 MB × 4, 4 rings | 0.53 | **0.64** |

A gap to colibrì remains, and dm-crypt was the standing hypothesis for it: on a
LUKS volume reads are CPU-bound on decryption, so *N* blocking `pread`s keep *N*
cores decrypting where the ring's completion model can leave cores idle. Testing
that means adding a pread engine behind the same `read_regions` choke point and
measuring — not asserting.

> **Retracted 2026-08-09 — the `0.84` vs `2.02` row was a two-variable
> comparison.** The pread engine landed, the test ran, and it does not support the
> hypothesis it was built for.
>
> `COLI_IO_ENGINE=pread` **implies no O_DIRECT** (`concurrent.rs` skips the direct
> path for that engine). So the recorded pair pitted uring-**with**-O_DIRECT
> against pread-**without** it — two variables at once, which is exactly the
> confound [`validation-runbook.md`](validation-runbook.md) §1a warns about.
> Held constant on a Ryzen 5 5500 / LUKS NVMe (a *different* box from the i5-1235U
> table above), 5 reps per arm at the shipped 4 rings:
>
> | arm | median | spread | verdict |
> |---|---|---|---|
> | `uring` buffered | 1.12 GB/s | 5.4 % | — |
> | `pread` ×8 | 1.06 GB/s | 10.6 % | 5.7 % gap — **inside the noise floor** |
> | `uring` O_DIRECT | 0.86 GB/s | 10.3 % | **−23 %, outside both spreads** |
>
> **The syscall shape costs nothing measurable.** What the same run does show is
> that O_DIRECT is genuinely the slow arm — which is why `COLI_DIRECT` defaults
> off. Read the `0.84`/`2.02` row above as "O_DIRECT vs threaded buffered reads",
> not as "io_uring vs pread", and not as "C beats Rust".
>
> The dm-crypt question itself is **still open** — decryption may still cost, and
> [`M1-storage-config.md`](../bench-data/2026-08-09-prefetch-causes/M1-storage-config.md)
> records the one cheap read-only measurement that would settle it. What is closed
> is the inference that the engine gap *demonstrated* it.
>
> Full working: [`M5-io-engine.md`](../bench-data/2026-08-09-prefetch-causes/M5-io-engine.md).

*Provenance of the numbers:* peregrine's come from
[`crates/peregrine-io/examples/iobench.rs`](../crates/peregrine-io/examples/iobench.rs),
which now drives `read_direct_aligned` — the production path — rather than a
sibling with its own batching. colibrì's 2.02 GB/s comes from its `c/iobench.c`,
which is **not** io_uring at all but 8 OpenMP threads issuing blocking `pread`s;
its production decode path is a 512-deep io_uring queue. The two harnesses
measure different things, so read the ratio as "this box's best threaded reader
vs this lane", not as "C beats Rust".

Reproducing the two engine-comparable rows:

```bash
# io_uring read rate: FILE BLK_MB ITERS RINGS DIRECT
cargo run --release -p peregrine-io --example iobench -- \
    /path/to/model/out-00100.safetensors 64 4 8 1
# colibrì's equivalent: file blkMB n threads direct
make -C c iobench && ./c/iobench /path/to/model/out-00101.safetensors 64 16 8 1

# tokenizer, no weights loaded
cargo run --release -p peregrine-serve -- --model /path/to/model \
    --bench-tokenizer corpus.txt
```

Single run per cell, no variance bars; a different container shard per
measurement so no row is served from page cache.

---

## Two B=16 numbers, and why both are right

This page headlines **0.280 tok/s** at B=16; the 2026-08-01 pass below reads
**0.224**. Same box, same model, different configuration — and the difference
is recorded here rather than left for a reader to trip over, because carrying
two live figures for one quantity is a mistake this repo has already made
twice.

| | 0.280 | 0.224 |
|---|---|---|
| Expert cache | 10–12 GB | **2 GB** (the box was hosting a live VM) |
| Harness | in-sweep (`bench 1 4 16`, one process) | **one batch size per fresh process** |
| Repeats | single run | 3, spread ±3 % |

Neither supersedes the other, but they answer different questions. **Use 0.224
when comparing configurations** — the isolated harness is the sound one, and
the pass below shows the in-sweep form biases later batch points badly (0.143
vs 0.224 for the same arm, +57 %). **Use 0.280 as the best-observed figure**
for a cache-generous run, remembering it is a single unrepeated point.

The 4.4× batching ratio survives both: it reproduces at 0.056 → 0.244 in the
isolated pass, so it is the one headline here that is not harness-dependent.

---

## Benchmark pass — 2026-08-01 (post-improvement re-measure)

This is the re-run [`benchmarks.md`](benchmarks.md) called for: *"the study's
measurements predate the adaptive-runtime and completion sweeps … re-running
with the knobs live is the next benchmarking pass."*

Engine at `c23f7b5` (workspace defect audit / tokenizer fast paths), the same
GLM-5.2 744B int4 model and the same box as
[the original study](peregrine-vs-colibri.md).

**What this pass predates.** The knobs measured here are the §1–§11 set — all of
which change *how* expert bytes are fetched. The work that followed this pass acts
on *how many*: adaptive prefill chunking, the cross-request prefix cache, adaptive
top-k (`COLI_ROUTE_MIN_SHARE`) and the int2 producer, tracked in
[`todo.md`](../todo.md) §12–§13. None of them is measured below, and the finding
that motivated opening that axis is the one in the first TL;DR bullet.

### TL;DR

- **The adaptive/I-O knob bundle produces no measurable throughput gain.**
  At B=16 it is **1.00×** baseline (0.225 vs 0.224 tok/s, 3 reps each). Disk-read
  counts are byte-identical between the two, which corroborates it: the knobs
  change *how* the bytes are fetched, not how many, and on this disk-bound
  workload how doesn't matter.
- **The CUDA lane is a real but modest win: 1.09× at B=16** (0.244 vs 0.224,
  tight across 3 reps). These are the **first GPU-lane numbers ever measured**
  for peregrine — the original study was CPU-streaming only.
- **`peregrine bench 1 4 16` is not a sound harness for comparing configs.**
  Running the batch points in one process biases the later ones badly: baseline
  B=16 reads **0.143 tok/s in-sweep vs 0.224 isolated (+57%)**. Every number
  below that matters was re-measured one batch size per fresh process.
- Batching itself reproduces the published headline: **4.4× aggregate at B=16**
  (0.056 → 0.244 tok/s).

### Setup

| | |
|---|---|
| Engine | `c23f7b5`, `--release` (fat LTO); GPU arm additionally `--features cuda` (nvcc 13.3, links `libcudart.so.13`) |
| Model | `GLM-5.2-colibri-int4-with-int8-mtp`, 358 GB on disk, 78 layers, vocab 154 880 |
| Box | Ryzen 5 5500 (6C/12T, AVX2, no VNNI), 46 GB DDR4, RTX 3060 12 GB (sm_86), LUKS-encrypted NVMe |
| Shared env | `MALLOC_ARENA_MAX=2`, `COLI_ECACHE_GB=2`, `COLI_ROUTE_STATS_PERSIST=0`, `COLI_BENCH_STEPS=3` |
| Containment | each run in a `systemd-run --scope -p MemoryMax=24G` cgroup |

`COLI_ROUTE_STATS_PERSIST=0` matters for validity: persistence is on by default
and would have let each arm warm the next through `route_stats.json` in the
model directory. Every arm here starts cold.

**Cache is 2 GB, not the study's 10–12 GB.** The box was hosting a live VM, and
the study records an OOM kill at 24.7 GB RSS. Peak RSS came in at **16.4–17.3 GB**
across every run, so the cap was never approached. Shrinking the cache costs
almost nothing here — measured cross-token expert hit rate is 0.6 %.

#### Arms

| Arm | Knobs on top of defaults |
|---|---|
| `baseline` | none — every adaptive knob defaults to historical behavior, so plain defaults *are* the published study's config |
| `improved` | `COLI_DIRECT`² `COLI_REGBUF`¹ `COLI_IO_TUNE` `COLI_LANE_BALANCE` `COLI_SHAPE_SPECIALIZE` `COLI_HYPER_SCHED` `COLI_PREFETCH_TUNE` `COLI_ENTROPY_ADAPT` `COLI_REPLICATE_K=8` |
| `gpu` | `improved` + `COLI_GPU=1` on the `cuda` build |

¹ **`COLI_REGBUF` turned out to be inert.** Audited after this pass:
`register_read_buffers()` / `IORING_OP_READ_FIXED` are implemented and tested,
but nothing calls them outside tests and no code reads the variable — the
streaming path always takes the plain read. So the `improved` arm was eight live
knobs and one no-op. It does not change the conclusion (the bundle was already
1.00×, with byte-identical disk reads), but the arm description would otherwise
overstate what was exercised. Tracked in [`todo.md`](../todo.md) §4.

² **`COLI_DIRECT` was crippled, which this pass could not have known.**
Discovered 2026-08-02: `Reactor::read_direct_aligned` — the call the streaming
lane makes under `COLI_DIRECT=1` — submitted **one region at a time**, so the
O_DIRECT lane ran at queue depth 1 regardless of `COLI_IO_BATCH` or
`COLI_IO_RINGS`, while the buffered sibling submitted all 96 regions of an expert
batch at once. Direct reads were consequently *slower* than buffered ones.

So the nine-knob bundle above was really **eight live knobs, one no-op, and one
running at a fraction of its intended depth** — and the headline **1.004×** rests
on that. Unlike footnote ¹ this one *may* change the conclusion, because
`COLI_DIRECT` is the knob most directly aimed at the disk-bandwidth bound the
whole §1–§11 program targets. The fix measured 1.2–1.3× on the I/O lane in
isolation on a LUKS laptop; its effect on end-to-end decode is unmeasured.

**Re-running this pass is the first thing to do on hardware that can load the
model** — see the [validation runbook](validation-runbook.md#2-re-run-the-2026-08-01-knob-pass--its-conclusion-is-suspect).
Until then, treat 1.004× as provisional rather than as settled evidence that
faster byte movement is exhausted.

### Results — B=16, isolated, 3 repeats

The headline measurement. One batch size per process, fresh load each time.

| Arm | rep 1 | rep 2 | rep 3 | **median** | vs baseline |
|---|---|---|---|---|---|
| baseline | 0.225 | 0.224 | 0.209 | **0.224** | 1.00× |
| improved | 0.225 | 0.225 | 0.237 | **0.225** | **1.004×** |
| gpu | 0.241 | 0.246 | 0.244 | **0.244** | **1.089×** |

Aggregate tok/s. Spread is tight (±3 % worst case) — B=16 runs are long enough
to average out host contention.

`improved` and `baseline` agree to three digits on two of three reps. Treating
the knob bundle as a throughput lever on this workload is not supported.

### Results — B=1, isolated, 2 repeats

| Variant | rep 1 | rep 2 | median | note |
|---|---|---|---|---|
| baseline | 0.057 | 0.054 | 0.056 | |
| `direct_only` (O_DIRECT alone) | 0.063 | 0.065 | **0.064** | tight; only lead worth chasing |
| `bundle_nodirect` (bundle − O_DIRECT) | 0.058 | 0.040 | 0.049 | ±31 % |
| improved | 0.062 | 0.049 | 0.056 | ±23 % |
| gpu | 0.067 | 0.037 | 0.052 | ±45 % |

**B=1 is too noisy to rank anything.** Single-sequence runs are ~50 s, short
enough that host contention dominates; the same config swung 45 % between
repeats. The one suggestive result is `direct_only` — O_DIRECT *by itself*
measured 1.15× baseline with a tight spread on both reps, while the full bundle
containing O_DIRECT did not. That is a lead, not a finding: n=2.

### Why the first sweep looked so different

The initial single-process sweep produced numbers that did not survive
isolation:

| B | baseline in-sweep | baseline isolated | |
|---|---|---|---|
| 1 | 0.045 | 0.056 | +24 % |
| 16 | 0.143 | 0.224 | **+57 %** |

In-sweep, B=16 runs third in the same process, after B=1 and B=4 have already
churned the allocator and the warm cache. That penalty landed on whichever arm
was measured that way, and it manufactured an apparent **1.45× for `improved`
and 1.64× for `gpu`** that vanished under isolation. It also manufactured an
apparent B=1 *regression* for `improved` (0.028 tok/s) that no later run
reproduced — the closest repeat was 0.062.

`docs/benchmarks.md` currently recommends `peregrine bench 1 4 16` for exactly
this comparison. It is fine for observing batch scaling within one config; it
should not be used to compare configs against each other.

### GPU lane

First measurement of this lane. `COLI_GPU=1` on the `cuda` build:

```
[CUDA] device 0: NVIDIA GeForce RTX 3060, 12.5 GB VRAM, sm_86
peregrine: GPU tier holds 62 experts in VRAM
```

62 residents against ~19 200 experts routed per B=16 step is **0.3 % residency**,
and the disk-read counters show it: 7867 reads vs 7957 for the CPU arms, a 1.1 %
reduction. A 1.09× throughput gain from a 1.1 % byte reduction means the win is
not residency — it is the GPU lane overlapping compute that the CPU arms
serialize against I/O. That is the concurrent-scheduler thesis showing up in a
measurement for the first time, just at a scale this GPU can barely express.

**Caveat:** the GPU arm is *not* token-identical. Per the study's limitations,
"GPU experts compute in f32 (not bit-exact vs int4 CPU) by design." `baseline`
vs `improved` is a clean correctness-neutral A/B; `gpu` trades exactness for
speed and should be read as a different configuration, not a free win.

### Threats to validity

- **Contended host.** An OSX-KVM VM (~80 % CPU) and its VNC helper (~100 % CPU)
  ran throughout. This is the dominant noise source and the reason B=1 is
  unusable. B=16 medians are stable regardless. Package temp stayed at 31 °C, so
  thermal throttling is ruled out.
- **Page cache not dropped between runs** (no passwordless sudo). With a 358 GB
  model against ~28 GB free, carryover is a small fraction of the working set,
  and the O_DIRECT arms bypass the page cache entirely.
- **2 GB expert cache**, so absolute tok/s is below the study's 10–12 GB config.
  Cross-arm comparisons are unaffected — every arm used the same cache.
- **n=3 at B=16, n=2 at B=1.** Enough to kill the large claimed effects; not
  enough to resolve effects under ~5 %.
- **Short runs** (3 decode steps) to keep wall-clock tractable at ~0.2 tok/s.

### Reproducing

```bash
# headline: one batch size per process, repeated
BATCH=16 VARIANTS="baseline improved gpu" scripts/bench-b1-isolate.sh out 3

# B=1 knob isolation (adds direct_only / bundle_nodirect variants)
BATCH=1 scripts/bench-b1-isolate.sh out 2

# the original 3-arm sweep (batch scaling within a config only)
MEMMAX=24G COLI_ECACHE_GB=2 scripts/bench-arms.sh out baseline improved gpu
```

Raw output for this pass is committed under
[`bench-data/2026-08-01/`](../bench-data/2026-08-01) — `sweep/` (the original
3-arm in-process sweep), `b1-isolate/` and `b16-repeat/`. Every run contributes
the engine's stdout table (`.out`), its stderr including the `[ecache]`
disk-read counters (`.err`), and wall/RSS stats (`.stat`). `sweep/` additionally
captures each arm's exact environment (`.env`); for the isolate runs the
environment is the `variant_env` table in `scripts/bench-b1-isolate.sh`, and
the per-variant tallies are in `results.csv`.

---

## Benchmark pass — 2026-08-03 (three-arm sweep, archived as data)

Run at `be3ba8b` (still HEAD as of 2026-08-08), `scripts/bench-arms.sh`, three
arms × three batch sizes, 3 steps, `COLI_ECACHE_GB=4`, real GLM-5.2 744B int4.
Raw output under [`bench-data/2026-08-03/`](../bench-data/2026-08-03).

| arm | B=1 | B=4 | B=16 | wall_s | peak_rss_gb | major_faults |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.046 | 0.103 | 0.190 | 446.59 | 17.58 | **514 315** |
| improved | 0.044 | 0.117 | **0.224** | 398.23 | 17.10 | **0** |
| gpu | 0.044 | 0.108 | 0.209 | 433.31 | 17.36 | 626 |

**This section archives the data; it does not claim a result.** Two things in
that table look like findings and are not, and saying so is the whole point of
writing it up rather than leaving the directory unexplained.

**The GPU arm appears to regress** — 0.235 at B=16 on 2026-08-01 → 0.209 here,
now *below* `improved` at 0.224, inverting the previous ordering.

**But the arms are not comparable, and the fault counts say so.** Baseline took
**514 315 major faults**; `improved` took **zero**. Half a million major faults
means that arm spent the run refetching pages from disk under the `MemoryMax`
cap, and the 2026-08-01 baseline did not. A wall-clock comparison across two
arms whose fault behaviour differs by five hundred thousand events is not
measuring the engine. That alone can produce the apparent GPU regression with no
engine change whatsoever.

**Threats to validity.** Same contended host as the other passes. 3 steps is
short. And the fault asymmetry above is not a caveat on the numbers — it is a
reason not to use them for a cross-arm claim at all. Re-run before anyone
concludes the GPU lane got slower; the useful comparison is
`improved`-vs-`gpu` *within* this pass (0.224 vs 0.209, both at low fault
counts), which is the pair that shares its conditions.

**Reproducing.** `scripts/bench-arms.sh bench-data/<date> "1 4 16"`.

---

## Benchmark pass — 2026-08-07 (routing-statistics pass)

The first pass whose deliverable is **counters, not wall clock**. It exists to
settle §13's gate-mass mixed-precision question, which no amount of timing can
answer and which the synthetic model cannot address at all (4 experts, top-2, so
there is no weight tail to measure).

`COLI_UNION_STATS` and `COLI_GATE_STATS` had printed only from `serve`, which is
strictly single-sequence — so `s_n` was a prefill chunk length and both figures
described sharing across *positions of one prompt*, never across concurrent
sequences. The question is entirely about what happens as the batch grows.
`run_bench`, the only batched entry point, printed neither. Both now come from a
shared `report_gate_stats` / `report_union_stats`.

### Setup

GLM-5.2 744B int4 (`~/models/GLM-5.2-colibri-int4-with-int8-mtp`, 358 GB), 2 GB
expert cache, `COLI_BENCH_STEPS=2`, `COLI_ROUTE_STATS_PERSIST=0`,
`MemoryMax=24G`, **one batch size per fresh process**. Sequences start from
distinct tokens (`bench` seeds `(i*7+1) % vocab`) so their routing diverges.

### Result — batch-union sharing and the low-gate ceiling

| B | selections | distinct reads | share | all-low-gate | fraction | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 200 | 1 200 | 1.000× | **16** | 1.3 % | 0.048 |
| 4 | 4 800 | 2 235 | 2.148× | **15** | 0.7 % | 0.120 |
| 16 | 19 200 | 3 855 | 4.981× | **18** | 0.5 % | 0.243 |
| 16, γ=4 | 58 112 | 10 145 | 5.728× | **18** | 0.2 % | 0.112 |

**Gate-mass mixed-precision loading is closed by the fourth column.** The count
of experts that are low-gate for *every* row routing them is flat at 15–18 while
distinct reads grow 8.5×. That is the mechanism exactly: a read is issued per
union entry, not per row, so adding rows can only remove candidates. The whole
feature's ceiling at B=16 is 0.5 % of reads, and int2-vs-int4 saves ~25 % of
those bytes — **~0.12 % of expert bytes** for a dual-precision container, a
re-keyed warm cache and a precision-aware region locator.

Consistency check that the number is real: at B=1 the all-low-gate fraction
(1.3 %) equals the independently computed `[gate] below_1%` (1.3 %) exactly, as
it must — with one row, "every row wants it weakly" *is* "below 1 % of gate mass".

### Result — the gate tail, which sizes the lever that does exist

`[gate] routed=19200 below_0.5%=1.0% below_1%=1.3% below_2%=2.4% below_5%=12.5%`
(B=16; B=1 gives 1.0 / 1.3 / 2.8 / 14.3). So `COLI_ROUTE_MIN_SHARE=0.05` would
drop about an eighth of routed selections. **First real-checkpoint sizing this
knob has ever had.** It changes token values, so gate it with
`Model::prediction_flip_rate` before running it anywhere real.

### Result — speculation does not get its extra tokens for free

`COLI_DRAFT=4` at B=16 grew the union from 3 855 to **10 145 distinct reads
(2.63×)**. `todo.md` had claimed one union yields `1 + accepted` tokens; it does
not — draft rows route substantially different experts. Break-even needs an
accepted run above ~2.6, not above 1.

Measured verified throughput was **0.112 tok/s against 0.243 baseline (2.2×
slower)**. **That figure is harness-bound and should not be generalized**: the
bench seeds each sequence with one arbitrary token and runs two steps, so the MTP
head drafts with no context and acceptance is near worst case. The union growth
is *not* harness-bound — it depends on where routed sets fall, not on whether the
drafts were accepted. A real-workload acceptance rate is still owed.

### Threats to validity

Same contended host as previous passes (an OSX-KVM VM plus a VNC helper), and a
concurrent `nvcc` compile during the B=16 arm. **Timing numbers here are
accordingly soft**; the counters are exact and unaffected — they are counts of
routed experts, not measurements of how fast anything ran. Two steps per run is
short: enough for routing statistics, not for a throughput claim.

### Reproducing

```bash
scripts/bench-measure.sh out/union 16      # 3 arms at B=16
COLI_UNION_STATS=1 COLI_GATE_STATS=1 COLI_BENCH_STEPS=2 \
  COLI_MODEL=... target/release/peregrine bench 1   # one B per process
```

Raw output under [`bench-data/2026-08-07-union/`](../bench-data/2026-08-07-union),
now `b1/`, `b4/` and `b16/` with per-arm `.env` and a `BUILD.txt` in each.

### Addendum — 2026-08-08: the look-ahead arm was never a comparison

The B=1 and B=4 arms were re-run on 2026-08-08 because only `b16/` had been
archived, leaving two of the four rows above with no data in-tree. **The re-run
reproduced the published counters exactly** — B=1 `1 200 / 1 000× / 16 / 1.3 %`,
B=4 `4 800 / 2.148× / 15 / 0.7 %` — so the table was right; it just could not be
checked.

The re-run also fixed a defect in the harness that invalidated one arm.
`scripts/bench-measure.sh` defined `lookahead_batch` as
`COLI_ROUTER_LOOKAHEAD_BATCH=1`, but that knob **defaults to on**
(`model.rs:836-840` reads `!matches!(v, Ok("0") | Ok("false"))`), so the arm was
being compared against a baseline that already had it enabled. That is why the
original run's counters came back byte-identical to baseline and were written off
as a null result. With `baseline` now setting the knob to `0` explicitly:

| B | arm | distinct reads | disk_reads | wall_s |
|---|---|---:|---:|---:|
| 1 | baseline (look-ahead off) | 1 200 | 1 188 | 61.44 |
| 1 | lookahead_batch | 1 200 | **1 098** | 49.97 |
| 4 | baseline (look-ahead off) | 2 235 | 2 235 | 80.80 |
| 4 | lookahead_batch | 2 235 | **2 030** | 102.60 |

**The multi-row look-ahead cuts disk reads by 7.6 % at B=1 and 9.2 % at B=4** —
a real result on the repo's own test, and one the previous configuration could
not have produced at any sample size.

Two things it is not. The **union counters are identical across the two arms, as
they must be**: the look-ahead changes what is *prefetched*, not what is
*routed*, so `selections`/`distinct`/`share` cannot move and their agreement is a
consistency check rather than a null. And the **wall clock disagrees with itself**
(faster at B=1, slower at B=4) on a contended host with 2 steps per run, so read
the `disk_reads` column and ignore the seconds.

Absent at **every** B: `[predict-eval] recall=` and `[lookahead] issued=`.

**This was blamed on the `s_n == 1` gate until 2026-08-09, and that explanation is
wrong** — which matters, because it points at the wrong repair. The gate is in
`forward_hidden`, and `bench` never executes that function: `run_bench` calls
`forward_step_batched` → `forward_rows_inner`, and `forward_rows_inner` **has no
`score_and_stash` call at all**. The only call site in the tree is in
`forward_hidden`, reached from `Model::generate` and friends. So `bench 1` produces
no scoreboard either, and no amount of splitting the `s_n == 1` gate would change
that. `peregrine-serve` is in the same position, and additionally never calls
`predict_eval_report()`, so it could not print one if it had it.

The scoreboard is reachable **only from the stdio `GEN` protocol**:

```bash
printf 'GEN 64 1 2 3\nQUIT\n' | COLI_PREDICT_EVAL=1 COLI_PREDICT_EVAL_N=16 peregrine "$COLI_MODEL"
```

`COLI_PREDICT_EVAL_N=16` makes it comparable with the WASTE recall@16 table; the
default is the model's topk (8 on GLM-5.2). Getting the number from `bench` or from
the server needs a hook in `forward_rows_inner`, not a gate change.

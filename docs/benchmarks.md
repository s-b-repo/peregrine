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
| Decode, single sequence (steady state) | 0.054 tok/s | **0.077 tok/s** |
| **Batched decode, B=16 (aggregate)** | **0.280 tok/s** (4.4× over B=1) | — |
| Warm cache on a repeated forward | **3.58×** (100 % hit, 0 disk) | learned pin |
| Raw device read rate (after I/O ports) | **~980 MB/s** | ~870 MB/s — but see [the laptop re-measurement](#second-box-glm-52-on-a-7-gb-laptop), which inverts this |
| Warm-cache hit rate, sustained decode (10 GB cache) | 0.6 % (measured) | — |
| Tokenizer throughput | **204 MB/s** (gigatoken) | — (HF: 6 MB/s) |

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
  **Attributing all three to routing entropy is an inference, not a
  measurement** — the routing overlap has never been measured. Run
  `peregrine route-stats <routes.json> 256` (and `COLI_UNION_STATS=1` for the
  live per-step sharing factor) before relying on the entropy story; see the
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
[**Benchmark pass 2026-08-01**](benchmark-2026-08-01.md) measures the knobs
live (no measurable throughput gain) and reports the first CUDA-lane numbers
(1.09× at B=16).

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

A gap to colibrì remains, and dm-crypt is the standing hypothesis for it: on a
LUKS volume reads are CPU-bound on decryption, so *N* blocking `pread`s keep *N*
cores decrypting where the ring's completion model can leave cores idle. Testing
that means adding a pread engine behind the same `read_regions` choke point and
measuring — not asserting.

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

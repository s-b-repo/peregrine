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
| Raw device read rate (after I/O ports) | **~980 MB/s** | ~870 MB/s |
| Cross-token expert locality | 0.6 % (measured) | — |
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
- **The warm cache works but locality doesn't**: 3.58× on a repeated forward
  proves the mechanism; 0.6 % measured cross-token hit rate explains why it's
  ~1× on sustained decode. This independently corroborates colibrì's own
  findings that its PILOT prefetch is neutral and MTP is a net loss on
  disk-saturated MoE decode — all symptoms of high routing entropy.
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

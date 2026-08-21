# peregrine documentation

Documentation wiki for **peregrine** — a from-scratch Rust MoE inference
engine that drives CPU, GPU, RAM, and SSD concurrently. New here? Start with
[Getting started](getting-started.md); for the big picture read
[Architecture](architecture.md).

## Using peregrine

| Page | What's in it |
|---|---|
| [Getting started](getting-started.md) | prerequisites, build & test, first run, serving, troubleshooting |
| [Portability](portability.md) | what varies from one Linux box to the next — io_uring, page size, SIMD, O_DIRECT — and which fallbacks are tested |
| [`peregrine` CLI reference](cli-peregrine.md) | every subcommand (`demo`, `build`, `bench`, `build-automaton`, `dump-routes`, `galactic`, `compile-plan`), the stdio serve protocol |
| [Serving (`peregrine-serve`)](serving.md) | CLI flags, the OpenAI-compatible HTTP API, SSE streaming, auth, priority header, continuous batching |
| [Layout tools](layout-tools.md) | `peregrine-layout-reorg` and `peregrine-prune`: routing traces → disk schedules → physical checkpoint rewrite |
| [Tools](tools.md) | `peregrine-gen` (watch and time generation), `peregrine-requantize` (fewer bytes per expert), `peregrine-skipbound`, `peregrine-basisfit` (cross-expert factorization, priced as rate–distortion on activations) |
| [Configuration](configuration.md) | the complete env-var reference — every tuning knob, none of which can change the token stream |
| [Performance tuning](performance-tuning.md) | "decode is slow, what do I check" — the levers in **measured** order, including the ones that look like levers and are not |
| [Model format & artifacts](model-format.md) | model directory layout, weight naming, QT quant formats, safetensors extensions, every artifact JSON |
| [Measurement discipline](measurement.md) | how to get a number that means something here: medians over single runs, duty cycles over thread-summed counters, the page-cache trap, **the byte ledger** (11.3 GB/token is not one number) |
| [Benchmarks](benchmarks.md) | headline numbers + how to reproduce; summary of the full [peregrine-vs-colibri study](peregrine-vs-colibri.md) |
| [Validation runbook](validation-runbook.md) | what this workspace could not measure, and the procedure for settling it on a GPU box with the real checkpoint |

## How it works

| Page | What's in it |
|---|---|
| [Architecture](architecture.md) | the phased-vs-concurrent thesis, workspace map, memory hierarchy, the five layers of concurrency, design invariants |
| [The concurrent 3-lane scheduler](concurrent-scheduler.md) | task model, the GPU/CPU/I-O lanes, deterministic accumulation, dispatch-order shaping |
| [Adaptive runtime](adaptive-runtime.md) | lane telemetry, bubble tuner & lane balancer, IoTuner, sensor governors, learned schedulers, cross-session persistence |
| [Prefetch & caching](prefetch-and-caching.md) | the prediction spine, two-tier speculation, the warm RAM cache, GPU residency |
| [I/O & storage](io-and-storage.md) | the io_uring reactor, O_DIRECT lane, slab pool, zstd, hugepages, NUMA, topology probe, perf counters, **per-read latency distribution** and device-geometry alignment |
| [GPU / CUDA lane](gpu-cuda.md) | building with `cuda`, runtime gates, pinned staging & graphs, autotuning, what still needs hardware |
| [Tokenizer](tokenizer.md) | the vendored gigatoken BPE fast path: what's vendored, what's dropped, parity gates |

## Project

| Page | What's in it |
|---|---|
| [Testing & quality gates](testing-and-quality.md) | the 604-test suite, bit-identity philosophy, the bad-patterns audit, contribution ground rules |
| [Bad-pattern catalogue](BAD_PATTERNS.md) | the panic-vector / UB audit gate in detail |
| [Roadmap & status](roadmap.md) | completion dashboard, what shipped in which wave, the hardware-gated remainder |
| [Scale-out designs](scale-out-design.md) | the five items needing a second GPU, a second host, or a GDS stack — what each would be, and the `file:line` seam it hooks into |
| [peregrine vs colibrì](peregrine-vs-colibri.md) | the full same-hardware engineering study on the real GLM-5.2 744B model |
| [External audit response](external-audit-response.md) | issue #6's 21-section review, triaged against what has since been measured — including the Tier 1 items that are already shipped or already rejected |

## Faster decoding: designs and open ideas

Nineteen pages of design notes on breaking strict one-token-at-a-time
generation and on moving fewer bytes per token. Most are proposals, not shipped
work — [Speculative decoding alternatives](speculative-decoding-alternatives.md)
is the index of the first group and carries the per-approach status, the
economics that decide it, and the closures for approaches this engine cannot
take.

| Page | What's in it |
|---|---|
| [Speculative decoding alternatives](speculative-decoding-alternatives.md) | **start here** — every parallel-decoding approach scored against the two serving tracks, with status, priority order, and what is closed and why |
| [Blockwise decoding](blockwise-decoding.md) · [Medusa / EAGLE](medusa-eagle.md) · [Speculative routing](speculative-routing.md) | the individual sketches the index scores |
| [Expert-union future execution](expert-union-future-execution.md) · [Expert-level non-causal execution](expert-non-causal-execution.md) · [Adaptive causal](adaptive-causal.md) · [Causal inversion](causal-inversion.md) · [Backward routing](backward-routing.md) | executing several futures from one expert load |
| [Expert decomposition](expert-decomposition.md) · [Token-equivalence adaptive precision](token-equivalence-adaptive-precision.md) · [Residual algebra](residual-algebra.md) · [Speed vs bytes](speed-vs-bytes.md) | moving fewer bytes at the representation level |
| [Expert address prediction](expert-address-prediction.md) · [Compute before read](compute-before-read.md) · [Geometric cache](geometric-cache.md) · [Physical checkpoint](physical-checkpoint.md) | prediction, staging and layout |
| [Three most promising ideas](three-most-promising-ideas.md) · [Advanced optimization directions](advanced-optimization-directions.md) | the two summary pages |

## Long-form working documents

These are the large, append-only records rather than reference pages. They moved
here from the repository root on 2026-08-21; only `README.md` still lives at the
top level.

| Page | What's in it |
|---|---|
| [`todo.md`](todo.md) | the audited per-item roadmap — the primary record, cited from ~30 files |
| [`DESIGN.md`](DESIGN.md) | the original design document |
| [`todo.txt`](todo.txt) | the older plain-text predecessor of `todo.md`, kept for its early reasoning |
| [`ideas-tokens-per-sec-2026-08-15.md`](ideas-tokens-per-sec-2026-08-15.md) | ranked throughput ideas, and the **closed negatives — do NOT re-propose** list |
| [`ideas-from-colibri.md`](ideas-from-colibri.md) | what was worth taking from the predecessor engine |

Elsewhere: [`README.md`](../README.md) (project overview) ·
[`crates/peregrine-token/README.md`](../crates/peregrine-token/README.md)
(re-vendoring notes).

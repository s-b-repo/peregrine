# peregrine documentation

Documentation wiki for **peregrine** — a from-scratch Rust MoE inference
engine that drives CPU, GPU, RAM, and SSD concurrently. New here? Start with
[Getting started](getting-started.md); for the big picture read
[Architecture](architecture.md).

## Using peregrine

| Page | What's in it |
|---|---|
| [Getting started](getting-started.md) | prerequisites, build & test, first run, serving, troubleshooting |
| [`peregrine` CLI reference](cli-peregrine.md) | every subcommand (`demo`, `build`, `bench`, `build-automaton`, `dump-routes`, `galactic`, `compile-plan`), the stdio serve protocol |
| [Serving (`peregrine-serve`)](serving.md) | CLI flags, the OpenAI-compatible HTTP API, SSE streaming, auth, priority header, continuous batching |
| [Layout tools](layout-tools.md) | `peregrine-layout-reorg`: routing traces → disk schedules → physical checkpoint rewrite |
| [Configuration](configuration.md) | the complete env-var reference — every tuning knob, none of which can change the token stream |
| [Model format & artifacts](model-format.md) | model directory layout, weight naming, QT quant formats, safetensors extensions, every artifact JSON |
| [Benchmarks](benchmarks.md) | headline numbers + how to reproduce; summary of the full [peregrine-vs-colibri study](peregrine-vs-colibri.md) |

## How it works

| Page | What's in it |
|---|---|
| [Architecture](architecture.md) | the phased-vs-concurrent thesis, workspace map, memory hierarchy, the five layers of concurrency, design invariants |
| [The concurrent 3-lane scheduler](concurrent-scheduler.md) | task model, the GPU/CPU/I-O lanes, deterministic accumulation, dispatch-order shaping |
| [Adaptive runtime](adaptive-runtime.md) | lane telemetry, bubble tuner & lane balancer, IoTuner, sensor governors, learned schedulers, cross-session persistence |
| [Prefetch & caching](prefetch-and-caching.md) | the prediction spine, two-tier speculation, the warm RAM cache, LFRU tiering, GPU residency |
| [I/O & storage](io-and-storage.md) | the io_uring reactor, O_DIRECT lane, slab pool, zstd, hugepages, NUMA, topology probe, perf counters |
| [GPU / CUDA lane](gpu-cuda.md) | building with `cuda`, runtime gates, pinned staging & graphs, autotuning, what still needs hardware |
| [Tokenizer](tokenizer.md) | the vendored gigatoken BPE fast path: what's vendored, what's dropped, parity gates |

## Project

| Page | What's in it |
|---|---|
| [Testing & quality gates](testing-and-quality.md) | the 282-test suite, bit-identity philosophy, the bad-patterns audit, contribution ground rules |
| [Bad-pattern catalogue](BAD_PATTERNS.md) | the panic-vector / UB audit gate in detail |
| [Roadmap & status](roadmap.md) | completion dashboard, what shipped in which wave, the hardware-gated remainder |
| [peregrine vs colibrì](peregrine-vs-colibri.md) | the full same-hardware engineering study on the real GLM-5.2 744B model |

Root-level references: [`README.md`](../README.md) (project overview) ·
[`DESIGN.md`](../DESIGN.md) (the original design document) ·
[`todo.md`](../todo.md) (the audited per-item roadmap) ·
[`crates/peregrine-token/README.md`](../crates/peregrine-token/README.md)
(re-vendoring notes).

# Research: advanced techniques to reduce streamed-MoE bytes/token and SSD stalls

Date: 2026-08-23 · Source: [upstream issue #8](https://github.com/s-b-repo/peregrine/issues/8)
(`enhancement`), opened by `s-b-repo`. Companion to
[issue #7](ideas-github-issue-7-2026-08-23.md) — this one is the refined,
experiment-shaped successor of #7's fifteen-item list. Archived verbatim below;
local status annotations are at the end.

---

## Summary

GLM-5.2 on a single consumer machine is dominated by streamed expert I/O. In my current deployment, decode is about **18.9 s/token (~0.053 tok/s)**, with roughly **11.1 GB moved per decode step**. The measured step is approximately 8.7 s I/O, 2.8 s CPU, and 0.016 s GPU.

The goal of this issue is to investigate optimization paths that are **not already implemented or already measured/rejected** in Peregrine, with emphasis on reducing physical bytes/token rather than only increasing nominal SSD throughput.

The ideas below are research candidates, not assumptions that they will work. Each should have correctness/quality gates and real-checkpoint measurements.

## High-priority candidates

### 1. Fine-grained sub-expert decomposition / loading

Investigate whether expert tensors can be decomposed into smaller independently useful regions so the runtime does not have to fetch an entire expert when only part of its computation can be serviced.

Potential forms:

- split expert matrices into independently scheduled blocks;
- schedule `W_gate`, `W_up`, `W_down` or smaller blocks independently where dependencies permit;
- cache/reuse sub-expert ranges rather than whole experts;
- deduplicate identical physical ranges requested by concurrent consumers.

Primary metric: **physical bytes read/token** and critical-path latency.

### 2. Training-free cache-aware rerouting / expert substitution

Investigate whether the runtime can preserve the authoritative GLM router for correctness while selecting a cached or resident functionally similar expert when an exact expert would otherwise require a large SSD fetch.

Possible policy:

```
exact expert resident -> use exact expert
exact expert absent -> evaluate substitute candidates
substitute sufficiently similar -> use resident substitute
otherwise -> fetch exact expert
```

Need a strict output-quality/flip-rate gate. The experiment should measure disk bytes saved versus output deviation.

### 3. Segment-level routing/cache planning

Current routing/cache logic is largely token/layer oriented. Investigate planning a short decode segment as a unit and selecting a cache working set that maximizes coverage for the whole segment.

For example:

- predict a short horizon of expert activations;
- build a segment expert union;
- optimize a bounded RAM/VRAM working set for the segment;
- avoid repeatedly evicting/reloading experts across adjacent tokens.

Measure segment cache hit rate, unique physical bytes/token, and end-to-end tok/s.

### 4. Adaptive per-expert precision tiers

Do **not** pursue global int3 as the solution. Instead investigate whether experts can exist in different representations depending on residency/importance:

- high-value resident experts: existing int4/full representation;
- cold SSD experts: more compact representation where quality permits;
- promotion/demotion between SSD/RAM/VRAM representations asynchronously.

The objective is to make **precision another memory-hierarchy tier**, while preserving a measurable output-quality bound.

### 5. Multi-layer activation prediction

Go beyond one-step router look-ahead. Investigate a lightweight predictor that can propose likely expert sets for several future layers from the current hidden state/prompt, so SSD requests can be scheduled substantially earlier.

The native router remains authoritative; predictions are advisory only.

Measure:

- recall@K;
- useful-prefetch fraction;
- bytes of stale/wasted prefetch;
- latency hidden;
- impact under saturated multi-SSD I/O.

### 6. Expert functional-equivalence clustering

Cluster experts based on **activation/output behavior**, not only weight-space distance. Build equivalence classes where a resident expert may safely approximate an absent expert under a configurable tolerance.

Potential runtime use:

- exact expert available -> exact;
- exact absent + equivalent resident expert -> substitute;
- otherwise fetch.

This is related to expert merging/substitution but should be evaluated as an online inference mechanism.

### 7. Cross-expert subspace/factorization on the real GLM-5.2 checkpoint

Peregrine already has `peregrine-basisfit`, so this issue is specifically about the still-missing **real-checkpoint validation**, not duplicating the tool.

Required experiment:

- activation-space rate/distortion;
- shuffled-expert control;
- resident basis charged continuously while resident;
- physical bytes/token ledger;
- output flip-rate / tolerance gate.

The key question is whether shared cross-expert structure can materially reduce the ~370 GB routed-expert footprint without simply moving the cost into a non-resident basis.

### 8. Learn the SSD itself / completion-time scheduling

Peregrine already has adaptive I/O tuning, multiple rings, direct I/O, etc. Investigate a device-aware model:

```
L = f(device, address, request_size, queue_depth, temperature, filesystem_state)
```

Then choose the expert read order that minimizes the **maximum completion time of the critical expert set**, rather than maximizing aggregate MB/s.

This should be especially interesting on heterogeneous multi-drive systems where one device is much slower than the others.

Potential implementation gate: `COLI_SSD_AWARE_SCHED`.

### 9. Joint expert-survival cache policy

Extend the existing co-activation tracker into cache eviction. Rather than scoring experts independently, estimate whether evicting A will soon force B to reload.

Candidate victim score should include something like:

```
future_use(A) * reload_cost(A)
+ joint_survival(A,B)
+ critical_path_cost
```

This is explicitly a different use of co-activation from prefetching: it changes **what survives eviction**.

Potential gate: `COLI_JOINT_EVICTION`.

### 10. Multi-future speculative expert DAG

Investigate several low-cost future trajectories instead of a single speculative future, while sharing the union of expert loads and common computation across branches.

For example:

```
             future A
            /
current ---- future B
            \
             future C
```

The key objective is to amortize one SSD read across multiple possible future states rather than independently fetching for each branch.

This must be heavily gated because speculative work can be a net loss on disk-bound MoE if rejected paths trigger additional reads.

## Additional niche candidates

- physical-range request deduplication across concurrent lanes/batches;
- sub-expert/range-aware caching rather than expert-granularity caching;
- transition-aware expert placement using P(expert_B | expert_A), not only static heat;
- cache-aware routing that explicitly prices current device/queue latency;
- output-sensitive expert pruning using a router score × output-sensitivity criterion;
- lightweight retrieval of historically similar activation patterns as a prefetch signal;
- dynamic precision promotion/demotion based on expert hotness and criticality;
- critical-path priority scoring that combines router probability, expert cost, cache state, device latency, and queue depth.

## Measurement requirements

Before evaluating any optimization, instrument the byte ledger separately for:

- requested expert bytes;
- unique bytes after batch/consumer union;
- cache-served bytes;
- successful prefetch bytes;
- stale/wasted prefetch bytes;
- re-read bytes after eviction;
- actual device bytes.

Also record p50/p95/p99 expert-read latency by device, queue depth, request size, and completion time on the critical path.

Every lossy technique should report output flip-rate/logit or token agreement alongside bytes saved.

## Explicit exclusions

This issue is **not** proposing another pass over already-covered areas such as:

- normal adaptive prefetch;
- existing router look-ahead;
- continuous batching;
- parallel `io_uring` rings;
- `O_DIRECT` / registered files;
- NUMA pinning;
- existing expert layout methods;
- CUDA graphs / fused reduce;
- int2-g64;
- persistent-kernel work already investigated;
- the previously measured CPU/GPU split-GEMM route;
- global int3 as a solution.

## Suggested first experiments

1. Device-aware critical-path SSD scheduling.
2. Physical-range/request deduplication.
3. Sub-expert/range-level caching.
4. Training-free cache-aware rerouting/substitution.
5. Segment-level cache planning.
6. Real-checkpoint factorization with the corrected byte ledger.

The primary success criterion should be **lower physical bytes/token and lower end-to-end decode latency**, not merely higher synthetic storage throughput.

---

## Local annotations (not part of the issue)

Where each candidate already has peregrine prior art or a recorded closure:

| Candidate | Prior art in this repo |
|---|---|
| 1 sub-expert decomposition | [expert-decomposition.md](expert-decomposition.md), [speed-vs-bytes.md](speed-vs-bytes.md) |
| 4 adaptive precision tiers | [token-equivalence-adaptive-precision.md](token-equivalence-adaptive-precision.md); global int3/int2-g64 closures in [ideas-tokens-per-sec-2026-08-15.md](ideas-tokens-per-sec-2026-08-15.md) |
| 5 multi-layer prediction | **Shipped** as the multi-layer router look-ahead: `COLI_ROUTER_LOOKAHEAD_K` (default 1 = historical Δ=1) reaches k layers out with a geometrically halved window — `lookahead_horizon_widths`, bit-identity tested. Still open per the issue's metrics: recall@K / wasted-prefetch measurement on the real checkpoint |
| 6 equivalence clustering / 2 substitution | [token-equivalence-adaptive-precision.md](token-equivalence-adaptive-precision.md), [residual-algebra.md](residual-algebra.md) |
| 7 real-checkpoint factorization | `peregrine-basisfit` in [tools.md](tools.md); owed experiment tracked in [validation-runbook.md](validation-runbook.md) |
| 8 completion-time scheduling | **Shipped gated off** (`COLI_SSD_AWARE_SCHED=1`): per-device bandwidth EWMA + SJF in-group order + seconds-weighted ring homing — `ssdclock.rs`, bit-identity tested. Still open: queue-depth/address terms, cross-device ordering beyond device-pure groups |
| 9 joint eviction | co-activation tracker + `COLI_JOINT_EVICTION` proposal in [prefetch-and-caching.md](prefetch-and-caching.md) |
| 10 multi-future DAG | [expert-union-future-execution.md](expert-union-future-execution.md), [expert-non-causal-execution.md](expert-non-causal-execution.md) |
| byte-ledger instrumentation requirements | [measurement.md](measurement.md) |

Anything touching routing changes must clear the flip-rate gate; anything
touching precision must beat the recorded RTN-ladder failures. The closed
negatives list in [ideas-tokens-per-sec-2026-08-15.md](ideas-tokens-per-sec-2026-08-15.md)
applies before any candidate is scheduled.

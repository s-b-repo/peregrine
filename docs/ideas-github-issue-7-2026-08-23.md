# ideas — fifteen techniques not yet tried

Date: 2026-08-23 · Source: [upstream issue #7](https://github.com/s-b-repo/peregrine/issues/7),
opened by `s-b-repo`, no labels. The broader survey behind
[issue #8](ideas-github-issue-8-2026-08-23.md), which distils this list into
ten experiment-shaped candidates. Archived verbatim below; local status
annotations are at the end.

---

Excluding **int3** entirely, and excluding the techniques that Peregrine has already implemented or explicitly tested and rejected, there is still a meaningful set of deeper possibilities.

The important distinction is that several of the remaining ideas can attack your real bottleneck — **~11.3 GB of expert traffic per generated token** — rather than just making the existing I/O path slightly faster. Peregrine's own current roadmap says this is where the remaining gains are.

## Highest-priority ideas you have not already tried

| Rank | Technique | Peregrine status | Target |
|---|---|---|---|
| 1 | SSD completion-time scheduling | **Not implemented** | Reduce storage stall |
| 2 | Expert-request deduplication across layers/batch | **Not established as a standalone optimization** | Reduce bytes |
| 3 | Cross-token expert read reuse beyond normal cache | **Not established** | Reduce bytes |
| 4 | Joint co-activation eviction | **Proposal only** | Increase useful cache hits |
| 5 | Expert similarity / substitute routing | **Not implemented** | Reduce compulsory reads |
| 6 | Router fine-tuning for locality | **Not implemented** | Increase reuse structurally |
| 7 | Output-aware expert pruning | **Not implemented as a production path** | Reduce expert count |
| 8 | Expert merging / consolidation | **External tooling exists; not integrated** | Reduce model size |
| 9 | Cross-expert residual/factorization with real GLM checkpoint | **Not validated** | Reduce bytes dramatically |
| 10 | Multi-future speculative execution | **Proposal only** | Amortize expert loads |
| 11 | Learned lightweight next-layer predictor | **Not the same as existing statistical predictors** | Better prefetch |
| 12 | Cache-aware routing at runtime | **Not equivalent to current cache admission** | Reduce misses |
| 13 | Expert popularity snapshots / warm-start scheduling | Partially explored elsewhere, not equivalent to full Peregrine path | Cold-start latency |
| 14 | SSD queue-depth/latency feedback per device | **Not implemented as physical-device model** | Better multi-SSD utilization |
| 15 | Expert placement by *transition probability*, not just heat | **Not implemented** | Lower future misses |

### 1. SSD completion-time scheduling

Peregrine already has `io_uring`, multiple rings, direct I/O, adaptive worker counts and various ordering/layout mechanisms. What it does **not** have is a scheduler that learns:

```
(device, offset, size, queue depth) → expected completion time
```

and then chooses the order that minimizes the completion time of the **critical expert set**.

The project explicitly lists this as a proposal (`COLI_SSD_AWARE_SCHED`) and says the variable is not read by code.

This is especially interesting on your setup because:

```
sda   ~263 MB/s
sdb   ~249 MB/s
sdc    ~50 MB/s
sdd   ~257 MB/s
NVMe  ~277 MB/s
```

A global FIFO policy does not know enough about those differences.

### 2. Expert-request deduplication

This is different from ordinary caching.

Suppose a batch or concurrent execution asks for:

```
(layer 20, expert 73)
(layer 20, expert 73)
(layer 21, expert 73)
...
```

The engine can currently coalesce some work through batching/union behavior, but a more aggressive **request identity layer** can detect identical physical tensor ranges before they enter the I/O scheduler:

```
logical request
      ↓
physical range key
      ↓
one SSD read
      ↓
N consumers
```

That matters because your bottleneck is physical bytes, not logical expert invocations.

I did **not** find evidence that Peregrine has a generalized cross-lane physical-range request deduper.

### 3. Cross-token read reuse without relying on full expert retention

Normal caching asks:

> Is the whole expert resident?

A more aggressive mechanism is:

> Which *subranges* of an expert are likely to be reused?

For example:

```
expert 42
├── tensor A
├── tensor B
├── tensor C
└── tensor D
```

If some tensors/ranges have dramatically different reuse profiles, cache only the repeatedly used pieces.

This is potentially useful because Peregrine currently reasons primarily in expert-level units. I did not find a production sub-expert range cache in the current design.

This could be a substantial architectural change.

---

## 4. Joint co-activation eviction

This one is explicitly documented but **not implemented**.

Peregrine already records co-activation information. The proposed extension is to make eviction aware of relationships such as:

```
A → B happens frequently
```

rather than scoring A and B independently.

The repository explicitly proposes `COLI_JOINT_EVICTION=1` and says no code reads it.

Your measured routing overlap of **33.55% between consecutive tokens** gives this a real signal to exploit.

---

# 5. Expert similarity / substitute routing

This is much more radical.

Instead of insisting:

```
router → expert 137
```

you can construct a set of functionally similar experts:

```
expert 137
   ↙    ↓    ↘
136   141   198
```

and choose a local/resident substitute when the exact expert is expensive to fetch.

A recent 2026 paper, **OrderMoE**, explores similarity-based expert allocation and quality-aware substitute selection specifically to reduce remote expert invocation. ([arXiv](https://arxiv.org/abs/2607.17154 "OrderMoE: An expert similarity driven distributed edge MoE inference"))

For Peregrine, the attractive version would be:

```
exact expert available?
      │
   yes ──→ use it
      │
      no
      ↓
similar resident expert?
      │
   confidence high
      ↓
use substitute
```

This necessarily trades exactness for quality unless carefully constrained, so it belongs behind a quality gate.

But unlike faster I/O, it can potentially eliminate the read entirely.

---

# 6. Router fine-tuning for locality

This is one of the biggest things I found that is **not a runtime optimization already present in Peregrine**.

**ReMoE** modifies the router so that it prefers recently used experts, increasing temporal reuse without adding inference-time computation. The paper reports improved expert reuse and substantial decode gains under memory-constrained serving. ([arXiv](https://arxiv.org/abs/2605.27081 "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"))

Conceptually:

```
normal router:

token → A B C D

locality-aware router:

token → A A C D
          ↑
      recently hot
```

The important advantage is that the model itself starts producing a more cache-friendly workload.

Your current engine can optimize around the routing decisions, but it cannot make the GLM-5.2 router intrinsically more locality-friendly without changing/fine-tuning the model.

This is potentially much larger than another cache heuristic.

---

# 7. Output-aware expert pruning

Rather than trusting router probabilities alone:

```
router score = importance
```

estimate:

```
router score × actual output sensitivity
```

and avoid experts whose contribution is negligible.

This is related to the research distinction between **router importance** and **output sensitivity**. Peregrine currently has a document exploring a backward-routing/Jacobian approach, but that is a design document rather than a shipped execution path.

That means there is a potentially interesting experiment:

```
Top-8
 ↓
sensitivity estimate
 ↓
can 1 low-impact expert be skipped?
 ↓
Top-7 effective execution
```

A 1-of-8 reduction would theoretically remove ~12.5% of expert traffic for affected layers.

It must be gated by exact-output/quality tests.

---

# 8. Expert merging / consolidation

There is now external work implementing **REAM**, which merges less-important experts into protected centroids using router and output similarity. ([GitHub](https://github.com/puwaer/moe-expert-compress "puwaer/moe-expert-compress"))

This is different from runtime caching.

Instead of storing:

```
256 experts
```

you could potentially have:

```
256 logical experts
        ↓
shared/merged physical representation
```

and therefore reduce the disk footprint.

This is especially relevant to your workload because shrinking the checkpoint changes the fundamental bandwidth problem.

Peregrine itself has not integrated REAM-style expert surgery.

---

# 9. Cross-expert factorization

Peregrine already started investigating this with `peregrine-basisfit`, so I would **not** call the general idea new to Peregrine.

However, the important part is that the repository says the real-checkpoint experiment is still owed, and its own initial control exposed false positives.

So the thing you have **not actually done yet** is:

```
GLM-5.2 real checkpoint
        ↓
cross-expert basis
        ↓
activation-space rate/distortion
        ↓
byte-accurate cost
        ↓
flip-rate gate
```

That could potentially be much more powerful than quantizing from 4-bit to 3-bit.

---

# 10. Multi-future execution

The repository's own advanced-optimization document explicitly labels this as a proposal.

Instead of:

```
current → predicted next token
```

you maintain:

```
             ┌── future A
current ─────┼── future B
             └── future C
```

and share common expert loads among the branches.

The interesting part is not the speculation itself.

It's:

**load the union once.**

For example:

```
future A: {1,4,7,9}
future B: {1,4,8,9}
future C: {1,4,7,10}

union = {1,4,7,8,9,10}
```

rather than three completely separate fetches.

The repository explicitly describes this as not implemented.

---

# 11. Learned lightweight prefetch model

Peregrine already has several statistical predictors, so merely adding "another predictor" isn't interesting.

A different approach is the current **SpecPrefetch** direction: use a small learned adapter specifically for next-layer expert transfer while leaving the native router authoritative. This separates prediction from execution correctness. ([arXiv](https://arxiv.org/abs/2607.24787))

Possible architecture:

```
hidden state
    │
    ├── native router ───────→ authoritative experts
    │
    └── tiny predictor ──────→ speculative SSD fetches
```

That is a useful distinction for Peregrine.

---

# 12. Cache-aware routing

Another line of work directly changes routing based on what is cached.

Research on cache-conditional MoE reports large gains by making the router aware of expert locality rather than treating routing and caching as independent problems. ([arXiv](https://arxiv.org/abs/2412.00099 "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference"))

This differs from Peregrine's existing cache-admission mechanisms:

```
current:
router → cache reacts

new:
router ↔ cache co-optimize
```

That is potentially much more powerful.

The recent ReMoE results reinforce that locality-aware routing can materially improve offloaded inference. ([arXiv](https://arxiv.org/abs/2605.27081 "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"))

---

# 13. Learned cache replacement

Peregrine has sophisticated cache policies, but I did not find evidence of the **ML-based cache replacement** approach used in FlashMoE.

FlashMoE specifically targets SSD-backed MoE and reports an ML-based policy combining recency and frequency, with substantial cache-hit improvements over ordinary policies. ([arXiv](https://arxiv.org/abs/2601.17063))

The distinction is:

```
Peregrine:
hand-designed adaptive heuristic

FlashMoE-style:
learn replacement behavior from traces
```

This is worth benchmarking because your workload is explicitly SSD-bound.

---

# 14. Full device-specific queue model

This is related to #1 but more specialized.

Instead of asking:

> How many I/O workers should I have?

ask:

> What queue depth is optimal for each physical device for this exact expert-request distribution?

For example:

```
sda   QD 16
sdb   QD 32
sdc   QD 4
sdd   QD 32
nvme  QD 64
```

and dynamically change those based on latency percentiles and contention.

Peregrine has adaptive I/O tuning, but I found no evidence that it models each physical device at this level.

Your unusually slow `sdc` makes this particularly interesting.

---

# 15. Transition-aware expert placement

Peregrine already has heat/layout mechanisms.

A deeper placement objective is:

```
P(expert_B | expert_A)
```

rather than:

```
P(expert_B)
```

Then place frequently successive experts so their physical addresses minimize the storage completion cost.

For example:

```
A → B → C
```

could be physically arranged:

```
[ A ][ B ][ C ]
```

rather than scattered over different regions/drives.

This is more sophisticated than static popularity placement.

I did not find this exact transition-cost placement implemented.

---

# Things I found that are **already accounted for**

I would not waste your time reimplementing these:

- normal look-ahead prefetch
- adaptive prefetch
- continuous batching
- multi-ring `io_uring`
- `O_DIRECT`
- registered files
- NUMA pinning
- Louvain/spectral/Hilbert-style layout work
- adaptive I/O worker counts
- ordinary cache admission
- CUDA graphs
- GPU-side fused reduction
- MTP plumbing
- route-history persistence
- zstd
- int2-g64
- CUDA allocation defragmentation experiment
- CPU/GPU split-GEMM experiment
- persistent-kernel investigation

Those are already documented as shipped, measured, or explicitly rejected.

## The really interesting discovery

The biggest opportunity I see is **not another SSD optimization**.

It's changing the problem from:

```
600 experts
×
~18.9 MB
≈
11.3 GB/token
```

to:

```
fewer physical expert bytes/token
```

There are four fundamentally different ways to do that:

```
1. Don't fetch some experts
   → output-aware pruning

2. Fetch a cheaper substitute
   → similarity-aware routing

3. Store fewer physical parameters
   → merging/factorization

4. Make future routes naturally cacheable
   → locality-aware router fine-tuning
```

That is much deeper than another prefetch heuristic.

And there is evidence the field is moving in exactly this direction: recent work is exploring router locality, expert similarity, expert merging, adaptive expert prediction, and SSD-specific learned caching rather than only faster storage. ([arXiv](https://arxiv.org/abs/2605.27081 "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"))

One especially important warning: **don't assume speculative decoding is automatically useful here.** SpecMoEOff reports large gains in other offloaded-MoE settings, but Peregrine's own GLM-5.2 measurements have already shown that naïve MTP can become a net loss when every rejected verification path causes another huge expert read. ([arXiv](https://arxiv.org/abs/2508.21706))

So, for **your exact machine**, my deeper shortlist is:

**A. SSD completion scheduling**

**B. Physical request deduplication**

**C. Sub-expert/range caching**

**D. Learned cache replacement**

**E. Cache-aware routing**

**F. Router locality fine-tuning**

**G. Output-sensitive expert pruning**

**H. Expert substitution by similarity**

**I. Expert merging**

**J. Real-checkpoint cross-expert factorization**

**K. Lightweight learned prefetch predictor**

**L. Transition-aware physical expert placement**

Those are the directions I'd investigate before touching another kernel optimization.

---

## Local annotations (not part of the issue)

Corrections against repo state at archive time, per house rules
([measurement.md](measurement.md), closed negatives in
[ideas-tokens-per-sec-2026-08-15.md](ideas-tokens-per-sec-2026-08-15.md)):

- **Speculative decoding warning (#10 / SpecMoEOff):** correct and already
  encoded as the standing rule — speculation must reduce total bytes/token;
  `COLI_SPEC_CONF` + `COLI_DRAFT=5` measured +37 % tok/s and −22 % disk reads,
  while `COLI_DRAFT=4` without confidence gating was a recorded negative.
  See [speculative-decoding-alternatives.md](speculative-decoding-alternatives.md).
- **"Bigger warm cache":** the issue's rank-3 "cross-token reuse beyond normal
  cache" must not drift into re-proposing a bigger pool — that exact closure
  (one token routes ~11 GB vs a ~363 GB pool) stands unless the budget reaches
  a whole pass.
- Prior art pointers: sub-range/sub-expert reuse → [expert-decomposition.md](expert-decomposition.md);
  precision tiers → [token-equivalence-adaptive-precision.md](token-equivalence-adaptive-precision.md);
  multi-future union → [expert-union-future-execution.md](expert-union-future-execution.md);
  device/queue modelling → [io-and-storage.md](io-and-storage.md) and the
  device-pure-claims seed in [ideas-tokens-per-sec-2026-08-15.md](ideas-tokens-per-sec-2026-08-15.md);
  basisfit real-checkpoint debt → [validation-runbook.md](validation-runbook.md).

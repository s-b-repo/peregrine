# Advanced Optimization Directions

## 14. Reverse cache eviction

Traditional systems ask:

> Which cached item is least useful?

Instead estimate:

```
P(future use) × C(reload)
```

and evict the item with minimum expected future cost.

But go one level further.

If loading expert A will inevitably lead to loading B shortly afterward, retain them
as a coupled pair even if B has a low individual probability.

That produces a **joint cache policy**:

```
P(A, B)
```

rather than:

```
P(A) + P(B)
```

This could interact extremely well with the measured routing overlap in Peregrine
(33.55% consecutive overlap). The co-activation tracker in `predict.rs` already
records pair co-firing rates — this extends it from "prefetch these together" to
"evict these together".

### Implementation

1. Extend `CoActivation` (predict.rs) to track eviction-time survivorship: when
   expert A is evicted, record whether B was loaded shortly after.
2. Compute the joint survival probability P(A,B) vs P(A) + P(B) from historical data.
3. Under `COLI_CACHE_LFRU` or `COLI_CACHE_COMPRESS`, use the joint probability as
   the victim score instead of independent recency+heat.
4. Gate behind a new `COLI_JOINT_EVICTION=1` flag.

This is correctness-neutral — it only changes eviction order, and the reduce is
position-keyed.

## 15. Branchless multi-future execution

Instead of generating one speculative future:

```
A → B
```

construct several low-cost state trajectories:

```
A
├── B
├── C
└── D
```

But don't fully execute each — **share their common computation**.

This creates a **computation DAG of futures**. For branches (B, C, D), shared experts
and shared attention operations execute once. Only divergent portions are duplicated.

That could make speculative inference much cheaper than naïvely running multiple branches.

### Relationship to existing ideas

- **Speculative routing** (docs/speculative-routing.md): predicts one future route,
  drafts, verifies. Multi-future executes several draft branches with shared compute.
- **Expert-union future-token execution** (TIER 1 #3): loads the union of predicted
  experts once for multiple token positions. Multi-future generalizes this to
  multiple *branches* of execution, sharing the union across them.
- **Time-shifted inference** (opencode-prompt.txt): speculative state reservoir
  with one branch pre-computed. Multi-future pre-computes several.

### Implementation sketch

1. From the current hidden state h_t, use the router to predict the top-K likely
   next expert sets (already available via `router_ranks_for`).
2. For each candidate expert set, compute the shared prefix of the forward pass
   (attention up to the MoE layer, shared attention output).
3. Only diverge at the MoE layer — compute all K candidates' expert outputs in
   one batched `expert_group` call (the union), then split into K paths.
4. Verify against the full model when the actual token arrives; non-matching
   branches are discarded.

## 16. Learn the SSD itself

Peregrine already adapts its computation to telemetry. The next level would model
storage latency as a function:

```
L = f(address, queue_depth, request_size, temperature, filesystem, state)
```

Then the scheduler asks:

> Given 500 candidate expert reads, which ordering minimizes completion time?

This is a scheduling problem over the SSD's physical behavior.

Instead of maximizing nominal MB/s, optimize:

```
min  max_i  T_i
```

for the completion time of all required experts.

That could matter especially when expert requests have very different sizes —
a small expert read ahead of a large one may let compute start earlier, even if
the large one takes longer overall.

### Implementation sketch

1. Instrument each `read_experts_batched` call with per-expert latency.
2. Train a latency model L(address, size, queue_depth) from historical data.
3. Under `COLI_IO_ENGINE=uring`, replace the FIFO claim order with a
   latency-minimizing schedule: use the model to order the claim window so
   that completion times are minimized (shortest-job-first weighted by overlap).
4. Gate behind `COLI_SSD_AWARE_SCHED=1`.

This is correctness-neutral — completion order doesn't affect the position-keyed
reduce.

# Three Most Promising Ideas

If I were trying to push Peregrine beyond its current design, I would start with these:

## 1. Residual-space expert elimination

Don't ask:

> "Which experts did the router select?"

Ask:

> "Which selected experts contribute genuinely new information to the residual?"

Mathematically:

```
|(I - P_S) g_i E_i(x)|
```

becomes the selection criterion.

Where P_S is the projection onto the span of already-selected experts' outputs.
An expert whose output lies mostly within the span of others contributes little
new information and can be skipped.

This is a refinement of the existing **residual-space routing** idea — but more
aggressive. Instead of just pruning experts whose output is linearly dependent
(soft pruning with flip-gate validation), this computes the actual residual
contribution after loading and can drop experts mid-forward.

Potentially enormous reduction in expert computation and I/O.

### Relationship to existing code

- `peregrine-router` already computes gate weights g_i.
- The span projection P_S can be computed incrementally as experts are loaded.
- The flip-rate gate (`Model::prediction_flip_rate`) measures whether this changes
  outputs — the quality safety net already exists.

## 2. Cross-expert factorization

Search for:

```
W_i = B + Δ_i
```

or:

```
W_i = Σ_k a_{ik} B_k
```

Put B (shared basis) resident and stream only Δ_i (small per-expert delta).

This is the detailed treatment already in [expert-decomposition.md](expert-decomposition.md).
It attacks Peregrine's fundamental 11.3 GB/token bandwidth problem at the
**representation level** — not the scheduler, not the cache, but the actual
weight format.

The int3-g64 failure (0.447 flip rate, data-free quantization) is NOT evidence
against this — it used data-free rounding. Adaptive precision (see
[token-equivalence-adaptive-precision.md](token-equivalence-adaptive-precision.md))
uses a calibration signal, which is the standard approach that survived in llama.cpp.

## 3. Future-state computation

Don't speculate tokens.

**Predict future hidden states + future expert addresses**, preconstruct them,
and discard incorrect futures.

This is much closer to Peregrine's real bottleneck than ordinary speculative decoding.
The bottleneck is not "compute is slow" — it's "11.3 GB of expert data moves per token".
Speculative decoding doesn't reduce that; it multiplies it (each draft row streams
its own expert union). Future-state computation pre-stages the expert reads *before*
the routing decision, converting latency into bandwidth utilization.

### Three sub-components

1. **Future expert address prediction** (already in [expert-address-prediction.md](expert-address-prediction.md)):
   predict the byte ranges that will be needed, issue async reads early.

2. **Future hidden state prediction** (related to [causal-inversion.md](causal-inversion.md)):
   predict Ŝ_{t+k} = f(S_t) for expert IDs, magnitudes, and KV structure.

3. **State reservoir** (related to time-shifted inference in opencode-prompt.txt):
   maintain several future state candidates in parallel, pre-compute shared
   portions, discard incorrect branches at low cost.

## The fundamental insight

The current Peregrine architecture asks:

> How can we move 11.3 GB faster?

The more radical question is:

> Why does inference need to move 11.3 GB at all?

That changes the optimization target from:

```
faster inference
```

to:

```
┌─────────────────────────────────────────────┐
│  minimum information movement required      │
│  for a given output fidelity               │
└─────────────────────────────────────────────┘
```

That is where the largest discontinuous improvement lies — not in another 5–20%
scheduler optimization, but in changing what gets moved.

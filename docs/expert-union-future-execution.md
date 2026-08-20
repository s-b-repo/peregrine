# Expert-Union Future-Token Execution

## Core idea

Instead of loading experts per-token:

```
Token 1 → experts {A B C D}  → read A B C D
Token 2 → experts {A B C E}  → read A B C E
Token 3 → experts {A B D E}  → read A B D E
Token 4 → experts {A C D E}  → read A C D E
```

Predict the union of all future experts and load them once:

```
union → {A B C D E}  → read A B C D E  (once)
```

Then compute all candidate positions from that single load.

## Why this matters

Peregrine's bottleneck is 11.3 GB/token of expert data. For 4 speculative tokens,
naive execution reads 4× the expert data. Expert-union execution reads 1× the union.

With 33.55% consecutive-token overlap, the union of 4 tokens is roughly:
- 4×8 = 32 raw expert loads
- minus overlaps ≈ 32 - 4×(2.68 repeats) ≈ 21 unique experts
- vs 4×8 = 32 without sharing
= ~34% reduction in expert I/O for the speculative branch

## Relationship to existing code

- **Speculative routing** (docs/speculative-routing.md): predicts next N token routes.
  Expert-union is what you do with those predictions — load the union, not each independently.
- **Batch=16 already shows this**: B=16 jumps from 0.064 → 0.280 tok/s because routed
  experts are shared across sequences. Expert-union extends the same principle to future
  tokens of one sequence.
- **Geometric cache** (docs/geometric-cache.md): caches by state similarity. If two
  states are geometrically close, they likely share expert sets — prime targets for
  union execution.

## Implementation plan

1. **Router prediction** (peregrine-router): extend `router_ranks_for` to predict top-K
   expert sets for K future positions, not just one.

2. **Union computation**: compute the set-union of predicted expert IDs across K positions.

3. **Preload**: issue a single batched SSD read for the union (already exists in
   `read_experts_batched`, just needs to be called with the larger set).

4. **Multi-position execution**: compute all K speculative positions using the union,
   but only execute the divergent expert portions separately. Shared experts compute
   once.

5. **Verification**: verify all K positions, discard incorrect ones. Use existing
   flip-rate gate to validate quality.

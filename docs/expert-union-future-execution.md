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

## Shipped path

The optional runtime path is `COLI_FUTURE_PREFETCH=history|learned|both`. It runs after
speculative tokens are proposed and before their verify forward:

1. The pending token and every draft path are projected through route history,
   a learned token-context table, or both.
2. Candidate experts are unioned and ranked per layer.
3. The deduplicated union is submitted in bounded deep batches on the existing
   prefetch rings (`COLI_PREFETCH_BATCH`).
4. The normal verify forward still owns routing, computation, acceptance, and KV
   commit; speculative reads can only help or be wasted.

The learned artifact is built with:

```bash
peregrine build-router-predictor /path/to/model 4096 --text corpus.txt
```

The default remains off until a workload-specific table and bandwidth measurement
justify enabling it. See [configuration](configuration.md#coli_future_prefetch).

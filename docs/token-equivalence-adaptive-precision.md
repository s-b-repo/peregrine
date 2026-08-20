# Token Equivalence Classes & Adaptive Precision

## Token equivalence classes

Another unusual direction is identifying states where exact continuation computation can be reused.

Suppose two hidden states satisfy:

```
|h_a - h_b| < ε
```

and their downstream transformation remains within a bounded tolerance.

Then their expensive expert computation might be reusable.

The engine could maintain:

```
state fingerprint
      ↓
previously computed continuation
```

This becomes a form of **semantic memoization** of inference computation, not token caching.

The challenge is finding a mathematically safe equivalence criterion — a fingerprint
that guarantees the downstream expert outputs are within acceptable tolerance.

### Relationship to existing ideas

- **Geometric cache** (docs/geometric-cache.md): caches by state distance, but only
  skips routing. Token equivalence classes could skip the entire expert compute.
- **Time-shifted inference** (opencode-prompt.txt): speculative state reservoir. Token
  equivalence is the cheap variant — if two states are equivalent, one computation suffices.
- **Compute-before-read** (docs/compute-before-read.md): if a state's continuation is
  already computed, no reads are needed at all.

### Implementation sketch

1. Define a fingerprint function f(h) → fingerprint that is cheap to compute.
2. Maintain a map: fingerprint → (expert outputs, residual).
3. At each decode step, compute f(h_t). If a match exists within ε, reuse the cached
   continuation. Otherwise compute fresh and cache.
4. The equivalence ε must be validated against output flip rate.

This is correctness-neutral if the tolerance bound holds — the reused continuation is
within ε of what would be computed fresh.

## Adaptive precision via uncertainty propagation

Instead of:

```
hot expert = INT4
cold expert = INT2
```

use error propagation.

For each operation estimate:

```
δy ≈ J · δx
```

where J is the local Jacobian (sensitivity).

Then assign precision to the computation according to how much its numerical error
affects the final logits.

### Precision assignment

```
high sensitivity → high precision
low sensitivity → lower precision
```

This could produce a radically nonuniform model — not just per expert, potentially:

**per tensor × per token × per route**

### Implications

- An expert that is numerically sensitive to a particular input only needs high precision
  for that input; for other inputs it can safely use lower precision.
- The precision decision is per-computation, not per-parameter. A single expert matrix
  could be int8 for some tokens and int2 for others, depending on the local sensitivity.
- The Jacobian J can be estimated cheaply via the existing `COLI_PERF_COUNTERS` LLC-miss
  infrastructure or via backward Jacobian-vector products (see backward-routing.md).

### Priority

High — this directly addresses the 11.3 GB/token problem by reducing the bytes that
actually need to be moved at full precision. The int3-g64 failure (0.447 flip rate)
used data-free quantization; adaptive precision uses a calibration signal, which is
the standard approach that survived in llama.cpp (IQ4_XS, IQ3 survived because of
importance weighting).

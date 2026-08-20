# Adaptive Causal / Non-Causal Execution

There is an even more interesting hybrid: adaptive causal/non-causal execution.

## Concept

```
easy region
     ↓
parallel decoding   (8 tokens at once)

hard region
     ↓
normal autoregressive decoding (1 token at a time)
```

The engine dynamically determines whether a token/block is predictable enough
for parallel generation:

- easy → 8 tokens at once
- medium → 4 tokens at once
- hard → 1 token

## Combining with Peregrine's routing statistics

Blocks where expert overlap is high (33.55% consecutive overlap) are likely
predictable and suitable for parallel generation; blocks with low overlap are
harder and need step-by-step generation.

## Implementation

Use the router's confidence scores or entropy to gate between parallel-block
decoding and token-by-token decoding. High-confidence blocks (router entropy <
threshold) use blockwise parallel decoding; low-confidence blocks fall back to
sequential.

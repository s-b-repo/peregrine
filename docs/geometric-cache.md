# Geometric Expert Cache

## Problem

LRU asks: What was used recently?

But inference states are not arbitrary.

## Key idea

If two hidden states are similar: |x_t - x_{t-1}| << ε
their expert behavior may also be similar.

Therefore cache decisions could use:
  d(x_t, x_cached)
instead of time alone.

## Geometric caching

You could maintain:
  hidden-state region → predicted expert set

and cache expert sets associated with regions of state space.

The cache becomes a map:
  X → E
rather than an ordinary LRU.

## Implementation

1. Embeds each hidden state x_t into a low-dimension space (PCA or hashing)
2. Computes distance d(x_t, x_cached) to cached states
3. If distance < threshold, reuse the cached expert set (skip routing)
4. Uses Peregrine's 33.55% consecutive overlap to seed initial cache entries

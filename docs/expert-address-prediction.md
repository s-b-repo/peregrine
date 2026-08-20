# Expert-Address Prediction

## Problem

An enormous hidden cost is not just reading expert weights.
It is determining which disk locations need to be read.

## Method

Suppose: f(x_t) → e_{t+1}

Then instead of waiting for routing to finish, the engine predicts future
storage addresses.

That means the storage subsystem could begin reading the likely future expert
regions before the router finishes.

Not merely:
  "prefetch expert 17."

But:
  "prefetch byte ranges 18.3–20.7 MB, 61.2–79.1 MB, ... because those are the
  likely future execution addresses."

## Key insight

The prediction target is therefore the physical I/O graph.

## Implementation

1. Maps each expert ID to its byte offset range in the GGUF/flat file
2. Uses the router's distribution p(e|x) to predict the top-K byte ranges
3. Issues async pread() calls for those byte ranges before routing completes
4. Uses Peregrine's 33.55% consecutive overlap to predict byte-range reuse

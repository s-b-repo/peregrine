# Compute-Before-Read Scheduling

## Problem

Peregrine currently thinks largely in terms of:

```
read → compute
```

But some computations do not depend on the cold expert weights.

## Solution

So schedule:

```
known state
 ├── compute everything independent of cold experts
 ├── predict likely future state
 ├── construct memory addresses
 └── then fetch only what's still necessary
```

## Dependency graph scheduling

For a sufficiently large model, the scheduler could turn the forward pass into
a dependency graph instead of a layer-by-layer pipeline. Rather than:

```
Layer 1 → Layer 2 → Layer 3 → ...
```

you execute any mathematically available operation.

This is closer to critical-path scheduling over the actual tensor dependency
graph. The SSD is then just another asynchronous node in that graph.

## Implementation

1. Identifies operations that only depend on already-loaded weights
2. Executes those first (compute-bound, GPU-friendly)
3. Predicts and pre-fetches remaining weights for downstream ops
4. Schedules SSD reads as async I/O that completes while compute runs

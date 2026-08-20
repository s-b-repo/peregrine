# Expert Decomposition: Shared Basis + Small Deltas

## Core idea

Split experts into reusable mathematical components.

Instead of storing:

```
expert = W_up + W_gate + W_down
```

independently for every expert, search for decomposition:

```
W_i = B + Δ_i
```

or:

```
W_i = Σ_k a_{ik} B_k
```

Then:

```
shared basis → resident
small expert delta → SSD
```

## Impact

This could massively change the bandwidth equation.

For example, conceptually:

```
11.3 GB/token
    ↓
shared basis resident
    +
small per-expert deltas
    ↓
perhaps dramatically less SSD traffic
```

## Why this is different from ordinary quantization

The goal is **cross-expert compression**, not intra-expert compression. Ordinary quantization
makes a single expert smaller. Expert decomposition makes a *group* of experts smaller *together*
because they share a basis.

## Relationship to existing Tier 1 targets

- **TIER 1 #11: Expert fusion (store shared basis B_i, stream small deltas)** — this doc is the
  detailed treatment of that item.
- **Algebraic expert cancellation** (see opencode-prompt.txt) — the shared basis approach is a
  more general form: instead of requiring experts to be linearly dependent *at compute time*,
  we restructure storage so they are *already* decomposed at load time.
- **INT3 quantization** — decomposition + int3 could stack: the shared basis is the large
  residual matrix, per-expert deltas are small enough that even int3 quantizes them well.

## Implementation plan

1. **Cluster experts by weight-space similarity**: compute pairwise distances between experts
   (e.g. Frobenius norm of W_i - W_j) and cluster into groups of 8-32 experts that share structure.

2. **Compute shared basis per cluster**: for each cluster, compute the best-fit shared basis
   B_k (e.g. via SVD or low-rank approximation of the stacked expert matrices).

3. **Store (basis, deltas)**: write the shared basis to VRAM-resident storage and the small
   per-expert deltas to SSD-streamed storage. The delta for expert i is W_i - Σ_k a_{ik} B_k.

4. **Runtime reconstruction**: at inference, load the shared basis once (resident) and stream
   only the small deltas. Compute: output = B · Σ_k a_{ik} + Δ_i.

## Key questions to investigate

- What is the compression ratio? If a 4096×1536 expert matrix (6.3 MB at f16) decomposes into
  a 512×1536 shared basis (1.6 MB) + a 4096×512 delta (2.1 MB), that's 3.7 MB vs 6.3 MB = 41% reduction.
  Across 11.3 GB/token, even a 40% reduction is transformative.

- Does it survive the flip-rate gate? Decomposition changes weight values, so it must pass
  `Model::prediction_flip_rate` against the unmodified checkpoint.

- What's the compute overhead? The reconstruction adds a matrix add per expert, but the
  compute is on the CPU lane (no GPU needed) and the weight loading is dramatically reduced.

## Priority

This is a **TIER 1** optimization — it attacks the fundamental 11.3 GB/token bottleneck directly
rather than optimizing the I/O path around it. However, it requires model conversion tooling
(`peregrine-requantize` changes) and must pass the flip-rate quality gate.

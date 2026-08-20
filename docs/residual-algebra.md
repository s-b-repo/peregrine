# Residual Algebra

## Core idea

The experts in a MoE model produce outputs that are summed (residual-style):

```
R(x) = Σ_i g_i(x) · E_i(x)
```

Where g_i are the router weights and E_i are the expert outputs.

This is a **linear combination** of expert outputs. The key insight is that linear
algebra lets us restructure this computation to reduce memory movement.

## Algebraic expert cancellation

Suppose experts E_1, E_2, E_3 produce outputs that are approximately linearly
dependent:

```
E_3(x) ≈ a · E_1(x) + b · E_2(x)
```

Then we don't need to load E_3 at all — its contribution can be computed from E_1 and E_2:

```
g_3 · E_3(x) ≈ g_3 · (a · E_1(x) + b · E_2(x))
```

This is the residual-space elimination idea: skip loading an expert if its output
can be reconstructed from already-loaded experts.

## Low-rank shared basis

More generally, if we find a low-rank decomposition across all experts:

```
E_i(x) ≈ Σ_j a_{ij} · B_j(x)
```

Where B_j are shared basis functions (or matrices in weight space), then:

```
R(x) = Σ_i g_i · E_i(x) ≈ Σ_j (Σ_i g_i · a_{ij}) · B_j(x)
```

The entire expert layer reduces to computing the basis functions B_j (which are
shared, so resident in fast memory) and combining them with coefficients that are
cheap to compute from the router weights.

## Relationship to expert decomposition

- **Expert decomposition** (docs/expert-decomposition.md): W_i = B + Δ_i — decomposes
  the weight matrix, not the output. Stores shared basis in VRAM, streams deltas.
- **Residual algebra**: E_i(x) ≈ Σ_j a_{ij} B_j(x) — decomposes the output/function.
  If the basis functions B_j are simple (e.g. linear projections of the input),
  the entire expert computation can be replaced by a cheap shared basis evaluation.

## Implementation sketch

1. **Basis discovery**: given the full set of expert weight matrices {W_i}, compute
   a low-rank shared basis. SVD of the stacked matrix [W_1; W_2; ...; W_N] gives
   the dominant directions.

2. **Residual threshold**: for each expert, compute the residual after projection
   onto the shared basis. If |residual| < ε, the expert is "covered" by the basis.

3. **Runtime**: at inference, evaluate only the basis functions (resident in VRAM),
   then reconstruct each expert's output via its coefficients. Only experts with
   significant residuals need to be loaded from SSD.

This could reduce the 11.3 GB/token bandwidth by an order of magnitude if the expert
space has low effective rank (which MoE routing often produces — most of the output
mass is concentrated in a few dominant directions).

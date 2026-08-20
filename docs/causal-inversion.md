# Causal Inversion: Predict Computation, Not Token

## Today's approach

```
token → router → experts → weights → compute → logits → next token
```

## Inversion

Instead, define a prediction space over future computational states:

```
current hidden state
         ↓
  predict future expert activations
  predict future KV structure
  predict future hidden-state subspace
         ↓
  prepare only the necessary state
         ↓
  compute exact token
```

## Key insight

The prediction target isn't the next token.
It is: Ŝ_{t+k} = f(S_t)
where (S) could be:
  - expert IDs
  - expert contribution magnitudes
  - attention regions
  - low-dimensional hidden-state subspaces
  - expected residual direction

Then the engine prepares the future state before it exists.
This could allow prediction errors to be discarded without affecting correctness.

## Implementation

Add a prediction head to peregrine-router that outputs future
expert IDs + magnitudes K positions ahead. Use this to batch-preload experts
before they are actually needed.

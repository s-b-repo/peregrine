# Speed vs Bytes: The Critical Warning

## Not every "parallel decoding" method will make Peregrine faster.

For this engine the metric should be:

```
SSD bytes/token
```

Not FLOPs saved.

And secondarily:

```
VRAM/RAM bytes/token
```

## Why

The repo's own results demonstrate why. Peregrine can increase computational
parallelism yet remain disk-bound.

Recent MoE research reaches the same conclusion: ordinary speculative decoding
can actually increase expert traffic and slow inference. Utility-aware systems
therefore selectively enable speculation and dynamically choose draft length.

## New optimization category: "Causality relaxation / token-parallel inference"

Test in this order:

1. Expert-route speculative decoding
2. Self-speculative MoE
3. Blockwise verification
4. Jacobi parallel decoding
5. Medusa/EAGLE-style heads
6. Expert-union future-token execution
7. Adaptive causal/non-causal switching
8. True non-autoregressive/diffusion models

The first three are most compatible with the existing Peregrine architecture
(no need to abandon the GLM checkpoint) while attacking the biggest bottleneck:
the enormous amount of expert data moved per token.

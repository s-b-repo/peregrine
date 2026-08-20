# Speculative Routing for Peregrine

Even more interesting: speculative routing

Peregrine already has routing prediction.
The next step would be to turn that into an execution mechanism instead of merely a prefetch mechanism.

## Current conceptual flow

```
token
  ↓
router
  ↓
experts
  ↓
SSD/RAM
  ↓
compute
  ↓
next token
```

## Aggressive speculative system

```
token
  ↓
cheap router prediction
  ↓
predict next 4–8 token routes
  ↓
prefetch ONLY predicted experts
  ↓
draft generation
  ↓
full verification
```

## Key insight

The prediction operates at the expert-set level, not just the token level.

Peregrine's own measurements show consecutive-token expert overlap of 33.55% vs a
3.12% independence baseline, meaning there is real routing structure to exploit.

This could make the SSD cache and prefetch mechanisms substantially more useful.

## Implementation plan

1. Extend crate `peregrine-router` to predict next N=4-8 expert sets given current token
2. Prefetch only predicted experts (already exists, but scope it to predictions)
3. Draft generation with predicted experts, verify with full model in 1 pass
4. If verification fails, fall back to full model (no quality loss)
5. This implements: Speculative decoding + Draft-model routing prediction +
   Expert-only speculative execution + Multi-sequence expert union decoding

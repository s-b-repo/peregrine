# Blockwise Parallel Decoding

Another middle ground between 1-token-at-a-time and full non-autoregressive:

## Normal (sequential)

```
A → B → C → D → E
```

## Blockwise

```
A → [B C D E]
         ↓
    verify/refine
```

You don't completely abandon causality, but you make the unit of inference
a block instead of a token. This is potentially much more realistic for
GLM-style models than completely non-autoregressive generation.

## Implementation

Combine blockwise decoding with speculative routing:
1. Predict next block of 4-8 tokens
2. Draft-generate all tokens in the block
3. Verify the entire block in 1 pass
4. If any token in the block fails verification, fall back to token-by-token
   for that block

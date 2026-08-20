# Speculative Decoding Alternatives for Peregrine

The alternatives I would investigate for Peregrine

| Approach | Breaks strict 1-token-at-a-time generation? | Potential for Peregrine | Status |
|---|---|---|---|
| Speculative decoding | Yes | High | Partly explored |
| Self-speculative decoding | Yes | Very high | Not fully explored |
| Multi-token prediction / MTP | Yes | High | Present, but disk-bound behavior is a problem |
| Jacobi / iterative parallel decoding | Yes | Very high | Not apparent in Peregrine |
| Lookahead decoding | Yes | High | Not apparent |
| Medusa-style multiple heads | Yes | High | Not apparent |
| EAGLE-style draft heads | Yes | High | Not apparent |
| Tree speculative decoding | Yes | Medium/high | Not apparent |
| Draft-model routing prediction | Yes | Extremely interesting for MoE | Not fully explored |
| Expert-only speculative execution | Yes | Extremely interesting | Not apparent |
| Non-autoregressive generation | Yes | Potentially enormous | Requires model/training support |
| Mask-predict / iterative refinement | Yes | Potentially high | Requires compatible model |
| Diffusion LLM decoding | Yes | Potentially huge | Different model family |
| Discrete diffusion / masked generation | Yes | Potentially huge | Different model family |
| Blockwise parallel decoding | Partially | High | Not comprehensively explored |
| Token-tree execution | Partially | High | Not apparent |
| Multi-sequence expert union decoding | Partially | Very high | Batching exists, but could go much further |

## KEY INSIGHT: Relationship to Tier 1 priorities

The alternative approaches map directly to the Tier 1 optimization targets in `scripts/opencode-prompt.txt`:

1. **Expert-route speculative decoding** (Tier 1 #1) = Speculative decoding + Draft-model routing prediction
2. **Expert-union future-token execution** (Tier 1 #3) = Expert-only speculative execution + Multi-sequence expert union decoding
3. **Pre-fill/decode expert-union sharing** (Tier 1 #4) = Blockwise parallel decoding + Multi-sequence expert union decoding

## Implementation priority order

1. Expert-route speculative decoding (uses GPU-resident hot experts for draft tokens)
2. Self-speculative MoE (no separate draft model needed)
3. Blockwise verification (predict 4-8 tokens, verify in 1 pass)
4. Jacobi parallel decoding (parallel token updates)
5. Medusa/EAGLE-style heads (lightweight draft heads)
6. Expert-union future-token execution (load union of predicted experts once)
7. Adaptive causal/non-causal switching (high overlap → parallel, low → sequential)
8. True non-autoregressive/diffusion models (requires different model architecture)

The first three are most compatible with existing Peregrine architecture (no need to abandon the GLM checkpoint) while attacking the biggest bottleneck: the enormous amount of expert data moved per token.

## Critical metric

For this engine the metric should be **SSD bytes/token**, not FLOPs saved. Speculative techniques must reduce total bytes/token or they slow inference.

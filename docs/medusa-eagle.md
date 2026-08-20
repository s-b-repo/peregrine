# Medusa / EAGLE-Style Decoding

These are another category worth testing.

Instead of a separate draft model, attach lightweight prediction heads:

```
main model
     │
     ├── head → token +1
     ├── head → token +2
     ├── head → token +3
     └── head → token +4
```

The big model then verifies several candidates together.

## Advantage for Peregrine

One streamed expert load potentially validating multiple tokens.

## Danger

Verification still touches the expert union for multiple positions, so it has
to be measured against the additional bytes. Recent work on MoE speculation
specifically warns that naive speculation can increase data movement
substantially.

## Combined approach

Medusa heads + blockwise + speculative routing — predict 4-8 next tokens,
draft-verify in 1 pass with shared expert load. If verification fails, fall back.

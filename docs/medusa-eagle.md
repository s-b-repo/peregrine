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

## Where this sits (2026-09-02)

[EAGLE-3](https://arxiv.org/abs/2503.01840) is one of the six drafter
architectures Tencent's [AngelSpec](https://arxiv.org/abs/2607.25852)
training workbench unifies behind a single config flag. The status table on
[the alternatives page](speculative-decoding-alternatives.md) already scores
the EAGLE family's place here, and AngelSpec does not move it: this engine has
the EAGLE-1 move (`MtpHead` drafts from the hidden state), the tree substrate
is EAGLE-2's dynamic tree in other clothes, and AngelSpec is a *training*
workbench — the checkpoints it would produce are precisely the artifact this
engine cannot make (its flagship, DFly, reports 4.79 average accepted length
on Hunyuan 3). Notes and sources:
[the AngelSpec cross-read](dflash.md#the-angelspec-cross-read-2026-09-02) on
[what peregrine took from DFlash](dflash.md).

## Combined approach

Medusa heads + blockwise + speculative routing — predict 4-8 next tokens,
draft-verify in 1 pass with shared expert load. If verification fails, fall back.

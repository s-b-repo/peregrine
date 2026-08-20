# Expert-Level Non-Causal Execution

This is the idea most worth researching for Peregrine.

## The core insight

Instead of asking:
    "What is the next token?"

Ask:
    "What experts will the next K positions probably need?"

Then execute the union of those experts once.

## Example

```
Token 1 → experts {A B C D}
Token 2 → experts {A B C E}
Token 3 → experts {A B D E}
Token 4 → experts {A C D E}
union → {A B C D E}
```

Instead of:
```
read A B C D
read A B C E
read A B D E
read A C D E
```

You load:
```
A B C D E
```
once and process all candidate positions.

## Why it matters

That is directly aligned with Peregrine's core bottleneck — disk-bound expert
reads on a 10 GB cache with only 0.6% hit rate.

The existing B=16 result already demonstrates the principle: batching improves
aggregate throughput because routed experts can be shared across sequences.
Peregrine measured 0.064 → 0.280 tok/s at B=16.

The unanswered question is whether the same sharing can be exploited across
future tokens of one sequence.

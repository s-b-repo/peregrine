# Three Most Promising Ideas

If I were trying to push Peregrine beyond its current design, I would start with these:

## 1. Residual-space expert elimination

Don't ask:

> "Which experts did the router select?"

Ask:

> "Which selected experts contribute genuinely new information to the residual?"

Mathematically:

```
|(I - P_S) g_i E_i(x)|
```

becomes the selection criterion.

Where P_S is the projection onto the span of already-selected experts' outputs.
An expert whose output lies mostly within the span of others contributes little
new information and can be skipped.

This is a refinement of the existing **residual-space routing** idea — but more
aggressive. Instead of just pruning experts whose output is linearly dependent
(soft pruning with flip-gate validation), this computes the actual residual
contribution after loading and can drop experts mid-forward.

Potentially enormous reduction in expert computation and I/O.

### Relationship to existing code

- `peregrine-router` already computes gate weights g_i.
- The span projection P_S can be computed incrementally as experts are loaded.
- The flip-rate gate (`Model::prediction_flip_rate`) measures whether this changes
  outputs — the quality safety net already exists.

### The runtime objection, and the proposed answer (`orionzion`, 2026-08-21)

The criterion can only be evaluated **after** the ~18.9 MB read it exists to avoid, so as
stated it is an offline pruning pass, not a scheduler change. The suggested way out is a
**tiny resident per-expert surrogate** — a low-rank sketch, sized so every expert's surrogate
stays resident at once — predicting *projected output novelty* against the already-selected
span before the weights are issued.

Two costs must be charged to the same budget or the result is not readable: the surrogates'
own resident bytes, and the **false-negative** rate, which drops an expert that mattered and
is therefore a flip-rate cost rather than a byte cost. The bar is set by prior art in this
repo — `peregrine-skipbound` measured a weight-norm bound and found it added **0.12 points**
over the gate weight alone, because `C_e` barely varies across similarly-trained experts. A
surrogate has to beat **the gate**, not zero.

## 2. Cross-expert factorization

Search for:

```
W_i = B + Δ_i
```

or:

```
W_i = Σ_k a_{ik} B_k
```

Put B (shared basis) resident and stream only Δ_i (small per-expert delta).

This is the detailed treatment already in [expert-decomposition.md](expert-decomposition.md).
It attacks Peregrine's fundamental 11.3 GB/token bandwidth problem at the
**representation level** — not the scheduler, not the cache, but the actual
weight format.

The int3-g64 failure (0.447 flip rate, data-free quantization) is NOT evidence
against this — it used data-free rounding. Adaptive precision (see
[token-equivalence-adaptive-precision.md](token-equivalence-adaptive-precision.md))
uses a calibration signal, which is the standard approach that survived in llama.cpp.

### How to measure it, and the control it needs (`orionzion`, 2026-08-21)

**Weight-space reconstruction error is the wrong objective.** A basis can lower
`‖W_i − (B + Δ_i)‖_F` while making `Δ_i` high-entropy and **hostile to the int4/block
quantization the residual then has to survive** — the fit looks good and the container does
not shrink. That is the failure mode that decides this idea, and Frobenius error cannot see it.

Score it instead as **rate–distortion on activations**: fit the basis per layer, then measure
the *residual bytes required to preserve downstream logits* under the existing flip-rate gate.
Sweep **basis rank and residual quantization jointly** — they are not separable, since a rank
that looks wasteful may buy a residual that quantizes far better. Report **bytes/token with the
resident basis charged once**, read amplification, and flip rate, so the result is comparable to
`--tier-hot-frac` and int2-g64 rather than living on its own scale.

**The control: shuffled experts at equal rank.** If a learned grouping does not beat *random*
groups at equal rank and equal residual precision, the basis is capturing **layer-wide
structure** rather than cross-expert redundancy. The saving would still be real, but it would be
attributable to one per-layer mean and no grouping search — and calling it cross-expert
compression would be the same error as reporting a warm-cache hit rate as a routing statistic.
Print both arms side by side.

### Both of the above are now implemented — `peregrine-basisfit`

```bash
peregrine calib-capture <model-dir> calib.json 512 --text corpus.txt
peregrine-basisfit <model-dir> --calib calib.json --rank 2 --groups 8 \
    --residual int2-g64 --control
```

Fits the basis, quantizes the residual through the same producer and consumer
the loader uses, scores the error weighted by calibrated per-channel `mean|x|`,
and runs the shuffled control against **four** random partitions rather than
one. Both arms default to the same precision one rung below the container, so
streamed bytes are identical and the only variable is the basis; the resident
basis is charged separately and never folded into the comparison.

**The control needed four draws, not one.** Against a single shuffled partition
the demo container reported its grouping as load-bearing at a 6.87 % margin;
against the best of four it reads 2.56 % against a 4.63 % spread of the draws
themselves — a negative. The floor is `max(15 %, spread)`, not a sign test.

**What is still owed: the sweep on real GLM-5.2 weights.** The harness is
validated on fixtures only, and a fixture cannot tell you whether *this*
checkpoint's experts share anything.

## 3. Future-state computation

Don't speculate tokens.

**Predict future hidden states + future expert addresses**, preconstruct them,
and discard incorrect futures.

This is much closer to Peregrine's real bottleneck than ordinary speculative decoding.
The bottleneck is not "compute is slow" — it's "11.3 GB of expert data moves per token".
Speculative decoding doesn't reduce that; it multiplies it (each draft row streams
its own expert union). Future-state computation pre-stages the expert reads *before*
the routing decision, converting latency into bandwidth utilization.

### Three sub-components

1. **Future expert address prediction** (already in [expert-address-prediction.md](expert-address-prediction.md)):
   predict the byte ranges that will be needed, issue async reads early.

2. **Future hidden state prediction** (related to [causal-inversion.md](causal-inversion.md)):
   predict Ŝ_{t+k} = f(S_t) for expert IDs, magnitudes, and KV structure.

3. **State reservoir** (related to time-shifted inference in opencode-prompt.txt):
   maintain several future state candidates in parallel, pre-compute shared
   portions, discard incorrect branches at low cost.

## The fundamental insight

The current Peregrine architecture asks:

> How can we move 11.3 GB faster?

The more radical question is:

> Why does inference need to move 11.3 GB at all?

That changes the optimization target from:

```
faster inference
```

to:

```
┌─────────────────────────────────────────────┐
│  minimum information movement required      │
│  for a given output fidelity               │
└─────────────────────────────────────────────┘
```

That is where the largest discontinuous improvement lies — not in another 5–20%
scheduler optimization, but in changing what gets moved.

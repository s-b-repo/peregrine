# What peregrine took from DFlash

[**DFlash**](https://github.com/z-lab/dflash) (z-lab; Chen, Liang & Liu,
[arXiv:2602.06036](https://arxiv.org/abs/2602.06036), plus the
[DFlash 2](https://inco.ai/blog/dflash2/) follow-up) is a **block diffusion**
draft model for speculative decoding. Its draft is not autoregressive: it takes
the whole verify block as mask tokens, cross-attends to hidden states pulled
from several layers of the target, and emits every position of the block in
**one** forward. DFlash 2 adds a low-rank candidate selector to put back the
sequential dependency that parallel drafting throws away.

This page records what was taken, what was measured, and — at least as
importantly — what could not be taken and why, in the style of
[`ideas-from-colibri.md`](ideas-from-colibri.md) and the closed negatives in
[`ideas-tokens-per-sec-2026-08-15.md`](ideas-tokens-per-sec-2026-08-15.md).

Read [speculative decoding alternatives](speculative-decoding-alternatives.md)
first for the economics every row below is scored against.

## Summary

| Taken from | peregrine | Status |
|---|---|---|
| `_rejection_sample`'s `draft_indices` / `draft_probs` pair | [`draftdist.rs`](../crates/peregrine-model/src/draftdist.rs), `speculative_sample_at` | **Shipped.** A draft's `q` is stored at its support: ~620 KB → tens of bytes per draft position |
| — the borrow that forced a second copy | `accept_run_sampled` | **Shipped.** A second ~620 KB copy per draft position removed |
| `_sampling_probs`'s `torch.topk` stage | `Sampler::with_top_k`, `top_k` on the HTTP API | **Shipped.** peregrine had no rank cutoff at all |
| `_sampling_probs`'s *shape* — rank first, order only the survivors | `Sampler::truncate` | **Shipped.** Replaces a full vocabulary sort per sampled token: **4.8–5.1×** on decode-shaped rows |
| Block-diffusion parallel drafting | — | **Not taken.** Needs a trained draft; see below |
| Multi-layer context feature (`build_target_layer_ids`) | — | **Not taken.** Same reason |
| `GroupedDynamicCausalConv` | — | **Not taken.** Same reason |
| DFlash 2's `CandidateSelector` | — | **Not taken**, but its *shape* is recorded below as the cheapest known way to build a candidate tree |

## 1. A draft distribution is stored at its support

Rejection sampling needs exactly two things from the distribution `q` a draft
was drawn from: `q(drafted)`, and the residual `(p − q)+`. peregrine stored `q`
densely — one `f32` per vocabulary entry, per draft position, per sequence,
resident between ticks. `Model::mtp_draft_sampled` priced that honestly in its
own doc comment (*"~2.4 MB per sequence at GLM-5.2's vocab and `g = 4`"*) and
then declined to fix it, for a reason that was correct: a sparse `q` **alongside**
a dense one is two representations of one distribution, and the two coming apart
is exactly how speculation stops being distribution-preserving while still
looking reasonable.

DFlash answers that objection by making the support the *only* form.
`_rejection_sample` takes `draft_probs` with an optional `draft_indices` and
branches on `draft_indices is None` in the two places a support matters:

```python
q        = (draft_probs * (draft_indices == draft_tokens[..., None])).sum(-1)
residual = target_probs[0, accepted].scatter_add_(0, draft_indices[0, accepted],
                                                  -draft_probs[0, accepted]).clamp_min_(0)
```

`DraftDist` is that: one type, `idx: Option<Vec<u32>>`, dense and sparse as two
variants of the same value rather than two code paths. `Sampler` already
truncated to a nucleus, so the support was *already* small — the engine was
allocating a mostly-zero vector to hold it.

- GLM-5.2's vocabulary is 154 880 tokens, so a dense `q` is **619 520 B**.
- A nucleus keeping 64 entries is **512 B**. A greedy draft's one-hot is **8 B**.
- It falls back to dense when the support passes half the vocabulary, where an
  index next to each probability would cost *more* (8 B/entry against 4 B/entry)
  and lookups would be `O(k)` instead of `O(1)`. `top_k` is what bounds that
  case: it caps the support no matter how flat the row underneath it is.

**Why this is a change of representation and not of algorithm.** `q` is zero off
its support, so `q(t)` and `(p − q)+` are the same numbers either way, and
`speculative_sample_at` performs the arithmetic in the same order and the same
precision (`f32` subtract, `f32` clamp, then widen) as the dense reference. That
equality is a test, not an argument — `sparse_and_dense_acceptance_agree_token_for_token`
runs both encodings on the same uniforms and requires the same token, the same
gate the SIMD integer-dot kernels get against their scalar reference. A second
test re-checks the property the mechanism exists for, because two identically
wrong implementations would agree with each other.

### The second copy

`accept_run_sampled` also did this, once per draft position per round:

```rust
let p = sampler.distribution(r).to_vec();      // ~620 KB allocated and copied
let (u_accept, u_resample) = (sampler.uniform(), sampler.uniform());
```

The `to_vec()` bought nothing but the release of a borrow — `distribution`
borrows the sampler and the two uniform draws need it mutably. Drawing the
uniforms *first* removes the copy, and the RNG stream is untouched because
`distribution` consumes none of it.

## 2. `top_k`, which peregrine did not have

`Sampler` supported temperature and nucleus (top-p) and nothing else. DFlash's
`_sampling_probs` applies `torch.topk` before the softmax and `top_p` after it,
and the model families this engine serves publish `top_k` in their own
recommended settings — the DFlash benchmark invocations run Qwen at
`--top-k 20` and `--top-k 64`. An OpenAI-compatible endpoint that accepted the
field and ignored it was answering a different question than the client asked.

`Sampler::with_top_k(k)` is a builder rather than a fourth argument to
`Sampler::new`, so the twenty-odd existing construction sites — nearly all tests
asserting something about temperature or the nucleus — keep saying what they
mean. `0` is off, which is also what vLLM and SGLang mean by `top_k: 0`.

The two cutoffs compose in DFlash's order — **top-k first, renormalize over it,
then apply the nucleus to those probabilities**. The other order would measure
the nucleus against mass the top-k has already discarded, so the same
`(top_k, top_p)` pair would mean something different here than at the client that
sent it. Concretely: 64 equally likely tokens under `top_k=64, top_p=0.95` leave
61, and that is asserted.

On the HTTP API `top_k` is a request field alongside `top_p`, and it joins the
response-memo key. It cannot change a *greedy* answer and only greedy requests
are memoizable — but the key is the whole request semantics rather than the
subset that happens to matter today, which is the same reason `top_p` is already
in it.

## 3. The truncation stopped being a full sort of the vocabulary

The pre-existing `dist_build` answered "which tokens are in the nucleus?" with
`sort_by` over every index in the vocabulary — for GLM-5.2, ordering 154 880
entries with an indirect comparator to find, usually, the first few dozen.
DFlash's `_sampling_probs` has the shape that avoids it: rank first, order only
what survives.

A top-k gives the rank boundary outright. A nucleus does not, so `nucleus_bound`
derives one in a single pass: bucket the probabilities by IEEE-754 exponent (256
buckets), walk from the top accumulating mass, and stop at the bucket where the
nucleus mass is reached. Every entry outside those buckets is strictly smaller
than every entry inside them, so those `n` entries **are** the ranked top `n` —
which makes it a bound and not a heuristic. One `select_nth_unstable_by` and a
sort of `n` then settle the answer exactly.

Two details are load-bearing:

- The partial comparator breaks probability ties by **ascending index**. The
  selection is unstable, so without a tiebreak two equally likely tokens could
  rank either way and a seeded request would stop being reproducible. The full
  sort it replaces was *stable*, which already put equal probabilities in
  ascending index order — so this reproduces it exactly rather than merely
  plausibly. (Ranking the whole vocabulary keeps using a stable sort and needs no
  tiebreak, which is also one comparison cheaper.)
- `the_o_v_selection_reproduces_the_full_sort_bit_for_bit` keeps a copy of the
  old routine in the test module and requires the two to produce the same `f32`
  vector, across nucleus values from 0.0 to 1.0 and including rows deliberately
  full of ties.

### Measured

```
cargo run --release -p peregrine-model --example nucleusbench
```

Ryzen 5 5500, single thread, GLM-5.2's 154 880-token vocabulary, `temp = 1.0`,
`top_p = 0.95`, median of 120 runs. Logits are Zipf-shaped (`lo[rank] =
−α·ln(rank+1)`, shuffled into the vocabulary) — a short dominant head over a long
near-flat tail, with `α` sweeping how peaked the row is, and with it how many
tokens the nucleus keeps. The **new** side calls the shipped
`Sampler::distribution`; the old side is the pre-2026-08-22 `dist_build` body,
and the two are checked to produce the identical `f32` vector before anything is
timed.

| α | nucleus keeps | full sort | rank-bounded | |
|---:|---:|---:|---:|---:|
| 3.0 | 3 | 5221 µs | 1029 µs | **5.07×** |
| 2.0 | 12 | 5305 µs | 1103 µs | **4.81×** |
| 1.5 | 218 | 5251 µs | 1073 µs | **4.89×** |
| 1.2 | 16 748 | 5276 µs | 2039 µs | **2.59×** |
| 1.0 | 82 787 | 5419 µs | 4477 µs | **1.21×** |
| 0.8 | 122 430 | 5446 µs | 5639 µs | 0.97× |
| 0.5 | 139 807 | 5452 µs | 5714 µs | 0.95× |

The softmax is ~500 µs of every row and is common to both, so on the truncation
alone the top rows are ~9×.

**The last two rows are the honest cost and are worth stating plainly.** When the
nucleus keeps ~90 % of the vocabulary, the bound costs a pass over the
probabilities and saves nothing, and the result is 5 % slower. That regime is a
high temperature with `top_p` near 1 — a nucleus that is barely truncating.
Everything a decode distribution normally looks like is in the top three rows.

Two earlier designs were measured and rejected, recorded so they are not
re-proposed:

- **Grow the rank from 64 by ×8 until the mass is reached.** 5.8× on peaked rows
  and **0.65×** on flat ones — four selection passes plus a 32 768-element sort,
  all wasted, before falling through to the full sort anyway.
- **One probe at 64, then the full sort on a miss.** Uniformly 0.80× outside the
  peaked case: the probe is pure loss whenever it misses, and it misses exactly
  when the row is flat.

## What could not be taken, and why

DFlash's headline — **the draft emits the whole block in one forward** — is worth
more on this engine than on the ones it was measured on, and that is precisely
why it is worth being clear that it is not portable here.

On the streaming track a draft *step* is the expensive thing: `MtpHead` is a
sparse MoE layer stored int8, ~300 MB of SSD at `topk=8`, and `mtp_draft_with`
runs it once per draft position with no batch-union amortization (see
[why MTP is disk-bound](speculative-decoding-alternatives.md#why-mtp-is-disk-bound-and-what-would-fix-it)).
Collapsing γ steps into one forward would collapse γ expert unions into one. It
is the largest single win available on that path.

It needs a draft that was **trained** to consume a block of mask tokens
conditioned on the target's hidden states. Feeding mask tokens to peregrine's
autoregressive MTP head produces a draft with no relationship to the sequence;
speculation would stay correct (every draft is verified) and acceptance would
collapse to noise, which on a disk-bound engine is strictly worse than not
speculating. The same applies to the pieces around it: the multi-layer context
feature needs the trained `fc` that projects `len(target_layer_ids) × hidden`
down to `hidden`, `GroupedDynamicCausalConv` is a learned data-dependent kernel,
and DFlash 2's `CandidateSelector` needs its two trained codebooks.

**But the usual closure does not apply, and that is the finding worth recording.**
[The alternatives page](speculative-decoding-alternatives.md#closed-here-and-why)
closes non-autoregressive and diffusion decoding with *"what does not exist is a
checkpoint"*. For DFlash that is only half true:

- **Qwen3.5-27B — the resident track — has a published DFlash draft** in the
  [DFlash collection](https://huggingface.co/collections/z-lab/dflash). So does
  GLM 5.1. This is not a training project on that endpoint; it is an
  implementation project.
- **GLM-5.2 — the streaming track, where the win would be largest — does not.**
  The collection lists GLM 5.1, not 5.2.

So the shape of the opportunity is inverted from where the value is, in the same
way [trees are](speculative-decoding-alternatives.md#trees-are-an-mla-track-mechanism-and-that-is-a-real-constraint):
the track with a checkpoint is the one where an extra draft forward is nearly
free, and the track where a draft forward costs 300 MB of SSD has no checkpoint.

What building it on the resident track would take, sized rather than hand-waved:
a new draft architecture (`Qwen3DFlashAttention` cross-attends to the target
hiddens as extra keys/values rather than to its own history), a loader for its
config's `dflash_config` block, a mask-token embedding path, and the block
verify loop — which is the part peregrine already has, since `CandidateTree::chain`
plus `accept_run` *is* the block accept rule and `forward_rows_batched` already
runs arbitrary multi-row layouts. It is a real project, not a phase, and it is
not blocked on training.

### The candidate-selector shape, recorded

DFlash 2's `CandidateSelector` is the cheapest known way to turn per-position
top-k candidates into one chosen path:

```text
score(candidate c at position t) = logit(c) + ⟨ pred_codebook(chosen_{t−1}) ⊙ W·h_t , succ_codebook(c) ⟩
```

— a rank-`r` bilinear compatibility term between the predecessor actually chosen
and each candidate, walked greedily over positions. It is worth recording because
peregrine has the *substrate* for it (`CandidateTree` lays candidates out in DFS
order over ordinary consecutive cache slots) and no *builder*: `CandidateTree::chain`
is the only constructor speculation uses, so the MTP head commits to one token per
depth and the tree is only ever used to hedge two draft sources against each other.

It is not proposed as work. On the streaming track every sibling is another
verify row against a 2.63× union growth factor, and a weight-free substitute for
the codebooks would be a new heuristic with no measurement behind it. It is here
so that if a tree builder is ever wanted, this shape is the one to reach for
first.

## Also read

- [Speculative decoding alternatives](speculative-decoding-alternatives.md) — the
  index this page hangs off, and the economics
- [Configuration](configuration.md) · [Serving](serving.md) — where `top_k` shows up
- [`ideas-from-colibri.md`](ideas-from-colibri.md) — the same exercise against the
  predecessor engine

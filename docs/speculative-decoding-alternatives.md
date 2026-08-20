# Speculative Decoding Alternatives for Peregrine

The alternatives worth investigating for peregrine, re-scored against what the
engine measurably is rather than against the literature's assumptions.

**Read the economics first.** Everything below divides into one fraction:

```
speedup = (1 + accepted_tokens) / union_growth_factor
```

and which term matters depends on which of the two serving tracks you are on.

| Track | Regime | What an extra verify row costs |
|---|---|---|
| **GLM-5.2** (`serve-glm.sh`) | 363 GB of experts streamed, 10.85 GB/token, io duty 93 % | **disk bytes.** γ=4 at B=16 grew the union 3 855 → 10 145 distinct reads (**2.63×**), so break-even needs an accepted run above ~2.6, not above 1 |
| **Qwen3.5-27B** (`serve-qwen.sh`, `:8132`) | `COLI_STREAM=0`, fully resident | **compute on weights already in RAM/VRAM.** `union_growth_factor ≈ 1`; the numerator is all that matters |

The original version of this page described only the first, which is why several
rows below were mis-scored.

## Status

| Approach | Breaks strict 1-token-at-a-time generation? | Potential for Peregrine | Status |
|---|---|---|---|
| Speculative decoding | Yes | High | **Shipping.** `COLI_DRAFT=5` + `COLI_SPEC_CONF=0.65` is the GLM default: +37 % tok/s, −22 % disk reads, accept rate 10 % → 83 % (REPEATS=3) |
| Self-speculative decoding | Yes | Very high | **Shipping.** The MTP head *is* self-speculation — there is no separate draft model to load |
| Multi-token prediction / MTP | Yes | High | **Shipping.** "Disk-bound" is the *draft* path, not the verify path — see [the four causes](#why-mtp-is-disk-bound-and-what-would-fix-it) |
| Jacobi / iterative parallel decoding | Yes | Very high | Not built. Needs the tree substrate first |
| Lookahead decoding | Yes | High | Not built. Its n-gram pool half is the cheapest unbuilt item on this page |
| Medusa-style multiple heads | Yes | ~~High~~ **Low** | **Declined** — see [below](#closed-here-and-why) |
| EAGLE-style draft heads | Yes | High | **Already present in EAGLE-1 form**: `MtpHead` drafts from the *hidden state*, which is EAGLE's defining move. The upgrade is EAGLE-2's dynamic tree |
| Tree speculative decoding | Yes | Medium/high | **Substrate built** (`tree.rs`, `Model::forward_tree_rows`, `accept_tree`) and proven end to end; what remains is the `batch.rs` half. **MLA only** — see below |
| Draft-model routing prediction | Yes | Extremely interesting for MoE | Deferred — competes with a router look-ahead that already measures 92.5 % recall |
| Expert-only speculative execution | Yes | Extremely interesting | Deferred, same reason |
| Non-autoregressive generation | Yes | Potentially enormous | **Not buildable here** — needs a NAR checkpoint |
| Mask-predict / iterative refinement | Yes | Potentially high | **Not buildable here.** Nearest in-tree analogue is RLM (`rlm.rs`), which already refines a hidden without appending KV |
| Diffusion LLM decoding | Yes | Potentially huge | **Different model family** — [sized below](#closed-here-and-why), not dismissed |
| Discrete diffusion / masked generation | Yes | Potentially huge | Same |
| Blockwise parallel decoding | Partially | High | `COLI_DRAFT` *is* a width-1 block, and `CandidateTree::chain` is that block in the tree substrate's terms |
| Token-tree execution | Partially | High | Same substrate; same state |
| Multi-sequence expert union decoding | Partially | Very high | **Shipping.** Continuous batching + `COLI_FUSE_PREFILL`. "Could go much further" = the `EXPERT_BUDGET` union cap, which is a *byte* lever tracked in `ideas-tokens-per-sec-2026-08-15.md`, not a decoding one |

### Recurrent architectures could not speculate at all until 2026-08-20

Worth stating separately because it gated the whole table on the fast endpoint.
`spec_reject_is_kv_only()` returned false for `Arch::HybridGdn`, so
`peregrine-serve` refused to draft on Qwen3.5 — the *resident* track, where
speculation's economics are textbook-favourable — and printed a line saying so.
A hybrid's linear-attention layers keep a delta-rule state rather than rows, so
a rejected draft cannot be truncated away.

`COLI_SPEC_GDN=1` wires the rollback the `GdnState` API was written for:
snapshot before the verify forward, drop it on full acceptance, and on partial
acceptance restore plus re-advance over exactly the committed rows (all such
sequences in one forward). Output-neutral. See
[`configuration.md`](configuration.md#coli_spec_gdn) for the cost, which is
real: ~151 MB per drafting sequence per tick at 27B dims.

### Trees are an MLA-track mechanism, and that is a real constraint

The substrate landed with a limitation worth stating up front, because it
inverts the obvious expectation.

A tree needs sibling rows that branch from a shared context. A **recurrent**
(GDN) layer keeps one delta-rule state per sequence and advances it row by row,
so two siblings would chain into each other's state instead of branching —
and giving each branch its own state costs a full `GdnState` copy per branch per
layer (~3.1 MB × 48 at 27B dims). The batched **GQA** path takes no key set at
all, so a mask there would be silently ignored. `Model::forward_tree_rows`
refuses both rather than answering plausibly and wrongly.

So trees work on **GLM/MLA — the streaming track**, where an extra verify row
costs disk bytes, and *not* on **Qwen3.5 — the resident track**, where extra
rows are nearly free. That is exactly backwards from where the value is, and it
is why tree width on the track that supports it has to be spent against the
union-cost gate rather than set to a constant. The hybrid speculates in chains.

What the substrate does provide, proven rather than argued:

- `CandidateTree` lays candidates out in **DFS order as ordinary consecutive
  cache slots**, so `LayerKv` did not change at all — its append-only,
  strictly-ascending invariant is untouched, and a rejected branch disappears
  through the same `truncate` a rejected chain does.
- `RowLayout::sel` gives each row its ancestors (siblings are absent from each
  other's key sets) and `RowLayout::rope_pos` gives each row its tree *depth*,
  so siblings share a logical position instead of being rotated as a sequence.
- `accept_tree` walks from the root taking whichever child matches that node's
  own argmax, and **reduces exactly to `accept_run` on a chain** — asserted, so
  a tree changes how many tokens a forward commits and never which.
- `a_tree_row_cannot_see_its_siblings` checks the masking on a real forward
  against the same branch run alone, and also requires that an *unmasked* third
  row would differ — otherwise it would pass with `sel` ignored.

### Why MTP is disk-bound, and what would fix it

On the *streaming* track only, and it is the draft path rather than the verify
path — the verify forward already shares one expert union across `B·(1+γ)` rows.

1. The MTP head is a **sparse MoE layer with its own expert pool** at layer
   index `n_layers`.
2. Its experts are stored **int8**: 37,748,736 bytes each against 18,874,368 at
   int4 — 2× a normal expert. The requantizer deliberately left that row alone.
3. Each draft step runs at `s_n = 1`, so γ steps issue γ **serial, disjoint**
   unions with no cross-row amortization: ~300 MB of SSD per draft step at
   topk=8, ≈1.2 GB per round before the verify forward starts.
4. The draft `ForwardCtx` passes `expert_index: None, heat: None,
   layout_schedule: None, affinity: None` — so reads take the slow resolve
   path, skip the `(fd, offset)` sort, never gain residency heat, and are never
   prefetched or VRAM-promoted.

All four are fixable and all four are output-neutral. (1)–(2) especially:
**a draft is only ever accepted where it equals the verify forward's own
argmax, so degrading the draft head cannot change a served token** — it can only
change the acceptance rate. Requantizing that one layer to int4 therefore needs
no flip-rate gate, just an assertion that the gate still reads 0.000.

### Closed here, and why

Recorded in the style of `ideas-tokens-per-sec-2026-08-15.md`'s closed
negatives, so they are not re-proposed without new evidence.

- **Medusa-style multiple heads — declined, not deferred.** Medusa attaches N
  *independent* per-offset heads. `MtpHead` is one head applied *recursively*,
  conditioned on the hidden state each step — strictly more informed, which is
  the whole reason EAGLE beat Medusa. Building Medusa here would be replacing a
  better mechanism with a worse one. It is also not an engine task: neither
  GLM-5.2 nor Qwen3.5 ships Medusa weights, so it is a *training* project.
- **Non-autoregressive / mask-predict / diffusion decoding — no checkpoint.**
  A masked-diffusion LM (LLaDA/Dream shape) would need a new `Arch` variant
  (`config.rs` hard-errors on an unknown `model_type` by design), bidirectional
  attention, and an iterative unmask loop. Two of the three already exist —
  `sel` can express a non-causal mask, and `forward_rows_batched` runs arbitrary
  multi-row layouts. What does not exist is a checkpoint. Separate project, not
  a phase.
- **Naive draft-model speculative decoding on the streaming track.** Two
  independent engines measured it: colibrì recorded ~5 % acceptance and ~3×
  slower; peregrine measured `COLI_DRAFT=4` at 1.57× slower. Each drafted row
  streams its own expert union. The adapted form that *does* pay is the MTP head
  plus a confidence floor.

## Relationship to the Tier 1 targets

The mapping below is to the goal list in `scripts/opencode-prompt.txt`. Kept for
traceability; note that the priority order further down supersedes it, because
that list was written before the two tracks were distinguished.

1. **Expert-route speculative decoding** (Tier 1 #1) = Speculative decoding + Draft-model routing prediction
2. **Expert-union future-token execution** (Tier 1 #3) = Expert-only speculative execution + Multi-sequence expert union decoding
3. **Pre-fill/decode expert-union sharing** (Tier 1 #4) = Blockwise parallel decoding + Multi-sequence expert union decoding

## Implementation priority order

Ordered by expected value against the *measured* bottleneck of each track, not
by how much of the literature a row covers.

1. **Speculation on the resident track** — `COLI_SPEC_GDN`. **Done
   (2026-08-20).** The one change that turned the whole table on for `:8132`.
2. **A second draft source: n-gram / prompt-lookup** — `COLI_DRAFT_NGRAM`.
   **Done.** Zero weights, zero model calls, no training. The only draft source
   that on the streaming track does not first pay ~300 MB of MTP expert reads,
   and it works on a checkpoint with no MTP head at all. Pays on both tracks.
3. **Tree / blockwise verification.** **Substrate done**, `batch.rs` half open.
   One mechanism; "tree spec", "token-tree", "blockwise" and the tree half of
   EAGLE-2 all reduce to it. Needed no attention-mask work and no `LayerKv`
   change. Sequenced behind (5) because trees turned out to be MLA-only, so
   their width is spent on the track where rows cost bytes.
4. **Cut the MTP draft path's bytes** (streaming track). **Two of four done** —
   the draft path has the `ExpertIndex`, and `mtp_draft_batched` turns B
   disjoint unions per step into one. `--mtp-target int4` on that layer is
   built and needs a conversion run to become a number; pinning its hot experts
   is open, and needs a residency mechanism that is not the heat table (the
   draft context withholds heat on purpose).
5. **Union-cost draft admission** (streaming track). **Instrument done, gate
   deliberately not.** `decode.tokens_emitted` / `decode.rows` against the
   `ecache` counters give tokens per expert read directly. `COLI_SPEC_UNION_MAX`
   — the cost-side twin of `COLI_SPEC_CONF`'s acceptance-side floor — stays
   unbuilt until those counters say whether speculative rows are already
   union-cheap under the 0.65 floor. Tuning a gate against an unmeasured
   quantity is the failure `measurement.md` opens with.

**What none of this has yet is a number.** Every item above is env-gated and
default-off, and this workspace cannot load the real checkpoint. The A/Bs owed
are: `COLI_SPEC_GDN` at B ∈ {1, 8, 32} on `:8132` (the ~151 MB/sequence snapshot
is charged per sequence while a forward's weight read is shared, so the
crossover is a batch width); `COLI_DRAFT_NGRAM` on a coding/IT corpus where its
acceptance argument is supposed to hold; and the draft-path byte items measured
as tokens-per-disk-read rather than wall clock. Use
`scripts/bench-serve-envarms.sh` — rotating arms, medians, `REPEATS≥3` — and
treat a gap smaller than the measured spread as unresolved.

Deferred with reasons: draft-model routing prediction and expert-only
speculative execution are *prefetch* ideas, and the router look-ahead they would
compete with already measures 92.5 % recall / 46.3 % precision — while 98.6 % of
speculative reads were arriving too late to use at 93 % io duty. Revisit only
once the counters show speculative rows are union-cheap.

## Critical metric

On the **streaming** track the metric is **SSD bytes per accepted token**, not
FLOPs and not tokens per forward: a speculative technique that does not reduce
bytes per accepted token slows inference there, whatever it does to the token
count.

On the **resident** track that metric is meaningless — no expert bytes move —
and the ordinary one applies: accepted tokens per forward, against whatever the
rollback costs.

Quoting one at the other is how this page's original ranking went wrong.

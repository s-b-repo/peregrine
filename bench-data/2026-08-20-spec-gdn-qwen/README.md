# `COLI_SPEC_GDN` on resident Qwen3.5-27B — 2026-08-20

**Verdict: a measured net loss on this container, 1.55× slower, and the cause is
the MTP head's acceptance rate — not the recurrent snapshot the knob was gated
on.** `COLI_SPEC_GDN` stays off by default; nothing in production regressed.

This is the A/B that `docs/speculative-decoding-alternatives.md` listed as owed
when the knob landed earlier the same day.

## Setup

| | |
|---|---|
| Model | `/srv/modelstripe/qwen/Qwen3.8-27B-peregrine` — Qwen3.5-27B, int4, `Arch::HybridGdn` |
| Residency | `COLI_STREAM=0`, fully resident: 14.3 GB weights, peak 18.3 GB |
| Build | `target/release/peregrine-serve`, **no CUDA** (`gpu=unavailable`) — CPU only |
| Arms | `off` = `COLI_DRAFT=5 COLI_SPEC_CONF=0.65` (speculation refused: the arch needs the rollback); `on` = same **+ `COLI_SPEC_GDN=1`** |
| Load | B=1, one greedy request, `max_tokens=48` |
| Rounds | 3, **arm order rotated** (`off on` / `on off` / `off on`) |
| Isolation | fresh process per arm — `EngineKnobs` resolve once at spawn, so an in-process toggle would compare an arm with itself |
| Prompt | distinct per (arm, round): an exact greedy repeat is answered by the **response memo** without touching the engine |
| Tokens | counted server-side from `decode.tokens_emitted`, not from the text |

Box: 46 GB RAM, load average ~2.4 with a browser holding ~35 % of 12 threads —
the contention `measurement.md` warns about. That is why the spread below
matters as much as the gap.

## Result

```
off tok/s: [0.7307, 0.7856, 0.7879]   median 0.7856   spread  7.3 % of median
on  tok/s: [0.4931, 0.5064, 0.5278]   median 0.5064   spread  6.9 % of median

gap: 0.645x  (−35.5 %)   →  speculation is 1.55x SLOWER
```

**The gap (35.5 %) is five times the spread (~7 %), so it resolves.** Each arm
ate exactly one cold-page-cache run — `off` round 1 at 0.731 and `on` round 2 at
0.506 — which is what rotating the order is for.

| counter | off | on |
|---|---|---|
| rows per token | 0.979 | 1.229 |
| drafts proposed / accepted | 0 / 0 | 41 / 4 — **9.8 %** |
| `conf_stops` (floor cut the draft) | 0 | 131 |
| `gdn_replays` per 48 tokens | 0 | 11 (median) |
| `gdn_snapshot` per 48 tokens | 0 | 1 945 MB (median) |

`rows_per_token` of **0.979** in the off arm is the predicted sub-1.0 baseline:
a request's first token is sampled from the prefill's last position and costs no
decode row.

## What the cost is actually made of — and it is not what the knob was gated on

`COLI_SPEC_GDN_MAX_B` exists because the recurrent snapshot is ~3.1 MB per
linear layer, ≈151 MB per drafting sequence per tick, and that was expected to
be the thing that stopped this paying at width. The measurement confirms the
*size* and refutes the *conclusion*:

- 1 945 MB over 11–14 drafting ticks ≈ **160 MB per tick** — the 151 MB estimate
  was right.
- At ~8 GB/s that is **≈0.25 s** of memcpy across a run whose total overhead is
  **≈33 s**. Under 1 %.
- The 11 `gdn_replays` are 11 *extra forwards*. At the off arm's 1.27 s per tick
  that is **≈14 s**, the largest identified component of the overhead.

So the snapshot is not the problem; the **replays** are, and replays are a
direct function of acceptance. The risk note written when the knob landed had
the cost model backwards, and said the mitigation was tuning the confidence
floor. It is not: see below.

## Root cause: the head, not the floor

The floor was the obvious suspect — 131 `conf_stops` says it fired on nearly
every draft. So the floor was removed entirely and the depth dropped to 1, which
makes `accepted / proposed` the head's **raw top-1 agreement** with the model:

```
COLI_DRAFT=1  COLI_SPEC_CONF=0        proposed 42   accepted 4   → 9.5 %
```

**9.5 % with no floor at all.** The floor is not mis-tuned; the MTP head simply
disagrees with the main model about nine times in ten. For scale, the streaming
GLM-5.2 container measures **83 %** accepted at depth 5 with the same floor.

That is the failure signature the loader itself predicts, verbatim:

> A mis-loaded head cannot corrupt output: every draft it proposes is verified
> by the main model, so the failure mode is a **zero acceptance rate**, not
> wrong tokens.

### One hypothesis tested and rejected

`mtp_draft_with` applies the main model's `final_norm` to the incoming hidden at
`g == 0` before the head's own `hnorm`. That is GLM-shaped; the Qwen dialect's
`pre_fc_norm_hidden` may expect the pre-final-norm hidden directly. Patched out
for `HybridGdn` and re-measured:

```
with final_norm (shipped)   4 / 42  =  9.5 %
without final_norm          6 / 40  = 15.0 %      P(≥6 | rate 0.095) = 0.18
```

**Not significant**, so the change was reverted rather than shipped. The served
text was **byte-identical** across both, which is the safety property working:
a draft cannot change an emitted token, so this experiment was free.

### Still open

Candidates, none confirmed, roughly in order of how cheaply they could be ruled
out:

1. **Concat order into `mtp.fc`.** `[embed | hidden]` vs `[hidden | embed]` is
   shape-identical — `[5120, 10240]` either way — so a swap is completely
   silent and would produce exactly this symptom.
2. **The `final_norm` step above**, at a sample large enough to resolve 9.5 %
   from 15 % (needs ~500 proposals, not 40).
3. **The conversion.** `peregrine-import-hf` may be mis-mapping the head; the
   main stack is demonstrably fine, and the head is checked by nothing.
4. The norm convention was **checked and cleared**: `mtp.norm.weight` is stored
   1-centered (mean +1.25) and so is the main stack's final norm (mean +0.94),
   and both take the same zero-centered `+1.0`. Consistent, and the main model
   generates coherently.

## Reproducing

```bash
bash bench-data/2026-08-20-spec-gdn-qwen/ab.sh <outdir> 3 48
# per arm, isolated:
COLI_STREAM=0 COLI_DRAFT=5 COLI_SPEC_CONF=0.65 [COLI_SPEC_GDN=1] \
  target/release/peregrine-serve --model $QWEN --port 8202 --model-id qwen \
  --max-batch 4 --max-tokens 4096
```

`raw.jsonl` holds one line per arm-run; `*.before.json` / `*.after.json` are the
raw `/metrics` scrapes the deltas come from.

## Threats to validity

- **CPU only.** No CUDA in this build, so the per-forward cost is far higher
  than the production `serve-qwen.sh` path (`COLI_GPU=1 COLI_GPU_DENSE=1`).
  A cheaper forward makes the 11 replays cheaper too, so the *ratio* would
  shift — but the 9.8 % acceptance is a property of the head, not the device,
  and it is what makes this a loss.
- **B=1 only.** The snapshot is charged per sequence while a forward's weight
  read is shared, so the snapshot's share grows with batch width. It is under
  1 % of the overhead here; at B=32 it would be ~32× that against a forward
  that has not grown proportionally. Untested.
- **One prompt shape**, 48 tokens, three repeats. Enough to resolve a 35 % gap
  against a 7 % spread; not enough to resolve 9.5 % from 15 %.
- Browser contention throughout, identical in both arms and absorbed by the
  rotation.

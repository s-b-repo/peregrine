# Cross-engine review: what colibrì has that peregrine does not

Date: 2026-08-09 · Source: `~/colibri` @ `2026-07-16` (`issue_diskio.md`,
`issue_budget.md`, `c/glm.c`) · Compared against peregrine's measured state after
the io-claim fix (16.08 s/tok, io duty 84 %, 10.85 GB/token).

peregrine is a spin-off of colibrì, and `mlp.rs:5` records the port as *"minus the
streaming/tiering (M2), CACHE_ROUTE, and EXPERT_BUDGET opt-ins"*. Those two named
omissions turn out to be the two biggest ideas colibrì has, and both attack **bytes
read per token** — the quantity that bounds everything on this box.

Everything below is ranked by expected value against peregrine's *measured*
bottleneck, not by how clever it is.

---

## 1. `EXPERT_BUDGET` — cap the batch-union, not the per-position top-K

**colibrì measured: decode 0.18 → 0.33 tok/s (~2×), prefill 38.7 s → 8.9 s (~4×)**
on a 24 GB host with a tiny cache. Grounded in MoE-Spec (arXiv 2602.16052): *top 32
of 64 experts capture 93 % of routing weight*.

The mechanism, from `c/glm.c` `moe()`: after the batch-union `uniq[0..nu)` is built
but **before** the resolve/load loop, if `nu > EXPERT_BUDGET`, aggregate each unique
expert's gate weight across all positions that route to it, sort descending, keep the
top N, and strip the rest from every position's `idxs[]`/`keff[]` with renormalisation.
Dropped experts are *never resolved, never read, never computed*.

**Why peregrine needs this specifically.** peregrine has `COLI_ROUTE_MIN_SHARE`, and
it is **a different lever**: it trims within one position's top-K (colibrì's `TOPP`),
which at B=1 decode is the only union there is. But peregrine's two most expensive
regimes are exactly the ones with a *large* union:

| regime | per-layer union | peregrine's measured cost |
|---|---|---|
| decode B=1 | 8 | 10.85 GB/token |
| **prefill chunk** | **~69** | **~68 % of a short request** |
| batched decode B=16 | 7.76× a single (M2) | the whole point of batching |

`ROUTE_MIN_SHARE` cannot touch the cross-position union. A union budget attacks
prefill — which `peregrine-gen` measured at **143 s of a 197 s request** — head on.

Changes token values, so it needs `Model::prediction_flip_rate`, which peregrine
already has wired. Effort: colibrì did it in **54 lines, one file**.

## 2. `CACHE_ROUTE` — make the hit rate a decision instead of an outcome

This is the one that explains the number peregrine cannot reach. colibrì reports a
**27.6 % expert-cache hit rate**; peregrine measures **0.5 %** and every attempt to
raise it by capacity has failed (tripling the cache bought 3.8 % fewer reads).

`CACHE_ROUTE` (paper 2412.00099, "max-rank"), from `c/glm.c:1095`:

> Keep true top-J always; fill remaining slots preferring pin∪LRU experts ranked
> within top-M (or cumulative mass `ROUTE_P`).

So of top-8, the first `ROUTE_J` (default 2) are taken as the router demands, and the
remaining 6 slots are filled *preferentially from experts already in cache* — provided
they still rank inside a top-M window (default 12). `ROUTE_ALPHA` scales the
substitutes' gate mass before renormalisation; `ROUTE_AGREE=1` reports overlap % and
mean KL against the true top-K so the quality cost is measured rather than assumed.

**This inverts the problem.** Every cache strategy peregrine has tried takes routing as
given and tries to predict it — and M2 proved routing *is* predictable (33.55 % overlap
vs a 3.12 % null) without that helping, because a 2 GB cache against an 11.3 GB working
set evicts everything before reuse. `CACHE_ROUTE` instead biases routing toward what is
already resident. The hit rate stops being a property of the workload and becomes a
knob with a measurable quality cost.

Highest risk of anything here — it changes which experts compute a token — and highest
ceiling. `ROUTE_AGREE` is the honest gate, and peregrine has `prediction_flip_rate` for
the end-to-end check.

## 3. A pin tier separate from the LRU

colibrì carries `PIN`, `PIN_GB`, `PIN_FILL`, `AUTOPIN`, `REPIN` — a **byte-sized pinned
residency** distinct from the LRU, and `expert_is_resident()` probes `pin ∪ LRU`.

peregrine has `COLI_PREFETCH_PROTECT`, which is a *priority* within one budget, not a
separate tier with its own size. Given peregrine's measured finding that protection has
**opposite signs either side of one token's working set** (+193 hits below the
threshold, −40 % above it), an explicitly sized pin tier that holds a stable hot core
while the LRU churns underneath is a better-shaped mechanism than a priority bit.

Prerequisite for `CACHE_ROUTE`, which needs a residency probe to prefer against.

## 4. `DISK_SPLIT` — worth reading before the RAID0 lands

colibrì has a `DISK_SPLIT` knob peregrine has no analogue for. Given the array plan
(3-way RAID0 across SATA SSDs), how colibrì splits reads across devices is worth
reading before assuming md's striping is the right layer to do it at. **Unread — flagged,
not recommended.**

---

## Confirmations and negative results — do not spend time here

**peregrine already wins on read coalescing.** colibrì issues **4 reads per expert**
(one ~19 MB O_DIRECT `pread` for gate/up/down, plus 3 tiny scale `pread`s).
peregrine's `COLI_EXPERT_MERGE` issues **2** (one scales run, one weights run). Its own
`issue_diskio.md` lists expert coalescing as a *non-opportunity* — "don't change this".
peregrine is ahead; nothing to take.

**mmap for MoE: a measured regression, twice.** llama.cpp discussion #18758 is widely
cited for "mmap ≥10× faster than O_DIRECT for MoE", and colibrì implemented the mmap
expert path on that basis — then **reverted it as a measured regression**. The llama.cpp
result holds when the model fits in ~RAM, where the page cache serves re-faults for free.
At 358 GB against 46 GB it does not apply. peregrine has no mmap path; **it should not
grow one**, and this is why.

**O_DIRECT: the two engines disagree, and peregrine has the measurement.** colibrì uses
O_DIRECT by default and records it as settled. peregrine measured it at **0.86 vs 1.12
GB/s (−23 %)** with repeats, and defaults it off. Keep peregrine's default; the buffered
arm keeps kernel readahead that O_DIRECT discards.

**Speculative decode: colibrì's code agrees with today's measurement, not with our docs.**
`c/glm.c` on `DRAFT`: *"measured on the real run (2026-07-03) acceptance ~5 % → every
rejected draft still pays for its experts from disk = ~3× slower"*, hence default OFF.
peregrine's `configuration.md` recommended "use 4–6" until today, when `COLI_DRAFT=4`
measured **1.57× slower** for precisely that reason. Two independent engines, same
conclusion. Already corrected.

**The `/proc/meminfo` fopen storm is not an opportunity here.** colibrì ranks it medium-ROI
(re-read every ~16 tokens). peregrine reads it the same way, but at 16 s/token that is
~3 syscalls per 256 seconds. Genuinely nothing.

---

## Worth a look, unevaluated

- **`GRAMMAR` as a third draft source.** colibrì uses grammar-forced bytes (JSON keys,
  punctuation, enum values) as drafts at acceptance ≈ 1 — free tokens on constrained
  output, and they work where an int4 MTP head does not. Since speculation on a
  disk-bound engine only loses when drafts are *rejected*, an acceptance-1 source
  sidesteps the exact failure mode both engines measured.
- **`COUPLE` / `COUPLE_D` / `COUPLE_K`** — co-activation coupling. peregrine has
  `COLI_HYPER_SCHED` and `COLI_FUSE_THRESHOLD`; whether these are the same idea is
  unchecked.
- **`PILOT_TWO` / `PILOT_REAL`** — colibrì's two-tier speculative prefetch. peregrine's
  own prefetch is measured at 0.2 % yield, so the shape of a working one is of interest.

## Suggested order

1. **`EXPERT_BUDGET`** — smallest change, largest measured win, attacks prefill which is
   the dominant cost, and the quality gate already exists.
2. **Pin tier** — cheap, useful alone, and the prerequisite for (3).
3. **`CACHE_ROUTE`** — the only idea that can move a 0.5 % hit rate, gated on `ROUTE_AGREE`.

All three change token values. None should land without `prediction_flip_rate` numbers
next to the throughput numbers, and per
[`docs/measurement.md`](docs/measurement.md), none should be believed from a single run.

# What causes the ~0.4 % warm-cache hit rate — measured, 2026-08-09

Box: Ryzen 5 5500 (6C/12T), 46 GB RAM, LUKS NVMe · Model: GLM-5.2 int4, 358 GB
container, 78 layers (75 sparse), 256 experts/layer, top-8 · CPU-only
`peregrine-serve`, B=1.

The question was how to improve `hit_rate=0.4 %` and `prefetch accuracy=21.9 %`.
Four hypotheses were tested. **Three were refuted, including the one that looked
strongest going in**, and the fourth turned out to be partly an artefact of how the
measurement was set up.

| file | what it settles |
|---|---|
| `M1-storage-config.md` | scheduler already `none`; LUKS sector size still unverified (needs root) |
| `M2-routing-structure.md` | **routing overlap 33.55 % vs a 3.12 % independence null** — routing *is* predictable |
| `M3-prefetch-arms.md` | prefetch is the only source of hits at 4 GB; it issues ~420 reads to save 0–30 |
| `M4-capacity.md` | **cache size is the lever**: 4.29 → 12.88 GB gives 2.9× hits and −3.8 % disk reads |
| `M5-io-engine.md` | pread ≈ io_uring (1.20 vs 1.19 GB/s); the recorded 2.4× gap was an O_DIRECT confound |
| `unbounded-queue-blocks-shutdown.md` | shutdown drains 93 GB of reads nothing will use |

## The answer

**Cache capacity, and nothing else tested here.** At 8 decode tokens with prefetch
off in both arms:

| cache | slots | hits | hit_rate | disk_reads |
|---|---:|---:|---:|---:|
| 4.29 GB | 227 | 193 | 1.9 % | 9751 |
| **12.88 GB** | **681** | **564** | **5.7 %** | **9380** |

681 slots is above one decode token's 600-expert working set; 227 is well below,
and the hit rate tracks the threshold. **This is the only change tested that
reduced `disk_reads` at all** — 371 fewer reads, for free, where prefetch issues
~420 to save ~20.

The user's framing was right: this is the "needs more RAM" branch, and the number
is **~11.3 GB per concurrent decode stream** (600 experts × 18.9 MB). Below it,
cross-token reuse is structurally impossible however good the predictor is; above
it, reuse appears immediately.

## The one unambiguous result

**Disk reads are constant to 0.5 % across every configuration tested** — 6314 to
6344, across 4 GB and 12.9 GB caches, prefetch on and off, protection on and off.
Seven runs, one number. Nothing in the caching or prediction machinery changed how
many bytes this request moved.

Against that, prefetch issues ~420 extra reads per request and saves between 0 and
30. **That trade is 14:1 against at its best observed value**, and it is the only
conclusion here robust enough to act on: the hit counts themselves are noise
(`default` gave 2, 19 and 30 hits on three identical runs), while read counts are
counts, not rates, and survive a contended box.

## What was refuted

1. **"Routing is too high-entropy to predict."** The repo carried this as an
   inference and flagged it as unmeasured. Measured: 33.55 % consecutive-token
   overlap against a 3.12 % null. Refuted.
2. **"Prefetch poisons the eviction order."** `protect_from` does give a never-used
   speculative slab priority over a routed demand slab — the code does exactly
   that. Turning protection off did not raise the hit rate; it produced the lowest
   non-zero count observed. Unsupported.
3. **"The io_uring lane is 2.4× slower than blocking pread."** Controlled for the
   O_DIRECT confound (`pread` silently disables it), the two are the same rate:
   1.20 vs 1.19 GB/s. The recorded gap compared uring-with-O_DIRECT against
   pread-without. Refuted.

The capacity hypothesis initially looked refuted too — a 12.9 GB cache moved the
hit rate not at all — and that turned out to be the *measurement's* fault, not the
hypothesis's. See below.

## What the design got wrong, and it matters

The first six runs used a **2-token completion on a 12-token prompt**. Prefill is
~5 144 of the 6 344 cache lookups — **81 % of the measurement** — and prefill has
no cross-token reuse by construction: each expert in a chunk's per-layer union is
read once. With only two decode tokens there is exactly **one** cross-token
transition available.

So even flawless reuse could not have exceeded ~3 % hit rate on this workload, and
the "0.4 %" being investigated is substantially a property of the request shape,
not of the cache. **Repeating the capacity comparison at 8 decode tokens
(`M4-decode8/`) is what turned a refuted hypothesis into the answer** — same two
cache sizes, same code, 193 → 564 hits.

This is worth stating plainly rather than quietly re-running: the headline number
this investigation started from is not wrong, but it is not the number anyone
thought it was, and it was measured on a workload that could not have shown the
effect being looked for.

## Standing caveats

- Single run per arm, except `default` (three runs, spread 2–30 hits).
- No passwordless sudo, so the OS page cache cannot be dropped between arms. This
  affects wall clock and not `hits`/`misses`/`disk_reads`, which are counters
  internal to peregrine's own cache.
- One arm (`default`, contended) overlapped a stale process draining 93 GB; kept
  and labelled rather than discarded, because with `rerun-clean/` it is the only
  estimate of run-to-run spread available.

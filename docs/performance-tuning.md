[« Docs index](README.md)

# Performance tuning

"Decode is slow — what do I actually check?"

This page walks the levers in **measured** order, with the numbers. It is the
counterpart to [Configuration](configuration.md), which is an alphabetical
reference: that page tells you what a knob does, this one tells you whether it is
worth turning. Every figure here traces to a file under `bench-data/`.

Before changing anything, read [Measurement discipline](measurement.md) — most of
the wrong answers in this repo's history came from a plausible number taken once.

---

## Step 0: find the ceiling before you tune

Decode on a streaming MoE is bytes over bandwidth. Establish both before touching a
knob, because they bound everything else:

```bash
# 1. what one token has to read — printed at shutdown
[workingset] 10.85 GB per token; prefetch-protect on (budget cannot hold a pass)

# 2. what the device gives this engine's own reactor
./target/release/examples/iobench <shard> 64 8 4 0 uring 15
median 1.12 GB/s over 15 reps (min 1.11, max 1.17, spread 5.4% of median)
```

`10.85 GB ÷ 1.12 GB/s = 9.7 s` — **the floor**. No scheduler change goes below it.
Measured decode is 16.08 s/tok, so ~60 % of a token is the device and the rest is
everything else.

Then get the split from the shutdown block:

```
[lane] moe wall 40.6s over 3 forwards (13.53s each); io duty 84% of 4 rings, cpu 0.5 workers busy
```

**`io duty` is the number that matters.** Below ~50 % the lane is starved and the
problem is upstream of the disk; near 90 % you are genuinely device-bound and the
only remaining levers are fewer bytes or a faster drive.

---

## The levers, in measured order

| lever | effect | status |
|---|---|---|
| **io claim batch** | **21.83 → 16.08 s/tok**, io duty 24 → 84 % | shipped |
| Fewer bytes (`--tier-hot-frac` / int2) | **49 %** fewer bytes; compounds with the drive | needs space for a 2nd container |
| Faster device | at 3.4 GB/s the floor drops 9.7 → 3.2 s | needs hardware |
| `COLI_FUSE_PREFILL=1` | 2 streams: **322 s vs 356 s**, output byte-identical | **adopt** |
| Cache capacity (`COLI_ECACHE_GB`) | 3× cache → 3.8 % fewer reads, ~1 % wall | marginal |
| Prefetch | **keep it on** — off measured *slower* | default |
| `COLI_FORCE_ASYNC` | no resolvable effect | leave alone |
| `COLI_DRAFT=4` | **1.57× slower** | **reject** |
| `COLI_IO_RINGS=8` | refuses to load — RAM-capped | unavailable here |

### Shipped: the io claim batch

The single biggest software win found so far, and it was a bug. Rings claimed work
in fixed batches of `COLI_IO_BATCH` (16) while a decode token routes **8 experts per
layer**, so one ring took the whole layer and the other three broke out without
reading. Detail in [the concurrent scheduler](concurrent-scheduler.md#the-three-lanes).

Nothing to configure — but if you change `COLI_IO_BATCH` by hand, remember it is now
an **upper bound**, not the claim size.

### Fewer bytes: the only lever that is not capped by the drive

```bash
peregrine-requantize "$MODEL" "$OUT" --dry-run --target int2
# 383.73 GB -> 195.28 GB (50.9% of source)
```

Halving bytes halves the floor, and it multiplies with a faster drive rather than
competing with it. Lossy — gate on `Model::prediction_flip_rate`. See
[Tools](tools.md#peregrine-requantize), including why a flat `--tier-hot-frac`
sweep means your heat profile is too thin rather than the tiering being useless.

### `COLI_FUSE_PREFILL=1`: adopt it, but know what it was measured on

Without it a mixed tick runs prefill and decode as **two disjoint forwards**, each
streaming its own ~11 GB routed union. With it they share one forward.

Measured on two concurrent streams: **322 s against a 356 s baseline**, and stream
one's completion was byte-identical to the single-stream baseline — so the
documented output-neutrality has an empirical check, not only its two unit tests.

**The 1.11× understates it.** That test used `max_tokens=4`, which is ~68 % prefill
(5,144 expert lookups against 2,400), and union sharing is a *decode* effect. Re-run
at 16+ decode tokens before drawing conclusions about batching.

### Rejected: `COLI_DRAFT=4`

`configuration.md` recommends 4–6 on the grounds that the "MTP is a net loss"
figures were taken at depth 2. **On streaming GLM-5.2 that does not hold** — depth 4
measured p50 24.3 s/token against 15.5 s without it, a **1.57× regression**:

```
gaps: [20.2, 45.7, 52.5, 48.2, 0.0, 24.3, 20.4] s     6 forwards -> 8 tokens
```

Speculation genuinely accepts — the `0.0` is two tokens from one forward — but each
forward verifies γ+1 rows and its routed union grows faster than acceptance repays.
Output is sequence-identical at temperature 0, as documented; it is simply slower.
The advice may still hold where experts are **resident** rather than streamed.

### Pending measurement: the 2026-08-15 ds4-port wave

Four knobs ported from antirez's ds4/DwarfStar engine landed 2026-08-15, every
one **off by default** until its A/B (running in `scripts/overnight-2026-08-15.sh`,
results → `bench-data/2026-08-15-*/`) licenses a change. Listed here so nobody
mistakes "documented" for "measured":

- **Asymmetric expert requantization** — `peregrine-requantize --down keep
  --keep-last-layers 6 --target int3-g64`: gate/up to int3-g64, `down_proj`
  byte-identical, last 6 expert layers (incl. the MTP row) untouched. The one
  untested point on the sub-int4 ladder after todo.md §13 closed every uniform
  rung (uniform int3-g64: flip_rate 0.514). Predicted 383.73 → 355.25 GB
  (−7.4 % container bytes); gate before trusting.
- **`COLI_SPEC_CONF`** (default 0, off) — stop an MTP draft early when the
  head's top-token probability drops under the floor. Depth-only: `accept_run`
  is untouched, so greedy output is bit-identical by construction (test:
  `the_confidence_floor_never_changes_a_greedy_stream`). Rationale: every
  drafted token is a verify row streaming expert bytes; the `COLI_DRAFT=4`
  rejection above is exactly the cost this floor prunes. ds4 ships 0.6–0.7.
- **`COLI_ECACHE_GB=auto`** (+ `COLI_ECACHE_AUTO_FRAC`, default 0.80) — ds4's
  budget rule: a fraction of post-load `MemAvailable`, still capped by the
  transient reserve + 1 GiB safety. Numeric spellings unchanged.
- **`COLI_KV_STORE_DIR`** (+ `COLI_KV_STORE_MB` cap, default 16384;
  `COLI_KV_STORE_TRIM`, default 32) — disk-persisted KV sessions: completed
  prefixes ≥ 256 tokens checkpoint to disk and a restarted server restores
  them instead of re-prefilling (~176 KiB stored per token vs ~10.85 GB of
  expert reads re-streamed). Fingerprint + checksum + full-token compare gate
  every load; a bad file means a cold prefill, never a wrong token. Note the
  checkpoint files contain the session's token ids in the clear — point the
  knob at storage with the same privacy expectations as the server log.

### Unavailable here: `COLI_IO_RINGS=8`

Streaming buffers scale with ring count — 11.5 GB at 4 rings, **21.1 GB at 8** — and
the pre-load RAM guard refuses rather than being OOM-killed mid-load:

```
[ram] avail 27.4 GB | resident 10.9 + KV 0.7 + stream 21.1 + slack 1.4 -> peak 36.3 GB
Error: ... 6.8 GB short, so the kernel would OOM-kill this run part-way through loading
```

On a 46 GB box with a VM resident, ring count is capped at 4 by **RAM**, not by the
device.

---

## Things that look like levers and are not

- **Bigger warm cache.** The expert pool is 256/layer × 75 sparse layers ×
  18.9 MB ≈ **363 GB**; one token routes 11.3 GB of it. Tripling a 4.29 GB cache
  bought 3.8 % fewer reads and moved decode 356 → 353 s. A cache larger than half
  of `MemAvailable` makes it *worse* — a hit that has been paged out is a page
  fault, and `ram.rs::cache_cliff_warning` says so at load.
- **VRAM as an expert tier.** 12 GB soldered, ~10.4 GB free — *smaller than one
  token's 11.3 GB working set*, and smaller than the 12.88 GB RAM cache that only
  reached 5.7 %. GPUDirect Storage would not change the arithmetic and is
  unavailable anyway (`nvidia-fs` absent).
- **Turning prefetch off.** It measured **slower** (23.8 vs 21.9 s/tok), and
  `disk_reads` is identical either way — `prefetch_reads` is a *subset* of
  `disk_reads`, not additional I/O. The "issues 420 reads to save 20" reading
  counted the same bytes twice.
- **A better router/predictor.** Routing is not the problem: consecutive tokens
  share **33.55 %** of routed experts against a 3.12 % independence null. The low
  hit rate is capacity, not entropy.

---

## Where the remaining headroom is

At 84 % io duty the lane is close to saturated but not at the device's rate:
10.85 GB in 13.53 s of MoE wall ≈ **0.80 GB/s**, against **1.12 GB/s** from
`iobench` on the same drive at 4 rings. **Roughly 1.4× is still in the I/O path.**
Current suspects, in the order the measurements point:

1. Submit depth per claim — a decode claim is now 2 experts, and with
   `COLI_EXPERT_MERGE` on that is ~4 reads per ring, against `iobench`'s 8-deep.
2. The warm-cache lock: taken per expert by every io thread, and the cache runs
   100 % full so each miss also evicts.

Past that, the floor binds and the answer is fewer bytes or a faster drive.

## Related pages

[Measurement discipline](measurement.md) · [Configuration](configuration.md) ·
[Benchmarks](benchmarks.md) · [Prefetch & caching](prefetch-and-caching.md) ·
[I/O & storage](io-and-storage.md) · [Tools](tools.md)

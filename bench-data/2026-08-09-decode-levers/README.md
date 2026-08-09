# Decode levers, measured — one bug fix worth 1.36×, and three levers that do not pay

Date: 2026-08-09 · GLM-5.2 int4 · root LUKS/NVMe · measured with `peregrine-gen`
(separates ttft from decode, reports `min/p50/p95/max`) and the `[lane]` duty-cycle
line added the same day.

## The finding: one ring was doing four rings' work on every decode token

`concurrent.rs`'s I/O lane claims work with

```rust
let start = io_work_ref.fetch_add(batch, Ordering::Relaxed);
if start >= n_plans { break; }        // no work left for this ring
```

`batch` was `experts_per_batch()` — `COLI_IO_BATCH`, default **16**. A decode token
routes **8 experts per layer**. So ring 0 claimed all 8 and rings 1–3 got starts
16/32/48, every one `>= n_plans`, and **broke without issuing a single read**.

The default is correct for prefill, where a chunk's routed union is ~69 experts per
layer. It collapses decode to one ring. Fixed by ceil-dividing the claim while keeping
the configured value as a ceiling:

```rust
let n_rings = reactors.len().max(1);
let batch = experts_per_batch().min(n_plans.div_ceil(n_rings)).max(1);
```

Decode gets `ceil(8/4) = 2` and all four rings run; prefill gets `ceil(69/4) = 18`,
clamped back to 16, so its deep submits are unchanged.

| at defaults | before | after |
|---|---|---|
| decode | 21.83 s/tok | **16.08 s/tok** |
| p50 inter-token | 21.9 s | **15.5 s** |
| io duty | 24% of 4 rings | **84%** |
| moe wall / forward | 19.38 s | 13.53 s |

**1.36×.** Independently reproduced before any code change by setting `COLI_IO_BATCH=2`,
which gave 14.80 s/tok and 90% duty — the env probe and the fix agree.

### How it was found

Not by reading the code. `io_us`/`cpu_us`/`gpu_us`/`reduce_us` are summed over *threads*,
so they cannot distinguish a saturated lane from an idle one. Adding
`LaneTimings::lane_wall_us` — the **wall** clock of the 3-lane region — turns them into
duty cycles, and `io duty 24% of 4 rings` pointed straight at the claim loop. The four
counters had been collected since the lane was written and read only by the bubble tuner,
which `snapshot_and_reset`s them every forward.

## Levers measured after the fix

| lever | result | verdict |
|---|---|---|
| `COLI_FUSE_PREFILL=1` | 2 streams, 8 tokens: **322 s vs 356 s** | **adopt** — 1.11%, free, output byte-identical |
| `COLI_DRAFT=4` (MTP) | p50 **24.3 s vs 15.5 s**, mean 30.2 vs 15.5 | **reject** — 1.57× slower |
| `COLI_IO_RINGS=8` | **refuses to load** on this box | **unavailable** — RAM-capped |

### `COLI_FUSE_PREFILL=1` — small, positive, and provably output-neutral

Two concurrent streams, 4 tokens each: **322 s against a 356 s baseline**. Stream 1's
completion is byte-identical to the single-stream baseline, so the documented
output-neutrality now has an empirical check and not only its two unit tests.

**The 1.11× understates it, and the test is the reason.** At `max_tokens=4` these runs are
~68% prefill (5 144 lookups against 2 400), and union sharing is a *decode* effect — M2's
7.76×-for-16 is about decode. A prefill-dominated request structurally cannot show it.
Re-run at 16+ decode tokens before concluding anything about batching.

### `COLI_DRAFT=4` — a loss, and the documented advice is not supported

`docs/configuration.md` says *"Use 4–6: the 'MTP is a net loss' figures were taken at depth
2"*. At depth 4 it is **worse**, not better:

```
gaps: [20.2, 45.7, 52.5, 48.2, 0.0, 24.3, 20.4] s     6 forwards -> 8 tokens
```

The `0.0` is two tokens from one forward, so speculation **is** accepting — 1.33 tokens per
forward. But each forward verifies γ+1 = 5 rows and its routed union grows faster than
acceptance repays: the 45–52 s forwards are the cost. Output prefix-matches the baseline, so
it is sequence-identical as documented; it is simply slower. Single run, but 1.57× on p50 is
far outside this box's ~20% spread.

### `COLI_IO_RINGS=8` — bounded by RAM, not by the device

```
[ram] avail 27.4 GB | resident 10.9 + KV 0.7 + stream 21.1 + slack 1.4 -> peak 36.3 GB
Error: ... 6.8 GB short, so the kernel would OOM-kill this run part-way through loading
```

Streaming buffers scale with ring count — 11.5 GB at 4 rings, **21.1 GB at 8** — and the
pre-load guard refuses rather than being killed mid-load. **Ring count is capped at 4 on this
box** (46 GB with an OSX-KVM guest holding ~8), so the ring-count lever is closed until
either the buffers shrink or the VM stops.

## Where decode stands

At 84% io duty the lane is close to saturated: 10.85 GB in 13.53 s of MoE wall ≈ **0.80
GB/s**, against the **1.12 GB/s** `iobench` gets on the same drive at 4 rings. So roughly
**1.4× remains in the I/O path**, and past that the device floor (9.7 s/token) binds.

## Still untested

`COLI_GPU=1` (gpu was 0.0 s in every run; CUDA now builds), `COLI_KV_DTYPE=f16`, and a
proper median-based re-measurement of prefetch on/off — the current claim that prefetch
*helps* ~8% rests on one run per arm.

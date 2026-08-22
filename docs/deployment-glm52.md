[« Docs index](README.md)

# Deploying GLM-5.2 on cortix: five drives, no RAID, CUDA

This is the concrete deployment this repo was written for — a 744B MoE that does
not fit in RAM, streamed off every drive in the box at once. It records what the
machine actually is, what was measured on it, and which knobs earned their place.

## Why no RAID

The box previously ran `md0`, a RAID0 of `sdb`+`sdd`. Measured, that array reads
**1.0 GB/s** — and its two members read **514** and **547 MB/s** individually.
So RAID0 bought nothing over addressing the drives directly, and cost two things
that matter here:

- **One queue instead of two.** peregrine's I/O lane claims expert batches off a
  per-device cursor with cross-device work stealing (`COLI_IO_DEVICE_SCHED`). An
  md array presents a single block device, so the scheduler cannot see that there
  are two spindles behind it and cannot steal between them.
- **`n × slowest member`.** md stripes evenly. That is the wrong shape for a box
  whose drives span 91 MB/s to 669 MB/s — a stripe including the HDD would run
  the whole array at HDD speed. peregrine instead splits **bandwidth-
  proportionally**, so a slow drive is given proportionally less to do and never
  becomes the bottleneck.

RAID0 also couples failure: one drive death loses the array. For weights that are
re-downloadable at ~1.5 MB/s (≈20 days for this model) that is not free.

## The drives

Measured 2026-08-22, `dd iflag=direct bs=1M count=3000`, caches dropped between:

| Device | Hardware | Read | Share | Model bytes |
|---|---|---:|---:|---:|
| `nvme1n1` | Intel 600p 128 GB, **CPU-direct M.2** | 669 MB/s | 28.5 % | 103.6 GB |
| `sdd` | Crucial BX500 932 GB SATA SSD | 547 MB/s | 23.0 % | 83.7 GB |
| `sda` | Kingston A400 894 GB SATA SSD | 537 MB/s | 22.7 % | 102.8 GB¹ |
| `sdb` | Acer RE100 954 GB SATA SSD | 514 MB/s | 21.9 % | 79.4 GB |
| `sdc` | Hitachi 699 GB **HDD**, 5400 rpm @ 3 Gb/s | 91 MB/s | 3.9 % | 14.2 GB |
| | | **2.36 GB/s** | | **383.7 GB** |

¹ `sda` also holds the 20.5 GB trunk, which goes to group 0.

**`sde` (WD 1 TB) is deliberately excluded.** It reports **139
`Current_Pending_Sector`** — sectors that failed to read and await reallocation —
and logged 183 kernel errors in one boot. SMART's overall verdict is still
`PASSED` because pending sectors do not trip that flag; do not read it as health.
A pending sector under a model shard surfaces as a read that retries and stalls,
which `COLI_IO_RECOVERY` would mask as slowness rather than failure. There is no
capacity reason to take the risk: the five drives above hold 3.6 TB for a 384 GB
model.

**The OS drive is excluded on purpose.** It is the *fastest* device here
(LUKS NVMe, ~1.12 GB/s), so excluding it costs ~15 % of aggregate read bandwidth
— 2.36 GB/s against 2.79. That is paid deliberately: saturating the root NVMe
previously drove I/O pressure to `some avg60=52 %` and put plasmashell into
uninterruptible sleep, which presents exactly like "KDE has hung". Keeping the
model off root keeps the desktop responsive under load, and frees the fastest
drive for the KV session store, where it is not competing with expert streaming.

## The split

Each layer's experts are divided across all five groups, so a single layer's read
hits all five drives at once:

```bash
peregrine-reshard \
    --model /path/to/GLM-5.2 \
    --out   /srv/m-sda/GLM-5.2-r5 \
    --groups sda:537,sdb:514,sdc:91,sdd:547,nv1:669 \
    --verify
```

Weights are the measured MB/s; the tool normalises them. Splitting by *layer*
instead — all of layer 10 on one drive — would be much worse: layers are streamed
sequentially, so only one drive would be active at a time and aggregate bandwidth
would collapse to that of a single device.

`--verify` byte-compares every tensor against the source after writing. It
roughly doubles the run (~2 h write + ~1 h verify here, bounded by the DRAMless
A400's sustained write). Worth it: a silently corrupt weight shard produces
plausible garbage, not an error.

Afterwards each group's files move to their drive and `model_paths.json` lists
the rest. `SafeTensors::open` merges the directories; the group name in a
filename is documentation, not addressing.

## The knobs, and what licenses each

Set by [`scripts/serve-glm52-cuda.sh`](../scripts/serve-glm52-cuda.sh). Every one
cites its measurement; the script also lists the knobs that *sound* like wins,
with the number that disqualified them, so they are not re-enabled from first
principles.

| Setting | Effect | Evidence |
|---|---|---|
| `--max-batch 16` | **4.4×** aggregate | 0.056 → 0.244 tok/s. The knee is at or below B=32 |
| `COLI_KV_STORE_DIR` | **6.4–6.7×** *prefill* | cold 2620.5 s → warm 391.9 s, output byte-identical. The biggest win for coding, where a long file is resent every turn. Does nothing for decode |
| `COLI_DRAFT=5` **+** `COLI_SPEC_CONF=0.65` | **+37 %** | 0.060 → 0.082 median, REPEATS=3; 22 % fewer expert reads. **These travel together** — `COLI_DRAFT` alone measured **1.57× slower** (24.3 vs 15.5 s/tok) because the read union grew 2.63× |
| `COLI_GPU=1` | **1.09 %**→1.09× | 0.244 vs 0.224, 3 reps. The win is lane *overlap*, not residency: 62 residents against ~19 200 experts routed per step is 0.3 % |
| `COLI_PREFETCH_STALE_DROP=1` | **+6.9 %** | 0.072 → 0.077 median, REPEATS=3; speculative reads −68 % |
| `COLI_ECACHE_GB=8` | best in-tree B=16 number | 0.088 tok/s. **Never `=auto`** (OOM-killed at 0/16 streams), never past ~half MemAvailable |
| `COLI_IO_BATCH=8` | decode 21.83 → 16.08 s/tok | io duty 24 % → 84 %. 8 rather than 16 cuts the stream reserve 11.5 → ~6.4 GB |

### Accuracy

One setting here changes token values: **`COLI_GPU=1`**. GPU experts compute in
**f32** where the CPU path is int4, so the GPU arm is *more* accurate but not
bit-identical to a CPU baseline. If you need bit-exactness against the CPU
reference, unset it and lose the 1.09×.

Everything else above is output-neutral by construction and by test.
`COLI_ROUTE_MIN_SHARE` is the knob that trades accuracy for bytes, and it is off:
at τ=0.05 it cut 12.5 % of reads and flipped **27.9 %** of top-1 predictions.

### Unmeasured, and flagged as such

`COLI_IO_DEVICE_SCHED=1` is enabled because this is the exact topology it was
written for — five devices spanning 91 to 669 MB/s, where a device-blind cursor
lets the HDD straggle and hold up a wave. **No bench-data arm has ever run it.**
A/B against `COLI_IO_DEVICE_SCHED=0` before believing it.

## Where the ceiling is

One token routes ~11 GB of experts. At 2.36 GB/s aggregate that is a **~4.6 s
floor per token** from the device alone, before any compute. No scheduler change
goes below it, and no cache that fits in 46 GB changes its shape.

The only lever that is not capped by the drives is **fewer bytes** — and every
sub-int4 rung has failed its quality gate here (int2-g64 `flip_rate = 1.000`,
asymmetric int3-g64 `0.447`). That is the open problem, not the scheduler.

## Serving

Bearer auth is mandatory: the script refuses to start without
`PEREGRINE_API_KEY`, because the Cloudflare tunnel makes the port reachable and
an unauthenticated endpoint at ~0.08 tok/s is exhausted by a single request.
Ingress is path-restricted at the edge — `/metrics` is 404'd there, not merely
unadvertised.

## Using it for coding — read this before wiring up a client

The endpoint is OpenAI-compatible (`/v1/chat/completions`, SSE streaming), so any
OpenAI-shaped client works — aider, Continue, Cline, opencode:

```bash
export OPENAI_BASE_URL=https://glm.cybersec.org.za/v1
export OPENAI_API_KEY=<the key in ~/.config/peregrine/glm52.env>
```

**Measured on this box, 2026-08-22, after the migration:** a 24-token completion
took **453 s — 18.9 s/token**, prefill included. That is not an estimate; it is
the number the deployed stack produced. One decode step moves **11.12 GB** and
splits `io 8.7 s wall (43.5 s summed across the 5 rings) / cpu 2.8 s / gpu 0.016 s
/ reduce 0.17 s`. The device is ~75 % of the step, exactly as the floor predicts.

(An earlier draft of this page predicted ~8 s/token by halving a figure measured
on the old storage. That was optimistic by more than 2x. 18.9 s/token is the
measured one.)

That is **~95 minutes for a 300-token code edit**. Batching does not rescue
interactive use: at B=16 the batch-union means 16 streams share one set of expert
reads, so *aggregate* throughput rises 4.4x while each individual stream gets
*slower*. Batching is for serving many requests at once, not for making one
request fast.

So:

- **Interactive coding assistant: no.** Not at 18.9 s/token. No knob changes
  this, because it is the device floor and not the scheduler — `gpu_us` is
  0.016 s against `io_us` of 8.7 s wall.
- **Batch / async work: yes.** Submit a refactor or a review, come back later.
  This is what the model is good for on this hardware, and B=16 is right for it.
- **For interactive work, use a resident model instead.** Anything that fits in
  VRAM+RAM without streaming will be two to three orders of magnitude faster.

The one lever that *would* change this is **fewer bytes per token**, and every
sub-int4 rung has failed its quality gate here (int2-g64 `flip_rate = 1.000`,
asymmetric int3-g64 `0.447`). Until a rung passes, 10.85 GB/token is fixed and so
is the floor.

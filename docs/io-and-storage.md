[« Docs index](README.md)

# I/O & storage

`peregrine-io` owns every byte that moves between disk and RAM: an io_uring
reactor, the aligned slab pool, the warm expert cache, memory hints
(hugepages/NUMA), the topology probe, sensors, and perf counters. Design rule:
**minimal syscalls, zero userspace copies of bulk weight bytes**.

## The io_uring reactor (`ring.rs`)

A **custom single-owner reactor thread** built directly on the `io-uring`
crate — deliberately not `tokio-uring`, whose per-op Future model fights the
batched-submit / io-wq-worker-cap ownership the design needs.

- **Registered files** (`IOSQE_FIXED_FILE`): the streaming rings register
  every shard fd (buffered + O_DIRECT twins, `SafeTensors::shard_fds`) right
  after construction, so read SQEs name a fixed-file slot instead of
  re-resolving the fd each time. Rings run `COOP_TASKRUN`; **not**
  `SINGLE_ISSUER` — the scheduler's lane threads are respawned per layer, and
  a single-issuer ring would reject the new task with `EEXIST`.
- **Owned-completion lane** (`COLI_IO_COMPLETION`, default on): the demand
  path submits each claimed expert's regions with Reactor-owned landing
  buffers and reaps completions incrementally, forwarding every expert to the
  CPU pool the moment its last region lands — no whole-wave barrier, so
  compute on expert 1 overlaps the reads of experts 2..N. Bytes are
  byte-identical to the wave path (same regions, same carve); only delivery
  timing changes, which the `pos`-keyed reduce absorbs by construction.
  `COLI_IO_COMPLETION=0` restores the blocking wave;
  `COLI_IO_ENGINE=pread|regbuf` also implies the wave (those measurement arms
  are wave-shaped on purpose).
- **SQPOLL** (`COLI_SQPOLL=1`, default off): a kernel submission-polling
  thread that removes even the one enter syscall per submit batch
  (unprivileged since kernel 5.13). Opt-in because the poll kthread
  busy-polls a core until `COLI_SQPOLL_IDLE_MS` (default 2000) of quiet —
  usually the wrong trade on a CPU-contended box; the bench arm decides.
  `COLI_SQPOLL_CPU` pins the kthread. Streaming rings only, with graceful
  fallback to the plain `COOP_TASKRUN` ring if the kernel refuses. Under
  SQPOLL the tuner's `sq_full` signal reads near-zero (the kthread drains
  the SQ continuously); the read-µs EWMA remains the live signal.
- **Registered buffers**: `register_read_buffers()` + `IORING_OP_READ_FIXED`
  serve the `COLI_IO_ENGINE=regbuf` measurement arm (`read_fixed_many`, one
  in-flight op per registered buffer, sized from the batch's largest region
  and grown once). Fixed buffers skip per-I/O page pinning but copy out of
  the registered landing buffer, where `read_many` has the kernel write the
  caller's buffer directly — at ~6 MB expert regions the copy plausibly costs
  more than the pinning saves, so this stays a measurement engine, not a
  default.
- **Mirror striping** (`COLI_IO_MIRROR_ROOTS`): one logical range splits into
  one chunk per mirror drive inside a single `read_striped` submit
  (`StripedRead`), so every drive reads concurrently; sub-`n_drives×128 KiB`
  ranges ride whole, round-robin. Buffered-only, byte-identical to the
  `pread_striped` oracle; mirror fds register as fixed files next to shard
  fds. The trunk loader (`SafeTensors::read_at`) and the MoE wave lane stripe
  transparently; the owned-completion lane spreads whole regions round-robin.
- **Batched submits**: `read_experts_batched()` issues one deep submit per claim.
  An expert costs **two SQEs when its regions coalesce, six when they do not** —
  its three weight tensors are contiguous on disk (18,874,368 bytes at int4,
  37,748,736 on the int8 MTP layer) and its three `.qs` scales are a second run
  pooled at the front of the shard, so the common case is two reads. Experts
  straddling a shard boundary (~0.45 % of this container) keep all six.
  `COLI_EXPERT_MERGE=0` forces six, for A/B against a bit-identity assertion.
  Extents come from the load-time expert map (`ExpertIndex`), and each merged
  buffer is split back into six regions as refcounted `Bytes::View` windows —
  **no copy**, and admission to the warm cache is then free too, since a view is
  already shared. The split is computed from each tensor's own offset, never its
  position: on disk the three projections sit in alphabetical order (down, gate,
  up), so a positional split would load one matrix's bytes into another and still
  produce plausible activations.
  `read_many()` itself does **not** coalesce — it emits one SQE per `ReadReq`.
  This entry claimed "merge contiguous regions before submit" until 2026-08-09,
  describing an optimization no code performed; the merging now happens a level
  up, where the expert's extents are known.
- **fadvise integration**: `POSIX_FADV_WILLNEED` batched before main-path
  reads (`COLI_FADVISE_MAIN`, default on) — on the completion lane the hint
  covers the **next** claim window, issued on spare SQ slots while the current
  window streams (reads always outrank hints for slots); optional
  `POSIX_FADV_DONTNEED` after each streamed read for RSS-bounded runs
  (`COLI_FADVISE_DROP`).
- **Adaptive io-wq cap**: the [`IoTuner`](adaptive-runtime.md#iotuner)
  adjusts `iowq_max_workers` between forwards from an EWMA + SQ-full deltas
  (`COLI_IO_TUNE`, default on).
- **Retry ladder**: on a batched-read failure each region re-issues via
  `read_exact_retry` with linear backoff for transient
  `EIO`/`EAGAIN`/`EINTR` (`COLI_IO_RECOVERY`, default on).

Everything goes through the reactor — config, safetensors headers, all weight
loading. Only `tokenizer.json` and `/proc/meminfo` are read synchronously
(one-time).

## O_DIRECT lane

`COLI_DIRECT=1` enables the zero-copy DMA path: a twin fd opened with
O_DIRECT, 4 KiB base/length alignment, and `Reactor::read_direct_aligned`
DMA-ing each region's 4096-aligned superset straight into an owned
`AlignedBuf`, exposing the exact `[off, off+len)` sub-slice. This bypasses the
page cache entirely — the right call when streaming ~11 GB/token of
0.6 %-reuse expert data that would otherwise evict useful pages. Measured on
the reference box it raised raw device read rate by ~21 % (see
[Benchmarks](benchmarks.md)).

Weight loading itself is `pread` + `fadvise(DONTNEED)` for flat RSS — a
deliberate choice over mmap.

### Measuring the lane in isolation

[`examples/iobench.rs`](../crates/peregrine-io/examples/iobench.rs) drives the
`Reactor` against any file with no model loaded, mirroring colibrì's `iobench`
argument order so the two are directly comparable:

```bash
cargo run --release -p peregrine-io --example iobench -- FILE BLK_MB ITERS RINGS DIRECT
```

Each ring submits its whole batch in one call, so queue depth = `ITERS` — a
depth-1 loop measures per-request latency, not the device's parallel read rate,
and reads ~40 % low.

**Fixed 2026-08-02: the O_DIRECT lane itself ran at queue depth 1.**
`read_direct_aligned` looped one region at a time, each calling `read_many` with
a single-element slice, so a 96-region expert batch became 96 sequential ring
round-trips — the "batched reads" win never reached the direct path, and direct
reads measured *slower* than buffered ones. It now allocates every region's
landing buffer up front, submits them in one `read_many` (which chunks
internally to the ring's entry count), and completes short reads per region.
Peak memory is unchanged, since each region already owned its buffer for the
returned `Bytes`. Worth 1.2–1.3× on a LUKS NVMe; the residual gap to colibrì's
threaded-pread harness is still open, with dm-crypt CPU-boundness as the
hypothesis. See [Benchmarks](benchmarks.md#second-box-glm-52-on-a-7-gb-laptop).

## Slab pool (`slab.rs`)

A pool of aligned slabs for expert landing buffers. (Not lock-free, despite
older prose: `slab.rs` states outright that the outer `Mutex<Reactor>` already
serializes access, so a lock-free swap here would be dead weight.)
`checkout_tagged` / `checkin_tagged` return and verify a generation-tagged
`SlabHandle { gen }` so a straggler speculative load cannot write into a recycled
slab. **Wired 2026-08-06** — the O_DIRECT landing path (`ring.rs`, the sole
production call site) now checks out and returns tagged handles.

Two things had to be fixed before wiring, not after. The check asserted
`handle.gen < self.gen`, which **every** handle the pool ever issued satisfies,
because `gen` only increases — so the double-return it advertised could not be
detected, and wiring it first would merely have made an unfireable check
reachable. The pool now tracks the generations actually *outstanding*, and a
handle absent from that set — returned twice, or issued by a different pool — is
rejected with its buffer dropped rather than pushed onto the free-list, where it
would let two holders write the same allocation. And the check was a
`debug_assert!`, compiled out of exactly the build where a late completion can
happen; it now runs in release. Expert reads are
zero-copy into the weight: the landing region is a `peregrine_io::Bytes` the
streamed `QtWeight` moves in, and kernels read it via `Deref<[u8]>`.

## Warm cache & tiering

Covered in depth in [Prefetch & caching](prefetch-and-caching.md): the
budgeted warm RAM cache (`COLI_ECACHE_GB`) with Bloom-filter miss shortcut,
transparent zstd, negative TTL, heat-gated admission and idle recompression.

Eviction picks the lowest `(priority, recency)` — priority-weighted LRU
(`warmcache.rs::evict_to_budget`) — unless `COLI_CACHE_LFRU=1`, which replaces
the recency component with `tier::lfru_score` (`(heat << 8) | recency`) while
leaving priority as the primary key. Default off and bit-for-bit the historical
tuple; see [prefetch & caching](prefetch-and-caching.md#tiering--gpu-residency).

## Compression (`peregrine-core/src/compress.rs`)

One shared zstd codec threads through both storage levels:

- **On disk** — read support only. `SafeTensors` decodes compressed containers
  produced elsewhere, but no first-party path *writes* one:
  `pack::Blob::with_compression` has test-only callers, and
  `peregrine-requantize` writes uncompressed. The header carries
  `"compression": "zstd"` + `"uncompressed_nbytes"` and reads decompress
  transparently. See [Model format](model-format.md#safetensors-header-extensions).
- **In RAM** — `COLI_CACHE_COMPRESS` zstd-compresses warm-cache slabs on
  admit (~1.2× smaller — measured) and decodes on hit.

## Memory hints (`mem.rs`)

- **Hugepages** (`COLI_HUGEPAGE`, default on): `MADV_HUGEPAGE` applied at
  every ≥ 2 MB allocation choke point — `AlignedBuf::with_capacity`,
  registered read buffers, safetensors landing buffers. Ranges are narrowed
  to whole pages inside the caller's allocation so the destructive
  `MADV_DONTNEED` companion never touches neighboring mappings.
- **NUMA binding** (`COLI_NUMA_PIN=1`): `bind_local_if_enabled`
  (`sched_getcpu` → node → `mbind`) binds every ≥ 2 MB buffer to the
  allocating thread's node **before first touch**; worker threads pin
  round-robin across node-grouped CPUs (the `peregrine-par` pool via a
  std-only worker-startup hook, and the prefetch pool at spawn). CPUs are
  enumerated node-grouped, so consecutive workers fill a node before spilling
  to the next socket.
- **Malloc arenas**: both binaries call `cap_malloc_arenas()` at startup
  (`M_ARENA_MAX=2`); setting `MALLOC_ARENA_MAX` yourself or `COLI_NO_ARENA_CAP`
  makes it a no-op.
- **Trunk wiring** (`COLI_MLOCK=1`, `wire_resident`): `mlockall(MCL_CURRENT)`
  called from `Model::load` *after* the resident weights are in and *before* the
  warm cache fills, so it pins the trunk and leaves every later allocation —
  cache slabs, KV, streaming buffers — ordinary reclaimable memory.

  `MCL_CURRENT` and deliberately not `MCL_FUTURE`. That asymmetry is the whole
  design: a streaming MoE engine lives with a large resident trunk and a cache
  sized to the remainder, which is exactly the shape that invites the kernel to
  reclaim trunk pages to grow the page cache — and every reclaimed trunk page is
  re-read at disk speed in the middle of a token. The cache is *supposed* to be
  the part the kernel can take back; wiring it too would turn a gentle slowdown
  into an allocation failure.

  **It does not raise the memory ceiling, it removes variance.** A model that did
  not fit still does not fit — it now fails honestly instead of swapping, which is
  the better of the two. Needs `RLIMIT_MEMLOCK` headroom (`ulimit -l unlimited`,
  or `CAP_IPC_LOCK`); refusal is the normal outcome on a desktop default and is
  reported with the limit and the fix rather than failing the load. Note the
  interaction with `COLI_REGBUF_SLOTS`, which charges pinned pages against the
  same limit.

## Memory budgeting

`mem_available_bytes` is the smaller of `/proc/meminfo` `MemAvailable` and what
the process's cgroup permits (v2 `memory.max` − `memory.current`, else v1
`limit_in_bytes` − `usage_in_bytes`; unlimited sentinels are recognized by
magnitude and defer to the host figure).

`/proc/meminfo` is **not namespaced**, so inside a container it reports the host.
A 4 GB container on a 512 GB host would otherwise read 512 GB of headroom, size
its warm cache for hardware it is not running on, and be OOM-killed by the cgroup
while `[ram]` printed a comfortable projection. The parsing is pure and
unit-tested (`ram.rs`); the projection itself is unchanged.

See also the [expert-cache cliff](prefetch-and-caching.md#borrowed-negative-results):
a cache that fits by this arithmetic can still be too large, because `MemAvailable`
counts reclaimable page cache. `cache_cliff_warning` says so at load.

## Topology, sensors, perf (`topo.rs`, `sensors.rs`, `perf.rs`)

- **Topology probe**: logical CPUs, NUMA nodes
  (`/sys/devices/system/node/nodeN/cpulist`), and PCIe link speed/width per
  BDF (`/sys/bus/pci/devices/<bdf>/current_link_{speed,width}`); single-node
  fallback keeps every caller safe off-Linux.
- **Sensors**: `/sys/class/thermal` package temperature and a wrap-aware RAPL
  energy meter (`/sys/class/powercap`) — inputs to the
  [sensor governors](adaptive-runtime.md#sensor-governors).
- **Perf counters**: a real `perf_event_open(2)` LLC-miss counter
  (`PerfCounter::open_cache_misses`, hand-declared `PERF_ATTR_SIZE_VER0` attr
  layout, thread-following, user-space-only). Gated on
  `COLI_PERF_COUNTERS=1`; degrades to `None` wherever the kernel refuses.

## `unsafe` policy

`peregrine-io` is one of the three engine crates allowed `unsafe` — confined
to io_uring submission, aligned-buffer allocation, and the OS-interface
helpers (`madvise`, `sched_setaffinity`/`mbind`, `perf_event_open`). The
[bad-patterns audit](BAD_PATTERNS.md) reports any `unsafe` outside the
expected crates.

## Per-read latency distribution (`COLI_IO_LATENCY`)

Every other I/O figure on this page is a mean or a steady-state aggregate. None
of them can answer *"what is p99 for one token, and how much of it is page-fault
handling rather than device time"* — and **a mean is exactly the statistic an
SSD garbage-collection tail survives**. Periodic multi-hundred-millisecond
stalls move a mean by a few percent while dominating the tail, and `IoTuner`
smooths that signal by design.

`peregrine-io/src/latency.rs` is therefore a **histogram, not another EWMA**:
log-scale, four sub-buckets per octave, 0 µs–16 s, sampled **per completion** in
`Reactor::read_many` (a wave mean would average away the single slow read being
hunted), plus a `RUSAGE_THREAD` minor/major fault delta over the same window.
Per-ring histograms are merged at shutdown, because a per-ring p99 answers "how
slow was ring 2" when the question is "how slow was a read".

### Reading it honestly

- The tail figure is **p99/p50**. The first version used p99/mean and was wrong
  in the only case that matters: one large outlier drags the mean above p99, so
  a fat tail reads as flat.
- `max/p50` is reported beside it. **p99 cannot resolve fewer than `count/100`
  stalls** — one stall in a hundred reads reports flat, correctly — so a window
  where p99 is flat and `max` is not prints "rare stalls p99 cannot resolve at
  this sample count", never "no tail".
- Undersampled quantiles print `n/a(20<100)`. A p99 from twenty samples is the
  slowest of twenty wearing the name.
- **Submit→complete is not device service time.** It includes queueing behind
  the ring's depth cap and io-wq scheduling. The report says so rather than
  letting a fat tail read as a statement about the drive.

### Measured on the real GLM-5.2 container, 2026-08-21

2400 expert reads, B=1, cold:

```
[latency] expert-read: n=2400 mean=61745.9us p50=40959us p90=114687us
                       p99=262143us p99.9=3381313us max=3381313us
[latency] expert-read: minor-faults=0 (0.00/read) major-faults=0
                       tail(p99/p50)=6.4x worst(max/p50)=82.6x
[latency] p99 is flat but the slowest read is >=10x the median: rare stalls
          p99 cannot resolve at this sample count.
```

**There is a fat tail, and p99 alone would have missed it.** The p99/p50 ratio is
6.4× — under the 10× threshold, i.e. "flat". The slowest read is **3.38 seconds
against a 41 ms median, 82.6×**. That is exactly the case the `max/p50` column
exists for, and the reason the report distinguishes "no tail" from "rare stalls
this window cannot resolve". A ratio-on-p99-alone instrument would have printed
reassurance.

**The host/device split is unambiguous here: zero.** `minor-faults=0`,
`major-faults=0` over 2400 reads, so essentially none of that latency is
page-fault handling — it is queueing plus device time. This answers directly a
question that had been put to this project and could not previously be answered.

Note what it does *not* settle: submit→complete includes queueing behind the
ring's depth cap, so a 3.38 s outlier is not yet proof the drive stalled. The
next step is an `iobench` arm at fixed queue depth to separate the two.

## Device geometry and layout alignment (`peregrine align-cost`)

The engine aligns O_DIRECT reads to 4096 (`slab::ALIGN`) because the *syscall*
requires it. That is not the granularity the **drive** serves at: NVMe parts
commonly advertise a 128 KB optimal transfer. An expert is read through six
regions, so a region whose start is arbitrary pays a straddled unit at each end,
six times per expert per routing.

`peregrine-io/src/geometry.rs` probes `optimal_io_size` from sysfs and models
what aligning region starts would cost and save. **The constant is probed, not
chosen** — an unprobeable device reports `ASSUMED` rather than quoting the
fallback as a measurement.

Note what this is *not* about. The idea arrived as "remap tensors to 2 MB
hugepages at the filesystem layer", which targets TLB pressure — and TLB
pressure is not in this read path, because **nothing `mmap`s the checkpoint**.
Reads go through io_uring or `pread` into owned buffers. What a straddled read
costs is **queue depth**: it becomes two device operations, and the second is
what stalls the pipeline.

Measured expectation on real expert shapes: a ~3 MB region spans ~23 units, so
starting mid-unit costs about **4 % more units for about 0.5 % more disk**. Small
but real — and a unit count is not a latency, so the writer stays untouched
until `iobench` confirms it on the real shards with a cold cache.
## The bytes/token wall on a SATA stripe (2026-09-26, GLM-5.3-Flash)

Issue #9 asks for ≥2 tok/s aggregate. This section records what the *host* in
front of the engine can deliver for that model, and which of the 2026-09 SSD-MoE
papers' levers survive contact with it. Everything below is measured on this box
unless a line says `projected`.

### What the three drives deliver

`sda` + `sdb` + `sdd`, the HDD removed, expert shards physically split across all
three (`GLM-5.3-Flash-4ssd/model_paths.json` → `glm53-experts` on each drive), so
one token's sweep is a genuinely parallel read:

| measurement | value |
|---|---|
| single device, cold sequential | ~0.40 GB/s |
| **aggregate across the three, cold** | **1.10 – 1.23 GB/s** |
| queue-depth scaling beyond 2 rings | **flat** |

Flat beyond two rings is the important row: the ceiling is the *media*, not the
submission path, so no amount of io_uring tuning, ring count, or batching buys
bytes that the drives cannot serve. The engine's own lane accounting agrees —
`io 72.2 s (91%)  cpu 6.9 s (9%)` — it is not doing anything except waiting.

### What one token costs

Locked configuration (`scripts/serve-glm53-flash-cuda.sh` defaults: lookahead 64,
RSS guard 20 GB, `COLI_ECACHE_GB=6`, GPU expert tier off, prefetch-protect off),
B=1, 4 tokens, byte ledger from `[ledger]`:

| quantity | value |
|---|---|
| bytes per expert | 12,615,680 |
| requested | 4.239 GB/token |
| unique (union) | 4.239 GB/token — 0.0% saved by batch union at B=1 |
| **cache-served** | **18.2% of unique** |
| **from disk** | **3.469 GB/token** |
| re-read (evicted) | 2.6% of disk traffic |
| prefetch waste | 0.0% (prefetch off) |
| working set vs cache | 4.04 GB routed against 6.00 GB = **148% of a pass** |
| throughput | 0.144 tok/s cold, **0.24 tok/s warm** |

### The arithmetic that closes single-stream 2 tok/s

At 3.469 GB/token off the devices, the SATA ceiling allows **0.317–0.354 tok/s**.
The warm measurement is 0.24, i.e. **~72% of the I/O-bound ceiling** — about 28%
of remaining slack is overlap efficiency, and that is the *entire* inventory of
software-side headroom that does not involve moving bytes.

2 tok/s single-stream therefore needs **6.9 GB/s of device bytes**, or 8.5 GB/s
if nothing were cached — **6–7× the media ceiling**. This is not a scheduling
deficit and cannot be tuned out on SATA. Any claim of 2 tok/s at B=1 on this box
is arithmetically false, and `docs/deployment-glm53-flash.md` should be read with
that sentence beside it.

A second, independent confirmation of the model comes from
[arXiv 2608.18261](https://arxiv.org/abs/2608.18261), which measures 0.44 tok/s
warm serving Qwen3-235B-A22B Q4_K_M on 8 GB of VRAM and notes it is "in exact
agreement with a bytes-per-token ÷ bandwidth model". Same equation, different
hardware, same agreement — which is why the division above is trusted here.

### Which 2026-09 levers survive a 1.2 GB/s ceiling

Five papers bear directly on this. Each is recorded with what it claims and what
this box says about it — because two of them do not transfer, and the reason why
is the useful part.

**[2609.18110 SSD-LLaMA](https://arxiv.org/abs/2609.18110) — the critique that
names our engine.** It benchmarks *Colibri* (peregrine's upstream) explicitly and
finds that Colibri achieves **higher read bandwidth** than the proposed system
while paying **higher per-token I/O traffic**; SSD-LLaMA moves 1.5× less data per
token than KTransformers and 1.8× less than llama.cpp, and claims 77.7% of peak
SSD read bandwidth against a 43.2% baseline. Read against the table above, the
first half of that critique cannot apply here: peregrine runs at ~91% lane
occupancy and ~72% of its own device ceiling, on SATA where the peak *is* ~1.2
GB/s. It is not the bandwidth-inefficient case. The second half applies exactly —
the bytes per token are the wall. Of its three mechanisms, pack layout with O(1)
byte offsets is already in the tree (`[expertmap] indexed=19456 coalescing=98.7%
→ 39936 reads per full sweep vs 116736 unmerged`), bit-width is a measured dead
end *for data-free rounding* (below), and **partial-tensor tiering — keeping
`gate`/`up` hot and `down` cold rather than caching an expert whole — is the one
mechanism genuinely absent here.**

**[2609.18063 Edge0](https://arxiv.org/abs/2609.18063) — prediction as routing,
not prediction and verification.** Its claim is that a prerouter predicting the
next layer's routing *one token ahead*, consumed *as* the routing so the staged
set equals the routed set, reaches ~20 tok/s for a 35B MoE inside 3 GiB of peak
active memory — and that the advantage is "conservation, not prediction quality".
This is the one paper that explains a negative result of ours better than we
did. Our predict-*and-verify* prefetch measured **+3 hits for 4034 extra reads
and +41% wall time**, and our own scoreboard puts `router-lookahead` at **92.5%
recall / 46.3% precision** at width 16. Under verify, that 46.3% precision is
wasted I/O, which is exactly what was measured. Under *prediction-as-routing*,
the same 92.5% recall is ~92.5% of the sweep staged before it is needed and 46.3%
precision becomes a **quality** cost instead of an I/O cost. With those numbers
the device bytes fall enough to reach 1–3 tok/s on this hardware — the only lever
in this document that arithmetically reaches 2 tok/s at B=1. It is also the only
one that requires training a recovery adapter: it changes which experts compute
a token, so `peregrine flip-rate` gates it and a LoRA has to pay the loss back.
It is a project, not a knob.

**[2608.18261 — a negative result worth more than most positive ones](https://arxiv.org/abs/2608.18261).**
Three numbers here land directly on open questions in this repo:

- **Expert reuse between adjacent tokens is 2.0× chance.** Our own trace of
  GLM-5.3-Flash gives 22% consecutive-token overlap against an 11.6% uniform-null
  baseline — **1.9× chance**, on a different model, from different tooling. The
  locality is real, weak, and now corroborated twice; it is also the hard ceiling
  on single-stream caching, which is what the 18.2% served figure is bumping
  against.
- **Batching collapses.** The paper builds "a plausible request-batching scheme
  that should amortize one disk sweep over many streams" and reports it collapses
  at batch 32 under paging thrash. This is the scheme the ≥2 tok/s aggregate
  target rests on, and it fails on *memory*, not bandwidth — which matches this
  box exactly (`Swap: 15/15 GB used, 0 free` during a B=1 run).
- **Routing locality is not trainable at acceptable cost.** Locality training cut
  cache misses by up to 60% but *every* configuration failed a pre-registered
  ≤1% perplexity gate; miss reduction and perplexity cost were tightly coupled.
  That closes idea #6 ("router fine-tuning for locality") in
  `docs/ideas-github-issue-7-2026-08-23.md` on external evidence, and it closes
  it cheaply — no training run needs to be budgeted for it.

**[2601.17063 FlashMoE](https://arxiv.org/abs/2601.17063) — and why its 2.6× is
not ours.** An ML replacement policy combining recency and frequency improves hit
rate by up to 51% over LRU/LFU and reports up to 2.6× speedup on a real desktop.
The *mechanism* is one this repo already discovered from the other side, and
harder: at sub-one-pass capacity, pure LRU returns **0 hits out of 9944** lookups
while the cache sits 100% full, because a deterministic front-to-back layer sweep
makes LRU retain exactly the complement of what is needed next (see
`bench-data/2026-08-10-sweep-eviction/`). What does not transfer is the
*magnitude*, and the reason is capacity regime: FlashMoE's gain is a
below-threshold gain, and this configuration runs at **148% of a pass**, where
plain recency already captures ~83% of the reuse that exists (18.2% served
against a 22% overlap ceiling). Applying their +51% relative hit-rate gain to
these numbers buys **~+13% tok/s, not 2.6×**. Replacement policy is the most
cited lever in this literature and the least available here.

**[2512.09277 METRO](https://arxiv.org/abs/2512.09277) — the right question,
wrong substrate.** Balancing *activated experts* rather than tokens per worker
reduces decode latency 11–22% (up to 4.11× decode throughput at fixed SLO) in
expert-parallel GPU serving, because token-balancing inflates the distinct-expert
count and memory-bound workers pay for the union. The insight is precisely the
union amplification this box shows — at B=16 with k=8 of 288, the expected
distinct set is ~105 experts, i.e. ~6.6 expert-equivalents per token instead of
8. But it is about spreading a union across GPUs, and at B=1 there is no union to
balance (`share=1.000x` by construction). Transferable as arithmetic, not as a
mechanism to implement.

### Is "many async streams" a throughput multiplier? Measured: no, it is a trade

The question deserves the measured answer rather than the intuitive one, because
the machinery already exists and has already been run on this model.

`crates/peregrine-serve/src/batch.rs` is a continuous-batching engine: an async
HTTP handler submits requests over two priority channels, **one** engine thread
drains high- then normal-priority each tick, admits up to `MAX_BATCH` sequences,
and runs **one batched step across all active rows**. The I/O under it is already
fully async and already parallel — one io_uring ring per device, device-pure claim
groups (`COLI_IO_DEVICE_SCHED`). Request-level concurrency, async submission, and
parallel device reads are not missing features; they are the shipped shape.

Measured on GLM-5.3-Flash (`docs/deployment-glm53-flash.md`, serialized runs):

| Arm | tok/s | Per stream |
|---|---:|---:|
| sustained B=1 | **0.154** | 0.154 |
| B=16, CPU shared path | **0.335** aggregate | 0.021 → **48 s/token/user** |
| B=16, GPU shared path | 0.299 aggregate | 0.019 |

16× the streams buys **2.2× the aggregate** and hands each user forty-eight
seconds per token. Three walls explain why that is the whole of it:

1. **Async removes waiting, not bytes.** Delivered bandwidth is ~0.83 GB/s
   against a measured 1.10–1.23 GB/s ceiling, and the lane is already `io 91%`
   occupied at B=1 — there is no idle bubble for a second stream to fill. The
   three drives are DRAMless SATA parts (Kingston SA400, Acer RE100, Crucial
   BX500) each at ~0.4 GB/s, so the stripe is delivering ~90% of the *sum of its
   parts*; parallelism is not the missing ingredient, media is.
2. **Union amortization is capped near 1.23×.** Batching's only byte-reducing
   mechanism is that B streams share experts. At k=8 of 288, expected distinct
   experts = 288·(1 − (1 − 8/288)^16) ≈ **104 of the 128 requested** = 0.815
   bytes/token. That lifts the aggregate ceiling from 1.2/3.47 = 0.35 to
   1.2/2.83 = **0.42 tok/s** — and the measured 0.335 sits below even that.
3. **Memory is the wall that actually breaks it.** The batch's per-step routed
   set is ~104 experts × 43 layers × 12 MB ≈ **56 GB** against a 6 GiB cache —
   941% of a pass — on a box whose swap read `15/15 GB used, 0 free` during a
   *B=1* run. This is exactly the failure 2608.18261 documented: a correctly
   built request-batching scheme that "should amortize one disk sweep over many
   streams" **collapsed at batch 32 under paging thrash**, dying on memory rather
   than bandwidth. The serve script's own comment records the same event here at
   B=16 with `COLI_ECACHE_GB=10`: MemAvailable 28.7 → 3.0 GB, swapping began,
   which is why `COLI_RSS_GUARD_GB` exists and refuses rather than swaps.

The shape to avoid is N separate B=1 server processes: that multiplies byte
demand by N (N × 3.47 GB/token against one fixed 1.2 GB/s) and gives each
process its own 4.04 GB working set to evict the others' with.

There is, however, a real async gap the current design cannot close with streams
at all. The engine runs **exactly one decode step at a time**, and step t+1's
layer L cannot be fetched before step t computes it, because routing for L is
produced by layer L−1. That is why cross-step prefetching failed here — predict
*and verify* measured **+3 hits for 4034 extra reads and +41% wall time**. The
version of "all work is async" that converts into bytes is prediction consumed
*as* routing (2609.18063), where the speculation is used rather than checked.

### Levers, ranked by measured magnitude

| # | Lever | Effect on device GB/token | Status |
|---|---|---|---|
| 1 | **Reclaim RSS; stop the indexer** | none directly — unblocks 2 and 3 | `baloo_file` holds **4.38 GB** RSS and has read **107 GB** cumulatively with **nothing excluded** from indexing `/srv`. Needs consent to stop. |
| 2 | **Calibrated int3-g64** | 3.47 → ~3.0 (**−13.8%**, the RTN-measured byte effect) → **+14% tok/s** | Code-complete (`pack.rs:192`, `peregrine-requantize --calib`), **unrun** — needs row 1's RAM. Voids if calibration cannot beat RTN's int3 `flip_rate` of 0.514. |
| 3 | **Overlap out to distance 2–7** | unknown; moves the 22% ceiling up or confirms it | The one open question `bench-data/2026-08-10-sweep-eviction/` left. Cheap, and the only way caching gets big. |
| 4 | **NVMe hot tier** | 600p adds **669 MB/s** → ceiling +54%, ideal 0.35 → 0.53 tok/s | Only **21 GB** free on `/srv/model600p`, and the census needed to size it **does not exist** (below). Must be iobench'd before any claim. |
| 5 | **Partial-tensor tiering** | bounded by the 1/3 gate/up/down split | Unimplemented; the SSD-LLaMA mechanism genuinely absent here. |
| 6 | **Prediction-as-routing + LoRA** | 92.5% recall staged → ~0.3 GB/token → **1–3 tok/s** | The only lever that reaches issue #9's target arithmetically. Changes computed routing, so it is a training project gated by `flip-rate`, not a knob. |
| 7 | **Batch wide (B>1)** | −18.5% at B=16 → aggregate ceiling 0.42 tok/s | Measured and memory-gated; 48 s/token/user. For aggregate serving, not latency. |
| 8 | **Better replacement policy** | 18.2% → 27.4% hit → 3.08 GB/token = **+13%** | Real but last: at 148% of a pass, plain recency already captures ~83% of the reuse that exists. |
| 9 | **Faster media** | 3.47 GB/token at ~4 GB/s = **>1 tok/s** | The honest answer to "2 tok/s at B=1" is a different drive, not different code. |

Rows 1–5 stacked optimistically reach **~0.4 tok/s**, not 2. Issue #9's target on
this box needs row 6 or row 9, and that should be said plainly rather than
discovered late.

### Corrections recorded by this pass

- **`bench-data/2026-09-22-4drive-split/README.md` calls the array "4 HDDs" and
  explains its rates as "HDD-class". Three of the four devices are SATA SSDs.**
  Per sysfs `rotational`: `sda` Kingston SA400S37960G = 0, `sdb` Acer RE100 = 0,
  `sdd` Crucial CT1000BX500SSD1 = 0; the spinning parts are `sdc` (Hitachi
  HTS547575) and `sde` (WDC WD10SPZX), both 1. The GLM-5.3 placement already
  excludes `sdc`, which is why the three-SSD stripe measures **1.10–1.23 GB/s**
  where that README recorded 0.87 GB/s aggregate: the straggler was removed, the
  array was not improved. Its 85%-of-sum contention story should be re-derived
  before it is cited.
- **The "browsers/Steam holding ~16 GB of deleted tmpfs files" comment in
  `scripts/serve-glm53-flash-cuda.sh` does not reproduce.** Deleted-but-still-open
  descriptors across the top RSS processes total **under 1 MB**. The actual
  holders are `baloo_file` at **4.38 GB** and this box's dev/agent tooling at
  ~7.5 GB. The RSS guard's *policy* is right; only its stated rationale was stale.
- **`route_stats.json` cannot size a hot tier.** `heat` and `learn` are `null`
  and `hist` carries **4 positions per layer** (168 layer-positions, 962 of 12,384
  slots). Fitting a concentration curve to that returns artifacts — a naive read
  says "top 16 experts per layer cover 78.4% of volume", true only because 4
  positions × 8 experts cannot produce more than 32 distinct experts per layer. A
  hot-expert tier needs hundreds of positions, which the calibration run in row 2
  produces as a side effect.
- The container at `/srv/m-sdd/GLM-5.3-Flash-peregrine` (101 GB) is an **aborted
  import** — 30 expert layers holding 47/103/288 slots, against the served
  container's 43 × 288 — not a lower-bit sibling. Do not serve it. The served
  manifest is `GLM-5.3-Flash-4ssd`, whose `model_paths.json` points at
  `glm53-experts` on `m-sda`/`m-sdb`/`m-sdd`.

### Standing caveats

- Every byte-ledger figure here is a **4-token run**; token 1 is cold and pulls
  the served fraction down. The warm 0.24 tok/s baseline is a separate measurement
  from the 0.144 tok/s ledger run.
- Throughput numbers are single runs. The retained profile's table in
  `docs/deployment-glm53-flash.md` came from `scripts/bench-glm53-cuda-matrix.sh`
  in fresh serialized processes; arms must not overlap.
- Bandwidth ceilings come from `peregrine-io`'s `iobench` with O_DIRECT and no
  root for cache drops; the 1.10–1.23 GB/s spread is run-to-run, not a confidence
  interval.
- `iobench` reports decimal GB/s while the byte ledger reports GiB. Mixing them
  moves derived tok/s by ~7%; the ~72%-of-ceiling claim is not robust to that unit
  sloppiness at the last significant digit.
- Projected rows are arithmetic on measured bytes, not benchmarks.








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

- **Registered files** (`IOSQE_FIXED_FILE`) plus `SINGLE_ISSUER` and
  `COOP_TASKRUN` ring flags.
- **Registered buffers**: `register_read_buffers()` + `IORING_OP_READ_FIXED`
  are implemented and tested, but **not reachable from the streaming path** —
  nothing calls them outside tests, and `COLI_REGBUF` is read by no code.
- **Batched submits**: `read_many()` / `read_experts_batched()` merge
  contiguous regions before submit; the streaming lane keeps
  `COLI_IO_BATCH` experts × 6 regions ≈ 96 reads in flight per ring.
- **fadvise integration**: `POSIX_FADV_WILLNEED` batched before main-path
  reads (`COLI_FADVISE_MAIN`, default on); optional `POSIX_FADV_DONTNEED`
  after each streamed read for RSS-bounded runs (`COLI_FADVISE_DROP`).
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
slab — **but nothing calls them**. The live path is the untagged
`checkout`/`checkin`, so that protection is implemented and not in effect. Expert reads are
zero-copy into the weight: the landing region is a `peregrine_io::Bytes` the
streamed `QtWeight` moves in, and kernels read it via `Deref<[u8]>`.

## Warm cache & tiering

Covered in depth in [Prefetch & caching](prefetch-and-caching.md): the
budgeted warm RAM cache (`COLI_ECACHE_GB`) with Bloom-filter miss shortcut,
transparent zstd, negative TTL, heat-gated admission and idle recompression.

Eviction picks the lowest `(priority, recency)` — priority-weighted LRU
(`warmcache.rs::evict_to_budget`). The LFRU scoring in `tier.rs`
(`(heat << 8) | recency`) is **not** wired to it; see
[prefetch & caching](prefetch-and-caching.md#tiering--gpu-residency).

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

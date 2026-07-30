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
  (the streaming path keeps plain reads by default — see `COLI_REGBUF`).
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

## Slab pool (`slab.rs`)

A lock-free pool of aligned slabs for expert landing buffers.
`checkout_tagged` / `checkin_tagged` return and verify a generation-tagged
`SlabHandle { gen }`, so a straggler speculative load can never write into a
recycled slab; use-after-checkin is caught in debug builds. Expert reads are
zero-copy into the weight: the landing region is a `peregrine_io::Bytes` the
streamed `QtWeight` moves in, and kernels read it via `Deref<[u8]>`.

## Warm cache & tiering

Covered in depth in [Prefetch & caching](prefetch-and-caching.md): the
budgeted warm RAM cache (`COLI_ECACHE_GB`) with Bloom-filter miss shortcut,
transparent zstd, negative TTL, heat-gated admission and idle recompression;
plus the LFRU tier scoring (`(heat << 8) | recency`) used for
eviction/promotion decisions.

## Compression (`peregrine-core/src/compress.rs`)

One shared zstd codec threads through both storage levels:

- **On disk** — `pack::Blob::with_compression(Compression::Zstd)` writes
  tensors compressed; the safetensors header carries
  `"compression": "zstd"` + `"uncompressed_nbytes"` and reads decompress
  transparently. See [Model format](model-format.md#safetensors-header-extensions).
- **In RAM** — `COLI_CACHE_COMPRESS` zstd-compresses warm-cache slabs on
  admit (~2–3× smaller) and decodes on hit.

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

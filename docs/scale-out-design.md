# Designs for the work this box cannot run

Five roadmap items need hardware this development machine does not have: a
second GPU, a second host, or an NVIDIA GDS driver stack. They stay unbuilt
rather than half-built, because the alternative — shipping code behind a
hardware check that can never fire here — is the "implemented, tested, and
reachable by nothing" state the `[R]` reachability pass exists to remove.

This page is what replaces the code: enough design that starting is a matter of
having the hardware, not of rediscovering the problem.

**The rule each section obeys: it must name a `file:line` seam.** A section that
cannot point at the line that changes is a wish, and wishes belong in `todo.md`,
not here.

## Read this before the table

**None of the five cuts bytes read per token on a single host**, which is this
repo's test for whether anything can move tokens/second
(`todo.md` § "Which shipped work can actually move tokens/second"). Two of them
cut bytes per token *per host* by splitting the container across machines, which
is a different and much stronger claim — and the honest reason to want them.

| Item | Needs | Cuts bytes/token? | Size |
|---|---|---|---|
| [Multi-GPU expert ownership](#1-multi-gpu-expert-ownership-and-migration) | ≥2 GPUs | Yes, per host — 2× VRAM is ~2× the resident set | ~600–900 lines |
| [Distributed sharding](#5-distributed-inference-across-hosts) | ≥2 hosts | Yes, ~1/N per host | A different program |
| [NVLink-aware placement](#3-nvlink-aware-expert-placement) | ≥2 NVLink GPUs | No — cuts inter-GPU bytes | ~150, after #1 |
| [VRAM replication](#4-runtime-expert-replication-in-vram) | ≥2 GPUs | No — reduces the distinct resident set | ~100, after #1 |
| [GPUDirect Storage](#2-gpudirect-storage) | GDS driver stack | No — cuts copies per byte | ~400 lines |

---

## 1. Multi-GPU expert ownership and migration

**What it is.** Partition the resident expert set across devices instead of
filling one, so the VRAM tier scales with the number of cards.

**Why this box cannot run it.** One RTX 3060. Everything below would compile and
would be exercised by exactly zero tests here — and note the specific trap:
every GPU-gated test self-skips on `init(&[0]) < 1` and reports `ok`, so a
two-device path on a one-device box is untested *and green*.

**Where it hooks in.** This is the best-evidenced seam in the file, because the
architecture is already multi-device and only the entry point is not:

- `crates/peregrine-model/src/gpu.rs:1444` — `peregrine_cuda::init(&[0])`
- `crates/peregrine-model/src/gpu.rs:1447` — `let device = 0;`

Those two lines are the whole hardcode. Everything downstream already threads
`device` as a parameter (`upload_expert(.., self.device, ..)`, `coli_cuda_pipe_*(device, ..)`,
`coli_cuda_graph_begin(device)`), and the C side already builds a context array
`static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES]` (`cuda/backend_cuda.cu:62`)
with `COLI_CUDA_MAX_DEVICES` at 16 (`cuda/backend_cuda.h:21`); `coli_cuda_init`
accepts a device list of that length (`:486`).

The design: one `GpuTier` per device, each with its own `experts` map and graph
cache; an owner map `(layer, expert) → device`; and a `plan_residency` that
**partitions** rather than fills. Migration is the existing `reheat` evict/admit
swap generation with a device in the key.

**The constraint that will bite.** Cross-device partial sums must be gathered in
a fixed device order, never in completion order. The reduce is `pos`-keyed
precisely so that float addition happens in one order regardless of timing
(`concurrent.rs`, `backend_cuda.cu`'s `grouped_reduce` note); summing on arrival
would make the low-order bits a function of which card finished first — the same
defect that got CPU/GPU split GEMM declined.

**How you would know it worked.** Per-device `residents` and `bytes` on
`/metrics`, plus a migration counter. Two-opposite-outcomes test: the same heat
table and budget across 1 vs 2 devices must produce **different** placements and
**identical** output.

---

## 2. GPUDirect Storage

**What it is.** `cuFileRead` moves shard bytes from NVMe straight into VRAM,
skipping the host bounce buffer.

**Why this box cannot run it.** No `nvidia-fs` kernel module, and a consumer
card. More to the point, it would optimize the wrong thing here: the container
under test lives on a USB 2.0 volume at ~41.7 MB/s, and GDS removes a memory
copy that is three orders of magnitude away from being the bottleneck.

**Where it hooks in.** `read_regions` (`crates/peregrine-model/src/concurrent.rs:278`)
is already the single choke point behind which `COLI_IO_ENGINE` selects `uring`,
`pread` or `regbuf` (`:391`). GDS is a fourth engine there, not a new path —
which is what makes it a bounded change. It does need one new thing on the CUDA
side: a device pointer handed out *before* the bytes exist, so the upload path
becomes `cuFileRead → device pointer` instead of `read → slab → pipe_upload`.

**Cost.** ~400 lines, a `cufile` link in `build.rs`, `COLI_GDS`, and
`gds_reads`/`gds_bytes` on `/metrics`.

**The interaction to watch.** An expert's VRAM bytes now arrive asynchronously,
so the CUDA graph cache's `scratch_gen` guard has to cover a buffer that is
allocated before it is filled.

---

## 3. NVLink-aware expert placement

**What it is.** Place co-activating experts on devices with a fast link between
them, so a cross-device gather rides NVLink rather than PCIe.

**Why this box cannot run it.** One GPU — and consumer Ampere has no NVLink at
all, so even a second 3060 would not exercise it.

**Where it hooks in.** `crates/peregrine-io/src/topo.rs` already probes PCIe link
speed and width per BDF and maps NUMA nodes; NVLink adds
`cudaDeviceGetP2PAttribute`/NVML to that probe. The *consumer* is item 1's owner
map, and the co-activation data already exists — `peregrine-tools`' Louvain and
co-occurrence passes emit it as `schedule.json`.

**Cost.** ~150 lines given item 1. Meaningless before it.

---

## 4. Runtime expert replication in VRAM

**What it is.** Keep a hot expert resident on *several* devices so a routed
token does not wait on a cross-device fetch.

**Why this box cannot run it.** The CPU-side half already shipped
(`COLI_REPLICATE_K`, which warms hot GPU residents into the CPU warm cache).
The VRAM side needs a second device to mean anything.

**Where it hooks in.** `plan_residency`/`plan_precision` in `gpu.rs` already rank
residents by heat from the persisted `HeatTable`; replication is a third knob
(`COLI_VRAM_REPLICATE_K`) applied before item 1's partition.

**The trade-off to state up front:** every replica costs a residency slot on
every device it lands on, so replication and capacity are directly opposed.

**Do this part now, without the hardware:** the break-even is a *routing*
question, not a hardware one. The fraction of routed selections the top-K
experts carry is measurable today on one GPU with `COLI_GATE_STATS` and
`route_stats.json`. If the top-K carry little of the mass, replication is
answered before a second card is bought.

---

## 5. Distributed inference across hosts

**The strongest of the five, and the one whose seam already exists.**

**What it is.** Shard the expert set across machines; each host holds and streams
`1/N` of the container.

**Why this box cannot run it.** One host.

**Where it hooks in.** `install_moe_engine` / `MoeEngine::moe_forward(&ForwardCtx, MoeCall)`
(`crates/peregrine-model/src/concurrent.rs:628`) is a first-call-wins trait
object that `peregrine-sched` already uses as an alternative engine, and
`crates/peregrine-model/tests/moe_engine_report.rs` pins that the reported engine
follows actual dispatch rather than the requested one. A `RemoteMoeEngine`
implementing that trait is the **entire** model-side integration point.
Everything else is transport, a shard map, and failure semantics.

**Why it is worth wanting, in this repo's own units.** An expert weighs
19–151 MB. Even 10 GbE moves ~1.2 GB/s, against this box's 41.7 MB/s USB link
and ~0.9 GB/s NVMe. **The network is faster than the disk here** — that
asymmetry, not elegance, is the argument. Sharding N ways cuts per-host bytes
per token by roughly `1/N`, which is the only change on this page that alters
the disk-bound story qualitatively.

**Same ordering constraint as item 1:** a network gather that completes out of
order must be re-sorted before summing, never summed on arrival.

**How you would know it worked.** Per-shard bytes and round-trip latency on
`/metrics`, and an end-to-end equality test against a single-host run — the
partition is a placement decision, so the output should be reproducible.

---

## What is deliberately not here

**Persistent CUDA kernels** are unbuilt for a different reason: the hardware is
present and they are still declined, because they conflict with a shipped
feature. See `docs/gpu-cuda.md` § "Persistent kernels — declined".

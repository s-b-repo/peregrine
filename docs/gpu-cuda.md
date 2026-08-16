[« Docs index](README.md)

# GPU / CUDA lane

The GPU lane is FFI to the **vendored, validated CUDA kernels** from colibrì
(`cuda/backend_cuda.cu` / `.h`) — fused quantized matmuls, Tensor Core WMMA
(W4A16 / INT4), SwiGLU, attention+RoPE — behind the `cuda` cargo feature of
`peregrine-cuda`. Rewriting those kernels in Rust was deliberately deferred:
they are already token-exactness-validated upstream, and re-wrapping them
would re-open that validation.

## Building

```bash
# default build: the CUDA backend is a stub — no nvcc, no GPU needed
cargo build --release

# on an NVIDIA host with CUDA installed:
cargo build -p peregrine-cuda --features cuda     # build.rs runs nvcc on cuda/backend_cuda.cu
cargo test  -p peregrine-cuda --features cuda     # GPU-gated tests, incl. graph capture
cargo build --release --features cuda -p peregrine-engine
```

`build.rs` compiles `cuda/backend_cuda.cu` with `nvcc` and links
`-lcudart -lstdc++`. The FFI is a flat `extern "C"` ABI over opaque
`ColiCudaTensor*` handles, with added `_async` non-syncing stream variants for
the concurrent scheduler.

## Runtime gates

Building with the feature does not enable the GPU — runtime is opt-in:

| Var | Effect |
|---|---|
| `COLI_GPU=1` | enable the GPU lane (the VRAM expert tier is built only when set) |
| `COLI_GPU_INT4=1` | prefer int4 VRAM residents (~8x denser). Needs per-row int4 sources; a grouped-int4 or int8 expert falls back to f32 on its own rather than failing the tier |
| `COLI_GPU_F32_FRAC=<0..1>` | adaptive per-expert precision: the hottest fraction of residents is promoted to f32 at `reheat`, the rest stay int4; format tracked per expert with re-upload on change |
| `COLI_PCIE_BUDGET_MB=<n>` | cap on bytes one `reheat` generation may upload across PCIe. Residency churn is otherwise unbounded (~18.9 MB int4 / ~151 MB f32 per expert, every 256 decode steps); the coldest deferred experts are reconsidered next generation. Watch `[gpu] transfer_frac` to size it |
| `COLI_CUDA_PROFILE=1` | accumulate per-call `h2d_ms` / `kernel_ms` / `d2h_ms` timings |
| `COLI_CUDA_TC_*` | Tensor Core dispatch knobs (int4 / W4A16 gates, min-row thresholds) |

## What the lane does

- **Batched expert compute**: ready VRAM-resident experts coalesce into
  `expert_group` calls (fused gate/up/silu/down on GPU). Layer-level
  accumulation stays on the deterministic host-side reduce by default, or moves
  onto the device under `COLI_CUDA_FUSED_REDUCE` — see below.
- **Pinned staging + async copies**: `cudaMallocHost` staging with
  `cudaMemcpyAsync` on a persistent non-blocking stream, double-buffered so
  H2D(n+1) ∥ kernel(n) ∥ D2H(n−1).
- **Persistent scratch pools**: pre-allocated scratch slots reused across
  layers — no per-layer allocation churn.
- **CUDA graph cache (`COLI_CUDA_GRAPH`)**: `expert_group` captures its launch
  sequence per *shape* — `(arm, expert count, D, I, per-expert rows)` — and
  replays it. That split is what makes it work at all: in decode the shape
  repeats constantly (at B=1 every routed expert contributes one row) while the
  contents change every call, and the contents ride in through pinned staging
  buffers the graph copies from. Rewriting the pinned descriptor buffer between
  replays is what lets one graph serve a *different residency generation* at the
  same shape.
  **The design hazard is stale device pointers, and it is silent.** `reserve` is
  grow-only and frees before it reallocates, so a graph captured before a larger
  call holds dangling pointers into VRAM the allocator has since handed out —
  correct-looking numbers from freed memory, no error anywhere. A per-context
  `scratch_gen` counter, bumped inside the reserve helpers rather than at their
  call sites (`dc->y` *is* `ctx->y`, so an attention call can invalidate an
  expert-group graph), invalidates any entry captured under an older generation.
  `a_grown_scratch_buffer_invalidates_cached_graphs` is the regression guard.
  **Two arms are excluded**: `COLI_CUDA_TC_W4A16` passes device weight pointers
  as kernel arguments, so a replay would compute against the previous
  generation's experts; and `COLI_CUDA_PROFILE`'s event records are not part of
  the work being replayed. Both fall through to the eager path and are counted
  as `graph_uncacheable` on `/metrics`, so "the knob is on and buying nothing"
  is visible rather than merely slow.
  *Still open*: capturing a whole `forward_layer` — see "What still needs
  hardware" below. This captures the GPU work decode actually does today.
- **Fused device reduce (`COLI_CUDA_FUSED_REDUCE`)**: the gate-weighted
  accumulation of GPU-resident experts runs on the device, so the D2H carries
  `s_n` rows instead of `Σrows` — ~5× fewer at B=16 on the measured GLM-5.2
  unions, and exactly 1× at B=1, which is why a B=1 measurement of this cannot
  distinguish "it worked" from "the regime had no win".
  **CSR, no atomics.** `f32 +=` is not associative, so an atomic scatter would
  return a different vector each run on identical input while every
  tolerance-based test kept passing. Each `(row, dim)` is written by exactly one
  thread summing its contributions in ascending y-row order, which is
  batch-union (`pos`) order; `fused_reduce_is_bit_stable_across_repeats` asserts
  identical bits across repeats.
  It **does** move the GPU arm's low bits relative to the host reduce (GPU
  experts sum among themselves before meeting the CPU lane's contributions), so
  it is opt-in. The host adds the device partial in a fixed position — after all
  CPU contributions, in its own pass — so lane arrival order never reaches the
  arithmetic.
- **Every `pipe_*` op runs on `ctx->stream`** (fixed 2026-08-07). That stream is
  `cudaStreamNonBlocking`, so it does *not* implicitly synchronize with the
  legacy default stream — and `pipe_rmsnorm`, `pipe_rope`, `pipe_rows_add`,
  `pipe_gemm` and `pipe_copy2d` used to launch on the default stream while
  `pipe_silu_mul` and `pipe_add` used `ctx->stream`. A chain mixing them was
  **unordered**, not merely uncapturable. Two live-path instances too: the int4
  offset→signed conversion in `tensor_upload` / `tensor_update` ran on the
  default stream while `expert_group` consumes those weights on `ctx->stream`;
  both now synchronize before returning. When adding a `pipe_*` op, launching it
  anywhere but `ctx->stream` is a correctness bug, not a style choice — and a
  silent one, since capture omits foreign-stream ops without an error.
- **Dynamic residency**: `reheat()` re-selects the hottest experts every 256
  steps from the routing heat table. `COLI_GPU_TIER_SWAP=lfru|freq` replaces the
  whole-set re-plan with an incremental one-swap-per-layer policy
  (`peregrine-io::pick_lfru` / `pick_swap`), whose 25 %-plus-4-count hysteresis
  brakes PCIe churn at the source rather than truncating it after the fact. Initial placement is a greedy heat/bytes
  knapsack (`solve_residency_sized`) seeded from the previous session's
  `route_stats.json`, so a warm start places by last session's routing; with no
  usable stats it falls back to the deterministic round-robin spread.
  `COLI_REPLICATE_K` additionally mirrors the hottest residents into the CPU
  warm cache so a lane-balancer downgrade pays no disk read.
- **Mixed-precision residency**: `COLI_GPU_F32_FRAC` promotes the hottest
  fraction of residents to f32. The resident count and that fraction are sized
  *together* (`plan_precision_fitted`) — an f32 resident costs ~8× an int4 one,
  so sizing the set from the int4 footprint first and promoting afterwards let
  the promotions eat the whole budget before the int4 tail was reached (~67
  residents instead of ~204 at `frac=0.25` on a 10 GB budget). A promoted
  expert that no longer fits falls back to int4 rather than being evicted.
- **Mixed-format sources**: int4 residency is a preference. An expert whose
  source is grouped-int4 or int8 falls back to the f32 path on its own, instead
  of failing the whole tier at the first non-per-row-int4 expert. Such an expert
  is asked once, not once per generation — the tier remembers that its source
  cannot be int4, otherwise it re-uploaded forever.
- **Format-split dispatch**: the kernel picks its path from `all_s4` over the
  *whole* call, so one f32 resident would drop every expert in that call onto
  the generic scalar path. `compute` issues one `expert_group` per residency
  format and restores job order before returning, leaving the position-keyed
  reduce untouched. (This alone does not guarantee Tensor Cores — the TC path
  also requires every job in a call to clear the min-row threshold.)
- **PCIe budget**: `COLI_PCIE_BUDGET_MB` bounds how many bytes one `reheat`
  generation may upload, deferring the coldest experts to the next generation
  rather than bursting gigabytes into the lane it is feeding.

## Autotuning

`WmmaTuner` (`peregrine-model/src/wmma_tune.rs`) records per-shape
`(D, I, count, max_rows) → TileConfig` EWMAs and picks the winning tile per
shape. Wired into `GpuTier::compute` behind `COLI_CUDA_AUTOTUNE=1`, persisted to
`<dir>/kernel_tuning.json`, restored at load.

**Only the W4A16 arm is tunable, and only over three shapes.** WMMA fragment
shapes are compile-time (`wmma::fragment<..,M,N,K,..>` has no runtime form), so
"tunable tile size" can only mean selecting among instantiations — the kernels
are templated and the backend emits the three legal fp16 shapes (16×16×16,
32×8×16, 8×32×16). The int4 Tensor Core arm has exactly one legal `s4` fragment
(8×8×32), so `TileConfig::Int4Tc` records which kernel ran rather than offering
a choice; tagging it keeps an int4 measurement from sharing a table row with the
fp16 numbers, where the faster *arm* would read as the faster *tile*.

Three things this deliberately does not do:

- **It does not guess which arm ran.** `coli_cuda_expert_group_tiled` reports the
  arm, because that gate gets its own compute-capability and row-count checks — a
  group that missed the W4A16 row floor silently ran the scalar kernel, and
  crediting that time to the selected tile fills the table with measurements of a
  kernel the tile never touched. Tile-insensitive arms are not recorded at all.
- **It does not trust a restored winner.** `WmmaTuner::select` re-explores every
  legal shape before exploiting one, so a table carried between machines (or
  written by a different GPU) costs an exploration round rather than pinning a
  wrong answer. `kernel_tuning.json` is a separate file from `route_stats.json`
  for the same reason: one describes this host's GPU, the other the workload.
- **It does not claim bit-identity.** This file used to say fragment sizes
  "only affect performance". All three legal shapes share `K = 16` and the same
  k-loop, so identical per-element sum order is *expected* — but that is an
  argument about hardware reduction order, not a measurement.

## Persistent kernels — declined (2026-08-08)

Launch once and let threadblocks loop `dequeue → compute → enqueue`. The
hardware is present, so this is not deferred for want of a card — it is
declined, for four reasons worth writing down so it is not reopened without new
evidence:

1. **It conflicts head-on with the CUDA graph cache**, which shipped the same
   day. Both exist to remove per-launch overhead, and a persistent kernel cannot
   be graph-captured — capture would record a launch that never returns. So
   adopting it means *deleting* a tested, counter-instrumented feature to
   replace it with an untested one solving the same problem, and
   `graph_cacheable_arm` would need a third state to describe the overlap.
2. **It monopolises `ctx->stream`.** There is one non-blocking stream per
   device, so a resident kernel starves `pipe_*`, the attention entry points and
   `coli_cuda_matmul` — or forces them onto another stream, which reintroduces
   the cross-stream ordering defect fixed on 2026-08-07.
3. **It defeats the `scratch_gen` guard.** The graph cache survives a scratch
   reallocation by discarding the stale graph and recapturing. You cannot
   discard a kernel that is still running, so every `note_realloc` would have to
   drain and relaunch.
4. **It fails the repo's throughput test.** Launch overhead is not bytes read
   per token, and this workload is disk-bound.

It becomes interesting only alongside a device-resident forward, which is a
different project — see the note at the end of this section.

## What still needs hardware

**Correction (2026-08-06): this list used to say "needs `nvcc` + real hardware
this workspace lacks". The dev box has both** — CUDA 13.3 and an RTX 3060 — and
`benchmarks.md` records a measured run on it. What is genuinely out of reach
here is a *second* GPU, a second host, and a GDS driver stack.

**Each of those five now has a design rather than a bullet**:
[Designs for the work this box cannot run](scale-out-design.md) — multi-GPU
expert ownership and migration, GPUDirect Storage, NVLink-aware placement, VRAM
replication, and distributed sharding across hosts, each naming the `file:line`
seam that would change. The short version for the GPU lane: the tier's
single-device assumption is exactly two lines (`gpu.rs:1444` `init(&[0])` and
`:1447` `let device = 0`), because everything downstream already threads
`device` as a parameter and the `.cu` side already builds up to
`COLI_CUDA_MAX_DEVICES` contexts.

Of the items that are *not* hardware-blocked, persistent CUDA kernels are
declined above; the `cudaMallocAsync` pool is **closed on measurement
(2026-08-13)** — `coli_cuda_largest_free_block` (a binary search over real
`cudaMalloc` probes, ~13 of them at a 2 MB grain) reports the largest single
allocation that still succeeds, and after three rounds of the worst churn the
engine can produce (interleaved frees of the two expert block sizes with every
gap refilled at the *other* format) the RTX 3060 still hands back **96.7 % of
free VRAM as one block**. Fragmentation is the gap between that and 100 %, and
a pool cannot buy back ~3 % headroom the runtime itself holds.
`vram_churn_of_the_two_expert_block_sizes_leaves_free_memory_in_one_block`
pins it at a ≥ 50 % bar — generous enough to survive other tenants on the
card, three orders above the ~24 MB collapse a genuinely fragmenting
allocator would show — so if a future driver regresses this, the item reopens
by a test failure instead of a hunch. Idle-cycle GPU compute splits into an
engine-idle half that an in-tree negative result already argues against and a
mid-forward-spill half that is a different feature. See `todo.md`.

**CUDA Graphs into decode and the GPU-side fused reduce shipped on 2026-08-08**
(`COLI_CUDA_GRAPH`, `COLI_CUDA_FUSED_REDUCE` above) — as did the WMMA
autotuner's CUDA half (`COLI_CUDA_AUTOTUNE`). They were **compiled and unit-tested but had not executed a single kernel on
this box** until the 2026-08-08 14:08 reboot cleared a driver/library mismatch
(module 610.43.02 vs library 610.57) that made `cudaGetDeviceCount` fail and
every GPU-gated test skip itself. They now execute — 17 tests, all passing.

That first real run is why two things in this section changed. The graph cache's
scratch-generation guard turned out never to have been exercised: the test
covering it sized the scratch at the large shape *before* the graphed sequence,
so no reallocation ever happened and there was nothing to invalidate. Forcing
the guard off now SIGSEGVs the test, which is the correct behaviour of a
mechanism that stops a replay from reading freed VRAM. And `expert_group_tiled`
reported the wrong arm for a `None` tile, which silently cost the autotuner its
int4-arm samples — see `todo.md` §0 for both.

**The throughput claims in this section are still predictions.** That run
established correctness, not tok/s.

Three clarifications, because this list used to be wrong in both directions:

- **PCIe bandwidth scheduling is done**, not hardware-gated — see
  `COLI_PCIE_BUDGET_MB` above. The byte costs follow from the residency
  format, so the policy needs no device measurement.
- **CPU/GPU split GEMM is open by choice, not by hardware.** Splitting one
  expert's rows across lanes is small plumbing, but the CPU half computes int4
  and the GPU half f32, and a split point derived from the bubble tuner's
  wall-clock EWMA would make low-order output bits depend on machine timing.
- **A whole-layer graph is still much larger than "wiring", and is not done.**
  `COLI_CUDA_GRAPH` captures the launch sequence inside `expert_group` — the GPU
  work decode actually performs. Capturing all of `forward_layer` is a different
  project: every other step is host-resident and only MoE expert weights ever
  reach VRAM, so it needs device-resident attention/norm/router/embed weights and
  a device KV cache first. The primitives for that half exist in the `.cu`
  (`pipe_*`, `attention_absorb_kvdev`, `attention_project_batch_dev_out`) and are
  still reachable only from `peregrine-cuda`'s tests — sizing on GLM-5.2 shapes
  says it would fit (`kv_b` ≈ 490 MB at int4 across 78 layers, device KV ≈ 717 MB
  at 4 k context), but it competes with expert residency and none of it has run
  on this box. Shipping it behind a knob nobody could execute would recreate
  exactly the "implemented and tested with no caller" state the graph work just
  removed.

## Why this matters

On a single consumer box both engines are disk-bound and the GPU lane is
mostly latent. The concurrent design pays off in the **residency regime** —
enough VRAM to hold the expert working set — which is exactly where colibrì
reports 6.84 tok/s on 6× RTX 5090 with its *phased* loop leaving the overlap
on the table. See [Benchmarks](benchmarks.md).

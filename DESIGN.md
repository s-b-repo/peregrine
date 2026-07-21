# Rust rewrite of colibrì: true CPU∥GPU∥SSD∥RAM concurrency

> This is the design document for **peregrine** — the Rust spin-off of colibrì.
> It is preserved as originally written; references to `rust/` map to this repo's
> root and references to `c/` are to the upstream [colibrì](https://github.com/JustVugg/colibri)
> engine (the CUDA source is vendored here under `cuda/`).

## Context

the engine use RAM, CPU, GPU, and SSD in
parallel *at the same time* to improve throughput, and to build a Rust version that uses
io_uring to cut syscalls.

**What the research found:**

- **io_uring already exists in the C engine** (`URING=1`, Linux-only, `c/uring.h`). It is a
  hand-rolled raw-syscall ring (no liburing), batches up to 512 reads / 64 expert-loads per
  submit, forces `IOSQE_ASYNC`, and caps io-wq workers. It already replaces the blocking
  loader threads on the expert path. A Rust version *re-implements* this idea rather than
  inventing it.
- **CPU + GPU + SSD + RAM overlap already partly exists.** `PIPE=1` overlaps disk reads with
  matmul; `CUDA_DENSE`/`COLI_CUDA_ATTN` put dense/attention on the GPU while the CPU streams
  experts; a three-tier VRAM/RAM/disk hierarchy is implemented (`pin_load`, `resource_plan.py`,
  `tier.h`). The documented optimization stack already reaches **1.41 tok/s (4.3×)**.
- **The one real gap — the throughput lever.** Under CUDA, the MoE inner loop is *phased, not
  concurrent*: VRAM-resident experts are collected and deferred (`c/glm.c:3079-3081`), RAM/disk
  experts are computed on the CPU **inline** (`c/glm.c:3084-3101`), and the GPU expert group is
  dispatched **only after** that loop finishes (`c/glm.c:3124-3127`). The in-code comment
  measures the waste: *"9343 experts in VRAM sat unused during prefill — 81s of expert-matmul
  all on CPU, GPU groups 21ms total"* (`c/glm.c:2921-2923`). So on the same MoE block, CPU-expert
  compute and GPU-expert compute never run at once. Metal has a GPU∥disk overlap that CUDA lacks.

**Chosen direction** A **full from-scratch Rust engine**, targeting
**Linux + NVIDIA CUDA**, whose goal is to **beat current tok/s** by closing that phased gap —
a completion-driven scheduler where the GPU lane, CPU lane, and io_uring SSD lane all drain the
same layer's experts simultaneously. Byte-exact parity with the C engine is **not** required:
the project already documents that batched/GPU forwards round differently and that the bar is
"every emitted token is the argmax of a *valid* forward" (`README.md:60`). We hold that bar and
validate token-exactness on the single-token CPU path against the existing `transformers` oracle.

**Intended outcome:** a `rust/` cargo workspace producing a `peregrine-engine` binary that is a
drop-in for `c/glm` (same stdio serve protocol), matches the reference architecture semantics,
and delivers higher decode/prefill tok/s than the C engine on the same NVIDIA box by running all
four resources concurrently.

---

## Architecture — the concurrent 3-lane scheduler (the centerpiece)

Per MoE layer, after routing (Phase A) and batch-union dedup (Phase B), each **unique** expert
becomes exactly one `ExpertTask` (this structurally enforces the "compute each expert once, apply
to all its rows" invariant — stronger than the C `seen[]` scan). Tasks are classified O(1) by
residency into three lanes that run **as concurrent actors, not sequential phases**:

- **GPU lane** — one dispatcher thread per CUDA device. Consumes VRAM-resident tasks *and*
  disk-miss tasks the I/O lane promotes after a read completes. Coalesces ready tasks into
  `coli_cuda_expert_group`-style calls on the device's non-blocking stream, double-buffered
  pinned staging so H2D(n+1) ∥ kernel(n) ∥ D2H(n−1). Driven **continuously** via new
  non-syncing `_async` entry points + a CUDA event per batch (vs. the ABI's per-call
  `cudaStreamSynchronize`).
- **CPU lane** — a physical-core-pinned pool (`crossbeam` deque + `core_affinity`), outer
  parallelism over *experts* (inverting the C engine, which runs experts serially with each
  matmul internally OMP-parallel). Each worker runs one expert's SwiGLU: fused gate+up → silu·up
  → down → weighted scatter. Inner O-row tiling stays SIMD, no nested threading.
- **I/O lane** — one reactor thread owning the `io-uring` ring. Submits the coalesced ~19 MB
  gate/up/down read + 3 scale reads with `IOSQE_ASYNC` (mirroring `uring_load_add`,
  `c/glm.c:1863`). **On each CQE completion it never blocks** — it stamps the task's slab and
  routes the now-ready expert to whichever compute lane is shorter (GPU if VRAM staging free,
  else CPU). This is the CPU∥GPU∥SSD hand-off the C engine lacks.

**Accumulation (correct + non-serializing):** top-K means two experts can write the same output
row, so raw scatter races. Decode (small S): tiny lane-private accumulators summed once at layer
end (no locks). Prefill (large S): per-expert contiguous `partial` staging + one indexed reduce
sorted by `(row, expert-rank)` → deterministic run-to-run. A single atomic `remaining` counter
signals layer completion.

**PILOT prefetch, concurrent:** L+1's predicted-router experts are submitted into the *same*
ring as low-priority speculative reads tagged with a distinct `user_data` band; slab-arena
generation tags (the lock-free idea from `PipePool`'s gen-tagged cursor, `c/glm.c:2010`) replace
the C mutex + inflight barrier, so a straggler speculative load can never write a wrong-generation
slab.

**Why it beats the phased design:** for a decode block of 3 VRAM + 2 RAM + 3 disk experts, the C
wall-clock ≈ `max(disk_chain, cpu_5_experts) + gpu_3_experts` (GPU idle during the CPU phase); the
Rust wall-clock ≈ `max(gpu_lane, cpu_lane, disk_lane)` — the slowest single lane, not the sum.

---

## Crate & toolchain choices (Linux + CUDA + io_uring)

- **io_uring:** the `io-uring` crate (tokio-rs) with a **custom single-owner reactor thread** —
  *not* `tokio-uring` (its per-op Future model fights the batched-submit / `IOSQE_ASYNC` /
  io-wq-worker-cap ownership model the C ring uses). Keep O_DIRECT twin-fd + 4 KB base/len
  alignment and the 16 KB-aligned slab arena.
- **CUDA:** raw `extern "C"` FFI to the existing, validated `backend_cuda.o` first (flat ~40-fn
  ABI over opaque `ColiCudaTensor*`, `c/backend_cuda.h`). `build.rs` runs `nvcc` on
  `../c/backend_cuda.cu`, links `-lcudart -lstdc++` (mirrors `c/Makefile:191-193`). Add a few
  `_async` non-syncing stream variants for the scheduler. Defer `cudarc` (re-wrapping WMMA
  kernels = re-validation tax).
- **CPU parallelism:** custom physical-core-pinned pool, **not** rayon's logical-core global
  pool (README warns quantized kernels regress when SMT siblings contend for memory channels).
- **SIMD:** `std::arch` intrinsics with `is_x86_feature_detected!` runtime dispatch, **not**
  `portable_simd` — token-exactness needs exact AVX2 `maddubs`+`madd` / VNNI `dpbusd` / i8mm
  `smmla` accumulation order. This is the most correctness-sensitive module.
- **safetensors:** hand-rolled pread-based index (mirror `c/st.h` with `fadvise(DONTNEED)` +
  O_DIRECT to keep RSS flat), header via `serde_json` — **not** the `safetensors` crate (mmaps,
  no DONTNEED/O_DIRECT control). `memmap2` behind a `COLI_MMAP` flag. `half` for bf16/f16→f32.
- **tokenizer:** `tokenizers` crate to bootstrap, validated id-for-id against `c/tok.h`; hand-port
  `tok.h` (~400 lines) if the pretokenizer/added-token handling diverges.
- **Support:** `crossbeam`, `core_affinity`, `bytemuck`, `parking_lot`, `serde_json`, `clap`.

---

## Repo layout & build

New cargo workspace `rust/`, sibling of `c/` and `desktop/` (keeps the flat `c/` runtime intact):

```
rust/crates/
  peregrine-core/     # QT formats (fmt 0..4), Cfg, safetensors index      (↔ c/st.h)
  peregrine-kernels/  # std::arch int4/int8/int2 + f32 matmul, token-exact  (↔ matmul_qt_ex, glm.c:978)
  peregrine-io/       # io-uring reactor (I/O lane), slab arena, LRU/pin    (↔ c/uring.h, tier.h)
  peregrine-cuda/     # -sys FFI to backend_cuda.h + build.rs(nvcc) + wrapper
  peregrine-model/    # MLA, router, MoE, DSA, MTP + the 3-lane scheduler   (↔ glm.c forward)
  peregrine-engine/   # binary: clap CLI + stdio serve protocol (drop-in for c/glm)
```

- Build: `cargo build --release --features cuda`; CPU-only drops the `cuda` feature (pure-CPU,
  like `make` without `CUDA=1`).
- Integration: `c/coli` gains a `--engine rust` / `COLI_ENGINE=rust` branch; the Rust binary
  speaks the existing `openai_server.py` stdio protocol (`Popen([exe, cap])`, `READY`/`END`/
  `CANCEL` sentinels, `c/openai_server.py:457-476`) so `serve`/`web`/desktop need zero changes.
- Feature flags mirror C env knobs (`URING`, `DIRECT`, `PIPE_WORKERS`, `PIN_GB`, `CUDA_EXPERT_GB`,
  `RAM_GB`, `TOPP`, `DRAFT`, `DSA`), preserving C precedence (explicit flag > env > auto).

---

## Milestones (each independently verifiable; start on the 2.4 MB tiny-random model)

| M | Goal | Verify |
|---|---|---|
| **M0** | Workspace; parse `config.json`/tokenizer/safetensors header; load tiny-random model | tensor inventory matches `st.h`; tokenizer round-trips id-for-id vs `tok.h` |
| **M1** | **CPU-only int4 forward, token-exact**: MLA (q/kv-LoRA, partial RoPE, absorption), sigmoid router, dense+shared+routed MoE, SIMD int4/int8/int2 kernels | `TF=1` 32/32 + greedy 20/20 vs oracle `ref_glm.json` (the README:57 bar) |
| **M2** | io_uring streaming + LRU/pin tiers on the **real 744B model**, CPU-only | coherent decode; hit-rate/disk-wait counters same order as `./glm`; warm-cache A/B within "valid forward" tolerance |
| **M3** | CUDA expert lane via FFI (still phased): link `backend_cuda.o`, upload VRAM tier, route VRAM experts through `coli_cuda_expert_group` | `tools/benchmark_cuda_fixture.py` on 313M fixture; CPU vs CUDA same tokens |
| **M4** | **The concurrent 3-lane scheduler** (centerpiece): completion-driven dispatch, CPU∥GPU∥SSD on the same layer, sharded/indexed accumulation, PILOT via ring | **tok/s beats C engine** on a matched box; argmax stream unchanged vs M3; profiler shows GPU busy during CPU/disk |
| **M5** | MLA weight-absorption + DSA lightning indexer (top-2048, auto-detected from `out-idx-*`) | DSA-off reproduces dense attention token-for-token (README:67); absorption TF 32/32 |
| **M6** | MTP speculative decode (int8 head draft + batch-union verify) | 39–59% acceptance / 2.2–2.8 tok/fw on int8-head model (README:60); rejection-sampling correct under sampling |
| **M7** | serve / OpenAI drop-in: stdio `READY`/`END`/`CANCEL` child | `openai_server.py` spawns Rust binary unchanged; `curl` chat streams; web UI works |

De-risking order: CPU-only tiny model first (M0/M1) → reuse `.cu` via FFI before the scheduler
(M3 before M4) → keep the phased path as a correctness oracle for M4.

---

## Reuse decisions (keep via FFI / process boundary)

- **`c/backend_cuda.cu` kernels via FFI** — validated WMMA/quant/attention over the flat C ABI;
  compiled by `peregrine-cuda/build.rs`. Rewriting them early buys nothing and costs re-validation.
- **`openai_server.py` gateway + `coli` CLI** — the Rust binary is a drop-in for `c/glm`; gateway,
  web UI, desktop unchanged.
- **Oracle/eval tooling** (`tools/make_glm_oracle.py`, `eval_glm.py`, `ref_glm.json`, `ref.json`)
  — the correctness gate at every milestone, reused verbatim.
- **`resource_plan.py`** (header-only planner) and the **FP8→int4 converter** + int4/int8 container
  format (incl. int8 MTP heads) — unchanged; the Rust engine consumes the same files.

---

## Risks & mitigations

- **Token-exact hand-written SIMD (highest risk):** `std::arch` (controlled accumulation order);
  port `qrow_i8` / `matmul_*_idot` / `matmul_i4_pair` exactly; validate each kernel bit-identical
  to a NOPACK-style f32 reference before integration; chase byte-parity only on the S=1 CPU path.
- **Re-validation enormity:** reuse the exact oracle harness every milestone; tiny model keeps the
  loop at seconds; gate on TF 32/32 + greedy 20/20.
- **io_uring under O_DIRECT:** keep 4 KB base/len alignment + 16 KB slab arena; buffered twin-fd
  fallback when unaligned; property-test the alignment arithmetic.
- **Memory/OOM:** port `cap_for_ram` auto-sizing from `MemAvailable`; bound the slab arena; reserve
  ~2 GB/device VRAM headroom before placing the expert tier.
- **Scheduler correctness:** the one-`ExpertTask`-per-unique-expert queue enforces the batch-union
  invariant; generation-tagged slabs prevent stale speculative writes; keep the phased path
  available as a differential oracle for M4.

---

## Critical files (C references to port/bind)

- `c/glm.c` — `moe()` phased loop **2658–3191** (the exact serialization to replace: CPU inline
  3084–3101, GPU group after 3124–3147, the measured-waste comment 2921–2923); `matmul_qt_ex`
  978; `expert_load` 1641; `layer_forward_rows` 3629; `spec_decode` 4146; `pin_load` 5432.
- `c/uring.h` — io_uring ownership / `IOSQE_ASYNC` / io-wq model for the Rust I/O lane.
- `c/backend_cuda.h` / `c/backend_cuda.cu` — flat C ABI to FFI; add `_async` stream variants.
- `c/st.h` — safetensors index + fadvise/O_DIRECT streaming behavior.
- `c/tier.h` — LFRU eviction/promotion math for the tier manager.
- `c/openai_server.py` — `READY`/`END`/`CANCEL` stdio protocol the Rust binary must implement.

## Verification (end-to-end)

1. Per-milestone oracle gate: `SNAP=./glm_tiny TF=1 <rust-engine> ...` reproduces `ref_glm.json`
   (TF 32/32) and greedy 20/20 — the same bar the C engine passes (`README.md:57`).
2. M4 throughput proof: run the documented optimized stack config on the same NVIDIA box for both
   engines; the Rust `peregrine-engine` must exceed the C `tok/s`, with a profiler trace showing GPU,
   CPU, and io_uring lanes busy simultaneously within a layer (the phased C trace shows them
   sequential).
3. Drop-in proof: point `c/openai_server.py` at the Rust binary and run a `curl` chat completion +
   the web dashboard — unchanged behavior confirms the stdio protocol match.

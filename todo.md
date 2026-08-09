# peregrine — Optimization Checklist

Roadmap of throughput/latency optimizations for streamed-MoE inference, distilled from `todo.txt`,
with **implementation status audited against the codebase** (2026-07-24; §1 Prefetching completed 2026-07-25;
**adaptive & disk-layout wave shipped 2026-07-30** — this update).

Goal: **maximize tokens/sec for giant streamed MoE models** by eliminating the remaining sources of
idle time — SSD latency, GPU launch overhead, and suboptimal expert placement — on top of the existing
concurrent CPU/GPU/SSD scheduler + `io_uring`.

**Status legend:** `- [x]` ✅ Done · `- [ ] 🟡` Partial (scaffolding / incomplete) · `- [ ]` ⬜ Not started
**Ratings** (where the source ranked them) — gain ★☆☆☆☆→★★★★★ · Difficulty: Easy/Medium/Hard

---

## 📊 Completion Dashboard

| Scope | ✅ Done | 🟡 Partial | ⬜ Not started | Total | Completion |
|---|---:|---:|---:|---:|---:|
| **Full roadmap** | 127 | 1 | 9 | 137 | **~93% strict · ~93% weighted** |
| **Priority shortlist** | 17 | 0 | 3 | 20 | **85% strict · 85% weighted** |

*Strict = Done ÷ Total. Weighted = (Done + ½·Partial) ÷ Total. "Fast matrix multiplication" is excluded.
Total is now 127, up from the 108 this table tracked through 2026-08-01: the 93 source items, 2
shipped extras in §6 (adaptive prefill/decode window, telemetry feedback loop), 2 VRAM-residency
fixes in §3, 1 open dispatch item in §2, the 10 items of §12-§13 — an axis the original 11 sections
had no category for (attention/serving memory, and reading fewer experts rather than reading them
faster) — and 17 more the research waves added or split out. Adding them *lowers* the percentage,
which is the honest reading: the roadmap was ~89% done against a scope that excluded them, and the
strict figure has barely moved since because each wave opens roughly as much as it closes. Counts are generated from the checkboxes below — recount with
`awk '/^## 1\./,/^## ❄️/' todo.md | grep -c '^- \[x\]'`.*

**Per-section:** Prefetch **11/11 ✅** · GPU 8/10 · Caching **13/13 ✅** · I/O 10/11 · Memory/NUMA
**8/8 ✅** · Scheduling 16/18 · Disk-layout **10/10 ✅** · Workload **5/5 ✅** · Compilation
**5/5 ✅** · Self-optimizing **10/10 ✅** · Multi-GPU 0/4 · Attention/serving **7/7 ✅** ·
Workload-reduction **15/16**.

*2026-08-03 wave (+5, all from a cross-read of two parallel projects —
[WASTE](https://github.com/sqliteai/waste) and [deltafin](https://github.com/gavamedia/deltafin)):
the router look-ahead and its scoreboard in §1, trunk wiring and cgroup-aware budgeting in §5, the
response memo in §12. The borrowed **negative** results are recorded in
[prefetch-and-caching.md](docs/prefetch-and-caching.md#borrowed-negative-results) — they close off
per-expert bit allocation, routed-tail truncation and prefill-path look-ahead, and they put an
open question against §1's entire statistical predictor stack.*

## 📌 What is actually left (2026-08-08)

**Ten roadmap line-items (1 partial + 9 not started).** The 2026-08-08 wave
closed §2's CUDA-graph decode wiring and its GPU-side fused reduce — the last
two engineering 🟡s — and took the `[R]` dead-code count **29 → 22** by wiring
`moe_engine_installed`, `pick_lfru`/`pick_swap`, `speculative_sample`, and
`WmmaTuner`'s `best_for`/`default_w4a16`/`default_int4tc`.

**Later the same day the box was rebooted, and the two verification blockers
below both came off.** What the GPU lane's first real execution found is in §0
just below this list; the int2 conversion
finished at 58 shards; and `peregrine flip-rate` now exists, so the int2 quality
gate is a running measurement rather than a missing tool. The line-item counts
are unchanged — no roadmap item opened or closed — but three claims in this file
were wrong and are corrected below.

**1. Needs hardware this box does not have (5).** §4/§11: GPUDirect Storage,
multi-GPU ownership + migration, NVLink placement, VRAM replication,
distributed sharding. One GPU, one host, consumer card with no GDS stack.
**Each now has a design** — [`docs/scale-out-design.md`](docs/scale-out-design.md),
one section apiece, every one naming the `file:line` seam that changes. The two
worth wanting are multi-GPU ownership (the hardcode is exactly two lines,
`gpu.rs:1444` and `:1447` — everything downstream already threads `device`) and
distributed sharding (a `RemoteMoeEngine` implementing the existing
`MoeEngine` trait is the *entire* model-side integration point, and at 19–151 MB
per expert the network is faster than this box's disk). Neither is started; the
point of the page is that starting is a hardware question, not a design one.

**2. CUDA work this box could do (3) — resolved by evidence, not by building all
three.** None of them cuts bytes read per token, which is this file's own test
for moving tok/s, so each got the cheapest measurement that could justify it
rather than an implementation on spec:

- **Persistent kernels — declined.** They remove per-launch overhead, which is
  what the CUDA graph cache shipped on 2026-08-08 already does, and the two
  cannot coexist: a persistent kernel cannot be graph-captured, so adopting it
  means *deleting* a tested, counter-instrumented feature to replace it with an
  untested one solving the same problem. It also monopolises `ctx->stream` (one
  non-blocking stream per device), starving `pipe_*`, attention and matmul —
  which recreates the cross-stream ordering defect fixed on 2026-08-07 — and it
  defeats the `scratch_gen` guard, whose whole mechanism is *discard and
  recapture* and you cannot discard a running kernel. Reasons recorded in
  `docs/gpu-cuda.md` so this is not reopened without new evidence.
- **`cudaMallocAsync` defrag pool — measure first, and expect to close it.**
  There are exactly **two** distinct block sizes (`int4_bytes`, `f32_bytes`) and
  startup is a monotone allocation burst, which is the textbook non-fragmenting
  workload. The decisive number is largest-free-block vs free-bytes, which
  `coli_cuda_mem_info` is one binary-searched `cudaMalloc` away from reporting.
  Build the probe, not the pool.
- **Idle-cycle GPU compute — the item was two items, and one of them is
  declined.** *Engine-idle* GPU warming has a ready-made hook
  (`Model::idle_maintenance`) and a result already in the tree against it:
  `model.rs:3482-3492` records speculative work tripling the demand hit rate for
  **+6.9 % bytes and no wall-clock movement**. It fails the byte test in the
  wrong direction. *Mid-forward GPU spill* is the half worth keeping open, and
  it is not an idle-cycle feature at all — `lane.rs:232-241` already documents
  the blocker in code: `Placement::GpuSpill` is advisory-only because acting on
  it needs `&mut GpuTier` during a forward that holds the tier by `&`.

**3. Open by choice (1) — declined; the replacement is named.** §6 CPU/GPU split
GEMM fails this file's own test twice: it cuts **zero** bytes per token (same
routed union) and it *adds* a streamed read, because the CPU half needs whole
weight matrices on an engine whose whole problem is bytes. A timing-independent
`COLI_SPLIT_FRAC` would have fixed the determinism objection while leaving both
of those intact. What replaces it, by the same test:

- **`COLI_ROUTE_MIN_SHARE`, sized by the flip-rate gate — no new code.** The
  2026-08-07 counters give it a real ceiling for the first time: at B=16,
  **12.5 % of routed selections carry under 5 % of the gate mass**
  (`below_0.5%=1.0%`, `below_1%=1.3%`, `below_2%=2.4%`, `below_5%=12.5%`). The
  knob ships and is off only because nobody had a quality number, and the number
  is now one command away — run `peregrine flip-rate` with the knob set on the
  candidate side and unset on the source, *same container both times*, so there
  is no conversion cost at all. This is the highest value-per-hour item on the
  roadmap.
- **int2-g64 is not the answer** — measured this pass at `flip_rate = 1.000`.
  See §13. `int3-g64` (12.7 % smaller than int4) is the untested rung.
- Speculative, named so it is not rediscovered: expert-delta or shared-basis
  coding is the only idea here with a ceiling below 2 bits/weight, but it breaks
  `QtInfo::detect`'s per-tensor self-description and there is no evidence
  GLM-5.2's experts share a basis. One afternoon of singular-value spectra on a
  few real expert matrices would open or close it.

**4. Machine time, not engineering (1) — now spent, and the answer is negative.**
§13 int2. The conversion finished (118 478/118 478 tensors, 58 shards,
383.73 GB → **286.29 GB**, 74.6 %), the `peregrine flip-rate` runner was built,
and the gate ran: **`flip_rate = 1.000000` on 512 positions of prose — zero
top-1 agreement.** The container decodes correctly (cosine similarity 0.916
against the source on an independently-decoded tensor); the *scheme* is what
fails, at 45 % relative reconstruction error, which is the textbook figure for
four-level round-to-nearest. **int2-g64 is closed as a measured negative**, not
carried as a pending item. Details in §13.

  **Measured, and the answer is no: `flip_rate = 1.000000`, 512 of 512
  positions.** 512 positions of English prose, tokenized with the source
  container's own `tokenizer.json`, one teacher-forcing forward each — 760 s for
  the int4 source, 2 348 s for int2-g64 over the USB 2.0 link. Zero agreement.

  **The container is not broken; the scheme is.** That distinction is worth the
  paragraph, because "we shipped a corrupt container" and "2-bit RTN does not
  work" call for opposite responses. Decoding
  `model.layers.13.mlp.experts.195.gate_proj.weight` out of both containers
  independently — outside the engine, straight from the safetensors bytes —
  recovers the *same tensor* at **cosine similarity 0.916**, so the producer, the
  `2·o·ng` scale cardinality and `QtInfo::detect`'s Int2G64 branch all agree.
  What it also shows is **45.05 % relative reconstruction error**, and that
  number is not a bug either: four levels spanning each group's own range give a
  step of `range/3`, the range of 64 roughly-Gaussian samples is ≈5σ, so the
  predicted RMS error is `5σ/(3·√12) ≈ 0.48σ` — within a few percent of what was
  measured. The format is behaving exactly as designed. **Round-to-nearest at 2
  bits is simply not enough for this model**, and 45 % per tensor compounded
  across 78 layers × 3 projections is why top-1 agreement is not merely degraded
  but gone.

  So the 2× byte saving (11.35 → 5.69 GB/token) is **not available this way**.
  What would be needed is a 2-bit method that does not quantize each weight
  independently — vector quantization or incoherence processing, as AQLM and
  QuIP# do — which is a research project, not a converter flag. Recorded as a
  closed negative result rather than a pending one: the container exists, the
  gate ran, the answer is in. `int3-g64` (3.5 bits effective, 12.7 % smaller than
  int4) is the untested point on this ladder and the only one still worth a run.

  **The "blocked on the cable" claim this bullet used to make was wrong**, and
  worth correcting because it stopped a measurement that was always available.
  It cited ~2.3 minutes per token from 5.69 GB/token at the USB link's 41.7 MB/s
  — but that is the *decode* cost, and the gate runs under **teacher forcing**:
  `Model::teacher_forcing` does **one** forward over the whole prompt and returns
  an argmax per position. Reads are the routed *union*, which saturates — at
  top-8 over 256 experts, a 128-position forward already touches ~98 % of the
  container, so 512 positions cost ~2 % more bytes than 128 and buy 4× the
  statistics. The cable charges ~2 h **once**, not per token. Separately, a flip
  rate is a *quality* number: a slow link changes how long it takes to obtain,
  not what it means.

---

### §0 · What the first post-reboot GPU run found

*This section replaces the "the GPU is currently unusable" warning that stood
here. The box rebooted 2026-08-08 14:08; kernel module and userspace CUDA driver
are both **610.57.04**, `nvidia-smi` reports the RTX 3060 12 GB, and
`cargo test -p peregrine-cuda --features cuda` now **runs** instead of
self-skipping.*

The six tests that had never asserted anything asserted, and **one of them
failed**. Closing the section's other open claim — that only 16×16×16 had ever
run — then required a test that did not exist, and writing it turned up a second,
unrelated bug. Two defects from one afternoon of finally executing the code is
the argument for the warning having been written at all.

**1. `a_grown_scratch_buffer_invalidates_cached_graphs` failed, and the test was
wrong, not the guard.** `reserve` is grow-only, so the test's own eager
reference at the large shape sized the scratch *before* the graphed sequence
ran. The graphed large call then found `*cap >= bytes`, returned without
freeing, `note_realloc` never ran, the generation never moved, and there was
nothing to invalidate. The test asserted the consequence of a premise it had
destroyed two lines earlier. Fixed by taking the large reference *after* the
graphed sequence (eager recomputes from the weights either way) and by making
the large shape dominate every other shape in the module.

  Verified by mutation rather than by re-reading it: with `s->gen !=
  ctx->scratch_gen` forced false, the test **SIGSEGVs** — a replay reading freed
  VRAM, exactly the silent failure the guard exists for. The guard was correct
  all along; nothing had ever exercised it.

**2. The three fp16 WMMA instantiations now all execute.**
`every_w4a16_tile_instantiation_agrees_with_the_default` drives 16×16×16, 32×8×16
and 8×32×16 and checks them against the default (a tolerance check, not
bit-identity — the fragment shapes accumulate in different orders). Because
`w4a16_*_dispatch` **silently falls back to 16×16×16** on an unmatched triple, a
passing agreement test cannot by itself prove the other two ran; that was
established by perturbing `fill_fragment` under `TM==32` and then `TM==8` and
confirming the matching tile — and only that tile — fails.

**3. Writing that test surfaced a real bug in `expert_group_tiled`.** It matched
on `tile` to choose the C entry, so `tile: None` went to the **untiled**
`coli_cuda_expert_group`, which has no `arm_out` parameter — leaving `arm` at its
`-1` sentinel, which `GroupArm::from_c` maps to `Generic`. So
`expert_group_tiled(.., None)` reported `Generic` no matter which arm ran.
`gpu.rs` passes exactly that whenever the tuner selects an int4 tile
(`w4a16_dims()` returns `None` for `Int4Tc`), so its `match arm` dropped the
int4-arm observation the comment beside it says it deliberately keeps — and
dropped it *permanently* once `Int4Tc` became the recorded best for a shape, or
was loaded as one from `kernel_tuning.json`. Now keyed on whether the caller
asked for the arm; `{0,0,0}` is the C side's default tile. `expert_group` still
reaches the kernels through the historical untiled entry.

**Still true:** no *throughput* claim in §2 has been measured. 17/17
`peregrine-cuda` tests and the full workspace suite pass, which is a correctness
statement. Run the runbook's GPU arms before believing any tok/s number.

**Also outstanding, outside the roadmap sections:** the device-resident forward
half of the graph work — `pipe_*`, `attention_absorb_kvdev`,
`attention_project_batch_dev_out` are still reachable only from
`peregrine-cuda`'s tests. Sizing says it fits (`kv_b` ≈ 490 MB at int4 across 78
layers, device KV ≈ 717 MB at 4 k context), but shipping it behind a knob that
cannot be executed would recreate exactly the "implemented and tested with no
caller" state this wave removed. `docs/gpu-cuda.md` records the plan.

---

*Historical note — the framing this section replaced:* **the 19 open items fell into three groups**, and only the first was blocked by this workspace:

1. **Needs hardware (5, not 10 — corrected 2026-08-06).** This entry said ten items needed "`nvcc`
   + NVIDIA hardware this workspace lacks". **The dev box has both**: CUDA 13.3 at `/opt/cuda` and
   one RTX 3060 12 GB, all six `peregrine-cuda` tests pass on it (graph capture included), and
   `docs/benchmarks.md` §GPU lane records a measured run at 1.09×. The claim survived because the
   *other* half of it was true: `build.rs` mapped "no nvcc" and "the `.cu` does not compile" onto
   the same warning and a success exit, so no `.cu` edit was ever compiled and the lane looked
   untouchable. That is fixed — a present-but-failing toolkit now fails the build.
   What is actually out of reach here is a **second GPU, a second host, and a GDS driver stack**:
   §11 multi-GPU ×3 + distributed hosts, and §4 GPUDirect Storage. The other five — §2 persistent
   CUDA kernels, CUDA-graph decode wiring 🟡, GPU-side fused reduce 🟡, GPU defrag pool, and §6
   idle-cycle GPU compute — are buildable and measurable on this machine today.
2. **Open by choice (1).** §6 CPU/GPU split GEMM. The plumbing is small (row sets are disjoint, so
   the `pos`-keyed reduce order is untouched), but the CPU half computes int4 and the GPU half f32,
   and a split point taken from the bubble tuner's wall-clock EWMA would make low-order output bits a
   function of machine timing — the same prompt giving different logits run to run. Today's
   divergence is at expert granularity and deterministic given the residency set; this would be
   strictly worse. It also *adds* a streamed disk read, since the CPU half needs whole weight
   matrices. Reopen only with a timing-independent split policy and an explicitly downgraded guarantee.
3. **Pure CPU work, nothing blocking but effort (8).** §12 fuse prefill into the decode batch, KV
   quantization, paged/block-pooled KV; §13 int2 checkpoint conversion 🟡, heat-tiered on-disk
   precision. These need no hardware at all — they are open because each is a substantial change to a
   core invariant (the batched one-token-per-sequence API, the `pos == len` append contract, or a
   requantizing converter this repo does not have), plus three features that ship tested
   but unreachable and only need wiring: §4's registered-buffer read path, §1/§10's
   `perf_event_open` counter, and §5's slab generation tagging — all found by the [R]
   reachability pass (`docs/BAD_PATTERNS.md`). **This is where the remaining throughput is**,
   and it is now measured rather than inferred: the [2026-08-01 pass](docs/benchmarks.md#benchmark-pass--2026-08-01-post-improvement-re-measure)
   put nine §1–§11 knobs together at **1.004×** with byte-identical disk reads.

Training-loop and research-scale items were completed in their user-approved pragmatic forms, marked
**pragmatic** inline.

**Headline:** the big **adaptive-runtime wave** landed this pass. The three lanes are now instrumented
(`LaneTimings` + `BubbleTuner`) and the placement decision they drive (`LaneBalancer`) is live; the
io_uring worker cap is EWMA-tuned by `IoTuner`; the warm cache gained a Bloom-filter miss shortcut,
negative-TTL eviction and transparent zstd compression; the checkpoint reader/writer speak zstd
end-to-end; a new **`peregrine-layout-reorg`** binary emits `schedule.json` (greedy or Louvain
communities) that the loader consumes to coalesce batched disk reads; per-forward routing history +
GPU heat table now persist across sessions in `route_stats.json`; the batching engine has a
two-priority queue, a latency-SLA-adaptive batch cap, and an optional decode-heavy window;
`PredictSource::PhaseAware` boosts recency on Jaccard-distance shift; runtime expert replication
warms hot GPU-residents into the CPU warm cache so a bias flip pays no disk. A follow-up sweep the
same day landed the tails: **NUMA thread pinning** wired at the par-pool + prefetch-pool spawn sites
(via a std-only worker-startup hook), a **heat-threshold cache-admission gate**, a **spectral
(Fiedler) ordering** method in the layout tool, a **real `perf_event_open` LLC-miss counter**
(hand-declared VER0 ABI), **background recompression** of cold cache slots on engine-idle ticks, and
**end-to-end token-class plumbing** (HTTP-handler classifier → `EngineRequest.class` → per-class
prefetch-breadth env overrides). Everything is env-gated and bit-identical when off. A third wave (**the
completion sweep**) then closed every non-hardware item: sensor governors (thermal / RAPL power /
memory-bandwidth) steering an adjustable worker count, routing-entropy-adaptive prefetch, NUMA
`mbind` allocation + hierarchical two-level pool dispatch, per-expert adaptive mixed precision,
co-activation-driven expert fusion + hypergraph scheduling, macro-state routing compression, the
`galactic` one-shot preprocessing pass, Hilbert / 2-opt / tier-placement layout methods, physical
checkpoint self-rewrite (`--apply`), online bandit + Q-learning schedulers, per-shape dispatch
specialization, kblock tensor-layout auto-conversion, and the `compile-plan` profile-guided
execution plan. Beyond the roadmap, the serve layer gained the **vendored gigatoken BPE tokenizer** (`peregrine-token`, MIT, stable-toolchain subset) as its sole runtime tokenizer — parity-gated id-for-id against the HF `tokenizers` test oracle, 34× measured locally.

**Since then, three further waves.** A **VRAM-residency pass** fixed defects the roadmap's own
bookkeeping had hidden: the heat/bytes residency knapsack was ✅ and documented as live while having
*zero production callers* (initial placement was round-robin by index); promoting residents to f32
collapsed the resident set ~3× because the count and the promoted fraction were sized independently;
one non-per-row-int4 expert truncated the whole tier; and an f32-fallback expert was re-uploaded on
every reheat generation forever. It also shipped format-split expert dispatch and a PCIe upload budget.

Then an **attention/serving pass** opened §12–§13 — an axis the original eleven sections had no
category for. Caching and prefetch plateau on this workload, so the remaining wins are not in moving
11.3 GB/token faster but in moving less of it. (The **0.6%** figure usually cited here is a
*warm-cache hit rate* at a 10 GB cache, not a routing statistic — corrected 2026-08-02 in
`docs/peregrine-vs-colibri.md` §5.2; `peregrine route-stats` measures the routing quantity.) Landed: **adaptive prefill chunking** (a fixed 64-token chunk made prefill *quadratic* —
an 8192-token prompt re-derived ~524k KV rows instead of 8192, per layer, across 78 layers), a
**cross-request KV prefix cache** (a shared system prompt was prefilled once per request), **adaptive
top-k**, and the **first int2 producer** for a format that had been consumable-but-unwritable since M1.

Two measurement tools shipped alongside, because the remaining levers are lossy and the suite is built
entirely on bit-identity anchors — which a lossy change fails by construction, leaving no way to say
what it cost. `COLI_GATE_STATS` tallies how much of the routed set carries a negligible gate share;
`Model::prediction_flip_rate` is the first non-equality quality gate in the repo.

---

## 🎯 Which shipped work can actually move tokens/second

The measurement that decides this is the repo's own: the 2026-08-01 pass ran a
nine-knob adaptive/IO bundle at **1.004× with byte-identical disk reads**
(0.225 vs 0.224 tok/s, B=16, GLM-5.2 744B int4). The knobs changed *how* bytes
were fetched, not how many, and this workload is disk-bound. So the test for
any change is one question: **does it cut bytes read per token?**

Sorting the landed work by that test — because "it made things faster" is the
claim this repo has most often had to walk back.

**Can move tok/s**

- **Speculation (`COLI_DRAFT`)** — *this entry was written on an assumption the
  2026-08-07 run falsified.* It claimed "one expert union now yields
  `1 + accepted` tokens instead of 1", i.e. that the drafts ride along on the
  verify row's reads for free. **They do not.** At B=16, γ=4 the union grew from
  3 855 distinct expert reads to **10 145 — 2.63×** (`COLI_UNION_STATS`). Draft
  rows route substantially different experts, so speculation buys at most `1+γ`
  tokens for 2.63× the bytes on an engine whose whole problem is bytes.
  Break-even needs an accepted run above ~2.6, not above 1.

  Measured verified throughput on that run was **0.112 tok/s against a 0.243
  baseline — 2.2× slower**. **Read that number with its harness in mind**: the
  bench seeds each sequence with a single arbitrary token and runs two steps, so
  the MTP head drafts with essentially no context and acceptance is near its
  worst case. What is *not* harness-dependent is the union growth, because it is
  a property of where the routed sets fall, not of whether the drafts were right.
  A real-workload acceptance measurement is still owed; the published 2.46 at
  depth 2 for this model class is now a figure to test, not to rely on.
- **Prefill fusion (`COLI_FUSE_PREFILL`)** — one union instead of two, but only
  on ticks that have both prefill and decode.
- **An int2-g64 container** — ~25% fewer expert bytes, but needs a converted
  checkpoint and a flip-rate measurement first.

**Cannot move tok/s, whatever else they are worth**

- **f16 KV, the `Arc` prefix, the growth cap** — memory and admission, not
  bytes streamed. They raise *concurrency* under `COLI_KV_BUDGET_MB`, which
  lifts aggregate throughput, not per-stream.
- **DSA (`COLI_DSA`)** — caps attention compute at long context; expert reads
  are unchanged.
- **Expert pruning (`peregrine-prune`)** — smaller working set, identical
  activated parameters. Its own report says so, in its own output.
- **The adaptive/IO knob bundle** — this is the 1.004× itself.

This is a decomposition, not a measurement. No tok/s number is claimed for any
of it: this workspace cannot load the checkpoint, and the synthetic model
(3 layers, 16 hidden, fully resident, no streaming at all) is five orders of
magnitude away and says nothing. `docs/validation-runbook.md` §2 and §6 are the
procedures — and §6 specifically says to measure fusion with `COLI_UNION_STATS`
rather than wall clock, because wall clock cannot distinguish "it worked" from
"the workload had no mixed ticks".

---

## 💽 Can the disk read match or beat colibrì?

Asked often enough to be worth settling in writing. The short answer is that
the gap most people quote is not measuring what it looks like, the engine-side
lever already exists as a knob, and the largest lever is not code at all.

**The 0.84 vs 2.02 GB/s gap is partly a harness artefact.** `benchmarks.md`
records why: colibrì's 2.02 comes from `c/iobench.c`, which is **not io_uring
at all** but 8 OpenMP threads issuing blocking `pread`s — while **its own
production decode path is a 512-deep io_uring queue**. The two harnesses
measure different things. On the other box the comparison *inverts*: peregrine
~980 MB/s against colibrì's ~870 MB/s after the O_DIRECT and parallel-ring
work. There is no stable "colibrì is 2.4× faster" fact to chase.

**colibrì's approach is already selectable.** `COLI_IO_ENGINE` takes `uring`
(the historical batched submit), `pread` (N blocking-`pread` threads —
colibrì's shape) or `regbuf`, all behind the same `read_regions` choke point,
so output is **byte-identical** and the three A/B against a bit-identity
assertion rather than being eyeballed. If threaded `pread` wins on a given
drive, it is one env var away. The dev-box run was **inconclusive and is
recorded as such** — `pread` led on one file pair (2.02 vs 1.68 GB/s) and
trailed on another (1.16 vs 1.26), at different sizes and page-cache states.

**The largest lever is storage configuration, not the engine.** A LUKS volume
at the 512-byte default sector size measured **~10% of raw throughput**;
`--sector-size 4096` alone restored it to ~50%, roughly **5×**. Nothing in the
I/O lane competes with that. Linux I/O schedulers cost **14–57%** versus `none`
on NVMe. Neither is code, and both are checked in one command each
(runbook §1b).

**Order to work in**, so effort lands where the ratio is:

1. `cryptsetup luksDump | grep -i sector` (want 4096) and
   `cat /sys/block/nvme*/queue/scheduler` (want `[none]`). If the volume is
   512-byte, that is the 5× and everything below is noise.
2. Run the three engines on **real shards, cold cache, O_DIRECT** — runbook
   §1a — using a different shard per run so the page cache flatters no arm.
3. Only then consider code. If `pread` wins reproducibly on model shards, the
   honest change is to make it the default *and say why*, not to add a fourth
   mechanism.

Step 2 cannot be done in this workspace: no model shards, and the attempt on
system libraries is what produced the inconclusive result above.

---

## ✅ Foundation already shipped (baseline the roadmap builds on)

These aren't roadmap line-items but represent the substantial completed groundwork:

- [x] **Real custom CUDA backend** — fused quantized matmuls, Tensor Core WMMA (W4A16/INT4), SwiGLU, attention+RoPE `cuda/backend_cuda.cu`
- [x] **io_uring with registered files** (`IOSQE_FIXED_FILE`) `ring.rs:55-105`
- [x] **Genuine 3-lane concurrency** — I/O ∥ CPU ∥ GPU within a single MoE layer, deterministic merge `concurrent.rs:267-521`
- [x] **Continuous batching** — chunked prefill interleaved with decode `batch.rs:82-189`
- [x] **Bit-identical fork-join thread pool** `peregrine-par/lib.rs:83-278`
- [x] **3-tier memory hierarchy** SSD → warm RAM cache → GPU VRAM `warmcache.rs`, `gpu.rs`, `concurrent.rs`
- [x] **Quantization** — per-row INT4/INT8, grouped INT4 w/ fine-grained scales `qt.rs`, `quant.rs`
- [x] **Per-lane wall-time telemetry** — `LaneTimings` accumulator inside `moe_forward_concurrent`, drained + fed to `BubbleTuner` between forwards `lane.rs`, `model.rs::publish_lane_timings`
- [x] **Zstd codec** — shared `peregrine_core::compress` module, threaded into both on-disk and warm-RAM paths `compress.rs`
- [x] **gigatoken BPE tokenizer fast path** — vendored stable-toolchain subset of marcelroed/gigatoken v0.10.0 (MIT) in `peregrine-token`; the sole serve tokenizer (HF `tokenizers` is a dev-only parity oracle); id-for-id parity-gated; 34× measured locally (`--bench-tokenizer`). Hot-path files verified byte-identical to upstream (per-file diff); facade adds upstream's bulk shapes — `encode_into` (whole-buffer, ~3× the per-line row) and `encode_batch` (persistent forked-worker pool, id-for-id serial-identical; 872 MB/s warm on a 2P+8E laptop vs 129 per-line) — bench reports line/whole/parN rows
- [x] **Documentation wiki** (2026-07-30) — full docs under `docs/` ([index](docs/README.md)): getting started, `peregrine` CLI + stdio protocol, HTTP serving/API, layout tools, complete env-knob reference, model format + artifact inventory, architecture / scheduler / adaptive-runtime / prefetch / I/O / GPU / tokenizer deep dives, testing & quality gates, roadmap summary; README gained a Documentation section

---

## ⭐ Top Priority Shortlist

Highest expected throughput per unit effort. **17 done, 0 partial, 3 to go** —
counted from the checkboxes below, not asserted: this line said "15 done" while
the dashboard said 16, which is the drift the recount command exists to catch.
The three not started are persistent CUDA kernels, GPUDirect Storage and
multi-GPU. CUDA Graphs into decode closed 2026-08-08.

- [x] ✅ **Layer look-ahead prefetch** — per-layer emission mid-forward, staggered ahead of the compute cursor (`PrefetchCtx::emit_layer`) _(★★★★★ · Medium)_
- [x] ✅ **Router look-ahead** — run layer L+1's *own* router on layer L's output and prefetch that ranking, filling the layer boundary the readers would otherwise spend idle. Output-neutral, decode-only, no artifact, works on a cold process. The one predictor here that asks the router instead of its history `model.rs::LookaheadCtx` _(★★★★★ · Medium)_
- [ ] ⬜ **Persistent CUDA kernels** — launch once, threadblocks loop `dequeue → compute → enqueue` _(★★★★★ · Hard, CUDA-only)_
- [x] ✅ **CUDA Graphs wired into decode (`COLI_CUDA_GRAPH`)** (2026-08-08) — `expert_group` now captures its launch sequence **per shape** — `(arm, count, D, I, per-expert rows)` — and replays it. That is the split CUDA Graphs exist for: in decode the shape repeats constantly (at B=1 every routed expert contributes one row) while the *contents* change every call, and the contents ride in through pinned staging buffers the graph copies from. Rewriting the pinned descriptor buffer between replays is what lets one graph serve a different residency generation at the same shape (`a_replayed_graph_picks_up_new_expert_weights`). **The hazard is stale device pointers and it is silent**: `reserve` is grow-only and frees before it reallocates, so a graph captured before a larger call points into VRAM the allocator has re-issued — plausible numbers from freed memory, no error. A `scratch_gen` counter bumped *inside* the reserve helpers (not at their call sites — `dc->y` **is** `ctx->y`, so an attention call can invalidate an expert-group graph) discards those; `a_grown_scratch_buffer_invalidates_cached_graphs` is the guard. The `COLI_CUDA_TC_W4A16` arm is excluded because it passes device weight pointers as *kernel arguments*, and `COLI_CUDA_PROFILE` because its event records are not part of the replayed work; both are counted as `graph_uncacheable` on `/metrics`, so "the knob is on and replaying nothing" is visible rather than merely slow. **Not the whole-layer graph** — that still needs device-resident attention/norm/router/embed and a device KV cache, and is stated as open in `docs/gpu-cuda.md` `backend_cuda.cu`, `peregrine-cuda/src/lib.rs` _(★★★★☆ · Medium, CUDA-only)_
- [x] ✅ **Dynamic expert VRAM cache** — `reheat()` re-selects hottest experts by routing frequency every 256 steps `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [x] ✅ **Triple-buffered pipeline** — read ∥ compute ∥ (H2D∥kernel∥D2H) overlap `concurrent.rs:344-509`, `backend_cuda.cu:653-739` _(★★★★☆ · Medium)_
- [x] ✅ **Adaptive expert cache — LFRU wired (`COLI_CACHE_LFRU`)** (2026-08-06) — `tier.rs`'s `(heat<<8)|recency` score was implemented and tested with **no caller**, while the live policy was priority-weighted LRU; this entry claimed the opposite ("not plain LRU") until 2026-08-02. `lfru_score` is now the second component of `evict_to_budget`'s victim key and `decay` halves frequency every 4096 hits. **Priority stays the primary key** — the protected set is a separate mechanism, so LFRU reorders within a class instead of overriding `COLI_PREFETCH_PROTECT` (`priority_still_dominates_under_lfru`). **The frequency source is the interesting decision**: `HeatTable` is the obvious one and the wrong one, because it is constructed only when a GPU tier is, so on any CPU-only run the policy would have degraded to the LRU it replaces while the knob still read as on. A `Slot` counts its own hits instead — the same quantity, over exactly the slots a victim choice ranges over. Off = the historical `(prio, used)` tuple, and the two tests run the *same* access sequence and require **opposite** victims, so the knob cannot quietly stop being a knob `warmcache.rs`, `tier.rs`
- [x] ✅ **Pinned memory + async copies** — `cudaMallocHost` staging + `cudaMemcpyAsync` `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [x] ✅ **Huge pages** — `MADV_HUGEPAGE` on every buffer ≥ 2 MB, single choke point `peregrine-io/src/mem.rs`; `COLI_HUGEPAGE` _(★★★☆☆ · Easy)_
- [x] ✅ **Lock-free work stealing** — atomic `io_work.fetch_add` across N io_uring rings `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [x] ✅ **Adaptive CPU/GPU work balancing** — `BubbleTuner` EWMA over `LaneTimings` publishes a `Bias`; `LaneBalancer::choose` downgrades cold GPU-resident experts to the CPU lane when GPU is the bottleneck `lane.rs`, `model.rs::build_balancer`; `COLI_LANE_BALANCE`
- [x] ✅ **Runtime expert replication for hot experts** — `Model::enqueue_expert_replicas` warms the top-K hottest GPU-resident experts into the CPU warm cache from `reheat`; `COLI_REPLICATE_K`
- [ ] ⬜ **GPUDirect Storage** — no direct SSD→VRAM path _(needs GDS driver stack)_
- [x] ✅ **Dynamic prefetch distance tuning** — `PrefetchTuner` EWMA over used/wasted adapts warm breadth
- [x] ✅ **Learned cache admission & eviction** — predictive protected-set eviction (predictor + heat → cache priority); pragmatic (heuristic, not a trained model)
- [x] ✅ **Hardware-counter-driven scheduler feedback (`COLI_PERF_PREFETCH_FEEDBACK`)** — the counter itself is real and works: `PerfCounter::open_cache_misses` (`perf_event_open`, hand-declared VER0 attr layout, thread-following) with `read()`/`reset()`, and `telemetry::open_l3_miss_counter` gates it on `COLI_PERF_COUNTERS=1`. **This entry said "nothing calls the opener" until 2026-08-06; that was stale** — `peregrine-engine`'s `serve` opens it on the decode thread and prints the total at shutdown (the §10 entry below was the accurate one, and two entries for one feature disagreeing is its own lesson). The **consumer** landed the same day: `Model::attach_perf_counter` takes ownership from the binary (the model cannot open its own — `perf_event_open` follows the calling thread, and a `Model` is built on whichever thread loaded the checkpoint), primes a baseline so the first delta is not the whole counter, and `llc_trend` steers the prefetch distance from the per-forward delta. **Two deliberate restraints, because §10 argues against this feature and the shortlist argues for it.** It is a *second* opt-in — `COLI_PERF_PREFETCH_FEEDBACK=1` on top of `COLI_PERF_COUNTERS=1` — so a measurement knob never silently becomes a control loop; and the **direction is documented as a hypothesis, not a result**: the counter follows the decode thread, so it sees attention, the router matmul and the reduce, and *not* the io_uring workers or the `peregrine-par` pool that actually stream and compute experts. A rising miss rate there most plausibly tracks a growing KV cache, which prefetch breadth cannot help. The control law is pure and unit-tested (`llc_trend`: seeding holds, ±10% dead band, rising widens) because `perf_event_open` is refused on most VMs — a test needing a live counter would pass by not running. If enabling it cannot be shown to improve `[prefetch] used/wasted` at constant disk reads, the honest end state is deleting the loop and keeping the report `perf.rs`, `telemetry.rs`, `model.rs`, `main.rs`
- [x] ✅ **Offline checkpoint re-layout from routing traces** — `peregrine-layout-reorg` binary consumes `dump-routes` JSON and emits `schedule.json`; loader picks it up and orders `EPlan`s by the schedule `crates/peregrine-tools/src/reorg.rs`, `model.rs::load_layout_schedule`
- [x] ✅ **Online kernel autotuning** — `WmmaTuner` records per-shape `(D, I, count, max_rows) → TileConfig` EWMAs and persists across sessions; picks the winning tile per shape `wmma_tune.rs` _(dispatch-side wiring in `backend_cuda.cu` is CUDA-only follow-up)_
- [x] ✅ **Pipeline bubble detection & rebalancing** — `BubbleTuner` hysteresis (α = 0.3, dominance 1.5, k = 3 consecutive); consumed by the LaneBalancer `lane.rs::BubbleTuner`
- [ ] ⬜ **Multi-GPU expert ownership & migration** — single device (hardcoded `device=0`); requires ≥ 2 GPUs

---

## 1. Prefetching & Speculation — 11/11 ✅

Shared spine in `predict.rs` (`RouteHistory` K-deep + `PredictSource` momentum/automaton/phase-aware +
`PrefetchTuner` + `TransitionTable`) feeding a per-layer emitter (`PrefetchCtx::emit_layer`) and a
parallel-async lane pool. All bit-identical (prefetch/eviction/prediction affect performance only)
and clippy-clean.

- [x] ✅ Layer look-ahead prefetch — per-layer emission mid-forward (`PrefetchCtx::emit_layer` from the `forward_hidden` loop), staggered ahead of the compute cursor instead of one bulk dump; `COLI_PREFETCH_LOOKAHEAD` _(★★★★★ · Medium)_
- [x] ✅ Speculative expert prefetch — next token's experts warmed on a background ring `model.rs`, `concurrent.rs`
- [x] ✅ Expert "momentum" prediction — recency-weighted vote over K-deep `RouteHistory` (`COLI_ROUTE_HIST_DEPTH`, default 4); depth-1 == legacy `predict.rs`
- [x] ✅ Global Expert Transition Automaton — offline FSA: `build-automaton`/`dump-routes` CLI → config-tagged `automaton.json`, auto-loaded at construction, blended with momentum `predict.rs::TransitionTable`
- [x] ✅ Speculative multi-path execution — top-N ranked candidates split into warm (tier 1) + fadvise-hint (tier 2) tiers; `COLI_PREFETCH_WARM_PATHS`/`_HINT_PATHS`
- [x] ✅ Asynchronous page-cache warming — `PrefetchMsg::Hint` wires `fadvise_willneed` for low-confidence tier (gated `!direct`)
- [x] ✅ Dynamic prefetch distance tuning — `PrefetchTuner` EWMA over prefetch used/wasted → adapts warm breadth; `COLI_PREFETCH_TUNE`/`_DIST`/`_DIST_MAX`
- [x] ✅ Predictive cache eviction — resident predicted-∪-hot experts protected via an opaque cache priority (`WarmCache` `(prio, recency)` victim order); all-equal == pure LRU; `COLI_PREFETCH_PROTECT`
- [x] ✅ Background verification of speculative expert loads — opt-in `COLI_PREFETCH_VERIFY` re-reads + byte-compares each load (`verify_mismatch` counter, never panics); shutdown accuracy log (`[prefetch] used/wasted/accuracy/fadvise/verify`)

**Bonus (beyond §1):** per-sequence prefetch in the **batched serving engine** with a parallel-async prefetch-lane
pool (`COLI_PREFETCH_LANES`) — each concurrent stream predicts + prefetches from its own routing history
(per-row `route_log_multi`, `batch.rs` field-split unzip). Plus
**`PredictSource::PhaseAware`** — wraps any inner source and boosts newest-frame vote when Jaccard
distance between the top two frames exceeds a basis-points threshold.

**Audited 2026-08-08, and the audit found four things this entry had wrong.** All of the above ships and
is reachable — `batch.rs:854` calls `enqueue_seq_prefetch` in the live decode loop — which is exactly why
the entry read as settled and was not. (a) **`PhaseAware` was inert.** Production built it with `boost: 2`
against a newest-frame momentum weight of `depth`; at the default depth 4 a newest-frame expert scored 4+2
and one that had just dropped out scored 3+2+1 — a *tie*, broken by ascending expert id. Both unit tests
passed throughout because they build their own source with `boost: 100`, so they proved the mechanism and
said nothing about the constant. Now derived: `predict::phase_boost(depth)` returns the full momentum scale
and dominates by construction, pinned by `phase_boost_outranks_every_stale_expert`, which fails at 2.
(b) **`COLI_PHASE_THRESHOLD` governed nothing** — its only reader was `PhaseTracker`, which has no
production caller, while the predictor used a hardcoded `6000` bp. Now converted at the boundary and read
by the predictor (`phase_threshold_bp`). (c) **The lane key was the sequence's index in `active`**, which
`retain` compacts every tick, so a live stream migrated lanes whenever an earlier one retired. Now a
monotonic id assigned at admission. (d) The entry named `forward_step_batched`; the serving engine calls
`forward_rows_batched_hidden` (`batch.rs:818`) and `forward_step_batched` is a 10-line wrapper used by the
bench and tests. **`COLI_PREFETCH_LANES` still defaults to 1**, so the "parallel-async" pool is a single
lane unless raised `predict.rs`, `model.rs`, `workload.rs`, `batch.rs`

**Trying to measure the lane default found two more, and neither was in the prefetch code.** The sweep
itself did not run — see [`bench-data/2026-08-08-serve-prefetch`](bench-data/2026-08-08-serve-prefetch/README.md)
for the budget that killed it (~50 s/token single-stream on the 358 GB container; 8 concurrent streams
took >50 min for 4 tokens at load average 27, so a 4-arm × 3-repeat sweep is a multi-day run on a machine
someone is using). Setting it up is what exposed the rest. (e) **`peregrine-serve` never applied
`COLI_PREDICT_SOURCE`** — `apply_predictor_override` was called only by the stdio binary, so `PhaseAware`
was unreachable from the batched engine, i.e. from the only path that has per-sequence prefetch and
therefore the only place it was meant to matter. (f) **The server never joined its engine thread.** The
join handle was bound as `_engine_join` and dropped, so the process exited while the thread still owned
the `Model` and `Model::drop` never ran — which silently cost **`route_stats.json` persistence on the HTTP
path entirely** (`COLI_ROUTE_STATS_PERSIST` worked only in the stdio binary) on top of the shutdown
counters. Now joined with a 30 s bound, because the engine exits when both request senders drop and those
live in an `Arc` a detached SSE task can still hold; a server that will not exit is a worse trade than a
lost counter. The `[ecache]`/`[prefetch]` lines are now reported from the engine thread, which is the only
thing that owns the model — **the first numbers ever seen from the serving path**:
`hits=14 misses=3510 disk_reads=3510 prefetch_reads=433 hit_rate=0.4%` and
`used=14 wasted=50 accuracy=21.9%` on one 2-token request. Too small to conclude from, and consistent with
§5.2's plateau finding at this capacity ratio `peregrine-serve/src/main.rs`, `batch.rs`

- [x] ✅ **Router look-ahead** (2026-08-03) — every predictor above is a statistic over the router's *past answers*; this one asks the router. At the end of layer `L`, apply layer `L+1`'s own `post_ln` + router to layer `L`'s output and prefetch that ranking's top `COLI_ROUTER_LOOKAHEAD_N` (default 6) — one extra `E×D` matvec against resident weights, no artifact, no format change, works on the first token of a cold process where every history-based predictor is still empty. Correctness-neutral: the authoritative router still runs at `L+1` and still decides, pinned by `router_lookahead_cannot_move_a_token` (streamed decode bit-identical to resident). **Decode only** — WASTE built the prefill-chunk version and measured bytes read +6.9 % with flat wall clock, because a chunk layer's speculative records are exactly what eviction takes first and its readers are never idle. Speculative reads stay out of `misses`. `model.rs::LookaheadCtx`, `router.rs::route_ranks`; `COLI_ROUTER_LOOKAHEAD` _(★★★★★ · Medium)_
- [x] ✅ **Predictor scoreboard** (2026-08-03) — `COLI_PREDICT_EVAL=1` scores the router look-ahead, the configured `PredictSource` and a previous-token baseline against the routing that actually happened, and prints recall + precision-by-rank at shutdown (`[predict-eval]`). Built because the §1 spine is entirely correctness-neutral, which means **no test can catch a predictor that has degraded to noise** — it costs throughput silently. Pure and unit-tested (`predeval.rs`); an arm that abstains is counted as silent rather than wrong, and recall counts distinct coverage so a degenerate arm cannot report >1. **This is the open question against every other item in this section**: WASTE measured held-out co-occurrence at 29.0 % recall@16 against "reuse the previous token's set" at 29.5 %, i.e. no better than the baseline the cache already exploits for free. Whether `automaton.json` / `macrostates.json` beat it *here* is now measurable and unmeasured `predeval.rs`, `model.rs::score_and_stash`

## 2. GPU Execution — 7/9

- [ ] ⬜ Persistent CUDA kernels _(★★★★★ · Hard, CUDA-only)_ — kernels launched per-batch, no threadblock loop
- [x] ✅ CUDA Graphs into decode (`COLI_CUDA_GRAPH`) — per-shape capture/replay of the `expert_group` launch sequence; see the shortlist entry for the `scratch_gen` invalidation guard and the two excluded arms. The whole-`forward_layer` graph remains open and is tracked under §2's device-resident work, not here `backend_cuda.cu`, `peregrine-cuda/src/lib.rs` _(★★★★☆ · Medium, CUDA-only)_
- [x] ✅ **Fused MoE pipeline — the layer-level reduce moved onto the device (`COLI_CUDA_FUSED_REDUCE`)** (2026-08-08) — `expert_group` fused gate/up/silu/down and then returned `Σrows × D` floats for the host to accumulate. `coli_cuda_expert_group_reduce` takes a CSR (`row_ptr`/`row_idx`/`rw`) and folds them on the GPU, so the D2H carries `s_n × D` instead: ~5× fewer rows at B=16 on the measured GLM-5.2 unions, and **exactly 1× at B=1** — which is why a B=1 measurement of this cannot distinguish "it worked" from "the regime had no win", and why the runbook specifies both. **CSR, not atomics, and that is the whole design**: `f32 +=` is not associative, so an atomic scatter would return a different vector each run on identical input while every tolerance test kept passing. Each `(row, dim)` is written by exactly one thread summing in ascending y-row order — which is batch-union (`pos`) order, because the host fills `x` by walking `gplans` in `pos` order — and `fused_reduce_is_bit_stable_across_repeats` requires identical **bits** across three runs. It does move the GPU arm's low bits against the host reduce (GPU experts sum among themselves before meeting the CPU lane), so it is opt-in; the host adds the device partial in a fixed position, after all CPU contributions, so lane arrival order never reaches the arithmetic. `ReduceLayout::build` is pure and tested without a GPU, because the ordering *is* the correctness `backend_cuda.cu`, `peregrine-cuda/src/lib.rs`, `gpu.rs::compute_reduced`, `concurrent.rs`
- [x] ✅ Zero-copy GPU uploads via pinned memory `backend_cuda.cu:356-359,610-611` _(★★★☆☆ · Easy)_
- [x] ✅ Persistent GPU memory pools — 24 pre-allocated scratch slots reused across layers `backend_cuda.cu:892-899`
- [x] ✅ Format-split expert dispatch — `all_s4` is computed over the whole routed group (`backend_cuda.cu:638,645`), so **one** wider-precision resident dropped every expert in that `expert_group` call off the int4 Tensor-Core and packed-W4 fast paths. `GpuTier::compute` now issues one `expert_group` per residency format via the pure `partition_by_format` / `scatter_by_index` pair, restoring job order before returning so `concurrent.rs`'s positional zip and the `pos`-keyed reduce are untouched. Repeated calls per layer are safe: the kernel syncs before returning (`backend_cuda.cu:745-750`) and scratch is per-device grow-only (`:341-359`). Partition/scatter pure and unit-tested (round-trip is a bijection, malformed input errors rather than misaligning); the dispatch loop is type-checked under `--features cuda`. **Does not always reach Tensor Cores** — `:674` additionally gates TC on *every* job in the call clearing `tc_min` rows, a separate partition axis `gpu.rs` _(★★★☆☆ · Medium, CUDA-only)_
- [ ] ⬜ GPU memory defragmentation during decode — residents fixed at startup; `cudaMallocAsync` pool is the planned fix (CUDA-only)
- [x] ✅ Online kernel autotuning for GEMM tile sizes — `WmmaTuner` records per-shape kernel_ms and picks the best tile config, persists as `kernel_tuning.json`; the CUDA-side dispatch selector is a follow-up `wmma_tune.rs`
- [x] ✅ Runtime SIMD kernel selection (CPU) — AVX2 vs AVX-VNNI chosen at runtime `idot.rs:40-61`
- [x] ✅ **Stream-ordering defect in the `pipe_*` primitives — found while preparing graph capture, and it was not only a capture problem** (2026-08-07). `ctx->stream` is created `cudaStreamNonBlocking`, so it does **not** implicitly synchronize with the legacy default stream. `pipe_rmsnorm`, `pipe_rmsnorm_s`, `pipe_rope`, `pipe_rope_base`, `pipe_rows_add`, `pipe_gemm` and `pipe_copy2d` all launched with no stream argument — i.e. on the default stream — while `pipe_silu_mul` and `pipe_add` used `ctx->stream`. **Any chain mixing them had no ordering guarantee at all**: a `silu_mul` could read a buffer a `rmsnorm` had not finished writing. Nothing was wrong in practice only because no live path builds such a chain; it would have surfaced the moment the device-resident forward wired one, as intermittently wrong logits with no failing test. Two further instances in the **live** GPU path: `tensor_upload` and `tensor_update` convert int4 offset-encoded weights with a default-stream kernel whose consumers (`expert_group`) run on `ctx->stream` — a `reheat` refresh has a far shorter window than a startup upload — now synchronized before return, which is the contract every caller already assumed. Pinned by `graph_capture_records_ops_only_from_the_context_stream`, which is deterministic rather than race-dependent: capture records `ctx->stream`, so an op on another stream is silently **absent from the graph**, and the test replays onto a poisoned buffer so a skipped normalization cannot pass on leftovers from the eager pass `cuda/backend_cuda.cu`, `peregrine-cuda/src/lib.rs`

## 3. Caching & VRAM Residency — 12/12 ✅

- [x] ✅ Adaptive expert cache — LFRU wired behind `COLI_CACHE_LFRU`; see the shortlist entry `tier.rs:18-56`, `warmcache.rs::evict_to_budget` _(★★★★☆ · Medium)_
- [x] ✅ Quantized RAM cache — warm cache holds quantized bytes verbatim; hits return a byte-identical `ExpertSlab`; **transparent zstd compression** on admit under `COLI_CACHE_COMPRESS` shrinks the resident footprint **~1.2x** (measured — the payload is packed int4 nibbles whose only structure is nibble-value skew, so the entropy stage is the whole win and an LZ-only codec would score ~1.0x) at the cost of one decode per hit; `WarmCache::compression_ratio` reports the achieved figure at shutdown `warmcache.rs`
- [x] ✅ Dynamic GPU residency — `reheat()` heat-ranked VRAM re-selection `gpu.rs:363-382` _(★★★★☆ · Hard)_
- [x] ✅ Heat-wave scheduling — `PhaseTracker` (Jaccard EWMA) + `PredictSource::PhaseAware` blend a boost vote onto the newest frame during a shift `workload.rs`, `predict.rs`
- [x] ✅ "Negative" caching — `COLI_CACHE_NEGATIVE_TTL` evicts unhit slots ahead of pure-LRU order (unprotected slots only, guarded by keep-at-least-one) `warmcache.rs::evict_to_budget`
- [x] ✅ Persistent Expert Residency Solver — `gpu.rs::solve_residency_sized` — heat / bytes-per-expert knapsack; deterministic ties; falls back to round-robin on a cold heat table. **Now actually wired**: `build_with` called round-robin `plan_residency` and the solver had zero production callers, so the item shipped ✅ while the feature was dead code. Initial placement is additionally seeded from the previous session's `route_stats.json` via `peek_persisted_heat` (the heat table is otherwise restored *after* the tier is built, so a warm start had no heat to rank by) `gpu.rs`, `model.rs`
- [x] ✅ Cache admission from estimated future reuse — heat-threshold gate: a streamed expert is admitted only once its routing heat reaches `COLI_CACHE_ADMIT_MIN_HEAT` (default 0 = admit all; 1 = cache from the second routing on, filtering one-off experts) `concurrent.rs::cache_admit_min_heat`, `HeatTable::get`
- [x] ✅ Learned cache admission & eviction — predictive protected-set eviction (predictor + heat → opaque cache priority, `WarmCache` `(prio,recency)` victim order); see §1 predictive eviction
- [x] ✅ Bloom filter / probabilistic cache lookup — 2048-bit Bloom (two hashes) short-circuits the miss-path in `WarmCache::get`; rebuilt on eviction so the hint stays tight `warmcache.rs::Bloom`
- [x] ✅ Self-consistent mixed-precision residency sizing — `plan_precision_fitted` sizes the resident set and its f32 share **together** (`R = budget / (frac·hi + (1-frac)·lo)`) over every sparse candidate. Previously `reheat` ranked the hottest `capacity` experts (a count derived from the *int4* footprint), promoted `frac` of them to f32 (~8x each), then dropped the overrun — so the promotions exhausted VRAM before the int4 tail was reached: **~67 residents instead of ~204** at `frac=0.25` on a 10 GB budget (GLM-5.2 shape). A promoted expert that no longer fits now falls back to int4 instead of being evicted — the hottest expert was the one being dropped. Pure, unit-tested `gpu.rs`
- [x] ✅ Mixed-format VRAM residency — `upload_expert` treats int4 as a *preference*: a grouped-int4 or int8 source falls back to the f32 path for that expert alone. It previously returned `Err`, which the caller reads as "stop and keep what landed", so one odd expert truncated the tier and cost every expert behind it. Per-expert formats are recorded at upload so byte accounting and re-upload-on-change stay correct `gpu.rs`
- [x] ✅ Runtime expert replication for hot experts — `Model::enqueue_expert_replicas` reads the top-`COLI_REPLICATE_K` hottest GPU-resident experts from `HeatTable` and enqueues prefetches so their bytes land in the warm cache too; a bias-driven downgrade then pays no disk `model.rs`

## 3b. Unreachable subsystems (found 2026-08-02)

*Four features that ship, pass their own tests, are documented as live, and no
production path reaches. Catalogued in [BAD_PATTERNS](docs/BAD_PATTERNS.md#the-2026-08-02-triage).
Each needs a wire-or-delete decision; none is a correctness risk today, because
code that never runs cannot be wrong — it can only mislead.*

- [x] ✅ **LFRU eviction wired** (`tier.rs`) — `lfru_score` + `decay` now drive `warmcache.rs::evict_to_budget` under `COLI_CACHE_LFRU`; see §3's entry. **`pick_lfru`/`pick_swap` remain unreachable on purpose**: they are the *fixed-slot swap* form of the policy — a pinned set of constant size, which is `GpuTier::reheat`'s shape, not a byte-budgeted cache's. Wiring them here would have meant reshaping a byte-budget evictor into a slot-swapper to satisfy a reachability count. They are a `reheat` decision and are tracked there
- [x] ✅ **DSA sparse attention wired (`COLI_DSA`)** — `dsa.rs` shipped complete and unit-tested with **zero call sites**: `Indexer` was never constructed and no key was ever appended, so the largest single workload reduction available on long context (`index_topk=2048` against a growing cache) sat inert. **The blocker was structural, not effort**: `Indexer` conflated per-*layer* weights (`wq`/`wk`/`wp`/`k_norm_*`) with per-*sequence* cache state (`keys`/`len`), so one indexer per layer on a `Model` serving concurrent requests would have let two sequences interleave keys into one buffer — silent cross-sequence corruption, not a crash. Split into `IndexerWeights` (per layer, on `LayerW`) and a key cache that lives **inside `LayerKv` as a third stream** beside `lc`/`rc`, because it has exactly the latents' lifecycle: appended in order, rewound by the same speculative `truncate`, shared across a common prefix by the same refcount. A parallel structure would have had to re-implement all three, and `indexer_keys_ride_the_kv_cache_through_sharing_and_rewind` pins that it does not have to. **Keys are cached every step, scoring is conditional**: a selection at position *t* needs keys for every earlier position, so only the scoring is gated on `len > index_topk` — and below that threshold the skip is *exactly* output-neutral, since attention over at most `index_topk` keys already is the selection (`dsa_is_inert_without_an_indexer_and_below_index_topk`, bit-exact on both counts). **`project_batched` now returns `qr`**, the post-`rmsnorm` q-LoRA row it used to drop: letting the indexer re-derive its own would have been a silent pre- vs post-norm fork that no output test could catch. **Single-sequence path only** — selection is implemented against the dense core's `sel`, and the batched engine runs the *absorb* core, which has no sparse form; documented rather than left looking wired. Determinism needed nothing: `select_topk` already sorts value-descending with an index tie-break, so the GLM-5 non-deterministic-top-k hazard does not apply. Off by default; **changes token values** `dsa.rs`, `attention.rs`, `model.rs`, `testkit.rs`
- [x] ✅ **`peregrine-sched` is now the correctness oracle** (2026-08-06) — no crate depends on it and production MoE is `concurrent.rs::moe_forward_concurrent`, so a second implementation of the same computation was a second thing to keep correct with nothing checking it. `streamed_matches_the_production_concurrent_path` closes that: it builds a `testkit` container, runs the **production** path over it through a real `ForwardCtx`, and points this crate's `DiskQt`s at **the same container bytes** via `SafeTensors::region` — the same triples `concurrent.rs`'s private `tplan` builds from — so the two engines read one file rather than two. Both entry points take the router weights as arguments, so routing is identical by construction and only the expert path is under test. Tolerance, not bits: the lanes accumulate in different orders and `f32 +=` is not associative; bit-identity is asserted *within* each engine, never across them. Guarded against passing vacuously by requiring the reference output to be non-zero, and verified to bite (a 1.5× `routed_scale` on one side fails it). The crate's own `concurrent_matches_sequential` — which compares against the *resident* reference, not the concurrent path, and whose name said otherwise (`docs/testing-and-quality.md` flagged it) — is renamed `streamed_matches_the_resident_reference` `peregrine-sched/src/lib.rs`
- [x] ✅ **MTP speculative decode — wired in both binaries** — `generate_speculative` was unreachable from either. The stdio server takes `--draft N` / `COLI_DRAFT`, and the batched HTTP engine speculates per sequence with all sequences' `1 + γ` rows in *one* forward, so B sequences share one routed-expert union — B separate speculative decodes would stream B unions off the disk that is already the bottleneck. Greedy requests only (argmax acceptance is sequence-identical; temperature > 0 is merely distribution-preserving), drafts capped by the remaining `max_new`, and speculated rows recorded into a scratch history so a rejected draft never warms experts for a token that never existed. **colibrì's "net loss" figure was taken at depth 2**, where 2.46 accepted is already 82% of that configuration's ceiling of 3 — which is why the default guidance is 4–6, and why the reason to wire it was never completeness

- [ ] 🟡 **`PhaseTracker` (`workload.rs`) — a fifth, found 2026-08-08, and deliberately left unwired.** Nothing in the engine constructs one; the live phase signal is `PredictSource::PhaseAware`, which compares the newest two frames per prediction and holds no state. The cost was not zero while it sat there: `COLI_PHASE_THRESHOLD` was documented with a default of 0.6 and **`PhaseTracker` was its only reader**, so the knob the docs offered governed nothing, and the predictor that *should* have obeyed it used a hardcoded `6000` bp. That half is fixed — the threshold is now converted at the boundary and read by the predictor — so what remains unreachable is the struct, not the setting. **Kept rather than deleted because it is not a duplicate**: it holds an EWMA of frame-to-frame distance plus `since_change`, i.e. a *window* after a shift, and `PhaseAware` can express no such thing — it re-decides instantaneously at every layer. Whether a window beats the instantaneous check is precisely what `COLI_PREDICT_EVAL` exists to answer, and until it does, wiring this would be a guess with a control loop attached — the failure mode §3c's own note warns about, facing the other way. The reason is recorded at the definition `workload.rs`

## 3c. Dead-code sweep (2026-08-07, extended 2026-08-08)

*User mandate: "don't allow any dead code — wire it", and "don't delete, use
everything". `[R]` went **40 → 29** (2026-08-07), then **29 → 22** (2026-08-08).*

*The 2026-08-08 five, each with the reason it had no caller — which in every
case was a missing decision, not a missing line:*
*`moe_engine_installed` needed somewhere the **effective** dispatch differs from
the requested one (a failed ring), so it reports after the install attempt while
a new `MoeEngine::name` feeds `/metrics`.*
*`pick_lfru`/`pick_swap` are fixed-slot hot-store rules and `GpuTier` keys
residency by `(layer, expert)` — but **per layer the resident set is exactly a
slot array**, which is the shape they were written for, so they became the
incremental `COLI_GPU_TIER_SWAP` policy. LFRU also needed a recency clock, which
went into `HeatTable` beside the heat rather than into a second table with a
second bump site that could drift.*
*`speculative_sample` needed the temperature > 0 accept path plus, crucially, a
draft **drawn from a distribution it can hand to the verifier** —
`mtp_draft_sampled` draws and describes in one call because the guarantee is
void if `q` is not what the draft came from.*
*`WmmaTuner`'s three needed a `.cu` signature change: the W4A16 kernels are now
templated on the WMMA fragment shape and the backend **reports which arm ran**,
so a group that missed the arm's row floor is not credited to the selected tile.*

*Wired 2026-08-07: the startup banner
(`cpu_kernel_tier`, `is_available`, `status`, `pcie_link_by_bdf`,
`gpu_pcie_links`), `set_protected`, `routing_entropy_ewma`, `ewma_snapshot` +
`last_lane_timings` (via a real `GET /metrics`), `moe_streamed` (via
`COLI_MOE_ENGINE`), `set_force_async` (`COLI_FORCE_ASYNC`), `set_predictor`
(`COLI_PREDICT_SOURCE`), `ecache_disk_reads_for_layer`, `device_count`.*

**Wiring found three live defects that tests could not.** `is_available()` was
`device_count() > 0` — contexts initialized, not devices present — so it
reported "unavailable" on a working RTX 3060 and could never gate the `init` it
was meant to gate (`probe_device_count` added). The `pipe_*` primitives split
across two streams with no ordering between them. And `tensor_upload`'s int4
conversion raced its own consumers. None was reachable, so none was wrong in
practice — which is exactly why none was found.

**A fourth, found 2026-08-08 by auditing the fix rather than the feature — and this one is
demonstrated, not reasoned.** The 2026-08-07 pass moved every `pipe_*` *compute* primitive onto
`ctx->stream` and stopped there. The two **staging** primitives, `pipe_upload` and `pipe_download`, were
still blocking `cudaMemcpy` on the legacy default stream, which does not synchronize with a
`cudaStreamNonBlocking` stream — so a download issued after queued kernels could return bytes those
kernels had not written yet. It was missed for the same reason the original set was: the only callers are
the graph-capture tests, and they always sync via `graph.launch()` first, so the primitives the pass *did*
fix were exactly the ones the tests chained. On this box's RTX 3060 the new test
`a_download_observes_work_already_queued_on_the_context_stream` reads **164 where 200 is correct** against
the old code — 36 of 200 queued kernels still pending — and passes once both staging ops are issued on
`ctx->stream` and drained before return. `pipe_peer_copy` got the same treatment, plus a source-side drain,
because cross-device the producer and consumer sit on two different non-blocking streams.
**Also corrected here:** this entry says `is_available` "could never gate the `init` it was meant to
gate". It still does not — `is_available` is banner-only (`peregrine-model/src/lib.rs:104`) and
`GpuTier::build_with` probes with `init(&[0]) < 1` directly (`gpu.rs:1444`). That is fine behavior, but
the wiring the sentence implies did not happen `cuda/backend_cuda.cu`, `peregrine-cuda/src/lib.rs`

**Three entries are deliberately not wired, and say so at their definitions.**
`plan_precision` (wiring it reinstates the 542→67 residency bug
`plan_precision_fitted` exists to fix; it cannot delegate either, because the
two use different equal-heat tie-breaks), `bf16_roundtrip` (nothing in the
engine *writes* bf16, so a round-trip helper has no honest caller), and
`solve_residency_greedy` (a uniform cost makes it top-N-by-heat, which
`rank_by_heat` already does *while preserving `plan_residency`'s round-robin
tie-break* — this one breaks ties by ascending layer/expert, and on a partially
warm table the difference is pure residency churn). Inventing call sites for
these would make the metric agree while making the code worse, which is the
`[R]` failure mode facing the other way. Of the 29 remaining, 18 are the
test-only env-override builders that exist so parallel tests do not race on
process-global state, and one (`teacher_forcing`) is a scanner false positive —
`crates/*/tests/` is outside its glob. Of the 22 that remain after 2026-08-08,
the same three are still deliberately unwired for the reasons above.

## 4. I/O & Storage — 9/11

- [x] ✅ **Registered io_uring buffers wired (`COLI_REGBUF` / `COLI_IO_ENGINE=regbuf`)** — `register_read_buffers()` + `IORING_OP_READ_FIXED` were implemented and tested with **nothing calling them outside tests**, while the env var was documented *and set in a published benchmark arm*. Wiring the existing `read_fixed` would have been worse than leaving it inert: it looped `submit_and_wait(1)` per region, the same depth-1 defect that made the O_DIRECT lane slower than buffered, so `read_fixed_many` had to land first. **Two findings argue against making it a default and are documented rather than buried**: registered buffers are pinned pages charged against `RLIMIT_MEMLOCK` (8 MB typical, against the 96 MB a 16-slot pool of ~6 MB expert regions needs, so registration returns `ENOMEM` and the engine falls back with an advisory naming the limit), and the fixed path *copies out* where `read_many` has the kernel write the destination directly — at 6 MB regions that memcpy plausibly exceeds the pinning it saves, since the published gains are at 4–64 KB `ring.rs`, `concurrent.rs`
- [x] ✅ Batch I/O intelligently — `read_many()` / `read_experts_batched()` merge contiguous regions `ring.rs:251-288`, `concurrent.rs:225-254`
- [x] ✅ **Batch the O_DIRECT lane too** — the batching above never reached it. `read_direct_aligned` (the call `read_regions` makes under `COLI_DIRECT=1`) looped one region at a time, each its own `read_many` on a single-element slice, so a 96-region expert batch was 96 sequential ring round-trips at **queue depth 1** regardless of `COLI_IO_BATCH`/`COLI_IO_RINGS`. Direct reads consequently measured *slower* than buffered ones — the symptom that `COLI_DIRECT`'s "direct I/O regresses" default-off note recorded without diagnosing. Now: allocate every landing buffer up front, one `read_many` for the whole batch (it chunks internally to the ring's entries), per-region short-read completion. Byte-identical output, peak memory unchanged (each region already owned its buffer for the returned `Bytes`). Measured **1.2–1.3×** across four configurations on a LUKS NVMe, ±0.01 GB/s run-to-run. Gated by `direct_aligned_batch_exceeding_ring_capacity_matches_serial`, which submits 40 regions through an 8-entry ring and compares every one against a serial read `ring.rs`
- [x] ✅ Double / triple buffering — 3-lane concurrent overlap `concurrent.rs:344-509` _(★★★★☆ · Medium)_
- [x] ✅ Compressed expert storage (Zstd) — `pack::Blob::with_compression(Compression::Zstd)`; the safetensors header carries `"compression": "zstd"` + `"uncompressed_nbytes"`; `SafeTensors::read_raw`/`read_f32` decompress transparently `compress.rs`, `pack.rs`, `safetensors.rs`
- [x] ✅ Transparent expert compression in RAM — `SlotBytes::Compressed { six, orig_lens }`; per-region zstd on admit + decode-on-hit; `uncompressed_bytes_seen`/`compressed_bytes_seen` counters; `COLI_CACHE_COMPRESS` `warmcache.rs`
- [x] ✅ Background expert recompression when idle — `WarmCache::recompress_one_cold` converts the coldest raw slot to zstd; the batch engine sweeps while no requests are pending, interruptible per slot; `COLI_CACHE_COMPRESS_IDLE` `warmcache.rs`, `batch.rs`, `Model::idle_maintenance`
- [ ] ⬜ GPUDirect Storage (GDS) support _(needs vendor stack)_
- [x] ✅ Learned SSD read scheduler — pragmatic: batched read + main-path `fadvise_willneed_many` before submit, `COLI_FADVISE_MAIN` `ring.rs::fadvise_willneed_many`, `concurrent.rs::read_experts_batched`
- [x] ✅ Disk queue-depth autotuning — `IoTuner` EWMA + per-forward SQ-full deltas (counted at every ring push via `Reactor::push_counted`, drained by `take_sq_full`) drive grow/halve of `set_iowq_max_workers` on every reactor `iotune.rs`, `ring.rs`, `model.rs::publish_lane_timings`
- [x] ✅ Adaptive `io_uring` SQ/CQ sizing — `IoTuner::step` grows/halves the `(bounded, unbounded)` cap; `COLI_IO_TUNE`; last applied cap exposed on `Model::last_iowq()` `iotune.rs`
- [x] ✅ Fault-tolerant I/O recovery + degraded-mode execution — on a batched-read failure the buffered path re-issues each region via `Reactor::read_exact_retry` (linear backoff, transient EIO/EAGAIN/EINTR); `COLI_IO_RECOVERY` `concurrent.rs::read_regions_with_retry`, `ring.rs`

## 5. Memory & NUMA — 7/8

- [x] ✅ Huge pages (2 MB / 1 GB) — `advise_hugepages` (`MADV_HUGEPAGE`) applied at every ≥ 2 MB allocation choke point: `AlignedBuf::with_capacity`, `Reactor::register_read_buffers`, the safetensors `read_*` landing buffers `peregrine-io/src/mem.rs`, `safetensors.rs::maybe_hugepage`; `COLI_HUGEPAGE` (default on)
- [x] ✅ Automatic huge-page allocation and promotion — implicit via the `≥ 2 MB` threshold above; `MAP_HUGETLB` explicit-hugetlb variant is planned as a future opt-in
- [x] ✅ NUMA-aware scheduling — worker threads pinned round-robin across node-grouped CPUs: the `peregrine-par` pool (via a std-only worker-startup hook, `set_worker_start_hook`) and the prefetch pool both pin at spawn; opt-in `COLI_NUMA_PIN=1` `model.rs::numa_pin_worker`, `peregrine-par/lib.rs`
- [x] ✅ NUMA-aware RAM allocation and thread placement — `bind_local_if_enabled` (`sched_getcpu` → node → `mbind`) binds every ≥ 2 MB `AlignedBuf` to the allocating thread's node **before first touch**; thread placement via the pin hook `mem.rs::current_numa_node`, `slab.rs`
- [x] ✅ **int3-g64 AVX2** — the format shipped scalar-only, so an int3 checkpoint computed slower than the int4 it replaced, and it was the one kernel in `idot.rs` without the `_scalar` + dispatched-SIMD pair the convention requires. The low plane is byte-for-byte the int2 layout so its unpack is shared; the high plane's 1-bit-per-value is broadcast eight lanes per source byte and tested against a bit mask. Bit-exact against the reference (`int3_g64_avx2_matches_scalar`, which packs its own fixtures so the two kernels are compared to each other rather than to a shared producer bug) `idot.rs`
- [x] ✅ **Slab recycling by generation — fixed, then wired** (2026-08-06) — the pool was live (`checkout`/`checkin`) while the **generation-tagged** variants had zero callers including tests, and `docs/io-and-storage.md` described the protection as active. Found by the [R] reachability pass. **Wiring it as written would have been worse than leaving it inert**: `checkin_tagged` asserted `handle.gen < self.gen`, which every handle the pool ever issued satisfies — `gen` is monotonic — so the double-return it advertised was undetectable, and wiring would only have made an unfireable check *reachable*, retiring the audit finding without buying the safety. The pool now tracks the generations **outstanding**; a handle absent from that set (returned twice, or from another pool) is rejected and its buffer dropped rather than pushed onto the free-list, where it would let two holders write one allocation. The check also moved from `debug_assert!` to a release-path advisory — a straggler completion is a production phenomenon, so a check compiled out of release guards the one build that cannot hit it. `ring.rs`'s O_DIRECT landing path is the sole production call site and now takes tagged handles; `a_handle_returned_twice_is_rejected` and `a_handle_from_another_pool_is_rejected` both assert via the aliasing they prevent (a pool of one must never hand out two) rather than by inspecting the counter `slab.rs`, `ring.rs`
- [x] ✅ **Wire the resident trunk** (2026-08-03) — `COLI_MLOCK=1` runs `mlockall(MCL_CURRENT)` after the weights load and *before* the warm cache fills, so the trunk is pinned and every later allocation stays ordinary reclaimable memory. `MCL_CURRENT` not `MCL_FUTURE` is the design, not an oversight: the cache is supposed to be the part the kernel can take back, and wiring it too converts a gentle slowdown into an allocation failure. **This does not raise the ceiling, it removes variance** — a streaming engine with a large trunk and a cache sized to the remainder is exactly the shape that invites the kernel to reclaim trunk pages to grow the page cache, and each one is re-read at disk speed mid-token. Refusal (the normal outcome on a desktop `RLIMIT_MEMLOCK`) is reported with the limit and the fix, never fatal `peregrine-io/src/mem.rs::wire_resident`
- [x] ✅ **cgroup-aware memory budgeting** (2026-08-03) — `/proc/meminfo` is not namespaced, so inside a container `MemAvailable` is the *host's*: a small container on a large machine sized its caches for hardware it was not running on and got OOM-killed while every projection reported room to spare. `mem_available_bytes` is now `min(MemAvailable, cgroup v2 memory.max − memory.current, cgroup v1 limit − usage)`, with unlimited sentinels recognized by magnitude. Parsing is pure and unit-tested; both reference projects hit this independently `ram.rs::effective_available`

*Note: weight loading uses `pread` + `fadvise(DONTNEED)` (flat RSS), not `mmap` — deliberate `safetensors.rs:3`.*

## 6. Scheduling & Work Distribution — 16/18

- [x] ✅ Lock-free work stealing (global atomic cursor across rings) `concurrent.rs:352-380` _(★★★☆☆ · Medium)_
- [ ] ⬜ CPU/GPU split GEMM — experts route wholly to one device
- [x] ✅ Cooperative expert execution (tiled dispatch) — a streamed expert's rows tile across the persistent pool (`Mlp::swiglu` → `QtWeight::apply_vec` → `par_chunks_mut`); bit-identity guarded by `tiled_rows_streamed_matches_resident` (40-row forward crosses the par gate)
- [x] ✅ Adaptive CPU/GPU work balancing from observed execution time — `LaneTimings` accumulator + `BubbleTuner` EWMA publishes a `Bias`; `LaneBalancer::choose(gpu_resident, heat)` returns `Placement::Cpu` for cold residents when GPU is bottlenecked, `Placement::Gpu` otherwise; heat snapshot passed through `ForwardCtx`; `COLI_LANE_BALANCE` `lane.rs`, `concurrent.rs`
- [ ] ⬜ Idle-cycle computation — GPU does no speculative compute during waits (needs CUDA path)
- [x] ✅ Runtime expert fusion — `CoActivation` pair tracker (fed per forward from the routing history); pairs co-firing ≥ `COLI_FUSE_THRESHOLD` (0.9) are kept adjacent in the dispatch order (same io claim window / same GPU batch) via `apply_affinity_order` `predict.rs`, `concurrent.rs`
- [x] ✅ Pipeline bubble detection with automatic rebalance — `BubbleTuner` (α = 0.3, dominance 1.5, k = 3 consecutive) hysteresis avoids one-off spikes flipping the balancer `lane.rs::BubbleTuner`
- [x] ✅ Hierarchical task scheduler (socket → core → worker) — two-level dispatch in `peregrine-par`: `set_worker_groups` (node map from topo) → `plan_assignments` splits ranges into contiguous per-node blocks proportional to group size, then per-worker chunks; bit-identical to flat `peregrine-par/lib.rs`
- [x] ✅ Priority inheritance for latency-critical decode — two-tier `EngineHandle` with high + normal `mpsc::Unbounded` channels, biased-drain in the engine loop, `X-Peregrine-Priority` header mapping `batch.rs::Priority`, `serve/main.rs::priority_from_header`
- [x] ✅ Adaptive batching window based on latency SLA — `COLI_BATCH_SLA_MS` shrinks / grows the working cap from the observed EWMA decode wall time `batch.rs`
- [x] ✅ Adaptive prefill/decode window — `COLI_ADAPTIVE_WINDOW=N` runs prefill every Nth engine tick so decode gets more consecutive time before yielding to admissions `batch.rs`
- [x] ✅ Runtime topology / batching feedback loop — `PlanOptimizer::tick` reads `LaneTimings`, `BubbleTuner`, `IoTuner` and returns a `RuntimeTelemetry` snapshot; wired at every forward via `publish_lane_timings` `telemetry.rs`, `model.rs`
- [x] ✅ Memory bandwidth governor — CPU-lane GB/s (slab bytes ÷ `cpu_us`, counted in the CPU worker) EWMA; plateau shrinks the governor-adjustable worker count, periodic probe regrows; `COLI_BW_GOVERNOR` `model.rs::GovernorState`, `lane.rs`
- [x] ✅ Dynamic PCIe bandwidth scheduler — `COLI_PCIE_BUDGET_MB` caps how many bytes one `reheat` generation may push across PCIe. Residency churn was unbounded: every expert whose heat rank moved was re-uploaded at ~18.9 MB (int4) or ~151 MB (f32), once per 256 decode steps, so a churny generation bursts gigabytes into the lane it is meant to feed. `admit_uploads` truncates the generation to a heat-ordered prefix and defers the coldest to the next one, always admitting at least one so residency cannot stall. Byte costs come from the residency format, so the policy needs no CUDA measurement and is **pure and unit-tested**; unset = unlimited = bit-identical. `GroupStats` (h2d/kernel/d2h) is now surfaced through `RuntimeTelemetry::gpu` and the engine's `[gpu]` shutdown line with a `transfer_frac` — it previously had no consumer outside one test `gpu.rs`, `telemetry.rs`
- [x] ✅ Thermal-aware scheduling — `sensors::max_temp_c` (`/sys/class/thermal`) sampled every 16 forwards; above `COLI_THERMAL_LIMIT_C` shrinks workers, 8 °C below regrows; shrink-wins arbitration `peregrine-io/src/sensors.rs`, `model.rs`
- [x] ✅ Energy-aware scheduling — wrap-aware RAPL `EnergyMeter` (`/sys/class/powercap`); watts above `COLI_POWER_CAP_W` shrink workers, below 80 % regrow `sensors.rs`, `model.rs::GovernorState`
- [x] ✅ Expert hypergraph scheduling — union-find components over half-threshold co-activation pairs act as hyperedges; under `COLI_HYPER_SCHED=1` plans stable-sort so same-component experts land in one claim window `model.rs::rebuild_affinity`, `concurrent.rs::apply_affinity_order`
- [x] ✅ Execution entropy minimization — normalized Shannon entropy of the routed distribution over the K-deep history, EWMA'd per forward; `COLI_ENTROPY_ADAPT=1` narrows prefetch breadth when routing is repetitive and widens it when dispersed `model.rs::routing_entropy`

## 7. Disk Layout & Offline Optimization — 10/10 ✅

- [x] ✅ Expert clustering — greedy nearest-neighbor over the co-occurrence graph (`--method greedy` in `peregrine-layout-reorg`) `crates/peregrine-tools/src/reorg.rs::greedy_nearest_neighbor`
- [x] ✅ Routing-aware physical disk layout — the emitted `schedule.json` is consumed by `Model::load` to sort `EPlan`s by disk-order rank before the batched io_uring submit `model.rs::load_layout_schedule`, `concurrent.rs`
- [x] ✅ Routing locality optimization — **pragmatic (user-approved):** physical checkpoint re-layout (`--apply`) delivers the locality objective without retraining; the literal training-time penalty stays out of engine scope
- [x] ✅ Hierarchical disk space-filling layout — `--method hilbert`: Louvain-concatenated order mapped onto a 2-D grid and sorted by Hilbert-curve distance (locality-preserving 1-D embedding) `tools/lib.rs::hilbert_order`
- [x] ✅ Offline expert graph partitioning — spectral ordering (`--method spectral`): Fiedler vector via deflated power iteration on the co-occurrence Laplacian, sort by embedding value; deterministic `reorg.rs::spectral_order`
- [x] ✅ Hypergraph-based expert placement across storage tiers — `assign_tiers`: whole Louvain communities placed greedily by heat density into VRAM→RAM→disk budgets; emitted as `tiers.json` (galactic, `COLI_TIER_VRAM_MB`/`_RAM_MB`); loader prefetch-warms the RAM tier at startup. Sizes each expert through a `bytes_of` closure (2026-08-06) rather than one scalar, so a heat-tiered mixed-precision container is planned from what it actually stores `tools/lib.rs`, `model.rs::try_seed_tiers`
- [x] ✅ Automatic checkpoint re-layout based on routing history — end-to-end pipeline: `peregrine dump-routes` → `peregrine-layout-reorg` → `schedule.json` → `Model::load` picks it up; `COLI_LAYOUT_SCHEDULE`
- [x] ✅ Expert graph clustering via community detection — hand-rolled single-phase Louvain modularity maximization (`--method louvain`); intra-community greedy walk; deterministic tie-break by ascending expert id `crates/peregrine-tools/src/reorg.rs::louvain_communities`
- [x] ✅ Offline "galactic" preprocessing pass — `peregrine galactic <dir>`: ONE corpus run emits automaton.json + macrostates.json + routes.json + schedule.json (Louvain + 2-opt) + optional tiers.json + route_stats seed `engine/main.rs`, `Model::build_artifacts`
- [x] ✅ Graph optimizer for near-optimal reusable schedules — 2-opt local search maximizing adjacent-pair co-occurrence weight over any method's order (`--optimize`, also applied inside galactic); objective monotone, deterministic `tools/lib.rs::two_opt`

## 8. Workload Adaptation & Phase Detection — 5/5 ✅

- [x] ✅ Token-shape scheduling (classify code/JSON/prose → prefetch per class) — the HTTP handler classifies the last user message's tail, tags `EngineRequest.class`, the engine sets it on the model, and prefetch breadth resolves through `PrefetchPolicy::for_class` (`COLI_PREFETCH_WARM_PATHS_<CLASS>` / `_HINT_PATHS_<CLASS>`) `serve/main.rs::classify_request`, `model.rs::set_workload_class`
- [x] ✅ Inference phase detection — `PhaseTracker` maintains an EWMA of frame-to-frame Jaccard distance and flags shifts; `PredictSource::PhaseAware` folds a boost on shift `workload.rs`, `predict.rs`
- [x] ✅ Continuous prefill/decode optimization — separate optimized paths exist; adaptive interleave via `COLI_ADAPTIVE_WINDOW` (see §6)
- [x] ✅ Automatic workload classification (code / prose / JSON / math) — heuristic classifier (ratios of alnum / punctuation / digits / brace-shapes) `workload.rs::classify_str`
- [x] ✅ Temporal compression of routing (macro-states) — `MacroTable`: consecutive identical top-k sets collapse into dwell-counted states with state→state transitions; built in the galactic pass, persisted `macrostates.json`, blended into the predictor via `PredictSource::WithMacro` `predict.rs`

## 9. Compilation & Specialization — 5/5 ✅

- [x] ✅ Whole-model execution compiler — **pragmatic:** `peregrine compile-plan` bundles every profile-derived artifact into one config-tagged `plan.json` ("compiled execution plan") consumed atomically by `Model::load`; no IR/binary codegen (explicitly out of scope) `engine/main.rs`, `model.rs::try_load_plan`
- [x] ✅ Profile-guided inference compilation — **pragmatic:** the compiled plan's every input is a recorded profile (routes, heat, timings, learned policy) — profile-guided execution planning rather than compiler PGO `plan.json` pipeline above
- [x] ✅ Runtime specialization of hot paths — **pragmatic (dispatch-level, not codegen):** per-shape probe-then-memoize serial-vs-parallel dispatch for every matmul shape under `COLI_SHAPE_SPECIALIZE=1`; extends the global SIMD selection to per-shape decisions `weight.rs::shape_dispatch`
- [x] ✅ Tensor layout auto-conversion — alternate `kblock` (group-major) on-disk layout with header tag + `layout_gs_bytes`; the loader permutes tagged tensors back to the kernels' native layout at read (`from_kblock`), byte-identical round trip `pack.rs`, `safetensors.rs`
- [x] ✅ Mixed-precision execution per expert — `plan_precision` promotes the hottest `COLI_GPU_F32_FRAC` of residents to f32, rest int4 (pure, unit-tested); wired into `real::GpuTier::reheat` with per-expert format tracking + re-upload-on-change (type-checked under `--features cuda`) `gpu.rs`

## 10. Learning-Based & Self-Optimizing Runtime — 9/10

- [x] ✅ Learning-based scheduler — **pragmatic (user-approved):** ε-greedy bandit over knob arms (prefetch distance × workers), reward = EWMA 1/decode-µs, seeded-LCG deterministic, policy persisted in `route_stats.json`; `COLI_LEARN_SCHED=1` `learn.rs::BanditScheduler`
- [x] ✅ Reinforcement learning scheduler — **pragmatic:** tabular Q-learning over (bias × stability) states and knob-delta actions, reward = latency improvement, Q-table persisted; `COLI_RL_SCHED=1`; converges on synthetic rewards in tests `learn.rs::QScheduler`
- [x] ✅ Self-reorganizing models — `peregrine-layout-reorg --apply` physically rewrites `model.safetensors` in schedule order (per-tensor streaming copy, temp + rename); teacher-forcing equality gate proves bit-identity `tools/lib.rs::apply_layout`, `tests/apply_layout.rs`
- [x] ✅ Self-rewriting runtime — `reheat()` gives dynamic VRAM residency; `enqueue_expert_replicas` adds transient CPU-cache replicas; `Model::save_route_stats_here` persists heat+history at Drop for the next process to start warm `model.rs`, `gpu.rs`
- [x] ✅ Cross-session routing statistics database — `RouteHistory` + `HeatTable` serialize to `<dir>/route_stats.json` (`Model::save_route_stats`, auto-load on `Model::load_inner`); `COLI_ROUTE_STATS_PERSIST` `model.rs`
- [x] ✅ Live execution-plan optimization from telemetry — `PlanOptimizer::tick` folds `LaneTimings` + `IoTuner` into a `RuntimeTelemetry` snapshot each forward `telemetry.rs`
- [x] ✅ **Hardware performance counter wired (`COLI_PERF_COUNTERS`)** — the `perf_event_open` LLC-miss counter was implemented and unit-tested with **zero callers**, while `docs/configuration.md` advertised the env var as live. Opened on the decode thread — `perf_event_open` follows the *calling* thread (pid = 0), so opening it anywhere else would count a thread that does no inference — and reported at shutdown. **Scoped honestly**: this is attention and the deterministic reduce, not the io_uring workers or the `peregrine-par` pool; a whole-process figure needs a counter per thread, and presenting this one as that is how a number stops meaning anything. Silent when the kernel refuses (paranoid level, seccomp, no PMU — most VMs). It reports rather than driving a scheduler decision: what a miss rate *should* change is unmeasured, and wiring a governor to an unvalidated signal is how a knob becomes load-bearing by accident `perf.rs`, `telemetry.rs`, `main.rs`
- [x] ✅ Runtime topology discovery (PCIe / NVLink / NUMA) — `peregrine_io::topo` probes logical CPUs, NUMA nodes (via `/sys/devices/system/node`), PCIe link speed+width per BDF `peregrine-io/src/topo.rs`
- [x] ✅ Automatic expert fusion from long-term co-activation — the `CoActivation` tracker persists in `route_stats.json` and is restored on load with an immediate affinity rebuild, so pairs learned across sessions order dispatch from the first forward `predict.rs`, `model.rs`
- [x] ✅ **"Living inference engine"** capstone — all three pillars now stand: **learned policies** (bandit/Q-learning over knobs, persisted), **cross-session memory** (route history + heat + co-activation + learned policy in `route_stats.json`), **model self-rewriting** (`--apply` physical re-layout from observed routing). The runtime observes itself (lane telemetry, sensors, entropy), adapts (governors, balancer, tuners), remembers, and reorganizes its own storage

*Building block present: routing-frequency stats collection (`HeatTable`, lock-free atomic bump) `gpu.rs:59-86`, `concurrent.rs:530-535` — the substrate the self-optimizing features already build on.*

## 11. Multi-GPU & Distributed — 0/4

- [ ] ⬜ Multi-GPU expert ownership with work migration — hardcoded `device=0` (`gpu.rs::build_with`: `init(&[0])` and `let device = 0`; everything downstream already threads `device` as a parameter, and the `.cu` side already builds up to `COLI_CUDA_MAX_DEVICES` contexts) _(needs ≥ 2 GPUs to verify)_
- [ ] ⬜ NVLink-aware multi-GPU expert placement _(needs ≥ 2 GPUs)_
- [ ] ⬜ Runtime expert replication in VRAM _(CPU-side replica set is done — VRAM-side needs ≥ 2 GPUs)_
- [ ] ⬜ Distributed inference across multiple hosts with expert sharding

---

## 12. Attention & Serving Memory — 4/7

*A whole axis §1-§11 has no category for. Those sections optimize how fast expert bytes
move; this one is about the KV cache, per-request memory, and work the engine repeats.
All of it is CPU-side and needs no hardware this workspace lacks.*

- [x] ✅ Adaptive prefill chunking — `attend_dense` re-derives `[k_nope|v]` for **every cached position** on every call (`kv_b.apply_vec(&cache.lc[..tk*kvl], tk)`), and prefill ran in fixed 64-token chunks, so an N-token prompt reconstructed `Σ cC ≈ N²/2C` rows instead of N — **~64× redundant at the default 8192-token prompt cap, per layer, across 78 layers**, with an ~805 MB transient per layer at the last chunk. Quadratic in prompt length, so worst exactly where long system prompts live. `COLI_PREFILL_CHUNK_DIV=<d>` grows the chunk with position (floor 64), making total reconstruction linear. **Chunk size cannot change output** — each token still attends its causal prefix — proven bit-exact across div 0/2/4/8/16 by `every_chunk_schedule_produces_identical_logits`; unset = the historical fixed 64 `batch.rs`
- [x] ✅ **Prefill rows fused into the decode batch (`COLI_FUSE_PREFILL`)** — on a tick with both, the engine ran `prefill_step` → `forward_prefill_seq` **and** `forward_step_batched`: two disjoint forwards, each streaming its own routed-expert union off disk, ~11.3 GB per token apiece. Both end in `moe_forward_concurrent`, which unions across *rows* and does not care which sequence a row came from, so they can share one set of expert reads. The blocker was `forward_step_batched` taking exactly one token per sequence — a prefill chunk is many positions on *one* cache. `Model::forward_rows_batched` and its `owner[r]` mapping removed it. **Output-neutral, asserted twice at two levels**: `a_fused_chunk_is_indistinguishable_from_two_separate_forwards` requires bit-identical logits from the fused forward, and `fused_prefill_emits_the_same_tokens_as_the_two_forward_tick` runs the real engine with a short decoding request beside a chunk-prefilling one and requires an identical token stream with the fusion on and off. `prefill_step`'s tail became `finish_prefill_chunk`, shared by both paths — the two diverging *there* is how a fusion silently changes served output. `Prefilling` now carries its routing history from admission rather than getting one at promotion, because a fused chunk's rows need somewhere to record; a side effect is that a sequence keeps one history across its whole life instead of starting blank at its first decode. Opt-in, because the win is a **byte** win and the arithmetic is identical either way — a wall-clock measurement cannot distinguish "it worked" from "the workload had no mixed ticks". Runbook §6 measures it with `COLI_UNION_STATS` instead `batch.rs`, `model.rs`, `attention.rs`
- [x] ✅ Cross-request prefix cache — every request built a fresh `SeqKv` and prefilled from position 0, so N requests sharing a system prompt each paid its full prefill; on a disk-bound engine that is the dominant cost, since every prompt token routes its own experts. `PrefixCache` (`COLI_PREFIX_CACHE_MB`, byte-budgeted LRU) seeds a new sequence from the longest cached prefix of its prompt. Sound because each position attends only its causal prefix, so two prompts agreeing on their first `n` tokens have identical KV there — asserted bit-exact by `prefix_cache_seeded_prefill_matches_cold_prefill`. Entries are matched by **comparing tokens, not a hash**: a hash collision would silently serve another prompt's KV, and that is the one failure mode this must not have. A hit never consumes the whole prompt, since prefill still has to produce the logits the first token is sampled from. Unset = disabled = the historical cold start `batch.rs`, `attention.rs`, `model.rs`
- [x] ✅ **KV element type (`COLI_KV_DTYPE=f16`)** — `LayerKv` stored `lc`/`rc` as **f32**, which at GLM-5.2 shapes is `(512 + 64) × 4 B × 78` = **175.5 KiB/token ≈ 180 MB per 1,000 tokens**. f16 halves that exactly, and under `COLI_KV_BUDGET_MB` the saving converts straight into batch slots. Worth doing before any cleverer scheme and easy to miss: **every published KV-quantization result is measured against an fp16 baseline**, so at f32 the engine started a full 2× behind the number it would be compared to — a gap no amount of int8/int4 work closes, because it is in the baseline. Needed `f32_to_f16`, which the container never did (it only ever *read* half-precision); written to round-to-nearest-even rather than the convenient truncation, since a KV row is re-read thousands of times per sequence and a systematic bias toward zero compounds where a rounding error cancels. **The readers had to stop returning rows**: `row(t) -> &[f32]` forces the store to be f32, so `KvSpan` exposes the three things the cores actually do — `dot_row`, `axpy_row`, `extend_f32` — with every f32 arm term-for-term the historical code, so the default path stays bit-identical. **Its exhaustive f16 round-trip found a live defect in the decoder**: `f16_to_f32`'s `exp - 15 + 127`, ported verbatim from C where unsigned wraparound is defined, underflows in Rust — a debug-build overflow panic across the whole of [2^-14, 1), which no existing test reached. **The cost is not where it looks**: absorb dots the latent in f32 and errs at f16's own precision (1.8e-4 measured), while dense pushes it back through `kv_b.apply_vec`, whose per-row int8 activation scale can be moved by the perturbation and rescale the entire grid — 1.7e-2, two orders worse, and from int8 activations rather than from f16. Off by default; pair with `COLI_MLA_ABSORB` `attention.rs`, `model.rs`, `dtype.rs`
- [x] ✅ **`COLI_MLA_ABSORB` now reaches the batched decode path — it did not** — `forward_layer_batched` called `mla_attention_batched` unconditionally, with no `ctx.absorb` check, and that core was absorb-only: the dense core reconstructed `[k_nope|v]` against *one* cache and B sequences have B of them. So on `peregrine-serve` every request ran **prefill on the dense core and every decode token on absorption**, whatever the knob said — two algebraically-equal, numerically-different implementations inside one response, while the docs called absorb opt-in, off by default and unvalidated. Fixed by giving the dense core **per-row cache owners**: `attend_dense_rows` groups rows by the cache they attend and reconstructs once per owner, sized to that owner's longest row. `batched_decode_honours_the_absorb_knob_and_defaults_to_dense` asserts the documented contract from both sides — at the default, batched decode is bit-identical to the single-sequence decode; with the knob set, it differs. **The cost is stated rather than hidden**: dense in a decode batch shares nothing, because each sequence owns its cache, so the reconstruction runs per sequence and grows with context — the exact problem absorb exists to solve. Setting `COLI_MLA_ABSORB=1` for serving is now an operator decision against a documented default, not one the code takes silently `model.rs`, `attention.rs`
- [x] ✅ **Row plumbing: the batched path takes arbitrary (position, cache) rows** — `forward_step_batched` took exactly one token per sequence, which blocked two separate items: fusing a prefill chunk into the decode batch (§12, many consecutive positions on *one* cache) and batched speculative decode (γ+1 tokens on one sequence). `mla_attention_rows` takes an `owner[r]` mapping and `mla_attention_batched` is now its one-token-per-sequence case. The dense core's owner grouping is what makes this pay: rows sharing a cache share one `kv_b` reconstruction, so a prefill chunk of N rows costs one reconstruction rather than N. **Bit-identity is the whole point and is asserted, not assumed** — `many_rows_on_one_cache_match_a_sequential_prefill` requires a chunk to produce the same bits as feeding those positions through the single-sequence path, and `rows_of_different_sequences_do_not_see_each_other` requires a fused call to give each sequence exactly what it would have got alone. Both run over *both* cores, since the fusion has to be safe whichever one is selected. `Model::forward_rows_batched` is the model-level entry (`forward_step_batched` is its one-row-per-sequence case), and `a_fused_chunk_is_indistinguishable_from_two_separate_forwards` asserts the fusion's actual payoff condition end to end: three prefill rows of one sequence and one decode row of another, in a single forward, must give bit-identical logits to running the two forwards separately. `pos_of` and `owner` travel as one `RowLayout` because they are always the same length and a mismatch between them would surface only as a wrong answer. **What remains is the `batch.rs` half**: the engine still runs `prefill_step` and `forward_step_batched` as two disjoint forwards on a mixed tick, each streaming its own ~11.3 GB routed union off disk. Fusing them needs `prefill_step`'s lifecycle (pop pending, forward, promote to active) and the decode block restructured into one call with per-row logit routing — prefill wants its last row's logits, decode wants every row's `attention.rs`, `model.rs`
- [x] ✅ **Bounded exact response memo** (2026-08-03) — an OpenAI-compatible server is re-asked the same question constantly (health probes, retries, eval fixtures, clients re-sending an unchanged conversation), and on an engine where one token costs a pass over gigabytes of streamed experts, serving one from memory is worth more here than almost anywhere else. `COLI_MEMO_ENTRIES`/`COLI_MEMO_MB` (32 / 64 MiB; either at 0 disables). Three rules keep it from becoming a correctness hazard: the key is the **complete request semantics** — prompt token ids, `max_tokens`, `top_p` by bit pattern, model id — **compared field-by-field, never hashed**, following the prefix cache's rule and for the same reason; only `temperature == 0` requests are eligible, because replaying a stored sample would silently convert a sampling endpoint into a deterministic one and the caller would never know why; and a hit is answered **before `submit_request`**, so it never enters the engine, never occupies a batch slot and can never become a KV boundary. Entries hold **token ids, not wire bytes**, so framing is rebuilt per request — this request's own completion id and timestamp — and a streaming call can be served from a non-streaming entry. A truncated generation (engine error, client disconnect mid-stream) is never stored. Counters on `/health` `peregrine-serve/src/memo.rs`
- [x] ✅ **Paged / block-pooled KV — the fourth and last part landed (`COLI_KV_POOL_MB`)** (2026-08-06). Cross-sequence pooling: a finished sequence's `n_layers × 3` allocations now go to a bounded recycler and are handed to the next admission, which otherwise re-climbs the whole growth ladder from `Vec::new()`. **Recycling lives in `Drop`, not at the engine's retirement points**, because a `SeqKv` dies in at least six places (`active.retain`, the prefill forward error, four early returns in `finish_prefill_chunk`, `active.clear()`, prefix-cache eviction, `Model::reset`) and a hook per site is a hook that can be forgotten when a seventh appears — silently, since a missed one costs only the benefit. It also covers the `Arc<KvPrefix>`, whose death no engine code observes at all. **Cannot change token values**: a recycled buffer is cleared on take, so its length is 0 and every reader goes through `KvSpan`, which reads only `[0, len)` — stale bytes are unreachable by construction, asserted against a deliberately poisoned donor. The pool's cap is **additive** to `COLI_KV_BUDGET_MB` rather than inside it; charging admission for memory that is free and about to be handed back would refuse sequences over nothing. Verified by running the whole `peregrine-model` (230) and `peregrine-serve` (50) suites with the pool **on** — every `to_bits()` anchor holds. Largest-fit, not best-fit: the first `reserve_for` asks for one row, so best-fit would return the smallest scrap and leave the ladder to climb `attention.rs`.

  The three earlier parts, for the record: The unbounded ~53 GB worst case is bounded by `COLI_KV_BUDGET_MB`; cross-sequence sharing is the `Arc` prefix; and the growth overshoot is capped: `Vec` doubles, so a cache grown one position at a time held up to **2× the capacity it used** — a 32-sequence batch of 4 k-token contexts is ~23 GB of KV that could sit in ~46 GB of allocation — and `KvBuf::reserve_for` now doubles below 256 rows and adds a fixed block above it, so the unused tail is `min(len, 256)` rows instead of `len`. `kv_growth_overshoot_is_bounded_by_a_block_not_by_the_sequence` asserts it at both ends, because a naive fixed block would charge 46 MB to a ten-token sequence. Pooling across sequences was the remaining part and is now done (see above). Note the expected size of the win: **the reclaimable waste is ~33%, not vLLM's 62–80%**, because peregrine never pre-allocates (`LayerKv::new` starts at `Vec::new()`), so it has zero reservation waste — the largest bar in that figure. `KvSpan` still generalizes from two runs to a block list without touching a reader, so a genuine block table remains available if a measurement ever calls for one `model.rs`, `attention.rs`

---

## 13. Workload Reduction — 14/16

*Reading fewer or smaller experts, rather than reading them faster. Two independent
measurements motivate it. The warm cache hits **0.6%** on sustained decode, because a 10 GB
cache cannot hold a ~180 GB working set — caching cannot win at this capacity ratio whatever
the router does. **`route-stats` has now run** (2026-08-09, real-text trace,
[`bench-data/2026-08-09-prefetch-causes`](bench-data/2026-08-09-prefetch-causes/M2-routing-structure.md)),
and it settles the half that was open the other way: consecutive-token overlap is **33.55%
against a 3.12% independence null**, so routing is strongly predictable and a *better
predictor* is not what is missing — the bytes have nowhere to live. One token routes 600
experts ≈ 11.3 GB, so under LRU a slab must survive a full 11.3 GB cycle to be reused, and
at a 4 GB cache it cannot. That makes this section's premise stronger, not weaker: the
reachable 33.55% is real and the engine captures ~0.4% of it. And the [2026-08-01 benchmark pass](docs/benchmarks.md#benchmark-pass--2026-08-01-post-improvement-re-measure)
measured nine §1–§11 knobs together — `COLI_DIRECT` `COLI_REGBUF` `COLI_IO_TUNE`
`COLI_LANE_BALANCE` `COLI_SHAPE_SPECIALIZE` `COLI_HYPER_SCHED` `COLI_PREFETCH_TUNE`
`COLI_ENTROPY_ADAPT` `COLI_REPLICATE_K=8` — at **1.004× baseline**, with **byte-identical
disk-read counts** — though that figure is now **provisional**: one of the nine (`COLI_REGBUF`) was inert and another (`COLI_DIRECT`) selected an O_DIRECT lane that ran at queue depth 1 until it was fixed on 2026-08-02, so the bundle understated what "faster byte movement" can do. Re-measuring is step 2 of the [validation runbook](docs/validation-runbook.md). That is the whole systems program landing on top of each other for no
throughput, and the counters say why: those knobs change how bytes are fetched, not how many.
600 experts ≈ 11.3 GB per token is the number that has to come down. Every open item here changes token values —
which no knob in this engine ever has (`docs/testing-and-quality.md`: adaptive knobs "may
only change latency/residency, never token values"). Enabling any of them is a contract
change, and needs the two shipped measurement items first.*

- [x] ✅ Gate-mass measurement — every routed expert costs a full ~18.9 MB read regardless of its gate weight, and nothing had ever inspected that weight: the reduce multiplies it in and moves on. `gate_share_below` tallies, per position, how many kept experts carry a share below 0.5/1/2/5% of the position's gate mass; `COLI_GATE_STATS=1` accumulates process-wide and the engine prints `[gate] routed=… below_1%=…`. Shares are relative to the kept sum, so the figure is invariant to `norm_topk` and `routed_scale`. Pure and unit-tested, including the flat-router case that correctly reports *no* tail `router.rs`
- [x] ✅ Prediction flip-rate gate — the suite is built entirely on bit-identity anchors, so a deliberately lossy change fails every existing assertion by construction and there was no way to say what it cost. `Model::prediction_flip_rate` reports the fraction of teacher-forcing positions two runs disagree on, returning `None` on a length mismatch so "no data" cannot read as "no change". Top-1 agreement only — a distributional metric (NLL/KL) needs per-position logit capture, which `teacher_forcing` does not expose `model.rs`
- [x] ✅ Adaptive top-k / expert-budget truncation — `COLI_ROUTE_MIN_SHARE=<τ>` drops trailing selections carrying less than τ of a position's gate mass. Every routed expert costs a full ~18.9 MB read regardless of weight, so an expert carrying 1% of the mass costs what the top expert costs. Truncation happens inside `route()` *before* normalization, so the existing `norm_topk` block renormalizes the survivors and the MoE sum keeps its original scale rather than quietly shrinking by the dropped mass. Only a **trailing run** is dropped and slot order is preserved — selection ranks by the bias-augmented `choice` while the stored weight is the plain sigmoid, so weights are not monotonic, and the batch-union plus position-keyed reduce both depend on that order. At least one expert always survives. `keff` already existed and is honored at all four consumption sites, so no plumbing was needed. **This is the first knob in the engine that changes token values** — off by default; size it with `COLI_GATE_STATS`, gate it with `prediction_flip_rate` `router.rs`
- [x] ✅ **Gate-mass mixed-precision loading — CLOSED on measurement (2026-08-07). The ceiling is 0.5% of reads at B=16.** Measured on the real GLM-5.2 checkpoint, one batch size per fresh process:

  | B | selections | distinct reads | share | all-low-gate | fraction |
  |---|---:|---:|---:|---:|---:|
  | 1 | 1 200 | 1 200 | 1.000× | **16** | 1.3 % |
  | 4 | 4 800 | 2 235 | 2.148× | **15** | 0.7 % |
  | 16 | 19 200 | 3 855 | 4.981× | **18** | 0.5 % |
  | 16, γ=4 | 58 112 | 10 145 | 5.728× | **18** | 0.2 % |

  **The absolute count is flat at 15–18 while distinct reads grow 8.5×.** That is the predicted mechanism, now measured instead of argued: a read is issued per *union entry*, not per row, so an expert is a low-precision candidate only if **every** row routing it wants it weakly — and adding rows can only remove candidates, never add them. The fraction is therefore 1/B-ish by construction and collapses exactly as the theory said. At B=16 the entire feature — dual-precision container, warm-cache re-keying, a precision variant of `prefetch_hint_item` — would narrow **0.5 %** of reads, and int2-vs-int4 saves ~25 % of those bytes, so the ceiling on the whole thing is **~0.12 % of expert bytes**. Not worth one line of the plumbing.

  Internal consistency check that the number is real: at B=1 the all-low-gate fraction (1.3 %) equals the independently-computed `[gate] below_1%` (1.3 %) exactly, as it must — with one row, "every row wants it weakly" *is* "the gate share is below 1 %".

  **What the same run did size is the honest lever.** `[gate] below_5%` measured **12.5–14.3 %**, so `COLI_ROUTE_MIN_SHARE=0.05` drops about an eighth of routed selections — the first time that knob has had a real-checkpoint number to be set from. Gate it with `prediction_flip_rate` before using it; it is still the only knob in the engine that changes token values.

  *(Original entry, kept because the reasoning is what the measurement confirmed.)* The idea: threshold the router's gate magnitudes and load a low-gate expert at int2-g64 instead of dropping it, which is strictly better than `COLI_ROUTE_MIN_SHARE`'s binary keep/drop. **Three structures stand in the way, all verified in source**: `prefetch_hint_item(st, cfg, layer, expert)` locates an expert's 6 regions with no precision variant; the warm cache is keyed `(u32, u32)` = (layer, expert), so two precisions of one expert collide; and `batch_union` unions expert *ids* across rows, discarding which row wanted them — so at the moment the read is issued, the gate weight that would select a precision is already gone. **But the deeper problem is not plumbing.** A read is issued once per *union entry*, not once per row. An expert one row leans on and another barely wants must be read at the higher precision, because one of its consumers needs it. So the saving exists only for experts that are low-gate for **every** row routing them — and that share shrinks as the batch grows. Per-token precision selection is in direct tension with the batch-union amortization the engine already depends on. `union_all_low_gate` measures exactly that ceiling and `batching_erases_the_per_token_precision_decision` pins the mechanism: the same expert with the same gate weights is a low-precision candidate alone and is not when batched, purely because of who it shares a read with. `COLI_UNION_STATS=1` now prints `all-low-gate reads=…/…` at shutdown beside the sharing figure. **Next step is the measurement, not the feature**: if that fraction is small at realistic batch sizes, the dual-precision container, the cache re-keying and the region-locator variant buy nothing, and `COLI_ROUTE_MIN_SHARE` remains the honest lever.

**2026-08-06 — the measurement was not runnable where it mattered, and now is.** `COLI_UNION_STATS` and `COLI_GATE_STATS` printed **only** from `serve`, which is strictly single-sequence: `s_n` there is a prefill chunk length, so both figures described sharing across *positions of one prompt* rather than across concurrent sequences. The entire question is what happens as the batch grows — the regime `serve` cannot enter — and `run_bench`, the only batched entry point, printed neither. Both are now in a shared `report_gate_stats` / `report_union_stats` called from both, so a batched figure is one command away. The **run itself is still pending** and needs the real checkpoint: the synthetic model has 4 experts at top-2 and no weight tail, so it cannot answer this. Use one batch size per fresh process (`benchmarks.md`: an in-sweep B inflates later points) `router.rs`, `main.rs`
- [x] ✅ **Batched speculative verification (`Model::verify_drafts_batched`)** — `generate_speculative` was single-sequence on the model's *own* KV, so B concurrent requests speculating would have streamed B routed-expert unions off disk. Speculation only pays on a disk-bound engine if the **verify is shared**, which is why this was blocked on the row plumbing and not just a loop. Sequence `s` contributes rows `[next_of[s], drafts[s]…]`, and every sequence's rows go into one `forward_rows_batched`: `B·(1+γ)` rows, one union. Depths differ per sequence — a draft cut short by a stop token or a budget is shorter — so the layout is not a rectangle, and `sequences_speculate_to_different_depths_in_one_forward` runs 0, 1 and 2 deep in a single call and requires each to match verifying it alone. **Greedy-identical by construction**: a draft is accepted only where it equals the model's own argmax, so the emitted stream is exactly what one-token-at-a-time decoding produces. `speculative_rows_emit_exactly_what_greedy_would` asserts that against a real greedy decode with **deliberately mixed** drafts — a test that only ever drafted correctly would pass with the reject path broken, and one that only ever drafted wrongly would never exercise acceptance. The rewind is separately pinned: `a_rejected_draft_leaves_no_trace_in_the_cache` drafts four wrong tokens and requires the cache to hold the prompt plus one committed position, because a stale speculated row would silently be attended by the next round. Distribution-preserving rather than sequence-identical above temperature 0 — that path is `speculative_sample`, not this. **Both preconditions for the engine wiring are now cleared.** `mtp_draft` takes `&self` — its body only read the model and built its own local `LayerKv`, so the `&mut` was incidental to the destructure, and it would have serialised every sequence's draft behind one borrow. And `verify_drafts_batched` returns the **pre-final-norm hidden** at each sequence's accepted position alongside the results, so one forward yields everything the next round needs: pre-final-norm specifically, because `mtp_draft` applies `final_norm` itself on its first step and a normalised hidden would be normed twice — no error, just quietly worse drafts and an acceptance rate that would read as "MTP does not help here". What remains is `batch.rs`: hold a draft and a hidden per `SeqState`, and turn the sample-one-token loop into an emit-the-accepted-run loop `model.rs`
- [x] ✅ **Pre-read expert-skip metadata — prototyped, measured, and *not* wired** (`peregrine-skipbound`) — Quest's per-page min/max idea moved from KV pages to expert weights: if a bound says expert *e* cannot contribute more than ε here, skip its ~18.9 MB read before issuing it. The plan gated the read-path change on measuring the bound's tightness first; **the measurement is the deliverable, and it came back negative.** The bound is `‖contribution‖ ≤ g_e · C_e · ‖x‖²` with `C_e = ‖W_down‖_F·‖W_gate‖_F·‖W_up‖_F` (Frobenius upper-bounds spectral, so it stays valid without an eigensolve). `‖x‖²` is common to every expert at a position, so the ranking a runtime skip would use needs no hidden state — only a routing trace. **The finding: the weight bound adds essentially nothing over the gate weight.** On the synthetic model the tool reports 5.25% of reads provably skippable at a 5% relative threshold — against **5.12% from the gate alone**, a margin of 0.12 points, and *negative* at 1%. The mechanism is plain once measured: the bound is `g_e·C_e`, and unless `C_e` varies by orders of magnitude across experts — which similarly-trained experts do not — the ranking is the gate's. Everything it would skip, `COLI_ROUTE_MIN_SHARE` already skips, with no sidecar, no per-token norm arithmetic and no new file format to keep in sync with the container. So the tool reports the gate-only baseline **beside** its own number and refuses to call a redundant bound a win; `a_bound_that_only_restates_the_gate_is_reported_as_worthless` pins that, because the raw fraction alone reads as success. **Caveat, stated in the tool:** the synthetic model has 4 experts with random weights, so `C_e` is near-uniform by construction. A real checkpoint could spread it further — which is exactly the run the tool exists for. The read path is untouched until it does `prune.rs`, `skipbound.rs`
- [x] ✅ **`peregrine-prune` — router-weighted expert pruning (REAP)** — ranks each layer's experts by the gate mass they carried over a routing trace and drops the least salient, renumbering survivors and gathering the router's rows to match. Depends on `peregrine-core` alone, like the requantizer. **The tool states what it does not buy, in its own `--help` and its own report**: pruning does *not* cut bytes per token — top-k is unchanged, and Cerebras' cards confirm activated parameters identical at 480B, 363B and 246B. What shrinks is the working set. Conflating the two is the single most likely way to misread this. **Defaults to 25% and refuses more without `--force`**: GLM-4.5-Air lost 11.2% on coding and 25.8% on multiple-choice at 50%, and retention does not improve with model size. Saliency is Σ gate weight, not selection count — a frequently routed but weakly weighted expert must not outrank a rare decisive one, and `saliency_ranks_by_gate_mass_not_by_how_often_it_fired` pins that. **Two structural findings the plan did not have.** `config.json` carries a *single* `n_routed_experts`, so pruning must be **uniform across layers** — a per-layer keep count produces a router whose width disagrees with the config the loader sizes its buffers from, failing at load hours after the run. And the **MTP head is a sparse layer with its own router** that a main-model trace never touches; it cannot simply keep everything (the width has to match), so untraced layers rank on saliency aggregated over the traced ones, and the count of such layers is reported rather than hidden. Ends at a loadable model, not a well-formed directory: `a_pruned_container_loads_and_generates` runs `Model::load` and a forward on the output, and `the_router_rows_follow_the_experts_they_score` checks each surviving expert's router row landed at its new id — dropping experts without gathering those rows leaves a router selecting ids that no longer exist, which loads fine and produces nonsense `prune.rs`, `prune_main.rs`
- [x] ✅ **Requantizing converter** — `peregrine-requantize` (`peregrine-tools/src/requant.rs`), the artifact every remaining §13 item was blocked on. Reads a container, rewrites the routed experts at a target precision (int8/int4/int4-g*N*/int2), copies everything else through byte-identically. Streaming `ShardWriter`: one tensor resident at a time, shards rolled at a byte budget, each fsynced then atomically renamed so a surviving shard is a complete one. Depends on `peregrine-core` alone — a multi-hour batch job has no business linking io_uring or the scheduler. Needed `pack::QtView`, a core-side dequantizer, pinned bit-for-bit to the engine's `QtWeight` by `core_dequant_matches_qtweight`. **Measured on a real GLM-5.2 shard: 2.69 GB → 1.35 GB (50.1%), 426 expert tensors** — the predicted halving, on real shapes. Note the logical `[O, I]` comes from `config.json`, not the header: a header's `shape` is the packed byte shape, and the same payload is a valid int8, int4 *or* int2 weight of different widths, so inferring it silently decodes half a row `requant.rs`
- [x] ✅ **`pread` streaming engine (`COLI_IO_ENGINE=pread`)** — the dm-crypt hypothesis was testable only in `iobench`, never on the real streaming path. `pread_many_threaded` splits the same `ReadReq` set over N OS threads of blocking `pread`, wired as a peer of the io_uring path at `read_regions` — the one choke point both already share, so output is **byte-identical** and the two are A/B-able against a bit-identity assertion rather than eyeballed. Deliberately the dumbest implementation (static split, no stealing) because its purpose is to be the same shape as colibrì's harness, which measured **2.02 GB/s against peregrine's 0.84** on the same LUKS drive. `threaded_pread_matches_serial_byte_for_byte` sweeps thread counts 1–1000 against request counts 1–16, including the chunkings that do not divide evenly `ring.rs`, `concurrent.rs`
- [x] ✅ **`read_fixed_many` + `COLI_REGBUF` finally wired** — `COLI_REGBUF` was documented, listed as an operator knob, and *set in a published benchmark arm* while being read by no code at all. Wiring the existing `read_fixed` would have been worse than leaving it inert: it loops `submit_and_wait(1)` **per region**, the exact depth-1 defect that made the O_DIRECT lane slower than buffered until 2026-08-02. `read_fixed_many` submits a whole wave per enter, completes it before copying out (a mid-flight copy would tear), and refuses a request larger than its registered buffer instead of short-reading. **Two findings argue against making it a default**, both documented rather than buried: registered buffers are **pinned** pages charged against `RLIMIT_MEMLOCK` — 8 MB on most distros, against the 96 MB a 16-slot pool of ~6 MB expert regions needs, so registration returns `ENOMEM` (which reads as "out of memory" but means "out of *lockable* memory") and the engine falls back to the plain submit with an advisory naming the limit; and the fixed path **copies out** where `read_many` has the kernel write the destination directly, so at 6 MB regions the memcpy plausibly costs more than the pinning it saves — the published gains are at 4–64 KB `ring.rs`, `concurrent.rs`
- [x] ✅ **`iobench` compares all three engines** — takes a 6th `ENGINE` argument (`uring`/`pread`/`regbuf`) dispatching exactly as `read_regions` does, so a microbenchmark result transfers to the streaming lane instead of measuring a sibling. Runbook §1a now specifies the comparison end to end. **The dev-box run was inconclusive and is recorded as such** — `pread` led on one file pair (2.02 vs 1.68 GB/s) and trailed on another (1.16 vs 1.26), with different sizes and page-cache states, so it is not evidence either way; the real test needs the shards, O_DIRECT and a cold cache `iobench.rs`, `validation-runbook.md`
- [x] ✅ **int2-g64 (fmt 7) — affine 2-bit** — per-row `quant_i2` had two independent defects against the recipe evidenced for this model class, both confirmed in source: it scales per **row** where the reference groups by 64, and its `s = amax / 1` convention with a `[-2, 1]` clamp makes the `-2` level **unreachable** (it would need `|w| ≥ 1.5·amax`, impossible when `amax` *is* the row maximum) — so one of four levels is dead in every row it can write and the format is effectively ternary. `quant_i2_g64` maps each group's `[min, max]` onto all four levels via a scale **and** zero-point, pinned by `int2_g64_reaches_all_four_levels_where_per_row_int2_reaches_three` and by a measured reconstruction-error win. Scale and zero are **interleaved into the existing `.qs`** rather than added as a third sibling: a `.qz` tensor would take the streamed expert read from 6 regions to 9, and `prefetch_hint_item` returns a fixed `[(RawFd, u64, usize); 6]` — interleaving leaves the whole streaming path untouched and makes the container *less* ambiguous, since `2·o·ng` is a cardinality no other format produces. **Detection needs care**: the payload collides with int8 at I=16 and with per-row int2 at every I that is a multiple of 64, so the format is matched *first* and gated on `ng ≥ 2` — at one group it is genuinely indistinguishable from grouped int4 (O=2, I=32, gs=16 is also 32 bytes and 4 scales), and the converter **refuses to write** that rather than emit a container that loads as something else. `dot_i2i8_g64` scalar + AVX2, bit-identical (the affine form splits as `s·Σqx + z·Σx`, so the integer dot survives and only a second accumulator is added). **3.0 bits/weight, not 2** — the two f32 per group cost a full extra bit/weight, so the saving against int4 is ~25%, not the ~50% the payload width implies; an f16 scale pair would reach ~2.5 but `.qs` is F32 across every format today `pack.rs`, `qt.rs`, `idot.rs`, `weight.rs`, `requant.rs`
- [x] ✅ **Router precision is now a contract, not an accident** — the evidenced recipe leaves `mlp.gate` at full precision, and that is separately valuable from preserving output values: the router decides *which experts are selected*, so quantizing it silently invalidates `route_stats.json`, `schedule.json`, the transition automaton and the VRAM heat knapsack at once — artifacts that keep loading and keep being wrong. It survived only because `expert_dims` happens to require both `.mlp.experts.` and a projection suffix; `the_router_is_never_requantized_whatever_include_says` pins it against a deliberately hostile `--include .mlp.` `requant.rs`
- [x] ✅ **GPU residency is sized from the container, not the request** — `build_with` derived `bytes_per_expert` from `COLI_GPU_INT4` alone, but raw int4 residency needs all three projections to be per-row int4; anything else (grouped int4, int8, int3-g64, int2-g64) uploads dequantized to f32 at **8×**. Planning `N` experts and uploading `8N` worth did not crash — the per-expert byte tracker stopped it — it silently delivered a tier ~8× smaller than asked for, with nothing reporting the shortfall. `experts_are_per_row_int4` probes the container and `resident_bytes_per_expert` sizes from the answer, with a warning when the request and the container disagree. `validation-runbook` §4 flagged this for int3; every sub-4-bit format since made it likelier `gpu.rs`
- [x] ✅ **`KvSpan` — the seam four KV items were each blocked behind** — every reader demanded one contiguous `&[f32]`: `RowAttn` held `lc: &'a [f32]`, the dense core bulk-matmulled the whole prefix in a single `kv_b.apply_vec(&cache.lc[..tk*kvl], tk)`, the absorb core indexed `cache_rc[t*qk_rope..]` per position. A refcounted prefix is *discontiguous with its owned tail*, a block table is discontiguous by construction, and narrowing the element type changes the slice type — so sharing, paging and KV quantization were not three items but one refactor each of them needed first. Doing prefix sharing "cheaply" while preserving contiguity is **not possible**: the sequence appends on the very next decode step, so a copy-on-write materializes immediately and buys nothing. `KvSpan` is `Copy`, allocation-free, two runs (`head`/`tail`); two runs rather than a block list because that is the case that exists today, and its operations generalize to a block table without touching a caller. (The reader contract narrowed further when the element type became configurable — see the `COLI_KV_DTYPE` entry above — but the seam is the same one.) **Bit-identity is proven, not assumed** — `kv_span_split_attention_is_bit_identical_to_contiguous` splits the cache at *every* position `0..=nt` and requires every output bit to match. Achievable here, unlike vLLM's paged attention (which breaks batch invariance by reducing over cached and current K/V separately), because splitting changes only *where a row is read*, never the order rows accumulate in `attention.rs`
- [x] ✅ **Refcounted prefix sharing — the admission-path deep copy is gone** — `SeqKv::clone_prefix` copied the entire shared prefix into the new sequence on **every** admission: ~350 MB for a 2 k-token shared system prompt at 175.5 KiB/token, which is the exact workload the prefix cache exists to serve, so the cache's hit path paid a memcpy proportional to its own benefit. `LayerKv` now holds an immutable `Arc` prefix plus a private tail; a `clone_prefix` whose depth falls inside an already-frozen prefix is a refcount bump and copies nothing, and the prefix cache's entries are themselves `clone_prefix` results, so every hit takes that path. A shallower match reuses the same allocation at a narrower view rather than re-copying. Bit-identical by construction — same bytes, same order, never written — and asserted by `a_shared_prefix_produces_the_same_bits_as_a_private_copy` on **both** cores. The speculative rewind can land *inside* a shared prefix, so `truncate` narrows this cache's view instead of the buffer; truncating the buffer would silently shorten every concurrent sequence seeded from it, corruption nothing downstream would catch (`rewinding_into_a_shared_prefix_leaves_other_holders_intact`). **`COLI_KV_BUDGET_MB` had to learn this in the same change** or it would cancel it: charging a shared prompt to every viewer would refuse admissions over RAM that was never allocated, so `resident_kv` counts each private tail plus one charge per distinct allocation, identified by the allocation itself `attention.rs`, `model.rs`, `batch.rs`
- [ ] 🟡 **int2 expert storage — converted; the quality gate is running** (2026-08-08). `pack::quant_i2` is the format's **first producer**. int2 had been fully consumable since M1 (container detector, scalar + AVX2 dot kernels, dequantizer, CUDA `row_bytes`/`weight_at`) and completely unproducible, so no checkpoint ever used it. Four 2-bit fields per byte, biased `+2`, verified against the decoders' own bit layout and against `QtInfo::detect` inferring fmt 3 from byte count alone. **The conversion that was blocking this was started on 2026-08-08 at `--target int2-g64`, not `int2`** — per-row int2's `amax / 1` convention makes the `-2` level unreachable (it would need `|w| ≥ 1.5·amax`, impossible when `amax` *is* the row maximum), so the per-row format is effectively ternary and `quant_i2_g64`'s scale + zero-point per 64 values is strictly better. Measured plan on the real GLM-5.2 checkpoint: **118 478 tensors, 58 368 requantized, 383.73 GB → 286.29 GB (74.6 %)** across 58 shards — *not* the "exactly halves" the original entry projected from the weight plane alone, because per-group scales and the copied-through non-expert tensors are real bytes. Output at `/mnt/models/GLM-5.2-int2g64` (root is at 98 %, so it had nowhere else to go), which is USB 2.0 — the job wrote at a measured ~12.5 MB/s and took **~6.5 h**. **It was killed once at 76 GB / 16 shards** when the harness tore down its process group, and restarted detached (`setsid nohup`, log at `int2g64-convert.log`) — the partial output was removed rather than resumed, because the converter is a one-shot stream and 16 shards with no index is not a container, just 76 GB that looks like one.

  **The conversion has since finished** and this entry's own gate — "stays 🟡 until that line appears" — is satisfied: the log's final line reports 58 shards at 286.29 GB, and `.requant-progress.json` reads `tensors_done: 118478, shards: 58`, which the converter writes only after `requantize()` returns `Ok`. `du -sh` gives 267 GiB, which *is* 286.7 GB — the two figures differ by units, not by a truncated run.

  **What remains is the flip rate, and the thing that was actually missing was a runner, not a cable.** `peregrine-requantize` has always ended by telling the operator to measure `prediction_flip_rate`, and nothing in the tree could: `requant.rs` explains that the tool links `peregrine-core` alone on purpose and that the gate "belongs in the engine binary instead", where it had never been wired. It is wired now — `peregrine flip-rate <source> <candidate> [--text FILE] [--tokens N]`, loading the two containers **one at a time** so the slower one is not measured while the faster one evicts it from the page cache. Validated three ways before being pointed at the real checkpoint: a container against itself flips **0.000** (determinism), an int4→int4 no-op requantize flips **0.000** (a lossless conversion reads as lossless), and int4→per-row-int2 on the same fixture flips **0.727** (it detects real loss). The earlier "hardware-blocked, ~2.3 minutes a token" reading is corrected under "What is actually left" at the top of this file: that is the decode cost, the gate runs under teacher forcing, and the routed union saturates. This entry stays 🟡 until the number is in this file `pack.rs`, `requant.rs`, `engine/main.rs`

- [x] ✅ **int3-g64 (fmt 5) support** — peregrine could not load a container colibrì wrote with `--xbits 3`: `QtFmt` stopped at 4 and anything else was a hard `Unknown` error. Now a first-class format end to end — detector (scale *cardinality* is part of the discriminator, since a narrow tensor's int3 size can collide with a row format's), `QtView`/`QtWeight` dequant, `dot_i3i8_g64`, `pack::quant_i3_g64`, and a `--target int3-g64` converter scheme. 3.5 bits/weight effective: 12.7% smaller than int4 once the per-group scales are counted, not the 25% the weight plane alone suggests. colibrì has no int8-activation path for it ("int3 has no IDOT in v1: int8 activations don't compose with per-group accumulation without a kernel restructure") — `dot_i4i8_grouped` already *is* that restructure here, so int3 composed with it for free. **Byte-identical to colibrì's encoder**, frozen against a vector that engine produced; that fixture caught a rounding mismatch (`f32::round` vs `np.rint`, halves away from zero vs to even) which a round-trip test structurally cannot see `pack.rs`, `qt.rs`, `idot.rs`, `weight.rs`
- [x] ✅ **Heat-tiered on-disk precision** — `peregrine-requantize --tier-hot-frac <f> --tier-hot <scheme>` keeps the hottest fraction of each layer's experts (ranked from the source's `route_stats.json`) at one precision and the rest at another, in one container. Confirmed to need **no loader change**: `QtInfo::detect` is per-tensor, and a mixed container loads and generates. Refuses without routing data rather than tiering everything cold, which would look like it worked; ties and all-zero (untraced) layers go cold, so nothing is promoted on no evidence. **The tool reports the real trade before you spend hours on it**: tiering saves disk by expert *count* but per-token bytes by routing *frequency*, and on a skewed distribution keeping the hot 25% at int4 measured **140% of an all-int2 checkpoint** — the hedge costs 40% of the byte saving. `assign_tiers`' scalar `bytes_per_expert` was left assuming uniformity by this change and is now a per-expert closure — see the VRAM-planner entry below `requant.rs`
- [x] ✅ **The VRAM-planner half of heat-tiered precision** (2026-08-06) — the converter half shipped in the entry above; what remained was that **both** planners sized experts uniformly, which a tiered container makes false. `assign_tiers` took a scalar `bytes_per_expert` documented as safe because "int4 experts are same-shaped" — true until `--tier-hot-frac` began emitting int4 and int2 experts inside one layer, differing ~40% in size. It now takes `bytes_of(expert)`, mirroring `solve_residency_sized`'s existing closure, and the galactic caller passes `Model::expert_bytes_on_disk(layer, e)` **per expert** instead of probing `(first_dense, 0)` once. `tiers_size_each_expert_from_the_container_not_from_a_probe` runs the same graph, heat and budget both ways and requires **different** placements, so the closure cannot quietly stop being read. The VRAM side had the mirror defect: `build_with` passed `|_, _| bytes_per_expert` into the closure-taking solver, and since `experts_are_per_row_int4` deliberately answers "no" for a mixed container, every expert — including the int4 majority — was budgeted at the f32 size, giving a tier ~8× smaller than the VRAM allowed. Split into a per-expert `expert_is_per_row_int4` probe; a uniform container plans identically `tools/lib.rs`, `gpu.rs`, `engine/main.rs`

---

## ❄️ Deprioritized / Noted (excluded from %)

- [ ] ~~Fast matrix multiplication (Strassen, Coppersmith–Winograd, Williams')~~ — asymptotically cheaper but almost never wins for LLM inference. **Skip unless proven otherwise.** (Confirmed absent — custom tiled GEMM used instead.)

- [ ] ~~Compressed VRAM residency ("zram for VRAM" — nvCOMP/LZ4 block compression of resident experts, decompressed to scratch before dispatch)~~ — **closed on measurement.** The payload is packed int4 nibbles whose only compressible structure is nibble-value skew: measured **1.18x** (repo fixture) to **1.26x** (Gaussian weights) against an order-0 entropy ceiling of ~1.27x, with store-only at exactly 1.000x. All of the win is entropy coding, so LZ4/Snappy — most of what nvCOMP offers — would return ~1.0x, and a trained dictionary has no repetition to exploit.

  The analogy also inverts. zram pays off because decompressing in RAM (~5 GB/s) beats the swap device it replaces (~0.5 GB/s SSD). In VRAM the roles swap: the "swap device" is host RAM over PCIe at ~25 GB/s while VRAM itself runs 500–1000 GB/s, so any decompressor feeding a matmul is **slower than reading the uncompressed weights**. Kernels also read raw device pointers out of `GroupDesc` with no lazy-decode hook (`backend_cuda.cu:639-647`), so decode would have to run before the dispatch ladder into scratch — spending on scratch the capacity it just saved, against the design rule at `safetensors.rs:312-317` that the fast path hands raw bytes to kernels with no decompress step.

  What the question was really reaching for is **precision as compression**, which the ladder in §3 delivers: an f32→int4 residency step is an exact 8x with no decoder at all. **Skip unless the payload stops being quantized.**

---

## 🔧 Env-var reference (new & existing gates)

| Env var | Default | Effect |
|---|---|---|
| `COLI_HUGEPAGE` | on | `MADV_HUGEPAGE` on every ≥ 2 MB allocation |
| `COLI_FADVISE_MAIN` | on | `POSIX_FADV_WILLNEED` batched before each main-path read |
| `COLI_FADVISE_DROP` | off | `POSIX_FADV_DONTNEED` after each streamed read (RSS-bounded) |
| `COLI_IO_TUNE` | on | Adaptive `set_iowq_max_workers` from `IoTuner` recommendation |
| `COLI_IO_RECOVERY` | on | Per-region retry ladder on batched-read failure |
| `COLI_BATCH_SLA_MS` | unset | Shrink working batch cap on p95-latency overrun |
| `COLI_ADAPTIVE_WINDOW` | 1 | Run prefill every Nth engine tick (decode-heavy window) |
| `COLI_LANE_BALANCE` | off | `LaneBalancer` overrides static residency decision |
| `COLI_REPLICATE_K` | 0 | Top-K hot GPU-residents also warmed into the CPU warm cache |
| `COLI_NUMA_PIN` | off | Pin par-pool + prefetch-pool workers round-robin across NUMA-node CPUs |
| `COLI_CACHE_ADMIT_MIN_HEAT` | 0 | Warm-cache admission gate: cache an expert only at ≥ N routings |
| `COLI_CACHE_COMPRESS_IDLE` | off | Background recompression of cold cache slots while the engine is idle |
| `COLI_PREFETCH_WARM_PATHS_<CLASS>` | unset | Per-workload-class prefetch breadth override (CODE/JSON/MATH/PROSE/MIXED) |
| `COLI_CACHE_COMPRESS` | off | Zstd-compress WarmCache slabs on admit, decode on hit |
| `COLI_CACHE_NEGATIVE_TTL` | 0 | Evict unhit warm-cache slots older than N clock ticks |
| `COLI_CACHE_LFRU` | off | Warm-cache victims ranked by `tier::lfru_score` (`heat<<8\|recency`) within a priority class instead of by recency alone. Frequency is per-slot hit count, so it works without a GPU heat table |
| `COLI_ROUTE_STATS_PERSIST` | on | Save `route_stats.json` at Drop; auto-load matching one on `Model::load` |
| `COLI_LAYOUT_SCHEDULE` | on | Use `<dir>/schedule.json` (if present) to pre-sort disk reads |
| `COLI_PHASE_THRESHOLD` | 0.6 | Jaccard distance above which `PhaseTracker` declares a phase change |
| `COLI_ROUTER_LOOKAHEAD` / `_N` | on / 6 | Router look-ahead: prefetch layer L+1's experts by running its own router on layer L's output. Decode-only, output-neutral |
| `COLI_PREDICT_EVAL` / `_N` | off / topk | Score every predictor against the real routing; `[predict-eval]` recall + precision-by-rank at shutdown |
| `COLI_MLOCK` | off | `mlockall(MCL_CURRENT)` after load, before the cache fills — wires the trunk, leaves the cache reclaimable |
| `COLI_MEMO_ENTRIES` / `COLI_MEMO_MB` | 32 / 64 | Exact response memo in `peregrine-serve`; greedy requests only, keyed on full request semantics |
| `COLI_PERF_COUNTERS` | off | Open a real `perf_event_open` LLC-miss counter (needs kernel grant) |
| `COLI_PERF_PREFETCH_FEEDBACK` | off | Let the LLC-miss delta steer the prefetch distance (rising → widen). Needs `COLI_PERF_COUNTERS` too; direction is an untested hypothesis, deliberately a second opt-in |
| `COLI_THERMAL_LIMIT_C` | unset | Thermal governor: shrink workers above this package temperature |
| `COLI_POWER_CAP_W` | unset | Power governor: shrink workers when RAPL watts exceed the cap |
| `COLI_BW_GOVERNOR` | off | Memory-bandwidth governor: shrink workers on GB/s plateau |
| `COLI_ENTROPY_ADAPT` | off | Routing-entropy-adaptive prefetch breadth (needs the distance tuner) |
| `COLI_FUSE_THRESHOLD` | 0.9 | Co-firing rate above which expert pairs are fused (kept adjacent) |
| `COLI_HYPER_SCHED` | off | Hyperedge (co-activation component) grouping of dispatch order |
| `COLI_LEARN_SCHED` | off | ε-greedy bandit over knob configurations (persisted policy) |
| `COLI_RL_SCHED` | off | Tabular Q-learning scheduler (bandit wins if both set) |
| `COLI_TIER_VRAM_MB` / `_RAM_MB` | unset | Galactic pass: tier byte budgets → emit `tiers.json` |
| `COLI_TIER_SEED` | on | Prefetch-warm the planned RAM tier at model load |
| `COLI_SHAPE_SPECIALIZE` | off | Per-shape probe-then-memoize matmul dispatch |
| `COLI_DEBUG` | off | Surface advisory-operation failures (hints, pinning, persistence) on stderr |
| `COLI_GPU_F32_FRAC` | unset | Adaptive per-expert precision: hottest fraction of residents promoted to f32 (cuda) |
| `COLI_PCIE_BUDGET_MB` | unlimited | Cap on bytes one `reheat` generation may upload across PCIe; defers the coldest to the next generation (cuda) |
| `COLI_PREFILL_CHUNK_DIV` | 0 (fixed 64) | Grow the prefill chunk as `pos/d` (floor 64), making KV reconstruction linear instead of quadratic. Output-neutral |
| `COLI_GATE_STATS` | off | Tally how much of the routed set carries a negligible gate share; reported as `[gate]` at shutdown |
| `COLI_UNION_STATS` | off | Tally batch-union sharing (`selections/distinct`); reported as `[union]` at shutdown |
| `COLI_MOE_ENGINE` | `concurrent` | MoE expert dispatch: `concurrent` (the 3-lane default) or `sched` (`peregrine-sched::moe_streamed`, the 2-lane ancestor — no GPU lane, no warm cache, no prefetch). Installed by the binaries, since `peregrine-sched` depends on `peregrine-model` and the reverse would be a dependency cycle |
| `COLI_CUDA_GRAPH` | off | Capture each `expert_group` launch shape into a CUDA graph and replay it. Bit-identical by construction; off by default because a stale baked device pointer fails silently. `/metrics` reports captures/replays/invalidations/uncacheable (cuda) |
| `COLI_CUDA_FUSED_REDUCE` | off | Fuse the layer-level gate-weighted reduce onto the device: D2H carries `s_n` rows instead of `Σrows`. **Moves the GPU arm's low bits** (order, not content); CSR-ordered so it is stable run to run (cuda) |
| `COLI_CUDA_AUTOTUNE` | off | Online WMMA tile selection for the `COLI_CUDA_TC_W4A16` arm, persisted to `kernel_tuning.json`. A second opt-in on top of that arm's own gate (cuda) |
| `COLI_GPU_TIER_SWAP` | `replan` | VRAM residency policy: `replan` (historical whole-set re-rank) or `lfru`/`freq` (one swap per layer per generation, with `tier.rs`'s 25 %+4 hysteresis) (cuda) |
| `COLI_DRAFT_SAMPLED` | off | Extend `COLI_DRAFT` to temperature > 0 via rejection sampling. Emits the request's exact **distribution**, not its exact **sequence** — two uniforms per draft vs one per token, so a seeded request is reproducible against itself only |
| `COLI_FORCE_ASYNC` | on | Force `IOSQE_ASYNC` on buffered reads. `0` runs them inline — worth trying where completion is fast enough that the io-wq hand-off is pure overhead |
| `COLI_PREDICT_SOURCE` | unset | Force a prefetch predictor (`momentum` / `phase-aware`). Load already picks the strongest the artifacts allow, so this selects a **weaker** arm — what `COLI_PREDICT_EVAL`'s scoreboard needs to be actionable |
| `COLI_IO_ENGINE` | `uring` | Streaming read engine: `uring`, `pread` (blocking, N threads), or `regbuf` (registered fixed buffers). Byte-identical output |
| `COLI_IO_THREADS` | workers | Threads for the `pread` engine |
| `COLI_REGBUF_SLOTS` | 16 | Registered-buffer count for the `regbuf` engine (pinned pages — see `RLIMIT_MEMLOCK`) |
| `COLI_KV_BUDGET_MB` | 0 (off) | Resident-KV byte ceiling for admission, alongside the `--max-batch` count. Charges each private tail plus one per distinct shared prefix |
| `COLI_KV_POOL_MB` | 0 (off) | Cross-sequence recycler for KV latent allocations: a retired sequence's buffers go to the next admission instead of the allocator. **Additive to `COLI_KV_BUDGET_MB`**, not inside it — total KV RSS ≈ budget + pool. Output-neutral (recycled buffers are cleared; readers see `[0, len)`) |
| `COLI_KV_DTYPE` | `f32` | Element type for stored KV latents (`f32`/`f16`). f16 halves resident KV; **changes token values**. Pair with `COLI_MLA_ABSORB` |
| `COLI_DSA` | off | Run the DSA lightning indexer where the checkpoint carries one; each query attends only the top `index_topk` cached keys. **Changes token values**. Single-sequence path only |
| `COLI_DRAFT` | 0 (off) | MTP speculative-decode draft depth (`--draft N`); greedy, so output-identical |
| `COLI_PREFIX_CACHE_MB` | 0 (off) | Cross-request KV prefix cache budget; seeds a new request from the longest cached prefix of its prompt |
| `COLI_ROUTE_MIN_SHARE` | 0 (off) | Drop trailing routed experts below this share of a position's gate mass. **The only knob that changes token values** |

---

## Notes

- **Audit basis:** statuses verified against source; file:line evidence inline. `Done` = actually implemented and covered by a test — bit-identical or round-trip for everything except the one deliberately lossy knob (`COLI_ROUTE_MIN_SHARE`), which is gated by a bounded prediction flip rate instead. `Partial` = scaffolding present but not yet on the hot path.
- **Documentation:** the full docs wiki is [`docs/`](docs/README.md) (2026-07-30) — user guides (CLI, HTTP API, configuration, model format) and subsystem deep dives; this file remains the per-item engineering audit.
- **Compilation & test invariant:** **588** tests pass workspace-wide, up 11 this pass: `flip_rate_gate.rs` pins the lossy-container gate in both directions (a gate stuck at 0.000 is indistinguishable from a lossless conversion), and `peregrine-serve`'s `main.rs` gained a **first** test module — until 2026-08-08 the entire HTTP tool-calling surface, prompt assembly included, was reachable only through a running server with a loaded model, clippy clean on everything tracked, `--strict` bad-patterns audit green on tracked files, `[R]` at **21** — `prediction_flip_rate` left the list when `peregrine flip-rate` gave it a production caller, which is the point of the `[R]` pass: a quality gate nothing can invoke is not a gate. The `peregrine-cuda --features cuda` suite is **17 passing and, since the 2026-08-08 14:08 reboot, actually executing** — driver 610.57.04 on both sides, `init(&[0])` returns 1, and the GPU-gated tests no longer return early. It went 16 → 17 with `every_w4a16_tile_instantiation_agrees_with_the_default`; the run that first executed them found one vacuous test and one real bug (§0). Keep reading a green CUDA suite with suspicion anyway: **these tests self-skip on `init(&[0]) < 1` and report `ok`**, so the next driver upgrade without a reboot makes the suite green and meaningless again. `nvidia-smi` before believing it.
- **What's left is *not* all CUDA-shaped** (it used to be, and then it was over-corrected). As of 2026-08-08: five items need hardware this box lacks, three are CUDA work it could do, one is open by choice, and one is wall clock. See the four-way split under the dashboard.
- **Validation caveat:** synthetic-model tests catch correctness; throughput impact needs a real model to measure — and now partly has been, in the [2026-08-01 pass](docs/benchmarks.md#benchmark-pass--2026-08-01-post-improvement-re-measure) (§1–§11 knob bundle 1.004×, CUDA lane 1.09×, batching 4.4× reproduced). Note that pass also found `peregrine bench 1 4 16` unsound for comparing configurations: running every batch point in one process inflates the later ones (baseline B=16 reads 0.143 in-sweep vs 0.224 isolated), which had manufactured an apparent 1.45×/1.64× that vanished under isolation. Use one batch size per fresh process. The pattern is "many small adaptive knobs, each bit-identical when off" — evaluate combined. Two knobs now need a *real checkpoint* to size at all: `COLI_ROUTE_MIN_SHARE` (how much gate mass the tail actually carries) and any int2 use (its accuracy cost). `COLI_GATE_STATS` and `prediction_flip_rate` exist to answer exactly those, and both are unanswerable on the synthetic model — 4 experts, top-2, so there is no weight tail to measure.

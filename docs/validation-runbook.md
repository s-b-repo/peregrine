[« Docs index](README.md)

# Validation runbook (GPU box)

Everything this workspace could not measure, in the order worth measuring it.

The development laptop cannot load GLM-5.2 at all — the dense set is 10.59 GB
against ~2 GB free — so several shipped features carry no real-checkpoint
evidence, and one published conclusion was measured against a defect that has
since been fixed. This page is the procedure for settling them on a machine with
≥ 16 GB RAM, an NVIDIA GPU, and the 349 GB int4 container.

**Rule for every step: write down what you measured, including when it disagrees
with what this repo currently claims.** Several entries below exist because a
number went unchallenged.

## 0. Setup

```bash
export COLI_MODEL=/path/to/glm52_i4       # the runners default to another box's path
cargo build --release --bins              # CPU binaries
cargo build --release --features cuda -p peregrine-engine
# scripts/bench-arms.sh expects the GPU binary here; otherwise pass BIN_GPU=
mkdir -p target/cuda/release && cp target/release/peregrine target/cuda/release/peregrine
```

Confirm the model actually loads before benchmarking anything:

```bash
echo "" | ./target/release/peregrine "$COLI_MODEL"    # expect [ram] line, then READY
```

The `[ram]` line is the pre-load projection. **Record it** — step 5 compares it
against measured RSS, which is the only way to know whether the projection is
trustworthy.

## 1. Does the O_DIRECT fix hold on real hardware?

The lane submitted one region at a time until 2026-08-02, so direct reads were
*slower* than buffered. Batching them measured 1.2–1.3× on a LUKS laptop; the
open question is whether direct now overtakes buffered on an uncontended NVMe.

```bash
SH=$COLI_MODEL/out-00060.safetensors
cargo run --release -p peregrine-io --example iobench -- $SH 64 4 8 1   # O_DIRECT
cargo run --release -p peregrine-io --example iobench -- $SH 64 4 8 0   # buffered
# colibrì's harness for scale (8 threads of blocking pread, not io_uring)
make -C /path/to/colibri/c iobench && /path/to/colibri/c/iobench $SH 64 16 8 1
```

Use a **different shard per run** — the page cache will otherwise flatter the
buffered arm.

### 1a. The dm-crypt hypothesis is now testable end to end

The pread engine this called for exists. `iobench` takes a 6th argument
selecting the engine, and it is the *same* three `concurrent.rs::read_regions`
dispatches on, so a result here transfers to the streaming lane:

```bash
for E in uring pread regbuf; do
  cargo run --release -p peregrine-io --example iobench -- $SH 64 8 8 1 $E
done
# and end to end, where it actually matters (output is byte-identical, so this
# is purely a rate comparison):
COLI_IO_ENGINE=pread COLI_IO_THREADS=8 scripts/bench-b1-isolate.sh out 3
```

**What was learned on the dev box, and what was not.** The three engines were
run against buffered reads of large system libraries on a LUKS volume. The
result was *inconclusive* — `pread` led on one pair (2.02 vs 1.68 GB/s) and
trailed on another (1.16 vs 1.26) — because the files differ in size and
page-cache state, so it was never a controlled comparison. It does not
reproduce, and is not evidence either way. The real test needs the model
shards, O_DIRECT, and a cold cache.

**`regbuf` has a hard operational limit worth knowing before you plan around
it.** Registered buffers are **pinned** pages, charged against
`RLIMIT_MEMLOCK` — 8 MB by default on most distros. A pool sized for real
~6 MB expert regions (16 slots = 96 MB) fails registration outright with
`ENOMEM`, which reads as "out of memory" but means "out of *lockable* memory".
The engine falls back to the plain submit with an advisory line rather than
failing the read; to actually exercise it, raise the limit:

```bash
ulimit -l unlimited        # or: systemd-run -p LimitMEMLOCK=infinity ...
```

Even then, weigh it before adopting: `read_fixed_many` **copies out** of the
registered buffer where `read_many` has the kernel write the destination
directly. At ~6 MB regions that memcpy plausibly costs more than the
page-pinning it saves — the published gains for fixed buffers are at 4–64 KB.
Treat `regbuf` as a measurement, not a default.

### 1b. Before blaming the engine, check the storage configuration

Two settings dominate anything the I/O lane can do, and neither is code:

```bash
sudo cryptsetup luksDump /dev/<model-volume> | grep -i sector   # want 4096
cat /sys/block/nvme*/queue/scheduler                            # want [none]
```

A LUKS volume formatted with the 512-byte default sector size measured ~10% of
raw throughput on a high-bandwidth array; `--sector-size 4096` alone restored it
to ~50% (≈5×). It is set at `luksFormat` time, so changing it means
`cryptsetup reencrypt`. **Skip** `--perf-no_read_workqueue`: it cuts CPU to near
zero but did not move throughput. Linux I/O schedulers cost 14–57% versus `none`
on NVMe.

## 2. Re-run the 2026-08-01 knob pass — its conclusion is suspect

The [2026-08-01 pass](benchmarks.md#benchmark-pass--2026-08-01-post-improvement-re-measure) measured nine adaptive knobs
together at **1.004×** with byte-identical disk reads, and that result is the
foundation of `todo.md` §13's "moving bytes faster is spent" thesis. Two of the
nine were not what they seemed:

- `COLI_REGBUF` is inert — that document footnotes it.
- `COLI_DIRECT` selected the depth-1 lane fixed in step 1. **Nothing records
  this**, because the defect was found afterwards.

So the bundle was eight live knobs, one no-op, and one crippled.

```bash
scripts/bench-arms.sh out/rerun-$(date +%F) baseline improved gpu
```

Run against a **freshly built binary from this tree** — an older one reproduces
the old number and proves nothing. If `improved` still lands at ~1.00×, the
thesis survives with better evidence than it had. If it moves, §13's premise and
the roadmap's ordering both need revisiting.

## 3. Quality of everything lossy

Three features ship default-off precisely because their end-to-end cost could
not be measured here. Each needs a flip rate against the int4 container over a
fixed prompt set, via `Model::prediction_flip_rate`.

```bash
# int2: half the expert bytes (measured 2.69 GB -> 1.35 GB on one real shard)
./target/release/peregrine-requantize "$COLI_MODEL" /path/glm52_i2 --target int2 --dry-run
./target/release/peregrine-requantize "$COLI_MODEL" /path/glm52_i2 --target int2
# int3-g64: 12.7% fewer bytes, better-conditioned than per-row int4
./target/release/peregrine-requantize "$COLI_MODEL" /path/glm52_i3 --target int3-g64
# int2-g64: affine 2-bit, 3.0 bits/weight once the two f32 per group are counted
# (~25% under int4, not the ~50% the payload width implies), and the format whose
# accuracy cost is least known. Unlike per-row int2 it uses all four levels and
# scales per 64 values, so it should land well ahead of `--target int2`; measure
# both rather than assuming.
./target/release/peregrine-requantize "$COLI_MODEL" /path/glm52_i2g --target int2-g64
```

`--dry-run` first: it predicts output size from headers alone and is exact (it
matched the real run to the byte on a GLM-5.2 shard). Both containers exist at
once, so check free space against its figure.

Then compare each against the int4 source on the same prompts, and measure
`COLI_MLA_ABSORB=1` the same way. Absorb is the one where a *bad* result is most
likely: it is algebraically equal to the dense path but not numerically, and on
the synthetic model end-to-end logit divergence reached 2.6 absolute.

**Expect the ordering int4 > int3-g64 > int2-g64 on quality and the reverse on
size, and check it rather than trusting it.** peregrine's int4 is a *QAT*
baseline (GLM-5's int4 was applied during SFT), so every post-training scheme
here is fitting a target that was already trained at 4 bits — the usual
"PTQ vs fp16" deltas understate the damage. Also note the two independent
measurements that byte reduction converts to wall-clock at only 7–24% on
memory-bound systems; peregrine's disk share is larger so it should do better,
but treat any byte saving as an upper bound on the speedup.

Record flip rates next to the byte savings. A halved working set at an
unacceptable flip rate is not a win, and the point of measuring is to be able to
say which it is.

## 4. The CUDA lane — never exercised

No GPU has ever run this code.

```bash
cargo test -p peregrine-cuda --features cuda      # incl. graph capture/replay
scripts/bench-arms.sh out/gpu-$(date +%F) gpu
```

Two specifics worth checking beyond "does it run":

- **int3 on the GPU tier.** `gpu.rs` uploads raw only when all three projections
  are per-row int4; int3 falls to `dequant()` and uploads f32 — 8× the budgeted
  bytes. The VRAM knapsack sizes residents from a *single* `bytes_per_expert`, so
  an int3 container with `COLI_GPU_INT4=1` may plan N experts and upload 8N worth.
  Expect eviction thrash or CUDA OOM; confirm before trusting a GPU int3 run.
- **`build.rs` succeeds without `nvcc`**, so nothing in this repo has ever
  syntax-checked the `.cu` file against a real toolchain.

## 5. Decode throughput and peak RSS

The row [benchmarks.md](benchmarks.md) cannot fill for *either* engine.

```bash
COLI_BENCH_STEPS=3 ./target/release/peregrine bench 1 4 16
/usr/bin/time -v ./target/release/peregrine "$COLI_MODEL" < prompts.txt
```

Peak RSS is the number that matters most here, because it grades the pre-load
projection from step 0. If measured RSS materially exceeds the projected peak,
the projection's slack model is wrong and the **runtime RSS guard**
(`COLI_RSS_GUARD_GB`) is doing load-bearing work rather than acting as a
backstop — which is exactly the drift colibrì hit (74.4 GB projected, 115.6 GB
actual, three kernel kills). Watch for the guard's `[ram] RSS … over the … budget`
line; if it fires often, fix the projection rather than leaning on the guard.

### 3a. What does f16 KV actually cost?

The byte saving is exact and needs no measurement: `COLI_KV_DTYPE=f16` halves
resident KV, asserted as an equality in the suite. The **accuracy** cost is the
open question, and the synthetic model says something specific enough to be
worth confirming or refuting on a real checkpoint:

| core | measured relative error, tiny model | mechanism |
|---|---|---|
| absorb (`COLI_MLA_ABSORB=1`) | 1.8e-4 | f16's own precision — the latent is dotted in f32 |
| dense (default) | 1.7e-2 | `kv_b.apply_vec` quantizes activations to int8 at `amax / 127`; a perturbation that moves the row maximum rescales the whole grid |

Two orders of magnitude, and the cause is the int8 activation path, not f16. So
measure **four** arms, not two — the interaction is the finding:

```bash
for kv in f32 f16; do for absorb in 0 1; do
  COLI_KV_DTYPE=$kv COLI_MLA_ABSORB=$absorb \
    ./target/release/peregrine "$COLI_MODEL" < prompts.txt > out.$kv.$absorb
done; done
```

Then `prediction_flip_rate` each against the `f32`/`absorb=0` reference. If the
gap does not reproduce, the tiny model's 8-wide latent was amplifying and f16 is
cheaper than it looks; if it does, `COLI_MLA_ABSORB` stops being an independent
knob and becomes a prerequisite. Either answer changes the recommendation, which
is why this is measured rather than asserted.

Note `COLI_MLA_ABSORB` is itself listed as unvalidated on a real checkpoint, so
this run grades both at once.

### 3b. DSA: the one item whose payoff grows with context

Everything else in this runbook is measured at a fixed prompt length. DSA is
not: it caps attention work at `index_topk` keys, so its benefit is zero at
short context and grows without bound. Measure it as a **curve**, not a point.

The dev box cannot produce it — the laptop-converted 349 GB container was built
without `--indexer`, so `COLI_DSA` is inert there and the synthetic fixture's
6-token prompts say nothing about a 2048-key threshold. Two prerequisites:

```bash
# 1. a container that actually carries indexer tensors
grep -o 'indexer_projections' "$COLI_MODEL"/*.json | head    # none => reconvert with --indexer
# 2. prompts long enough to cross index_topk (check config.json; GLM-5.2 ships 2048)
```

Then sweep prompt length across the threshold, dense vs sparse:

```bash
for n in 512 1024 2048 4096 8192; do
  for dsa in 0 1; do
    COLI_DSA=$dsa ./target/release/peregrine "$COLI_MODEL" < prompt.$n.txt
  done
done
```

What to look for, in order of what would change the recommendation:

- **Below `index_topk` the two arms must be bit-identical.** If they are not,
  the activation rule is wrong and everything above is untrustworthy.
- **Above it, where does the curve cross?** Scoring every cached key is itself
  `O(context)`, so DSA trades a cheap linear pass for an expensive one. The
  crossover point is the whole result; a single long-prompt number hides it.
- **`prediction_flip_rate` at each length.** Selection drops real attention
  mass, and the dropped fraction grows with context — this is the one lossy
  knob whose accuracy cost is *not* constant across the sweep.

Note the batched serving engine runs the absorb core, which has no sparse form,
so a `peregrine-serve` run sees DSA during prefill only. Compare against
`peregrine` (stdio) before concluding the effect is small.

### 5b. Is the remaining KV fragmentation worth a block pool?

`KvBuf::reserve_for` capped the growth overshoot at 256 rows; what is left is
allocator churn across sequences, and whether that is worth a block pool is a
measurement, not an argument. Run a long serve session and compare **peak RSS
against live KV**:

```bash
COLI_KV_BUDGET_MB=16384 ./target/release/peregrine-serve --max-batch 16 &
# drive it for an hour with varied prompt lengths, then:
grep VmHWM /proc/$(pgrep -f peregrine-serve)/status
```

`VmHWM` minus the sum of `SeqKv::bytes()` over live sequences is the headroom a
pool could reclaim. Two readings decide it:

- **If the gap is flat over the session**, it is allocator slack that a pool
  would just relocate — stop, and record the number so the item can be closed
  rather than left open forever.
- **If the gap grows** with a workload of mixed prompt lengths, that is real
  fragmentation from freeing 78 differently-sized allocations per sequence, and
  it is the case a block pool exists for.

Note the ceiling before spending on it: peregrine never pre-allocates, so it has
none of the reservation waste that is the largest bar in vLLM's 62–80% figure.
~33% is the honest upper bound here.

### 5a. Does prefix sharing show up in RSS and admission?

Refcounted prefixes and the KV byte budget are the two halves of one change, and
only a real checkpoint can show whether they add up. The synthetic tests prove
the bytes are identical and the allocation is counted once; what they cannot
show is the size of the win, because it scales with the *shared prompt*.

Drive N concurrent requests that share a long system prompt, with the prefix
cache on and the byte budget set below what N private copies would need:

```bash
COLI_PREFIX_CACHE_MB=8192 COLI_KV_BUDGET_MB=16384 \
  ./target/release/peregrine-serve --max-batch 16 &
# then N clients, same system prompt, different user turns
```

Two numbers, both from one run:

- **Peak RSS** against `N × prompt_tokens × 175.5 KiB`. Before sharing, the
  admission path copied the whole prefix per request; after, the prefix appears
  once, so RSS should track `1 ×` the prompt plus each sequence's generated
  tail. A peak still scaling with `N` means the requests are not matching the
  cache — check the prefix-cache hit counter before concluding sharing failed.
- **Admitted concurrency** at a budget that would have refused most of the
  batch. This is the half that is easy to get wrong in the *other* direction: if
  concurrency does not rise, confirm `resident_kv` is deduping rather than that
  the budget is simply large enough to be inert.

Also worth one run with `--draft 4`: speculative rewind is the path that
truncates *into* a shared prefix, and `rewinding_into_a_shared_prefix_leaves_other_holders_intact`
covers it only at synthetic scale.

## 6. Does fusing prefill into the decode forward actually save the bytes?

`COLI_FUSE_PREFILL` is the one item whose payoff is *entirely* an I/O saving:
the arithmetic is identical either way, asserted bit-for-bit, so a wall-clock
measurement alone cannot tell you whether it worked or whether the workload
simply had no mixed ticks.

Measure the bytes, not the clock. `COLI_UNION_STATS=1` reports how many routed
selections each distinct expert read served:

```bash
for f in 0 1; do
  COLI_FUSE_PREFILL=$f COLI_UNION_STATS=1 \
    ./target/release/peregrine-serve --max-batch 16 &
  # drive it with a mix: several short decoding requests plus a long new prompt,
  # so ticks actually contain both. Then read the [union] line at shutdown.
done
```

What to look for:

- **`share` should rise with fusion on.** A prefill chunk's rows and the decode
  rows now union into one read set. If it does not move, the workload never
  produced a mixed tick — add a long prompt while short ones are decoding, and
  check the prefill is actually chunked (`COLI_PREFILL_CHUNK_DIV`, prompt
  longer than 64 tokens).
- **Token streams must be identical between the two arms.** They are asserted
  identical on the synthetic model; confirm it once on the real one, because
  that is the property the whole item rests on.
- **Watch decode latency, not just throughput.** Fusion makes one tick do more
  work, so a decode step now waits on a chunk it previously ran beside. If
  `COLI_BATCH_SLA_MS` starts shrinking the working cap, the chunk is too large
  — that is what `COLI_PREFILL_CHUNK_DIV` is for.

## 7. Is a pruned checkpoint worth its quality cost?

`peregrine-prune` is the only item here whose output is a *different model*,
so it is the one where "did it work" is entirely a quality question. Prune at
the default 25% and measure before considering more:

```bash
peregrine dump-routes "$COLI_MODEL" > routes.json    # the workload you serve
./target/release/peregrine-prune "$COLI_MODEL" --trace routes.json --dry-run
./target/release/peregrine-prune "$COLI_MODEL" /path/glm52_p25 --trace routes.json
```

Three measurements, in this order:

- **`prediction_flip_rate` against the source**, on the traced workload *and*
  on one the trace did not cover. The second is the one that matters: the
  published failure mode is a model that holds up on the calibration
  distribution and collapses off it.
- **Working-set size, not per-token bytes.** Compare peak RSS and warm-cache
  hit rate. Per-token bytes will not move — top-k is unchanged — and a
  benchmark that reports throughput alone cannot tell you whether pruning did
  anything at all.
- **The aggregate-fallback count** the tool prints. If more than the MTP head
  fell back, the trace was too short and the plan is partly guesswork; extend
  it and re-run before drawing conclusions.

Only if 25% is clean is 50% (`--force`) worth trying, and then only with the
off-distribution flip rate as the acceptance gate rather than a benchmark
average.

## 8. Expert-skip bounds: confirm or close the negative result

The offline prototype already ran here and came back negative — the weight
bound adds ~0.12 points of skippable reads over the gate weight alone. That is
a synthetic-model result with 4 random-weight experts, so `C_e` is near-uniform
by construction; a real checkpoint is the only thing that can overturn it.

```bash
peregrine dump-routes "$COLI_MODEL" > routes.json
./target/release/peregrine-skipbound "$COLI_MODEL" --trace routes.json
```

Read the **"the bound adds"** column, not the "with bound" one. The engine can
already drop low-gate experts through `COLI_ROUTE_MIN_SHARE`; the only thing
that justifies a sidecar, per-token norm arithmetic and a new file format is
the margin over that baseline.

- **Margin still under ~1 point** → close the item. Size `COLI_ROUTE_MIN_SHARE`
  with `COLI_GATE_STATS` instead; it is the same saving with none of the
  machinery.
- **Margin materially larger on 256 experts** → then `C_e` does spread on a
  real checkpoint, and the next step is *still* not the read path: re-measure
  on a second workload, since the fraction is a property of the routing
  distribution as much as of the weights.

## What to do with the results

Write them into [benchmarks.md](benchmarks.md). Where a measurement contradicts
existing prose, **correct the prose** rather than appending a newer number
beside it — this repo has already been bitten by two live figures for the same
quantity. If a number cannot be obtained, say so explicitly; an absent
measurement stated plainly is worth more than an implied one.

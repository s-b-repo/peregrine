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

[`benchmark-2026-08-01.md`](benchmark-2026-08-01.md) measured nine adaptive knobs
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

## What to do with the results

Write them into [benchmarks.md](benchmarks.md). Where a measurement contradicts
existing prose, **correct the prose** rather than appending a newer number
beside it — this repo has already been bitten by two live figures for the same
quantity. If a number cannot be obtained, say so explicitly; an absent
measurement stated plainly is worth more than an implied one.

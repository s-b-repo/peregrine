[« Docs index](README.md)

# Tools

The standalone binaries that are not the engine or the server. Offline layout
tools — `peregrine-layout-reorg` and `peregrine-prune` — have their own page:
[Layout tools](layout-tools.md).

| Binary | What it does | When you reach for it |
|---|---|---|
| [`peregrine-gen`](#peregrine-gen) | streaming client: watch generation live and time it | any time you want a decode number that means something |
| [`peregrine-requantize`](#peregrine-requantize) | rewrite a container at lower precision | when bytes-per-expert is the bottleneck |
| [`peregrine-skipbound`](#peregrine-skipbound) | measure whether reads could be skipped at all | before anyone touches the read path for skipping |
| [`bench-serve-arrivals.py`](#bench-serve-arrivalspy) | open-loop load: fixed request *rate*, not fixed concurrency | finding the saturation knee, and anything about queue time |

---

## `peregrine-gen`

A streaming client for `peregrine-serve`'s `POST /v1/chat/completions`. It prints
the completion as it arrives and reports the *shape* of the run — which `curl -N`
cannot, and which the batch harness reports only as one lumped `decode_s`.

```
peregrine-gen [options] [prompt ...]

  --host <h>          server host                    (default 127.0.0.1)
  --port <p>          server port                    (default 8137)
  --model <id>        model id sent in the request   (default glm-5.2)
  --max-tokens <n>    completion length cap          (default 64)
  --temperature <f>   sampling temperature           (default 0.0)
  --system <text>     prepend a system message
  --json <file>       write per-token timing records
  --quiet             no live status line
```

The prompt comes from the arguments, or from stdin when none are given.

```
$ peregrine-gen --max-tokens 4 "Explain how a mixture-of-experts layer routes a token."
**Mixture-of
── peregrine-gen ──────────────────────────────────────────────
  generated    4 tokens, 12 chars
  ttft         2m 29s  (prefill + first token)
  total        3m 42s
  decode rate  16.08 s/tok (0.062 tok/s)  (excludes ttft)
  inter-token  min 15.0s  p50 15.5s  p95 17.6s  max 17.6s
  finish       stop
```

While it waits, a status line ticks with a spinner, the running rate, and how long
the current token has been outstanding.

**Text goes to stdout, statistics to stderr.** So `peregrine-gen "…" > out.txt`
captures a clean completion with the summary still on the terminal, and
`2> stats.txt` captures plain text with no escape codes.

Four behaviours are specific to this engine rather than generic:

- **Seconds per token, not tokens per second.** Streaming experts off disk puts
  decode in the tens of seconds per token at GLM-5.2 shapes, where `tok/s` reads
  `0.06` and carries no information. The unit flips automatically above 1 tok/s.
- **The status line redraws from its own thread.** The socket blocks for as long as
  a token takes; a line that only moved on arrival would be indistinguishable from
  a hung process.
- **TTFT is excluded from the interval percentiles.** Prefill happens once. Folding
  it in would drag every percentile with it.
- **The spread is the headline.** A warm-cache token and one that streams a full
  routed union differ by an order of magnitude, so a slowest/fastest ratio ≥ 2× is
  called out explicitly.

Use it for A/B work: run a baseline, run an arm, compare `p50` — and read
[Measurement discipline](measurement.md) before believing a small gap.

No new dependencies: raw HTTP/1.1 over `std::net::TcpStream` including
chunked-transfer decoding, plus `serde_json` for the SSE payloads.

---

## `peregrine-requantize`

Rewrites a container's expert tensors at lower precision. This is the only lever on
read volume that scales with the *model* rather than the hardware — a faster drive
and a bigger cache both stop at what the device can deliver; halving the bytes
halves them everywhere.

```
peregrine-requantize <indir> <outdir> [options]

  --target <scheme>    int8 | int4 | int4-g<N> | int2      (default int2)
  --include <substr>   only tensors containing this        (default .mlp.experts.)
  --shard-bytes <N>    roll output shards at N bytes       (default 5000000000)
  --dry-run            report the size plan from headers, write nothing
  --tier-hot-frac <f>  heat-tiered: keep this fraction of each layer's experts
                       (by routing heat) at --tier-hot, the rest at --target
  --tier-hot <scheme>  precision for the hot fraction      (default int4)
  --mtp-target <scheme>  precision for the MTP head's own expert pool
```

### `--mtp-target`: the one rung on this ladder with no quality gate

The MTP head is a sparse MoE layer with its **own** expert pool, at layer index
`n_layers`, and in a GLM-5.2 container it is the last **int8** rung:
**37,748,736 bytes per expert against 18,874,368 at int4**. It is also read in
the worst possible regime — once per *draft step*, at `s_n = 1`, so unlike the
main stack it gets no batch-union amortization at all. At topk=8 that is roughly
300 MB of SSD per draft step.

`--mtp-target int4` halves it, and **needs no flip-rate gate**, which nothing
else on this ladder can say. A draft is accepted only where it equals the
verify forward's own argmax, so a coarser draft head can change the *acceptance
rate* and cannot change a served token. Run `peregrine flip-rate` anyway — as an
assertion that it reads **0.000**, which is what proves the override landed on
the head layer and nowhere else.

It is deliberately more specific than the other selectors: it beats both
`--keep-last-layers` (whose window counts `n_layers + 1` slots and therefore
covers the head) and `--down`, because a flag naming one layer is a statement
about that whole layer. Without it, both behave exactly as before.

**Always `--dry-run` first** — it reads only headers and tells you what the run
will cost in space:

```
$ peregrine-requantize "$MODEL" /mnt/models/glm52-int2 --dry-run --target int2
peregrine-requantize --dry-run: 118478 tensors (58368 would requantize to int2)
  | 383.73 GB -> 195.28 GB (50.9% of source), about 40 shard(s) at 5.0 GB each
both containers must fit at once: plan for 195.28 GB of free space.
```

Requantizing is **lossy — it changes token values.** Gate every output with
`Model::prediction_flip_rate` against the source container before relying on it.

### Heat tiering, and its prerequisite

`--tier-hot-frac` keeps each layer's hottest experts at `--tier-hot` and demotes the
cold tail to `--target`, so accuracy is spent where routing actually goes. It needs
a **routing-heat profile**, and the checkpoint almost certainly does not have one:

```
peregrine-requantize: heat tiering needs routing data — route_stats.json has no `heat` array
Produce it by running the model with COLI_ROUTE_STATS_PERSIST=1, or with `peregrine dump-routes`.
Refusing rather than tiering everything cold, which would look like it worked.
```

**The heat table is allocated only when the GPU tier is built** (`model.rs`:
`gpu.as_ref().map(|_| HeatTable::new(…))`), so producing it needs a run with
`COLI_GPU=1` **and** a binary built `--features cuda` — even though the decision it
feeds is pure storage. `scripts/heat-pass.sh` automates the run; the accumulation
itself lives on the CPU MoE path, so only the allocation is GPU-coupled.

Expect a tiered container to be **larger** than an all-`--target` one, and size
the disk from a `--dry-run` *with the same `--tier-hot-frac`*:

| run | output | of source |
|---|---|---|
| `--target int2` (all cold) | 195.28 GB | 50.9 % |
| `--tier-hot-frac` any > 0 | 249.30 GB | 65.0 % |

### Two fixes worth knowing about (2026-08-09)

Both were silent, and both are in `crates/peregrine-tools/src/requant.rs`:

- **`from_route_stats` could never accept a real heat file.** It demanded exactly
  `n_layers × n_experts`, but `HeatTable` is built `n_layers + 1` rows — the MTP
  head sits at index `n_layers` and routes a full set. At GLM-5.2 shapes that is
  20,224 produced against 19,968 demanded, so `--tier-hot-frac` refused *every*
  heat file the engine could produce. The extra row is now accepted and dropped.
- **`--dry-run` ignored the tier.** `plan_sizes` sized every expert at
  `plan.target`, so a tiered dry-run printed the all-cold plan and under-stated
  required free space by **54 GB**. It now mirrors `requantize`'s per-expert choice.

### Interpreting a flat `--tier-hot-frac` sweep

If every fraction reports the same per-token bytes, the heat profile is too thin,
not the tiering. Heat gathered from a single request makes the hot set *definitionally*
the routed set — every expert that gets routed has non-zero heat — so per-token bytes
come out at 100 % of an all-`--tier-hot` container whatever the fraction. Gather heat
over a broad corpus and evaluate on a *different* workload. `requant.rs` says the same
thing at the top of `HeatTier`: measure the skew before building a tiered checkpoint.

---

## `peregrine-skipbound`

An **offline prototype and a measurement**, not a runtime feature. It asks whether
expert reads could be skipped before they are issued — attacking 11.3 GB/token at
the root rather than compressing it.

```
peregrine-skipbound <model-dir> [--trace <routes.json>] [--out <bounds.json>] [--no-write]
```

The bound, transplanted from Quest's per-page min/max keys to expert weights:

```
||contribution|| <= gate * C_e * ||x||^2,   C_e = ||W_down||_F ||W_gate||_F ||W_up||_F
```

`||x||²` is common to every expert at a position, so the ranking a runtime skip
would use needs no hidden state — only the gate weights already in the trace.

**It deliberately stops short of the read path.** An upper bound is one-sided: a
small bound proves an expert cannot matter, a large one proves nothing. If few
reads clear the threshold, a runtime check costs per-token work and skips nothing —
a loose bound is not a partial win, it is zero wins plus overhead. So the tightness
measurement is the deliverable, and touching `read_regions` is gated on it.

## `peregrine-basisfit`

Cross-expert factorization (`W_e = B + Δ_e`), priced as **rate–distortion on
activations**. Fits a shared basis across each layer's routed experts, holds it
resident, and measures what the engine would actually pay. Measures only — no
container is written.

```bash
peregrine calib-capture <model-dir> calib.json 512 --text corpus.txt
peregrine-basisfit <model-dir> --calib calib.json --rank 2 --groups 8 \
    --residual int2-g64 --control
```

**Why it does not measure reconstruction error.** `‖W − (B + Δ)‖_F` is minimized
*by construction* when `B` is the group mean and `Δ` is stored exactly. That
experiment cannot fail, and its success says nothing about the container: a
basis can lower Frobenius error while making the residual **high-entropy and
hostile to the int4/block quantization it then has to survive**. Weight-space
error is structurally blind to that, because it never quantizes anything.

So the residual is actually quantized — through the same `quant_*` producer and
`QtView` consumer the loader uses, because an independently written "simulated"
quantizer would be measuring itself — and the error is weighted by calibrated
per-channel `mean|x|`.

**Both arms default to the same precision**, one rung below the container, so
streamed bytes are identical and the only variable is whether the residual
quantizes better than the weight. The resident basis is charged separately and
**both charges are always printed**: once (if it stays resident) and per token
(if it is evicted and re-read). At a 0.6 % warm-cache hit rate the second is the
likelier one, and quoting only the first assumes the favourable answer.

**`--control` is not optional in spirit.** It compares the learned grouping
against **four** shuffled partitions at equal rank. One draw is not enough: on
the demo container a single draw reported the grouping "load-bearing" at a
6.87 % margin, and against the best of four the same container reads 2.56 %
against a 4.63 % spread — a negative. The floor is `max(15 %, spread)`.

**Limitations it prints for you.** Distortion is a *diagonal* approximation of
`E_x‖(W−Ŵ)x‖²` (channels weighted independently; cross-channel structure
invisible). `down_proj` has **no** activation signal at all — the sidecar covers
hidden channels at the MoE input and `down_proj` is indexed by the intermediate
width — so it is scored unweighted and named in the caveats. And distortion is
not flip rate: gate the winning arm with `peregrine flip-rate` before believing
any of it.

Proposed by `orionzion` (2026-08-21). The sweep on real GLM-5.2 weights is still
owed; the harness is validated on fixtures only.

## Related pages

[Layout tools](layout-tools.md) · [Serving](serving.md) ·
[Measurement discipline](measurement.md) · [Performance tuning](performance-tuning.md) ·
[Model format & artifacts](model-format.md)


---

## `bench-serve-arrivals.py`

```
scripts/bench-serve-arrivals.py --rate <req/s> [--duration S] [--max-tokens N]
                                [--host H] [--port P] [--model ID] [--seed N]
                                [--repeat-prompt]
```

The other two serving clients are **closed-loop**: `bench-serve-lanes.py` starts
N streams and waits for them, `bench-serve-gaps.py` does the same and reports
inter-token gaps. In a closed loop the offered load is a *consequence* of the
server's speed — a slower server gets fewer requests, and the queue can never
grow. That is the one regime in which queue time is structurally invisible.

This one submits on a Poisson process at `--rate` and never waits for a
completion, so the server's own queue becomes the thing under test. Sweeping
`--rate` shows a knee rather than a smooth curve:

```
--- offered rate 50/s ---
  achieved=46.4/s   queue_mean=32us      ttft_p95=1.17ms
--- offered rate 400/s ---
  achieved=247.8/s  queue_mean=619910us  ttft_p95=1392.62ms
```

Below the knee, achieved ≈ offered and queue wait is microseconds. Past it,
achieved plateaus at what the engine can actually do while queue wait grows
without bound — and note what that second row says: **at saturation the queue
wait is roughly half the TTFT.** That span ends before the first token is
generated, so no client-side timer can see it; it comes from `/metrics`'
`queue` block, deltaed across the run.

**Prompts vary per request by default, and that is not cosmetic.**
`peregrine-serve` answers an exact repeat of a greedy request from its response
memo without touching the engine at all. The first version of this script
offered 61 requests and the engine admitted **one** — `measurement.md`'s
"benchmark that measures its own cache" in a new costume. `--repeat-prompt`
restores the old behaviour for the rare case where the memo *is* the subject,
and the script warns loudly whenever admissions fall far below completions.

503s are reported as data, not failures: past the knee they are
`COLI_QUEUE_DEPTH` shedding, which is the healthy outcome — an unbounded queue
would surface as latency instead.

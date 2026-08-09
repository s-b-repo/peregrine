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
```

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

## Related pages

[Layout tools](layout-tools.md) · [Serving](serving.md) ·
[Measurement discipline](measurement.md) · [Performance tuning](performance-tuning.md) ·
[Model format & artifacts](model-format.md)

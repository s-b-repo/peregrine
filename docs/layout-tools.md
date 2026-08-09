[« Docs index](README.md)

# Offline layout tools

`peregrine-tools` ships the `peregrine-layout-reorg` binary: it turns a
recorded routing trace into a disk-layout schedule (and, optionally, a
physically rewritten checkpoint) so the batched io_uring submit issues
contiguous-offset reads first. Everything is deterministic — the same trace
always produces the same artifacts — and correctness-neutral (the scheduler's
reduce is position-keyed, so read order can never change output).

## The pipeline

```bash
# 1. record a routing trace (synthetic deterministic corpus, default 256 forwards)
cargo run --release --bin peregrine -- dump-routes /path/to/model routes.json 512

# 2. compute a schedule (and optionally rewrite the checkpoint)
cargo run --release --bin peregrine-layout-reorg -- \
    --routes routes.json --out /path/to/model --method louvain --optimize

# 3. nothing else — the loader picks up /path/to/model/schedule.json automatically
```

Or run everything at once with `peregrine galactic <model-dir>`, which emits
`automaton.json`, `macrostates.json`, `routes.json`, `schedule.json`
(Louvain + 2-opt), optional `tiers.json`, and a `route_stats.json` seed in one
corpus pass — see the [CLI reference](cli-peregrine.md#galactic-model-dir-corpus-len).

## `peregrine-layout-reorg` flags

| Flag | Default | Meaning |
|---|---|---|
| `--routes <routes.json>` | **required** | routing trace from `dump-routes` (bare `[forward][layer][expert_id]` array) |
| `--out <model_dir>` | **required** | output directory (created if missing); `schedule.json` lands here |
| `--method <m>` | `cluster` | ordering method: `cluster`/`greedy`, `louvain`/`community`, `spectral`, `hilbert` |
| `--optimize` | off | refine each layer's order with 2-opt local search |
| `--apply` | off | physically rewrite `model.safetensors` in schedule order |
| `--help` / `-h` | | usage |

There are no tier flags on this binary — tier placement (`tiers.json`) is
driven by `peregrine galactic` under `COLI_TIER_VRAM_MB` / `COLI_TIER_RAM_MB`.

## Ordering methods

All operate on a per-layer expert co-occurrence graph built from the trace,
and all break ties by ascending expert id:

- **greedy** (alias `cluster`) — start at the node with the largest total
  incident co-occurrence weight, repeatedly hop to the heaviest unvisited
  neighbor; unreached experts append at the tail.
- **louvain** (alias `community`) — single-phase Louvain modularity
  maximization (integer arithmetic, bounded sweeps), communities emitted
  largest-first with a greedy walk ordering members inside each.
- **spectral** — sort experts by their Fiedler-vector value (second eigenvector
  of the graph Laplacian, via deflated power iteration) — the classical
  min-cut 1-D ordering.
- **hilbert** — lay the Louvain order onto a 2-D grid and re-sort by
  Hilbert-curve distance, mapping 2-D locality onto 1-D disk adjacency.
- **`--optimize` (2-opt)** — first-improvement segment reversals maximizing
  adjacent-pair co-occurrence weight; the objective is monotone
  non-decreasing, passes bounded at 8.

## Outputs

**`<out>/schedule.json`** — always written:

```json
{"version": 1, "n_layers": N, "order": [[expert ids per layer], …]}
```

Consumed by `Model::load` (gate `COLI_LAYOUT_SCHEDULE`, default on): at MoE
entry each layer's streamed expert plans sort by the schedule's rank;
unscheduled experts keep their original order and append. Anything but
`version: 1` is ignored.

**`<out>/model.safetensors`** — rewritten only under `--apply` (below).

## `--apply`: physical checkpoint rewrite

`--apply` rewrites the checkpoint so the schedule's order is the *physical*
disk order — the layout win without needing the schedule hint at runtime.

- **Single-shard only**: multi-shard checkpoints are rejected before any byte
  is written (`apply_layout supports single-shard checkpoints only`).
- Non-expert tensors are emitted first in original order; expert tensors
  follow, grouped by layer and ordered by schedule rank (experts absent from
  the schedule land at the end of their layer).
- Written via temp dir + atomic same-filesystem rename
  (`.relayout.tmp/model.safetensors` → `model.safetensors`).
- **Bit-identity gate**: the integration test `apply_layout_is_bit_identical`
  asserts teacher-forcing outputs are identical before/after a deliberately
  reversed schedule, and that the physical offsets actually moved.

Caveats worth knowing:

- Peak memory during the rewrite is roughly two copies of the checkpoint (all
  tensor payloads are staged before the single write), so budget RAM
  accordingly for large single-shard files.
- zstd-compressed tensors are **re-encoded** at level 3 — decoded bytes are
  identical, the compressed file bytes may not be.
- A `kblock`-tiled checkpoint loses its on-disk tiling tag after `--apply`
  (semantically safe — the reader had been de-tiling anyway — but the tiling
  optimization is dropped).
- There is no backup of the original file; the rename is atomic, but keep a
  copy if the checkpoint is precious.

## Library surface

`peregrine_tools` (the lib) exposes the building blocks the binary and
`galactic` share: `read_routes`, `build_cooccurrence`,
`greedy_nearest_neighbor`, `louvain_communities`, `spectral_order`,
`hilbert_order`, `two_opt`, `write_schedule`, `assign_tiers` / `write_tiers`
(whole Louvain communities placed greedily by heat density into
VRAM → RAM → disk byte budgets), `trace_heat`, and `apply_layout`.

---

## `peregrine-prune` — router-weighted expert pruning (REAP)

Ranks each layer's experts by the gate mass they carried over a routing trace,
drops the least salient, renumbers the survivors, and gathers the router's rows
to match.

```bash
peregrine dump-routes "$COLI_MODEL" routes.json 512 --text corpus.txt  # trace the workload you serve
peregrine-prune "$COLI_MODEL" --trace routes.json --dry-run
peregrine-prune "$COLI_MODEL" /path/glm52_pruned25 --trace routes.json
```

### What it buys, and what it does not

**Pruning does not reduce bytes per token.** Top-k is unchanged, so the same
`k` experts are read for every position whatever the pool size — Cerebras'
own model cards report identical activated parameters at 480B, 363B and 246B.
What shrinks is the **working set**: fewer distinct experts to hold, cache,
prefetch and lay out. On a disk-bound engine that is a residency win, not a
bandwidth win, and the two are routinely conflated. The tool says so in its
`--help` and again in its report, because a summary that read as a bandwidth
win would be actively misleading.

### Three things to get right before running it

**Use 25%, not 50%.** `--frac` defaults to 0.25 and refuses more without
`--force`. GLM-4.5-Air lost 11.2% on coding and 25.8% on multiple-choice at
50%, and retention does not improve with model size — the honest reading is
that GLM-family models degrade unusually.

**The calibration trace dominates the result.** Generic web text collapsed code
performance in the published runs. Saliency is measured from *your* trace; one
that does not contain the workload you serve will prune the experts that
workload needs.

**Saliency is gate mass, not selection count.** A frequently routed but weakly
weighted expert must not outrank a rare decisive one. If the trace carries no
weights the ranking degrades to counting, and the report says so rather than
letting a weaker signal pass as the stronger one.

### Two structural constraints worth knowing

`config.json` carries a **single** `n_routed_experts` for the whole model, so
pruning is necessarily **uniform across layers**. A per-layer keep count cannot
be expressed: it would produce a router whose width disagrees with the config
the loader sizes its buffers from, and that fails at load — hours after the
conversion finished.

The **MTP head is a sparse layer with its own router and expert pool**, and a
main-model trace never touches it. It cannot simply keep everything, because
the width has to match, so it is ranked on saliency aggregated across the
layers that *were* traced. The report counts how many layers took that
fallback; more than one means the calibration run was too short.

### After the run

A pruned checkpoint that was never compared is a guess. Measure it with
`Model::prediction_flip_rate` against the source container, on the workload you
traced — see [validation-runbook.md](validation-runbook.md).

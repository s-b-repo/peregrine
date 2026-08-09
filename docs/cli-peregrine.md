[« Docs index](README.md)

# `peregrine` CLI reference

The engine binary (crate `peregrine-engine`). All arguments are positional —
there are no flags; an unrecognized first argument is treated as a model
directory (serve mode). Exit code is `0` on success, `1` on any error
(printed as `peregrine: <error>` on stderr). Build with `--features cuda` on
an NVIDIA host to enable the GPU lane (runtime opt-in via `COLI_GPU`).

```
peregrine demo
peregrine build <dir>
peregrine bench [B ...]                                (model from COLI_MODEL)
peregrine build-automaton <model-dir> [corpus-len]
peregrine dump-routes <model-dir> <out.json> [corpus-len]
peregrine galactic <model-dir> [corpus-len]
peregrine compile-plan <model-dir>
peregrine <model-dir>                                  serve mode (stdio protocol)
```

## Subcommands

### `demo`

Self-contained end-to-end smoke test: builds a tiny synthetic model in a temp
directory, loads it, greedy-generates 8 tokens from a fixed prompt, prints the
result, and deletes the temp dir. Needs no model, no GPU, no network.

### `build <dir>`

Writes a tiny deterministic GLM-5.2-shaped demo model to `<dir>`
(`config.json` + `model.safetensors`; hidden 16, 3 layers — 1 dense + 2 MoE —
4 routed experts top-2, vocab 32, fixed seed). Useful as a serve/bench target
and as the shape reference for a [valid model directory](model-format.md).

### `bench [B ...]`

Aggregate decode-throughput sweep over batch sizes. The model comes **only**
from `COLI_MODEL` (no positional argument). Default batch set is `1 4 16`;
each row runs `COLI_BENCH_STEPS` decode steps (default 3) and reports
aggregate and per-sequence tok/s. Non-numeric or zero batch args are silently
dropped.

```bash
COLI_MODEL=/path/to/model cargo run --release --bin peregrine -- bench 1 4 16
```

### `build-automaton <model-dir> [corpus-len]`

Loads the model in forced streaming mode, runs a deterministic synthetic
corpus (default length 256), and writes the expert-transition FSA to
`<model-dir>/automaton.json` (config-tagged; auto-loaded on the next
`Model::load`). See [Prefetch & prediction](prefetch-and-caching.md).

### `dump-routes <model-dir> <out.json> [corpus-len] [--text FILE]`

Writes the raw per-forward routing trace to `<out.json>` (any path). The trace is
a bare nested array `[forward][layer][expert-id]` — the input format of
[`peregrine-layout-reorg`](layout-tools.md) and of
[`route-stats`](#route-stats-routesjson-n_experts).

**Pass `--text` for anything that reads the trace as a statement about routing.**
Without it the corpus is uniform-random token ids, and random ids route randomly:
`route-stats` over such a trace reports consecutive-token overlap at the
independence null *however the router behaves*, which reads as proof that
prefetch has nothing to predict when it is only proof that the corpus was noise.
`--text` encodes a real file with the tokenizer that travels with the container,
truncated to `corpus-len`. The subcommand warns loudly when it is absent.

Layout tools are less sensitive to this — they want co-occurrence over whatever
workload you serve — but they are not *insensitive*, and the same flag applies.

### `galactic <model-dir> [corpus-len]`

The one-shot offline preprocessing pass: ONE corpus run emits every artifact —
`automaton.json`, `macrostates.json`, `routes.json`, `schedule.json`
(Louvain ordering + per-layer 2-opt), optional `tiers.json` (only when
`COLI_TIER_VRAM_MB` and/or `COLI_TIER_RAM_MB` are set), and a
`route_stats.json` seed. All are picked up automatically at the next load.

### `flip-rate <source-dir> <candidate-dir> [--text FILE] [--tokens N]`

The quality gate for a **lossy** container — a `peregrine-requantize` output
measured against the checkpoint it was converted from. Every other gate in this
repo is bit-identity, which a requantized container fails by construction, so
`prediction_flip_rate` (top-1 agreement under teacher forcing) is what stands in
its place. Prints `positions`, `flips` and `flip_rate` on stdout.

```bash
peregrine flip-rate ~/models/GLM-5.2-int4 /mnt/models/GLM-5.2-int2g64 \
  --text sample.txt --tokens 512
```

- **The models load one at a time.** Two streaming loads would hold two warm
  caches against one page cache, and the slower container would be measured
  while the faster one evicted it. Peak RSS is one model's.
- **`--text` uses the *source* container's `tokenizer.json`**, so the ids are
  the ones that container was converted from. Without it the run uses
  uniform-random token ids and says so loudly on stderr — that is a smoke test
  of the harness, not a quality figure for the container.
- **Pick `--tokens` large.** One forward covers every position, and the bytes
  read are the routed *union*, which saturates: at top-8 over 256 experts a
  128-position forward already touches ~98% of the container, so 512 positions
  cost ~2% more bytes than 128 and buy 4× the statistics. There is no per-token
  read cost to economize on here.
- Top-1 on one text is a **floor** on quality, not a summary — a container can
  hold argmax everywhere and still shift the distribution underneath. The gate
  itself is pinned in both directions by `flip_rate_gate.rs`: zero flips against
  an identical container, non-zero against a deliberately lossy one.

### `compile-plan <model-dir>`

Pure file bundling, no model load: folds whatever artifacts exist
(`automaton.json`, `macrostates.json`, `schedule.json`, `tiers.json`, and the
learned policy from `route_stats.json`) into one `<model-dir>/plan.json`
(`"version": 1`) that `Model::load` consumes atomically — the profile-guided
execution plan. Missing parts are skipped silently; if *nothing* is found it
errors with a pointer to run `peregrine galactic` first.

### Serve mode (default)

With a model directory (positional or `COLI_MODEL` — **the env var wins** when
both are given), the binary speaks colibrì's stdio protocol as a drop-in for
`c/glm` behind `openai_server.py`.

## The stdio protocol

Sentinels are byte-exact matches for colibrì's:

- `READY` frame: `\x01\x01READY\x01\x01\n` — emitted once after the model loads.
- `END` frame: `\x01\x01END\x01\x01\n` — terminates every request.

Requests are read line-by-line from stdin:

| Input line | Behavior |
|---|---|
| `GEN <ngen> <tok0> <tok1> ...` | greedy-generates `ngen` tokens from the prompt ids; replies with the generated ids (decimal, space-separated, one line), then `END`. Only generated tokens are returned, not the prompt. |
| `QUIT` | exits cleanly (no `END`) |
| empty line | ignored (no `END`) |
| EOF | exits cleanly |
| anything else | ignored, but still answered with an `END` frame |

Notes:

- Generation is deterministic (greedy; temperature 0).
- A malformed `GEN` prints `peregrine: bad GEN request: <msg>` to stderr and
  still emits `END` with no data line. `ngen == 0` or an empty prompt likewise
  produce just `END`.
- After each generation, cache/prefetch diagnostics go to stderr when an
  expert cache is active: `[ecache] hits=… misses=… disk_reads=… hit_rate=…`
  and `[prefetch] used=… wasted=… accuracy=… fadvise=… verify_mismatch=…`.
- A broken pipe (peer closed mid-response) exits `0`, not `1`.

```bash
cargo run --bin peregrine -- build /tmp/demo-model
COLI_MODEL=/tmp/demo-model cargo run --bin peregrine
# → READY
GEN 8 1 5 9 2
# → <8 token ids>
# → END
QUIT
```

## Environment variables read by the binary

| Var | Used by | Effect |
|---|---|---|
| `COLI_MODEL` | serve, `bench` | model directory (wins over the positional arg in serve mode; the only source for `bench`) |
| `COLI_BENCH_STEPS` | `bench` | decode steps per batch size (default 3) |
| `COLI_TIER_VRAM_MB` / `COLI_TIER_RAM_MB` | `galactic` | tier byte budgets; `tiers.json` is emitted only when at least one is > 0 |

Everything else (`COLI_STREAM`, `COLI_ECACHE_GB`, `COLI_DIRECT`, `COLI_GPU`,
prefetch/cache/governor knobs, …) is read inside the model/io crates during
load — see [Configuration](configuration.md).

The offline passes (`build-automaton`, `galactic`, `dump-routes`) force
streaming mode; on a model too small to trigger auto-streaming they still work
because the loader is invoked with streaming forced on. The synthetic corpus
is a fixed-seed LCG, so the same corpus length always produces byte-identical
artifacts.

## Related pages

- [Model format & artifacts](model-format.md) — what each emitted JSON file
  contains and who consumes it.
- [Layout tools](layout-tools.md) — turning `routes.json` into `schedule.json`
  and a physically rewritten checkpoint.
- [Serving](serving.md) — the native HTTP server (`peregrine-serve`).

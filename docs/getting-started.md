[« Docs index](README.md)

# Getting started

## Prerequisites

- **Linux.** io_uring is the I/O backbone; a reasonably modern kernel
  (5.11+ for the registered-file/`SINGLE_ISSUER` ring features) is assumed.
- **Stable Rust** (edition 2021 workspace; no nightly features anywhere).
- **Optional — NVIDIA GPU + CUDA toolkit** (`nvcc`) for the GPU lane. The
  default build is CPU-only and the entire test suite runs without a GPU.

## Build & test

```bash
cargo build --release                    # optimized (fat LTO)
cargo test --workspace                   # 373 tests, CPU-only
cargo clippy --workspace --all-targets   # clean
scripts/audit-bad-patterns.sh --strict   # quality gate (see docs/BAD_PATTERNS.md)

# GPU lane (NVIDIA host with CUDA installed):
cargo build -p peregrine-cuda --features cuda
cargo test  -p peregrine-cuda --features cuda
```

## First run: the demo

No model needed — `demo` builds a tiny synthetic one, loads it, generates,
and cleans up:

```bash
cargo run -p peregrine-engine --bin peregrine -- demo
```

## Serve a model

Two binaries serve, for different callers:

**1. `peregrine` — stdio protocol** (drop-in for colibrì's `c/glm` behind
`openai_server.py`):

```bash
cargo run --bin peregrine -- build /tmp/demo-model      # write a tiny model
COLI_MODEL=/tmp/demo-model cargo run --bin peregrine    # emits READY, then:
#   GEN <ngen> <tok0> <tok1> ...   → greedy-generates, replies, emits END
#   QUIT
```

**2. `peregrine-serve` — native OpenAI-compatible HTTP** (continuous
batching, SSE streaming):

```bash
cargo run --release -p peregrine-serve -- --model /path/to/model --port 8080
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Note: the HTTP server also needs `<model-dir>/tokenizer.json` (a BPE
HuggingFace `tokenizer.json` — SentencePiece models are rejected at boot).
The tiny `build` model has none, so use the stdio engine with it.

## Real models

`Model::load` accepts any real int4/int8 container model directory in the
GLM-5.2 weight-naming scheme (`model.layers.N.self_attn.*`,
`mlp.experts.M.*`, …) — the format colibrì's FP8→int4 converter emits. Details
in [Model format](model-format.md). The `COLI_MODEL` env var name is kept
from colibrì for drop-in compatibility.

Streaming vs resident mode is decided automatically from available RAM;
`COLI_STREAM=1|0` overrides. On a big-MoE model the routed experts stream from
SSD per token; the dense sub-model stays resident.

## Make it faster (optional offline passes)

```bash
# everything at once: automaton + macro-states + routing trace + disk schedule (+ tiers)
cargo run --release --bin peregrine -- galactic /path/to/model 512

# bundle the artifacts into one atomically-consumed plan
cargo run --release --bin peregrine -- compile-plan /path/to/model

# or run the layout step manually / with a different method:
cargo run --release --bin peregrine -- dump-routes /path/to/model routes.json 512
cargo run --release --bin peregrine-layout-reorg -- \
    --routes routes.json --out /path/to/model --method louvain --optimize
```

Artifacts land in the model directory and are picked up automatically at the
next load. See [Layout tools](layout-tools.md) and
[Prefetch & caching](prefetch-and-caching.md).

## Benchmarks

```bash
COLI_MODEL=/path/to/model cargo run --release --bin peregrine -- bench 1 4 16
cargo run --release -p peregrine-serve -- --model /path/to/model \
    --bench-tokenizer big_text_file.txt
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `peregrine: usage: …` | missing positional args — see the [CLI reference](cli-peregrine.md) |
| `tokenizer: gigatoken can't load this model's tokenizer.json …` | SentencePiece/non-BPE tokenizer — unsupported by design ([why](tokenizer.md)) |
| `[peregrine] compressed checkpoint detected — disabling expert streaming` | zstd-compressed tensors force resident mode; keep experts uncompressed to stream |
| `this engine requires n_group=1 (GLM-5.2)` | the model config uses grouped routing — unsupported |
| Advisory failures seem swallowed | set `COLI_DEBUG=1` to surface them on stderr |
| Perf counter never opens | needs `perf_event_paranoid ≤ 2` and a PMU; it degrades to off by design |
| Want the serial A/B baseline | `COLI_PAR_THREADS=1` (fully serial, bit-identical) |

Every tuning knob is documented in [Configuration](configuration.md); none of
them can change the token stream.

[« Docs index](README.md)

# Full guide: from checkout to served completions

End-to-end path: build → point at a model → serve → call it from any
OpenAI-compatible client → measure. Reference pages carry the depth
([Configuration](configuration.md), [Serving](serving.md),
[Performance tuning](performance-tuning.md)); this page is the one
unbroken run.

## 1. Build

```bash
cargo build --release -p peregrine-serve   # HTTP server
cargo build --release -p peregrine-engine  # stdio/CLI engine (target/release/peregrine)
```

CUDA lane (optional, NVIDIA): build with `--features cuda` into
`target-cuda/` — runtime opt-in via `COLI_GPU=1`. CPU-only is the default.

## 2. Model directory

A directory with `config.json`, `*.safetensors` shards, and
`tokenizer.json`. Shards may be **split across drives** via
`model_paths.json` (`{"paths": ["/srv/m-sdb/...", ...]}` — sorted by file
name, duplicates refused), or **mirrored per drive** via
`COLI_IO_MIRROR_ROOTS` for striped parallel reads
([Configuration](configuration.md#coli_io_mirror_roots)).
Default to `. ./peregrine.env` in this repo — it sets `COLI_MODEL` and the
machine's known-good knobs.

## 3. Serve

```bash
peregrine-serve --model /path/to/model --port 8080
# vLLM launch lines map over:
peregrine-serve --model /path/to/model --served-model-name qwen3 \
  --max-model-len 4096 --max-num-seqs 8
```

Full flag table + vLLM spellings: [Serving](serving.md#cli-flags).
`COLI_CTX=1` serves prompts from the DCR-planned working set instead of the
raw prompt ([Context plane](context-plane.md)).

## 4. Call it (any OpenAI client works)

```bash
# non-streaming
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "What is the server ip?"}],
  "max_tokens": 128}' | jq -r .choices[0].message.content

# streaming
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3", "messages": [{"role": "user", "content": "hi"}],
  "stream": true, "max_tokens": 64}'

# hosted tools (server executes; pure ones speculate mid-stream)
peregrine-serve --model /m --host-tool tokenize --host-tool detokenize_batch
# newer OpenAI fields accepted: max_completion_tokens, top_k
```

## 5. Per-part counterparts

| Need | Command |
|---|---|
| Tokenizer only (no weights) | `peregrine-serve --model /m --bench-tokenizer file.txt` · `cargo run -p peregrine-token --example tokcount -- file.txt` (per-input token counts) |
| Token batch shapes (HF API) | `batch_encode` / `batch_decode` on `GigaTokenizer` ([Tokenizer](tokenizer.md#framework-compatible-batch-api)) |
| Context runtime CLI | `cargo run -p peregrine-ctx -- demo` · `ingest` / `ask` / `plan` / `bench` |
| IO lane in isolation | `cargo run --release -p peregrine-io --example iobench -- FILE BLK_MB ITERS RINGS DIRECT ENGINE REPS` |
| Watch generation live | `peregrine-gen --port 8137 --max-tokens 8 < prompt.txt` |
| Layout / prune / requant | [Layout tools](layout-tools.md) |

## 6. Measure

- Tokenizer: `--bench-tokenizer` rows (line/whole/parN).
- IO: 4-drive split method in
  [`bench-data/2026-09-22-4drive-split/`](../bench-data/2026-09-22-4drive-split/README.md).
- Serving: `/metrics` (tokens, lane timings, RLM, sPTC) + the byte ledger
  ([Measurement](measurement.md)).
- Rules: medians over reps, never single passes; O_DIRECT or dropped caches
  for device numbers; the harness warns when spread swallows the gap.

## 7. When it breaks

- `413 prompt is N tokens` → raise `--max-prompt-tokens`/`--max-model-len`,
  or `COLI_CTX=1` to serve a planned working set instead.
- Unknown `--host-tool` → boot error by design (see [sPTC](sptc.md)).
- Slow decode → [Performance tuning](performance-tuning.md) in measured
  order; `COLI_BLOCK_EXEC=sync|mt|par` selects the block thread shape.

[« Docs index](README.md)

# Serving: `peregrine-serve` HTTP server

`peregrine-serve` is the native OpenAI-compatible HTTP server (axum/tokio) with
continuous batching. One dedicated engine thread (`peregrine-batch`) owns the
model; HTTP handlers only exchange token ids with it over channels. The crate
is `#![forbid(unsafe_code)]`.

```bash
cargo run --release -p peregrine-serve -- --model /path/to/model --port 8080
```

Boot prints `[tokenizer] gigatoken BPE active, vocab=<n>` and
`peregrine-serve listening on http://<host>:<port>` to stderr. Shutdown is
graceful on Ctrl-C / SIGINT (in-flight connections drain; no SIGTERM handler).

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model <dir>` | **required** (env: `COLI_MODEL`) | model dir with `config.json`, `*.safetensors`, `tokenizer.json` |
| `--host <host>` | `127.0.0.1` | localhost-only by default |
| `--port <port>` | `8080` | |
| `--api-key <key>` | unset (env: `PEREGRINE_API_KEY`) | enables bearer auth when set |
| `--max-tokens <n>` | `1024` | hard server cap on generated tokens per request |
| `--max-prompt-tokens <n>` | `8192` | hard cap on encoded prompt tokens (413 above it) |
| `--model-id <id>` | `glm-5.2` | id reported by `/v1/models` and echoed in responses |
| `--max-batch <n>` | `32` | continuous-batching width ceiling |
| `--bench-tokenizer <file>` | unset | tokenizer throughput bench, then exit (no weights loaded) |

## Endpoints

### `GET /health`

No auth. `200` →
`{"status":"ok","tokenizer":"gigatoken","memo":{"hits":0,"misses":0,"entries":0,"bytes":0}}`.

`memo.hits` counts requests answered from the [response memo](#response-memo) without
entering the model at all — the one rate here that measures work *not* done.

### `GET /v1/models`

Auth required when an API key is configured. Returns one model card:

```json
{"object":"list","data":[{"id":"glm-5.2","object":"model","owned_by":"peregrine"}]}
```

### `POST /v1/chat/completions`

Auth required when an API key is configured. Request fields (all optional
except that `messages` must be non-empty; unknown fields are ignored):

| Field | Type | Default | Behavior |
|---|---|---|---|
| `messages` | `[{role, content}]` (both strings) | — | required non-empty; array-form `content` is not accepted |
| `max_tokens` | int | `256` | clamped to `[1, --max-tokens]` |
| `temperature` | float | `0.0` (greedy) | clamped to `[0.0, 2.0]`; note the default differs from OpenAI's `1.0` |
| `top_p` | float | `0.95` | clamped to `[0.0, 1.0]` |
| `stream` | bool | `false` | SSE streaming when `true` |

`model`, `n`, `stop`, `seed`, `logprobs`, penalties, `tools`,
`response_format`, and `stream_options` are not supported (ignored). Stop ids
come from the model's `config.json` `eos_token_id` and are never emitted.
Non-greedy sampling is seeded from the clock, so it is not reproducible;
`temperature: 0` is deterministic.

Prompts are built with the GLM chat template: `[gMASK]<sop>` then
`<|system|>` / `<|user|>` / `<|assistant|>` + `\n` + content per message
(unknown roles map to `<|user|>`), ending with a trailing `<|assistant|>\n`.

**Non-streaming response** (`200`):

```json
{"id":"chatcmpl-<nanos>","object":"chat.completion","model":"glm-5.2",
 "choices":[{"index":0,"message":{"role":"assistant","content":"..."},
             "finish_reason":"stop"}]}
```

`finish_reason` is always `"stop"` (never `"length"`); there is no `usage` or
`created` field.

**Streaming response** (`200`, `text/event-stream`; keep-alive comment every
15 s). Content chunks:

```
data: {"object":"chat.completion.chunk","model":"glm-5.2","choices":[{"index":0,"delta":{"content":"<text>"},"finish_reason":null}]}
```

then a final `finish_reason:"stop"` chunk with an empty delta, then
`data: [DONE]`. There is no initial role delta and chunks carry no `id`.
Deltas are decoded-text suffixes — a token that doesn't lengthen the decoded
string emits no chunk. If an engine error occurs mid-stream, the stream emits
`data: {"error":{"message":"..."}}` and ends without a finish chunk or
`[DONE]`. A client disconnect aborts the request and the sequence is retired
on the next engine step.

**Errors** use the OpenAI shape `{"error":{"message":"…","type":"…"}}`:

| Status | Trigger |
|---|---|
| 400 | `messages must not be empty` |
| 401 | missing or invalid API key (`Authorization: Bearer <key>` required) |
| 413 | prompt exceeds `--max-prompt-tokens` |
| 500 | tokenizer failure or engine error |

(Unparseable JSON bodies are rejected by the framework before the handler and
return plain-text 4xx, not the OpenAI shape.)

### `X-Peregrine-Priority` header

On `/v1/chat/completions` only. `high`, `1`, or `true` (case-insensitive) →
high priority; anything else, or absence, → normal. Each engine tick drains
the high queue first. Priority reorders **admission only** — it never changes
any request's token stream.

## Response memo

`COLI_MEMO_ENTRIES` (default 32) and `COLI_MEMO_MB` (default 64) bound a cache of
completed responses; either at `0` disables it. A hit answers the request *before*
it reaches the engine, so it consumes no batch slot and produces no KV state.

This is worth more here than on a typical inference server: one token costs a pass
over gigabytes of streamed experts, and an OpenAI-compatible endpoint is re-asked the
same question constantly — health checks, retried requests, eval fixtures, clients
that re-send an unchanged conversation.

Three rules keep it from being a correctness hazard.

- **The key is the complete request semantics** — prompt token ids, `max_tokens`,
  `top_p` (by bit pattern) and the model id — **compared field-by-field, never
  hashed.** Same rule as the prefix cache, same reason: a hash collision would serve
  one caller another caller's answer, silently and unboundedly.
- **Only `temperature == 0` is eligible.** Sampling draws against a clock-derived
  seed; replaying a stored draw would quietly turn a sampling endpoint into a
  deterministic one, and a user asking twice for variety would get the same text with
  no indication why. Greedy decoding is reproducible by contract, so that is where
  memoization is honest.
- **Entries hold token ids, not wire bytes.** The framing is rebuilt per request —
  its own completion id and `created` — so a streaming request can be served from an
  entry a non-streaming request created, and no request's identifiers leak into
  another's response.

A generation that did not finish (engine error, client disconnect mid-stream) is
never stored: a truncated answer replayed as a complete one is worse than no memo.

## Continuous batching

- **Chunked prefill:** prompts advance 64 tokens per engine tick, round-robin,
  interleaved with decode — admitting a long prompt never stalls the in-flight
  batch. Bit-identical to whole-prompt prefill (asserted by
  `engine_chunked_prefill_matches_reference`).
- **Working cap:** `--max-batch` is the ceiling. With `COLI_BATCH_SLA_MS=<ms>`
  an EWMA of per-step wall time shrinks the cap when over the SLA and regrows
  it (up to the ceiling) when comfortably under.
- **Decode-heavy window:** `COLI_ADAPTIVE_WINDOW=N` runs prefill only every
  Nth tick (default 1; prefill always runs when nothing is decoding).
- **Reheat:** every 256 batched decode steps the GPU tier re-selects its
  hottest residents (no-op without a GPU tier).
- **Idle maintenance:** when nothing is in flight the engine runs bounded
  warm-cache recompression sweeps (`COLI_CACHE_COMPRESS_IDLE`), aborting the
  moment a request arrives.
- **Workload classes:** the handler classifies each request's last user
  message tail (code/JSON/math/prose/mixed) to select per-class prefetch
  breadth; the setting is batch-global, latest admission wins.

Why batching pays: decoding B sequences together reads each routed expert
**once per step and shares it across the batch** — a measured 4.4× aggregate
gain at B=16 on the real 744B model. See [Benchmarks](benchmarks.md).

## Runtime tokenizer

The sole runtime tokenizer is the vendored gigatoken BPE engine
([details](tokenizer.md)). It loads `<model-dir>/tokenizer.json`; a tokenizer
gigatoken can't handle (SentencePiece/non-BPE) is a **hard boot error** —
`tokenizer: gigatoken can't load this model's tokenizer.json …` — never a
silent fallback. The instance is process-persistent, so its pretoken memo
cache warms across requests (repeated chat-template prefixes encode from
cache).

`--bench-tokenizer <file>` runs without loading weights and prints three
throughput rows: `gigatoken/line` (one encode per line — the serve pattern,
and the row behind the documented 34×-vs-HF measurement),
`gigatoken/whole` (one call over the whole file — single-core engine
capability, ~3× the line row), and `gigatoken/parN p1/p2` (parallel
`encode_batch` over ~256 KiB slices, cold then warm — the warm row is the
batch-layer steady state). See
[Tokenizer → Throughput anatomy](tokenizer.md#throughput-anatomy).

## Examples

```bash
# non-streaming
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'

# streaming, high priority, with auth
curl -sN localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer SECRET' \
  -H 'x-peregrine-priority: high' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

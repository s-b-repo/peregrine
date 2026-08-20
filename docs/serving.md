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
| `messages` | `[{role, content, …}]` | — | required non-empty. `content` may be a string, `null`, absent, or an **array of `{type, text}` parts** (parts without `text` are dropped, the rest concatenated). Assistant turns may carry `tool_calls`; `role: "tool"` is accepted and replayed as an observation |
| `tools` | `[{type, function:{name, description, parameters}}]` | none | tool schemas, rendered into the system turn — see [Tool calling](#tool-calling) |
| `tool_choice` | string or object | none | only the exact string `"none"` is honoured, and it disables tools for that request. Any other value (including `"auto"`, `"required"`, or an object naming a function) leaves all tools active — peregrine does not force a call |
| `max_tokens` | int | `256` | clamped to `[1, --max-tokens]` |
| `temperature` | float | `0.0` (greedy) | clamped to `[0.0, 2.0]`; note the default differs from OpenAI's `1.0` |
| `top_p` | float | `0.95` | clamped to `[0.0, 1.0]` |
| `stream` | bool | `false` | SSE streaming when `true` |

`model`, `n`, `stop`, `seed`, `logprobs`, penalties,
`response_format`, and `stream_options` are not supported (ignored). Stop ids
come from the model's `config.json` `eos_token_id` and are never emitted.
Non-greedy sampling is seeded from the clock, so it is not reproducible;
`temperature: 0` is deterministic.

Prompts are built with the GLM chat template: `[gMASK]<sop>` then
`<|system|>` / `<|user|>` / `<|assistant|>` + `\n` + content per message
(unknown roles map to `<|user|>`), ending with a trailing `<|assistant|>\n`.

**Non-streaming response** (`200`):

```json
{"id":"chatcmpl-<nanos>","object":"chat.completion","created":1755100000,
 "model":"glm-5.2",
 "choices":[{"index":0,"message":{"role":"assistant","content":"..."},
             "finish_reason":"stop"}],
 "usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}
```

`finish_reason` is `"stop"`, or `"tool_calls"` when the generation produced
tool calls (see §Tool calling); `"length"` is never emitted.

**Streaming response** (`200`, `text/event-stream`; keep-alive comment every
15 s). The first chunk carries the assistant role delta; every chunk carries
the completion `id` and `created`. Content chunks:

```
data: {"id":"chatcmpl-<nanos>","object":"chat.completion.chunk","created":...,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":"<text>"},"finish_reason":null}]}
```

then a final finish chunk with an empty delta, then `data: [DONE]`.
Deltas are decoded-text suffixes — a token that doesn't lengthen the decoded
string emits no chunk. If an engine error occurs mid-stream, the stream emits
`data: {"error":{"message":"..."}}` and ends without a finish chunk or
`[DONE]`. A client disconnect aborts the request and the sequence is retired
on the next engine step. A submit against a full admission queue
(`COLI_QUEUE_DEPTH` set) is refused up front with an OpenAI-shaped `503`
`overloaded_error` — retry later.

**Errors** use the OpenAI shape `{"error":{"message":"…","type":"…"}}`:

| Status | Trigger |
|---|---|
| 400 | `messages must not be empty` |
| 401 | missing or invalid API key (`Authorization: Bearer <key>` required) |
| 413 | prompt exceeds `--max-prompt-tokens` |
| 500 | tokenizer failure or engine error |

(Unparseable JSON bodies are rejected by the framework before the handler and
return plain-text 4xx, not the OpenAI shape.)

### Tool calling

OpenAI clients send tool *schemas* in a `tools` array and expect calls back in
`choices[].message.tool_calls`. GLM-5.2 reads schemas from the system turn and
emits calls as `<tool_call>` markup its tokenizer has dedicated tokens for.
peregrine bridges the two in both directions, so an OpenAI-shaped client needs
no changes.

**Request → prompt.** Schemas render into a `# Tools` block inside the **first**
system turn (a system turn is synthesized if the request has none). This is
GLM's own template placement: a bare tools turn with no system content trains
the model to answer the schema instead of using it. `tool_choice: "none"`
suppresses the block entirely; see the field table for why no other value
changes anything. Assistant turns carrying `tool_calls` are rendered back into
`<tool_call>` markup so a multi-turn conversation replays to the model in the
form it emitted, and `role: "tool"` turns become `<|observation|>` with the
result wrapped in `<tool_response>`.

**Output → response.** A streaming filter splits the token stream into visible
text, tool calls, and `<think>` reasoning, holding back any suffix that could
still turn out to be a marker — so a `<tool_call>` arriving one character at a
time never leaks a fragment to the client. Then:

- `finish_reason` is `"tool_calls"` when the turn produced any call, else `"stop"`.
- `content` is `null` on a calls-only turn, and the text otherwise.
- Call ids are `call_<completion-id-without-the-chatcmpl-prefix>_<index>`, stable
  between the streaming and non-streaming paths for the same completion.
- Argument values are typed **from the declared schema**, falling back to
  sniffing only for undeclared parameters. This is what keeps a shell command
  `1.10` a string instead of the float `1.1`, and a file named `true` a string
  instead of a boolean.
- `<think>` blocks are dropped, not shown.

**One deliberate divergence from OpenAI's streaming shape:** peregrine emits one
*whole* call per SSE chunk, where OpenAI fragments `arguments` across deltas. The
markup is not a well-formed call until its closing tag arrives, so emitting
partial arguments would mean emitting text that may never become valid JSON.
Clients that accumulate `arguments` deltas still work — they receive one delta
containing the whole string.

**Truncation.** A call cut off at the token cap is still parsed and returned:
the call is the point of the turn, so a recovered one beats a discarded one. An
unclosed `<think>` block is *not* recovered — reasoning is not for the client —
so a response truncated mid-reasoning is an empty one.

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

- **Chunked prefill:** one pending prompt advances per engine tick (round-robin
  across pending prompts), interleaved with decode — admitting a long prompt
  never stalls the in-flight batch. The chunk grows geometrically with position
  (`max(64, pos/4)`; `COLI_PREFILL_CHUNK_DIV`, `0` = fixed 64), which keeps
  total prefill work linear in prompt length instead of quadratic.
  Bit-identical to whole-prompt prefill (asserted by
  `engine_chunked_prefill_matches_reference`).
- **Fused prefill (default on):** the prefill chunk rides the decode batch's
  forward, sharing one routed-expert union instead of streaming two disjoint
  ones (`COLI_FUSE_PREFILL=0` restores the two-forward tick).
  `COLI_MAX_BATCH_ROWS` optionally bounds the fused forward's total rows
  (chunk yields first; draft depth bounds itself to fit).
- **Prefix cache (default 2048 MB):** completed prompts *and their generated
  tokens* seed later requests sharing a token prefix — a multi-turn client's
  next request is a refcount bump, not a re-prefill (`COLI_PREFIX_CACHE_MB`,
  `0` disables). Matching is exact-token comparison, never a hash.
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

## Observability: `GET /metrics`

Unauthenticated JSON snapshot, published by the engine thread once per tick
(the thread owns the `Model`; handlers only clone the latest snapshot). Fields
beyond the lane timings and cache counters documented in
[configuration.md](configuration.md): `spec` (MTP drafts
`proposed`/`accepted`/`accept_rate` — each proposed draft is a verify row of
expert reads, so the accept rate is what says whether `COLI_DRAFT`'s depth
pays), `rlm` (recursive-refinement passes and tokens, `COLI_RLM`),
`io_slab_in_use` (O_DIRECT landing buffers in flight; pinned at the pool cap
means reads are serializing on buffers), and `memo` (response-memo counters).
On a recurrent arch under `COLI_SPEC_GDN`, `spec` also carries
`gdn_snapshot_bytes` / `gdn_replays` / `gdn_replay_rows` — the cost side of that
rollback, which is what decides whether it pays at a given batch width.

`decode` (`tokens_emitted`, `rows`) is the **numerator** `/metrics` was missing.
The denominator was already there: `ecache.hits + misses` is every routed-expert
entry the streaming lane resolved and `ecache.disk_reads` the subset that
reached the device. Delta both across two scrapes and the metric this engine is
ultimately tuned on falls out directly:

```
tokens per expert read = Δdecode.tokens_emitted / Δ(ecache.hits + ecache.misses)
tokens per disk read   = Δdecode.tokens_emitted / Δecache.disk_reads
rows per token         = Δdecode.rows           / Δdecode.tokens_emitted
```

The first two are *SSD bytes per accepted token* — the figure
[`speculative-decoding-alternatives.md`](speculative-decoding-alternatives.md)
names as the only one that decides a speculative technique on the streaming
track — measured rather than derived. The third is speculation's row overhead
and is what an expert-union budget would be spent against.

Read `rows per token` with one correction: a request's **first** token is
sampled from the prefill's last position and costs no decode row, so the ratio
sits just below 1.0 unspeculated and approaches 1.0 as requests lengthen. Above
1.0 is speculation's overhead.

`queue` (`wait_us`, `admits`, `max_us`, `mean_us`) is admission latency — the
span between `submit` and a request becoming a `Prefilling`. Every other latency
instrument here starts counting once a request is *already being served*, so
queue time was indistinguishable from slow decode; with `COLI_QUEUE_DEPTH`
shedding at the door, this is what separates "at capacity" from "over capacity".
Counted per admission, not per submit: a refused request never waited.

`energy_uj` is cumulative CPU-package energy, so `Δenergy_uj ÷
Δdecode.tokens_emitted` is microjoules per token. It reads `null` unless the host
grants the RAPL counter — it is root-only on current kernels (the PLATYPUS
mitigation) — and `null` rather than `0` because zero energy and no permission
are different facts. Treat it as a **floor** on system energy: RAPL sees the
package, and the component doing the most work on this engine is the SSD, which
it cannot see. See [external-audit-response.md](external-audit-response.md) for
the udev rule and the caveats in full.
Cumulative counters are meant to be delta'd across two scrapes. At shutdown
the engine prints `[prefix-cache]`, `[spec]` and `[rlm]` summary lines to
stderr (each silent when its feature never engaged).

## Speculation and refinement

With an MTP head in the checkpoint, `COLI_DRAFT=<g>` drafts `g` tokens per
sequence per tick and verifies them in the same batched forward (greedy
requests take argmax-identity acceptance; sampled requests join only under
`COLI_DRAFT_SAMPLED`, via distribution-preserving rejection sampling).
`COLI_DRAFT_NGRAM=<n>` adds a **second draft source** to the same verify path:
prompt-lookup, which proposes whatever followed the last occurrence of the
current suffix ([note](configuration.md#coli_draft_ngram)). It costs a backward
scan rather than a model call, takes priority over the head whenever it matches,
works on a checkpoint with no MTP head at all, and is reported separately under
`ngram` on `/metrics` — pooling a free source's accept rate with an expensive
one's hides which is doing the work.

On a **recurrent** arch (Qwen3.5-hybrid) drafting additionally needs
`COLI_SPEC_GDN=1`: a linear-attention layer keeps a delta-rule state rather
than rows, so a rejected draft cannot be truncated away and has to be rolled
back by snapshot/restore plus a re-advance over the accepted rows
([note](configuration.md#coli_spec_gdn)). Same greedy stream either way; the
snapshot is ~151 MB per drafting sequence per tick at 27B dims, which is why it
is a knob and why `/metrics` reports `gdn_snapshot_bytes` beside the accept
rate.

`COLI_RLM=1` additionally refines *uncertain* decision rows (top-2 margin /
entropy policy, `COLI_RLM_*` knobs) with recursive replay passes over the
request's own KV before the token is picked — composed with speculation
exactly as the stdio engine's `generate_speculative` does: raw logits decide
acceptance, only the post-acceptance contested row refines, sampled
speculative runs are never refined. Both are off by default and structurally
bit-identical when off.

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

## `peregrine-gen` — watch generation live, and time it

A streaming client that prints the completion as it arrives and reports what the
engine did. It exists because `curl -N` shows you the text but nothing about the
*shape* of the run, and `scripts/bench-prefetch-arms.sh` reports one lumped
`decode_s` per request — which on a mostly-prefill request is largely prefill.

```bash
peregrine-gen "Explain how a mixture-of-experts layer routes a token."
peregrine-gen --port 8137 --max-tokens 8 --json timings.json < prompt.txt
```

```
── peregrine-gen ──────────────────────────────────────────────
  generated    2 tokens, 3 chars
  ttft         2m 23s  (prefill + first token)
  total        2m 38s
  decode rate  15.23 s/tok (0.066 tok/s)  (excludes ttft)
  inter-token  min 15.1s  p50 15.1s  p95 15.1s  max 15.1s
```

**Text goes to stdout, statistics to stderr**, so `peregrine-gen "..." > out.txt`
captures a clean completion while the summary stays on the terminal, and
`2> stats.txt` captures plain text with no escape codes.

Three things are tuned to this engine rather than generic:

- **Seconds per token, not tokens per second.** Streaming experts off disk puts
  decode in the tens of seconds per token at GLM-5.2 shapes, where `tok/s` reads
  `0.07` and says nothing. The unit flips automatically above 1 tok/s.
- **The status line ticks from its own thread.** A token can take a minute; a line
  that only redrew on arrival would be indistinguishable from a hang.
- **The inter-token spread is the headline.** A token served from the warm cache
  and one that streams a full ~11.3 GB routed union differ by an order of
  magnitude, so `min/p50/p95/max` carry the signal an average destroys. A
  slowest/fastest ratio ≥ 2× is called out explicitly, and `--json` writes every
  interval for offline analysis.

TTFT is measured to the first delta and **excluded from the interval
percentiles** — prefill is work that happens once, and folding it in would drag
every percentile with it.

No new dependencies: raw HTTP/1.1 over `std::net::TcpStream` including
chunked-transfer decoding, plus `serde_json` for the event payloads.

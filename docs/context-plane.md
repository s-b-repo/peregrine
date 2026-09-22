[« Docs index](README.md)

# Context plane (`peregrine-ctx`, `COLI_CTX`)

`peregrine-ctx` is the **ported subnext/DCR runtime** (crate `dcr`):
unlimited external state plus a small, planned working set. Upstream files
live under `crates/peregrine-ctx/src/` verbatim apart from the crate prefix
(`dcr::` → `peregrine_ctx::`) and the CLI name (`peregrine-ctx`), so future
re-ports diff cleanly. The CLI keeps its upstream shape:

```bash
cargo run -p peregrine-ctx -- demo
cargo run -p peregrine-ctx -- bench
peregrine-ctx ingest notes/ --store .ctx.json
peregrine-ctx ask "what is the server ip?" --budget 600
```

## How serving uses it

The plane lives **behind the HTTP layer, not inside the token-level engine**
— context is a text-level concern, and the engine stays tokenizer-free.
`crates/peregrine-serve/src/ctxplane.rs` is the seam, and it is used as
**context, not compression**: with `COLI_CTX=1` the prompt the model sees is
the runtime's planned working set on every request — message turns ingested
role-tagged (tool schemas as a system turn), relevance-planned for the last
user turn under `max_prompt_tokens`, rendered as context — instead of the raw
rendered prompt. With the knob off (or when assembly yields nothing) the raw
prompt serves exactly as before.

- Stateless per call: a fresh runtime per request, no cross-request memory
  (the server is stateless per request), deterministic for identical turns.
- Ingestion failure degrades to the raw prompt, never a partial context: a
  turn that fails to ingest aborts the whole assembly, and an empty assembly
  is refused the same way.
- The over-budget error still applies to the assembled ids: assembly plans
  *what* the model attends to, the budget still caps *how much*.
- The serving line says what happened:
  `peregrine: [ctx] serving planned working set (N raw chars → M assembled
  chars)`.

## Cost

Per-request assembly is ~100 ms at 800 turns on CPU (7150 input tokens →
321-token set at budget 512; output pins at relevance, not at the cap), off
the decode critical path. Method and numbers:
[`bench-data/2026-09-22-ctx-assembly/`](../bench-data/2026-09-22-ctx-assembly/README.md).

## Knobs

| Var | Default | Effect |
|---|---|---|
| `COLI_CTX` | off | `1` enables the plane. Off by default: engagement changes tokens, so it must be asked for |

The per-call plan budget is always `max_prompt_tokens` — the plane exists to
fit the prompt into the budget the server already enforces, not to set its
own.

## What it is not

- **Not cross-request memory.** No session store exists; every request
  plans from its own turns. A persistent `.context` container across
  requests is future work (the runtime's `context_store` already defines
  the format).
- **Not in the RLM loop.** Recursion still keys on logit margins
  (`COLI_RLM`); the plane feeds prompts, not passes. Wiring planned sets
  into refinement passes is open.
- **Not GPU.** The runtime is zero-dependency CPU text indices — no model
  calls, no device code.

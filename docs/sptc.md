[« Docs index](README.md)

# sPTC — speculative programmatic tool calling

**The model streams a tool call one token at a time; this server can already be
running the tool before the call finishes.** When the operator hosts tools
(`peregrine-serve --host-tool tokenize --host-tool count_tokens`), every
*complete* argument pair of a still-streaming call is offered to a shadow
executor — a bounded worker pool that runs the tool on the arguments so far
and files the result under the exact key the finished call will carry. If the
model closes the call with the same arguments (the common case: a prefix is a
prefix), the result is already on the shelf at close time. If it does not, the
guess is discarded and the call executes exactly as it would have. The stream
is byte-identical either way: speculation changes *when* a result exists,
never *what* it is.

Without sPTC, the tool round trip is: finish the call → the client executes it
→ the client re-sends the whole conversation → the server re-renders and
re-prefills before the model learns the answer. With a hosted pure tool, the
answer exists at close time, attached to the same response.

## The three safety rules

1. **Only pure tools are speculated — and only pure tools can be hosted.**
   Purity is not a flag a caller can get wrong: the `pure_tool!` macro
   (sptc.rs) is the *only* way to construct a `HostedTool` — its run closure
   is private — so "deterministic, side-effect-free, safe to run on a partial
   guess of the arguments" is a property of the type, not of registration
   discipline. A wrong guess wastes CPU; nothing else.
2. **Verification is by exact argument match, never by trust.** The key is
   the canonical JSON of the final parsed arguments (sorted keys — serde
   `Map` is a `BTreeMap` here — and the same schema-typed `coerce` the
   client-facing parse uses). No hash, no fuzzy match: one extra character is
   a miss, and a miss only costs the speculation.
3. **Hosted tools have no ambient authority.** A hosted tool receives its
   JSON arguments and nothing else — no disk, no network, no engine state. A
   client cannot smuggle a computation into this process's privileges: it can
   only invoke what the operator explicitly hosted. Input and output are
   byte-capped both sides (64 KiB in, 1 MiB out); a rejected input surfaces
   as an `error` member on the attached result, never as a silent success.

## Wire contract

Everything standard is untouched. A **hosted** tool's `tool_calls` chunk (and
the non-streaming `message.tool_calls[i]`) carries one additive member:

```json
{
  "index": 0, "id": "call_…", "type": "function",
  "function": { "name": "count_tokens", "arguments": "{\"text\":\"…\"}" },
  "peregrine_result": { "count": 42 }
}
```

Unhosted calls carry no `peregrine_result` and are byte-identical to a server
with no sPTC at all — standard OpenAI clients ignore the extra member, and
harnesses that know it save their own round trip. A tool that failed reports
`"peregrine_result": {"error": "…"}`; a failure is cached like a hit, because
a pure tool's failure is as deterministic as its success.

The three built-ins are tokenizer queries over the server's own tokenizer
(same ids `/v1/debug/tokenize` reports): `tokenize {text} → {ids, count}`,
`detokenize {ids} → {text}`, `count_tokens {text} → {count}`. An agent that
budgets `max_tokens` before writing a call no longer has to guess.

## Knobs and lifecycle

| Knob | Effect |
|---|---|
| `--host-tool NAME` (repeatable) | Host a tool from the closed set. An unknown name is a **boot error** — a misconfiguration must not silently turn hosted calls back into client round trips. |
| `COLI_SPTC=0` (or `false`/`off`) | Kill switch for *speculation*; hosted tools still execute, synchronously at close. |
| *(absent)* | No tools hosted → the subsystem does not exist: no threads, no table, no per-token work. |

Internal shape, for operators reading `/metrics`'s `sptc` block: two shadow
worker threads, a 16-deep speculation queue (full queue ⇒ the guess is
skipped, never a stall — the sync run at close is always the floor), and a
256-entry / 8 MiB LRU table of verified results. `speculated` counts enqueued
guesses; `verified_hits` the ones the model's close confirmed;
`verified_hits ≈ speculated` with `sync_runs ≈ 0` means the shadow is saving
its wait per hosted call — the reverse says it is only burning CPU on guesses
that never verify.

## What it is not

- **Not an agent loop.** The server executes the tool and hands the result
  back; it does not continue the turn on the client's behalf. The OpenAI
  protocol is unchanged.
- **Not for side-effectful tools.** A tool that writes, launches, or reaches
  the network cannot be hosted — not "should not": the only constructor
  enforces the pure contract, and the shipped set is read-only tokenizer
  math.
- **Not free when it misses.** A discarded guess costs one bounded
  in-process run. That is the whole downside, and `/metrics` says how often
  it happens.

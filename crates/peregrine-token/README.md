# peregrine-token

A **vendored, stable-toolchain subset** of
[marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) **v0.10.0** (MIT
— see `LICENSE-MIT-gigatoken`): GB/s BPE tokenization, the **sole runtime
tokenizer** of `peregrine-serve` (the HuggingFace `tokenizers` crate is a
dev-dependency parity oracle only).

## What is vendored (upstream-verbatim where possible)

| Module | Upstream path | Notes |
|---|---|---|
| `bpe/tiktoken.rs` | `src/bpe/tiktoken.rs` | the memoizing BPE engine (encode/decode, added tokens, fork) |
| `bpe/mod.rs`, `bpe/pretoken_cache.rs` | `src/bpe/…` | merge kernels, byte remapping, pretoken memo cache |
| `pretokenize/**` | `src/pretokenize/**` | SIMD pretokenizers (`std::arch` AVX-512/AVX2/NEON, runtime-dispatched) + reference implementations |
| `hf.rs` | `src/load_tokenizer/hf.rs` | HuggingFace `tokenizer.json` loader (BPE branch) |
| `tiktoken_load.rs` | `src/load_tokenizer/tiktoken.rs` | `.tiktoken` rank-file loader |
| `input.rs` | `src/input/mod.rs` (trimmed) | `Resource`/`Document`/`DocRef` only |
| `tests/fixtures/gpt2_tokenizer.json` | upstream fixture | verbatim openai-community/gpt2 file |

## What is dropped, and why

- **SentencePiece engine** — upstream's only `portable_simd` user, which is what
  makes the whole upstream crate nightly-only. Dropping it keeps this workspace
  on **stable Rust** (edition 2024 crate-locally for let-chains).
  `byte_fallback` models get a descriptive error; serve treats that as a
  hard boot error (no runtime fallback).
- **PyO3 / numpy bindings** — would link libpython into `peregrine-serve`
  (verified absent: `ldd` shows no libpython).
- **batch / parquet / jsonl / hub layers** — peregrine feeds in-memory prompts.

Local modifications are marked with `// Local modification (vendoring):`
comments; data-dependent upstream tests (`~/data/owt_train.txt`, tiktoken rank
files, HF-cache lookups) skip gracefully when the fixture is absent — matching
upstream's own stated test policy.

## Correctness gates

- 68 vendored upstream tests (including the fancy-regex pretokenizer oracles
  and GPT-2 fixture encodes) pass on stable.
- The facade is anchored by the canonical GPT-2 check
  (`"Hello world"` → `[15496, 995]`) and round-trip tests.
- `crates/peregrine-serve/tests/tokenizer_parity.rs` asserts **id-for-id
  equality with the HF `tokenizers` crate** over an edge-case corpus
  (unicode, CJK/RTL, chat markup, contractions, empty inputs).
- Measured locally via `peregrine-serve --bench-tokenizer`: **204 MB/s vs
  6 MB/s HF (34×)** on a 12 MB mixed corpus, identical ids.

## Lint policy

This crate keeps upstream style and is exempt from the engine crates'
panic-free / clippy gates (crate-level allows in `lib.rs`; excluded from
`scripts/audit-bad-patterns.sh` — see `docs/BAD_PATTERNS.md`). The
`GigaTokenizer` facade at the bottom of `lib.rs` is the only peregrine-authored
surface.

## Re-vendoring

Upstream is not on crates.io. To update: clone upstream at the new tag, re-copy
the module table above, re-apply the marked local modifications (SP strip in
`hf.rs`/`bpe/mod.rs`, path renames `load_tokenizer::hf` → `hf`, test skip
guards), and re-run the parity + facade suites.

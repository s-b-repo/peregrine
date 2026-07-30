[« Docs index](README.md)

# Tokenizer: the vendored gigatoken fast path

`peregrine-token` is a **vendored, stable-toolchain subset** of
[marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) **v0.10.0**
(MIT — `LICENSE-MIT-gigatoken`): GB/s-class BPE tokenization. It is the
**sole runtime tokenizer** of `peregrine-serve`; the HuggingFace `tokenizers`
crate survives only as a dev-dependency parity oracle and is not linked into
the serve binary (verified: `ldd` clean, no libpython).

Measured on the reference box: **204 MB/s vs 6 MB/s HF (34×)** on a 12 MB
mixed corpus, identical ids (`peregrine-serve --bench-tokenizer <file>`).

## What is vendored

| Module | Upstream path | Notes |
|---|---|---|
| `bpe/tiktoken.rs` | `src/bpe/tiktoken.rs` | memoizing BPE engine (encode/decode, added tokens, fork) |
| `bpe/mod.rs`, `bpe/pretoken_cache.rs` | `src/bpe/…` | merge kernels, byte remapping, pretoken memo cache |
| `pretokenize/**` | `src/pretokenize/**` | SIMD pretokenizers (`std::arch` AVX-512/AVX2/NEON, runtime-dispatched) + reference implementations |
| `hf.rs` | `src/load_tokenizer/hf.rs` | HuggingFace `tokenizer.json` loader (BPE branch) |
| `tiktoken_load.rs` | `src/load_tokenizer/tiktoken.rs` | `.tiktoken` rank-file loader |
| `input.rs` | `src/input/mod.rs` (trimmed) | `Resource`/`Document`/`DocRef` only |

The `GigaTokenizer` facade at the bottom of `lib.rs` is the only
peregrine-authored surface.

## What is dropped, and why

- **SentencePiece engine** — upstream's only `portable_simd` user, which is
  what makes upstream nightly-only. Dropping it keeps the workspace on
  **stable Rust**. `byte_fallback` models get a descriptive error, and serve
  treats that as a **hard boot error** — there is no silent runtime fallback.
- **PyO3 / numpy bindings** — would link libpython into `peregrine-serve`.
- **batch / parquet / jsonl / hub layers** — peregrine feeds in-memory prompts.

Local modifications are marked with `// Local modification (vendoring):`
comments.

## Runtime behavior in serve

- One process-persistent instance behind a mutex; encode is `&mut` because
  the pretoken memo cache learns — so the cache **warms across requests**,
  and repeated chat-template prefixes encode from cache.
- Streaming decodes the id list per emitted token and handles partial UTF-8
  at token boundaries via lossy conversion (a token that doesn't lengthen the
  decoded text emits no SSE chunk).
- Boot prints `[tokenizer] gigatoken BPE active, vocab=<n>`.

## Correctness gates

- 68 vendored upstream tests pass on stable (fancy-regex pretokenizer
  oracles, GPT-2 fixture encodes).
- The facade is anchored by the canonical GPT-2 check
  (`"Hello world"` → `[15496, 995]`) and round-trip tests.
- `crates/peregrine-serve/tests/tokenizer_parity.rs` asserts **id-for-id
  equality with the HF `tokenizers` crate** over an edge-case corpus
  (unicode, CJK/RTL, chat markup, contractions, empty inputs) plus decode
  round trips.

## Lint & audit policy

The crate keeps upstream style and is exempt from the engine crates'
panic-free/clippy gates and excluded from the
[bad-patterns audit](BAD_PATTERNS.md) — its correctness gate is the parity
suite plus the vendored upstream tests.

## Re-vendoring

Upstream is not on crates.io. To update: clone upstream at the new tag,
re-copy the module table above, re-apply the marked local modifications
(SentencePiece strip in `hf.rs`/`bpe/mod.rs`, path renames, test skip
guards), and re-run the parity + facade suites. Full notes:
[`crates/peregrine-token/README.md`](../crates/peregrine-token/README.md).

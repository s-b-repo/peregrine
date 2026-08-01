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
That number is the *serve-pattern* row (one short encode per line) — the
engine itself runs several times faster on bulk input; see
[Throughput anatomy](#throughput-anatomy) below.

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

## Throughput anatomy

Upstream's headline numbers (tens of GB/s) are its **batch API fanned out
across many cores** on AVX-512/M-series hardware — ~170–390 MB/s per thread.
The vendored engine is hot-path **byte-identical** to upstream (verified by
per-file diff at v0.10.0; the only local changes are test-skip guards, module
paths, and the SentencePiece strip), so per-core speed is at parity. What
differs is everything around the engine:

1. **Call granularity.** One `encode` call per short string pays a fixed
   setup cost (SIMD pretokenizer state, span-batch machinery, added-token
   scan, output alloc) that dominates ~50-byte inputs. One call over a whole
   document runs ~3× faster on the same core, same code.
2. **Parallelism.** Upstream ships a batch layer (dropped in vendoring —
   serve encodes one prompt at a time). The facade now restores its shape as
   [`encode_batch`](../crates/peregrine-token/src/lib.rs): contiguous
   byte-balanced chunks over a **persistent pool of `fork`ed workers**
   (pre-sized caches, built once, warm across calls), id-for-id identical to
   serial, with a serial fallback if a worker dies. Small inputs
   (< 2 MB/worker) stay serial — worker construction only pays on bulk.
3. **Hardware.** Reference numbers elsewhere in these docs came from
   laptop/desktop-class AVX2 machines; upstream's came from a Zen 5 X3D,
   an M4 Max, and a 288-thread EPYC.

`--bench-tokenizer` reports all three regimes. Example (i5-1235U laptop,
2P+8E cores, 48 MB corpus, GPT-2):

| Row | What it measures | MB/s |
|---|---|---:|
| `gigatoken/line` | one `encode` per line — the serve pattern, and the HF-comparison row | 129 |
| `gigatoken/whole` | one `encode_into` call over the file — single-core engine capability | 384 |
| `gigatoken/par12 p1` | `encode_batch`, cold (includes one-time worker construction) | 275 |
| `gigatoken/par12 p2` | `encode_batch`, steady state (warm persistent pool) | **872** |

The same laptop on the **GLM-5.2 vocabulary** (154 880 tokens vs GPT-2's
50 257), 98.6 MB corpus, 27.8 M ids:

| Row | MB/s | vs GPT-2 above |
|---|---:|---|
| `gigatoken/line` | 105 | 0.81× |
| `gigatoken/whole` | 259 | 0.67× |
| `gigatoken/par12 p1` | 236 | 0.86× |
| `gigatoken/par12 p2` | **707** | 0.81× |

A ~3× larger merge table costs ~20–35 % throughput — the ratios between the four
regimes hold. For reference, Python `tokenizers` (HF) on the identical corpus and
vocabulary manages **1.71 MB/s** line-at-a-time, so the comparable `gigatoken/line`
row is **61× faster**.

For serving, none of this is a bottleneck: a request is one encode over one
prompt (µs) against a model forward (ms–s). The bulk paths exist for corpus
work and to keep the vendored subset honest against upstream's numbers.

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

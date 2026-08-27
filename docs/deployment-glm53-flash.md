[« Docs index](README.md)

# GLM-5.3-Flash (`Arch::Glm5Next`): native support and deployment

GLM-5.3-Flash is the model GitHub issue #9 asked this engine to target: a
~330B-parameter hybrid MoE whose decode working set is **~2.7× smaller per
token than GLM-5.2's**, which is what makes the issue's 2 tok/s aggregate
target arithmetic rather than aspiration. This doc records what the
architecture is, what was implemented (2026-08-27), the end-to-end conversion
pipeline, and the honest bytes/token math.

## The architecture

`zai-org/GLM-5.3-Flash`, `model_type: glm5_next` (multimodal wrapper) /
`glm5_next_text` (the stack peregrine runs; the vision tower is skipped at
import). 45 layers + 1 GLM-dialect MTP layer:

| piece | shape | peregrine implementation |
|---|---|---|
| **KDA linear attention** (34 layers) | Kimi Delta Attention: q/k/v convolved together (4-tap depthwise), **per-channel** forget gate `-5·σ(e^A_log·(f_b(f_a(x))+dt_bias))`, per-head β, sigmoid-gated RMSNorm output | `gdn.rs::kda_forward`, sharing `GdnState` with the Qwen hybrid (Glm5Next lays `lin_*` out so the geometry matches) |
| **NoPE MLA** (11 layers + MTP) | GLM-5.2's MLA with `qk_rope_head_dim = 0` — no lane is ever rotated | the existing MLA path; `rope_interleave` is a structural no-op at span 0 |
| **k-pool DSA indexer** | scores *pools* of 4 consecutive tokens (learned per-channel within-pool softmax over `compress_gate(x) + ape`), keeps `index_topk/4` pools, always appends the incomplete tail | `dsa.rs::KpoolWeights`; the `LayerKv` index stream stores `k ‖ gate` rows (`row_width = 2·hd`) |
| **mHC hyper-connections** | 4 parallel residual streams; every attn/FFN site collapses them with learned σ-weights and re-mixes through a Sinkhorn-normalized 4×4 matrix | `hyper.rs`; `forward_layer_hc` / `forward_layer_batched_hc` run the stack on `[s_n, 4·hidden]` between embedding and the mean collapse |
| **Clamped SwiGLU** | gate ≤ 10, up ∈ [−10, 10], in every dense/shared/routed MLP | `Mlp::limit`, carried through the resident path **and** the streamed rebuild (`streamed == resident` stays bit-identical — `glm5next_streamed_experts_match_resident`) |
| **MoE** | 288 routed / 8 active + 1 shared per sparse layer (42 + MTP), sigmoid router + correction bias, `routed_scaling_factor 2.5` | the GLM-5.2 router verbatim (`n_group = 1`) |
| **MTP layer 45** | GLM dialect (`eh_proj`/`enorm`/`hnorm`/`shared_head.norm`), DSA attention, sparse MoE, **no hc tensors** (conventional residual) | the existing GLM MTP path with a `full_attn` override |

Container precision (the `glm5-next` import contract in
`tools/src/import.rs::classify_glm5next`): int4 + `.qs` for the big
projections and every expert; **int8** for the gate-critical matrices (mHC
`fn`, KDA `f_*`/`g_*`/`b_proj`, the k-pool compress gate — anything whose
output passes through σ/softmax/decay); F32 for norms, convs, per-channel
vectors and the router. The FP8 (e4m3, 128×128 block scales) checkpoint is
dequantized exactly at import (`Dtype::F8E4M3`, `read_dense_dequant`).

The importer renames `model.language_model.` → `model.` so the expert-
streaming lane, `peregrine-reshard`, `peregrine-layout-reorg` and every other
tool written against `model.layers.{l}.mlp.experts.{e}.` work on this model
unchanged.

## What is deliberately not enabled (yet)

- **GPU expert tier (`COLI_GPU`)** — refused with a boot note: the CUDA
  expert kernels compute `silu(gate)·up` with no clamp. Lifting this needs
  the clamp in `backend_cuda.cu`; until then experts stay on the CPU lane.
- **Router look-ahead / predict-eval under hc** — the predictors rank the
  next layer's router over the hidden state, and mid-stack that state is the
  `[4·hidden]` stream form. Wiring a per-layer collapse for prediction is
  future work; the look-ahead is silently off for this arch.
- **Model-resident speculative decode** — falls back to greedy (KDA states
  cannot rewind a rejected draft; the serve engine's SeqKv
  snapshot/restore rollback is the supported speculation path, as for the
  Qwen hybrid).
- **Prefix cache / disk sessions / token trees / RLM** — excluded for the
  same recurrent-state reasons as `Arch::HybridGdn` (and RLM/trees refuse
  loudly).
- **MTP-layer semantics** are implemented as GLM-5.2's (collapsed hidden →
  `eh_proj(concat(enorm(embed), hnorm(hidden)))` → one DSA+MoE layer).
  The HF implementation ignores layer 45 entirely, so there is no reference
  to gate against until a serving stack (vLLM/SGLang) lands one — treat
  drafts as unverified against upstream, which is safe: every draft is
  verified by the main model, so a wrong head costs acceptance rate, not
  correctness.

## Pipeline on this box

**1. Download** (~328 GB; the HF CDN was measured at ~10 MiB/s on this box
2026-08-27, so budget ~9 hours; unthrottled by default — set `LIMIT=1M` if
the link is ever needed for something else):

```sh
/srv/m-sdc/glm53-flash-fp8/download.sh     # resumable (aria2c -c)
tail -f /srv/m-sdc/glm53-flash-fp8/download.log
```

**2. Import** (FP8 → int4/int8/F32 container). Measured 2026-08-27: 328.33 GB
→ **161.39 GB** in 1h42m (CPU-bound at ~55 MB/s), 76 108 tensors → 37 524
int4 + 273 int8 + 626 float, 347 vision skipped — the int4/int8 counts match
the architecture's expected inventory exactly.

```sh
cargo build --release -p peregrine-tools --bin peregrine-import-hf
target/release/peregrine-import-hf /srv/m-sdc/glm53-flash-fp8 /srv/m-sdd/GLM-5.3-Flash-peregrine
```

**3. Smoke + ledger.** First real measurement (single drive, cold cache,
`COLI_UNION_STATS=1 COLI_BENCH_STEPS=2 peregrine bench 1`):

```text
[workingset] one token routes 4.04 GB of experts against a 2.00 GB cache
loaded 45 layers, vocab 154880, mtp=yes, in 10.3s  (resident 5.2 GB)
B=1: 0.119 tok/s   [ledger] from disk 4.239 GB/token, re-read 4.3%
```

8.4 s/token = 4.24 GB over ~505 MB/s — exactly one SATA drive's ceiling, so
the decode is purely disk-bound and the reshard multiplies it. (GLM-5.2 on
the same box: 0.064–0.067 tok/s at 11.3 GB/token.)

For the quality gate, dump a teacher-forcing reference from transformers
(needs a version carrying `glm5_next`; the pattern is
`scripts/qwen-parity-reference.py`) and run
`peregrine flip-rate <container> --reference-json <dump> --text <corpus>`.
**Not runnable on this box**: teacher-forcing the 330B reference needs the
bf16/FP8 model resident, far beyond 32 GB RAM — it needs a big-RAM host or a
layer-at-a-time offline comparer (future work).

**4. Reshard across the four SSDs** (bandwidth-proportional, byte-verified;
the whole pipeline is `/srv/m-sdd/glm53-reshard.sh`):

```sh
target/release/peregrine-reshard --model /srv/m-sdd/GLM-5.3-Flash-peregrine \
    --out /srv/m-sdb/glm53-reshard --groups sda:530,sdb:514,sdc:530,sdd:547 --verify
# per-drive placement: experts-l*-<g>.safetensors → /srv/m-<g>/glm53-experts/
# (trunk-sda-* rides sda, the first group), then a model dir holding only the
# sidecars + model_paths.json:
#   {"paths": ["/srv/m-sda/glm53-experts", ..., "/srv/m-sdd/glm53-experts"]}
```

The dry-run plan lands 37–47 GB per drive at 24.3–25.7% expected per-token
routed share each.

**5. Serve** with the machinery GLM-5.2 earned: `COLI_IO_DEVICE_SCHED` (on
by default), `COLI_SSD_AWARE_SCHED=1`, the warm cache, `COLI_ENTROPY_ADAPT`,
and the batch engine. The GLM prompt dialect is selected automatically
(`uses_chatml_prompt` is false for this arch); verify the shipped
`chat_template.jinja` against `build_prompt`'s markup once the download
completes before trusting served chats.

## The bytes/token arithmetic (issue #9's denominator)

Routed decode working set at int4, B=1, cold cache:

```text
per expert   ≈ 3 × (2048×4096)/2 B + scales   ≈ 12.6 MB
per token    = 8 experts × 42 sparse layers    ≈ 4.2 GB   (GLM-5.2: 11.3 GB)
```

Four SATA SSDs at ~2.1 GB/s aggregate put the **no-cache, no-batching floor
at ~0.5 tok/s** — already ~7× the GLM-5.2 baseline. The measured GLM-5.2
levers apply on top and multiply: consecutive-token expert overlap (33.55%
measured on GLM-5.2's router — remeasure here), the B=16 batch union (4.4×
aggregate measured), and hot-expert residency in RAM. That is the credible
route to the issue's ≥2 tok/s aggregate target; every figure above is
arithmetic or transplanted measurement, not a benchmark of this model — the
byte ledger (`COLI_UNION_STATS=1`, now including the re-read-after-eviction
column) is the instrument to replace them with real ones.

Issue #9's full item-by-item triage lives in
[ideas-github-issue-9-2026-08-27.md](ideas-github-issue-9-2026-08-27.md).

## Test coverage

`glm5next_*` in `peregrine-model` (config parse/refusals, stack
load-decode-step-consistency, streamed==resident bit-identity, batched decode
== per-sequence, k-pool sparse-subset engagement, MTP draft, SwiGLU clamp),
`kda_*`/`hyper::` unit tests (one-call==stepwise, snapshot rewind,
doubly-stochastic Sinkhorn), `f8e4m3` decode tests in `peregrine-core`, and
the importer acceptance `the_glm53_fixture_imports_renames_dequantizes_and_generates`
(FP8 fixture → import → production loader → decode).

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

**4. Reshard across the drives** (bandwidth-proportional, byte-verified;
the whole pipeline is `/srv/m-sdd/glm53-reshard.sh`). **Weights are the
MEASURED MB/s — measure first** (`dd iflag=direct bs=8M` per drive, solo).
This deployment's scar, preserved so it is not re-earned: an equal-weight
first attempt assumed "four SSDs" and benched **slower than one drive**
(0.058 vs 0.105 tok/s O_DIRECT B=1) — because `sdc` is not an SSD, it is a
Hitachi 5400 rpm laptop HDD reading 77–87 MB/s, and an equal split gates
*every* token on it. Same lesson deployment-glm52.md records ("a stripe
including the HDD would run the whole array at HDD speed").

```sh
target/release/peregrine-reshard --model /srv/m-sdd/GLM-5.3-Flash-peregrine \
    --out /srv/m-sdb/glm53-reshard --groups sda:470,sdb:540,sdc:85,sdd:570 --verify
# per-drive placement: experts-l*-<g>.safetensors → /srv/m-<g>/glm53-experts/
# (trunk-sda-* rides sda, the first group), then a model dir holding only the
# sidecars + model_paths.json:
#   {"paths": ["/srv/m-sda/glm53-experts", ..., "/srv/m-sdd/glm53-experts"]}
```

With measured weights the plan lands 28.1/32.3/5.2/34.4% per-token routed
share (the HDD carries 7.9 GB), balancing every device at ~2.6 s of reads
per token — a ~0.39 tok/s B=1 ceiling before caching and batching. GLM-5.2's
five-way split also uses `/srv/model600p` (nvme, 669 MB/s); it sits at 21 GB
free, so GLM-5.3 skips it — a capacity-capped ~`nv1:130` group there is a
possible +4% if that drive is ever cleared.

**5. Serve** with the machinery GLM-5.2 earned: `COLI_IO_DEVICE_SCHED` (on
by default), `COLI_SSD_AWARE_SCHED=1`, the warm cache, `COLI_ENTROPY_ADAPT`,
and the batch engine. The shipped profile is `scripts/serve-glm53-flash-cuda.sh`.
The GLM prompt dialect is selected automatically (`uses_chatml_prompt` is
false for this arch); verify the shipped `chat_template.jinja` against
`build_prompt`'s markup once the download completes before trusting served
chats.

## Measured CUDA profile

The retained profile combines device-side fused reduction, fused shared/dense
residency for row-count 1, the four-drive concurrent stream, a 6 GiB warm cache,
and MTP disabled. Measurements are tok/s; B=16 values are aggregate.

### 2026-10-08: five SSDs, and the cache policies finally measured

`scripts/reshard-glm53-5ssd.sh` re-striped the container across **five** SSDs
(the two new ones: `sdc`, formerly the Kali drive, and `sde`, formerly a Windows
install — both wiped and formatted ext4 on the user's instruction), weighting each
device by its **measured** O_DIRECT read bandwidth rather than a carried-over
constant. Every tensor was byte-verified before the stage was accepted
(`verification passed (byte-identical)`), and the new container lives beside the
old one at `/srv/m-sdd/GLM-5.3-Flash-5ssd` with per-device dirs named
`glm53-experts-5ssd`, so the three-drive container is untouched and still serves.

| device | measured read (O_DIRECT, 8M, median of 12) | spread |
|---|---:|---:|
| sda Kingston A400 | 434 MB/s | **307 %** |
| sdb Acer RE100 | 537 | 1 % |
| sdc Crucial 120G | 562 | 0 % |
| sdd Crucial 1T | 561 | 5 % |
| sde Seagate 120G | 551 | 0 % |

Interleaved A/B, 6 decode steps, CUDA profile, aggregate tok/s:

| container | median | runs | GB/token from disk | slowest device's learned rate |
|---|---:|---|---:|---:|
| 3 SSD | 0.187 | 0.182 / 0.187 / 0.197 | 3.139 | 185 MB/s |
| **5 SSD** | **0.227** | 0.207 / 0.227 / 0.232 | 3.139 | **379 MB/s** |

**+21 %**, 3 of 3 reps, against an 11 % spread. Bytes per token are identical —
the model did not change — so this is bandwidth, and the mechanism is in the
per-device rates: with five drives *every* device sits above 379 MB/s, where with
three, one was at 185 MB/s and gated every token.

Note the 21 % against 2.65/1.47 = **+80 %** of raw array bandwidth. The missing
59 % is the engine-side inefficiency measured on 2026-10-06 and still the largest
open gap on this box: it sustains roughly a third of what the array can deliver.

**Cache size, the first policy win that is not noise.** `COLI_ECACHE_GB=14` takes
the bytes a token reads from 3.139 to **2.704 GB** (−14 %; every CUDA run of its
arm reproduces both numbers exactly, including on a freshly rebuilt binary) because one token routes 4.04 GB and the cache now holds ~1.3
passes instead of ~0.6. The tok/s gain (+4 %) is inside this box's noise; the byte
saving is not.

**`COLI_CACHE_LFRU=1` is a measured loss, and the literature is on the other side
of it.** FlashMoE (arXiv 2601.17063) reports a frequency+recency replacement policy
beating LRU/LFU by **51 % hit rate** on an SSD-offloaded MoE; here it *adds* 14 %
to the bytes (3.581 vs 3.139 GB/token), because `tier::lfru_score` ranks by
`heat << 8 | (255 - age)` and one routing-frequency count outweighs every
recency advantage — right in principle, wrong when `heat` is sparse relative to a
cache holding several passes. `COLI_JOINT_EVICTION=1` changed no bytes. Both stay
off, now for a recorded reason rather than a default.

Raw runs: [`bench-data/2026-10-08-5ssd/`](../bench-data/2026-10-08-5ssd/).

### Re-measured 2026-10-06: the router look-ahead is a net loss here

The shipped profile ran `COLI_ROUTER_LOOKAHEAD_N=64`. On the current **three-SSD**
stripe (`GLM-5.3-Flash-4ssd`, the HDD removed) it no longer pays: interleaved
arms, 6 decode steps, CUDA profile, aggregate tok/s.

| Arm | B=1 (median of 4) | B=8 (median of 2) | B=16 (median of 3) |
|---|---:|---:|---:|
| `N=64` (was shipped) | 0.095 | 0.193 | 0.275 |
| `COLI_ROUTER_LOOKAHEAD=0` | **0.163** (+72%) | **0.308** (+60%) | 0.271 (no difference) |
| spread of the shipped arm across its reps | 14.8% | 17.7% | 34.5% |

The shipped profile **after** this change, B=1, three runs: 0.244 / 0.148 /
0.246 aggregate tok/s, median **0.244** — 2.6× the N=64 median above, at
3.139 GB/token from disk. The box is noisy (0.148 and 0.246 back to back on the
same configuration), so quote the median and the count.

Raw runs and every knob that was measured and *not* acted on are in
[`bench-data/2026-10-06-lookahead/`](../bench-data/2026-10-06-lookahead/).

The look-ahead's window **pre-streams every predicted candidate**
(`COLI_PREFETCH_WARM_PATHS` defaults to "all"), and each device has exactly one
ring (`device-pure claims on (3 devices, … one ring per device)`), so a
speculative read sits in the same FIFO as the read the token is waiting for. The
ledger prices the bytes at 12.9% of disk traffic; the queue delay is the larger
term. Turning it off costs the cache some warmth — re-read-after-eviction rises
from 1.8% to 11.3% of disk traffic at B=1 — and that is more than repaid.

`COLI_PREFETCH_HINT_PATHS=64` is the cheap half of the same idea: a page-cache
`fadvise` rather than a device read. It is suppressed under O_DIRECT, so on this
lane it is inert rather than harmful; it earns its place on a buffered lane.

**This does not generalise.** The previous claim on this file — "N=64 is the
measured pick on this box (+14% over off)" — was measured against a *four*-drive
stripe that included a 5.4k RPM HDD, and it does not reproduce on three SSDs.
Re-measure per container.

### Two open findings from the same session

- **The warm cache serves nothing at B>1.** At B=16 the ledger prints
  `cache-served 0.0%` while `re-read (evicted) 22.0% of disk traffic`, and
  `COLI_ECACHE_GB` of 6, 12 and 18 GB produce byte-identical ledgers. A cache
  that is filled, evicts, and never serves is worse than no cache: it reserves
  6–18 GB of RAM. Not diagnosed here — flagged because 22% of the batched lane's
  disk traffic is the size of the win.
- **The io_uring path is ~22% below `dd` on the same drive.** `dd iflag=direct
  bs=8M` measures 370/536/560 MB/s on sda/sdb/sdd; `iobench` through the engine's
  own reactor measures 0.43–0.46 GB/s on sdd regardless of block size (1–64 MB) or
  ring count (1–16). During decode the three drives together sustain ~0.6 GB/s of
  useful bytes against a ~1.3 GB/s ceiling for this path.

### The retained CUDA profile (older arms, four-drive stripe)

| Arm | Result | Decision |
|---|---:|---|
| Original B=1, one step | 0.574 | baseline |
| Fused reduction B=1, one step | 0.709 | retain |
| Best new B=1, one step | 0.812 | retain |
| New sustained B=1, three steps | 0.154 | production shape |
| Sustained without shared/dense tier | 0.118 | reject; about 30% slower |
| New B=16 | 0.299 | diagnostic |
| Original B=16 | 0.335 | use CPU shared path above B=1 |
| MTP depth 1 | 0.106, zero accepted | reject |
| 12 GiB expert cache | 0.131 | reject versus 0.154 at 6 GiB |

`COLI_GPU_DENSE_MAX_BATCH=1` keeps the B=1 fused shared path while the CPU
shared path serves wider batches. `COLI_CUDA_FUSED_REDUCE=1`,
`COLI_GPU_DENSE=1`, `COLI_ECACHE_GB=6`, and `COLI_DRAFT=0` remain independent
and compose in the same process. `scripts/bench-glm53-cuda-matrix.sh` reruns
the matrix in fresh, serialized processes; performance arms must not overlap.

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

[« Docs index](README.md)

# Model format & artifacts

What a model directory must contain, how weights are encoded, and every
sidecar artifact the runtime reads or writes.

## Model directory layout

`Model::load` requires exactly two things:

| File | Required | Notes |
|---|---|---|
| `config.json` | yes | GLM-5.2 / DeepSeek-V3-class config (fields below) |
| `*.safetensors` (≥ 1) | yes | every file with the `safetensors` extension, loaded in **lexical filename order**; there is no `model.safetensors.index.json` support |
| `tokenizer.json` | serve only | required by `peregrine-serve` (gigatoken BPE); the stdio engine works on raw token ids without it |

Optional sidecars are consumed automatically and best-effort (missing,
malformed, or stale files are silently ignored): `automaton.json`,
`macrostates.json`, `route_stats.json`, `tiers.json`, `schedule.json`,
`plan.json` — see [the artifact table](#runtime--offline-artifacts).

`peregrine build <dir>` writes a minimal valid directory (tiny synthetic
GLM-5.2-shaped model, fixed seed — the executable spec of the format). Note it
includes no `tokenizer.json`, so it serves over stdio but not HTTP.

## Weight naming scheme (GLM-5.2)

Top level: `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`.
Per layer `model.layers.{i}.`:

- Norms: `input_layernorm.weight`, `post_attention_layernorm.weight`
- MLA attention: `self_attn.q_a_proj.weight`, `self_attn.q_a_layernorm.weight`,
  `self_attn.q_b_proj.weight`, `self_attn.kv_a_proj_with_mqa.weight`,
  `self_attn.kv_a_layernorm.weight`, `self_attn.kv_b_proj.weight`,
  `self_attn.o_proj.weight`
- Dense layers (`i < first_k_dense_replace`): `mlp.gate_proj.weight`,
  `mlp.up_proj.weight`, `mlp.down_proj.weight`
- MoE layers: `mlp.gate.weight` (router, **must be F32**),
  `mlp.gate.e_score_correction_bias`,
  `mlp.shared_experts.{gate,up,down}_proj.weight`, and
  `mlp.experts.{e}.{gate,up,down}_proj.weight` for each routed expert

Every quantized weight `W` has a sibling F32 scale tensor **`W.qs`**.

Optional sub-models are auto-detected:

- **MTP head** — an extra layer at index `n_layers` plus `eh_proj.weight`,
  `enorm.weight`, `hnorm.weight`, `shared_head.norm.weight`.
- **DSA lightning indexer** — `self_attn.indexer_projections.{wq_b,wk,weights_proj}`
  and `self_attn.indexer.k_norm.{weight,bias}`.

In streaming mode, routed-expert tensors are presence-checked at load and then
read per token over io_uring; the `.mlp.experts.` substring is also how the
loader sums routed bytes for the streaming-mode heuristic.

## QT quantization formats

The format is **inferred from byte counts, never declared**. For a weight of
shape `[O, I]` with `ns` scale floats in `W.qs`:

| fmt | Variant | Payload bytes | Scales |
|---|---|---|---|
| 0 | F32 | no `.qs` sibling — rejected on the quantized expert path | 0 |
| 1 | Int8 (per-row) | `O·I` | `O` |
| 2 | Int4 (per-row) | `O·⌈I/2⌉` packed nibbles | `O` |
| 3 | Int2 | the remaining case (nominally `O·⌈I/4⌉`) | `O` |
| 4 | Int4 grouped | same payload as fmt 2 but `ns > O` | `O·⌈I/gs⌉` |

- Int4 nibbles are biased `+8` into `[0,15]`; even index in the low nibble.
- Scales: int8 = `amax/127` per row, int4 = `amax/7`.
- The group size `gs` is probed from `[16, 32, 48, 64, 96, 128, 192, 256]`
  (finest first); the grouped scale layout `sc[o·ng + g]` matches
  `convert_fp8_to_int4.py --group-size`.
- Supported dtypes in headers: `F32`, `BF16`, `F16`, `U8`/`I8`.

## safetensors header extensions

peregrine hand-rolls its safetensors reader (pread-based, io_uring, flat RSS —
no mmap) and understands two extensions in each tensor's header entry:

| Keys | Semantics |
|---|---|
| `"compression": "zstd"` + `"uncompressed_nbytes": <n>` | tensor payload is zstd-compressed on disk; `data_offsets` describe the compressed bytes, `uncompressed_nbytes` the logical length. Reads decompress transparently. Unknown compression tags degrade to raw. Written at fixed zstd level 3. |
| `"layout": "kblock"` + `"layout_gs_bytes": <n>` | alternate group-major byte tiling (`[g][o]` instead of row-major `[o][g]`) for sequential per-group streaming; the reader auto-inverts to the kernels' native layout (`from_kblock`), byte-identical round trip. Both keys must be present; either alone is ignored. |

**Interaction note:** a checkpoint containing any compressed tensor forces
resident-expert mode — the loader prints
`[peregrine] compressed checkpoint detected — disabling expert streaming`.

## config.json fields

Main fields parsed (validated to sane ranges at load):

`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`n_routed_experts`, `num_experts_per_tok` (top-k), `moe_intermediate_size`,
`intermediate_size`, `first_k_dense_replace`, `q_lora_rank`, `kv_lora_rank`,
`qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `n_shared_experts`,
`vocab_size`, `norm_topk_prob`, `rms_norm_eps`, `routed_scaling_factor`,
`rope_parameters.rope_theta`, `eos_token_id` (scalar or array; first 8 used
as stop ids), and the DSA fields `index_topk`, `index_n_heads`,
`index_head_dim`, `indexer_types` / `index_topk_freq` /
`index_skip_topk_offset`.

Constraint: **`n_group` must be exactly 1** (GLM-5.2) — anything else is a
load error.

## Runtime & offline artifacts

All artifacts live in the model directory, are correctness-neutral, and load
best-effort. Load order is automaton → macrostates → route_stats → tiers →
**plan.json last** (a plan part wins over its standalone file).

| File | Written by | Read by | Shape (top level) |
|---|---|---|---|
| `automaton.json` | `peregrine build-automaton`, `galactic` | predictor at load | `{tag, n_layers, edges: [[layer, from, to, count]…]}` |
| `macrostates.json` | `galactic` | predictor at load | `{tag, n_layers, layers: [[set, dwell, [[exit, count]…]]…]}` |
| `routes.json` | `galactic`, `dump-routes` (any path) | `peregrine-layout-reorg` only — not the runtime | bare nested array `[forward][layer][expert_id]` |
| `schedule.json` | `peregrine-layout-reorg`, `galactic` | loader → disk-read ordering | `{version: 1, n_layers, order: [[expert_id…]…]}` |
| `tiers.json` | `galactic` (with `COLI_TIER_*_MB`) | loader — **only the `ram` list** (prefetch-warm, capped at 256 entries) | `{version: 1, vram: [[layer, expert]…], ram: […]}` |
| `route_stats.json` | `Model` at Drop (`COLI_ROUTE_STATS_PERSIST`), seeded by `galactic` | loader — warm-start | `{tag, hist, heat, coact, learn}`; `learn` is the bandit/Q policy |
| `plan.json` | `peregrine compile-plan` | loader, consumed atomically | `{version: 1, automaton?, macrostates?, schedule?, tiers?, learn?}` |
| `kernel_tuning.json` | *(planned)* — `WmmaTuner` has the (de)serializers but no code path writes/reads the file yet | — | `{version: 1, rows: […]}` |

**Config-tag guards.** The tag is
`L{n_layers}E{n_experts}H{hidden}I{moe_inter}D{first_dense}V{vocab}` (e.g.
`L5E8H128I32D3V256`). `automaton.json`, `macrostates.json`, the `hist` part of
`route_stats.json`, and the automaton/macrostate parts of `plan.json` are
rejected on tag mismatch (silently — the runtime falls back to its default
behavior). `schedule.json` is guarded by `version == 1` only; `tiers.json`
entries are validated per item.

Kill switches: `COLI_ROUTE_STATS_PERSIST=0`, `COLI_LAYOUT_SCHEDULE=0`,
`COLI_TIER_SEED=0`. Full knob table: [Configuration](configuration.md).

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

Optional sidecars are consumed automatically and best-effort — none of them can
change what the engine computes, only how fast: `automaton.json`,
`macrostates.json`, `route_stats.json`, `tiers.json`, `schedule.json`,
`plan.json` — see [the artifact table](#runtime--offline-artifacts).

**A missing sidecar is silent; a broken one is not.** An absent file is the
normal case and says nothing. A file that is present but unreadable or malformed
is still ignored — correctness never depends on one — but reports through
`note_advisory_err`, so `COLI_DEBUG=1` names the file and the reason. Until
2026-08-02 the two were indistinguishable, and a syntax error in `plan.json`
produced default behavior with nothing anywhere to explain why the artifact had
no effect. A **stale** file (one that fails its `config_tag` fingerprint) is a
third case and remains silent by design: it is a correct file for a different
model, and the guard exists precisely so mixing them is a no-op.

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
| 3 | Int2 | `O·⌈I/4⌉` | `O` |
| 4 | Int4 grouped | same payload as fmt 2 but `ns > O` | `O·⌈I/gs⌉` |
| 5 | Int3 grouped-64 | `O·⌈I/64⌉·24` **and** `ns == O·⌈I/64⌉` | `O·⌈I/64⌉` |
| 7 | Int2 grouped-64, **affine** | `O·⌈I/64⌉·16` **and** `ns == 2·O·⌈I/64⌉`, requires `⌈I/64⌉ ≥ 2` | `2·O·⌈I/64⌉` (`[scale, zero]` interleaved) |

- Int4 nibbles are biased `+8` into `[0,15]`; even index in the low nibble.
- **Int2-g64** (fmt 7) is one 16-byte plane per 64-value group — four 2-bit
  fields per byte, the same packing as int3-g64's low plane — plus **two** f32
  per group in `.qs`, interleaved `[scale, zero]`. Fields are unsigned `[0,3]`
  and the bias is the group's own zero-point, so an affine mapping of
  `[min, max]` onto four levels.

  **3.0 bits/weight effective, not 2.** The two f32 per group are 8 bytes per
  64 values — a full extra bit/weight on top of the 2-bit payload. It is still
  the smallest scheme here (int3-g64 is 3.5, per-row int4 ~4.0, so 25% below
  int4), but "2-bit" describes the payload, not the container, and the
  distinction is worth 33% of the file. Storing the scale pair as f16 would
  reach ~2.5 bits/weight; `.qs` is F32 across every format today, so that is a
  container-wide change rather than a per-format one.

  It exists because per-row **Int2 (fmt 3) is effectively ternary**: its
  `s = amax / 1` convention with a `[-2, 1]` clamp makes the `-2` level
  unreachable (it would need `|w| ≥ 1.5·amax`, impossible when `amax` is the
  row's own maximum), so one of four levels is dead in every row it can write.
  Affine grouping fixes that by construction and adds finer scales;
  `int2_g64_reaches_all_four_levels_where_per_row_int2_reaches_three` pins the
  difference.

  Two f32 per group rather than a third `.qz` sibling tensor is deliberate: a
  third sibling would take the streamed expert read from 6 regions to 9, and
  `prefetch_hint_item` returns a fixed `[(RawFd, u64, usize); 6]`. Interleaving
  keeps the whole streaming path untouched.

  **At least two groups per row are required.** At one group the
  `(bytes, scales)` pair is identical to grouped int4 (O=2, I=32, gs=16 is also
  32 bytes and 4 scales), so the format is undetectable and
  `peregrine-requantize` refuses to write it rather than emit a container that
  loads as something else. Real routed experts are 1536–5120 wide, so this
  excludes only fixtures.
- **Int3-g64** (colibrì's fmt 5) is two planes per 64-value group: a 16-byte low
  plane holding bits 0-1 in the int2 layout, then an 8-byte high plane holding
  bit 2 one bit per value. Values are biased `+4` into `[0,7]`, decoding to
  `[-4, 3]`, with one f32 scale per group — 3.5 bits/weight effective, 12.7%
  smaller than int4 once the scales are counted. The scale *cardinality* is part
  of the discriminator, not a consequence of it, because a narrow tensor's int3
  size can collide with a row format's. `quant_i3_g64` is **byte-identical to
  colibrì's encoder**, frozen by `int3_g64_bytes_match_colibri_byte_for_byte`
  against a vector that engine produced — which is how peregrine can read a
  container colibrì wrote. That test caught a rounding mismatch (`f32::round`
  rounds halves away from zero; `np.rint` rounds to even) that a round-trip test
  structurally cannot see, since each encoder decodes its own output correctly.
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

## Mixed-precision containers

A container may hold **different formats per expert** — `QtInfo::detect` runs
per tensor and is re-detected per expert per forward, so nothing in the loader
assumes uniformity. `peregrine-requantize --tier-hot-frac` produces one, keeping
the hottest experts of each layer at a higher precision.

Read the economics before building one: tiering saves *disk* in proportion to
expert **count** but *per-token bytes* in proportion to routing **frequency**,
and frequency is what makes an expert hot. On a skewed trace, keeping the hot
25% at int4 measured **140% of an all-int2 checkpoint** — the accuracy hedge
gave back 40% of the byte saving. The tool prints that ratio before converting.

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

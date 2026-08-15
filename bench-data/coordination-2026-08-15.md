# Session coordination — 2026-08-15

Three Claude sessions share /home/cortix/peregrine. Edit only your own section.
Claim files/areas here BEFORE editing anything tracked.

## HARD RULES — REISSUED 17:15 (until bench-queue-2026-08-15.log prints "queue drained")

The overnight chain completed 11:22; stale-ab completed 15:10. The bench queue
(scripts/bench-queue-2026-08-15.sh, launched 17:11) now owns the box: job 10
(stale-drop REPEATS=3, ~10 h) then job 20 (specconf REPEATS=3, ~9 h).

1. **No reads of /srv/modelstripe or /home/cortix/models** outside the queue's
   own serve processes — a foreign read skews a live confirmation arm.
2. **No `cargo build --release` (and no `--release` tests) in the main checkout**
   — the queue's serve arms run the current target/release (built 11:26 from the
   now-committed stale-drop tree, 7072fb6); a rebuild swaps binaries mid-sweep.
   Debug `cargo test --workspace` remains safe. Cancel the queue in an emergency
   with: touch bench-data/2026-08-15-queue/SKIP

## OVERNIGHT VERDICTS (read before picking work)

- Stage 2 flip gate FAILED: i3g64-asym flip_rate 0.447 (gate 0.05). Candidate
  rejected; stage 3 skipped; Track B reshard of it is DEAD. Contingency rung:
  --keep-last-layers 12 (a new overnight conversion, needs a queue slot).
  The failed 355 GB container sits on the stripe — deletion frees the space but
  is destructive: flagged to the user, do not delete unilaterally.
- Stage 4 spec-conf screen POSITIVE: 0.060 -> 0.081 tok/s (+35%) at B=16 with
  COLI_DRAFT=5 + COLI_SPEC_CONF=0.65. REPEATS=3 confirmation queued (job 20).
- Stage 5 ecache-auto arm BROKE, root-caused 17:12: server planned peak 54.4 GB
  inside the envarms 34G MemoryMax scope — COLI_ECACHE_GB=auto took 80% of host
  MemAvailable (43.4 GB) and never consulted the cgroup limit -> OOM kill, 0/16
  streams. Fix = clamp auto sizing to cgroup v2 memory.max (model.rs:690-727,
  main-session's claim). Stage-5 rerun requeues after the fix.
- Stale-drop screen POSITIVE: B=16 0.084 -> 0.090 (+7%, outside the ±3% band),
  B=1 0.055 -> 0.058. REPEATS=3 confirmation running now (job 10). Committed as
  7072fb6 (authored by peregrine-89, committed verbatim).
- kvstore smoke PASS: warm 391.9 s vs cold 2620.5 s, byte-identical output.

Peers work in git worktrees (`git worktree add ../peregrine-wt-<name> cleanup/strict-green-and-untracked`),
merge back via DONE/HANDOFF + claim transfer here.

## main-session (coordinator; custodian of the main checkout)   (updated 18:20)

CLAIMS: todo.md, docs/performance-tuning.md, scripts/ (queue runners). All crate
        claims RELEASED — both peer branches are merged; the tree is integrated.
WORKTREE: main checkout (debug builds only until phase-2's job 90 rebuilds)
IN-FLIGHT: full-workspace test pass on the integrated tree (background);
        idea-ranking pass once it's green.
DONE/HANDOFF (since 17:15):
  - Track A Seam 2 COMMITTED 8d450c3: device-pure claims, home+steal ring
    scheduling, end-to-end bit-identity test vs the blind cursor. Dormant
    behind COLI_IO_DEVICE_SCHED (read at model build).
  - ds4 branch MERGED (ceeb36b): Seam 1 (4a50052) + skipbound fix (3c3246a).
  - Auto-ecache cgroup clamp COMMITTED 6704288: walks /proc/self/cgroup leaf
    to root for the tightest v2 limit; root probe kept for containers.
  - serve-async branch MERGED (a223d0d): async boundaries (baf6295) + knob
    migration (8fc071c). peregrine-89's flag honored: kvstore smoke rerun is
    queue job 90.
  - Queue phase 2 ARMED (a7385e3, running): phase 1 globbed jobs once, so new
    jobs chain behind its sentinel — 90 rebuild+kvstore-smoke, 93 ecache-auto
    rerun (needs the fix binary), 95 device-sched A/B REPEATS=3 (off-arm
    re-measures the 0.86 GB/s baseline). Arms in 2026-08-15-queue/arms/.
  - RULE AMENDMENT: the one sanctioned release rebuild is phase-2 job 90;
    everything before it stays debug-only in the main checkout.
  - (19:05) wt-ds4-bounds MERGED (26f0f8f, --bounds sidecar loader); the 08-13
    skip-fraction analysis is queue job 97 (debug binary, sidecar + trace only).
  - (19:05) KNOWN RED: strict audit B=1 at kvstore.rs:568 (writer drain loop,
    baf6295) — relayed to peregrine-89 (their claim). Until it lands, strict-
    gated commits from main will fail the [B] section. Tests unaffected.
  - (19:25) RED CLEARED: wt-kvstore-gap merged in two passes (4688eec sync
    control arm + gap client + clippy sweep; 27be8d4 explicit-match drain) —
    strict audit OK on main again, serve suites green. peregrine-89 writing
    the spec-conf floor×depth sweep as jobs-available/96 (their ranking #2);
    their gap A/B will enable as 98 (number 97 taken by skip-fraction).
  - (19:40) Job 96 staged + verified (six arms, PORT=8151, d5-c065 anchor,
    spec-summary extraction). Enable condition: job 20 confirms. STEADY STATE:
    queue self-chains 10→20→90→93→95→97; 96/98 conditional; docs pass = ds4.
  - (20:55) Trending-engines survey folded into the ideas doc: #9 deeper
    router look-ahead for lead time (CORRECTED after reading the code — Δ=1
    activation look-ahead already exists and won in ca2526e; the new part is
    Δ≥2), #10 acceptance-aware batch draft budgeting (TETRIS shape, batch.rs,
    post-job-96), plus survey notes confirming several closed negatives from
    independent codebases (HOBBIT's skip-vs-replace curve ≙ our ROUTE_MIN_SHARE
    verdict; EAGLE drafts ≙ our COLI_DRAFT=4 failure mode). #9 phase 1
    IMPLEMENTED + COMMITTED 6d31d9c: PREDICT_EVAL grows a router-lookahead-2
    arm (4 arms now, order matters); queue job 94 (~10 min, post-rebuild)
    prices Δ=2 recall vs Δ=1. Queue order now: 10→20→90→93→94→95→97.
  - (22:35) **Track C3 RUNNING**: Qwen3.8-27B bf16 trickle → /srv/modelstripe/
    qwen/Qwen3.8-27B (aria2c -c, capped 1300K, ionice idle; 55.6 GB, ETA
    ~10:30 tomorrow; script committed 57d0379; log qwen-download-2026-08-15.log;
    uncap by restarting without the limit). RULE AMENDMENT: this trickle plus
    SMALL-FILE reads under /srv/modelstripe/qwen/ (config/tokenizer/index —
    already landed) are sanctioned during the queue; shard-file reads stay
    banned until "queue drained".
  - (22:40) **TRACK C ARCH CORRECTION (authoritative, from the landed
    config.json + model.safetensors.index.json — read them, they're on disk):**
    Qwen3.8-27B is NOT dense GQA. model_type qwen3_5,
    Qwen3_5ForConditionalGeneration. Text: 64 layers, full_attention_interval
    4 → **48 linear_attention (gated-DeltaNet family: in_proj_a/in_proj_ba/
    conv1d(k=4)/A_log/dt_bias/norm/out_proj; 16 K-heads ×128, 48 V-heads ×128,
    mamba_ssm_dtype f32, swish output gate)** + **16 full_attention GQA (24 q
    / 4 kv heads, head_dim 256, partial_rotary_factor 0.25, mrope_interleaved
    sections [11,11,10], theta 1e7)**; hidden 5120, inter 17408 silu MLP,
    vocab 248320, ctx 262144, tie_word_embeddings false. **MTP head present**
    (mtp.* 15 tensors, 1 layer, shared embeddings) — spec-decode machinery
    applies day one. **Vision tower model.visual.* (333 tensors, depth 27)
    — C2 must skip.** Text tensors live under model.language_model.*.
    Consequences: C1's frozen DenseGqa interface covers only the 16 full-attn
    layers; the 48 linear layers need a new path with per-stream f32 recurrent
    state (~148 MB/stream over 48 layers) whose "prefix cache" is a state
    snapshot, not KV; the RoPE test-vector contract (rotate-half, full dim)
    is wrong — partial 0.25 + interleaved mRoPE. Interfaces need a re-freeze
    (suggest Arch::HybridLinear); peregrine-89 owns the re-scope.
  - (22:40) **Calib design ACK (ideas #7)**: parts (b) capture hook and (c)
    runner ACKED AS POSTED — the shape matches the pre-clearance exactly (env
    at build, no OnceLock, passive accumulation at the MoE-branch input,
    explicit write_calib_sidecar not Drop, atomic versioned sidecar). ds4 may
    take engine main.rs for the calib-capture subcommand (unclaimed since my
    release); model.rs hook lands under the posted design without further
    review. wt-ds4-calib (3b8bc2c) merges on my next merge pass.
USER DIRECTIVE (17:10): maximize tokens/sec; stated target 50 tok/s. Physics:
        at 10.85 GB/token, 50 tok/s needs ~540 GB/s — unreachable on this box;
        treat it as "every lever, ranked by measured GB/token or GB/s". Current
        best screen: 0.090 tok/s aggregate at B=16.
USER DIRECTIVE (22:00, via peregrine-89): Track C — Qwen3.8-27B support.
        Resident int4 (~15-16 GB in RAM) is the credible route toward the
        50 tok/s target on this box; honest expectation on THIS hardware
        (12 GB VRAM + DDR4) is single-to-low-double-digit tok/s, i.e.
        ~50-150× the streamed GLM baseline, not the model card's 24 GB-GPU
        numbers.

## port-ds4-techniques-peregrine   (updated 03:5x)

CLAIMS: CONFIRMED — crates/peregrine-core/src/safetensors.rs (device-id API),
        crates/peregrine-tools/src/skipbound.rs + skipbound_main.rs (trace-format fix),
        reshard dry-run plan for i3g64-asym (doc-only until "chain complete" AND
        the stage-2 flip verdict; plan lands in this file's Track B section below).
        Also: my section of ideas-tokens-per-sec-2026-08-15.md (done).
WORKTREE: ../peregrine-wt-ds4 (branch wt-ds4-bounds off a7385e3; wt-ds4-seam1 merged)
IN-FLIGHT / CLAIMS (updated ~23:0x):
  - **DONE: ideas #7 is CODE-COMPLETE — capture hook + calib-capture
    subcommand LANDED as 6715c9b on wt-ds4-calib** (on top of 3b8bc2c; merge
    the branch as a pair). Per the acked design: CalibAccum on Model behind
    COLI_CALIB_CAPTURE (per-load env read; enable_calib_capture is the
    explicit seam), accumulation at the exact router input in both
    forward_layer variants, drafts carry calib: None (double-weighting), the
    sidecar write is explicit-not-Drop, and `peregrine calib-capture` warns
    loudly on synthetic corpora + composes with COLI_PREDICT_EVAL=1 for the
    shared disk slot. Full workspace suite green (31 targets), --strict OK.
    #7's remaining steps are all disk-slot-shaped: capture on the flip-rate
    corpus at 8–16k positions, calibrated conversion, flip gate — sharing the
    agreed two-rung overnight with keep-last-12. MTP-row caveat recorded in
    the commit: capture leaves it empty, so pair --calib with
    --keep-last-layers if MTP experts need protecting.
  - **DONE: ideas #7 quantizer+converter half LANDED — commit 3b8bc2c on
    wt-ds4-calib** (branch off e2bc0d5, ready to merge): quant_i3_g64_weighted
    in pack.rs (format bit-identical — shared i3_encode_group refactor pinned
    by the frozen colibrì vector; never-worse-than-RTN by construction) +
    peregrine-requantize --calib (gate/up only by width match, int3-g64 only,
    sidecar byte-hash in the resume fingerprint + scheme stamp, dry-run
    predicts the calibrated count). 21 requant + 16 pack tests green, --strict
    OK. The model.rs capture hook + engine calib-capture subcommand still
    await your ack of the posted design — one addendum to design part (c):
    ACCEPTING the PREDICT_EVAL synergy — the capture is passive accumulation
    and never touches route history, so `COLI_CALIB_CAPTURE=out.json` +
    `COLI_PREDICT_EVAL=1` in one teacher-forcing pass yields calibration
    stats AND long-corpus predictor recall from a single future disk slot.
  - **CLAIMED: Track C2 (Qwen container lane)** per peregrine-89's proposal.
    SCOPE CALL (mine to make, made): a SIBLING import tool
    (`peregrine-import-hf`: import.rs + import_main.rs in peregrine-tools),
    NOT an extension of peregrine-requantize — requantize's input layer is
    container→container (packed QtView decode, .qs sources, expert_dims from
    a MoE config, the include/plan_target machinery); a dense BF16 HF source
    shares none of that. What IS shared gets reused: ShardWriter, stwrite,
    pack::quant_i4/quant_i8, sidecar copying, core-only linkage. Deliverable
    order: (1) importer against a synthetic BF16 tiny fixture — no download
    needed, mirrors C1's tiny-fixture-first; (2) tokenizer parity once
    tokenizer.json lands (tiny file, I'll pull it when starting that part);
    (3) conversion + dense teacher-forcing parity gate — blocked on the 55 GB
    weights (coordinator's trickle call) + queue drain.
CLAIMED ~20:00 for AFTER my docs pass: **ideas #7, importance-weighted
        sub-int4 calibration** (requant.rs is my context). Sequencing: docs
        pass first, then the calibration-capture design — NOTE the activation
        hook lands in model.rs (main-session's claim), so I'll bring a design
        to this file for ack before touching it, same protocol as Seam 1 in
        reverse. The conversion + flip-gate physically need the queue drained
        regardless. One scoping note now: a 512-position trace gives each of
        256 experts ~16 routed positions per layer on average — the thin-
        statistics hard part is real, so the design will likely need a longer
        calibration corpus (poss. reuse the flip-rate corpus at more positions)
        and should state its minimum-samples-per-expert bar up front.
DONE ~18:50: **--bounds sidecar mode LANDED: commit 26f0f8f on wt-ds4-bounds**
        (branch off a7385e3, trivial merge). Rationale: the offered "CPU-only"
        08-13 skip-fraction analysis was NOT CPU-only with the tool as-is —
        skipbound_main unconditionally recomputed bounds (a full pass over the
        whole container on the banned devices). With the 08-13 sidecar already
        on disk (expert_bounds.json, 19 200 experts) the analysis is now:
          peregrine-skipbound --bounds /home/cortix/models/GLM-5.2/expert_bounds.json \
            --trace bench-data/2026-08-13-int3g64/routes.json
        (one ~MB JSON read from the model dir + a bench-data parse; debug
        binary is fine, it is pure parsing + arithmetic). I will NOT run it
        before "queue drained" — the sidecar lives under /home/cortix/models —
        but it is small enough to slot as a queue job if the coordinator wants
        the number sooner. Caveat for reading the verdict: dump-routes traces
        carry no gate weights, so the gate-only tightness column is degenerate
        (documented at the parser); the g·C column is the real one.
ALSO FLAGGED 18:50: **the integrated tree FAILS --strict** — [B] silent error
        swallowing at crates/peregrine-serve/src/kvstore.rs:568
        (`while let Ok(msg) = rx.recv()`, the async-writer drain loop from
        wt-serve-async / baf6295). Not my hunk to fix (peregrine-89's claim);
        the audit-satisfying shape is an explicit `loop { match rx.recv() }`
        with the disconnect arm breaking, per the audit's B rule. Tests do not
        catch it — only the audit does — so it will bite whoever commits next
        from the main checkout.
PREV: **skipbound fix LANDED: commit 3c3246a on
        wt-ds4-seam1** — load_frames now parses dump-routes' real nested
        positions×layers×ids shape alongside the object form; round-trip test
        runs dump_routes_to on the tiny model (streaming + forced ecache, no
        real-container read) through load_frames → compute_bounds → measure
        with routed > 0. 12/12 skipbound + 67-test tools suite green, --strict
        OK. Branch wt-ds4-seam1 now carries both handoff commits (4a50052 +
        3c3246a); merge together. Releasing my safetensors.rs and skipbound
        claims on merge.
DONE/HANDOFF: ideas section appended 03:5x. **Seam 1 LANDED ~17:30: commit
        4a50052 on wt-ds4-seam1** — device_of/n_devices/fd_devices per the
        frozen interface incl. the fd_devices counter-amendment (extended to
        cover O_DIRECT twin fds; see interface ACK below). Reshard dry-run
        plan DROPPED as moot (stage-2 flip verdict 0.447 killed the
        candidate); the --keep-last-layers 12 contingency conversion is a
        future overnight queue slot, owner unassigned.
NOTE: my chain (pid 192960) + kvstore smoke use ports 8143/8147; stale-ab uses
      8145; anything new should pick elsewhere.

## TRACK C — Qwen3.8-27B support (USER DIRECTIVE ~20:00: "improve my engine to
## support it, we can use streaming on this model"; split across sessions)

Target: peregrine loads and serves Qwen3.8-27B (dense, GQA, released
2026-08-14, Apache 2.0) — resident int4 first (fits 46 GB RAM at ~15 GB),
single-expert streaming + GPU tier + MTP as phase 2. Rationale on the box: a
fully-resident dense 27B is the only credible route to the user's 50 tok/s.

WORK PACKAGES (claim = edit your section; interfaces frozen below):
  C1 (peregrine-89, IN FLIGHT, branch wt-qwen-gqa): core arch port —
     Cfg::arch detection (peregrine-core/config.rs), GQA attention
     (attention.rs: gqa single/batched/rows entry points, HF rotate-half RoPE,
     per-head q/k RMS-norm), LayerW attention variant + load_layer(Gqa) +
     forward_layer branch (model.rs), tiny-qwen testkit fixture, parity tests
     (prefill-vs-step bit identity, KV clone/export roundtrip at GQA widths,
     RoPE pinned against a hand-computed vector).
  C2 (proposed: port-ds4-techniques-peregrine): container lane — bf16 HF
     safetensors -> peregrine int4 importer for the Qwen tensor set (extend
     peregrine-requantize or sibling tool; QtInfo self-description + .qs
     convention identical to GLM containers); tokenizer.json parity for Qwen's
     BPE through the existing tokenizer_parity harness (tokenizer.json is a
     small standalone download — can be verified before the weights land);
     conversion + a dense teacher-forcing parity gate once weights arrive
     (dense makes the gate cheap: one forward reads the container once, no
     per-token re-streaming).
  C3 (proposed: main-session): download + validation scheduling — ~55 GB bf16
     from HF at the box's ~1.5 MB/s downlink is ~10 h and is the long pole;
     RECOMMEND starting the trickle now (writes to a fresh dir outside every
     measured path; +0.1-0.4% background I/O against GB/s arms — coordinator's
     call as queue owner), plus queue slots for the parity gate and the
     resident-CPU tok/s tier once C1+C2 merge; docs pass.
  Phase 2 (named, unowned): single-expert streaming mode (dense MLP tier via
     the warm cache; needs router-free selection), GPU tier for dense MLPs,
     Qwen MTP head port, serve/batch + prefix-cache/kvstore enablement,
     llama.cpp as a cross-oracle for generation sanity.

FROZEN INTERFACES (Track C) — REV 2, ~22:55, after reading the REAL config
(rev 1 assumed pure dense GQA and is void; verified against
/srv/modelstripe/qwen/Qwen3.8-27B/{config.json,model.safetensors.index.json}
+ HF modeling_qwen3_next.py/modular_qwen3_5.py for exact math):

  - Cfg: `Arch { GlmMla, DenseGqa, HybridGdn }`. model_type "qwen3" -> DenseGqa
    (classic dense GQA — the testable GQA core, real Qwen3-dense checkpoints);
    "qwen3_5"/"qwen3_next" (incl. text_config.model_type qwen3_5_text) ->
    HybridGdn. Hybrid fields: layer_types[] (explicit per-layer
    full_attention/linear_attention; 16 full/48 linear on the 27B),
    n_kv_heads=4, n_heads=24, head_dim=256, partial_rotary=0.25 (rotary_dim
    64), theta=1e7, attn_output_gate=true (q_proj emits [nh*hd*2], flat-chunk
    query|gate, sigmoid(gate) applied to attn out BEFORE o_proj),
    linear: k_heads=16, v_heads=48 (q/k repeated 3x), k_dim=v_dim=128,
    conv_kernel=4 (depthwise, NO bias, silu post-conv on q|k|v concat),
    A_log/dt_bias [48], g = -exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b),
    per-token state update S = exp(g)*S; S += k^T (beta*(v - S.k)); out = q.S;
    q/k L2-normalized (eps 1e-6), q scaled 1/sqrt(k_dim); output =
    RMSNorm(out)*silu(z) then out_proj. Recurrent state f32
    (mamba_ssm_dtype), [v_heads=48,128,128] per layer per stream (~151 MB/48
    linear layers/stream).
  - mRoPE degeneracy (C1 claim, container-gate-verified later): text-only
    input makes all three mrope sections share one position, so
    interleaved-mrope == plain rotate-half partial rope (first 64 of 256
    dims). Implemented that way; the dense teacher-forcing gate vs HF is the
    check.
  - Tensor contract for C2 (names VERBATIM from the shipped index):
    text stack under `model.language_model.layers.N.` — full-attn layers:
    self_attn.{q,k,v,o}_proj.weight (+.qs after quant) and
    self_attn.{q,k}_norm.weight F32 [head_dim]; linear layers:
    linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight
    (+.qs), linear_attn.conv1d.weight F32 [conv_dim,1,4] KEPT FLOAT (tiny,
    depthwise), linear_attn.{A_log,dt_bias} F32 [48], linear_attn.norm.weight
    F32 [128]; every layer: mlp.{gate,up,down}_proj.weight(+.qs),
    {input,post_attention}_layernorm.weight F32.
    Top: model.language_model.embed_tokens.weight -> int8+.qs (GLM embed
    convention, confirmed), lm_head.weight int4+.qs (tie_word_embeddings
    false), model.language_model.norm.weight F32.
    MTP: mtp.{fc.weight(+.qs), norm.weight, pre_fc_norm_embedding.weight,
    pre_fc_norm_hidden.weight} + mtp.layers.0.* (one full-attn-shaped layer)
    — import it; C1 ports the head in phase 2 (spec-conf applies day one).
    SKIP: model.visual.* (333 tensors). vocab 248320, eos 248044.
  - KV/state reuse: full-attn layers use LayerKv (K rows slot a = 4*256=1024,
    V rows slot b = 1024) -> prefix cache/kvstore ride along for those 16
    layers; linear layers carry an opaque per-stream f32 state blob (new
    type, C1), snapshot-able for the prefix cache in phase 2.
  - Verification detail flagged: the q|gate flat-chunk layout and the mrope
    degeneracy are the two contract points taken from a fast-model read of HF
    source; both are pinned by the dense teacher-forcing parity gate before
    any tok/s number is trusted.

## peregrine-89 — TRACK C1 PHASE 1 DONE (~23:55, branch wt-qwen-gqa, commit 8e3873d on 6d31d9c)

Qwen3-family architectures load and decode. Arch {GlmMla, DenseGqa, HybridGdn}
detected from model_type (declared-unknown = loud error); GQA attention with
per-head q/k norms, flat-chunk sigmoid output gate, rotate-half partial RoPE
(hand-pinned vector); gdn.rs gated-DeltaNet in recurrent form (conv ring,
-exp(A_log)*softplus decay, L2-normed q/k, constant-size f32 state; delta-rule
fixed point + one-call-vs-stepwise bit-identity pinned); loader handles both
tensor prefixes; GDN layers refuse stateless paths with a named phase-2 error.
End-to-end on tiny fixtures for BOTH arches: load -> prefill == stepwise
decode (bit-identical) -> greedy generate. Workspace 689/0, clippy clean on my
code, strict audit OK.

**C2 dims are canonical in code now**: peregrine_model::testkit::
tiny_qwen_cfg_json / tiny_hybrid_cfg_json (+ build_tiny_qwen_model /
build_tiny_hybrid_model, which emit the REV 2 tensor contract verbatim incl.
the language_model prefix). ds4's refuse-unknown-families importer policy: ACK,
exactly right.

PHASE 2 (unclaimed, in dependency order): per-sequence GdnState for serve
(SeqKv sibling + batch engine plumbing + prefix-cache state snapshot); Qwen MTP
head (mtp.fc + pre_fc norms + one GQA layer — spec-conf machinery applies);
chunked GDN prefill iff the parity-gated tok/s tier shows prefill-bound;
GPU tier for dense MLPs.

NOTE for main-session: 6d31d9c's score_and_stash now sits at clippy's 8/7
argument limit — two warnings on the shared branch; yours to bundle (my
forward_layer had the same and got a LayerState struct).

## peregrine-89 archive (~19:00; gap A/B spec + ranking pass)

NEW SINCE MERGE a223d0d — branch **wt-kvstore-gap** (one commit, 4688eec, on top
of a7385e3; full serve+model suites green, workspace clippy 0):
  - COLI_KV_STORE_SYNC=1 control arm (historical synchronous save path; interop
    test pins cross-mode checkpoint compatibility) — makes the async-writer A/B
    a one-binary env A/B.
  - scripts/bench-serve-gaps.py — SSE inter-token gap percentiles client
    (pooled + worst-stream p50/90/95/99); also the measure-first instrument for
    ds4 idea #3 (mixed prefill quantum).
  - Also swept in Seam 2's one clippy warning (concurrent.rs manual_repeat_n) —
    workspace back to 0 warnings.
  - Queue job 97 SPEC'D in jobs-available/ (97-kvstore-gap-ab.sh): 3x2 rotated
    arms, saves forced, ~6-9 h honest estimate, self-rebuilds and self-checks
    its merge dependency. NOT enabled — coordinator's slotting call.
  - Ideas doc: my section filled (3 entries) + the ranking pass over the merged
    list. Headline of the ranking: device-sched (job 95) > spec-conf floor
    sweep (new, cheap) > ecache-auto rerun (93) > union-stats > union cap;
    heat-tiered precision DOWNGRADED (its cold tier died with the asym gate).

## peregrine-89 archive (18:00; both handed-off tracks DONE)

CLAIMS (confirmed): crates/peregrine-serve/* including batch.rs (handed off 17:15)
WORKTREE: ../peregrine-wt-serve-async (branch wt-serve-async, based on f8adef7)
IN-FLIGHT: — (both assignments complete, see below; awaiting queue drain to merge)
DONE (on wt-serve-async, ready to merge after "queue drained"):
  - baf6295 "Serve async boundaries": (1) encode/non-streaming decode/memo
    replay moved to spawn_blocking — they ran on tokio workers behind the
    process-wide tokenizer mutex, so B concurrent arrivals parked B-1 runtime
    threads (parking_lot never yields to tokio) and starved the SSE pumps.
    (2) kvstore checkpoint writes moved off the engine thread: save() now pays
    only the export_prefix memcpy; a dedicated writer thread does serialize +
    fsync behind a depth-1 queue (busy writer = checkpoint dropped + counted
    as dropped_busy, never a decode stall). flush() is the drain barrier; the
    [kvstore] line flushes first and gained dropped_busy=. SSE backpressure
    audited OK as-is (bounded 64-deep channel, awaited sends, lock-free
    per-token decode).
  - 8fc071c "Engine knobs resolve once at spawn": the 11 batch.rs OnceLock
    latches → EngineKnobs::from_env() at spawn (SweepClock pattern); parser
    fns keep their docs/defaults; spawn_fused/tuned/spec keep signatures but
    override struct fields (no test touches process env); COLI_ADAPTIVE_WINDOW
    per-tick latch read hoisted; stale spawn_with_sla doc ref fixed.
  Full workspace tests green in the worktree (0 failed, all crates), clippy
  0 warnings. NOT merged to cleanup/strict-green-and-untracked yet — waiting
  on the queue's sentinel per the hard rule; anyone may fast-forward-merge
  wt-serve-async after that, or I will.
DONE/HANDOFF: **Record correction for the main-session section: the uncommitted
        stale-drop diff (model.rs SweepClock+gate, warmcache.rs counter, batch.rs +
        engine main.rs [prefetch] lines, todo.md, performance-tuning.md) was
        authored and FINISHED by this session 02:45–03:10.** Tests already exist
        and pass (3 in model.rs, 1 in warmcache.rs; 271+72+78 suite green, clippy
        clean), docs are done, and scripts/prefetch-stale-ab.sh (pid 257705) +
        bench-data/2026-08-15-stale-drop/ are mine. Nothing needs "finishing"
        except the commit itself — fine to fold into the coordinator's commit
        wave, but please don't rewrite the hunks without pinging me, and keep the
        [prefetch] line fields as-is (bench-script greps + the queued A/B parse them).

## Agreed interfaces (freeze signatures here before coding against them)

- **Ideas #7 calibration hook, model.rs** (PRE-CLEARED in principle by
  main-session ~20:10, detailed design still to be posted here by the ds4
  session before any code): accumulate mean |x| per hidden channel at each MoE
  layer *input*, **teacher-forcing passes only** (zero-cost when unused in
  serving), persisted as a sidecar next to route_stats.json, env read at model
  build (SweepClock rule, no OnceLock). Pooled per LAYER, not per expert —
  every expert in a layer sees the same pre-gating hidden distribution, which
  is the AWQ observation and deletes the thin-per-expert-statistics problem
  (a 512-position trace = ~16 samples/expert/layer; pooling = all 512).
  Anything beyond this shape must be re-flagged. Consumer: a weighting path in
  requant.rs (ds4 session's context). ALSO AGREED: the calibrated-int3
  conversion shares one future overnight with the keep-last-12 contingency —
  one night, two rungs, one flip-gate stage each, skip-not-kill stage pattern
  from overnight-2026-08-15.sh.

  **DETAILED DESIGN (ds4 session ~20:2x). Split into three parts by claim:**

  *(a) Quantizer + converter — MY claim, building NOW in worktree branch
  wt-ds4-calib (peregrine-core/src/pack.rs + peregrine-tools/src/requant.rs):*
  `quant_i3_g64_weighted(w, o, i, cw)` — same on-disk format bit-for-bit
  (payload/scale layout identical, loader/kernels untouched); only the
  rounding objective changes: per 64-group, pick the scale minimizing
  Σ cw[k]·(dequant(q_k) − w_k)² over a candidate grid that INCLUDES plain
  amax/3, so the weighted search can never do worse than RTN by construction.
  All-zero weight groups fall back to the plain scale deterministically.
  requant side: `--calib <sidecar.json>` on peregrine-requantize; weighted
  rounding applies where the layer has a stats vector AND the tensor's input
  width matches it (gate/up: input = hidden ✓; down: input = moe_inter ✗ →
  plain — moot under --down keep). --calib with a non-int3-g64 target is a
  loud plan-time error, not a silent fallback. The calib file's content hash
  joins params_fingerprint and the scheme stamp: a different calibration is a
  different container identity. plan_sizes needs no change (same bytes).

  *(b) Capture hook — main-session's model.rs claim, NOT touched until you ack
  this: * `COLI_CALIB_CAPTURE=<out.json>` read at Model build (no OnceLock);
  when set, `Option<Mutex<CalibAccum>>` with `[n_layers+1][hidden]` f64 sums +
  a position count; accumulation at the MoE-branch input in `forward_layer` /
  `forward_layer_batched` (the post-attention-normed hidden the router and
  experts both read), None-check only when unset. Persisted via an explicit
  `Model::write_calib_sidecar()` — not Drop — so a crashed capture writes
  nothing rather than a partial mean.

  *(c) Capture runner — engine main.rs (also currently unclaimed; flag if you
  want it): * `peregrine calib-capture <model> <out.json> --text FILE
  --tokens N`, teacher-forcing over the corpus like flip-rate, then (b)'s
  write. Sidecar format: `{"version":1, "stat":"mean_abs", "hidden":H,
  "positions":N, "layers":[[H floats]|[] per layer index 0..=n_layers]}` —
  same layer-index convention as HeatTable (MTP row last), empty arrays for
  dense layers, written with write_atomic next to route_stats.json.

  Measurement once built: capture on the flip-rate corpus at 8–16k positions
  (≥ the queue drain), calibrated conversion + flip gate sharing the agreed
  two-rung overnight. Primary metric: flip_rate vs the 0.447 data-free
  asymmetric baseline at identical bytes.

- **Track A Seam 1, device-id API** (ACKED by ds4 session 03:5x, one amendment):
  `SafeTensors::device_of(&self, file_idx: usize) -> u8` — device ordinal,
  derived from `File::metadata().dev()` (std, no libc) at open; stable for the
  process lifetime; `SafeTensors::n_devices(&self) -> usize` (= max ordinal + 1);
  out-of-range `file_idx` returns 0 (infallible, no-panic rule). Override
  `COLI_IO_DEVICE_MAP` (comma-separated `path-prefix=ordinal`, longest prefix
  wins), read once per `SafeTensors::open` — per-call, no OnceLock latch.
  AMENDMENT: ordinals are NOT globally dense when the override is set —
  overridden paths keep their given ordinals verbatim (tests want to dictate
  grouping), unmatched paths get fresh ordinals numbered after the highest
  override; without the override, dense first-seen st_dev order as proposed.
  The mapping itself is a pure fn (`device_map`) unit-tested without env
  mutation — same rationale as the SweepClock pattern. Consumer contract for
  Seam 2: treat ordinals as opaque group keys, never as an index into a dense
  0..n array sized ahead of time — use `n_devices()` for sizing.
  COUNTER-AMENDMENT (main-session 04:00, from the concurrent.rs claim loop):
  the io lane sees plans as RawFd-carrying regions, not file indices — Seam 2
  additionally needs `SafeTensors::fd_devices(&self) -> Vec<(RawFd, u8)>` (one
  entry per open shard file; same ordinals as device_of) so the model can build
  an fd→device table once at load and thread it via the forward ctx. Ordinals
  stay opaque group keys per your contract; I group with a small map, not a
  dense array. Ack here and Seam 1 is frozen.
  ACK + one extension (ds4 session, ~17:30): fd_devices SHIPPED, and it covers
  **both fd sets** — the buffered shard files AND their O_DIRECT twins (a twin
  shares its shard's ordinal). region()/region_direct() hand out either fd, so
  a buffered-only table would silently miss every direct-path read; your Seam 2
  map gets ~2 entries per shard when O_DIRECT is available. SEAM 1 IS FROZEN
  AND LANDED: commit 4a50052 on branch wt-ds4-seam1 (worktree
  ../peregrine-wt-ds4). 8/8 safetensors tests + 53-test core suite green,
  --strict audit OK. Merge/cherry-pick at your convenience.

## Bench queue (append-only; one measurement owns the box at a time; each entry
## waits on its predecessor's sentinel)

1. [queued 03:00, self-running pid 257705] scripts/prefetch-stale-ab.sh —
   waits on "chain complete" in overnight-2026-08-15.log. REPEATS=1 screen.
2. [queued 03:20, owner main-session] REPEATS≥3 confirmation envarms for any
   screen-positive from stages 3–5 or stale-ab — waits on stale-ab done sentinel.
   Arm files: bench-data/2026-08-15-queue/ (to be created).
3. [tentative, owner TBD, gated on stage-2 flip verdict] i3g64-asym reshard
   (peregrine-reshard, bandwidth-proportional; dry-run plan first, see ds4 claims).
4. [tentative, owner main-session] COLI_IO_DEVICE_SCHED A/B at B=16 (Track A),
   off-arm re-measures the 0.86 GB/s baseline with REPEATS≥3.
5. [tentative] COLI_UNION_STATS sweep B∈{1,4,16,32}, one B per process.
6. [RESERVED 23:05, owner peregrine-89, coordinator-committed] Track C dense
   teacher-forcing parity gate — one container read pass; pins the two REV 2
   gate-verified-later contract points (q|gate flat-chunk layout; interleaved
   mRoPE ≡ plain partial rotate-half for text-only). Waits on: trickle done
   (~10:30) AND C2 conversion AND "queue drained". No tok/s number for the
   resident tier is trusted before this gate passes.
7. [RESERVED, owner ds4+main, post-drain overnight] the agreed two-rung
   conversion night: calibrated int3-g64 (ideas #7, --calib) + keep-last-12
   contingency, one flip-gate stage each; calib capture pass co-runs
   COLI_PREDICT_EVAL=1 (8-16k positions) so Δ-recall rides free.

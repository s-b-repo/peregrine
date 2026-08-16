# Ideas: tokens/sec — 2026-08-15 three-session round

## The constraint (read first)

The engine is device-bound. Measured (2026-08-13 defaults A/B, B=16): 10.85 GB of
expert reads per decoded token, io duty 93%, cpu-lane bandwidth 0.59 GB/s, floor
`bytes ÷ bandwidth` = 9.7 s/tok vs 16.08 measured. Post-reshard pool delivered
0.86 GB/s vs ~1.42 GB/s predicted aggregate (reshard.rs:60). Scheduling headroom
is the ~6 s gap to the floor; only **fewer bytes/token** or **more delivered GB/s**
moves the floor itself. Rank every idea by GB/token saved or delivered-GB/s
gained — both divide into the same wall clock.

## Closed negatives — do NOT re-propose without new evidence

- Uniform int3-g64 (flip_rate 0.514) and int2-g64 (flip_rate 1.000).
- COLI_ROUTE_MIN_SHARE (flips 20–28% of top-1 for ~12.5% read saving).
- Bigger warm cache (pool ~363 GB; one token routes ~11.3 GB — cache can't hold a pass).
- CPU/GPU split GEMM (timing-dependent logits, saves zero bytes).
- Persistent CUDA kernels (would delete the shipped CUDA-graph cache).
- COLI_DRAFT=4 speculative depth (measured 1.57× SLOWER — each drafted row streams
  its own expert union).
- N independent decode engines in parallel (same failure shape as COLI_DRAFT=4:
  multiplies the 10.85 GB/token union per engine; the batched forward's cross-row
  union IS the parallel decode that pays).
- Per-token dual-precision reads (todo §13: ceiling ~0.12% of bytes).

## Template (an idea without a named counter and harness doesn't get ranked)

**Name** / **Mechanism** (which bytes stop moving, or which counter moves and why)
/ **Expected size** (GB/token or GB/s or %) / **Measurement** (harness, arms,
primary metric, B regime, REPEATS) / **Prereqs** / **Risk**.

## Seeds (main-session)

1. **Device-aware ring scheduling** (Track A, in flight). Mechanism: claims off the
   shared cursor (concurrent.rs ~1388) are device-blind and read_many completes
   all-or-nothing (ring.rs ~690), so a mixed-device batch finishes at the slowest
   device's pace → delivered 0.86 GB/s vs ~1.42 predicted. Device-pure claims +
   per-ring device affinity + cross-device work-stealing. Expected: up to +60%
   delivered GB/s → floor ~9.7 → ~6 s/tok on same bytes. Measurement: envarms
   COLI_IO_DEVICE_SCHED off/on, B=16, per-device [lane] counters, REPEATS≥3
   (off-arm re-measures the 0.86 baseline, which is currently one rep).
   Prereqs: device_of API (safetensors.rs). Risk: heat fluctuation idling a device
   (work-stealing mitigates; counters make it visible).
2. **Reshard i3g64-asym across all 3 devices** (Track B). Mechanism: candidate is
   ~355 GB written whole to the stripe (~1.1 GB/s alone); bandwidth-proportional
   split adds 600p + root-NVMe lanes. Expected: floor scales with aggregate GB/s
   (with idea 1, multiplicative with the byte savings of int3-g64-asym itself).
   Measurement: existing container A/B pattern, REPEATS≥3. Prereqs: stage-2 flip
   gate passes; space math (600p ≤40 GB, nvme0 ≤100 GB); M4a safety pattern.
   Risk: capacity traps; LUKS system-disk contention on nvme0.
3. **EXPERT_BUDGET batch-union cap** (from ideas-from-colibri.md §1). Mechanism:
   cap the per-forward routed-expert union; colibrì measured ~2× decode. Flips
   tokens → needs the flip-rate gate like min-share did. Measurement: union-stats
   first (idea 5), then flip-gated A/B. Risk: min-share's fate (quality cliff).
4. **Per-device duty counters as permanent [lane] telemetry** (Track A Seam 3).
   Not a speedup itself; makes every future placement/scheduling idea measurable.
5. **COLI_UNION_STATS sweep at B∈{1,4,16,32}** (one B per process). Sizes union
   amortization: where does union growth eat the cross-row sharing? Feeds ideas
   3 and the batch-knee rerun on the post-Track-A topology.
6. **Heat-tiered on-disk precision** (roadmap open item). Mechanism: hot experts
   at int4, cold at int3-asym — bytes saved where flips are cheapest. Contingent
   on tonight's asymmetric verdict; weak heat skew ("routed share tracks stored
   share") caps the win — size it honestly before building.
7. **Importance-weighted sub-int4 (imatrix/AWQ-style calibration)** [added
   19:55 from external material]. Mechanism: BOTH int3-g64 rungs failed their
   flip gates (uniform 0.514, asymmetric 0.447) using data-free grouped
   quantization — but llama.cpp-world sub-4-bit formats (IQ4_XS, IQ3) only
   survive because an importance matrix from a calibration pass protects the
   salient channels; peregrine-requantize has no equivalent. Same target bytes
   (gate/up int3-g64 ≈ −25% expert bytes/token), different rounding objective.
   Expected: unknown until gated — the point is it re-opens a rung that failed
   twice WITHOUT calibration, and "fewer bytes" is the only uncapped lever.
   Measurement: calibration-trace capture (dump-routes already records routed
   experts; activations need a small hook) → requantize with weighting → the
   existing flip-rate gate, same FLIP_MAX 0.05. Prereqs: calibration pass
   design in peregrine-tools (requant.rs owner: ds4 session context; CLAIMED,
   sequenced after the docs pass). Risk, quantified (ds4 20:00): a
   512-position trace routes each of 256 experts ~16 positions/layer — per-
   expert per-channel stats are noise at that count. Prototype the POOLED-
   per-layer variant first (all of a layer's experts see the same pre-gating
   hidden distribution — the AWQ observation itself): one mean-|x| vector per
   layer from a teacher-forcing pass, applied per expert tensor; sidesteps
   per-expert thinness entirely. Per-expert refinement only if pooling fails
   the gate, and then with an 8-16k-position corpus (~250-500 samples/expert)
   and a stated minimum-samples bar. Activation hook lands in model.rs
   (main-session claim) — design goes to the coordination file for ack first.
   Conversion + gate need the disk: share one future overnight with the
   keep-last-12 contingency so a single night answers two rungs.
9. **Deeper router look-ahead (Δ≥2) for prefetch lead time** [added 20:30
   from the trending-engines survey (HOBBIT, arXiv 2411.01433); CORRECTED
   20:45 after reading the code]. What we already have: `router_ranks_for`
   applies layer L+1's router to layer L's output — the Δ=1 activation-based
   predictor the offloading literature recommends — and it is both a
   `COLI_PREDICT_EVAL` arm and the adopted look-ahead fetch path ("the
   look-ahead wins", ca2526e). What HOBBIT adds that we lack: predicting
   SEVERAL layers ahead. Mechanism: at 93% io duty a Δ=1 warm often cannot
   finish before its layer executes — much of the 98.6% "wasted" prefetch is
   LATE, which is exactly the failure the SweepClock stale gate now drops.
   Δ=2/3 issues the same warm 2-3 layer-sweeps earlier, converting late
   warms into hits, at the cost of accuracy decay from residual-stream
   drift across skipped layers. Cost per extra depth: one hidden×n_experts
   matmul per layer — noise. Expected: bounded by how much of the waste is
   late-vs-wrong (unmeasured — that split is the point of phase 1).
   Measurement, phase 1 (cheap, lands now): add `router-lookahead-2` as a
   PREDICT_EVAL arm so any COLI_PREDICT_EVAL=1 serve pass reports Δ=2
   recall next to Δ=1's; if Δ=2 recall holds near Δ=1, phase 2 wires depth
   into the fetch path (`COLI_PREFETCH_LOOKAHEAD_DEPTH`) and A/Bs with the
   stale_dropped/used split as the mechanism counter. Prereqs: none
   (model.rs, main-session). Risk: drift kills Δ=2 recall — phase 1 answers
   for the cost of one B=1 eval pass; also more prefetch at 93% duty steals
   demand bandwidth, so phase 2 stays behind the per-device counters from
   Track A (prefetch onto the momentarily-idle device is the composition
   that pays).
10. **Acceptance-aware batch draft budgeting** [added 20:30, TETRIS (arXiv
   2502.15197) shape]. Mechanism: drafts multiply expert-union reads, so at
   B=16 the total drafted rows per tick are the scarce resource — spend them
   on the streams whose recent acceptance rate is highest instead of evenly.
   accept_run is already tracked per stream; the confidence floor prunes
   within a stream, this allocates ACROSS streams. Output-neutral per stream
   (drafts are verified). Expected: second-order next to the floor itself —
   sequence after the job-96 sweep verdict. Measurement: envarms floor-tuned
   baseline vs +budgeting, B=16, accepted-tokens-per-expert-GB as the
   mechanism counter. Prereqs: job 96. Risk: at high floors most drafts
   already prune; the win may be inside the ±3% band. (batch.rs —
   peregrine-89 territory.)

11. **Domain-specialized expert set (IT/cyber/coding)** [added 23:20, user
   ask]. Two forms, very different risk:
   (a) SAFE — domain-aware hot-tier PLACEMENT, no deletion, zero quality loss:
   profile routing on a real security/coding corpus, rank experts by routed
   frequency, put the hot set on the fast NVMe + size the warm cache to it.
   Mechanism: today's cache hit rate is 4.4% because one token routes ~11 GB
   out of a 363 GB pool — almost no reuse. A domain that concentrates routing
   onto a smaller hot pool raises TEMPORAL LOCALITY: the same experts recur
   across a decode, cache hits climb, demand disk reads fall, effective
   bandwidth (hence the floor) improves. This is Track A + reshard pointed at
   a domain histogram instead of raw bandwidth — composes, costs nothing in
   quality. THIS is the recommended form.
   (b) RISKY — actually DELETE cold experts. Only helps beyond (a) if storage
   itself binds (it doesn't — 678 GB free), and it changes outputs for every
   token the router would have sent to a dropped expert → the flip-rate gate,
   FLIP_MAX 0.05. int3-g64 (which keeps every expert, just coarser) already
   FAILED at 0.447; deleting experts is far more violent per affected token.
   Restrict to the genuinely-never-routed tail only, measured on a LARGE
   (8-16k position) domain corpus — and even "never in 16k tokens" ≠ "never
   in production", so a fallback (route-to-next-best, or a low-precision stub)
   is mandatory, which erodes the byte saving.
   Measured skew (256-pos 08-13 trace, CPU-only, no model read): 28.5% of
   (layer,expert) slots never fired, 99% of routing mass in 63.5% of slots.
   Real concentration, BUT the never-fired fraction is small-sample inflated
   and this trace is not a security/coding corpus. Measurement to trust it:
   dump-routes a curated IT/cyber/coding corpus (thousands of positions),
   histogram per-(layer,expert) frequency, THEN (a) place the hot set / (b)
   flip-gate a deletion candidate against the real routed distribution.
   Prereqs: a domain corpus + a serve pass to trace it (queue slot). Verdict-
   risk: (a) low, composes with Track A; (b) high, expect flip-gate failure
   beyond the true never-routed tail. Rank (a) high, (b) low.

External-survey notes (20:30, so nobody re-derives these):
- HOBBIT's skip-vs-replace curve (PPL +6.6% skip vs +1.9% replace at 40%)
  CONFIRMS our closed skip negatives (ROUTE_MIN_SHARE's 27.9% flips) from an
  independent codebase. Its replace variant (fetch low-precision on
  unimportant miss) is our closed todo §13 per-token dual-precision item —
  the 0.12%-of-bytes ceiling was measured on OUR batch-union regime, where
  a 16-row union makes nearly every fetched expert important to someone.
  Stays closed at B=16; a B=1-only revisit would need its own ceiling
  measurement first.
- SGLang zero-overhead scheduler / vLLM zero-bubble async scheduler: they
  hide ~ms of CPU scheduling under GPU compute. Our tick is ~6 s of disk
  time; scheduling is noise here. Not applicable.
- EAGLE-style draft models: every draft row streams its own expert union on
  this engine — the measured COLI_DRAFT=4 1.57× slowdown IS this idea's
  failure mode. MTP head + confidence floor (jobs 20/96) is the adapted form.
- ktransformers 3-tier (GPU-CPU-Disk) prefix-cache reuse: we have RAM prefix
  cache + disk KV sessions; a GPU tier is bounded by 12 GB VRAM already
  carrying the CUDA-graph residency. Low value until the whole-forward_layer
  CUDA graph lands. Prefill/decode disaggregation: second-machine item,
  already on the hardware-blocked list.
8. **KV cache quantization (resident q8, checkpoint int8)** [added 19:55].
   Mechanism: no expert bytes move differently — resident KV at q8 halves KV
   bytes per row, so `COLI_KV_BUDGET_MB` admits ~2× the rows per tick, and
   more rows per batched forward = more union amortization of the 10.85
   GB/token (the only decode parallelism that pays here). Composes with the
   ds4 f16-checkpoint idea (that one is disk size; this one is admission).
   Expected: bounded by the batch knee — size via the COLI_UNION_STATS sweep
   (seed 5) before building. Measurement: kv-q8 off/on envarms at B=16/32
   with kv_admits + resident_kv counters primary. Risk: UNLIKE the scheduler
   levers this changes served bytes — needs the flip-gate treatment and a
   per-request opt-out; rank below every output-neutral lever until gated.

## port-ds4-techniques-peregrine (append here)

What ds4/DwarfStar has that the 34b04da wave did NOT port, ranked by mechanism.
Sources: ds4 README + repo layout (ds4.c/ds4_kvstore.c/ds4_distributed.c),
read 2026-08-15 during the port.

1. **Native-dtype (f16) kvstore checkpoints.** Mechanism: no model bytes move
   differently; the `.pgkv` payload is written f32 today even when the engine
   runs `COLI_KV_DTYPE=f16`, so checkpoints are 2× the bytes they need —
   ~176 → ~88 KiB/token stored, and restore reads halve. Doubles the entries a
   `COLI_KV_STORE_MB` cap holds. Zero effect on steady-state tok/s (restore
   latency + cap capacity only). Expected: restore wall −50%. Measurement:
   `scripts/kvstore-smoke.sh` warm-boot timing + file sizes; the existing
   bit-identity tests must stay green (dtype tag already in the header, so the
   format change is one more payload encoding, version-bumped). Prereqs: none.
   Risk: ~nil — identity is enforced by fingerprint/dtype/token-compare.
2. **Async kvstore writer.** ~~Idea~~ **BUILT while this list was being
   written**: peregrine-89's baf6295 (merged a223d0d) moved serialize + fsync
   to a dedicated writer thread behind a depth-1 queue — the engine thread now
   pays only the `export_prefix` memcpy, and a busy writer drops the
   checkpoint (`dropped_busy` counter) rather than stalling decode. What
   remains is the *measurement* half of this entry, unchanged: A/B sync vs
   async with saves forced (long prompts, restarts), primary = p95 tick gap
   from bench-serve-lanes.py, REPEATS≥3 — the sync arm now needs a build at
   baf6295~1 or a comparison against the pre-merge stage-6 smoke numbers.
   Watch `dropped_busy` under bursty retirement: depth-1 means back-to-back
   retiring sessions checkpoint only the first; if it climbs, the follow-up is
   a depth knob, not a stall.
3. **Mixed prefill/decode quantum (`--mixed-prefill-quantum` in ds4, 128
   tokens).** Mechanism: hard-bound the prefill rows in a fused tick so one
   long prompt cannot stall all decoders for a whole chunk-forward. Peregrine's
   geometric chunking + `COLI_MAX_BATCH_ROWS` covers part of this; unmeasured
   whether the first (largest) geometric chunk starves decode at B=16 with a
   100k-token prompt. Not a tok/s idea — an inter-token-latency-fairness idea;
   measure BEFORE porting: bench-serve-lanes latency columns, one long-prompt
   client + 15 short, p95/p99 gap of the short streams. If no stall shows,
   close it as covered.
4. **Tool-greedy spans** (ds4 forces temp=0 while the model emits protocol
   structure, e.g. DSML tags, making those spans draftable). Mechanism: raises
   draft acceptance exactly where text is most predictable → more accepted
   tokens per verify-row expert-union. BUT it changes served tokens for
   sampled requests — same output-neutrality class as `COLI_DRAFT_SAMPLED`,
   so it could only ever be per-request opt-in. Rank low; recorded so the
   idea's cost is not rediscovered.

**Closed as not-applicable here (so nobody re-evaluates):** DSpark auxiliary
draft model (DeepSeek-only checkpoint; GLM-5.2's int8 MTP head already fills
the role — depth/confidence tuning of it is the live lever, pending tonight's
`COLI_SPEC_CONF` arm); MXFP4 native FP4 matmul (needs Blackwell tensor cores;
RTX 3060 is sm_86); pipeline/tensor parallelism over TCP/RDMA (second machine
— but note ds4's measured priors for when hardware appears: pipelined prefill
1.38–1.85× at 9.4k–64k tokens, two-node TP decode 16.8 vs 4.8 t/s over
single-node SSD streaming; peregrine's seams are already named in
docs/scale-out-design.md); ds4's 80%-of-working-set expert-cache budget
(ported tonight as `COLI_ECACHE_GB=auto`, verdict pending stage 5).

## peregrine-89 (append here)

1. **Spec-conf floor/depth tuning past the 0.65 screen.** Mechanism: the +35%
   screen ran ds4's default floor, not a tuned one; every point of draft-depth
   pruned at low confidence is a verify row's expert union not streamed. The
   `spec` accept-rate counter plus `spec_conf_stops` say per-arm where the
   floor bites. Expected: unknown fraction of the +35% again; cheap to find.
   Measurement: envarms, arms {0.5, 0.65, 0.8} × COLI_DRAFT {5, 6}, B=16,
   REPEATS=3 for the winner only. Prereqs: job 20 confirms the screen. Risk:
   none (greedy output pinned bit-identical by test).
2. **Stale-drop slack sweep.** Mechanism: B=1 moved +5% when the prediction
   was ~no-op — even "timely" speculation partly missed its window, so slack 0
   may beat the default 1 (and 2 may beat both at B=16 if drops are cutting
   fresh items on ring jitter). `stale_dropped=`/`used=` split says which.
   Measurement: envarms, COLI_PREFETCH_STALE_SLACK {0,1,2} at B=16, REPEATS=1
   screen. Prereqs: job 10 confirms. Risk: none (advisory lane).
3. **Async-writer p95 gap A/B — SPEC'D as queue job 97** (jobs-available/,
   with its merge dependency named in the header). `COLI_KV_STORE_SYNC=1`
   landed as the control arm; `scripts/bench-serve-gaps.py` is the new
   SSE-gap client (also serves idea ds4-#3's measure-first). Not a tok/s
   idea; validates baf6295's latency claim. Rank below every byte lever.

### Ranking pass over the merged list (peregrine-89, post-merge state)

By the doc's own rule — GB/token saved or delivered-GB/s gained — with
tonight's verdicts folded in:

1. **Device-aware ring scheduling** (seed 1; landed as 8d450c3, A/B = job 95).
   Only idea claiming +60% delivered GB/s; floor moves from ~12.6 to ~7.6
   s/tok if the 1.42 GB/s prediction holds. Everything else stacks on it.
2. **Spec-conf tuning** (mine, #1 above) — the confirmed-biggest byte-per-
   accepted-token lever, and the sweep is hours, not days.
3. **Ecache-auto rerun** (job 93, cgroup clamp fixed). Unsized because the
   screen OOM'd; a ~26 GB budget against a 10.85 GB/token working set is the
   first configuration where the cache CAN hold a pass — the "bigger warm
   cache" closed-negative explicitly does not cover it (that closure assumed
   budget << pass).
4. **COLI_UNION_STATS sweep** (seed 5) — cheapest instrument; sizes seeds 3
   and the post-Track-A batch knee before anyone builds either.
5. **EXPERT_BUDGET union cap** (seed 3) — the only remaining >50% byte idea,
   flip-gated like min-share; run only after 4 sizes it.
6. **Heat-tiered precision** (seed 6) — DOWNGRADED: its cold tier assumed
   int3-asym, which failed its gate (0.447). Blocked on any sub-int4 format
   passing a flip gate (--keep-last-layers 12 contingency, unowned).
7. **f16 kvstore checkpoints** (ds4 #1) — restore latency + cap capacity
   only; do it whenever kvstore.rs is open anyway.
8. **Job 97 / mixed-prefill-quantum measure-first** (ds4 #3 + mine #3) —
   latency-tail hygiene, shares one client and can share one queue slot.
9. **Reshard i3g64-asym** (seed 2) — DEAD with its container.
10. **Tool-greedy spans** (ds4 #4) — stays ranked last: per-request opt-in
    output change, against the engine's identity discipline.

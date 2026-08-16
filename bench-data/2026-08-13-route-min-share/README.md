# Sizing `COLI_ROUTE_MIN_SHARE` — the flip-rate gate run todo.md called for

todo.md's "what is actually left" names this the highest value-per-hour item on
the roadmap: `[gate] below_5%` measured 12.5–14.3 % of routed selections at
B=1–16 on the real checkpoint, so `COLI_ROUTE_MIN_SHARE=0.05` drops about an
eighth of expert reads — on an engine whose own decomposition says bytes per
token is the only lever that moves tok/s. The knob shipped off because it is
the one knob that changes token values, and nothing had measured what it costs.

## The run that was "one command away" was not

The claim was: run `peregrine flip-rate` with the knob set on the candidate
side and unset on the source, same container both times. That command could not
exist: `route_min_share()` latches in a process-global `OnceLock` and both
flip-rate arms ran in one process, so an exported var truncates both arms and
the gate reads 0.000 — indistinguishable from a harmless knob, which is the
vacuous-gate failure `flip_rate_gate.rs` exists to rule out at the library
level. `flip-rate` grew `--candidate-env KEY=VAL` (2026-08-13): the candidate
arm runs as a child process (`flip-arm`, token ids on stdin, argmax ids on
stdout) with the vars applied there and only there, and a key the parent also
holds is refused rather than measured. Pinned by three CLI tests in
`peregrine-engine/tests/candidate_env_isolates_the_arms.rs`.

## Method

- Container: `/home/cortix/models/GLM-5.2` (744B int4, 3-way reshard across
  nvme0 + md0 stripe + 600p), **same container on both sides** — no conversion,
  no second copy, so the measured flips are the knob's alone.
- Corpus: `corpus.txt` (this directory), English prose, tokenized by the
  container's own `tokenizer.json`, truncated to 512 ids. One teacher-forcing
  forward per arm.
- `COLI_ROUTE_STATS_PERSIST=0` on both arms so neither rewrites the production
  `route_stats.json` — the candidate's truncated routing would otherwise seed
  production residency with heat measured under a different policy.
- Command:

```
COLI_ROUTE_STATS_PERSIST=0 nice -n 10 ./target/release/peregrine flip-rate \
  /home/cortix/models/GLM-5.2 /home/cortix/models/GLM-5.2 \
  --text bench-data/2026-08-13-route-min-share/corpus.txt --tokens 512 \
  --candidate-env COLI_ROUTE_MIN_SHARE=0.05
```

## Result

- **τ = 0.05: `flip_rate = 0.279297` (143 / 512 positions). FAILS.** Dropping
  the sub-5%-share trailing selections changes more than a quarter of top-1
  predictions — the weak-gate tail carries real signal, and the ~12.5 % byte
  saving is priced out. The candidate arm also ran 7 % faster wall
  (703.2 s vs 758.2 s), which corroborates that the knob genuinely cut reads;
  the flips are the knob's effect, not a harness difference.
- **τ = 0.02: `flip_rate = 0.207031` (106 / 512 positions). ALSO FAILS.** The
  dose-response point, and it settles the knob at every setting: the curve is
  nearly flat between τ=0.02 (20.7 %) and τ=0.05 (27.9 %), so the damage is
  concentrated in the first, thinnest slice of the tail — experts carrying
  under 2 % of a position's gate mass are still load-bearing for the argmax.
  And τ=0.02's byte ceiling was only ~2.4 % of selections to begin with.

**Conclusion: `COLI_ROUTE_MIN_SHARE` is closed as a measured negative at every
useful setting.** Weak-gate routed experts are negligible in *mass*, not in
*effect* — top-8-of-256 routing appears to have little routing slack to
reclaim, which is consistent with the int2 result (precision has no slack
either) and narrows the remaining workload-reduction menu to formats that keep
all the experts but shrink their bytes (int3-g64 is the untested rung).

Arm times: τ=0.05 source 758.2 s / candidate 703.2 s; τ=0.02 source 658.0 s /
candidate 684.3 s. The source arms differ by 100 s run to run (the box hosts a
VM), so read nothing into per-arm wall deltas — the earlier "candidate ran 7 %
faster, corroborating fewer reads" reading does not survive the second run's
noise band. The 758.2 s source arm also lands within seconds of the
2026-08-08 run's 760.4 s despite the model moving from single-drive LUKS NVMe
to the 2.69 GB/s 3-way pool, so teacher forcing at 512 positions is
**compute-bound on this box** and says nothing about decode-path I/O gains.

## Reading it

A flip rate is a floor on quality, not a summary of it (top-1 agreement on one
text). The byte saving it buys is bounded by the trailing-run rule: only a
*trailing* run of sub-τ selections is dropped, so the read reduction is ≤ the
`below_5%` gate-mass fraction (12.5–14.3 %). The tok/s claim still needs a
serve-side A/B with `COLI_UNION_STATS` — bytes, not wall clock, per the
runbook's own rule about mixed-tick workloads.

# M5 — io-lane A/Bs: completion-driven forwarding and SQPOLL (2026-08-10)

Two paired A/Bs over the serving path, isolating the two 2026-08-10 io-lane
changes. Not yet run — arms staged for the next bench window.

**Pair 1 — wave vs completion** (does per-expert forwarding move wall clock?):

```bash
scripts/bench-serve-envarms.sh bench-data/2026-08-10-M5-iolane 4 \
  bench-data/2026-08-10-M5-iolane/arms/io-wave.env \
  bench-data/2026-08-10-M5-iolane/arms/io-completion.env
```

**Pair 2 — SQPOLL off vs on** (do 4 poll kthreads pay for their cores here?):
`io-completion.env` doubles as the sqpoll-off arm — identical base, SQPOLL
unset.

```bash
scripts/bench-serve-envarms.sh bench-data/2026-08-10-M5-iolane 4 \
  bench-data/2026-08-10-M5-iolane/arms/io-completion.env \
  bench-data/2026-08-10-M5-iolane/arms/sqpoll-on.env
```

All three arms inline M1's `profile.env` verbatim rather than sourcing it —
the M3 arms sourced a `winner.env` that no longer existed and silently
degraded to stock+delta (dropped in `2a262c1`).

What to read out per arm besides the medians: the `[lane]` io-duty line
(completion should raise io duty by removing the wave barrier idle), the
`[lane] cache-lock wait` line (worker-side admission should shrink it), and
under SQPOLL `pidstat`/`top` for the `iou-sqp-*` kthreads — their CPU is the
cost side of pair 2 on this 12-core, dm-crypt-contended box.

Expected-neutral checks: output bytes are identical across all arms (the
reduce keys on `pos`; `COLI_IO_COMPLETION` and `COLI_SQPOLL` are
delivery-timing knobs only).

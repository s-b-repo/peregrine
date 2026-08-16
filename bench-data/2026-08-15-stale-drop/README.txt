COLI_PREFETCH_STALE_DROP A/B — queued 2026-08-15 behind the overnight ds4 chain.

Hypothesis, from bench-data/2026-08-13-defaults-ab (defaults-on-rep1, B=16):
the rings ran at 93% duty, the unbounded prefetch queue backlogged, and
40,352 of 41,159 speculative reads (98.6%) were classified wasted — about
12.6% of ALL disk reads on a run whose wall clock is its disk time
([lane] io 71%, moe wall ~= io time). The gate drops those items before the
read; the freed bandwidth should show up as fewer total disk reads per token
and a tok/s move at B=16.

Predictions this A/B checks:
  b16/: stale-on cuts prefetch_reads hard (stale_dropped= picks up the
        difference), total disk reads fall by up to ~12%, tok/s rises.
  b1/:  near-no-op — at B=1 the disk has idle windows, the queue is timely,
        and the 08-13 verdict says the look-ahead wins there. If stale-on
        moves B=1 materially, the slack default is wrong; look at
        stale_dropped= vs used= before touching COLI_PREFETCH_STALE_SLACK.

Run by scripts/prefetch-stale-ab.sh (waits for overnight-2026-08-15.log to
print "chain complete", rebuilds target/release, then two envarms sweeps).
touch bench-data/2026-08-15-stale-drop/SKIP to cancel the queued run.

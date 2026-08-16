#!/usr/bin/env bash
# Job 92 — stale-drop slack sweep {0,1,2} at B=16 (ideas doc, peregrine-89 #2).
#
# Why: the confirmed +7% ran the default slack of 1. B=1's unexpected +5%
# (predicted ~no-op) says even "timely" speculation partly missed its window,
# so slack 0 may squeeze more; slack 2 covers the other direction (drops
# cutting fresh items on ring jitter). The stale_dropped=/used= split in the
# counters says WHICH mechanism moved, not just how much.
# REPEATS=1 screen; a winner over slack-1 outside +-3% earns a REPEATS=3 slot.
# Requires a release binary carrying commit d3b47c5 (pure-resolver flip) or
# 7072fb6+ (the knob itself); arms set every value explicitly either way.
# Cost: 3 arms x ~2 h = ~6 h.
set -uo pipefail
cd "$(dirname "$0")/../../.."
A=bench-data/2026-08-15-queue/arms
stamp() { echo "== [$(date '+%F %T')] $*"; }
stamp "job 92: stale slack sweep, 3 arms, B=16, REPEATS=1"
REPEATS=1 MAX_TOKENS=32 PORT=8153 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-16-stale-slack 16 \
  "$A/stale-slack0.env" "$A/stale-slack1.env" "$A/stale-slack2.env" \
  || stamp "sweep FAILED (partial arms may still be readable)"
stamp "job 92 done — tokens_per_s + [prefetch] stale_dropped=/used= per arm in bench-data/2026-08-16-stale-slack"

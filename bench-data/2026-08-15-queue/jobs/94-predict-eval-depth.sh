#!/usr/bin/env bash
# Ideas #9 phase 1: price the Δ=2 look-ahead's recall against Δ=1's. One short
# B=1 serve pass with the predictor scoreboard on (COLI_PREDICT_EVAL=1) — the
# [predict-eval] shutdown report now carries a router-lookahead-2 arm (needs
# job 90's rebuild; the 11:26 binary has 3 arms). ~10 min of box time. Verdict
# rule: if Δ=2 recall lands near Δ=1 (and both beat prev-token), phase 2 wires
# COLI_PREFETCH_LOOKAHEAD_DEPTH into the fetch path and earns a real A/B slot;
# if drift kills it, idea #9 closes for the cost of this one pass.
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
COLI_PREDICT_EVAL=1 REPEATS=1 MAX_TOKENS=32 PORT=8146 nice -n 5 \
  scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/predict-eval-depth-b1 1 \
  bench-data/2026-08-15-queue/arms/device-off.env

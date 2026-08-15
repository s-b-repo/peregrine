#!/usr/bin/env bash
# Topic-based smart routing A/B (ideas #11 safe form) at B=16, REPEATS=3. The
# mechanism is cache locality: topic-hot experts stay resident across mixed
# traffic, so watch the [ecache] hit_rate / disk_reads split as the primary
# counter, then tok/s. Needs job 90's rebuild (COLI_TOPIC_ROUTING is new).
#
# NOTE: the win shows only when traffic actually interleaves TOPICS (coding +
# prose + json requests in one server lifetime) — a single-topic sweep never
# evicts a rival topic's experts, so the tiebreak never bites. This arm uses
# the default mixed corpus; if it reads flat, the follow-up is a deliberately
# topic-interleaved client, not a verdict. Enable AFTER the byte-lever jobs.
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
REPEATS=3 MAX_TOKENS=32 PORT=8146 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-queue/topic-routing-b16 16 \
  bench-data/2026-08-15-queue/arms/topic-off.env \
  bench-data/2026-08-15-queue/arms/topic-on.env

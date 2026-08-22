#!/usr/bin/env bash
# One-shot measurement chain for the 2026-08-15 ds4-port wave — sequenced so
# nothing heavy shares the model stripe with the asymmetric conversion. Each
# stage logs under bench-data/ and a failure skips to the next stage rather
# than killing the chain (a lost stage is a rerun; a lost night is not).
#
#   setsid nohup scripts/overnight-2026-08-15.sh > overnight-2026-08-15.log 2>&1 &
#
# Stages:
#   0. preflight: asymmetric dry-run + free-space check on the stripe
#   1. the conversion itself: experts to int3-g64, down_proj kept, last 6
#      expert layers (incl. the MTP row) untouched — the ds4 recipe, the one
#      untested point on the sub-int4 ladder after docs/todo.md §13 closed every
#      uniform rung
#   2. flip-rate quality gate vs the int4 source (same corpus as the 08-13
#      gate; uniform int3-g64 measured 0.514 there). FLIP_MAX only gates
#      whether stage 3 spends bench hours — the license decision stays human.
#   3. container A/B at B=1 and B=16 (gate-passed only). The candidate lives
#      whole on the stripe while the source is the 3-way split, so tok/s is
#      topology-confounded: the primary metric is GB read/token from the
#      [ecache] counters, tok/s is secondary until an optional reshard.
#   4. COLI_SPEC_CONF A/B on the *source* container (draft 5 vs draft 5 +
#      confidence floor 0.65) — independent of the gate verdict
#   5. COLI_ECACHE_GB fixed-8 vs auto A/B — independent of the gate verdict
#   6. kvstore smoke: same request before/after a server restart must be
#      byte-identical, with the second prefill collapsed by the disk KV
#
# Stages 4/5/6 need serve binaries built AFTER the 2026-08-15 code wave; the
# rebuild happens while stage 1 streams the stripe, hours before they run.
set -u
cd "$(dirname "$0")/.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
CAND=/srv/modelstripe/GLM-5.2-i3g64-asym
CORPUS=bench-data/2026-08-13-route-min-share/corpus.txt
FLIP_MAX=${FLIP_MAX:-0.05}
DATA=bench-data/2026-08-15-i3g64-asym
mkdir -p "$DATA"
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "stage 0: asymmetric dry-run + free-space preflight"
if ! nice -n 10 ./target/release/peregrine-requantize "$COLI_MODEL" "$CAND" \
    --target int3-g64 --down keep --keep-last-layers 6 --dry-run \
    > "$DATA/dry-run.log" 2>&1; then
  stamp "stage 0 dry-run FAILED — aborting (nothing else is meaningful without the container)"
  exit 1
fi
need_gb=$(grep -o 'plan for [0-9.]* GB' "$DATA/dry-run.log" | tail -1 | awk '{print $3}')
free_gb=$(df --output=avail -B1G /srv/modelstripe | tail -1 | tr -d ' ')
stamp "stage 0: needs ~${need_gb:-?} GB, ${free_gb:-?} GB free on the stripe"
if [ -z "$need_gb" ] || ! awk -v n="$need_gb" -v f="$free_gb" 'BEGIN { exit !(f >= n * 1.2) }'; then
  stamp "stage 0 FAILED: want 1.2x headroom over ${need_gb:-unknown} GB — aborting."
  stamp "note: /srv/modelstripe/GLM-5.2-int3g64 (310 GB, the failed uniform rung) is deletable if space is the blocker"
  exit 1
fi

stamp "stage 1: asymmetric int3-g64 conversion (disk-heavy; ionice idle)"
if nice -n 10 ionice -c3 ./target/release/peregrine-requantize "$COLI_MODEL" "$CAND" \
    --target int3-g64 --down keep --keep-last-layers 6 \
    > "$DATA/convert.log" 2>&1; then
  stamp "stage 2: flip-rate gate (512 positions; uniform int3-g64 scored 0.514 here)"
  COLI_ROUTE_STATS_PERSIST=0 nice -n 10 ./target/release/peregrine flip-rate \
    "$COLI_MODEL" "$CAND" --text "$CORPUS" --tokens 512 \
    > "$DATA/flip-rate.log" 2>&1 \
    || stamp "stage 2 FAILED (continuing)"
  rate=$(awk '/^flip_rate/ { print $2 }' "$DATA/flip-rate.log")
  if [ -n "${rate:-}" ] && awk -v r="$rate" -v m="$FLIP_MAX" 'BEGIN { exit !(r <= m) }'; then
    stamp "stage 2: flip_rate $rate <= $FLIP_MAX — spending bench hours on stage 3"
    stamp "stage 3a: source container serve sweep (B=1,16; REPEATS=1)"
    REPEATS=1 MAX_TOKENS=32 PORT=8143 nice -n 5 scripts/bench-serve-batch.sh \
      "$DATA/serve-src" 1 16 || stamp "stage 3a FAILED (continuing)"
    stamp "stage 3b: candidate container serve sweep (B=1,16; REPEATS=1)"
    COLI_MODEL="$CAND" REPEATS=1 MAX_TOKENS=32 PORT=8143 nice -n 5 scripts/bench-serve-batch.sh \
      "$DATA/serve-cand" 1 16 || stamp "stage 3b FAILED (continuing)"
  else
    stamp "stage 2: flip_rate ${rate:-unparseable} > $FLIP_MAX — skipping stage 3; contingency is --keep-last-layers 12, else docs/todo.md §13 gains another closed rung"
  fi
else
  stamp "stage 1 conversion FAILED — skipping the gate and stage 3"
fi

stamp "stage 4: COLI_SPEC_CONF A/B at B=16 (REPEATS=1, source container)"
REPEATS=1 MAX_TOKENS=32 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-specconf 16 \
  bench-data/2026-08-15-specconf/arms/draft5.env \
  bench-data/2026-08-15-specconf/arms/draft5-conf065.env \
  || stamp "stage 4 FAILED (continuing)"

stamp "stage 5: COLI_ECACHE_GB fixed-8 vs auto A/B at B=16 (REPEATS=1)"
REPEATS=1 MAX_TOKENS=32 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-15-ecache-auto 16 \
  bench-data/2026-08-15-ecache-auto/arms/ecache8.env \
  bench-data/2026-08-15-ecache-auto/arms/ecache-auto.env \
  || stamp "stage 5 FAILED (continuing)"

stamp "stage 6: kvstore restart smoke (identity + prefill collapse)"
nice -n 5 scripts/kvstore-smoke.sh bench-data/2026-08-15-kvstore \
  || stamp "stage 6 FAILED"

stamp "chain complete"

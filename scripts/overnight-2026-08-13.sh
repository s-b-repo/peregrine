#!/usr/bin/env bash
# One-shot measurement chain for the 2026-08-13 wave — sequenced so nothing
# heavy shares the model stripe with the int3-g64 conversion. Each stage logs
# under bench-data/ and a failure skips to the next stage rather than killing
# the chain (a lost stage is a rerun; a lost night is not).
#
#   setsid nohup scripts/overnight-2026-08-13.sh > overnight-2026-08-13.log 2>&1 &
#
# Stages:
#   1. COLI_PREDICT_EVAL scoreboard, stdio GEN (the todo §3b PhaseTracker gate)
#   2. int3-g64 dry-run, then the real conversion onto /srv/modelstripe
#   3. int3-g64 flip-rate quality gate vs the int4 source (same corpus as the
#      min-share gate, --candidate-env not needed: two real containers)
#   4. dump-routes + skipbound on the real checkpoint (runbook §8)
#   5. serve-path defaults A/B at B=16 (the honesty check on the flipped
#      defaults; REPEATS=1 — the arms are ~100 min each, well past the box's
#      ±3% settling horizon)
#   6. runbook §2 nine-knob re-run: baseline / improved / gpu arms (first
#      GPU tok/s number, and re-validates the provisional 1.004x with
#      sqpoll/regbuf actually reporting)
set -u
cd "$(dirname "$0")/.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
CORPUS=bench-data/2026-08-13-route-min-share/corpus.txt
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "stage 1: predict-eval GEN 64"
COLI_PREDICT_EVAL=1 COLI_PREDICT_EVAL_N=16 COLI_ROUTE_STATS_PERSIST=0 \
  nice -n 10 ./target/release/peregrine "$COLI_MODEL" \
  > bench-data/2026-08-13-predict-eval/gen64.out \
  2> bench-data/2026-08-13-predict-eval/gen64.log <<'EOF' || stamp "stage 1 FAILED (continuing)"
GEN 64 1 2 3
QUIT
EOF

stamp "stage 2: int3-g64 dry-run"
nice -n 10 ./target/release/peregrine-requantize "$COLI_MODEL" /srv/modelstripe/GLM-5.2-int3g64 \
  --target int3-g64 --dry-run > bench-data/2026-08-13-int3g64/dry-run.log 2>&1 \
  || stamp "stage 2 dry-run FAILED (continuing)"

stamp "stage 2b: int3-g64 conversion (disk-heavy; ionice idle)"
if nice -n 10 ionice -c3 ./target/release/peregrine-requantize "$COLI_MODEL" /srv/modelstripe/GLM-5.2-int3g64 \
  --target int3-g64 > int3g64-convert-2026-08-13.log 2>&1; then
  stamp "stage 3: int3-g64 flip-rate gate (512 positions)"
  COLI_ROUTE_STATS_PERSIST=0 nice -n 10 ./target/release/peregrine flip-rate \
    "$COLI_MODEL" /srv/modelstripe/GLM-5.2-int3g64 \
    --text "$CORPUS" --tokens 512 \
    > bench-data/2026-08-13-int3g64/flip-rate.log 2>&1 \
    || stamp "stage 3 FAILED (continuing)"
else
  stamp "stage 2b conversion FAILED — skipping the int3 gate"
fi

stamp "stage 4: dump-routes + skipbound"
COLI_ROUTE_STATS_PERSIST=0 nice -n 10 ./target/release/peregrine dump-routes "$COLI_MODEL" \
  bench-data/2026-08-13-int3g64/routes.json --text "$CORPUS" \
  > bench-data/2026-08-13-int3g64/dump-routes.log 2>&1 \
  && nice -n 10 ./target/release/peregrine-skipbound "$COLI_MODEL" \
    --trace bench-data/2026-08-13-int3g64/routes.json \
    > bench-data/2026-08-13-int3g64/skipbound.log 2>&1 \
  || stamp "stage 4 FAILED (continuing)"

stamp "stage 5: serve defaults A/B at B=16 (REPEATS=1)"
REPEATS=1 MAX_TOKENS=32 nice -n 5 scripts/bench-serve-envarms.sh \
  bench-data/2026-08-13-defaults-ab 16 \
  bench-data/2026-08-13-defaults-ab/arms/defaults-on.env \
  bench-data/2026-08-13-defaults-ab/arms/defaults-off.env \
  || stamp "stage 5 FAILED (continuing)"

stamp "stage 6: runbook §2 nine-knob re-run (baseline improved gpu)"
nice -n 5 scripts/bench-arms.sh out/rerun-2026-08-13 baseline improved gpu \
  || stamp "stage 6 FAILED"

stamp "chain complete"

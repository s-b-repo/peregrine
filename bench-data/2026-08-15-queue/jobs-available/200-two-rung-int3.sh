#!/usr/bin/env bash
# The two-rung sub-int4 night (agreed ds4-session + coordinator, 08-15): the
# last two evidence-backed points on the int3 ladder, one flip gate each.
#
#   Rung A — data-free, deeper hedge: --keep-last-layers 12 (the 0.447
#            contingency; expected to fail, cheap to know).
#   Rung B — calibrated (ideas #7): capture mean-|x| channel stats on the
#            committed corpus (COLI_PREDICT_EVAL=1 co-runs per the locked
#            synergy — predictor recall rides the same disk slot), then
#            convert with --calib + the full asym recipe (best-chance point:
#            if THIS fails, every RTN variant incl. calibrated is closed and
#            sub-4-bit means vector quantization, todo.md §13).
#
# SPACE SAFETY: the stripe holds the failed 355 GB asym container (user's
# deletion call — untouched). Each rung's candidate is script-owned: a rung
# that FAILS its gate deletes its own container before the next conversion,
# so peak extra usage is one container (~360 GB) against the ~650 GB free.
# A rung that PASSES is kept and announced loudly.
#
# Enable (move to ../jobs/) only after "queue drained" work is scheduled —
# this owns the disk for ~8-9 h. Requires the post-6715c9b release binaries
# (job 90's rebuild provides them).
set -u
cd "$(dirname "$0")/../../.."
export COLI_MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2}
CORPUS=bench-data/2026-08-13-route-min-share/corpus.txt
DATA=bench-data/2026-08-16-two-rung-int3
FLIP_MAX=${FLIP_MAX:-0.05}
mkdir -p "$DATA"
stamp() { echo "== [$(date '+%F %T')] $*"; }
VERDICTS="$DATA/VERDICTS.txt"
: > "$VERDICTS"

gate() { # $1 candidate dir, $2 label → echoes flip_rate or "unparseable"
  COLI_ROUTE_STATS_PERSIST=0 nice -n 10 ./target/release/peregrine flip-rate \
    "$COLI_MODEL" "$1" --text "$CORPUS" --tokens 512 \
    > "$DATA/flip-$2.log" 2>&1 || true
  awk '/^flip_rate/ { print $2; found=1 } END { if (!found) print "unparseable" }' "$DATA/flip-$2.log"
}

run_rung() { # $1 label, $2 outdir, then converter args...
  local label=$1 outdir=$2
  shift 2
  stamp "rung $label: convert -> $outdir"
  if ! nice -n 10 ionice -c3 ./target/release/peregrine-requantize "$COLI_MODEL" "$outdir" "$@" \
      > "$DATA/convert-$label.log" 2>&1; then
    stamp "rung $label conversion FAILED (continuing)"
    echo "$label: CONVERSION FAILED" >> "$VERDICTS"
    rm -rf "$outdir"
    return 1
  fi
  stamp "rung $label: flip gate"
  local rate
  rate=$(gate "$outdir" "$label")
  echo "$label: flip_rate $rate (gate $FLIP_MAX)" >> "$VERDICTS"
  if awk -v r="$rate" -v m="$FLIP_MAX" 'BEGIN { exit !(r <= m) }' 2>/dev/null; then
    stamp "rung $label PASSES ($rate <= $FLIP_MAX) — KEEPING $outdir. License decision stays human."
    echo "$label: PASS — container kept at $outdir" >> "$VERDICTS"
  else
    stamp "rung $label fails ($rate) — deleting its script-owned container"
    rm -rf "$outdir"
    echo "$label: FAIL — container deleted" >> "$VERDICTS"
  fi
}

# ---- Rung A: data-free keep-last-12 ----------------------------------------
run_rung "A-keeplast12" /srv/modelstripe/GLM-5.2-i3g64-kl12 \
  --target int3-g64 --down keep --keep-last-layers 12 \
  || stamp "rung A did not complete (continuing to rung B)"

# ---- Rung B: calibrated ----------------------------------------------------
stamp "rung B: calibration capture on the committed corpus (predict-eval co-run)"
if COLI_PREDICT_EVAL=1 COLI_PREDICT_EVAL_N=16 COLI_ROUTE_STATS_PERSIST=0 \
    nice -n 10 ./target/release/peregrine calib-capture \
    "$COLI_MODEL" "$DATA/calib_channels.json" 8192 --text "$CORPUS" \
    > "$DATA/calib-capture.log" 2>&1; then
  run_rung "B-calibrated" /srv/modelstripe/GLM-5.2-i3g64-calib \
    --target int3-g64 --down keep --keep-last-layers 6 \
    --calib "$DATA/calib_channels.json" \
    || stamp "rung B did not complete"
else
  stamp "rung B capture FAILED — calibrated rung not attempted"
  echo "B-calibrated: CAPTURE FAILED, rung not attempted" >> "$VERDICTS"
fi

stamp "two-rung night complete; verdicts:"
cat "$VERDICTS"

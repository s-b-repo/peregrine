#!/usr/bin/env bash
# COLI_SPEC_GDN A/B on the resident Qwen3.5-27B container.
#
# Discipline, per docs/measurement.md:
#   * isolated fresh process per arm (EngineKnobs resolve once at spawn, so an
#     in-process toggle would compare an arm with itself)
#   * arm order rotated across rounds, so a warming trend cannot be read as an
#     arm difference
#   * distinct prompt per (arm, round) — peregrine-serve answers an exact repeat
#     of a greedy request from its response memo without touching the engine
#   * tokens counted server-side from decode.tokens_emitted, not from the text
set -u
M=/srv/modelstripe/qwen/Qwen3.8-27B-peregrine
OUT=$1; ROUNDS=${2:-3}; NTOK=${3:-48}; PORT=8202
mkdir -p "$OUT"
: > "$OUT/raw.jsonl"

start_server () {   # $1 = arm, $2 = round -> echoes pid
  local arm=$1 round=$2
  if [ "$arm" = "on" ]; then
    COLI_STREAM=0 COLI_DRAFT=5 COLI_SPEC_CONF=0.65 COLI_SPEC_GDN=1 \
      ./target/release/peregrine-serve --model "$M" --port "$PORT" --model-id qwen \
      --max-batch 4 --max-tokens 4096 > "$OUT/$arm-r$round.log" 2>&1 &
  else
    COLI_STREAM=0 COLI_DRAFT=5 COLI_SPEC_CONF=0.65 \
      ./target/release/peregrine-serve --model "$M" --port "$PORT" --model-id qwen \
      --max-batch 4 --max-tokens 4096 > "$OUT/$arm-r$round.log" 2>&1 &
  fi
  echo $!
}

run_arm () {
  local arm=$1 round=$2
  local pid; pid=$(start_server "$arm" "$round")
  local ok=0
  for _ in $(seq 1 80); do
    sleep 3
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then ok=1; break; fi
  done
  if [ "$ok" -eq 0 ]; then
    echo "{\"arm\":\"$arm\",\"round\":$round,\"error\":\"server did not start\"}" >> "$OUT/raw.jsonl"
    kill "$pid" 2>/dev/null
    return
  fi
  curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/$arm-r$round.before.json"
  local t0 t1
  t0=$(date +%s.%N)
  curl -s --max-time 3000 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about heat transfer in metals. Variation $arm-$round.\"}],\"max_tokens\":$NTOK,\"temperature\":0.0}" \
    > "$OUT/$arm-r$round.resp.json"
  t1=$(date +%s.%N)
  curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/$arm-r$round.after.json"
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 2
  OUT="$OUT" ARM="$arm" RND="$round" T0="$t0" T1="$t1" python3 - >> "$OUT/raw.jsonl" <<'PY'
import json, os
out, arm, rnd = os.environ["OUT"], os.environ["ARM"], int(os.environ["RND"])
t0, t1 = float(os.environ["T0"]), float(os.environ["T1"])
b = json.load(open(f"{out}/{arm}-r{rnd}.before.json"))
a = json.load(open(f"{out}/{arm}-r{rnd}.after.json"))
r = json.load(open(f"{out}/{arm}-r{rnd}.resp.json"))
tok = a["decode"]["tokens_emitted"] - b["decode"]["tokens_emitted"]
rows = a["decode"]["rows"] - b["decode"]["rows"]
sp, sb = a["spec"], b["spec"]
d = lambda k: sp.get(k, 0) - sb.get(k, 0)
wall = t1 - t0
print(json.dumps({
    "arm": arm, "round": rnd, "wall_s": round(wall, 2), "tokens": tok,
    "tok_per_s": round(tok / wall, 4) if wall > 0 and tok else None,
    "rows": rows, "rows_per_token": round(rows / tok, 3) if tok else None,
    "proposed": d("proposed"), "accepted": d("accepted"), "conf_stops": d("conf_stops"),
    "gdn_replays": d("gdn_replays"), "gdn_snapshot_mb": round(d("gdn_snapshot_bytes") / (1 << 20), 1),
    "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
}))
PY
}

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="off on"; else order="on off"; fi
  for arm in $order; do
    echo "round $r arm $arm" >&2
    run_arm "$arm" "$r"
  done
done
echo "ALL DONE" >&2

#!/usr/bin/env bash
# reshard-glm53-5ssd.sh — re-stripe the GLM-5.3-Flash container across every SSD
# by **measured** read bandwidth, then verify and leave the old container intact.
#
# Why this exists: a stripe's per-device byte shares must be proportional to each
# device's *read speed*, because every token is gated on its slowest drive. The
# weights are measured here, never carried over from another box — the
# deployment doc's own scar is an equal-weight split that benched slower than one
# drive, because a 5.4 kRPM HDD sat in it.
#
# Fastest read format on Linux for this workload: O_DIRECT with large blocks.
# It bypasses the page cache (so the number is the device, not the cache) and
# halves the copies. 8 MiB is the block peregrine-io's reads are aligned to, so
# the probe measures the geometry the engine will actually use.
#
#   usage: scripts/reshard-glm53-5ssd.sh --measure | --plan | --apply [--confirm]
#
#   --measure   per-device read bandwidth (O_DIRECT, ROUNDS x REPS, median, and
#               the observed spread — a device whose rate swings 2x has a share
#               that is a guess, and you want to see that before re-striping)
#   --plan      dry-run the re-stripe with those weights; writes nothing
#   --apply     re-stripe + byte-verify + move shards into place + write a NEW
#               model dir pointing at them. The source container is never
#               touched, so the old one keeps serving until you switch.
#
# Capacity is checked before anything is written: every device must hold its own
# share, and the stage needs room for the whole container again because moving a
# shard across filesystems is a copy.
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

MODEL=${MODEL:-/srv/m-sdd/GLM-5.3-Flash-4ssd}
NEWMODEL=${NEWMODEL:-/srv/m-sdd/GLM-5.3-Flash-5ssd}
STAGE=${STAGE:-/srv/m-sdb/glm53-reshard-5}
RESHARD=${RESHARD:-target/release/peregrine-reshard}
# Per-device expert directory. **Not** the default on purpose for a re-stripe:
# the shard names are the same (`experts-l10-<group>.safetensors`), so writing
# into an existing `glm53-experts/` would overwrite the container that is
# currently serving, mid-flight. A new directory per stripe keeps the old one
# intact, which is the whole point of staging rather than re-striping in place.
EXPERTDIR=${EXPERTDIR:-glm53-experts-5ssd}
BLOCK_MB=${BLOCK_MB:-8}
REPS=${REPS:-4}
ROUNDS=${ROUNDS:-3}

declare -A DEV=(
    [sda]=/srv/m-sda
    [sdb]=/srv/m-sdb
    [sdc]=/srv/m-sdc
    [sdd]=/srv/m-sdd
    [sde]=/srv/m-sde
)

MODE=${1:---plan}
CONFIRM=0
for a in "$@"; do
    [ "$a" = "--confirm" ] && CONFIRM=1
done

median() {
    sort -n | awk '{a[NR]=$1} END{if(!NR)print 0; else printf "%d", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}

# One O_DIRECT read of the probe, printing dd's own achieved MB/s.
read_mb_s() {
    dd if="$1" of=/dev/null bs="${BLOCK_MB}M" count=48 iflag=direct 2>&1 >/dev/null |
        awk '{for(i=1;i<=NF;i++) if($i=="MB/s") print $(i-1)}'
}

measure() {
    local out=() joined="" d m probe vals med lo hi n spread
    for d in "${!DEV[@]}"; do
        m=${DEV[$d]}
        if [ ! -d "$m" ]; then
            printf '  %-4s %-16s MISSING (not mounted)\n' "$d" "$m" >&2
            continue
        fi
        vals=$(mktemp)
        for _ in $(seq "$ROUNDS"); do
            probe="$m/.reshard-bwtest"
            if ! dd if=/dev/urandom of="$probe" bs="${BLOCK_MB}M" count=48 conv=fsync status=none 2>/dev/null; then
                printf '  %-4s %-16s not writable\n' "$d" "$m" >&2
                rm -f "$vals"
                break
            fi
            for _ in $(seq "$REPS"); do
                read_mb_s "$probe" >>"$vals"
            done
            rm -f "$probe"
        done
        med=$(sort -n "$vals" | median)
        if [ "${med:-0}" -le 0 ]; then
            printf '  %-4s %-16s no reading succeeded\n' "$d" "$m" >&2
            continue
        fi
        lo=$(sort -n "$vals" | head -1)
        hi=$(sort -n "$vals" | tail -1)
        n=$(grep -c . "$vals" 2>/dev/null || echo 0)
        rm -f "$vals"
        spread=$(awk -v lo="$lo" -v hi="$hi" 'BEGIN{ if (lo+0 > 0) printf "%d", (hi-lo)/lo*100; else print 0 }')
        printf '  %-4s %-16s %4d MB/s   (min %d, max %d, %d samples, spread %s%%)\n' \
            "$d" "$m" "$med" "$lo" "$hi" "$n" "$spread" >&2
        out+=("$d:$med")
    done
    [ "${#out[@]}" -gt 0 ] || return 1
    for entry in "${out[@]}"; do joined="${joined:+$joined,}$entry"; done
    printf '%s\n' "$joined"
}

printf '== measured read bandwidth (O_DIRECT, %sM blocks, %s rounds x %s reads, median) ==\n' \
    "$BLOCK_MB" "$ROUNDS" "$REPS" >&2
GROUP_SPEC=$(measure) || { echo "no device could be measured — nothing to do" >&2; exit 2; }
printf 'groups: %s\n\n' "$GROUP_SPEC" >&2

case "$MODE" in
--measure) exit 0 ;;
--plan) ;;
--apply) ;;
*) echo "usage: $0 --measure | --plan | --apply [--confirm]" >&2; exit 2 ;;
esac

[ -x "$RESHARD" ] || { echo "missing $RESHARD (cargo build --release -p peregrine-tools)" >&2; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "missing container $MODEL" >&2; exit 2; }

TOTAL=$(python3 scripts/reshard-container-mb.py "$MODEL" 2>/dev/null || echo 0)
echo "== capacity ==" >&2
printf '  container %s MB; the stage needs about the same again\n' "$TOTAL" >&2
for d in "${!DEV[@]}"; do
    m=${DEV[$d]}
    [ -d "$m" ] || continue
    printf '  %-4s %-16s %s MB free\n' "$d" "$m" "$(df -Pm "$m" | awk 'NR==2{print $4}')" >&2
done
echo >&2

echo "== plan ==" >&2
"$RESHARD" --model "$MODEL" --out "$STAGE" --groups "$GROUP_SPEC" \
    ${ROUTE_STATS:+--route-stats "$ROUTE_STATS"} --dry-run >&2 || exit 1
echo >&2

if [ "$MODE" = "--plan" ]; then
    echo "dry run only; pass --apply --confirm to execute" >&2
    exit 0
fi
if [ "$CONFIRM" -ne 1 ]; then
    echo "refusing to write ~${TOTAL} MB without --confirm" >&2
    exit 2
fi

echo "== re-striping (writes ~${TOTAL} MB and saturates the disks; keep the box idle) ==" >&2
ROUTE_STATS=${ROUTE_STATS:-$MODEL/route_stats.json}
"$RESHARD" --model "$MODEL" --out "$STAGE" --groups "$GROUP_SPEC" \
    --route-stats "$ROUTE_STATS" --verify >&2 || exit 1

for d in "${!DEV[@]}"; do
    m=${DEV[$d]}
    [ -d "$m" ] || continue
    mkdir -p "$m/$EXPERTDIR"
    # Two shapes, and missing either one breaks the container *after* the stage
    # is gone: expert shards are `experts-l<layer>-<group>.safetensors`, while the
    # trunk rides the FIRST group under a rolling name, `trunk-<group>-NNNNN.
    # safetensors`. A single `*-$group.safetensors` glob matches the first and
    # silently misses the second — which is exactly how the 5-drive container came
    # up missing `model.embed_tokens.weight` after a byte-identical verification.
    find "$STAGE" -maxdepth 1 \
        \( -name "*-$d.safetensors" -o -name "trunk-$d-*.safetensors" \) \
        -exec mv -v {} "$m/$EXPERTDIR/" \; >&2
done

mkdir -p "$NEWMODEL"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja route_stats.json; do
    [ -f "$MODEL/$f" ] && cp -v "$MODEL/$f" "$NEWMODEL/" >&2
done

{
    printf '{"paths":['
    first=1
    for d in "${!DEV[@]}"; do
        m=${DEV[$d]}
        [ -d "$m/$EXPERTDIR" ] || continue
        [ "$first" -eq 1 ] || printf ','
        printf '"%s/%s"' "$m" "$EXPERTDIR"
        first=0
    done
    printf ']}\n'
} >"$NEWMODEL/model_paths.json"

printf '\nnew container: %s\n' "$NEWMODEL" >&2
cat <<EOF >&2
Now measure it before switching — a re-stripe that is 5 % slower must be a
one-line change to the model dir, not an incident:

  MODEL_DIR=$NEWMODEL ./scripts/bench-lookahead-arms.sh bench-data/\$(date +%F)-reshard 3
  MODEL_DIR=$NEWMODEL COLI_MODEL=$NEWMODEL ./scripts/serve-glm53-flash-cuda.sh

If it is faster, point peregrine.env at it and delete the stage:
  rm -rf $STAGE
EOF
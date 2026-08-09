#!/usr/bin/env bash
# M4a: verify pre-copied shard groups against their originals and swap the
# model dir's shard files for symlinks onto the split tier.
#
# Run ONLY in the M4a window (after the M3 gate): the swap changes which
# devices the engine streams from. Originals are moved to .trash-presplit/,
# never deleted here — deletion happens by hand only after a model load plus
# a temperature-0 reference-transcript byte-compare passes.
#
#   usage: scripts/swap-shard-links.sh <plan-dir> --map name=/mount [...] [--execute]
#
# Without --execute this is a dry-run: verify checksums, print what would
# happen. A missing pre-copy is copied on the spot (unthrottled — this runs in
# a measurement gap by definition).
set -euo pipefail

PLANDIR=${1:?usage: swap-shard-links.sh <plan-dir> --map name=/mount [...] [--execute]}
shift
declare -A MAP
EXECUTE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --map) IFS='=' read -r n p <<< "$2"; MAP[$n]=$p; shift 2 ;;
        --execute) EXECUTE=1; shift ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
PLAN="$PLANDIR/plan.tsv"
[ -f "$PLAN" ] || { echo "no $PLAN" >&2; exit 1; }

# The model dir is recorded in move.sh by plan-shard-split.py.
MDIR=$(grep -m1 '^MDIR=' "$PLANDIR/move.sh" | sed "s/^MDIR='//; s/'$//")
[ -d "$MDIR" ] || { echo "model dir $MDIR not found" >&2; exit 1; }
BASE=$(basename "$MDIR")
mkdir -p "$MDIR/.trash-presplit"

fail=0 swapped=0 verified=0
while IFS=$'\t' read -r shard bytes device; do
    [ "$shard" = "shard" ] && continue          # header
    [ "$device" = "keep" ] && continue
    mount=${MAP[$device]:-}
    [ -n "$mount" ] || { echo "!! no --map for device '$device'" >&2; exit 1; }
    src="$MDIR/$shard"
    dst="$mount/$BASE/$shard"
    if [ -L "$src" ]; then
        echo "ok   $shard already a symlink"
        continue
    fi
    if [ ! -f "$dst" ]; then
        echo "copy $shard -> $device (missing from pre-copy)"
        [ "$EXECUTE" = 1 ] && { mkdir -p "$mount/$BASE"; cp "$src" "$dst.part" && mv "$dst.part" "$dst"; }
    fi
    if [ "$EXECUTE" = 1 ]; then
        a=$(sha256sum < "$src" | cut -d' ' -f1)
        b=$(sha256sum < "$dst" | cut -d' ' -f1)
        if [ "$a" != "$b" ]; then
            echo "!! checksum mismatch: $shard (src $a dst $b)" >&2
            fail=1
            continue
        fi
        verified=$((verified + 1))
        mv "$src" "$MDIR/.trash-presplit/$shard"
        ln -sfn "$dst" "$src"
        swapped=$((swapped + 1))
        echo "swap $shard -> $device (verified)"
    else
        echo "plan $shard -> $device$([ -f "$dst" ] && echo ' (pre-copied)')"
    fi
done < "$PLAN"

echo
if [ "$EXECUTE" = 1 ]; then
    echo "$verified verified, $swapped swapped; originals in $MDIR/.trash-presplit"
    echo "NEXT: boot the server, byte-compare the temperature-0 reference"
    echo "transcripts (bench-data/2026-08-09-M0-baseline/reference-completions/),"
    echo "and only then consider deleting .trash-presplit."
    [ "$fail" = 0 ] || { echo "!! MISMATCHES ABOVE — nothing further until resolved"; exit 1; }
else
    echo "dry-run only; re-run with --execute in the M4a window"
fi

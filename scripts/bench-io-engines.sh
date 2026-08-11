#!/usr/bin/env bash
# Authoritative `COLI_IO_ENGINE` A/B: uring-buffered vs pread vs uring-O_DIRECT.
#
# Every figure this comparison has produced before was wrong in at least one of
# four ways, so the design is mostly a list of things not to do again:
#
#   * **Every rep is a cold read, enforced not assumed.** `iobench --reps`
#     re-reads one file, so rep 1 is cold and reps 2..N come from the page cache —
#     a buffered arm then measures RAM. Here each rep gets its own shard *and*
#     the shard is evicted with `posix_fadvise(DONTNEED)` first, which needs no
#     root. Without the eviction, "fresh shard" only means "not cached by this
#     run": a first attempt drew 1.41 GB/s from a shard an earlier run had left
#     resident, against 0.86 for the same engine on a cold one.
#   * **No offset wrap.** `rings × iters × blk` must stay under the shard size, or
#     the later reads of a pass hit what the same pass just cached. Shards are
#     **not** uniform — this container has one of 0.89 GB and a 16-byte empty one
#     among 2.69 GB siblings — so undersized shards are skipped rather than
#     silently producing an inflated number.
#   * **4 rings, not 8.** `COLI_IO_RINGS` defaults to 4; a harness measuring a
#     lane the engine never runs is worse than no harness.
#   * **Medians, not single passes.** Five identical passes on this box have
#     spanned 35 % of their median.
#
# The O_DIRECT arm is reported but is **not** comparable to the other two:
# `COLI_IO_ENGINE=pread` silently disables O_DIRECT, so uring-with-O_DIRECT
# against pread-without is a two-variable comparison. The engine question is
# settled by the two buffered arms; O_DIRECT answers a different one.
#
#   usage: scripts/bench-io-engines.sh <out-dir> [first-shard-index]
set -uo pipefail

MODEL=${COLI_MODEL:-/home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp}
BIN=${BIN_IOBENCH:-target/release/examples/iobench}
RINGS=${RINGS:-4}
ITERS=${ITERS:-24}
BLK_MB=${BLK_MB:-16}
REPS=${REPS:-5}

OUT=${1:?usage: bench-io-engines.sh <out-dir> [first-shard-index]}
FIRST=${2:-130}
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "no iobench at $BIN (cargo build --release -p peregrine-io --example iobench)" >&2; exit 1; }

echo "== $RINGS rings x $ITERS iters x ${BLK_MB} MB = $((RINGS * ITERS * BLK_MB)) MB per rep, $REPS reps, one fresh shard each"
free -g | awk '/^Mem:/{print "== mem: "$2"G total, "$7"G available"}'

NEED=$((RINGS * ITERS * BLK_MB * 1024 * 1024))
# Shards big enough that the offsets cannot wrap, newest-index first so the arms
# draw from one homogeneous pool.
mapfile -t POOL < <(find "$MODEL" -name 'out-[0-9]*.safetensors' -size +"$((NEED / 1024))"k | sort)
echo "== shard pool: ${#POOL[@]} of $(find "$MODEL" -name 'out-[0-9]*.safetensors' | wc -l) are >= $((NEED / 1024 / 1024)) MB"

# Evict a file from the page cache. No root needed, and it is what makes each rep
# a device read rather than a RAM read.
drop() { python3 -c "
import os,sys
fd=os.open(sys.argv[1],os.O_RDONLY)
os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
os.close(fd)" "$1" 2>/dev/null; }

i=$FIRST
for arm in "uring 0" "pread 0" "uring 1"; do
    set -- $arm
    engine=$1 direct=$2
    label="$engine$([ "$direct" = 1 ] && echo "-direct" || echo "-buffered")"
    : > "$OUT/$label.raw"
    for _ in $(seq 1 "$REPS"); do
        f=${POOL[$((i % ${#POOL[@]}))]}
        i=$((i + 1))
        drop "$f"
        # reps=1: the repetition is this loop, so each pass reads a shard that was
        # just evicted from the page cache.
        "$BIN" "$f" "$BLK_MB" "$ITERS" "$RINGS" "$direct" "$engine" 1 2>/dev/null \
            | grep -oE '\-> [0-9.]+ GB/s' | grep -oE '[0-9.]+' >> "$OUT/$label.raw"
    done
    printf "%-16s " "$label"
    python3 - "$OUT/$label.raw" <<'PY'
import statistics, sys
v = [float(x) for x in open(sys.argv[1]) if x.strip()]
if not v:
    print("no samples"); raise SystemExit
m = statistics.median(v)
spread = (max(v) - min(v)) / m * 100 if m else 0
print(f"median {m:.2f} GB/s  (min {min(v):.2f}, max {max(v):.2f}, spread {spread:.1f}%)  n={len(v)}")
PY
done

echo
echo "Read the two buffered arms against each other; the spread column is the"
echo "noise floor, and a gap smaller than it is not a result."

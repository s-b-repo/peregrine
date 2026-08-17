#!/usr/bin/env bash
# Gate-0(b): re-download the two corrupt shards (out= names so HF's LFS redirect
# can't hash-rename them), verify non-zero, then re-import the container.
set -u
D=/srv/modelstripe/qwen/Qwen3.8-27B
BASE=https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main
cd "$D"
stamp(){ echo "== [$(date '+%F %T')] $*"; }

printf '%s\n  out=%s\n%s\n  out=%s\n' \
  "$BASE/model-00001-of-00018.safetensors" model-00001-of-00018.safetensors \
  "$BASE/model-00002-of-00018.safetensors" model-00002-of-00018.safetensors > refix-urls.txt

stamp "re-downloading shards 1-2 (uncapped, resumable)"
ionice -c3 aria2c -c -j2 -x2 -s2 --auto-file-renaming=false -i refix-urls.txt \
  > refix-dl.log 2>&1
rc=$?
stamp "aria2c rc=$rc"
[ $rc -eq 0 ] || { stamp "download FAILED — see $D/refix-dl.log"; exit 1; }

stamp "verifying non-zero (the corruption tell)"
for f in model-00001-of-00018.safetensors model-00002-of-00018.safetensors; do
  bad=$(python3 - "$f" <<'PY'
import os,sys
f=sys.argv[1]; sz=os.path.getsize(f); z=0
with open(f,'rb') as fh:
    for i in range(20):
        fh.seek(int(sz*i/20)); b=fh.read(4*1024*1024)
        if b.count(0) > len(b)*0.98: z+=1
print(z)
PY
)
  stamp "  $f: $bad/20 windows >98% zero"
  [ "$bad" -le 1 ] || { stamp "STILL CORRUPT: $f — aborting re-import"; exit 1; }
done
stamp "shards verified non-zero"

stamp "re-importing (delete old container, rebuild int4)"
rm -rf "$D-peregrine"
cd "$HOME/peregrine"
bash bench-data/2026-08-15-queue/jobs-available/110-c2-import-qwen.sh > qwen-reimport.log 2>&1
stamp "re-import done — tail:"; tail -4 "$HOME/peregrine/qwen-reimport.log"
stamp "GATE-0(b) COMPLETE — container rebuilt from clean shards"

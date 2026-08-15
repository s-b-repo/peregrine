#!/usr/bin/env bash
# Track C3: Qwen3.8-27B bf16 source download — 55.6 GB, 18 shards + config +
# tokenizer (repo also ships crc32.txt for post-verify). WAN-bound: this box's
# downlink ceiling is ~1.5 MB/s, so the cap below (1300K) trades ~1.5 h of tail
# for usable interactive headroom; restart without the limit to uncap.
# Resumable (aria2c -c) and ionice-idle; at ~1.3 MB/s the writes are ~0.1% of
# the stripe's measured bandwidth — sanctioned beside the running bench queue
# in bench-data/coordination-2026-08-15.md.
#
#   setsid nohup scripts/download-qwen3.8-27b.sh > qwen-download-2026-08-15.log 2>&1 &
set -u
DEST=/srv/modelstripe/qwen/Qwen3.8-27B
BASE=https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main
mkdir -p "$DEST"
cd "$DEST"

files=(
  config.json generation_config.json model.safetensors.index.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  chat_template.jinja preprocessor_config.json video_preprocessor_config.json
  crc32.txt
)
for i in $(seq -w 1 18); do
  files+=("model-000${i}-of-00018.safetensors")
done

printf '%s\n' "${files[@]/#/$BASE/}" > urls.txt
echo "== [$(date '+%F %T')] starting: $(wc -l < urls.txt) files -> $DEST"
ionice -c3 aria2c -c -j2 -x1 -s1 \
  --max-overall-download-limit=1300K \
  --auto-file-renaming=false \
  --summary-interval=300 \
  -i urls.txt
rc=$?
echo "== [$(date '+%F %T')] aria2c exited rc=$rc"
if [ $rc -eq 0 ] && [ -s crc32.txt ]; then
  echo "== verifying shard checksums against crc32.txt"
  fail=0
  while read -r want name; do
    [ -f "$name" ] || { echo "MISSING $name"; fail=1; continue; }
    got=$(python3 -c "import zlib,sys; print(format(zlib.crc32(open(sys.argv[1],'rb').read())&0xffffffff,'08x'))" "$name")
    [ "$got" = "$want" ] || { echo "CRC MISMATCH $name want=$want got=$got"; fail=1; }
  done < crc32.txt
  [ $fail -eq 0 ] && echo "== all checksums OK — source container complete" \
                  || echo "== CHECKSUM FAILURES — re-run this script to repair (aria2c -c resumes)"
fi

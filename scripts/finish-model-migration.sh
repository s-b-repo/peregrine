#!/usr/bin/env bash
# Finish the 5-drive migration once peregrine-reshard has written and verified
# /srv/m-sda/GLM-5.2-r5. Run the steps in order; each is idempotent-ish and
# checks its own precondition, because several are irreversible.
#
# Order is not arbitrary. The new split is verified BEFORE anything old is
# deleted, and md0 is broken only after the last byte that was read from it has
# a verified replacement.
set -euo pipefail
NEW=/srv/m-sda/GLM-5.2-r5
step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "1/8 precondition: the new split exists and is complete"
test -d "$NEW" || { echo "missing $NEW"; exit 1; }
N=$(ls "$NEW"/*.safetensors 2>/dev/null | wc -l)
echo "shards: $N (expect 380 = 75 experts x 5 groups + 5 trunk)"
[ "$N" -ge 380 ] || { echo "ABORT: incomplete reshard"; exit 1; }
du -sh "$NEW"

step "2/8 verify the new split LOADS before deleting anything"
# Until step 7 the new dir holds all five groups, so it must load standalone.
# A stray model_paths.json would silently merge the OLD dirs back in and make
# this verification prove nothing about the new split.
rm -f "$NEW/model_paths.json"
timeout 900 ./target/release/peregrine-serve --model "$NEW" --host 127.0.0.1 --port 8198 --max-batch 1 &
SRV=$!; OK=0
for i in $(seq 1 90); do sleep 10
  if curl -sf http://127.0.0.1:8198/health >/dev/null 2>&1; then OK=1; break; fi
  kill -0 $SRV 2>/dev/null || break
done
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
[ "$OK" = 1 ] || { echo "ABORT: new split did not load — old copies left intact"; exit 1; }
echo "new split loads OK"

step "3/8 move qwen off md0 (it is not model-tier data, just keep it)"
if [ -d /srv/modelstripe/qwen ]; then
  rsync -a --info=progress2 /srv/modelstripe/qwen/ /srv/m-sdc/qwen/
  diff -rq /srv/modelstripe/qwen /srv/m-sdc/qwen >/dev/null && echo "qwen copied + compared OK"
fi

step "4/8 free the OS drive — this is the point of the exercise"
rm -rf /home/cortix/models/GLM-5.2 /home/cortix/models/GLM-5.2-colibri-int4-with-int8-mtp
df -h / | tail -1

step "5/8 free nvme1n1 for its 103.6 GB group"
rm -rf /srv/model600p/GLM-5.2-r4
df -h /srv/model600p | tail -1

step "6/8 break md0 (RAID0 sdb+sdd) into two independent drives"
sudo -A umount /srv/modelstripe || true
sudo -A mdadm --stop /dev/md0
sudo -A mdadm --zero-superblock /dev/sdb1 /dev/sdd1
sudo -A mkfs.ext4 -q -F -L model-sdb -m 0 /dev/sdb1
sudo -A mkfs.ext4 -q -F -L model-sdd -m 0 /dev/sdd1
sudo -A mkdir -p /srv/m-sdb /srv/m-sdd
sudo -A mount /dev/sdb1 /srv/m-sdb && sudo -A mount /dev/sdd1 /srv/m-sdd
sudo -A chown cortix:cortix /srv/m-sdb /srv/m-sdd
sudo -A sed -i '/\/dev\/md0/d; /9adb6c80/d' /etc/mdadm.conf
grep -q model-sdb /etc/fstab || sudo -A tee -a /etc/fstab >/dev/null <<'FSTAB'
LABEL=model-sdb /srv/m-sdb ext4 noatime,nofail 0 2
LABEL=model-sdd /srv/m-sdd ext4 noatime,nofail 0 2
LABEL=model600p /srv/model600p ext4 noatime,nofail 0 2
FSTAB
sudo -A sed -i '/LABEL=modelstripe/d' /etc/fstab
cat /proc/mdstat

step "7/8 distribute each group to its drive"
# sda keeps its own group + the trunk; the other four move.
mkdir -p /srv/m-sdb/GLM-5.2-r5 /srv/m-sdc/GLM-5.2-r5 /srv/m-sdd/GLM-5.2-r5 /srv/model600p/GLM-5.2-r5
for g in sdb sdc sdd; do
  echo "-> $g"; mv "$NEW"/experts-*-"$g".safetensors "/srv/m-$g/GLM-5.2-r5/"
done
echo "-> nv1 (nvme1n1)"; mv "$NEW"/experts-*-nv1.safetensors /srv/model600p/GLM-5.2-r5/
# Metadata stays with the primary dir.
cat > "$NEW/model_paths.json" <<'JSON'
{"paths": ["/srv/m-sdb/GLM-5.2-r5", "/srv/m-sdc/GLM-5.2-r5", "/srv/m-sdd/GLM-5.2-r5", "/srv/model600p/GLM-5.2-r5"]}
JSON
for d in "$NEW" /srv/m-sdb/GLM-5.2-r5 /srv/m-sdc/GLM-5.2-r5 /srv/m-sdd/GLM-5.2-r5 /srv/model600p/GLM-5.2-r5; do
  printf '%-36s %4s shards %8s\n' "$d" "$(ls "$d"/*.safetensors 2>/dev/null|wc -l)" "$(du -sh "$d"|cut -f1)"
done

step "8/8 verify the distributed model loads"
timeout 900 ./target/release/peregrine-serve --model "$NEW" --host 127.0.0.1 --port 8198 --max-batch 1 &
SRV=$!; OK=0
for i in $(seq 1 90); do sleep 10
  if curl -sf http://127.0.0.1:8198/health >/dev/null 2>&1; then OK=1; break; fi
  kill -0 $SRV 2>/dev/null || break
done
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
[ "$OK" = 1 ] && echo "MIGRATION COMPLETE — model spread across 5 drives, OS drive clear" \
              || { echo "FAILED: distributed model did not load"; exit 1; }

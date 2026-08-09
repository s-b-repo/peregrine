#!/usr/bin/env bash
# Build the model storage tier for the 2-tok/s campaign (M4). RUN AS ROOT:
#
#   sudo scripts/build-model-tier.sh --yes
#
# Creates, DESTRUCTIVELY:
#   md0 = RAID0(sdb1, sdd1), 512 KiB chunk, ext4 -m 0, LABEL=modelstripe,
#         mounted /srv/modelstripe  (~1.86 TiB, ~1.0 GB/s expected)
#   nvme1n1p1 ext4 -m 0, LABEL=model600p, mounted /srv/model600p
#         (~119 GiB on the CPU-direct M.2 link, ~0.77 GB/s expected)
#
# Safety: every target device is verified BY MODEL STRING before any write, so
# a drive-letter shuffle across reboots cannot aim this at the wrong disk. The
# two SATA members get equal 928 GiB partitions so the stripe is 2-wide across
# its whole span (md zones a mixed-size RAID0; the tail zone would run at
# single-drive speed and silently slow whatever lands there).
#
# Deliberately plain ext4, no LUKS: public model weights, and dm-crypt would
# tax the exact byte stream this tier exists to accelerate.
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
[ "${1:-}" = "--yes" ] || {
    echo "This DESTROYS all data on the Acer RE100 (sdb), the Crucial BX500 (sdd)," >&2
    echo "and the Intel 600p (nvme1n1). Re-run with --yes to proceed." >&2
    exit 1
}

expect_model() { # $1=dev $2=model-substring
    local m
    m=$(lsblk -ndo MODEL "$1" 2>/dev/null | tr -d ' ')
    case "$m" in
        *"$2"*) echo "  $1 = $m ok" ;;
        *) echo "!! $1 reports model '$m', expected *$2* — REFUSING, device letters may have moved" >&2
           exit 1 ;;
    esac
}

echo "== identity checks"
expect_model /dev/sdb  "RE100"
expect_model /dev/sdd  "BX500"
# The 600p reports its model number, not its marketing name.
expect_model /dev/nvme1n1 "SSDPEKKW128"

echo "== sanity: targets must not be mounted or in any md array"
for d in sdb sdd nvme1n1; do
    if lsblk -nro MOUNTPOINT "/dev/$d" | grep -q .; then
        echo "!! /dev/$d has a mounted filesystem — REFUSING" >&2; exit 1
    fi
done
if grep -qE 'sdb|sdd|nvme1n1' /proc/mdstat; then
    echo "!! a target is already a RAID member (/proc/mdstat) — REFUSING" >&2; exit 1
fi
[ -e /dev/md0 ] && { echo "!! /dev/md0 already exists — REFUSING" >&2; exit 1; }
if ! grep -q '\[UU\]' /proc/mdstat; then
    echo "!! md1 is not clean [UU] — finish/repair the backup mirror first" >&2; exit 1
fi

echo "== wiping signatures"
for part in /dev/sdb?* /dev/sdd?* /dev/nvme1n1p*; do
    [ -e "$part" ] && mdadm --zero-superblock "$part" 2>/dev/null || true
done
wipefs -a /dev/sdb /dev/sdd /dev/nvme1n1

echo "== partitioning (equal 928 GiB SATA members; whole 600p)"
parted -s /dev/sdb  mklabel gpt mkpart stripe 1MiB 928GiB
parted -s /dev/sdd  mklabel gpt mkpart stripe 1MiB 928GiB
parted -s /dev/nvme1n1 mklabel gpt mkpart data 1MiB 100%
udevadm settle

echo "== md0 RAID0, 512 KiB chunk"
mdadm --create /dev/md0 --run --level=0 --raid-devices=2 --chunk=512K \
      --metadata=1.2 /dev/sdb1 /dev/sdd1

echo "== filesystems (lazy init off: pay the inode-table writes now, in one"
echo "   burst, instead of trickling background I/O into tonight's benchmark arms)"
mkfs.ext4 -q -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0 -L modelstripe /dev/md0
mkfs.ext4 -q -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0 -L model600p /dev/nvme1n1p1

echo "== mounts + persistence"
mkdir -p /srv/modelstripe /srv/model600p
mount /dev/md0 /srv/modelstripe
mount /dev/nvme1n1p1 /srv/model600p
chown cortix:cortix /srv/modelstripe /srv/model600p
mdadm --detail --scan /dev/md0 >> /etc/mdadm.conf
grep -q 'LABEL=modelstripe' /etc/fstab || \
    echo 'LABEL=modelstripe /srv/modelstripe ext4 noatime,nofail 0 2' >> /etc/fstab
grep -q 'LABEL=model600p' /etc/fstab || \
    echo 'LABEL=model600p  /srv/model600p  ext4 noatime,nofail 0 2' >> /etc/fstab

echo "== done"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,LABEL /dev/sdb /dev/sdd /dev/nvme1n1
cat /proc/mdstat
df -h /srv/modelstripe /srv/model600p
echo
echo "Next: qualify with iobench (>=5 reps per fs) before any shard moves."

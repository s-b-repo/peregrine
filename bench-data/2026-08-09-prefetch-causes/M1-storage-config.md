# M1 — storage configuration (runbook §1b)

Date: 2026-08-09 · Box: Ryzen 5 5500, 46 GB RAM, model on `/dev/mapper/luks-3208…`
(LUKS2 → ext4 → `nvme0n1p2`).

Checked first because §1b records the largest unverified effect on the table: a
512-byte LUKS sector measured ~10 % of raw throughput where 4096 restored ~5×,
and an I/O scheduler costs 14–57 % versus `none` on NVMe. If either were wrong
here, every other number in this directory was taken against a crippled device.

## Result: the two tunable items are already optimal; one is unresolved

| Item | Value | Verdict |
|---|---|---|
| `nvme0n1` scheduler | `[none]` mq-deadline kyber bfq | **optimal** — nothing to win |
| `nvme1n1` scheduler | `[none]` mq-deadline kyber bfq | optimal |
| `nvme0n1` `max_hw_sectors_kb` | 128 | **hardware limit, not tunable** — `max_sectors_kb` cannot exceed it |
| `nvme0n1` `max_sectors_kb` | 128 | at the hardware ceiling |
| `dm-0` (LUKS) `logical_block_size` | 512 | suggestive, **not conclusive** — see below |
| `dm-0` `max_sectors_kb` | 128 | inherited from the backing device |
| LUKS **encryption sector size** | **unknown** | needs root |

## The one that is still open

`cryptsetup luksDump` requires root and was not run. The `logical_block_size: 512`
on the dm device is *suggestive* of a 512-byte LUKS sector but does not establish
it: that field reports the backing device's logical block size (the NVMe also
reports 512), and dm-crypt's encryption sector is a separate LUKS2 header field.

To settle it:

```bash
sudo cryptsetup luksDump /dev/nvme0n1p2 | grep -i sector
```

If it reads 512, §1b's recorded ~5× applies and **reformatting the volume at
`--sector-size 4096` would dominate every other lever in this investigation** —
including anything in the prefetch code. It is also destructive (a reformat, not a
tunable), so it is a decision, not a knob.

## What this rules in

`max_hw_sectors_kb=128` means a 6 MB expert weight region is split into ~48
block-layer requests regardless of engine or queue depth. That is normal for NVMe
(MDTS), not a defect, and it is the same for every arm — so it cannot explain a
difference *between* arms, only the absolute rate. Worth knowing when reading any
GB/s figure taken here against the 0.84 / 2.02 GB/s pair from a different box.

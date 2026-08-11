# Full move off the root NVMe (user decision, 2026-08-09 evening)

Supersedes the bandwidth-proportional keep/stripe/600p split in
../2026-08-09-M4a-split/. The user chose maximum space relief over the last
~1.3 GB/s of pool bandwidth, accepting the stated consequence: the streaming
pool becomes stripe (1.00 GB/s qualified pending) + 600p (0.57 measured),
byte share 71.5/28.5 vs bandwidth share 63.7/36.3 (600p capacity-capped at
110 GB), effective ~1.4 GB/s, aggregate ceiling ~1.2-1.3 tok/s on this
storage. The 2 tok/s line needs a storage change from here (the CPU-direct
M.2 slot + a >=2 TB TLC NVMe remains the documented path).

Sequence: rep-1 old-layout stock B=16 kept as the before-datum ->
full-speed rsync -> sha256 verify + symlink swap (originals in
.trash-presplit until morning review) -> temperature-0 reference-transcript
byte-compare gate -> M0new/M1/M2 ladder rerun entirely on the new layout.

## Correction (20:25): the old-layout B=16 before-datum was lost

The runner gated on the existence of b16-rep1.json, but the harness's shell
redirect creates that file when the client STARTS — so the gate fired on an
empty file and the runner killed rep 1 mid-arm (~85% done). Empty JSON
deleted. The old-layout aggregate at B=16 was therefore never successfully
measured (attempt 1: 3600 s timeout guillotine; attempt 2: this). The
"before" story rests on the single-stream lineage (21.83 -> 16.08 s/tok) and
the temperature-0 reference transcripts, which are intact and are the gate
input. Runner-pattern lesson, recorded for future harnesses: gate on JSON
*validity* or process exit, never on file existence next to a redirect.

## Pivot 2 (21:10): reshard replaces the file-level split

The user asked whether reading from both drives at once would be faster.
Answer: literal dual full copies are impossible (600p holds 117 GB against a
384 GB model), but the heat-aware reshard achieves the same effect — every
sparse layer's experts split across BOTH devices, so each layer's reads run
on both spindles concurrently instead of whichever device its ~1.8 shards
landed on. Dry-runs showed the heat skew is weak (routed share tracks stored
share), so the 600p is capacity-capped at weight 0.42 = 107.8 GB stored,
29.7% routed share, effective pool ~1.42 GB/s (stripe-bound). Group files
are written to the stripe, verified byte-for-byte, the 600p group moved
over, and a symlinked model dir assembled; the NVMe originals stay untouched
as rollback until morning review. Found and fixed on the way: reshard's
route-stats loader refused the engine's (n_layers+1)-row heat table — the
same MTP-row trap requant hit, fixed the same way.

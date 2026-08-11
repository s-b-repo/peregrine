# M0 notes

- The reference transcripts (reference-completions/) are from the first launch,
  temperature 0, and remain the bit-identity gold standard.
- **First arm attempt (2026-08-09 ~16:00) was VOID and its files deleted**: the
  client's hardwired 3600 s per-request timeout guillotined every stream at 0
  tokens. A stock-config (unfused-prefill) B=16 request measured >60 min
  end-to-end — prefill without COLI_FUSE_PREFILL dominates deep batches. That
  is itself a finding: fused prefill is not a nicety at depth, it is the
  difference between an arm finishing and not. The rep also overlapped the
  model-tier mkfs burst (~2 min of shared-uplink writes).
- B=32 was moved out of M0 into the M2 sweep (it is one of M2's points anyway);
  a stock B=32 arm costs ~2.5 h and teaches nothing M2 won't.
- The rerun (from ~19:10) uses the cold-start harness: page cache dropped
  before every arm via the narrow sudoers rule. M0 B=16 medians are therefore
  the stock referent for M1's arms, measured under identical methodology.
- Throughout the rerun a throttled (8 MB/s) rsync pre-copy of shard groups to
  the new tier runs only AFTER M1 launches; M0 arms see no pre-copy traffic.

# Physical Checkpoint As An Optimization Surface

The checkpoint doesn't have to be a passive data file. Treat its physical layout as a
compiled execution artifact.

## The optimization problem

For each expert:

```
E_i = (size, frequency, coactivation, latency, device)
```

Construct an optimization problem:

```
minimize_π  E[I/O(π)]
```

where **π** is the physical placement of tensors.

But go beyond ordering individual experts.

## Co-activation superblocks

```
disk block
├─ expert A
├─ expert F
├─ expert Q
└─ expert Z
```

If these experts frequently occur together, one I/O operation can bring all of them in.

The checkpoint becomes a compiled representation of the workload.

## Key insight

The disk layout is a compile-time optimization that the runtime scheduler consumes.
This is complementary to the `--apply` flag in `peregrine-layout-reorg` (see
[Layout tools](layout-tools.md#--apply-physical-checkpoint-rewrite)), which rewrites
the checkpoint so the schedule's order is the physical disk order. The superblock
approach takes this further by grouping experts into co-activation clusters rather
than just ordering them by a 1-D traversal.

## Implementation plan

1. **Collect co-activation statistics**: instrument the router to log which expert sets
   are routed together across the full model (not just per-layer). The existing
   `CoActivation` tracker (`predict.rs`, thresholded by
   [`COLI_FUSE_THRESHOLD`](configuration.md)) already records pair co-firing
   rates — extend this to full set co-occurrence. (An earlier revision of this
   page named a `COLI_COACT_THRESHOLD`; no such knob exists or ever did. The
   pair threshold is `COLI_FUSE_THRESHOLD`, default 0.9, and the hyperedge
   grouping uses half of it.)

2. **Build a co-activation graph**: nodes = experts across all layers, edge weight =
   co-occurrence frequency weighted by gate mass. This is a global graph, not
   per-layer, because blocks that span layers can share I/O when pre-fetching.

3. **Solve co-activation clustering**: use a hypergraph partitioner (the existing
   `COLI_HYPER_SCHED` infrastructure in [concurrent-scheduler.md](concurrent-scheduler.md#dispatch-order-shaping)
   already does this per-forward — extend it to the checkpoint layout). Group experts
   into blocks of size `B` (e.g. 64 MB) that minimize total I/O bytes.

4. **Rewrite the checkpoint**: physically lay out co-activation superblocks so a single
   read of `B` bytes satisfies multiple expert sets. This goes beyond `schedule.json`
   (which is a 1-D ordering hint) — it creates 2-D clusters of co-hot experts.

5. **Runtime prefetch**: when the router predicts experts {A, B, C}, find their
   superblock and issue a single range-read. Use `peregrine route-stats` output to
   validate that superblocks are actually hit.

## Priority

Low-hanging fruit — reordering the checkpoint requires no model code changes, just a
data transformation. The `peregrine-layout-reorg --apply` flag already provides the
rewrite infrastructure. The superblock approach is an enhancement on top: instead of
a greedy 1-D ordering, use the co-activation graph to find 2-D clusters.

The critical gating question: does the co-activation structure persist across
different prompts, or is it prompt-specific? If global, the superblock layout is a
one-time investment that pays for all future inference. If per-domain, the checkpoint
needs domain-tagging (which the config-tag system in [model-format.md](model-format.md#config-tag-guards)
already supports).

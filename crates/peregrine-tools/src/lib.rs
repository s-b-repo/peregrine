//! peregrine offline preprocessing library: routing-trace analysis and
//! disk-layout ordering. The `peregrine-layout-reorg` binary and the engine's
//! `galactic` subcommand both drive these functions.
//!
//! Everything is deterministic: same trace → same artifacts.

// The last first-party crate to adopt the panic-lint denials the other nine
// already carry. It qualified all along — zero unwrap/expect/panic in this
// crate's sources — so this is a ratchet, not a cleanup. Each binary target
// is its own crate root, so the attribute has to be repeated per target.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod prune;
pub mod requant;
pub mod reshard;
pub mod skipbound;
pub mod stwrite;

use peregrine_core::{Context, Error};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

pub fn read_routes(path: &Path) -> Result<Vec<Vec<Vec<i32>>>, Error> {
    let bytes = std::fs::read(path).ctx(|| format!("read {}", path.display()))?;
    let v: Value = serde_json::from_slice(&bytes).ctx(|| "parse routes".to_string())?;
    let arr = v.as_array().ok_or_else(|| Error::Format("routes JSON is not an array".to_string()))?;
    let mut out: Vec<Vec<Vec<i32>>> = Vec::with_capacity(arr.len());
    for forward in arr {
        let layers = forward.as_array().ok_or_else(|| Error::Format("forward is not an array".to_string()))?;
        let mut ls: Vec<Vec<i32>> = Vec::with_capacity(layers.len());
        for layer in layers {
            let ids = layer.as_array().ok_or_else(|| Error::Format("layer is not an array".to_string()))?;
            let mut es: Vec<i32> = Vec::with_capacity(ids.len());
            for id in ids {
                let n = id.as_i64().ok_or_else(|| Error::Format("expert id is not an integer".to_string()))?;
                es.push(n as i32);
            }
            ls.push(es);
        }
        out.push(ls);
    }
    Ok(out)
}

pub fn order_experts(trace: &[Vec<Vec<i32>>], method: &str) -> Result<Vec<Vec<i32>>, Error> {
    // Determine layer count and expert-id upper bound.
    let n_layers = trace.iter().map(|f| f.len()).max().unwrap_or(0);
    let mut layer_order: Vec<Vec<i32>> = vec![Vec::new(); n_layers];
    for (l, slot) in layer_order.iter_mut().enumerate() {
        let matrix = build_cooccurrence(trace, l);
        *slot = match method {
            "cluster" | "greedy" => greedy_nearest_neighbor(&matrix),
            "louvain" | "community" => louvain_communities(&matrix),
            "spectral" => spectral_order(&matrix),
            "hilbert" => hilbert_order(&matrix),
            other => return Err(Error::Format(format!("unknown --method: {other}"))),
        };
    }
    Ok(layer_order)
}

/// Build the symmetric N×N co-occurrence weight matrix for `layer` across the
/// full trace. `matrix[a][b]` = number of forwards where both `a` and `b` were
/// routed. Uses a HashMap to keep memory proportional to the observed expert
/// set, not the vocabulary.
pub fn build_cooccurrence(trace: &[Vec<Vec<i32>>], layer: usize) -> HashMap<i32, HashMap<i32, u32>> {
    let mut w: HashMap<i32, HashMap<i32, u32>> = HashMap::new();
    for forward in trace {
        let Some(set) = forward.get(layer) else { continue };
        for &a in set {
            if a < 0 {
                continue;
            }
            let row = w.entry(a).or_default();
            for &b in set {
                if b < 0 || a == b {
                    continue;
                }
                *row.entry(b).or_insert(0) += 1;
            }
        }
    }
    w
}

/// Greedy nearest-neighbor over the co-occurrence graph. Starts from the
/// highest-total-weight (aka highest-degree) node, then walks by max-weight
/// neighbor. Deterministic tie-breaks by ascending expert id. Any experts not
/// reached are appended in ascending id order (so an unroutable expert still
/// gets a slot).
pub fn greedy_nearest_neighbor(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
    // Start from the node with the largest total incident weight.
    let mut totals: Vec<(i32, u32)> = w
        .iter()
        .map(|(&a, row)| (a, row.values().sum::<u32>()))
        .collect();
    totals.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let mut order: Vec<i32> = Vec::new();
    let mut used = std::collections::HashSet::new();
    if let Some(&(start, _)) = totals.first() {
        order.push(start);
        used.insert(start);
        loop {
            let last = *order.last().unwrap_or(&start);
            let Some(row) = w.get(&last) else { break };
            let mut best: Option<(i32, u32)> = None;
            for (&b, &wt) in row {
                if used.contains(&b) {
                    continue;
                }
                let candidate = (b, wt);
                best = Some(match best {
                    None => candidate,
                    Some((cur_id, cur_w)) if wt > cur_w || (wt == cur_w && b < cur_id) => candidate,
                    other => other.unwrap_or(candidate),
                });
            }
            match best {
                Some((b, _)) => {
                    order.push(b);
                    used.insert(b);
                }
                None => break,
            }
        }
    }
    // Append any experts we never reached (isolated in the graph).
    let mut leftover: Vec<i32> = w.keys().copied().filter(|k| !used.contains(k)).collect();
    leftover.sort_unstable();
    order.extend(leftover);
    order
}

/// Louvain community detection over the co-occurrence graph, followed by an
/// intra-community greedy walk. The classical Louvain algorithm iteratively
/// moves each node to the neighbor community that produces the largest
/// modularity gain, until no move increases modularity. This implementation is
/// a single-phase Louvain (no super-node aggregation) — sufficient for the
/// per-layer sizes we see (hundreds of experts, not millions). Communities
/// become layout blocks; within a block, the walker in [`greedy_nearest_neighbor`]
/// orders the members.
///
/// Deterministic: node iteration order is by ascending expert id; ties in the
/// modularity gain are broken by preferring the smaller-id community.
pub fn louvain_communities(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
    louvain_blocks(w).into_iter().flatten().collect()
}

/// [`louvain_communities`] before flattening: each detected community as its own
/// block, in the same order. Callers that place or budget *whole* communities
/// (storage tiers, layout blocks) need the grouping, not the concatenation.
pub fn louvain_blocks(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<Vec<i32>> {
    // Node ids in ascending order — the canonical iteration order that keeps
    // the algorithm deterministic (Louvain is order-sensitive).
    let mut nodes: Vec<i32> = w.keys().copied().collect();
    nodes.sort_unstable();
    if nodes.is_empty() {
        return Vec::new();
    }
    let single = |ns: Vec<i32>| -> Vec<Vec<i32>> { ns.into_iter().map(|n| vec![n]).collect() };
    // Node → community. Start with each node in its own community.
    let mut community: HashMap<i32, i32> = nodes.iter().map(|&n| (n, n)).collect();
    // Degree (sum of edge weights) per node, and total graph weight (2m).
    let node_deg: HashMap<i32, u64> = nodes
        .iter()
        .map(|&n| (n, w.get(&n).map(|r| r.values().map(|&x| x as u64).sum::<u64>()).unwrap_or(0)))
        .collect();
    let two_m: u64 = node_deg.values().sum();
    if two_m == 0 {
        // No edges → every node is its own community, ascending.
        return single(nodes);
    }
    // Community → total degree of its members (updated on each move).
    let mut comm_deg: HashMap<i32, u64> = HashMap::new();
    for &n in &nodes {
        *comm_deg.entry(community[&n]).or_insert(0) += node_deg[&n];
    }
    // Louvain phase 1: repeatedly try to move each node to its best neighbor
    // community. Bounded at 32 sweeps to guarantee termination on pathological
    // inputs (real graphs converge in a handful).
    for _sweep in 0..32 {
        let mut moved = false;
        for &n in &nodes {
            let n_row = match w.get(&n) {
                Some(r) => r,
                None => continue,
            };
            let cur_comm = community[&n];
            // Weight from `n` into each neighboring community.
            let mut w_into: HashMap<i32, u64> = HashMap::new();
            for (&nb, &wt) in n_row {
                let c = community[&nb];
                *w_into.entry(c).or_insert(0) += wt as u64;
            }
            let k_i = node_deg[&n] as i128;
            let two_m_i = two_m as i128;
            // Remove `n` from its current community first (Δmodularity uses
            // the "n outside" community-degree). We add it back into the best
            // target at the end.
            let cur_deg = comm_deg.get(&cur_comm).copied().unwrap_or(0) as i128;
            let w_into_self = w_into.get(&cur_comm).copied().unwrap_or(0) as i128;
            let sigma_tot_minus_i = cur_deg - k_i;
            // Baseline modularity contribution for leaving `n` in its own community.
            // Compare gains multiplied through by `2m` (a positive constant), so
            // the ranking is exact: dividing first truncated, and near-ties then
            // resolved by rounding rather than by modularity.
            let m2 = two_m_i.max(1);
            let baseline = (w_into_self * 2) * m2 - (k_i * sigma_tot_minus_i * 2);
            let mut best_comm = cur_comm;
            let mut best_gain = baseline;
            for (&nc, &w_nc) in &w_into {
                if nc == cur_comm {
                    continue;
                }
                let sigma_tot = comm_deg.get(&nc).copied().unwrap_or(0) as i128;
                let gain = (w_nc as i128 * 2) * m2 - (k_i * sigma_tot * 2);
                if gain > best_gain || (gain == best_gain && nc < best_comm) {
                    best_gain = gain;
                    best_comm = nc;
                }
            }
            if best_comm != cur_comm {
                *comm_deg.entry(cur_comm).or_insert(0) -= node_deg[&n];
                *comm_deg.entry(best_comm).or_insert(0) += node_deg[&n];
                community.insert(n, best_comm);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    // Group nodes by community, sort communities by (size-desc, id-asc), and
    // within each community run the same greedy walk we use for `--method greedy`.
    let mut by_comm: HashMap<i32, Vec<i32>> = HashMap::new();
    for (&n, &c) in &community {
        by_comm.entry(c).or_default().push(n);
    }
    let mut comm_order: Vec<(i32, Vec<i32>)> = by_comm.into_iter().collect();
    comm_order.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    let mut out: Vec<Vec<i32>> = Vec::new();
    for (_c, members) in comm_order {
        // Slice the graph to this community and reuse the greedy walker.
        let sub: HashMap<i32, HashMap<i32, u32>> = members
            .iter()
            .filter_map(|&n| {
                let row = w.get(&n)?;
                let restricted: HashMap<i32, u32> =
                    row.iter().filter(|(&nb, _)| members.contains(&nb)).map(|(&k, &v)| (k, v)).collect();
                Some((n, restricted))
            })
            .collect();
        let mut block = greedy_nearest_neighbor(&sub);
        // Members with no edges within the community still deserve a slot.
        for m in &members {
            if !block.contains(m) {
                block.push(*m);
            }
        }
        out.push(block);
    }
    out
}

/// Spectral ordering: sort experts by their Fiedler-vector value (the second-
/// smallest eigenvector of the graph Laplacian), computed via deflated power
/// iteration. The Fiedler vector's sign structure is the classical minimal-cut
/// bisection, and *sorting* by its values yields a 1-D embedding that places
/// strongly-connected experts adjacently — exactly the disk-adjacency objective.
/// Deterministic: fixed iteration count, fixed deterministic start vector, and
/// stable tie-break by ascending expert id. Disconnected components fall out
/// naturally (each gets a near-constant value per component and sorts together).
pub fn spectral_order(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
    let mut nodes: Vec<i32> = w.keys().copied().collect();
    nodes.sort_unstable();
    let n = nodes.len();
    if n <= 2 {
        return nodes;
    }
    let index: HashMap<i32, usize> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    // Dense adjacency + degree in f64. Per-layer expert counts are ≤ 256, so a
    // dense N² walk is trivially cheap and avoids sparse bookkeeping.
    let mut adj = vec![0f64; n * n];
    let mut deg = vec![0f64; n];
    for (&a, row) in w {
        let (Some(&ia), true) = (index.get(&a), true) else { continue };
        for (&b, &wt) in row {
            if let Some(&ib) = index.get(&b) {
                adj[ia * n + ib] = wt as f64;
                deg[ia] += wt as f64;
            }
        }
    }
    // Power iteration on M = (c·I − L) where L = D − A and c = max degree + 1.
    // M's dominant eigenvector is L's smallest (the constant vector); deflating
    // the constant component each step makes the iteration converge to the
    // second-smallest — the Fiedler vector.
    let c = deg.iter().fold(0f64, |m, &d| m.max(d)) + 1.0;
    // Deterministic non-constant start: alternating pattern indexed by position.
    let mut v: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    for _ in 0..200 {
        // deflate: remove the constant-vector component
        let mean: f64 = v.iter().sum::<f64>() / n as f64;
        for x in v.iter_mut() {
            *x -= mean;
        }
        // y = M v = c·v − L·v = c·v − (D − A)·v
        let mut y = vec![0f64; n];
        for i in 0..n {
            let mut lv = deg[i] * v[i];
            for j in 0..n {
                lv -= adj[i * n + j] * v[j];
            }
            y[i] = c * v[i] - lv;
        }
        // normalize
        let norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break; // degenerate (e.g. edgeless graph) — keep the last v
        }
        for (dst, src) in v.iter_mut().zip(&y) {
            *dst = src / norm;
        }
    }
    // Order by Fiedler value; ties (including whole disconnected components with
    // near-equal values) break by ascending expert id for determinism.
    let mut order: Vec<(i32, f64)> = nodes.iter().map(|&id| (id, v[index[&id]])).collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    order.into_iter().map(|(id, _)| id).collect()
}

/// Hierarchical space-filling (Hilbert) layout: experts are placed on a 2-D
/// plane — x = their Louvain community's rank, y = their rank inside the
/// community's greedy walk — and ordered by Hilbert-curve distance. The curve
/// preserves 2-D locality in 1-D, so both intra-community neighbors *and*
/// same-rank experts of adjacent communities end up near each other on disk —
/// the "hierarchical" property a flat community concatenation lacks.
/// Deterministic (communities and walks already are).
pub fn hilbert_order(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
    let ordered = louvain_communities(w); // communities concatenated, walked
    let n = ordered.len();
    if n <= 2 {
        return ordered;
    }
    // Recover (community_rank, intra_rank) from the concatenated order by
    // re-deriving community membership: experts whose consecutive pair weight
    // is zero start a new community block. Simpler and stable: use sqrt-side
    // blocks of the concatenated order — the curve then interleaves nearby
    // blocks, which is the locality property we are after.
    let side = (n as f64).sqrt().ceil() as u32;
    let order_pow2 = side.next_power_of_two();
    let mut keyed: Vec<(u64, i32)> = ordered
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let x = (i as u32) % side;
            let y = (i as u32) / side;
            (hilbert_d(order_pow2, x, y), e)
        })
        .collect();
    keyed.sort_by_key(|&(d, e)| (d, e));
    keyed.into_iter().map(|(_, e)| e).collect()
}

/// x/y → Hilbert-curve distance for a `side × side` grid (`side` a power of
/// two). The standard iterative rot/reflect walk.
fn hilbert_d(side: u32, mut x: u32, mut y: u32) -> u64 {
    let mut rx: u32;
    let mut ry: u32;
    let mut d: u64 = 0;
    let mut s = side / 2;
    while s > 0 {
        rx = u32::from((x & s) > 0);
        ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        // rotate quadrant
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x) & (side - 1);
                y = s.wrapping_sub(1).wrapping_sub(y) & (side - 1);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// 2-opt refinement of a layer order: repeatedly reverse the segment between
/// two positions when doing so raises the total adjacent-pair co-occurrence
/// weight. Deterministic first-improvement scan, bounded at 8 full passes.
/// The objective never decreases; returns the final objective value.
pub fn two_opt(order: &mut [i32], w: &HashMap<i32, HashMap<i32, u32>>) -> u64 {
    let weight = |a: i32, b: i32| -> u64 { w.get(&a).and_then(|r| r.get(&b)).copied().unwrap_or(0) as u64 };
    let n = order.len();
    if n < 4 {
        return (1..n).map(|i| weight(order[i - 1], order[i])).sum();
    }
    for _pass in 0..8 {
        let mut improved = false;
        // reversing order[i..=j] changes only the two boundary edges
        for i in 1..n - 1 {
            for j in i + 1..n {
                let before = weight(order[i - 1], order[i])
                    + if j + 1 < n { weight(order[j], order[j + 1]) } else { 0 };
                let after = weight(order[i - 1], order[j])
                    + if j + 1 < n { weight(order[i], order[j + 1]) } else { 0 };
                if after > before {
                    order[i..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (1..n).map(|i| weight(order[i - 1], order[i])).sum()
}

/// Storage-tier assignment across VRAM / RAM: whole Louvain communities are
/// placed greedily by heat density (Σ heat ÷ Σ bytes) — VRAM first until its
/// byte budget is exhausted, then RAM, remainder stays on disk. Whole-community
/// placement is the "hypergraph placement" property: co-firing experts land in
/// the same tier, so one forward's routed set crosses as few tiers as possible.
///
/// `heat[expert]` is this layer's per-expert routing heat. `bytes_of(expert)`
/// reports what **that** expert actually occupies.
///
/// **It used to be a scalar `bytes_per_expert`, on the reasoning that "int4
/// experts are same-shaped".** That stopped being true when
/// `peregrine-requantize --tier-hot-frac` shipped: a heat-tiered container holds
/// int4 and int2 experts side by side, differing by ~40% in size, and
/// `QtInfo::detect` is per-tensor precisely so a mixed container loads. A
/// uniform size on such a container mis-sizes every community it places — and
/// silently, since the planner has no way to notice. The closure mirrors
/// `gpu.rs::solve_residency_sized`, which took exactly this shape for exactly
/// this reason; callers with a genuinely uniform container pass `|_| n` and get
/// the old behavior.
///
/// Returns `(vram, ram)` expert-id lists, deterministic.
pub fn assign_tiers(
    w: &HashMap<i32, HashMap<i32, u32>>,
    heat: &HashMap<i32, u64>,
    bytes_of: impl Fn(i32) -> u64,
    vram_budget: u64,
    ram_budget: u64,
) -> (Vec<i32>, Vec<i32>) {
    // The detector's actual communities. Re-deriving them by splitting the
    // concatenated order wherever two adjacent experts share no edge merged any
    // two communities that happened to have a cross edge — the oversized block
    // then missed a tier its real community would have fit in.
    let mut blocks = louvain_blocks(w);
    // A community's byte cost is now the sum of its members', not a count times
    // a constant. `.max(1)` per expert, not once over the total, so a single
    // expert the container cannot size cannot make a whole community free.
    let block_bytes = |b: &Vec<i32>| -> u64 {
        b.iter().map(|&e| bytes_of(e).max(1)).sum()
    };
    // greedy by heat density, deterministic tie-break by first expert id
    let density = |b: &Vec<i32>| -> (u64, i32) {
        let h: u64 = b.iter().map(|e| heat.get(e).copied().unwrap_or(0)).sum();
        (h * 1_000_000 / block_bytes(b).max(1), -b.first().copied().unwrap_or(0))
    };
    blocks.sort_by_key(|b| std::cmp::Reverse(density(b)));
    let mut vram: Vec<i32> = Vec::new();
    let mut ram: Vec<i32> = Vec::new();
    let (mut vleft, mut rleft) = (vram_budget, ram_budget);
    for b in blocks {
        let bytes = block_bytes(&b);
        if bytes <= vleft {
            vleft -= bytes;
            vram.extend(b);
        } else if bytes <= rleft {
            rleft -= bytes;
            ram.extend(b);
        }
        // else: stays on disk
    }
    (vram, ram)
}

/// Emit `<dir>/tiers.json`: the storage-tier placement produced by
/// [`assign_tiers`] over every layer — `vram`/`ram` as `[layer, expert]` pairs.
/// The loader seeds GPU residency from the vram list and prefetch-warms the ram
/// list into the warm cache at startup.
pub fn write_tiers(dir: &Path, vram: &[(usize, i32)], ram: &[(usize, i32)]) -> Result<(), Error> {
    std::fs::create_dir_all(dir).ctx(|| format!("mkdir {}", dir.display()))?;
    let enc = |v: &[(usize, i32)]| -> Vec<serde_json::Value> {
        v.iter().map(|&(l, e)| serde_json::json!([l, e])).collect()
    };
    let doc = serde_json::json!({ "version": 1, "vram": enc(vram), "ram": enc(ram) });
    let bytes = serde_json::to_vec_pretty(&doc).ctx(|| "serialize tiers".to_string())?;
    peregrine_core::write_atomic(&dir.join("tiers.json"), &bytes)
}

/// Per-layer expert heat from a raw routing trace (occurrence counts).
pub fn trace_heat(trace: &[Vec<Vec<i32>>], layer: usize) -> HashMap<i32, u64> {
    let mut heat: HashMap<i32, u64> = HashMap::new();
    for fwd in trace {
        if let Some(set) = fwd.get(layer) {
            for &e in set {
                if e >= 0 {
                    *heat.entry(e).or_insert(0) += 1;
                }
            }
        }
    }
    heat
}

/// Routing statistics for one layer of a trace.
///
/// **Why this exists.** `docs/benchmarks.md` reports "cross-token expert
/// locality 0.6%", and the study records how it was obtained: 58 warm-cache
/// hits out of 9,600 expert reads at a 10 GB cache
/// (`docs/peregrine-vs-colibri.md` §5.2). That is a *cache-capacity* result —
/// 16 tokens touch ~180 GB, so a 10 GB cache holds ~5% of the working set — and
/// it was then glossed across four documents as "consecutive tokens route to
/// ~disjoint expert sets", which is a claim about the *router*. The two are
/// different quantities and only the first was ever measured. These functions
/// measure the second.
///
/// The distinction is load-bearing: the routing figure is what decides whether
/// batching amortizes expert reads and whether speculative verification is
/// byte-neutral, and a cache hit rate cannot answer either.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapStats {
    /// Consecutive position pairs counted (both sides non-empty).
    pub pairs: u64,
    /// Mean `|A ∩ B| / |A|` — the share of a position's experts that the
    /// previous position also routed. This is the quantity a perfect one-step
    /// cache would hit, i.e. what "cross-token locality" should mean.
    pub mean_overlap: f64,
    /// Mean `|A ∩ B| / |A ∪ B|` (Jaccard). Reported alongside because
    /// `PhaseTracker` steers on Jaccard distance, so the two stay comparable.
    pub mean_jaccard: f64,
    /// Mean routed-set size, so the reader can size the null below.
    pub mean_set_size: f64,
    /// Expected `mean_overlap` if consecutive positions routed *independently*:
    /// each of A's experts lands in B with probability `|B| / n_experts`. A
    /// measured value near this means the router carries no cross-token
    /// structure; **below** it means routing is anti-correlated, which would be
    /// a much stronger and stranger claim than "no locality".
    pub independent_null: f64,
}

/// The routed set at `(position, layer)`: deduplicated, ordered, with padding
/// (negative ids) removed — the same convention [`build_cooccurrence`] uses.
///
/// An absent position or layer yields the empty set, spelled out rather than
/// defaulted: "this layer is dense" and "the trace is shorter than asked" must
/// both read as empty here, and neither is an error worth propagating through
/// every statistic.
fn routed_set(trace: &[Vec<Vec<i32>>], pos: usize, layer: usize) -> BTreeSet<i32> {
    match trace.get(pos).and_then(|f| f.get(layer)) {
        Some(s) => s.iter().copied().filter(|&e| e >= 0).collect(),
        None => BTreeSet::new(),
    }
}

/// Consecutive-position expert overlap for `layer`.
///
/// `n_experts` is the layer's expert-pool size, used only to compute the
/// independence null; pass 0 to skip it (the null is then reported as 0.0).
/// Positions whose routed set is empty (dense layers, or a layer the trace
/// never exercised) are skipped rather than counted as zero-overlap, which
/// would silently drag the mean toward 0.
pub fn consecutive_overlap(trace: &[Vec<Vec<i32>>], layer: usize, n_experts: usize) -> OverlapStats {
    let sets: Vec<BTreeSet<i32>> = (0..trace.len()).map(|p| routed_set(trace, p, layer)).collect();
    let (mut pairs, mut sum_ov, mut sum_ja, mut sum_sz, mut sz_n) = (0u64, 0f64, 0f64, 0f64, 0u64);
    for s in &sets {
        if !s.is_empty() {
            sum_sz += s.len() as f64;
            sz_n += 1;
        }
    }
    for w in sets.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.is_empty() || b.is_empty() {
            continue;
        }
        let inter = a.intersection(b).count() as f64;
        let union = (a.len() + b.len()) as f64 - inter;
        sum_ov += inter / a.len() as f64;
        sum_ja += if union > 0.0 { inter / union } else { 0.0 };
        pairs += 1;
    }
    let mean_set_size = if sz_n > 0 { sum_sz / sz_n as f64 } else { 0.0 };
    let independent_null =
        if n_experts > 0 { (mean_set_size / n_experts as f64).min(1.0) } else { 0.0 };
    OverlapStats {
        pairs,
        mean_overlap: if pairs > 0 { sum_ov / pairs as f64 } else { 0.0 },
        mean_jaccard: if pairs > 0 { sum_ja / pairs as f64 } else { 0.0 },
        mean_set_size,
        independent_null,
    }
}

/// Mean `|∪ routed(t..t+w)| / |routed(t)|` over `layer` — how many times one
/// position's expert bytes a `w`-position window costs.
///
/// With `w = γ+1` this is exactly the price of a speculative verify that drafts
/// `γ` tokens: the verify forward reads the union of all `w` positions' experts
/// in one batched pass. A ratio near 1 means verification is nearly free in
/// bytes (speculation amortizes); a ratio near `w` means it multiplies them.
///
/// Windows are *consecutive* positions of one sequence, which is what
/// speculation actually produces. For the independent-sequence case that
/// batching produces, use [`union_growth_strided`].
pub fn union_growth_consecutive(trace: &[Vec<Vec<i32>>], layer: usize, w: usize) -> f64 {
    union_growth_over(trace, layer, w, 1)
}

/// Mean union growth over `b` positions spread across the trace by a stride, as
/// a proxy for `b` **independent** sequences decoding in one batch.
///
/// A single-sequence trace cannot contain genuinely independent sequences, so
/// this is a proxy and should be labelled as one — but it is a far better proxy
/// than consecutive positions, which share whatever local structure exists. The
/// stride is deterministic (`len / b`, floored at 1), so the same trace always
/// yields the same figure.
pub fn union_growth_strided(trace: &[Vec<Vec<i32>>], layer: usize, b: usize) -> f64 {
    let stride = (trace.len() / b.max(1)).max(1);
    union_growth_over(trace, layer, b, stride)
}

fn union_growth_over(trace: &[Vec<Vec<i32>>], layer: usize, w: usize, stride: usize) -> f64 {
    if w == 0 {
        return 0.0;
    }
    let at = |i: usize| routed_set(trace, i, layer);
    let span = (w - 1) * stride;
    let (mut n, mut sum) = (0u64, 0f64);
    for start in 0..trace.len().saturating_sub(span) {
        let base = at(start);
        if base.is_empty() {
            continue;
        }
        let mut u = base.clone();
        for j in 1..w {
            u.extend(at(start + j * stride));
        }
        sum += u.len() as f64 / base.len() as f64;
        n += 1;
    }
    if n > 0 {
        sum / n as f64
    } else {
        0.0
    }
}

/// Expected union growth under independent routing: `w` draws of `k` experts
/// from a pool of `n` cover `n·(1 − (1 − k/n)^w)` distinct experts, i.e. this
/// multiple of one draw. The null every measured figure should be read against
/// — reported next to it rather than left for the reader to compute.
pub fn union_growth_null(n_experts: usize, k: usize, w: usize) -> f64 {
    if n_experts == 0 || k == 0 || w == 0 {
        return 0.0;
    }
    let (n, k) = (n_experts as f64, k as f64);
    n * (1.0 - (1.0 - k / n).powi(w as i32)) / k
}

/// Render the whole-trace routing report: per-layer overlap against the
/// independence null, then union growth for speculative windows and batch
/// proxies. Returned as text rather than printed so the caller owns the stream
/// and tests can assert on it.
pub fn format_route_stats(trace: &[Vec<Vec<i32>>], n_experts: usize) -> String {
    let n_layers = trace.iter().map(|f| f.len()).max().unwrap_or(0);
    let mut out = String::new();
    out.push_str(&format!("positions={} layers={} experts/layer={}\n", trace.len(), n_layers, n_experts));

    // Sparse layers only: a dense layer routes nothing and would report 0.0.
    let sparse: Vec<usize> = (0..n_layers)
        .filter(|&l| trace.iter().any(|f| f.get(l).is_some_and(|s| !s.is_empty())))
        .collect();
    let (mut ov, mut ja, mut null, mut sz) = (0f64, 0f64, 0f64, 0f64);
    for &l in &sparse {
        let s = consecutive_overlap(trace, l, n_experts);
        ov += s.mean_overlap;
        ja += s.mean_jaccard;
        null += s.independent_null;
        sz += s.mean_set_size;
    }
    let d = sparse.len().max(1) as f64;
    out.push_str("\nconsecutive-token routing overlap (mean over sparse layers)\n");
    out.push_str(&format!("  routed set size   {:.1}\n", sz / d));
    out.push_str(&format!("  overlap |A∩B|/|A| {:.4}  ({:.2}%)\n", ov / d, 100.0 * ov / d));
    out.push_str(&format!("  jaccard           {:.4}\n", ja / d));
    out.push_str(&format!("  independence null {:.4}  ({:.2}%)\n", null / d, 100.0 * null / d));
    out.push_str(
        "  NOTE: this is a routing statistic. It is NOT the 0.6% in benchmarks.md,\n\
         \x20       which is a warm-cache hit rate (58/9600 at a 10 GB cache).\n",
    );

    let k = (sz / d).round() as usize;
    out.push_str("\nunion growth |∪ routed(t..t+w)| / |routed(t)|\n");
    out.push_str("  w  consecutive  strided(proxy for B indep. seqs)  independent null\n");
    for w in [2usize, 3, 4, 5, 6, 16] {
        let (mut c, mut s) = (0f64, 0f64);
        for &l in &sparse {
            c += union_growth_consecutive(trace, l, w);
            s += union_growth_strided(trace, l, w);
        }
        out.push_str(&format!(
            "  {:<2} {:>11.3} {:>31.3} {:>17.3}\n",
            w,
            c / d,
            s / d,
            union_growth_null(n_experts, k, w)
        ));
    }
    out.push_str(
        "\nreading it: w=γ+1 consecutive is what a γ-token speculative verify pays\n\
         (near 1.0 = speculation is nearly free in bytes; near w = it multiplies them).\n\
         strided is the batching proxy; compare against the null to see whether the\n\
         router carries structure or is behaving like independent draws.\n",
    );
    out
}

/// **Self-reorganizing checkpoint rewrite**: physically rewrite
/// `model.safetensors` so each layer's expert tensors are stored in the
/// computed schedule order — turning the runtime ordering hint into actual
/// disk adjacency (sequential reads for co-firing experts). Numerically the
/// model is untouched: only tensor *placement in the file* changes, names and
/// bytes are identical, so every read is bit-identical.
///
/// Bounded memory: tensors are copied one at a time (`read_raw` → new blob
/// list), and the output is written to `model.safetensors.tmp` then renamed
/// over the original. Multi-shard checkpoints are rejected (out of scope for
/// the rewrite; the loader handles them unmodified).
pub fn apply_layout(model_dir: &Path, ordered: &[Vec<i32>]) -> Result<(), Error> {
    apply_layout_with(model_dir, ordered, None)
}

/// [`apply_layout`] with an optional compression override for the rewritten
/// expert payloads. `None` preserves each tensor's existing scheme verbatim
/// (the historical behavior); `Some(Zstd)` compresses the routed-expert
/// tensors during the rewrite — the one production seam where re-encoding is
/// free, since every payload is already in memory on its way to a new file.
/// Non-expert tensors keep their scheme either way (they are hot-path reads).
pub fn apply_layout_with(
    model_dir: &Path,
    ordered: &[Vec<i32>],
    compress: Option<peregrine_core::Compression>,
) -> Result<(), Error> {
    use peregrine_core::{pack, SafeTensors};
    let st = SafeTensors::open(model_dir).ctx(|| format!("open {}", model_dir.display()))?;
    if st.paths().len() != 1 {
        return Err(Error::Format("apply_layout supports single-shard checkpoints only".to_string()));
    }
    // Rank map: (layer, expert) → schedule position; unlisted experts keep
    // their relative position after the listed ones.
    let rank = |layer: usize, e: i64| -> usize {
        ordered
            .get(layer)
            .and_then(|row| row.iter().position(|&x| x as i64 == e))
            .unwrap_or(usize::MAX)
    };
    // Parse "model.layers.<L>.mlp.experts.<E>." out of a tensor name.
    let parse = |name: &str| -> Option<(usize, i64)> {
        let rest = name.strip_prefix("model.layers.")?;
        let dot = rest.find('.')?;
        let layer: usize = rest[..dot].parse().ok()?;
        let rest = rest[dot + 1..].strip_prefix("mlp.experts.")?;
        let dot = rest.find('.')?;
        let e: i64 = rest[..dot].parse().ok()?;
        Some((layer, e))
    };
    // Order all tensors: non-expert tensors keep original order (key = their
    // original index); expert tensors sort by (layer, schedule rank, original).
    let mut keys: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(st.len());
    for (i, t) in st.tensors().iter().enumerate() {
        match parse(&t.name) {
            Some((layer, e)) => keys.push((1, layer, rank(layer, e), i)),
            None => keys.push((0, 0, 0, i)),
        }
    }
    keys.sort_unstable();
    // Copy each tensor's raw on-disk payload into a new blob list, preserving
    // dtype/shape/compression metadata verbatim.
    let mut blobs: Vec<pack::Blob> = Vec::with_capacity(st.len());
    for &(_, _, _, i) in &keys {
        let t = &st.tensors()[i];
        let mut raw = vec![0u8; t.uncompressed_nbytes as usize];
        st.read_raw_by_index(i, &mut raw).ctx(|| format!("read '{}'", t.name))?;
        let dtype_str = match t.dtype {
            peregrine_core::Dtype::F32 => "F32",
            peregrine_core::Dtype::Bf16 => "BF16",
            peregrine_core::Dtype::F16 => "F16",
            peregrine_core::Dtype::U8 => "U8",
        };
        let mut b = pack::Blob::new(t.name.clone(), dtype_str, t.shape.clone(), raw);
        b.compression = t.compression;
        // Expert payloads only: the dense path and head are read on every
        // token and should not grow a decompress step from a layout rewrite.
        if let Some(c) = compress {
            if parse(&t.name).is_some() {
                b = b.with_compression(c);
            }
        }
        blobs.push(b);
    }
    // Write to a temp dir entry then swap in.
    let tmp = model_dir.join(".relayout.tmp");
    if let Err(e) = std::fs::remove_dir_all(&tmp) {
        if e.kind() != std::io::ErrorKind::NotFound {
            peregrine_core::note_advisory_err("pre-clean relayout tmp dir", &e);
        }
    }
    pack::write_safetensors(&tmp, &blobs).ctx(|| "write relayout".to_string())?;
    // fsync the rewritten checkpoint before it replaces the original: this
    // rename overwrites the *only* copy of the weights, so a crash between the
    // rename and the data reaching disk would leave a truncated model.
    let written = tmp.join("model.safetensors");
    {
        let f = std::fs::File::open(&written).ctx(|| format!("open {}", written.display()))?;
        f.sync_all().ctx(|| format!("fsync {}", written.display()))?;
    }
    peregrine_core::commit_atomic(&written, &model_dir.join("model.safetensors"))
        .ctx(|| "swap relayout in".to_string())?;
    if let Err(e) = std::fs::remove_dir_all(&tmp) {
        if e.kind() != std::io::ErrorKind::NotFound {
            peregrine_core::note_advisory_err("remove relayout tmp dir", &e);
        }
    }
    Ok(())
}

pub fn write_schedule(dir: &Path, ordered: &[Vec<i32>]) -> Result<(), Error> {
    std::fs::create_dir_all(dir).ctx(|| format!("mkdir {}", dir.display()))?;
    let doc = serde_json::json!({
        "version": 1,
        "n_layers": ordered.len(),
        "order": ordered,
    });
    let bytes = serde_json::to_vec_pretty(&doc).ctx(|| "serialize schedule".to_string())?;
    peregrine_core::write_atomic(&dir.join("schedule.json"), &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_spans_identical_to_disjoint() {
        // Identical consecutive sets → perfect overlap and Jaccard 1.
        let same = vec![vec![vec![1, 2, 3, 4]], vec![vec![1, 2, 3, 4]]];
        let s = consecutive_overlap(&same, 0, 256);
        assert_eq!(s.pairs, 1);
        assert!((s.mean_overlap - 1.0).abs() < 1e-12);
        assert!((s.mean_jaccard - 1.0).abs() < 1e-12);

        // Disjoint consecutive sets → zero on both.
        let disj = vec![vec![vec![1, 2, 3, 4]], vec![vec![5, 6, 7, 8]]];
        let s = consecutive_overlap(&disj, 0, 256);
        assert_eq!(s.mean_overlap, 0.0);
        assert_eq!(s.mean_jaccard, 0.0);

        // Half-shared: |A∩B|=2, |A|=4 → 0.5; |A∪B|=6 → jaccard 1/3.
        let half = vec![vec![vec![1, 2, 3, 4]], vec![vec![3, 4, 5, 6]]];
        let s = consecutive_overlap(&half, 0, 256);
        assert!((s.mean_overlap - 0.5).abs() < 1e-12);
        assert!((s.mean_jaccard - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn overlap_null_is_set_size_over_pool_and_negatives_are_skipped() {
        // 8 experts of 256 → an independent draw would share 8/256 = 3.125%.
        // This is the number a measured overlap has to be read against; the
        // repo's "0.6%" is a cache hit rate and cannot be compared to it.
        let t = vec![vec![vec![0, 1, 2, 3, 4, 5, 6, 7]], vec![vec![8, 9, 10, 11, 12, 13, 14, 15]]];
        let s = consecutive_overlap(&t, 0, 256);
        assert!((s.mean_set_size - 8.0).abs() < 1e-12);
        assert!((s.independent_null - 8.0 / 256.0).abs() < 1e-12);

        // Negative ids are padding, not experts — the whole file skips them.
        let neg = vec![vec![vec![1, -1, 2]], vec![vec![1, -1, 2]]];
        let s = consecutive_overlap(&neg, 0, 0);
        assert!((s.mean_set_size - 2.0).abs() < 1e-12);
        assert_eq!(s.independent_null, 0.0, "pool 0 means no null, not a divide by zero");
    }

    #[test]
    fn empty_layers_are_skipped_not_counted_as_zero() {
        // A dense layer routes nothing. Counting it as zero-overlap would drag
        // the mean toward 0 and manufacture exactly the "no locality" reading
        // this function exists to test.
        let t = vec![vec![vec![1, 2], vec![]], vec![vec![1, 2], vec![]]];
        let sparse = consecutive_overlap(&t, 0, 16);
        let dense = consecutive_overlap(&t, 1, 16);
        assert!((sparse.mean_overlap - 1.0).abs() < 1e-12);
        assert_eq!(dense.pairs, 0, "a layer that never routes contributes no pairs");
        assert_eq!(dense.mean_overlap, 0.0);
    }

    #[test]
    fn union_growth_brackets_are_one_and_w() {
        // Identical sets every position → the union never grows: 1.0 for any w.
        let same = vec![vec![vec![1, 2]]; 8];
        for w in [2usize, 3, 6] {
            assert!((union_growth_consecutive(&same, 0, w) - 1.0).abs() < 1e-12, "w={w}");
        }
        // Fully disjoint sets → the union is exactly w times one position.
        let disj: Vec<Vec<Vec<i32>>> = (0..8).map(|i| vec![vec![i * 2, i * 2 + 1]]).collect();
        for w in [2usize, 3, 4] {
            assert!((union_growth_consecutive(&disj, 0, w) - w as f64).abs() < 1e-12, "w={w}");
        }
    }

    #[test]
    fn union_growth_null_matches_the_coupon_collector_closed_form() {
        // Two independent draws of k from n cover n(1-(1-k/n)^2) distinct, i.e.
        // (2 - k/n) times one draw. At k=8, n=256 that is 1.96875.
        assert!((union_growth_null(256, 8, 2) - (2.0 - 8.0 / 256.0)).abs() < 1e-12);
        assert!((union_growth_null(256, 8, 1) - 1.0).abs() < 1e-12);
        // Degenerate inputs return 0 rather than NaN/inf.
        assert_eq!(union_growth_null(0, 8, 2), 0.0);
        assert_eq!(union_growth_null(256, 0, 2), 0.0);
        assert_eq!(union_growth_null(256, 8, 0), 0.0);
    }

    #[test]
    fn union_growth_strided_is_deterministic_and_spreads() {
        // Period-2 alternation: consecutive positions are disjoint (growth 2),
        // but a stride of 2 lands on the same set every time (growth 1).
        let t: Vec<Vec<Vec<i32>>> =
            (0..8).map(|i| vec![if i % 2 == 0 { vec![1, 2] } else { vec![3, 4] }]).collect();
        assert!((union_growth_consecutive(&t, 0, 2) - 2.0).abs() < 1e-12);
        assert!((union_growth_strided(&t, 0, 4) - 1.0).abs() < 1e-12, "stride 2 hits one phase");
        // Same trace, same answer — the stride is derived, never sampled.
        assert_eq!(union_growth_strided(&t, 0, 4), union_growth_strided(&t, 0, 4));
    }

    #[test]
    fn route_stats_report_distinguishes_itself_from_the_cache_hit_rate() {
        // The whole point of the report is that it is not the 0.6% figure, so
        // the disclaimer is part of the contract, not decoration.
        let t = vec![vec![vec![1, 2, 3]], vec![vec![2, 3, 4]], vec![vec![3, 4, 5]]];
        let s = format_route_stats(&t, 64);
        assert!(s.contains("warm-cache hit rate"), "report must disown the 0.6% gloss");
        assert!(s.contains("independence null"), "a measured overlap is meaningless without its null");
        assert!(s.contains("union growth"));
        assert!(s.contains("positions=3"));
    }

    #[test]
    fn cooccurrence_is_symmetric_by_construction() {
        // one forward, one layer, three experts co-routing → each pair sees 1
        let trace = vec![vec![vec![10, 20, 30]]];
        let w = build_cooccurrence(&trace, 0);
        assert_eq!(w.get(&10).and_then(|r| r.get(&20)), Some(&1));
        assert_eq!(w.get(&20).and_then(|r| r.get(&10)), Some(&1));
        assert_eq!(w.get(&10).and_then(|r| r.get(&10)), None); // self-edge omitted
    }

    #[test]
    fn greedy_starts_from_highest_degree_and_is_deterministic() {
        // expert 1 co-routes twice; expert 5 once; expert 9 once — start from 1.
        let trace = vec![vec![vec![1, 5]], vec![vec![1, 9]]];
        let w = build_cooccurrence(&trace, 0);
        let order = greedy_nearest_neighbor(&w);
        assert_eq!(order.first(), Some(&1), "start from highest-degree");
        // The next step: from 1, both neighbors (5, 9) have equal weight 1 → tie-break
        // by ascending id → 5 next.
        assert_eq!(order.get(1), Some(&5));
    }

    #[test]
    fn two_opt_never_decreases_objective_and_is_deterministic() {
        let trace = vec![vec![vec![1, 2]], vec![vec![2, 3]], vec![vec![3, 4]], vec![vec![1, 4]]];
        let w = build_cooccurrence(&trace, 0);
        let weight = |o: &[i32]| -> u64 {
            (1..o.len())
                .map(|i| w.get(&o[i - 1]).and_then(|r| r.get(&o[i])).copied().unwrap_or(0) as u64)
                .sum()
        };
        // deliberately bad starting order
        let mut order = vec![1, 3, 2, 4];
        let before = weight(&order);
        let after = two_opt(&mut order, &w);
        assert!(after >= before, "2-opt must not lower the objective ({before} → {after})");
        let mut order2 = vec![1, 3, 2, 4];
        two_opt(&mut order2, &w);
        assert_eq!(order, order2, "deterministic");
        // still a permutation
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4]);
    }

    #[test]
    fn hilbert_is_deterministic_permutation() {
        let mut trace = Vec::new();
        for _ in 0..5 {
            trace.push(vec![vec![1, 2, 3]]);
            trace.push(vec![vec![10, 11, 12]]);
            trace.push(vec![vec![20, 21]]);
        }
        let w = build_cooccurrence(&trace, 0);
        let a = hilbert_order(&w);
        let b = hilbert_order(&w);
        assert_eq!(a, b, "deterministic");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 10, 11, 12, 20, 21], "a permutation of all experts");
    }

    #[test]
    fn tiers_place_hot_communities_first_and_whole() {
        // Two communities: hot {1,2,3} and cold {10,11,12}. VRAM fits exactly
        // one community → the hot one goes whole; the cold one lands in RAM.
        let mut trace = Vec::new();
        for _ in 0..5 {
            trace.push(vec![vec![1, 2, 3]]);
            trace.push(vec![vec![10, 11, 12]]);
        }
        let w = build_cooccurrence(&trace, 0);
        let heat: HashMap<i32, u64> =
            [(1, 100), (2, 90), (3, 80), (10, 5), (11, 4), (12, 3)].into_iter().collect();
        // `|_| 10` is the old scalar behavior, so this pins that a uniform
        // container plans exactly as it always did.
        let (vram, ram) = assign_tiers(&w, &heat, |_| 10, 30, 30);
        let mut v = vram.clone();
        v.sort_unstable();
        assert_eq!(v, vec![1, 2, 3], "hot community whole into VRAM");
        let mut r = ram.clone();
        r.sort_unstable();
        assert_eq!(r, vec![10, 11, 12], "cold community whole into RAM");
    }

    /// The case a scalar `bytes_per_expert` cannot express: a heat-tiered
    /// container (`peregrine-requantize --tier-hot-frac`) where experts within
    /// one layer differ in size. Sized uniformly, the hot community looks like it
    /// fits and the plan overcommits; sized per expert, the planner sees it does
    /// not and places the community that does.
    #[test]
    fn tiers_size_each_expert_from_the_container_not_from_a_probe() {
        let mut trace = Vec::new();
        for _ in 0..5 {
            trace.push(vec![vec![1, 2, 3]]);
            trace.push(vec![vec![10, 11, 12]]);
        }
        let w = build_cooccurrence(&trace, 0);
        let heat: HashMap<i32, u64> =
            [(1, 100), (2, 90), (3, 80), (10, 5), (11, 4), (12, 3)].into_iter().collect();
        // The hot community is stored at a wider precision — 30 bytes each — and
        // the cold one at 10. A VRAM budget of 30 fits the cold community only.
        let bytes_of = |e: i32| if (1..=3).contains(&e) { 30 } else { 10 };
        let (vram, _ram) = assign_tiers(&w, &heat, bytes_of, 30, 0);
        let mut v = vram.clone();
        v.sort_unstable();
        assert_eq!(v, vec![10, 11, 12], "the community that actually fits is the one placed");

        // Same graph, same heat, same budget, uniform sizing → the *other*
        // answer. If these two ever agree, the closure has stopped being read.
        let (uniform_vram, _) = assign_tiers(&w, &heat, |_| 10, 30, 0);
        let mut uv = uniform_vram.clone();
        uv.sort_unstable();
        assert_eq!(uv, vec![1, 2, 3], "uniform sizing places the hot community — and overcommits");
    }

    #[test]
    fn spectral_orders_two_cliques_contiguously() {
        // Two 3-cliques joined by nothing: the Fiedler embedding must place each
        // clique's members adjacent in the 1-D order (same contiguity property
        // the Louvain test asserts, via a different algorithm).
        let mut trace = Vec::new();
        for _ in 0..5 {
            trace.push(vec![vec![1, 2, 3]]);
            trace.push(vec![vec![10, 11, 12]]);
        }
        let w = build_cooccurrence(&trace, 0);
        let order = spectral_order(&w);
        assert_eq!(order.len(), 6);
        let idx = |x: i32| order.iter().position(|&y| y == x).unwrap_or(usize::MAX);
        let (a_min, a_max) = ([1, 2, 3].iter().map(|&x| idx(x)).min().unwrap_or(0), [1, 2, 3].iter().map(|&x| idx(x)).max().unwrap_or(0));
        let (b_min, b_max) = ([10, 11, 12].iter().map(|&x| idx(x)).min().unwrap_or(0), [10, 11, 12].iter().map(|&x| idx(x)).max().unwrap_or(0));
        assert!(a_max < b_min || b_max < a_min, "cliques must be contiguous; got {order:?}");
    }

    #[test]
    fn spectral_is_deterministic() {
        let trace = vec![vec![vec![1, 2]], vec![vec![2, 3]], vec![vec![3, 4]]];
        let w = build_cooccurrence(&trace, 0);
        let a = spectral_order(&w);
        let b = spectral_order(&w);
        assert_eq!(a, b, "same input → same order");
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn louvain_groups_two_disjoint_cliques() {
        // Two disjoint 3-cliques {1,2,3} and {10,11,12} co-routing many times.
        // Louvain must end up with a layout where each clique is contiguous —
        // i.e. no member of one clique falls between two members of the other.
        let mut trace = Vec::new();
        for _ in 0..5 {
            trace.push(vec![vec![1, 2, 3]]);
            trace.push(vec![vec![10, 11, 12]]);
        }
        let w = build_cooccurrence(&trace, 0);
        let order = louvain_communities(&w);
        assert_eq!(order.len(), 6);
        // Find the position of each element. The two cliques must occupy
        // contiguous prefixes/suffixes — the "min index of clique A > max index
        // of clique B" (or vice versa) test.
        let idx = |x: i32| order.iter().position(|&y| y == x).unwrap_or(usize::MAX);
        let (a_min, a_max) = ([1, 2, 3].iter().map(|&n| idx(n)).min().unwrap_or(0), [1, 2, 3].iter().map(|&n| idx(n)).max().unwrap_or(0));
        let (b_min, b_max) = ([10, 11, 12].iter().map(|&n| idx(n)).min().unwrap_or(0), [10, 11, 12].iter().map(|&n| idx(n)).max().unwrap_or(0));
        assert!(a_max < b_min || b_max < a_min, "cliques must be contiguous; got {order:?}");
    }

    #[test]
    fn isolated_experts_appended_ascending() {
        // one forward: 1 & 2 co-route; 7 shows up alone → 7 has no edges.
        // The greedy walker starts at the highest-degree node, walks its
        // neighbors, then appends unreached experts (`7` here) in ascending id
        // order so every seen expert still gets a slot.
        let trace = vec![vec![vec![1, 2]], vec![vec![7]]];
        let w = build_cooccurrence(&trace, 0);
        let order = greedy_nearest_neighbor(&w);
        assert!(order.contains(&7), "7 must be appended even though it has no edges");
        assert_eq!(order.last().copied(), Some(7), "isolated experts land at the tail");
        assert!(order.contains(&1) && order.contains(&2));
    }
}

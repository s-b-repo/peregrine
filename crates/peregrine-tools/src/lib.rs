//! peregrine offline preprocessing library: routing-trace analysis and
//! disk-layout ordering. The `peregrine-layout-reorg` binary and the engine's
//! `galactic` subcommand both drive these functions.
//!
//! Everything is deterministic: same trace → same artifacts.

use peregrine_core::{Context, Error};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
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
    // Node ids in ascending order — the canonical iteration order that keeps
    // the algorithm deterministic (Louvain is order-sensitive).
    let mut nodes: Vec<i32> = w.keys().copied().collect();
    nodes.sort_unstable();
    if nodes.is_empty() {
        return Vec::new();
    }
    // Node → community. Start with each node in its own community.
    let mut community: HashMap<i32, i32> = nodes.iter().map(|&n| (n, n)).collect();
    // Degree (sum of edge weights) per node, and total graph weight (2m).
    let node_deg: HashMap<i32, u64> = nodes
        .iter()
        .map(|&n| (n, w.get(&n).map(|r| r.values().map(|&x| x as u64).sum::<u64>()).unwrap_or(0)))
        .collect();
    let two_m: u64 = node_deg.values().sum();
    if two_m == 0 {
        // No edges → every node stays in its own community; append ascending.
        return nodes;
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
            let baseline = (w_into_self * 2) - (k_i * sigma_tot_minus_i * 2) / two_m_i.max(1);
            let mut best_comm = cur_comm;
            let mut best_gain = baseline;
            for (&nc, &w_nc) in &w_into {
                if nc == cur_comm {
                    continue;
                }
                let sigma_tot = comm_deg.get(&nc).copied().unwrap_or(0) as i128;
                let gain = (w_nc as i128 * 2) - (k_i * sigma_tot * 2) / two_m_i.max(1);
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
    let mut out = Vec::new();
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
        out.extend(greedy_nearest_neighbor(&sub));
        // Members with no edges within the community still deserve a slot.
        for m in &members {
            if !out.contains(m) {
                out.push(*m);
            }
        }
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
/// `heat[expert]` is this layer's per-expert routing heat; `bytes_per_expert`
/// is uniform (int4 experts are same-shaped). Returns `(vram, ram)` expert-id
/// lists, deterministic.
pub fn assign_tiers(
    w: &HashMap<i32, HashMap<i32, u32>>,
    heat: &HashMap<i32, u64>,
    bytes_per_expert: u64,
    vram_budget: u64,
    ram_budget: u64,
) -> (Vec<i32>, Vec<i32>) {
    // communities from the same detector the layout uses
    let ordered = louvain_communities(w);
    // rebuild community blocks: split on zero-weight adjacency (community
    // boundaries in the concatenated walk have no intra-edge) — plus singletons.
    let mut blocks: Vec<Vec<i32>> = Vec::new();
    for &e in &ordered {
        let extend = blocks.last().is_some_and(|b: &Vec<i32>| {
            b.last().is_some_and(|&prev| w.get(&prev).and_then(|r| r.get(&e)).copied().unwrap_or(0) > 0)
        });
        if extend {
            if let Some(b) = blocks.last_mut() {
                b.push(e);
                continue;
            }
        }
        blocks.push(vec![e]);
    }
    // greedy by heat density, deterministic tie-break by first expert id
    let density = |b: &Vec<i32>| -> (u64, i32) {
        let h: u64 = b.iter().map(|e| heat.get(e).copied().unwrap_or(0)).sum();
        let bytes = (b.len() as u64) * bytes_per_expert.max(1);
        (h * 1_000_000 / bytes, -b.first().copied().unwrap_or(0))
    };
    blocks.sort_by_key(|b| std::cmp::Reverse(density(b)));
    let mut vram: Vec<i32> = Vec::new();
    let mut ram: Vec<i32> = Vec::new();
    let (mut vleft, mut rleft) = (vram_budget, ram_budget);
    for b in blocks {
        let bytes = (b.len() as u64) * bytes_per_expert.max(1);
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
    let out = dir.join("tiers.json");
    let mut f = std::fs::File::create(&out).ctx(|| format!("create {}", out.display()))?;
    f.write_all(&bytes).ctx(|| "write tiers.json".to_string())?;
    Ok(())
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
    std::fs::rename(tmp.join("model.safetensors"), model_dir.join("model.safetensors"))
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
    let out_path = dir.join("schedule.json");
    let mut f = std::fs::File::create(&out_path).ctx(|| format!("create {}", out_path.display()))?;
    f.write_all(&bytes).ctx(|| "write schedule.json".to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (vram, ram) = assign_tiers(&w, &heat, 10, 30, 30);
        let mut v = vram.clone();
        v.sort_unstable();
        assert_eq!(v, vec![1, 2, 3], "hot community whole into VRAM");
        let mut r = ram.clone();
        r.sort_unstable();
        assert_eq!(r, vec![10, 11, 12], "cold community whole into RAM");
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

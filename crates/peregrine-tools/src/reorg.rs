//! `peregrine-layout-reorg`: rearrange experts in a routing-aware order so
//! contiguous batched reads coalesce more of the disk queue.
//!
//! Input:  `--routes routes.json` — the raw routing trace emitted by
//!         `peregrine dump-routes` (a `Vec<Vec<Vec<i32>>>` — forwards × layers ×
//!         routed-expert-ids).
//! Output: `--out out_dir/schedule.json` — a per-layer ordered expert-id list,
//!         used at load time as an eviction/prefetch ordering hint. The
//!         `Model::load` path can look up this file next to the checkpoint.
//!
//! Method: greedy nearest-neighbor over the per-layer co-occurrence graph. For
//! each layer, build an N×N weight matrix `W[a][b] = frames where a and b were
//! both routed`. Start from the highest-degree expert and repeatedly append the
//! not-yet-added neighbor with the highest edge weight. Simple, deterministic,
//! and O(N² · frames).
//!
//! This is intentionally a small pass — the roadmap has larger community
//! detection (Louvain) and spectral variants; adding them here is a follow-up
//! that just swaps the ordering function.

use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-layout-reorg: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut routes_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut method: String = "cluster".to_string();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--routes" => {
                i += 1;
                routes_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--method" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    method = v.clone();
                }
            }
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    let routes = routes_path.ok_or_else(|| "missing --routes <routes.json>".to_string())?;
    let out = out_path.ok_or_else(|| "missing --out <out_dir>".to_string())?;
    let trace = read_routes(&routes)?;
    let ordered = order_experts(&trace, &method)?;
    write_schedule(&out, &ordered)?;
    eprintln!(
        "wrote per-layer schedule for {} layers to {}/schedule.json",
        ordered.len(),
        out.display()
    );
    Ok(())
}

fn usage() {
    eprintln!(
        "usage: peregrine-layout-reorg --routes <routes.json> --out <out_dir> [--method cluster|greedy|louvain|spectral]"
    );
}

fn read_routes(path: &Path) -> Result<Vec<Vec<Vec<i32>>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("parse routes: {e}"))?;
    let arr = v.as_array().ok_or_else(|| "routes JSON is not an array".to_string())?;
    let mut out: Vec<Vec<Vec<i32>>> = Vec::with_capacity(arr.len());
    for forward in arr {
        let layers = forward.as_array().ok_or_else(|| "forward is not an array".to_string())?;
        let mut ls: Vec<Vec<i32>> = Vec::with_capacity(layers.len());
        for layer in layers {
            let ids = layer.as_array().ok_or_else(|| "layer is not an array".to_string())?;
            let mut es: Vec<i32> = Vec::with_capacity(ids.len());
            for id in ids {
                let n = id.as_i64().ok_or_else(|| "expert id is not an integer".to_string())?;
                es.push(n as i32);
            }
            ls.push(es);
        }
        out.push(ls);
    }
    Ok(out)
}

fn order_experts(trace: &[Vec<Vec<i32>>], method: &str) -> Result<Vec<Vec<i32>>, String> {
    // Determine layer count and expert-id upper bound.
    let n_layers = trace.iter().map(|f| f.len()).max().unwrap_or(0);
    let mut layer_order: Vec<Vec<i32>> = vec![Vec::new(); n_layers];
    for (l, slot) in layer_order.iter_mut().enumerate() {
        let matrix = build_cooccurrence(trace, l);
        *slot = match method {
            "cluster" | "greedy" => greedy_nearest_neighbor(&matrix),
            "louvain" | "community" => louvain_communities(&matrix),
            "spectral" => spectral_order(&matrix),
            other => return Err(format!("unknown --method: {other}")),
        };
    }
    Ok(layer_order)
}

/// Build the symmetric N×N co-occurrence weight matrix for `layer` across the
/// full trace. `matrix[a][b]` = number of forwards where both `a` and `b` were
/// routed. Uses a HashMap to keep memory proportional to the observed expert
/// set, not the vocabulary.
fn build_cooccurrence(trace: &[Vec<Vec<i32>>], layer: usize) -> HashMap<i32, HashMap<i32, u32>> {
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
fn greedy_nearest_neighbor(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
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
fn louvain_communities(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
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
fn spectral_order(w: &HashMap<i32, HashMap<i32, u32>>) -> Vec<i32> {
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
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
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
    let _ = inv_sqrt_n; // kept for clarity of the deflation derivation
    // Order by Fiedler value; ties (including whole disconnected components with
    // near-equal values) break by ascending expert id for determinism.
    let mut order: Vec<(i32, f64)> = nodes.iter().map(|&id| (id, v[index[&id]])).collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    order.into_iter().map(|(id, _)| id).collect()
}

fn write_schedule(dir: &Path, ordered: &[Vec<i32>]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let doc = serde_json::json!({
        "version": 1,
        "n_layers": ordered.len(),
        "order": ordered,
    });
    let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| format!("serialize: {e}"))?;
    let out_path = dir.join("schedule.json");
    let mut f = std::fs::File::create(&out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
    f.write_all(&bytes).map_err(|e| format!("write schedule.json: {e}"))?;
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

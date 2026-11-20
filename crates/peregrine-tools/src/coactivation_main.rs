//! `peregrine-coactivation-experiment`: measure the §17c idea #5 claim —
//! ordering experts on disk by *measured* co-activation makes each routed set
//! occupy fewer contiguous runs on disk, so batched reads coalesce.
//!
//! The library's statistics (`consecutive_overlap`, `union_growth_*`) describe
//! the *trace* and are layout-independent; the run-break count is the one metric
//! that depends on the disk order, and it is the quantity idea #5 claims to cut.
//! So this binary computes it directly, over the layouts that
//! [`peregrine_tools::order_experts`] actually produces.
//!
//! With no argument it runs on a SYNTHETIC trace whose experts fall into
//! interleaved co-activation clusters — built so the identity (id-order) layout
//! is deliberately bad and a co-activation ordering should recover contiguity.
//! Given a path it reads a real routing trace (`dump-routes` output) via
//! [`peregrine_tools::read_routes`], so the same harness validates the claim on
//! production data the moment a trace exists. CPU-only, no model required.
//!
//! A "run" is a maximal block of disk indices that are consecutive integers; a
//! routed set that lands on indices {4,5,6, 19,20} is two runs — two seeks, two
//! contiguous reads. Fewer mean-runs-per-position is the win.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use peregrine_core::Error;
use peregrine_tools::{order_experts, read_routes};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("peregrine-coactivation-experiment: {e}");
        std::process::exit(1);
    }
}

/// A deterministic pseudo-random stream (SplitMix64), so every run of this
/// experiment is bit-identical and no `rand` dependency is pulled in.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }
}

/// Build a synthetic single-layer trace. `clusters` co-activation groups, each
/// `per_cluster` experts, expert `e` assigned to cluster `e % clusters` so the
/// groups are maximally *interleaved* in id-space — the identity layout then
/// breaks every multi-expert read into one run per expert. Each position routes
/// `k` experts, mostly from one active cluster, with one cross-cluster "noise"
/// pick a fraction of the time. Returns (trace, n_experts).
fn synthetic_trace(
    clusters: usize,
    per_cluster: usize,
    positions: usize,
    k: usize,
) -> (Vec<Vec<Vec<i32>>>, usize) {
    let n = clusters * per_cluster;
    let mut rng = Rng(0x5EED_1234_5678_9ABC);
    let mut trace = Vec::with_capacity(positions);
    for _ in 0..positions {
        let active = rng.below(clusters);
        // member ids of the active cluster: active, active+clusters, ...
        let members: Vec<i32> =
            (0..per_cluster).map(|j| (active + j * clusters) as i32).collect();
        let mut picked: Vec<i32> = Vec::with_capacity(k);
        while picked.len() < k.min(per_cluster) {
            let cand = members[rng.below(members.len())];
            if !picked.contains(&cand) {
                picked.push(cand);
            }
        }
        // ~1-in-8 positions add a cross-cluster expert (realistic noise).
        if rng.below(8) == 0 {
            let noise = rng.below(n) as i32;
            if !picked.contains(&noise) {
                picked.push(noise);
            }
        }
        trace.push(vec![picked]); // one layer
    }
    (trace, n)
}

/// Largest routed expert id + 1 across the trace (the pool size for a real one).
fn expert_pool_size(trace: &[Vec<Vec<i32>>]) -> usize {
    let mut max_id = -1i32;
    for frame in trace {
        for layer in frame {
            for &e in layer {
                if e > max_id {
                    max_id = e;
                }
            }
        }
    }
    (max_id + 1).max(0) as usize
}

/// Turn a disk order (expert ids, best-first) into expert_id -> disk_index.
/// Any id in `0..n` missing from `order` (never co-fired, so never placed) is
/// appended; it is never read, so its slot does not affect the metric.
fn disk_index(order: &[i32], n: usize) -> HashMap<i32, usize> {
    let mut idx: HashMap<i32, usize> = HashMap::with_capacity(n);
    let mut next = 0usize;
    // Ordered ids first (best-first, padding dropped), then any id in 0..n the
    // order never placed — appended so the layout stays a full permutation.
    for e in order.iter().copied().filter(|&e| e >= 0).chain(0..n as i32) {
        idx.entry(e).or_insert_with(|| {
            let slot = next;
            next += 1;
            slot
        });
    }
    idx
}

/// `true` iff `layout` places every expert `0..n` exactly once (bijective) —
/// the correctness half of idea #5's certificate.
fn is_bijection(layout: &HashMap<i32, usize>, n: usize) -> bool {
    if layout.len() != n {
        return false;
    }
    let mut slots = vec![false; n];
    for (&e, &slot) in layout {
        if e < 0 || e as usize >= n || slot >= n || slots[slot] {
            return false;
        }
        slots[slot] = true;
    }
    slots.into_iter().all(|s| s)
}

/// Mean number of maximal contiguous disk runs per routed set, over `layer`.
/// This is the quantity idea #5 claims a co-activation layout reduces.
fn mean_run_breaks(trace: &[Vec<Vec<i32>>], layer: usize, layout: &HashMap<i32, usize>) -> f64 {
    let (mut total_runs, mut counted) = (0u64, 0u64);
    for frame in trace {
        let set = match frame.get(layer) {
            Some(s) => s,
            None => continue,
        };
        let mut slots: Vec<usize> = set
            .iter()
            .filter(|&&e| e >= 0)
            .filter_map(|e| layout.get(e).copied())
            .collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            continue;
        }
        let mut runs = 1u64;
        for w in slots.windows(2) {
            if w[1] != w[0] + 1 {
                runs += 1;
            }
        }
        total_runs += runs;
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total_runs as f64 / counted as f64 }
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    let trace_path = args.iter().skip(1).find(|a| !a.starts_with("--")).map(PathBuf::from);

    let (trace, n, source) = match &trace_path {
        Some(p) => {
            let t = read_routes(p)?;
            let n = expert_pool_size(&t);
            (t, n, format!("real trace {}", p.display()))
        }
        None => {
            let (t, n) = synthetic_trace(8, 16, 4000, 6);
            (t, n, "synthetic: 8 interleaved clusters x 16 experts, 4000 positions, k=6".to_string())
        }
    };
    if n == 0 {
        return Err(Error::Format("trace routes no experts; nothing to lay out".into()));
    }
    let layer = 0usize;

    println!("peregrine idea #5 — co-activation disk layout, run-break experiment");
    println!("source: {source}");
    println!("experts: {n}, positions: {}\n", trace.len());

    // Baselines: identity (id order) and a deterministic shuffle (null).
    let identity: Vec<i32> = (0..n as i32).collect();
    let mut shuffled = identity.clone();
    let mut rng = Rng(0xA5A5_F00D_1234_5678);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, rng.below(i + 1));
    }

    let mut rows: Vec<(String, f64, bool)> = Vec::new();
    let id_layout = disk_index(&identity, n);
    let identity_runs = mean_run_breaks(&trace, layer, &id_layout);
    rows.push(("identity (id order)".into(), identity_runs, is_bijection(&id_layout, n)));
    let sh_layout = disk_index(&shuffled, n);
    rows.push(("shuffled (null)".into(), mean_run_breaks(&trace, layer, &sh_layout), is_bijection(&sh_layout, n)));

    let mut best: Option<(String, f64)> = None;
    for method in ["greedy", "louvain", "spectral", "hilbert"] {
        let order = order_experts(&trace, method)?;
        let layer_order = order.get(layer).cloned().unwrap_or_default();
        let layout = disk_index(&layer_order, n);
        let runs = mean_run_breaks(&trace, layer, &layout);
        let bij = is_bijection(&layout, n);
        rows.push((format!("co-activation: {method}"), runs, bij));
        if bij && best.as_ref().is_none_or(|(_, b)| runs < *b) {
            best = Some((method.to_string(), runs));
        }
    }

    println!("{:<28} {:>14} {:>10}", "layout", "mean runs/read", "bijective");
    println!("{}", "-".repeat(54));
    for (name, runs, bij) in &rows {
        println!("{:<28} {:>14.3} {:>10}", name, runs, if *bij { "yes" } else { "NO" });
    }

    // Verdict.
    if rows.iter().any(|(_, _, bij)| !bij) {
        return Err(Error::Format("a layout was not bijective — idea #5's correctness half failed".into()));
    }
    let (best_method, best_runs) = best.ok_or_else(|| Error::Format("no co-activation layout produced".into()))?;
    let reduction = if identity_runs > 0.0 {
        100.0 * (identity_runs - best_runs) / identity_runs
    } else {
        0.0
    };
    println!(
        "\nbest co-activation method: {best_method} — {best_runs:.3} runs/read vs {identity_runs:.3} identity ({reduction:.1}% fewer run-breaks)"
    );
    if best_runs < identity_runs {
        println!("CLAIM HELD: co-activation ordering cut contiguous-run breaks on this trace.");
        Ok(())
    } else {
        Err(Error::Format(format!(
            "CLAIM NOT HELD on this trace: best {best_runs:.3} did not beat identity {identity_runs:.3}"
        )))
    }
}

#[cfg(test)]
// Tests may assert with unwrap/panic; production code in this file stays denied.
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn layout_of(order: &[i32], n: usize) -> HashMap<i32, usize> {
        disk_index(order, n)
    }

    #[test]
    fn run_breaks_counts_maximal_contiguous_blocks() {
        let layout = layout_of(&[0, 1, 2, 3, 4, 5, 6], 7);
        // {0,1,2} is one run; {0,2,4} is three; {0,1,5,6} is two.
        assert_eq!(mean_run_breaks(&[vec![vec![0, 1, 2]]], 0, &layout), 1.0);
        assert_eq!(mean_run_breaks(&[vec![vec![0, 2, 4]]], 0, &layout), 3.0);
        assert_eq!(mean_run_breaks(&[vec![vec![0, 1, 5, 6]]], 0, &layout), 2.0);
    }

    #[test]
    fn run_breaks_ignores_padding_and_dedups() {
        let layout = layout_of(&[0, 1, 2], 3);
        // -1 padding dropped, duplicate 1 collapsed -> {0,1} is one run.
        assert_eq!(mean_run_breaks(&[vec![vec![1, -1, 0, 1]]], 0, &layout), 1.0);
    }

    #[test]
    fn bijection_accepts_permutation_rejects_collision() {
        assert!(is_bijection(&layout_of(&[2, 0, 1], 3), 3));
        let mut bad: HashMap<i32, usize> = HashMap::new();
        bad.insert(0, 0);
        bad.insert(1, 0); // two experts, same slot
        bad.insert(2, 2);
        assert!(!is_bijection(&bad, 3));
    }

    #[test]
    fn missing_ids_are_appended_to_complete_the_permutation() {
        // order names only expert 2; 0 and 1 must still get slots.
        let layout = layout_of(&[2], 3);
        assert!(is_bijection(&layout, 3));
        assert_eq!(layout.get(&2), Some(&0));
    }

    #[test]
    fn co_activation_beats_identity_on_interleaved_clusters() {
        // Two clusters interleaved in id-space: evens co-fire, odds co-fire.
        let (trace, n) = synthetic_trace(2, 8, 1500, 4);
        let identity: Vec<i32> = (0..n as i32).collect();
        let id_runs = mean_run_breaks(&trace, 0, &disk_index(&identity, n));
        let order = order_experts(&trace, "greedy").unwrap();
        let co_runs = mean_run_breaks(&trace, 0, &disk_index(&order[0], n));
        assert!(
            co_runs < id_runs,
            "co-activation {co_runs:.3} should beat identity {id_runs:.3}"
        );
    }

    #[test]
    fn synthetic_trace_is_deterministic() {
        let a = synthetic_trace(4, 8, 500, 3);
        let b = synthetic_trace(4, 8, 500, 3);
        assert_eq!(a, b);
    }
}

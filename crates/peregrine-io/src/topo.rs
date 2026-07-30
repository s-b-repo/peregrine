//! Runtime hardware topology discovery.
//!
//! Best-effort, read-only. Every probe has a portable fallback (`None`, `1`,
//! `logical_cpus`); nothing panics or errors. Consumers use the returned facts to
//! size worker pools, pin threads to NUMA nodes, and pick per-device residency
//! budgets — but every consumer must still tolerate the fallback shape (single
//! NUMA node, single GPU, unknown PCIe link).

use std::path::Path;

/// Number of logical CPUs the OS reports. Falls back to `1` when unknown.
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// One NUMA node's CPUs. `id` matches the kernel's node number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaNode {
    pub id: u32,
    /// Logical CPU ids belonging to this node, ascending. Never empty on a
    /// well-formed system.
    pub cpus: Vec<u32>,
}

/// Enumerate NUMA nodes from `/sys/devices/system/node/nodeN/cpulist`.
/// Returns `[NumaNode{0, all-cpus}]` on non-Linux or when the sysfs tree is
/// missing (single-node fallback). Consumers can safely assume `nodes.len() >= 1`.
pub fn numa_nodes() -> Vec<NumaNode> {
    #[cfg(target_os = "linux")]
    {
        let root = Path::new("/sys/devices/system/node");
        if let Ok(entries) = std::fs::read_dir(root) {
            let mut nodes: Vec<NumaNode> = Vec::new();
            for e in entries.flatten() {
                let name = e.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(rest) = name.strip_prefix("node") else { continue };
                let Ok(id) = rest.parse::<u32>() else { continue };
                let list_path = e.path().join("cpulist");
                let Ok(text) = std::fs::read_to_string(&list_path) else { continue };
                let cpus = parse_cpu_list(text.trim());
                if !cpus.is_empty() {
                    nodes.push(NumaNode { id, cpus });
                }
            }
            if !nodes.is_empty() {
                nodes.sort_by_key(|n| n.id);
                return nodes;
            }
        }
    }
    // Fallback: one node with every logical CPU.
    let all: Vec<u32> = (0..logical_cpus() as u32).collect();
    vec![NumaNode { id: 0, cpus: all }]
}

/// Parse a Linux `cpulist` string ("0-3,7,9-11") into ascending CPU ids. Bad
/// tokens are silently skipped — matches the kernel's own leniency.
fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((a, b)) = token.split_once('-') {
            let (a, b) = (a.trim().parse::<u32>(), b.trim().parse::<u32>());
            if let (Ok(a), Ok(b)) = (a, b) {
                for c in a.min(b)..=a.max(b) {
                    out.push(c);
                }
            }
        } else if let Ok(c) = token.parse::<u32>() {
            out.push(c);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// PCIe link information for one device.
#[derive(Clone, Debug, PartialEq)]
pub struct PcieLink {
    /// `current_link_speed`, kernel string (e.g. "16 GT/s PCIe" or "8 GT/s"),
    /// verbatim. Rarely worth parsing further; the string is enough to log.
    pub speed: String,
    /// `current_link_width`, lane count (e.g. `16`).
    pub width: u32,
}

/// Read the PCIe link info for `bdf` (`0000:01:00.0`). `None` on non-Linux,
/// missing sysfs entries, or unparseable values.
pub fn pcie_link_by_bdf(bdf: &str) -> Option<PcieLink> {
    #[cfg(target_os = "linux")]
    {
        let base = Path::new("/sys/bus/pci/devices").join(bdf);
        let speed = std::fs::read_to_string(base.join("current_link_speed")).ok()?;
        let width = std::fs::read_to_string(base.join("current_link_width")).ok()?;
        let width: u32 = width.trim().parse().ok()?;
        Some(PcieLink { speed: speed.trim().to_string(), width })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = bdf;
        None
    }
}

/// The single-shot topology snapshot every subsystem consults at startup.
///
/// Cached lazily — the first caller pays the sysfs read; subsequent callers
/// see the same numbers. Numbers do not update mid-process (would require an
/// invalidation story); a fresh process reads fresh values.
pub fn snapshot() -> &'static Topology {
    use std::sync::OnceLock;
    static SNAP: OnceLock<Topology> = OnceLock::new();
    SNAP.get_or_init(|| Topology { logical_cpus: logical_cpus(), numa: numa_nodes() })
}

/// A single snapshot of what we discovered about this box. GPU count is
/// deliberately not included — callers that care depend on `peregrine-cuda`,
/// which owns that probe.
pub struct Topology {
    pub logical_cpus: usize,
    pub numa: Vec<NumaNode>,
}

impl Topology {
    /// Whether the box is multi-NUMA (drives NUMA pinning defaults).
    pub fn multi_numa(&self) -> bool {
        self.numa.len() > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_list_parses_ranges_and_singletons() {
        assert_eq!(parse_cpu_list("0-3,7,9-11"), vec![0, 1, 2, 3, 7, 9, 10, 11]);
        assert_eq!(parse_cpu_list(""), Vec::<u32>::new());
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list("garbage,4,x-y,7-8"), vec![4, 7, 8]);
    }

    #[test]
    fn logical_cpus_is_at_least_one() {
        assert!(logical_cpus() >= 1);
    }

    #[test]
    fn numa_nodes_never_empty() {
        let n = numa_nodes();
        assert!(!n.is_empty(), "always at least the single-node fallback");
        assert!(n.iter().all(|node| !node.cpus.is_empty()), "no empty node");
    }

    #[test]
    fn snapshot_is_stable() {
        let a = snapshot();
        let b = snapshot();
        assert_eq!(a.logical_cpus, b.logical_cpus);
        assert_eq!(a.numa.len(), b.numa.len());
    }
}

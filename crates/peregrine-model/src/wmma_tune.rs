//! Online WMMA tile autotuner.
//!
//! Records observed `kernel_ms` per `(D, I, count, max_rows)` shape and picks
//! the winning tile configuration. Persists as `<model_dir>/kernel_tuning.json`
//! so a repeat run skips the exploration phase.
//!
//! This is the Rust-side substrate; the actual CUDA dispatch still gates on
//! the `COLI_CUDA_TC_*` env knobs. When [`WmmaTuner::best_for`] returns a
//! recommendation, the dispatcher can override those knobs on a per-shape
//! basis. Bit-identical to the fixed-tile path (WMMA fragment sizes only
//! affect performance).

use std::collections::HashMap;

/// Kernel shape triple. Coarse enough to keep the table small (per-layer inter-
/// sizes rarely differ) yet specific enough to discriminate the meaningful
/// tuning axes.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct KernelShape {
    pub d: u32,        // hidden dim
    pub i: u32,        // intermediate dim
    pub count: u16,    // number of experts routed
    pub max_rows: u16, // max rows in any expert
}

/// A tile configuration. Two flavors so far — W4A16 (gate/up) and INT4 TC
/// (down). Expand as the CUDA backend gains more templated variants.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum TileConfig {
    /// `(M, N, K)` for the W4A16 path. Only the pairs the C++ backend accepts
    /// are valid (16×16×16 today; 32×32×16 after templating).
    W4A16 { m: u16, n: u16, k: u16 },
    /// `(M, N, K)` for the INT4 TC path (8×8×32 default; 16×16×32 after
    /// templating).
    Int4Tc { m: u16, n: u16, k: u16 },
}

impl TileConfig {
    pub fn default_w4a16() -> TileConfig {
        TileConfig::W4A16 { m: 16, n: 16, k: 16 }
    }
    pub fn default_int4tc() -> TileConfig {
        TileConfig::Int4Tc { m: 8, n: 8, k: 32 }
    }
}

/// Table of (shape, tile) → observed EWMA microseconds. Cheap to consult; the
/// dispatcher looks up its shape before every launch.
pub struct WmmaTuner {
    /// One row per shape: the best tile so far + a recency counter that lets
    /// the tuner occasionally re-check alternatives (defense against a stale
    /// winner after workload drift).
    best: HashMap<KernelShape, TileConfig>,
    /// Explored (shape, tile) → EWMA microseconds. Higher is worse.
    ewma_us: HashMap<(KernelShape, TileConfig), f32>,
    alpha: f32,
}

impl WmmaTuner {
    pub fn new() -> WmmaTuner {
        WmmaTuner { best: HashMap::new(), ewma_us: HashMap::new(), alpha: 0.3 }
    }

    /// Record one measurement. Updates the EWMA and potentially the best tile.
    pub fn observe(&mut self, shape: KernelShape, tile: TileConfig, us: f32) {
        let key = (shape, tile);
        let prev = self.ewma_us.get(&key).copied().unwrap_or(us);
        let next = (1.0 - self.alpha) * prev + self.alpha * us;
        self.ewma_us.insert(key, next);
        // Update best when this tile has the lowest EWMA for its shape.
        let mut best_us = f32::INFINITY;
        let mut best_tile: Option<TileConfig> = None;
        for ((s, t), &v) in &self.ewma_us {
            if *s == shape && v < best_us {
                best_us = v;
                best_tile = Some(*t);
            }
        }
        if let Some(t) = best_tile {
            self.best.insert(shape, t);
        }
    }

    /// The current best-known tile for a shape (or `None` if never observed).
    pub fn best_for(&self, shape: KernelShape) -> Option<TileConfig> {
        self.best.get(&shape).copied()
    }

    /// Serialize the table (deterministic ordering).
    pub fn to_json(&self) -> serde_json::Value {
        let mut rows: Vec<serde_json::Value> = self
            .best
            .iter()
            .map(|(s, t)| serde_json::json!([s.d, s.i, s.count, s.max_rows, encode_tile(*t)]))
            .collect();
        rows.sort_by_key(|a| a.to_string());
        serde_json::json!({ "version": 1, "rows": rows })
    }

    /// Parse a table from [`Self::to_json`] output. Malformed rows are silently
    /// skipped — this is a hint, not a source of truth.
    pub fn from_json(v: &serde_json::Value) -> WmmaTuner {
        let mut t = WmmaTuner::new();
        if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
            for row in rows {
                let arr = match row.as_array() {
                    Some(a) if a.len() == 5 => a,
                    _ => continue,
                };
                let (d, i, count, max_rows) = match (
                    arr[0].as_u64(),
                    arr[1].as_u64(),
                    arr[2].as_u64(),
                    arr[3].as_u64(),
                ) {
                    (Some(a), Some(b), Some(c), Some(d)) => (a as u32, b as u32, c as u16, d as u16),
                    _ => continue,
                };
                let shape = KernelShape { d, i, count, max_rows };
                if let Some(tile) = decode_tile(&arr[4]) {
                    t.best.insert(shape, tile);
                }
            }
        }
        t
    }
}

impl Default for WmmaTuner {
    fn default() -> WmmaTuner {
        WmmaTuner::new()
    }
}

fn encode_tile(t: TileConfig) -> serde_json::Value {
    match t {
        TileConfig::W4A16 { m, n, k } => serde_json::json!(["w4a16", m, n, k]),
        TileConfig::Int4Tc { m, n, k } => serde_json::json!(["int4tc", m, n, k]),
    }
}

fn decode_tile(v: &serde_json::Value) -> Option<TileConfig> {
    let a = v.as_array()?;
    if a.len() != 4 {
        return None;
    }
    let kind = a[0].as_str()?;
    let m = a[1].as_u64()? as u16;
    let n = a[2].as_u64()? as u16;
    let k = a[3].as_u64()? as u16;
    match kind {
        "w4a16" => Some(TileConfig::W4A16 { m, n, k }),
        "int4tc" => Some(TileConfig::Int4Tc { m, n, k }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_picks_lowest_ewma_tile() {
        let mut t = WmmaTuner::new();
        let shape = KernelShape { d: 128, i: 256, count: 8, max_rows: 64 };
        let a = TileConfig::W4A16 { m: 16, n: 16, k: 16 };
        let b = TileConfig::W4A16 { m: 32, n: 32, k: 16 };
        // A is faster on average.
        for _ in 0..5 {
            t.observe(shape, a, 100.0);
            t.observe(shape, b, 200.0);
        }
        assert_eq!(t.best_for(shape), Some(a));
    }

    #[test]
    fn json_round_trip() {
        let mut t = WmmaTuner::new();
        let shape = KernelShape { d: 128, i: 256, count: 8, max_rows: 64 };
        t.observe(shape, TileConfig::W4A16 { m: 16, n: 16, k: 16 }, 100.0);
        let j = t.to_json();
        let t2 = WmmaTuner::from_json(&j);
        assert_eq!(t2.best_for(shape), Some(TileConfig::W4A16 { m: 16, n: 16, k: 16 }));
    }
}

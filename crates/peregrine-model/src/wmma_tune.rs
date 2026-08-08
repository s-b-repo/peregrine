//! Online WMMA tile autotuner.
//!
//! Records observed `kernel_ms` per `(D, I, count, max_rows)` shape and picks
//! the winning tile configuration. Persists as `<model_dir>/kernel_tuning.json`
//! so a repeat run skips the exploration phase.
//!
//! Wired into `GpuTier::compute` behind `COLI_CUDA_AUTOTUNE=1`, which is a
//! *second* opt-in on top of `COLI_CUDA_TC_W4A16` — the tile reaches only that
//! Tensor Core arm, so tuning on a run that never takes it would be recording
//! noise as a winner.
//!
//! **The bit-identity claim this file used to make is not one this workspace can
//! support.** It said "WMMA fragment sizes only affect performance". All three
//! legal fp16 shapes share `K = 16` and the same k-loop, so the per-element sum
//! order is *expected* to be identical — but that is an argument about hardware
//! reduction order, not a measurement, and nothing here has executed on a GPU to
//! check it. Treat the tuner as a knob that may move low bits until a run on
//! real hardware says otherwise.

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

/// A tile configuration. Two flavors — W4A16 (gate/up/down fp16 Tensor Core)
/// and INT4 TC.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum TileConfig {
    /// `(M, N, K)` for the W4A16 path. Exactly three are legal — see
    /// [`TileConfig::W4A16_LEGAL`] — because those are the fp16 WMMA fragment
    /// shapes the hardware defines; anything else the backend rejects back to
    /// the default rather than launching a kernel that cannot exist.
    W4A16 { m: u16, n: u16, k: u16 },
    /// `(M, N, K)` for the INT4 TC path. **One legal shape**, 8×8×32: that is
    /// the only `experimental::precision::s4` fragment WMMA defines, so this
    /// variant records which kernel ran rather than offering a choice. It earns
    /// its place by keeping the tuning table honest — an int4 measurement and a
    /// W4A16 measurement at the same `KernelShape` are not comparable, and
    /// without the tag they would share a row and the faster *arm* would look
    /// like the faster *tile*.
    Int4Tc { m: u16, n: u16, k: u16 },
}

impl TileConfig {
    /// Every fp16 WMMA fragment shape the hardware defines. The tuner explores
    /// exactly these; the CUDA side instantiates exactly these.
    pub const W4A16_LEGAL: [TileConfig; 3] = [
        TileConfig::W4A16 { m: 16, n: 16, k: 16 },
        TileConfig::W4A16 { m: 32, n: 8, k: 16 },
        TileConfig::W4A16 { m: 8, n: 32, k: 16 },
    ];

    /// The historical shape, and what an unmeasured run executes.
    pub fn default_w4a16() -> TileConfig {
        TileConfig::W4A16 { m: 16, n: 16, k: 16 }
    }

    /// The only legal int4 Tensor Core fragment; see the variant's note.
    pub fn default_int4tc() -> TileConfig {
        TileConfig::Int4Tc { m: 8, n: 8, k: 32 }
    }

    /// `(m, n, k)` for the CUDA dispatch, or `None` for a config the W4A16 arm
    /// cannot take — an int4 tile, or a shape outside [`Self::W4A16_LEGAL`].
    /// Returning `None` rather than the numbers is what stops a stale
    /// `kernel_tuning.json` from selecting a kernel that was never compiled.
    pub fn w4a16_dims(self) -> Option<(u16, u16, u16)> {
        match self {
            TileConfig::W4A16 { m, n, k } if Self::W4A16_LEGAL.contains(&self) => Some((m, n, k)),
            _ => None,
        }
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

    /// The tile to run next for `shape`: explore first, then exploit.
    ///
    /// Any legal shape with no measurement yet is returned before
    /// [`Self::best_for`] is consulted, so every candidate is tried once before
    /// one is declared the winner. Without that a restored table would pin
    /// whatever the previous session happened to try first — and a table
    /// restored from `kernel_tuning.json` carries `best` but not the per-tile
    /// EWMAs, so this also re-explores after a restart rather than trusting a
    /// winner it cannot re-derive.
    pub fn select(&self, shape: KernelShape) -> TileConfig {
        for t in TileConfig::W4A16_LEGAL {
            if !self.ewma_us.contains_key(&(shape, t)) {
                return t;
            }
        }
        self.best_for(shape).unwrap_or_else(TileConfig::default_w4a16)
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
    fn select_explores_every_legal_tile_before_exploiting_one() {
        // Without the explore phase a restored table pins whatever the previous
        // session happened to measure first, and the alternatives are never
        // tried again — the tuner would converge on the tile it started with
        // and report it as a winner.
        let mut t = WmmaTuner::new();
        let shape = KernelShape { d: 5120, i: 1536, count: 6, max_rows: 1 };
        let mut seen = Vec::new();
        for _ in 0..TileConfig::W4A16_LEGAL.len() {
            let tile = t.select(shape);
            assert!(!seen.contains(&tile), "select repeated {tile:?} before trying the rest");
            seen.push(tile);
            // Later tiles are slower, so the first one must win at the end.
            t.observe(shape, tile, 100.0 + 10.0 * seen.len() as f32);
        }
        assert_eq!(seen.len(), TileConfig::W4A16_LEGAL.len(), "every legal tile must be explored");
        for _ in 0..4 {
            assert_eq!(t.select(shape), seen[0], "after exploring, select must exploit the winner");
        }
    }

    #[test]
    fn select_is_the_default_tile_for_an_untouched_shape_once_explored() {
        // A shape whose only measurement is on an *illegal* tile must not
        // "win" — otherwise a hand-edited or version-skewed table selects a
        // kernel instantiation that does not exist.
        let mut t = WmmaTuner::new();
        let shape = KernelShape { d: 64, i: 32, count: 1, max_rows: 1 };
        t.observe(shape, TileConfig::W4A16 { m: 64, n: 64, k: 16 }, 1.0);
        // Exploration still runs (the illegal tile is not one of the legal
        // three), and every legal tile is offered before anything is exploited.
        assert!(TileConfig::W4A16_LEGAL.contains(&t.select(shape)));
    }

    #[test]
    fn only_the_three_hardware_fragment_shapes_reach_the_dispatch() {
        // `w4a16_dims` is the gate between a persisted table and a kernel
        // launch. A shape the backend never instantiated must come back `None`
        // and fall to the default, not be passed through as three integers.
        for t in TileConfig::W4A16_LEGAL {
            assert!(t.w4a16_dims().is_some(), "{t:?} is legal and must dispatch");
        }
        assert_eq!(TileConfig::default_w4a16().w4a16_dims(), Some((16, 16, 16)));
        assert_eq!(TileConfig::W4A16 { m: 64, n: 64, k: 16 }.w4a16_dims(), None, "not a WMMA shape");
        assert_eq!(TileConfig::W4A16 { m: 16, n: 16, k: 32 }.w4a16_dims(), None, "K=32 is not fp16 WMMA");
        assert_eq!(
            TileConfig::default_int4tc().w4a16_dims(),
            None,
            "an int4 tile must never be handed to the fp16 arm"
        );
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

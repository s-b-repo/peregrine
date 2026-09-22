//! Per-block execution mode: `sync` (single-thread) vs `mt` (scoped OS
//! threads) vs `par` (persistent [`peregrine_par`] pool).
//!
//! Every transformer block runs the same math in all three modes — residual
//! adds stay in fixed order, `par_*` row partitions are disjoint — so output
//! is bit-identical and the mode is a pure throughput knob for token
//! generation speed (`COLI_BLOCK_EXEC=sync|mt|par`, default `par`).
//!
//! * `sync` — fully serial: every `par_*` gate is forced shut and elementwise
//!   work runs inline. Also dictates the global pool width to 1 via
//!   [`BlockExecMode::apply_to_pool`], which must run before the pool's first
//!   use (the pool is built once, lazily).
//! * `mt` — multi-threaded with scoped OS threads for elementwise work
//!   (residual adds) and the shared pool for heavy kernels. Same bytes as
//!   `par`, different thread shape: no pool dispatch on the hottest tiny ops.
//! * `par` — the historical path: the persistent pool everywhere.

/// Which thread shape one transformer block executes with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockExecMode {
    /// Fully serial single thread.
    Sync,
    /// Scoped OS threads per block for elementwise work, pool for kernels.
    Mt,
    /// Persistent pool everywhere (historical default).
    Par,
}

impl BlockExecMode {
    /// Parse an env value. Unknown/empty ⇒ `Par` (historical behavior).
    pub fn resolve_with(val: Option<&str>) -> BlockExecMode {
        match val.map(str::trim) {
            Some("sync") => BlockExecMode::Sync,
            Some("mt") => BlockExecMode::Mt,
            Some("par") => BlockExecMode::Par,
            _ => BlockExecMode::Par,
        }
    }

    /// Resolve `COLI_BLOCK_EXEC`, cached once. A later export cannot change a
    /// loaded model (same latch discipline as the mirror roots).
    pub fn resolve() -> BlockExecMode {
        use std::sync::OnceLock;
        static MODE: OnceLock<BlockExecMode> = OnceLock::new();
        *MODE.get_or_init(|| Self::resolve_with(std::env::var("COLI_BLOCK_EXEC").ok().as_deref()))
    }

    /// Short name for boot logs and the exec-mode report.
    pub fn name(self) -> &'static str {
        match self {
            BlockExecMode::Sync => "sync",
            BlockExecMode::Mt => "mt",
            BlockExecMode::Par => "par",
        }
    }

    /// Whether any parallel dispatch is allowed.
    pub fn allows_parallel(self) -> bool {
        !matches!(self, BlockExecMode::Sync)
    }

    /// Force a `par_*` minimum-parallel gate shut under `sync`, else pass it
    /// through. Lets every existing `PAR_*_MIN` call site honor the mode with
    /// a one-line wrap and no signature change.
    pub fn par_gate(self, gate: usize) -> usize {
        match self {
            BlockExecMode::Sync => usize::MAX,
            BlockExecMode::Mt | BlockExecMode::Par => gate,
        }
    }

    /// Dictate the global pool width for `sync` (must run before the pool's
    /// first `par_*` call). Returns whether the override took; `false` means a
    /// pool already exists and the caller must say so rather than claim serial.
    pub fn apply_to_pool(self) -> bool {
        match self {
            BlockExecMode::Sync => peregrine_par::set_global_threads(1),
            BlockExecMode::Mt | BlockExecMode::Par => true,
        }
    }

    /// Elementwise `dst += src` honoring the mode. Serial under `sync` or
    /// below the threading floor; otherwise each mode uses its own thread
    /// shape over disjoint chunks in fixed order — bit-identical to the
    /// serial loop.
    ///
    /// * `Mt` — scoped OS threads (no pool dispatch on this tiny op).
    /// * `Par` — the persistent pool (`par_rows_mut` with stride 1, so one
    ///   index per task; the pool's nesting guard keeps this deadlock-safe).
    ///
    /// The floor (64K floats) keeps small decodes off the spawn path: one
    /// token at `d=6144` is 6K floats per residual, serial-cheap.
    pub fn residual_add(self, dst: &mut [f32], src: &[f32]) {
        let n = dst.len().min(src.len());
        if n == 0 {
            return;
        }
        if n < 65_536 {
            for i in 0..n {
                dst[i] += src[i];
            }
            return;
        }
        match self {
            BlockExecMode::Sync => {
                for i in 0..n {
                    dst[i] += src[i];
                }
            }
            BlockExecMode::Mt => {
                let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(16);
                let chunk = n.div_ceil(threads);
                if chunk == 0 || threads <= 1 {
                    for i in 0..n {
                        dst[i] += src[i];
                    }
                    return;
                }
                // Scoped threads over disjoint chunks; each index written once.
                std::thread::scope(|s| {
                    for (dst_c, src_c) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
                        s.spawn(move || {
                            for (d, v) in dst_c.iter_mut().zip(src_c.iter()) {
                                *d += *v;
                            }
                        });
                    }
                });
            }
            BlockExecMode::Par => {
                peregrine_par::par_rows_mut(dst, 1, n, peregrine_par::PAR_ROWS_MIN, |i, slot| {
                    slot[0] += src[i];
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_with_parses_all_modes() {
        assert_eq!(BlockExecMode::resolve_with(Some("sync")), BlockExecMode::Sync);
        assert_eq!(BlockExecMode::resolve_with(Some("mt")), BlockExecMode::Mt);
        assert_eq!(BlockExecMode::resolve_with(Some("par")), BlockExecMode::Par);
        assert_eq!(BlockExecMode::resolve_with(None), BlockExecMode::Par);
        assert_eq!(BlockExecMode::resolve_with(Some("bogus")), BlockExecMode::Par);
        assert_eq!(BlockExecMode::resolve_with(Some(" SYNC ")), BlockExecMode::Par);
    }

    #[test]
    fn sync_shuts_every_par_gate() {
        assert_eq!(BlockExecMode::Sync.par_gate(2), usize::MAX);
        assert_eq!(BlockExecMode::Sync.par_gate(0), usize::MAX);
        assert_eq!(BlockExecMode::Mt.par_gate(4), 4);
        assert_eq!(BlockExecMode::Par.par_gate(4), 4);
        assert!(!BlockExecMode::Sync.allows_parallel());
        assert!(BlockExecMode::Mt.allows_parallel());
        assert!(BlockExecMode::Par.allows_parallel());
    }

    #[test]
    fn residual_add_matches_serial_in_every_mode() {
        let n = 70_000usize;
        let mut src = vec![0f32; n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i as f32 * 0.618034).sin();
        }
        let mut ref_dst = vec![1.25f32; n];
        for i in 0..n {
            ref_dst[i] += src[i];
        }
        for mode in [BlockExecMode::Sync, BlockExecMode::Mt, BlockExecMode::Par] {
            let mut dst = vec![1.25f32; n];
            mode.residual_add(&mut dst, &src);
            assert_eq!(dst.len(), ref_dst.len());
            for (i, (a, b)) in dst.iter().zip(&ref_dst).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "mode {:?} diverged at {i}", mode);
            }
        }
        // Small vectors stay serial-equivalent too (below the thread floor).
        let mut small = vec![0.5f32; 137];
        let ssrc = vec![0.25f32; 137];
        let mut expect = small.clone();
        for i in 0..expect.len() {
            expect[i] += ssrc[i];
        }
        BlockExecMode::Mt.residual_add(&mut small, &ssrc);
        assert!(small.iter().zip(&expect).all(|(a, b)| a.to_bits() == b.to_bits()));
        // Mismatched/empty inputs never panic.
        let mut empty: Vec<f32> = Vec::new();
        BlockExecMode::Par.residual_add(&mut empty, &[]);
    }
}

//! Pre-read expert-skip metadata: the **offline prototype**, and the
//! measurement that decides whether the read path should ever be touched.
//!
//! The idea, transplanted from Quest's per-page min/max keys to expert weights:
//! if a cheap bound says expert *e* cannot contribute more than ε to this
//! position's output, its ~18.9 MB read can be skipped before it is issued.
//! That attacks 11.3 GB/token at the root instead of compressing it.
//!
//! **This module deliberately stops short of the read path.** The plan gated
//! that change on measuring the bound's tightness first, and the measurement is
//! the deliverable, because the idea only works if the bound is *tight enough
//! often enough*. A loose bound is not a partial win — it is zero skips and a
//! per-token cost to compute them.
//!
//! # The bound
//!
//! A routed expert contributes `g_e · W_down · (silu(W_gate · x) ⊙ (W_up · x))`,
//! where `g_e` is its gate weight. Since `|silu(t)| ≤ |t|` and
//! `‖a ⊙ b‖₂ ≤ ‖a‖₂ · ‖b‖₂`,
//!
//! ```text
//!   ‖contribution‖ ≤ g_e · ‖W_down‖ · ‖W_gate‖ · ‖W_up‖ · ‖x‖²
//!                  = g_e · C_e · ‖x‖²
//! ```
//!
//! `C_e` is a property of the weights alone, so it is computable offline and
//! shipped beside the container. Frobenius norms stand in for spectral ones:
//! `‖·‖₂ ≤ ‖·‖_F`, so the bound stays valid, and computing it needs no
//! eigensolve — just a pass over the dequantized weights.
//!
//! **`‖x‖²` is common to every expert at a position**, which is what makes this
//! usable without knowing it: comparing `g_e·C_e` across the routed set ranks
//! them exactly as the true bounds do. So the measurement below never needs a
//! hidden state, only a routing trace.
//!
//! # What the measurement answers
//!
//! An upper bound is one-sided evidence. A *small* bound proves the expert
//! cannot matter; a large one proves nothing. The only question worth asking
//! is: **what fraction of routed experts have a bound small enough, relative to
//! the largest in their own routed set, that skipping them is provably safe?**
//! If that fraction is near zero the idea is dead and no read-path work is
//! justified; [`Tightness`] reports it directly.

use std::collections::BTreeMap;

use peregrine_core::config::Cfg;
use peregrine_core::pack::QtView;
use peregrine_core::qt::QtInfo;
use peregrine_core::safetensors::SafeTensors;
use peregrine_core::Error;

/// `C_e` per (layer, expert): the weight-only factor of the contribution bound.
#[derive(Debug, Default, Clone)]
pub struct Bounds {
    pub c: BTreeMap<(usize, usize), f64>,
}

impl Bounds {
    /// Serialize as the sidecar a runtime skip would read.
    pub fn to_json(&self) -> serde_json::Value {
        let rows: Vec<serde_json::Value> = self
            .c
            .iter()
            .map(|(&(l, e), &c)| serde_json::json!({ "layer": l, "expert": e, "c": c }))
            .collect();
        serde_json::json!({
            "version": 1,
            "bound": "||contribution|| <= gate * c * ||x||^2, c = ||W_down||_F * ||W_gate||_F * ||W_up||_F",
            "experts": rows,
        })
    }

    /// The inverse of [`Self::to_json`], so a sidecar the tool already wrote
    /// can be measured against a trace **without re-reading the container** —
    /// `compute_bounds` is a pass over every routed expert (~hundreds of GB),
    /// which is a steep toll for an analysis whose other input is a JSON trace.
    /// Rows with missing or non-numeric fields are refused, not skipped: a
    /// silently thinner sidecar would make every absent expert read as
    /// "unbounded" and quietly shrink the denominator of the one fraction this
    /// tool exists to report.
    pub fn from_json(v: &serde_json::Value) -> Result<Bounds, Error> {
        let version = v.get("version").and_then(|n| n.as_u64());
        if version != Some(1) {
            return Err(Error::Format(format!("bounds sidecar: unsupported version {version:?}")));
        }
        let rows = v
            .get("experts")
            .and_then(|e| e.as_array())
            .ok_or_else(|| Error::Format("bounds sidecar: no `experts` array".into()))?;
        let mut b = Bounds::default();
        for (i, row) in rows.iter().enumerate() {
            let (Some(l), Some(e), Some(c)) = (
                row.get("layer").and_then(|x| x.as_u64()),
                row.get("expert").and_then(|x| x.as_u64()),
                row.get("c").and_then(|x| x.as_f64()),
            ) else {
                return Err(Error::Format(format!("bounds sidecar: malformed row {i}: {row}")));
            };
            b.c.insert((l as usize, e as usize), c);
        }
        if b.c.is_empty() {
            return Err(Error::Format("bounds sidecar: zero experts — measuring against it would report every position unbounded".into()));
        }
        Ok(b)
    }
}

/// Read a bounds sidecar (the file `peregrine-skipbound` writes by default as
/// `<model-dir>/expert_bounds.json`).
pub fn load_bounds(path: &std::path::Path) -> Result<Bounds, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Format(format!("read bounds {}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Format(format!("parse bounds {}: {e}", path.display())))?;
    Bounds::from_json(&v)
}

/// Frobenius norm of one quantized weight, read row by row.
///
/// Row-at-a-time so peak memory is one row plus the packed copies, never a
/// dense `[o, i]`: this runs over every expert of a 358 GB container.
fn frobenius(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<f64, Error> {
    let info = QtInfo::detect(st, name, o as i64, i as i64);
    let Some(row_bytes) = QtView::row_bytes(info.fmt, i) else {
        // An f32 tensor: read it directly rather than pretending it is packed.
        let mut dense = vec![0f32; o * i];
        st.read_f32(name, &mut dense)?;
        return Ok(dense.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt());
    };
    let mut q = vec![0u8; o * row_bytes];
    st.read_raw(name, &mut q)?;
    let mut scale = vec![0f32; info.scale_count.max(0) as usize];
    st.read_f32(&format!("{name}.qs"), &mut scale)?;
    let view = QtView { fmt: info.fmt, o, i, gs: info.gs as usize, q: &q, scale: &scale };
    let mut row = vec![0f32; i];
    let mut acc = 0.0f64;
    for r in 0..o {
        view.dequant_row_into(r, &mut row);
        acc += row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    }
    Ok(acc.sqrt())
}

/// Compute `C_e` for every routed expert in the container.
///
/// A missing projection makes that expert's bound **absent** rather than zero:
/// a zero bound reads as "provably skippable", which is exactly the wrong
/// answer to give about a weight the tool could not find.
pub fn compute_bounds(indir: &std::path::Path) -> Result<Bounds, Error> {
    let cfg = Cfg::load(indir)?;
    let st = SafeTensors::open(indir)?;
    let (hidden, inter) = (cfg.hidden as usize, cfg.moe_inter as usize);
    let mut out = Bounds::default();
    for l in cfg.first_dense as usize..cfg.n_layers as usize {
        for e in 0..cfg.n_experts as usize {
            let p = |s: &str| format!("model.layers.{l}.mlp.experts.{e}.{s}");
            let (g, u, d) = (p("gate_proj.weight"), p("up_proj.weight"), p("down_proj.weight"));
            let have = [&g, &u, &d].iter().all(|n| st.tensors().iter().any(|t| &&t.name == n));
            if !have {
                continue;
            }
            let c = frobenius(&st, &g, inter, hidden)?
                * frobenius(&st, &u, inter, hidden)?
                * frobenius(&st, &d, hidden, inter)?;
            out.c.insert((l, e), c);
        }
    }
    if out.c.is_empty() {
        return Err(Error::Format(format!("{}: no routed experts found", indir.display())));
    }
    Ok(out)
}

/// How often the bound is tight enough to license a skip.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Tightness {
    /// Routed (position, expert) pairs examined.
    pub routed: u64,
    /// Pairs whose bound is below each relative threshold — i.e. provably
    /// contributing less than that share of the largest bound in their own
    /// routed set. Parallel to [`THRESHOLDS`].
    pub skippable: Vec<u64>,
    /// Pairs the **gate weight alone** would have called negligible, at the
    /// same thresholds. The comparison that matters: `COLI_ROUTE_MIN_SHARE`
    /// already drops low-gate experts, so the weight bound only earns its
    /// per-token cost by the margin it adds over this.
    pub skippable_gate_only: Vec<u64>,
    /// Positions whose routed set had no usable bound (unknown experts).
    pub positions_unbounded: u64,
}

/// Relative-bound thresholds the report is cut at. 1% is the aggressive end of
/// what an error budget might tolerate; 0.01% is where a skip is unarguable.
pub const THRESHOLDS: [f64; 4] = [0.0001, 0.001, 0.01, 0.05];

impl Tightness {
    /// Fraction of routed reads the bound proves skippable at `THRESHOLDS[k]`.
    pub fn fraction(&self, k: usize) -> f64 {
        Self::frac(self.routed, self.skippable.get(k))
    }

    /// The same fraction from the gate weight alone — the baseline the bound
    /// has to beat to be worth computing.
    pub fn fraction_gate_only(&self, k: usize) -> f64 {
        Self::frac(self.routed, self.skippable_gate_only.get(k))
    }

    fn frac(routed: u64, hits: Option<&u64>) -> f64 {
        match (routed, hits) {
            (0, _) | (_, None) => 0.0,
            (n, Some(&s)) => s as f64 / n as f64,
        }
    }

    /// The verdict this measurement exists to produce.
    ///
    /// Deliberately blunt in the negative case. The read-path change is
    /// expensive and irreversible-ish; "the prototype said the bound is loose"
    /// has to be as easy to read as a success would be.
    pub fn verdict(&self) -> String {
        let mut s = format!(
            "{} routed expert-reads examined\n\
             {:>10} {:>12} {:>12} {:>10}\n",
            self.routed, "threshold", "with bound", "gate only", "the bound adds"
        );
        for (k, t) in THRESHOLDS.iter().enumerate() {
            let (b, g) = (self.fraction(k) * 100.0, self.fraction_gate_only(k) * 100.0);
            s.push_str(&format!("{:>9.4}% {:>11.2}% {:>11.2}% {:>13.2}%\n", t * 100.0, b, g, b - g));
        }
        if self.positions_unbounded > 0 {
            s.push_str(&format!(
                "  {} position(s) had no usable bound — those experts are missing from the sidecar\n",
                self.positions_unbounded
            ));
        }
        // Two independent ways this can fail, checked in that order.
        //
        // First: nothing is skippable at all, so there is no feature here.
        // Second: plenty is skippable, but the *gate weight alone* already
        // identified it — and the engine can act on that today via
        // `COLI_ROUTE_MIN_SHARE`, with no sidecar, no per-token norm arithmetic
        // and no new file format to keep in sync with the container. A bound
        // that merely restates the gate is work for nothing, and reporting the
        // raw fraction would read as a win.
        let last = THRESHOLDS.len() - 1;
        let best = self.fraction(last);
        let margin = best - self.fraction_gate_only(last);
        if best >= 0.01 && margin < 0.01 {
            s.push_str(
                "\nVERDICT: the weight bound adds nothing the gate weight does not already say. \n\
                 Whatever it would skip, `COLI_ROUTE_MIN_SHARE` already skips — without a sidecar, \n\
                 without per-token norm arithmetic, and without a new file format to keep in sync \n\
                 with the container. Do not wire this; size that knob instead.",
            );
            return s;
        }
        s.push_str(if best < 0.01 {
            "\nVERDICT: the bound is too loose to be worth wiring. Fewer than 1% of reads are \n\
             provably skippable even at the most permissive threshold, so a runtime check would \n\
             cost per-token work and eliminate almost nothing. Do not touch the read path."
        } else if best < 0.10 {
            "\nVERDICT: marginal. Some reads are provably skippable, but not enough to pay for a \n\
             per-token bound check plus the prefetch disruption a skipped read causes. Tighten the \n\
             bound (spectral rather than Frobenius norms) and re-measure before wiring anything."
        } else {
            "\nVERDICT: worth pursuing. A material fraction of reads are provably skippable. Next \n\
             step is still not the read path: confirm on a second workload, since this fraction is \n\
             a property of the routing distribution and not of the weights alone."
        });
        s
    }
}

/// One position's routed set: expert ids and their gate weights.
pub type Frame = (usize, Vec<i32>, Vec<f32>);

/// Measure the bound's tightness over a routing trace.
///
/// `‖x‖²` cancels within a position, so this compares `g_e · C_e` against the
/// largest in the same routed set. That is the comparison a runtime skip would
/// make, which is why it can be evaluated without ever seeing a hidden state.
pub fn measure(bounds: &Bounds, frames: &[Frame]) -> Tightness {
    let mut t = Tightness {
        skippable: vec![0; THRESHOLDS.len()],
        skippable_gate_only: vec![0; THRESHOLDS.len()],
        ..Tightness::default()
    };
    for (layer, experts, weights) in frames {
        // `(g·C, g)` per routed expert: the bound, and the gate alone.
        let pairs: Vec<(f64, f64)> = experts
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| {
                if e < 0 {
                    return None; // an unfilled top-k slot
                }
                let c = bounds.c.get(&(*layer, e as usize))?;
                let g = f64::from(weights.get(i).copied().unwrap_or(1.0).abs());
                Some((g * c, g))
            })
            .collect();
        let scored: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let gate_max = pairs.iter().map(|p| p.1).fold(0.0f64, f64::max);
        let Some(&max) = scored.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) else {
            t.positions_unbounded += 1;
            continue;
        };
        // `<=` rather than `!(> 0.0)` so a NaN bound — a degenerate container,
        // not a skip opportunity — takes this branch too instead of dividing.
        if max <= 0.0 || !max.is_finite() {
            t.positions_unbounded += 1;
            continue;
        }
        for &(v, g) in &pairs {
            t.routed += 1;
            for (k, thr) in THRESHOLDS.iter().enumerate() {
                if v / max < *thr {
                    if let Some(slot) = t.skippable.get_mut(k) {
                        *slot += 1;
                    }
                }
                if gate_max > 0.0 && g / gate_max < *thr {
                    if let Some(slot) = t.skippable_gate_only.get_mut(k) {
                        *slot += 1;
                    }
                }
            }
        }
    }
    t
}

/// Parse a routing trace. Two shapes are accepted:
///
/// - **What `peregrine dump-routes` actually writes** —
///   `Model::dump_routes_to` serializes `Vec<Vec<Vec<i32>>>`: a bare array of
///   *positions*, each an array indexed by layer of routed-expert-id arrays
///   (dense / unrouted layers empty). This reader originally did not parse it:
///   it looked only for `{"layer": ..}` objects, silently `continue`d past
///   every nested row, and reported "no usable frames" against real traces —
///   which is exactly how the 2026-08-13 run lost its "reads skipped by
///   bounds" number (`bench-data/2026-08-13-int3g64/skipbound.log`).
/// - The object form `{"frames": [{"layer": N, "experts": [..],
///   "weights": [..]}]}` (or a bare array of those objects) — hand-written
///   fixtures and any future weight-carrying producer.
///
/// The nested shape carries no gate weights; [`measure`] defaults an absent
/// weight to `1.0`, so the `g·C` column still ranks by the Frobenius bound
/// while the gate-only column is degenerate for such traces — a property of
/// the trace, reported rather than silently wrong.
pub fn load_frames(path: &std::path::Path) -> Result<Vec<Frame>, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Format(format!("read trace {}: {e}", path.display())))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::Format(format!("parse trace {}: {e}", path.display())))?;
    let arr = match v.get("frames").and_then(|f| f.as_array()) {
        Some(f) => f.clone(),
        None => match v.as_array() {
            Some(f) => f.clone(),
            None => return Err(Error::Format(format!("trace {}: expected frames", path.display()))),
        },
    };
    let mut out = Vec::with_capacity(arr.len());
    for f in &arr {
        // dump-routes nested form: this element is one position's per-layer
        // routed sets. Layer index is the array position; empty layers (dense,
        // or not yet routed) contribute no frame.
        if let Some(pos_layers) = f.as_array() {
            for (layer, ids) in pos_layers.iter().enumerate() {
                let Some(ids) = ids.as_array() else { continue };
                let ids: Vec<i32> = ids.iter().filter_map(|e| e.as_i64()).map(|e| e as i32).collect();
                if !ids.is_empty() {
                    out.push((layer, ids, Vec::new()));
                }
            }
            continue;
        }
        let Some(layer) = f.get("layer").and_then(|l| l.as_u64()) else { continue };
        let Some(experts) = f.get("experts").and_then(|e| e.as_array()) else { continue };
        let ids: Vec<i32> = experts.iter().filter_map(|e| e.as_i64()).map(|e| e as i32).collect();
        let weights: Vec<f32> = match f.get("weights").and_then(|w| w.as_array()) {
            Some(w) => w.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect(),
            None => Vec::new(),
        };
        out.push((layer as usize, ids, weights));
    }
    if out.is_empty() {
        return Err(Error::Format(format!("trace {}: no usable frames", path.display())));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds_of(rows: &[(usize, usize, f64)]) -> Bounds {
        let mut b = Bounds::default();
        for &(l, e, c) in rows {
            b.c.insert((l, e), c);
        }
        b
    }

    #[test]
    fn a_sidecar_round_trips_and_malformed_rows_are_refused_not_skipped() -> Result<(), Error> {
        // from_json is what lets a trace analysis skip the container pass, so
        // it must reproduce exactly what to_json wrote…
        let b = bounds_of(&[(0, 0, 1.5), (0, 3, 0.25), (7, 200, 1e-6)]);
        let back = Bounds::from_json(&b.to_json())?;
        assert_eq!(back.c, b.c, "sidecar round-trip must be exact");
        // …and refuse rather than thin out: a dropped row would read as
        // "unbounded" downstream and shrink the measured denominator silently.
        let mut v = b.to_json();
        if let Some(rows) = v.get_mut("experts").and_then(|e| e.as_array_mut()) {
            rows.push(serde_json::json!({ "layer": 1, "expert": "not-a-number", "c": 0.5 }));
        }
        assert!(Bounds::from_json(&v).is_err(), "a malformed row must fail the load");
        // Version and emptiness are hard errors too.
        assert!(Bounds::from_json(&serde_json::json!({"version": 2, "experts": []})).is_err());
        assert!(Bounds::from_json(&serde_json::json!({"version": 1, "experts": []})).is_err());
        Ok(())
    }

    #[test]
    fn load_frames_reads_the_shape_dump_routes_actually_writes() -> Result<(), Error> {
        // The 2026-08-13 defect in miniature: a positions × layers × ids array
        // (dump-routes' real output) used to yield "no usable frames". Layer
        // index comes from array position; empty (dense) layers emit nothing.
        let dir = std::env::temp_dir().join(format!("peregrine_sb_shape_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("routes.json");
        std::fs::write(&path, r#"[[[],[1,2],[3]],[[],[],[4,5]]]"#)?;
        let frames = load_frames(&path)?;
        assert_eq!(
            frames,
            vec![
                (1usize, vec![1, 2], Vec::<f32>::new()),
                (2, vec![3], Vec::new()),
                (2, vec![4, 5], Vec::new()),
            ],
            "layer = array index, empty layers skipped, no weights in this shape"
        );
        // The object form keeps working alongside it.
        std::fs::write(&path, r#"{"frames":[{"layer":0,"experts":[7],"weights":[0.5]}]}"#)?;
        let frames = load_frames(&path)?;
        assert_eq!(frames, vec![(0usize, vec![7], vec![0.5])]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_real_dump_routes_trace_round_trips_into_measurable_frames() -> Result<(), Error> {
        // The acceptance for the fix: dump-routes on the synthetic tiny model
        // (no real-model disk read anywhere), feed the file to this reader,
        // and the measurement pipeline must see frames — the exact end-to-end
        // path that silently produced nothing on 2026-08-13.
        let dir = std::env::temp_dir().join(format!("peregrine_sb_rt_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        // Streaming + a small forced cache: `dump_routes` requires the routing
        // history, which only a streaming model builds.
        let mut m = peregrine_model::Model::load_streaming_ecache(&dir, true, 1 << 20)?;
        let corpus: Vec<i32> = (0..12).map(|k| (k * 5 + 1) % 32).collect();
        let trace_path = dir.join("routes.json");
        let n = m.dump_routes_to(&corpus, &trace_path)?;
        assert_eq!(n, corpus.len(), "one frame per forward");

        let frames = load_frames(&trace_path)?;
        assert!(!frames.is_empty(), "the reader must parse its own producer's output");
        let bounds = compute_bounds(&dir)?;
        let t = measure(&bounds, &frames);
        assert!(t.routed > 0, "bounded routed experts must be examined, got {t:?}");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_flat_bound_licenses_no_skips_and_the_verdict_says_so() {
        // The outcome the plan warned about and the reason this is a prototype
        // rather than a feature: if every routed expert's bound is comparable,
        // nothing is provably negligible and a runtime check is pure cost.
        let b = bounds_of(&[(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
        let frames = vec![(0usize, vec![0, 1, 2, 3], vec![0.25, 0.25, 0.25, 0.25])];
        let t = measure(&b, &frames);
        assert_eq!(t.routed, 4);
        assert_eq!(t.skippable, vec![0, 0, 0, 0], "identical bounds are never relatively small");
        assert_eq!(t.skippable_gate_only, vec![0, 0, 0, 0], "and equal gates say nothing either");
        assert!(t.verdict().contains("Do not touch the read path"), "{}", t.verdict());
    }

    #[test]
    fn a_skewed_bound_is_counted_at_every_threshold_it_clears() {
        // Expert 3 contributes at most a millionth of expert 0's ceiling, so it
        // clears every threshold; expert 2 clears only the loosest.
        let b = bounds_of(&[(0, 0, 1.0), (0, 1, 1.0), (0, 2, 0.02), (0, 3, 1e-6)]);
        let frames = vec![(0usize, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0])];
        let t = measure(&b, &frames);
        assert_eq!(t.routed, 4);
        // thresholds: 0.01%, 0.1%, 1%, 5%
        assert_eq!(t.skippable, vec![1, 1, 1, 2], "expert 3 everywhere; expert 2 only at 5%");
        assert!((t.fraction(3) - 0.5).abs() < 1e-12);
        assert!(t.verdict().contains("worth pursuing"), "{}", t.verdict());
    }

    #[test]
    fn the_gate_weight_is_part_of_the_bound_not_just_the_weights() {
        // The whole point of a *pre-read* skip: the same expert is negligible at
        // one position and decisive at another, and only the gate tells them
        // apart. A bound computed from weights alone could never skip anything
        // that top-k had already selected.
        let b = bounds_of(&[(0, 0, 1.0), (0, 1, 1.0)]);
        let heavy = measure(&b, &[(0usize, vec![0, 1], vec![1.0, 1.0])]);
        assert_eq!(heavy.skippable[3], 0, "equal gates, nothing skippable");
        let skewed = measure(&b, &[(0usize, vec![0, 1], vec![1.0, 1e-5])]);
        assert_eq!(skewed.skippable[3], 1, "a negligible gate makes the same expert skippable");
    }

    #[test]
    fn an_expert_with_no_bound_is_not_treated_as_zero() {
        // A missing weight must never read as "provably skippable" — that is the
        // one way this idea could silently drop a contribution that mattered.
        let b = bounds_of(&[(0, 0, 1.0)]);
        // Expert 7 has no entry: it is excluded from the scored set entirely.
        let t = measure(&b, &[(0usize, vec![0, 7], vec![1.0, 1.0])]);
        assert_eq!(t.routed, 1, "the unbounded expert is not counted as examined");
        assert_eq!(t.skippable[3], 0, "and certainly not counted as skippable");
        // A position where *nothing* is bounded is reported, not silently dropped.
        let t = measure(&b, &[(0usize, vec![7, 8], vec![1.0, 1.0])]);
        assert_eq!(t.positions_unbounded, 1);
        assert_eq!(t.routed, 0);
    }

    #[test]
    fn unfilled_topk_slots_are_not_expert_zero() {
        // `-1` marks a slot the router did not fill (COLI_ROUTE_MIN_SHARE
        // truncation). Reading it as expert 0 would score a real expert against
        // a position that never routed it.
        let b = bounds_of(&[(0, 0, 1.0), (0, 1, 1e-9)]);
        let t = measure(&b, &[(0usize, vec![0, 1, -1, -1], vec![1.0, 1.0, 0.0, 0.0])]);
        assert_eq!(t.routed, 2);
    }

    #[test]
    fn the_verdict_is_graded_not_binary() {
        // Gate-only zero, so the whole skippable fraction is margin the bound
        // actually added — the case where grading by absolute fraction is the
        // right question.
        let mk = |frac: f64| Tightness {
            routed: 1000,
            skippable: vec![0, 0, 0, (1000.0 * frac) as u64],
            skippable_gate_only: vec![0, 0, 0, 0],
            positions_unbounded: 0,
        };
        assert!(mk(0.005).verdict().contains("too loose"));
        assert!(mk(0.05).verdict().contains("marginal"));
        assert!(mk(0.30).verdict().contains("worth pursuing"));
        // Even the good case refuses to recommend the read path directly.
        assert!(mk(0.30).verdict().contains("still not the read path"));
    }

    #[test]
    fn a_bound_that_only_restates_the_gate_is_reported_as_worthless() {
        // **The question this tool exists to settle.** `COLI_ROUTE_MIN_SHARE`
        // already drops low-gate experts with no sidecar, no per-token norm
        // arithmetic and no new file format. A bound whose skips the gate would
        // have made anyway is work for nothing, and that has to be as loud as a
        // success would be — otherwise the fraction alone reads as a win.
        let t = Tightness {
            routed: 1000,
            skippable: vec![0, 0, 0, 300],
            skippable_gate_only: vec![0, 0, 0, 295],
            positions_unbounded: 0,
        };
        let v = t.verdict();
        assert!(v.contains("adds nothing the gate weight does not already say"), "{v}");
        assert!(v.contains("COLI_ROUTE_MIN_SHARE"), "it must name the knob that already does this");
        assert!(!v.contains("worth pursuing"), "30% skippable must not read as a win when 29.5% was free");
        // …and a bound that genuinely adds margin is still graded on its own.
        let t = Tightness { skippable_gate_only: vec![0, 0, 0, 50], ..t };
        assert!(t.verdict().contains("worth pursuing"), "{}", t.verdict());
    }

    #[test]
    fn the_sidecar_records_the_bound_it_used() -> Result<(), Error> {
        // A bound file whose formula is not written down is unusable a month
        // later: the consumer has to know it is quadratic in ||x|| and that the
        // norms are Frobenius, or it will apply it wrongly.
        let b = bounds_of(&[(3, 7, 2.5)]);
        let j = b.to_json();
        let s = serde_json::to_string(&j).map_err(|e| Error::Format(e.to_string()))?;
        assert!(s.contains("||x||^2"), "the formula must travel with the numbers");
        assert!(s.contains("W_down"));
        assert!(s.contains("\"layer\":3"));
        assert!(s.contains("\"expert\":7"));
        Ok(())
    }

    /// Writes the tiny fixture to `PEREGRINE_FIXTURE_DIR` when set, so the
    /// binary can be exercised by hand against a real container. Inert
    /// otherwise — a test that wrote to a fixed path would race the suite.
    #[test]
    fn emit_fixture_for_manual_runs() -> Result<(), Error> {
        let Ok(dir) = std::env::var("PEREGRINE_FIXTURE_DIR") else { return Ok(()) };
        peregrine_model::testkit::build_tiny_model(std::path::Path::new(&dir))?;
        Ok(())
    }

    #[test]
    fn bounds_are_computed_from_a_real_container() -> Result<(), Error> {
        // The offline half, end to end: every routed expert of the fixture gets
        // a finite, positive C_e.
        let dir = std::env::temp_dir().join(format!("peregrine_skipbound_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let b = compute_bounds(&dir)?;
        let sparse = (cfg.n_layers - cfg.first_dense) as usize;
        assert_eq!(b.c.len(), sparse * cfg.n_experts as usize, "one bound per routed expert");
        assert!(b.c.values().all(|c| c.is_finite() && *c > 0.0), "a bound of zero would license a wrong skip");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

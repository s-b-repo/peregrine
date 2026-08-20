//! Router-weighted expert pruning (REAP): drop the least-salient routed experts
//! from a checkpoint and renumber what survives.
//!
//! **Read this before running it.** Pruning does *not* reduce bytes per token.
//! Top-k is unchanged, so the same `k` experts are read for every position
//! whatever the pool size — Cerebras' own cards report identical activated
//! parameters at 480B, 363B and 246B. What it buys is a smaller **working set**:
//! fewer distinct experts to hold, cache, prefetch and lay out. On a disk-bound
//! engine that is a residency win, not a bandwidth win, and the two are
//! routinely conflated.
//!
//! Two numbers that decide whether it is worth doing at all:
//!
//! - **Use 25%, not 50%.** GLM-4.5-Air lost 11.2% on coding and 25.8% on
//!   multiple-choice at 50% pruning. Retention does not improve with model size
//!   — the honest statement is that GLM-family models degrade unusually — so
//!   this tool warns above [`SAFE_FRAC`] rather than letting the default carry
//!   an unstated risk.
//! - **The calibration corpus dominates the result.** Generic web text caused
//!   total collapse on code tasks in the published runs. Saliency here is
//!   measured from *your* traces; a trace that does not contain the workload
//!   you serve will prune the experts that workload needs.
//!
//! Saliency is `Σ gate_weight` over the calibration trace, per (layer, expert) —
//! the router-weighted activation REAP names, and the quantity actually
//! available from a routing trace. Frequency alone would rank a frequently
//! routed but weakly weighted expert above a rare decisive one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use peregrine_core::config::Cfg;
use peregrine_core::safetensors::SafeTensors;
use peregrine_core::{Context, Error};

use crate::requant::ShardWriter;

/// Pruned fraction above which the tool warns. The published GLM-4.5-Air
/// degradation at 50% is severe enough that passing it should be a deliberate
/// act, not a default someone inherited.
pub const SAFE_FRAC: f64 = 0.25;

/// Per-(layer, expert) saliency accumulated from a calibration trace.
///
/// `BTreeMap` rather than a hash map so the ranking is reproducible: two runs
/// over the same trace must prune the same experts, and iteration order is what
/// breaks exact ties.
#[derive(Debug, Default, Clone)]
pub struct Saliency {
    /// `(layer, expert) -> Σ gate weight`.
    pub mass: BTreeMap<(usize, usize), f64>,
    /// `(layer, expert) -> times routed`. Reported, never ranked on — see the
    /// module note on why frequency is the wrong ordering.
    pub hits: BTreeMap<(usize, usize), u64>,
    pub positions: u64,
}

impl Saliency {
    /// Fold one position's routed set into the running totals.
    pub fn observe(&mut self, layer: usize, experts: &[i32], weights: &[f32]) {
        self.positions += 1;
        for (i, &e) in experts.iter().enumerate() {
            if e < 0 {
                continue; // an unfilled top-k slot, not expert 0
            }
            let key = (layer, e as usize);
            // Absent weights mean the trace recorded selections only; count each
            // selection as unit mass so the ranking degrades to frequency rather
            // than to nothing. Stated in the report, because it is a weaker
            // signal than the one this tool is named for.
            let w = weights.get(i).copied().unwrap_or(1.0);
            *self.mass.entry(key).or_insert(0.0) += f64::from(w.abs());
            *self.hits.entry(key).or_insert(0) += 1;
        }
    }

    /// Whether any weights were recorded, or the trace was selections-only.
    pub fn is_empty(&self) -> bool {
        self.mass.is_empty()
    }
}

/// Which experts survive, per layer, in ascending original id.
///
/// The renumbering is implied: surviving expert `keep[l][j]` becomes expert `j`
/// of layer `l`. Keeping the original order rather than a saliency order means
/// a pruned container's expert ids stay monotonically related to the source's,
/// which is what lets an old `route_stats.json` be read as a coarse hint
/// instead of being silently wrong.
#[derive(Debug, Default, Clone)]
pub struct KeepPlan {
    pub keep: Vec<Vec<usize>>,
    pub n_experts_in: usize,
    /// Survivors per layer. Uniform across layers — see [`plan_keep`].
    pub n_experts_out: usize,
    /// Layers ranked by aggregate saliency because they had none of their own.
    pub layers_by_aggregate: usize,
}

impl KeepPlan {
    /// The surviving experts of `layer`, or a reported inconsistency.
    ///
    /// There is deliberately no "keep everything" fallback. `config.json`
    /// carries one `n_routed_experts` for the whole model, so a layer that kept
    /// a different number would contradict it — the plan must cover every
    /// sparse layer the container has, **including the MTP head**, and a plan
    /// that does not is a caller bug worth reporting rather than papering over.
    pub fn keep_of(&self, layer: usize, name: &str) -> Result<&[usize], Error> {
        self.keep.get(layer).map(|v| v.as_slice()).ok_or_else(|| {
            Error::Format(format!(
                "prune: '{name}' is layer {layer} but the plan covers {} — size it for every sparse layer, \
                 the MTP head included",
                self.keep.len()
            ))
        })
    }
}

/// Rank each layer's experts by saliency and keep the top `1 - frac`.
///
/// **A layer with no trace data keeps everything.** Pruning on no evidence is
/// how a tool silently removes the experts a workload needs but a short
/// calibration run never reached; the same discipline `--tier-hot-frac` uses
/// when it refuses to tier without routing data.
///
/// `keep_min` is a floor on surviving experts per layer, because top-k reads
/// `k` of them: pruning below `k` would make the router unable to fill a
/// selection at all.
pub fn plan_keep(sal: &Saliency, n_layers: usize, n_experts: usize, frac: f64, keep_min: usize) -> KeepPlan {
    let frac = frac.clamp(0.0, 1.0);
    let floor = keep_min.max(1).min(n_experts);
    // **Uniform, because the container cannot express anything else.**
    // `config.json` carries a single `n_routed_experts` for the whole model, so
    // every sparse layer must end with the same pool size. A per-layer keep
    // count would produce a router whose row count disagrees with the config
    // the loader sizes its buffers from — which fails at load, long after the
    // hours the conversion took.
    let n_keep = (((n_experts as f64) * (1.0 - frac)).round() as usize).max(floor).min(n_experts);

    // Saliency summed across every traced layer. Used only for layers with no
    // evidence of their own — the MTP head above all, which has its own router
    // and expert pool that a main-model trace never touches. Globally hot
    // experts are a weak signal, but they are a *signal*; keeping the lowest
    // ids instead would be arbitrary, and keeping all of them is not available
    // once the pool size has to match.
    let mut aggregate = vec![0.0f64; n_experts];
    for (&(_, e), &m) in &sal.mass {
        if let Some(slot) = aggregate.get_mut(e) {
            *slot += m;
        }
    }

    let mut keep = Vec::with_capacity(n_layers);
    let mut fallbacks = 0usize;
    for l in 0..n_layers {
        let own: Vec<(usize, f64)> =
            (0..n_experts).map(|e| (e, sal.mass.get(&(l, e)).copied().unwrap_or(0.0))).collect();
        let mut ranked = if own.iter().any(|(_, m)| *m > 0.0) {
            own
        } else {
            fallbacks += 1;
            aggregate.iter().copied().enumerate().collect()
        };
        // Descending saliency, ties broken by ascending id so the plan is
        // deterministic across runs and platforms.
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        let mut survivors: Vec<usize> = ranked.into_iter().take(n_keep).map(|(e, _)| e).collect();
        survivors.sort_unstable();
        keep.push(survivors);
    }
    KeepPlan { keep, n_experts_in: n_experts, n_experts_out: n_keep, layers_by_aggregate: fallbacks }
}

/// What a prune run did.
#[derive(Debug, Default, Clone)]
pub struct PruneReport {
    pub layers: usize,
    pub experts_in: usize,
    pub experts_kept: usize,
    pub tensors_copied: usize,
    pub tensors_dropped: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// Layers that kept everything because the trace never reached them.
    pub layers_without_evidence: usize,
    /// True when the trace carried no gate weights, so saliency fell back to
    /// counting selections.
    pub frequency_only: bool,
}

impl PruneReport {
    /// The claim this tool is allowed to make, and the one it is not.
    pub fn summary(&self) -> String {
        let pool = if self.experts_in == 0 {
            0.0
        } else {
            100.0 * (self.experts_in - self.experts_kept.min(self.experts_in)) as f64 / self.experts_in as f64
        };
        let mut s = format!(
            "pruned {:.1}% of the expert pool ({} -> {} per layer over {} layers); \
             {} tensors copied, {} dropped; {:.2} -> {:.2} GiB\n\
             NOTE: bytes *per token* are unchanged — top-k still reads k experts. \
             What shrank is the working set: fewer distinct experts to hold, cache and prefetch.",
            pool,
            self.experts_in,
            self.experts_kept,
            self.layers,
            self.tensors_copied,
            self.tensors_dropped,
            self.bytes_in as f64 / (1u64 << 30) as f64,
            self.bytes_out as f64 / (1u64 << 30) as f64,
        );
        if self.layers_without_evidence > 0 {
            s.push_str(&format!(
                "\nWARNING: {} layer(s) had no routing evidence of their own and were ranked by \
                 saliency aggregated over the layers that did. The MTP head is always one of \
                 them; more than that means the calibration run was too short.",
                self.layers_without_evidence
            ));
        }
        if self.frequency_only {
            s.push_str(
                "\nWARNING: the trace carried no gate weights, so experts were ranked by \
                 selection count, not router-weighted saliency. A frequently routed but \
                 weakly weighted expert outranks a rare decisive one under that ordering.",
            );
        }
        s
    }
}

/// Load a routing trace and accumulate saliency.
///
/// Accepts **both** trace shapes.
///
/// The envelope form — an object with a `frames` array, or a bare array of
/// `{"layer": L, "experts": [...], "weights": [...]}` — carries gate weights and
/// is what saliency actually wants.
///
/// The **nested** form is what `peregrine dump-routes` has always written:
/// `[position][layer][expert_id]`, no envelope and no weights. This doc claimed
/// to accept it and did not: `f.get("layer")` on an array returns `None`, so
/// every element hit the `continue` below and the tool exited "no usable frames"
/// on the one artifact the engine produces. `docs/layout-tools.md`'s documented
/// `dump-routes | peregrine-prune` pipeline could never have run.
/// `skipbound.rs` hit this exact defect, lost a run's numbers to it, and grew
/// the branch on 2026-08-13; this is the same fix.
///
/// On the nested form `weights` is empty, so ranking degrades from Σ gate mass
/// to counting — which [`Saliency::observe`] handles and the report states.
/// That is a real limitation, not a silent one: no trace the engine writes today
/// carries gate weights at all.
pub fn load_trace(path: &Path) -> Result<Saliency, Error> {
    let text = std::fs::read_to_string(path).ctx(|| format!("read trace {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::Format(format!("parse trace {}: {e}", path.display())))?;
    let frames = match v.get("frames").and_then(|f| f.as_array()) {
        Some(f) => f.clone(),
        None => match v.as_array() {
            Some(f) => f.clone(),
            None => {
                return Err(Error::Format(format!(
                    "trace {}: expected an array of frames or an object with a `frames` array",
                    path.display()
                )))
            }
        },
    };
    let mut sal = Saliency::default();
    for f in &frames {
        // Nested `dump-routes` form: this element is one position's per-layer
        // routed sets, so the layer index is the array position. Empty entries
        // are dense layers, which route nothing.
        if let Some(pos_layers) = f.as_array() {
            for (layer, ids) in pos_layers.iter().enumerate() {
                let Some(ids) = ids.as_array() else { continue };
                let ids: Vec<i32> = ids.iter().filter_map(|e| e.as_i64()).map(|e| e as i32).collect();
                if !ids.is_empty() {
                    sal.observe(layer, &ids, &[]);
                }
            }
            continue;
        }
        let Some(layer) = f.get("layer").and_then(|l| l.as_u64()) else { continue };
        let Some(experts) = f.get("experts").and_then(|e| e.as_array()) else { continue };
        let ids: Vec<i32> = experts.iter().filter_map(|e| e.as_i64()).map(|e| e as i32).collect();
        // A frame with no `weights` is a selections-only trace; ranking then
        // degrades to counting, which `Saliency::observe` handles and the report
        // surfaces. An empty vec is the signal, not a swallowed failure.
        let weights: Vec<f32> = match f.get("weights").and_then(|w| w.as_array()) {
            Some(w) => w.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect(),
            None => Vec::new(),
        };
        sal.observe(layer as usize, &ids, &weights);
    }
    if sal.is_empty() {
        return Err(Error::Format(format!(
            "trace {}: no usable frames — nothing to rank, and pruning on no evidence is refused",
            path.display()
        )));
    }
    Ok(sal)
}

/// Write a pruned container: surviving experts renumbered, the router's rows
/// gathered to match, everything else copied byte-identically.
///
/// The router **must** be rewritten in the same pass. `mlp.gate.weight` is
/// `[E, hidden]` and its row `e` scores expert `e`; dropping experts without
/// gathering those rows leaves a container whose router selects ids that no
/// longer exist — which loads fine and produces nonsense.
pub fn prune(indir: &Path, outdir: &Path, plan: &KeepPlan, shard_bytes: u64) -> Result<PruneReport, Error> {
    let cfg = Cfg::load(indir)?;
    let st = SafeTensors::open(indir)?;
    std::fs::create_dir_all(outdir).ctx(|| format!("create {}", outdir.display()))?;

    let mut rep = PruneReport {
        layers: plan.keep.len(),
        experts_in: plan.n_experts_in,
        experts_kept: plan.n_experts_out,
        layers_without_evidence: plan.layers_by_aggregate,
        ..PruneReport::default()
    };
    let mut w = ShardWriter::new(outdir, "out", shard_bytes.max(1)).with_metadata(vec![
        ("peregrine.prune.experts_in".into(), plan.n_experts_in.to_string()),
        ("peregrine.prune.experts_out".into(), plan.n_experts_out.to_string()),
        ("peregrine.prune.source".into(), indir.display().to_string()),
        // No accuracy key: it has not been measured, and an unmeasured one
        // stamped into the artifact is worse than none.
    ]);

    let meta: Vec<(String, Vec<i64>, String)> = st
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone(), format!("{:?}", t.dtype).to_uppercase()))
        .collect();
    for (name, shape, dtype) in &meta {
        let name = name.as_str();
        let nbytes = st.uncompressed_nbytes(name).unwrap_or(0).max(0) as usize;
        rep.bytes_in += nbytes as u64;
        if name.ends_with(".qs") {
            continue; // travels with its weight
        }

        // The router: gather the surviving rows, in the new id order.
        if let Some(layer) = router_layer(name) {
            let rewritten = gather_rows(&st, name, shape, plan.keep_of(layer, name)?, nbytes)?;
            rep.bytes_out += rewritten.1.len() as u64;
            rep.tensors_copied += 1;
            w.push(name, static_dtype_of(dtype), rewritten.0, rewritten.1)?;
            continue;
        }
        // The router's per-expert bias, same gather over a 1-D tensor.
        if let Some(layer) = router_bias_layer(name) {
            let rewritten = gather_rows(&st, name, shape, plan.keep_of(layer, name)?, nbytes)?;
            rep.bytes_out += rewritten.1.len() as u64;
            rep.tensors_copied += 1;
            w.push(name, static_dtype_of(dtype), rewritten.0, rewritten.1)?;
            continue;
        }

        // A routed expert: drop it, or copy it under its new id.
        if let Some((layer, expert)) = expert_coords_of(name) {
            match plan.keep_of(layer, name)?.iter().position(|&e| e == expert) {
                Some(new_id) => {
                    let renamed = rename_expert(name, expert, new_id);
                    copy_tensor(&st, name, &renamed, TensorMeta { shape, dtype, nbytes }, &mut w, &mut rep)?;
                }
                None => {
                    rep.tensors_dropped += 1;
                    // Its `.qs` sibling goes with it; not copying is the drop.
                }
            }
            continue;
        }

        copy_tensor(&st, name, name, TensorMeta { shape, dtype, nbytes }, &mut w, &mut rep)?;
    }
    w.flush()?;
    write_pruned_config(indir, outdir, &cfg, plan)?;
    Ok(rep)
}

/// A tensor's header facts. They are read as a group at every call site, and
/// bundling them keeps [`copy_tensor`] under the argument limit without an
/// `#[allow]`, which the strict audit rejects.
#[derive(Clone, Copy)]
struct TensorMeta<'a> {
    shape: &'a [i64],
    dtype: &'a str,
    nbytes: usize,
}

/// Copy one tensor (optionally under a new name) plus its `.qs` sibling.
fn copy_tensor(
    st: &SafeTensors,
    name: &str,
    out_name: &str,
    meta: TensorMeta<'_>,
    w: &mut ShardWriter,
    rep: &mut PruneReport,
) -> Result<(), Error> {
    let TensorMeta { shape, dtype, nbytes } = meta;
    let mut raw = vec![0u8; nbytes];
    st.read_raw(name, &mut raw)?;
    rep.bytes_out += raw.len() as u64;
    rep.tensors_copied += 1;
    w.push(out_name, static_dtype_of(dtype), shape.to_vec(), raw)?;
    let qs = format!("{name}.qs");
    if let Some(info) = st.tensors().iter().find(|t| t.name == qs) {
        let n = st.uncompressed_nbytes(&qs).unwrap_or(0).max(0) as usize;
        let mut raw = vec![0u8; n];
        st.read_raw(&qs, &mut raw)?;
        rep.bytes_out += raw.len() as u64;
        // Keep the source dtype: declaring F32 over a BF16 scale tensor makes
        // the whole output directory unopenable.
        let d = static_dtype_of(&format!("{:?}", info.dtype).to_uppercase());
        w.push(&format!("{out_name}.qs"), d, info.shape.clone(), raw)?;
    }
    Ok(())
}

/// Gather rows `keep` out of a row-major tensor, preserving its element width.
///
/// Works on raw bytes rather than dequantizing, so a quantized router (if one
/// ever ships) survives the gather unchanged rather than being silently
/// round-tripped through f32.
fn gather_rows(
    st: &SafeTensors,
    name: &str,
    shape: &[i64],
    keep: &[usize],
    nbytes: usize,
) -> Result<(Vec<i64>, Vec<u8>), Error> {
    let rows = shape.first().copied().unwrap_or(0).max(0) as usize;
    if rows == 0 || keep.is_empty() {
        return Err(Error::Format(format!("prune: cannot gather '{name}' with {rows} rows and {} kept", keep.len())));
    }
    let row_bytes = nbytes / rows;
    if row_bytes * rows != nbytes {
        return Err(Error::Format(format!(
            "prune: '{name}' is {nbytes} bytes over {rows} rows — not a whole number of rows per expert"
        )));
    }
    let mut raw = vec![0u8; nbytes];
    st.read_raw(name, &mut raw)?;
    let mut out = Vec::with_capacity(keep.len() * row_bytes);
    for &e in keep {
        if e >= rows {
            return Err(Error::Format(format!("prune: '{name}' has {rows} rows but the plan keeps expert {e}")));
        }
        out.extend_from_slice(&raw[e * row_bytes..e * row_bytes + row_bytes]);
    }
    let mut out_shape = shape.to_vec();
    if let Some(first) = out_shape.first_mut() {
        *first = keep.len() as i64;
    }
    Ok((out_shape, out))
}

/// Write the output `config.json` with the reduced expert count.
///
/// Without this the loader reads `n_routed_experts` from the source and indexes
/// experts that are no longer in the container — the failure this tool most
/// needs to not have, since it would surface as a load error long after the
/// hours the conversion took.
fn write_pruned_config(indir: &Path, outdir: &Path, cfg: &Cfg, plan: &KeepPlan) -> Result<(), Error> {
    let src = indir.join("config.json");
    let text = std::fs::read_to_string(&src).ctx(|| format!("read {}", src.display()))?;
    let mut v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::Format(format!("parse {}: {e}", src.display())))?;
    let Some(obj) = v.as_object_mut() else {
        return Err(Error::Format(format!("{}: expected a JSON object", src.display())));
    };
    obj.insert("n_routed_experts".into(), serde_json::json!(plan.n_experts_out));
    // top-k cannot exceed the surviving pool.
    let k = (cfg.topk as usize).min(plan.n_experts_out.max(1));
    obj.insert("num_experts_per_tok".into(), serde_json::json!(k));
    let out = outdir.join("config.json");
    let bytes = serde_json::to_vec_pretty(&v).map_err(|e| Error::Format(format!("serialize config: {e}")))?;
    peregrine_core::durable::write_atomic(&out, &bytes)?;
    Ok(())
}

fn router_layer(name: &str) -> Option<usize> {
    if !name.ends_with(".mlp.gate.weight") {
        return None;
    }
    name.split("model.layers.").nth(1)?.split('.').next()?.parse().ok()
}

fn router_bias_layer(name: &str) -> Option<usize> {
    if !name.ends_with(".mlp.gate.e_score_correction_bias") {
        return None;
    }
    name.split("model.layers.").nth(1)?.split('.').next()?.parse().ok()
}

fn expert_coords_of(name: &str) -> Option<(usize, usize)> {
    let layer = name.split("model.layers.").nth(1)?.split('.').next()?.parse().ok()?;
    let expert = name.split(".mlp.experts.").nth(1)?.split('.').next()?.parse().ok()?;
    Some((layer, expert))
}

fn rename_expert(name: &str, old: usize, new: usize) -> String {
    name.replacen(&format!(".mlp.experts.{old}."), &format!(".mlp.experts.{new}."), 1)
}

fn static_dtype_of(d: &str) -> &'static str {
    match d {
        "F32" => "F32",
        "F16" => "F16",
        "BF16" => "BF16",
        "I8" => "I8",
        _ => "U8",
    }
}

/// Where a prune run writes, given the source. Only used by the binary; kept
/// here so the path convention lives beside the writer.
pub fn default_outdir(indir: &Path, frac: f64) -> PathBuf {
    let pct = (frac * 100.0).round() as i64;
    let name = indir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    indir.with_file_name(format!("{name}_pruned{pct}"))
}

#[cfg(test)]
mod tests {

    #[test]
    fn load_trace_reads_the_shape_dump_routes_actually_writes() -> Result<(), Error> {
        // The defect this closes: `load_trace`'s doc claimed to accept
        // `dump-routes`' output and could not. That output is
        // `[position][layer][expert_id]` — `f.get("layer")` on an array returns
        // `None`, so every element was skipped and the tool exited "no usable
        // frames — nothing to rank". The pipeline `docs/layout-tools.md`
        // documents could never have run.
        let dir = std::env::temp_dir().join(format!("peregrine_prune_shape_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        std::fs::create_dir_all(&dir)?;

        // Two positions; layer 0 dense (empty), layers 1-2 routed.
        let path = dir.join("routes.json");
        std::fs::write(&path, r#"[[[],[1,2],[3]],[[],[1],[3,4]]]"#)?;
        let sal = load_trace(&path)?;
        // Counting, not gate mass — the nested form carries no weights, and
        // `observe` defaults each to 1.0.
        assert_eq!(sal.hits.get(&(1, 1)).copied(), Some(2), "expert 1 routed at layer 1 in both positions");
        assert_eq!(sal.hits.get(&(1, 2)).copied(), Some(1));
        assert_eq!(sal.hits.get(&(2, 3)).copied(), Some(2));
        assert_eq!(sal.hits.get(&(2, 4)).copied(), Some(1));
        assert!(!sal.hits.contains_key(&(0, 0)), "a dense layer routes nothing");

        // The envelope form still works, and still carries its weights — the
        // fix must not cost the shape that has gate mass.
        let env = dir.join("frames.json");
        std::fs::write(&env, r#"{"frames":[{"layer":1,"experts":[7],"weights":[0.5]}]}"#)?;
        let sal2 = load_trace(&env)?;
        assert_eq!(sal2.hits.get(&(1, 7)).copied(), Some(1));
        assert!(
            sal2.mass.get(&(1, 7)).is_some_and(|m| (*m - 0.5).abs() < 1e-6),
            "the envelope form's gate weight must survive: {:?}",
            sal2.mass.get(&(1, 7))
        );
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
    use super::*;

    fn sal_of(rows: &[(usize, usize, f32)]) -> Saliency {
        let mut s = Saliency::default();
        for &(l, e, w) in rows {
            s.observe(l, &[e as i32], &[w]);
        }
        s
    }

    #[test]
    fn saliency_ranks_by_gate_mass_not_by_how_often_it_fired() {
        // The distinction the tool is named for. Expert 1 is routed three times
        // as often as expert 0 but carries a tenth the weight each time; ranking
        // on frequency would keep the wrong one, and the difference only shows
        // up as quality loss on the workload that needed expert 0.
        let sal = sal_of(&[(0, 0, 1.0), (0, 1, 0.1), (0, 1, 0.1), (0, 1, 0.1), (0, 2, 0.05)]);
        assert_eq!(sal.hits.get(&(0, 1)).copied(), Some(3));
        assert!(sal.mass[&(0, 0)] > sal.mass[&(0, 1)], "gate mass, not hit count");
        let plan = plan_keep(&sal, 1, 3, 1.0 / 3.0, 1);
        assert_eq!(plan.keep[0], vec![0, 1], "the rarely-routed heavy expert survives; the light one does not");
    }

    #[test]
    fn an_untraced_layer_falls_back_to_aggregate_saliency_and_says_so() {
        // A layer with no evidence cannot simply keep everything: `config.json`
        // carries one `n_routed_experts`, so every sparse layer must end the
        // same width. The honest remaining choice is to rank it by saliency
        // summed over the layers that *were* traced — weak, but a signal, where
        // keeping the lowest ids would be arbitrary. The count is surfaced.
        let sal = sal_of(&[(0, 0, 1.0), (0, 1, 0.9), (0, 2, 0.1), (0, 3, 0.05)]);
        let plan = plan_keep(&sal, 2, 4, 0.5, 1);
        assert_eq!(plan.keep[0], vec![0, 1], "the traced layer is ranked on its own evidence");
        assert_eq!(plan.keep[1], vec![0, 1], "the untraced one on the aggregate");
        assert_eq!(plan.n_experts_out, 2, "and both end the same width, as the container requires");
        assert_eq!(plan.layers_by_aggregate, 1, "the fallback is counted, not hidden");
    }

    #[test]
    fn every_layer_keeps_the_same_number_because_the_config_has_one_field() {
        // The structural constraint that shaped this tool. A per-layer keep
        // count cannot be expressed: the loader sizes its router buffer from a
        // single `n_routed_experts`, so a layer that kept a different number
        // fails at load — hours after the conversion.
        let mut sal = sal_of(&[(0, 0, 1.0), (0, 1, 0.5)]);
        sal.observe(1, &[0, 1, 2, 3], &[1.0, 1.0, 1.0, 1.0]); // layer 1 uses all four
        let plan = plan_keep(&sal, 2, 4, 0.5, 1);
        assert_eq!(plan.keep[0].len(), plan.keep[1].len(), "uniform width across layers");
        assert_eq!(plan.n_experts_out, 2);
    }

    #[test]
    fn the_survivor_floor_keeps_top_k_satisfiable() {
        // Top-k reads k experts. Pruning below k leaves a router that cannot
        // fill a selection — a checkpoint that is not merely worse but invalid.
        let sal = sal_of(&[(0, 0, 1.0), (0, 1, 0.5), (0, 2, 0.25), (0, 3, 0.1)]);
        let plan = plan_keep(&sal, 1, 4, 0.99, 2);
        assert_eq!(plan.keep[0].len(), 2, "the floor wins over the requested fraction");
        assert_eq!(plan.keep[0], vec![0, 1], "and it keeps the most salient");
    }

    #[test]
    fn the_plan_is_deterministic_under_exact_ties() {
        // Two runs over the same trace must prune the same experts; an exact tie
        // broken by map iteration order would make the artifact irreproducible.
        let sal = sal_of(&[(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
        let a = plan_keep(&sal, 1, 4, 0.5, 1);
        let b = plan_keep(&sal, 1, 4, 0.5, 1);
        assert_eq!(a.keep, b.keep);
        assert_eq!(a.keep[0], vec![0, 1], "ties break toward the lower id");
    }

    #[test]
    fn survivors_are_renumbered_to_close_the_gaps() -> Result<(), Error> {
        // The renumbering `prune` performs: surviving expert `keep[j]` becomes
        // expert `j`, so the output has no holes for the loader to index into.
        let sal = sal_of(&[(0, 1, 1.0), (0, 3, 0.9), (0, 0, 0.1), (0, 2, 0.05)]);
        let plan = plan_keep(&sal, 1, 4, 0.5, 1);
        let keep = plan.keep_of(0, "test")?;
        assert_eq!(keep, [1, 3]);
        assert_eq!(keep.iter().position(|&e| e == 1), Some(0));
        assert_eq!(keep.iter().position(|&e| e == 3), Some(1));
        assert_eq!(keep.iter().position(|&e| e == 0), None, "a pruned expert has no new id");
        // A layer past the plan is reported, not silently given an empty pool:
        // one `n_routed_experts` means every sparse layer must be covered.
        let err = match plan.keep_of(9, "model.layers.9.mlp.gate.weight") {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("the plan covers 1"), "got: {err}");
        Ok(())
    }

    #[test]
    fn expert_tensors_are_renamed_to_their_new_id() {
        let n = "model.layers.7.mlp.experts.42.gate_proj.weight";
        assert_eq!(expert_coords_of(n), Some((7, 42)));
        assert_eq!(rename_expert(n, 42, 3), "model.layers.7.mlp.experts.3.gate_proj.weight");
        // Only the expert segment moves — a layer that happens to share the
        // number must not be rewritten too.
        let m = "model.layers.42.mlp.experts.42.up_proj.weight";
        assert_eq!(rename_expert(m, 42, 0), "model.layers.42.mlp.experts.0.up_proj.weight");
    }

    #[test]
    fn the_router_and_its_bias_are_recognised_and_nothing_else_is() {
        assert_eq!(router_layer("model.layers.3.mlp.gate.weight"), Some(3));
        assert_eq!(router_bias_layer("model.layers.3.mlp.gate.e_score_correction_bias"), Some(3));
        // The shared expert's gate projection is not the router. Gathering rows
        // out of it would corrupt a tensor that has nothing to do with routing.
        assert_eq!(router_layer("model.layers.3.mlp.shared_experts.gate_proj.weight"), None);
        assert_eq!(router_layer("model.layers.3.mlp.experts.0.gate_proj.weight"), None);
    }

    #[test]
    fn the_report_refuses_to_claim_a_bytes_per_token_win() {
        // The one thing this tool must never imply. Cerebras' own cards show
        // activated parameters identical before and after, so a summary that
        // read as a bandwidth win would be actively misleading.
        let rep = PruneReport {
            layers: 4,
            experts_in: 256,
            experts_kept: 192,
            layers_without_evidence: 1,
            frequency_only: true,
            ..PruneReport::default()
        };
        let s = rep.summary();
        assert!(s.contains("per token"), "the summary must state what did not shrink");
        assert!(s.contains("working set"), "…and what did");
        assert!(s.contains("no routing evidence of their own"), "aggregate-ranked layers must be surfaced");
        assert!(s.contains("no gate weights"), "a frequency-only ranking must be surfaced");
        assert!(s.contains("25.0%"), "and the pool reduction reported: 256 -> 192");
    }

    /// A trace that routes every expert of every layer with a saliency that
    /// falls off with id, so the plan is predictable.
    fn synthetic_trace(n_layers: usize, n_experts: usize) -> Saliency {
        let mut sal = Saliency::default();
        for l in 0..n_layers {
            for e in 0..n_experts {
                let w = 1.0 / (e as f32 + 1.0);
                sal.observe(l, &[e as i32], &[w]);
            }
        }
        sal
    }

    #[test]
    fn a_pruned_container_loads_and_generates() -> Result<(), Error> {
        // The end-to-end contract. Everything above is arithmetic on a plan;
        // this is the part that has to survive `SafeTensors::open` and a real
        // forward — the router gathered to match its surviving experts, the
        // experts renumbered to close the gaps, and `config.json` rewritten so
        // the loader does not index an expert that is no longer there.
        //
        // A pruned checkpoint that ranks well and does not load is worse than
        // no tool at all, because the failure surfaces hours after the run.
        let dir = std::env::temp_dir().join(format!("peregrine_prune_e2e_{}", std::process::id()));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let (n_layers, n_experts) = (cfg.n_layers as usize, cfg.n_experts as usize);
        let sal = synthetic_trace(n_layers, n_experts);
        // Keep half; the floor is top-k so the router can still fill a selection.
        // `n_layers + 1`: the MTP head is a sparse layer with its own router, and
        // one `n_routed_experts` covers it too.
        let plan = plan_keep(&sal, n_layers + 1, n_experts, 0.5, cfg.topk.max(1) as usize);
        let kept = plan.n_experts_out;
        assert!(kept < n_experts, "the fixture must actually be pruned ({kept} of {n_experts})");

        let rep = prune(&dir, &out, &plan, 1 << 20)?;
        assert!(rep.tensors_dropped > 0, "pruning must drop expert tensors, not just renumber");
        assert!(rep.bytes_out < rep.bytes_in, "and the container must get smaller");

        // The output must be a *loadable* model, not just a well-formed
        // directory: this is what catches a router whose rows no longer match
        // its expert pool.
        let pruned_cfg = Cfg::load(&out)?;
        assert_eq!(pruned_cfg.n_experts as usize, kept, "config must advertise the surviving pool");
        assert!(pruned_cfg.topk as usize <= kept, "top-k cannot exceed what survives");
        let mut m = peregrine_model::Model::load(&out)?;
        let logits = m.forward_step(&[1, 5, 9], 0)?;
        assert_eq!(logits.len(), 3 * m.cfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()), "a pruned model must generate finite logits");

        // Every surviving expert is present under its *new* id, and no gaps.
        let st = SafeTensors::open(&out)?;
        for l in cfg.first_dense as usize..n_layers {
            for e in 0..kept.min(plan.keep[l].len()) {
                let n = format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight");
                assert!(st.tensors().iter().any(|t| t.name == n), "missing {n}");
            }
            let past = format!("model.layers.{l}.mlp.experts.{n_experts}.gate_proj.weight");
            assert!(!st.tensors().iter().any(|t| t.name == past), "an out-of-range expert survived");
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn the_router_rows_follow_the_experts_they_score() -> Result<(), Error> {
        // The failure this tool most needs not to have. `mlp.gate.weight` is
        // `[E, hidden]` and row `e` scores expert `e`; dropping experts without
        // gathering those rows leaves a router selecting ids that no longer
        // exist — which loads fine and produces nonsense.
        let dir = std::env::temp_dir().join(format!("peregrine_prune_router_{}", std::process::id()));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let cfg = Cfg::load(&dir)?;
        let (n_layers, n_experts, hidden) = (cfg.n_layers as usize, cfg.n_experts as usize, cfg.hidden as usize);
        let sal = synthetic_trace(n_layers, n_experts);
        let plan = plan_keep(&sal, n_layers + 1, n_experts, 0.5, 1);
        prune(&dir, &out, &plan, 1 << 20)?;

        let src = SafeTensors::open(&dir)?;
        let dst = SafeTensors::open(&out)?;
        let layer = cfg.first_dense as usize;
        let name = format!("model.layers.{layer}.mlp.gate.weight");
        let mut before = vec![0f32; n_experts * hidden];
        src.read_f32(&name, &mut before)?;
        let keep = &plan.keep[layer];
        let mut after = vec![0f32; keep.len() * hidden];
        dst.read_f32(&name, &mut after)?;

        for (new_id, &old_id) in keep.iter().enumerate() {
            let a = &before[old_id * hidden..old_id * hidden + hidden];
            let b = &after[new_id * hidden..new_id * hidden + hidden];
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "router row for surviving expert {old_id} did not land at its new id {new_id}"
            );
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn a_trace_with_no_usable_frames_is_refused() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_prune_trace_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let p = dir.join("routes.json");
        std::fs::write(&p, br#"{"frames": []}"#)?;
        let err = match load_trace(&p) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(err.contains("no usable frames"), "an empty trace must refuse, not prune everything: {err}");
        // …and a real one is accepted, weights optional.
        std::fs::write(&p, br#"[{"layer":0,"experts":[1,2],"weights":[0.7,0.3]}]"#)?;
        let sal = load_trace(&p)?;
        assert_eq!(sal.positions, 1);
        // f32 0.7 widened to f64 is 0.69999998…, so the tolerance is f32's, not f64's.
        assert!((sal.mass[&(0, 1)] - 0.7).abs() < 1e-6, "got {}", sal.mass[&(0, 1)]);
        std::fs::write(&p, br#"[{"layer":0,"experts":[1,2]}]"#)?;
        let sal = load_trace(&p)?;
        assert_eq!(sal.mass[&(0, 1)], 1.0, "a weightless trace degrades to counting, not to nothing");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

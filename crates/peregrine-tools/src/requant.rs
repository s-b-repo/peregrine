//! Requantize an existing container to a different on-disk precision.
//!
//! `todo.md` §13's conclusion is that the remaining throughput lever is reading
//! *fewer or smaller* experts: cross-token expert locality measures 0.6%, and
//! nine adaptive I/O knobs together measured 1.004× with byte-identical disk
//! reads. Two of that section's items — int2 expert storage and heat-tiered
//! precision — were blocked on nothing but a producer. int2 has been fully
//! consumable since M1 (detector, kernels, dequantizer, CUDA decode) and had
//! never been written by anything.
//!
//! At GLM-5.2 shapes int2 halves the dominant payload: 18.92 → 9.48 MB per
//! expert, 11.35 → 5.69 GB per token, 363 → 182 GB of working set.
//!
//! **This changes token values.** Requantizing is lossy by construction, so
//! unlike every knob in the engine it cannot be gated by bit-identity — measure
//! a converted container with `Model::prediction_flip_rate` against its source
//! before relying on it.
//!
//! Output carries a `__metadata__` provenance stamp (scheme, include filter,
//! source dir) so "which scheme is this container?" is answerable from the
//! artifact rather than from whoever ran the command. Deliberately **absent**
//! until it can be measured: a `flip_rate` key — stamping an unmeasured one
//! would be worse than none.
//!
//! Not yet built, and not claimed as built: an `--ablate` mode ranking schemes
//! on a small model before anyone spends hours on a large one. It needs
//! `Model::prediction_flip_rate`, which lives in `peregrine-model`; wiring that
//! in would drag the whole inference stack into a batch tool that deliberately
//! links `peregrine-core` alone, so it belongs in the engine binary instead.
//!
//! **Memory: peak is roughly `shard_bytes` plus one dense tensor.** An earlier
//! version of this comment claimed "peak RAM is a row, not a checkpoint", which
//! was wrong in two ways — [`requantize`] materialises a whole `[o, i]` f32 to
//! quantize from, and [`ShardWriter`] buffers finished pieces until the shard
//! budget is reached, because safetensors puts its header before the payload and
//! the offsets must be known first. At the 5 GB default that is 5 GB resident.
//! **Lower `--shard-bytes` on a small machine**; the output is a directory of
//! shards either way, so shard size is a memory knob, not a format choice.

use peregrine_core::pack::{quant_i2, quant_i2_g64, quant_i3_g64, quant_i4, quant_i4_grouped, quant_i8, QtView};
use peregrine_core::config::Cfg;
use peregrine_core::qt::QtInfo;
use peregrine_core::safetensors::SafeTensors;
use peregrine_core::{Context, Error};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Target precision for the tensors a plan selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Int8,
    Int4,
    Int4Grouped(usize),
    Int2,
    /// colibrì's fmt 5: int3 with a scale per 64 values, 3.5 bits/weight.
    Int3G64,
    /// Affine int2 with a scale *and* zero-point per 64 values — 2.06
    /// bits/weight. Prefer this over [`Target::Int2`]: per-row int2's `amax / 1`
    /// convention leaves one of its four levels unreachable, so it is
    /// effectively ternary, and its scales are per row rather than per group.
    Int2G64,
}

impl Target {
    pub fn parse(s: &str) -> Option<Target> {
        match s {
            "int8" => Some(Target::Int8),
            "int4" => Some(Target::Int4),
            "int2" => Some(Target::Int2),
            "int3-g64" => Some(Target::Int3G64),
            "int2-g64" => Some(Target::Int2G64),
            _ => s
                .strip_prefix("int4-g")
                .and_then(|g| g.parse::<usize>().ok())
                .filter(|g| *g > 0 && g.is_multiple_of(16))
                .map(Target::Int4Grouped),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Target::Int8 => "int8".into(),
            Target::Int4 => "int4".into(),
            Target::Int2 => "int2".into(),
            Target::Int3G64 => "int3-g64".into(),
            Target::Int2G64 => "int2-g64".into(),
            Target::Int4Grouped(g) => format!("int4-g{g}"),
        }
    }

    /// Packed payload bytes for a `[o, i]` weight in this format.
    pub fn payload_bytes(&self, o: usize, i: usize) -> usize {
        match self {
            Target::Int8 => o * i,
            Target::Int4 | Target::Int4Grouped(_) => o * i.div_ceil(2),
            Target::Int2 => o * i.div_ceil(4),
            Target::Int3G64 => {
                o * i.div_ceil(peregrine_core::pack::I3_GROUP) * peregrine_core::pack::I3_GROUP_BYTES
            }
            Target::Int2G64 => {
                o * i.div_ceil(peregrine_core::pack::I2G_GROUP) * peregrine_core::pack::I2G_GROUP_BYTES
            }
        }
    }

    /// Number of f32 scales for a `[o, i]` weight in this format.
    pub fn scale_count(&self, o: usize, i: usize) -> usize {
        match self {
            Target::Int4Grouped(g) => o * i.div_ceil(*g),
            Target::Int3G64 => o * i.div_ceil(peregrine_core::pack::I3_GROUP),
            // Two f32 per group: scale and zero-point, interleaved.
            Target::Int2G64 => {
                o * i.div_ceil(peregrine_core::pack::I2G_GROUP) * peregrine_core::pack::I2G_SCALES_PER_GROUP
            }
            _ => o,
        }
    }

    fn quantize(&self, w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<f32>) {
        match self {
            Target::Int8 => quant_i8(w, o, i),
            Target::Int4 => quant_i4(w, o, i),
            Target::Int2 => quant_i2(w, o, i),
            Target::Int3G64 => quant_i3_g64(w, o, i),
            Target::Int2G64 => quant_i2_g64(w, o, i),
            Target::Int4Grouped(g) => quant_i4_grouped(w, o, i, *g),
        }
    }
}

/// Per-expert precision chosen from measured routing heat.
///
/// **The trade is not the obvious one.** Tiering saves *disk* in proportion to
/// expert **count**, but saves *per-token bytes* in proportion to routing
/// **frequency** — and frequency is exactly what makes an expert hot. Keeping
/// the hottest 25% at int4 costs 11.84 MB/expert/token under uniform routing
/// (−37%), but 15.14 MB (−20%) if that quarter carries 60% of routings. Skew
/// buys accuracy and costs bytes.
///
/// So **measure the skew before building a tiered checkpoint**. With the 0.6%
/// cross-token locality this engine measures, near-uniform is plausible — in
/// which case tiering degenerates to "int2 everything with an accuracy hedge",
/// which is a reasonable product but should be described honestly.
#[derive(Debug, Clone)]
pub struct HeatTier {
    /// Fraction of experts (per layer, by heat rank) kept at `hot`.
    pub hot_frac: f64,
    pub hot: Target,
    pub cold: Target,
    /// `heat[layer * n_experts + expert]`, as persisted in `route_stats.json`.
    pub heat: Vec<u32>,
    pub n_experts: usize,
}

impl HeatTier {
    /// Load heat from a `route_stats.json` beside the model.
    ///
    /// Deliberately no fingerprint check here: the caller already has the config
    /// this container was built for, and the array length is validated against
    /// it, which is the property that matters (a wrong-length array would
    /// misalign every layer).
    pub fn from_route_stats(dir: &Path, n_layers: usize, n_experts: usize) -> Result<Vec<u32>, Error> {
        let path = dir.join("route_stats.json");
        let bytes = std::fs::read(&path).ctx(|| format!("read {}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| Error::Format(format!("route_stats.json: {e}")))?;
        let arr = v
            .get("heat")
            .and_then(|h| h.as_array())
            .ok_or_else(|| Error::Format("route_stats.json has no `heat` array".into()))?;
        let mut heat: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
        let want = n_layers * n_experts;
        // The producer is one row longer than the consumer wants. `HeatTable` is
        // built `n_layers + 1` rows (`model.rs`) because the MTP head sits at
        // layer index `n_layers` and routes a full set of experts — a 2026-08-09
        // fix for that head never accumulating heat. The extra row is real data,
        // but it is not a layer this container requantizes, so drop it.
        //
        // Demanding an exact `n_layers * n_experts` made the two permanently
        // irreconcilable: at GLM-5.2 shapes the table is 79 x 256 = 20 224 and
        // this asked for 78 x 256 = 19 968, so `--tier-hot-frac` refused every
        // heat file the engine could produce, however the run was done.
        if heat.len() == want + n_experts {
            heat.truncate(want);
        }
        if heat.len() != want {
            return Err(Error::Format(format!(
                "route_stats.json heat is {} entries, expected {want} (n_layers {n_layers} x \
                 n_experts {n_experts}, or one row more for the MTP head) — a mismatched trace \
                 would misalign every layer",
                heat.len(),
            )));
        }
        Ok(heat)
    }

    /// Target for one expert, by heat rank within its own layer.
    fn target_for(&self, layer: usize, expert: usize) -> Target {
        let base = layer * self.n_experts;
        let Some(row) = self.heat.get(base..base + self.n_experts) else {
            return self.cold;
        };
        let mine = row.get(expert).copied().unwrap_or(0);
        // Rank by "how many are strictly hotter"; ties fall to the cold side, so
        // an all-zero (untraced) layer tiers entirely cold rather than
        // arbitrarily promoting the low-numbered experts.
        let hotter = row.iter().filter(|&&h| h > mine).count();
        let keep = ((self.n_experts as f64) * self.hot_frac).round() as usize;
        if mine > 0 && hotter < keep {
            self.hot
        } else {
            self.cold
        }
    }

    /// Bytes-per-token ratio against an all-`cold` checkpoint, given this heat
    /// distribution — the number that says whether tiering is worth it.
    pub fn weighted_ratio(&self, o: usize, i: usize) -> f64 {
        let (mut hot_w, mut total_w) = (0f64, 0f64);
        for (idx, &h) in self.heat.iter().enumerate() {
            let (layer, expert) = (idx / self.n_experts, idx % self.n_experts);
            let w = h as f64;
            total_w += w;
            if matches!(self.target_for(layer, expert), t if t == self.hot) {
                hot_w += w;
            }
        }
        if total_w == 0.0 {
            return 1.0;
        }
        let (hb, cb) = (self.hot.payload_bytes(o, i) as f64, self.cold.payload_bytes(o, i) as f64);
        let share = hot_w / total_w;
        (share * hb + (1.0 - share) * cb) / cb
    }
}

/// What to convert and how.
#[derive(Debug, Clone)]
pub struct Plan {
    pub target: Target,
    /// Optional heat-tiered precision; when set it overrides `target` per expert.
    pub tier: Option<HeatTier>,
    /// Only tensors whose name contains this substring are requantized.
    /// Defaults to `.mlp.experts.` — the routed experts are 97% of the bytes,
    /// and colibrì's own ablation found that leaving the head and dense path at
    /// their original precision recovers accuracy for essentially no bytes.
    pub include: String,
    /// Roll to a new output shard once it reaches this many bytes.
    pub shard_bytes: u64,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { target: Target::Int2, tier: None, include: ".mlp.experts.".into(), shard_bytes: 5_000_000_000 }
    }
}

/// One tensor's contribution to an output shard: its header entry plus the bytes
/// to stream. Held one at a time — never a whole shard.
struct Piece {
    name: String,
    dtype: &'static str,
    shape: Vec<i64>,
    bytes: Vec<u8>,
}

/// Name of the resume manifest written beside the output shards.
const PROGRESS_FILE: &str = ".requant-progress.json";

/// Identity of a conversion, so a resumed run cannot silently mix schemes.
///
/// colibrì's converter learned this the hard way and guards it the same way
/// (`convert_fp8_to_int4.py:350`): re-running with different flags over a
/// half-finished directory interleaves two bit-widths in one container, which
/// then loads without complaint and computes garbage.
fn params_fingerprint(plan: &Plan) -> String {
    format!("target={} include={} shard_bytes={}", plan.target.label(), plan.include, plan.shard_bytes)
}

/// How far a previous run got. Recorded only at shard boundaries, because a
/// shard is the unit that is atomically complete — it is fsynced and renamed
/// into place, so one that exists is whole.
#[derive(Debug, Clone, Copy, Default)]
struct Progress {
    tensors_done: usize,
    shards: usize,
}

fn read_progress(dir: &Path, plan: &Plan) -> Result<Progress, Error> {
    let path = dir.join(PROGRESS_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Progress::default()),
        Err(e) => return Err(Error::Io(e)),
    };
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::Format(format!("{PROGRESS_FILE}: {e}")))?;
    let want = params_fingerprint(plan);
    let got = v.get("params").and_then(|p| p.as_str()).unwrap_or("");
    if got != want {
        return Err(Error::Format(format!(
            "{} records an unfinished conversion with different settings:\n  \
             existing: {got}\n  requested: {want}\n\
             Resuming would interleave two schemes in one container, which loads without \
             complaint and computes garbage. Finish that run with its original settings, or \
             delete the output directory and start over.",
            dir.join(PROGRESS_FILE).display()
        )));
    }
    Ok(Progress {
        tensors_done: v.get("tensors_done").and_then(|n| n.as_u64()).unwrap_or(0) as usize,
        shards: v.get("shards").and_then(|n| n.as_u64()).unwrap_or(0) as usize,
    })
}

fn write_progress(dir: &Path, plan: &Plan, p: Progress) -> Result<(), Error> {
    let v = serde_json::json!({
        "version": 1,
        "params": params_fingerprint(plan),
        "tensors_done": p.tensors_done,
        "shards": p.shards,
    });
    let bytes = serde_json::to_vec_pretty(&v).map_err(|e| Error::Format(format!("progress: {e}")))?;
    peregrine_core::durable::write_atomic(&dir.join(PROGRESS_FILE), &bytes)
}

/// Accumulates pieces and writes them as safetensors shards.
///
/// safetensors puts its header first, so sizes must be known before any payload
/// is written. Pieces are therefore buffered per shard and flushed when the
/// shard's byte budget is reached — bounded by `shard_bytes`, not by the
/// checkpoint. Each shard is fsynced then atomically renamed, so a shard that
/// exists is complete; [`requantize`] records progress at those boundaries and
/// skips them on a re-run.
pub struct ShardWriter {
    dir: PathBuf,
    prefix: String,
    shard_bytes: u64,
    pending: Vec<Piece>,
    pending_bytes: u64,
    /// Next shard number to write. Set past prior shards when resuming.
    index: usize,
    /// `__metadata__` entries stamped into every shard header. Each shard is
    /// self-describing, so a stray one is still identifiable.
    meta: Vec<(String, String)>,
    pub written: Vec<PathBuf>,
}

impl ShardWriter {
    pub fn new(dir: &Path, prefix: &str, shard_bytes: u64) -> ShardWriter {
        ShardWriter {
            dir: dir.to_path_buf(),
            prefix: prefix.to_string(),
            shard_bytes: shard_bytes.max(1),
            pending: Vec::new(),
            pending_bytes: 0,
            index: 0,
            meta: Vec::new(),
            written: Vec::new(),
        }
    }

    /// Stamp provenance into every shard this writer produces.
    ///
    /// A requantized container changes token values by existing, so "which
    /// scheme is this?" must be answerable from the artifact rather than from
    /// whoever ran the command. `SafeTensors::open` skips `__metadata__`, so
    /// this is additive and older readers are unaffected.
    pub fn with_metadata(mut self, meta: Vec<(String, String)>) -> ShardWriter {
        self.meta = meta;
        self
    }

    /// Shard file this index would produce — also the resume probe.
    pub fn shard_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}-{index:05}.safetensors", self.prefix))
    }

    pub fn push(&mut self, name: &str, dtype: &'static str, shape: Vec<i64>, bytes: Vec<u8>) -> Result<(), Error> {
        self.pending_bytes += bytes.len() as u64;
        self.pending.push(Piece { name: name.to_string(), dtype, shape, bytes });
        if self.pending_bytes >= self.shard_bytes {
            self.flush()?;
        }
        Ok(())
    }

    /// Write the buffered pieces as one shard. A no-op when nothing is pending,
    /// so calling it at the end is always safe.
    pub fn flush(&mut self) -> Result<(), Error> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let path = self.shard_path(self.index);
        let mut header = serde_json::Map::new();
        if !self.meta.is_empty() {
            let mut m = serde_json::Map::new();
            for (k, v) in &self.meta {
                m.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            header.insert("__metadata__".into(), serde_json::Value::Object(m));
        }
        let mut cursor: i64 = 0;
        for p in &self.pending {
            let start = cursor;
            let end = start + p.bytes.len() as i64;
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), serde_json::Value::String(p.dtype.to_string()));
            entry.insert("shape".into(), serde_json::json!(p.shape));
            entry.insert("data_offsets".into(), serde_json::json!([start, end]));
            header.insert(p.name.clone(), serde_json::Value::Object(entry));
            cursor = end;
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header))
            .map_err(|e| Error::Format(format!("serialize shard header: {e}")))?;
        std::fs::create_dir_all(&self.dir).ctx(|| format!("create {}", self.dir.display()))?;
        let tmp = path.with_extension("safetensors.part");
        {
            let f = std::fs::File::create(&tmp).ctx(|| format!("create {}", tmp.display()))?;
            let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
            w.write_all(&(hdr.len() as u64).to_le_bytes()).ctx(|| "write header length".to_string())?;
            w.write_all(&hdr).ctx(|| "write header".to_string())?;
            for p in &self.pending {
                w.write_all(&p.bytes).ctx(|| format!("write payload for '{}'", p.name))?;
            }
            let f = w.into_inner().map_err(|e| Error::Format(format!("flush shard: {e}")))?;
            // Durability before the rename: a shard that exists must be complete,
            // because the resume path treats existence as "already done".
            f.sync_all().ctx(|| format!("fsync {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path).ctx(|| format!("commit {}", path.display()))?;
        self.written.push(path);
        self.pending.clear();
        self.pending_bytes = 0;
        self.index += 1;
        Ok(())
    }
}

/// What a conversion did, for the caller to report.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub tensors_total: usize,
    pub tensors_requantized: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub shards: usize,
    pub skipped_unsupported: Vec<String>,
    /// Model-directory sidecars copied alongside the shards. Without at least
    /// `config.json` the output is not loadable.
    pub sidecars: Vec<String>,
}

impl Report {
    pub fn ratio(&self) -> f64 {
        if self.bytes_in == 0 {
            return 1.0;
        }
        self.bytes_out as f64 / self.bytes_in as f64
    }
}

/// Predict a conversion's output size from headers alone — no payload is read
/// and nothing is written.
///
/// Worth having because the alternative is discovering the answer several hours
/// and a few hundred gigabytes into a run: the GLM-5.2 container is 349 GB in
/// and ~182 GB out, and both have to fit at once. The sizes are exactly derivable
/// from the target format and the logical `[O, I]`, which is what
/// [`Target::payload_bytes`] and [`Target::scale_count`] compute.
pub fn plan_sizes(indir: &Path, plan: &Plan) -> Result<Report, Error> {
    let cfg = Cfg::load(indir)?;
    let st = SafeTensors::open(indir)?;
    let mut rep = Report::default();
    for t in st.tensors() {
        rep.tensors_total += 1;
        let nbytes = st.uncompressed_nbytes(&t.name).unwrap_or(0).max(0) as u64;
        rep.bytes_in += nbytes;
        if t.name.ends_with(".qs") {
            continue;
        }
        match (t.name.contains(&plan.include), expert_dims(&t.name, &cfg)) {
            (true, Some((o, i))) => {
                rep.tensors_requantized += 1;
                // Mirror `requantize`'s per-expert choice exactly (see the
                // `expert_coords` match there): a tier overrides the uniform
                // target per expert, and sizing every expert at `plan.target`
                // ignored that. `--dry-run --tier-hot-frac` therefore reported
                // the all-cold size whatever the fraction — identical output for
                // every `--tier-hot-frac`, and a "plan for N GB of free space"
                // line that under-states a tiered run by the whole difference
                // between hot and cold for the experts kept hot.
                let target = match (&plan.tier, expert_coords(&t.name)) {
                    (Some(tr), Some((layer, expert))) => tr.target_for(layer, expert),
                    _ => plan.target,
                };
                rep.bytes_out += (target.payload_bytes(o, i) + target.scale_count(o, i) * 4) as u64;
            }
            _ => {
                rep.bytes_out += nbytes;
                let qs = format!("{}.qs", t.name);
                rep.bytes_out += st.uncompressed_nbytes(&qs).unwrap_or(0).max(0) as u64;
            }
        }
    }
    rep.shards = (rep.bytes_out / plan.shard_bytes.max(1) + 1) as usize;
    Ok(rep)
}

/// Requantize every tensor in `indir` matching `plan.include`, copying the rest
/// through unchanged, into `outdir`.
pub fn requantize(indir: &Path, outdir: &Path, plan: &Plan) -> Result<Report, Error> {
    let cfg = Cfg::load(indir)?;
    let st = SafeTensors::open(indir)?;
    check_writable(plan, &cfg)?;
    std::fs::create_dir_all(outdir).ctx(|| format!("create {}", outdir.display()))?;

    // Resume. A shard is fsynced then atomically renamed, so one that exists is
    // complete; the manifest records how many input tensors those shards cover.
    // Without this a re-run restarted from shard 0 while the previous run's
    // higher-numbered shards survived, leaving the same tensor name in two files
    // — which `SafeTensors::open` rejects outright, so an interrupted conversion
    // produced a directory that could not be opened at all.
    let resume = read_progress(outdir, plan)?;
    if resume.tensors_done > 0 {
        eprintln!(
            "peregrine-requantize: resuming — {} tensors already written across {} shard(s)",
            resume.tensors_done, resume.shards
        );
    }
    let scheme = match &plan.tier {
        Some(t) => format!("heat-tiered hot={} cold={} hot_frac={}", t.hot.label(), t.cold.label(), t.hot_frac),
        None => plan.target.label(),
    };
    let mut w = ShardWriter::new(outdir, "out", plan.shard_bytes).with_metadata(vec![
        ("peregrine.requantize.scheme".into(), scheme),
        ("peregrine.requantize.include".into(), plan.include.clone()),
        ("peregrine.requantize.source".into(), indir.display().to_string()),
        // Deliberately absent until measured: a `flip_rate` key. Stamping an
        // unmeasured one would be worse than none.
    ]);
    w.index = resume.shards;
    let mut rep = Report::default();
    let meta: Vec<(String, Vec<i64>, String)> = st
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone(), format!("{:?}", t.dtype).to_uppercase()))
        .collect();

    let mut shards_seen = 0usize;
    for (idx, (name, shape, dtype)) in meta.iter().enumerate() {
        let name = name.as_str();
        rep.tensors_total += 1;
        if idx < resume.tensors_done {
            continue;
        }
        let nbytes = st.uncompressed_nbytes(name).unwrap_or(0).max(0) as usize;
        rep.bytes_in += nbytes as u64;

        // Scales travel with their weight; a `.qs` sibling is emitted by the arm
        // that rewrites its weight, never copied on its own.
        if name.ends_with(".qs") {
            continue;
        }
        let selected = name.contains(&plan.include);
        // The header's `shape` is the *packed* byte shape (`[rows, bytes_per_row]`),
        // not logical `[O, I]` — and logical `I` cannot be recovered from the
        // payload alone, since the same bytes are a valid int8, int4 or int2
        // weight of different widths. The engine resolves this from `config.json`
        // and so must this: guessing is how a container gets silently misdecoded.
        let two_d = expert_dims(name, &cfg);

        if let (true, Some((o, i))) = (selected, two_d) {
            let info = QtInfo::detect(&st, name, o as i64, i as i64);
            match QtView::row_bytes(info.fmt, i) {
                Some(rb) => {
                    let mut q = vec![0u8; o * rb];
                    st.read_raw(name, &mut q)?;
                    let mut scale = vec![0f32; info.scale_count as usize];
                    st.read_f32(&format!("{name}.qs"), &mut scale)?;
                    let view = QtView { fmt: info.fmt, o, i, gs: info.gs as usize, q: &q, scale: &scale };

                    // Dequantize row-by-row into a reusable scratch: peak RAM is
                    // one row plus the two packed copies, never a dense [o, i].
                    let mut dense = vec![0f32; o * i];
                    for r in 0..o {
                        view.dequant_row_into(r, &mut dense[r * i..(r + 1) * i]);
                    }
                    // A tier overrides the uniform target per expert; anything
                    // whose coordinates cannot be parsed falls back to the
                    // uniform target rather than being silently mis-tiered.
                    let target = match (&plan.tier, expert_coords(name)) {
                        (Some(t), Some((layer, expert))) => t.target_for(layer, expert),
                        _ => plan.target,
                    };
                    let (nq, nsc) = target.quantize(&dense, o, i);
                    rep.bytes_out += (nq.len() + nsc.len() * 4) as u64;
                    rep.tensors_requantized += 1;
                    let out_rb = (nq.len() / o.max(1)) as i64;
                    w.push(name, "U8", vec![o as i64, out_rb], nq)?;
                    let sc_bytes: Vec<u8> = nsc.iter().flat_map(|v| v.to_le_bytes()).collect();
                    w.push(&format!("{name}.qs"), "F32", vec![nsc.len() as i64], sc_bytes)?;
                    continue;
                }
                None => {
                    // F32/Unknown: no packed row form to decode. Copy through and
                    // say so rather than guessing at the bytes.
                    rep.skipped_unsupported.push(name.to_string());
                }
            }
        }

        // Pass-through: byte-identical copy, including this tensor's `.qs` if it
        // has one, so untouched weights survive the rewrite exactly.
        let mut raw = vec![0u8; nbytes];
        st.read_raw(name, &mut raw)?;
        rep.bytes_out += raw.len() as u64;
        w.push(name, static_dtype(dtype), shape.clone(), raw)?;
        let qs = format!("{name}.qs");
        if let Some(qs_info) = st.tensors().iter().find(|t| t.name == qs) {
            let n = st.uncompressed_nbytes(&qs).unwrap_or(0).max(0) as usize;
            let qs_shape = qs_info.shape.clone();
            // Keep the *source* dtype. Hardcoding "F32" over a BF16 scale tensor
            // declares 4n bytes for a 2n-byte payload, and `SafeTensors::open`
            // then rejects the entire output directory — from a tensor this tool
            // only copied.
            let qs_dtype = static_dtype(&format!("{:?}", qs_info.dtype).to_uppercase());
            let mut raw = vec![0u8; n];
            st.read_raw(&qs, &mut raw)?;
            rep.bytes_out += raw.len() as u64;
            w.push(&qs, qs_dtype, qs_shape, raw)?;
        }

        // Record only at shard boundaries: a shard is the unit that is atomically
        // complete, so `tensors_done` is exactly what the shards on disk cover.
        if w.written.len() != shards_seen {
            shards_seen = w.written.len();
            write_progress(outdir, plan, Progress { tensors_done: idx + 1, shards: w.index })?;
        }
    }
    w.flush()?;
    write_progress(outdir, plan, Progress { tensors_done: meta.len(), shards: w.index })?;

    // A directory of shards is not a model. `Cfg::load` needs `config.json`, and
    // serving needs `tokenizer.json`; without them the output cannot be loaded at
    // all, and hand-copying a *mismatched* config silently decodes every expert
    // wrong, since `expert_dims` resolves logical [O, I] from it.
    for side in ["config.json", "tokenizer.json", "generation_config.json", "tokenizer_config.json"] {
        let src = indir.join(side);
        if src.exists() {
            std::fs::copy(&src, outdir.join(side)).ctx(|| format!("copy {side}"))?;
            rep.sidecars.push(side.to_string());
        }
    }
    rep.shards = w.index;
    Ok(rep)
}

/// Refuse a plan that would write a container the loader cannot detect back.
///
/// int2-g64 needs **at least two 64-value groups** per row to be distinguishable:
/// at one group its `(bytes, scales)` pair collides with grouped int4, and
/// `QtInfo::detect` would resolve the result as some other format and decode
/// garbage that loads without complaint. That is the exact failure the
/// detector's "deliberately not assume int2" comment exists to avoid, so the
/// converter must not manufacture it.
///
/// Checked once against the config's expert dims rather than per tensor: every
/// routed projection shares those two widths, so one check covers the run and an
/// operator learns before the hours are spent, not after.
fn check_writable(plan: &Plan, cfg: &Cfg) -> Result<(), Error> {
    let targets = [Some(plan.target), plan.tier.as_ref().map(|t| t.hot), plan.tier.as_ref().map(|t| t.cold)];
    if !targets.iter().flatten().any(|t| *t == Target::Int2G64) {
        return Ok(());
    }
    // gate/up are [moe_inter, hidden] and down is [hidden, moe_inter], so the
    // narrowest input width across the three is `min(hidden, moe_inter)`.
    let narrowest = (cfg.hidden as usize).min(cfg.moe_inter as usize);
    let groups = narrowest.div_ceil(peregrine_core::pack::I2G_GROUP);
    if groups < 2 {
        return Err(Error::Format(format!(
            "int2-g64 needs at least two {}-value groups per row, but this model's narrowest \
             expert projection is {narrowest} wide ({groups} group). A single-group container \
             is indistinguishable from grouped int4 on disk and would load as the wrong format. \
             Use int3-g64 or int4-g{} instead.",
            peregrine_core::pack::I2G_GROUP,
            peregrine_core::pack::I2G_GROUP,
        )));
    }
    Ok(())
}

/// Logical `[O, I]` of a routed-expert projection, from the config — the only
/// place those dims exist. `None` for anything that is not one of the three
/// expert projections, which is then copied through untouched.
fn expert_dims(name: &str, cfg: &Cfg) -> Option<(usize, usize)> {
    if !name.contains(".mlp.experts.") {
        return None;
    }
    let (h, m) = (cfg.hidden as usize, cfg.moe_inter as usize);
    if h == 0 || m == 0 {
        return None;
    }
    match name {
        n if n.ends_with(".gate_proj.weight") || n.ends_with(".up_proj.weight") => Some((m, h)),
        n if n.ends_with(".down_proj.weight") => Some((h, m)),
        _ => None,
    }
}

/// `(layer, expert)` from a routed-expert tensor name, e.g.
/// `model.layers.7.mlp.experts.42.gate_proj.weight` -> `(7, 42)`.
fn expert_coords(name: &str) -> Option<(usize, usize)> {
    let layer = name.split("model.layers.").nth(1)?.split('.').next()?.parse().ok()?;
    let expert = name.split(".mlp.experts.").nth(1)?.split('.').next()?.parse().ok()?;
    Some((layer, expert))
}

/// safetensors dtype tags the writer emits. Anything unrecognized is carried as
/// `U8`, which is how packed payloads are already stored.
fn static_dtype(d: &str) -> &'static str {
    match d {
        "F32" => "F32",
        "F16" => "F16",
        "BF16" => "BF16",
        "I8" => "I8",
        _ => "U8",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dirs(tag: &str) -> Result<(PathBuf, PathBuf), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_rq_{}_{}", std::process::id(), tag));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        Ok((dir, out))
    }

    #[test]
    fn target_parsing_accepts_the_documented_schemes() {
        assert_eq!(Target::parse("int2"), Some(Target::Int2));
        assert_eq!(Target::parse("int4"), Some(Target::Int4));
        assert_eq!(Target::parse("int8"), Some(Target::Int8));
        assert_eq!(Target::parse("int4-g64"), Some(Target::Int4Grouped(64)));
        assert_eq!(Target::parse("int4-g63"), None, "group size must be a multiple of 16");
        assert_eq!(Target::parse("int3-g64"), Some(Target::Int3G64));
        assert_eq!(Target::parse("int3"), None, "the group size is part of the format name");
        assert_eq!(Target::parse("nonsense"), None);
    }

    #[test]
    fn int2_is_half_of_int4_and_a_quarter_of_int8() {
        let (o, i) = (2048usize, 6144usize);
        assert_eq!(Target::Int4.payload_bytes(o, i), o * i / 2);
        assert_eq!(Target::Int2.payload_bytes(o, i), o * i / 4);
        assert_eq!(Target::Int8.payload_bytes(o, i), o * i);
        // The headline: an expert projection at GLM-5.2 shapes.
        assert_eq!(Target::Int4.payload_bytes(o, i), 6_291_456);
        assert_eq!(Target::Int2.payload_bytes(o, i), 3_145_728);
    }

    #[test]
    fn grouped_scales_are_per_group_per_row() {
        assert_eq!(Target::Int4Grouped(64).scale_count(8, 128), 8 * 2);
        assert_eq!(Target::Int4.scale_count(8, 128), 8);
    }

    #[test]
    fn expert_dims_come_from_the_config_not_the_header() -> Result<(), Error> {
        // The header's `shape` is the packed byte shape, and logical `I` cannot
        // be recovered from a payload (the same bytes are a valid int8, int4 or
        // int2 weight of different widths). Getting this wrong silently decodes
        // half a row — which is exactly what the first draft did.
        let cfg = Cfg::from_json(&serde_json::json!({
            "hidden_size": 6144, "num_hidden_layers": 4, "num_attention_heads": 8,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 2048,
            "intermediate_size": 4096, "first_k_dense_replace": 1, "n_shared_experts": 1,
            "vocab_size": 256, "q_lora_rank": 64, "kv_lora_rank": 32,
            "qk_nope_head_dim": 24, "qk_rope_head_dim": 8, "v_head_dim": 32,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5, "eos_token_id": [1],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}
        }))?;
        let p = "model.layers.4.mlp.experts.7.";
        assert_eq!(expert_dims(&format!("{p}gate_proj.weight"), &cfg), Some((2048, 6144)));
        assert_eq!(expert_dims(&format!("{p}up_proj.weight"), &cfg), Some((2048, 6144)));
        assert_eq!(expert_dims(&format!("{p}down_proj.weight"), &cfg), Some((6144, 2048)));
        assert_eq!(expert_dims("model.layers.4.self_attn.o_proj.weight", &cfg), None, "not an expert");
        assert_eq!(expert_dims("model.layers.4.mlp.shared_experts.up_proj.weight", &cfg), None);
        Ok(())
    }

    #[test]
    fn writer_round_trips_pieces_into_readable_shards() -> Result<(), Error> {
        // The writer builds its own safetensors header, so a mis-sized offset
        // would corrupt every following tensor. Read the shard back with the
        // engine's own reader rather than trusting the bytes we just wrote.
        let dir = std::env::temp_dir().join(format!("peregrine_requant_w_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture must be removable: {e}");
        }
        let mut w = ShardWriter::new(&dir, "out", 1 << 30);
        let a: Vec<u8> = (0..=255u8).collect();
        let b: Vec<u8> = (0..64u8).map(|v| v.wrapping_mul(7)).collect();
        w.push("alpha", "U8", vec![16, 16], a.clone())?;
        w.push("beta", "U8", vec![8, 8], b.clone())?;
        w.flush()?;
        assert_eq!(w.written.len(), 1, "both pieces fit one shard");

        let st = SafeTensors::open(&dir)?;
        let mut got_a = vec![0u8; a.len()];
        st.read_raw("alpha", &mut got_a)?;
        let mut got_b = vec![0u8; b.len()];
        st.read_raw("beta", &mut got_b)?;
        assert_eq!(got_a, a, "first tensor round-trips");
        assert_eq!(got_b, b, "second tensor round-trips at its offset");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    fn tier_fixture(hot_frac: f64) -> HeatTier {
        // 3 layers x 4 experts, decaying heat: 100, 75, 50, 25 per layer.
        let n_experts = 4;
        let heat: Vec<u32> = (0..3).flat_map(|_| (0..4).map(|e| 100 - e * 25)).collect();
        HeatTier { hot_frac, hot: Target::Int4, cold: Target::Int2, heat, n_experts }
    }

    #[test]
    fn heat_tiering_promotes_by_rank_within_a_layer() {
        let t = tier_fixture(0.25); // keep 1 of 4
        assert_eq!(t.target_for(0, 0), Target::Int4, "hottest expert kept at int4");
        for e in 1..4 {
            assert_eq!(t.target_for(0, e), Target::Int2, "expert {e} is not in the hot quarter");
        }
        // Ranking is per layer, not global — layer 2's hottest is also promoted.
        assert_eq!(t.target_for(2, 0), Target::Int4);
    }

    #[test]
    fn an_untraced_layer_tiers_entirely_cold() {
        // All-zero heat means "no evidence", and promoting on no evidence would
        // silently keep whichever experts happen to sort first. Ties go cold.
        let mut t = tier_fixture(0.5);
        t.heat = vec![0; 12];
        for e in 0..4 {
            assert_eq!(t.target_for(0, e), Target::Int2, "expert {e} has no heat to justify int4");
        }
    }

    #[test]
    fn weighted_ratio_measures_routing_mass_not_expert_count() {
        // The trade that is easy to get backwards: tiering saves disk by expert
        // *count* but per-token bytes by routing *frequency*. Here the hot 25%
        // carries 40% of the mass (100 of 250), so keeping it at int4 costs far
        // more than "a quarter of the experts" suggests.
        let t = tier_fixture(0.25);
        let (o, i) = (2048usize, 6144usize);
        let ratio = t.weighted_ratio(o, i);
        assert!(ratio > 1.3 && ratio < 1.5, "expected ~1.4x an all-int2 checkpoint, got {ratio}");
        // Sanity: promoting nothing is exactly the cold checkpoint.
        assert!((tier_fixture(0.0).weighted_ratio(o, i) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn output_is_a_loadable_model_directory() -> Result<(), Error> {
        // Shards alone are not a model: `Cfg::load` needs `config.json`. Before
        // this, every converted container had to be hand-completed with a `cp`,
        // and copying a *mismatched* config silently decodes every expert wrong
        // because `expert_dims` resolves logical [O, I] from it.
        let (dir, out) = fixture_dirs("loadable")?;
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let rep = requantize(&dir, &out, &Plan::default())?;
        assert!(rep.sidecars.iter().any(|s| s == "config.json"), "config.json must travel with the shards");
        assert!(out.join("config.json").exists(), "and actually be there");
        peregrine_core::config::Cfg::load(&out)?; // the check that used to fail
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn an_interrupted_run_resumes_and_a_changed_scheme_is_refused() -> Result<(), Error> {
        // Re-running used to restart at shard 0 while the previous run's higher
        // shards survived, putting one tensor name in two files — which
        // `SafeTensors::open` rejects, so an interrupted conversion produced a
        // directory that could not be opened at all.
        let (dir, out) = fixture_dirs("resume")?;
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let plan = Plan { shard_bytes: 4000, ..Plan::default() };

        let first = requantize(&dir, &out, &plan)?;
        assert!(first.shards > 1, "small budget must roll several shards");
        assert!(first.tensors_requantized > 0, "first run does the work");

        let again = requantize(&dir, &out, &plan)?;
        assert_eq!(again.tensors_requantized, 0, "a completed run re-does nothing");
        assert_eq!(again.shards, first.shards, "and adds no shards");
        // The decisive check: the directory is still openable, i.e. no tensor
        // name landed in two shards.
        let st = SafeTensors::open(&out)?;
        assert!(!st.tensors().is_empty(), "output remains readable after a re-run");

        // A different scheme over the same directory must be refused, not merged.
        let other = Plan { target: Target::Int8, ..plan.clone() };
        let verdict = requantize(&dir, &out, &other);
        assert!(verdict.is_err(), "changing the scheme mid-directory must not proceed");
        if let Err(e) = verdict {
            let msg = format!("{e}");
            assert!(msg.contains("different settings"), "explains what clashed: {msg}");
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn dry_run_sizes_agree_with_what_the_converter_writes() -> Result<(), Error> {
        // A size plan is only useful if it is the same number the run produces —
        // otherwise it is a guess dressed as a forecast. Build a small container,
        // predict it, convert it, compare. (On the real GLM-5.2 shard both paths
        // report 2.69 GB -> 1.35 GB, 426 tensors.)
        let dir = std::env::temp_dir().join(format!("peregrine_requant_dry_{}", std::process::id()));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let plan = Plan { target: Target::Int2, ..Plan::default() };
        let predicted = plan_sizes(&dir, &plan)?;
        let actual = requantize(&dir, &out, &plan)?;
        assert_eq!(predicted.tensors_total, actual.tensors_total, "tensor count");
        assert_eq!(predicted.tensors_requantized, actual.tensors_requantized, "requantized count");
        assert_eq!(predicted.bytes_in, actual.bytes_in, "input bytes");
        assert_eq!(predicted.bytes_out, actual.bytes_out, "predicted output bytes must be exact");
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn the_router_is_never_requantized_whatever_include_says() -> Result<(), Error> {
        // **Contract, not an accident.** The evidenced 2-bit recipe for this
        // model class leaves the router (`mlp.gate`) at full precision, and that
        // is separately valuable from preserving output values: the router
        // decides *which experts are selected*, so quantizing it would silently
        // invalidate `route_stats.json`, `schedule.json`, the transition
        // automaton and the VRAM heat knapsack all at once — artifacts that
        // would keep loading and keep being wrong.
        //
        // Today the router survives because `expert_dims` requires both
        // `.mlp.experts.` and a projection suffix. That is a guarantee worth
        // pinning: a future `include` widened to `.mlp.` must not quietly start
        // rewriting it.
        let dir = std::env::temp_dir().join(format!("peregrine_requant_router_{}", std::process::id()));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let src = SafeTensors::open(&dir)?;
        let routers: Vec<String> =
            src.tensors().iter().map(|t| t.name.clone()).filter(|n| n.contains(".mlp.gate")).collect();
        assert!(!routers.is_empty(), "fixture must actually contain a router to protect");

        // Deliberately hostile filter: `.mlp.` matches the router by substring.
        let plan = Plan { target: Target::Int2, include: ".mlp.".into(), ..Plan::default() };
        requantize(&dir, &out, &plan)?;

        let dst = SafeTensors::open(&out)?;
        for name in &routers {
            let mut a = vec![0u8; src.uncompressed_nbytes(name).unwrap_or(0).max(0) as usize];
            let mut b = vec![0u8; dst.uncompressed_nbytes(name).unwrap_or(0).max(0) as usize];
            assert_eq!(a.len(), b.len(), "{name}: size changed, so it was rewritten");
            src.read_raw(name, &mut a)?;
            dst.read_raw(name, &mut b)?;
            assert_eq!(a, b, "{name}: router bytes must survive byte-for-byte");
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn int2_g64_size_arithmetic_matches_what_the_producer_emits() {
        // `plan_sizes` predicts free-space requirements from `payload_bytes` and
        // `scale_count` alone, never by quantizing. If either drifts from
        // `quant_i2_g64` the forecast is wrong in exactly the situation it
        // exists for — an operator checking whether a multi-hour run will fit.
        // The tiny fixture is one group wide, so this checks the arithmetic
        // directly at a realistic width instead.
        for (o, i) in [(8usize, 128usize), (4, 192), (3, 100), (5, 1536)] {
            let w: Vec<f32> = (0..o * i).map(|k| (k % 23) as f32 * 0.1 - 1.0).collect();
            let (q, sc) = peregrine_core::pack::quant_i2_g64(&w, o, i);
            assert_eq!(q.len(), Target::Int2G64.payload_bytes(o, i), "payload bytes for [{o},{i}]");
            assert_eq!(sc.len(), Target::Int2G64.scale_count(o, i), "scale count for [{o},{i}]");
        }
        // And it must be the cheapest scheme on offer at GLM-5.2 expert widths,
        // scales included — the whole reason to add it.
        let (o, i) = (1536usize, 5120usize);
        let bytes = |t: Target| t.payload_bytes(o, i) + t.scale_count(o, i) * 4;
        assert!(bytes(Target::Int2G64) < bytes(Target::Int3G64), "int2-g64 must beat int3-g64");
        assert!(bytes(Target::Int3G64) < bytes(Target::Int4), "int3-g64 must beat int4");
        // **3.0 bits/weight, not 2.** The two f32 per group are 8 bytes per 64
        // values — a full extra bit/weight on top of the 2-bit payload. Pinned
        // here because "2-bit" describes the payload and the container is 50%
        // larger than that name suggests; an f16 scale pair would reach ~2.5.
        let bpw = 8.0 * bytes(Target::Int2G64) as f64 / (o * i) as f64;
        assert!((2.9..3.1).contains(&bpw), "expected ~3.0 bits/weight, got {bpw:.3}");
        // The saving against int4 is therefore ~25%, not the ~50% the payload
        // width alone implies — the same "count the scales" correction int3-g64
        // documents (12.7%, not 25%).
        let cut = 1.0 - bytes(Target::Int2G64) as f64 / bytes(Target::Int4) as f64;
        assert!((0.20..0.30).contains(&cut), "expected ~25% smaller than int4, got {:.1}%", 100.0 * cut);
    }

    #[test]
    fn int2_g64_refuses_a_model_too_narrow_to_encode_unambiguously() -> Result<(), Error> {
        // The worst outcome for a converter is a container that writes cleanly
        // and *loads as the wrong format*. int2-g64 at one group per row is
        // byte-and-scale-identical to grouped int4, so on a narrow model the
        // run must fail up front rather than produce something that decodes
        // garbage without complaint. The tiny fixture is 16 wide — one group.
        let dir = std::env::temp_dir().join(format!("peregrine_requant_i2g_{}", std::process::id()));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let plan = Plan { target: Target::Int2G64, ..Plan::default() };
        let msg = match requantize(&dir, &out, &plan) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        assert!(!msg.is_empty(), "a single-group int2-g64 run must refuse, not write an undetectable container");
        assert!(msg.contains("two"), "the error must say what the requirement is: {msg}");
        assert!(msg.contains("int3-g64"), "…and point at a format that does work: {msg}");

        // Every other target still converts the same fixture, so the guard is
        // specific to the format that needs it rather than a blanket refusal.
        let ok = Plan { target: Target::Int3G64, ..Plan::default() };
        let predicted = plan_sizes(&dir, &ok)?;
        let actual = requantize(&dir, &out, &ok)?;
        assert_eq!(predicted.bytes_out, actual.bytes_out, "int3-g64 prediction must be exact");
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn the_shard_budget_rolls_files() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_requant_r_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture must be removable: {e}");
        }
        let mut w = ShardWriter::new(&dir, "out", 1024);
        for k in 0..4 {
            w.push(&format!("t{k}"), "U8", vec![512], vec![k as u8; 512])?;
        }
        w.flush()?;
        assert_eq!(w.written.len(), 2, "4 x 512 B at a 1 KiB budget = 2 shards");
        // Every shard must be independently readable, and no `.part` may survive.
        let st = SafeTensors::open(&dir)?;
        assert_eq!(st.tensors().len(), 4, "all four tensors visible across shards");
        let leftovers = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "part"))
            .count();
        assert_eq!(leftovers, 0, "no uncommitted .part files");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

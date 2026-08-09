//! Expert-level resharding: re-pack a sharded safetensors MoE checkpoint so
//! each sparse layer's routed experts are split across bandwidth-proportional
//! device groups (M4b of the throughput campaign — `docs/performance-tuning.md`).
//!
//! Why file-level placement cannot do this: each source shard holds ~142
//! experts and one layer spans only ~1.8 adjacent shards, so however whole
//! files are placed, one layer's reads land on at most two devices.
//! Splitting *experts* into per-(layer, group) files lets every token's expert
//! reads fan out across all devices in proportion to their bandwidth.
//!
//! Scope is deliberately narrow: this tool **groups and re-packs, nothing
//! else**. Physical placement happens later by moving each group's files onto
//! its device and symlinking them back into one model directory, so there is
//! no `--layout` flag. Two invariants the engine depends on:
//!
//! - **Bytes are copied verbatim** from the source regions — no dequantize/
//!   requantize, no transformation. Unlike `peregrine-requantize` this tool is
//!   bit-identity-gated end to end (`--verify` proves it against the disk).
//! - Within an output file each expert's six regions (gate/up/down ×
//!   weight/scale) are laid out **contiguously, in exactly the order the
//!   streaming lane reads them** (`concurrent.rs`'s `read_expert`), so the
//!   engine's same-fd/abutting-offset read coalescing still merges an expert
//!   into one submit after the move.
//!
//! Everything that is not a routed-expert tensor — embeddings, attention,
//! norms, routers, shared experts, `lm_head`, and the MTP layer (including its
//! own `.mlp.experts.` tensors, which the decode loop never streams) — travels
//! with the FIRST group listed, in rolling `trunk-<group>-NNNNN.safetensors`
//! files.

use crate::requant::expert_coords;
use crate::stwrite::{write_streaming, PieceMeta};
use peregrine_core::config::Cfg;
use peregrine_core::dtype::Dtype;
use peregrine_core::safetensors::{SafeTensors, TensorInfo};
use peregrine_core::{Context, Error};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Streaming copy/compare chunk. Bounds resident memory per tensor regardless
/// of tensor size — a 2.69 GB source shard is never materialised.
const COPY_CHUNK: usize = 4 << 20;

/// Trunk files roll to a new file once they reach this size (≤ ~5 GB each, so
/// a group's trunk can still be placed file-by-file).
pub const TRUNK_SHARD_BYTES: u64 = 5_000_000_000;

/// The six per-expert regions, in the exact order the engine's streaming lane
/// submits them (`concurrent.rs read_expert`: gate/up/down × weight, scale).
/// The output writes them contiguously in this order so exactly-abutting
/// offsets on one fd still coalesce into one read.
const EXPERT_REGION_ORDER: [&str; 6] = [
    "gate_proj.weight",
    "gate_proj.weight.qs",
    "up_proj.weight",
    "up_proj.weight.qs",
    "down_proj.weight",
    "down_proj.weight.qs",
];

/// One storage group: a name (becomes part of every output filename) and a
/// relative bandwidth weight.
#[derive(Debug, Clone)]
pub struct GroupSpec {
    pub name: String,
    pub weight: f64,
}

/// Parse `--groups <name>:<weight>,<name>:<weight>,...`.
pub fn parse_groups(s: &str) -> Result<Vec<GroupSpec>, Error> {
    let mut out: Vec<GroupSpec> = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, w) = part
            .split_once(':')
            .ok_or_else(|| Error::Format(format!("--groups entry '{part}' is not <name>:<weight>")))?;
        let weight: f64 = w
            .parse()
            .map_err(|_| Error::Format(format!("--groups entry '{part}': weight '{w}' is not a number")))?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Error::Format(format!("--groups entry '{part}': weight must be a positive number")));
        }
        // The name is embedded in filenames, so keep it filesystem-inert.
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(Error::Format(format!(
                "--groups name '{name}' must be non-empty [A-Za-z0-9_-] (it becomes part of filenames)"
            )));
        }
        if out.iter().any(|g| g.name == name) {
            return Err(Error::Format(format!("--groups name '{name}' listed twice")));
        }
        out.push(GroupSpec { name: name.to_string(), weight });
    }
    if out.is_empty() {
        return Err(Error::Format("--groups is empty".into()));
    }
    Ok(out)
}

/// What to reshard and how.
#[derive(Debug, Clone)]
pub struct Options {
    pub groups: Vec<GroupSpec>,
    /// Optional `route_stats.json`-shaped file (`{"heat": [layer*n_experts+e]}`,
    /// the array `Model::try_load_route_stats` persists). Absent → uniform heat.
    pub route_stats: Option<PathBuf>,
    /// Roll trunk files at this many bytes (a knob for tests; the CLI uses
    /// [`TRUNK_SHARD_BYTES`]).
    pub trunk_shard_bytes: u64,
}

impl Options {
    pub fn new(groups: Vec<GroupSpec>) -> Options {
        Options { groups, route_stats: None, trunk_shard_bytes: TRUNK_SHARD_BYTES }
    }
}

/// One output file: its name, owning group, and the tensor names it carries in
/// write order.
#[derive(Debug, Clone)]
pub struct FilePlan {
    pub file_name: String,
    pub group: usize,
    pub tensors: Vec<String>,
    pub bytes: u64,
}

/// The full, auditable decision: which expert goes to which group, which file
/// carries which tensors, and what each group is expected to serve.
#[derive(Debug, Clone)]
pub struct Plan {
    pub groups: Vec<GroupSpec>,
    /// layer → (expert → group index). Only sparse (routed) layers appear.
    pub assignment: BTreeMap<usize, BTreeMap<usize, usize>>,
    /// Expert files first (layer-major), then the rolling trunk files.
    pub files: Vec<FilePlan>,
    /// Total output bytes per group (trunk included, in the first group).
    pub group_bytes: Vec<u64>,
    /// Routed-expert bytes per group — the quantity the weights balance.
    pub routed_bytes: Vec<u64>,
    /// Expected share of per-token routed bytes per group, under the heat
    /// distribution used for the assignment. This is the number that should
    /// track the weights: per-token reads are heat-weighted, disk is not.
    pub token_share: Vec<f64>,
    pub tensors_total: usize,
}

impl Plan {
    pub fn files_in_group(&self, g: usize) -> usize {
        self.files.iter().filter(|f| f.group == g).count()
    }
}

/// What a write did, for the caller to report.
#[derive(Debug, Clone)]
pub struct WriteReport {
    pub files: Vec<PathBuf>,
    pub bytes_written: u64,
    pub sidecars: Vec<String>,
    pub manifest: PathBuf,
}

/// Rank of a routed-expert tensor within its expert's on-disk run: the six
/// known regions in engine read order, anything unexpected after them by name.
fn region_rank(name: &str) -> (usize, &str) {
    let suffix = name
        .split(".mlp.experts.")
        .nth(1)
        .and_then(|rest| rest.split_once('.'))
        .map(|(_, s)| s)
        .unwrap_or("");
    let rank = EXPERT_REGION_ORDER.iter().position(|&s| s == suffix).unwrap_or(EXPERT_REGION_ORDER.len());
    (rank, suffix)
}

/// Split the source index into routed experts (layer → expert → ordered tensor
/// names) and trunk (everything else, in source-index order). `mtp_cutoff` is
/// the model's hidden-layer count: expert tensors on layers at or past it (the
/// MTP layer) are trunk, because the decode loop never streams them.
fn scan(st: &SafeTensors, mtp_cutoff: Option<usize>) -> (BTreeMap<usize, BTreeMap<usize, Vec<String>>>, Vec<String>) {
    let mut experts: BTreeMap<usize, BTreeMap<usize, Vec<String>>> = BTreeMap::new();
    let mut trunk: Vec<String> = Vec::new();
    for t in st.tensors() {
        match expert_coords(&t.name) {
            Some((layer, expert)) if mtp_cutoff.is_none_or(|n| layer < n) => {
                experts.entry(layer).or_default().entry(expert).or_default().push(t.name.clone());
            }
            _ => trunk.push(t.name.clone()),
        }
    }
    for layer in experts.values_mut() {
        for names in layer.values_mut() {
            names.sort_by(|a, b| region_rank(a).cmp(&region_rank(b)));
        }
    }
    (experts, trunk)
}

/// Load the flat `heat` array out of a `route_stats.json`-shaped file — the
/// same `[layer * n_experts + expert]` shape `Model::try_load_route_stats`
/// restores and `HeatTier::from_route_stats` validates. Like the latter, no
/// config-fingerprint gate (the caller chose the file), but the length *is*
/// checked: a mismatched trace would misalign every layer.
fn load_heat(path: &Path, n_layers: usize, n_experts: usize) -> Result<Vec<u64>, Error> {
    let bytes = std::fs::read(path).ctx(|| format!("read {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    let arr = v
        .get("heat")
        .and_then(|h| h.as_array())
        .ok_or_else(|| Error::Format(format!("{} has no `heat` array", path.display())))?;
    let heat: Vec<u64> = arr.iter().filter_map(|x| x.as_u64()).collect();
    if heat.len() != n_layers * n_experts {
        return Err(Error::Format(format!(
            "{}: heat is {} entries, expected {} (n_layers {n_layers} x n_experts {n_experts}) — \
             a mismatched trace would misalign every layer",
            path.display(),
            heat.len(),
            n_layers * n_experts
        )));
    }
    Ok(heat)
}

/// Effective heat for one layer's experts. A thin row (fewer observations than
/// experts) says nothing about skew, so it falls back to uniform 1 — otherwise
/// counts are floored at 1 so never-routed experts still spread across groups
/// instead of piling onto whichever group the argmin visits first.
fn effective_heat(row: &[u64]) -> Vec<f64> {
    let sum: u64 = row.iter().sum();
    if sum < row.len() as u64 {
        return vec![1.0; row.len()];
    }
    row.iter().map(|&h| h.max(1) as f64).collect()
}

/// Build the full plan: scan, heat, greedy assignment, file layout. Reads only
/// headers and sidecars — `--dry-run` stops here having written nothing.
pub fn plan(model: &Path, opts: &Options) -> Result<Plan, Error> {
    let st = SafeTensors::open(model)?;
    plan_with(&st, model, opts)
}

fn plan_with(st: &SafeTensors, model: &Path, opts: &Options) -> Result<Plan, Error> {
    let ngroups = opts.groups.len();
    // The config is the only place the MTP cutoff and the heat-array stride
    // live. Without it the tool still reshards (every `.mlp.experts.` layer is
    // treated as routed), but says so — an MTP layer would then be split too.
    let cfg = Cfg::load(model).ok();
    if cfg.is_none() {
        eprintln!(
            "peregrine-reshard: no readable config.json in {} — treating every \
             `.mlp.experts.` layer as routed (no MTP cutoff)",
            model.display()
        );
    }
    let mtp_cutoff = cfg.as_ref().map(|c| c.n_layers as usize);
    let (experts, trunk) = scan(st, mtp_cutoff);
    if experts.is_empty() {
        return Err(Error::Format(format!(
            "{}: no routed-expert tensors (`.mlp.experts.`) found — nothing to reshard",
            model.display()
        )));
    }

    let nbytes = |name: &str| -> u64 { st.nbytes(name).unwrap_or(0).max(0) as u64 };

    // Heat, flat [layer * n_experts + expert]. Stride and row count come from
    // the config when present, else from the names themselves.
    let n_experts = cfg
        .as_ref()
        .map(|c| c.n_experts as usize)
        .unwrap_or_else(|| experts.values().flat_map(|l| l.keys()).max().map_or(0, |m| m + 1));
    let n_layers = mtp_cutoff.unwrap_or_else(|| experts.keys().max().map_or(0, |m| m + 1));
    let heat: Option<Vec<u64>> = match &opts.route_stats {
        Some(p) => Some(load_heat(p, n_layers, n_experts)?),
        None => None,
    };

    // Greedy per layer: hottest expert first, each into the group with the
    // smallest assigned_heat/weight at that moment (ties → first listed).
    // Every group's expected per-token routed bytes come out proportional to
    // its weight; with uniform heat that is also the byte count.
    let mut assignment: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
    let mut token_share_num = vec![0f64; ngroups];
    let mut token_share_den = 0f64;
    for (&layer, layer_experts) in &experts {
        let ids: Vec<usize> = layer_experts.keys().copied().collect();
        let raw_row: Vec<u64> = ids
            .iter()
            .map(|&e| heat.as_ref().and_then(|h| h.get(layer * n_experts + e)).copied().unwrap_or(0))
            .collect();
        let eff = effective_heat(&raw_row);
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by(|&a, &b| eff[b].partial_cmp(&eff[a]).unwrap_or(std::cmp::Ordering::Equal).then(ids[a].cmp(&ids[b])));
        let mut loads = vec![0f64; ngroups];
        let slot = assignment.entry(layer).or_default();
        let layer_heat: f64 = eff.iter().sum();
        for i in order {
            let g = (0..ngroups)
                .min_by(|&a, &b| {
                    let ra = loads[a] / opts.groups[a].weight;
                    let rb = loads[b] / opts.groups[b].weight;
                    ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            loads[g] += eff[i];
            slot.insert(ids[i], g);
            let expert_bytes: u64 = layer_experts[&ids[i]].iter().map(|n| nbytes(n)).sum();
            let contrib = eff[i] / layer_heat * expert_bytes as f64;
            token_share_num[g] += contrib;
            token_share_den += contrib;
        }
    }
    let token_share: Vec<f64> = token_share_num
        .iter()
        .map(|&n| if token_share_den > 0.0 { n / token_share_den } else { 0.0 })
        .collect();

    // File layout: one file per (layer, group) with members, experts ascending,
    // each expert's regions contiguous in engine read order.
    let mut files: Vec<FilePlan> = Vec::new();
    let mut group_bytes = vec![0u64; ngroups];
    let mut routed_bytes = vec![0u64; ngroups];
    for (&layer, layer_experts) in &experts {
        for (g, spec) in opts.groups.iter().enumerate() {
            let mut tensors: Vec<String> = Vec::new();
            let mut bytes = 0u64;
            for (&e, names) in layer_experts {
                if assignment[&layer][&e] != g {
                    continue;
                }
                for n in names {
                    bytes += nbytes(n);
                    tensors.push(n.clone());
                }
            }
            if tensors.is_empty() {
                continue;
            }
            group_bytes[g] += bytes;
            routed_bytes[g] += bytes;
            files.push(FilePlan {
                file_name: format!("experts-l{layer}-{}.safetensors", spec.name),
                group: g,
                tensors,
                bytes,
            });
        }
    }

    // Trunk: everything else rides with the first group, rolled at the budget.
    // Source-index order keeps each `.qs` next to its weight.
    let budget = opts.trunk_shard_bytes.max(1);
    let mut cur: Vec<String> = Vec::new();
    let mut cur_bytes = 0u64;
    let mut trunk_idx = 0usize;
    let roll = |cur: &mut Vec<String>, cur_bytes: &mut u64, files: &mut Vec<FilePlan>, trunk_idx: &mut usize| {
        if cur.is_empty() {
            return;
        }
        files.push(FilePlan {
            file_name: format!("trunk-{}-{:05}.safetensors", opts.groups[0].name, *trunk_idx),
            group: 0,
            tensors: std::mem::take(cur),
            bytes: *cur_bytes,
        });
        *trunk_idx += 1;
        *cur_bytes = 0;
    };
    for name in &trunk {
        let nb = nbytes(name);
        if cur_bytes > 0 && cur_bytes + nb > budget {
            roll(&mut cur, &mut cur_bytes, &mut files, &mut trunk_idx);
        }
        cur.push(name.clone());
        cur_bytes += nb;
        group_bytes[0] += nb;
    }
    roll(&mut cur, &mut cur_bytes, &mut files, &mut trunk_idx);

    Ok(Plan {
        groups: opts.groups.clone(),
        assignment,
        files,
        group_bytes,
        routed_bytes,
        token_share,
        tensors_total: st.tensors().len(),
    })
}

fn dtype_str(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::Bf16 => "BF16",
        Dtype::F16 => "F16",
        Dtype::U8 => "U8",
    }
}

/// Header entry for a verbatim copy: source dtype/shape exactly, plus any
/// `compression`/`layout` tags the source carried — dropping either would make
/// the bytes decode as something they are not.
fn piece_meta(t: &TensorInfo) -> PieceMeta {
    let mut extra: Vec<(String, serde_json::Value)> = Vec::new();
    if let Some(tag) = t.compression.tag() {
        extra.push(("compression".into(), serde_json::json!(tag)));
        extra.push(("uncompressed_nbytes".into(), serde_json::json!(t.uncompressed_nbytes)));
    }
    if let Some((tag, gs)) = &t.layout {
        extra.push(("layout".into(), serde_json::json!(tag)));
        extra.push(("layout_gs_bytes".into(), serde_json::json!(gs)));
    }
    PieceMeta {
        name: t.name.clone(),
        dtype: dtype_str(t.dtype).to_string(),
        shape: t.shape.clone(),
        nbytes: t.nbytes.max(0) as u64,
        extra,
    }
}

/// Lazily-opened plain fds onto the source shards, for chunked `pread` copies.
/// (`SafeTensors` reads go through io_uring and materialise whole tensors; a
/// verbatim copier wants bounded chunks, so it reads the files directly.)
struct SourceFiles<'a> {
    st: &'a SafeTensors,
    open: HashMap<usize, std::fs::File>,
}

impl<'a> SourceFiles<'a> {
    fn new(st: &'a SafeTensors) -> SourceFiles<'a> {
        SourceFiles { st, open: HashMap::new() }
    }

    fn file(&mut self, idx: usize) -> Result<&std::fs::File, Error> {
        if !self.open.contains_key(&idx) {
            let path = &self.st.paths()[idx];
            let f = std::fs::File::open(path).ctx(|| path.display().to_string())?;
            self.open.insert(idx, f);
        }
        Ok(&self.open[&idx])
    }

    /// Stream one tensor's on-disk region into `w`, verbatim, in bounded chunks.
    fn copy_region(&mut self, t: &TensorInfo, w: &mut dyn Write) -> Result<(), Error> {
        let f = self.file(t.file_idx)?;
        let mut buf = vec![0u8; COPY_CHUNK.min(t.nbytes.max(0) as usize).max(1)];
        let mut off = t.off;
        let mut remaining = t.nbytes.max(0) as usize;
        while remaining > 0 {
            let n = COPY_CHUNK.min(remaining);
            f.read_exact_at(&mut buf[..n], off).ctx(|| format!("read '{}' @ {off}", t.name))?;
            w.write_all(&buf[..n]).ctx(|| format!("write '{}'", t.name))?;
            off += n as u64;
            remaining -= n;
        }
        Ok(())
    }
}

/// Execute a plan: write every output file, copy the model sidecars, and emit
/// `manifest.json`. Refuses an output directory that already holds shards —
/// there is no resume here, and mixing two runs' files would put one tensor
/// name in two files, which `SafeTensors::open` rejects outright.
pub fn write(model: &Path, out: &Path, plan: &Plan) -> Result<WriteReport, Error> {
    let st = SafeTensors::open(model)?;
    std::fs::create_dir_all(out).ctx(|| format!("create {}", out.display()))?;
    for entry in std::fs::read_dir(out).ctx(|| out.display().to_string())? {
        let p = entry.ctx(|| out.display().to_string())?.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            return Err(Error::Format(format!(
                "{} already contains {} — refusing to mix two reshard runs in one directory \
                 (delete the old output, or point --out elsewhere)",
                out.display(),
                p.file_name().and_then(|n| n.to_str()).unwrap_or("shards")
            )));
        }
    }

    let mut src = SourceFiles::new(&st);
    let mut rep = WriteReport {
        files: Vec::new(),
        bytes_written: 0,
        sidecars: Vec::new(),
        manifest: out.join("manifest.json"),
    };
    for fp in &plan.files {
        let mut infos: Vec<TensorInfo> = Vec::with_capacity(fp.tensors.len());
        for name in &fp.tensors {
            infos.push(
                st.find(name)
                    .ok_or_else(|| Error::Format(format!("planned tensor '{name}' vanished from the source")))?
                    .clone(),
            );
        }
        let pieces: Vec<PieceMeta> = infos.iter().map(piece_meta).collect();
        let meta = vec![
            ("peregrine.reshard.source".to_string(), model.display().to_string()),
            ("peregrine.reshard.group".to_string(), plan.groups[fp.group].name.clone()),
        ];
        let path = out.join(&fp.file_name);
        write_streaming(&path, &meta, &pieces, |i, w| src.copy_region(&infos[i], w))?;
        rep.bytes_written += fp.bytes;
        rep.files.push(path);
    }

    // A directory of shards is not a model: once the group files are symlinked
    // back together, `Cfg::load` still needs `config.json` and serving needs
    // the tokenizer. Same sidecar set `peregrine-requantize` carries.
    for side in ["config.json", "tokenizer.json", "generation_config.json", "tokenizer_config.json"] {
        let s = model.join(side);
        if s.exists() {
            std::fs::copy(&s, out.join(side)).ctx(|| format!("copy {side}"))?;
            rep.sidecars.push(side.to_string());
        }
    }

    let manifest = manifest_json(model, plan);
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| Error::Format(format!("manifest: {e}")))?;
    peregrine_core::durable::write_atomic(&rep.manifest, &bytes)?;
    Ok(rep)
}

/// The auditable record: per group its files, bytes, and expected per-token
/// share, plus the full layer → expert → group assignment.
fn manifest_json(model: &Path, plan: &Plan) -> serde_json::Value {
    let groups: Vec<serde_json::Value> = plan
        .groups
        .iter()
        .enumerate()
        .map(|(g, spec)| {
            let files: Vec<&str> =
                plan.files.iter().filter(|f| f.group == g).map(|f| f.file_name.as_str()).collect();
            serde_json::json!({
                "name": spec.name,
                "weight": spec.weight,
                "files": files,
                "total_bytes": plan.group_bytes[g],
                "routed_bytes": plan.routed_bytes[g],
                "expected_per_token_routed_share": plan.token_share[g],
            })
        })
        .collect();
    let mut assignment = serde_json::Map::new();
    for (layer, experts) in &plan.assignment {
        let mut row = serde_json::Map::new();
        for (e, &g) in experts {
            row.insert(e.to_string(), serde_json::Value::String(plan.groups[g].name.clone()));
        }
        assignment.insert(layer.to_string(), serde_json::Value::Object(row));
    }
    serde_json::json!({
        "version": 1,
        "tool": "peregrine-reshard",
        "source": model.display().to_string(),
        "groups": groups,
        "assignment": assignment,
    })
}

/// One output file's verification outcome.
#[derive(Debug, Clone)]
pub struct FileVerdict {
    pub file: String,
    pub tensors: usize,
    pub mismatches: Vec<String>,
}

/// The full verification outcome.
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    pub files: Vec<FileVerdict>,
    /// Source tensors with no counterpart in the output.
    pub missing: Vec<String>,
    /// Output tensors with no counterpart in the source.
    pub extra: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.files.iter().all(|f| f.mismatches.is_empty())
    }
}

/// Byte-compare EVERY source tensor's on-disk region against the output,
/// streaming in bounded chunks (never a whole 2.69 GB shard in RAM). Also
/// checks dtype/shape/nbytes and that the tensor *sets* match exactly. Pure
/// reads — usable against a freshly-written or a pre-existing `--out`.
pub fn verify(model: &Path, out: &Path) -> Result<VerifyReport, Error> {
    let src = SafeTensors::open(model)?;
    let dst = SafeTensors::open(out)?;
    let mut per_file: BTreeMap<usize, FileVerdict> = BTreeMap::new();
    for (i, p) in dst.paths().iter().enumerate() {
        per_file.insert(
            i,
            FileVerdict {
                file: p.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string(),
                tensors: 0,
                mismatches: Vec::new(),
            },
        );
    }
    let mut rep = VerifyReport::default();
    let mut sfiles = SourceFiles::new(&src);
    let mut dfiles = SourceFiles::new(&dst);
    let mut sbuf = vec![0u8; COPY_CHUNK];
    let mut dbuf = vec![0u8; COPY_CHUNK];

    for t in src.tensors() {
        let Some(o) = dst.find(&t.name) else {
            rep.missing.push(t.name.clone());
            continue;
        };
        let verdict = per_file.get_mut(&o.file_idx).expect("every output tensor has a file");
        verdict.tensors += 1;
        if o.dtype != t.dtype || o.shape != t.shape || o.nbytes != t.nbytes {
            verdict.mismatches.push(format!(
                "{}: header differs (dtype/shape/nbytes {:?}/{:?}/{} vs {:?}/{:?}/{})",
                t.name, o.dtype, o.shape, o.nbytes, t.dtype, t.shape, t.nbytes
            ));
            continue;
        }
        let sf = sfiles.file(t.file_idx)?;
        let df = dfiles.file(o.file_idx)?;
        let mut remaining = t.nbytes.max(0) as usize;
        let (mut soff, mut doff) = (t.off, o.off);
        while remaining > 0 {
            let n = COPY_CHUNK.min(remaining);
            sf.read_exact_at(&mut sbuf[..n], soff).ctx(|| format!("verify read '{}' (source)", t.name))?;
            df.read_exact_at(&mut dbuf[..n], doff).ctx(|| format!("verify read '{}' (output)", t.name))?;
            if sbuf[..n] != dbuf[..n] {
                let at = sbuf[..n].iter().zip(&dbuf[..n]).position(|(a, b)| a != b).unwrap_or(0);
                verdict
                    .mismatches
                    .push(format!("{}: byte {} differs", t.name, (soff - t.off) as usize + at));
                break;
            }
            soff += n as u64;
            doff += n as u64;
            remaining -= n;
        }
    }
    for t in dst.tensors() {
        if !src.has(&t.name) {
            rep.extra.push(t.name.clone());
        }
    }
    rep.files = per_file.into_values().collect();
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requant::ShardWriter;

    fn fixture_dirs(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("peregrine_reshard_{}_{}", std::process::id(), tag));
        let out = dir.with_extension("out");
        for d in [&dir, &out] {
            if let Err(e) = std::fs::remove_dir_all(d) {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
            }
        }
        (dir, out)
    }

    /// Deterministic per-tensor bytes: same name → same content, different
    /// names → (overwhelmingly) different content, so a region copied to the
    /// wrong place cannot pass the byte comparison.
    fn synth_bytes(name: &str, n: usize) -> Vec<u8> {
        let mut x: u64 = 0xcbf29ce484222325;
        for b in name.bytes() {
            x = (x ^ b as u64).wrapping_mul(0x100000001b3);
        }
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Tiny synthetic model: 2 sparse layers x 8 routed experts (six regions
    /// each, `concurrent.rs` naming), trunk tensors (embeddings, attention,
    /// norms, router, shared experts, lm_head), and an MTP layer (index 2 ==
    /// num_hidden_layers) that carries its own expert tensors. Written through
    /// `ShardWriter` with a small budget so the source is genuinely
    /// multi-shard, like the real checkpoint.
    fn build_fixture(dir: &Path) -> Result<(), Error> {
        std::fs::create_dir_all(dir)?;
        let cfg = serde_json::json!({
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "intermediate_size": 32, "first_k_dense_replace": 0, "n_shared_experts": 1,
            "vocab_size": 64, "q_lora_rank": 16, "kv_lora_rank": 16,
            "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 8,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5, "eos_token_id": [1],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg)?)?;

        fn push(w: &mut ShardWriter, name: &str, dtype: &'static str, shape: Vec<i64>, n: usize) {
            w.push(name, dtype, shape, synth_bytes(name, n)).expect("fixture push");
        }
        fn expert(w: &mut ShardWriter, layer: usize, e: usize) {
            let p = format!("model.layers.{layer}.mlp.experts.{e}.");
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                push(w, &format!("{p}{proj}.weight"), "U8", vec![8, 8], 64);
                push(w, &format!("{p}{proj}.weight.qs"), "F32", vec![8], 32);
            }
        }
        let mut w = ShardWriter::new(dir, "model", 1000);
        push(&mut w, "model.embed_tokens.weight", "BF16", vec![8, 16], 256);
        for layer in 0..2usize {
            push(&mut w, &format!("model.layers.{layer}.input_layernorm.weight"), "F32", vec![16], 64);
            push(&mut w, &format!("model.layers.{layer}.self_attn.kv_a_proj.weight"), "U8", vec![16, 8], 128);
            push(&mut w, &format!("model.layers.{layer}.self_attn.kv_a_proj.weight.qs"), "F32", vec![16], 64);
            push(&mut w, &format!("model.layers.{layer}.mlp.gate.weight"), "F32", vec![8, 16], 512);
            push(&mut w, &format!("model.layers.{layer}.mlp.shared_experts.up_proj.weight"), "U8", vec![8, 16], 128);
            push(&mut w, &format!("model.layers.{layer}.mlp.shared_experts.up_proj.weight.qs"), "F32", vec![8], 32);
            for e in 0..8usize {
                expert(&mut w, layer, e);
            }
        }
        // The MTP layer sits past num_hidden_layers and has expert-shaped
        // names; the decode loop never streams it, so it must ride trunk.
        push(&mut w, "model.layers.2.eh_proj.weight", "U8", vec![16, 16], 256);
        expert(&mut w, 2, 0);
        push(&mut w, "lm_head.weight", "BF16", vec![8, 16], 256);
        w.flush()?;
        assert!(w.written.len() > 1, "fixture must be multi-shard to be representative");
        Ok(())
    }

    fn groups3() -> Vec<GroupSpec> {
        parse_groups("a:1,b:1,c:2").expect("static group spec")
    }

    #[test]
    fn group_spec_parsing_accepts_the_cli_shape_and_rejects_nonsense() {
        let g = parse_groups("nvme:3,ssd:2.5,hdd:1").expect("valid spec");
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].name, "nvme");
        assert_eq!(g[1].weight, 2.5);
        assert!(parse_groups("").is_err(), "empty spec");
        assert!(parse_groups("nvme").is_err(), "missing weight");
        assert!(parse_groups("nvme:fast").is_err(), "non-numeric weight");
        assert!(parse_groups("nvme:0").is_err(), "zero weight");
        assert!(parse_groups("nvme:1,nvme:2").is_err(), "duplicate name");
        assert!(parse_groups("a/b:1").is_err(), "name would escape into the path");
    }

    #[test]
    fn round_trip_preserves_every_tensor_byte_for_byte() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("roundtrip");
        build_fixture(&dir)?;
        let plan = plan(&dir, &Options::new(groups3()))?;
        write(&dir, &out, &plan)?;

        let src = SafeTensors::open(&dir)?;
        let dst = SafeTensors::open(&out)?;
        assert_eq!(src.len(), dst.len(), "every tensor must land exactly once");
        for t in src.tensors() {
            let n = src.uncompressed_nbytes(&t.name).unwrap_or(0).max(0) as usize;
            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            src.read_raw(&t.name, &mut a)?;
            dst.read_raw(&t.name, &mut b)?;
            assert_eq!(a, b, "{}: bytes must survive the re-pack exactly", t.name);
            let o = dst.find(&t.name).expect("present");
            assert_eq!(o.dtype, t.dtype, "{}: dtype preserved", t.name);
            assert_eq!(o.shape, t.shape, "{}: shape preserved", t.name);
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn uniform_heat_grouping_puts_bytes_exactly_proportional_to_weights() -> Result<(), Error> {
        // 8 equal-size experts into weights 1:1:2 must split 2/2/4 per layer —
        // with uniform heat the byte shares are exactly the weight shares.
        let (dir, _out) = fixture_dirs("uniform");
        build_fixture(&dir)?;
        let p = plan(&dir, &Options::new(groups3()))?;
        let total: u64 = p.routed_bytes.iter().sum();
        let shares: Vec<f64> = p.routed_bytes.iter().map(|&b| b as f64 / total as f64).collect();
        for (share, want) in shares.iter().zip([0.25, 0.25, 0.5]) {
            assert!((share - want).abs() < 1e-9, "routed byte shares {shares:?} must match weights");
        }
        for (share, want) in p.token_share.iter().zip([0.25, 0.25, 0.5]) {
            assert!((share - want).abs() < 1e-9, "token shares {:?} must match weights", p.token_share);
        }
        // Per layer too, not just in aggregate: every sparse layer must be
        // spread across all three groups, else one layer's reads still serialize
        // onto a subset of devices — the exact failure this tool exists to fix.
        for layer in [0usize, 1] {
            for g in 0..3 {
                let n = p.assignment[&layer].values().filter(|&&x| x == g).count();
                assert_eq!(n, [2, 2, 4][g], "layer {layer} group {g}");
            }
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn skewed_heat_balances_expected_per_token_bytes_not_expert_counts() -> Result<(), Error> {
        let (dir, _out) = fixture_dirs("skew");
        build_fixture(&dir)?;
        // Heat is flat [layer * n_experts + expert] over num_hidden_layers=2
        // rows — the route_stats.json shape try_load_route_stats persists.
        let row = [32u64, 16, 8, 4, 2, 1, 1, 0];
        let heat: Vec<u64> = row.iter().chain(row.iter()).copied().collect();
        let rs = dir.join("route_stats.json");
        std::fs::write(&rs, serde_json::to_vec(&serde_json::json!({"heat": heat}))?)?;

        let mut opts = Options::new(parse_groups("x:1,y:1")?);
        opts.route_stats = Some(rs);
        let p = plan(&dir, &opts)?;
        // Equal weights: expected per-token bytes must come out near 50/50 even
        // though the heat is wildly skewed — which forces unequal expert counts.
        assert!(
            (p.token_share[0] - 0.5).abs() < 0.1,
            "token shares {:?} should track the 1:1 weights",
            p.token_share
        );
        let l0 = &p.assignment[&0];
        assert_ne!(l0[&0], l0[&1], "the two hottest experts must not share a group");
        let counts = [
            l0.values().filter(|&&g| g == 0).count(),
            l0.values().filter(|&&g| g == 1).count(),
        ];
        assert_ne!(counts[0], counts[1], "balancing heat with this skew requires unequal expert counts");
        // A heat array of the wrong length must refuse, not misalign every layer.
        let bad = dir.join("route_stats_bad.json");
        std::fs::write(&bad, serde_json::to_vec(&serde_json::json!({"heat": [1, 2, 3]}))?)?;
        opts.route_stats = Some(bad);
        assert!(plan(&dir, &opts).is_err(), "a 3-entry heat array for 16 slots must be rejected");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn an_experts_six_regions_stay_contiguous_and_in_read_order() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("contig");
        build_fixture(&dir)?;
        let p = plan(&dir, &Options::new(groups3()))?;
        write(&dir, &out, &p)?;
        let dst = SafeTensors::open(&out)?;
        for layer in [0usize, 1] {
            for e in 0..8usize {
                let g = p.assignment[&layer][&e];
                let file = format!("experts-l{layer}-{}.safetensors", p.groups[g].name);
                let prefix = format!("model.layers.{layer}.mlp.experts.{e}.");
                let mut cursor: Option<(usize, u64)> = None;
                for suffix in EXPERT_REGION_ORDER {
                    let t = dst.find(&format!("{prefix}{suffix}")).expect("region present in output");
                    let fname = dst.paths()[t.file_idx].file_name().and_then(|n| n.to_str()).unwrap_or("?");
                    assert_eq!(fname, file, "{prefix}{suffix}: lands in its (layer, group) file");
                    if let Some((fidx, end)) = cursor {
                        assert_eq!(t.file_idx, fidx, "{prefix}{suffix}: same file as the previous region");
                        assert_eq!(
                            t.off, end,
                            "{prefix}{suffix}: must start exactly where the previous region ends \
                             (abutting offsets are what merge_run coalesces into one read)"
                        );
                    }
                    cursor = Some((t.file_idx, t.off + t.nbytes as u64));
                }
            }
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn trunk_and_mtp_tensors_travel_with_the_first_group_and_roll_at_the_budget() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("trunk");
        build_fixture(&dir)?;
        let mut opts = Options::new(groups3());
        opts.trunk_shard_bytes = 600; // force several trunk files
        let p = plan(&dir, &opts)?;
        write(&dir, &out, &p)?;
        let dst = SafeTensors::open(&out)?;
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate.weight",                       // router
            "model.layers.0.mlp.shared_experts.up_proj.weight",     // shared expert
            "model.layers.1.self_attn.kv_a_proj.weight",            // attention
            "model.layers.2.eh_proj.weight",                        // MTP trunk
            "model.layers.2.mlp.experts.0.gate_proj.weight",        // MTP *expert* — still trunk
            "lm_head.weight",
        ] {
            let t = dst.find(name).unwrap_or_else(|| panic!("{name} missing from output"));
            let fname = dst.paths()[t.file_idx].file_name().and_then(|n| n.to_str()).unwrap_or("?");
            assert!(fname.starts_with("trunk-a-"), "{name} must ride the first group's trunk, got {fname}");
        }
        let trunk_files = p.files.iter().filter(|f| f.file_name.starts_with("trunk-a-")).count();
        assert!(trunk_files > 1, "a 600-byte budget must roll multiple trunk files, got {trunk_files}");
        for f in p.files.iter().filter(|f| f.file_name.starts_with("trunk-")) {
            assert!(f.bytes <= 600 || f.tensors.len() == 1, "{}: over budget with room to roll", f.file_name);
        }
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn manifest_records_files_bytes_and_the_greedy_assignment() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("manifest");
        build_fixture(&dir)?;
        let p = plan(&dir, &Options::new(groups3()))?;
        write(&dir, &out, &p)?;
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("manifest.json"))?)?;
        let groups = v.get("groups").and_then(|g| g.as_array()).expect("groups array");
        assert_eq!(groups.len(), 3);
        for (g, spec) in p.groups.iter().enumerate() {
            assert_eq!(groups[g]["name"], serde_json::json!(spec.name));
            assert_eq!(groups[g]["total_bytes"], serde_json::json!(p.group_bytes[g]));
            let files = groups[g]["files"].as_array().expect("file list");
            assert_eq!(files.len(), p.files_in_group(g), "group {g} file count");
        }
        // The assignment is the audit trail: every (layer, expert) must be
        // recorded under the group name the plan chose.
        for (layer, experts) in &p.assignment {
            for (e, &g) in experts {
                assert_eq!(
                    v["assignment"][layer.to_string()][e.to_string()],
                    serde_json::json!(p.groups[g].name),
                    "layer {layer} expert {e}"
                );
            }
        }
        // Re-running into the same directory must refuse, not interleave.
        assert!(write(&dir, &out, &p).is_err(), "a second write into a populated --out must refuse");
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }

    #[test]
    fn verify_passes_a_faithful_copy_and_catches_a_single_flipped_byte() -> Result<(), Error> {
        let (dir, out) = fixture_dirs("verify");
        build_fixture(&dir)?;
        let p = plan(&dir, &Options::new(groups3()))?;
        write(&dir, &out, &p)?;

        let clean = verify(&dir, &out)?;
        assert!(clean.ok(), "faithful copy must verify: {:?}", clean);
        assert_eq!(
            clean.files.iter().map(|f| f.tensors).sum::<usize>(),
            p.tensors_total,
            "verification must cover every tensor"
        );

        // Corrupt one payload byte in one output file, far from any header.
        let (path, off) = {
            let dst = SafeTensors::open(&out)?;
            let t = dst.find("model.layers.0.mlp.experts.0.up_proj.weight").expect("present");
            (dst.paths()[t.file_idx].clone(), t.off + 3)
        };
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&path)?;
        let mut b = [0u8; 1];
        f.read_exact_at(&mut b, off)?;
        f.write_all_at(&[b[0] ^ 0xff], off)?;

        let dirty = verify(&dir, &out)?;
        assert!(!dirty.ok(), "a flipped byte must fail verification");
        let hits: Vec<&String> = dirty.files.iter().flat_map(|f| &f.mismatches).collect();
        assert!(
            hits.iter().any(|m| m.contains("model.layers.0.mlp.experts.0.up_proj.weight")),
            "the mismatch must name the corrupted tensor: {hits:?}"
        );
        std::fs::remove_dir_all(&dir)?;
        std::fs::remove_dir_all(&out)?;
        Ok(())
    }
}

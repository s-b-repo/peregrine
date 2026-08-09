//! On-demand safetensors indexing and reading — the Rust port of `c/st.h`.
//!
//! Like the C engine this uses positioned reads (`pread`) rather than mmap, so
//! tensor pages do not stay resident in the process (the RSS fix). O_DIRECT
//! twin fds and `fadvise(DONTNEED)` streaming belong to the M2 I/O lane
//! (`peregrine-io`); this crate is the index plus straightforward converting reads.

use crate::compress::{decode, Compression};
use crate::dtype::{bf16_to_f32, f16_to_f32, Dtype};
use crate::{Context, Error};
use parking_lot::Mutex;
use peregrine_io::Reactor;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

/// Cap on the safetensors header size — real headers are KB..a few MB. A crafted
/// file declaring a huge `hlen` would force a giant allocation before any read.
const ST_MAX_HEADER: u64 = 512 << 20;

/// One tensor's location within a shard file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub file_idx: usize,
    /// absolute byte offset of the data within the file
    pub off: u64,
    /// On-disk byte length. Equals the logical tensor size for uncompressed
    /// entries; equals the compressed payload size when [`Self::compression`]
    /// is [`Compression::Zstd`].
    pub nbytes: i64,
    pub dtype: Dtype,
    pub numel: i64,
    pub shape: Vec<i64>,
    /// Compression scheme applied to the on-disk payload. `None` = raw bytes
    /// (the historical format); `Zstd` = decompress via [`crate::compress::decode`].
    pub compression: Compression,
    /// Original (post-decompression) size, in bytes. Equals `nbytes` for
    /// uncompressed entries.
    pub uncompressed_nbytes: i64,
    /// On-disk byte-layout tag + its `gs_bytes` parameter. `None` = the
    /// kernels' native row-major layout; `Some(("kblock", gs))` = group-block
    /// transposed, auto-converted back to native by `read_raw`.
    pub layout: Option<(String, usize)>,
}

/// Index over all `*.safetensors` shards in a model directory.
pub struct SafeTensors {
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    files: Vec<File>,
    /// O_DIRECT twin of `files[i]` for bulk expert streaming (bypasses the page
    /// cache). `None` when O_DIRECT is unavailable for that shard (filesystem
    /// rejects the open or aligned reads); callers then use the buffered fd.
    direct_files: Vec<Option<File>>,
    paths: Vec<PathBuf>,
    /// The io_uring lane every read goes through (interior mutability keeps the
    /// read methods `&self`). Serialized: one positioned read at a time.
    reactor: Mutex<Reactor>,
}

impl SafeTensors {
    /// Read exactly `buf.len()` bytes at `off` from shard `file_idx` through the
    /// io_uring reactor. The single choke point for every disk read here.
    fn read_at(&self, file_idx: usize, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        // O_DIRECT trunk load (`COLI_DIRECT_LOAD=1`), when this shard has a
        // working direct fd.
        //
        // This is the consumer `region_direct`'s doc has named since it was
        // written — "the block alignment O_DIRECT requires is applied by the
        // reader (`Reactor::read_direct_many`)" — while `read_direct_many` had
        // no caller outside its own test. The streaming lane uses the *other*
        // direct entry point (`read_direct_aligned`, which returns owned aligned
        // buffers); this one copies into a caller's buffer, which is exactly the
        // shape a loader reading into `out` needs, and is why the two exist.
        //
        // Worth having because the trunk is read **once**. Every byte of it that
        // lands in the page cache afterwards is a byte the warm expert cache
        // cannot use, on a box where that cache is the thing under pressure —
        // which is the problem `COLI_FADVISE_DROP` attacks after the fact and
        // O_DIRECT avoids entirely. Off by default: it is the model-load path,
        // and a fallback that silently produced different bytes would be the
        // worst possible failure, so the gate is explicit and the fallback is a
        // plain retry on the buffered fd.
        if direct_load_enabled() {
            if let Some(dfd) = self.direct_files.get(file_idx).and_then(|f| f.as_ref()) {
                let mut req = [peregrine_io::ReadReq { fd: dfd.as_raw_fd(), offset: off, buf, tag: 0 }];
                let outcome = self.reactor.lock().read_direct_many(&mut req);
                match outcome {
                    Ok(res) if res.first().copied().unwrap_or(-1) >= 0 => return Ok(()),
                    // Any direct failure falls through to the buffered path
                    // below, which reads the same bytes from the same offsets.
                    Ok(_) => peregrine_io::note_advisory_err(
                        "O_DIRECT trunk read returned an error code (using buffered)",
                        &"short or failed direct read",
                    ),
                    Err(e) => peregrine_io::note_advisory_err("O_DIRECT trunk read (using buffered)", &e),
                }
            }
        }
        let fd = self.files[file_idx].as_raw_fd();
        // parking_lot mutex does not poison, so the lock never fails
        self.reactor
            .lock()
            .read_exact(fd, off, buf)
            .ctx(|| format!("{}: io_uring read @ {off}", self.paths[file_idx].display()))
    }
}

/// Whether the trunk load reads through O_DIRECT (`COLI_DIRECT_LOAD=1`).
///
/// Separate from `COLI_DIRECT`, which selects the *streaming expert* lane. They
/// are different reads with different economics: experts are re-read constantly
/// and want the page cache bypassed to stop evicting the warm cache; the trunk
/// is read once at load and wants the same thing for a different reason. Fusing
/// them under one variable would make it impossible to A/B either.
fn direct_load_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_DIRECT_LOAD").as_deref(), Ok("1") | Ok("true")))
}

impl SafeTensors {
    /// Index every `model*.safetensors` shard in `dir` — plus, when the dir
    /// carries a `model_paths.json` (`{"paths": ["/mnt/fast/model-part", ...]}`),
    /// every shard in each listed directory. This is how a model split across
    /// several drives is served without a symlink farm: the primary dir holds
    /// the sidecars and the paths file, each drive holds its own folder of
    /// shards, and relative entries resolve against the primary dir.
    ///
    /// Shards sort by **file name**, not full path (matching the C engine's
    /// ordering so fused-expert offsets line up across shards — and so the
    /// order is independent of which drive a shard lives on). A file name
    /// appearing in two directories is a hard error: silently preferring one
    /// copy would make the load depend on listing order.
    pub fn open(dir: &Path) -> Result<SafeTensors, Error> {
        let mut roots: Vec<PathBuf> = vec![dir.to_path_buf()];
        let paths_file = dir.join("model_paths.json");
        if paths_file.exists() {
            let bytes = std::fs::read(&paths_file).ctx(|| paths_file.display().to_string())?;
            let v: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Format(format!("{}: {e}", paths_file.display())))?;
            let arr = v.get("paths").and_then(|p| p.as_array()).ok_or_else(|| {
                Error::Format(format!(
                    "{}: expected {{\"paths\": [\"/dir\", ...]}}",
                    paths_file.display()
                ))
            })?;
            for p in arr {
                let s = p.as_str().ok_or_else(|| {
                    Error::Format(format!("{}: non-string entry in paths", paths_file.display()))
                })?;
                let extra = if Path::new(s).is_absolute() { PathBuf::from(s) } else { dir.join(s) };
                if !extra.is_dir() {
                    return Err(Error::Format(format!(
                        "{}: {} is not a directory (drive not mounted?)",
                        paths_file.display(),
                        extra.display()
                    )));
                }
                roots.push(extra);
            }
        }
        let mut shard_paths: Vec<PathBuf> = Vec::new();
        for root in &roots {
            for entry in std::fs::read_dir(root).ctx(|| root.display().to_string())? {
                // a failed directory entry is surfaced, not silently dropped
                let path = entry.ctx(|| root.display().to_string())?.path();
                if path.extension().is_some_and(|x| x == "safetensors") {
                    shard_paths.push(path);
                }
            }
        }
        shard_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        for w in shard_paths.windows(2) {
            if w[0].file_name() == w[1].file_name() {
                return Err(Error::Format(format!(
                    "shard {} exists in two model directories ({} and {}) — \
                     refusing to guess which copy to serve",
                    w[1].file_name().unwrap_or_default().to_string_lossy(),
                    w[0].display(),
                    w[1].display()
                )));
            }
        }
        if shard_paths.is_empty() {
            return Err(Error::Format(format!("no .safetensors shards in {}", dir.display())));
        }

        // one io_uring lane for every read: the shard headers here at open time
        // and all tensor data later. Depth covers a per-layer expert batch.
        let mut reactor = Reactor::new(256).ctx(|| "io_uring reactor init".to_string())?;

        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut files: Vec<File> = Vec::with_capacity(shard_paths.len());
        let mut direct_files: Vec<Option<File>> = Vec::with_capacity(shard_paths.len());

        for (file_idx, path) in shard_paths.iter().enumerate() {
            let f = File::open(path).ctx(|| path.display().to_string())?;
            // O_DIRECT twin for bulk expert streaming; kept only if a probe aligned
            // read succeeds. Any failure → buffered fallback (never fatal).
            let direct = open_direct(path);
            let fsz = f.metadata().ctx(|| path.display().to_string())?.len();
            let read = |reactor: &mut Reactor, off: u64, buf: &mut [u8]| -> Result<(), Error> {
                reactor.read_exact(f.as_raw_fd(), off, buf).ctx(|| format!("{}: io_uring read @ {off}", path.display()))
            };

            // Check the size before reading: an 8-byte read against a shorter
            // file surfaces an opaque io_uring short-read error instead of
            // "this is not a safetensors file".
            if fsz < 8 {
                return Err(Error::Format(format!(
                    "{}: too small to be a safetensors file ({fsz} bytes)",
                    path.display()
                )));
            }
            let mut lenbuf = [0u8; 8];
            read(&mut reactor, 0, &mut lenbuf)?;
            let hlen = u64::from_le_bytes(lenbuf);
            if hlen > fsz - 8 || hlen > ST_MAX_HEADER {
                return Err(Error::Format(format!(
                    "{}: bad safetensors header length {hlen} (file {fsz} bytes)",
                    path.display()
                )));
            }

            let mut hdr = vec![0u8; hlen as usize];
            read(&mut reactor, 8, &mut hdr)?;
            let data_start: u64 = 8 + hlen;
            let root: Value = serde_json::from_slice(&hdr).ctx(|| format!("{}: header not JSON", path.display()))?;
            let obj = root
                .as_object()
                .ok_or_else(|| Error::Format(format!("{}: header not a JSON object", path.display())))?;

            for (name, m) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dt = m.get("dtype").and_then(|v| v.as_str());
                let offs = m.get("data_offsets").and_then(|v| v.as_array());
                let shp = m.get("shape").and_then(|v| v.as_array());
                let (dt, offs, shp) = match (dt, offs, shp) {
                    (Some(dt), Some(offs), Some(shp)) if offs.len() >= 2 => (dt, offs, shp),
                    _ => {
                        return Err(Error::Format(format!(
                            "{}: tensor '{name}' malformed dtype/data_offsets/shape",
                            path.display()
                        )))
                    }
                };
                let dtype = Dtype::parse(dt)
                    .ok_or_else(|| Error::Format(format!("unsupported dtype: {dt}")))?;
                let a0 = offs[0].as_i64().unwrap_or(-1);
                let b0 = offs[1].as_i64().unwrap_or(-1);
                // Bounds in u64 with checked arithmetic: `data_start as i64 + b0`
                // wraps negative for a crafted offset near i64::MAX (release
                // builds do not check overflow), which passed this test and then
                // asked for an exabyte-sized allocation.
                let bounds_ok = match (u64::try_from(a0), u64::try_from(b0)) {
                    (Ok(a), Ok(b)) => {
                        b >= a && data_start.checked_add(b).is_some_and(|end| end <= fsz)
                    }
                    _ => false,
                };
                if !bounds_ok {
                    return Err(Error::Format(format!(
                        "{}: tensor '{name}' data_offsets [{a0},{b0}] out of file bounds ({fsz})",
                        path.display()
                    )));
                }
                let shape: Vec<i64> = shp.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
                if shape.iter().any(|&d| d < 0) {
                    return Err(Error::Format(format!("{}: tensor '{name}' has a negative dimension", path.display())));
                }
                let numel: i64 = shape
                    .iter()
                    .try_fold(1i64, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| Error::Format(format!("{}: tensor '{name}' shape overflows", path.display())))?;
                let on_disk_nbytes = b0 - a0;
                // Optional compression + original size. Missing tag ⇒ raw bytes.
                let compression = Compression::from_tag(m.get("compression").and_then(|v| v.as_str()));
                let uncompressed_nbytes = match compression {
                    Compression::None => on_disk_nbytes,
                    _ => m
                        .get("uncompressed_nbytes")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(on_disk_nbytes),
                };
                // The shape and the payload length are independent fields; nothing
                // tied them together, so a short payload for a declared shape read
                // partially and left the tail of the destination at zero while
                // returning Ok. U8 tensors are quantized containers whose element
                // count is deliberately not `numel * 1`, so they are exempt.
                if !matches!(dtype, Dtype::U8) {
                    let want = numel.checked_mul(dtype.elem_size() as i64).ok_or_else(|| {
                        Error::Format(format!("{}: tensor '{name}' size overflows", path.display()))
                    })?;
                    if want != uncompressed_nbytes {
                        return Err(Error::Format(format!(
                            "{}: tensor '{name}' declares {:?} shape {shape:?} ({want} bytes) but carries {uncompressed_nbytes}",
                            path.display(),
                            dtype
                        )));
                    }
                }
                let layout = match (m.get("layout").and_then(|v| v.as_str()), m.get("layout_gs_bytes").and_then(|v| v.as_u64())) {
                    (Some(tag), Some(gs)) => Some((tag.to_string(), gs as usize)),
                    _ => None,
                };
                let idx = tensors.len();
                tensors.push(TensorInfo {
                    name: name.clone(),
                    file_idx,
                    off: data_start + a0 as u64,
                    nbytes: on_disk_nbytes,
                    dtype,
                    numel,
                    shape,
                    compression,
                    uncompressed_nbytes,
                    layout,
                });
                if index.insert(name.clone(), idx).is_some() {
                    // Two shards claiming the same tensor (e.g. a directory
                    // holding both a full and a sharded copy) would resolve half
                    // the model to the wrong file.
                    return Err(Error::Format(format!(
                        "{}: tensor '{name}' appears in more than one shard",
                        path.display()
                    )));
                }
            }
            files.push(f);
            direct_files.push(direct);
        }
        Ok(SafeTensors { tensors, index, files, direct_files, paths: shard_paths, reactor: Mutex::new(reactor) })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
    pub fn numel(&self, name: &str) -> Option<i64> {
        self.find(name).map(|t| t.numel)
    }
    /// **On-disk** byte length — the compressed payload size for a zstd tensor.
    /// Callers deciding what a tensor *contains* (quantized container format,
    /// element counts) want [`Self::uncompressed_nbytes`] instead; this is for
    /// callers that are about to read those bytes off the disk.
    pub fn nbytes(&self, name: &str) -> Option<i64> {
        self.find(name).map(|t| t.nbytes)
    }

    /// Logical byte length after any decompression — the size of the buffer a
    /// reader ends up with, and the only length that describes the tensor's
    /// *contents*. Equals [`Self::nbytes`] for uncompressed entries.
    pub fn uncompressed_nbytes(&self, name: &str) -> Option<i64> {
        self.find(name).map(|t| t.uncompressed_nbytes)
    }

    /// Raw on-disk location of a tensor's data: `(fd, absolute_offset, nbytes)`.
    /// Lets the I/O lane stream the tensor **in place** from the checkpoint (no
    /// re-coalescing to a sidecar file). The `fd` stays valid as long as this
    /// `SafeTensors` is alive, so a streaming `Model` must keep it resident.
    pub fn region(&self, name: &str) -> Option<(RawFd, u64, usize)> {
        self.find(name).map(|t| (self.files[t.file_idx].as_raw_fd(), t.off, t.nbytes as usize))
    }

    /// Like [`Self::region`] but returns the shard's **O_DIRECT** fd (same offset
    /// and length), or `None` when O_DIRECT is unavailable for that shard — the
    /// caller then falls back to [`Self::region`]. The block alignment O_DIRECT
    /// requires is applied by the reader ([`peregrine_io::Reactor::read_direct_many`]).
    pub fn region_direct(&self, name: &str) -> Option<(RawFd, u64, usize)> {
        let t = self.find(name)?;
        let df = self.direct_files.get(t.file_idx)?.as_ref()?;
        Some((df.as_raw_fd(), t.off, t.nbytes as usize))
    }

    /// Whether any shard opened a working O_DIRECT fd (so the direct path is usable).
    pub fn has_any_direct(&self) -> bool {
        self.direct_files.iter().any(Option::is_some)
    }

    /// The compression scheme applied to `name` (`Compression::None` for
    /// uncompressed tensors or unknown names). Callers streaming raw bytes
    /// through the concurrent MoE lane should check this — compressed
    /// tensors need [`Self::read_raw`], which decompresses.
    pub fn compression(&self, name: &str) -> Compression {
        self.find(name).map(|t| t.compression).unwrap_or(Compression::None)
    }

    /// Whether any tensor in this index uses on-disk compression. When `true`
    /// the caller should force resident-expert mode (streamed expert reads
    /// hand raw bytes to the CPU/GPU kernels without a decompress step).
    pub fn has_compressed_tensors(&self) -> bool {
        self.tensors.iter().any(|t| !matches!(t.compression, Compression::None))
    }

    /// Read a tensor as f32, converting BF16/F16/F32. `out` must hold `numel`
    /// floats. Errors on a U8 (quantized) tensor — use [`Self::read_raw`].
    pub fn read_f32(&self, name: &str, out: &mut [f32]) -> Result<i64, Error> {
        let t = self.tensor(name)?;
        if t.dtype == Dtype::U8 {
            return Err(Error::Format(format!("read_f32 on quantized (U8) tensor '{name}'")));
        }
        let need = t.numel as usize;
        if out.len() < need {
            return Err(Error::Format(format!(
                "read_f32 '{name}': out buffer {} < numel {need}",
                out.len()
            )));
        }
        let (dtype, off, on_disk_nbytes, fidx) = (t.dtype, t.off, t.nbytes as usize, t.file_idx);
        let mut disk = vec![0u8; on_disk_nbytes];
        maybe_hugepage(&mut disk);
        self.read_at(fidx, off, &mut disk)?;
        let raw = match t.compression {
            Compression::None => disk,
            other => decode(&disk, other, t.uncompressed_nbytes as usize)
                .ctx(|| format!("read_f32 '{name}' decompress"))?,
        };
        convert_f32(dtype, &raw, &mut out[..need])?;
        Ok(need as i64)
    }

    /// Read the raw bytes of tensor number `idx` (index into [`Self::tensors`]).
    /// Same semantics as [`Self::read_raw`] but addressed positionally — the
    /// offline relayout tool walks the index, not names.
    pub fn read_raw_by_index(&self, idx: usize, out: &mut [u8]) -> Result<(), Error> {
        let name = self
            .tensors
            .get(idx)
            .map(|t| t.name.clone())
            .ok_or_else(|| Error::Format(format!("tensor index {idx} out of range")))?;
        self.read_raw(&name, out)
    }

    /// Read the raw bytes of a tensor (no dtype conversion) — for the already
    /// int4/int8/int2-quantized U8 container payloads. `out` must be
    /// `uncompressed_nbytes`.
    pub fn read_raw(&self, name: &str, out: &mut [u8]) -> Result<(), Error> {
        let t = self.tensor(name)?;
        let need = t.uncompressed_nbytes as usize;
        // Exact, not "at least": an oversized buffer used to be filled partially
        // and the tail left at zero. For a quantized payload zero nibbles decode
        // to -8 (maximum-magnitude negative weights), so the model would produce
        // confident garbage rather than an error.
        if out.len() != need {
            return Err(Error::Format(format!(
                "read_raw '{name}': out buffer {} bytes, tensor is {need}",
                out.len()
            )));
        }
        maybe_hugepage(&mut out[..need]);
        let (off, fidx) = (t.off, t.file_idx);
        match t.compression {
            Compression::None => self.read_at(fidx, off, &mut out[..need])?,
            other => {
                let mut disk = vec![0u8; t.nbytes as usize];
                maybe_hugepage(&mut disk);
                self.read_at(fidx, off, &mut disk)?;
                let raw = decode(&disk, other, need).ctx(|| format!("read_raw '{name}' decompress"))?;
                out[..need].copy_from_slice(&raw);
            }
        }
        // Layout auto-conversion: a "kblock"-tagged payload is stored group-major
        // on disk; permute back to the kernels' native row-major layout so every
        // consumer sees identical bytes regardless of on-disk tiling.
        if let Some((tag, gs_bytes)) = &t.layout {
            if tag == "kblock" {
                let o = t.shape.first().copied().unwrap_or(0).max(0) as usize;
                let native = crate::pack::from_kblock(&out[..need], o, *gs_bytes).ok_or_else(|| {
                    Error::Format(format!("read_raw '{name}': kblock layout does not tile (o={o}, gs_bytes={gs_bytes})"))
                })?;
                out[..need].copy_from_slice(&native);
            }
        }
        Ok(())
    }

    /// Read `n_elems` starting at element `elem_off` (converted to f32). Used for
    /// GLM's fused-expert blocks where one tensor is `[E, ...]` and only one
    /// expert's sub-range is read.
    pub fn read_slice_f32(
        &self,
        name: &str,
        elem_off: i64,
        n_elems: i64,
        out: &mut [f32],
    ) -> Result<(), Error> {
        let t = self.tensor(name)?;
        if t.dtype == Dtype::U8 {
            return Err(Error::Format(format!("read_slice_f32 on quantized (U8) tensor '{name}'")));
        }
        if !matches!(t.compression, Compression::None) {
            // Slicing reads bytes at an offset; a compressed payload has no
            // addressable element boundaries, so the bytes would be decoded as
            // if they were floats.
            return Err(Error::Format(format!(
                "read_slice_f32 '{name}': tensor is compressed — use read_f32 (whole-tensor decompress)"
            )));
        }
        let esz = t.dtype.elem_size() as i64;
        // All of this is attacker/corruption-reachable through the header, so it
        // is checked: a negative offset used to wrap to ~2^64 when cast, and the
        // slice could read past the tensor into the next one's bytes.
        if elem_off < 0 || n_elems < 0 {
            return Err(Error::Format(format!("read_slice_f32 '{name}': negative offset/length")));
        }
        let last = elem_off
            .checked_add(n_elems)
            .ok_or_else(|| Error::Format(format!("read_slice_f32 '{name}': range overflows")))?;
        if last > t.numel {
            return Err(Error::Format(format!(
                "read_slice_f32 '{name}': elements [{elem_off},{last}) exceed the tensor's {} elements",
                t.numel
            )));
        }
        let byte_off = elem_off
            .checked_mul(esz)
            .and_then(|b| u64::try_from(b).ok())
            .and_then(|b| t.off.checked_add(b))
            .ok_or_else(|| Error::Format(format!("read_slice_f32 '{name}': byte offset overflows")))?;
        let nb = n_elems
            .checked_mul(esz)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or_else(|| Error::Format(format!("read_slice_f32 '{name}': byte length overflows")))?;
        let boff = byte_off;
        if out.len() < n_elems as usize {
            return Err(Error::Format(format!("read_slice_f32 '{name}': out buffer too small")));
        }
        let (dtype, fidx) = (t.dtype, t.file_idx);
        let mut raw = vec![0u8; nb];
        maybe_hugepage(&mut raw);
        self.read_at(fidx, boff, &mut raw)?;
        convert_f32(dtype, &raw, &mut out[..n_elems as usize])?;
        Ok(())
    }

    fn tensor(&self, name: &str) -> Result<&TensorInfo, Error> {
        self.find(name).ok_or_else(|| Error::Format(format!("missing tensor: {name}")))
    }
}

/// Ask the kernel to back a large landing buffer with transparent huge pages
/// (2 MB). Threshold matches the huge-page size — below that the advice does
/// nothing useful and just spends a syscall. All-safe wrapper around the
/// `unsafe` `libc::madvise` in `peregrine-io::mem`.
fn maybe_hugepage(buf: &mut [u8]) {
    if buf.len() >= 2 * 1024 * 1024 {
        peregrine_io::advise_hugepages_slice(buf);
    }
}

fn convert_f32(dtype: Dtype, raw: &[u8], out: &mut [f32]) -> Result<(), Error> {
    // `zip` stops at the shorter side, so a source that is too short used to
    // leave the tail of `out` untouched (zeros, or stale contents) and still
    // return Ok — half a weight row silently dequantizing to 0.0.
    let esz = dtype.elem_size();
    let want = out.len().saturating_mul(esz);
    if raw.len() < want {
        return Err(Error::Format(format!(
            "convert_f32: source holds {} bytes but {} elements ({want} bytes) were requested",
            raw.len(),
            out.len()
        )));
    }
    match dtype {
        Dtype::F32 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Dtype::Bf16 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Dtype::F16 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        // Callers (read_f32/read_slice_f32) reject U8 before converting; keep this
        // total (no `unreachable!`) so a misuse is a surfaced error, not a panic.
        Dtype::U8 => return Err(Error::Format("convert_f32 called on a U8 tensor".into())),
    }
    Ok(())
}

/// Open a second, O_DIRECT fd for `path`, kept only if a probe aligned read
/// succeeds (some filesystems accept the open but reject aligned reads with
/// EINVAL). `None` on any failure — the caller falls back to the buffered fd.
/// All-safe: the `O_DIRECT` open is std, and the probe's `unsafe` lives in
/// `peregrine-io` (this crate is `#![forbid(unsafe_code)]`).
#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let df = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path).ok()?;
    peregrine_io::probe_direct(df.as_raw_fd()).then_some(df)
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> Option<File> {
    None
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Dependency-free synthetic safetensors writer for tests (no torch/numpy).
    use serde_json::json;
    use std::path::Path;

    /// A tensor to embed: (name, dtype string, shape, little-endian raw bytes).
    pub struct Blob<'a> {
        pub name: &'a str,
        pub dtype: &'a str,
        pub shape: Vec<i64>,
        pub bytes: Vec<u8>,
    }

    /// Write a single-shard `model.safetensors` into `dir`.
    pub fn write_safetensors(dir: &Path, blobs: &[Blob]) -> Result<(), crate::Error> {
        write_safetensors_named(dir, "model.safetensors", blobs)
    }

    /// Same, but with a caller-chosen file name — the multi-directory tests
    /// need distinct shard names spread across several dirs.
    pub fn write_safetensors_named(dir: &Path, file: &str, blobs: &[Blob]) -> Result<(), crate::Error> {
        let mut header = serde_json::Map::new();
        let mut cursor: i64 = 0;
        let mut data: Vec<u8> = Vec::new();
        for b in blobs {
            let start = cursor;
            let end = start + b.bytes.len() as i64;
            header.insert(
                b.name.to_string(),
                json!({"dtype": b.dtype, "shape": b.shape, "data_offsets": [start, end]}),
            );
            data.extend_from_slice(&b.bytes);
            cursor = end;
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&data);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(file), out)?;
        Ok(())
    }

    pub fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    pub fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("coli_st_{}_{}", std::process::id(), tag));
        if let Err(e) = std::fs::remove_dir_all(&d) {
            if e.kind() != std::io::ErrorKind::NotFound {
                peregrine_io::note_advisory_err("pre-clean test tmpdir", &e);
            }
        }
        d
    }

    #[test]
    fn index_and_read_roundtrip() -> Result<(), Error> {
        let dir = tmpdir("roundtrip");
        write_safetensors(
            &dir,
            &[
                Blob { name: "a", dtype: "F32", shape: vec![2], bytes: f32_bytes(&[1.0, 2.0]) },
                Blob { name: "b", dtype: "BF16", shape: vec![3], bytes: bf16_bytes(&[1.0, 2.0, -4.0]) },
                Blob { name: "w.qs", dtype: "U8", shape: vec![4], bytes: vec![10, 20, 30, 40] },
            ],
        )?;
        let st = SafeTensors::open(&dir)?;
        assert_eq!(st.len(), 3);
        assert!(st.has("a") && st.has("b") && st.has("w.qs"));
        assert_eq!(st.numel("b"), Some(3));

        let mut a = [0f32; 2];
        st.read_f32("a", &mut a)?;
        assert_eq!(a, [1.0, 2.0]);

        let mut b = [0f32; 3];
        st.read_f32("b", &mut b)?;
        assert_eq!(b, [1.0, 2.0, -4.0]);

        let mut raw = [0u8; 4];
        st.read_raw("w.qs", &mut raw)?;
        assert_eq!(raw, [10, 20, 30, 40]);

        // reading a U8 tensor as f32 is an error
        let mut junk = [0f32; 4];
        assert!(st.read_f32("w.qs", &mut junk).is_err());

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn model_paths_json_merges_directories_bit_identically() -> Result<(), Error> {
        // The same two shards, once in a single dir and once split across a
        // primary + a second dir listed in model_paths.json, must load with
        // identical tensor sets and identical bytes — and a shard name present
        // in two directories must be refused, not silently resolved.
        let both = tmpdir("mp_both");
        let prim = tmpdir("mp_prim");
        let sec = tmpdir("mp_sec");
        let a = || Blob { name: "alpha", dtype: "F32", shape: vec![3], bytes: f32_bytes(&[1.0, -2.0, 3.5]) };
        let b = || Blob { name: "beta.qs", dtype: "U8", shape: vec![4], bytes: vec![9, 8, 7, 6] };
        write_safetensors_named(&both, "out-00000.safetensors", &[a()])?;
        write_safetensors_named(&both, "out-00001.safetensors", &[b()])?;
        write_safetensors_named(&prim, "out-00000.safetensors", &[a()])?;
        write_safetensors_named(&sec, "out-00001.safetensors", &[b()])?;
        std::fs::write(
            prim.join("model_paths.json"),
            format!(r#"{{"paths": ["{}"]}}"#, sec.display()),
        )?;

        let st1 = SafeTensors::open(&both)?;
        let st2 = SafeTensors::open(&prim)?;
        assert_eq!(st1.len(), st2.len());
        for st in [&st1, &st2] {
            assert!(st.has("alpha") && st.has("beta.qs"));
        }
        let (mut f1, mut f2) = ([0f32; 3], [0f32; 3]);
        st1.read_f32("alpha", &mut f1)?;
        st2.read_f32("alpha", &mut f2)?;
        assert_eq!(f1, f2);
        let (mut r1, mut r2) = ([0u8; 4], [0u8; 4]);
        st1.read_raw("beta.qs", &mut r1)?;
        st2.read_raw("beta.qs", &mut r2)?;
        assert_eq!(r1, r2);

        // duplicate shard name across dirs → hard error naming both homes
        write_safetensors_named(&prim, "out-00001.safetensors", &[b()])?;
        let Err(err) = SafeTensors::open(&prim) else {
            return Err(Error::Format("duplicate shard name was accepted".into()));
        };
        assert!(format!("{err:?}").contains("two model directories"));

        for d in [&both, &prim, &sec] {
            std::fs::remove_dir_all(d)?;
        }
        Ok(())
    }

    #[test]
    fn region_direct_matches_region_or_none() -> Result<(), Error> {
        // The O_DIRECT twin must expose the same (offset, length) as the buffered
        // region on a different fd — or be `None` (consistent with has_any_direct)
        // when the filesystem rejects O_DIRECT.
        let dir = tmpdir("regiondirect");
        write_safetensors(&dir, &[Blob { name: "w.qs", dtype: "U8", shape: vec![8], bytes: vec![1, 2, 3, 4, 5, 6, 7, 8] }])?;
        let st = SafeTensors::open(&dir)?;
        let buf = st.region("w.qs").ok_or_else(|| Error::Format("no region".into()))?;
        match st.region_direct("w.qs") {
            None => assert!(!st.has_any_direct(), "None region_direct ⇒ no direct fds"),
            Some(d) => {
                assert!(st.has_any_direct());
                assert_eq!(d.1, buf.1, "same offset");
                assert_eq!(d.2, buf.2, "same length");
                assert_ne!(d.0, buf.0, "twin uses a different fd");
            }
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn slice_read() -> Result<(), Error> {
        let dir = tmpdir("slice");
        write_safetensors(
            &dir,
            &[Blob { name: "x", dtype: "F32", shape: vec![6], bytes: f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) }],
        )?;
        let st = SafeTensors::open(&dir)?;
        let mut slice = [0f32; 3];
        st.read_slice_f32("x", 2, 3, &mut slice)?;
        assert_eq!(slice, [2.0, 3.0, 4.0]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn rejects_truncated_header() -> Result<(), Error> {
        let dir = tmpdir("bad");
        std::fs::create_dir_all(&dir)?;
        // declare an 8 GB header in a tiny file
        let mut out = Vec::new();
        out.extend_from_slice(&(8u64 << 30).to_le_bytes());
        out.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), out)?;
        assert!(SafeTensors::open(&dir).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

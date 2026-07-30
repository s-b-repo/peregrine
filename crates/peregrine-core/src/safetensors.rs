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
        let fd = self.files[file_idx].as_raw_fd();
        // parking_lot mutex does not poison, so the lock never fails
        self.reactor
            .lock()
            .read_exact(fd, off, buf)
            .ctx(|| format!("{}: io_uring read @ {off}", self.paths[file_idx].display()))
    }
}

impl SafeTensors {
    /// Index every `model*.safetensors` shard in `dir` (sorted by name, matching
    /// the C engine's ordering so fused-expert offsets line up across shards).
    pub fn open(dir: &Path) -> Result<SafeTensors, Error> {
        let mut shard_paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir).ctx(|| dir.display().to_string())? {
            // a failed directory entry is surfaced, not silently dropped
            let path = entry.ctx(|| dir.display().to_string())?.path();
            if path.extension().is_some_and(|x| x == "safetensors") {
                shard_paths.push(path);
            }
        }
        shard_paths.sort();
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

            let mut lenbuf = [0u8; 8];
            read(&mut reactor, 0, &mut lenbuf)?;
            let hlen = u64::from_le_bytes(lenbuf);
            if fsz < 8 || hlen > fsz - 8 || hlen > ST_MAX_HEADER {
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
                let dtype = Dtype::from_str(dt)
                    .ok_or_else(|| Error::Format(format!("unsupported dtype: {dt}")))?;
                let a0 = offs[0].as_i64().unwrap_or(-1);
                let b0 = offs[1].as_i64().unwrap_or(-1);
                if a0 < 0 || b0 < a0 || data_start as i64 + b0 > fsz as i64 {
                    return Err(Error::Format(format!(
                        "{}: tensor '{name}' data_offsets [{a0},{b0}] out of file bounds ({fsz})",
                        path.display()
                    )));
                }
                let shape: Vec<i64> = shp.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
                let numel: i64 = shape.iter().product::<i64>().max(if shape.is_empty() { 1 } else { 0 });
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
                index.insert(name.clone(), idx);
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
    pub fn nbytes(&self, name: &str) -> Option<i64> {
        self.find(name).map(|t| t.nbytes)
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
                .map_err(|e| Error::Format(format!("read_f32 '{name}' decompress: {e}")))?,
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
        if out.len() < need {
            return Err(Error::Format(format!(
                "read_raw '{name}': out buffer {} < nbytes {need}",
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
                let raw = decode(&disk, other, need).map_err(|e| Error::Format(format!("read_raw '{name}' decompress: {e}")))?;
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
        let esz = t.dtype.elem_size() as i64;
        let boff = t.off + (elem_off * esz) as u64;
        let nb = (n_elems * esz) as usize;
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
        let _ = peregrine_io::advise_hugepages_slice(buf);
    }
}

fn convert_f32(dtype: Dtype, raw: &[u8], out: &mut [f32]) -> Result<(), Error> {
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
        std::fs::write(dir.join("model.safetensors"), out)?;
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
        let _ = std::fs::remove_dir_all(&d);
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

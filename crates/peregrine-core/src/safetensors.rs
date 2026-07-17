//! On-demand safetensors indexing and reading — the Rust port of `c/st.h`.
//!
//! Like the C engine this uses positioned reads (`pread`) rather than mmap, so
//! tensor pages do not stay resident in the process (the RSS fix). O_DIRECT
//! twin fds and `fadvise(DONTNEED)` streaming belong to the M2 I/O lane
//! (`peregrine-io`); this crate is the index plus straightforward converting reads.

use crate::dtype::{bf16_to_f32, f16_to_f32, Dtype};
use crate::Error;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
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
    pub nbytes: i64,
    pub dtype: Dtype,
    pub numel: i64,
    pub shape: Vec<i64>,
}

/// Index over all `*.safetensors` shards in a model directory.
pub struct SafeTensors {
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    files: Vec<File>,
    paths: Vec<PathBuf>,
}

impl SafeTensors {
    /// Index every `model*.safetensors` shard in `dir` (sorted by name, matching
    /// the C engine's ordering so fused-expert offsets line up across shards).
    pub fn open(dir: &Path) -> Result<SafeTensors, Error> {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::Format(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(Error::Format(format!("no .safetensors shards in {}", dir.display())));
        }

        let mut st = SafeTensors {
            tensors: Vec::new(),
            index: HashMap::new(),
            files: Vec::with_capacity(shard_paths.len()),
            paths: shard_paths.clone(),
        };

        for (file_idx, path) in shard_paths.iter().enumerate() {
            let f = File::open(path).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
            let fsz = f.metadata().map_err(Error::Io)?.len();

            let mut lenbuf = [0u8; 8];
            read_exact_at(&f, &mut lenbuf, 0, path)?;
            let hlen = u64::from_le_bytes(lenbuf);
            if fsz < 8 || hlen > fsz - 8 || hlen > ST_MAX_HEADER {
                return Err(Error::Format(format!(
                    "{}: bad safetensors header length {hlen} (file {fsz} bytes)",
                    path.display()
                )));
            }

            let mut hdr = vec![0u8; hlen as usize];
            read_exact_at(&f, &mut hdr, 8, path)?;
            let data_start: u64 = 8 + hlen;
            let root: Value = serde_json::from_slice(&hdr)
                .map_err(|e| Error::Format(format!("{}: header not JSON: {e}", path.display())))?;
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
                let idx = st.tensors.len();
                st.tensors.push(TensorInfo {
                    name: name.clone(),
                    file_idx,
                    off: data_start + a0 as u64,
                    nbytes: b0 - a0,
                    dtype,
                    numel,
                    shape,
                });
                st.index.insert(name.clone(), idx);
            }
            st.files.push(f);
        }
        Ok(st)
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
        let mut raw = vec![0u8; t.nbytes as usize];
        read_exact_at(&self.files[t.file_idx], &mut raw, t.off, &self.paths[t.file_idx])?;
        convert_f32(t.dtype, &raw, &mut out[..need]);
        Ok(t.numel)
    }

    /// Read the raw bytes of a tensor (no dtype conversion) — for the already
    /// int4/int8/int2-quantized U8 container payloads. `out` must be `nbytes`.
    pub fn read_raw(&self, name: &str, out: &mut [u8]) -> Result<(), Error> {
        let t = self.tensor(name)?;
        let need = t.nbytes as usize;
        if out.len() < need {
            return Err(Error::Format(format!(
                "read_raw '{name}': out buffer {} < nbytes {need}",
                out.len()
            )));
        }
        read_exact_at(&self.files[t.file_idx], &mut out[..need], t.off, &self.paths[t.file_idx])
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
        let mut raw = vec![0u8; nb];
        read_exact_at(&self.files[t.file_idx], &mut raw, boff, &self.paths[t.file_idx])?;
        convert_f32(t.dtype, &raw, &mut out[..n_elems as usize]);
        Ok(())
    }

    fn tensor(&self, name: &str) -> Result<&TensorInfo, Error> {
        self.find(name).ok_or_else(|| Error::Format(format!("missing tensor: {name}")))
    }
}

fn convert_f32(dtype: Dtype, raw: &[u8], out: &mut [f32]) {
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
        Dtype::U8 => unreachable!("convert_f32 called on U8"),
    }
}

fn read_exact_at(f: &File, buf: &mut [u8], off: u64, path: &Path) -> Result<(), Error> {
    f.read_exact_at(buf, off)
        .map_err(|e| Error::Format(format!("{}: read {} bytes @ {off}: {e}", path.display(), buf.len())))
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
    pub fn write_safetensors(dir: &Path, blobs: &[Blob]) {
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
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&data);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
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
    fn index_and_read_roundtrip() {
        let dir = tmpdir("roundtrip");
        write_safetensors(
            &dir,
            &[
                Blob { name: "a", dtype: "F32", shape: vec![2], bytes: f32_bytes(&[1.0, 2.0]) },
                Blob { name: "b", dtype: "BF16", shape: vec![3], bytes: bf16_bytes(&[1.0, 2.0, -4.0]) },
                Blob { name: "w.qs", dtype: "U8", shape: vec![4], bytes: vec![10, 20, 30, 40] },
            ],
        );
        let st = SafeTensors::open(&dir).unwrap();
        assert_eq!(st.len(), 3);
        assert!(st.has("a") && st.has("b") && st.has("w.qs"));
        assert_eq!(st.numel("b"), Some(3));

        let mut a = [0f32; 2];
        st.read_f32("a", &mut a).unwrap();
        assert_eq!(a, [1.0, 2.0]);

        let mut b = [0f32; 3];
        st.read_f32("b", &mut b).unwrap();
        assert_eq!(b, [1.0, 2.0, -4.0]);

        let mut raw = [0u8; 4];
        st.read_raw("w.qs", &mut raw).unwrap();
        assert_eq!(raw, [10, 20, 30, 40]);

        // reading a U8 tensor as f32 is an error
        let mut junk = [0f32; 4];
        assert!(st.read_f32("w.qs", &mut junk).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slice_read() {
        let dir = tmpdir("slice");
        write_safetensors(
            &dir,
            &[Blob { name: "x", dtype: "F32", shape: vec![6], bytes: f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) }],
        );
        let st = SafeTensors::open(&dir).unwrap();
        let mut slice = [0f32; 3];
        st.read_slice_f32("x", 2, 3, &mut slice).unwrap();
        assert_eq!(slice, [2.0, 3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_truncated_header() {
        let dir = tmpdir("bad");
        std::fs::create_dir_all(&dir).unwrap();
        // declare an 8 GB header in a tiny file
        let mut out = Vec::new();
        out.extend_from_slice(&(8u64 << 30).to_le_bytes());
        out.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
        assert!(SafeTensors::open(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

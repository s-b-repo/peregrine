//! Shared safetensors file writing: header construction plus the durable
//! stream-fsync-rename commit ceremony.
//!
//! Factored out of `requant`'s [`ShardWriter`](crate::requant::ShardWriter)
//! flush path so `peregrine-reshard` can write exact-named files without
//! duplicating it — and, unlike the buffering writer, *stream* payloads: the
//! reshard tool knows every piece's size up front from the source index, so the
//! header can be written first and the payload copied through a bounded chunk
//! buffer instead of holding a whole shard in RAM (this box is RAM-contended;
//! a (layer, group) file of GLM-5.2 experts is ~1.6 GB).

use peregrine_core::{Context, Error};
use std::io::Write;
use std::path::Path;

/// One tensor's header entry for a file about to be written. `nbytes` is the
/// on-disk payload length; data offsets are derived from the piece order, so
/// the payload must be streamed in exactly this order.
pub struct PieceMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub nbytes: u64,
    /// Extra header fields carried verbatim (`compression`,
    /// `uncompressed_nbytes`, `layout`, `layout_gs_bytes`). A verbatim copier
    /// must forward these: dropping a `layout` tag would make the reader treat
    /// a kblock-tiled payload as row-major bytes, and dropping a `compression`
    /// tag would hand zstd frames to the dequantizer as weights.
    pub extra: Vec<(String, serde_json::Value)>,
}

/// Serialize the safetensors header JSON (without the 8-byte length prefix)
/// for `pieces` in order, with optional `__metadata__` entries.
pub fn header_bytes(meta: &[(String, String)], pieces: &[PieceMeta]) -> Result<Vec<u8>, Error> {
    let mut header = serde_json::Map::new();
    if !meta.is_empty() {
        let mut m = serde_json::Map::new();
        for (k, v) in meta {
            m.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(m));
    }
    let mut cursor: u64 = 0;
    for p in pieces {
        let start = cursor;
        let end = start
            .checked_add(p.nbytes)
            .ok_or_else(|| Error::Format(format!("'{}': data offsets overflow u64", p.name)))?;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String(p.dtype.clone()));
        entry.insert("shape".into(), serde_json::json!(p.shape));
        entry.insert("data_offsets".into(), serde_json::json!([start, end]));
        for (k, v) in &p.extra {
            entry.insert(k.clone(), v.clone());
        }
        if header.insert(p.name.clone(), serde_json::Value::Object(entry)).is_some() {
            // `SafeTensors::open` rejects duplicate names across shards; within
            // one file the later entry would silently shadow the earlier one.
            return Err(Error::Format(format!("'{}': tensor listed twice in one output file", p.name)));
        }
        cursor = end;
    }
    serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| Error::Format(format!("serialize safetensors header: {e}")))
}

/// Counts what flows through so a piece writing the wrong number of bytes is an
/// error, not a silent shift of every following tensor's offsets.
struct CountingWriter<W: Write> {
    inner: W,
    n: u64,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let k = self.inner.write(buf)?;
        self.n += k as u64;
        Ok(k)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write one complete safetensors file at `path`: header first, then each
/// piece's payload streamed through `fill(piece_index, writer)`.
///
/// The file is assembled as `<path>.part`, fsynced, then atomically renamed —
/// so a file that exists is whole (the same durability contract `ShardWriter`
/// has always had, and the property resume/verify logic leans on). `fill` must
/// write exactly `pieces[i].nbytes` bytes; the count is enforced.
pub fn write_streaming(
    path: &Path,
    meta: &[(String, String)],
    pieces: &[PieceMeta],
    mut fill: impl FnMut(usize, &mut dyn Write) -> Result<(), Error>,
) -> Result<(), Error> {
    let hdr = header_bytes(meta, pieces)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ctx(|| format!("create {}", dir.display()))?;
    }
    let tmp = path.with_extension("safetensors.part");
    {
        let f = std::fs::File::create(&tmp).ctx(|| format!("create {}", tmp.display()))?;
        let mut w =
            CountingWriter { inner: std::io::BufWriter::with_capacity(1 << 20, f), n: 0 };
        w.write_all(&(hdr.len() as u64).to_le_bytes()).ctx(|| "write header length".to_string())?;
        w.write_all(&hdr).ctx(|| "write header".to_string())?;
        let mut expected = 8 + hdr.len() as u64;
        for (i, p) in pieces.iter().enumerate() {
            fill(i, &mut w)?;
            expected += p.nbytes;
            if w.n != expected {
                return Err(Error::Format(format!(
                    "'{}': payload wrote {} bytes, header declares {} — refusing to commit a \
                     file whose offsets are already wrong",
                    p.name,
                    (w.n + p.nbytes).saturating_sub(expected),
                    p.nbytes
                )));
            }
        }
        let f = w.inner.into_inner().map_err(|e| Error::Format(format!("flush {}: {e}", tmp.display())))?;
        // Durability before the rename: a file that exists must be complete.
        f.sync_all().ctx(|| format!("fsync {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).ctx(|| format!("commit {}", path.display()))?;
    Ok(())
}

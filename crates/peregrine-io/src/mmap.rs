use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

#[derive(Clone)]
pub struct MmapFile {
    map: Arc<Mmap>,
}

impl MmapFile {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self { map: Arc::new(map) })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::MmapFile;
    use std::path::PathBuf;

    #[test]
    fn maps_a_regular_file_and_keeps_clones_alive() -> std::io::Result<()> {
        let path = std::env::temp_dir().join(format!(
            "peregrine-mmap-{}-{}.bin",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, b"mapped bytes")?;
        let first = MmapFile::open(&path)?;
        let second = first.clone();
        assert_eq!(first.as_bytes(), b"mapped bytes");
        assert_eq!(second.as_bytes(), first.as_bytes());
        assert_eq!(first.len(), second.len());
        assert!(!first.is_empty());
        std::fs::remove_file(path)
    }

    #[test]
    fn rejects_a_missing_file() {
        let path = PathBuf::from("/definitely/not/a/peregrine/mmap/file");
        assert!(MmapFile::open(&path).is_err());
    }
}

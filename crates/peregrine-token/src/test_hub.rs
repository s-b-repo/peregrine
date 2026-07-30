//! Test-only lookup of HuggingFace-cached tokenizer files — a filesystem-only
//! shim of upstream's test_hub (the hub *download* module is not vendored).
//! Nothing is ever downloaded: tests skip when a file is absent, except GPT-2,
//! which falls back to the committed fixture in tests/fixtures.

use std::path::PathBuf;

/// `filename` from a model repo's `main` snapshot in the local HF cache
/// (`~/.cache/huggingface/hub` or `$HF_HOME/hub`), or `None` when the repo,
/// ref, or file is not cached.
pub(crate) fn cached_hub_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    let hub_root = std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))?;
    let repo_dir = hub_root.join(format!("models--{}", repo_id.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs/main")).ok()?;
    let path = repo_dir.join("snapshots").join(commit.trim()).join(filename);
    path.exists().then_some(path)
}

/// A model repo's tokenizer.json from the local HF cache.
pub(crate) fn hf_tokenizer_json(repo_id: &str) -> Option<PathBuf> {
    cached_hub_file(repo_id, "tokenizer.json")
}

/// GPT-2's tokenizer.json: the HF cache copy when present, else the committed
/// fixture (a verbatim copy of the openai-community/gpt2 file).
pub(crate) fn gpt2_tokenizer_json() -> PathBuf {
    hf_tokenizer_json("openai-community/gpt2").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt2_tokenizer.json")
    })
}

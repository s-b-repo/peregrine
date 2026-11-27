# Changelog

All notable changes to this project are documented in this file by [Release Please].

## [0.1.4](https://github.com/s-b-repo/peregrine/compare/v0.1.3...v0.1.4) (2026-11-27)

### Features

- Stream complete transformer layers through a bounded asynchronous `io_uring` window for native MLA, GQA, and hybrid GDN checkpoints. `peregrine-reshard --layer-drive` distributes layers across drive directories and writes `model_paths.json`.
- Load one routed expert at a time with `COLI_STREAM=demand`, including compressed expert weights.
- Use a shared OpenAI-compatible server interface for native and Omni models, with Hugging Face tokenizer fallback and a common launcher.

### Fixes

- Validate completion limits, request bodies, authentication, and Omni conversation history.
- Harden persisted state and KV cache writes, handle interrupted kernel I/O, and update Tokio to 1.54.1.
- Configure Release Please to update crate versions and the lockfile within a virtual Cargo workspace.

[Release Please]: https://github.com/googleapis/release-please

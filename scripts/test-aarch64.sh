#!/usr/bin/env bash
# Build and RUN the CPU kernels as real aarch64 code, on an x86_64 host.
#
# Why this exists: `peregrine-kernels` is the token-exactness anchor, and its
# NEON/`sdot` kernels must produce the identical i32 accumulator to the scalar
# reference. Compiling for ARM proves nothing about that — the equivalence tests
# have to actually execute ARM instructions. This script makes that possible
# without ARM hardware, so the aarch64 path is covered by the same assertions as
# the x86 one instead of shipping on reasoning.
#
# How it works, and why each piece is needed:
#   * target `aarch64-unknown-linux-musl`, not `-gnu`: the musl target ships
#     self-contained crt objects and a static libc, so no cross C toolchain is
#     required. `peregrine-kernels` has zero dependencies, so nothing pulls in C.
#   * linker `rust-lld`, shipped with the toolchain — again, no cross gcc.
#   * runner `qemu-aarch64` (user-mode emulation) executes the test binary.
#
# Requirements: `rustup target add aarch64-unknown-linux-musl` and a
# `qemu-aarch64` binary on PATH (Arch: `qemu-user`; Debian: `qemu-user-static`).
#
# NOTE ON TIMING: this verifies CORRECTNESS only. Emulated wall-clock says
# nothing about throughput on real ARM silicon, so do not quote a speedup from
# it — see docs/measurement.md.
set -euo pipefail

TARGET=aarch64-unknown-linux-musl

if ! rustup target list --installed | grep -qx "$TARGET"; then
  echo "missing target; run: rustup target add $TARGET" >&2
  exit 1
fi
if ! command -v qemu-aarch64 > /dev/null; then
  echo "missing qemu-aarch64 (Arch: pacman -S qemu-user; Debian: apt install qemu-user-static)" >&2
  exit 1
fi

LLD=$(find "$(rustc --print sysroot)" -name rust-lld -type f | head -1)
if [ -z "$LLD" ]; then
  echo "rust-lld not found in $(rustc --print sysroot)" >&2
  exit 1
fi

export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER="$LLD"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C linker-flavor=ld.lld -C link-self-contained=yes -C target-feature=+crt-static"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64"

# `-p peregrine-kernels` only: the rest of the workspace pulls zstd-sys and other
# C crates, which would need a cross C toolchain this script deliberately avoids.
exec cargo test -p peregrine-kernels --target "$TARGET" "$@"

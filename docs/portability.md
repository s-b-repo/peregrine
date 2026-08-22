[« Docs index](README.md)

# Portability: what varies from one Linux box to the next

peregrine targets **Linux**, but "Linux" is not one machine. This page records
which machine-to-machine differences the engine actually handles, which it
handles badly, and — the part that matters most — **which claims here are tested
and which are only reasoned**.

Everything below was established on one host: x86_64, 4 KB pages, io_uring
available, no GPU in the test build. The *fallback paths* are exercised on this
box by forcing them, and the aarch64 kernels are executed under emulation
(`scripts/test-aarch64.sh`). What has **not** happened is a run on real
non-x86_64 silicon, so every performance claim here is scoped to x86_64.

## Matrix

| Surface | What varies | What the engine does | Tested? |
|---|---|---|---|
| **io_uring** | absent on kernels < 5.1, `kernel.io_uring_disabled=2`, seccomp-blocked containers (Docker's default profile blocks `io_uring_setup`) | Probes once, prints one line, resolves the engine to `pread`. Loading, streaming, and metadata reads all have ring-free paths; **zero** reactors are constructed | **Yes** — `streaming_runs_with_zero_rings` re-execs with `COLI_IO_ENGINE=pread`, asserts the `rings=0 engine=pread` boot line, and requires logits identical to the resident path |
| **Page size** | 4 KB on x86_64; aarch64 kernels are commonly built 16 KB or 64 KB (RHEL/CentOS aarch64 ships 64 KB), ppc64le 64 KB | `page_size()` reads `sysconf(_SC_PAGESIZE)` once. Used by the RSS guard and by `madvise` range narrowing | **Yes** — `page_size_is_a_sane_power_of_two`, `narrowing_stays_inside_the_range_and_lands_on_page_boundaries` |
| **CPU SIMD** | AVX2 / AVX-VNNI on x86_64; NEON / `dotprod` on aarch64 | Runtime-detected on both. aarch64 gets `sdot` where the CPU has ARMv8.2 dot-product, `smull`/`sadalp` NEON otherwise, scalar below that | **Yes, on real ARM instructions** — `scripts/test-aarch64.sh` runs the equivalence suite under `qemu-aarch64`, including a guard that the dispatcher did not silently pick scalar. Throughput on real ARM silicon is **unmeasured** |
| **O_DIRECT** | unsupported on tmpfs, overlayfs, many network filesystems | Probed per shard at open (`probe_direct`); any failure is a buffered fallback, never fatal. Also forced off under the `pread` engine, which has no aligned buffers | Partly — the probe is exercised, ringless O_DIRECT is *disabled* rather than tested |
| **NUMA** | single-node desktops vs multi-socket servers | Opt-in (`COLI_NUMA_PIN`); inert and harmless when absent | Yes — `numa_primitives_are_inert_unless_the_knob_is_on` |
| **RAPL energy** | root-only on current kernels (PLATYPUS mitigation); **does not exist on ARM at all** | `energy_uj` reads `null`, deliberately not `0` — no permission and no energy are different facts | Yes on this box (reads `null` unprivileged) |
| **GPU** | present / absent / different vendor | Optional; the default build is CPU-only and the whole suite runs without one | Yes — this is the normal build here |
| **Hugepages** | THP on / off / madvise-only | Advisory `MADV_HUGEPAGE`; a refusal returns `false` and changes nothing | Yes |
| **Core count / RAM** | 4-core laptop to 128-core server | Workers default from `logical_cpus()`; the warm cache sizes from `MemAvailable` | Partly — sized here, not swept |

## ARM SIMD: implemented, and verified without ARM hardware

`peregrine-kernels` used to carry **13 `target_arch = "x86_64"` gates and zero
for aarch64**, so every int dot product on ARM fell through to the scalar
reference. It now has both:

| kernel | x86_64 | aarch64 |
|---|---|---|
| int8·int8 | `vpdpbusd` (VNNI) / `maddubs` (AVX2) | `sdot` (ARMv8.2 dotprod) / `smull`+`sadalp` (NEON) |
| int4·int8 | `maddubs` (AVX2) | `sdot` / `smull`+`sadalp` |
| int2·int8 | `maddubs` (AVX2) | `sdot` |

`sdot` is the direct analogue of VNNI's `vpdpbusd` and is *easier* to use —
signed×signed→i32, so none of the `maddubs` sign-trick ceremony is needed.
`dot_i4i8_grouped` delegates per group to `dot_i4i8`, and `matmul.rs` has no arch
gates of its own, so GLM-5.2's hot path (int4 weights with group scales) picks
this up with no further work.

Two implementation notes worth keeping:

- **`sdot` is emitted as inline `asm!`, not `vdotq_s32`.** That intrinsic is
  still nightly-only (`stdarch_neon_dotprod`,
  [rust-lang/rust#117224](https://github.com/rust-lang/rust/issues/117224)) and
  this workspace is stable-only. `asm!` is stable, so the instruction is
  reachable without a nightly toolchain.
- **The int4 nibble unpack matches the x86 one exactly.** `vzip1q_u8`/`vzip2q_u8`
  are the NEON spelling of `_mm_unpacklo_epi8`/`_mm_unpackhi_epi8`, which is what
  restores element order from the interleaved low/high-nibble layout.

### How it is verified without an ARM box

This was the actual blocker, not writing the kernels: the scalar path is the
engine's token-exactness anchor, and a SIMD kernel that disagrees with it
produces silently wrong output rather than a crash. Compiling for ARM proves
nothing about that.

`scripts/test-aarch64.sh` executes the equivalence suite as **real aarch64
instructions** on an x86_64 host:

```bash
scripts/test-aarch64.sh     # 16 tests, ~1s
```

It works because `peregrine-kernels` has **zero dependencies**, so the
`aarch64-unknown-linux-musl` target (self-contained crt objects, static libc)
links with the toolchain's own `rust-lld` — no cross C toolchain — and
`qemu-aarch64` user-mode emulation runs the result. The suite asserts each ARM
kernel against the scalar reference *and* asserts `cpu_kernel_tier()` reports a
NEON backend, so a passing run cannot mean "both sides ran the scalar path".

### What is still not known

**Throughput on real ARM silicon.** Emulated wall-clock says nothing about IPC,
cache behaviour or memory bandwidth on a Graviton or Ampere part, so no speedup
number is quoted here — per [measurement.md](measurement.md), a number nobody
measured is worse than no number. What is established is that the vector kernels
are *correct* and *reached*; what they are worth is an open measurement on
hardware this project does not have.

## Also known

- **`peregrine-sched::Streamer` still requires a ring** and documents itself as
  such. It has no constructor anywhere outside its own crate, so nothing on the
  serving path reaches it; adding a fallback to code nothing calls would be
  motion, not progress.
- **Prefetch is now non-fatal.** Each lane wants its own ring, and a pool that
  will not spawn used to fail the load. Prefetch is speculative warming — every
  expert it loads is re-read correctly on a miss — so a failure now logs an
  advisory and the engine runs without it.
- **`COLI_DIRECT=1` under `pread` reports "off"**, with the reason, rather than
  taking `EINVAL` on every read.

## If you are bringing this up on a new machine

```bash
cargo test --workspace          # the fallbacks are covered here, on any host
target/release/peregrine demo   # synthetic model, no checkpoint needed
```

Then read the boot lines. They are the cheapest evidence of what you actually
got:

```
peregrine: [io] rings=4 sqpoll=off fixed_files=registered   # io_uring path
peregrine: [io] rings=0 engine=pread (no io_uring) threads=8 # fallback path
peregrine: O_DIRECT streaming off — the pread engine has no aligned buffers; buffered fallback
```

A silent degrade that halves throughput is worse than a loud one, so each of
these prints unconditionally when it applies.

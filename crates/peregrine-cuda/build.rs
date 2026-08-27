//! Compile and link the existing, validated GPU kernels (`cuda/backend_cuda.cu`)
//! when the `cuda` feature is on. No-op otherwise, so the default workspace
//! build needs neither nvcc nor a GPU. Mirrors `c/Makefile` (CUDA=1 path).
//!
//! **One kernel source, two vendors.** The kernels are written against the CUDA
//! runtime API, which is also what ROCm's HIP targets: the same file compiles
//! under `nvcc` (NVIDIA) or — after a mechanical `hipify-perl` pass — under
//! `hipcc` (AMD). The host ABI (`backend_cuda.h`, every `coli_cuda_*` symbol) is
//! identical either way, so the Rust FFI and everything above it are
//! vendor-agnostic; only this script and the linked runtime library differ.
//!
//! Vendor selection:
//! - `PEREGRINE_GPU_BACKEND=cuda|hip` forces one;
//! - unset/`auto` prefers whatever toolchain is present (nvcc first);
//! - `intel` is recognized and REJECTED here: no SYCL port of the kernels
//!   exists yet (see `docs/gpu-vendors.md`), and silently building nothing
//!   would read as support.

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return; // feature off → pure-CPU build, nothing to do
    }
    let Ok(out) = std::env::var("OUT_DIR") else {
        // OUT_DIR is always set by cargo for build scripts; surface it as a warning
        // rather than panicking if a non-cargo invocation ever omits it.
        println!("cargo:warning=OUT_DIR unset; skipping GPU backend compile");
        return;
    };

    let requested = std::env::var("PEREGRINE_GPU_BACKEND").unwrap_or_else(|_| "auto".to_string());
    let linked = match requested.as_str() {
        "cuda" => build_cuda(&out),
        "hip" => build_hip(&out),
        "auto" => {
            if nvcc_present() {
                build_cuda(&out)
            } else {
                build_hip(&out)
            }
        }
        other => panic!(
            "PEREGRINE_GPU_BACKEND={other} is not a backend this build can produce \
             (cuda | hip; intel is designed but not ported — see docs/gpu-vendors.md)"
        ),
    };
    write_backend_name(&out, linked);
}

/// Emit the compiled-backend identity for [`status()`] at runtime.
fn write_backend_name(out: &str, linked: &str) {
    let text =
        format!("/// Written by build.rs — which vendor's runtime was actually linked.\npub const BACKEND: &str = {linked:?};\n");
    let _ = std::fs::write(format!("{out}/vendor.rs"), text); // OUT_DIR was writable when we got this far
}

#[allow(clippy::needless_bool)]
fn nvcc_present() -> bool {
    let home = std::env::var("CUDA_HOME").unwrap_or_else(|_| detect_cuda_home());
    Path::new(&format!("{home}/bin/nvcc")).exists()
}

// ---------------- NVIDIA: nvcc compiles the source directly ----------------

/// Returns the linked backend name for the banner, or "" when nothing was.
fn build_cuda(out: &str) -> &'static str {
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| detect_cuda_home());
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    // repo layout: rust/crates/peregrine-cuda/build.rs → ../../cuda/backend_cuda.cu
    let src = "../../cuda/backend_cuda.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let obj = format!("{out}/backend_cuda.o");
    let nvcc = format!("{cuda_home}/bin/nvcc");

    // "no toolkit here" and "the kernel source does not compile" are different
    // facts and must not produce the same outcome. Collapsing them — which this
    // script did until 2026-08-06, by mapping both onto one `cargo:warning` and
    // a success exit — means a `.cu` syntax error is indistinguishable from a
    // CPU-only host, on a host that *has* nvcc. Nothing greps build warnings, so
    // every edit to `backend_cuda.cu` went unverified and the build stayed green.
    // Absent toolkit stays a warning (a pure-CPU host must still build the
    // workspace); a toolkit that is present and rejects the source is a hard
    // failure, because on that host it is a real compile error.
    if !Path::new(&nvcc).exists() {
        println!("cargo:warning=nvcc not found at {nvcc}; CUDA backend NOT linked (build on an NVIDIA host with CUDA installed, or set CUDA_HOME)");
        return "";
    }

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "-std=c++17",
            &format!("-arch={arch}"),
            "-Xcompiler",
            "-fPIC",
            "-c",
            src,
            "-o",
            &obj,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("{nvcc} failed to compile {src} ({s}) — this host has a CUDA toolkit, so this is a compile error in the kernel source, not a missing toolchain"),
        Err(e) => panic!("{nvcc} exists but could not be executed: {e}"),
    }

    archive(out, &obj);
    // cudart lives in `lib64` on standard installs and `targets/<triple>/lib`
    // on Arch (`lib64` is a symlink there, but emit both so a missing symlink
    // still links). Only existing dirs are emitted to avoid linker noise.
    for cand in [
        format!("{cuda_home}/lib64"),
        format!("{cuda_home}/targets/x86_64-linux/lib"),
    ] {
        if Path::new(&cand).exists() {
            println!("cargo:rustc-link-search=native={cand}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    "CUDA (NVIDIA)"
}

/// Locate the CUDA toolkit root when `CUDA_HOME` is unset: prefer the
/// conventional `/usr/local/cuda`, else derive it from `nvcc` on `PATH`
/// (`<root>/bin/nvcc`), else fall back to the Arch default `/opt/cuda`.
fn detect_cuda_home() -> String {
    if Path::new("/usr/local/cuda/bin/nvcc").exists() {
        return "/usr/local/cuda".to_string();
    }
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout);
            let nvcc = Path::new(p.trim());
            // <root>/bin/nvcc → <root>
            if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
                return root.to_string_lossy().into_owned();
            }
        }
    }
    "/opt/cuda".to_string()
}

// ---------------- AMD: hipify-perl then hipcc, same source ----------------

const HIPIFY: &str = "hipify-perl";

fn build_hip(out: &str) -> &'static str {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rerun-if-changed=../../cuda/backend_cuda.cu");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=PEREGRINE_GPU_BACKEND");
    println!("cargo:rerun-if-env-changed=HIPC_ARCH");

    let hipcc = format!("{rocm}/bin/hipcc");
    let hipify = format!("{rocm}/bin/{HIPIFY}");
    let have_rocm = Path::new(&hipcc).exists();

    // Same philosophy as the nvcc branch: an explicitly requested HIP build on a
    // host without ROCm is a warning-and-no-backend (the workspace must keep
    // building everywhere), never a silent "supported".
    if !have_rocm {
        println!(
            "cargo:warning=hipcc not found at {hipcc}; AMD/HIP backend NOT linked \
             (install ROCm or set ROCM_PATH)"
        );
        return "";
    }
    if !Path::new(&hipify).exists() {
        println!(
            "cargo:warning={hipify} not found in {rocm}/bin; AMD/HIP backend NOT linked \
             (hipify-perl ships with ROCm's hip-extras)"
        );
        return "";
    }

    // 1. Mechanical CUDA→HIP translation of the kernel source into OUT_DIR —
    //    the vendored file stays untouched.
    let hip_src = format!("{out}/backend_hip.cpp");
    let translated = Command::new(&hipify).args(["../../cuda/backend_cuda.cu", "-o", &hip_src]).status();
    match translated {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("{hipify} failed on ../../cuda/backend_cuda.cu ({s})"),
        Err(e) => panic!("{hipify} exists but could not be executed: {e}"),
    }

    // 2. Compile the translated source for the local or named AMD target(s).
    //    `HIPC_ARCH` may name several (`gfx1100;gfx90a`). Unset, ask the
    //    installed ROCm which agents are present (`rocm_agent_enumerator`);
    //    a host with ROCm but no AMD device — the compile-verification case —
    //    gets a broad CDNA+RDNA set instead of `native`, which hipcc rejects
    //    outright when there is no device to inspect.
    let obj = format!("{out}/backend_hip.o");
    let arch = std::env::var("HIPC_ARCH").unwrap_or_else(|_| detect_hip_archs(&rocm));
    let offload: Vec<String> =
        arch.split(';').filter(|a| !a.is_empty()).map(|a| format!("--offload-arch={a}")).collect();
    let mut args: Vec<&str> = vec!["-O3", "-std=c++17", "-fPIC"];
    args.extend(offload.iter().map(String::as_str));
    args.extend(["-c", hip_src.as_str(), "-o", obj.as_str()]);
    let compiled = Command::new(&hipcc).args(&args).status();
    match compiled {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("{hipcc} failed to compile {hip_src} ({s}) — ROCm is installed, so treat this as a compile error in the hipified source"),
        Err(e) => panic!("{hipcc} exists but could not be executed: {e}"),
    }

    archive(out, &obj);
    for cand in [format!("{rocm}/lib"), format!("{rocm}/lib64")] {
        if Path::new(&cand).exists() {
            println!("cargo:rustc-link-search=native={cand}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    "HIP (AMD ROCm)"
}

/// The `;`-separated gfx targets to compile for when `HIPC_ARCH` is unset:
/// what `rocm_agent_enumerator` reports (dropping its gfx000 CPU placeholder),
/// or — no device / no enumerator — a broad current CDNA + RDNA set, so a
/// GPU-less ROCm host still compile-verifies every kernel.
fn detect_hip_archs(rocm: &str) -> String {
    if let Ok(out) = Command::new(format!("{rocm}/bin/rocm_agent_enumerator")).output() {
        if out.status.success() {
            let agents: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .filter(|a| a.starts_with("gfx") && *a != "gfx000")
                .map(str::to_string)
                .collect();
            if !agents.is_empty() {
                return agents.join(";");
            }
        }
    }
    "gfx90a;gfx942;gfx1030;gfx1100".to_string()
}

/// Archive one compiled object into the static lib the Rust FFI links.
fn archive(out: &str, obj: &str) {
    let lib = format!("{out}/libcoli_cuda_backend.a");
    // A failed `ar` leaves no archive, and the link below would then fail with
    // undefined symbols naming the kernels rather than the archiver that never
    // ran — same reasoning as the nvcc branch.
    match Command::new("ar").args(["crus", &lib, obj]).status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("ar crus {lib} failed ({s})"),
        Err(e) => panic!("could not run ar to archive {obj}: {e}"),
    }
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=coli_cuda_backend");
}

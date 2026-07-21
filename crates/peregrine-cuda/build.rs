//! Compile and link the existing, validated CUDA kernels (`cuda/backend_cuda.cu`)
//! when the `cuda` feature is on. No-op otherwise, so the default workspace
//! build needs neither nvcc nor a GPU. Mirrors `c/Makefile` (CUDA=1 path).

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return; // feature off → pure-CPU build, nothing to do
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| detect_cuda_home());
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    // repo layout: rust/crates/peregrine-cuda/build.rs → ../../cuda/backend_cuda.cu
    let src = "../../cuda/backend_cuda.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // OUT_DIR is always set by cargo for build scripts; surface it as a warning
    // rather than panicking if a non-cargo invocation ever omits it.
    let Ok(out) = std::env::var("OUT_DIR") else {
        println!("cargo:warning=OUT_DIR unset; skipping CUDA backend compile");
        return;
    };
    let obj = format!("{out}/backend_cuda.o");
    let nvcc = format!("{cuda_home}/bin/nvcc");

    let compiled = std::process::Command::new(&nvcc)
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
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !compiled {
        println!("cargo:warning=nvcc unavailable or failed; CUDA backend NOT linked (build on an NVIDIA host with CUDA installed)");
        return;
    }

    let lib = format!("{out}/libcoli_cuda_backend.a");
    let _ = std::process::Command::new("ar").args(["crus", &lib, &obj]).status();
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=coli_cuda_backend");
    // cudart lives in `lib64` on standard installs and `targets/<triple>/lib`
    // on Arch (`lib64` is a symlink there, but emit both so a missing symlink
    // still links). Only existing dirs are emitted to avoid linker noise.
    for cand in [
        format!("{cuda_home}/lib64"),
        format!("{cuda_home}/targets/x86_64-linux/lib"),
    ] {
        if std::path::Path::new(&cand).exists() {
            println!("cargo:rustc-link-search=native={cand}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Locate the CUDA toolkit root when `CUDA_HOME` is unset: prefer the
/// conventional `/usr/local/cuda`, else derive it from `nvcc` on `PATH`
/// (`<root>/bin/nvcc`), else fall back to the Arch default `/opt/cuda`.
fn detect_cuda_home() -> String {
    if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
        return "/usr/local/cuda".to_string();
    }
    if let Ok(out) = std::process::Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout);
            let nvcc = std::path::Path::new(p.trim());
            // <root>/bin/nvcc → <root>
            if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
                return root.to_string_lossy().into_owned();
            }
        }
    }
    "/opt/cuda".to_string()
}

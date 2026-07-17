//! Compile and link the existing, validated CUDA kernels (`cuda/backend_cuda.cu`)
//! when the `cuda` feature is on. No-op otherwise, so the default workspace
//! build needs neither nvcc nor a GPU. Mirrors `c/Makefile` (CUDA=1 path).

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return; // feature off → pure-CPU build, nothing to do
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    // repo layout: rust/crates/peregrine-cuda/build.rs → ../../cuda/backend_cuda.cu
    let src = "../../cuda/backend_cuda.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out = std::env::var("OUT_DIR").unwrap();
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
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

//! Microbenchmark for the packed-int4 · int8 dot kernel at the shapes a resident
//! dense model actually runs, so a kernel change is judged on the dimensions it
//! has to be fast at rather than a synthetic one.
//!
//! Qwen3.8-27B's MLP is the dominant consumer (66.7% of per-token weight bytes):
//! `gate`/`up` are `[17408, 5120]` and `down` is `[5120, 17408]`, so the dot runs
//! at `n = 5120` and `n = 17408`. GLM's expert shape (`n = 4096`) is included as
//! the streaming-model reference.
//!
//!   cargo run --release -p peregrine-kernels --example idotbench
//!
//! Prints GB/s of packed weight bytes consumed — the figure to compare across
//! kernel revisions. Correctness is not this file's job: `i4_simd_matches_scalar`
//! owns that, and any kernel change must keep it green.

fn main() {
    // (name, n) — the reduction length, i.e. the input width of the matmul row.
    let shapes = [("qwen mlp gate/up (n=5120)", 5120usize), ("qwen mlp down (n=17408)", 17408), ("glm expert (n=4096)", 4096)];
    for (name, n) in shapes {
        // Deterministic inputs; the kernel's cost does not depend on values.
        let w4: Vec<u8> = (0..n / 2).map(|i| (i * 37 + 11) as u8).collect();
        let x: Vec<i8> = (0..n).map(|i| ((i * 17 + 3) % 251) as i8 - 125).collect();

        // ~4e9 elements per trial so each trial runs order-seconds: a 6 ms timing
        // is noise (frequency ramp, scheduler), which is how a first cut of this
        // bench "showed" a 3x regression that was really jitter.
        let reps = (4_000_000_000u64 / n as u64).max(1000) as usize;
        // Best of three: noise only ever ADDS time, so the minimum is the least
        // contaminated estimate of the kernel's own cost.
        let mut sink = peregrine_kernels::dot_i4i8(&w4, &x, n); // untimed warm-up
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                sink = sink.wrapping_add(peregrine_kernels::dot_i4i8(&w4, &x, n));
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let bytes = (n / 2) as f64 * reps as f64;
        println!(
            "{name:<26} {reps:>8} reps  best {:>6.3} s  {:>7.2} GB/s  {:>8.1} ns/dot  (sink {sink})",
            best,
            bytes / best / 1e9,
            best / reps as f64 * 1e9,
        );
    }
}

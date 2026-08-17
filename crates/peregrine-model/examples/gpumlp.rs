//! Per-layer MLP bench: the CPU path against the VRAM-resident device path, at
//! one real layer's shape, without loading a model.
//!
//!   cargo run --release -p peregrine-model --features cuda --example gpumlp -- [HIDDEN] [INTER] [REPS]
//!
//! Defaults are Qwen3.8-27B's dense layer (5120 x 17408) and 9 reps. One layer's
//! int4 weights are ~134 MB, so this fits VRAM left over by another process —
//! which is the point: it answers "is the device path worth wiring further"
//! before anything needs 8.6 GB of card.
//!
//! **Reports the median and the spread, and takes decode's shape seriously.**
//! Decode is `s_n = 1`, and until 2026-08-17 that was the case where this
//! engine's matmuls ran single-threaded — a bench taken at a larger batch would
//! have measured a path decode never takes and made the CPU look ~6x better
//! than it was. `S` is fixed at 1 here for that reason; change it only if you
//! also change what you claim the number means.
//!
//! What it deliberately does NOT measure: the H2D/D2H copies are inside the
//! timed region because they are inside the real call — `dense_mlp_w4a16`
//! uploads `x` and downloads `y` every invocation. That is a genuine cost of
//! the current design at `s_n = 1` (two ~20 KB transfers per layer per token),
//! not an artifact, and hiding it would flatter the device.

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (hidden, inter, reps) = (n(1, 5120), n(2, 17408), n(3, 9));
    let s_n = 1usize; // decode

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (hidden, inter, reps, s_n);
        eprintln!("gpumlp: built without --features cuda; nothing to compare");
    }

    #[cfg(feature = "cuda")]
    {
        use peregrine_core::pack::quant_i4;
        use peregrine_model::gpu::GpuDenseTier;
        use peregrine_model::mlp::Mlp;
        use peregrine_model::weight::{QtWeight, QuantFmt};

        // Quantized through the CONTAINER's own encoder, so the bytes here are
        // the bytes a real checkpoint holds — including the offset-binary nibble
        // convention the upload path converts on its way to the device.
        let w4 = |w: &[f32], o: usize, i: usize| {
            let (q, sc) = quant_i4(w, o, i);
            QtWeight::new(QuantFmt::Int4, o, i, q, sc)
        };

        let mut seed = 0xC0FFEEu64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        eprintln!("gpumlp: building a {inter}x{hidden} SwiGLU triple (~{:.0} MB int4)",
                  (3.0 * inter as f64 * hidden as f64 / 2.0) / 1e6);
        let gf: Vec<f32> = (0..inter * hidden).map(|_| rnd() * 0.1).collect();
        let uf: Vec<f32> = (0..inter * hidden).map(|_| rnd() * 0.1).collect();
        let df: Vec<f32> = (0..hidden * inter).map(|_| rnd() * 0.1).collect();
        let mlp = Mlp { gate: w4(&gf, inter, hidden), up: w4(&uf, inter, hidden), down: w4(&df, hidden, inter) };
        drop((gf, uf, df));
        let x: Vec<f32> = (0..s_n * hidden).map(|_| rnd()).collect();

        if peregrine_cuda::init(&[0]) < 1 {
            eprintln!("gpumlp: no CUDA device — CPU numbers only");
        }
        let mut tier = GpuDenseTier::new(0);
        let resident = match tier.try_add(0, &mlp.gate, &mlp.up, &mlp.down, 64 << 20) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("gpumlp: upload refused: {e}");
                false
            }
        };
        if !resident {
            let free = peregrine_cuda::mem_info(0).map(|(f, _)| f).unwrap_or(0);
            eprintln!("gpumlp: layer did not fit ({:.2} GB free) — CPU numbers only", free as f64 / 1e9);
        }

        let median = |mut v: Vec<f64>| -> (f64, f64) {
            v.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            let m = v[v.len() / 2];
            let spread = if m > 0.0 { (v[v.len() - 1] - v[0]) / m * 100.0 } else { 0.0 };
            (m, spread)
        };
        let time = |f: &mut dyn FnMut()| -> Vec<f64> {
            f(); // warm: first call pays allocation and, on the device, context setup
            (0..reps)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect()
        };

        let mut sink = 0f32;
        let cpu = time(&mut || {
            let y = mlp.swiglu(&x, s_n);
            sink += y[0];
        });
        let (cpu_ms, cpu_spread) = median(cpu);
        println!("cpu  (int4 w, int8 act): {cpu_ms:8.3} ms  spread {cpu_spread:.0}%");

        if resident {
            let gpu = time(&mut || {
                if let Some(Ok(y)) = tier.mlp(0, &x, s_n, hidden) {
                    sink += y[0];
                }
            });
            let (gpu_ms, gpu_spread) = median(gpu);
            println!("gpu  (int4 w, fp16 act): {gpu_ms:8.3} ms  spread {gpu_spread:.0}%");
            println!("speedup: {:.2}x per layer", cpu_ms / gpu_ms.max(1e-9));
            // The projection everyone actually wants, stated as a projection:
            // one layer's measured gap scaled to a 64-layer stack, MLPs only.
            println!(
                "projection: 64 MLP layers {:.0} ms cpu vs {:.0} ms gpu (MLP work only, attention/GDN excluded)",
                cpu_ms * 64.0,
                gpu_ms * 64.0
            );
        }
        std::hint::black_box(sink);
    }
}

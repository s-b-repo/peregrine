//! Microbenchmark for `Sampler::dist_build`'s truncation: the full vocabulary
//! sort it used to be against the rank-bounded selection it is now.
//!
//! The number in `docs/dflash.md` comes from here. It is a real benchmark rather
//! than an argument from complexity because the two routines cross over: the
//! selection wins by ~5× when the nucleus keeps a few dozen tokens and loses by
//! ~5 % when it keeps 90 % of the vocabulary, and only a measurement says where.
//!
//!   cargo run --release -p peregrine-model --example nucleusbench
//!
//! The **new** side calls the shipped `Sampler::distribution`, so this measures
//! the engine and not a copy of it. The **old** side is the pre-2026-08-22
//! `dist_build` body, kept here because the whole point is to compare against
//! what was replaced; `the_o_v_selection_reproduces_the_full_sort_bit_for_bit`
//! owns the correctness half and is the reason this file only has to time them.
//!
//! Logits are Zipf-shaped — `lo[rank] = -alpha * ln(rank + 1)`, shuffled into the
//! vocabulary — which is the shape a decode row has: a short dominant head over a
//! long, near-flat tail. `alpha` sweeps how peaked it is, and with it how many
//! tokens the nucleus keeps, which is the variable the two routines differ on.

use peregrine_model::Sampler;
use std::time::Instant;

/// GLM-5.2's vocabulary (`docs/tokenizer.md`).
const VOCAB: usize = 154_880;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / 16_777_216.0
    }
}

/// The pre-2026-08-22 `dist_build`: softmax, a full stable sort of the
/// vocabulary, keep the smallest prefix reaching the nucleus, renormalize.
fn old_dist_build(p: &mut Vec<f32>, idx: &mut Vec<usize>, lo: &[f32], temp: f32, nucleus: f32) -> usize {
    let v = lo.len();
    p.resize(v, 0.0);
    let mx = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let invt = 1.0 / temp.max(1e-4);
    let mut s = 0f64;
    for (pi, &l) in p.iter_mut().zip(lo.iter()) {
        *pi = ((l - mx) * invt).exp();
        s += f64::from(*pi);
    }
    for pi in p.iter_mut() {
        *pi /= s as f32;
    }
    if nucleus >= 1.0 || nucleus.is_nan() {
        return v;
    }
    idx.clear();
    idx.extend(0..v);
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut cum = 0f64;
    let mut keep = v;
    for i in 0..v {
        cum += f64::from(p[idx[i]]);
        if cum >= f64::from(nucleus) {
            keep = i + 1;
            break;
        }
    }
    for i in keep..v {
        p[idx[i]] = 0.0;
    }
    let mut s2 = 0f64;
    for i in 0..keep {
        s2 += f64::from(p[idx[i]]);
    }
    for i in 0..keep {
        p[idx[i]] /= s2 as f32;
    }
    keep
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    const TEMP: f32 = 1.0;
    const NUCLEUS: f32 = 0.95;
    const ITERS: usize = 120;
    println!("vocab {VOCAB}, temp {TEMP}, top_p {NUCLEUS}, median of {ITERS}\n");
    println!("{:>5}  {:>9}  {:>10}  {:>12}  {:>7}", "alpha", "keeps", "full sort", "rank-bounded", "");
    for alpha in [3.0f32, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5] {
        let mut rng = Lcg(0x1234_5678 ^ u64::from(alpha.to_bits()));
        let mut order: Vec<usize> = (0..VOCAB).collect();
        for i in (1..VOCAB).rev() {
            let j = (rng.f() * (i as f32 + 1.0)) as usize % (i + 1);
            order.swap(i, j);
        }
        let mut lo = vec![0f32; VOCAB];
        for (rank, &slot) in order.iter().enumerate() {
            lo[slot] = -alpha * ((rank + 1) as f32).ln();
        }

        let (mut p, mut idx) = (vec![0f32; VOCAB], Vec::new());
        let keep = old_dist_build(&mut p, &mut idx, &lo, TEMP, NUCLEUS);
        let mut sampler = Sampler::new(TEMP, NUCLEUS, 1);
        // Cheap agreement check, so a timing run can never report a speedup for a
        // routine that stopped producing the same distribution.
        if sampler.distribution(&lo) != p.as_slice() {
            println!("alpha {alpha}: MISMATCH — the two routines disagree; timings withheld");
            continue;
        }

        let (mut told, mut tnew) = (Vec::with_capacity(ITERS), Vec::with_capacity(ITERS));
        for _ in 0..ITERS {
            let t = Instant::now();
            std::hint::black_box(old_dist_build(&mut p, &mut idx, &lo, TEMP, NUCLEUS));
            told.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            std::hint::black_box(sampler.distribution(&lo));
            tnew.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let (a, b) = (median(told), median(tnew));
        println!("{alpha:>5.1}  {keep:>9}  {a:>8.0} us  {b:>10.0} us  {:>6.2}x", a / b);
    }
    println!("\nThe softmax is common to both and is ~500 us of every row.");
}

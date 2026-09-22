//! Context-assembly benchmark: ingest N multi-turn histories and time the
//! DCR plan+render path at a fixed budget.
//!
//! Usage: `cargo run --release -p peregrine-ctx --example ctxbench`
//!
//! Prints per scale: turns, ingested tokens (heuristic), ingest ms, plan ms,
//! assembled tokens, and whether the assembly fits the budget.

use peregrine_ctx::tokens::estimate_tokens;
use peregrine_ctx::Dcr;

fn history(n: usize) -> Vec<(String, String)> {
    let facts = [
        ("server ip", "10.0.4.12"),
        ("deploy window", "02:00"),
        ("hourly rate", "140"),
        ("retry budget", "7"),
        ("on-call", "sorin"),
        ("region", "eu-west-1"),
    ];
    (0..n)
        .map(|i| {
            let (k, v) = facts[i % facts.len()];
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            (role.to_string(), format!("turn {i}: the {k} is {v}."))
        })
        .collect()
}

fn main() {
    let budget = 512usize;
    println!("budget={budget} heuristic-tokens (len/4)");
    println!("{:>8} {:>12} {:>10} {:>10} {:>12} {:>6}", "turns", "in-tok", "ingestms", "planms", "out-tok", "fits?");
    for &n in &[50usize, 200, 800] {
        let turns = history(n);
        let in_tok: usize = turns.iter().map(|(_, t)| estimate_tokens(t)).sum();
        let mut rt = Dcr::new(budget);
        let t0 = std::time::Instant::now();
        for (role, text) in &turns {
            let tagged = format!("[{role}]\n{text}");
            if let Err(e) = rt.ingest(&tagged, None) {
                eprintln!("ingest failed at synthetic turn: {e}");
                std::process::exit(1);
            }
        }
        let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let context = rt.plan("what is the server ip?", Some(budget));
        let assembled = rt.prompt_for(&context);
        let plan_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let out_tok = estimate_tokens(&assembled);
        let carries = assembled.contains("10.0.4.12");
        println!(
            "{n:>8} {in_tok:>12} {ingest_ms:>10.1} {plan_ms:>10.1} {out_tok:>12} {:>6}",
            if out_tok <= budget && carries { "yes" } else { "NO" }
        );
        assert!(carries, "working set lost the queried fact at {n} turns");
    }
}

//! Swapping the vector channel for your own embedder.
//!
//! The bundled embedder is a 256-dimensional hashing embedding: offline,
//! deterministic, dependency-free, and it finds material that shares
//! vocabulary. That last property is why paraphrase retrieval is a documented
//! poor fit — `cargo run --release -- bench --channels` measures how much it
//! actually agrees with the lexical channel instead of leaving it asserted.
//!
//! Replacing it is a constructor argument. Implement `Embedder`, put it on the
//! `Ladder`, and every vector the runtime builds — node L2 vectors and query
//! vectors alike — comes from yours.
//!
//! Run with: `cargo run --release --example custom_embedder`

use peregrine_ctx::embed::Embedder;
use peregrine_ctx::runtime::Dcr;

/// A deliberately trivial stand-in for a learned model.
///
/// It buckets by first letter, which is useless for retrieval and useful for
/// this example: if the runtime is really routing through the trait, swapping
/// this in has to visibly change which nodes a query seeds from. An embedder
/// that made no difference would prove nothing.
#[derive(Debug)]
struct FirstLetterEmbedder;

impl Embedder for FirstLetterEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim()];
        for word in text.split_whitespace() {
            if let Some(c) = word.chars().next().filter(char::is_ascii_alphabetic) {
                let i = (c.to_ascii_lowercase() as usize) - ('a' as usize);
                v[i] += 1.0;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn dim(&self) -> usize {
        26
    }
}

// In a real deployment this is where a sentence-transformer or an API-backed
// embedder goes. Note what that costs: the runtime stops being offline and
// deterministic, so every figure in RESULTS.md would need re-measuring against
// the new channel rather than inherited.

fn main() {
    let transcript = [
        ("t1", "Goal: restore checkout by 09:00 UTC."),
        ("t2", "The server ip is 10.0.4.12 and the port is 8080."),
        ("t3", "Update: the server ip is 10.0.9.7 after the failover."),
        ("t4", "The deploy window is 02:00-04:00 UTC."),
    ];

    for (label, custom) in [("bundled hashing embedder", false), ("custom embedder", true)] {
        let mut dcr = Dcr::new(400);
        if custom {
            dcr.ladder.embedder = Box::new(FirstLetterEmbedder);
        }
        for (id, text) in &transcript {
            dcr.ingest(text, Some(id)).expect("ingest");
        }
        let context = dcr.plan("what is the server ip?", None);
        println!(
            "{label:<26} dim={:<4} admitted={:<3} tokens={}",
            dcr.ladder.embedder.dim(),
            context.entries.len(),
            context.tokens
        );
        for entry in &context.entries {
            println!("    [{}] {}", entry.level.as_str(), entry.label);
        }
    }

    println!();
    println!("Both plans answer; the second seeds from a channel that knows nothing");
    println!("but first letters. The seam is real — which is the only claim this");
    println!("example makes. It is not evidence that a better embedder helps, and");
    println!("no figure in RESULTS.md transfers to a runtime with a different one.");
}

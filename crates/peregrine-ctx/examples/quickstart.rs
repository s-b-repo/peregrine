//! Smallest useful thing: ingest a transcript, ask a question, audit the answer.
//!
//! ```bash
//! cargo run --release --example quickstart
//! ```

use peregrine_ctx::Dcr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = Dcr::new(600);

    runtime.ingest("Goal: restore checkout by 09:00 UTC.", None)?;
    runtime.ingest(
        "The error was \"connection refused\" when talking to the inventory host.",
        None,
    )?;
    runtime.ingest("The server ip is 10.0.4.12 and the port is 8080.", None)?;
    runtime.ingest(
        "The blocker is firewall rule 37, which drops checkout traffic.",
        None,
    )?;
    runtime.ingest(
        "Decision: roll back to build 4471 because the blocker is firewall rule 37.",
        None,
    )?;

    // ... 200 turns of noise later ...
    for i in 0..200 {
        runtime.ingest(
            &format!("Chatter {i}: dashboards refreshed, queue at {i} items, nothing to do."),
            None,
        )?;
    }

    // ... and a correction nobody would find by scrolling.
    runtime.ingest(
        "Correction: actually the server ip is 10.0.9.7, we misread the dashboard.",
        None,
    )?;

    let answer = runtime.ask("what is the server ip?", None);
    println!("{}", answer.text);
    println!(
        "({} tokens in the window, out of {} tokens of history)\n",
        answer.tokens, runtime.telemetry.history_tokens
    );

    // Every answer is walkable back to raw spans.
    if let Some(node_id) = answer.cited.first() {
        println!("{}", runtime.explain(node_id)?);
    }

    // And you can see exactly what the model was shown, and why.
    println!("\n{}", runtime.explain_plan(&answer.context));
    Ok(())
}

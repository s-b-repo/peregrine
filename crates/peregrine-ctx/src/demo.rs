//! Worked example — one task traced through all four ladder levels.
//!
//! A transcript that contains a correction, an exact string worth quoting, a
//! multi-hop decision to justify, and a derivation to recompute. Runs offline.

use crate::graph::DcrError;
use crate::llm::LocalReasoner;
use crate::nodes::Kind;
use crate::runtime::Dcr;

pub const TRANSCRIPT: &[(&str, &str)] = &[
    (
        "t01",
        "Goal: restore checkout by 09:00 UTC.\n\n\
         Constraint: never restart the payment service during business hours.",
    ),
    (
        "t02",
        "04:12 paging on-call. The service is alpha-checkout and the owner is team-payments.",
    ),
    (
        "t03",
        "The error was \"connection refused\" when talking to the inventory host.",
    ),
    ("t04", "The server ip is 10.0.4.12 and the port is 8080."),
    (
        "t05",
        "The blocker is firewall rule 37, which drops traffic to the checkout subnet.",
    ),
    (
        "t06",
        "Decision: roll back to build 4471 because the blocker is firewall rule 37.",
    ),
    (
        "t07",
        "The engineer count is 3 and the incident hours are 4.",
    ),
    (
        "t08",
        "Standup notes: coffee machine still broken, ticket queue at 14 items.",
    ),
    (
        "t09",
        "Correction: actually the server ip is 10.0.9.7, we misread the dashboard.",
    ),
    ("t10", "The hourly rate is 180 USD."),
    (
        "t11",
        "Deploy log for build 4471: the rollout started at 04:03 and moved through canary, \
         then 10 percent, then 50 percent of the fleet without incident. At 04:11 the checkout \
         pods began failing readiness probes against the inventory host, the load balancer \
         drained them, and the rollout controller entered a retry loop. The retry budget was \
         exhausted after 7 attempts and the final failure code was ERR_CONN_REFUSED_37, at \
         which point the controller stopped and paged the on-call engineer.",
    ),
];

pub fn build(budget: usize, noise: usize) -> Result<Dcr, DcrError> {
    let mut runtime = Dcr::new(budget);
    for (doc_id, text) in TRANSCRIPT {
        runtime.ingest(text, Some(doc_id))?;
    }
    for i in 0..noise {
        runtime.ingest(
            &format!(
                "Chatter {i}: dashboards refreshed, {} alerts acknowledged, nothing actionable, \
                 standby continues.",
                i % 7
            ),
            Some(&format!("noise{i:03}")),
        )?;
    }
    runtime.register("incident_cost", |inputs| {
        let get = |name: &str| {
            inputs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };
        get("rate") * get("hours") * get("engineers")
    });
    Ok(runtime)
}

pub fn run_demo(budget: usize) -> Result<Dcr, DcrError> {
    let mut runtime = build(budget, 60)?;
    let mut reasoner = LocalReasoner::new();
    let rule = "=".repeat(72);

    println!("{rule}\nINGESTED\n{rule}");
    println!(
        "{} documents, {} L0 spans, {} tokens of history",
        runtime.raw.documents().len(),
        runtime.raw.len(),
        runtime.telemetry.history_tokens
    );
    println!("state: {}", runtime.graph.stats());

    let mut turn = |runtime: &mut Dcr, title: &str, query: &str| {
        println!("\n{rule}\n{title}\n{rule}");
        let answer = runtime.ask_with(query, None, &mut reasoner);
        println!("{}", answer.context.render());
        println!("\nQUESTION: {query}");
        println!("ANSWER:   {}", answer.text);
        println!(
            "[{} tokens of {} budget, {} escalation(s)]",
            answer.tokens, answer.context.budget, answer.escalations
        );
    };

    turn(
        &mut runtime,
        "1. VALUE LOOKUP — answered from L2 state, after a correction",
        "what is the server ip?",
    );
    turn(
        &mut runtime,
        "2. EXACT QUOTE — routed straight to L0 by query type",
        "quote the exact error message",
    );
    turn(
        &mut runtime,
        "3. ESCALATION — the compact form is too thin, so the model asks for L0",
        "how many retry attempts were made before the failure?",
    );
    turn(
        &mut runtime,
        "4. JUSTIFICATION — retrieval follows dependency edges",
        "why did we roll back?",
    );

    println!("\n{rule}\n5. RECOMPUTE — L3 derivation, memoised into C_t\n{rule}");
    // These three facts are ingested at the top of this function, so the lookups
    // succeed on the demo corpus. Handle the empty case anyway rather than
    // unwrap: a demo that panics because someone edited the transcript is a
    // worse failure than one that says why it stopped.
    let latest = |key: &str| runtime.graph.by_key(key, true).last().copied();
    let (Some(rate), Some(hours), Some(engineers)) = (
        latest("hourly.rate"),
        latest("incident.hours"),
        latest("engineer.count"),
    ) else {
        println!("(recompute demo skipped: expected facts not extracted from the transcript)");
        return Ok(runtime);
    };
    let deps: Vec<String> = [rate, hours, engineers]
        .iter()
        .map(|&i| runtime.graph.node(i).id.clone())
        .collect();
    let inputs = vec![
        ("rate".to_string(), 180.0),
        ("hours".to_string(), 4.0),
        ("engineers".to_string(), 3.0),
    ];
    let cost = runtime.compute(
        "incident_cost",
        inputs.clone(),
        deps.clone(),
        Some("incident.cost"),
    )?;
    let cost_id = runtime.graph.node(cost).id.clone();
    println!(
        "incident.cost = {} (node {cost_id})",
        runtime.graph.node(cost).value
    );
    println!("execution: {}", runtime.execution.stats());
    let again = runtime.compute("incident_cost", inputs, deps, Some("incident.cost"))?;
    println!(
        "\nsame derivation again -> {}",
        runtime.graph.node(again).value
    );
    println!(
        "execution: {}  <- memo hit, no recomputation",
        runtime.execution.stats()
    );

    println!("\n{rule}\n6. INVALIDATION — a corrected input marks the derivation stale\n{rule}");
    runtime.ingest(
        "Correction: the hourly rate is 210 USD, finance updated the figure.",
        Some("t12"),
    )?;
    println!(
        "incident.cost status -> {}",
        runtime.graph.node(cost).status.as_str()
    );
    println!("{}", runtime.explain(&cost_id)?);

    println!("\n{rule}\n7. AUDIT PATH\n{rule}");
    let decision = runtime.graph.by_kind(Kind::Decision, true)[0];
    let decision_id = runtime.graph.node(decision).id.clone();
    println!("{}", runtime.explain(&decision_id)?);

    println!(
        "\n{rule}\n8. WORKSPACE REBUILD — destroy the working set, rebuild from memory\n{rule}"
    );
    println!(
        "{}",
        runtime.rebuild_workspace("what is the server ip?", None)
    );

    println!("\n{rule}\nTELEMETRY\n{rule}");
    print!("{}", runtime.report());
    Ok(runtime)
}

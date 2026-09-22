//! Context plane for chat prompts (`COLI_CTX`), backed by `peregrine-ctx`
//! (the ported subnext/DCR runtime).
//!
//! When on, the prompt the model sees is the runtime's planned working set:
//! message turns ingested role-tagged, relevance-planned for the last user
//! turn under the token budget, rendered as context. The plane is a context
//! source, not a compression fallback — it assembles what the model should
//! attend to on every request, not only over-budget ones. With the knob off
//! the raw rendered prompt serves exactly as before (bit-identical).
//!
//! Stateless per call — a fresh runtime per request, no cross-request memory
//! (the server is stateless per request), deterministic for identical turns.
//! An ingestion failure or empty assembly falls back to the raw prompt, never
//! a partial context: a turn that fails to ingest aborts the whole assembly.

use peregrine_ctx::Dcr;

/// Context-plane master switch (`COLI_CTX=1`). Off by default: engagement
/// changes tokens, so it must be asked for, like every other knob that can
/// move a token.
pub fn ctx_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COLI_CTX").as_deref(), Ok("1") | Ok("true")))
}

/// Assemble `turns` (`(role, text)` oldest-first) into a working set for
/// `query` under `budget` tokens. Returns `None` when the plane is off, when
/// nothing ingested, or when assembly yields no text — every `None` means
/// "serve the raw prompt", never "serve a partial context".
pub fn assemble_context(turns: &[(String, String)], query: &str, budget: usize) -> Option<String> {
    if !ctx_enabled() {
        return None;
    }
    assemble_context_enabled(turns, query, budget)
}

/// [`assemble_context`] without the env gate — the testable core. Same
/// contract: `None` on empty input, failed ingest, or empty assembly.
pub fn assemble_context_enabled(
    turns: &[(String, String)],
    query: &str,
    budget: usize,
) -> Option<String> {
    if turns.is_empty() || budget == 0 {
        return None;
    }
    let mut rt = Dcr::new(budget);
    let mut ingested = 0usize;
    for (role, text) in turns {
        if text.trim().is_empty() {
            continue;
        }
        let tagged = format!("[{role}]\n{text}");
        if let Err(e) = rt.ingest(&tagged, None) {
            peregrine_core::note_advisory_err("ctx ingest (falling back to raw prompt)", &e);
            return None;
        }
        ingested += 1;
    }
    if ingested == 0 {
        return None;
    }
    let context = rt.plan(query, Some(budget));
    let assembled = rt.prompt_for(&context);
    if assembled.trim().is_empty() {
        return None;
    }
    Some(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_never_assembles() {
        assert!(assemble_context_enabled(&[], "q", 100).is_none());
        assert!(assemble_context_enabled(&[("user".to_string(), "hi".to_string())], "q", 0).is_none());
        // Whitespace-only turns ingest nothing: no partial context.
        assert!(assemble_context_enabled(&[("user".to_string(), "   ".to_string())], "q", 100).is_none());
    }

    #[test]
    fn assembly_covers_the_query_terms() -> Result<(), String> {
        let turns = vec![
            ("system".to_string(), "You are a network assistant.".to_string()),
            ("user".to_string(), "The server ip is 10.0.4.12.".to_string()),
            ("user".to_string(), "The deploy window is 02:00.".to_string()),
            ("user".to_string(), "What is the server ip?".to_string()),
        ];
        let out = assemble_context_enabled(&turns, "what is the server ip?", 200)
            .ok_or_else(|| "a small multi-turn history must assemble".to_string())?;
        if !out.contains("10.0.4.12") {
            return Err(format!("working set must carry the queried fact: {out}"));
        }
        if !out.contains("QUESTION") {
            return Err(format!("assembled prompt keeps the question frame: {out}"));
        }
        Ok(())
    }
}

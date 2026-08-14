//! Workload classification.
//!
//! [`TokenClass`] — a coarse bucket over recent tokens by their id-space
//! statistics (proxying "is this code / prose / JSON / math?"). The HTTP
//! handler is the natural producer since it has the tokenizer; the engine
//! consumes a class tag from the request. Correctness-neutral: it only steers
//! prefetch breadth, never the numerics.
//!
//! A `PhaseTracker` (EWMA of frame-to-frame routing-set Jaccard distance,
//! carrying a post-shift *window* the instantaneous `PhaseAware` check cannot
//! express) lived here unwired from 2026-07-30 to 2026-08-14, kept on the
//! grounds that `COLI_PREDICT_EVAL` could one day say whether the window beat
//! the instantaneous check. The scoreboard ran on the real checkpoint
//! (2026-08-14, 4 725 layer transitions, width 16) and answered a bigger
//! question instead: the **router look-ahead** dominates the entire
//! statistical stack — recall 92.5 % / precision 46.3 % vs PhaseAware's
//! 58.1 % / 29.3 % (the prev-token baseline read 38.9 % / 38.3 %). A window
//! refinement to an arm 34 recall points behind the shipped default has no
//! path to relevance, so the struct is deleted rather than carried;
//! `bench-data/2026-08-13-predict-eval/` holds the run.

/// A coarse workload class inferred from the recent token stream. Deliberately
/// broad — the class is a *prefetch hint*, not a routing decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TokenClass {
    #[default]
    Prose,
    Code,
    Json,
    Math,
    Mixed,
}

/// Bucketize a short string over a raw slice of decoded characters. Meant for
/// the *tail* of the prompt (say, the last 128 characters — long enough to
/// be representative, short enough to be O(1)). Deterministic and stateless.
pub fn classify_str(tail: &str) -> TokenClass {
    if tail.trim().is_empty() {
        return TokenClass::Prose;
    }
    let (mut alpha, mut digits, mut punct, mut brace_curly, mut brace_paren, mut brace_bracket, mut ws) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let mut math_ops = 0u32;
    for ch in tail.chars() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => ws += 1,
            c if c.is_ascii_alphabetic() => alpha += 1,
            c if c.is_ascii_digit() => digits += 1,
            '{' | '}' => brace_curly += 1,
            '(' | ')' => brace_paren += 1,
            '[' | ']' => brace_bracket += 1,
            '+' | '-' | '*' | '/' | '=' | '<' | '>' | '^' => math_ops += 1,
            c if c.is_ascii_punctuation() => punct += 1,
            _ => {}
        }
    }
    let total = alpha + digits + punct + brace_curly + brace_paren + brace_bracket + ws + math_ops;
    if total == 0 {
        return TokenClass::Prose;
    }
    let punct_ratio = (punct + brace_curly + brace_paren + brace_bracket) as f32 / total as f32;
    let digit_ratio = digits as f32 / total as f32;
    let math_ratio = math_ops as f32 / total as f32;
    // JSON: curly braces present AND a dense punctuation stream (quotes,
    // colons, commas — all counted as `punct`). Short JSON snippets rarely
    // exceed a 10 % curly-brace ratio, so gate on presence rather than share.
    if brace_curly >= 2 && punct_ratio > 0.30 {
        return TokenClass::Json;
    }
    // Math: heavy on operators + digits.
    if math_ratio > 0.10 && digit_ratio > 0.10 {
        return TokenClass::Math;
    }
    // Code: many brackets/parens or high punctuation.
    if brace_paren + brace_bracket > 3 || punct_ratio > 0.15 {
        return TokenClass::Code;
    }
    // Mixed vs Prose: if the alpha share is modest, it is likely mixed content.
    if alpha as f32 / total as f32 > 0.60 {
        TokenClass::Prose
    } else {
        TokenClass::Mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_json_and_code() {
        assert_eq!(classify_str(r#"{"key": "value", "n": 42}"#), TokenClass::Json);
        assert_eq!(classify_str("fn foo(x: i32) -> i32 { x + 1 }"), TokenClass::Code);
    }

    #[test]
    fn classify_math_and_prose() {
        assert_eq!(classify_str("2 + 3 = 5 and 6 * 7 = 42"), TokenClass::Math);
        assert_eq!(classify_str("The quick brown fox jumps over the lazy dog."), TokenClass::Prose);
    }
}

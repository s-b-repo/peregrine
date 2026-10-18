//! Context assembly & KV-cache integrity helpers.

/// Static header, dynamic trailer: per-turn stats moved out of the rendered
/// context header into an appended trailer so the KV-cache prefix stays stable.
#[derive(Debug, Clone, Default)]
pub struct StaticHeaderDynamicTrailer;

impl StaticHeaderDynamicTrailer {
    pub fn render_trailer(tokens: usize, budget: usize, admitted: usize, considered: usize, qtype: &str) -> String {
        format!(
            "\n# k={}/{} tokens, {} of {} candidates, query type: {}",
            tokens, budget, admitted, considered, qtype
        )
    }

    pub fn render_header() -> String {
        "# ACTIVE CONTEXT".to_string()
    }
}

/// Stable content ordering: sort by content-addressed node id within kind groups.
pub fn stable_content_ordering(ids: &mut [String]) {
    ids.sort();
}

pub fn stable_ordering(ids: &mut [String]) {
    stable_content_ordering(ids);
}

/// Escalation-as-append: render a mid-turn re-plan as an appended block.
#[derive(Debug, Clone, Default)]
pub struct EscalationAsAppend;

impl EscalationAsAppend {
    pub fn append_escalated(base: &str, escalated: &str) -> String {
        format!("{base}\n\n## ESCALATED\n{escalated}")
    }

    pub fn append(base: &str, escalated: &str) -> String {
        Self::append_escalated(base, escalated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailer_contains_stats() {
        let t = StaticHeaderDynamicTrailer::render_trailer(10, 20, 2, 5, "open");
        assert!(t.contains("k=10/20"));
        assert!(t.contains("2 of 5 candidates"));
    }

    #[test]
    fn header_is_static() {
        assert_eq!(StaticHeaderDynamicTrailer::render_header(), "# ACTIVE CONTEXT");
    }

    #[test]
    fn stable_order_sorts() {
        let mut ids = vec!["b".to_string(), "a".to_string()];
        stable_content_ordering(&mut ids);
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
        let mut ids2 = vec!["z".to_string(), "a".to_string()];
        stable_ordering(&mut ids2);
        assert_eq!(ids2, vec!["a".to_string(), "z".to_string()]);
    }

    #[test]
    fn escalation_append() {
        let s = EscalationAsAppend::append_escalated("base", "esc");
        assert!(s.contains("## ESCALATED"));
        assert!(s.starts_with("base"));
        let s2 = EscalationAsAppend::append("base", "esc");
        assert_eq!(s, s2);
    }
}

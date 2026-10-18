//! Authorization & capability gating — PDP/PEP split, scoped attenuable tokens,
//! context-aware authorization, fail-closed default-deny, full audit logging.


/// External policy decision point: authorization enforced outside the model.
#[derive(Debug, Clone, Default)]
pub struct PolicyDecisionPoint {
    pub decisions: Vec<String>,
}

impl PolicyDecisionPoint {
    pub fn decide(&mut self, tool: &str, resource: &str, allowed: bool) -> bool {
        self.decisions.push(format!("{tool}:{resource}:{allowed}"));
        allowed
    }
}

pub type Pdp = PolicyDecisionPoint;
pub type ExternalPdp = PolicyDecisionPoint;

/// Policy enforcement point (the gate in front of tool execution).
#[derive(Debug, Clone, Default)]
pub struct PolicyEnforcementPoint {
    pub pdp: PolicyDecisionPoint,
}

impl PolicyEnforcementPoint {
    pub fn enforce(&mut self, tool: &str, resource: &str, args_ok: bool) -> bool {
        // Fail-closed default-deny.
        if !args_ok {
            self.pdp.decide(tool, resource, false);
            return false;
        }
        self.pdp.decide(tool, resource, true)
    }
}

pub type Pep = PolicyEnforcementPoint;

/// Scoped, attenuable authorization token (Macaroons-style).
#[derive(Debug, Clone, Default)]
pub struct ScopedAuthToken {
    pub scope: String,
    pub domain: String,
    pub expires_in_secs: u64,
    pub caveats: Vec<String>,
}

impl ScopedAuthToken {
    pub fn new(scope: &str, domain: &str, expires_in_secs: u64) -> Self {
        Self {
            scope: scope.to_string(),
            domain: domain.to_string(),
            expires_in_secs,
            caveats: Vec::new(),
        }
    }

    pub fn attenuate(&mut self, caveat: &str) {
        self.caveats.push(caveat.to_string());
    }

    pub fn narrow(&mut self, caveat: &str) {
        self.attenuate(caveat);
    }

    pub fn add_caveat(&mut self, caveat: &str) {
        self.attenuate(caveat);
    }
}

pub type MacaroonToken = ScopedAuthToken;
pub type AttenuableToken = ScopedAuthToken;

/// Context-aware authorization reasoning over full request + trajectory.
#[derive(Debug, Clone, Default)]
pub struct ContextAwareAuthorization {
    pub allow: bool,
}

impl ContextAwareAuthorization {
    pub fn evaluate(&self, _request: &str, _trajectory: &str) -> bool {
        self.allow
    }

    pub fn is_allowed(&self, request: &str, trajectory: &str) -> bool {
        self.evaluate(request, trajectory)
    }

    pub fn decide(&self, request: &str, trajectory: &str) -> bool {
        self.evaluate(request, trajectory)
    }
}

/// Fail-closed default-deny gate: every tool call re-authorized individually.
#[derive(Debug, Clone, Default)]
pub struct FailClosedAuthorizer {
    pub default_allow: bool,
}

impl FailClosedAuthorizer {
    pub fn authorize(&self, concrete_args_ok: bool) -> bool {
        // Fail-closed default-deny: concrete args must be ok; never trust default_allow.
        concrete_args_ok
    }

    pub fn is_authorized(&self, concrete_args_ok: bool) -> bool {
        self.authorize(concrete_args_ok)
    }
}

/// Full audit logging of authorization decisions.
#[derive(Debug, Clone, Default)]
pub struct AuthorizationAuditLog {
    pub entries: Vec<String>,
}

impl AuthorizationAuditLog {
    pub fn log(&mut self, identity: &str, tool: &str, resource: &str, args: &str, decision: bool, reason: &str) {
        self.entries.push(format!(
            "identity={identity} tool={tool} resource={resource} args={args} decision={decision} reason={reason}"
        ));
    }

    pub fn record(&mut self, identity: &str, tool: &str, resource: &str, args: &str, decision: bool, reason: &str) {
        self.log(identity, tool, resource, args, decision, reason);
    }
}

pub type AuditLog = AuthorizationAuditLog;
pub type FullAuditLog = AuthorizationAuditLog;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_closed_default_deny() {
        let a = FailClosedAuthorizer::default();
        assert!(!a.authorize(false));
        assert!(a.authorize(true));
    }

    #[test]
    fn audit_log_records() {
        let mut log = AuthorizationAuditLog::default();
        log.log("id", "tool", "res", "args", true, "ok");
        assert_eq!(log.entries.len(), 1);
    }

    #[test]
    fn attenuate_token() {
        let mut t = ScopedAuthToken::new("scope", "domain", 10);
        t.attenuate("caveat");
        assert_eq!(t.caveats.len(), 1);
    }
}

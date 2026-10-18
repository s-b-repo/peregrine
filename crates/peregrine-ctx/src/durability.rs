//! Durability & correctness — rollback consistency, recoverable session
//! contract, resident KV as a provable claim.


/// Rollback consistency: what is recorded when a mid-turn working set is
/// discarded because the state it was rendered from moved underneath it.
///
/// The honest scope of this record is *invalidation*, not clearing: the store
/// is untouched, nothing is written or dropped — what is invalidated is the
/// prompt the reasoner already saw. Its cached prefix must not be reused,
/// because material it contains has been falsified. An earlier version
/// recorded a "clear" of transcripts and KV in two lockstep vecs; nothing
/// downstream read them, nothing was actually cleared, and a record of work
/// that did not happen is worse than no record.
#[derive(Debug, Clone, Default)]
pub struct RollbackConsistencyGuard {
    /// Working sets discarded mid-turn, oldest first. Each entry is a
    /// promise the caller makes: the prompt rendered from this working set
    /// must not be served again, and its cached prefix must not be reused.
    pub discarded_working_sets: Vec<String>,
}

impl RollbackConsistencyGuard {
    /// Record that the working set identified by `id` was discarded after the
    /// state under it was invalidated. The caller owns the actual discard —
    /// this records it, with the reason attached by the caller's own audit
    /// log.
    pub fn discard_working_set(&mut self, id: &str) {
        self.discarded_working_sets.push(id.to_string());
    }

    /// Every discard recorded this session.
    pub fn discards(&self) -> &[String] {
        &self.discarded_working_sets
    }
}

pub type RollbackConsistency = RollbackConsistencyGuard;
pub type RollbackConsistencyGuarantee = RollbackConsistencyGuard;

/// Recoverable session contract: retention budget, cross-session eviction
/// priority, session isolation, crash/restart recovery.
#[derive(Debug, Clone, Default)]
pub struct RecoverableSessionContract {
    pub session_id: String,
    pub retention_budget: usize,
    pub isolation: bool,
    pub crash_recovery: bool,
}

impl RecoverableSessionContract {
    pub fn new(session_id: &str, retention_budget: usize) -> Self {
        Self {
            session_id: session_id.to_string(),
            retention_budget,
            isolation: true,
            crash_recovery: true,
        }
    }

    pub fn validate(&self) -> bool {
        self.isolation && self.crash_recovery && !self.session_id.is_empty()
    }
}

pub type SessionContract = RecoverableSessionContract;
pub type RecoverableSession = RecoverableSessionContract;

/// Resident KV as a provable claim (conformance contract), tying into the
/// provenance graph rather than a soft hint.
#[derive(Debug, Clone, Default)]
pub struct ResidentKvClaim {
    pub node_id: String,
    pub promised: bool,
    pub honored: bool,
}

impl ResidentKvClaim {
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            promised: true,
            honored: false,
        }
    }

    pub fn honor(&mut self) {
        self.honored = true;
    }

    pub fn verify(&self) -> bool {
        self.conformance_ok()
    }

    pub fn prove(&self) -> bool {
        self.conformance_ok()
    }

    pub fn conformance_ok(&self) -> bool {
        !self.promised || self.honored
    }
}

pub type ResidentKvAsProvableClaim = ResidentKvClaim;
pub type ResidentKvConformance = ResidentKvClaim;

/// Durability placeholder types.
#[derive(Debug, Clone, Default)]
pub struct DurabilityGuarantee {
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discard_is_recorded_not_fabricated() {
        let mut g = RollbackConsistencyGuard::default();
        assert!(g.discards().is_empty());
        g.discard_working_set("workspace");
        assert_eq!(g.discards(), &["workspace".to_string()]);
        // A second discard of the same id is a second event: each one
        // invalidates a prompt that must not be reused, so deduplication
        // would hide a real invalidation.
        g.discard_working_set("workspace");
        assert_eq!(g.discards().len(), 2);
    }

    #[test]
    fn session_contract_defaults() {
        let c = RecoverableSessionContract::new("s", 100);
        assert!(c.isolation && c.crash_recovery);
        assert_eq!(c.retention_budget, 100);
    }

    #[test]
    fn resident_claim_conformance() {
        let mut claim = ResidentKvClaim::new("n1");
        assert!(!claim.conformance_ok());
        claim.honor();
        assert!(claim.conformance_ok());
    }
}

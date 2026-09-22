//! The scrubber — bit-rot detection and repair.
//!
//! Signatures and the checkpoint chain detect *modification*. They do not fix
//! anything, and they do not distinguish a malicious edit from a disk that
//! flipped a bit in a sector nobody has read for a year. This module is the
//! second half: walk every object, recompute its digest, and repair the ones
//! that no longer match from a copy that can prove itself.
//!
//! ```text
//! read object → recompute hash → compare
//!                                  │
//!                        ┌─────────┴─────────┐
//!                     matches            mismatch
//!                        │                   │
//!                     healthy      find a replica copy
//!                                            │
//!                                  verify the replica ON ITS OWN
//!                                  (digest == address, and it proves
//!                                   against the committed Merkle root)
//!                                            │
//!                              ┌─────────────┴─────────────┐
//!                          verified                   unverified
//!                              │                           │
//!                         repair primary              quarantine
//!                         emit audit event            emit audit event
//! ```
//!
//! The rule that makes this safe is the one that is easiest to get wrong:
//! **never repair from an unverified replica.** A copy is only allowed to fix
//! another copy once its own bytes hash to the address being repaired *and*
//! that pair proves against the root the store committed to. Otherwise a
//! scrubber is just a mechanism for spreading corruption evenly.
//!
//! # What repair cannot do
//!
//! With no replicas configured there is nothing to repair *from*. Detection
//! still works — that is the value of running it — but the only available
//! action is quarantine. Losing an object also changes the Merkle root, so a
//! store that has quarantined something can no longer reproduce the root its
//! last checkpoint committed to. That is not a bug to paper over: the loss is
//! real, and [`ScrubOptions::seal`] records it as a new generation rather than
//! letting the store quietly claim a root it can no longer build.

use std::fmt;

use crate::context_store::{ContextError, ContextStore};
use crate::hash::sha256;
use crate::trust::Signer;

/// One thing that happened during a scrub, in the order it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditEvent {
    Healthy(String),
    /// The object's bytes no longer hash to its address.
    Corrupt { object: String, detail: String },
    /// A replica was found but could not prove itself, so it was not used.
    ReplicaRejected { object: String, source: String },
    Repaired { object: String, source: String },
    Quarantined { object: String, reason: String },
    /// Corrupt, and no replica exists to repair it from.
    Unrepairable(String),
    Sealed { generation: u64 },
}

impl fmt::Display for AuditEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short = |id: &String| id.chars().take(16).collect::<String>();
        match self {
            AuditEvent::Healthy(id) => write!(f, "ok         {}", short(id)),
            AuditEvent::Corrupt { object, detail } => {
                write!(f, "CORRUPT    {} — {detail}", short(object))
            }
            AuditEvent::ReplicaRejected { object, source } => write!(
                f,
                "REJECTED   {} — replica at {source} does not verify; not used for repair",
                short(object)
            ),
            AuditEvent::Repaired { object, source } => {
                write!(f, "repaired   {} from {source}", short(object))
            }
            AuditEvent::Quarantined { object, reason } => {
                write!(f, "quarantine {} — {reason}", short(object))
            }
            AuditEvent::Unrepairable(id) => write!(
                f,
                "LOST       {} — corrupt and no verified replica exists",
                short(id)
            ),
            AuditEvent::Sealed { generation } => {
                write!(f, "sealed     generation {generation} records the repair")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScrubOptions {
    /// Repair corrupt objects from a verified replica.
    pub repair: bool,
    /// Move objects that cannot be repaired out of the store.
    pub quarantine: bool,
    /// Commit a new generation when the object set changed, so the store's
    /// committed root matches what it can actually produce.
    pub seal: bool,
    /// Logical timestamp for the sealing checkpoint.
    pub timestamp: u64,
}

impl Default for ScrubOptions {
    fn default() -> Self {
        // Read-only by default: a scrub that mutates the store on a bare
        // invocation is a surprise nobody wants at 3am.
        ScrubOptions {
            repair: false,
            quarantine: false,
            seal: false,
            timestamp: 0,
        }
    }
}

impl ScrubOptions {
    /// Detect, repair from verified replicas, quarantine what is unrepairable,
    /// and seal the result.
    pub fn repairing(timestamp: u64) -> ScrubOptions {
        ScrubOptions {
            repair: true,
            quarantine: true,
            seal: true,
            timestamp,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScrubReport {
    pub checked: usize,
    pub healthy: usize,
    pub corrupt: Vec<String>,
    pub repaired: Vec<String>,
    pub quarantined: Vec<String>,
    pub unrepairable: Vec<String>,
    pub replicas_configured: usize,
    pub sealed_generation: Option<u64>,
    pub events: Vec<AuditEvent>,
}

impl ScrubReport {
    /// Everything checked out, or everything that did not was repaired.
    pub fn clean(&self) -> bool {
        self.corrupt.len() == self.repaired.len()
    }
}

impl fmt::Display for ScrubReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "scrubbed {} objects: {} healthy, {} corrupt, {} repaired, {} quarantined",
            self.checked,
            self.healthy,
            self.corrupt.len(),
            self.repaired.len(),
            self.quarantined.len()
        )?;
        if self.replicas_configured == 0 {
            writeln!(
                f,
                "no replicas configured — corruption can be detected here but not repaired"
            )?;
        } else {
            writeln!(f, "{} replica location(s) available", self.replicas_configured)?;
        }
        if let Some(generation) = self.sealed_generation {
            writeln!(f, "sealed as generation {generation}")?;
        }
        for event in &self.events {
            // Healthy objects are the boring majority; only surface them when
            // the whole store is small enough for the list to be readable.
            if matches!(event, AuditEvent::Healthy(_)) && self.checked > 20 {
                continue;
            }
            writeln!(f, "  {event}")?;
        }
        Ok(())
    }
}

/// Walk the store, verify every object, and act on what fails.
pub fn scrub(
    store: &mut ContextStore,
    options: &ScrubOptions,
    signer: Option<&dyn Signer>,
) -> Result<ScrubReport, ContextError> {
    let mut report = ScrubReport {
        replicas_configured: store.replicas().len(),
        ..ScrubReport::default()
    };

    let ids: Vec<String> = store.object_ids().cloned().collect();
    for id in ids {
        report.checked += 1;
        let path = store.object_path(&id);
        let found = std::fs::read(&path);
        let detail = match &found {
            Ok(bytes) if sha256(bytes).to_hex() == id => {
                report.healthy += 1;
                report.events.push(AuditEvent::Healthy(id.clone()));
                continue;
            }
            Ok(bytes) => format!("computed {}", sha256(bytes).short(16)),
            Err(e) => format!("unreadable: {e}"),
        };

        report.corrupt.push(id.clone());
        report.events.push(AuditEvent::Corrupt {
            object: id.clone(),
            detail,
        });

        if options.repair && repair_from_replica(store, &id, &mut report)? {
            continue;
        }

        report.unrepairable.push(id.clone());
        report.events.push(AuditEvent::Unrepairable(id.clone()));

        if options.quarantine {
            let reason = "failed integrity verification; no verified replica available";
            store.quarantine(&id, reason)?;
            report.quarantined.push(id.clone());
            report.events.push(AuditEvent::Quarantined {
                object: id.clone(),
                reason: reason.to_string(),
            });
        }
    }

    // The object set changed, so the committed root no longer describes the
    // store. Record that as a new generation rather than leaving a manifest
    // pointing at a root nothing can rebuild.
    if options.seal && !report.quarantined.is_empty() {
        let checkpoint = store.commit(options.timestamp, signer)?;
        report.sealed_generation = Some(checkpoint.generation);
        report.events.push(AuditEvent::Sealed {
            generation: checkpoint.generation,
        });
    }

    Ok(report)
}

/// Try every replica in turn. A copy repairs the primary only once it has
/// proved itself against the committed root — never merely because it exists.
fn repair_from_replica(
    store: &mut ContextStore,
    object_id: &str,
    report: &mut ScrubReport,
) -> Result<bool, ContextError> {
    for (path, bytes) in store.replica_copies(object_id) {
        let source = path.display().to_string();
        let hash = sha256(&bytes);

        // Two independent checks: the bytes are what this address means, and
        // that pairing is the one the store committed to.
        if hash.to_hex() != object_id || !store.proves(object_id, &hash) {
            report.events.push(AuditEvent::ReplicaRejected {
                object: object_id.to_string(),
                source,
            });
            continue;
        }

        store.restore_object(object_id, &bytes)?;
        report.repaired.push(object_id.to_string());
        report.events.push(AuditEvent::Repaired {
            object: object_id.to_string(),
            source,
        });
        return Ok(true);
    }
    Ok(false)
}

/// The message an operator sees — and the model does not.
///
/// A failed object is never handed to the reasoner to adjudicate. The runtime
/// says what broke and what the trusted replacement is; the security decision
/// belongs to the runtime, not to a model reading suspect bytes.
pub fn quarantine_notice(object_id: &str, generation: u64) -> String {
    format!(
        "Object {} failed integrity verification. \
         Trusted replacement available: object {}@gen{generation}.",
        &object_id[..object_id.len().min(12)],
        &object_id[..object_id.len().min(12)]
    )
}

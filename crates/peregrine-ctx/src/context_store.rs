//! `.context` — the container format.
//!
//! A single `.dcr.json` is a file that can be edited with any text editor and
//! reloaded without anything noticing. That is fine for a scratch store and
//! wrong for memory a reasoner is going to trust. This module makes the store
//! a **content-addressed, append-only, tamper-evident container**:
//!
//! ```text
//! .context/
//! ├── manifest              format, agent, root hash, highest generation, key ids
//! ├── generation.hwm        anti-rollback high-water mark
//! ├── objects/ab/cdef…      immutable, addressed by SHA-256 of their canonical form
//! ├── objects/ab/cdef….m    sidecar: hash, size, replicas, locations, verification
//! ├── checkpoints/000001    generation, merkle root, parent root, chain, policy hash
//! ├── signatures/000001     detached signature over the checkpoint
//! ├── indexes/              derived; rebuilt on load, never authoritative
//! └── quarantine/           objects that failed verification, plus why
//! ```
//!
//! Three properties fall out of that layout:
//!
//! * **An object's id is its digest.** Modifying an object changes its address,
//!   so there is no such thing as an edited object — only a new one.
//! * **A checkpoint chains to its parent.** `chain_n = H(chain_{n-1} ‖ Δ_n)`,
//!   so altering any historical generation invalidates every later one.
//! * **Generations only go up.** The high-water mark is checked on open, which
//!   is what stops an old-but-correctly-signed state being swapped in.
//!
//! # What this does and does not defend against
//!
//! Without a [`Signer`], the container is **tamper-evident, not
//! tamper-proof**: it detects bit rot, truncation, partial writes and casual
//! edits, because those do not come with a recomputed chain. An attacker with
//! write access to the directory *can* rewrite the objects, the checkpoints,
//! the manifest and `generation.hwm` together and produce a store that verifies
//! — nothing here prevents that, and no amount of hashing can. Signatures over
//! checkpoints are what raise the bar, and real anti-rollback needs the
//! high-water mark held somewhere the attacker cannot reach (a TPM, a secure
//! element, a remote witness). The format records enough to accept that
//! hardware later; this crate does not pretend to be it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::hash::{Digest, sha256, tagged};
use crate::json::{Json, parse};
use crate::merkle::{MerkleTree, Step, verify_proof};
use crate::trust::{
    AdmissionPolicy, Admitted, Candidate, ContextGateway, KeyId, KeyRole, Rejection, SCHEMA_VERSION,
    Signer, TrustLabel, Verification, Verifier, schema_hash,
};

pub const FORMAT_VERSION: u64 = 1;

const OBJECTS: &str = "objects";
const CHECKPOINTS: &str = "checkpoints";
const SIGNATURES: &str = "signatures";
const QUARANTINE: &str = "quarantine";
const INDEXES: &str = "indexes";
const MANIFEST: &str = "manifest";
const HWM: &str = "generation.hwm";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    Io(String),
    Parse(String),
    /// The bytes on disk do not hash to the address they were stored under.
    Corrupt { object: String, detail: String },
    /// A checkpoint does not chain to its parent.
    ChainBroken { generation: u64, detail: String },
    /// A state older than the high-water mark was offered.
    Rollback { found: u64, floor: u64 },
    /// The gateway refused an object.
    Refused {
        object: String,
        rejection: Rejection,
    },
    Missing(String),
    /// An object id was rebound to different content.
    Immutable(String),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::Io(m) => write!(f, "io error: {m}"),
            ContextError::Parse(m) => write!(f, "parse error: {m}"),
            ContextError::Corrupt { object, detail } => {
                write!(f, "object {object} failed integrity verification: {detail}")
            }
            ContextError::ChainBroken { generation, detail } => write!(
                f,
                "checkpoint {generation} does not chain to its parent: {detail}"
            ),
            ContextError::Rollback { found, floor } => write!(
                f,
                "refusing generation {found}: the highest generation accepted here is {floor} \
                 — this is a rollback"
            ),
            ContextError::Refused { object, rejection } => {
                write!(f, "object {object} refused at the gateway: {rejection}")
            }
            ContextError::Missing(what) => write!(f, "missing: {what}"),
            ContextError::Immutable(id) => write!(
                f,
                "object {id} already exists with different content; objects are immutable"
            ),
        }
    }
}

impl std::error::Error for ContextError {}

fn io<E: fmt::Display>(e: E) -> ContextError {
    ContextError::Io(e.to_string())
}

/// What an object holds. The kind is part of the hashed record, so an object
/// cannot be reinterpreted as a different kind than it was written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObjectKind {
    Document,
    Span,
    Node,
    Edge,
    /// Read counters, one object per generation.
    ///
    /// These live outside the node objects on purpose. A node's address is its
    /// content, so folding a read counter into it would mint a new copy of
    /// every admitted node on every query — an append-only store that grows
    /// with *reads* rather than with knowledge. Usage is telemetry about the
    /// memory, not part of it.
    Usage,
}

impl ObjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectKind::Document => "document",
            ObjectKind::Span => "span",
            ObjectKind::Node => "node",
            ObjectKind::Edge => "edge",
            ObjectKind::Usage => "usage",
        }
    }

    pub fn parse(text: &str) -> Option<ObjectKind> {
        match text {
            "document" => Some(ObjectKind::Document),
            "span" => Some(ObjectKind::Span),
            "node" => Some(ObjectKind::Node),
            "edge" => Some(ObjectKind::Edge),
            "usage" => Some(ObjectKind::Usage),
            _ => None,
        }
    }
}

/// One immutable object. Everything in here is hashed; nothing outside it is.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectRecord {
    pub kind: ObjectKind,
    /// The runtime-level name (`s_1f2e…`, `claim_9a…`). Distinct from the
    /// object id, which is the digest — one is a name, the other an address.
    pub logical_id: String,
    /// The generation this object first appeared in.
    ///
    /// Deliberately **not** part of the hashed body: an address must depend on
    /// content alone. Stamping the generation into the content would give
    /// unchanged material a new address at every commit, so the store would
    /// grow by its whole size on each write and content addressing would buy
    /// nothing. It is carried in the sidecar and restored on read.
    pub generation: u64,
    pub label: TrustLabel,
    /// True when this object was derived rather than observed. Derived objects
    /// with no `source_objects` are refused: that is a fact with no path back
    /// to raw material.
    pub derived: bool,
    pub source_objects: Vec<String>,
    pub created_by: String,
    pub body: Json,
}

impl ObjectRecord {
    pub fn new(kind: ObjectKind, logical_id: impl Into<String>, body: Json) -> ObjectRecord {
        ObjectRecord {
            kind,
            logical_id: logical_id.into(),
            generation: 0,
            label: TrustLabel::Trusted,
            derived: false,
            source_objects: Vec::new(),
            created_by: String::new(),
            body,
        }
    }

    pub fn derived_from(mut self, sources: Vec<String>) -> Self {
        self.derived = true;
        self.source_objects = sources;
        self
    }

    pub fn labelled(mut self, label: TrustLabel) -> Self {
        self.label = label;
        self
    }

    /// The exact bytes that are hashed and written. Canonical, so a record
    /// loaded and re-serialised reproduces its own address.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.to_json().to_canonical_string().into_bytes()
    }

    pub fn content_hash(&self) -> Digest {
        sha256(&self.canonical_bytes())
    }

    /// Content address: the digest, in hex. `objects/<first 2>/<rest>`.
    pub fn object_id(&self) -> String {
        self.content_hash().to_hex()
    }

    fn to_json(&self) -> Json {
        Json::obj(vec![
            ("schema", Json::str(SCHEMA_VERSION)),
            ("kind", Json::str(self.kind.as_str())),
            ("logical_id", Json::str(&self.logical_id)),
            ("label", Json::str(self.label.as_str())),
            ("derived", Json::Bool(self.derived)),
            (
                "source_objects",
                Json::Arr(self.source_objects.iter().map(Json::str).collect()),
            ),
            ("created_by", Json::str(&self.created_by)),
            ("body", self.body.clone()),
        ])
    }

    fn from_json(data: &Json) -> Result<ObjectRecord, ContextError> {
        let field = |name: &str| -> Result<&Json, ContextError> {
            data.get(name)
                .ok_or_else(|| ContextError::Parse(format!("object has no {name}")))
        };
        let kind = ObjectKind::parse(field("kind")?.as_str().unwrap_or_default())
            .ok_or_else(|| ContextError::Parse("unknown object kind".into()))?;
        Ok(ObjectRecord {
            kind,
            logical_id: field("logical_id")?.as_str().unwrap_or_default().to_string(),
            // Filled in from the sidecar by the caller: it is not in the body.
            generation: 0,
            label: TrustLabel::parse(field("label")?.as_str().unwrap_or_default())
                .ok_or_else(|| ContextError::Parse("unknown trust label".into()))?,
            derived: data.get("derived").and_then(Json::as_bool).unwrap_or(false),
            source_objects: data
                .get("source_objects")
                .map(Json::string_list)
                .unwrap_or_default(),
            created_by: data
                .get("created_by")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            body: field("body")?.clone(),
        })
    }
}

/// The sidecar. Deliberately *not* part of the hashed record: it describes
/// where the bytes live and how many copies exist, which changes as replicas
/// come and go without the object itself changing.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectMeta {
    pub hash: Digest,
    pub size: usize,
    pub generation: u64,
    pub verification: Verification,
    pub locations: Vec<String>,
}

impl ObjectMeta {
    fn to_json(&self) -> Json {
        Json::obj(vec![
            ("hash", Json::str(self.hash.to_hex())),
            ("size", Json::num(self.size as f64)),
            ("generation", Json::num(self.generation as f64)),
            ("verification", Json::str(self.verification.as_str())),
            (
                "replicas",
                Json::num(self.locations.len().max(1) as f64),
            ),
            (
                "locations",
                Json::Arr(self.locations.iter().map(Json::str).collect()),
            ),
        ])
    }

    fn from_json(data: &Json) -> Option<ObjectMeta> {
        Some(ObjectMeta {
            hash: Digest::parse_hex(data.get("hash")?.as_str()?)?,
            size: data.get("size").and_then(Json::as_usize).unwrap_or_default(),
            generation: data
                .get("generation")
                .and_then(Json::as_u64)
                .unwrap_or_default(),
            verification: data
                .get("verification")
                .and_then(Json::as_str)
                .and_then(Verification::parse)
                .unwrap_or(Verification::Unsigned),
            locations: data
                .get("locations")
                .map(Json::string_list)
                .unwrap_or_default(),
        })
    }
}

/// A signed (or at least chained) state of the whole container.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// The schema string as it was written.
    ///
    /// Carried rather than re-derived from the constant: an earlier version
    /// rebuilt this field from `SCHEMA_VERSION` when recomputing the chain, so
    /// whatever was on disk never entered the digest and could be edited for
    /// free. A field the verifier reconstructs is a field the verifier does not
    /// check.
    pub schema: String,
    pub generation: u64,
    pub merkle_root: Digest,
    pub parent_root: Digest,
    /// `H(parent_chain ‖ canonical(delta))` — the append-only history.
    pub chain: Digest,
    pub parent_chain: Digest,
    pub timestamp: u64,
    pub policy_hash: Digest,
    pub schema_hash: Digest,
    pub object_count: usize,
    /// Object ids added since the parent checkpoint. This *is* the delta the
    /// chain commits to.
    pub added: Vec<String>,
}

impl Checkpoint {
    /// The bytes a signature covers: the whole checkpoint, chain digest
    /// included, so a signature commits to the history as well as the state.
    pub fn signing_bytes(&self) -> Vec<u8> {
        self.to_json().to_canonical_string().into_bytes()
    }

    /// Everything the chain digest covers: the whole checkpoint except the
    /// digest itself.
    ///
    /// Covering the *whole* body rather than a chosen few fields is the point.
    /// An earlier version chained only the root and the delta, which left the
    /// timestamp, the object count and the policy hash editable without
    /// breaking anything — a signature would have caught it, and an unsigned
    /// store would not have.
    fn body_json(&self) -> Json {
        Json::obj(vec![
            ("schema", Json::str(&self.schema)),
            ("generation", Json::num(self.generation as f64)),
            ("merkle_root", Json::str(self.merkle_root.to_hex())),
            ("parent_root", Json::str(self.parent_root.to_hex())),
            ("parent_chain", Json::str(self.parent_chain.to_hex())),
            ("timestamp", Json::num(self.timestamp as f64)),
            ("policy_hash", Json::str(self.policy_hash.to_hex())),
            ("schema_hash", Json::str(self.schema_hash.to_hex())),
            ("object_count", Json::num(self.object_count as f64)),
            ("added", Json::Arr(self.added.iter().map(Json::str).collect())),
        ])
    }

    fn to_json(&self) -> Json {
        let Json::Obj(mut fields) = self.body_json() else {
            return self.body_json();
        };
        fields.push(("chain".to_string(), Json::str(self.chain.to_hex())));
        Json::Obj(fields)
    }

    /// The chain digest this checkpoint should carry, given its parent.
    fn compute_chain(&self) -> Digest {
        tagged(
            "dcr:chain",
            &[
                self.parent_chain.as_bytes(),
                self.body_json().to_canonical_string().as_bytes(),
            ],
        )
    }

    fn from_json(data: &Json) -> Option<Checkpoint> {
        let digest = |name: &str| Digest::parse_hex(data.get(name)?.as_str()?);
        let schema = data.get("schema")?.as_str()?;
        // Refuse a schema this runtime does not implement, *and* carry the
        // value so the chain covers it. Either check alone leaves a gap: the
        // first reopens the moment migration is added, the second would accept
        // a well-chained checkpoint written by a future format.
        if schema != SCHEMA_VERSION {
            return None;
        }
        Some(Checkpoint {
            schema: schema.to_string(),
            generation: data.get("generation")?.as_u64()?,
            merkle_root: digest("merkle_root")?,
            parent_root: digest("parent_root")?,
            chain: digest("chain")?,
            parent_chain: digest("parent_chain")?,
            timestamp: data.get("timestamp").and_then(Json::as_u64).unwrap_or(0),
            policy_hash: digest("policy_hash")?,
            schema_hash: digest("schema_hash")?,
            object_count: data
                .get("object_count")
                .and_then(Json::as_usize)
                .unwrap_or_default(),
            added: data.get("added").map(Json::string_list).unwrap_or_default(),
        })
    }

}

/// The container's front matter.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub format_version: u64,
    pub agent_id: String,
    pub root_hash: Digest,
    pub highest_generation: u64,
    pub encryption_key_id: Option<KeyId>,
    pub signing_key_id: Option<KeyId>,
    /// Whether the signing key offers real cryptographic protection. A store
    /// signed by [`crate::trust::InsecureDevSigner`] says `false` here, so an
    /// audit cannot mistake it for a protected store.
    pub signing_is_cryptographic: bool,
    pub schema_hash: Digest,
    pub policy_hash: Digest,
    pub created_at: u64,
}

impl Manifest {
    fn to_json(&self) -> Json {
        let key = |k: &Option<KeyId>| match k {
            Some(key) => Json::str(key.to_string()),
            None => Json::Null,
        };
        Json::obj(vec![
            ("format_version", Json::num(self.format_version as f64)),
            ("schema", Json::str(SCHEMA_VERSION)),
            ("agent_id", Json::str(&self.agent_id)),
            ("root_hash", Json::str(self.root_hash.to_hex())),
            (
                "highest_generation",
                Json::num(self.highest_generation as f64),
            ),
            ("encryption_key_id", key(&self.encryption_key_id)),
            ("signing_key_id", key(&self.signing_key_id)),
            (
                "signing_is_cryptographic",
                Json::Bool(self.signing_is_cryptographic),
            ),
            ("schema_hash", Json::str(self.schema_hash.to_hex())),
            ("policy_hash", Json::str(self.policy_hash.to_hex())),
            ("created_at", Json::num(self.created_at as f64)),
        ])
    }

    fn from_json(data: &Json) -> Result<Manifest, ContextError> {
        let missing = |what: &str| ContextError::Parse(format!("manifest has no {what}"));
        let key = |name: &str| -> Option<KeyId> {
            let text = data.get(name)?.as_str()?;
            let (role, id) = text.split_once(':')?;
            Some(KeyId::new(KeyRole::parse(role)?, id))
        };
        Ok(Manifest {
            format_version: data
                .get("format_version")
                .and_then(Json::as_u64)
                .ok_or_else(|| missing("format_version"))?,
            agent_id: data
                .get("agent_id")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            root_hash: data
                .get("root_hash")
                .and_then(Json::as_str)
                .and_then(Digest::parse_hex)
                .ok_or_else(|| missing("root_hash"))?,
            highest_generation: data
                .get("highest_generation")
                .and_then(Json::as_u64)
                .unwrap_or_default(),
            encryption_key_id: key("encryption_key_id"),
            signing_key_id: key("signing_key_id"),
            signing_is_cryptographic: data
                .get("signing_is_cryptographic")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            schema_hash: data
                .get("schema_hash")
                .and_then(Json::as_str)
                .and_then(Digest::parse_hex)
                .ok_or_else(|| missing("schema_hash"))?,
            policy_hash: data
                .get("policy_hash")
                .and_then(Json::as_str)
                .and_then(Digest::parse_hex)
                .unwrap_or_default(),
            created_at: data.get("created_at").and_then(Json::as_u64).unwrap_or(0),
        })
    }
}

/// What `verify` found. A report rather than a bool, because "one object is
/// corrupt" and "the chain is broken" call for different responses.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VerifyReport {
    pub objects_checked: usize,
    pub objects_failed: Vec<String>,
    pub checkpoints_checked: usize,
    pub chain_intact: bool,
    pub root_matches: bool,
    pub signatures_checked: usize,
    pub signatures_failed: Vec<u64>,
    pub quarantined: usize,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.objects_failed.is_empty()
            && self.signatures_failed.is_empty()
            && self.chain_intact
            && self.root_matches
    }
}

impl fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "objects   : {} checked, {} failed",
            self.objects_checked,
            self.objects_failed.len()
        )?;
        writeln!(
            f,
            "checkpoints: {} checked, chain {}",
            self.checkpoints_checked,
            if self.chain_intact { "intact" } else { "BROKEN" }
        )?;
        writeln!(
            f,
            "merkle root: {}",
            if self.root_matches {
                "matches manifest"
            } else {
                "DOES NOT MATCH MANIFEST"
            }
        )?;
        writeln!(
            f,
            "signatures : {} checked, {} failed",
            self.signatures_checked,
            self.signatures_failed.len()
        )?;
        write!(f, "quarantine : {} objects", self.quarantined)?;
        for id in &self.objects_failed {
            write!(f, "\n  FAILED {}", &id[..id.len().min(16)])?;
        }
        Ok(())
    }
}

/// The container.
#[derive(Debug)]
pub struct ContextStore {
    root: PathBuf,
    manifest: Manifest,
    /// object id → sidecar. Sorted, so the Merkle tree is deterministic.
    objects: BTreeMap<String, ObjectMeta>,
    checkpoints: Vec<Checkpoint>,
    /// Written since the last checkpoint — the next delta.
    pending: BTreeSet<String>,
    /// Extra directories holding full copies. Empty means no redundancy, and
    /// the scrubber says so rather than implying otherwise.
    replicas: Vec<PathBuf>,
    policy: AdmissionPolicy,
}

impl ContextStore {
    // -- lifecycle ---------------------------------------------------------

    /// Create a new container. Fails if one is already there — creating over a
    /// store would be an overwrite, and nothing here overwrites.
    pub fn create(root: impl AsRef<Path>, agent_id: &str) -> Result<ContextStore, ContextError> {
        let root = root.as_ref().to_path_buf();
        if root.join(MANIFEST).exists() {
            return Err(ContextError::Immutable(format!(
                "{} already holds a container",
                root.display()
            )));
        }
        for dir in [OBJECTS, CHECKPOINTS, SIGNATURES, QUARANTINE, INDEXES] {
            std::fs::create_dir_all(root.join(dir)).map_err(io)?;
        }
        let policy = AdmissionPolicy::default();
        let store = ContextStore {
            manifest: Manifest {
                format_version: FORMAT_VERSION,
                agent_id: agent_id.to_string(),
                root_hash: crate::merkle::empty_root(),
                highest_generation: 0,
                encryption_key_id: None,
                signing_key_id: None,
                signing_is_cryptographic: false,
                schema_hash: schema_hash(),
                policy_hash: policy.policy_hash(),
                created_at: 0,
            },
            root,
            objects: BTreeMap::new(),
            checkpoints: Vec::new(),
            pending: BTreeSet::new(),
            replicas: Vec::new(),
            policy,
        };
        store.write_manifest()?;
        store.write_hwm(0)?;
        Ok(store)
    }

    /// Open an existing container, checking the manifest, the chain and the
    /// anti-rollback high-water mark. Object bytes are *not* all re-hashed
    /// here — that is [`ContextStore::verify`], which is a scrub, not a load.
    pub fn open(root: impl AsRef<Path>) -> Result<ContextStore, ContextError> {
        let root = root.as_ref().to_path_buf();
        let manifest = Manifest::from_json(&read_json(&root.join(MANIFEST))?)?;
        if manifest.format_version > FORMAT_VERSION {
            return Err(ContextError::Parse(format!(
                "container format {} is newer than this runtime's {FORMAT_VERSION}",
                manifest.format_version
            )));
        }
        if manifest.schema_hash != schema_hash() {
            return Err(ContextError::Refused {
                object: MANIFEST.to_string(),
                rejection: Rejection::SchemaMismatch {
                    expected: schema_hash(),
                    actual: manifest.schema_hash,
                },
            });
        }

        // Anti-rollback: the mark is read before anything else is believed.
        let hwm = read_hwm(&root)?;
        if manifest.highest_generation < hwm {
            return Err(ContextError::Rollback {
                found: manifest.highest_generation,
                floor: hwm,
            });
        }

        let mut objects = BTreeMap::new();
        for (id, path) in object_paths(&root)? {
            let meta_path = path.with_extension("m");
            let meta = if meta_path.exists() {
                ObjectMeta::from_json(&read_json(&meta_path)?).ok_or_else(|| {
                    ContextError::Parse(format!("sidecar for {id} is unreadable"))
                })?
            } else {
                // A sidecar can be rebuilt from the object; the object cannot
                // be rebuilt from the sidecar. Missing sidecar is recoverable.
                let bytes = std::fs::read(&path).map_err(io)?;
                ObjectMeta {
                    hash: sha256(&bytes),
                    size: bytes.len(),
                    generation: manifest.highest_generation,
                    verification: Verification::Unsigned,
                    locations: vec![relative(&root, &path)],
                }
            };
            objects.insert(id, meta);
        }

        let mut checkpoints = Vec::new();
        for path in checkpoint_paths(&root)? {
            let checkpoint = Checkpoint::from_json(&read_json(&path)?).ok_or_else(|| {
                ContextError::Parse(format!("checkpoint {} is unreadable", path.display()))
            })?;
            checkpoints.push(checkpoint);
        }
        checkpoints.sort_by_key(|c| c.generation);
        verify_chain(&checkpoints)?;

        if let Some(last) = checkpoints.last() {
            if last.generation != manifest.highest_generation {
                return Err(ContextError::ChainBroken {
                    generation: last.generation,
                    detail: format!(
                        "manifest claims generation {} but the newest checkpoint is {}",
                        manifest.highest_generation, last.generation
                    ),
                });
            }
        }

        let policy = AdmissionPolicy {
            generation_ceiling: manifest.highest_generation,
            require_signature: manifest.signing_key_id.is_some(),
            ..AdmissionPolicy::default()
        };

        Ok(ContextStore {
            root,
            manifest,
            objects,
            checkpoints,
            pending: BTreeSet::new(),
            replicas: Vec::new(),
            policy,
        })
    }

    /// Register a directory holding a full copy of the object store. Repair
    /// reads from these — and only ever from a copy that verifies on its own.
    pub fn add_replica(&mut self, path: impl AsRef<Path>) {
        self.replicas.push(path.as_ref().to_path_buf());
    }

    pub fn replicas(&self) -> &[PathBuf] {
        &self.replicas
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn generation(&self) -> u64 {
        self.manifest.highest_generation
    }

    pub fn root_hash(&self) -> Digest {
        self.manifest.root_hash
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn object_ids(&self) -> impl Iterator<Item = &String> {
        self.objects.keys()
    }

    pub fn meta(&self, object_id: &str) -> Option<&ObjectMeta> {
        self.objects.get(object_id)
    }

    pub fn policy(&self) -> &AdmissionPolicy {
        &self.policy
    }

    // -- objects -----------------------------------------------------------

    /// Write an object. Returns its content address.
    ///
    /// Idempotent by construction: the same content produces the same address,
    /// so re-putting an object is a no-op rather than a duplicate. Writing is
    /// tmp-plus-rename, so a crash mid-write leaves either the old state or the
    /// new one, never half an object.
    pub fn put(&mut self, mut record: ObjectRecord) -> Result<String, ContextError> {
        let generation = self.manifest.highest_generation + 1;
        record.generation = generation;
        if record.created_by.is_empty() {
            record.created_by = self.manifest.agent_id.clone();
        }
        if record.derived && record.source_objects.is_empty() {
            return Err(ContextError::Refused {
                object: record.logical_id.clone(),
                rejection: Rejection::ProvenanceMissing,
            });
        }

        let bytes = record.canonical_bytes();
        let hash = sha256(&bytes);
        let id = hash.to_hex();
        let path = self.object_path(&id);

        if let Some(existing) = self.objects.get(&id) {
            // Same address, same content — by definition. Nothing to write.
            if existing.hash == hash {
                return Ok(id);
            }
            return Err(ContextError::Immutable(id));
        }

        std::fs::create_dir_all(path.parent().unwrap_or(&self.root)).map_err(io)?;
        write_atomic(&path, &bytes)?;

        let mut locations = vec![relative(&self.root, &path)];
        for replica in &self.replicas {
            let target = replica.join(OBJECTS).join(&id[..2]).join(&id[2..]);
            std::fs::create_dir_all(target.parent().unwrap_or(replica)).map_err(io)?;
            write_atomic(&target, &bytes)?;
            locations.push(target.display().to_string());
        }

        let meta = ObjectMeta {
            hash,
            size: bytes.len(),
            generation,
            verification: Verification::Unsigned,
            locations,
        };
        write_atomic(
            &path.with_extension("m"),
            meta.to_json().to_json_string().as_bytes(),
        )?;

        self.objects.insert(id.clone(), meta);
        self.pending.insert(id.clone());
        Ok(id)
    }

    /// Read an object, verifying its bytes against its address on the way out.
    ///
    /// Verification happens here rather than at the caller because "read the
    /// object" and "check the object" must not be two things a caller can
    /// forget to do in the right order.
    pub fn get(&self, object_id: &str) -> Result<ObjectRecord, ContextError> {
        let bytes = self.read_object_bytes(object_id)?;
        let mut record = ObjectRecord::from_json(&parse_bytes(&bytes)?)?;
        record.generation = self
            .objects
            .get(object_id)
            .map(|m| m.generation)
            .unwrap_or_default();
        // Re-serialising must reproduce the address, which also proves the
        // canonical encoding is stable across a round-trip.
        if record.content_hash().to_hex() != object_id {
            return Err(ContextError::Corrupt {
                object: object_id.to_string(),
                detail: "canonical re-encoding does not reproduce the address".into(),
            });
        }
        Ok(record)
    }

    /// Read and pass through the gateway. This is the only path that should
    /// feed a reasoner: `get` proves the bytes are intact, `admit` decides
    /// whether intact content is allowed to be believed.
    pub fn admit(
        &self,
        object_id: &str,
        verifier: Option<&dyn Verifier>,
    ) -> Result<(ObjectRecord, Admitted), ContextError> {
        let bytes = self.read_object_bytes(object_id)?;
        let mut record = ObjectRecord::from_json(&parse_bytes(&bytes)?)?;
        record.generation = self
            .objects
            .get(object_id)
            .map(|m| m.generation)
            .unwrap_or_default();
        let mut gateway = ContextGateway::new(self.policy.clone());
        if let Some(verifier) = verifier {
            gateway = gateway.with_verifier(verifier);
        }
        let declared = self
            .objects
            .get(object_id)
            .map(|m| m.hash)
            .unwrap_or_else(|| sha256(&bytes));
        let candidate = Candidate {
            object_id,
            bytes: &bytes,
            declared_hash: declared,
            generation: record.generation,
            schema_hash: schema_hash(),
            label: record.label,
            signature: None,
            signing_key: None,
            source_objects: &record.source_objects,
            derived: record.derived,
        };
        let admitted = gateway
            .admit(&candidate)
            .map_err(|rejection| ContextError::Refused {
                object: object_id.to_string(),
                rejection,
            })?;
        Ok((record, admitted))
    }

    fn read_object_bytes(&self, object_id: &str) -> Result<Vec<u8>, ContextError> {
        let path = self.object_path(object_id);
        if !path.exists() {
            return Err(ContextError::Missing(format!("object {object_id}")));
        }
        let bytes = std::fs::read(&path).map_err(io)?;
        let actual = sha256(&bytes);
        if actual.to_hex() != object_id {
            return Err(ContextError::Corrupt {
                object: object_id.to_string(),
                detail: format!("computed {}", actual.short(16)),
            });
        }
        Ok(bytes)
    }

    pub fn object_path(&self, object_id: &str) -> PathBuf {
        object_path_in(&self.root, object_id)
    }

    /// Every copy of an object held outside the primary store, with the path it
    /// came from. Used by the scrubber; each copy still has to prove itself
    /// before it is allowed to repair anything.
    pub fn replica_copies(&self, object_id: &str) -> Vec<(PathBuf, Vec<u8>)> {
        self.replicas
            .iter()
            .filter_map(|replica| {
                let path = object_path_in(replica, object_id);
                let bytes = std::fs::read(&path).ok()?;
                Some((path, bytes))
            })
            .collect()
    }

    /// Put known-good bytes back at an object's address.
    ///
    /// Refuses anything that does not hash to the address it is being written
    /// to, so a repair cannot be the vector that installs corrupt content.
    pub fn restore_object(&mut self, object_id: &str, bytes: &[u8]) -> Result<(), ContextError> {
        let hash = sha256(bytes);
        if hash.to_hex() != object_id {
            return Err(ContextError::Corrupt {
                object: object_id.to_string(),
                detail: format!(
                    "refusing to restore bytes that hash to {}",
                    hash.short(16)
                ),
            });
        }
        let path = self.object_path(object_id);
        write_atomic(&path, bytes)?;
        let meta = ObjectMeta {
            hash,
            size: bytes.len(),
            generation: self
                .objects
                .get(object_id)
                .map(|m| m.generation)
                .unwrap_or(self.manifest.highest_generation),
            verification: Verification::Verified,
            locations: vec![relative(&self.root, &path)],
        };
        write_atomic(
            &path.with_extension("m"),
            meta.to_json().to_json_string().as_bytes(),
        )?;
        self.objects.insert(object_id.to_string(), meta);
        Ok(())
    }

    // -- checkpoints -------------------------------------------------------

    /// Seal everything written since the last checkpoint into a new
    /// generation: new Merkle root, chained to the parent, optionally signed.
    pub fn commit(
        &mut self,
        timestamp: u64,
        signer: Option<&dyn Signer>,
    ) -> Result<Checkpoint, ContextError> {
        let tree = self.tree();
        let parent = self.checkpoints.last().cloned();

        // Nothing was written and the root is unchanged: there is no new state
        // to attest to. Minting a generation anyway would make every read look
        // like a write in the audit log.
        if self.pending.is_empty() && tree.root() == self.manifest.root_hash {
            if let Some(parent) = parent {
                return Ok(parent);
            }
        }

        let generation = self.manifest.highest_generation + 1;
        let parent_root = parent
            .as_ref()
            .map(|c| c.merkle_root)
            .unwrap_or_else(crate::merkle::empty_root);
        let parent_chain = parent.as_ref().map(|c| c.chain).unwrap_or_default();
        let added: Vec<String> = self.pending.iter().cloned().collect();
        let merkle_root = tree.root();

        let mut checkpoint = Checkpoint {
            schema: SCHEMA_VERSION.to_string(),
            generation,
            merkle_root,
            parent_root,
            chain: Digest::default(),
            parent_chain,
            timestamp,
            policy_hash: self.policy.policy_hash(),
            schema_hash: schema_hash(),
            object_count: self.objects.len(),
            added,
        };
        checkpoint.chain = checkpoint.compute_chain();

        write_atomic(
            &self.checkpoint_path(generation),
            checkpoint.to_json().to_json_string().as_bytes(),
        )?;

        if let Some(signer) = signer {
            let key = signer.key_id();
            if key.role != KeyRole::ContextSigning {
                return Err(ContextError::Refused {
                    object: format!("checkpoint {generation}"),
                    rejection: Rejection::SignatureInvalid { key },
                });
            }
            let signature = signer.sign(&checkpoint.signing_bytes());
            let record = Json::obj(vec![
                ("key", Json::str(key.to_string())),
                ("cryptographic", Json::Bool(signer.is_cryptographic())),
                ("generation", Json::num(generation as f64)),
                ("signature", Json::str(hex(&signature))),
            ]);
            write_atomic(
                &self.signature_path(generation),
                record.to_json_string().as_bytes(),
            )?;
            self.manifest.signing_key_id = Some(key);
            self.manifest.signing_is_cryptographic = signer.is_cryptographic();
            self.policy.require_signature = true;
        }

        // Order matters on a crash: the checkpoint exists before the manifest
        // points at it, and the high-water mark rises last. A crash between
        // any two of these leaves a store that opens, at the older generation.
        self.manifest.highest_generation = generation;
        self.manifest.root_hash = merkle_root;
        self.manifest.policy_hash = self.policy.policy_hash();
        self.policy.generation_ceiling = generation;
        self.write_manifest()?;
        self.write_hwm(generation)?;

        self.checkpoints.push(checkpoint.clone());
        self.pending.clear();
        Ok(checkpoint)
    }

    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Inclusion proof for one object against the current root.
    pub fn proof(&self, object_id: &str) -> Option<Vec<Step>> {
        self.tree().proof(object_id)
    }

    fn tree(&self) -> MerkleTree {
        MerkleTree::build(self.objects.iter().map(|(id, meta)| (id.clone(), meta.hash)))
    }

    // -- verification ------------------------------------------------------

    /// Full scrub of the container: every object re-hashed, every checkpoint
    /// re-chained, the root compared against the manifest, and any signatures
    /// checked if a verifier is supplied.
    pub fn verify(&self, verifier: Option<&dyn Verifier>) -> VerifyReport {
        let mut report = VerifyReport {
            chain_intact: true,
            ..VerifyReport::default()
        };

        for (id, meta) in &self.objects {
            report.objects_checked += 1;
            match std::fs::read(self.object_path(id)) {
                Ok(bytes) if sha256(&bytes) == meta.hash && meta.hash.to_hex() == *id => {}
                _ => report.objects_failed.push(id.clone()),
            }
        }

        report.checkpoints_checked = self.checkpoints.len();
        report.chain_intact = verify_chain(&self.checkpoints).is_ok();
        report.root_matches = self.tree().root() == self.manifest.root_hash;

        if let Some(verifier) = verifier {
            for checkpoint in &self.checkpoints {
                let path = self.signature_path(checkpoint.generation);
                if !path.exists() {
                    continue;
                }
                report.signatures_checked += 1;
                let ok = read_json(&path)
                    .ok()
                    .and_then(|data| {
                        let key = data.get("key")?.as_str()?;
                        let (role, id) = key.split_once(':')?;
                        let key = KeyId::new(KeyRole::parse(role)?, id);
                        let signature = unhex(data.get("signature")?.as_str()?)?;
                        Some(
                            !verifier.is_revoked(&key)
                                && verifier.verify(
                                    &key,
                                    &checkpoint.signing_bytes(),
                                    &signature,
                                ),
                        )
                    })
                    .unwrap_or(false);
                if !ok {
                    report.signatures_failed.push(checkpoint.generation);
                }
            }
        }

        report.quarantined = quarantine_entries(&self.root).unwrap_or_default().len();
        report
    }

    /// Move a failed object out of the store and record why.
    ///
    /// The reason record is for an operator, not for the reasoner: a model is
    /// never asked to decide whether a failed object should be believed.
    pub fn quarantine(&mut self, object_id: &str, reason: &str) -> Result<PathBuf, ContextError> {
        let path = self.object_path(object_id);
        let target = self.root.join(QUARANTINE).join(object_id);
        std::fs::create_dir_all(self.root.join(QUARANTINE)).map_err(io)?;
        if path.exists() {
            std::fs::rename(&path, &target).map_err(io)?;
        }
        let meta_path = path.with_extension("m");
        if meta_path.exists() {
            let _ = std::fs::remove_file(&meta_path);
        }
        let record = Json::obj(vec![
            ("object_id", Json::str(object_id)),
            ("reason", Json::str(reason)),
            ("generation", Json::num(self.manifest.highest_generation as f64)),
            ("state", Json::str(Verification::Quarantined.as_str())),
        ]);
        write_atomic(
            &self.root.join(QUARANTINE).join(format!("{object_id}.why")),
            record.to_json_string().as_bytes(),
        )?;
        self.objects.remove(object_id);
        self.pending.remove(object_id);
        Ok(target)
    }

    pub fn quarantined(&self) -> Vec<(String, String)> {
        quarantine_entries(&self.root).unwrap_or_default()
    }

    /// Is this object still provable against the current root? The check a
    /// repaired replica has to pass before its bytes are put back.
    pub fn proves(&self, object_id: &str, content: &Digest) -> bool {
        match self.proof(object_id) {
            Some(steps) => verify_proof(object_id, content, &steps, &self.manifest.root_hash),
            None => false,
        }
    }

    // -- persistence helpers ----------------------------------------------

    fn write_manifest(&self) -> Result<(), ContextError> {
        write_atomic(
            &self.root.join(MANIFEST),
            self.manifest.to_json().to_json_string().as_bytes(),
        )
    }

    /// The high-water mark, with a guard digest so truncation is visible.
    ///
    /// This is honest-but-weak on its own: an attacker who can write the
    /// directory can recompute the guard. Real anti-rollback needs this value
    /// somewhere the attacker cannot write. The file is the interface for that
    /// hardware, not a substitute for it.
    fn write_hwm(&self, generation: u64) -> Result<(), ContextError> {
        let record = Json::obj(vec![
            ("highest_generation", Json::num(generation as f64)),
            (
                "guard",
                Json::str(tagged("dcr:hwm", &[&generation.to_be_bytes()]).to_hex()),
            ),
        ]);
        write_atomic(
            &self.root.join(HWM),
            record.to_json_string().as_bytes(),
        )
    }

    fn checkpoint_path(&self, generation: u64) -> PathBuf {
        self.root.join(CHECKPOINTS).join(format!("{generation:06}"))
    }

    fn signature_path(&self, generation: u64) -> PathBuf {
        self.root.join(SIGNATURES).join(format!("{generation:06}"))
    }
}

// -- free helpers ---------------------------------------------------------

fn object_path_in(root: &Path, object_id: &str) -> PathBuf {
    if object_id.len() < 3 {
        return root.join(OBJECTS).join(object_id);
    }
    root.join(OBJECTS)
        .join(&object_id[..2])
        .join(&object_id[2..])
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Write via a temporary file and rename. A partial write must never be
/// mistakable for a whole object, and rename within a directory is the closest
/// thing the filesystem offers to atomic.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ContextError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(io)?;
    std::fs::rename(&tmp, path).map_err(io)
}

fn read_json(path: &Path) -> Result<Json, ContextError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ContextError::Io(format!("{}: {e}", path.display())))?;
    parse(&text).map_err(ContextError::Parse)
}

fn parse_bytes(bytes: &[u8]) -> Result<Json, ContextError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| ContextError::Parse(format!("object is not utf-8: {e}")))?;
    parse(text).map_err(ContextError::Parse)
}

fn read_hwm(root: &Path) -> Result<u64, ContextError> {
    let path = root.join(HWM);
    if !path.exists() {
        return Ok(0);
    }
    let data = read_json(&path)?;
    let generation = data
        .get("highest_generation")
        .and_then(Json::as_u64)
        .ok_or_else(|| ContextError::Parse("high-water mark is unreadable".into()))?;
    let guard = data
        .get("guard")
        .and_then(Json::as_str)
        .and_then(Digest::parse_hex);
    if guard != Some(tagged("dcr:hwm", &[&generation.to_be_bytes()])) {
        return Err(ContextError::Corrupt {
            object: HWM.to_string(),
            detail: "guard digest does not match the recorded generation".into(),
        });
    }
    Ok(generation)
}

fn object_paths(root: &Path) -> Result<Vec<(String, PathBuf)>, ContextError> {
    let objects = root.join(OBJECTS);
    let mut found = Vec::new();
    if !objects.is_dir() {
        return Ok(found);
    }
    for shard in std::fs::read_dir(&objects).map_err(io)?.flatten() {
        if !shard.path().is_dir() {
            continue;
        }
        let prefix = shard.file_name().to_string_lossy().to_string();
        for entry in std::fs::read_dir(shard.path()).map_err(io)?.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".m") || name.ends_with(".tmp") {
                continue;
            }
            found.push((format!("{prefix}{name}"), path));
        }
    }
    found.sort();
    Ok(found)
}

fn checkpoint_paths(root: &Path) -> Result<Vec<PathBuf>, ContextError> {
    let dir = root.join(CHECKPOINTS);
    let mut found = Vec::new();
    if !dir.is_dir() {
        return Ok(found);
    }
    for entry in std::fs::read_dir(&dir).map_err(io)?.flatten() {
        let path = entry.path();
        if path.is_file() && !path.to_string_lossy().ends_with(".tmp") {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

fn quarantine_entries(root: &Path) -> Result<Vec<(String, String)>, ContextError> {
    let dir = root.join(QUARANTINE);
    let mut found = Vec::new();
    if !dir.is_dir() {
        return Ok(found);
    }
    for entry in std::fs::read_dir(&dir).map_err(io)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(id) = name.strip_suffix(".why") else {
            continue;
        };
        let reason = read_json(&entry.path())
            .ok()
            .and_then(|d| d.get("reason").and_then(Json::as_str).map(str::to_string))
            .unwrap_or_default();
        found.push((id.to_string(), reason));
    }
    found.sort();
    Ok(found)
}

/// Walk the checkpoints in order and confirm each one chains to its parent.
fn verify_chain(checkpoints: &[Checkpoint]) -> Result<(), ContextError> {
    let mut previous: Option<&Checkpoint> = None;
    for checkpoint in checkpoints {
        let (parent_chain, parent_root, expected_generation) = match previous {
            Some(parent) => (parent.chain, parent.merkle_root, parent.generation + 1),
            None => (Digest::default(), crate::merkle::empty_root(), 1),
        };
        if checkpoint.generation != expected_generation {
            return Err(ContextError::ChainBroken {
                generation: checkpoint.generation,
                detail: format!("expected generation {expected_generation}"),
            });
        }
        if checkpoint.parent_chain != parent_chain || checkpoint.parent_root != parent_root {
            return Err(ContextError::ChainBroken {
                generation: checkpoint.generation,
                detail: "parent pointers do not match the previous checkpoint".into(),
            });
        }
        if checkpoint.chain != checkpoint.compute_chain() {
            return Err(ContextError::ChainBroken {
                generation: checkpoint.generation,
                detail: "chain digest does not match this checkpoint's contents".into(),
            });
        }
        previous = Some(checkpoint);
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const HEX: [u8; 16] = *b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(text.len() / 2);
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

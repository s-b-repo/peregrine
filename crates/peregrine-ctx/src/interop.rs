//! The **responder** half of the `unifur-subnext-interop` file channel.
//!
//! unifur (`/home/cortix/Downloads/inc/unifur/src/interop.rs`) owns the
//! contract: it writes files into a shared directory and waits for this side
//! to answer. This module is the other end — the side that actually holds the
//! context, answers queries about it, and takes writes into it.
//!
//! Without it every `nina chat` turn degrades to its bare prompt: the model
//! asks, nothing answers, and the "context lives outside the weights" claim is
//! untested end to end. With it, the claim is a file on disk that a model
//! reads and a runtime writes.
//!
//! # What is served
//!
//! A [`ChannelRequest`] carries a [`ContextQuery`]; the answer is a
//! [`ChannelResponse`] holding the [`ContextHit`]s of a planned working set.
//! The hits are the *same* admissions `dcr ask --show-context` would print,
//! so the answer is budget-bounded and provenance-bearing rather than a raw
//! similarity dump: every hit names the source span it came from.
//!
//! # What is taken
//!
//! A [`ContextPut`] or a [`MemoryAppend`] handed over on the write channel is
//! ingested through [`Dcr::ingest`], the one entry point the peer's note
//! commits to, and is retrievable by any later query. `ContextPut.text` is
//! stored as the document. `MemoryAppend.payload` is a JSON value the peer's
//! own doc comment calls *"owned by the memory log, not the model"*, so this
//! side does not pick a rendering for it: it is indexed in canonical JSON,
//! which is lossless and stable under key reordering.
//!
//! The memory log is the `.context` container — objects under a Merkle root,
//! each generation chained to its parent by [`Dcr::save_context`]. A write
//! therefore enters an append-only, tamper-evident log rather than a
//! rewritable file. Nothing in this file can edit a weight: there is no
//! training path here to call.
//!
//! The store is sealed **before** the acknowledgements go out: a crash between
//! the two leaves a peer that will retry, and a retry is free because L0 treats
//! identical text under a used id as a no-op. The other order would leave a
//! peer holding an acknowledgement for a document that only ever existed in
//! memory, and "do not retry" is the one instruction an acknowledgement gives.
//! Without [`Responder::with_store`] a write lives in the live runtime only,
//! which is right for a one-shot pass and wrong for the loop.
//!
//! # Accepted is not the same as servable, and the ack says so
//!
//! The query half serves planned *nodes*, not documents. A [`ContextPut`] whose
//! text the indexer turns into no typed state is stored, acknowledged, and will
//! never appear in a working set — prose like "restart the ledger with
//! ledgerctl" yields nothing, while "the ledger restart command is ledgerctl"
//! yields a node that will be served. That is the query half's design, not a
//! defect in this one, but on the write side it is a trap: the peer is certain
//! it just handed over a document, so an empty answer later reads as "the store
//! lost it" rather than "the indexer had nothing to extract".
//!
//! So [`WriteAck::nodes`] and [`WriteAck::spans`] are reported at write time.
//! A peer learns from the acknowledgement that nothing was extractable, while
//! it can still act on it, instead of inferring it later from a silence.
//!
//! # The write acknowledgement, and why a write is not fire-and-forget
//!
//! A query has a natural reply and the peer polls for it. A write does not.
//! The peer has to be able to tell **accepted and ingested** from **not
//! ingested**, and "no answer" has to stay distinguishable from "an answer
//! saying there was nothing to find". Fire-and-forget leaves exactly one way
//! for unifur to find out: query for what it just put, and read the empty
//! working set. That state is the one this file works hardest not to
//! manufacture — it reads at the model as "the history does not contain it",
//! which is a false statement about the store dressed as a result.
//!
//! So a write is acknowledged, and the acknowledgement is **success only**.
//! There is no failure envelope, for the same reason there is no failure
//! envelope on the query side: a write that could not be ingested gets no file
//! at all, and the peer times out on the rule it already uses. An
//! `{"ok": false}` reply would add a second, softer failure channel, and the
//! first thing a peer does with the wrong kind of file is treat it as an
//! answer. Absence is the signal; presence is the promise.
//!
//! | file | written by | contents |
//! |---|---|---|
//! | `<id>.request.json` | unifur | a bare [`ChannelRequest`] |
//! | `<id>.response.json` | this responder | [`ChannelResponse`], hits or none |
//! | `<id>.put.json` | unifur | a bare [`ContextPut`] **or** [`MemoryAppend`] |
//! | `<id>.put-ack.json` | this responder | a bare [`WriteAck`] |
//!
//! Each body is the declared message and nothing else — there is no write
//! envelope, because `contract.json` names six messages and not one of them
//! wraps a write. Which of the two a `.put.json` holds is read off the wire
//! rather than out of a wrapper: `payload` is the only key [`MemoryAppend`]
//! has, and inventing a discriminator field is the same move as inventing a
//! `version`.
//!
//! **The file stem is the channel id, not `ContextPut.id`.** The stem is what
//! the peer correlates on and what this side validates as a file name, so it is
//! held to `[A-Za-z0-9-_]` exactly like a request id. `ContextPut.id` is the
//! peer's own document id; it is used as a DCR document id and never as a
//! path, so it is not restricted. That is the one asymmetry between the two
//! halves, and it is the reason the "a request whose id disagrees with its
//! file name is refused" rule has no write-side twin: there is nothing on a
//! write file to disagree with.
//!
//! **The write suffixes are an extension.** `contract.json` at version 1
//! declares `request_suffix` and `response_suffix` and is silent about writes,
//! so a peer on the deployed build simply never drops a `.put.json`. Inert is
//! the safe direction for an extension to fail in; misinterpreted is the one
//! this module is organised to prevent. [`WriteAck`] therefore carries its own
//! [`WRITE_ACK_VERSION`] rather than borrowing [`CHANNEL_ENVELOPE_VERSION`],
//! so the two halves can move independently when the peer adopts this.
//!
//! # The `field` hint
//!
//! [`ContextQuery::field`] is unifur's taxonomy (`pentest`, `coding`, …). DCR
//! does not own that taxonomy — its documents carry a filesystem `source` — so
//! the field is recorded in the poll report and **not** used to filter. A
//! filter here would have to guess at a mapping between two vocabularies, and
//! guessing wrong returns an empty working set, which reads at the model as
//! "the history does not contain it". An honest over-retrieval is the better
//! failure, and the field is reported so the operator can see what was asked.
//!
//! # Idempotence
//!
//! unifur's `RequestChannel::gc` deletes answered request/response pairs but
//! leaves unanswered requests for the responder to find. A response that
//! already exists is therefore never rewritten — the responder is a poller of
//! a directory it does not own, and a request file can reappear if a peer
//! crashed between writing the response and collecting it. The same rule
//! governs the write half through the acknowledgement file, so a `put.json`
//! whose peer never collected its ack is answered once and then left alone.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::hash::sha256;
use crate::json::{self, Json};
use crate::runtime::Dcr;

/// Schema version of [`ContextPut`]. Mirrors unifur's `CONTEXT_PUT_VERSION`.
pub const CONTEXT_PUT_VERSION: u32 = 1;
/// Schema version of [`ContextQuery`]. Mirrors unifur's `CONTEXT_QUERY_VERSION`.
pub const CONTEXT_QUERY_VERSION: u32 = 1;
/// Schema version of [`ContextHit`]. Mirrors unifur's `CONTEXT_HIT_VERSION`.
pub const CONTEXT_HIT_VERSION: u32 = 1;
/// Schema version of [`MemoryAppend`]. Mirrors unifur's `MEMORY_APPEND_VERSION`.
pub const MEMORY_APPEND_VERSION: u32 = 1;
/// Schema version of the [`ChannelRequest`]/[`ChannelResponse`] envelopes.
pub const CHANNEL_ENVELOPE_VERSION: u32 = 1;
/// Schema version of the [`WriteAck`], the write channel's own answer.
///
/// Deliberately *not* [`CHANNEL_ENVELOPE_VERSION`]. The write half is an
/// extension to a contract that does not declare it, so it versions apart
/// from the half that is declared; sharing one constant would let a future
/// bump of the query envelope silently re-label the write acknowledgement as
/// something it is not.
pub const WRITE_ACK_VERSION: u32 = 1;
/// Suffix of a submitted request file inside the channel directory.
pub const REQUEST_SUFFIX: &str = ".request.json";
/// Suffix of the responder's answer file inside the channel directory.
pub const RESPONSE_SUFFIX: &str = ".response.json";
/// Suffix of an inbound write file: a bare [`ContextPut`] or [`MemoryAppend`].
pub const WRITE_SUFFIX: &str = ".put.json";
/// Suffix of this responder's acknowledgement of an accepted write.
///
/// Contains a `-` so it cannot end with [`WRITE_SUFFIX`], and so a responder
/// re-reading its own directory never mistakes the acknowledgement it wrote
/// for a write waiting to be ingested.
pub const WRITE_ACK_SUFFIX: &str = ".put-ack.json";
/// Protocol name both repos implement, as recorded in `contract.json`.
pub const PROTOCOL: &str = "unifur-subnext-interop";

/// The channel directory unifur submits into, and this responder watches.
///
/// unifur's `interop::INTEROP_DIR` is the same literal; the two are separate
/// crates with no shared source of truth, so [`Responder::new`] takes the
/// directory as an argument and this is only the default `dcr serve-context`
/// uses. Unifur creates the directory when it submits, which is why this one
/// is created rather than required.
pub const DEFAULT_CHANNEL_DIR: &str = "/srv/m-sda/unifur/interop";

/// Reject an unknown or missing schema version loudly.
///
/// The rule is byte-identical to unifur's `check_version`: a *missing*
/// `version` key parses as `0` and is rejected the same way, because a
/// silently-reinterpreted envelope is how two repos drift apart while both
/// report success.
fn check_version(kind: &str, got: u32, supported: u32) -> Result<(), String> {
    if got != supported {
        return Err(format!(
            "{kind} message version {got} is not supported by this build \
             (supported: {supported}; a missing version parses as 0 and is \
             rejected the same way)"
        ));
    }
    Ok(())
}

/// A context document handed from unifur to subnext for storage.
///
/// Field for field the shape unifur serialises: `meta` is a [`Json`] here
/// against a `serde_json::Value` there, which is the same "any JSON value"
/// widened into a crate with no dependencies. The five keys and the version
/// are the whole message — there is no field here that `contract.json` does
/// not declare.
///
/// The `id` is the *peer's* document id, not the channel id: the channel id is
/// the file-name stem, and only that stem is ever treated as a path. It lands
/// in the store as `interop:<kind>:<id>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPut {
    pub version: u32,
    pub id: String,
    pub kind: String,
    pub text: String,
    pub meta: Json,
}

impl ContextPut {
    /// Build a message at the current schema version.
    pub fn new(
        id: impl Into<String>,
        kind: impl Into<String>,
        text: impl Into<String>,
        meta: Json,
    ) -> Self {
        Self {
            version: CONTEXT_PUT_VERSION,
            id: id.into(),
            kind: kind.into(),
            text: text.into(),
            meta,
        }
    }

    /// Parse a put, rejecting versions this build does not implement.
    ///
    /// A *missing* `version` is the same error as an unknown one, for the same
    /// reason as everywhere else here: unifur's `#[serde(default)]` turns an
    /// absent `version` into `0`, and a document stored under a shape nobody
    /// agreed on is worse than a refusal.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value = json::parse(text).map_err(|e| format!("invalid ContextPut JSON: {e}"))?;
        let version = value.get("version").and_then(Json::as_u64).unwrap_or(0) as u32;
        check_version("ContextPut", version, CONTEXT_PUT_VERSION)?;
        Ok(Self {
            version,
            id: value
                .get("id")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            kind: value
                .get("kind")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            text: value
                .get("text")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            // Absent and explicitly null are the same thing here: unifur
            // serialises `Value::Null` when the caller left `meta` unused.
            meta: value.get("meta").cloned().unwrap_or(Json::Null),
        })
    }

    /// Serialize in the shape unifur's `ContextPut` deserializes.
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("id", Json::str(self.id.clone())),
            ("kind", Json::str(self.kind.clone())),
            ("text", Json::str(self.text.clone())),
            ("meta", self.meta.clone()),
        ])
    }

    /// The provenance source this put declares, if it declares one.
    ///
    /// `meta` is documented on the peer's side as "provenance hints, tags",
    /// and `source` is the one key in it with an unambiguous meaning here: the
    /// corpus a document came from, which is exactly what
    /// [`Provenance::source`] reports on the way back. A put with no
    /// `meta.source` gets no source, and its hits come back with an empty
    /// one — never a string invented here to fill the field.
    ///
    /// The other keys are not read. That is a limit, not an oversight: what
    /// DCR keeps about a document is `(id, text, ts, source)` — see
    /// `RawStore::add_document` — so an unrecognised tag has no honest
    /// address, and prefixing it onto the text would corrupt what gets
    /// retrieved for it.
    pub fn source_hint(&self) -> Option<String> {
        self.meta
            .get("source")
            .and_then(Json::as_str)
            .map(str::to_string)
    }
}

/// A write into subnext's Merkle-chained long-term memory: past chats and runs
/// become retrievable context, never weight edits.
///
/// One field, `payload`, exactly as `contract.json` declares it.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryAppend {
    pub version: u32,
    pub payload: Json,
}

impl MemoryAppend {
    /// Build a message at the current schema version.
    pub fn new(payload: Json) -> Self {
        Self {
            version: MEMORY_APPEND_VERSION,
            payload,
        }
    }

    /// Parse an append, rejecting versions this build does not implement —
    /// and refusing a missing version exactly as an unknown one.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value = json::parse(text).map_err(|e| format!("invalid MemoryAppend JSON: {e}"))?;
        let version = value.get("version").and_then(Json::as_u64).unwrap_or(0) as u32;
        check_version("MemoryAppend", version, MEMORY_APPEND_VERSION)?;
        Ok(Self {
            version,
            payload: value.get("payload").cloned().unwrap_or(Json::Null),
        })
    }

    /// Serialize in the shape unifur's `MemoryAppend` deserializes.
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("payload", self.payload.clone()),
        ])
    }

    /// The document id this append is stored under.
    ///
    /// Content-addressed, and that is the point rather than a detail. L0 is
    /// immutable: a second document under a used id carrying different bytes
    /// is a hard error, so an append that derived its id from a counter would
    /// make every retry after a lost acknowledgement fail. Deriving it from
    /// the payload means the same payload twice is a no-op under one id, and
    /// two peers appending the same payload converge on one entry instead of
    /// colliding on one.
    pub fn document_id(&self) -> String {
        let digest = sha256(self.indexed_text().as_bytes());
        format!("interop:memory:{}", digest.short(16))
    }

    /// The text this append is indexed as.
    ///
    /// The payload's *canonical* JSON: keys sorted, so a peer that re-serialises
    /// the same value differently still lands on the same document id, and the
    /// bytes are the payload itself with nothing added. Canonical rather than
    /// pretty, because the indexer matches tokens and whitespace is not one.
    pub fn indexed_text(&self) -> String {
        self.payload.to_canonical_string()
    }
}

/// The acknowledgement of an accepted write, written as `<id>.put-ack.json`.
///
/// Every field is an identity or a measurement. There is no `ok` flag and no
/// error string, because the file's *existence* is the promise: it is written
/// only once the write is in the store, and a write that failed gets no file
/// at all, so the peer times out on the same rule the query half already uses.
/// An acknowledgement that could also say "no" would be a second failure
/// channel, and a peer would have to learn which of the two to believe.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteAck {
    pub version: u32,
    /// The channel id — the file stem of both write files.
    pub id: String,
    /// The DCR document id the write landed under. This is the string a later
    /// hit reports as `provenance.path`, so it is how a peer finds what it put.
    pub document: String,
    /// Spans the indexer produced. Zero means the write was accepted and held
    /// nothing indexable — an empty `text`, or a payload that is only
    /// whitespace. That is a fact about the document, not a failure.
    pub spans: usize,
    /// Typed state nodes extracted from those spans.
    pub nodes: usize,
}

impl WriteAck {
    /// Build an acknowledgement at the current write-envelope version.
    pub fn new(
        id: impl Into<String>,
        document: impl Into<String>,
        spans: usize,
        nodes: usize,
    ) -> Self {
        Self {
            version: WRITE_ACK_VERSION,
            id: id.into(),
            document: document.into(),
            spans,
            nodes,
        }
    }

    /// Serialize for `<id>.put-ack.json`.
    pub fn to_json(&self) -> String {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("id", Json::str(self.id.clone())),
            ("document", Json::str(self.document.clone())),
            ("spans", Json::num(self.spans as f64)),
            ("nodes", Json::num(self.nodes as f64)),
        ])
        .to_json_string()
    }
}

/// The DCR document id a [`ContextPut`] is stored under.
///
/// `kind` is inside the id because unifur documents it as a discriminator
/// (`"document"`, `"memory"`, …), and a peer that puts a document and a memory
/// under one id has meant two different things — which under one id would be a
/// silent supersession, the exact failure L0's immutability exists to make
/// loud. The `interop:` prefix keeps these documents out of the namespace of
/// whatever corpus the responder was started with, so a peer's `doc-9` cannot
/// collide with a file the operator ingested as `doc-9`.
///
/// An empty `kind` is stored as an empty segment rather than defaulted to
/// `document`. Guessing would let two puts that said different things land on
/// one id, and a collision that is loud beats one that is not.
fn document_id_for(put: &ContextPut) -> String {
    format!("interop:{}:{}", put.kind, put.id)
}

/// A retrieval query: unifur asks for the `k` most relevant documents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContextQuery {
    pub version: u32,
    pub text: String,
    pub k: usize,
    pub field: String,
}

impl ContextQuery {
    /// Parse a query, rejecting versions this build does not implement.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value = json::parse(text).map_err(|e| format!("invalid ContextQuery JSON: {e}"))?;
        let version = value.get("version").and_then(Json::as_u64).unwrap_or(0) as u32;
        check_version("ContextQuery", version, CONTEXT_QUERY_VERSION)?;
        Ok(Self {
            version,
            text: value
                .get("text")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            k: value.get("k").and_then(Json::as_usize).unwrap_or_default(),
            field: value
                .get("field")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
        })
    }

    /// Serialize in the shape unifur's `ContextQuery` deserializes.
    pub fn to_json(&self) -> String {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("text", Json::str(self.text.clone())),
            ("k", Json::num(self.k as f64)),
            ("field", Json::str(self.field.clone())),
        ])
        .to_json_string()
    }
}

/// Where a retrieved hit came from: which source, and which document in it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Provenance {
    pub source: String,
    pub path: String,
}

/// One scored, retrieved context document.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextHit {
    pub version: u32,
    pub id: String,
    pub score: f64,
    pub text: String,
    pub provenance: Provenance,
}

impl ContextHit {
    /// Serialize in the shape unifur's `ContextHit` deserializes.
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("id", Json::str(self.id.clone())),
            ("score", Json::num(self.score)),
            ("text", Json::str(self.text.clone())),
            (
                "provenance",
                Json::obj(vec![
                    ("source", Json::str(self.provenance.source.clone())),
                    ("path", Json::str(self.provenance.path.clone())),
                ]),
            ),
        ])
    }
}

/// The request envelope written as `<id>.request.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelRequest {
    pub version: u32,
    pub id: String,
    pub query: ContextQuery,
}

impl ChannelRequest {
    /// Parse an envelope, rejecting unknown envelope versions.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value =
            json::parse(text).map_err(|e| format!("invalid ChannelRequest JSON: {e}"))?;
        let version = value.get("version").and_then(Json::as_u64).unwrap_or(0) as u32;
        check_version("ChannelRequest", version, CHANNEL_ENVELOPE_VERSION)?;
        let query_text = value
            .get("query")
            .map(|q| q.to_json_string())
            .ok_or_else(|| "ChannelRequest has no 'query' field".to_string())?;
        Ok(Self {
            version,
            id: value
                .get("id")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            query: ContextQuery::from_json(&query_text)?,
        })
    }

    /// The file-name stem of this request's two envelope files.
    pub fn request_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}{REQUEST_SUFFIX}", self.id))
    }

    /// Where this request's answer goes.
    pub fn response_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}{RESPONSE_SUFFIX}", self.id))
    }
}

/// The response envelope written as `<id>.response.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelResponse {
    pub version: u32,
    pub id: String,
    pub hits: Vec<ContextHit>,
}

impl ChannelResponse {
    /// Build a response envelope at the current version.
    pub fn new(id: impl Into<String>, hits: Vec<ContextHit>) -> Self {
        Self {
            version: CHANNEL_ENVELOPE_VERSION,
            id: id.into(),
            hits,
        }
    }

    /// Serialize in the shape unifur's `ChannelResponse` deserializes.
    pub fn to_json(&self) -> String {
        Json::obj(vec![
            ("version", Json::num(self.version as f64)),
            ("id", Json::str(self.id.clone())),
            (
                "hits",
                Json::Arr(self.hits.iter().map(ContextHit::to_json).collect()),
            ),
        ])
        .to_json_string()
    }
}

/// Write via a temporary file and rename, so a polling peer never reads a
/// partial envelope. Matches the rule both sides document in `contract.json`.
fn write_atomic(path: &Path, contents: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, contents)
        .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })
}

/// Channel ids become file names, so the character set is restricted exactly
/// as unifur's `validate_request_id` restricts it. An answer written to a path
/// built from an unvalidated id would let a peer choose the file name — for
/// either half of the channel, which is why both call this one function.
fn validate_request_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!(
            "invalid channel id '{id}': only [A-Za-z0-9-_] is allowed \
             (the id becomes a file name in the channel directory)"
        ));
    }
    Ok(())
}

/// The channel id a channel file names: its file-name stem, or `None` when the
/// name does not carry `suffix`.
///
/// `None` rather than an empty string, because an empty id passes
/// [`validate_request_id`]'s companion checks nowhere and would silently
/// become `""` at the far end of a `format!`. A name that is not a channel
/// file is skipped instead.
fn channel_id_of(path: &Path, suffix: &str) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    Some(name.strip_suffix(suffix)?.to_string())
}

/// What one pass over the channel directory did.
#[derive(Debug, Clone, Default)]
pub struct PollReport {
    /// Requests answered this pass.
    pub answered: usize,
    /// Requests already carrying a response; left untouched.
    pub skipped_answered: usize,
    /// Fields seen in the queries answered, with counts. The `field` hint is
    /// reported rather than filtered on — see the module docs.
    pub fields: BTreeSet<String>,
    /// Total hits served across the pass.
    pub hits: usize,
    /// Writes ingested **and acknowledged** this pass. A write whose ingest
    /// failed, or whose acknowledgement could not be written, is in
    /// `failures` and not counted here: the peer was not told, so nothing was
    /// delivered, and a number that said otherwise would be the one place this
    /// report could lie.
    pub written: usize,
    /// Writes already carrying an acknowledgement; left untouched.
    pub skipped_acked: usize,
    /// [`ContextPut`]s among `written`.
    pub documents: usize,
    /// [`MemoryAppend`]s among `written`.
    pub memory_appends: usize,
    /// Spans indexed across `written`.
    pub spans: usize,
    /// Per-request and per-write failures. A failure is never converted into a
    /// response or an acknowledgement: a peer with no answer times out, which
    /// is the honest outcome, whereas a fabricated empty hit list reads as "the
    /// history does not contain it".
    pub failures: Vec<String>,
}

impl PollReport {
    /// One line per answered request, one per ingested write, then the failures.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if self.fields.is_empty() {
            out.push_str("no fields requested\n");
        } else {
            out.push_str(&format!(
                "fields asked for: {}\n",
                self.fields
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        out.push_str(&format!(
            "answered {} request(s), skipped {} already answered, {} hit(s) served\n",
            self.answered, self.skipped_answered, self.hits
        ));
        out.push_str(&format!(
            "ingested {} write(s) — {} document(s), {} memory append(s), {} span(s) — \
             skipped {} already acknowledged\n",
            self.written,
            self.documents,
            self.memory_appends,
            self.spans,
            self.skipped_acked
        ));
        for failure in &self.failures {
            out.push_str(&format!("failure: {failure}\n"));
        }
        out
    }
}

/// A write file read and ingested, not yet acknowledged.
///
/// Split out from the [`WriteAck`] so the ingest and the file write stay
/// separately reportable: `written` counts what the peer can see, and a store
/// that took a document the peer never learned about is a fact about the
/// store, not a delivered write.
struct Ingested {
    document: String,
    spans: usize,
    nodes: usize,
    /// True for a [`MemoryAppend`], false for a [`ContextPut`].
    memory: bool,
}

/// Watches a channel directory and answers the requests it finds.
pub struct Responder {
    dir: PathBuf,
    runtime: Dcr,
    /// Cap on hits per answer. `min(query.k, this)`: the query's own `k` is
    /// the peer's stated appetite, but a peer asking for 10_000 hits would
    /// otherwise turn a working set into a corpus dump — which is precisely
    /// the thing the budget exists to prevent.
    max_hits: usize,
    /// Where ingested writes are sealed after the pass that ingested them.
    ///
    /// Set by [`Responder::with_store`]. Without one a write lives in the live
    /// runtime only, which is fine for `serve-context --once` — the caller
    /// persists on the way out — and wrong for the loop, where a responder that
    /// restarts takes the peer's acknowledged documents with it.
    store: Option<PathBuf>,
}

impl Responder {
    /// Open a channel at `dir`, creating it on demand. The directory is
    /// never assumed to exist: unifur creates it when it submits, but a
    /// responder that starts first must not fail on its absence.
    pub fn new(dir: &Path, runtime: Dcr, max_hits: usize) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("creating channel dir '{}': {e}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            runtime,
            max_hits: max_hits.max(1),
            store: None,
        })
    }

    /// Seal ingested writes into `store` after the pass that ingested them.
    ///
    /// A `.context` container directory when the path already holds a
    /// container manifest, a single runtime file otherwise — the same two
    /// stores `Dcr::save_context` and `Dcr::save` accept. The manifest is asked
    /// for rather than the path shape because the responder is handed a store
    /// that already exists: `dcr serve-context` opens its `--store` before it
    /// builds one, so "is there a container here" is the question that has an
    /// answer, and a store path that is neither fails loudly at the seal
    /// instead of quietly becoming a runtime file.
    pub fn with_store(mut self, store: &Path) -> Self {
        self.store = Some(store.to_path_buf());
        self
    }

    /// Seal the current runtime into the configured store, if there is one.
    ///
    /// Called once per pass, between the ingests and the acknowledgements, so
    /// a document is on disk before a peer is told it is. Never called after a
    /// pass that ingested nothing: every generation is a new Merkle root, and a
    /// store that grew one per idle poll would have a chain recording the
    /// responder's wake-ups rather than what it learned.
    fn seal(&self) -> Result<(), String> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        if store.join(crate::context_store::MANIFEST).exists() {
            let checkpoint = self
                .runtime
                .save_context(store, None)
                .map_err(|e| format!("sealing {}: {e}", store.display()))?;
            eprintln!(
                "dcr: sealed generation {} · root {} · {} object(s) into {}",
                checkpoint.generation,
                checkpoint.merkle_root.short(12),
                checkpoint.object_count,
                store.display()
            );
        } else {
            self.runtime
                .save(store)
                .map_err(|e| format!("saving {}: {e}", store.display()))?;
        }
        Ok(())
    }

    /// The directory this responder reads.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The runtime behind the answers, for persistence.
    pub fn runtime_mut(&mut self) -> &mut Dcr {
        &mut self.runtime
    }

    /// Every channel file ending in `suffix`, sorted by name so two
    /// responders racing the same directory take files in the same order.
    fn pending_files(&self, suffix: &str) -> Result<Vec<PathBuf>, String> {
        let entries = std::fs::read_dir(&self.dir)
            .map_err(|e| format!("read channel dir '{}': {e}", self.dir.display()))?;
        let mut found: Vec<PathBuf> = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|e| format!("read channel dir '{}': {e}", self.dir.display()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(suffix) {
                found.push(entry.path());
            }
        }
        found.sort();
        Ok(found)
    }

    /// Ingest one peer write file and report what it became.
    ///
    /// Which of the two messages the body holds is read off the wire: `payload`
    /// is the only key [`MemoryAppend`] has. `contract.json` declares no write
    /// envelope and no discriminator, so a wrapper or an added `type` field
    /// would be a second thing for the two repos to agree on that the contract
    /// does not already record.
    fn ingest_write(&mut self, body: &str) -> Result<Ingested, String> {
        let value = json::parse(body).map_err(|e| format!("invalid write JSON: {e}"))?;
        Ok(if value.get("payload").is_some() {
            let append = MemoryAppend::from_json(body)?;
            let document = append.document_id();
            let result = self
                .runtime
                .ingest(&append.indexed_text(), Some(&document))
                .map_err(|e| format!("appending a memory payload: {e}"))?;
            Ingested {
                document,
                spans: result.spans.len(),
                nodes: result.nodes.len(),
                memory: true,
            }
        } else {
            let put = ContextPut::from_json(body)?;
            let document = document_id_for(&put);
            let result = self
                .runtime
                .ingest_with_source(&put.text, Some(&document), put.source_hint())
                .map_err(|e| format!("putting context document '{}': {e}", put.id))?;
            Ingested {
                document,
                spans: result.spans.len(),
                nodes: result.nodes.len(),
                memory: false,
            }
        })
    }

    /// Ingest every write in the channel that does not already have an
    /// acknowledgement, seal, then acknowledge each one.
    ///
    /// Three steps in that order, and each one exists because of the step
    /// after it. Ingest precedes the acknowledgement, so an acknowledgement
    /// never describes a document the store refused. The seal precedes the
    /// acknowledgements, so a crash between the two leaves a peer that will
    /// retry — and a retry is free, because L0 treats identical text under a
    /// used id as a no-op. The other order would leave a peer holding an
    /// acknowledgement for a document that only ever existed in memory, and
    /// "do not retry" is the one instruction an acknowledgement gives.
    ///
    /// A write that fails any step gets no file at all: the peer times out,
    /// which is the same honest outcome a failed query gets.
    fn apply_writes(&mut self, report: &mut PollReport) {
        let paths = match self.pending_files(WRITE_SUFFIX) {
            Ok(paths) => paths,
            Err(err) => {
                report.failures.push(err);
                return;
            }
        };
        let mut ingested: Vec<(String, PathBuf, Ingested)> = Vec::new();
        for path in paths {
            let Some(id) = channel_id_of(&path, WRITE_SUFFIX) else {
                continue;
            };
            if let Err(err) = validate_request_id(&id) {
                report.failures.push(format!("{}: {err}", path.display()));
                continue;
            }
            let ack_path = self.dir.join(format!("{id}{WRITE_ACK_SUFFIX}"));
            if ack_path.exists() {
                report.skipped_acked += 1;
                continue;
            }
            let body = match std::fs::read_to_string(&path) {
                Ok(body) => body,
                Err(err) => {
                    report
                        .failures
                        .push(format!("read {}: {err}", path.display()));
                    continue;
                }
            };
            match self.ingest_write(&body) {
                Ok(result) => ingested.push((id, ack_path, result)),
                Err(err) => report.failures.push(format!("{}: {err}", path.display())),
            }
        }
        if ingested.is_empty() {
            return;
        }
        if let Err(err) = self.seal() {
            // Nothing is acknowledged: the peer retries, and the retry is a
            // no-op against content already in the runtime.
            report.failures.push(err);
            return;
        }
        for (id, ack_path, result) in ingested {
            let ack = WriteAck::new(&id, &result.document, result.spans, result.nodes);
            match write_atomic(&ack_path, &ack.to_json()) {
                Ok(()) => {
                    report.written += 1;
                    report.spans += result.spans;
                    if result.memory {
                        report.memory_appends += 1;
                    } else {
                        report.documents += 1;
                    }
                }
                // On disk, but the peer was not told. Recoverable by the peer's
                // next poll, and still a failure rather than a delivered write.
                Err(err) => report.failures.push(err),
            }
        }
    }

    /// Plan a working set and map its admissions onto [`ContextHit`]s.
    ///
    /// The `score` is the planner's `utility` — the quantity the knapsack
    /// actually maximised — not a post-hoc similarity. A consumer sorting by
    /// `score` therefore gets the planner's own ranking.
    fn answer(&mut self, request: &ChannelRequest) -> Result<ChannelResponse, String> {
        let query = &request.query;
        let k = if query.k == 0 {
            self.max_hits
        } else {
            query.k.min(self.max_hits)
        };
        let context = self.runtime.plan(&query.text, None);
        let mut hits: Vec<ContextHit> = context
            .entries
            .iter()
            .take(k)
            .map(|entry| {
                let node = self.runtime.graph.node(entry.node);
                let (source, path) = self.provenance_of(&node.source_spans);
                ContextHit {
                    version: CONTEXT_HIT_VERSION,
                    id: entry.node_id.clone(),
                    score: entry.utility as f64,
                    text: entry.rendered.clone(),
                    provenance: Provenance { source, path },
                }
            })
            .collect();
        // Best first. The knapsack emits admissions in *budget* order, not
        // utility order, and the consumer takes the head of the list until its
        // own budget runs out — so an unsorted answer silently gives the model
        // the worst `n` of the set it could have had. Ties keep admission
        // order, which is stable, so the same query gives the same list.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ChannelResponse::new(request.id.clone(), hits))
    }

    /// Resolve an admitted node's spans back to the document they came from.
    ///
    /// A node with no surviving span is reported with empty provenance rather
    /// than a fabricated one: unifur renders `source=` into the prompt, and an
    /// invented source string would be an unsourced fact wearing a citation.
    fn provenance_of(&self, span_ids: &[String]) -> (String, String) {
        for span_id in span_ids {
            if let Some(span) = self.runtime.raw.span_by_id(span_id) {
                let doc_id = span.doc_id.clone();
                let source = self
                    .runtime
                    .raw
                    .documents()
                    .iter()
                    .find(|d| d.id == doc_id)
                    .and_then(|d| d.source.clone())
                    .unwrap_or_default();
                return (source, doc_id);
            }
        }
        (String::new(), String::new())
    }

    /// Answer every request in the channel that does not already have one.
    ///
    /// One pass. A failure on one request is recorded and the pass continues:
    /// a single malformed envelope must not strand every other peer's query.
    fn answer_requests(&mut self, report: &mut PollReport) {
        let paths = match self.pending_files(REQUEST_SUFFIX) {
            Ok(paths) => paths,
            Err(err) => {
                report.failures.push(err);
                return;
            }
        };
        for path in paths {
            let Some(id) = channel_id_of(&path, REQUEST_SUFFIX) else {
                continue;
            };
            if let Err(err) = validate_request_id(&id) {
                report.failures.push(format!("{}: {err}", path.display()));
                continue;
            }
            let response_path = self.dir.join(format!("{id}{RESPONSE_SUFFIX}"));
            if response_path.exists() {
                report.skipped_answered += 1;
                continue;
            }
            let body = match std::fs::read_to_string(&path) {
                Ok(body) => body,
                Err(err) => {
                    report
                        .failures
                        .push(format!("read {}: {err}", path.display()));
                    continue;
                }
            };
            match ChannelRequest::from_json(&body) {
                Ok(request) => {
                    if request.id != id {
                        report.failures.push(format!(
                            "{}: envelope answers request '{}' but its file name says '{id}'",
                            path.display(),
                            request.id
                        ));
                        continue;
                    }
                    report.fields.insert(request.query.field.clone());
                    match self.answer(&request) {
                        Ok(response) => {
                            let hits = response.hits.len();
                            match write_atomic(&response_path, &response.to_json()) {
                                Ok(()) => {
                                    report.answered += 1;
                                    report.hits += hits;
                                }
                                Err(err) => report.failures.push(err),
                            }
                        }
                        Err(err) => report
                            .failures
                            .push(format!("plan {}: {err}", request.query.text)),
                    }
                }
                Err(err) => report.failures.push(format!("{}: {err}", path.display())),
            }
        }
    }

    /// One pass over the channel: ingest the writes, then answer the queries.
    ///
    /// **Writes first, deliberately.** A peer that drops a put and a query into
    /// the same directory inside one turn means the query to be answered against
    /// the put. Answering it first would serve a store one write behind and
    /// report the result as authoritative — the same "the history does not
    /// contain it" lie, reached by a different route.
    ///
    /// Within each half, sorted file names and per-file failure isolation: one
    /// peer's malformed write must not strand another peer's query, and two
    /// responders racing this directory must take the same files in the same
    /// order.
    pub fn poll_once(&mut self) -> Result<PollReport, String> {
        let mut report = PollReport::default();
        self.apply_writes(&mut report);
        self.answer_requests(&mut report);
        Ok(report)
    }

    /// Poll until interrupted, sleeping `idle` between empty passes.
    pub fn serve(&mut self, idle: Duration) -> Result<(), String> {
        loop {
            let report = self.poll_once()?;
            if report.answered == 0 && report.written == 0 && report.failures.is_empty() {
                std::thread::sleep(idle);
            } else if !report.failures.is_empty() {
                eprint!("{}", report.render());
            }
        }
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it.
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::wildcard_imports,
    clippy::exit
)]
mod tests {
    use super::*;
    use crate::context_store::ContextStore;

    /// A channel directory unique to this test: parallel tests share `/tmp`.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "dcr-interop-{}-{}-{tag}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn responder_with(dir: &Path, docs: &[(&str, &str)]) -> Responder {
        let mut runtime = Dcr::new(400);
        for (text, doc_id) in docs {
            runtime.ingest(text, Some(doc_id)).unwrap();
        }
        Responder::new(dir, runtime, 8).unwrap()
    }

    fn request(id: &str, text: &str, k: usize, field: &str) -> String {
        Json::obj(vec![
            ("version", Json::num(CHANNEL_ENVELOPE_VERSION as f64)),
            ("id", Json::str(id)),
            (
                "query",
                Json::obj(vec![
                    ("version", Json::num(CONTEXT_QUERY_VERSION as f64)),
                    ("text", Json::str(text)),
                    ("k", Json::num(k as f64)),
                    ("field", Json::str(field)),
                ]),
            ),
        ])
        .to_json_string()
    }

    /// A `.put.json` body carrying a `ContextPut`, as unifur would write it.
    fn put(id: &str, kind: &str, text: &str, meta: Json) -> String {
        ContextPut::new(id, kind, text, meta).to_json().to_json_string()
    }

    /// A `.put.json` body carrying a `MemoryAppend`.
    fn append(payload: Json) -> String {
        MemoryAppend::new(payload).to_json().to_json_string()
    }

    /// Drop a body into the channel the way a peer would.
    fn submit(dir: &Path, channel_id: &str, body: &str) -> PathBuf {
        let path = dir.join(format!("{channel_id}{WRITE_SUFFIX}"));
        std::fs::write(&path, body).unwrap();
        path
    }

    /// The acknowledgement for a channel id, if the responder wrote one.
    fn ack_of(dir: &Path, channel_id: &str) -> Option<Json> {
        let path = dir.join(format!("{channel_id}{WRITE_ACK_SUFFIX}"));
        let body = std::fs::read_to_string(path).ok()?;
        Some(json::parse(&body).unwrap())
    }

    /// The hits a channel id was answered with, if it was answered.
    fn hits_of(dir: &Path, channel_id: &str) -> Vec<Json> {
        let path = dir.join(format!("{channel_id}{RESPONSE_SUFFIX}"));
        let body = std::fs::read_to_string(path).unwrap();
        let value = json::parse(&body).unwrap();
        value
            .get("hits")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .to_vec()
    }

    #[test]
    fn query_round_trips_through_the_wire_shape() {
        let query = ContextQuery {
            version: CONTEXT_QUERY_VERSION,
            text: "which port does the relay listen on?".to_string(),
            k: 8,
            field: "pentest".to_string(),
        };
        assert_eq!(ContextQuery::from_json(&query.to_json()).unwrap(), query);
    }

    #[test]
    fn an_unknown_version_is_refused_rather_than_reinterpreted() {
        // The drift failure mode this guards: unifur bumps the envelope,
        // subnext keeps parsing, and both report success while exchanging
        // different shapes.
        let bad = Json::obj(vec![
            ("version", Json::num(2.0)),
            ("id", Json::str("a")),
            ("query", Json::obj(vec![])),
        ])
        .to_json_string();
        assert!(ChannelRequest::from_json(&bad).unwrap_err().contains("version 2"));
    }

    #[test]
    fn a_missing_version_is_refused_the_same_way() {
        // serde's `#[serde(default)]` on unifur's side turns an absent
        // `version` into 0. If only the explicit wrong-version case were
        // checked, this would parse and silently mean something else.
        let bad = Json::obj(vec![("id", Json::str("a")), ("query", Json::obj(vec![]))])
            .to_json_string();
        assert!(ChannelRequest::from_json(&bad).unwrap_err().contains("version 0"));
    }

    #[test]
    fn a_request_is_answered_with_planned_hits_and_provenance() {
        let dir = temp_dir("answer");
        let mut responder = responder_with(
            &dir,
            &[
                ("The server ip is 10.0.4.12.", "fox.txt"),
                ("The hourly rate is 180 USD.", "garden.txt"),
            ],
        );
        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "what is the server ip?", 4, "pentest"),
        )
        .unwrap();

        let report = responder.poll_once().unwrap();
        assert_eq!(report.answered, 1);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert!(report.hits > 0, "a query about a stored fact must return it");

        let body = std::fs::read_to_string(dir.join("r1.response.json")).unwrap();
        let value = json::parse(&body).unwrap();
        assert_eq!(value.get("id").and_then(Json::as_str), Some("r1"));
        let hits = value.get("hits").and_then(Json::as_array).unwrap();
        assert!(!hits.is_empty());
        let first = &hits[0];
        assert_eq!(first.get("version").and_then(Json::as_u64), Some(1));
        assert!(
            first.get("text").and_then(Json::as_str).is_some_and(|t| !t.is_empty()),
            "a hit with no text is not context"
        );
        let prov = first.get("provenance").expect("every hit carries provenance");
        assert!(
            prov.get("path").and_then(Json::as_str).is_some(),
            "provenance names the document the node came from"
        );
        assert!(
            prov.get("source").and_then(Json::as_str).is_some(),
            "provenance names the source that document came from"
        );
        assert_eq!(report.fields.iter().collect::<Vec<_>>(), ["pentest"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_answered_request_is_not_answered_twice() {
        // unifur's `gc` deletes the pair, and a peer that crashed between
        // writing a response and collecting it can leave both files. The
        // responder must be idempotent: rewriting a response would race a
        // peer that is mid-read.
        let dir = temp_dir("idempotent");
        let mut responder = responder_with(&dir, &[("The server ip is 10.0.4.12.", "fox.txt")]);
        std::fs::write(dir.join("r1.request.json"), request("r1", "server ip", 2, "tech")).unwrap();
        assert_eq!(responder.poll_once().unwrap().answered, 1);

        let again = responder.poll_once().unwrap();
        assert_eq!(again.answered, 0);
        assert_eq!(again.skipped_answered, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_malformed_request_does_not_strand_the_others() {
        let dir = temp_dir("malformed");
        let mut responder = responder_with(&dir, &[("The server ip is 10.0.4.12.", "fox.txt")]);
        std::fs::write(dir.join("bad.request.json"), "{not json").unwrap();
        std::fs::write(dir.join("good.request.json"), request("good", "server ip", 2, "tech")).unwrap();

        let report = responder.poll_once().unwrap();
        assert_eq!(report.answered, 1, "the good request is still answered");
        assert_eq!(report.failures.len(), 1, "the bad one is reported, not hidden");
        assert!(
            !dir.join("bad.response.json").exists(),
            "a request that could not be parsed gets no fabricated empty answer"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_envelope_answering_a_different_id_is_refused() {
        let dir = temp_dir("mismatch");
        let mut responder = responder_with(&dir, &[("The server ip is 10.0.4.12.", "fox.txt")]);
        // File name says `r1`; the envelope says `other`. Answering would put
        // a response where a peer polling `other` is not looking, and leave
        // `r1` timing out.
        std::fs::write(dir.join("r1.request.json"), request("other", "server ip", 2, "tech")).unwrap();

        let report = responder.poll_once().unwrap();
        assert_eq!(report.answered, 0);
        assert_eq!(report.failures.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_traversal_id_is_refused_before_it_becomes_a_path() {
        // The id is a file name, so a peer that answered `../x` would write
        // outside the channel. `read_dir` cannot yield such a name, which is
        // why this tests the validator directly rather than through a file:
        // it is the guard for a future transport that accepts an id as data.
        for hostile in ["../evil", "a/b", "a\\b", "", ".", "..", "a b", "a;b"] {
            assert!(
                validate_request_id(hostile).is_err(),
                "{hostile:?} must not become a file name"
            );
        }
        for fine in ["ctx-3336511-1791718557487399007-0", "r1", "A_b-9"] {
            assert!(validate_request_id(fine).is_ok(), "{fine:?} is a legal id");
        }
    }

    #[test]
    fn k_is_capped_so_a_hungry_peer_cannot_turn_the_answer_into_a_dump() {
        let dir = temp_dir("cap");
        // Distinct subjects, deliberately: forty documents all claiming
        // `server.ip` are forty *contradictions*, and the planner correctly
        // supersedes all but the last, leaving nothing to serve. A corpus of
        // near-duplicate keys is a correctness case, not a size case.
        let docs: Vec<(String, String)> = (0..40)
            .map(|i| (format!("The host{i} ip is 10.0.{i}.12."), format!("d{i}.txt")))
            .collect();
        let borrowed: Vec<(&str, &str)> = docs
            .iter()
            .map(|(t, d)| (t.as_str(), d.as_str()))
            .collect();
        let mut responder = responder_with(&dir, &borrowed);
        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "ip", 10_000, "tech"),
        )
        .unwrap();

        responder.poll_once().unwrap();
        let body = std::fs::read_to_string(dir.join("r1.response.json")).unwrap();
        let hits = json::parse(&body).unwrap();
        let hits = hits.get("hits").and_then(Json::as_array).unwrap();
        assert_eq!(hits.len(), 8, "the responder's own max_hits is the ceiling");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_store_with_nothing_in_it_answers_with_no_hits() {
        // The honest empty: an empty store has nothing to plan from, and the
        // answer says so rather than inventing context. (Note this is *not*
        // the same as an unrelated query against a full store — the planner
        // deliberately over-retrieves there, so a narrow query still gets
        // back its working set. Over-retrieval is recoverable by the model's
        // budget; a fabricated citation is not.)
        let dir = temp_dir("empty");
        let mut responder = responder_with(&dir, &[]);
        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "what is the server ip?", 4, "tech"),
        )
        .unwrap();

        let report = responder.poll_once().unwrap();
        assert_eq!(report.answered, 1);
        assert_eq!(report.hits, 0);
        let body = std::fs::read_to_string(dir.join("r1.response.json")).unwrap();
        let hits = json::parse(&body).unwrap();
        let hits = hits.get("hits").and_then(Json::as_array).unwrap();
        assert!(
            hits.is_empty(),
            "an empty plan is reported as empty, not padded with the whole corpus"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hits_come_back_best_first_so_a_budgeted_consumer_keeps_the_best_ones() {
        // The peer takes the head of the list until its budget runs out, so an
        // unsorted answer hands it the worst n of what it could have had. The
        // knapsack itself emits in admission order, which is not utility order.
        let dir = temp_dir("order");
        let mut responder = responder_with(&dir, &[]);
        for i in 0..12 {
            responder
                .runtime_mut()
                .ingest(
                    &format!("The host{i} ip is 10.0.{i}.12."),
                    Some(&format!("d{i}.txt")),
                )
                .unwrap();
        }
        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "ip", 10, "tech"),
        )
        .unwrap();

        responder.poll_once().unwrap();
        let body = std::fs::read_to_string(dir.join("r1.response.json")).unwrap();
        let hits = json::parse(&body).unwrap();
        let hits = hits.get("hits").and_then(Json::as_array).unwrap();
        assert!(hits.len() > 2, "need a real set to order");
        let scores: Vec<f64> = hits
            .iter()
            .map(|h| h.get("score").and_then(Json::as_f64).unwrap_or(f64::NAN))
            .collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "scores must be non-increasing: {scores:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_query_gives_the_same_answer_twice() {
        // A responder that reorders by a non-total comparison, or that plans
        // against a mutated graph, would make the chat transcript
        // irreproducible: the same question asked twice would be grounded in
        // different context.
        let dir = temp_dir("stable");
        let mut responder = responder_with(&dir, &[]);
        for i in 0..8 {
            responder
                .runtime_mut()
                .ingest(
                    &format!("The host{i} ip is 10.0.{i}.12."),
                    Some(&format!("d{i}.txt")),
                )
                .unwrap();
        }
        let ask = |responder: &mut Responder, n: usize| {
            std::fs::write(
                dir.join(format!("r{n}.request.json")),
                request(&format!("r{n}"), "ip", 5, "tech"),
            )
            .unwrap();
            responder.poll_once().unwrap();
            let body = std::fs::read_to_string(dir.join(format!("r{n}.response.json"))).unwrap();
            let hits = json::parse(&body).unwrap();
            let hits = hits.get("hits").and_then(Json::as_array).unwrap().to_vec();
            // The id differs by construction; everything a reader relies on
            // must not.
            hits.iter()
                .map(|h| {
                    (
                        h.get("id").and_then(Json::as_str).unwrap_or_default().to_string(),
                        h.get("score").and_then(Json::as_f64).unwrap_or_default(),
                        h.get("text").and_then(Json::as_str).unwrap_or_default().to_string(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = ask(&mut responder, 1);
        let second = ask(&mut responder, 2);
        assert_eq!(first, second, "the same query must give the same context");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_put_becomes_retrievable_context_for_a_later_query() {
        // The whole point of the write half, end to end through the channel:
        // unifur hands over a document, and the next chat turn's working set
        // contains it. Without this the "context lives outside the weights"
        // claim is only ever read half of "outside".
        let dir = temp_dir("put-retrievable");
        let mut responder = responder_with(&dir, &[]);
        submit(
            &dir,
            "w1",
            &put(
                "doc-9",
                "document",
                "The Heliodrome mirror temperature is 77 kelvin.",
                Json::Null,
            ),
        );

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1, "the write was acknowledged");
        assert_eq!(report.documents, 1);
        assert_eq!(report.memory_appends, 0);
        assert!(report.failures.is_empty(), "{:?}", report.failures);

        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "what temperature is the Heliodrome mirror kept?", 6, "tech"),
        )
        .unwrap();
        let report = responder.poll_once().unwrap();
        assert_eq!(report.answered, 1);

        let hits = hits_of(&dir, "r1");
        assert!(
            hits.iter().any(|h| h
                .get("text")
                .and_then(Json::as_str)
                .is_some_and(|t| t.contains("77 kelvin"))),
            "a document that was put must come back to a query about it: {hits:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_memory_append_becomes_retrievable_context() {
        let dir = temp_dir("append-retrievable");
        let mut responder = responder_with(&dir, &[]);
        let payload = Json::obj(vec![
            ("run", Json::str("migration")),
            (
                "outcome",
                Json::str("the ledger import duration is 40 minutes"),
            ),
        ]);
        submit(&dir, "m1", &append(payload));

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(
            report.memory_appends, 1,
            "the payload key is what says `memory`"
        );
        assert_eq!(report.documents, 0);
        assert!(report.failures.is_empty(), "{:?}", report.failures);

        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "how long is the ledger import duration?", 6, "tech"),
        )
        .unwrap();
        responder.poll_once().unwrap();
        let hits = hits_of(&dir, "r1");
        assert!(
            hits.iter().any(|h| h
                .get("text")
                .and_then(Json::as_str)
                .is_some_and(|t| t.contains("40 minutes"))),
            "an appended run must be recallable by a later turn: {hits:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_acknowledgement_reports_a_document_the_indexer_could_not_read() {
        // The honest limit of the write half, stated as a control. The query
        // half serves planned *nodes*, not documents: a `ContextPut` whose text
        // the indexer turns into no typed state is stored and acknowledged, and
        // will never be served. That is the same "an empty working set reads as
        // the history does not contain it" trap, reached by a document the
        // peer is certain it just wrote — so `nodes` is in the acknowledgement,
        // and this is where a peer finds out.
        let dir = temp_dir("ack-nodes");
        let mut responder = responder_with(&dir, &[]);
        submit(
            &dir,
            "w1",
            &put(
                "doc-prose",
                "document",
                "Restart the ledger with ledgerctl.",
                Json::Null,
            ),
        );
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1, "it was accepted, and said so");
        let ack = ack_of(&dir, "w1").expect("an accepted write is acknowledged");
        assert_eq!(
            ack.get("nodes").and_then(Json::as_usize),
            Some(0),
            "a peer is told, at write time, that nothing was extractable"
        );
        assert!(
            ack.get("spans").and_then(Json::as_usize).unwrap_or_default() > 0,
            "the text was still stored, and the ack says how much of it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_acknowledgement_names_the_document_a_peer_can_look_for() {
        let dir = temp_dir("ack-shape");
        let mut responder = responder_with(&dir, &[]);
        submit(
            &dir,
            "w1",
            &put("doc-9", "document", "The relay listens on port 8443.", Json::Null),
        );
        responder.poll_once().unwrap();

        let ack = ack_of(&dir, "w1").expect("an accepted write is acknowledged");
        assert_eq!(ack.get("version").and_then(Json::as_u64), Some(1));
        assert_eq!(
            ack.get("id").and_then(Json::as_str),
            Some("w1"),
            "the acknowledgement must correlate with the file stem it answers"
        );
        assert_eq!(
            ack.get("document").and_then(Json::as_str),
            Some(document_id_for(&ContextPut::new("doc-9", "document", "", Json::Null)).as_str()),
            "the acknowledgement names the document the put landed under"
        );
        assert!(
            ack.get("spans").and_then(Json::as_usize).unwrap_or_default() > 0,
            "a document with content in it indexed something"
        );
        assert!(
            ack.get("ok").is_none(),
            "there is no success flag: the file's existence is the promise"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_that_could_not_be_ingested_gets_no_acknowledgement() {
        // L0 is immutable: the same document id with different bytes is a hard
        // error, not an edit. Converting that into an acknowledgement would
        // tell a peer its new text is retrievable when the store still holds
        // the old text — and nothing downstream could correct that, because
        // the peer's next action is to stop retrying.
        let dir = temp_dir("put-conflict");
        let mut responder = responder_with(&dir, &[]);
        submit(
            &dir,
            "w1",
            &put("doc-9", "document", "The relay listens on port 8443.", Json::Null),
        );
        assert_eq!(responder.poll_once().unwrap().written, 1);

        submit(
            &dir,
            "w2",
            &put("doc-9", "document", "The relay listens on port 9999.", Json::Null),
        );
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 0, "a refused write is not a write");
        assert_eq!(report.failures.len(), 1, "the refusal is reported, not hidden");
        assert!(
            ack_of(&dir, "w2").is_none(),
            "a write that failed must leave the peer with no answer to read"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_put_resubmitted_with_identical_text_is_accepted_again() {
        // The retry path. A peer whose acknowledgement it never collected
        // resubmits; L0 treats identical text under a used id as a no-op, so the
        // retry is acknowledged rather than refused. If it were not, the
        // timeout rule would make a lost ack unrecoverable.
        let dir = temp_dir("put-retry");
        let mut responder = responder_with(&dir, &[]);
        let body = put("doc-9", "document", "The relay listens on port 8443.", Json::Null);
        submit(&dir, "w1", &body);
        assert_eq!(responder.poll_once().unwrap().written, 1);
        let documents = responder.runtime_mut().raw.documents().len();

        // A different channel id carrying the same document: a distinct write,
        // because the acknowledgement is keyed on the channel id.
        submit(&dir, "w2", &body);
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(
            responder.runtime_mut().raw.documents().len(),
            documents,
            "L0 immutability means a retry adds no second copy"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_acknowledged_write_is_not_acknowledged_twice() {
        // Same rule as the response half, and for the same reason: a responder
        // is a poller of a directory it does not own, and a put file can
        // reappear if a peer crashed between writing and collecting the ack.
        let dir = temp_dir("ack-idempotent");
        let mut responder = responder_with(&dir, &[]);
        submit(&dir, "w1", &put("doc-9", "document", "Rate is 180 USD.", Json::Null));
        assert_eq!(responder.poll_once().unwrap().written, 1);
        let first = std::fs::read_to_string(dir.join("w1.put-ack.json")).unwrap();

        let again = responder.poll_once().unwrap();
        assert_eq!(again.written, 0);
        assert_eq!(again.skipped_acked, 1);
        assert_eq!(
            std::fs::read_to_string(dir.join("w1.put-ack.json")).unwrap(),
            first,
            "the acknowledgement on disk is not rewritten under a peer that may be reading it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writes_are_applied_before_the_queries_in_the_same_pass() {
        // A peer that drops a put and a query into the same directory inside
        // one turn means the query to be answered against the put. Answering it
        // first would serve a store one write behind and report it as
        // authoritative.
        let dir = temp_dir("write-then-read");
        let mut responder = responder_with(&dir, &[]);
        submit(
            &dir,
            "w1",
            &put("doc-9", "document", "The billing contact is rota@example.net.", Json::Null),
        );
        std::fs::write(
            dir.join("r1.request.json"),
            request("r1", "who is the billing contact?", 6, "tech"),
        )
        .unwrap();

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert_eq!(report.answered, 1, "one pass serves both halves");
        let hits = hits_of(&dir, "r1");
        assert!(
            hits.iter().any(|h| h
                .get("text")
                .and_then(Json::as_str)
                .is_some_and(|t| t.contains("rota@example.net"))),
            "the query must see the write that landed in the same pass: {hits:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_bad_write_does_not_strand_the_others() {
        let dir = temp_dir("bad-write");
        let mut responder = responder_with(&dir, &[]);
        std::fs::write(dir.join("broken.put.json"), "{not json").unwrap();
        submit(&dir, "good", &put("doc-9", "document", "Rate is 180 USD.", Json::Null));

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1, "the good write still lands");
        assert_eq!(report.failures.len(), 1, "the bad one is reported, not hidden");
        assert!(
            !dir.join("broken.put-ack.json").exists(),
            "an unparseable write gets no fabricated acknowledgement"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_put_never_lets_a_peer_choose_a_path() {
        // The asymmetry, stated as a control. The *channel* id is the file name
        // and is held to [A-Za-z0-9-_]; `ContextPut.id` is the peer's document
        // id and may contain anything, because it is only ever used as a map
        // key. A responder that validated the document id as a path would
        // refuse legitimate documents; one that trusted it as a path would let
        // a peer write outside the channel.
        let dir = temp_dir("doc-id");
        let mut responder = responder_with(&dir, &[]);
        let hostile = "../../etc/passwd";
        submit(&dir, "w1", &put(hostile, "document", "Rate is 180 USD.", Json::Null));
        assert_eq!(responder.poll_once().unwrap().written, 1);

        let ack = ack_of(&dir, "w1").expect("a document id is not a file name");
        assert_eq!(
            ack.get("document").and_then(Json::as_str),
            Some(format!("interop:document:{hostile}").as_str()),
            "the document id is stored verbatim, never used to build a path"
        );
        let outside = dir.parent().unwrap().join("etc");
        assert!(
            !outside.exists(),
            "nothing was written outside the channel directory"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn meta_source_becomes_provenance_and_nothing_else_is_invented() {
        // Two puts, one declaring a provenance source and one not. The second
        // must come back with an empty `source` rather than a string this side
        // made up: unifur renders `source=` into the prompt, and an invented
        // one is an unsourced fact wearing a citation.
        let dir = temp_dir("meta-source");
        let mut responder = responder_with(&dir, &[]);
        let tagged = Json::obj(vec![
            ("source", Json::str("cybercorpus/text/runbooks")),
            ("tags", Json::Arr(vec![Json::str("runbook")])),
        ]);
        submit(
            &dir,
            "w1",
            &put(
                "doc-a",
                "document",
                "The ledger restart command is ledgerctl.",
                tagged,
            ),
        );
        submit(
            &dir,
            "w2",
            &put("doc-b", "document", "The token rotation interval is hourly.", Json::Null),
        );
        assert_eq!(responder.poll_once().unwrap().written, 2);

        for (channel_id, needle) in [("r1", "ledgerctl"), ("r2", "token rotation")] {
            std::fs::write(
                dir.join(format!("{channel_id}.request.json")),
                request(channel_id, needle, 4, "tech"),
            )
            .unwrap();
        }
        responder.poll_once().unwrap();

        // One document yields more than one node, so compare the set of sources
        // rather than one hit's: what matters is that every hit out of that
        // document agrees, and that it is the declared value and nothing else.
        let sources_of = |channel_id: &str| -> BTreeSet<String> {
            hits_of(&dir, channel_id)
                .iter()
                .filter(|h| h
                    .get("provenance")
                    .and_then(|p| p.get("path"))
                    .and_then(Json::as_str)
                    .is_some_and(|p| p.starts_with("interop:document:")))
                .map(|h| {
                    h.get("provenance")
                        .and_then(|p| p.get("source"))
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect()
        };
        assert_eq!(
            sources_of("r1"),
            BTreeSet::from(["cybercorpus/text/runbooks".to_string()]),
            "a declared source is carried through"
        );
        assert_eq!(
            sources_of("r2"),
            BTreeSet::from([String::new()]),
            "an undeclared source stays empty rather than being fabricated"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_four_channel_suffixes_cannot_be_confused_for_one_another() {
        // Every suffix is matched with `ends_with` against a name this side did
        // not choose, so a suffix that is a suffix of another is a file read as
        // the wrong kind of message — silently, because a misparsed file is a
        // parse error and a misread *kind* is not. The one that could bite is
        // the acknowledgement: without its `-`, it would end with `.put.json`
        // and the responder would ingest its own output on the next pass.
        let names = ["r1.request.json", "r1.response.json", "w1.put.json", "w1.put-ack.json"];
        let suffixes = [REQUEST_SUFFIX, RESPONSE_SUFFIX, WRITE_SUFFIX, WRITE_ACK_SUFFIX];
        for name in names {
            let matching: Vec<&str> = suffixes
                .iter()
                .copied()
                .filter(|suffix| name.ends_with(suffix))
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "{name} must match exactly one channel suffix, matched {matching:?}"
            );
        }
    }

    #[test]
    fn a_missing_version_is_refused_on_both_write_messages() {
        // The drift failure mode the write half is most exposed to, because a
        // peer that has not adopted it yet will be sending something. serde's
        // `#[serde(default)]` on unifur's side turns an absent `version` into
        // 0; if only the explicit wrong-version case were checked, a document
        // would be stored under a shape nobody agreed on.
        for body in [
            Json::obj(vec![
                ("id", Json::str("doc-9")),
                ("text", Json::str("body")),
            ])
            .to_json_string(),
            Json::obj(vec![("payload", Json::str("body"))]).to_json_string(),
        ] {
            let put_error = ContextPut::from_json(&body).unwrap_err();
            assert!(
                put_error.contains("version 0"),
                "ContextPut: {put_error}"
            );
            let append_error = MemoryAppend::from_json(&body).unwrap_err();
            assert!(
                append_error.contains("version 0"),
                "MemoryAppend: {append_error}"
            );
        }
    }

    #[test]
    fn an_unknown_write_version_is_refused_rather_than_reinterpreted() {
        let future = Json::obj(vec![
            ("version", Json::num(2.0)),
            ("id", Json::str("doc-9")),
            ("kind", Json::str("document")),
            ("text", Json::str("body")),
            ("meta", Json::Null),
        ])
        .to_json_string();
        assert!(ContextPut::from_json(&future).unwrap_err().contains("version 2"));

        let future = Json::obj(vec![
            ("version", Json::num(7.0)),
            ("payload", Json::str("body")),
        ])
        .to_json_string();
        assert!(MemoryAppend::from_json(&future)
            .unwrap_err()
            .contains("version 7"));
    }

    #[test]
    fn a_write_file_at_an_unknown_version_is_not_ingested() {
        // The same rule end to end through the channel: refusing to parse must
        // also mean nothing reached the store and no acknowledgement appeared,
        // or a peer would take the acknowledgement as the version check.
        let dir = temp_dir("bad-version");
        let mut responder = responder_with(&dir, &[]);
        let future = Json::obj(vec![
            ("version", Json::num(2.0)),
            ("id", Json::str("doc-9")),
            ("kind", Json::str("document")),
            ("text", Json::str("Rate is 180 USD.")),
            ("meta", Json::Null),
        ])
        .to_json_string();
        submit(&dir, "w1", &future);

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 0);
        assert_eq!(report.failures.len(), 1);
        assert!(ack_of(&dir, "w1").is_none());
        assert!(
            responder.runtime_mut().raw.documents().is_empty(),
            "a shape this build cannot parse must not reach the store"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_append_ids_are_content_addressed_so_a_repeat_is_a_noop() {
        // The retry problem. An append carries no id of its own, so an id from a
        // counter would make every retry after a lost acknowledgement a second
        // entry; derived from the payload, the same append is the same
        // document, and L0 immutability makes the repeat a no-op.
        let payload = Json::obj(vec![
            ("b", Json::num(2.0)),
            ("a", Json::num(1.0)),
        ]);
        let reordered = Json::obj(vec![
            ("a", Json::num(1.0)),
            ("b", Json::num(2.0)),
        ]);
        let append_1 = MemoryAppend::new(payload);
        assert_eq!(append_1.document_id(), append_1.document_id());
        assert_eq!(
            append_1.document_id(),
            MemoryAppend::new(reordered).document_id(),
            "canonical JSON makes key order irrelevant to a payload's identity"
        );
        assert_ne!(
            append_1.document_id(),
            MemoryAppend::new(Json::str("something else")).document_id(),
            "two different payloads must not share an entry"
        );

        let dir = temp_dir("append-retry");
        let mut responder = responder_with(&dir, &[]);
        let body = append(Json::obj(vec![("a", Json::num(1.0))]));
        submit(&dir, "m1", &body);
        submit(&dir, "m2", &body);
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 2, "both channel ids are acknowledged");
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(
            responder.runtime_mut().raw.documents().len(),
            1,
            "and they are the same document in the store"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_responder_with_a_store_seals_what_it_acknowledged() {
        // The durability half of the promise. An acknowledgement says the store
        // has the document; a responder that says so and then restarts without
        // it has told the peer to stop retrying for something that is gone.
        // A `.context` container is the interesting case because that is the
        // Merkle-chained log the peer's note asks the writes to land in, and
        // reopening it is the only way to know a generation was really written.
        let dir = temp_dir("seal");
        let store = dir.join("store.context");
        ContextStore::create(&store, "interop-test")
            .expect("a container to seal into");
        let mut responder = responder_with(&dir, &[]).with_store(&store);
        submit(
            &dir,
            "w1",
            &put("doc-9", "document", "The relay port is 8443.", Json::Null),
        );

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(
            responder.runtime_mut().raw.documents().len(),
            1,
            "and it is in the runtime that was sealed"
        );

        let reopened = Dcr::open_context(&store, 400).expect("the sealed generation reopens");
        assert_eq!(
            reopened
                .raw
                .documents()
                .iter()
                .map(|d| d.id.clone())
                .collect::<Vec<_>>(),
            ["interop:document:doc-9"],
            "an acknowledged write must survive the process that acknowledged it"
        );

        // An idle pass must not grow the container: a store whose chain
        // recorded the responder's wake-ups would be a log of something other
        // than what it learned.
        let generations_before = ContextStore::open(&store)
            .map(|store| store.checkpoints().len())
            .unwrap_or_default();
        assert_eq!(responder.poll_once().unwrap().written, 0);
        let generations_after = ContextStore::open(&store)
            .map(|store| store.checkpoints().len())
            .unwrap_or_default();
        assert_eq!(
            generations_before, generations_after,
            "a pass that ingested nothing writes no generation"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_is_sealed_before_it_is_acknowledged() {
        // The ordering, as a control that can fail. A store path that cannot be
        // sealed must produce *no* acknowledgement: the alternative leaves the
        // peer holding permission to stop retrying for a document that exists
        // only in a runtime which is about to be gone. The contents matter —
        // a store path that silently fell back to `save` would pass the
        // durability check and lose the Merkle chain the peer's note asks for.
        let dir = temp_dir("seal-order");
        // A path that cannot be sealed: underneath a plain file, so neither
        // `save_context` nor `save` can put anything there. The channel
        // directory has to exist before the blocker goes down, which is why
        // the responder — and the directory — is made first.
        let responder = responder_with(&dir, &[]);
        let blocked = dir.join("not-a-dir/inner");
        std::fs::write(dir.join("not-a-dir"), "x").unwrap();
        let mut responder = responder.with_store(&blocked);
        submit(
            &dir,
            "w1",
            &put("doc-9", "document", "The relay port is 8443.", Json::Null),
        );

        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 0, "a write that cannot be sealed is not a write");
        assert!(!report.failures.is_empty(), "and the reason is reported");
        assert!(
            !dir.join("w1.put-ack.json").exists(),
            "no acknowledgement may outlive the durability it promises"
        );

        // The same write, resubmitted by the peer against a store that works,
        // is acknowledged and sealed. The retry is free because L0 treats
        // identical text under a used id as a no-op.
        let store = dir.join("good.context");
        ContextStore::create(&store, "interop-test").expect("a container to seal into");
        let mut responder =
            Responder::new(&dir, responder.runtime, 8).unwrap().with_store(&store);
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert!(
            Dcr::open_context(&store, 400).is_ok(),
            "and the acknowledged document is in the sealed container"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_responder_without_a_store_leaves_persistence_to_its_caller() {
        // `--once` seals on the way out; the default responder must not start
        // writing files the caller never named a destination for.
        let dir = temp_dir("no-seal");
        let mut responder = responder_with(&dir, &[]);
        submit(&dir, "w1", &put("doc-9", "document", "The relay port is 8443.", Json::Null));
        let report = responder.poll_once().unwrap();
        assert_eq!(report.written, 1);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            2,
            "the write and its acknowledgement, and nothing else"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_contract_json_unifur_writes_is_what_this_responder_speaks() {
        // The peer publishes its own contract document. If a field name or a
        // version drifts, this fails here rather than at 3am in a chat turn.
        let contract = Path::new("/srv/m-sda/unifur/interop/contract.json");
        if !contract.exists() {
            return; // the peer has not run yet; nothing to disagree with
        }
        let body = std::fs::read_to_string(contract).unwrap();
        let doc = json::parse(&body).unwrap();
        assert_eq!(
            doc.get("protocol").and_then(Json::as_str),
            Some(PROTOCOL),
            "protocol name must match the peer's contract.json"
        );
        let messages = doc.get("messages").and_then(Json::as_array).unwrap();
        let message = |name: &str| {
            messages
                .iter()
                .find(|m| m.get("name").and_then(Json::as_str) == Some(name))
                .unwrap_or_else(|| panic!("{name} missing from the peer contract"))
        };
        let version_of = |name: &str| {
            message(name)
                .get("version")
                .and_then(Json::as_u64)
                .unwrap_or_else(|| panic!("{name} has no version in the peer contract"))
        };
        assert_eq!(version_of("ContextPut"), CONTEXT_PUT_VERSION as u64);
        assert_eq!(version_of("ContextQuery"), CONTEXT_QUERY_VERSION as u64);
        assert_eq!(version_of("ContextHit"), CONTEXT_HIT_VERSION as u64);
        assert_eq!(version_of("MemoryAppend"), MEMORY_APPEND_VERSION as u64);
        assert_eq!(version_of("ChannelRequest"), CHANNEL_ENVELOPE_VERSION as u64);
        assert_eq!(version_of("ChannelResponse"), CHANNEL_ENVELOPE_VERSION as u64);

        // The declared field lists, against what this build actually speaks.
        // A field the peer does not declare would be one half of a message the
        // other half cannot produce; a declared field this build ignores is a
        // value silently lost, which the note says is the peer-owned payload.
        let fields_of = |name: &str| -> Vec<String> {
            message(name)
                .get("fields")
                .and_then(Json::as_array)
                .unwrap_or_default()
                .iter()
                .filter_map(|f| f.as_str().map(str::to_string))
                .collect()
        };
        assert_eq!(fields_of("ContextPut"), ["id", "kind", "text", "meta"]);
        assert_eq!(fields_of("MemoryAppend"), ["payload"]);
        assert_eq!(
            fields_of("ContextQuery"),
            ["text", "k", "field"],
            "the query half must not drift either"
        );

        // Everything this build serialises for a message, by name — so a field
        // added here without a peer adding it is a loud failure, not a message
        // the peer drops on the floor.
        let keys_of = |body: String| -> Vec<String> {
            json::parse(&body)
                .map(|v| match v {
                    Json::Obj(pairs) => pairs.into_iter().map(|(k, _)| k).collect(),
                    _ => Vec::new(),
                })
                .unwrap_or_default()
        };
        assert_eq!(
            keys_of(put("doc-9", "document", "body", Json::Null)),
            ["version", "id", "kind", "text", "meta"]
        );
        assert_eq!(
            keys_of(append(Json::Null)),
            ["version", "payload"]
        );
        assert_eq!(
            keys_of(ContextQuery {
                version: CONTEXT_QUERY_VERSION,
                text: "q".into(),
                k: 1,
                field: "tech".into(),
            }
            .to_json()),
            ["version", "text", "k", "field"]
        );

        let channel = doc.get("channel").expect("channel section");
        assert_eq!(
            channel.get("request_suffix").and_then(Json::as_str),
            Some(REQUEST_SUFFIX)
        );
        assert_eq!(
            channel.get("response_suffix").and_then(Json::as_str),
            Some(RESPONSE_SUFFIX)
        );
        // The write suffixes are this build's own extension: the deployed
        // contract declares `request_suffix` and `response_suffix` and is
        // silent about writes. If the peer ever declares them, these
        // constants are what has to agree — so it is asserted now, against
        // whatever it says, rather than discovered from a poll report in
        // production.
        for (key, ours) in [
            ("write_suffix", WRITE_SUFFIX),
            ("write_ack_suffix", WRITE_ACK_SUFFIX),
        ] {
            if let Some(declared) = channel.get(key).and_then(Json::as_str) {
                assert_eq!(declared, ours, "the peer's {key} must match this build");
            }
        }
    }
}


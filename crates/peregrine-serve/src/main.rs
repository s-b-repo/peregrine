//! peregrine-serve — a native, OpenAI-compatible HTTP server for the peregrine
//! engine (axum/tokio). Exposes `POST /v1/chat/completions` (streaming SSE and
//! non-streaming), `GET /v1/models`, and `GET /health`.
//!
//! Design & safety:
//! - A single [`batch`] engine thread owns the model and continuously batches all
//!   in-flight requests (one decode token per active sequence per step), so
//!   concurrent requests share expert reads instead of serializing behind a lock.
//!   Each handler submits a request and streams decoded token deltas back over an
//!   async channel.
//! - No panics anywhere (deny-lints below); every error becomes an OpenAI-shaped
//!   JSON body. Binds `127.0.0.1` by default; optional bearer `--api-key`;
//!   `max_tokens` and prompt-length caps; graceful Ctrl-C shutdown.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod batch;
mod kvstore;
mod memo;
mod tok;
mod tools;

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use batch::{EngineHandle, EngineOut, EngineRequest, Priority};
use clap::Parser;
use peregrine_core::Error;
use peregrine_model::{Model, Sampler};
use serde::{Deserialize, Serialize};
use tok::TokenBackend;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// CLI configuration.
#[derive(Parser, Clone)]
#[command(name = "peregrine-serve", about = "OpenAI-compatible server for peregrine")]
struct Args {
    /// Model directory (config.json + *.safetensors + tokenizer.json). Falls back to $COLI_MODEL.
    #[arg(long, env = "COLI_MODEL")]
    model: String,
    /// Bind host (default localhost for safety).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Require this bearer token on every request when set.
    #[arg(long, env = "PEREGRINE_API_KEY")]
    api_key: Option<String>,
    /// Hard cap on generated tokens per request.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Hard cap on prompt tokens per request.
    #[arg(long, default_value_t = 8192)]
    max_prompt_tokens: usize,
    /// Public model id reported by /v1/models and in responses.
    #[arg(long, default_value = "glm-5.2")]
    model_id: String,
    /// Max sequences decoded together per batched step (continuous-batching width).
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Benchmark the tokenizer backends on a text file and exit: encodes the
    /// file through the gigatoken tokenizer, reports MB/s. Needs
    /// `<model>/tokenizer.json`; no model weights are loaded.
    #[arg(long, value_name = "TEXT_FILE")]
    bench_tokenizer: Option<std::path::PathBuf>,
}

/// Shared, cloneable server state.
#[derive(Clone)]
struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    engine: EngineHandle,
    tokenizer: Arc<TokenBackend>,
    args: Args,
    /// Bounded exact response memo. Serves a byte-identical greedy request from a
    /// prior certified completion without entering the model. Deliberately *not*
    /// part of the engine: a memo hit must never become a KV boundary.
    memo: parking_lot::Mutex<memo::ResponseMemo>,
    /// `true` when the loaded model expects ChatML prompts (Qwen family), `false`
    /// for GLM's `[gMASK]<sop>` markup. Captured at load from the model's arch
    /// before it moves into the engine thread; selects `build_prompt`'s dialect.
    chatml_prompt: bool,
}

// ---- OpenAI request/response shapes ----

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    /// Tool schemas, in OpenAI shape. Rendered into the system turn for the
    /// model (see [`tools::render_preamble`]) — GLM-5.2 does not read a `tools`
    /// field, it reads the markup its tokenizer has tokens for.
    #[serde(default)]
    tools: Option<Vec<tools::ToolDef>>,
    /// Accepted for compatibility. `"none"` suppresses the tool preamble;
    /// `"auto"`, `"required"`, and a named-function choice all render the same
    /// schemas, because forcing a call is a decoding constraint this engine
    /// does not implement — claiming otherwise would be worse than ignoring it.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize, Clone)]
struct ChatMessage {
    role: String,
    /// Absent, `null`, a string, or an array of content parts — every shape the
    /// OpenAI-compatible clients actually send. An assistant turn that is
    /// *only* tool calls carries no content at all, so this cannot be required.
    #[serde(default)]
    content: Option<MessageContent>,
    /// Assistant turns replaying the model's own calls.
    ///
    /// A `role: "tool"` turn's `tool_call_id` needs no field here: serde ignores
    /// unknown members, and the prompt form is positional — results render in
    /// the order they arrive, so nothing correlates them by id.
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
}

/// The two content encodings OpenAI clients use, plus the parts form's
/// non-text members (images, files) which this text-only engine drops.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize, Clone)]
struct ContentPart {
    #[serde(default)]
    text: Option<String>,
}

impl ChatMessage {
    /// The message's text, with the parts form flattened. Non-text parts
    /// contribute nothing rather than a placeholder: a caption invented here
    /// would be indistinguishable to the model from something the user wrote.
    fn text(&self) -> String {
        match &self.content {
            None => String::new(),
            Some(MessageContent::Text(s)) => s.clone(),
            Some(MessageContent::Parts(ps)) => {
                ps.iter().filter_map(|p| p.text.as_deref()).collect::<Vec<_>>().join("")
            }
        }
    }
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}
#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

// ---- error → OpenAI JSON ----

struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
}
impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> ApiError {
        ApiError { status, kind, message: message.into() }
    }
    fn bad_request(m: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", m)
    }
    fn internal(m: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "message": self.message, "type": self.kind }
        });
        (self.status, Json(body)).into_response()
    }
}
impl From<Error> for ApiError {
    fn from(e: Error) -> ApiError {
        ApiError::internal(e.to_string())
    }
}

/// Convert a tokenizer error into our error type — the one boundary that
/// needs it, kept in a single helper.
fn tk<T>(r: Result<T, peregrine_core::Error>) -> Result<T, ApiError> {
    r.map_err(|e| ApiError::internal(format!("tokenizer: {e}")))
}

/// Build the GLM-5.2 prompt from chat messages (no chat_template ships in the
/// tokenizer): `[gMASK]<sop>` then `<|role|>\n{content}` per turn, ending with
/// an empty `<|assistant|>` turn to generate into.
/// `tools` are rendered into the *first* system turn (GLM's own template puts
/// them there); a conversation with no system turn gets one, since a tools
/// block appended to a user turn reads to the model as the user quoting
/// schemas at it.
fn build_prompt(messages: &[ChatMessage], tools: &[tools::ToolDef], chatml: bool) -> String {
    if chatml {
        return build_prompt_chatml(messages, tools);
    }
    let mut s = String::from("[gMASK]<sop>");
    let preamble = if tools.is_empty() { String::new() } else { tools::render_preamble(tools) };
    let mut preamble_placed = preamble.is_empty();
    if !preamble_placed && !messages.iter().any(|m| m.role == "system") {
        s.push_str("<|system|>\n");
        s.push_str(preamble.trim_start_matches('\n'));
        preamble_placed = true;
    }
    for m in messages {
        let role = match m.role.as_str() {
            "system" => "<|system|>",
            "assistant" => "<|assistant|>",
            "user" => "<|user|>",
            // A tool result is an observation turn — the role the model was
            // trained to read results in, not a user message about one.
            "tool" => "<|observation|>",
            // unknown roles are treated as user content, never trusted as markup
            _ => "<|user|>",
        };
        s.push_str(role);
        s.push('\n');
        if m.role == "tool" {
            s.push_str("<tool_response>\n");
            s.push_str(&m.text());
            s.push_str("\n</tool_response>");
        } else {
            s.push_str(&m.text());
        }
        if m.role == "system" && !preamble_placed {
            s.push_str(&preamble);
            preamble_placed = true;
        }
        // An assistant turn that called tools replays as the markup it emitted.
        if m.role == "assistant" {
            if let Some(calls) = &m.tool_calls {
                s.push_str(&tools::render_assistant_calls(calls));
            }
        }
    }
    s.push_str("<|assistant|>\n");
    s
}

/// Build a ChatML prompt (Qwen family): `<|im_start|>role\n{content}<|im_end|>`
/// per turn, ending with an open `<|im_start|>assistant\n` to generate into.
/// GLM's `[gMASK]<sop>` markup is invalid here — feeding it to Qwen tokenizes to
/// stray control tokens and degenerates the output, which is the bug this fixes.
fn build_prompt_chatml(messages: &[ChatMessage], tools: &[tools::ToolDef]) -> String {
    let mut s = String::new();
    let preamble = if tools.is_empty() { String::new() } else { tools::render_preamble(tools) };
    let mut preamble_placed = preamble.is_empty();
    // A tools block with no system turn gets its own, so the model does not read
    // it as the user quoting schemas (same rule as the GLM path).
    if !preamble_placed && !messages.iter().any(|m| m.role == "system") {
        s.push_str("<|im_start|>system\n");
        s.push_str(preamble.trim_start_matches('\n'));
        s.push_str("<|im_end|>\n");
        preamble_placed = true;
    }
    for m in messages {
        let role = match m.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user", // unknown roles are user content, never trusted as markup
        };
        s.push_str("<|im_start|>");
        s.push_str(role);
        s.push('\n');
        if m.role == "tool" {
            s.push_str("<tool_response>\n");
            s.push_str(&m.text());
            s.push_str("\n</tool_response>");
        } else {
            s.push_str(&m.text());
        }
        if m.role == "system" && !preamble_placed {
            s.push('\n');
            s.push_str(preamble.trim_start_matches('\n'));
            preamble_placed = true;
        }
        if m.role == "assistant" {
            if let Some(calls) = &m.tool_calls {
                s.push_str(&tools::render_assistant_calls(calls));
            }
        }
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    // Pre-close the reasoning block — what the shipped chat_template.jinja
    // emits for `enable_thinking=false`.
    //
    // Without it the model opens `<think>` itself (Qwen3.5 is trained to) and
    // reasons, and this server's OutputFilter DROPS `<think>…</think>` by
    // design. A request whose budget runs out before `</think>` then returns
    // completion_tokens == max_tokens with an EMPTY string and no content
    // deltas: the model worked, every token was reasoning, the filter
    // discarded all of it. That is why the failure was content-dependent
    // rather than length-dependent — it tracks how long the model reasons, not
    // how long the prompt is — and why "/no_think" in the user text did
    // nothing, since the model's own opening tag is not the user's to suppress.
    //
    // `COLI_QWEN_THINK=1` restores reasoning for callers who want it and will
    // budget for it: with the filter dropping the block, a small max_tokens
    // spends the whole budget on text nobody sees.
    if !matches!(std::env::var("COLI_QWEN_THINK").as_deref(), Ok("1") | Ok("true")) {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}

/// The tool schemas this request should expose to the model, honouring
/// `tool_choice: "none"`.
fn active_tools(req: &ChatRequest) -> &[tools::ToolDef] {
    if req.tool_choice.as_ref().and_then(|v| v.as_str()) == Some("none") {
        return &[];
    }
    req.tools.as_deref().unwrap_or(&[])
}

/// A monotonically-unique-ish seed for the sampler without extra deps.
fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E3779B9)
}

/// Submit a request to the batch engine and return its token-id stream. The
/// caller decodes ids to text — incrementally for streaming, in one shot
/// otherwise — so the engine stays tokenizer-free and both paths share it.
fn submit_request(
    state: &AppState,
    ids: &[u32],
    max_new: usize,
    temperature: f32,
    top_p: f32,
    priority: Priority,
    class: peregrine_model::TokenClass,
) -> Result<mpsc::UnboundedReceiver<EngineOut>, ApiError> {
    // Unbounded on purpose: the engine thread emits into this channel for every
    // active sequence, so a bounded channel lets one connected-but-not-reading
    // client block the engine and freeze *every* concurrent stream. The queue is
    // bounded in practice by `max_new` token ids (kilobytes), and the handler
    // still paces the client.
    let (tx, rx) = mpsc::unbounded_channel::<EngineOut>();
    let prompt: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    let sampler = Sampler::new(temperature, top_p, seed());
    state
        .inner
        .engine
        .submit(EngineRequest { prompt, max_new, sampler, out: tx, priority, class })
        .map_err(|r| match r {
            // Backpressure, not failure: the backlog is at COLI_QUEUE_DEPTH and
            // the honest answer is "retry", not a queue that grows until the
            // client times out anyway.
            batch::SubmitRefused::Full => ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "overloaded_error",
                "engine queue is full; retry later",
            ),
            batch::SubmitRefused::Down(m) => ApiError::internal(format!("batch engine is not running: {m}")),
        })?;
    Ok(rx)
}

/// Classify the workload from the tail of the last user message — the part of
/// the conversation the model is about to continue, so the best signal for
/// what routing distribution decode will see. The tail is capped at 512 chars
/// (classification is a ratio heuristic; more text doesn't sharpen it).
fn classify_request(messages: &[ChatMessage]) -> peregrine_model::TokenClass {
    // `match`, not `.map(..).unwrap_or_default()`: no-user-turn is a real case
    // here (a tools-only or system-only request), and the empty string is the
    // answer to it rather than a stand-in for one.
    let last_user = match messages.iter().rev().find(|m| m.role == "user") {
        Some(m) => m.text(),
        None => String::new(),
    };
    let last_user = last_user.as_str();
    let tail_start = last_user.len().saturating_sub(512);
    // step forward to a char boundary so the slice is valid UTF-8
    let mut start = tail_start;
    while start < last_user.len() && !last_user.is_char_boundary(start) {
        start += 1;
    }
    peregrine_model::classify_str(&last_user[start..])
}

/// Parse an `X-Peregrine-Priority` header value into a [`Priority`]. Unknown
/// values fall back to `Normal` — a deliberately-lax mapping so a client can't
/// break admission by sending an unrecognized string.
fn priority_from_header(v: Option<&str>) -> Priority {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("high") | Some("1") | Some("true") => Priority::High,
        _ => Priority::Normal,
    }
}

/// Resolve + validate common generation params against the server caps.
async fn resolve_params(state: &AppState, req: &ChatRequest) -> Result<(Vec<u32>, usize, f32, f32), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let prompt = build_prompt(&req.messages, active_tools(req), state.inner.chatml_prompt);
    // Encode is CPU-bound and serialized behind the process-wide encode mutex
    // (`tok.rs`: `encode` is `&mut`, so every request takes the same lock). Run it
    // on the blocking pool: a burst of B arrivals then parks blocking-pool threads
    // waiting on that mutex, not the runtime workers the SSE pump tasks and every
    // other endpoint are scheduled on. parking_lot does not yield to tokio, so
    // before this change B-1 of a burst's handlers each pinned a worker thread
    // doing nothing.
    let tokenizer = state.inner.tokenizer.clone();
    let ids = tk(tokio::task::spawn_blocking(move || tokenizer.encode(&prompt))
        .await
        .map_err(|e| ApiError::internal(format!("encode task: {e}")))?)?;
    if ids.len() > state.inner.args.max_prompt_tokens {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!("prompt is {} tokens; max is {}", ids.len(), state.inner.args.max_prompt_tokens),
        ));
    }
    let max_new = req.max_tokens.unwrap_or(256).min(state.inner.args.max_tokens).max(1);
    let temperature = req.temperature.unwrap_or(0.0).clamp(0.0, 2.0);
    let top_p = req.top_p.unwrap_or(0.95).clamp(0.0, 1.0);
    Ok((ids, max_new, temperature, top_p))
}

/// A header value as UTF-8, treating a non-UTF-8 value as absent — the lax
/// parse both header consumers (auth, priority) want. The rejection is
/// reported (`COLI_DEBUG=1`) rather than silently `.ok()`-dropped.
fn header_utf8(v: &axum::http::HeaderValue) -> Option<&str> {
    match v.to_str() {
        Ok(s) => Some(s),
        Err(e) => {
            peregrine_core::note_advisory_err("non-UTF-8 request header ignored", &e);
            None
        }
    }
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(want) = state.inner.args.api_key.as_deref() else {
        return Ok(());
    };
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(header_utf8)
        .and_then(|s| s.strip_prefix("Bearer "));
    if got.is_some_and(|g| constant_time_eq(g.as_bytes(), want.as_bytes())) {
        Ok(())
    } else {
        Err(ApiError::new(StatusCode::UNAUTHORIZED, "invalid_request_error", "missing or invalid API key"))
    }
}

/// Compare two secrets without leaking their common prefix length through
/// timing. `==` on strings returns at the first differing byte, which lets a
/// caller recover the key byte-by-byte.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let (hits, misses, entries, bytes) = state.inner.memo.lock().stats();
    Json(serde_json::json!({
        "status": "ok",
        "tokenizer": state.inner.tokenizer.name(),
        // Response-memo counters. `hits` are requests answered without entering the
        // model at all, so this is the one rate on this endpoint that describes work
        // *not* done. Zeroes when COLI_MEMO_ENTRIES/COLI_MEMO_MB disable it.
        "memo": { "hits": hits, "misses": misses, "entries": entries, "bytes": bytes }
    }))
}

/// `GET /metrics` — the engine's live telemetry.
///
/// This endpoint is why `PlanOptimizer::snapshot` says "safe to call from a
/// `/metrics` handler" and why `BubbleTuner::ewma_snapshot` says "used for
/// /metrics": both were written for a handler that did not exist, so the
/// adaptive runtime observed itself and then had nowhere to say so. Everything
/// here was already being computed every forward.
///
/// Deliberately **not** behind `check_auth`, matching `/health`: an operator
/// scraping liveness and an operator scraping load are the same operator, and
/// requiring a key on one but not the other is the kind of asymmetry that ends
/// with monitoring disabled. Nothing here is request content — only counters.
async fn metrics(State(state): State<AppState>) -> Json<serde_json::Value> {
    let t = state.inner.engine.telemetry();
    let (hits, misses, entries, bytes) = state.inner.memo.lock().stats();
    // `cpu_bytes` is carried in `LaneTimings` and was dropped here until
    // 2026-08-10, which mattered more than one field usually would: `io_us` and
    // `cpu_us` are **sums across threads** (rings and workers, different counts),
    // so neither is comparable to wall time or to the other without knowing both
    // thread counts. Bytes do not double-count that way, so `cpu_bytes` per
    // forward is the one figure here that converts to an aggregate rate directly.
    // (It is bytes fed to compute — warm-cache hits included — not disk bytes.)
    let lane = |l: &peregrine_model::LaneTimings| {
        serde_json::json!({
            "io_us": l.io_us,
            "cpu_us": l.cpu_us,
            "gpu_us": l.gpu_us,
            "reduce_us": l.reduce_us,
            "cpu_bytes": l.cpu_bytes,
        })
    };
    Json(serde_json::json!({
        "steps": t.steps,
        "active": t.active,
        "pending": t.pending,
        // Two lane views, and they answer different questions: `last` is what the
        // most recent token cost, `ewma` is which lane structurally dominates —
        // the one the balancer acts on. Reporting only one hides either a spike
        // or a trend.
        "lane_last": lane(&t.lane_last),
        "lane_ewma": lane(&t.lane_ewma),
        "bias": format!("{:?}", t.runtime.bias),
        "io_ewma_us": t.runtime.io_ewma_us,
        "io_sq_full": t.runtime.io_sq_full,
        "prefetch_accuracy": t.runtime.prefetch_accuracy,
        "cache_hit_rate": t.runtime.cache_hit_rate,
        // Cumulative and byte-convertible: delta these across two scrapes and
        // multiply by bytes-per-expert for a live disk rate. `disk_reads` counts
        // *experts* (six regions each), not regions or device requests.
        // `prefetch_reads` is the speculative lane, which contributes to device
        // load but to no lane timing.
        "ecache": t.ecache.map(|(h, m, d)| serde_json::json!({
            "hits": h, "misses": m, "disk_reads": d, "prefetch_reads": t.prefetch_reads,
        })),
        "routing_entropy_ewma": t.runtime.entropy_ewma,
        // MTP speculation: delta across scrapes for a live accept rate. Every
        // proposed draft is a verify row in the batched forward; the accept
        // rate is what says whether COLI_DRAFT's depth pays for those rows.
        "spec": {
            "proposed": t.spec_proposed,
            "accepted": t.spec_accepted,
            // Drafts the COLI_SPEC_CONF floor cut short (0 with the floor off).
            "conf_stops": t.spec_conf_stops,
            "accept_rate": if t.spec_proposed > 0 {
                t.spec_accepted as f64 / t.spec_proposed as f64
            } else { 0.0 },
        },
        // RLM recursive refinement (COLI_RLM): passes emitted and tokens that
        // triggered at least one.
        "rlm": { "passes": t.rlm.0, "tokens_recursed": t.rlm.1 },
        // Disk-persisted KV sessions (COLI_KV_STORE_DIR); null when off.
        "kvstore": t.kvstore.map(|(s, l, r)| serde_json::json!({
            "saved": s, "loaded": l, "tokens_restored": r,
        })),
        // O_DIRECT slab buffers in flight (null when experts are resident);
        // pinned at the pool cap = reads serializing on buffer availability.
        "io_slab_in_use": t.io_slab_in_use,
        // Which implementation is dispatching, read from the dispatch path
        // itself — not from `COLI_MOE_ENGINE`, which says what was requested.
        "moe_engine": peregrine_model::concurrent::moe_engine_name(),
        "gpu": {
            "calls": t.runtime.gpu.calls,
            "experts": t.runtime.gpu.experts,
            "rows": t.runtime.gpu.rows,
            "transfer_fraction": t.runtime.gpu.transfer_fraction(),
            // Whether COLI_CUDA_GRAPH is actually replaying. Zero replays with
            // nonzero captures means the launch-shape key is churning and the
            // knob is costing throughput rather than saving it.
            "graph_captures": t.runtime.gpu.graph_captures,
            "graph_replays": t.runtime.gpu.graph_replays,
            "graph_invalidations": t.runtime.gpu.graph_invalidations,
            "graph_uncacheable": t.runtime.gpu.graph_uncacheable,
        },
        "memo": { "hits": hits, "misses": misses, "entries": entries, "bytes": bytes },
    }))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<ModelList>, ApiError> {
    check_auth(&state, &headers)?;
    Ok(Json(ModelList {
        object: "list",
        data: vec![ModelCard { id: state.inner.args.model_id.clone(), object: "model", owned_by: "peregrine" }],
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;
    let (ids, max_new, temperature, top_p) = resolve_params(&state, &req).await?;
    let model_id = state.inner.args.model_id.clone();
    let priority = priority_from_header(headers.get("x-peregrine-priority").and_then(header_utf8));
    let class = classify_request(&req.messages);
    let tokenizer = state.inner.tokenizer.clone();

    let completion_id = format!("chatcmpl-{}", seed());
    let created = unix_seconds();
    let prompt_tokens = ids.len();

    // Response memo. Only greedy requests are eligible (see `MemoKey::eligible`), and
    // the key is the complete request semantics, so a one-token or one-option change
    // misses. A hit is answered here — before `submit_request` — so it never enters
    // the engine, never occupies a batch slot and never publishes KV state.
    let memo_key = memo::MemoKey::eligible(temperature).then(|| memo::MemoKey {
        ids: ids.clone(),
        max_new,
        top_p_bits: top_p.to_bits(),
        model: model_id.clone(),
    });
    if let Some(key) = &memo_key {
        let hit = state.inner.memo.lock().get(key);
        if let Some(out_ids) = hit {
            // Stored as token ids, so the transport framing is rebuilt fresh: this
            // request's own completion id and timestamp, and whichever wire format it
            // asked for. A streaming request can be served from an entry a
            // non-streaming one created. The replay decodes and re-chunks the whole
            // completion — CPU time proportional to its length — so it runs on the
            // blocking pool like every other decode.
            let stream = req.stream;
            let tool_defs = active_tools(&req).to_vec();
            let replay_tok = tokenizer.clone();
            let frame = ReplayFrame {
                completion_id: completion_id.clone(),
                model_id: model_id.clone(),
                created,
                prompt_tokens,
            };
            return tokio::task::spawn_blocking(move || {
                memo_response(stream, &tool_defs, &replay_tok, &out_ids, &frame)
            })
            .await
            .map_err(|e| ApiError::internal(format!("memo replay task: {e}")))?;
        }
    }

    let mut rx = submit_request(&state, &ids, max_new, temperature, top_p, priority, class)?;

    if req.stream {
        // SSE: an async task decodes engine token ids into text deltas and pushes
        // OpenAI chunk events; the response streams them.
        let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let mid = model_id.clone();
        let cid = completion_id.clone();
        let memo_state = state.inner.clone();
        // The schemas outlive the request body here: the filter types arguments
        // from them, and the spawned task owns everything it touches.
        let stream_tools: Vec<tools::ToolDef> = active_tools(&req).to_vec();
        tokio::spawn(async move {
            // Token payloads split multi-byte characters, so deltas come from an
            // incremental decoder that holds an unfinished character until the
            // next token completes it (see `IncrementalDecoder`).
            let mut dec = tok::IncrementalDecoder::new();
            // One lock acquisition per stream, then per-token decode is a
            // lock-free table read — N streams at token rate must not contend
            // on the encode mutex.
            let vocab = tokenizer.decode_handle();
            let mut out_ids: Vec<u32> = Vec::new();
            // Tool markup arrives split across tokens, so the decoded text goes
            // through the filter before any of it becomes a delta: a client must
            // never see half a `<tool_call>`.
            let mut filter = tools::OutputFilter::with_tools(&stream_tools);
            let mut emitted_calls = 0usize;
            // OpenAI clients expect the role in the first chunk.
            if sse_tx.send(Ok(chunk_event(&cid, &mid, created, None, Some("assistant"), None))).await.is_err() {
                return; // client disconnected before the first frame
            }
            while let Some(msg) = rx.recv().await {
                match msg {
                    EngineOut::Token(t) => {
                        out_ids.push(t);
                        let decoded = dec.push(vocab.token_bytes(t).unwrap_or(&[]));
                        if decoded.is_empty() {
                            continue; // token only extended an unfinished character
                        }
                        let delta = filter.push(&decoded);
                        for c in filter.take_calls() {
                            let call = c.to_openai(emitted_calls, &call_id(&cid, emitted_calls));
                            emitted_calls += 1;
                            if sse_tx.send(Ok(tool_call_chunk_event(&cid, &mid, created, call))).await.is_err() {
                                return; // client disconnected
                            }
                        }
                        if delta.is_empty() {
                            continue; // text held back as a possible partial marker
                        }
                        let ev = chunk_event(&cid, &mid, created, Some(&delta), None, None);
                        if sse_tx.send(Ok(ev)).await.is_err() {
                            return; // client disconnected
                        }
                    }
                    EngineOut::Error(m) => {
                        if sse_tx.send(Ok(sse_error(&m))).await.is_err() {
                            peregrine_core::note_advisory_err("SSE error event", &"client already disconnected");
                        }
                        return;
                    }
                }
            }
            // Reaching here means the engine's channel closed with no error, so the
            // completion is whole. Every earlier exit — an engine error, a client
            // disconnect mid-stream — returns above with a partial `out_ids` and
            // memoizes nothing: a truncated answer replayed as a complete one would
            // be worse than no memo at all.
            if let Some(key) = memo_key {
                memo_state.memo.lock().insert(key, out_ids);
            }
            // tail frames: a send error just means the client hung up first
            let tail = {
                let decoded_tail = dec.finish();
                let mut t = filter.push(&decoded_tail);
                t.push_str(&filter.finish());
                t
            };
            let tail_ev = if tail.is_empty() {
                None
            } else {
                Some(chunk_event(&cid, &mid, created, Some(&tail), None, None))
            };
            let mut hung_up = false;
            if let Some(ev) = tail_ev {
                hung_up = sse_tx.send(Ok(ev)).await.is_err();
            }
            // A call closed only by end-of-generation still reaches the client.
            for c in filter.take_calls() {
                let call = c.to_openai(emitted_calls, &call_id(&cid, emitted_calls));
                emitted_calls += 1;
                hung_up = hung_up || sse_tx.send(Ok(tool_call_chunk_event(&cid, &mid, created, call))).await.is_err();
            }
            let finish = if emitted_calls > 0 { "tool_calls" } else { "stop" };
            if hung_up
                || sse_tx.send(Ok(chunk_event(&cid, &mid, created, None, None, Some(finish)))).await.is_err()
                || sse_tx.send(Ok(Event::default().data("[DONE]"))).await.is_err()
            {
                peregrine_core::note_advisory_err("SSE stream tail", &"client disconnected before [DONE]");
            }
        });
        let stream = ReceiverStream::new(sse_rx);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming: collect the whole token stream, decode once, one JSON body.
        let mut out_ids: Vec<u32> = Vec::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                EngineOut::Token(t) => out_ids.push(t),
                EngineOut::Error(m) => return Err(ApiError::internal(m)),
            }
        }
        // The stream closed without an error, so this completion is whole and may be
        // memoized. The error arm above returns instead, so a failed generation is
        // never stored.
        if let Some(key) = memo_key {
            state.inner.memo.lock().insert(key, out_ids.clone());
        }
        // Decode goes through the same global tokenizer mutex as encode, so it
        // belongs on the blocking pool for the same reason (see `resolve_params`).
        let n_out = out_ids.len();
        let decoded = tokio::task::spawn_blocking(move || tokenizer.decode(&out_ids))
            .await
            .map_err(|e| ApiError::internal(format!("decode task: {e}")))?;
        let (text, calls) = split_output(&tk(decoded)?, active_tools(&req));
        Ok(Json(json_completion(
            &text,
            &calls,
            &completion_id,
            &model_id,
            created,
            prompt_tokens,
            n_out,
        ))
        .into_response())
    }
}

/// What the model is actually being asked: the rendered prompt string and the
/// exact ids it tokenizes to, for the same `messages` a completion would take.
///
/// Exists because two sessions independently lost hours to "what tokens are
/// these" during one debugging pass — the answer required rendering the
/// checkpoint's chat template through a reference tokenizer by hand, and it
/// turned out to be the whole explanation (the template opens a `<think>`
/// block this server's filter then drops). A server that can generate tokens
/// but cannot show them makes that class of question archaeology.
///
/// Authenticated like every other `/v1/*` route, and deliberately echoes the
/// prompt VERBATIM including control markup — that is the point: the bug being
/// hunted is usually in the markup.
async fn debug_tokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    let prompt = build_prompt(&req.messages, active_tools(&req), state.inner.chatml_prompt);
    let tokenizer = state.inner.tokenizer.clone();
    let p2 = prompt.clone();
    let ids = tk(tokio::task::spawn_blocking(move || tokenizer.encode(&p2))
        .await
        .map_err(|e| ApiError::internal(format!("encode task: {e}")))?)?;
    // Per-id decode alongside the ids: an id that renders to nothing is the
    // signature that matters, and only the pairing makes it visible.
    let vocab = state.inner.tokenizer.decode_handle();
    let pieces: Vec<String> = ids
        .iter()
        .map(|&t| String::from_utf8_lossy(vocab.token_bytes(t).unwrap_or(&[])).into_owned())
        .collect();
    Ok(Json(serde_json::json!({
        "prompt": prompt,
        "ids": ids,
        "pieces": pieces,
        "n_tokens": ids.len(),
    })))
}

/// The OpenAI `chat.completion` body. Shared by the generated and memoized paths so
/// a replayed response cannot drift in shape from a fresh one.
fn json_completion(
    text: &str,
    calls: &[tools::ParsedCall],
    completion_id: &str,
    model_id: &str,
    created: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> serde_json::Value {
    let mut message = serde_json::Map::new();
    message.insert("role".into(), serde_json::json!("assistant"));
    // `content` stays present-but-null when the turn was only tool calls: a
    // client that reads `content` unconditionally gets null, not the markup.
    message.insert(
        "content".into(),
        if text.is_empty() && !calls.is_empty() { serde_json::Value::Null } else { serde_json::json!(text) },
    );
    if !calls.is_empty() {
        let arr: Vec<serde_json::Value> =
            calls.iter().enumerate().map(|(i, c)| c.to_openai(i, &call_id(completion_id, i))).collect();
        message.insert("tool_calls".into(), serde_json::Value::Array(arr));
    }
    // A turn with calls finishes as `tool_calls`; agent clients branch on this
    // rather than on the presence of the array.
    let finish = if calls.is_empty() { "stop" } else { "tool_calls" };
    serde_json::json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": serde_json::Value::Object(message),
            "finish_reason": finish
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

/// Serve a memoized completion in whichever wire format this request asked for.
///
/// The memo holds token ids, so the framing is rebuilt here rather than replayed:
/// this request's own completion id and `created` timestamp, and its own choice of
/// SSE or JSON. That is what lets a streaming request be answered from an entry a
/// non-streaming one created — and it is why storing rendered wire bytes would have
/// been a mistake, since it would leak the original request's identifiers.
///
/// The SSE path re-chunks through the same [`tok::IncrementalDecoder`] as a live
/// stream, so a multi-byte character is split across deltas identically. It is sent
/// as one ready-made stream with no engine behind it.
/// The framing a memo replay rebuilds — always *this* request's identifiers and
/// timestamp, never the original's (see [`memo_response`]).
struct ReplayFrame {
    completion_id: String,
    model_id: String,
    created: u64,
    prompt_tokens: usize,
}

fn memo_response(
    stream: bool,
    tool_defs: &[tools::ToolDef],
    tokenizer: &TokenBackend,
    out_ids: &[u32],
    frame: &ReplayFrame,
) -> Result<Response, ApiError> {
    let (completion_id, model_id) = (frame.completion_id.as_str(), frame.model_id.as_str());
    let created = frame.created;
    if !stream {
        let (text, calls) = split_output(&tk(tokenizer.decode(out_ids))?, tool_defs);
        return Ok(Json(json_completion(
            &text,
            &calls,
            completion_id,
            model_id,
            created,
            frame.prompt_tokens,
            out_ids.len(),
        ))
        .into_response());
    }
    let mut events: Vec<Result<Event, std::convert::Infallible>> =
        vec![Ok(chunk_event(completion_id, model_id, created, None, Some("assistant"), None))];
    let mut dec = tok::IncrementalDecoder::new();
    // The replay runs through the same filter as a live stream, so a memoized
    // tool call comes back as a call and not as the markup that produced it.
    let mut filter = tools::OutputFilter::with_tools(tool_defs);
    let mut emitted_calls = 0usize;
    let vocab = tokenizer.decode_handle();
    for &t in out_ids {
        let decoded = dec.push(vocab.token_bytes(t).unwrap_or(&[]));
        if decoded.is_empty() {
            continue;
        }
        let delta = filter.push(&decoded);
        for c in filter.take_calls() {
            let call = c.to_openai(emitted_calls, &call_id(completion_id, emitted_calls));
            emitted_calls += 1;
            events.push(Ok(tool_call_chunk_event(completion_id, model_id, created, call)));
        }
        if !delta.is_empty() {
            events.push(Ok(chunk_event(completion_id, model_id, created, Some(&delta), None, None)));
        }
    }
    let tail = {
        let decoded_tail = dec.finish();
        let mut t = filter.push(&decoded_tail);
        t.push_str(&filter.finish());
        t
    };
    if !tail.is_empty() {
        events.push(Ok(chunk_event(completion_id, model_id, created, Some(&tail), None, None)));
    }
    for c in filter.take_calls() {
        let call = c.to_openai(emitted_calls, &call_id(completion_id, emitted_calls));
        emitted_calls += 1;
        events.push(Ok(tool_call_chunk_event(completion_id, model_id, created, call)));
    }
    let finish = if emitted_calls > 0 { "tool_calls" } else { "stop" };
    events.push(Ok(chunk_event(completion_id, model_id, created, None, None, Some(finish))));
    events.push(Ok(Event::default().data("[DONE]")));
    Ok(Sse::new(tokio_stream::iter(events)).keep_alive(KeepAlive::default()).into_response())
}

/// Split a whole generation into visible text and tool calls. The one-shot
/// counterpart of the streaming filter, so both wire formats agree on what was
/// content and what was markup.
fn split_output(raw: &str, tool_defs: &[tools::ToolDef]) -> (String, Vec<tools::ParsedCall>) {
    let mut f = tools::OutputFilter::with_tools(tool_defs);
    let mut text = f.push(raw);
    text.push_str(&f.finish());
    (text.trim().to_string(), f.take_calls())
}

/// A tool call's id, derived from the completion id so it is unique per
/// response and stable between the streaming and non-streaming renderings of
/// the same generation — a client correlates its tool result against this.
fn call_id(completion_id: &str, index: usize) -> String {
    format!("call_{}_{index}", completion_id.trim_start_matches("chatcmpl-"))
}

/// Wall-clock seconds since the epoch, for the OpenAI `created` field.
fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// One streaming chunk in OpenAI `chat.completion.chunk` shape. `id`/`created`
/// are stable across a response's chunks (clients correlate on them); `role` is
/// set only on the opening chunk, `finish` only on the closing one.
fn chunk_event(
    id: &str,
    model_id: &str,
    created: u64,
    delta: Option<&str>,
    role: Option<&str>,
    finish: Option<&str>,
) -> Event {
    let mut delta_obj = serde_json::Map::new();
    if let Some(r) = role {
        delta_obj.insert("role".into(), serde_json::json!(r));
    }
    if let Some(d) = delta {
        delta_obj.insert("content".into(), serde_json::json!(d));
    }
    let payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{ "index": 0, "delta": serde_json::Value::Object(delta_obj), "finish_reason": finish }]
    });
    Event::default().data(payload.to_string())
}

/// A streaming chunk carrying one whole tool call.
///
/// OpenAI streams a call in fragments (name first, `arguments` accumulated
/// across deltas); this emits it complete in a single delta instead. Both are
/// legal — a client concatenates argument fragments either way — and a whole
/// call is the only form this server can honestly send, since the markup is
/// not a valid call until its closing tag arrives.
fn tool_call_chunk_event(id: &str, model_id: &str, created: u64, call: serde_json::Value) -> Event {
    let payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": { "tool_calls": [call] },
            "finish_reason": serde_json::Value::Null
        }]
    });
    Event::default().data(payload.to_string())
}

fn sse_error(message: &str) -> Event {
    Event::default().data(serde_json::json!({ "error": { "message": message } }).to_string())
}

/// Tokenizer throughput bench, three rows for the three regimes:
///  - `line`  — one facade `encode` per non-empty line (the serve pattern of
///    many short encodes; the historical row, comparable to the documented
///    HF-ratio measurement),
///  - `whole` — one `encode_into` call over the entire file (the engine's
///    single-core capability; matches upstream's single-thread bench shape),
///  - `parN`  — `encode_batch` over the lines with N forked workers (what
///    upstream's batch layer does; engages only on bulk input).
///
/// Correctness is the parity test suite's job (`tests/tokenizer_parity.rs`,
/// HF oracle as a dev-dependency); this only measures throughput. Numbers are
/// local to this box; no docs-level claims.
fn bench_tokenizer(model_dir: &std::path::Path, file: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read(model_dir.join("tokenizer.json"))?;
    let text = std::fs::read_to_string(file)?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    if lines.is_empty() {
        return Err("bench file has no non-empty lines".into());
    }
    eprintln!("bench: {} lines, {:.1} MB", lines.len(), bytes as f64 / 1e6);

    let mut giga = peregrine_token::GigaTokenizer::from_hf_json_bytes(&json)
        .map_err(|e| format!("gigatoken: {e}"))?;

    // warmup (fills the pretoken memo cache the way a running server would;
    // the ids themselves are not needed here)
    for l in lines.iter().take(64) {
        giga.encode(l);
    }

    let mbs = |b: usize, s: f64| b as f64 / 1e6 / s.max(1e-9);

    // Three passes per row, best-of reported: a single pass on this box swings
    // by more than the few-percent effects the tokenizer changes under test
    // produce (same discipline as docs/measurement.md, scaled down to
    // milliseconds). Min, not mean — the floor is the machine's capability;
    // the excursions above it are scheduler noise.
    const PASSES: usize = 3;

    let mut line_ids = 0usize;
    let mut line_s = f64::INFINITY;
    for _ in 0..PASSES {
        let t0 = std::time::Instant::now();
        line_ids = 0;
        for l in &lines {
            line_ids += giga.encode(l).len();
        }
        line_s = line_s.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "gigatoken/line  : {:8.2} MB/s  ({line_ids} ids, best of {PASSES}: {line_s:.3}s)",
        mbs(bytes, line_s)
    );

    let mut out: Vec<u32> = Vec::with_capacity(text.len() / 3);
    let mut whole_s = f64::INFINITY;
    for _ in 0..PASSES {
        out.clear();
        let t0 = std::time::Instant::now();
        giga.encode_into(&text, &mut out);
        whole_s = whole_s.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "gigatoken/whole : {:8.2} MB/s  ({} ids, best of {PASSES}: {whole_s:.3}s)",
        mbs(text.len(), whole_s),
        out.len()
    );
    drop(out); // keep the parallel row's peak RSS to its own output

    // Batch input at upstream's granularity: documents, not lines — the file
    // sliced at line boundaries into ~256 KiB pieces (also keeps the output
    // to a few hundred Vecs instead of one per line).
    let mut docs: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;
    for line in text.split_inclusive('\n') {
        acc += line.len();
        if acc >= 256 << 10 {
            let end = start + acc;
            docs.push(&text[start..end]);
            start = end;
            acc = 0;
        }
    }
    if start < text.len() {
        docs.push(&text[start..]);
    }
    let workers = match std::thread::available_parallelism() {
        Ok(n) => n.get(),
        Err(e) => {
            eprintln!("bench: available_parallelism unknown ({e}); using 1 worker");
            1
        }
    };
    // p1 pays one-time worker construction (the pool persists on the
    // instance) and is inherently a single observation; p2 is the steady
    // state a long batch run sees, best of PASSES.
    let t0 = std::time::Instant::now();
    let par_ids: usize = giga.encode_batch(&docs, workers).iter().map(|v| v.len()).sum();
    let par_s = t0.elapsed().as_secs_f64();
    println!(
        "gigatoken/par{workers:<2} p1: {:8.2} MB/s  ({} docs, {par_ids} ids, {par_s:.3}s)",
        mbs(text.len(), par_s),
        docs.len()
    );
    let mut par_s = f64::INFINITY;
    let mut par_ids = 0usize;
    for _ in 0..PASSES {
        let t0 = std::time::Instant::now();
        par_ids = giga.encode_batch(&docs, workers).iter().map(|v| v.len()).sum();
        par_s = par_s.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "gigatoken/par{workers:<2} p2: {:8.2} MB/s  ({} docs, {par_ids} ids, best of {PASSES}: {par_s:.3}s)",
        mbs(text.len(), par_s),
        docs.len()
    );
    Ok(())
}

/// Install `peregrine-sched`'s two-lane engine when `COLI_MOE_ENGINE=sched`.
///
/// Mirrors `peregrine-engine`'s installer, and for the same structural reason:
/// `peregrine-sched` depends on `peregrine-model`, so only a binary can bridge
/// the two. Slower by construction (no GPU lane, no warm cache, no prefetch) —
/// it is an A/B against the default, not a default.
fn install_moe_engine() {
    // Matched through `as_deref()` — the idiom every other env gate in this
    // repo uses. `unwrap_or_default()` is a [P] hit and `Err(_)` a [B] one,
    // and both audits are right: neither says which case it is handling.
    let var = std::env::var("COLI_MOE_ENGINE");
    match var.as_deref() {
        Ok("sched") => {}
        // Unset, empty, or the default engine: nothing to install.
        Ok("") | Ok("concurrent") => return,
        Ok(other) => {
            eprintln!("peregrine: COLI_MOE_ENGINE={other} is not a known engine (concurrent|sched); using concurrent");
            return;
        }
        _ => return,
    }
    let depth: u32 = std::env::var("COLI_IO_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    match peregrine_sched::SchedEngine::new(depth) {
        Ok(engine) => {
            if peregrine_model::concurrent::install_moe_engine(Box::new(engine)) {
                eprintln!("peregrine-serve: MoE engine = sched (ring depth {depth}) — two-lane: no GPU lane, no warm cache, no prefetch");
            }
        }
        Err(e) => eprintln!("peregrine-serve: COLI_MOE_ENGINE=sched requested but the io_uring ring failed ({e}); using concurrent"),
    }
    // Confirm against the dispatch path rather than against the branch above.
    // Every message in this function reports an *intent*; this one reports what
    // `moe_forward_dispatch` will actually do, and the two differ whenever the
    // ring failed or something installed first.
    if !peregrine_model::concurrent::moe_engine_installed() {
        eprintln!("peregrine-serve: MoE dispatch = concurrent (no alternative engine installed)");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cap glibc arenas before the model spawns its worker pools, so the server no
    // longer needs `MALLOC_ARENA_MAX=2` in the environment to keep RSS flat.
    peregrine_model::cap_malloc_arenas();
    eprintln!("{}", peregrine_model::startup_banner());
    install_moe_engine();
    let args = Args::parse();
    let dir = std::path::PathBuf::from(&args.model);
    // Tokenizer throughput bench: encode a text file through both backends and
    // exit — no model weights loaded, so it runs anywhere tokenizer.json does.
    if let Some(file) = &args.bench_tokenizer {
        bench_tokenizer(&dir, file)?;
        return Ok(());
    }
    let mut model = Model::load(&dir)?;
    // `COLI_PREDICT_SOURCE`: force a specific prefetch predictor. The stdio binary
    // has always honoured this and the server never did, so the one knob that can
    // select `PhaseAware` was unreachable from the **batched** engine — which is
    // precisely where per-sequence prefetch lives, and so the only place the
    // phase-aware predictor was ever meant to matter. Applied before the model
    // moves into the engine thread; no thread affinity involved, unlike the perf
    // counter, which stays with whoever decodes.
    if let Some(name) = model.apply_predictor_override() {
        eprintln!("peregrine-serve: prefetch predictor = {name} (COLI_PREDICT_SOURCE)");
    }
    let tokenizer = TokenBackend::load(&dir).map_err(|e| format!("tokenizer: {e}"))?;
    // Capture the prompt dialect before the model moves into the engine thread.
    let chatml_prompt = model.uses_chatml_prompt();

    // One engine thread owns the model and continuously batches all requests.
    let (engine, engine_join) = batch::spawn(model, args.max_batch)?;

    let addr = format!("{}:{}", args.host, args.port);
    let state = AppState {
        inner: Arc::new(Inner {
            engine,
            tokenizer: Arc::new(tokenizer),
            args,
            memo: parking_lot::Mutex::new(memo::ResponseMemo::from_env()),
            chatml_prompt,
        }),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/debug/tokenize", post(debug_tokenize))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("peregrine-serve listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                eprintln!("peregrine-serve: ctrl-c handler failed ({e}); shutting down");
            }
            eprintln!("peregrine-serve shutting down");
        })
        .await?;

    // Wait for the engine thread to drain before returning from `main`.
    //
    // The join handle was `_engine_join` until 2026-08-08 — held, never joined —
    // so the process exited while the engine thread was still alive and
    // `Model::drop` never ran. That silently cost two things the server is
    // documented to do: **`route_stats.json` is written at Drop**, so the HTTP
    // server never persisted routing heat or co-activation across sessions
    // (`COLI_ROUTE_STATS_PERSIST` had no effect here, only in the stdio binary),
    // and the `[ecache]` / `[prefetch] used/wasted/accuracy` shutdown counters
    // never printed — which is why a lane-count sweep against this server could
    // not read the one diagnostic that says whether prefetch is earning its keep.
    //
    // Bounded, because the wait is not guaranteed to end: the engine exits when
    // both request senders drop, and those live in an `Arc<Inner>` that a
    // detached SSE pump task may still hold — graceful shutdown waits for
    // connections, not for `tokio::spawn`ed tasks. Losing the counters is a bad
    // trade for a server that will not exit, so this reports and moves on.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ok = engine_join.join().is_ok();
        if done_tx.send(ok).is_err() {
            // `main` hit the timeout below and stopped listening. Expected, not a
            // fault — but it means the drain finished *after* the deadline, which
            // is the one case where raising the deadline would have helped.
            peregrine_core::note_advisory_err(
                "engine join handoff",
                &"engine drained after main stopped waiting — consider a longer shutdown deadline",
            );
        }
    });
    match done_rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(true) => {}
        Ok(false) => eprintln!("peregrine-serve: batch engine thread panicked"),
        // Timeout and Disconnected mean different things and get different words:
        // the first is a slow drain, the second is a watchdog that died before
        // reporting — which would otherwise look like a clean shutdown.
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => eprintln!(
            "peregrine-serve: batch engine still busy after 30s; \
             route_stats.json and the [ecache]/[prefetch] counters may be incomplete"
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!("peregrine-serve: engine join watchdog exited without reporting")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The HTTP surface's own contracts — the ones that live in `main.rs` rather
    //! than in `tools.rs`.
    //!
    //! This module did not exist before 2026-08-08, which meant the entire
    //! tool-calling integration — prompt assembly, `tool_choice`, the OpenAI
    //! response shape — was reachable only through a running server with a
    //! loaded model, so nothing exercised it. `tools.rs` was well covered on its
    //! own; the wiring between it and the wire format was not covered at all.
    //!
    //! Fixtures go through `serde_json::from_value` rather than being built by
    //! hand: `ChatRequest`/`ChatMessage` are `Deserialize`-only, and going in
    //! through the wire shape tests the deserialization contract at the same
    //! time — which is precisely what `docs/serving.md` was stale about.

    use super::*;
    use serde_json::json;

    fn msgs(v: serde_json::Value) -> Result<Vec<ChatMessage>, serde_json::Error> {
        serde_json::from_value(v)
    }

    fn tool_defs() -> Result<Vec<tools::ToolDef>, serde_json::Error> {
        serde_json::from_value(json!([{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
            }
        }]))
    }

    #[test]
    fn the_preamble_lands_in_the_first_system_turn_and_only_there(
    ) -> Result<(), serde_json::Error> {
        let m = msgs(json!([
            {"role": "system", "content": "you are first"},
            {"role": "system", "content": "you are second"},
            {"role": "user", "content": "hi"}
        ]))?;
        let p = build_prompt(&m, &tool_defs()?, false);
        assert_eq!(p.matches("# Tools").count(), 1, "exactly one preamble:\n{p}");
        let first = p.find("you are first").unwrap_or(usize::MAX);
        let tools_at = p.find("# Tools").unwrap_or(usize::MAX);
        let second = p.find("you are second").unwrap_or(usize::MAX);
        assert!(first < tools_at && tools_at < second, "preamble belongs to the FIRST system turn:\n{p}");
        Ok(())
    }

    #[test]
    fn chatml_preclosed_reasoning_so_the_filter_has_something_to_emit() -> Result<(), serde_json::Error> {
        let m = msgs(json!([{"role": "user", "content": "Hi"}]))?;
        let p = build_prompt(&m, &[], true);
        assert!(
            p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "the reasoning block must arrive pre-closed; got {p:?}"
        );
        // What the filter does with each shape, which is why the default is
        // what it is: ordinary text reaches the client, an unclosed reasoning
        // block is swallowed whole.
        let mut f = tools::OutputFilter::with_tools(&[]);
        assert_eq!(f.push("Paris is the capital."), "Paris is the capital.");
        let mut f2 = tools::OutputFilter::with_tools(&[]);
        assert!(
            f2.push("<think>reasoning that never closes").is_empty(),
            "unclosed reasoning is dropped — the empty-response failure this default avoids"
        );
        Ok(())
    }

    #[test]
    fn chatml_dialect_renders_qwen_markup_not_glm() -> Result<(), serde_json::Error> {
        let m = msgs(json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]))?;
        let p = build_prompt(&m, &[], true);
        // ChatML markup, ending on an assistant turn with the reasoning block
        // already closed (see `chatml_preclosed_reasoning_so_the_filter_has_
        // something_to_emit` for why that closure is load-bearing rather than
        // cosmetic).
        assert!(p.contains("<|im_start|>system\nbe terse<|im_end|>\n"), "system turn:\n{p}");
        assert!(p.contains("<|im_start|>user\nhi<|im_end|>\n"), "user turn:\n{p}");
        assert!(p.contains("<|im_start|>assistant\n"), "assistant turn:\n{p}");
        // Never GLM's control tokens — feeding those to Qwen is the bug this fixes.
        assert!(!p.contains("[gMASK]") && !p.contains("<sop>") && !p.contains("<|user|>"), "no GLM markup:\n{p}");
        Ok(())
    }

    #[test]
    fn a_system_turn_is_synthesized_only_when_there_are_tools() -> Result<(), serde_json::Error> {
        // Same messages, opposite outcomes — the shape that keeps a conditional
        // from quietly becoming unconditional.
        let m = msgs(json!([{"role": "user", "content": "hi"}]))?;
        let with = build_prompt(&m, &tool_defs()?, false);
        let without = build_prompt(&m, &[], false);
        assert!(with.contains("<|system|>"), "tools with no system turn must synthesize one:\n{with}");
        assert!(with.contains("# Tools"), "{with}");
        assert!(!without.contains("<|system|>"), "no tools must not invent a system turn:\n{without}");
        Ok(())
    }

    #[test]
    fn a_tool_result_replays_as_an_observation_not_a_user_turn(
    ) -> Result<(), serde_json::Error> {
        let m = msgs(json!([
            {"role": "user", "content": "read it"},
            {"role": "tool", "content": "file contents"}
        ]))?;
        let p = build_prompt(&m, &[], false);
        assert!(p.contains("<|observation|>\n<tool_response>\nfile contents\n</tool_response>"), "{p}");
        Ok(())
    }

    #[test]
    fn an_unknown_role_is_treated_as_user_content_and_never_as_markup(
    ) -> Result<(), serde_json::Error> {
        let m = msgs(json!([{"role": "<|system|>evil", "content": "hi"}]))?;
        let p = build_prompt(&m, &[], false);
        assert!(p.contains("<|user|>\nhi"), "unknown role falls back to user:\n{p}");
        assert!(!p.contains("evil"), "the role string must never reach the prompt:\n{p}");
        Ok(())
    }

    #[test]
    fn an_assistant_turns_calls_replay_as_the_markup_the_model_emitted(
    ) -> Result<(), serde_json::Error> {
        // The round trip that ties this file to tools.rs: render a call into the
        // prompt, then parse that same markup back and require the same call.
        // A divergence between the render and parse sides is invisible to either
        // module's own tests.
        let m = msgs(json!([{
            "role": "assistant", "content": null,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read", "arguments": "{\"filePath\":\"/etc/hosts\"}"}
            }]
        }]))?;
        let p = build_prompt(&m, &[], false);
        // Slice out just the markup: the surrounding `[gMASK]<sop>` and
        // `<|assistant|>` role markers are prompt scaffolding, not model output,
        // and feeding them to the output filter would (correctly) return them as
        // visible text.
        let start = p.find("<tool_call>").unwrap_or(0);
        let end = p.rfind("</tool_call>").map_or(p.len(), |i| i + "</tool_call>".len());
        let (text, calls) = split_output(&p[start..end], &[]);
        assert_eq!(text, "", "the replayed markup is a call, not visible text");
        assert_eq!(calls.len(), 1, "prompt: {p}");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["filePath"], json!("/etc/hosts"));
        Ok(())
    }

    #[test]
    fn tool_choice_none_is_the_only_value_that_disables_tools(
    ) -> Result<(), serde_json::Error> {
        let base = json!({"messages": [{"role": "user", "content": "hi"}], "tools": [{
            "type": "function", "function": {"name": "bash"}
        }]});
        let with_choice = |c: serde_json::Value| -> Result<usize, serde_json::Error> {
            let mut v = base.clone();
            v["tool_choice"] = c;
            let req: ChatRequest = serde_json::from_value(v)?;
            Ok(active_tools(&req).len())
        };
        assert_eq!(with_choice(json!("none"))?, 0, "\"none\" disables tools");
        assert_eq!(with_choice(json!("auto"))?, 1, "\"auto\" leaves them active");
        assert_eq!(with_choice(json!("required"))?, 1, "peregrine never forces a call");
        // The object form is a *decision*, not an oversight: `.as_str()` is None,
        // so a client naming one function still gets all of them.
        assert_eq!(
            with_choice(json!({"type": "function", "function": {"name": "bash"}}))?,
            1,
            "the object form does not narrow the set"
        );
        let req: ChatRequest = serde_json::from_value(base)?;
        assert_eq!(active_tools(&req).len(), 1, "absent tool_choice leaves them active");
        Ok(())
    }

    #[test]
    fn content_may_be_a_string_an_array_of_parts_null_or_absent(
    ) -> Result<(), serde_json::Error> {
        let m = msgs(json!([
            {"role": "user", "content": "plain"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]},
            {"role": "user", "content": null},
            {"role": "user"}
        ]))?;
        assert_eq!(m[0].text(), "plain");
        assert_eq!(m[1].text(), "ab", "text parts concatenate; non-text parts are dropped");
        assert_eq!(m[2].text(), "");
        assert_eq!(m[3].text(), "");
        Ok(())
    }

    #[test]
    fn a_calls_only_turn_nulls_content_and_finishes_as_tool_calls() {
        let calls = vec![tools::ParsedCall { name: "read".into(), arguments: json!({"p": 1}) }];
        let v = json_completion("", &calls, "chatcmpl-abc", "m", 0, 1, 1);
        let choice = &v["choices"][0];
        assert_eq!(choice["message"]["content"], serde_json::Value::Null, "null, not \"\"");
        assert_eq!(choice["finish_reason"], json!("tool_calls"));
        assert_eq!(choice["message"]["tool_calls"][0]["id"], json!("call_abc_0"));

        // Opposite outcome over the same function: no calls → text and "stop".
        let plain = json_completion("hello", &[], "chatcmpl-abc", "m", 0, 1, 1);
        let choice = &plain["choices"][0];
        assert_eq!(choice["message"]["content"], json!("hello"));
        assert_eq!(choice["finish_reason"], json!("stop"));
        assert_eq!(choice["message"].get("tool_calls"), None, "no empty array when there are no calls");
    }

    #[test]
    fn a_call_id_is_stable_per_completion_and_unique_per_index() {
        assert_eq!(call_id("chatcmpl-xyz", 0), "call_xyz_0");
        assert_eq!(call_id("chatcmpl-xyz", 0), call_id("chatcmpl-xyz", 0), "stable");
        assert_ne!(call_id("chatcmpl-xyz", 0), call_id("chatcmpl-xyz", 1), "unique per index");
        assert_ne!(call_id("chatcmpl-a", 0), call_id("chatcmpl-b", 0), "unique per completion");
    }

    #[test]
    fn split_output_separates_visible_text_from_calls() -> Result<(), serde_json::Error> {
        let (text, calls) = split_output(
            "here you go<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>ls</arg_value>\n</tool_call>",
            &tool_defs()?,
        );
        assert_eq!(text, "here you go", "trimmed, and no markup");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], json!("ls"), "declared string stays a string");
        Ok(())
    }
}
